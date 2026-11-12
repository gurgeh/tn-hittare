"""The two hooks for the fitted move-probability update: the update function and batch mode."""

import numpy as np
import polars as pl
import pytest

from tnh.builder.synth import random_game
from tnh.engine.batch import run_batch
from tnh.engine.search import Engine, Settings
from tnh.engine.synth import SynthEvaluator

INF = float("inf")


def test_the_update_is_applied_where_probabilities_are_read():
    g = random_game(np.random.default_rng(11), depth=5, branching=(2, 3))
    seen = []

    def sharpen(node, first, facts):
        seen.append(set(facts))
        best = np.zeros_like(first)
        best[int(np.argmax(facts["value_mover"]))] = 1.0
        return 0.5 * first + 0.5 * best           # the mover finds his best move more often

    s = Settings(my_side="white", kappa=INF, price=0.002, batch=4)
    plain = Engine(SynthEvaluator(g), s)
    r0 = plain.search(nodes=60)
    upd = Engine(SynthEvaluator(g), s, update=sharpen)
    r1 = upd.search(nodes=60)
    assert seen and {"value_mover", "best_minus", "stand_in", "finished", "visits", "mover_elo", "opp_elo"} <= seen[0]
    assert r1.solve.value != pytest.approx(r0.solve.value)
    # the probabilities the solver used are the updated ones; obviousness still reads the first ones
    i = next(i for i in upd.game.order if upd.graph.nodes[i].expanded and not upd.graph.nodes[i].me_to_move)
    n = upd.graph.nodes[i]
    assert not np.allclose(n.p_model, n.info["p_first"])
    mine = [i for i in upd.game.order if upd.graph.nodes[i].expanded and upd.graph.nodes[i].me_to_move]
    for i in mine:
        n = upd.graph.nodes[i]
        for j in range(len(n.moves)):
            assert upd.game.move_cost(i, j) == (0 if n.info["p_first"][j] >= s.obvious else 1)


def test_batch_mode_writes_one_row_per_root_move(tmp_path):
    g = random_game(np.random.default_rng(12), depth=5, branching=(2, 3))
    ev = SynthEvaluator(g)
    root = g.nodes[g.root]
    positions = [{"moves": [], "mover_elo": 2200, "opp_elo": 2100},
                 {"moves": [root.moves[0], "m0"], "mover_elo": 2000, "opp_elo": 2000}]
    out = tmp_path / "batch.parquet"
    df = run_batch(ev, positions, nodes=30, out=out, settings=Settings(kappa=INF, batch=4))
    back = pl.read_parquet(out)
    assert back.shape == df.shape
    assert set(back["pos"].to_list()) == {0, 1}
    first = back.filter(pl.col("pos") == 0)
    assert first["move"].to_list() == root.moves
    assert first["p_first"].sum() == pytest.approx(1.0) and first["p_mixed"].sum() == pytest.approx(1.0)
    assert set(back["status"].to_list()) <= {"expanded", "stand_in", "frontier", "finished"}
    assert (back["expanded_below"] >= 0).all() and (back["visits"] >= 0).all()
    # the searched root value for the mover is the mixed expectation over his moves
    assert (first["p_mixed"] * first["value_mover"]).sum() == pytest.approx(first["root_after"][0], abs=1e-9)


def test_the_fitted_database_mix_is_the_default():
    from tnh import movemix
    g = random_game(np.random.default_rng(13), depth=4, branching=(2, 3))
    for n in g.nodes:
        if n.moves and not n.me_to_move:
            n.counts = np.arange(1, len(n.moves) + 1, dtype=float) * 7
    ev = SynthEvaluator(g)
    eng = Engine(ev, Settings(my_side="white", price=0.002, batch=4))
    eng.search(nodes=30)
    rule = movemix.load()
    i = next(i for i in eng.game.order if eng.graph.nodes[i].expanded and not eng.graph.nodes[i].me_to_move)
    n = eng.graph.nodes[i]
    assert eng.game.dist[i] == pytest.approx(rule.mix(n.counts, n.p_model))
    fitted = eng.last.value
    calls = sum(ev.calls.values())
    eng.set_settings(mix="kappa", kappa=100.0)
    eng.solve()
    q = eng.game.dist[i]
    N = n.counts.sum()
    assert q == pytest.approx((n.counts + 100 * n.p_model) / (N + 100))
    assert eng.last.value != pytest.approx(fitted)
    assert sum(ev.calls.values()) == calls
