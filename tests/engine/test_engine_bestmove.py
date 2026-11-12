"""Mode bestmove: expectimax to a reach horizon, with values that depend on the models only."""

import math
import sys

import chess
import chess.engine
import numpy as np
import pytest

from tnh.builder.synth import build, leaf, me, opp, random_game
from tnh.engine.bestmove import theta
from tnh.engine.search import Engine, Settings
from tnh.engine.synth import SynthEvaluator

INF = float("inf")


def settings(**kw):
    return Settings(**{"my_side": "white", "kappa": INF, "mix": "kappa", "mode": "bestmove", "prior_min": 0.02,
                       "blunder": 0.3, "alpha": 1.0, **kw})


def reference(g, ev, s, th):
    """Brute force on the full tree with the same horizon rule (independent of the engine)."""
    static = lambda i: (g.nodes[i].value if not g.nodes[i].moves else float(ev.static[i]))  # noqa: E731
    rho = {g.root: 1.0}
    for i in sorted(range(len(g.nodes)), key=lambda i: g.nodes[i].ply):
        n = g.nodes[i]
        if i not in rho or rho[i] < th or not n.moves:
            continue
        if n.me_to_move:
            p = n.p_model
            cand = [j for j in range(len(n.moves)) if p[j] >= s.prior_min and static(n.children[j]) >= static(i) - s.blunder] \
                or [int(np.argmax(p))]
            top = max(p[j] for j in cand)
            facs = [(n.children[j], 1.0 if i == g.root else (p[j] / top) ** s.alpha) for j in cand]
        else:
            facs = [(c, float(n.p_model[j])) for j, c in enumerate(n.children) if n.p_model[j] > 0]
        for c, f in facs:
            rho[c] = max(rho.get(c, 0.0), rho[i] * f)

    def V(i):
        n = g.nodes[i]
        if rho.get(i, 0.0) < th or not n.moves:
            return static(i)
        if n.me_to_move:
            p = n.p_model
            cand = [j for j in range(len(n.moves)) if p[j] >= s.prior_min and static(n.children[j]) >= static(i) - s.blunder] \
                or [int(np.argmax(p))]
            return max(V(n.children[j]) for j in cand)
        q = n.p_model
        vs = [V(c) for c in n.children]
        return (1 - s.prepared) * float(np.dot(q, vs)) + s.prepared * min(v for v, x in zip(vs, q) if x > 0)
    root = g.nodes[g.root]
    return V(g.root), {m: V(c) for m, c in zip(root.moves, root.children)}


@pytest.mark.parametrize("seed", range(10))
def test_equals_brute_force_expectimax_on_small_trees(seed):
    g = random_game(np.random.default_rng(seed), depth=6, branching=(2, 3))
    ev = SynthEvaluator(g)
    for prepared in (0.0, 0.25):
        s = settings(prepared=prepared)
        eng = Engine(ev, s)
        eng.search(depth=6)
        for k, d in eng.bm.completed.items():
            v, moves = reference(g, ev, s, theta(k))
            assert d.value == pytest.approx(v, abs=1e-12)
            for m, x in d.root_values.items():
                assert x == pytest.approx(moves[m], abs=1e-12)


def test_value_at_a_depth_does_not_depend_on_what_else_was_expanded_or_the_order():
    g = random_game(np.random.default_rng(42), depth=7, branching=(2, 3))
    ev = SynthEvaluator(g)
    a = Engine(ev, settings())
    a.search(depth=5)
    b = Engine(ev, settings())
    # expand a random set of positions first, outside any horizon, in a shuffled order
    rng = np.random.default_rng(1)
    b.search(depth=0)
    for _ in range(30):
        cand = [n.id for n in b.graph.nodes if n.children is None and "result_mover" not in n.info]
        if not cand:
            break
        b._expand(list(rng.choice(cand, size=min(5, len(cand)), replace=False)))
    b.search(depth=5)
    for k in range(6):
        assert b.bm.completed[k].value == pytest.approx(a.bm.completed[k].value, abs=1e-12)
        assert b.bm.completed[k].root_values == pytest.approx(a.bm.completed[k].root_values, abs=1e-12)


def test_completed_depths_never_change():
    g = random_game(np.random.default_rng(7), depth=7, branching=(2, 3))
    eng = Engine(SynthEvaluator(g), settings())
    eng.search(depth=3)
    snap = {k: (d.value, dict(d.root_values)) for k, d in eng.bm.completed.items()}
    eng.search(depth=6)
    for k, (v, rv) in snap.items():
        assert eng.bm.completed[k].value == v and eng.bm.completed[k].root_values == rv


def test_prepared_and_draw_score_as_specified():
    # his position O: reply x (q 0.9) leads to 0.9 for me, y (q 0.1) to 0.3
    O = opp(("x", 0.9, 0.5, leaf(0.9)), ("y", 0.1, 0.5, leaf(0.3)), key="O", static=0.5)
    g = build(me(("toO", 1.0, 0.5, O), key="root", static=0.5))
    ev = SynthEvaluator(g)
    for P in (0.0, 0.25):
        eng = Engine(ev, settings(prepared=P, prior_min=0.0))
        eng.search(depth=2)
        d = eng.bm.completed[2]
        assert d.root_values["toO"] == pytest.approx((1 - P) * (0.9 * 0.9 + 0.1 * 0.3) + P * 0.3)
    # a drawn finished game is worth DrawScore
    D = opp(("draw", 1.0, 0.5, leaf(0.5)), key="D", static=0.5)
    g2 = build(me(("toD", 1.0, 0.5, D), key="root", static=0.5))
    eng = Engine(SynthEvaluator(g2), settings(prior_min=0.0, draw_score=0.2))
    eng.search(depth=2)
    assert eng.bm.completed[2].root_values["toD"] == pytest.approx(0.2)


def test_a_new_root_recomputes_completed_depths_from_the_graph():
    g = random_game(np.random.default_rng(3), depth=7, branching=(2, 3))
    ev = SynthEvaluator(g)
    eng = Engine(ev, settings())
    eng.search(depth=4)
    root = g.nodes[g.root]
    m = root.moves[0]
    calls = sum(ev.calls.values())
    eng.set_position([m])
    r = eng.search(depth=2)                              # the child's shallow depths are known already
    assert r.complete and sum(ev.calls.values()) - calls < 10


def test_uci_bestmove_mode():
    eng = chess.engine.SimpleEngine.popen_uci([sys.executable, "-m", "tnh.engine.uci", "--fake"])
    try:
        eng.configure({"Mode": "bestmove", "Side": "white", "Alpha": "1"})
        b = chess.Board()
        for x in "c2c4 g8f6 g2g3 e7e6 f1g2 d7d5".split():
            b.push_uci(x)
        strings = []
        with eng.analysis(b, chess.engine.Limit(nodes=150), multipv=3) as an:
            for info in an:
                if "string" in info:
                    strings.append(info["string"])
        lines = [s for s in strings if s.startswith("preppy multipv=")]
        assert lines and "mode=bestmove" in lines[-1] and "complete=" in lines[-1] and "inside=" in lines[-1]
        res = eng.play(b, chess.engine.Limit(nodes=50))
        assert res.move in b.legal_moves
    finally:
        eng.quit()


def test_valued_positions_get_children_only_inside_and_values_do_not_care():
    from tnh.engine.chess_eval import ChessEvaluator, FakeBundle
    C4 = "c2c4 g8f6 g2g3 e7e6 f1g2 d7d5".split()
    eng = Engine(ChessEvaluator({"fake": FakeBundle()}), Settings(tempo="fake", my_side="white", mode="bestmove"))
    eng.set_position(C4)
    eng.search(depth=5)
    g = eng.graph
    valued_only = [n.id for n in g.nodes if n.info.get("valued") and n.children is None]
    assert valued_only                                    # evaluated for their value, no children made
    assert len(g.nodes) < 30 * sum(1 for n in g.nodes if n.children is not None) + 100
    before = {k: (d.value, dict(d.root_values)) for k, d in eng.bm.completed.items()}
    eng._materialize(valued_only)                         # give them children anyway
    eng.bm.completed = {}
    eng.bm._key = None
    eng.search(depth=5)
    for k, (v, rv) in before.items():
        assert eng.bm.completed[k].value == v and eng.bm.completed[k].root_values == rv


def test_info_during_a_depth_in_progress():
    from tnh.engine.chess_eval import ChessEvaluator, FakeBundle
    C4 = "c2c4 g8f6 g2g3 e7e6 f1g2 d7d5".split()
    eng = Engine(ChessEvaluator({"fake": FakeBundle()}), Settings(tempo="fake", my_side="white", mode="bestmove",
                                                                  batch=4))
    eng.set_position(C4)
    seen = []
    eng.search(depth=5, on_info=seen.append, info_every=0.0)   # an info line after every batch
    assert any(not r.complete or r.depth_in_progress > r.depth for r in seen)
    first = next(i for i, r in enumerate(seen) if r.lines)   # none before the root has children
    assert all(r.lines for r in seen[first:])
    final = {k: d.value for k, d in eng.bm.completed.items()}
    eng2 = Engine(ChessEvaluator({"fake": FakeBundle()}), Settings(tempo="fake", my_side="white", mode="bestmove"))
    eng2.set_position(C4)
    eng2.search(depth=5)
    assert final == {k: d.value for k, d in eng2.bm.completed.items()}
