"""The search on synthetic worlds: planted repertoires, the exact solve, caches and settings."""

import itertools
from dataclasses import replace

import numpy as np
import pytest

from tnh.builder.game import Game, ObviousCost, Params
from tnh.builder.search import candidates, solve
from tnh.builder.synth import build, leaf, me, opp, random_game
from tnh.engine.pricing import PL, solve_cap, solve_price
from tnh.engine.search import Engine, Settings
from tnh.engine.synth import SynthEvaluator

INF = float("inf")
BROAD = dict(my_side="white", kappa=INF, prior_min=0.0, blunder=1.0, batch=4, solve_every=10,
             c_puct=4.0, fpu=0.0, explore_uniform=0.3)


# -- the price solve ------------------------------------------------------------------------

def brute_net(game, mu):
    cand = candidates(game)
    ks = list(cand)
    best = -INF
    for ch in itertools.product(*([None] + cand[k] for k in ks)):
        L = {k: m for k, m in zip(ks, ch) if m is not None}
        best = max(best, game.evaluate(L) - mu * game.total_cost(L))
    return best


def test_pl_functions():
    f = PL.line(0.6).max(PL.line(0.8).shifted(0.05))
    assert f(0.0) == pytest.approx(0.0) and f(1.0) == pytest.approx(0.75)
    assert f(0.1) == pytest.approx(0.06)                  # learning not worth it at low reach
    assert f.scaled(0.5)(1.0) == pytest.approx(f(0.5))
    g = PL.add([f, PL.line(0.2)])
    assert g(0.7) == pytest.approx(f(0.7) + 0.14)


@pytest.mark.parametrize("seed", range(30))
def test_price_solve_is_exact_on_trees(seed):
    rng = np.random.default_rng(seed)
    g = random_game(rng, depth=int(rng.integers(3, 5)), branching=(2, 3), me_first=bool(seed % 2))
    params = Params(eps_white=0.1, cost=ObviousCost(0.5) if seed % 3 == 0 else Params().cost,
                    lam_node=0.1 if seed % 4 == 0 else 0.0)
    game = Game(g, params)
    while np.prod([1 + len(v) for v in candidates(game).values()]) > 20000:
        g = random_game(rng, depth=3, branching=(2, 3), me_first=bool(seed % 2))
        game = Game(g, params)
    for mu in (0.0, 0.002, 0.01, 0.05):
        s = solve_price(game, mu, ascent=False)
        assert s.net == pytest.approx(brute_net(game, mu), abs=1e-9)
        assert solve_price(game, mu).steps == []          # nothing for the ascent on a tree


def test_price_meets_the_budget_curve_on_its_hull():
    for seed in range(10):
        g = random_game(np.random.default_rng(100 + seed), depth=4, branching=(2, 3))
        game = Game(g, Params(eps_white=0.1))
        _, sols = solve(game, 6)
        for mu in (0.002, 0.01, 0.05):
            want = max(s.value - mu * s.cost for s in sols)
            assert solve_price(game, mu).net >= want - 1e-9


def test_cap_stays_within_its_budget():
    for seed in range(8):
        g = random_game(np.random.default_rng(200 + seed), depth=4, branching=(2, 3))
        game = Game(g, Params(eps_white=0.1))
        _, sols = solve(game, 4)
        for N in (1, 2, 3):
            s = solve_cap(game, N)
            assert s.cost <= N
            assert s.value <= sols[N].value + 1e-9            # never above the exact budget optimum


def test_dag_price_is_feasible_and_never_worse_than_its_first_decision():
    from tnh.engine.pricing import decide, net_value, subtree_functions
    for seed in range(15):
        g = random_game(np.random.default_rng(300 + seed), depth=5, branching=(2, 3), dag=True, pool=4)
        game = Game(g, Params(eps_white=0.1))
        mu = 0.005
        W, opts = subtree_functions(game, mu)
        first = decide(game, W, opts)
        s = solve_price(game, mu)
        assert s.net >= net_value(game, first, mu)[0] - 1e-12
        assert s.net == pytest.approx(net_value(game, s.learned, mu)[0])
        assert s.net <= brute_net(game, mu) + 1e-12 if np.prod([1 + len(v) for v in candidates(game).values()]) < 5000 else True


# -- the search -------------------------------------------------------------------------------

def engine_for(g, **kw):
    return Engine(SynthEvaluator(g), Settings(**{**BROAD, **kw}))


def exact_on_own_graph(eng):
    """The engine's values are the exact solve of its own graph, visits play no part."""
    game = Game(eng.graph, eng.params())
    want = solve_price(game, eng.last.mu)
    assert eng.last.net == pytest.approx(want.net, abs=1e-12)
    V = game.values(eng.last.learned)
    for i in game.order:
        assert eng.V[i] == pytest.approx(V[i], abs=1e-12)


@pytest.mark.parametrize("seed", range(12))
def test_search_values_are_exact_and_match_the_full_tree_once_expanded(seed):
    g = random_game(np.random.default_rng(seed), depth=4, branching=(2, 3), me_first=True)
    eng = engine_for(g, price=0.003)
    r = eng.search(nodes=10_000)
    exact_on_own_graph(eng)
    # tampering with the visit counts changes nothing in the values
    for i in list(eng.visits):
        eng.visits[i] = 7 * eng.visits[i] + 3
    before = eng.last.net
    eng.solve()
    assert eng.last.net == before
    if eng._fully_expanded():
        full = solve_price(Game(g, Params(eps_white=1.0, cost=ObviousCost(0.85), kappa=INF)), 0.003)
        assert r.solve.net == pytest.approx(full.net, abs=1e-9)
        assert r.solve.learned == full.learned


def deep_winner_world():
    D = me(("win", 0.01, 0.9, leaf(0.9)), ("usual", 0.99, 0.5, leaf(0.5)), key="D")
    E = me(("good", 0.5, 0.6, leaf(0.6)), ("meh", 0.5, 0.55, leaf(0.5)), key="E")
    M2 = me(("c", 0.95, 0.5, opp(("z", 0.9, 0.5, D), ("z2", 0.1, 0.5, leaf(0.5)), key="M2o")),
            ("d", 0.05, 0.48, leaf(0.45)), key="M2")
    return build(me(("a", 0.9, 0.5, opp(("x", 0.9, 0.5, M2), ("y", 0.1, 0.5, E), key="Ao")),
                    ("b", 0.1, 0.5, leaf(0.5)), key="root"))


def test_planted_deep_winner_is_found_with_default_exploration():
    eng = Engine(SynthEvaluator(deep_winner_world()), Settings(my_side="white", kappa=INF, price=0.01))
    r = eng.search(nodes=200)
    assert r.solve.learned.get("D") == "win"
    assert r.solve.cost >= 1
    # at a high price nothing deep is worth a move
    eng.set_settings(price=0.5)
    assert "D" not in eng.solve().learned


def test_planted_zero_cost_chain():
    D = me(("win", 0.02, 0.9, leaf(0.9)), ("usual", 0.98, 0.5, leaf(0.5)), key="D")
    node = D
    for i in reversed(range(3)):
        node = opp(("only", 1.0, 0.5, node), key=f"o{i}")
        node = me(("good", 0.9, 0.5, node), ("bad", 0.1, 0.45, leaf(0.4)), key=f"C{i}")
    eng = Engine(SynthEvaluator(build(node)), Settings(my_side="white", kappa=INF, price=0.05))
    r = eng.search(nodes=100)
    assert r.solve.learned == {"C0": "good", "C1": "good", "C2": "good", "D": "win"}
    assert r.solve.cost == 1                                 # the chain is obvious, so free
    eng.set_settings(price=0.6)                              # D gains 0.4 at reach 1: not at this price
    assert eng.solve().learned == {"C0": "good", "C1": "good", "C2": "good"}


def test_cap_and_price_on_a_planted_world():
    eng = engine_for(deep_winner_world(), price=0.0)
    eng.search(nodes=500)
    eng.set_settings(cap=1)
    s = eng.solve()
    assert s.cost <= 1 and s.learned.get("D") == "win"
    eng.set_settings(cap=0, price=0.0)
    assert eng.solve().cost >= s.cost


def test_prepared_share_drops_the_line_that_lives_on_one_model_error():
    O = opp(("blunder", 0.9, 0.7, leaf(0.7)), ("best", 0.1, 0.45, leaf(0.45)), key="O")
    O2 = opp(("r1", 0.8, 0.5, leaf(0.55)), ("r2", 0.2, 0.6, leaf(0.6)), key="O2")
    g = build(me(("P", 0.5, 0.5, O), ("Q", 0.5, 0.5, O2), key="root"))
    eng = engine_for(g, price=0.001)
    eng.search(nodes=50)
    assert eng.solve().learned["root"] == "P"
    eng.set_settings(prepared=0.7)
    assert eng.solve().learned["root"] == "Q"


def test_transpositions_are_one_position():
    g = random_game(np.random.default_rng(5), depth=5, branching=(2, 3), dag=True, pool=3)
    eng = engine_for(g, price=0.002)
    eng.search(nodes=2000)
    assert eng.stats["transpositions merged"] > 0
    keys = [n.key for n in eng.graph.nodes]
    assert len(keys) == len({(n.key, n.ply) for n in eng.graph.nodes})
    exact_on_own_graph(eng)


def test_reuse_after_position_moves():
    g = deep_winner_world()
    ev = SynthEvaluator(g)
    eng = Engine(ev, Settings(**{**BROAD, "price": 0.01}))
    eng.search(nodes=100)
    calls = sum(ev.calls.values())
    eng.set_position(["a", "x"])                     # into M2, already expanded
    assert eng.graph.nodes[eng.graph.root].key == "M2"
    eng.search(nodes=0)
    assert sum(ev.calls.values()) == calls           # nothing evaluated again
    assert eng.solve().learned.get("D") == "win"
    eng.set_position([])
    eng.search(nodes=0)
    assert sum(ev.calls.values()) == calls


def test_settings_invalidate_exactly_what_they_should():
    g = random_game(np.random.default_rng(9), depth=4, branching=(2, 3))
    for n in g.nodes:
        if n.moves and not n.me_to_move:
            n.counts = np.arange(1, len(n.moves) + 1, dtype=float)
    ev = SynthEvaluator(g)
    eng = Engine(ev, Settings(**{**BROAD, "kappa": 10.0, "price": 0.002}))
    eng.search(nodes=60)
    calls, counts = sum(ev.calls.values()), sum(ev.count_calls.values())
    expanded = eng.stats["expanded"]
    expanded_at_2200 = {i for i in eng.graph.reachable() if eng.graph.nodes[i].children is not None}
    for change in ({"price": 0.01}, {"kappa": 50.0}, {"obvious": 0.9}, {"prepared": 0.2}, {"cap": 2}, {"lines": 5}):
        info = eng.set_settings(**change)
        assert not info["evaluations_keyed_anew"] and info["positions_back_to_frontier"] == 0
        eng.solve()
        assert sum(ev.calls.values()) == calls and sum(ev.count_calls.values()) == counts
    eng.set_settings(cap=0)
    eng.set_settings(db_filter="blitz")              # counts again, no evaluations
    eng.solve()
    assert sum(ev.calls.values()) == calls and sum(ev.count_calls.values()) > counts
    info = eng.set_settings(my_elo=1800)             # every evaluation under a new key
    assert info["evaluations_keyed_anew"] and info["positions_back_to_frontier"] > 0
    eng.search(nodes=expanded)
    assert sum(ev.calls.values()) > calls
    c2 = sum(ev.calls.values())
    eng.set_settings(my_elo=2200)                    # the old ones are still in the cache
    now = {i for i in eng.graph.reachable() if eng.graph.nodes[i].children is not None}
    assert expanded_at_2200 <= now
    eng.solve()
    assert sum(ev.calls.values()) == c2


def test_deterministic():
    def run():
        g = random_game(np.random.default_rng(3), depth=5, branching=(2, 3))
        eng = Engine(SynthEvaluator(g), Settings(my_side="white", kappa=INF, price=0.002, batch=6))
        r = eng.search(nodes=80)
        return r.solve.learned, r.solve.net, [l.moves for l in r.lines], dict(r.stats)
    a, b = run(), run()
    assert a == b


def test_result_reports_masses_and_history():
    g = random_game(np.random.default_rng(4), depth=6, branching=(2, 3))
    eng = Engine(SynthEvaluator(g), Settings(my_side="white", kappa=INF, price=0.002, solve_every=10, solve_ratio=0.0))
    r = eng.search(nodes=40)
    m = r.mass
    assert m["stand_in"] > 0 and sum(m.values()) == pytest.approx(1.0)
    assert len(r.history) >= 3 and r.history[-1][1] == pytest.approx(r.solve.value)
    assert r.lines and r.lines[0].moves


def test_a_position_at_two_plies_one_unexpanded():
    # X occurs at ply 2 (expanded) and ply 4 (a leaf): costs and learning read the expanded one
    X2 = me(("win", 0.02, 0.9, leaf(0.9)), ("usual", 0.98, 0.5, leaf(0.5)), key="X")
    deep = opp(("t", 1.0, 0.5, me(("u", 1.0, 0.5, opp(("v", 1.0, 0.5, leaf(0.6, key="X")), key="o3")), key="m2")), key="o1")
    g = build(me(("a", 0.5, 0.5, opp(("x", 1.0, 0.5, X2), key="oa")), ("b", 0.5, 0.5, deep), key="root"))
    game = Game(g, Params(eps_white=1.0, cost=ObviousCost(0.85)))
    assert len(game.by_key["X"]) == 2
    assert game.cost_of("X", "win") == 1
    s = solve_price(game, 0.001)
    assert s.learned.get("X") == "win"
