"""The DP against enumeration, and the evaluator against simulation."""

import zlib

import numpy as np
import pytest

from tnh.builder.game import Game, Params
from tnh.builder.search import DP, brute_force, candidates, is_tree, maxplus, solve
from tnh.builder.synth import random_game


def test_maxplus_against_definition():
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = np.sort(rng.normal(size=rng.integers(1, 6)))
        b = np.sort(rng.normal(size=rng.integers(1, 6)))
        K = int(rng.integers(0, 10))
        out = maxplus(a, b, K)
        for k in range(len(out)):
            want = max(a[min(i, len(a) - 1)] + b[min(k - i, len(b) - 1)] for i in range(k + 1))
            assert out[k] == pytest.approx(want)


def random_counts(g, rng):
    for n in g.nodes:
        if n.moves and not n.me_to_move and rng.random() < 0.5:
            n.counts = rng.integers(0, 50, len(n.moves)).astype(float)


def random_cost(salt):
    return lambda n: 2 if zlib.crc32(f"{salt}{n.key}".encode()) % 3 == 0 else 1


@pytest.mark.parametrize("seed", range(60))
def test_dp_is_exact_on_trees(seed):
    rng = np.random.default_rng(seed)
    g = random_game(rng, depth=int(rng.integers(3, 6)), branching=(2, 3), me_first=bool(seed % 2))
    while len([n for n in g.nodes if n.me_to_move and n.moves]) > 14:   # keep the enumeration small
        g = random_game(rng, depth=int(rng.integers(3, 6)), branching=(2, 3), me_first=bool(seed % 2))
    random_counts(g, rng)
    params = Params(kappa=float(rng.choice([1.0, 20.0, np.inf])), lam=float(rng.choice([0.0, 0.2])),
                    lam_node=float(rng.choice([0.0, 0.1])),
                    eps_white=float(rng.choice([0.03, 0.1, 1.0])),
                    cost=random_cost(seed) if seed % 2 == 0 else Params().cost)
    game = Game(g, params)
    assert is_tree(game)
    n_cand = len(candidates(game))
    K = min(3, n_cand)
    _, sols = solve(game, K)
    for N in range(K + 1):
        bf, _ = brute_force(game, N)
        assert sols[N].dp_value == pytest.approx(bf, abs=1e-12), (N, sols[N].learned)
        assert sols[N].value == pytest.approx(bf, abs=1e-12)
        # local search finds nothing to improve on a tree
        assert not [x for x in sols[N].local_steps if x.split()[0] in ("add", "switch", "swap")]


@pytest.mark.parametrize("seed", range(10))
def test_dp_with_pruning_is_exact_over_the_positions_it_may_use(seed):
    rng = np.random.default_rng(100 + seed)
    g = random_game(rng, depth=5, branching=(2, 3))
    game = Game(g, Params(eps_white=0.1, min_reach=0.05))
    dp = DP(game, 3)
    allowed = [g.nodes[i].key for i in dp.active if game.learnable(i)]
    _, sols = solve(game, 3, local_search=False)
    for N in range(4):
        assert set(sols[N].learned) <= set(allowed)
        assert sols[N].value == pytest.approx(brute_force(game, N, keys=allowed)[0], abs=1e-12)


def test_dag_solutions_are_feasible_and_near_optimal():
    """On DAGs the DP is not exact. The repaired solution must be feasible (never above
    the enumerated optimum, within budget) and, over many seeds, close to it."""
    shares, raw_shares = [], []
    for seed in range(40):
        rng = np.random.default_rng(1000 + seed)
        g = random_game(rng, depth=5, branching=(2, 3), dag=True, pool=int(rng.integers(3, 6)))
        game = Game(g, Params(eps_white=0.05))
        e0 = game.evaluate({})
        _, sols = solve(game, 3)
        _, raw = solve(game, 3, local_search=False)
        for N in (1, 2, 3):
            bf, _ = brute_force(game, N)
            assert sols[N].value <= bf + 1e-12
            assert sols[N].cost <= N
            if bf - e0 > 1e-9:
                shares.append((sols[N].value - e0) / (bf - e0))
                raw_shares.append((raw[N].value - e0) / (bf - e0))
    assert np.mean(shares) > 0.99
    assert np.mean(raw_shares) > 0.85
    assert min(shares) > 0.9


def test_evaluate_matches_simulation_on_a_dag():
    rng = np.random.default_rng(7)
    g = random_game(rng, depth=6, branching=(2, 3), dag=True, pool=4)
    for n in g.nodes:
        if n.moves and not n.me_to_move:
            n.counts = rng.integers(0, 30, len(n.moves)).astype(float)
    game = Game(g, Params(kappa=10, lam_node=0.3, eps_white=0.1))
    cand = candidates(game)
    learned = {k: v[0] for k, v in list(cand.items())[:3]}
    exact = game.evaluate(learned)
    reach = game.reach(learned)
    sims, hits = 4000, np.zeros(len(g.nodes))
    total = 0.0
    for _ in range(sims):
        i = g.root
        while True:
            hits[i] += 1
            n = g.nodes[i]
            if not n.expanded:
                total += n.value
                break
            p = game.policy(i, learned)
            i = n.children[rng.choice(len(p), p=p)]
    est = total / sims
    assert abs(est - exact) < 4 * 0.5 / np.sqrt(sims)
    assert np.abs(hits / sims - reach).max() < 0.04


def test_marginals_are_exact_without_repetitions():
    rng = np.random.default_rng(3)
    g = random_game(rng, depth=5, branching=(2, 3), dag=True, pool=4)
    game = Game(g, Params(eps_white=0.1))
    cand = candidates(game)
    learned = {k: v[0] for k, v in list(cand.items())[:2]}
    v0 = game.evaluate(learned)
    for (k, m), gain in game.marginals(learned).items():
        trial = {kk: mm for kk, mm in learned.items() if kk != k} if m is None else {**learned, k: m}
        assert game.evaluate(trial) - v0 == pytest.approx(gain, abs=1e-12)


def test_curve_is_non_decreasing():
    rng = np.random.default_rng(11)
    g = random_game(rng, depth=6, branching=(2, 4))
    _, sols = solve(Game(g, Params(eps_white=0.05)), 12)
    vals = [s.value for s in sols]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))


def test_unsound_learned_move_is_refused():
    rng = np.random.default_rng(5)
    g = random_game(rng, depth=3, branching=(3, 3))
    game = Game(g, Params(eps_white=0.0))
    root = g.nodes[g.root]
    s = set(game.sound[root.id].tolist())
    bad = next(m for j, m in enumerate(root.moves) if j not in s)
    with pytest.raises(ValueError, match="not sound"):
        game.evaluate({root.key: bad})


@pytest.mark.parametrize("seed", range(25))
def test_per_opponent_pessimism_dp_is_exact_on_trees(seed):
    """The DP with per-node posteriors against enumeration under the obvious definition:
    (1 - lam) x value against the model + lam x value against the engine."""
    from tnh.builder.game import mixture_value
    rng = np.random.default_rng(500 + seed)
    g = random_game(rng, depth=int(rng.integers(3, 5)), branching=(2, 3), me_first=bool(seed % 2))
    lam = float(rng.choice([0.1, 0.3, 0.6]))
    params = Params(lam=lam, lam_node=float(rng.choice([0.0, 0.2])), eps_white=0.1)
    game = Game(g, params)
    _, sols = solve(game, 3)
    for N in range(4):
        bf, best = brute_force(game, N)
        assert sols[N].dp_value == pytest.approx(bf, abs=1e-12)
        assert mixture_value(g, params, sols[N].learned) == pytest.approx(bf, abs=1e-12)
        assert mixture_value(g, params, best) == pytest.approx(bf, abs=1e-12)


def test_per_opponent_on_dags_is_feasible_and_near_optimal():
    """On a DAG the posteriors are approximate in the DP, the reported value is exact."""
    from tnh.builder.game import mixture_value
    shares = []
    for seed in range(30):
        rng = np.random.default_rng(2000 + seed)
        g = random_game(rng, depth=5, branching=(2, 3), dag=True, pool=int(rng.integers(3, 6)))
        params = Params(lam=0.3, eps_white=0.05)
        game = Game(g, params)
        e0 = game.evaluate({})
        _, sols = solve(game, 3)
        for N in (1, 2, 3):
            bf, _ = brute_force(game, N)
            assert sols[N].value == pytest.approx(mixture_value(g, params, sols[N].learned), abs=1e-12)
            assert sols[N].value <= bf + 1e-12
            if bf - e0 > 1e-9:
                shares.append((sols[N].value - e0) / (bf - e0))
    assert np.mean(shares) > 0.99 and min(shares) > 0.75   # measured 0.997 and 0.797
