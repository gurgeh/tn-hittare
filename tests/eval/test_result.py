"""The result model's multinomial logit, and the look-ahead's arithmetic with a fake evaluator."""

import chess
import numpy as np

from tnh.eval.lookahead import Node, evaluate_nodes, exact, monte_carlo, terminal, to_root
from tnh.eval.result import fit_mlogit, losses, mlogit_probs, source_logits


def test_mlogit_recovers_planted_coefficients():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20000, 3))
    theta = np.array([0.3, 1.0, -0.5, 0.2, -0.4, 0.0, 0.7, -0.3])
    p = mlogit_probs(theta, X)
    y = np.array([rng.choice(3, p=r) for r in p])
    got = fit_mlogit(X, y)
    assert np.allclose(got, theta, atol=0.08), got
    ll, br = losses(mlogit_probs(got, X), y)
    assert ll.mean() <= losses(p, y)[0].mean() + 1e-9


def test_source_logits_floor_and_symmetry():
    w = np.array([[0.5, 0.3, 0.2], [0.0, 1.0, 0.0]])
    z = source_logits(w)
    assert np.isclose(z[0, 0], np.log(0.5 / 0.2))
    assert np.isfinite(z).all() and z[1, 0] == 0


class FakeEv:
    """Moves: the legal moves, uniform. Leaf for the side to move: depends on the material balance."""

    def evaluate(self, queries):
        moves, probs, leaf = [], [], []
        for q in queries:
            b = chess.Board(q.fen)
            ms = [m.uci() for m in b.legal_moves]
            moves.append(ms)
            probs.append(np.full(len(ms), 1.0 / len(ms)))
            mat = sum(len(b.pieces(t, b.turn)) - len(b.pieces(t, not b.turn)) for t in range(1, 6))
            win = 1 / (1 + np.exp(-mat))
            leaf.append(np.array([0.8 * win, 0.2, 0.8 * (1 - win)]))
        return moves, probs, np.array(leaf)


def test_exact_one_ply_is_the_weighted_child_average():
    ev = FakeEv()
    b = chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")    # White can take the queen
    root = Node(b, [], 2200, 2100, 0)
    re = evaluate_nodes(ev, [root])
    stats = {k: 0.0 for k in ("exact1_seconds", "exact2_seconds", "exact1_evals", "exact2_evals")}
    v1, v2 = exact(ev, [root], re, 0.0, stats)
    kids = [root.child(m.uci()) for m in b.legal_moves]
    manual = np.mean([to_root(e[2], 1) for e in evaluate_nodes(ev, kids)], axis=0)
    assert np.allclose(v1[0], manual)
    assert np.isclose(v1[0].sum(), 1) and np.isclose(v2[0].sum(), 1)
    assert stats["exact1_evals"] == len(kids)


def test_monte_carlo_values_are_from_the_roots_side_and_terminal_lines_keep_their_result():
    ev = FakeEv()
    # White to move can mate at once (Qg7#); the lines that do must keep the win at every later depth
    b = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    root = Node(b, [], 2200, 2200, 0)
    re = evaluate_nodes(ev, [root])
    stats = {k: 0.0 for k in ("mc_seconds", "mc_evals")}
    vals = monte_carlo(ev, [root], re, 16, 4, np.random.default_rng(0), stats)
    assert vals.shape == (1, 16, 4, 3)
    assert np.allclose(vals[:, :, :2].sum(-1), 1)          # depths 2 and 4; 8 and 16 lie beyond --depth 4
    assert np.isnan(vals[:, :, 2:]).all()
    # lines that mate keep a win for the root (index 0) at every later depth
    mated = [l for l in range(16) if np.allclose(vals[0, l, 0], [1, 0, 0])]
    for l in mated:
        assert np.allclose(vals[0, l, 1], [1, 0, 0])


def test_terminal_results():
    assert np.allclose(terminal(chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")), [0, 0, 1])
    assert np.allclose(terminal(chess.Board("7k/8/6QK/8/8/8/8/8 b - - 0 1")), [0, 1, 0])
    assert terminal(chess.Board()) is None
