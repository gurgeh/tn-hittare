"""Synthetic games where the best repertoire is known by construction."""

import numpy as np
import pytest

from tnh.builder.game import Game, Params, blend
from tnh.builder.search import brute_force, solve
from tnh.builder.synth import build, leaf, me, opp

SOUND = Params(eps_white=0.1)


def learned_at(game, N, K=None):
    _, sols = solve(game, K if K is not None else N)
    return sols[N].learned


def test_deep_often_reached_winning_move_is_chosen_first():
    # D lies four plies deep, is reached with probability .9*.9*.95*.9 = .69, and has a
    # winning move my model plays 1 % of the time. Everything else gains a little.
    D = me(("win", 0.01, 0.9, leaf(0.9)), ("usual", 0.99, 0.5, leaf(0.5)), key="D")
    E = me(("good", 0.5, 0.6, leaf(0.6)), ("meh", 0.5, 0.55, leaf(0.5)), key="E")
    M2 = me(("c", 0.95, 0.5, opp(("z", 0.9, 0.5, D), ("z2", 0.1, 0.5, leaf(0.5)))),
            ("d", 0.05, 0.48, leaf(0.45)), key="M2")
    root = me(("a", 0.9, 0.5, opp(("x", 0.9, 0.5, M2), ("y", 0.1, 0.5, E))),
              ("b", 0.1, 0.5, leaf(0.5)), key="root")
    game = Game(build(root), SOUND)
    assert learned_at(game, 1, K=4) == {"D": "win"}
    # and it stays in every larger repertoire
    _, sols = solve(game, 4)
    assert all(s.learned.get("D") == "win" for s in sols[1:])


@pytest.mark.parametrize("p_best", [0.999, 1.0])
def test_a_move_I_find_anyway_is_not_learned(p_best):
    # A has a crushing move (1.0 against 0.0) that my model already plays with p_best.
    A = me(("crush", p_best, 1.0, leaf(1.0)), ("blunder", 1 - p_best, 0.0, leaf(0.0)), key="A")
    Bs = [me(("good", 0.5, 0.8, leaf(0.8)), ("bad", 0.5, 0.75, leaf(0.5)), key=f"B{i}") for i in range(4)]
    root = opp(*[(f"r{i}", 0.2, 0.5, B) for i, B in enumerate(Bs)], ("r4", 0.2, 0.5, A), key="root")
    game = Game(build(root, me_white=True), Params(eps_white=0.1))
    _, sols = solve(game, 8)
    for s in sols[1:5]:
        assert "A" not in s.learned
    if p_best == 1.0:
        assert all("A" not in s.learned for s in sols)
        assert sols[8].cost == 4     # nothing left worth learning, so nothing more is spent


def line(depth, branching, tag):
    """My position: learning it gains the same in every copy. The good move leads to an
    opponent's position with `branching` equally likely replies, each to the same
    structure one level down; the bad move to a fixed leaf."""
    if depth == 0:
        return leaf(0.6)
    replies = [(f"r{j}", 1 / branching, 0.5, line(depth - 1, branching, f"{tag}{j}")) for j in range(branching)]
    return me(("good", 0.5, 0.5, opp(*replies, key=f"{tag}o")), ("bad", 0.5, 0.45, leaf(0.4)), key=tag)


def test_forcing_line_before_bushy_line():
    # A and B lead to the same structure; after A the opponent has one reply each time,
    # after B four. Unlearned both are worth 0.45, and the first position of each gains
    # the same. Deeper, a forcing position is reached four times as often.
    root = me(("A", 0.5, 0.5, opp(("only", 1.0, 0.5, line(2, 1, "X")), key="Ao")),
              ("B", 0.5, 0.5, opp(("only", 1.0, 0.5, line(2, 4, "Y")), key="Bo")), key="root")
    game = Game(build(root), SOUND)
    E = game.values({})
    assert E[game.by_key["Ao"][0]] == pytest.approx(E[game.by_key["Bo"][0]])
    _, sols = solve(game, 3)
    assert sols[2].learned == {"X": "good", "X0": "good"}
    assert sols[3].learned == {"root": "A", "X": "good", "X0": "good"}


def test_unsound_move_is_excluded_by_eps():
    # U scores 0.875 against the model but the engine says 0.2 where the best is 0.5.
    trap = opp(("falls", 0.9, 0.2, leaf(0.95)), ("refutes", 0.1, 0.2, leaf(0.2)), key="T")
    root = me(("U", 0.3, 0.2, trap), ("S", 0.7, 0.5, leaf(0.5)), key="root")
    g = build(root)
    sound = Game(g, Params(eps_white=0.05))
    assert all("root" not in s.learned for s in solve(sound, 2)[1])
    anything = Game(g, Params(eps_white=1.0))
    assert learned_at(anything, 1, K=2) == {"root": "U"}


def model_error_world():
    # O: the opponent model says he blunders 90 % of the time; the engine's best reply
    # is the other. O2: an honest position where the model already favours the best reply.
    O = opp(("blunder", 0.9, 0.7, leaf(0.7)), ("best", 0.1, 0.45, leaf(0.45)), key="O")
    O2 = opp(("r1", 0.8, 0.5, leaf(0.55)), ("r2", 0.2, 0.6, leaf(0.6)), key="O2")
    return build(me(("P", 0.5, 0.5, O), ("Q", 0.5, 0.5, O2), key="root"))


def test_pessimism_drops_the_line_that_lives_on_one_model_error():
    g = model_error_world()
    choice = {}
    for lam in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9):
        choice[lam] = learned_at(Game(g, Params(eps_white=0.1, lam=lam)), 1)["root"]
    assert choice[0.0] == "P" and choice[0.1] == "P"
    assert choice[0.7] == "Q" and choice[0.9] == "Q"
    # once dropped it stays dropped
    seq = [choice[l] for l in sorted(choice)]
    assert seq == sorted(seq)        # P... then Q...


def test_pessimism_lowers_the_predicted_score():
    g = model_error_world()
    vals = [solve(Game(g, Params(eps_white=0.1, lam=lam)), 1)[1][1].value for lam in (0, 0.25, 0.5, 1.0)]
    assert all(a > b for a, b in zip(vals, vals[1:]))
    assert vals[-1] == pytest.approx(0.55)      # an engine opponent: Q, then r1


def test_counts_win_where_common_model_where_rare():
    common = opp(("a", 0.1, 0.5, leaf(0.5)), ("b", 0.9, 0.5, leaf(0.5)), counts={"a": 900, "b": 100}, key="C")
    rare = opp(("a", 0.1, 0.5, leaf(0.5)), ("b", 0.9, 0.5, leaf(0.5)), counts={}, key="R")
    g = build(me(("toC", 0.5, 0.5, common), ("toR", 0.5, 0.5, rare), key="root"))
    game = Game(g, Params(kappa=50, eps_white=0.1))
    c = game.dist[g.get("C", 1).id]
    r = game.dist[g.get("R", 1).id]
    assert c == pytest.approx([(900 + 5) / 1050, (100 + 45) / 1050])
    assert r == pytest.approx([0.1, 0.9])
    assert blend(np.array([900, 100]), np.array([0.1, 0.9]), np.inf) == pytest.approx([0.1, 0.9])


def test_counts_decide_where_the_repertoire_goes():
    # After the common reply (by the counts) I have something to learn in A1, after the
    # reply the model prefers I have the same in B1. The search follows the counts.
    A1 = me(("good", 0.2, 0.7, leaf(0.7)), ("bad", 0.8, 0.65, leaf(0.5)), key="A1")
    B1 = me(("good", 0.2, 0.7, leaf(0.7)), ("bad", 0.8, 0.65, leaf(0.5)), key="B1")
    root = opp(("a", 0.1, 0.5, A1), ("b", 0.9, 0.5, B1), counts={"a": 900, "b": 100}, key="root")
    g = build(root)
    assert learned_at(Game(g, Params(kappa=50, eps_white=0.1)), 1) == {"A1": "good"}
    assert learned_at(Game(g, Params(kappa=np.inf, eps_white=0.1)), 1) == {"B1": "good"}


def test_cost_hook():
    # B is worth more than A but costs two units.
    A = me(("good", 0.5, 0.6, leaf(0.6)), ("bad", 0.5, 0.55, leaf(0.5)), key="A")
    B = me(("good", 0.1, 0.9, leaf(0.9)), ("bad", 0.9, 0.85, leaf(0.5)), key="B")
    g = build(opp(("a", 0.5, 0.5, A), ("b", 0.5, 0.5, B), key="root"))
    cost = lambda n: 2 if n.key == "B" else 1
    game = Game(g, Params(eps_white=0.1, cost=cost))
    _, sols = solve(game, 3)
    assert sols[1].learned == {"A": "good"}
    assert sols[2].learned == {"B": "good"}
    assert sols[3].learned == {"A": "good", "B": "good"}
    for N in range(4):
        assert sols[N].value == pytest.approx(brute_force(game, N)[0])


def test_learning_an_ancestor_to_steer_into_a_good_position():
    # Greedy one-at-a-time fails here: learning the root alone is worth nothing, D alone
    # little, both together a lot. The DP sees the pair.
    D = me(("win", 0.02, 0.9, leaf(0.9)), ("usual", 0.98, 0.5, leaf(0.5)), key="D")
    E = me(("fine", 0.5, 0.55, leaf(0.55)), ("ok", 0.5, 0.5, leaf(0.5)), key="E")
    root = me(("toD", 0.05, 0.5, opp(("x", 1.0, 0.5, D), key="Do")),
              ("toE", 0.95, 0.5, opp(("y", 1.0, 0.5, E), key="Eo")), key="root")
    game = Game(build(root), SOUND)
    _, sols = solve(game, 2)
    assert sols[2].learned == {"root": "toD", "D": "win"}
    assert sols[1].learned == {"E": "fine"}


def test_per_node_pessimism_is_weaker_than_per_opponent_on_long_lines():
    # Five only-moves in a row for the opponent, each found half the time by the model.
    # A miss gives me 1.0, finding them all 0.5.
    node = me(("end", 1.0, 0.5, leaf(0.5)), key="end")
    for i in reversed(range(5)):
        node = opp(("best", 0.5, 0.5, node), ("miss", 0.5, 1.0, leaf(1.0)), key=f"O{i}")
        node = me(("go", 1.0, 0.5, node), key=f"M{i}")
    g = build(node)
    from tnh.builder.game import mixture_value
    lam = 0.2
    per_node = Game(g, Params(lam=lam, eps_white=0.1)).evaluate({})
    per_opp = mixture_value(g, Params(lam=lam, eps_white=0.1))
    assert per_node == pytest.approx(0.6**5 * 0.5 + (1 - 0.6**5))
    assert per_opp == pytest.approx(lam * 0.5 + (1 - lam) * (0.5**5 * 0.5 + (1 - 0.5**5)))
    assert per_node - per_opp > 0.07
