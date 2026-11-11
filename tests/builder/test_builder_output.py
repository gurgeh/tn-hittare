"""The curve's steps, the price per position, the two gains and the played-out line."""

import pytest

from tnh.builder.game import Game, Params
from tnh.builder.output import budget_for_price, engine_line, frontier_mass, rows, steps, table
from tnh.builder.search import solve
from tnh.builder.synth import build, leaf, me, opp

import test_builder_planted as planted


def forcing_game():
    root = me(("A", 0.5, 0.5, opp(("only", 1.0, 0.5, planted.line(2, 1, "X")), key="Ao")),
              ("B", 0.5, 0.5, opp(("only", 1.0, 0.5, planted.line(2, 4, "Y")), key="Bo")), key="root")
    return Game(build(root), Params(eps_white=0.1))


def test_steps_and_price():
    game = forcing_game()
    _, sols = solve(game, 6)
    st = steps(sols)
    assert [s.N for s in st] == list(range(1, 7))
    for s, a, b in zip(st, sols, sols[1:]):
        assert s.marginal == pytest.approx(b.value - a.value)
    # the price picks the largest N whose marginal is at least mu
    for mu in (0.0, 0.01, 0.03, 0.05, 1.0):
        N = budget_for_price(sols, mu)
        assert all(s.marginal < mu for s in st if s.N > N)
        assert N == 0 or st[N - 1].marginal >= mu
    assert budget_for_price(sols, 1.0) == 0


def test_curve_gains_add_up_when_nested_and_losses_do_not():
    # the steering pair: root and D only work together
    D = me(("win", 0.02, 0.9, leaf(0.9)), ("usual", 0.98, 0.5, leaf(0.5)), key="D")
    F = me(("fine", 0.5, 0.52, leaf(0.52)), ("ok", 0.5, 0.5, leaf(0.5)), key="F")
    root = me(("toD", 0.05, 0.5, opp(("x", 0.8, 0.5, D), ("y", 0.2, 0.5, F), key="Do")),
              ("toE", 0.95, 0.5, leaf(0.5)), key="root")
    game = Game(build(root), Params(eps_white=0.1))
    _, sols = solve(game, 3)
    rs = rows(game, sols, 3)
    total = sols[3].value - sols[0].value
    nested = all(set(a.learned) <= set(b.learned) for a, b in zip(sols, sols[1:]))
    if nested:
        steps_seen = {}
        for r in rs:
            steps_seen[r.entered_at] = r.curve_gain
        assert sum(steps_seen.values()) == pytest.approx(total)
    assert sum(r.loss_if_forgotten for r in rs) > total + 1e-6     # the pair is counted twice
    assert "loss if forgotten" in table(rs)


def test_engine_line_plays_out_learned_moves_and_best_replies():
    game = forcing_game()
    _, sols = solve(game, 3)
    learned = sols[3].learned
    assert learned == {"root": "A", "X": "good", "X0": "good"}
    line, value, why = engine_line(game, learned, game.graph.root)
    assert line == ["A", "only", "good", "r0", "good", "r0"]
    assert why == "game over"            # a synthetic leaf has no moves
    assert value == pytest.approx(0.5)   # engine value of his last best reply, for me


def test_frontier_mass():
    game = forcing_game()
    _, sols = solve(game, 3)
    fm = frontier_mass(game, sols[3].learned)
    assert fm["total"] == pytest.approx(0.0)      # every leaf here is a finished game
