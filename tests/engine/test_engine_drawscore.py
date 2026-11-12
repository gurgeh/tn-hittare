"""DrawScore: what a draw is worth to me. 0.5 is today's behaviour bit for bit; other values
re-weight draws in the leaves and finished games, re-solve only."""

import numpy as np
import pytest

from tnh.builder.synth import build, leaf, me, opp
from tnh.engine.chess_eval import ChessEvaluator, FakeBundle
from tnh.engine.search import Engine, Settings
from tnh.engine.synth import SynthEvaluator

C4 = "c2c4 g8f6 g2g3 e7e6 f1g2 d7d5".split()


def test_half_reproduces_todays_values_exactly():
    def run(**kw):
        eng = Engine(ChessEvaluator({"fake": FakeBundle()}), Settings(tempo="fake", price=0.002, **kw))
        eng.set_position(C4)
        eng.search(nodes=150)
        return eng
    a, b = run(), run(draw_score=0.5)
    assert a.last.net == b.last.net and a.last.learned == b.last.learned
    assert all(a.V[i] == b.V[i] for i in a.V)


def drawish_and_sharp():
    # A and B have the same expected score for the opponent to move (0.5); A is drawish
    # (W 0.3, D 0.4, L 0.3), B sharp (0.5, 0, 0.5). The mover there is the opponent.
    A = opp(("x", 1.0, 0.5, opp_free("a2")), key="A", wdl=(0.3, 0.4, 0.3), static=0.5)
    B = opp(("y", 1.0, 0.5, opp_free("b2")), key="B", wdl=(0.5, 0.0, 0.5), static=0.5)
    return build(me(("toA", 0.5, 0.5, A), ("toB", 0.5, 0.5, B), key="root", static=0.5))


def opp_free(k):
    return me(("z", 1.0, 0.5, leaf(0.7)), key=k)


def test_a_lower_draw_score_lowers_drawish_positions_more_than_sharp_ones():
    g = drawish_and_sharp()
    ev = SynthEvaluator(g)
    eng = Engine(ev, Settings(my_side="white", kappa=float("inf"), price=0.0, expand_root=True))
    eng.search(nodes=0)                                   # the root and its two children
    A = next(n for n in eng.graph.nodes if n.key == "A")
    B = next(n for n in eng.graph.nodes if n.key == "B")
    assert A.value == pytest.approx(0.5) and B.value == pytest.approx(0.5)
    calls = sum(ev.calls.values())
    eng.set_settings(draw_score=0.3)
    eng.solve()
    assert A.value == pytest.approx(0.3 + 0.3 * 0.4)      # my W (his L) + 0.3 x D
    assert B.value == pytest.approx(0.5)
    assert sum(ev.calls.values()) == calls                # nothing evaluated again
    assert eng.last.learned.get("root") == "toB"
    eng.set_settings(draw_score=0.5)
    eng.solve()
    assert A.value == 0.5


def test_a_finished_drawn_game_is_worth_the_draw_score():
    g = build(me(("draw", 0.5, 0.5, leaf(0.5)), ("win", 0.5, 0.6, leaf(0.6)), key="root"))
    eng = Engine(SynthEvaluator(g), Settings(my_side="white", kappa=float("inf"), price=0.0))
    eng.search(nodes=10)
    d = next(n for n in eng.graph.nodes if n.key == "draw")
    assert d.value == 0.5
    eng.set_settings(draw_score=0.2)
    eng.solve()
    assert d.value == pytest.approx(0.2)


def test_draw_score_survives_a_root_move_and_needs_no_model_calls():
    fb = FakeBundle()
    eng = Engine(ChessEvaluator({"fake": fb}), Settings(tempo="fake", price=0.002, my_side="white"))
    eng.set_position(C4)
    eng.search(nodes=100)
    calls = fb.calls
    eng.set_settings(draw_score=0.3)
    v_root = eng.solve().value
    line = eng.lines(1)[0].moves
    eng.set_position(C4 + line[:1])                    # moving the root keeps the re-weighted values
    eng.set_position(C4)
    assert eng.solve().value == v_root
    assert fb.calls == calls
