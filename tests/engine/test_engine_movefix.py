"""The mistake correction: a share of the mover's probability onto his best move by static value."""

import numpy as np
import pytest

from tnh.builder.synth import build, leaf, me, opp
from tnh.engine.chess_eval import ChessEvaluator, FakeBundle
from tnh.engine.search import Engine, Settings
from tnh.engine.synth import SynthEvaluator

INF = float("inf")
C4 = "c2c4 g8f6 g2g3 e7e6 f1g2 d7d5".split()


class Planted:
    """A fixed share, for tests."""

    def __init__(self, s):
        self.s = s

    def share(self, elo, ply):
        return self.s

    def apply(self, p, best, elo, ply):
        out = (1 - self.s) * np.asarray(p, float)
        out[best] += self.s
        return out


def world():
    # his position O: x (q 0.8) leads to 0.7 for me (his worse move), y (q 0.2) to 0.4 (his best);
    # my position M: a (p 0.7) leads to 0.45, b (p 0.3) to 0.6 (my best)
    O = opp(("x", 0.8, 0.5, leaf(0.7)), ("y", 0.2, 0.5, leaf(0.4)), key="O", static=0.5)
    M = me(("a", 0.7, 0.5, leaf(0.45)), ("b", 0.3, 0.5, leaf(0.6)), key="M", static=0.5)
    Mo = opp(("only", 1.0, 0.5, M), key="Mo", static=0.5)
    return build(me(("toO", 0.5, 0.5, O), ("toM", 0.5, 0.5, Mo), key="root", static=0.5))


def engine(mode, s=0.5, **kw):
    eng = Engine(SynthEvaluator(world()), Settings(my_side="white", kappa=INF, mix="kappa", mode=mode,
                                                   prior_min=0.0, blunder=1.0, price=1.0, **kw))
    eng._fix, eng.fix_source = Planted(s), "planted"
    return eng


def test_off_reproduces_the_values_before_the_correction():
    import json
    from pathlib import Path
    path = Path.home() / "out/tn-hittare/scratch/engine/fix_baseline_before.json"
    if not path.exists():
        pytest.skip("no recorded baseline")
    before = json.loads(path.read_text())
    for P in (0.0, 0.25):
        eng = Engine(ChessEvaluator({"fake": FakeBundle()}), Settings(tempo="fake", my_side="white", price=0.002,
                                                                      prepared=P, mistake_fix="off"))
        eng.set_position(C4)
        eng.search(nodes=300)
        assert eng.last.net == before[f"priced/{P}"]["net"]
        eng = Engine(ChessEvaluator({"fake": FakeBundle()}), Settings(tempo="fake", my_side="white", mode="bestmove",
                                                                      prepared=P, mistake_fix="off",
                                                                      widen=0.0, reduce_margin=0.0))
        eng.set_position(C4)
        eng.search(depth=6)
        for k, d in eng.bm.completed.items():
            assert d.value == before[f"bestmove/{P}"][str(k)]["value"]


def test_the_share_moves_his_probabilities_and_my_model_play_in_the_priced_mode():
    eng = engine("repertoire", s=0.5)
    eng.search(nodes=50)
    g = eng.graph
    O = next(n for n in g.nodes if n.key == "O")
    M = next(n for n in g.nodes if n.key == "M")
    # his best (by static value for him) is y; mine is b
    assert eng.game.dist[O.id] == pytest.approx([0.5 * 0.8, 0.5 * 0.2 + 0.5])
    assert eng.game.dist[M.id] == pytest.approx([0.5 * 0.7, 0.5 * 0.3 + 0.5])
    # the cost rule still reads the first probabilities
    assert M.info["p_first"] == pytest.approx([0.7, 0.3])


def test_the_share_moves_his_probabilities_in_bestmove_mode():
    eng = engine("bestmove", s=0.5)
    eng.search(depth=2)
    d = eng.bm.completed[2]
    assert d.root_values["toO"] == pytest.approx(0.4 * 0.7 + 0.6 * 0.4)    # q = (0.4, 0.6)
    assert d.root_values["toM"] == pytest.approx(0.6)                      # my moves: max, no correction needed


def test_composition_with_prepared():
    s, P = 0.3, 0.25
    eng = engine("bestmove", s=s, prepared=P)
    eng.search(depth=2)
    q_best = 1 - (1 - P) * (1 - s) * (1 - 0.2)
    want = (1 - q_best) * 0.7 + q_best * 0.4
    assert eng.bm.completed[2].root_values["toO"] == pytest.approx(want)


def test_bestmove_with_the_fix_is_independent_of_expansion_order():
    from tnh.builder.synth import random_game
    g = random_game(np.random.default_rng(42), depth=7, branching=(2, 3))
    ev = SynthEvaluator(g)
    st = Settings(my_side="white", kappa=INF, mix="kappa", mode="bestmove", prior_min=0.02, blunder=0.3)
    a = Engine(ev, st)
    a._fix, a.fix_source = Planted(0.3), "planted"
    a.search(depth=5)
    b = Engine(ev, st)
    b._fix, b.fix_source = Planted(0.3), "planted"
    b.search(depth=0)
    rng = np.random.default_rng(1)
    for _ in range(20):
        cand = [n.id for n in b.graph.nodes if n.children is None and "result_mover" not in n.info]
        if not cand:
            break
        b._expand(list(rng.choice(cand, size=min(5, len(cand)), replace=False)))
    b.search(depth=5)
    for k in range(6):
        assert b.bm.completed[k].value == pytest.approx(a.bm.completed[k].value, abs=1e-12)


def test_switching_the_fix_is_a_re_solve():
    fb = FakeBundle()
    eng = Engine(ChessEvaluator({"fake": fb}), Settings(tempo="fake", my_side="white", price=0.002))
    eng.set_position(C4)
    eng.search(nodes=150)
    on, calls = eng.last.value, fb.calls
    eng.set_settings(mistake_fix="off")
    off = eng.solve().value
    assert fb.calls == calls and off != on
    assert eng.fix_share_root() == 0.0
    eng.set_settings(mistake_fix="fitted")
    assert eng.solve().value == on and eng.fix_share_root() > 0


def test_uci_mistake_fix_option_and_root_share():
    import sys
    import chess
    import chess.engine
    eng = chess.engine.SimpleEngine.popen_uci([sys.executable, "-m", "tnh.engine.uci", "--fake"])
    try:
        assert eng.options["MistakeFix"].default == "fitted" and eng.options["Blunder"].default == "0.3"
        b = chess.Board()
        for x in C4:
            b.push_uci(x)
        for value, want in (("fitted", True), ("off", False)):
            eng.configure({"MistakeFix": value, "Side": "white"})
            roots = []
            with eng.analysis(b, chess.engine.Limit(nodes=60)) as an:
                for info in an:
                    if "string" in info and info["string"].startswith("preppy root"):
                        roots.append(dict(t.split("=", 1) for t in info["string"].split()[2:] if "=" in t))
            assert roots and (float(roots[-1]["fix"]) > 0) == want
            assert roots[-1]["fix_source"] == ("fitted" if want else "off")
    finally:
        eng.quit()
