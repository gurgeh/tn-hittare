"""The engine on chess positions with fake models, through the API and through UCI."""

import sys

import chess
import chess.engine
import numpy as np
import pytest

from tnh.engine.api import Handle
from tnh.engine.chess_eval import ChessEvaluator, FakeBundle
from tnh.engine.search import Engine, Settings
from tnh.engine.uci import cp_of

C4 = "c2c4 g8f6 g2g3 e7e6 f1g2 d7d5".split()


def fake_engine(**kw):
    fb = FakeBundle({"g1f3": 2.0, "d2d4": 2.0, "e7e6": 1.5, "d7d5": 1.5})
    eng = Engine(ChessEvaluator({"fake": fb}), Settings(tempo="fake", **kw))
    return eng, fb


def test_root_moves_all_expanded_and_moves_read_lazily():
    eng, fb = fake_engine(price=0.002)
    eng.set_position(C4)
    eng.search(nodes=0)
    root = eng.graph.nodes[eng.graph.root]
    assert len(root.moves) == len(list(chess.Board(_fen(C4)).legal_moves))
    assert all(eng.graph.nodes[c].children is not None for c in root.children)
    assert fb.calls == 1 + len(root.moves)
    # positions created but never expanded have not had their moves read
    from tnh.engine.search import PENDING
    assert any(n.moves is PENDING for n in eng.graph.nodes)


def _fen(moves):
    b = chess.Board()
    for m in moves:
        b.push_uci(m)
    return b.fen()


def test_reuse_after_position_moves_on_chess():
    eng, fb = fake_engine(price=0.002)
    eng.set_position(C4)
    eng.search(nodes=200)
    calls = fb.calls
    line = eng.lines(1)[0].moves
    eng.set_position(C4 + line[:2])                    # into the main line, already grown
    root = eng.graph.nodes[eng.graph.root]
    assert root.children is not None
    fresh = sum(1 for c in root.children if eng.graph.nodes[c].children is None)
    eng.search(nodes=0)
    assert fb.calls - calls == fresh                   # only the new root's unexpanded moves were evaluated
    eng.set_position(C4)
    eng.search(nodes=0)
    assert fb.calls - calls == fresh


def test_handle_runs_in_the_background_and_stops():
    eng, _ = fake_engine(price=0.002)
    eng.set_position(C4)
    h = Handle(eng)
    seen = []
    h.start(on_info=seen.append, info_every=0.05)
    import time
    time.sleep(0.5)
    assert h.busy
    with pytest.raises(RuntimeError):
        h.require_idle()
    r = h.stop()
    assert not h.busy and r is not None and r.lines and seen


def test_adaptive_solve_interval():
    eng, _ = fake_engine(price=0.002, solve_every=20, solve_ratio=1e9)
    eng.set_position(C4)
    eng.search(nodes=200)
    # evaluation never outweighs a solve by 1e9: only the 10x-floor rule solves
    assert eng.stats["solves"] <= 200 // 200 + 3
    eng2, _ = fake_engine(price=0.002, solve_every=20, solve_ratio=0.0)
    eng2.set_position(C4)
    eng2.search(nodes=200)
    assert eng2.stats["solves"] >= 200 // 20 - 2


def test_cap_bisects_only_when_asked():
    eng, _ = fake_engine(cap=3)
    eng.set_position(C4)
    r = eng.search(nodes=150)
    assert r.solve.cost <= 3
    b = eng.stats["cap bisections"]
    eng.solve(final=False)
    assert eng.stats["cap bisections"] == b             # reuses the last price
    eng.solve()
    assert eng.stats["cap bisections"] == b + 1


def test_cp_mapping():
    assert cp_of(0.5) == 0
    assert cp_of(1 / (1 + 10 ** (-200 / 400))) == 200
    assert cp_of(0.0) == -cp_of(1.0) and cp_of(1.0) <= 2000


@pytest.fixture
def uci():
    eng = chess.engine.SimpleEngine.popen_uci([sys.executable, "-m", "tnh.engine.uci", "--fake"])
    yield eng
    eng.quit()


def test_uci_analyse_gives_legal_lines_with_scores(uci):
    uci.configure({"Price": "0.002", "MyElo": 2200, "OppElo": 2100, "Side": "white"})
    board = chess.Board(_fen(C4))
    infos = uci.analyse(board, chess.engine.Limit(nodes=120), multipv=3)
    assert len(infos) == 3
    for info in infos:
        b = board.copy()
        for mv in info["pv"]:
            assert mv in b.legal_moves
            b.push(mv)
        assert "score" in info
    assert infos[0]["score"].relative.score() >= infos[-1]["score"].relative.score() - 2000


def kv(s: str) -> dict:
    return dict(t.split("=", 1) for t in s.split()[1:] if "=" in t)


def test_uci_info_strings_and_bestmove(uci):
    assert uci.id["name"] == "preppy"
    board = chess.Board(_fen(C4))
    strings = []
    with uci.analysis(board, chess.engine.Limit(nodes=80), multipv=2) as an:
        for info in an:
            if "string" in info:
                strings.append(info["string"])
    lines = [kv(s) for s in strings if s.startswith("preppy multipv=")]
    roots = [kv(s) for s in strings if s.startswith("preppy root")]
    assert lines and roots
    x = lines[-1]
    assert x["multipv"] in ("1", "2") and 0 <= float(x["value"]) <= 1 and float(x["reach"]) <= 1
    assert int(x["nonobvious"]) >= 0
    for f in x["moves"].split(","):
        flag = f.split(":")[1]
        assert flag in ("L", "M") or 0 <= float(flag) <= 1
    assert roots[-1]["cap"] == "-" and int(roots[-1]["expanded"]) > 0
    res = uci.play(board, chess.engine.Limit(nodes=40))
    assert res.move in board.legal_moves


def test_uci_infinite_and_stop_many_times_keeps_the_graph(uci):
    board = chess.Board(_fen(C4))
    expanded = []
    for i in range(5):
        with uci.analysis(board, multipv=3) as an:            # go infinite
            n = 0
            for info in an:
                if "string" in info and info["string"].startswith("preppy root"):
                    expanded.append(int(kv(info["string"])["expanded"]))
                    n += 1
                    if n >= 1:
                        break
            an.stop()
            best = an.wait()
            assert best.move in board.legal_moves
        if i == 2:
            board.push(best.move)                           # step into a child: the graph is kept
    assert expanded == sorted(expanded)                     # never cleared


def test_uci_errors_are_reported_and_the_engine_goes_on():
    import subprocess
    p = subprocess.Popen([sys.executable, "-m", "tnh.engine.uci", "--fake"], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, text=True)
    out, _ = p.communicate("uci\nsetoption name Nonsense value 3\nsetoption name Price value x\nisready\n"
                           "position startpos moves e2e4\ngo nodes 20\nisready\n", timeout=60)
    assert p.returncode == 0                                # stdin closed: a clean exit
    assert "error=unknown_option:Nonsense" in out and "error=ValueError" in out
    assert "readyok" in out and "bestmove" in out


def test_uci_cap_and_opponent_to_move(uci):
    uci.configure({"Cap": 2, "Side": "black"})        # White to move at the root: the opponent's lines
    board = chess.Board(_fen(C4))
    infos = uci.analyse(board, chess.engine.Limit(nodes=100), multipv=3)
    assert len(infos) == 3 and all(i["pv"][0] in board.legal_moves for i in infos)


@pytest.mark.slow
def test_uci_real_models_smoke():
    eng = chess.engine.SimpleEngine.popen_uci([sys.executable, "-m", "tnh.engine.uci"])
    try:
        info = eng.analyse(chess.Board(_fen(C4)), chess.engine.Limit(nodes=60), multipv=2)
        assert info and info[0]["pv"]
    finally:
        eng.quit()
