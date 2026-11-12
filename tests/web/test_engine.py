"""The engine section's manager, against a fake UCI engine (`fake_uci.py`)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import chess
import pytest
from fastapi.testclient import TestClient

from _webfix import fake_hub, stats
from tnh import movemix as mm
from tnh.web.app import create_app
from tnh.web.engine import EngineError, EngineManager, EngineSpec, parse_info, parse_option
from tnh.web.stats import Shrink

FAKE = str(Path(__file__).with_name("fake_uci.py"))
PREPPY = ("MyElo", "OppElo", "Side", "Tempo", "Price", "Cap", "Obvious", "Prepared")


def specs(*extra: str) -> dict[str, EngineSpec]:
    return {
        "fake": EngineSpec("fake", "Fake", [sys.executable, FAKE, *extra], {"Hash": "32"}, PREPPY, short="FK"),
        "other": EngineSpec("other", "Other", [sys.executable, FAKE], short="OT"),
        "slow": EngineSpec("slow", "Slow", [sys.executable, FAKE, "--slow-start", "5"], short="SL"),
        "dies": EngineSpec("dies", "Dies", [sys.executable, FAKE, "--die-on-go"], short="DI"),
        "gpu": EngineSpec("gpu", "Gpu", [sys.executable, FAKE], gpu_mb=4000, short="GP"),
    }


@pytest.fixture()
def mgr():
    m = EngineManager(specs(), idle_s=30, start_timeout=10, stop_timeout=5, gpu_free=lambda: 1000)
    yield m
    m.close()


def wait_lines(m: EngineManager, n: int, timeout: float = 5.0) -> dict:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        s = m.snapshot()
        if len(s["lines"]) >= n:
            return s
        time.sleep(0.02)
    raise AssertionError(f"fewer than {n} lines: {m.snapshot()}")


def after(*uci: str) -> tuple[str, list[str]]:
    b = chess.Board()
    for m in uci:
        b.push_uci(m)
    return b.fen(), list(uci)


def test_parsers():
    o = parse_option("option name Side type combo default tomove var white var black var tomove")
    assert o == {"name": "Side", "type": "combo", "default": "tomove", "vars": ["white", "black", "tomove"]}
    assert parse_option("option name Clear Hash type button") == {"name": "Clear Hash", "type": "button", "vars": []}
    d = parse_info("info depth 12 seldepth 18 multipv 2 score cp -35 lowerbound wdl 100 800 100 nodes 5000 nps 9 pv e2e4 e7e5")
    assert (d["multipv"], d["kind"], d["value"], d["bound"], d["wdl"], d["pv"]) == (2, "cp", -35, "lowerbound",
                                                                                   [100, 800, 100], ["e2e4", "e7e5"])
    assert parse_info("info string tnh root value 0.5")["string"] == "tnh root value 0.5"
    assert parse_info("info depth 3 currmove e2e4") is None


def test_start_lines_and_white_view(mgr):
    gen = mgr.go("fake", *after(), multipv=3)
    s = wait_lines(mgr, 3)
    assert s["engine"] == "fake" and s["gen"] == gen and s["short"] == "FK"
    assert [x["multipv"] for x in s["lines"]] == [1, 2, 3]
    # first position (P = 1), White to move: cp 1000 + 10 K as the engine says
    assert [x["cp"] for x in s["lines"]] == [1010, 1020, 1030]
    assert s["lines"][0]["san"][0] == chess.Board().san(chess.Move.from_uci(s["lines"][0]["pv"][0]))
    assert any(x.startswith("tnh root") for x in s["strings"])
    # Black to move: the engine's cp is for Black, the page's for White
    mgr.go("fake", *after("e2e4"), multipv=2)
    s = wait_lines(mgr, 2)
    assert [x["cp"] for x in s["lines"]] == [-2010, -2020]


def test_a_new_position_never_shows_the_old_lines(mgr):
    mgr.go("fake", *after(), multipv=2)
    wait_lines(mgr, 2)
    for i, line in enumerate((("e2e4",), ("e2e4", "e7e5"), ("d2d4",), ("d2d4", "g8f6"))):
        fen, moves = after(*line)
        gen = mgr.go("fake", fen, moves, multipv=2)
        # the fake prints a round for the old position after stop: none of it may show
        for _ in range(10):
            s = mgr.snapshot()
            assert s["gen"] == gen and s["fen"] == fen
            for x in s["lines"]:
                assert abs(x["cp"]) // 1000 == i + 2, (line, x)      # P counts position commands
            time.sleep(0.01)


def test_settings_and_multipv_reach_the_engine(mgr):
    mgr.go("fake", *after(), multipv=1, settings={"Price": "0.002", "Side": "white", "Bogus": "1"})
    s = wait_lines(mgr, 1)
    joined = " | ".join(s["strings"])
    assert {o["name"] for o in s["options"]} == set(PREPPY)
    mgr.go("fake", *after(), multipv=4)
    assert len(wait_lines(mgr, 4)["lines"]) == 4
    # the fixed options and the settings were set; an unknown setting was not
    assert mgr._applied["Price"] == "0.002" and "Bogus" not in mgr._applied
    assert "MultiPV" in mgr._applied
    del joined


def test_switch_and_off_quit_the_process(mgr):
    mgr.go("fake", *after(), multipv=1)
    wait_lines(mgr, 1)
    first = mgr._proc.proc
    mgr.go("other", *after(), multipv=1)
    assert first.poll() is not None                      # the old one is gone
    s = wait_lines(mgr, 1)
    assert s["engine"] == "other"
    second = mgr._proc.proc
    mgr.go("off", *after(), multipv=1)
    assert second.poll() is not None and mgr._proc is None
    assert mgr.snapshot()["engine"] == "off" and mgr.snapshot()["lines"] == []


def test_errors_are_reported(mgr):
    with pytest.raises(EngineError, match="GPU"):
        mgr.go("gpu", *after(), multipv=1)
    with pytest.raises(EngineError, match="unknown engine"):
        mgr.go("nosuch", *after(), multipv=1)
    mgr.go("dies", *after(), multipv=1)                  # dies on go: found at the next command
    time.sleep(0.3)
    with pytest.raises(EngineError):
        mgr.go("dies", *after("e2e4"), multipv=1)
    assert mgr.snapshot()["error"] and mgr._proc is None


def test_start_timeout():
    m = EngineManager(specs(), start_timeout=0.5, gpu_free=lambda: None)
    try:
        with pytest.raises(EngineError, match="uciok"):
            m.go("slow", *after(), multipv=1)
        assert m._proc is None
    finally:
        m.close()


def test_idle_quit():
    m = EngineManager(specs(), idle_s=0.4, gpu_free=lambda: None)
    try:
        m.go("fake", *after(), multipv=1)
        p = m._proc.proc
        time.sleep(1.6)
        assert p.poll() is not None and m._proc is None
        assert "without a page" in (m.snapshot()["error"] or "")
    finally:
        m.close()


def test_game_over_has_no_search(mgr):
    fen, moves = after("f2f3", "e7e5", "g2g4", "d8h4")
    mgr.go("fake", fen, moves, multipv=2)
    time.sleep(0.1)
    assert mgr.snapshot()["lines"] == []


def test_endpoints_and_shutdown():
    m = EngineManager(specs(), gpu_free=lambda: None)
    app = create_app(stats(), fake_hub(), shrink=Shrink(rules={}, prob=mm.Rule()), engines=m)
    with TestClient(app) as c:
        names = [e["name"] for e in c.get("/api/uci/engines").json()["engines"]]
        assert names == list(specs())
        fen, moves = after("e2e4")
        r = c.post("/api/uci/go", json={"engine": "fake", "fen": fen, "moves": moves, "multipv": 2})
        assert r.status_code == 200
        wait_lines(m, 2)
        assert c.get("/api/uci/state").json()["lines"][0]["cp"] < 0
        assert c.post("/api/uci/go", json={"engine": "gpu", "fen": fen, "moves": moves}).status_code == 200
        assert c.post("/api/uci/go", json={"engine": "fake", "fen": fen, "moves": moves, "multipv": 99}).status_code == 400
        p = None
        c.post("/api/uci/go", json={"engine": "fake", "fen": fen, "moves": moves, "multipv": 1})
        p = m._proc.proc
    assert p.poll() is not None                          # the server's shutdown quit it


def test_preppy_fake_over_uci():
    """The repertoire engine itself (`python -m tnh.engine.uci --fake`: fake models, no
    database, no GPU), through the manager: lines, its info strings, a position change."""
    spec = EngineSpec("preppy", "preppy", [sys.executable, "-m", "tnh.engine.uci", "--fake"], {},
                      ("MyElo", "OppElo", "Side", "Price", "Cap", "Obvious", "Prepared"), short="preppy")
    m = EngineManager({"preppy": spec}, start_timeout=120, gpu_free=lambda: None)
    try:
        m.go("preppy", *after(), multipv=2, settings={"Side": "white", "MyElo": "2300", "OppElo": "2100"})
        s = wait_lines(m, 2, timeout=60)
        assert {o["name"] for o in s["options"]} >= {"MyElo", "Side", "Price"}
        end = time.monotonic() + 30
        while time.monotonic() < end and not (s.get("root") and all("preppy" in x for x in s["lines"])):
            time.sleep(0.2)
            s = m.snapshot()
        line = s["lines"][0]["preppy"]
        assert 0 <= line["white"] <= 1 and len(line["flags"]) == len(s["lines"][0]["pv"])
        assert line["flags"][0] in ("L", "M")                    # White (me) to move: my move first
        assert s["root"]["white"] is not None and s["root"]["expanded"] >= 1
        # Black to move: White's value is one minus the side to move's
        fen, moves = after("e2e4")
        m.go("preppy", fen, moves, multipv=2, settings={"Side": "white"})
        s = wait_lines(m, 1, timeout=60)
        end = time.monotonic() + 30
        while time.monotonic() < end and "preppy" not in s["lines"][0]:
            time.sleep(0.2)
            s = m.snapshot()
        x = s["lines"][0]["preppy"]
        assert x["white"] == pytest.approx(1 - x["value"])
        assert s["fen"] == fen and x["flags"][0] not in ("L", "M")   # his move: a probability
    finally:
        m.close()


def test_start_up_strings_are_not_shown_errors_are(mgr):
    mgr.go("fake", *after(), multipv=1)
    s = wait_lines(mgr, 1)
    time.sleep(0.1)
    s = mgr.snapshot()
    assert not any("Using" in x or "NNUE" in x for x in s["strings"])
    assert any(x.startswith("tnh root") for x in s["strings"])          # said during the search
    mgr.go("fake", *after("e2e4"), multipv=1, settings={"Price": "bad"})
    end = time.monotonic() + 3
    while time.monotonic() < end and not any("error" in x for x in mgr.snapshot()["strings"]):
        time.sleep(0.02)
    assert any("preppy error=" in x for x in mgr.snapshot()["strings"])


def test_lc0_contempt_mapping():
    """Lc0's contempt from the page, through the fake engine: off, on with me as White, then
    flipped so that I am Black, then off again."""
    from tnh.web.engine import lc0_contempt
    spec = EngineSpec("lc0f", "Lc0 (fake)", [sys.executable, FAKE], {},
                      ("Contempt", "WDLCalibrationElo", "ContemptMode"), short="Lc0", derive=lc0_contempt)
    m = EngineManager({"lc0f": spec}, gpu_free=lambda: None)
    try:
        def sent(settings, me):
            m.go("lc0f", *after(), multipv=1, settings=settings, me=me)
            wait_lines(m, 1)
            return m.snapshot()["applied"]

        assert sent({}, "white") == {"Contempt": "", "ContemptMode": "disable", "WDLCalibrationElo": "0"}
        assert sent({"Contempt": "0"}, "white")["ContemptMode"] == "disable"
        on = sent({"Contempt": "200", "WDLCalibrationElo": "2200"}, "white")
        assert on == {"Contempt": "200", "ContemptMode": "white_side_analysis", "WDLCalibrationElo": "2200"}
        assert sent({"Contempt": "200", "WDLCalibrationElo": "2200"}, "black")["ContemptMode"] == "black_side_analysis"
        assert sent({"Contempt": "-150"}, "black") == {"Contempt": "-150", "ContemptMode": "black_side_analysis",
                                                       "WDLCalibrationElo": "0"}
        off = sent({"Contempt": ""}, "black")
        assert off == {"Contempt": "", "ContemptMode": "disable", "WDLCalibrationElo": "0"}
        # the engine was told each time (the fake echoes every setoption)
        assert m._applied["ContemptMode"] == "disable"
        with pytest.raises(EngineError, match="Contempt"):
            m.go("lc0f", *after(), multipv=1, settings={"Contempt": "lots"}, me="white")
        assert "play" not in {lc0_contempt({"Contempt": c}, me)["ContemptMode"]
                              for c in ("", "0", "100", "-100") for me in ("white", "black")}
    finally:
        m.close()
