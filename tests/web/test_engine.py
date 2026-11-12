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
                      ("Contempt", "WDLCalibrationElo", "WDLEvalObjectivity", "WDLContemptAttenuation", "ContemptMode",
                       "MovesLeftThreshold"), short="Lc0", derive=lc0_contempt,
                      hidden={"WDLContemptAttenuation": {"name": "WDLContemptAttenuation", "type": "string",
                                                         "default": "1", "vars": []}})
    m = EngineManager({"lc0f": spec}, gpu_free=lambda: None)
    try:
        def sent(settings, me):
            m.go("lc0f", *after(), multipv=1, settings=settings, me=me)
            wait_lines(m, 1)
            return m.snapshot()["applied"]

        base = {"WDLEvalObjectivity": "1", "WDLContemptAttenuation": "1", "MovesLeftThreshold": "0.8"}
        assert sent({}, "white") == {"Contempt": "", "ContemptMode": "disable", "WDLCalibrationElo": "0", **base}
        assert sent({"Contempt": "0"}, "white")["ContemptMode"] == "disable"
        on = sent({"Contempt": "200", "WDLCalibrationElo": "2200"}, "white")
        assert on == {"Contempt": "200", "ContemptMode": "white_side_analysis", "WDLCalibrationElo": "2200", **base}
        assert sent({"Contempt": "200", "WDLCalibrationElo": "2200"}, "black")["ContemptMode"] == "black_side_analysis"
        assert sent({"Contempt": "-150"}, "black") == {"Contempt": "-150", "ContemptMode": "black_side_analysis",
                                                       "WDLCalibrationElo": "0", **base}
        off = sent({"Contempt": ""}, "black")
        assert off == {"Contempt": "", "ContemptMode": "disable", "WDLCalibrationElo": "0", **base}
        # the engine was told each time (the fake echoes every setoption)
        assert m._applied["ContemptMode"] == "disable"
        with pytest.raises(EngineError, match="Contempt"):
            m.go("lc0f", *after(), multipv=1, settings={"Contempt": "lots"}, me="white")
        assert "play" not in {lc0_contempt({"Contempt": c}, me)["ContemptMode"]
                              for c in ("", "0", "100", "-100") for me in ("white", "black")}
    finally:
        m.close()


def test_contempt_change_clears_lc0s_tree_not_preppys(tmp_path):
    """A change of Lc0's contempt settings (a flip included) sends ucinewgame before the next
    search; a position change does not; preppy never gets one on a setting change."""
    from tnh.web.engine import lc0_contempt
    log_l, log_p = tmp_path / "lc0.log", tmp_path / "preppy.log"
    lc0 = EngineSpec("lc0f", "Lc0 (fake)", [sys.executable, FAKE, "--log", str(log_l)], {},
                     ("Contempt", "WDLCalibrationElo", "ContemptMode"), derive=lc0_contempt,
                     reset_on=("Contempt", "ContemptMode", "WDLCalibrationElo"))
    pre = EngineSpec("preppyf", "preppy (fake)", [sys.executable, FAKE, "--log", str(log_p)], {}, PREPPY + ("DrawScore",))
    m = EngineManager({"lc0f": lc0, "preppyf": pre}, gpu_free=lambda: None)
    try:
        def news(path):
            return path.read_text().count("ucinewgame")

        m.go("lc0f", *after(), multipv=1, me="white")                      # first settings: disable
        wait_lines(m, 1)
        n0 = news(log_l)
        m.go("lc0f", *after("e2e4"), multipv=1, me="white")                # position only
        wait_lines(m, 1)
        assert news(log_l) == n0
        m.go("lc0f", *after("e2e4"), multipv=1, settings={"Contempt": "200"}, me="white")
        wait_lines(m, 1)
        assert news(log_l) == n0 + 1
        m.go("lc0f", *after("e2e4"), multipv=1, settings={"Contempt": "200"}, me="black")   # a flip
        wait_lines(m, 1)
        assert news(log_l) == n0 + 2
        cmds = log_l.read_text().splitlines()
        i = len(cmds) - 1 - cmds[::-1].index("ucinewgame")
        assert cmds[i - 1].startswith("setoption name ContemptMode value black_side_analysis")
        assert cmds[i + 1:i + 2] == ["isready"] and "go infinite" in cmds[i + 2:]
        m.go("preppyf", *after(), multipv=1, settings={"Price": "0.002", "Side": "white"})
        wait_lines(m, 1)
        m.go("preppyf", *after(), multipv=1, settings={"Price": "0.003", "Side": "black", "DrawScore": "0.6"})
        wait_lines(m, 1)
        assert news(log_p) == 0
    finally:
        m.close()


def lc0_like(*extra: str) -> EngineSpec:
    """The real Lc0 spec's settings, start options and rules, on the fake engine."""
    from tnh.web.engine import default_specs
    real = default_specs()["lc0"]
    return EngineSpec("lc0f", "Lc0 (fake)", [sys.executable, FAKE, "--strict", *extra],
                      {"WDLDrawRateReference": real.options["WDLDrawRateReference"]}, real.settings,
                      derive=real.derive, reset_on=real.reset_on, hidden=real.hidden, short="Lc0")


def test_lc0_guide_settings(tmp_path):
    """BT4's draw rate at start; objectivity and attenuation pass through and clear the tree;
    the hidden attenuation option is sent and accepted; MovesLeftThreshold 0.95 above 250."""
    from tnh.web.engine import lc0_contempt
    log = tmp_path / "lc0.log"
    m = EngineManager({"lc0f": lc0_like("--log", str(log))}, gpu_free=lambda: None)
    try:
        m.go("lc0f", *after(), multipv=1, me="white")
        wait_lines(m, 1)
        cmds = log.read_text().splitlines()
        assert "setoption name WDLDrawRateReference value 0.62" in cmds
        assert cmds.index("setoption name WDLDrawRateReference value 0.62") < cmds.index("go infinite")
        s = m.snapshot()
        assert {o["name"] for o in s["options"]} >= {"Contempt", "WDLEvalObjectivity", "WDLContemptAttenuation"}
        n = log.read_text().count("ucinewgame")
        m.go("lc0f", *after(), multipv=1, me="white",
             settings={"Contempt": "200", "WDLEvalObjectivity": "0", "WDLContemptAttenuation": "0.5"})
        wait_lines(m, 1)
        a = m.snapshot()["applied"]
        assert (a["WDLEvalObjectivity"], a["WDLContemptAttenuation"], a["MovesLeftThreshold"]) == ("0", "0.5", "0.8")
        assert "setoption name WDLContemptAttenuation value 0.5" in log.read_text()
        assert log.read_text().count("ucinewgame") == n + 1
        m.go("lc0f", *after(), multipv=1, me="white", settings={"Contempt": "300"})
        wait_lines(m, 1)
        assert m.snapshot()["applied"]["MovesLeftThreshold"] == "0.95"
        m.go("lc0f", *after(), multipv=1, me="black", settings={"Contempt": "-260"})
        wait_lines(m, 1)
        assert m.snapshot()["applied"]["MovesLeftThreshold"] == "0.95"
        assert lc0_contempt({"Contempt": "250"}, "white")["MovesLeftThreshold"] == "0.8"
        with pytest.raises(EngineError, match="0 to 1"):
            m.go("lc0f", *after(), multipv=1, me="white", settings={"WDLEvalObjectivity": "2"})
    finally:
        m.close()


def test_a_refused_option_is_an_error():
    spec = lc0_like()
    spec.options["NoSuchThing"] = "1"
    m = EngineManager({"lc0f": spec}, gpu_free=lambda: None)
    try:
        with pytest.raises(EngineError, match="refused.*No such option"):
            m.go("lc0f", *after(), multipv=1, me="white")
    finally:
        m.close()


def test_fill_from_ratings():
    from tnh.web.engine import lc0_from_ratings
    assert lc0_from_ratings(2200, 2200, "white", "classical") == {"Contempt": "0", "WDLCalibrationElo": "2400"}
    assert lc0_from_ratings(2200, 2000, "white", "classical") == {"Contempt": "200", "WDLCalibrationElo": "2400"}
    assert lc0_from_ratings(2200, 2000, "black", "classical") == {"Contempt": "-200", "WDLCalibrationElo": "2200"}
    assert lc0_from_ratings(2200, 2000, "white", "blitz") == {"Contempt": "200", "WDLCalibrationElo": "2200"}
    c = TestClient(create_app(stats(), fake_hub(), shrink=Shrink(rules={}, prob=mm.Rule())))
    r = c.get("/api/uci/fill", params={"white": 1900, "black": 2300, "me": "black", "tempo": "classical"})
    assert r.json() == {"Contempt": "400", "WDLCalibrationElo": "2500"}
    assert c.get("/api/uci/fill", params={"white": 1900, "black": 2300, "me": "both"}).status_code == 400


def test_preppy_line_parsing_is_lenient():
    from tnh.web.engine import _kv, _preppy_line
    kv = _kv("preppy multipv=1 value=0.6 net=- nonobvious=2 reach=0.1 moves=e2e4:L,e7e5:0.4 provisional=1 newkey=x")
    d = _preppy_line(kv, ["e2e4", "e7e5"], flip=True)
    assert d["provisional"] is True and d["white"] == pytest.approx(0.4) and d["net"] is None
    assert d["flags"] == ["L", "0.4"] and d["other"] == {"newkey": "x"}
    assert _preppy_line(_kv("preppy multipv=1 value=0.5"), [], flip=False)["provisional"] is False


def test_bestmove_strings_are_parsed():
    from tnh.web.engine import _kv, _preppy_line, _preppy_root
    kv = _kv("preppy multipv=2 value=0.545 net=- nonobvious=0 reach=0.2 provisional=0 moves=e2e4:L,e7e5:0.6 "
             "mode=bestmove depth=11 complete=1 inside=812 value_in_progress=0.549")
    d = _preppy_line(kv, ["e2e4", "e7e5"], flip=False)
    assert (d["mode"], d["depth"], d["complete"], d["inside"]) == ("bestmove", 11, True, 812)
    assert d["white"] == pytest.approx(0.545) and d["white_in_progress"] == pytest.approx(0.549) and d["other"] == {}
    d = _preppy_line(_kv("preppy multipv=1 value=0.4 mode=bestmove depth=3 complete=0 inside=5 value_in_progress=-"),
                     [], flip=True)
    assert d["white"] == pytest.approx(0.6) and d["in_progress"] is None and d["complete"] is False
    r = _preppy_root(_kv("preppy root value=0.55 mode=bestmove depth=11 complete=1 provisional=0 "
                         "depth_in_progress=12 expanded=3000"), flip=False)
    assert (r["mode"], r["depth"], r["complete"], r["depth_in_progress"], r["expanded"]) == ("bestmove", 11, True, 12, 3000)
    assert r["fix"] is None
    assert _preppy_root(_kv("preppy root value=0.5 mode=repertoire fix=fitted expanded=10"), flip=False)["fix"] == "fitted"


def test_preppy_whitelist():
    from tnh.web.engine import default_specs
    assert {"Blunder", "PriorMin", "MistakeFix", "Mode", "Alpha"} <= set(default_specs()["preppy"].settings)


def test_preppy_fake_bestmove_mode():
    spec = EngineSpec("preppy", "preppy", [sys.executable, "-m", "tnh.engine.uci", "--fake"], {},
                      ("MyElo", "OppElo", "Side", "Price", "Cap", "Obvious", "Prepared", "DrawScore", "Mode", "Alpha",
                       "Blunder", "PriorMin", "MistakeFix"), short="preppy")
    m = EngineManager({"preppy": spec}, start_timeout=120, gpu_free=lambda: None)
    try:
        m.go("preppy", *after(), multipv=2, settings={"Side": "white", "Mode": "bestmove", "Alpha": "0.5"})
        end = time.monotonic() + 60
        s = m.snapshot()
        while time.monotonic() < end and not (s["lines"] and all("preppy" in x for x in s["lines"]) and s.get("root")):
            time.sleep(0.2)
            s = m.snapshot()
        assert {o["name"] for o in s["options"]} >= {"Mode", "Alpha", "Blunder", "PriorMin"}
        mode = next(o for o in s["options"] if o["name"] == "Mode")
        assert mode["type"] == "combo" and set(mode["vars"]) == {"repertoire", "bestmove"}
        x = s["lines"][0]["preppy"]
        assert x["mode"] == "bestmove" and x["depth"] is not None and x["inside"] is not None
        assert s["root"]["mode"] == "bestmove" and s["root"]["depth_in_progress"] is not None
        assert m._applied["Mode"] == "bestmove" and m._applied["Alpha"] == "0.5"
    finally:
        m.close()
