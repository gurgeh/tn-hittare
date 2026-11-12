"""The repertoire layer, on a small run written by `tnh.builder.export` (not a hand-made file)."""

from __future__ import annotations

import chess
import pytest
from fastapi.testclient import TestClient

from _webfix import fake_hub, stats, write_run
from tnh.web.app import create_app
from tnh.web.repertoire import Store
from tnh import movemix as mm
from tnh.web.stats import Shrink



@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("repertoires")
    res, sols = write_run(d)
    return d, res, sols


@pytest.fixture(scope="module")
def client(run_dir):
    return TestClient(create_app(stats(), fake_hub(), shrink=Shrink(rules={}, prob=mm.Rule()), runs=Store(run_dir[0])))


def root_fen():
    b = chess.Board()
    b.push_uci("e2e4")
    b.push_uci("e7e5")
    return b


def test_list_and_summary(client, run_dir):
    _, res, sols = run_dir
    assert client.get("/api/runs").json()["runs"] == ["t"]
    s = client.get("/api/run", params={"name": "t"}).json()
    assert s["budgets"] == [1, 3] and s["me"] == "white"
    for N in (1, 3):
        assert s["solutions"][str(N)]["value"] == pytest.approx(sols[N].value, abs=1e-6)
        # one learned line per learned position, ordered by reach, each from the start
        lines = s["lines"][str(N)]
        assert len(lines) == len(sols[N].learned)
        assert [x["reach"] for x in lines] == sorted((x["reach"] for x in lines), reverse=True)
        assert all(x["moves"][:2] == ["e2e4", "e7e5"] for x in lines)
    assert {f["uci"] for f in s["forced"]} == {f["root_move"] for f in res["forced"]}
    assert client.get("/api/run", params={"name": "nosuch"}).status_code == 404


def test_position_at_the_root(client, run_dir):
    _, res, sols = run_dir
    b = root_fen()
    for N in (1, 3):
        d = client.get("/api/run/position", params={"name": "t", "budget": N, "fen": b.fen()}).json()
        assert d["in_graph"] and d["me_to_move"] and d["state"] == "expanded"
        rec = res["nodes"][str(chess.polyglot.zobrist_hash(b))]
        assert d["learned"] == rec["budgets"][str(N)]["learned"]
        assert d["value"] == pytest.approx(rec["budgets"][str(N)]["value"])   # me = White: unchanged
        assert d["reach"] == pytest.approx(1.0)
        mv = d["moves"]["g1f3"]
        assert {"loss", "sound", "obvious", "cost", "value_after", "p_model", "flags"} <= set(mv)


def test_a_learned_line_leads_to_its_learned_move(client):
    s = client.get("/api/run", params={"name": "t"}).json()
    line = s["lines"]["3"][-1]
    b = chess.Board()
    for m in line["moves"][:-1]:
        b.push_uci(m)
    d = client.get("/api/run/position", params={"name": "t", "budget": 3, "fen": b.fen()}).json()
    assert d["in_graph"] and d["learned"] == line["moves"][-1]


def test_opponent_position_and_outside(client, run_dir):
    _, res, _ = run_dir
    opp = next(r for r in res["nodes"].values() if r["to_move"] == "opp" and r["state"] == "expanded")
    b = chess.Board()
    b.set_epd(opp["epd"])
    d = client.get("/api/run/position", params={"name": "t", "budget": 1, "fen": b.fen()}).json()
    assert d["in_graph"] and not d["me_to_move"] and d["learned"] is None
    assert all("p_mixed" in m and "loss" not in m for m in d["moves"].values())
    d = client.get("/api/run/position", params={"name": "t", "budget": 1, "fen": chess.STARTING_FEN}).json()
    assert d == {"in_graph": False}
    r = client.get("/api/run/position", params={"name": "t", "budget": 2, "fen": chess.STARTING_FEN})
    assert r.status_code == 400


def test_black_runs_turn_values_to_white(run_dir, tmp_path):
    import json
    import shutil
    d, res, _ = run_dir
    shutil.copytree(d / "t", tmp_path / "b")
    r = json.loads((tmp_path / "b" / "result.json").read_text())
    r["meta"]["me"] = "black"
    (tmp_path / "b" / "result.json").write_text(json.dumps(r))
    run = Store(tmp_path).get("b")
    rec = res["nodes"][str(chess.polyglot.zobrist_hash(root_fen()))]
    assert run.position(root_fen(), 1)["value"] == pytest.approx(1 - rec["budgets"]["1"]["value"])


def test_pgn_comments_reach_the_page(client, run_dir):
    text = (run_dir[0] / "t" / "repertoire_N3.pgn").read_text()
    tree = client.post("/api/pgn", content=text).json()["tree"]

    def comments(nodes):
        for n in nodes:
            if n["comment"]:
                yield n["comment"]
            yield from comments(n["children"])
    cs = list(comments(tree))
    assert cs and all(c.startswith("reach ") for c in cs)
