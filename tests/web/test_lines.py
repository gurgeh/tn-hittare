"""Saved lines: the API over `tnh.web.lines.LineStore`, always in a temporary directory."""

from __future__ import annotations

import io

import chess
import chess.pgn
import pytest
from fastapi.testclient import TestClient

from _webfix import fake_hub, stats
from tnh import movemix as mm
from tnh.web.app import create_app
from tnh.web.lines import DEFAULT_DIR, LineStore, check_name, slug
from tnh.web.stats import Shrink

PGN = ('{before} 1. e4 {king pawn $1 [%clk 0:01:00]} e5 ( { opens } 1... c5 {Sicilian} 2. Nf3 ) '
       '2. Nf3 Nc6 3. Bb5 *')


@pytest.fixture
def client(tmp_path):
    store = LineStore(tmp_path / "lines")
    assert store.root != DEFAULT_DIR.expanduser()
    c = TestClient(create_app(stats(), fake_hub(), shrink=Shrink(rules={}, prob=mm.Rule()), lines=store))
    c.dir = store.root
    return c


def tree(client, pgn=PGN) -> dict:
    r = client.post("/api/pgn", content=pgn)
    assert r.status_code == 200, r.text
    return r.json()


def strip(nodes):
    return [{"uci": n["uci"], "comment": n["comment"], "pre": n["pre"], "children": strip(n["children"])}
            for n in nodes]


def save(client, name, t, overwrite=False):
    return client.post("/api/lines/save", json={**t, "name": name, "overwrite": overwrite})


def test_save_list_load_round_trip(client):
    t = tree(client)
    assert not client.dir.exists()                 # created on the first save
    r = save(client, "Ruy Lopez: main", t)
    assert r.status_code == 200, r.text
    assert r.json()["file"] == "ruy-lopez-main.pgn"
    text = (client.dir / "ruy-lopez-main.pgn").read_text()
    game = chess.pgn.read_game(io.StringIO(text))  # a plain PGN reader opens it
    assert not game.errors
    assert game.headers["Event"] == "Ruy Lopez: main"
    assert game.headers["Date"].count(".") == 2 and "?" not in game.headers["Date"]
    assert "FEN" not in game.headers and "SetUp" not in game.headers
    flat = " ".join(text.split())
    assert "{ king pawn $1 [%clk 0:01:00] }" in flat and "( { opens } 1... c5 { Sicilian } 2. Nf3 )" in flat

    ls = client.get("/api/lines").json()
    assert ls["dir"] == str(client.dir)
    [e] = ls["lines"]
    assert (e["name"], e["plies"], e["error"]) == ("Ruy Lopez: main", 5, None)
    assert e["preview"] == "1. e4 e5 2. Nf3 Nc6 3. Bb5"
    assert e["saved"]

    d = client.get("/api/lines/load", params={"name": "Ruy Lopez: main"}).json()
    assert d["name"] == "Ruy Lopez: main"
    assert d["comment"] == "before"
    assert strip(d["tree"]) == strip(t["tree"])
    assert d["headers"]["Event"] == "Ruy Lopez: main"


def test_overwrite_needs_the_flag(client):
    t = tree(client)
    assert save(client, "Main", t).status_code == 200
    shorter = tree(client, "1. d4 {queen pawn} *")
    r = save(client, "Main", shorter)
    assert r.status_code == 409 and r.json()["detail"]["code"] == "exists"
    assert "e4" in client.get("/api/lines/load", params={"name": "Main"}).json()["tree"][0]["san"]
    assert save(client, "Main", shorter, overwrite=True).status_code == 200
    d = client.get("/api/lines/load", params={"name": "Main"}).json()
    assert [n["san"] for n in d["tree"]] == ["d4"] and d["tree"][0]["comment"] == "queen pawn"
    assert len(client.get("/api/lines").json()["lines"]) == 1
    assert [p.name for p in client.dir.iterdir()] == ["main.pgn"]   # no temporary file left


def test_delete(client):
    assert save(client, "Gone", tree(client)).status_code == 200
    r = client.delete("/api/lines", params={"name": "Gone"})
    assert r.status_code == 200
    assert client.get("/api/lines").json()["lines"] == []
    r = client.delete("/api/lines", params={"name": "Gone"})
    assert r.status_code == 404 and r.json()["detail"]["code"] == "missing"
    assert client.get("/api/lines/load", params={"name": "Gone"}).status_code == 404


@pytest.mark.parametrize("name", ["", "a/b", "a\\b", ".hidden", "..", " padded", "padded ", "x" * 101,
                                  'say "hi"', "tab\there", "line\nbreak"])
def test_bad_names_refused(client, name):
    r = save(client, name, tree(client))
    assert r.status_code == 400 and r.json()["detail"]["code"] == "invalid", (name, r.text)
    assert not client.dir.exists()


def test_name_rules():
    assert check_name("Sämisch 5...c5 (sharp)") == "Sämisch 5...c5 (sharp)"
    assert slug("Sämisch: 5...c5") == "samisch-5-c5"
    assert slug("中文") == slug("中文") and slug("中文").startswith("line-") and slug("中文") != slug("日本")


def test_slug_collision_refused(client):
    t = tree(client)
    assert save(client, "Najdorf 6.Bg5", t).status_code == 200
    for overwrite in (False, True):
        r = save(client, "najdorf 6 bg5", t, overwrite=overwrite)
        assert r.status_code == 409, r.text
        d = r.json()["detail"]
        assert d["code"] == "collision" and "Najdorf 6.Bg5" in d["message"]
    assert [e["name"] for e in client.get("/api/lines").json()["lines"]] == ["Najdorf 6.Bg5"]
    # a name whose file holds another line is not found under it either
    assert client.get("/api/lines/load", params={"name": "najdorf 6 bg5"}).status_code == 404


def test_non_standard_start(client):
    fen = "8/8/8/4k3/8/8/4K3/7R w - - 0 1"
    t = tree(client, f'[FEN "{fen}"]\n[SetUp "1"]\n\n1. Rh5+ {{check}} Kd4 *')
    assert save(client, "Rook ending", t).status_code == 200
    game = chess.pgn.read_game(io.StringIO((client.dir / "rook-ending.pgn").read_text()))
    assert game.headers["FEN"] == fen and game.headers["SetUp"] == "1"
    d = client.get("/api/lines/load", params={"name": "Rook ending"}).json()
    assert d["root_fen"] == fen and d["tree"][0]["comment"] == "check"
    [e] = client.get("/api/lines").json()["lines"]
    assert (e["plies"], e["preview"]) == (2, "1. Rh5+ Kd4")


def test_unreadable_file_is_listed_with_its_error(client):
    client.dir.mkdir(parents=True)
    (client.dir / "broken.pgn").write_bytes(b"\xff\xfe not utf-8")
    [e] = client.get("/api/lines").json()["lines"]
    assert e["name"] == "broken" and e["error"]


def test_illegal_tree_refused(client):
    r = save(client, "Bad", {"tree": [{"uci": "e2e5", "children": []}]})
    assert r.status_code == 400 and "illegal" in r.json()["detail"]["message"]
    assert not client.dir.exists() or not any(client.dir.iterdir())


def test_no_store_no_lines():
    c = TestClient(create_app(stats(), fake_hub()))
    assert c.get("/api/lines").status_code == 404
