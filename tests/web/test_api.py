"""The JSON API against tables built by `tnh.data.edges.tables_from_games` from a handful of games."""

from __future__ import annotations

import chess
import pytest
from fastapi.testclient import TestClient

from _webfix import fake_hub, stats
from tnh.web.app import create_app
from tnh import movemix as mm
from tnh.web.stats import Shrink
from tnh.web.stats import Filters, two_proportion

START = chess.STARTING_FEN


def fen_after(*uci: str) -> str:
    b = chess.Board()
    for m in uci:
        b.push_uci(m)
    return b.fen()


@pytest.fixture(scope="module")
def client():
    hub = fake_hub()
    c = TestClient(create_app(stats(), hub, shrink=Shrink(rules={}, prob=mm.Rule())))   # the fixed fallback rules
    c.hub = hub
    return c


def get(client, fen, **params):
    r = client.get("/api/stats", params={"fen": fen, **params})
    assert r.status_code == 200, r.text
    return r.json()


def by_move(d):
    return {m["uci"]: m for m in d["moves"]}


def test_counts_and_white_score(client):
    d = get(client, START)
    assert d["white_to_move"] is True
    m = by_move(d)
    assert {k: v["n"] for k, v in m.items()} == {"e2e4": 4, "d2d4": 2, "c2c4": 2}
    assert d["status"] == "ok"
    assert d["total"]["played"] == 8
    assert d["total"]["reach"] == 8
    # e4: 1-0, 1/2, 0-1, 1/2 -> White scores 2/4
    assert m["e2e4"]["score"] == pytest.approx(0.5)
    assert m["e2e4"]["draw"] == pytest.approx(0.5)
    assert m["e2e4"]["share"] == pytest.approx(0.5)
    assert m["e2e4"]["san"] == "e4"
    assert m["e2e4"]["last_year"] == 2022
    # c4: 0-1, 1-0
    assert m["c2c4"]["score"] == pytest.approx(0.5)
    # the moves' own averages over rated games: d4 has one unrated game and one 2400-2400 game
    assert m["d2d4"]["elo"] == pytest.approx(2400)
    assert m["d2d4"]["opp_elo"] == pytest.approx(2400)
    # totals: 3 white wins, 3 draws, 2 black wins
    t = d["total"]
    assert (t["white_wins"], t["draws"], t["black_wins"]) == (3, 3, 2)
    assert t["score"] == pytest.approx((3 + 1.5) / 8)


def test_black_to_move_scores_for_black(client):
    d = get(client, fen_after("e2e4"))
    assert d["white_to_move"] is False
    m = by_move(d)
    assert m["e7e5"]["n"] == 3 and m["c7c5"]["n"] == 1
    # e5: 1-0, 1/2, 1/2 -> Black scores 1/3; c5: 0-1 -> Black scores 1
    assert m["e7e5"]["score"] == pytest.approx(1 / 3)
    assert m["c7c5"]["score"] == pytest.approx(1.0)
    assert m["e7e5"]["elo"] == pytest.approx((2200 + 2200 + 2300) / 3)   # the mover is Black
    assert m["e7e5"]["opp_elo"] == pytest.approx((2400 + 2200 + 2300) / 3)


def test_filters(client):
    m = by_move(get(client, START, tempo="classical"))
    assert {k: v["n"] for k, v in m.items()} == {"e2e4": 2, "d2d4": 1, "c2c4": 2}
    assert m["e2e4"]["score"] == pytest.approx(0.75)

    m = by_move(get(client, START, y0=2021, y1=2022))
    assert {k: v["n"] for k, v in m.items()} == {"e2e4": 2}

    # a year limit drops the game of unknown year
    m = by_move(get(client, START, y0=1900))
    assert m["e2e4"]["n"] == 3

    # mean rating band: 2400-2499 and up -> the three 2400/2500 games
    m = by_move(get(client, START, e0=2400))
    assert {k: v["n"] for k, v in m.items()} == {"d2d4": 1, "c2c4": 2}
    # below 2200 band edge: the 2000/2300 game (mean 2150, band 2000) only; unrated dropped
    m = by_move(get(client, START, e1=2200))
    assert {k: v["n"] for k, v in m.items()} == {"e2e4": 1}

    m = by_move(get(client, START, tempo="blitz,rapid"))
    assert {k: v["n"] for k, v in m.items()} == {"e2e4": 1, "d2d4": 1}

    # all tempi is the same as no filter
    assert get(client, START, tempo="classical,rapid,blitz,unknown")["total"]["played"] == 8

    # filters that leave nothing: said so, the unfiltered count kept
    d = get(client, START, y0=2025, tempo="blitz")
    assert d["status"] == "filtered" and d["moves"] == [] and d["total"]["reach"] == 8


def test_filters_bad_tempo(client):
    r = client.get("/api/stats", params={"fen": START, "tempo": "bullet"})
    assert r.status_code == 400


def test_transposition(client):
    fen = fen_after("d2d4", "g8f6", "c2c4")
    assert fen_after("c2c4", "g8f6", "d2d4").split()[:4] == fen.split()[:4]
    d = get(client, fen)
    assert d["total"]["reach"] == 4 and d["total"]["played"] == 4
    assert by_move(d)["e7e6"]["n"] == 3
    orders = {o["san"]: o["n"] for o in d["move_orders"]}
    assert orders == {"c4": 2, "d4": 2}
    # reach of a child counts every move order into it
    d = get(client, fen_after("d2d4", "g8f6"))
    assert by_move(d)["c2c4"]["reach"] == 4


def test_unknown_position(client):
    d = get(client, "8/8/8/4k3/8/8/4K3/7R w - - 0 1")
    assert d["status"] == "none"
    assert d["moves"] == [] and d["total"]["played"] is None and d["total"]["reach"] is None
    assert d["move_orders"] == [] and d["years"] == []


def test_position_where_every_game_ended(client):
    # 1.d4 Nf6 2.c4 e6 (and its transposition) is where three games stop
    d = get(client, fen_after("d2d4", "g8f6", "c2c4", "e7e6"))
    assert d["status"] == "ended" and d["total"]["reach"] == 3 and d["moves"] == []


def test_bad_fen(client):
    assert client.get("/api/stats", params={"fen": "not a fen"}).status_code == 400


def test_year_series(client):
    d = get(client, START)
    assert d["years"] == [2019, 2020, 2021, 2022, 2023, 2024]
    assert d["year_totals"] == [1, 1, 1, 1, 1, 2]
    assert by_move(d)["e2e4"]["series"] == [0, 1, 1, 1, 0, 0]
    assert d["trend"]["tested"] is False   # far too few games


def test_two_proportion():
    z, p = two_proportion(30, 100, 10, 100)
    assert z > 3 and p < 0.001
    z, p = two_proportion(10, 100, 10, 100)
    assert z == 0 and p == pytest.approx(1)


def test_filters_to_edgedb():
    assert Filters().edgedb() == {}
    assert Filters(2000, None, 2200, 2600, ("classical",)).edgedb() == {
        "years": (2000, 32767), "elo_bands": (2200, 2599), "tempi": ["classical"]}
    assert Filters(None, 2010).edgedb() == {"years": (1, 2010)}


def test_pgn_with_variations(client):
    pgn = '[Event "x"]\n\n1. e4 e5 (1... c5 2. Nf3) 2. Nf3 Nc6 *'
    r = client.post("/api/pgn", content=pgn, headers={"Content-Type": "text/plain"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["root_fen"] == START
    e4 = d["tree"][0]
    assert e4["san"] == "e4"
    assert [c["san"] for c in e4["children"]] == ["e5", "c5"]
    assert e4["children"][1]["children"][0]["uci"] == "g1f3"


def test_pgn_from_fen_tag(client):
    pgn = '[FEN "8/8/8/4k3/8/8/4K3/7R w - - 0 1"]\n[SetUp "1"]\n\n1. Rh5+ *'
    d = client.post("/api/pgn", content=pgn).json()
    assert d["root_fen"].startswith("8/8/8/4k3")
    assert d["tree"][0]["uci"] == "h1h5"


def test_pgn_illegal(client):
    r = client.post("/api/pgn", content="1. e4 e4 *")
    assert r.status_code == 400


COMMENTED = ('{before the first move} 1. e4 {main $1 [%clk 0:01:00] [%cal Ge2e4]} e5 '
             '( { opens the line } 1... c5 {Sicilian} 2. Nf3 {in the variation} ) 2. Nf3 *')


def _strip(nodes):
    """The tree without san, as the page sends it back."""
    return [{"uci": n["uci"], "comment": n["comment"], "pre": n["pre"], "children": _strip(n["children"])}
            for n in nodes]


def test_pgn_comments_parsed_as_text(client):
    d = client.post("/api/pgn", content=COMMENTED).json()
    assert d["comment"] == "before the first move"
    e4 = d["tree"][0]
    # NAGs and [%...] commands inside a comment stay text
    assert e4["comment"] == "main $1 [%clk 0:01:00] [%cal Ge2e4]"
    e5, c5 = e4["children"]
    assert (e5["comment"], e5["pre"]) == ("", "")
    assert (c5["san"], c5["comment"], c5["pre"]) == ("c5", "Sicilian", "opens the line")
    assert c5["children"][0]["comment"] == "in the variation"


def test_pgn_export_round_trip(client):
    d = client.post("/api/pgn", content=COMMENTED).json()
    r = client.post("/api/pgn/export", json=d)
    assert r.status_code == 200, r.text
    pgn = r.json()["pgn"]
    assert "{ main $1 [%clk 0:01:00] [%cal Ge2e4] }" in pgn
    assert "( { opens the line } 1... c5 { Sicilian } 2. Nf3 { in the variation } )" in pgn
    assert pgn.index("{ before the first move }") < pgn.index("1. e4")
    back = client.post("/api/pgn", content=pgn).json()
    assert back["comment"] == d["comment"]
    assert _strip(back["tree"]) == _strip(d["tree"])


def test_pgn_export_closing_brace_and_fen(client):
    fen = "8/8/8/4k3/8/8/4K3/7R w - - 0 1"
    body = {"root_fen": fen, "headers": {"White": "Me", "FEN": "stale", "SetUp": "1"},
            "tree": [{"uci": "h1h5", "comment": "check } {really}", "children": []}]}
    pgn = client.post("/api/pgn/export", json=body).json()["pgn"]
    assert f'[FEN "{fen}"]' in pgn and '[White "Me"]' in pgn and "stale" not in pgn
    assert "{ check ) {really) }" in pgn
    back = client.post("/api/pgn", content=pgn).json()
    assert back["root_fen"] == fen
    assert back["tree"][0]["comment"] == "check ) {really)"


def test_pgn_export_refuses_bad_trees(client):
    assert client.post("/api/pgn/export", json={"tree": [{"uci": "e2e5", "children": []}]}).status_code == 400
    assert client.post("/api/pgn/export", json={"tree": [{"san": "e4"}]}).status_code == 400
    assert client.post("/api/pgn/export", json={"root_fen": "nonsense"}).status_code == 400
    assert client.post("/api/pgn/export", content="not json").status_code == 400
    # an empty tree is a valid game
    assert client.post("/api/pgn/export", json={}).json()["pgn"].rstrip().endswith("*")


def test_models_off():
    from tnh.web.models import NoModels
    c = TestClient(create_app(stats(), NoModels()))
    assert c.get("/api/info").json()["models"] == {"disabled": True}
    r = c.get("/api/model", params={"fen": START})
    assert r.status_code == 503 and "--no-models" in r.json()["detail"]
    assert c.get("/api/stats", params={"fen": START}).status_code == 200


def test_trend_split_fits_short_tables():
    from tnh.web.stats import TrendRule
    r = TrendRule()
    assert r.recent(2024, 2026) == 1      # the 2024+ dev table: 2026 against 2024-2025
    assert r.recent(1998, 2026) == 3
    assert r.recent(2025, 2026) == 1
