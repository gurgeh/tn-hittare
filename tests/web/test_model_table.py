"""The model columns: the mixes with the database, the rating deltas, caching, batching, supersession."""

from __future__ import annotations

import chess
import pytest
from fastapi.testclient import TestClient

from _webfix import fake_hub, stats
from tnh.models.base import Query
from tnh.web.app import create_app
from tnh.web.fakes import FakeMoveModel, FakeValue
from tnh.web.models import Hub, Superseded, load_move_model, load_value_model, query
from tnh.web.stats import Shrink

START = chess.STARTING_FEN


def fen_after(*uci: str) -> str:
    b = chess.Board()
    for m in uci:
        b.push_uci(m)
    return b.fen()


@pytest.fixture()
def client():
    hub = fake_hub()
    c = TestClient(create_app(stats(), hub))
    c.hub = hub
    return c


def model(client, fen, moves="", **params):
    r = client.get("/api/model", params={"fen": fen, "moves": moves, "white": 2300, "black": 2100, **params})
    assert r.status_code == 200, r.text
    return r.json()


def white_view(wdl, white_to_move):
    w, d, l = wdl
    return (w if white_to_move else l) + d / 2, d


def test_mixes_at_the_start(client):
    d = model(client, START)
    s = Shrink()
    fm = FakeMoveModel().predict([Query(START, [], 2300, 2100)])[0]
    p = dict(zip(fm.moves, [float(x) for x in __import__("numpy").exp(fm.logp)]))
    e4 = d["moves"]["e2e4"]
    assert e4["p_model"] == pytest.approx(p["e2e4"], rel=1e-5)
    assert e4["p_mix"] == pytest.approx((4 + s.kappa * p["e2e4"]) / (8 + s.kappa), rel=1e-5)
    assert e4["p_db"] == pytest.approx(0.5)
    assert e4["delta"] == pytest.approx(e4["p_mix"] - e4["p_model"])

    # model 2 for the position after 1.e4, Black to move, at Black 2100 against White 2300
    b = chess.Board(fen_after("e2e4"))
    wdl = FakeValue().wdl([Query(b.fen(), ["e2e4"], 2100, 2300)])[0]
    score_model, draw_model = white_view(wdl, False)
    assert e4["score_model"] == pytest.approx(score_model)
    assert e4["score_db"] == pytest.approx(0.5)
    assert e4["score_mix"] == pytest.approx((4 * 0.5 + s.kappa_v * score_model) / (4 + s.kappa_v))
    assert e4["draw_mix"] == pytest.approx((4 * 0.5 + s.kappa_v * draw_model) / (4 + s.kappa_v))

    # a move the database lacks: the model's figures alone, its mix from count 0
    a3 = d["moves"]["a2a3"]
    assert a3["n"] == 0 and a3["score_db"] is None
    assert a3["p_mix"] == pytest.approx(s.kappa * a3["p_model"] / (8 + s.kappa))

    # the position's own line
    pos = d["position"]
    wdl0 = FakeValue().wdl([Query(START, [], 2300, 2100)])[0]
    assert pos["score_model"] == pytest.approx(white_view(wdl0, True)[0])
    assert pos["score_mix"] == pytest.approx((8 * (3 + 1.5) / 8 + s.kappa_v * pos["score_model"]) / (8 + s.kappa_v))


def test_listed_moves(client):
    d = model(client, START)
    assert {"e2e4", "d2d4", "c2c4"} <= set(d["listed"])
    p = d["moves"]
    assert all(p[u]["p_model"] >= 0.01 or p[u]["n"] > 0 for u in d["listed"])
    assert d["cost"]["value_queries"] == len(d["listed"]) + 1


def test_cache_and_batches(client):
    rec = client.hub._value
    d1 = model(client, START)
    assert d1["cost"]["value_computed"] == len(d1["listed"]) + 1
    assert max(rec.batches) <= 12
    d2 = model(client, START)
    assert d2["cost"]["value_computed"] == 0 and d2["cost"]["move_model_cached"] is True
    # other ratings: everything again
    r = client.get("/api/model", params={"fen": START, "white": 2000, "black": 2000})
    assert r.json()["cost"]["value_computed"] == len(r.json()["listed"]) + 1
    # a filter changes the database side only: the values are cached
    d3 = model(client, START, tempo="blitz")
    assert d3["cost"]["value_computed"] <= 1


def test_history_reaches_the_models(client):
    d = model(client, fen_after("e2e4", "e7e5"), "e2e4,e7e5")
    assert d["moves"]["g1f3"]["n"] == 2
    r = client.get("/api/model", params={"fen": START, "moves": "e2e4"})
    assert r.status_code == 400


def test_finished_game_is_scored_by_its_result():
    hub = Hub(FakeMoveModel(), FakeValue())
    b = chess.Board()
    for m in ("f2f3", "e7e5", "g2g4", "d8h4"):
        b.push_uci(m)
    assert b.is_checkmate()
    q = query(b, ["f2f3", "e7e5", "g2g4", "d8h4"], 2000, 2000)
    assert hub.values([q]) == [(0.0, 0.0, 1.0)]       # White to move, mated
    assert hub.counts["value_terminal"] == 1 and hub.counts["value_queries"] == 0


def test_superseded_between_batches():
    hub = Hub(FakeMoveModel(), FakeValue())
    b = chess.Board()
    qs = []
    for m in list(b.legal_moves)[:15]:
        c = b.copy()
        c.push(m)
        qs.append(query(c, [m.uci()], 2000, 2000))
    calls = iter([False, True])
    with pytest.raises(Superseded):
        hub.values(qs, cancelled=lambda: next(calls))
    assert hub.counts["value_queries"] == 12           # the first batch is kept
    assert hub.values(qs[:12]) and hub.counts["value_queries"] == 12


def test_model_specs():
    assert load_move_model("fake").name == "fake-move"
    assert load_value_model("import:tnh.web.fakes:FakeValue").name == "fake-value"
    with pytest.raises(ValueError):
        load_move_model("nosuch")
    with pytest.raises(TypeError):
        load_value_model("import:tnh.web.fakes:FakeMoveModel")


def test_hub_by_spec_loads_lazily():
    hub = Hub("fake", "fake")
    assert hub.status()["move_loaded"] is False
    hub.move_probs(START, [], 2000, 2000)
    assert hub.status()["move_loaded"] is True and hub.status()["move_model"] == "fake-move"


# --- the rating columns ----------------------------------------------------------------


def stats_of(client, fen):
    r = client.get("/api/stats", params={"fen": fen})
    assert r.status_code == 200
    return r.json()


def test_white_and_black_deltas(client):
    d = stats_of(client, START)
    k = Shrink().k_elo
    t = d["total"]
    # rated games from the start: e4 x4 (W 2400 2200 2000 2300, B 2200 2200 2300 2300),
    # d4 x1 (2400-2400), c4 x2 (2500-2500, 2400-2400)
    assert t["n_rated"] == 7
    assert t["white_elo"] == pytest.approx(16200 / 7)
    assert t["black_elo"] == pytest.approx(16300 / 7)
    e4 = {m["uci"]: m for m in d["moves"]}["e2e4"]
    assert e4["white_elo"] == pytest.approx(2225) and e4["black_elo"] == pytest.approx(2250)
    assert e4["w_elo_delta"] == pytest.approx(4 / (4 + k) * (2225 - 16200 / 7))
    assert e4["b_elo_delta"] == pytest.approx(4 / (4 + k) * (2250 - 16300 / 7))
    # the alternative form: White moves here, so White is the chooser
    assert e4["mover_elo_delta"] == pytest.approx(e4["w_elo_delta"])
    assert t["gap"] == pytest.approx(16200 / 7 - 16300 / 7)
    assert e4["gap"] == pytest.approx(-25)
    assert e4["gap_delta"] == pytest.approx(4 / (4 + k) * (-25 - t["gap"]))
    assert e4["gap_delta"] == pytest.approx(e4["w_elo_delta"] - e4["b_elo_delta"])


def test_alternative_form_with_black_to_move(client):
    d = stats_of(client, fen_after("e2e4"))
    m = {x["uci"]: x for x in d["moves"]}["e7e5"]
    # Black chooses: the mover's delta is Black's, the gap is Black minus White
    assert m["mover_elo_delta"] == pytest.approx(m["b_elo_delta"])
    assert m["gap"] == pytest.approx(m["black_elo"] - m["white_elo"])
    assert m["gap_delta"] == pytest.approx(m["b_elo_delta"] - m["w_elo_delta"])
    assert d["total"]["gap"] == pytest.approx(d["total"]["black_elo"] - d["total"]["white_elo"])


def test_unrated_moves_have_no_delta(client):
    d = stats_of(client, fen_after("d2d4", "g8f6", "c2c4"))
    for m in d["moves"]:
        if m["n_rated"] == 0:
            assert m["w_elo_delta"] is None and m["gap_delta"] is None
