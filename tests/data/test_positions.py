"""positions.parquet: which games give rows, pos_count, clocks per row."""

import datetime

import chess
import polars as pl
import pytest

import tnh.data.positions as positions
from tnh.data.parse import GAMES_SCHEMA


def game(gid, moves_san, welo, belo, tempo="classical", clocks=None, result="1-0", base=5400, inc=30, tc="90+30"):
    board = chess.Board()
    ucis = [board.push_san(m).uci() for m in moves_san]
    return {
        "game_id": gid, "source": "test", "tournament_id": "t", "event": "e", "site": "s", "round": "1",
        "date": datetime.date(2025, 1, 1), "year": 2025, "white": "w", "black": "b",
        "white_elo": welo, "black_elo": belo, "white_fide_id": None, "black_fide_id": None,
        "white_title": None, "black_title": None, "result": result, "time_control_raw": tc,
        "tempo": tempo, "tempo_source": "tag", "base_s": base, "inc_s": inc,
        "has_clock": clocks is not None, "n_plies": len(ucis), "moves_uci": " ".join(ucis),
        "clocks_s": clocks, "split": "train",
    }


@pytest.fixture()
def built(tmp_path, monkeypatch):
    monkeypatch.setattr(positions, "CORPUS", tmp_path)
    games = [
        # counted (both >= 1800) and emitted (classical)
        game("g1", ["e4", "e5", "Nf3", "Nc6"], 2000, 1900, clocks=[5420, 5425, 5300, 5000], result="0-1"),
        # counted, not emitted (rapid)
        game("g2", ["e4", "e5", "Nc3"], 2100, 2200, tempo="rapid"),
        # emitted, not counted (white below 1800)
        game("g3", ["e4", "c5"], 1700, 2000),
        # neither: a rating missing
        game("g4", ["e4", "e5"], None, 2000),
        # counted, rapid; reaches g1's final position by transposition
        game("g5", ["Nf3", "Nc6", "e4", "e5"], 1900, 1900, tempo="rapid"),
        # counted and emitted; its ply 4 is g1's and g5's final position
        game("g6", ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"], 2300, 2250, result="1/2-1/2"),
    ]
    pl.DataFrame(games, schema=GAMES_SCHEMA).write_parquet(tmp_path / "games.parquet")
    positions.build(max_ply=80, workers=1, chunk=2)
    return pl.read_parquet(tmp_path / "positions.parquet")


def test_schema_and_rows(built):
    assert built.schema == pl.Schema(positions.POSITIONS_SCHEMA)
    assert sorted(set(built["game_id"])) == ["g1", "g3", "g6"]
    assert built.filter(pl.col("game_id") == "g1")["ply"].to_list() == [0, 1, 2, 3]


def test_pos_count(built):
    pc = {(r["game_id"], r["ply"]): r["pos_count"] for r in built.iter_rows(named=True)}
    # counted games: g1, g2, g5, g6 (g3 has a rating below 1800, g4 lacks one)
    assert pc[("g1", 0)] == 4  # start position
    assert pc[("g1", 1)] == 3  # 1. e4: g1, g2, g6 (g5 plays Nf3 first)
    assert pc[("g1", 2)] == 3  # 1. e4 e5
    assert pc[("g1", 3)] == 2  # 2. Nf3: g1, g6
    assert pc[("g3", 1)] == 3  # g3 reaches 1. e4 but is not counted itself
    assert pc[("g3", 1)] == pc[("g1", 1)]
    # after 2... Nc6: the final position of g1, of g5 (by transposition) and ply 4 of g6
    assert pc[("g6", 4)] == 3
    assert pc[("g6", 5)] == 1


def test_pos_key_ignores_move_counters():
    b1, b2 = chess.Board(), chess.Board()
    for m in ["e4", "e5", "Nf3", "Nc6"]:
        b1.push_san(m)
    for m in ["Nf3", "Nc6", "e4", "e5"]:
        b2.push_san(m)
    assert b1.fen() != b2.fen()  # the half-move clock differs
    assert positions.pos_key(b1) == positions.pos_key(b2)
    # en passant square only when a capture is possible
    b3 = chess.Board()
    b3.push_san("e4")
    assert positions.pos_key(b3).endswith(" -")


def test_row_fields(built):
    g1 = built.filter(pl.col("game_id") == "g1").sort("ply")
    r0, r1, r2, r3 = g1.iter_rows(named=True)
    assert r0["white_to_move"] and not r1["white_to_move"]
    assert (r0["mover_elo"], r0["opp_elo"]) == (2000, 1900)
    assert (r1["mover_elo"], r1["opp_elo"]) == (1900, 2000)
    assert r0["result_mover"] == 0.0 and r1["result_mover"] == 1.0
    assert r0["n_legal"] == 20 and r0["played_uci"] == "e2e4"
    assert r0["fen"] == chess.STARTING_FEN
    # clocks before the move: unknown before each side's first move, then the clock after its previous move
    assert (r0["clock_mover_s"], r0["clock_opp_s"]) == (None, None)
    assert (r1["clock_mover_s"], r1["clock_opp_s"]) == (None, 5420)
    assert (r2["clock_mover_s"], r2["clock_opp_s"]) == (5420, 5425)
    assert (r3["clock_mover_s"], r3["clock_opp_s"]) == (5425, 5300)
    # time spent: 5420 -> 5300 with 30 s increment = 150 s
    assert r2["move_time_s"] == 150
    assert r3["move_time_s"] == 5425 - 5000 + 30
    assert r0["move_time_s"] is None
