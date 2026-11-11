"""Position statistics: hashing, counting, the two-game filter, the reader, batch invariance."""

import random

import chess
import chess.polyglot
import polars as pl
import pytest

import tnh.data.edges as edges_mod
from tnh.data.edges import EdgeDB, game_events, move_code, pos_hash, tables_from_games

SPECIAL = [
    # castling both sides, both colours
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 e1g1 d7d6 d2d3 c8g4 b1c3 d8d7 c1e3 e8c8",
    # en passant capture, and a double step with no pawn ready to capture
    "e2e4 a7a6 e4e5 d7d5 e5d6 c7c6 a2a4 b7b5 a4b5 a6b5",
    # promotion with capture, underpromotion
    "a2a4 b7b5 a4b5 a7a6 b5a6 c8b7 a6b7 b8c6 b7a8n d8a8",
    # losing castling rights by moving the rook, a rook capture on its home square
    "g2g3 b7b6 f1g2 c8b7 g1f3 b7f3 g2f3 a7a6 f3a8 h7h6 h1g1 h6h5",
]


def random_games(n: int, seed: int = 7, plies: int = 120) -> list[str]:
    rng = random.Random(seed)
    games = []
    for _ in range(n):
        b = chess.Board()
        ucis = []
        for _ in range(plies):
            moves = list(b.legal_moves)
            if not moves:
                break
            m = rng.choice(moves)
            ucis.append(m.uci())
            b.push(m)
        games.append(" ".join(ucis))
    return games


def games_frame(moves: list[str], results=None, years=None, welo=None, belo=None, tempi=None) -> pl.DataFrame:
    n = len(moves)
    return pl.DataFrame({
        "moves_uci": moves,
        "result": results or ["1-0"] * n,
        "year": years or [2024] * n,
        "white_elo": welo or [2000] * n,
        "black_elo": belo or [2000] * n,
        "tempo": tempi or ["classical"] * n,
    }, schema_overrides={"year": pl.Int32, "white_elo": pl.Int32, "black_elo": pl.Int32})


@pytest.mark.parametrize("moves", SPECIAL + random_games(30))
def test_incremental_hash_matches_polyglot(moves):
    reached, e_pos, e_move, e_child, e_white = game_events(moves, max_ply=200)
    board = chess.Board()
    expected = []
    for uci in moves.split():
        h = chess.polyglot.zobrist_hash(board)
        m = chess.Move.from_uci(uci)
        assert m in board.legal_moves, uci
        board.push(m)
        expected.append((h, move_code(m), chess.polyglot.zobrist_hash(board)))
    got = list(zip(e_pos, e_move, e_child, strict=True))
    # game_events lists each (position, move) once per game, at its first occurrence
    first = {}
    for h, c, ch in expected:
        first.setdefault((h, c), (h, c, ch))
    assert got == list(first.values())
    assert set(reached) == {h for h, _, _ in expected} | {expected[-1][2]}


def test_transpositions_share_a_hash():
    a, b = chess.Board(), chess.Board()
    for m in ["e4", "e5", "Nf3", "Nc6"]:
        a.push_san(m)
    for m in ["Nf3", "Nc6", "e4", "e5"]:
        b.push_san(m)
    assert pos_hash(a) == pos_hash(b)
    e, p = tables_from_games(games_frame(["e2e4 e7e5 g1f3 b8c6 f1b5", "g1f3 b8c6 e2e4 e7e5 f1c4"]))
    row = p.filter(pl.col("pos_hash") == pos_hash(a))
    assert row["n"].item() == 2
    assert sorted(e.filter(pl.col("pos_hash") == pos_hash(a))["move"]) == ["f1b5", "f1c4"]


def test_counts_and_results_from_each_side():
    games = games_frame(
        ["e2e4 e7e5 g1f3", "e2e4 e7e5 b1c3", "e2e4 c7c5", "d2d4 d7d5"],
        results=["1-0", "0-1", "1/2-1/2", "1-0"],
        welo=[2200, 2000, 1800, None], belo=[2100, 2300, 1900, 2000],
    )
    e, p = tables_from_games(games)
    # every row's n is the sum of its results
    assert (e["n"] == e["white_wins"] + e["draws"] + e["black_wins"]).all()
    db = EdgeDB.from_frames(e, p)
    start = db.moves(chess.Board())
    e4 = start.filter(pl.col("move") == "e2e4").row(0, named=True)
    assert (e4["n"], e4["white_wins"], e4["draws"], e4["black_wins"]) == (3, 1, 1, 1)
    assert e4["score"] == 0.5
    assert e4["mover_elo"] == pytest.approx((2200 + 2000 + 1800) / 3)
    assert e4["opp_elo"] == pytest.approx((2100 + 2300 + 1900) / 3)
    d4 = start.filter(pl.col("move") == "d2d4").row(0, named=True)
    assert d4["n"] == 1 and d4["mover_elo"] is None  # white's rating missing
    # after 1. e4, black to move: the score is black's
    b = chess.Board()
    b.push_san("e4")
    after = db.moves(b)
    e5 = after.filter(pl.col("move") == "e7e5").row(0, named=True)
    assert (e5["n"], e5["score"]) == (2, 0.5)
    c5 = after.filter(pl.col("move") == "c7c5").row(0, named=True)
    assert c5["score"] == 0.5
    assert e5["mover_elo"] == pytest.approx((2100 + 2300) / 2)
    # the sums of a move's rows over all years, bands and tempi are the games that played it
    g = e.filter((pl.col("pos_hash") == pos_hash(chess.Board())) & (pl.col("move") == "e2e4"))
    assert g["n"].sum() == 3 and g["n_rated"].sum() == 3
    assert dict(g.select("elo_band", "n").iter_rows()) == {2000: 2, 1800: 1}  # means 2150, 2150, 1850


def test_two_game_filter():
    e, p = tables_from_games(games_frame(["e2e4 e7e5 g1f3", "e2e4 c7c5 g1f3"]))
    hashes = set(p["pos_hash"])
    b = chess.Board()
    assert pos_hash(b) in hashes
    b.push_san("e4")
    assert pos_hash(b) in hashes  # both games
    b.push_san("e5")
    assert pos_hash(b) not in hashes  # one game
    assert set(e["pos_hash"]) <= hashes  # no row for a position below two games
    assert p["n"].min() >= 2


def test_repetition_counts_once_per_game():
    shuffle = "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8"
    e, p = tables_from_games(games_frame([shuffle, "g1f3 g8f6"]))
    start = p.filter(pl.col("pos_hash") == pos_hash(chess.Board()))
    assert start["n"].item() == 2
    nf3 = e.filter((pl.col("pos_hash") == pos_hash(chess.Board())) & (pl.col("move") == "g1f3"))
    assert nf3["n"].sum() == 2


def test_filters():
    games = games_frame(
        ["e2e4", "e2e4", "d2d4", "e2e4"],
        years=[2023, 2024, 2024, None],
        welo=[2400, 2000, 1500, 2000], belo=[2400, 2000, 1500, None],
        tempi=["classical", "classical", "rapid", "blitz"],
    )
    db = EdgeDB.from_frames(*tables_from_games(games))
    start = chess.Board()
    assert db.moves(start)["n"].sum() == 4
    assert db.moves(start, years=2024)["n"].sum() == 2
    assert db.moves(start, years=(2023, 2024))["n"].sum() == 3
    assert db.moves(start, years=0)["n"].sum() == 1  # unknown year
    assert db.moves(start, elo_bands=[2400])["n"].sum() == 1
    assert db.moves(start, elo_bands=0)["n"].sum() == 1  # a rating missing
    assert db.moves(start, tempi="rapid")["move"].to_list() == ["d2d4"]
    assert db.moves(start, tempi=["classical", "blitz"])["n"].sum() == 3
    ch = db.children(start)
    assert dict(ch.select("move", "n").iter_rows()) == {"e2e4": 3, "d2d4": 1}
    assert ch.filter(pl.col("move") == "e2e4")["child_n"].item() == 3
    assert ch.filter(pl.col("move") == "d2d4")["child_n"].item() is None  # one game
    assert db.total(start)["n"] == 4
    assert db.total(chess.STARTING_FEN)["n"] == 4


@pytest.mark.parametrize("batch_games", [1, 3, 1000])
def test_batched_build_matches_in_memory(tmp_path, monkeypatch, batch_games):
    moves = random_games(10, seed=3, plies=20) + ["e2e4 e7e5 g1f3"] * 3 + ["d2d4 d7d5"] * 2
    n = len(moves)
    games = games_frame(moves, results=["1-0", "0-1", "1/2-1/2"] * (n // 3) + ["1-0"] * (n % 3),
                        years=[2020 + i % 3 for i in range(n)]).with_columns(
        game_id=pl.Series([f"{i:016x}" for i in range(n)]))
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    games.write_parquet(corpus / "games.parquet", row_group_size=4)
    monkeypatch.setattr(edges_mod, "CORPUS", corpus)
    monkeypatch.setattr(edges_mod, "DATA_ROOT", tmp_path)
    edges_mod.build(name="t", out_dir=tmp_path / "out", fraction=None, since_year=None,
                    batch_games=batch_games, workers=1)
    e2 = pl.read_parquet(tmp_path / "out" / "edges_t.parquet")
    p2 = pl.read_parquet(tmp_path / "out" / "pos_index_t.parquet")
    e, p = tables_from_games(games)
    assert e2.equals(e)
    assert p2.equals(p)


def test_moves_by_year_and_last_year():
    games = games_frame(["e2e4", "e2e4", "d2d4", "e2e4", "d2d4"], years=[2022, 2024, 2024, None, 2021])
    db = EdgeDB.from_frames(*tables_from_games(games))
    m = db.moves(chess.Board())
    assert dict(m.select("move", "last_year").iter_rows()) == {"e2e4": 2024, "d2d4": 2024}
    y = db.moves_by_year(chess.Board())
    assert y.select("move", "year", "n").rows() == [
        ("e2e4", 0, 1), ("d2d4", 2021, 1), ("e2e4", 2022, 1), ("d2d4", 2024, 1), ("e2e4", 2024, 1)]
    assert y.filter(pl.col("year") == 2024)["share"].to_list() == [0.5, 0.5]
    assert db.moves_by_year(chess.Board(), years=(2022, 2024))["n"].sum() == 3
    only_unknown = db.moves(chess.Board(), years=0)
    assert only_unknown["last_year"].to_list() == [None]
