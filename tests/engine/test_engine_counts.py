"""EdgeDB.counts_many (the engine's batched counts) against the one-position reader."""

import chess
import polars as pl
import pytest

from tnh.data.edges import EdgeDB, tables_from_games


def _games(rows):
    return pl.DataFrame(rows, schema={"moves_uci": pl.String, "result": pl.String, "year": pl.Int32,
                                      "white_elo": pl.Int32, "black_elo": pl.Int32, "tempo": pl.String},
                        orient="row")


def test_counts_many_equals_moves_per_position():
    games = _games([("e2e4 e7e5 g1f3 b8c6", "1-0", 2020, 2200, 2200, "classical")] * 6
                   + [("e2e4 c7c5 g1f3 d7d6", "0-1", 2021, 2000, 2100, "classical")] * 4
                   + [("d2d4 d7d5 c2c4", "1/2-1/2", 2019, 1500, 1500, "blitz")] * 5
                   + [("e2e4 e7e5 f1c4", "1-0", 2022, 2400, 2400, "rapid")] * 3)
    db = EdgeDB.from_frames(*tables_from_games(games))
    boards = []
    b = chess.Board()
    boards.append(b.copy())
    b.push_uci("e2e4"); boards.append(b.copy())
    b.push_uci("e7e5"); boards.append(b.copy())
    b2 = chess.Board(); b2.push_uci("a2a3"); boards.append(b2)          # never played
    for filt in ({}, {"tempi": "classical"}, {"elo_bands": (2000, 2200)}, {"years": (2020, 2021), "tempi": ["classical", "rapid"]}):
        many = db.counts_many(boards, **filt)
        for bd in boards:
            one = db.moves(bd, **filt)
            want = dict(zip(one["move"].to_list(), one["n"].to_list()))
            got = many.filter(pl.col("pos_hash") == chess.polyglot.zobrist_hash(bd))
            assert dict(zip(got["move"].to_list(), got["n"].to_list())) == want, (filt, bd.fen())
    assert db.counts_many([]).height == 0
