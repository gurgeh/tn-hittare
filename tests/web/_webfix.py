"""A handful of games with known statistics, and the tables `tnh.data.edges` builds from them."""

from __future__ import annotations

import numpy as np
import polars as pl

from tnh.models.base import MoveValues, Prediction
from tnh.web.stats import Stats

# 1.d4 Nf6 2.c4 and 1.c4 Nf6 2.d4 reach the same position, twice each.
GAMES = [
    # moves_uci,                    result,    year, white, black, tempo
    ("e2e4 e7e5 g1f3 b8c6",          "1-0",     2020, 2400, 2200, "classical"),
    ("e2e4 e7e5 g1f3 g8f6",          "1/2-1/2", 2021, 2200, 2200, "classical"),
    ("e2e4 c7c5 g1f3",               "0-1",     2022, 2000, 2300, "rapid"),
    ("e2e4 e7e5 f1c4",               "1/2-1/2", None, 2300, 2300, "unknown"),
    ("d2d4 g8f6 c2c4 e7e6",          "1-0",     2019, None, None, "blitz"),
    ("d2d4 g8f6 c2c4 e7e6",          "1/2-1/2", 2024, 2400, 2400, "classical"),
    ("c2c4 g8f6 d2d4 e7e6",          "0-1",     2023, 2500, 2500, "classical"),
    ("c2c4 g8f6 d2d4 g7g6",          "1-0",     2024, 2400, 2400, "classical"),
]


def games() -> pl.DataFrame:
    cols = list(zip(*GAMES))
    return pl.DataFrame({
        "moves_uci": list(cols[0]), "result": list(cols[1]),
        "year": pl.Series(cols[2], dtype=pl.Int32),
        "white_elo": pl.Series(cols[3], dtype=pl.Int32), "black_elo": pl.Series(cols[4], dtype=pl.Int32),
        "tempo": list(cols[5]),
    })


def stats() -> Stats:
    from tnh.data.edges import tables_from_games
    edges, index = tables_from_games(games())
    return Stats.from_frames(edges, index, "test fixture")


class FakeHub:
    """Deterministic stand-in for the models: uniform over the legal moves, fixed values."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def maia(self, fen, moves, white_elo, black_elo):
        import chess
        from tnh.web.models import _board
        board = _board(fen, moves)
        self.calls.append(("maia", fen, tuple(moves), white_elo, black_elo))
        legal = [m.uci() for m in board.legal_moves]
        p = Prediction(legal, np.full(len(legal), -np.log(len(legal)), np.float32), (0.4, 0.3, 0.3))
        mover = white_elo if board.turn == chess.WHITE else black_elo
        opp = black_elo if board.turn == chess.WHITE else white_elo
        return {"model": "fake", "mover_elo": mover, "opp_elo": opp, "history": len(moves),
                "moves": {m: float(np.exp(x)) for m, x in zip(p.moves, p.logp)}, "wdl": list(p.wdl), "ms": 0.0}

    def engine(self, fen, moves, nodes):
        from tnh.web.models import _board
        board = _board(fen, moves)
        self.calls.append(("engine", fen, tuple(moves), nodes))
        legal = [m.uci() for m in board.legal_moves]
        mv = MoveValues(legal, np.linspace(0.6, 0.4, len(legal)).astype(np.float32),
                        np.linspace(50, -50, len(legal)).astype(np.float32), np.zeros(len(legal), np.int16), None)
        return {"model": "fake engine", "nodes": nodes, "ms": 0.0, "wdl": None,
                "moves": {m: {"value": float(v), "cp": float(c), "mate": 0}
                          for m, v, c in zip(mv.moves, mv.value, mv.cp)}}

    def status(self):
        return {"fake": True}
