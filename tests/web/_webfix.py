"""A handful of games with known statistics, and the tables `tnh.data.edges` builds from them."""

from __future__ import annotations

from pathlib import Path

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


def tables():
    from tnh.data.edges import parents_from_edges, tables_from_games
    edges, index = tables_from_games(games())
    return edges, index, parents_from_edges(edges)


def stats() -> Stats:
    from tnh.data.edges import EdgeDB
    edges, index, parents = tables()
    return Stats(EdgeDB.from_frames(edges, index, parents), "test", why="fixture")


def write_tables(corpus: Path, name: str = "dev") -> None:
    """The fixture tables as parquet files where `EdgeDB(name=...)` looks for them."""
    corpus.mkdir(parents=True, exist_ok=True)
    suffix = "" if name == "full" else f"_{name}"
    for t, df in zip(("edges", "pos_index", "parents"), tables()):
        df.write_parquet(corpus / f"{t}{suffix}.parquet", row_group_size=4, statistics=True)


class RecordingValue:
    """`FakeValue` that records the size of each batch it is asked for."""

    def __init__(self) -> None:
        from tnh.web.fakes import FakeValue
        self.inner = FakeValue()
        self.name = "fake-value"
        self.batches: list[int] = []

    def wdl(self, queries):
        self.batches.append(len(queries))
        return self.inner.wdl(queries)


def fake_hub():
    """The real `Hub` over the fake models."""
    from tnh.web.fakes import FakeMoveModel
    from tnh.web.models import Hub

    return Hub(FakeMoveModel(), RecordingValue())

