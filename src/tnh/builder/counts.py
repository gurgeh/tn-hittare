"""Where the opponent's database counts come from.

`CountSource.counts(key, moves)` gives the number of games that played each of
`moves` from the position `key` (`chess.polyglot.zobrist_hash`, the `pos_hash` of
the position statistics). None when the position is not in the source.

`EdgeDBCounts` wraps `tnh.data.edges.EdgeDB`, the read API of the position
statistics; tests build one with `EdgeDB.from_frames(*tables_from_games(games))`.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Protocol

import numpy as np


class CountSource(Protocol):
    def counts(self, key: Hashable, moves: list[str]) -> np.ndarray | None: ...


class NoCounts:
    def counts(self, key: Hashable, moves: list[str]) -> np.ndarray | None:
        return None


class DictCounts:
    """`{key: {move: games}}`, for tests."""

    def __init__(self, table: Mapping[Hashable, Mapping[str, int]]) -> None:
        self.table = table

    def counts(self, key: Hashable, moves: list[str]) -> np.ndarray | None:
        row = self.table.get(key)
        if row is None:
            return None
        unknown = set(row) - set(moves)
        if unknown:
            raise ValueError(f"counts for moves that are not legal at {key}: {sorted(unknown)}")
        return np.array([row.get(m, 0) for m in moves], dtype=np.float64)


class EdgeDBCounts:
    """Counts from `tnh.data.edges.EdgeDB`, filtered to the opponent's level.

    The filters are a parameter of the opponent model, not a global: the games that
    say what he plays are the games at his level (`elo_bands`, `years`, `tempi` as
    `EdgeDB.moves` takes them). A position below two games is not in the tables and
    gives None, which is where the model alone decides.
    """

    def __init__(self, db, *, years=None, elo_bands=None, tempi=None) -> None:
        self.db = db
        self.filters = {"years": years, "elo_bands": elo_bands, "tempi": tempi}
        self.lookups = 0

    def counts(self, key: Hashable, moves: list[str]) -> np.ndarray | None:
        self.lookups += 1
        rows = self.db.moves(key, **self.filters)
        if rows.height == 0:
            return None
        table = dict(zip(rows["move"].to_list(), rows["n"].to_list()))
        return DictCounts({key: table}).counts(key, moves)
