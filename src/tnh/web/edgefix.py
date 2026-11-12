"""A stopgap for a lookup bug in `tnh.data.edges._Sorted.rows` (as of b3f6925). Delete when fixed.

`_Sorted.rows` reads, for each key, only the last row group whose minimum is at or below
the key. A position whose rows straddle a row-group boundary loses the rows in the
earlier group. On the 2024+ dev table that is 157 positions of 320 boundaries, which lose
2,799 of their 6,258 edge rows, among them a position with 159,246 games; parents.parquet
has 21 such boundaries. `test_server.py` reproduces it with four-row groups.

`patch(db)` replaces the lookup on that one `EdgeDB`'s parquet tables with one that reads
every row group whose [min, max] holds a key. It reads through the `_Sorted` object's own
file handle and statistics; nothing else changes.
"""

from __future__ import annotations

import types

import numpy as np
import polars as pl

from tnh.data import edges


def _rows(self, keys, columns: list[str] | None = None) -> pl.DataFrame:
    k = np.unique(np.asarray(keys, np.uint64))
    # The file is sorted by key, so both the minima and the maxima are non-decreasing.
    lo = np.searchsorted(self.maxs, k, side="left")    # first group whose max >= key
    hi = np.searchsorted(self.mins, k, side="right")   # groups before this have min <= key
    groups = sorted({int(g) for a, b in zip(lo, hi) for g in self.groups[a:b]})
    if not groups:
        t = self.pf.schema_arrow.empty_table()
        return pl.from_arrow(t.select(columns) if columns else t)
    df = pl.from_arrow(self.pf.read_row_groups(groups, columns=columns))
    return df.filter(pl.col(self.key).is_in(pl.Series(k, dtype=pl.UInt64).implode()))


def patch(db: edges.EdgeDB) -> int:
    """Fix the lookups of `db`'s parquet tables in place; returns how many were patched."""
    n = 0
    for attr in ("_edges", "_pos", "_parents"):
        t = getattr(db, attr, None)
        if isinstance(t, edges._Sorted):
            t.rows = types.MethodType(_rows, t)
            n += 1
    return n
