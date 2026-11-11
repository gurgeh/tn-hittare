"""Which position rows count in an evaluation. See "The window" in docs/DESIGN.md.

    positions.filter(in_window(positions))

The expression is evaluated per game (``over("game_id")``), so apply it to
all the rows of each game, before any other filter that drops rows. Dropping
the early plies first would move the out-of-book point.
"""

from __future__ import annotations

import polars as pl

REQUIRED = ("game_id", "ply", "pos_count", "clock_mover_s")


def in_window(positions, *, book_count: int = 5, max_ply: int = 60, min_clock_s: int = 600) -> pl.Expr:
    """Boolean expression: the row is out of book, not too late, not in time trouble.

    - Out of book: the game leaves book at its first ply with
      ``pos_count < book_count``. That ply and every later one count; the move
      that led there (the ply before) does not. A game that never leaves book
      has no row in the window.
    - Not too late: ``ply < max_ply``.
    - Not in time trouble: ``clock_mover_s >= min_clock_s`` where the clock is
      known; rows without a clock rely on ``max_ply`` alone.

    ``positions`` is the frame the expression will be used on (DataFrame or
    LazyFrame). It is only used to check that the needed columns are there.
    """
    names = positions.collect_schema().names() if hasattr(positions, "collect_schema") else list(positions.columns)
    missing = [c for c in REQUIRED if c not in names]
    if missing:
        raise ValueError(f"in_window needs columns {missing}")

    first_rare_ply = (
        pl.when(pl.col("pos_count") < book_count).then(pl.col("ply")).min().over("game_id")
    )
    out_of_book = (pl.col("ply") >= first_rare_ply).fill_null(False)
    not_late = pl.col("ply") < max_ply
    no_time_trouble = pl.col("clock_mover_s").is_null() | (pl.col("clock_mover_s") >= min_clock_s)
    return out_of_book & not_late & no_time_trouble
