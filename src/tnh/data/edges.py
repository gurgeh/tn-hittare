"""Position statistics: ``corpus/edges.parquet`` and ``corpus/pos_index.parquet``.

    uv run python -m tnh.data.edges [--fraction F | --since-year Y] [--name NAME] [--workers 2]

The contract is "Position statistics" in docs/DESIGN.md. In short: one row
per (position, move, year, rating band, tempo) with game counts, results and
rating sums, over every kept game and plies below 80, for positions reached by
at least two games; and one row per such position with its total.

A position is the 64-bit polyglot Zobrist hash (``chess.polyglot.zobrist_hash``).
It is computed incrementally here, about twice as fast as hashing each board
from scratch; ``tests/data/test_edges.py`` checks it against python-chess on
every ply of a set of games.

What counts, per game:

- every position before plies 0 .. min(n_plies, 80) - 1, and the position
  after the last of those moves, once per game however often it recurs; that
  is the position's ``n``;
- every (position, move) at plies below 80, once per game.

Three tables: ``edges``, ``pos_index``, and ``parents`` (child_hash,
pos_hash, move, n; the edges summed over years, bands and tempi, sorted by
the child, for the positions leading into a position).

Two ways to build:

- ``tables_from_games(games)`` builds both tables in memory from a small frame
  of games. For test fixtures in other modules; it is the same aggregation
  as the full build, on one batch.
- ``build(...)`` (the command) works through ``games.parquet`` in batches,
  twice. Pass A counts how many games reach each position, written in 64
  buckets on the top six bits of the hash, and merges the counts per bucket
  into the set of positions with at least two games. Pass B replays the
  games again and aggregates edges, totals and EPDs for those positions only,
  again in buckets, which are then merged. Memory is set by the batch size and
  one bucket, not by the corpus; disk by the kept rows, since almost every
  position past the opening is reached by one game and never written in pass
  B. Bucket order is hash order, so the outputs are sorted by ``pos_hash``
  (``parents.parquet`` by ``child_hash``), which the readers' lookups rely on.

Reading: ``EdgeDB`` (moves from a position under filters, a position's total,
the counts of its children).
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import shutil
import time
from collections.abc import Iterator
from pathlib import Path

import chess
import chess.polyglot
import numpy as np
import polars as pl
import pyarrow.parquet as pq

from tnh.paths import CORPUS, DATA_ROOT

MAX_PLY = 80
MIN_GAMES = 2
N_BUCKETS = 64  # top six bits of the hash
BUCKET_SHIFT = 58
TEMPI = ("classical", "rapid", "blitz", "unknown")
TEMPO_CODE = {t: i for i, t in enumerate(TEMPI)}
RESULT_CODE = {"1-0": 0, "1/2-1/2": 1, "0-1": 2}
GAME_COLUMNS = ["moves_uci", "result", "year", "white_elo", "black_elo", "tempo"]

EDGES_SCHEMA = {
    "pos_hash": pl.UInt64,
    "move": pl.String,
    "child_hash": pl.UInt64,
    "year": pl.Int16,
    "elo_band": pl.Int16,
    "tempo": pl.String,
    "n": pl.UInt32,
    "white_wins": pl.UInt32,
    "draws": pl.UInt32,
    "black_wins": pl.UInt32,
    "sum_mover_elo": pl.UInt64,
    "sum_opp_elo": pl.UInt64,
    "n_rated": pl.UInt32,
}
PARENTS_SCHEMA = {
    "child_hash": pl.UInt64,
    "pos_hash": pl.UInt64,
    "move": pl.String,
    "n": pl.UInt32,
}
POS_INDEX_SCHEMA = {
    "pos_hash": pl.UInt64,
    "epd": pl.String,
    "n": pl.UInt32,
    "white_wins": pl.UInt32,
    "draws": pl.UInt32,
    "black_wins": pl.UInt32,
}

# --- hashing -----------------------------------------------------------------

_Z = chess.polyglot.POLYGLOT_RANDOM_ARRAY
_HASHER = chess.polyglot.ZobristHasher(_Z)


def _piece_key(piece_type: int, color: bool, square: int) -> int:
    return _Z[64 * ((piece_type - 1) * 2 + (1 if color else 0)) + square]


def _rest_key(board: chess.Board) -> int:
    return _HASHER.hash_castling(board) ^ _HASHER.hash_ep_square(board) ^ _HASHER.hash_turn(board)


def pos_hash(board: chess.Board) -> int:
    """The position key used in edges.parquet: ``chess.polyglot.zobrist_hash``."""
    return chess.polyglot.zobrist_hash(board)


# moves as 15-bit codes while aggregating: from (6 bits), to (6 bits), promotion piece (3 bits)
def move_code(move: chess.Move) -> int:
    return move.from_square | (move.to_square << 6) | ((move.promotion or 0) << 12)


_CODE_TO_UCI = [
    chess.Move(c & 63, (c >> 6) & 63, (c >> 12) or None).uci() if (c >> 12) in (0, 2, 3, 4, 5) else ""
    for c in range(1 << 15)
]


def game_events(
    moves_uci: str, max_ply: int = MAX_PLY, epds: dict[int, str] | None = None, wanted=None
) -> tuple[dict[int, int], list[int], list[int], list[int], list[bool]]:
    """One game -> ({reached position: first ply}, edge positions, move codes, child positions, white to move).

    Each position and each (position, move) once per game; plies below max_ply.
    With ``epds``, the EPD of every position reached that is not in it yet (and
    for which ``wanted(hash)`` is true, when given) is added to it.
    """
    board = chess.Board()
    pieces = _HASHER.hash_board(board)
    h = pieces ^ _rest_key(board)
    reached = {h: 0}
    if epds is not None and h not in epds and (wanted is None or wanted(h)):
        epds[h] = board.epd()
    seen_edges: set[tuple[int, int]] = set()
    e_pos: list[int] = []
    e_move: list[int] = []
    e_child: list[int] = []
    e_white: list[bool] = []
    for ply, uci in enumerate(moves_uci.split()):
        if ply >= max_ply:
            break
        move = chess.Move.from_uci(uci)
        frm, to = move.from_square, move.to_square
        mover = board.turn
        ptype = board.piece_type_at(frm)
        pieces ^= _piece_key(ptype, mover, frm)
        pieces ^= _piece_key(move.promotion or ptype, mover, to)
        if board.is_en_passant(move):
            cap_sq = to - 8 if mover == chess.WHITE else to + 8
            pieces ^= _piece_key(chess.PAWN, not mover, cap_sq)
        elif board.is_castling(move):
            rank = 0 if mover == chess.WHITE else 7
            if chess.square_file(to) == 6:  # kingside: rook h -> f
                r_from, r_to = chess.square(7, rank), chess.square(5, rank)
            else:
                r_from, r_to = chess.square(0, rank), chess.square(3, rank)
            pieces ^= _piece_key(chess.ROOK, mover, r_from) ^ _piece_key(chess.ROOK, mover, r_to)
        else:
            captured = board.piece_type_at(to)
            if captured:
                pieces ^= _piece_key(captured, not mover, to)
        board.push(move)
        child = pieces ^ _rest_key(board)
        code = move_code(move)
        if (h, code) not in seen_edges:
            seen_edges.add((h, code))
            e_pos.append(h)
            e_move.append(code)
            e_child.append(child)
            e_white.append(mover == chess.WHITE)
        if child not in reached:
            reached[child] = ply + 1
            if epds is not None and child not in epds and (wanted is None or wanted(child)):
                epds[child] = board.epd()
        h = child
    return reached, e_pos, e_move, e_child, e_white


# --- one batch -------------------------------------------------------------------


def _band(w: int | None, b: int | None) -> int:
    if w is None or b is None:
        return 0
    return int((w + b) / 2 // 200 * 200)


def _in_sorted(sorted_arr: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Membership of ``values`` in a sorted uint64 array."""
    if len(sorted_arr) == 0:
        return np.zeros(len(values), bool)
    pos = np.minimum(np.searchsorted(sorted_arr, values), len(sorted_arr) - 1)
    return sorted_arr[pos] == values


def batch_events(
    games: dict[str, list], rows: np.ndarray | None = None, max_ply: int = MAX_PLY, kept: np.ndarray | None = None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Columns of a batch of games -> (edge events, position events with EPDs).

    Kept compact: numpy arrays filled per game, not Python objects per event.
    ``rows`` are the games' rows in games.parquet (default 0, 1, ...); ``rep``
    is row * 256 + ply, so the smallest ``rep`` of a position is the first game
    to reach it, whose board gives the position's EPD. With ``kept`` (sorted
    hashes), only events of those positions are returned; EPDs are taken only
    for them.
    """
    e_cols = {k: [] for k in ("pos", "move", "child", "white", "g")}
    p_cols = {k: [] for k in ("pos", "g", "rep")}
    n = len(games["moves_uci"])
    rows = np.arange(n, dtype=np.uint64) if rows is None else np.asarray(rows, np.uint64)
    year = np.zeros(n, np.int16)
    band = np.zeros(n, np.int16)
    tempo = np.zeros(n, np.uint8)
    res = np.zeros(n, np.uint8)
    welo = np.zeros(n, np.uint16)
    belo = np.zeros(n, np.uint16)
    rated = np.zeros(n, bool)
    epds: dict[int, str] = {}
    wanted = None
    if kept is not None:
        def wanted(h: int) -> bool:
            i = np.searchsorted(kept, np.uint64(h))
            return i < len(kept) and int(kept[i]) == h
    for i in range(n):
        w, b = games["white_elo"][i], games["black_elo"][i]
        year[i] = games["year"][i] or 0
        band[i] = _band(w, b)
        tempo[i] = TEMPO_CODE[games["tempo"][i]]
        res[i] = RESULT_CODE[games["result"][i]]
        rated[i] = w is not None and b is not None
        welo[i], belo[i] = (w, b) if rated[i] else (0, 0)
        reached, ep, em, ec, ew = game_events(games["moves_uci"][i], max_ply, epds, wanted)
        e_cols["pos"].append(np.fromiter(ep, np.uint64, len(ep)))
        e_cols["move"].append(np.fromiter(em, np.uint16, len(em)))
        e_cols["child"].append(np.fromiter(ec, np.uint64, len(ec)))
        e_cols["white"].append(np.fromiter(ew, bool, len(ew)))
        e_cols["g"].append(np.full(len(ep), i, np.uint32))
        p_cols["pos"].append(np.fromiter(reached.keys(), np.uint64, len(reached)))
        p_cols["g"].append(np.full(len(reached), i, np.uint32))
        p_cols["rep"].append(rows[i] * np.uint64(256) + np.fromiter(reached.values(), np.uint64, len(reached)))
    cat = {k: (np.concatenate(v) if v else np.empty(0, np.uint64)) for k, v in e_cols.items()}
    pc = {k: (np.concatenate(v) if v else np.empty(0, np.uint64)) for k, v in p_cols.items()}
    if kept is not None:
        em_ = _in_sorted(kept, cat["pos"].astype(np.uint64))
        cat = {k: v[em_] for k, v in cat.items()}
        pm_ = _in_sorted(kept, pc["pos"].astype(np.uint64))
        pc = {k: v[pm_] for k, v in pc.items()}
    g = cat["g"].astype(np.int64)
    edges = pl.DataFrame({
        "pos_hash": cat["pos"].astype(np.uint64),
        "move": cat["move"].astype(np.uint16),
        "child_hash": cat["child"].astype(np.uint64),
        "year": year[g], "elo_band": band[g], "tempo": tempo[g], "res": res[g],
        "welo": welo[g], "belo": belo[g], "rated": rated[g], "mover_white": cat["white"].astype(bool),
    })
    pg = pc["g"].astype(np.int64)
    positions = pl.DataFrame({"pos_hash": pc["pos"].astype(np.uint64), "res": res[pg], "rep": pc["rep"].astype(np.uint64)})
    epd = pl.DataFrame({"pos_hash": np.fromiter(epds.keys(), np.uint64, len(epds)), "epd": list(epds.values())},
                       schema={"pos_hash": pl.UInt64, "epd": pl.String})
    positions = positions.join(epd, on="pos_hash", how="left")
    return edges, positions


def aggregate_edges(events: pl.DataFrame) -> pl.DataFrame:
    """Edge events (or partial aggregates) -> one row per (pos, move, year, band, tempo)."""
    if "n" not in events.columns:  # raw events
        events = events.with_columns(
            n=pl.lit(1, pl.UInt32),
            white_wins=(pl.col("res") == 0).cast(pl.UInt32),
            draws=(pl.col("res") == 1).cast(pl.UInt32),
            black_wins=(pl.col("res") == 2).cast(pl.UInt32),
            sum_mover_elo=pl.when(pl.col("mover_white")).then(pl.col("welo")).otherwise(pl.col("belo")).cast(pl.UInt64),
            sum_opp_elo=pl.when(pl.col("mover_white")).then(pl.col("belo")).otherwise(pl.col("welo")).cast(pl.UInt64),
            n_rated=pl.col("rated").cast(pl.UInt32),
        )
    keys = ["pos_hash", "move", "year", "elo_band", "tempo"]
    sums = ["n", "white_wins", "draws", "black_wins", "sum_mover_elo", "sum_opp_elo", "n_rated"]
    return events.group_by(keys).agg(pl.col("child_hash").first(), *[pl.col(c).sum() for c in sums])


def aggregate_positions(events: pl.DataFrame) -> pl.DataFrame:
    """Position events (or partial aggregates) -> one row per position."""
    if "n" not in events.columns:
        events = events.with_columns(
            n=pl.lit(1, pl.UInt32),
            white_wins=(pl.col("res") == 0).cast(pl.UInt32),
            draws=(pl.col("res") == 1).cast(pl.UInt32),
            black_wins=(pl.col("res") == 2).cast(pl.UInt32),
        )
    return events.group_by("pos_hash").agg(
        pl.col("n", "white_wins", "draws", "black_wins").sum(),
        pl.col("rep").min(),
        # the EPD of the first game to reach the position (smallest rep)
        pl.col("epd").sort_by("rep").first(),
    )


# --- finishing ------------------------------------------------------------------


def finish_edges(edges: pl.DataFrame, kept_positions: pl.DataFrame) -> pl.DataFrame:
    """Merged edge aggregates -> the edges.parquet schema, rows of kept positions only, sorted."""
    codes = pl.Series("uci", _CODE_TO_UCI, dtype=pl.String)
    out = edges.join(kept_positions.select("pos_hash"), on="pos_hash", how="semi")
    out = out.with_columns(
        move=codes.gather(out["move"].cast(pl.Int64)),
        tempo=pl.Series(TEMPI, dtype=pl.String).gather(out["tempo"].cast(pl.Int64)),
    )
    return out.select([pl.col(c).cast(t) for c, t in EDGES_SCHEMA.items()]).sort(
        "pos_hash", "move", "year", "elo_band", "tempo")


def tables_from_games(
    games: pl.DataFrame, *, max_ply: int = MAX_PLY, min_games: int = MIN_GAMES
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Both tables, in memory, from a small frame of games. For test fixtures.

    ``games`` needs the games.parquet columns ``moves_uci``, ``result``,
    ``year``, ``white_elo``, ``black_elo`` and ``tempo``; any other columns
    are ignored. Returns ``(edges, pos_index)`` in the schemas of
    ``edges.parquet`` and ``pos_index.parquet``, exactly as the full build
    would write them for this set of games. Not for large inputs: it holds
    every event in memory.
    """
    cols = {c: games[c].to_list() for c in GAME_COLUMNS}
    ev, pv = batch_events(cols, None, max_ply)
    edges = aggregate_edges(ev)
    positions = aggregate_positions(pv)
    kept = positions.filter(pl.col("n") >= min_games)
    return finish_edges(edges, kept), finish_pos_index(kept)


def finish_pos_index(kept: pl.DataFrame) -> pl.DataFrame:
    if kept["epd"].null_count():
        raise RuntimeError(f"{kept['epd'].null_count()} kept positions without an EPD")
    return kept.select([pl.col(c).cast(t) for c, t in POS_INDEX_SCHEMA.items()]).sort("pos_hash")


# --- reading -----------------------------------------------------------------------

Filter = int | str | tuple | list | None


def _key(pos) -> int:
    """A position as its hash: an int is taken as the hash, a Board or a FEN/EPD is hashed."""
    if isinstance(pos, int):
        return pos
    if isinstance(pos, chess.Board):
        return pos_hash(pos)
    board = chess.Board(pos) if len(pos.split()) >= 6 else chess.Board(pos + " 0 1")
    return pos_hash(board)


def _match(col: str, value: Filter) -> pl.Expr:
    """``value``: None (all), a single value, an inclusive (lo, hi) tuple, or a list of values."""
    if value is None:
        return pl.lit(True)
    if isinstance(value, tuple):
        lo, hi = value
        return pl.col(col).is_between(lo, hi)
    if isinstance(value, list):
        return pl.col(col).is_in(value)
    return pl.col(col) == value


PERFORMANCE_CAP = 800


def performance(score: pl.Expr, opp_elo: pl.Expr) -> pl.Expr:
    """Rating performance: the logistic inverse ``opp - 400 * log10(1 / score - 1)``.

    The one definition used everywhere here. The difference to the opponents'
    mean is capped at +-800 (FIDE's table stops there too), which is reached
    at a score of 100/101 or 1/101; 100 % and 0 % give exactly +-800. Null
    where ``opp_elo`` is null (no rated games).
    """
    lo = 1 / (1 + 10 ** (PERFORMANCE_CAP / 400))
    s = score.clip(lo, 1 - lo)
    return opp_elo - 400 * (1 / s - 1).log10()


def parents_from_edges(edges: pl.DataFrame) -> pl.DataFrame:
    """The parents table from the edges table: one row per (child, parent, move), all years,
    bands and tempi summed, sorted by ``child_hash``. ``n`` is the games that played the move."""
    return (
        edges.group_by("child_hash", "pos_hash", "move")
        .agg(pl.col("n").sum())
        .select([pl.col(c).cast(t) for c, t in PARENTS_SCHEMA.items()])
        .sort("child_hash", "pos_hash", "move")
    )


_EMPTY_MOVES = {"move": pl.String, "child_hash": pl.UInt64, "n": pl.UInt32}
_SUMS = ["n", "white_wins", "draws", "black_wins", "sum_mover_elo", "sum_opp_elo", "n_rated"]
_MOVE_COLS = ["n", "share", "white_wins", "draws", "black_wins", "score", "n_rated", "mover_elo", "opp_elo", "performance"]


class _Sorted:
    """Point and set lookups on a parquet file sorted by ``key``, through its row-group statistics.

    The row groups' key ranges are read once; a lookup reads only the row
    groups whose range can hold one of the keys, and filters them in memory.
    Polars' own pruning handles ``==`` well but not ``is_in`` over scattered
    keys, which is what a position's children and parents need.
    """

    def __init__(self, path: Path, key: str):
        self.pf = pq.ParquetFile(path)
        self.key = key
        col = self.pf.schema_arrow.get_field_index(key)
        md = self.pf.metadata
        self.groups, mins, maxs = [], [], []
        for i in range(md.num_row_groups):
            rg = md.row_group(i)
            if rg.num_rows == 0:
                continue
            st = rg.column(col).statistics
            if st is None or not st.has_min_max:
                raise ValueError(f"{path}: row group {i} has no min/max for {key}; the lookups need them")
            self.groups.append(i)
            mins.append(st.min)
            maxs.append(st.max)
        self.groups = np.array(self.groups, np.int64)
        self.mins, self.maxs = np.array(mins, np.uint64), np.array(maxs, np.uint64)

    def row_groups(self, keys) -> list[int]:
        """Every row group whose [min, max] holds one of the keys.

        A key's rows can run over several row groups (the start position's
        edges fill more than one), so for each key it is the whole range from
        the first group whose max reaches it to the last whose min does.
        """
        k = np.unique(np.asarray(keys, np.uint64))
        first = np.searchsorted(self.maxs, k, side="left")  # maxs and mins are sorted: the file is
        last = np.searchsorted(self.mins, k, side="right")  # groups first .. last - 1 hold k
        out: set[int] = set()
        for a, b in zip(first.tolist(), last.tolist(), strict=True):
            out.update(range(a, b))
        return sorted(self.groups[sorted(out)].tolist()) if out else []

    def rows(self, keys, columns: list[str] | None = None) -> pl.DataFrame:
        k = np.unique(np.asarray(keys, np.uint64))
        groups = self.row_groups(k)
        if not groups:
            t = self.pf.schema_arrow.empty_table()
            return pl.from_arrow(t.select(columns) if columns else t)
        read = None if columns is None else list(dict.fromkeys([self.key, *columns]))
        df = pl.from_arrow(self.pf.read_row_groups(groups, columns=read))
        df = df.filter(pl.col(self.key).is_in(pl.Series(k, dtype=pl.UInt64).implode()))
        return df if columns is None else df.select(columns)


def check_lookups(path: Path, key: str) -> int:
    """Check a written table against its reader: for every key at a row-group boundary, the
    reader's rows must equal a plain count of the key column. Raises on a mismatch; returns
    the number of keys checked. Reads only the key column (8 bytes a row)."""
    table = _Sorted(path, key)
    keys = np.unique(np.concatenate([table.mins, table.maxs]))
    col = pl.from_arrow(pq.read_table(path, columns=[key]))
    expected = col.filter(pl.col(key).is_in(pl.Series(keys, dtype=pl.UInt64).implode())).group_by(key).len()
    got = table.rows(keys, [key]).group_by(key).len()
    diff = expected.join(got, on=key, how="full", coalesce=True, suffix="_reader").filter(
        pl.col("len").fill_null(0) != pl.col("len_reader").fill_null(0))
    if diff.height:
        raise RuntimeError(f"{path}: the reader returns other rows than the file for {diff.height} keys:\n{diff.head()}")
    return len(keys)


class _InMemory:
    """The same interface over a frame, for tables built in memory."""

    def __init__(self, df: pl.DataFrame, key: str):
        self.df, self.key = df, key

    def rows(self, keys, columns: list[str] | None = None) -> pl.DataFrame:
        k = pl.Series(np.unique(np.asarray(keys, np.uint64)), dtype=pl.UInt64)
        df = self.df.filter(pl.col(self.key).is_in(k.implode()))
        return df.select(columns) if columns else df


class EdgeDB:
    """Queries on edges.parquet, pos_index.parquet and parents.parquet.

        db = EdgeDB()                     # corpus/edges.parquet, pos_index.parquet, parents.parquet
        db = EdgeDB(name="dev")           # corpus/edges_dev.parquet, ...
        db = EdgeDB.from_frames(e, p)     # the output of tables_from_games, for tests

    ``pos`` is a hash, a ``chess.Board`` or a FEN/EPD. Filters: ``years``
    (``year`` 0 is unknown), ``elo_bands`` (the floor of the mean rating to
    200, 0 when a rating is missing) and ``tempi``; each takes a value, an
    inclusive ``(lo, hi)`` tuple or a list.

    The files are sorted by their lookup key (``pos_hash``, ``child_hash`` for
    parents) and written in small row groups, so a lookup reads the one or
    two row groups whose min/max statistics contain the key.

    Counting: a position's ``n`` (``total``, ``child_n``) is games that
    reached it. A move's ``n`` is games that played that move from it, each
    game once per (position, move). A game that comes back to a position and
    plays another move the second time counts for both moves, so the moves'
    ``n`` can add up to more than the games that went on from the position.
    """

    def __init__(self, name: str = "full", directory: Path | None = None):
        d = directory or CORPUS
        suffix = "" if name == "full" else f"_{name}"
        self._edges = _Sorted(d / f"edges{suffix}.parquet", "pos_hash")
        self._pos = _Sorted(d / f"pos_index{suffix}.parquet", "pos_hash")
        self._parents = _Sorted(d / f"parents{suffix}.parquet", "child_hash")

    @classmethod
    def from_frames(cls, edges: pl.DataFrame, pos_index: pl.DataFrame, parents: pl.DataFrame | None = None) -> EdgeDB:
        db = cls.__new__(cls)
        db._edges, db._pos = _InMemory(edges, "pos_hash"), _InMemory(pos_index, "pos_hash")
        db._parents = _InMemory(parents if parents is not None else parents_from_edges(edges), "child_hash")
        return db

    def summary(self) -> dict:
        """Games in the table and their year span, from the start position (every game has it).

        ``games`` counts the kept games with at least one move; ``first_year`` and
        ``last_year`` ignore year 0 (unknown). One lookup.
        """
        start = chess.Board()
        total = self.total(start)
        years = self._edges.rows([pos_hash(start)], ["year"])["year"]
        known = years.filter(years > 0)
        return {
            "games": total["n"] if total else 0,
            "first_year": int(known.min()) if known.len() else None,
            "last_year": int(known.max()) if known.len() else None,
        }

    # -- single lookups --

    def total(self, pos) -> dict | None:
        """The position's row in pos_index (all years, bands, tempi), or None below two games."""
        rows = self._pos.rows([_key(pos)])
        return rows.row(0, named=True) if rows.height else None

    def _rows(self, h: int, years: Filter, elo_bands: Filter, tempi: Filter) -> pl.DataFrame:
        """The position's edge rows under the filters: one scan."""
        return self._edges.rows([h]).filter(_match("year", years) & _match("elo_band", elo_bands) & _match("tempo", tempi))

    def _child_totals(self, child_hashes: pl.Series) -> pl.DataFrame:
        return self._pos.rows(child_hashes.to_numpy(), ["pos_hash", "n"]).select(
            pl.col("pos_hash").alias("child_hash"), pl.col("n").alias("child_n"))

    @staticmethod
    def _summarise(rows: pl.DataFrame, white_to_move: bool, by: list[str]) -> pl.DataFrame:
        """Edge rows -> one row per move (and per ``by``), with the derived columns."""
        df = rows.group_by(["move", "child_hash", *by]).agg(
            pl.col(_SUMS).sum(), pl.col("year").filter(pl.col("year") > 0).max().alias("last_year"))
        wins = pl.col("white_wins") if white_to_move else pl.col("black_wins")
        rated = pl.col("n_rated").cast(pl.Float64)
        df = df.with_columns(
            share=(pl.col("n") / pl.col("n").sum().over(by)) if by else (pl.col("n") / pl.col("n").sum()),
            score=(wins + 0.5 * pl.col("draws")) / pl.col("n"),
            mover_elo=pl.when(rated > 0).then(pl.col("sum_mover_elo") / rated),
            opp_elo=pl.when(rated > 0).then(pl.col("sum_opp_elo") / rated),
        ).with_columns(performance=performance(pl.col("score"), pl.col("opp_elo")))
        if by:
            return df.select("move", "child_hash", *by, *_MOVE_COLS).sort([*by, "n", "move"], descending=[False, True, False])
        return df.select("move", "child_hash", *_MOVE_COLS, "last_year").sort("n", "move", descending=[True, False])

    def moves(self, pos, *, years: Filter = None, elo_bands: Filter = None, tempi: Filter = None) -> pl.DataFrame:
        """The moves played from the position under the filters, most played first.

        Columns: ``move``, ``child_hash``, ``n``, ``share`` (of the games here
        under these filters), ``white_wins``, ``draws``, ``black_wins``,
        ``score`` (for the side to move), ``n_rated`` (games with both
        ratings, which the next three rest on), ``mover_elo`` and ``opp_elo``
        (means), ``performance`` (see ``performance``), ``last_year`` (the
        last year the move was played, year 0 ignored; null when only games
        of unknown year played it).
        """
        h = _key(pos)
        total = self.total(h)
        if total is None:
            return pl.DataFrame(schema=_EMPTY_MOVES)
        return self._summarise(self._rows(h, years, elo_bands, tempi), total["epd"].split()[1] == "w", [])

    def moves_by_year(self, pos, *, years: Filter = None, elo_bands: Filter = None, tempi: Filter = None) -> pl.DataFrame:
        """As ``moves`` without ``last_year``, one row per move and year, sorted by year then games.

        ``share`` is the move's share of that year's games from the position;
        year 0 (unknown) is a row like any other.
        """
        h = _key(pos)
        total = self.total(h)
        if total is None:
            return pl.DataFrame(schema=_EMPTY_MOVES)
        return self._summarise(self._rows(h, years, elo_bands, tempi), total["epd"].split()[1] == "w", ["year"])

    def children(self, pos, *, years: Filter = None, elo_bands: Filter = None, tempi: Filter = None) -> pl.DataFrame:
        """Each move from the position with its games under the filters (``n``) and the
        child position's own total over all games (``child_n``, null below two games)."""
        m = self.moves(pos, years=years, elo_bands=elo_bands, tempi=tempi).select("move", "child_hash", "n")
        if m.height == 0:
            return m.with_columns(child_n=pl.lit(None, pl.UInt32))
        return m.join(self._child_totals(m["child_hash"]), on="child_hash", how="left", maintain_order="left")

    def panel(self, pos, *, years: Filter = None, elo_bands: Filter = None, tempi: Filter = None) -> dict:
        """Everything the database panel shows, from one scan of the position's rows.

        Returns ``{"total": total(pos), "moves": moves(...) plus child_n,
        "by_year": moves_by_year(...)}``; ``moves`` and ``by_year`` are empty
        frames and ``total`` None below two games. Three reads: the position's
        pos_index row, its edge rows, and its children's pos_index rows.
        """
        h = _key(pos)
        total = self.total(h)
        if total is None:
            empty = pl.DataFrame(schema=_EMPTY_MOVES)
            return {"total": None, "moves": empty.with_columns(child_n=pl.lit(None, pl.UInt32)), "by_year": empty}
        white = total["epd"].split()[1] == "w"
        rows = self._rows(h, years, elo_bands, tempi)
        moves = self._summarise(rows, white, [])
        if moves.height:
            moves = moves.join(self._child_totals(moves["child_hash"]), on="child_hash", how="left", maintain_order="left")
        else:
            moves = moves.with_columns(child_n=pl.lit(None, pl.UInt32))
        return {"total": total, "moves": moves, "by_year": self._summarise(rows, white, ["year"])}

    def totals_many(self, positions) -> pl.DataFrame:
        """``total`` for many positions in one read: the pos_index rows of those reached by
        two or more games (the others are absent), sorted by ``pos_hash``."""
        keys = [_key(p) for p in positions]
        return self._pos.rows(keys).sort("pos_hash") if keys else pl.DataFrame(schema=POS_INDEX_SCHEMA)

    def parents_many(self, positions, *, years: Filter = None, elo_bands: Filter = None,
                     tempi: Filter = None, with_epd: bool = True) -> pl.DataFrame:
        """``parents`` for many positions in one batch of reads.

        One row per (child, parent, move): ``child_hash``, ``pos_hash`` (the
        parent), ``move``, ``n`` (games that played it, under the filters) and,
        with ``with_epd``, the parent's ``epd``. Sorted by child, then games
        (most first), then parent. Children with no listed parent have no rows.
        Each underlying table is read once for the whole batch, so a batch of a
        few thousand children costs about what one lookup per row group touched.
        """
        keys = np.unique(np.asarray([_key(p) for p in positions], np.uint64))
        empty = {"child_hash": pl.UInt64, "pos_hash": pl.UInt64, "move": pl.String, "n": pl.UInt32}
        if with_epd:
            empty["epd"] = pl.String
        if len(keys) == 0:
            return pl.DataFrame(schema=empty)
        par = self._parents.rows(keys)
        if par.height and not (years is None and elo_bands is None and tempi is None):
            wanted = pl.Series(keys, dtype=pl.UInt64)
            par = (
                self._edges.rows(par["pos_hash"].unique().to_numpy())
                .filter(pl.col("child_hash").is_in(wanted.implode())
                        & _match("year", years) & _match("elo_band", elo_bands) & _match("tempo", tempi))
                .group_by("child_hash", "pos_hash", "move").agg(pl.col("n").sum())
            )
        if par.height == 0:
            return pl.DataFrame(schema=empty)
        par = par.select([pl.col(c).cast(t) for c, t in empty.items() if c != "epd"])
        if with_epd:
            epd = self._pos.rows(par["pos_hash"].unique().to_numpy(), ["pos_hash", "epd"])
            par = par.join(epd, on="pos_hash", how="left")
        return par.sort(["child_hash", "n", "pos_hash"], descending=[False, True, False])

    def parents(self, pos, *, years: Filter = None, elo_bands: Filter = None, tempi: Filter = None) -> pl.DataFrame:
        """The positions that lead into this one: parent ``pos_hash``, ``move``, ``n``
        (games that played it, under the filters) and the parent's ``epd``, most played first.

        Without filters it is one lookup in parents.parquet (sorted by
        ``child_hash``); with filters the parents' edge rows are read too.
        Only parents reached by two or more games are listed, as in edges.
        """
        out = self.parents_many([pos], years=years, elo_bands=elo_bands, tempi=tempi)
        return out.drop("child_hash").sort("n", "pos_hash", descending=[True, False])


# --- the full build ----------------------------------------------------------------


def _select(table, fraction: float | None, since_year: int | None) -> np.ndarray:
    """Row mask of a games batch: a fixed subset by game_id hash, or by year."""
    keep = np.ones(table.num_rows, bool)
    if fraction is not None:
        gid = table.column("game_id").to_pylist()
        keep &= np.fromiter((int(g[:8], 16) % 100_000 < fraction * 100_000 for g in gid), bool, len(gid))
    if since_year is not None:
        year = table.column("year").to_pylist()
        keep &= np.fromiter(((y or 0) >= since_year for y in year), bool, len(year))
    return keep


def iter_game_batches(path: Path, batch_games: int, fraction: float | None, since_year: int | None) -> Iterator[tuple[int, dict]]:
    """(batch number, columns) for the selected games, ``batch_games`` at a time.

    Rows are numbered by their place in games.parquet, so a later pass can find
    a game again from its row number. A batch is read as a few thousand rows
    at a time; the full file is never in memory.
    """
    pf = pq.ParquetFile(path)
    row = 0
    k = 0
    buf: dict[str, list] = {c: [] for c in [*GAME_COLUMNS, "row"]}
    for rb in pf.iter_batches(batch_size=20_000, columns=[*GAME_COLUMNS, "game_id"]):
        keep = _select(rb, fraction, since_year)
        idx = np.nonzero(keep)[0]
        if len(idx):
            sub = rb.take(idx)
            for c in GAME_COLUMNS:
                buf[c].extend(sub.column(c).to_pylist())
            buf["row"].extend((row + idx).tolist())
        row += rb.num_rows
        _drop_cache(path)
        if len(buf["row"]) >= batch_games:
            yield k, buf
            k += 1
            buf = {c: [] for c in [*GAME_COLUMNS, "row"]}
    if buf["row"]:
        yield k, buf


# Small row groups for point lookups: a lookup decodes the row groups whose min/max contain
# the key. pos_index and parents are read at scattered keys (a position's children, its
# parents), edges at one key whose rows lie together.
ROWS_PER_GROUP = {"edges": 32_768, "pos_index": 8_192, "parents": 8_192}


def _drop_cache(path: Path) -> None:
    """Write a file's dirty pages and drop it from the page cache.

    Under tnh-run the page cache of files a job reads and writes is charged to
    the job's memory; without this the partial files of a large build count
    against the cap as they accumulate.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _write(df: pl.DataFrame, path: Path, **kw) -> None:
    df.write_parquet(path, **kw)
    _drop_cache(path)


def _bucket(df: pl.DataFrame) -> pl.Series:
    return pl.Series((df["pos_hash"].to_numpy() >> np.uint64(BUCKET_SHIFT)).astype(np.uint8))


def _rss_mb() -> int:
    """Resident memory of this process now, MB."""
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") // 2**20


def _peak_mb() -> int:
    """Peak resident memory of this process, MB: VmHWM, which starts afresh at exec.

    Not getrusage's ru_maxrss: a spawned worker inherits the parent's figure
    across the fork and exec, so it would report the parent's peak.
    """
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) // 1024
    raise RuntimeError("no VmHWM in /proc/self/status")


def count_batch(args: tuple[int, dict, str, int]) -> tuple[int, int, int, int]:
    """Worker, pass A: how many games of the batch reach each position, written in buckets.

    Returns (batch, games, distinct positions, this process's peak MB).
    """
    k, cols, work, max_ply = args
    hashes = np.concatenate([
        np.fromiter(game_events(m, max_ply)[0].keys(), np.uint64) for m in cols["moves_uci"]
    ]) if cols["moves_uci"] else np.empty(0, np.uint64)
    keys, counts = np.unique(hashes, return_counts=True)
    del hashes
    df = pl.DataFrame({"pos_hash": keys, "n": counts.astype(np.uint32)})
    df = df.with_columns(_b=_bucket(df))
    for (b,), part in df.partition_by("_b", as_dict=True).items():
        d = Path(work) / "counts" / f"b{b:02d}"
        d.mkdir(parents=True, exist_ok=True)
        _write(part.drop("_b"), d / f"k{k:05d}.parquet", compression="lz4")
    return k, len(cols["moves_uci"]), len(keys), _peak_mb()


def merge_counts(args: tuple[int, str, int]) -> tuple[int, dict]:
    """Worker, merge A: the positions of one bucket reached by at least ``min_games`` games."""
    b, work, min_games = args
    w = Path(work)
    d = w / "counts" / f"b{b:02d}"
    hist = {"ge2": 0, "ge10": 0, "ge100": 0, "ge1000": 0}
    kept = np.empty(0, np.uint64)
    if d.exists():
        n = pl.read_parquet(d / "*.parquet").group_by("pos_hash").agg(pl.col("n").sum())
        hist = {f"ge{t}": int((n["n"] >= t).sum()) for t in (2, 10, 100, 1000)}
        kept = np.sort(n.filter(pl.col("n") >= min_games)["pos_hash"].to_numpy())
        del n
        shutil.rmtree(d)
    np.save(w / "kept" / f"b{b:02d}.npy", kept)
    return b, hist


def process_batch(args: tuple[int, dict, str, int]) -> tuple[int, int, int, int, int]:
    """Worker, pass B: edges and position totals of the batch, for kept positions only, in buckets.

    Returns (batch, games, edge rows written, position rows written, this process's peak MB).
    """
    k, cols, work, max_ply = args
    rows = cols.pop("row")
    kept = np.load(Path(work) / "kept.npy", mmap_mode="r")
    ev, pv = batch_events(cols, np.asarray(rows, np.uint64), max_ply, kept)
    edges = aggregate_edges(ev)
    del ev
    positions = aggregate_positions(pv)
    del pv
    for name, df in (("edges", edges), ("positions", positions)):
        df = df.with_columns(_b=_bucket(df))
        for (b,), part in df.partition_by("_b", as_dict=True).items():
            d = Path(work) / name / f"b{b:02d}"
            d.mkdir(parents=True, exist_ok=True)
            _write(part.drop("_b"), d / f"k{k:05d}.parquet", compression="lz4")
    return k, len(rows), edges.height, positions.height, _peak_mb()


def merge_bucket(args: tuple[int, str]) -> tuple[int, int, int]:
    """Worker, merge B: one bucket across batches, in the final schemas.

    Returns (bucket, kept positions, edge rows).
    """
    b, work = args
    w = Path(work)
    (w / "merged").mkdir(exist_ok=True)
    pd, ed = w / "positions" / f"b{b:02d}", w / "edges" / f"b{b:02d}"
    if pd.exists():
        kept = finish_pos_index(aggregate_positions(pl.read_parquet(pd / "*.parquet")))
        shutil.rmtree(pd)
    else:
        kept = pl.DataFrame(schema=POS_INDEX_SCHEMA)
    if ed.exists():
        edges = finish_edges(aggregate_edges(pl.read_parquet(ed / "*.parquet")), kept)
        shutil.rmtree(ed)
    else:  # only final positions fell in this bucket, or none
        edges = pl.DataFrame(schema=EDGES_SCHEMA)
    _write(edges, w / "merged" / f"edges_b{b:02d}.parquet")
    _write(kept, w / "merged" / f"pos_b{b:02d}.parquet")
    return b, kept.height, edges.height


def build(*, name: str, out_dir: Path, fraction: float | None, since_year: int | None,
          batch_games: int, workers: int, max_ply: int = MAX_PLY, min_games: int = MIN_GAMES,
          polars_threads: int = 1) -> None:
    """The full build: pass A counts positions, pass B aggregates the kept positions only.

    Two replays of the games instead of one: the partial files of a single
    pass hold every position and edge of every batch, almost all of them seen
    by one game, about 9 GB for the full corpus; after pass A only the kept
    positions' rows are written.
    """
    t0 = time.time()
    games_path = CORPUS / "games.parquet"
    work = DATA_ROOT / "scratch" / "edges" / f"work_{name}"
    if work.exists():
        shutil.rmtree(work)
    (work / "kept").mkdir(parents=True)
    # Workers are spawned, so they read this before polars starts its thread pool. One thread
    # each: the process count is the CPU budget, and every polars thread keeps its own arena.
    os.environ["POLARS_MAX_THREADS"] = str(polars_threads)
    ctx = multiprocessing.get_context("spawn")
    peak_disk = 0

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:7.0f} s] {msg}", flush=True)

    def disk() -> int:
        return sum(f.stat().st_size for f in work.rglob("*") if f.is_file())

    def run_batches(fn, label: str) -> int:
        nonlocal peak_disk
        n_games = 0
        # a fresh process per batch: memory a worker keeps after a batch does not add up
        with ctx.Pool(workers, maxtasksperchild=1) as pool:
            pending = []
            batches = iter_game_batches(games_path, batch_games, fraction, since_year)
            done = False
            while not done or pending:
                # at most one batch waiting per worker, so batches are read as they are needed
                while not done and len(pending) < workers + 1:
                    nxt = next(batches, None)
                    if nxt is None:
                        done = True
                        break
                    k, cols = nxt
                    if fn is count_batch:
                        cols.pop("row")
                    pending.append(pool.apply_async(fn, ((k, cols, str(work), max_ply),)))
                out = pending.pop(0).get()
                n_games += out[1]
                log(f"  {label} batch {out[0]}: {out[1]} games, rows {out[2:-1]}; {n_games:,} games so far; "
                    f"worker peak {out[-1]} MB, main {_rss_mb()} MB")
        peak_disk = max(peak_disk, disk())
        return n_games

    log(f"pass A: positions per batch of {batch_games} games, {workers} workers -> {work}")
    n_games = run_batches(count_batch, "A")
    hist = {"ge2": 0, "ge10": 0, "ge100": 0, "ge1000": 0}
    with ctx.Pool(workers, maxtasksperchild=8) as pool:
        for _, h in pool.imap_unordered(merge_counts, [(b, str(work), min_games) for b in range(N_BUCKETS)]):
            for k2, v in h.items():
                hist[k2] += v
    np.save(work / "kept.npy", np.concatenate([np.load(work / "kept" / f"b{b:02d}.npy") for b in range(N_BUCKETS)]))
    shutil.rmtree(work / "kept")
    log(f"pass A done: {n_games:,} games; positions with at least 2 / 10 / 100 / 1000 games: "
        f"{hist['ge2']:,} / {hist['ge10']:,} / {hist['ge100']:,} / {hist['ge1000']:,}")

    log("pass B: edges and totals of the kept positions")
    run_batches(process_batch, "B")
    kept_pos = kept_edges = 0
    with ctx.Pool(workers, maxtasksperchild=8) as pool:
        for _, kp, ke in pool.imap_unordered(merge_bucket, [(b, str(work)) for b in range(N_BUCKETS)]):
            kept_pos += kp
            kept_edges += ke
    if min_games == 2 and kept_pos != hist["ge2"]:
        raise RuntimeError(f"pass B kept {kept_pos:,} positions, pass A counted {hist['ge2']:,}")
    log(f"pass B done: {kept_pos:,} positions, {kept_edges:,} edge rows")

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if name == "full" else f"_{name}"
    edges_out = out_dir / f"edges{suffix}.parquet"
    pos_out = out_dir / f"pos_index{suffix}.parquet"
    par_out = out_dir / f"parents{suffix}.parquet"
    writers: dict[str, pq.ParquetWriter] = {}

    def append(key: str, path: Path, df: pl.DataFrame, rows_per_group: int) -> None:
        if df.height == 0 and key in writers:
            return  # an empty row group would carry no min/max, which the readers need
        t = df.to_arrow()
        if key not in writers:
            writers[key] = pq.ParquetWriter(str(path) + ".tmp", t.schema, compression="zstd")
        writers[key].write_table(t, row_group_size=rows_per_group)

    for b in range(N_BUCKETS):
        e = pl.read_parquet(work / "merged" / f"edges_b{b:02d}.parquet")
        p = pl.read_parquet(work / "merged" / f"pos_b{b:02d}.parquet")
        append("edges", edges_out, e, ROWS_PER_GROUP["edges"])
        append("pos", pos_out, p, ROWS_PER_GROUP["pos_index"])
        # the parents table is sorted on child_hash: bucket it again on the child
        par = parents_from_edges(e)
        par = par.with_columns(_b=pl.Series((par["child_hash"].to_numpy() >> np.uint64(BUCKET_SHIFT)).astype(np.uint8)))
        for (cb,), sub in par.partition_by("_b", as_dict=True).items():
            d = work / "parents" / f"c{cb:02d}"
            d.mkdir(parents=True, exist_ok=True)
            _write(sub.drop("_b"), d / f"b{b:02d}.parquet", compression="lz4")
        _drop_cache(work / "merged" / f"edges_b{b:02d}.parquet")
    for cb in range(N_BUCKETS):
        d = work / "parents" / f"c{cb:02d}"
        if d.exists():
            append("parents", par_out, pl.read_parquet(d / "*.parquet").sort("child_hash", "pos_hash", "move"),
                   ROWS_PER_GROUP["parents"])
    peak_disk = max(peak_disk, disk())
    for key, path in (("edges", edges_out), ("pos", pos_out), ("parents", par_out)):
        writers[key].close()
        Path(str(path) + ".tmp").replace(path)
        _drop_cache(path)
    for path, k in ((edges_out, "pos_hash"), (pos_out, "pos_hash"), (par_out, "child_hash")):
        log(f"checked the reader on {check_lookups(path, k):,} row-group boundary keys of {path.name}")
    shutil.rmtree(work)
    log(f"wrote {edges_out} ({edges_out.stat().st_size / 1e9:.3f} GB, {kept_edges:,} rows), "
        f"{pos_out} ({pos_out.stat().st_size / 1e9:.3f} GB, {kept_pos:,} rows) and "
        f"{par_out} ({par_out.stat().st_size / 1e9:.3f} GB, {pq.ParquetFile(par_out).metadata.num_rows:,} rows)")
    log("positions with at least 2 / 10 / 100 / 1000 games: "
        f"{hist['ge2']:,} / {hist['ge10']:,} / {hist['ge100']:,} / {hist['ge1000']:,}; "
        f"games {n_games:,}; work files peaked at {peak_disk / 1e9:.2f} GB")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="full", help="'full' writes corpus/edges.parquet; another name NAME writes edges_NAME.parquet")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: corpus/)")
    ap.add_argument("--fraction", type=float, default=None, help="a fixed fraction of the games, by game_id hash")
    ap.add_argument("--since-year", type=int, default=None, help="only games of this year and later")
    ap.add_argument("--batch-games", type=int, default=50_000)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)
    build(name=args.name, out_dir=args.out or CORPUS, fraction=args.fraction, since_year=args.since_year,
          batch_games=args.batch_games, workers=args.workers)


if __name__ == "__main__":
    main()
