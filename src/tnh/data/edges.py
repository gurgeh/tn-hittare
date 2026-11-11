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

Two ways to build:

- ``tables_from_games(games)`` builds both tables in memory from a small frame
  of games. For test fixtures in other modules; it is the same aggregation
  as the full build, on one batch.
- ``build(...)`` (the command) works through ``games.parquet`` in batches.
  Each batch is aggregated and written to disk split into 64 buckets on the
  top six bits of the position hash; each bucket is then merged across
  batches and filtered on the merged position counts. Memory is set by the
  batch size and the size of one bucket, not by the corpus. The
  representative EPD of each kept position is filled in by a last pass that
  replays only the games and plies needed. Bucket order is hash order, so
  the outputs are sorted by ``pos_hash``, which the readers' lookups rely on.

Reading: ``EdgeDB`` (moves from a position under filters, a position's total,
the counts of its children).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import shutil
import time
from collections.abc import Iterable, Iterator
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
    moves_uci: str, max_ply: int = MAX_PLY
) -> tuple[dict[int, int], list[int], list[int], list[int], list[bool]]:
    """One game -> ({reached position: first ply}, edge positions, move codes, child positions, white to move).

    Each position and each (position, move) once per game; plies below max_ply.
    """
    board = chess.Board()
    pieces = _HASHER.hash_board(board)
    h = pieces ^ _rest_key(board)
    reached = {h: 0}
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
        reached.setdefault(child, ply + 1)
        h = child
    return reached, e_pos, e_move, e_child, e_white


# --- one batch -------------------------------------------------------------------


def _band(w: int | None, b: int | None) -> int:
    if w is None or b is None:
        return 0
    return int((w + b) / 2 // 200 * 200)


def batch_events(games: dict[str, list], row0: int = 0, max_ply: int = MAX_PLY) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Columns of a batch of games -> (edge events, position events), one row per event.

    Kept compact: numpy arrays filled per game, not Python objects per event.
    ``rep`` is (row of the game in the input) * 256 + ply, for picking a
    representative game for each position's EPD.
    """
    e_cols = {k: [] for k in ("pos", "move", "child", "white", "g")}
    p_cols = {k: [] for k in ("pos", "g", "rep")}
    n = len(games["moves_uci"])
    year = np.zeros(n, np.int16)
    band = np.zeros(n, np.int16)
    tempo = np.zeros(n, np.uint8)
    res = np.zeros(n, np.uint8)
    welo = np.zeros(n, np.uint16)
    belo = np.zeros(n, np.uint16)
    rated = np.zeros(n, bool)
    for i in range(n):
        w, b = games["white_elo"][i], games["black_elo"][i]
        year[i] = games["year"][i] or 0
        band[i] = _band(w, b)
        tempo[i] = TEMPO_CODE[games["tempo"][i]]
        res[i] = RESULT_CODE[games["result"][i]]
        rated[i] = w is not None and b is not None
        welo[i], belo[i] = (w, b) if rated[i] else (0, 0)
        reached, ep, em, ec, ew = game_events(games["moves_uci"][i], max_ply)
        e_cols["pos"].append(np.fromiter(ep, np.uint64, len(ep)))
        e_cols["move"].append(np.fromiter(em, np.uint16, len(em)))
        e_cols["child"].append(np.fromiter(ec, np.uint64, len(ec)))
        e_cols["white"].append(np.fromiter(ew, bool, len(ew)))
        e_cols["g"].append(np.full(len(ep), i, np.uint32))
        p_cols["pos"].append(np.fromiter(reached.keys(), np.uint64, len(reached)))
        p_cols["g"].append(np.full(len(reached), i, np.uint32))
        p_cols["rep"].append((row0 + i) * 256 + np.fromiter(reached.values(), np.uint64, len(reached)))
    cat = {k: (np.concatenate(v) if v else np.empty(0)) for k, v in e_cols.items()}
    g = cat["g"].astype(np.int64)
    edges = pl.DataFrame({
        "pos_hash": cat["pos"].astype(np.uint64),
        "move": cat["move"].astype(np.uint16),
        "child_hash": cat["child"].astype(np.uint64),
        "year": year[g], "elo_band": band[g], "tempo": tempo[g], "res": res[g],
        "welo": welo[g], "belo": belo[g], "rated": rated[g], "mover_white": cat["white"].astype(bool),
    })
    pc = {k: (np.concatenate(v) if v else np.empty(0)) for k, v in p_cols.items()}
    pg = pc["g"].astype(np.int64)
    positions = pl.DataFrame({"pos_hash": pc["pos"].astype(np.uint64), "res": res[pg], "rep": pc["rep"].astype(np.uint64)})
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
        pl.col("n", "white_wins", "draws", "black_wins").sum(), pl.col("rep").min()
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


def epds_from_reps(reps: Iterable[int], moves_of_row) -> dict[int, str]:
    """{rep: epd} for reps (row * 256 + ply), replaying each needed game once."""
    by_row: dict[int, list[int]] = {}
    for rep in reps:
        by_row.setdefault(rep >> 8, []).append(rep & 255)
    out: dict[int, str] = {}
    for row, plies in by_row.items():
        wanted = set(plies)
        board = chess.Board()
        moves = moves_of_row(row).split()
        for ply in range(max(wanted) + 1):
            if ply in wanted:
                out[(row << 8) | ply] = board.epd()
            if ply < len(moves):
                board.push_uci(moves[ply])
    return out


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
    ev, pv = batch_events(cols, 0, max_ply)
    edges = aggregate_edges(ev)
    positions = aggregate_positions(pv)
    kept = positions.filter(pl.col("n") >= min_games)
    edges = finish_edges(edges, kept)
    epd = epds_from_reps(kept["rep"].to_list(), lambda row: cols["moves_uci"][row])
    pos_index = finish_pos_index(kept, epd)
    return edges, pos_index


def finish_pos_index(kept: pl.DataFrame, epd: dict[int, str]) -> pl.DataFrame:
    reps = kept["rep"].to_list()
    out = kept.with_columns(epd=pl.Series([epd[r] for r in reps], dtype=pl.String))
    return out.select([pl.col(c).cast(t) for c, t in POS_INDEX_SCHEMA.items()]).sort("pos_hash")


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


class EdgeDB:
    """Queries on edges.parquet and pos_index.parquet. Lookups read one row group, not the table.

        db = EdgeDB()                     # corpus/edges.parquet, corpus/pos_index.parquet
        db = EdgeDB(name="dev")           # corpus/edges_dev.parquet, pos_index_dev.parquet
        db = EdgeDB.from_frames(e, p)     # the output of tables_from_games, for tests

    ``pos`` is a hash, a ``chess.Board`` or a FEN/EPD. Filters: ``years``
    (``year`` 0 is unknown), ``elo_bands`` (the floor of the mean rating to
    200, 0 when a rating is missing) and ``tempi``; each takes a value, an
    inclusive ``(lo, hi)`` tuple or a list.
    """

    def __init__(self, name: str = "full", directory: Path | None = None):
        d = directory or CORPUS
        suffix = "" if name == "full" else f"_{name}"
        self._edges = pl.scan_parquet(d / f"edges{suffix}.parquet")
        self._pos = pl.scan_parquet(d / f"pos_index{suffix}.parquet")

    @classmethod
    def from_frames(cls, edges: pl.DataFrame, pos_index: pl.DataFrame) -> EdgeDB:
        db = cls.__new__(cls)
        db._edges, db._pos = edges.lazy(), pos_index.lazy()
        return db

    def total(self, pos) -> dict | None:
        """The position's row in pos_index (all years, bands, tempi), or None below two games.

        ``n`` counts every game that reached the position, including those that
        ended there or reached it at ply 80; the moves' ``n`` count the games that
        went on from it.
        """
        rows = self._pos.filter(pl.col("pos_hash") == _key(pos)).collect()
        return rows.row(0, named=True) if rows.height else None

    def moves(self, pos, *, years: Filter = None, elo_bands: Filter = None, tempi: Filter = None) -> pl.DataFrame:
        """The moves played from the position under the filters, most played first.

        Columns: ``move``, ``child_hash``, ``n``, ``share`` (of the games here
        under these filters), ``white_wins``, ``draws``, ``black_wins``,
        ``score`` (for the side to move), ``mover_elo`` and ``opp_elo`` (means
        over the rated games), ``performance`` (``opp_elo + 800 * (score -
        0.5)``, the linear approximation; null without rated games),
        ``last_year`` (the last year the move was played, year 0 ignored; null
        when only games of unknown year played it).
        """
        return self._moves(pos, years, elo_bands, tempi, by=[])

    def moves_by_year(self, pos, *, years: Filter = None, elo_bands: Filter = None, tempi: Filter = None) -> pl.DataFrame:
        """As ``moves``, one row per move and year, sorted by year then games.

        ``share`` is the move's share of that year's games from the position;
        year 0 (unknown) is a row like any other.
        """
        return self._moves(pos, years, elo_bands, tempi, by=["year"])

    def _moves(self, pos, years: Filter, elo_bands: Filter, tempi: Filter, by: list[str]) -> pl.DataFrame:
        h = _key(pos)
        total = self.total(h)
        if total is None:
            return pl.DataFrame(schema={"move": pl.String, "child_hash": pl.UInt64, "n": pl.UInt32})
        white = total["epd"].split()[1] == "w"
        df = (
            self._edges.filter(pl.col("pos_hash") == h)
            .filter(_match("year", years) & _match("elo_band", elo_bands) & _match("tempo", tempi))
            .group_by(["move", "child_hash", *by])
            .agg(pl.col("n", "white_wins", "draws", "black_wins", "sum_mover_elo", "sum_opp_elo", "n_rated").sum(),
                 pl.col("year").filter(pl.col("year") > 0).max().alias("last_year"))
            .collect()
        )
        wins = pl.col("white_wins") if white else pl.col("black_wins")
        rated = pl.col("n_rated").cast(pl.Float64)
        df = df.with_columns(
            share=(pl.col("n") / pl.col("n").sum().over(by)) if by else (pl.col("n") / pl.col("n").sum()),
            score=(wins + 0.5 * pl.col("draws")) / pl.col("n"),
            mover_elo=pl.when(rated > 0).then(pl.col("sum_mover_elo") / rated),
            opp_elo=pl.when(rated > 0).then(pl.col("sum_opp_elo") / rated),
        ).with_columns(performance=pl.col("opp_elo") + 800 * (pl.col("score") - 0.5))
        cols = ["move", "child_hash", *by, "n", "share", "white_wins", "draws", "black_wins", "score",
                "mover_elo", "opp_elo", "performance", *([] if by else ["last_year"])]
        if by:
            return df.select(cols).sort([*by, "n", "move"], descending=[False, True, False])
        return df.select(cols).sort("n", "move", descending=[True, False])

    def children(self, pos, *, years: Filter = None, elo_bands: Filter = None, tempi: Filter = None) -> pl.DataFrame:
        """Each move from the position with its games under the filters (``n``) and the
        child position's own total over all games from pos_index (``child_n``, null when
        fewer than two games reached it, which also means it is not in edges.parquet)."""
        m = self.moves(pos, years=years, elo_bands=elo_bands, tempi=tempi).select("move", "child_hash", "n")
        if m.height == 0:
            return m.with_columns(child_n=pl.lit(None, pl.UInt32))
        kids = self._pos.filter(pl.col("pos_hash").is_in(m["child_hash"].implode())).select(
            pl.col("pos_hash").alias("child_hash"), pl.col("n").alias("child_n")).collect()
        return m.join(kids, on="child_hash", how="left", maintain_order="left")


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
        if len(buf["row"]) >= batch_games:
            yield k, buf
            k += 1
            buf = {c: [] for c in [*GAME_COLUMNS, "row"]}
    if buf["row"]:
        yield k, buf


def _bucket(df: pl.DataFrame) -> pl.Series:
    return pl.Series((df["pos_hash"].to_numpy() >> np.uint64(BUCKET_SHIFT)).astype(np.uint8))


def process_batch(args: tuple[int, dict, str, int]) -> tuple[int, int, int, int]:
    """Worker, pass 1: aggregate one batch and write it split into buckets.

    Returns (batch, games, edge rows written, position rows written).
    """
    k, cols, work, max_ply = args
    rows = cols.pop("row")
    ev, pv = batch_events(cols, 0, max_ply)
    # reps refer to the row in games.parquet
    row_arr = np.asarray(rows, np.uint64)
    pv = pv.with_columns(rep=pl.Series(row_arr[(pv["rep"].to_numpy() >> 8).astype(np.int64)] * 256 + (pv["rep"].to_numpy() & 255)))
    edges = aggregate_edges(ev)
    del ev
    positions = aggregate_positions(pv)
    del pv
    for name, df in (("edges", edges), ("positions", positions)):
        df = df.with_columns(_b=_bucket(df))
        for (b,), part in df.partition_by("_b", as_dict=True).items():
            d = Path(work) / name / f"b{b:02d}"
            d.mkdir(parents=True, exist_ok=True)
            part.drop("_b").write_parquet(d / f"k{k:05d}.parquet", compression="lz4")
    return k, len(rows), edges.height, positions.height


def merge_bucket(args: tuple[int, str, int]) -> tuple[int, int, int, int]:
    """Worker, pass 2: merge one bucket across batches, filter on the merged position counts.

    Writes the bucket's edges (final schema) and its kept positions (with the
    representative game, EPD still to come). Returns (bucket, distinct
    positions, kept positions, kept edge rows).
    """
    b, work, min_games = args
    w = Path(work)
    (w / "merged").mkdir(exist_ok=True)
    if not (w / "positions" / f"b{b:02d}").exists():  # a small build can leave a bucket empty
        pl.DataFrame(schema=EDGES_SCHEMA).write_parquet(w / "merged" / f"edges_b{b:02d}.parquet")
        pl.DataFrame(schema={"pos_hash": pl.UInt64, "n": pl.UInt32, "white_wins": pl.UInt32, "draws": pl.UInt32,
                             "black_wins": pl.UInt32, "rep": pl.UInt64}).write_parquet(w / "merged" / f"pos_b{b:02d}.parquet")
        (w / "merged" / f"hist_b{b:02d}.json").write_text(json.dumps({"ge2": 0, "ge10": 0, "ge100": 0, "ge1000": 0}))
        return b, 0, 0, 0
    positions = aggregate_positions(pl.read_parquet(w / "positions" / f"b{b:02d}" / "*.parquet"))
    kept = positions.filter(pl.col("n") >= min_games)
    hist = positions.select(
        *[(pl.col("n") >= t).sum().alias(f"ge{t}") for t in (2, 10, 100, 1000)]
    ).row(0, named=True)
    del positions
    if (w / "edges" / f"b{b:02d}").exists():
        edges = finish_edges(aggregate_edges(pl.read_parquet(w / "edges" / f"b{b:02d}" / "*.parquet")), kept)
        shutil.rmtree(w / "edges" / f"b{b:02d}")
    else:  # only final positions fell in this bucket
        edges = pl.DataFrame(schema=EDGES_SCHEMA)
    edges.write_parquet(w / "merged" / f"edges_b{b:02d}.parquet", row_group_size=65_536)
    kept.sort("pos_hash").write_parquet(w / "merged" / f"pos_b{b:02d}.parquet")
    (w / "merged" / f"hist_b{b:02d}.json").write_text(json.dumps(hist))
    shutil.rmtree(w / "positions" / f"b{b:02d}")
    return b, hist["ge2"], kept.height, edges.height


def build(*, name: str, out_dir: Path, fraction: float | None, since_year: int | None,
          batch_games: int, workers: int, max_ply: int = MAX_PLY, min_games: int = MIN_GAMES) -> None:
    t0 = time.time()
    games_path = CORPUS / "games.parquet"
    work = DATA_ROOT / "scratch" / "edges" / f"work_{name}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    ctx = multiprocessing.get_context("spawn")

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:7.0f} s] {msg}", flush=True)

    log(f"pass 1: games in batches of {batch_games}, {workers} workers -> {work}")
    n_games = 0
    with ctx.Pool(workers) as pool:
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
                pending.append(pool.apply_async(process_batch, ((k, cols, str(work), max_ply),)))
            k, g, er, pr = pending.pop(0).get()
            n_games += g
            log(f"  batch {k}: {g} games, {er:,} edge rows, {pr:,} position rows; {n_games:,} games so far")
    partial_bytes = sum(f.stat().st_size for f in work.rglob("*.parquet"))
    log(f"pass 1 done: {n_games:,} games, partials {partial_bytes / 1e9:.2f} GB")

    log("pass 2: merge buckets")
    kept_pos = kept_edges = 0
    with ctx.Pool(workers) as pool:
        for b, _, kp, ke in pool.imap_unordered(merge_bucket, [(b, str(work), min_games) for b in range(N_BUCKETS)]):
            kept_pos += kp
            kept_edges += ke
    log(f"pass 2 done: {kept_pos:,} positions with >= {min_games} games, {kept_edges:,} edge rows")

    log("pass 3: representative EPDs")
    # (rep, pos_hash) of every kept position, sorted by rep: 16 bytes a position
    pos_files = sorted((work / "merged").glob("pos_b*.parquet"))
    rp = pl.concat([pl.read_parquet(f, columns=["rep", "pos_hash"]) for f in pos_files]).sort("rep")
    reps, hashes = rp["rep"].to_numpy(), rp["pos_hash"].to_numpy()
    del rp
    n_epd = 0
    for k, cols in iter_game_batches(games_path, batch_games, fraction, since_year):
        lo, hi = cols["row"][0] * 256, (cols["row"][-1] + 1) * 256
        i0, i1 = np.searchsorted(reps, lo), np.searchsorted(reps, hi)
        if i0 == i1:
            continue
        moves_of = dict(zip(cols["row"], cols["moves_uci"], strict=True))
        epd = epds_from_reps(reps[i0:i1].tolist(), moves_of.__getitem__)
        part = pl.DataFrame({"pos_hash": hashes[i0:i1], "epd": [epd[int(r)] for r in reps[i0:i1]]},
                            schema={"pos_hash": pl.UInt64, "epd": pl.String})
        part = part.with_columns(_b=_bucket(part))
        for (bb,), sub in part.partition_by("_b", as_dict=True).items():
            d = work / "epd" / f"b{bb:02d}"
            d.mkdir(parents=True, exist_ok=True)
            sub.drop("_b").write_parquet(d / f"k{k:05d}.parquet", compression="lz4")
        n_epd += i1 - i0
        log(f"  batch {k}: {i1 - i0:,} EPDs")
    del reps, hashes
    if n_epd != kept_pos:
        raise RuntimeError(f"{n_epd:,} EPDs for {kept_pos:,} kept positions")

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if name == "full" else f"_{name}"
    edges_out, pos_out = out_dir / f"edges{suffix}.parquet", out_dir / f"pos_index{suffix}.parquet"
    ew = pw = None
    hist = {"ge2": 0, "ge10": 0, "ge100": 0, "ge1000": 0}
    for b in range(N_BUCKETS):
        e = pl.read_parquet(work / "merged" / f"edges_b{b:02d}.parquet")
        p = pl.read_parquet(work / "merged" / f"pos_b{b:02d}.parquet")
        if p.height:
            p = p.join(pl.read_parquet(work / "epd" / f"b{b:02d}" / "*.parquet"), on="pos_hash", how="left")
        else:
            p = p.with_columns(epd=pl.lit(None, pl.String))
        if p["epd"].null_count():
            raise RuntimeError(f"bucket {b}: {p['epd'].null_count()} positions without an EPD")
        p = p.select([pl.col(c).cast(t) for c, t in POS_INDEX_SCHEMA.items()])
        for k2, v in json.loads((work / "merged" / f"hist_b{b:02d}.json").read_text()).items():
            hist[k2] += v
        te, tp = e.to_arrow(), p.to_arrow()
        if ew is None:
            ew = pq.ParquetWriter(str(edges_out) + ".tmp", te.schema, compression="zstd")
            pw = pq.ParquetWriter(str(pos_out) + ".tmp", tp.schema, compression="zstd")
        ew.write_table(te, row_group_size=65_536)
        pw.write_table(tp, row_group_size=65_536)
    ew.close()
    pw.close()
    Path(str(edges_out) + ".tmp").replace(edges_out)
    Path(str(pos_out) + ".tmp").replace(pos_out)
    shutil.rmtree(work)
    log(f"wrote {edges_out} ({edges_out.stat().st_size / 1e9:.3f} GB, {kept_edges:,} rows) and "
        f"{pos_out} ({pos_out.stat().st_size / 1e9:.3f} GB, {kept_pos:,} rows)")
    log("positions with at least 2 / 10 / 100 / 1000 games: "
        f"{hist['ge2']:,} / {hist['ge10']:,} / {hist['ge100']:,} / {hist['ge1000']:,}; "
        f"games {n_games:,}; partials peaked at {partial_bytes / 1e9:.2f} GB")


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
