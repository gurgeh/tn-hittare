"""Build ``corpus/positions.parquet`` from ``corpus/games.parquet``.

    uv run python -m tnh.data.positions [--max-ply 80] [--workers N] [--chunk 5000]

One row per move to predict, from the games where ``tempo`` is classical and
both ratings are known, for plies below ``max_ply`` (schema in
docs/DESIGN.md). Games are replayed in worker processes, a chunk of games at
a time; each chunk's rows go to a part file, so memory stays flat.

``pos_count`` is the number of games with both ratings at least 1800, of any
tempo, that reach the position at any ply (the start and the final position
included, each game counted once). The workers return a 64-bit hash of every
``pos_key`` those games reach; the counts are taken over all of them, and
then joined to the rows part by part. With 179 million distinct keys
(broadcasts and TWIC, 2026-11) the chance of any collision at all is about
1 in 1 000, and a collision would move one count by a few games.

Besides ``positions.parquet`` it writes ``book_exit.parquet``, for the
report: per game, the first ply where the count is below 2, 5, 20 and 100,
counted over all sources (``exit5_all``) and over each source alone
(``exit5_only_lichess_broadcast``), so that the effect of adding a source on
where games leave book can be measured without a second build.
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing
import os
import shutil
import time

import chess
import numpy as np
import polars as pl
import pyarrow.parquet as pq

from tnh.data.timecontrol import parse_time_control
from tnh.paths import CORPUS

COUNT_MIN_ELO = 1800
BOOK_EXIT_COUNTS = (2, 5, 20, 100)
RESULT_WHITE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}

POSITIONS_SCHEMA = {
    "game_id": pl.String,
    "ply": pl.Int32,
    "fen": pl.String,
    "played_uci": pl.String,
    "white_to_move": pl.Boolean,
    "mover_elo": pl.Int32,
    "opp_elo": pl.Int32,
    "clock_mover_s": pl.Int32,
    "clock_opp_s": pl.Int32,
    "move_time_s": pl.Int32,
    "pos_key": pl.String,
    "pos_count": pl.Int32,
    "n_legal": pl.Int32,
    "result_mover": pl.Float32,
    "split": pl.String,
}


def pos_key(board: chess.Board) -> str:
    """Board, side to move, castling rights, en passant square (only when a capture is legal)."""
    return board.epd()


def key_hash(key: str) -> int:
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little")


def clock_before(clocks: list[int | None] | None, ply: int) -> int | None:
    """The clock of the side that moves at ``ply``, before that ply: its clock after its previous move.

    Null before a side's first move. The clock then is not the base time:
    many DGT clocks add the increment before move 1, and the first readings
    of 90+30 games are typically 5 430-5 460 s, so the base would understate
    the clock and make the first move's time negative.
    """
    if clocks is None or ply < 2:
        return None
    return clocks[ply - 2]


def move_times(clocks: list[int | None] | None, tc_raw: str | None, tempo_source: str, inc_s: int | None) -> list[int | None]:
    """Seconds spent on each ply: clock before - clock after + increment (+ the next period's time).

    Null where a clock or the increment is unknown, and where the result is
    negative by more than 2 s (a clock correction or an unknown period); -2..0
    is read as 0.
    """
    if clocks is None or inc_s is None:
        return [None] * (len(clocks) if clocks else 0)
    tc = parse_time_control(tc_raw)[0] if tempo_source == "tag" else None
    out: list[int | None] = []
    for ply, after in enumerate(clocks):
        before = clock_before(clocks, ply)
        if before is None or after is None:
            out.append(None)
            continue
        bonus = tc.bonus_after_move(ply // 2 + 1) if tc is not None else 0
        t = before - after + inc_s + bonus
        out.append(None if t < -2 else max(0, t))
    return out


def process_chunk(args: tuple[int, str, int, str]) -> tuple[int, dict[str, np.ndarray], int, int]:
    """Worker: replay one row group of the input games.

    Writes the rows to a part file and returns, per source, the hashes of all
    positions reached by that source's counted games (each game once per position).
    """
    idx, input_path, max_ply, parts_dir = args
    table = pq.ParquetFile(input_path).read_row_group(idx)
    cols = {c: table.column(c).to_pylist() for c in table.column_names}
    hashes: dict[str, list[np.ndarray]] = {}
    rows: dict[str, list] = {c: [] for c in POSITIONS_SCHEMA if c != "pos_count"}
    rows["_h"] = []
    n_null_move_time = 0
    for i in range(len(cols["game_id"])):
        welo, belo = cols["white_elo"][i], cols["black_elo"][i]
        counted = welo is not None and belo is not None and welo >= COUNT_MIN_ELO and belo >= COUNT_MIN_ELO
        emit = cols["tempo"][i] == "classical" and welo is not None and belo is not None
        if not (counted or emit):
            continue
        moves = cols["moves_uci"][i].split()
        board = chess.Board()
        if emit:
            clocks = cols["clocks_s"][i]
            mt = move_times(clocks, cols["time_control_raw"][i], cols["tempo_source"][i], cols["inc_s"][i])
            res_white = RESULT_WHITE[cols["result"][i]]
            gid, split = cols["game_id"][i], cols["split"][i]
        game_hashes = set()
        for ply, uci in enumerate(moves):
            key = pos_key(board)
            h = key_hash(key)
            game_hashes.add(h)
            if emit and ply < max_ply:
                white = board.turn == chess.WHITE
                rows["game_id"].append(gid)
                rows["ply"].append(ply)
                rows["fen"].append(board.fen())
                rows["played_uci"].append(uci)
                rows["white_to_move"].append(white)
                rows["mover_elo"].append(welo if white else belo)
                rows["opp_elo"].append(belo if white else welo)
                rows["clock_mover_s"].append(clock_before(clocks, ply))
                rows["clock_opp_s"].append(clock_before(clocks, ply + 1))
                rows["move_time_s"].append(mt[ply] if mt else None)
                n_null_move_time += clocks is not None and mt[ply] is None and clocks[ply] is not None
                rows["pos_key"].append(key)
                rows["n_legal"].append(board.legal_moves.count())
                rows["result_mover"].append(res_white if white else 1.0 - res_white)
                rows["split"].append(split)
                rows["_h"].append(h)
            elif not counted and ply >= max_ply:
                break
            board.push_uci(uci)
        if counted:
            game_hashes.add(key_hash(pos_key(board)))
            hashes.setdefault(cols["source"][i], []).append(np.fromiter(game_hashes, dtype=np.uint64, count=len(game_hashes)))
    schema = {**{c: t for c, t in POSITIONS_SCHEMA.items() if c != "pos_count"}, "_h": pl.UInt64}
    part = pl.DataFrame(rows, schema=schema)
    part.write_parquet(os.path.join(parts_dir, f"part-{idx:05d}.parquet"))
    return idx, {src: np.concatenate(hs) for src, hs in hashes.items()}, len(part), n_null_move_time


def lookup(keys: np.ndarray, counts: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Count of each hash in ``h`` (0 where absent), from sorted unique ``keys``."""
    if len(keys) == 0:
        return np.zeros(len(h), dtype=np.int32)
    pos = np.minimum(np.searchsorted(keys, h), len(keys) - 1)
    return np.where(keys[pos] == h, counts[pos], 0).astype(np.int32)


def build(max_ply: int, workers: int, chunk: int) -> None:
    t0 = time.time()
    games = pl.read_parquet(
        CORPUS / "games.parquet",
        columns=["game_id", "source", "moves_uci", "clocks_s", "white_elo", "black_elo", "tempo", "result", "split",
                 "inc_s", "time_control_raw", "tempo_source", "n_plies"],
    )
    both = pl.col("white_elo").is_not_null() & pl.col("black_elo").is_not_null()
    counted = both & (pl.col("white_elo") >= COUNT_MIN_ELO) & (pl.col("black_elo") >= COUNT_MIN_ELO)
    emit = both & (pl.col("tempo") == "classical")
    games = games.filter(counted | emit).sort("game_id")
    n_emit = games.filter(emit).height
    est_rows = games.filter(emit).select(pl.col("n_plies").clip(upper_bound=max_ply).sum()).item()
    est_count_keys = games.filter(counted).select((pl.col("n_plies") + 1).sum()).item()
    print(
        f"{n_emit:,} classical games with both ratings -> {est_rows:,} position rows (about "
        f"{est_rows * 200 / 1e9:.1f} GB in memory if built at once; written in parts instead)\n"
        f"{games.filter(counted).height:,} games with both ratings >= {COUNT_MIN_ELO} -> up to {est_count_keys:,} "
        f"position hashes for pos_count ({est_count_keys * 8 / 1e9:.2f} GB, twice that while counting)",
        flush=True,
    )

    parts_dir = CORPUS / "_positions_parts"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True)
    # the workers read their games from one row group each, so the games are not
    # held in this process as Python objects
    input_path = parts_dir / "games_input.parquet"
    games.write_parquet(input_path, row_group_size=chunk)
    n_groups = pq.ParquetFile(input_path).num_row_groups
    del games

    by_source: dict[str, list[np.ndarray]] = {}
    n_rows = 0
    n_null_mt = 0
    tasks = [(i, str(input_path), max_ply, str(parts_dir)) for i in range(n_groups)]
    # spawn, not fork: forking after polars has started its thread pool can deadlock
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for k, (_, hs, n, nn) in enumerate(pool.imap_unordered(process_chunk, tasks), 1):
            for src, h in hs.items():
                by_source.setdefault(src, []).append(h)
            n_rows += n
            n_null_mt += nn
            if k % 20 == 0 or k == len(tasks):
                print(f"  {k}/{len(tasks)} chunks, {n_rows:,} rows, {time.time() - t0:.0f} s", flush=True)

    # counts over all sources (pos_count), and per source for the book-exit table
    per_source: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for src in sorted(by_source):
        h = np.concatenate(by_source.pop(src))
        per_source[src] = np.unique(h, return_counts=True)
        del h
        print(f"  {src}: {len(per_source[src][0]):,} distinct positions", flush=True)
    keys = np.unique(np.concatenate([k for k, _ in per_source.values()]))
    counts = np.zeros(len(keys), dtype=np.int64)
    for k_src, c_src in per_source.values():
        counts[np.searchsorted(keys, k_src)] += c_src
    print(f"{len(keys):,} distinct positions reached by games with both ratings >= {COUNT_MIN_ELO}", flush=True)

    out = CORPUS / "positions.parquet"
    tmp = CORPUS / "positions.parquet.tmp"
    writer = None
    written = 0
    exits = []
    for part_path in sorted(parts_dir.glob("part-*.parquet")):
        part = pl.read_parquet(part_path)
        h = part["_h"].to_numpy()
        part = part.with_columns(pl.Series("pos_count", lookup(keys, counts, h), dtype=pl.Int32))
        only = {f"pc_{src}": lookup(k, c, h) for src, (k, c) in per_source.items()}
        exits.append(
            part.select("game_id", "ply", "pos_count").with_columns(**{k: pl.Series(v) for k, v in only.items()})
            .group_by("game_id")
            .agg(
                *[pl.col("ply").filter(pl.col("pos_count") < bc).min().alias(f"exit{bc}_all") for bc in BOOK_EXIT_COUNTS],
                *[pl.col("ply").filter(pl.col(k) < bc).min().alias(f"exit{bc}_only_{k[3:]}") for k in only for bc in BOOK_EXIT_COUNTS],
            )
        )
        part = part.select([pl.col(c).cast(t) for c, t in POSITIONS_SCHEMA.items()])
        table = part.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
        writer.write_table(table, row_group_size=500_000)
        written += part.height
    if writer is not None:
        writer.close()
    tmp.replace(out)
    pl.concat(exits).sort("game_id").write_parquet(CORPUS / "book_exit.parquet")
    shutil.rmtree(parts_dir)
    print(f"wrote {written:,} rows to {out} ({out.stat().st_size / 1e9:.2f} GB), "
          f"{n_null_mt:,} rows with a clock after the move but no move time "
          f"(each side's first move among them: no clock before it, by design), "
          f"{time.time() - t0:.0f} s", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-ply", type=int, default=80)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=5000, help="games per worker task")
    args = ap.parse_args(argv)
    build(args.max_ply, args.workers or os.cpu_count() or 1, args.chunk)


if __name__ == "__main__":
    main()
