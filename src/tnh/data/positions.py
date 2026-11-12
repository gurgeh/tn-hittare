"""Build ``corpus/positions.parquet`` from ``corpus/games.parquet``.

    uv run python -m tnh.data.positions [--max-ply 80] [--workers N] [--chunk 5000]

One row per move to predict, from the games where ``tempo`` is classical and
both ratings are known, for plies below ``max_ply`` (schema in
docs/DESIGN.md). Games are replayed in worker processes, a chunk of games at
a time; each chunk's rows go to a part file, so memory stays flat.

``pos_count`` is the number of games with both ratings at least 1800, of any
tempo, that reach the position at any ply (the start and the final position
included, each game once). Each worker writes, per chunk, the sorted 64-bit
hashes of the positions its counted games reach, with counts and source; the
counts are then merged in 64 buckets on the top bits of the hash, so no step
holds every hash event, and joined to the rows part by part. With 179 million
distinct keys (broadcasts and TWIC, 2026-11) the chance of any collision at
all is about 1 in 1 000, and a collision would move one count by a few games.

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
from pathlib import Path

import chess
import numpy as np
import polars as pl
import pyarrow.parquet as pq

from tnh.data.timecontrol import parse_time_control
from tnh.paths import CORPUS, DATA_ROOT

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


def process_chunk(args: tuple) -> tuple[int, int, int, int]:
    """Worker: replay one row group of the input games.

    Writes the rows to a part file, and the positions reached by the chunk's
    counted games (each game once per position) as sorted unique hashes with
    their count and source: ``hashes/c<idx>_{keys,counts,src}.npy``.
    Returns (chunk, rows, rows with a clock but no move time, peak MB).

    ``args`` is (chunk, input path, max_ply, parts dir[, sources]); without
    ``sources`` (the four-element form ``tnh.eval.sample`` uses to rebuild
    rows) the source codes are the sorted sources of the chunk.
    """
    idx, input_path, max_ply, parts_dir = args[:4]
    sources = args[4] if len(args) > 4 else None
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
    part_path = Path(parts_dir) / f"part-{idx:05d}.parquet"
    part.write_parquet(part_path)
    _drop_cache(part_path)
    if sources is None:
        sources = sorted(hashes)
    keys_l, counts_l, src_l = [], [], []
    for si, src in enumerate(sources):
        if src in hashes:
            k, c = np.unique(np.concatenate(hashes.pop(src)), return_counts=True)
            keys_l.append(k)
            counts_l.append(c.astype(np.uint32))
            src_l.append(np.full(len(k), si, np.uint8))
    keys = np.concatenate(keys_l) if keys_l else np.empty(0, np.uint64)
    order = np.argsort(keys, kind="stable")
    hdir = Path(parts_dir) / "hashes"
    hdir.mkdir(exist_ok=True)
    for name, arr in (("keys", keys[order]),
                      ("counts", (np.concatenate(counts_l) if counts_l else np.empty(0, np.uint32))[order]),
                      ("src", (np.concatenate(src_l) if src_l else np.empty(0, np.uint8))[order])):
        np.save(hdir / f"c{idx:05d}_{name}.npy", arr)
        _drop_cache(hdir / f"c{idx:05d}_{name}.npy")
    return idx, len(part), n_null_move_time, _peak_mb()


N_BUCKETS = 64
BUCKET_SHIFT = 58


def merge_bucket(args: tuple[int, str, int, int]) -> tuple[int, int]:
    """Worker: the counts of one bucket of hashes (top six bits), over all chunks and per source.

    Reads each chunk's slice of the bucket from memory-mapped arrays, so only
    that slice is touched. Writes ``buckets/b<b>.npz`` (keys, total, one count
    array per source). Returns (bucket, distinct keys).
    """
    b, parts_dir, n_chunks, n_sources = args
    hdir = Path(parts_dir) / "hashes"
    lo, hi = np.uint64(b) << np.uint64(BUCKET_SHIFT), (np.uint64(b + 1) << np.uint64(BUCKET_SHIFT)) if b + 1 < N_BUCKETS else None
    ks, cs, ss = [], [], []
    for c in range(n_chunks):
        k = np.load(hdir / f"c{c:05d}_keys.npy", mmap_mode="r")
        i0 = np.searchsorted(k, lo)
        i1 = np.searchsorted(k, hi) if hi is not None else len(k)
        ks.append(np.array(k[i0:i1]))
        cs.append(np.array(np.load(hdir / f"c{c:05d}_counts.npy", mmap_mode="r")[i0:i1]))
        ss.append(np.array(np.load(hdir / f"c{c:05d}_src.npy", mmap_mode="r")[i0:i1]))
        del k
    keys = np.concatenate(ks) if ks else np.empty(0, np.uint64)
    counts = np.concatenate(cs) if cs else np.empty(0, np.uint32)
    src = np.concatenate(ss) if ss else np.empty(0, np.uint8)
    del ks, cs, ss
    uk, inv = np.unique(keys, return_inverse=True)
    out = {"keys": uk, "total": np.bincount(inv, weights=counts, minlength=len(uk)).astype(np.uint32)}
    for si in range(n_sources):
        m = src == si
        out[f"s{si}"] = np.bincount(inv[m], weights=counts[m], minlength=len(uk)).astype(np.uint32)
    np.savez(Path(parts_dir) / "buckets" / f"b{b:02d}.npz", **out)
    _drop_cache(Path(parts_dir) / "buckets" / f"b{b:02d}.npz")
    return b, len(uk)


def _drop_cache(path: Path) -> None:
    """Write a file's dirty pages and drop it from the page cache, which tnh-run counts as memory."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _peak_mb() -> int:
    """This process's peak resident memory (VmHWM, which starts afresh at exec), MB."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) // 1024
    raise RuntimeError("no VmHWM in /proc/self/status")


def lookup(keys: np.ndarray, counts: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Count of each hash in ``h`` (0 where absent), from sorted unique ``keys``."""
    if len(keys) == 0:
        return np.zeros(len(h), dtype=np.int32)
    pos = np.minimum(np.searchsorted(keys, h), len(keys) - 1)
    return np.where(keys[pos] == h, counts[pos], 0).astype(np.int32)


INPUT_COLUMNS = ["game_id", "source", "moves_uci", "clocks_s", "white_elo", "black_elo", "tempo", "result", "split",
                 "inc_s", "time_control_raw", "tempo_source", "n_plies"]


def build(max_ply: int, workers: int, chunk: int, out_dir: Path | None = None, fraction: float | None = None) -> None:
    """Replay the games in chunks, count positions in buckets on disk, then write the rows with pos_count.

    Memory: one input batch, one bucket of hashes per worker, and in the last
    step the merged counts of every distinct position (about 12 bytes plus 4
    per source each: some 3.5 GB for the 179 M positions of the 2026-11 corpus).
    """
    t0 = time.time()
    out_dir = out_dir or CORPUS
    parts_dir = DATA_ROOT / "scratch" / "positions" / "parts"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    (parts_dir / "hashes").mkdir(parents=True)
    (parts_dir / "buckets").mkdir()
    os.environ["POLARS_MAX_THREADS"] = "1"  # read by the spawned workers; the worker count is the CPU budget

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:6.0f} s] {msg}", flush=True)

    # 1. the input: the games that give rows or are counted, in row groups of `chunk`, streamed
    games_path = CORPUS / "games.parquet"
    input_path = parts_dir / "games_input.parquet"
    writer = None
    sources: set[str] = set()
    buf: list[pl.DataFrame] = []
    n_buf = n_in = n_emit = est_rows = 0
    for rb in pq.ParquetFile(games_path).iter_batches(batch_size=chunk, columns=INPUT_COLUMNS):
        g = pl.from_arrow(rb)
        both = pl.col("white_elo").is_not_null() & pl.col("black_elo").is_not_null()
        counted = both & (pl.col("white_elo") >= COUNT_MIN_ELO) & (pl.col("black_elo") >= COUNT_MIN_ELO)
        emit = both & (pl.col("tempo") == "classical")
        g = g.filter(counted | emit)
        if fraction is not None:
            g = g.filter(pl.col("game_id").str.slice(0, 8).str.to_integer(base=16) % 100_000 < int(fraction * 100_000))
        sources.update(g["source"].unique().to_list())
        n_emit += g.filter(emit).height
        est_rows += g.filter(emit).select(pl.col("n_plies").clip(upper_bound=max_ply).sum()).item()
        buf.append(g)
        n_buf += g.height
        while n_buf >= chunk:
            allg = pl.concat(buf)
            t = allg.head(chunk).to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(input_path, t.schema)
            writer.write_table(t, row_group_size=chunk)
            n_in += t.num_rows
            buf, n_buf = [allg.slice(chunk)], allg.height - chunk
        _drop_cache(games_path)
    if n_buf:
        t = pl.concat(buf).to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(input_path, t.schema)
        writer.write_table(t, row_group_size=chunk)
        n_in += t.num_rows
    writer.close()
    _drop_cache(input_path)
    n_groups = pq.ParquetFile(input_path).num_row_groups
    sources_l = sorted(sources)
    log(f"{n_in:,} games in {n_groups} chunks; {n_emit:,} classical games with both ratings -> {est_rows:,} rows")

    # 2. replay
    n_rows = n_null_mt = 0
    tasks = [(i, str(input_path), max_ply, str(parts_dir), sources_l) for i in range(n_groups)]
    ctx = multiprocessing.get_context("spawn")  # not fork: forking after polars has started its threads can deadlock
    with ctx.Pool(workers, maxtasksperchild=20) as pool:
        for k, (_, n, nn, peak) in enumerate(pool.imap_unordered(process_chunk, tasks), 1):
            n_rows += n
            n_null_mt += nn
            if k % 20 == 0 or k == len(tasks):
                log(f"  {k}/{len(tasks)} chunks, {n_rows:,} rows; worker peak {peak} MB")

    # 3. counts per bucket
    with ctx.Pool(workers, maxtasksperchild=8) as pool:
        n_keys = sum(n for _, n in pool.imap_unordered(
            merge_bucket, [(b, str(parts_dir), n_groups, len(sources_l)) for b in range(N_BUCKETS)]))
    shutil.rmtree(parts_dir / "hashes")
    loaded = [np.load(parts_dir / "buckets" / f"b{b:02d}.npz") for b in range(N_BUCKETS)]
    keys = np.concatenate([z["keys"] for z in loaded])
    counts = np.concatenate([z["total"] for z in loaded])
    per_source = {src: np.concatenate([z[f"s{si}"] for z in loaded]) for si, src in enumerate(sources_l)}
    del loaded
    shutil.rmtree(parts_dir / "buckets")
    log(f"{n_keys:,} distinct positions reached by games with both ratings >= {COUNT_MIN_ELO}")

    # 4. rows with pos_count, and the book-exit table
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "positions.parquet"
    tmp = out_dir / "positions.parquet.tmp"
    writer = None
    written = 0
    exits = []
    for part_path in sorted(parts_dir.glob("part-*.parquet")):
        part = pl.read_parquet(part_path)
        _drop_cache(part_path)
        h = part["_h"].to_numpy()
        part = part.with_columns(pl.Series("pos_count", lookup(keys, counts, h), dtype=pl.Int32))
        only = {f"pc_{src}": lookup(keys, c, h) for src, c in per_source.items()}
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
    _drop_cache(out)
    pl.concat(exits).sort("game_id").write_parquet(out_dir / "book_exit.parquet")
    shutil.rmtree(parts_dir)
    log(f"wrote {written:,} rows to {out} ({out.stat().st_size / 1e9:.2f} GB), "
        f"{n_null_mt:,} rows with a clock after the move but no move time "
        f"(each side's first move among them: no clock before it, by design)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-ply", type=int, default=80)
    ap.add_argument("--workers", type=int, default=2, help="replay processes (default 2; see AGENTS.md on the CPU budget)")
    ap.add_argument("--chunk", type=int, default=5000, help="games per worker task")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: corpus/)")
    ap.add_argument("--fraction", type=float, default=None, help="a fixed fraction of the games, by game_id hash (measurement)")
    args = ap.parse_args(argv)
    build(args.max_ply, args.workers, args.chunk, args.out, args.fraction)


if __name__ == "__main__":
    main()
