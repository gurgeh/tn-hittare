"""Fixed samples of position rows, the rows features are computed on.

    uv run python -m tnh.eval.sample --name test --split test --n 400000 [--extra-frac 0.25]
    uv run python -m tnh.eval.sample --name fit --split train --n 400000

Writes ``corpus/samples/<name>.parquet`` in the positions schema, plus these columns:

- ``tournament_id``, ``tempo_source``: from the game; the bootstrap clusters on the first.
- ``in_window``: ``tnh.data.window.in_window`` with its defaults (or the options given).
- ``out_of_book``: the row is at or after the game's out-of-book ply.
- ``part``: ``window`` for rows in the window, ``extra`` for rows outside it.
- ``exit_ply_<k>`` for each ``k`` in ``--exit-counts`` (default: the window's ``book_count``):
  the game's first ply with ``pos_count < k``, over all of its rows; null when the game
  never gets there before the corpus's ``max_ply``. ``tnh.eval.book`` counts plies from it.
- ``bucket``: the mover's rating, 100-point bucket (lower edge).
- ``weight``: population rows per sampled row in the row's stratum (part x bucket),
  so a weighted mean over the sample estimates the mean over the population.

And ``<name>.json``: the options, the population and sample size of each stratum;
and ``<name>.window.parquet``, the window rows alone, for feature runs that only
the window analyses need (the Elo sweep, the engines).

The population, in order (``in_window`` is evaluated on all of a game's rows
before anything else is dropped, as DESIGN.md requires):

1. Games: ``tempo == classical``, ``has_clock``, ``tempo_source != assumed``,
   both ratings in ``[elo_min, elo_max]`` (inclusive; the top value folds into the
   bucket below), ``split == --split``.
2. ``window`` rows: ``in_window``. ``extra`` rows: not ``in_window`` and either at
   ply >= ``--extra-min-ply`` or out of book. The extras are what the clock
   breakdown and the common window with the blitz reference need: rows in time
   trouble, after ``max_ply``, and still in book at ply 20 or later.

Memory: the corpus is read in batches of ``--batch-games`` games (narrow columns
only), keeping per stratum only the candidate rows with the smallest keys; the wide
columns are read for the chosen rows alone. The result does not depend on the batch
size.

Stratification: ``--n`` window rows split equally over the rating buckets; a
bucket with fewer rows gives all it has and the rest is shared by the others
(water-filling). ``extra`` rows: ``--extra-frac * n``, allocated the same way.
Within a stratum the rows with the smallest keys are taken, where the key is a
64-bit hash of (seed, game_id, ply). The draw is deterministic, does not depend on
row order, and a larger ``--n`` with the same seed gives a superset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from tnh.data.window import in_window
from tnh.eval.common import BUCKET, bucket_of
from tnh.paths import CORPUS

GAME_COLS = ["game_id", "tournament_id", "tempo", "tempo_source", "has_clock", "split", "white_elo", "black_elo"]


def splitmix64(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64)
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def row_keys(game_id: pl.Series, ply: pl.Series, seed: int) -> np.ndarray:
    """A deterministic pseudo-random 64-bit key per (game_id, ply). game_id is 16 hex chars."""
    hi = game_id.str.slice(0, 8).str.to_integer(base=16).cast(pl.UInt64).to_numpy()
    lo = game_id.str.slice(8, 8).str.to_integer(base=16).cast(pl.UInt64).to_numpy()
    g = (hi << np.uint64(32)) | lo
    with np.errstate(over="ignore"):
        x = splitmix64(g ^ splitmix64(np.full(len(g), seed, dtype=np.uint64)))
        return splitmix64(x + ply.to_numpy().astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15))


def allocate(available: dict, n: int) -> dict:
    """Equal allocation over strata, capped by what each has; leftovers go to the others."""
    alloc = {k: 0 for k in available}
    left = n
    open_ = {k for k, v in available.items() if v > 0}
    while left > 0 and open_:
        share = left // len(open_)
        if share == 0:
            for k in sorted(open_, key=lambda k: -available[k] + alloc[k])[:left]:
                alloc[k] += 1
            break
        for k in sorted(open_):
            take = min(share, available[k] - alloc[k])
            alloc[k] += take
            left -= take
        open_ = {k for k in open_ if alloc[k] < available[k]}
    return alloc


# --- working through the corpus in batches of games ------------------------------------
#
# positions.parquet has 212 million rows and is sorted by game_id. Nothing here holds more
# than one batch of games' narrow rows, plus the candidates: per stratum, the rows with the
# smallest keys, at most as many as the stratum could ever be allocated. Only the rows finally
# chosen are read with their wide columns (fen, pos_key, ...).

NARROW = ["game_id", "ply", "pos_count", "clock_mover_s", "mover_elo", "opp_elo"]


def drop_cache(path: Path) -> None:
    """Drop the page cache of a file just read. A capped job's memory counts the page cache it caused, so
    reading the 5 GB positions.parquet would otherwise look like 5 GB of memory to the cap."""
    import os
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def peak_gb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def eligible_games(games: pl.DataFrame, a, splits: list[str]) -> pl.DataFrame:
    g = games.select(GAME_COLS).filter(
        (pl.col("tempo") == "classical") & pl.col("has_clock") & (pl.col("tempo_source") != "assumed")
        & pl.col("white_elo").is_between(a.elo_min, a.elo_max) & pl.col("black_elo").is_between(a.elo_min, a.elo_max)
        & pl.col("split").is_in(splits)
    ).select("game_id", "tournament_id", "tempo_source").sort("game_id")
    frac = getattr(a, "tournament_frac", 1.0)
    if frac < 1.0:
        # whole tournaments at random with the seed; a weighted mean over them still estimates the
        # population mean (the weights are scaled by the same 1/frac everywhere, which cancels)
        import hashlib
        ts = g.get_column("tournament_id").unique().sort().to_list()
        keep = [t for t in ts if int.from_bytes(hashlib.blake2b(f"{a.seed}:{t}".encode(), digest_size=8).digest(),
                                                "little") / 2 ** 64 < frac]
        g = g.filter(pl.col("tournament_id").is_in(keep))
        print(f"tournaments: {len(keep)} of {len(ts)} drawn (frac {frac})")
    return g


def narrow_batch(positions: Path, ids: list[str], columns: list[str] = NARROW, extra=None) -> pl.DataFrame:
    """The narrow rows of the games ``ids`` (sorted). The range filter lets parquet skip row groups."""
    q = (pl.scan_parquet(positions).select(columns)
         .filter((pl.col("game_id") >= ids[0]) & (pl.col("game_id") <= ids[-1]))
         .filter(pl.col("game_id").is_in(ids)))
    if extra is not None:
        q = q.filter(extra)
    return q.collect().with_columns(pl.col("ply").cast(pl.Int64))


def candidates(positions: Path, ids: list[str], batch_games: int, annotate, strata: list[str],
               cap, seed: int, log=print) -> tuple[dict, pl.DataFrame]:
    """Population counts per stratum and, per stratum, the ``cap(stratum)`` rows with the smallest keys
    (a bound on what the stratum can ever be allocated).

    ``annotate`` turns one batch of whole games' narrow rows into rows with the ``strata`` columns
    (rows outside every stratum dropped); it must only look within a game."""
    counts: dict = {}
    keep = None
    for i in range(0, len(ids), batch_games):
        b = annotate(narrow_batch(positions, ids[i:i + batch_games]))
        drop_cache(positions)
        b = b.with_columns(pl.Series("_key", row_keys(b.get_column("game_id"), b.get_column("ply"), seed)))
        for *k, n in b.group_by(strata).len().iter_rows():
            counts[tuple(k)] = counts.get(tuple(k), 0) + n
        keep = b if keep is None else pl.concat([keep, b], how="vertical_relaxed")
        keep = trim(keep, strata, cap)
        log(f"  batch {i // batch_games + 1}/{-(-len(ids) // batch_games)}: {b.height} rows, "
            f"{keep.height} candidates, peak RSS {peak_gb():.2f} GB")
    return counts, keep


def trim(df: pl.DataFrame, strata: list[str], cap) -> pl.DataFrame:
    parts = []
    for key, g in df.group_by(strata, maintain_order=True):
        parts.append(g.sort("_key").head(cap(key)))
    return pl.concat(parts) if parts else df


def choose(cand: pl.DataFrame, counts: dict, strata: list[str], n_of, alloc_over: int) -> tuple[pl.DataFrame, list[dict]]:
    """Within each group of the first ``alloc_over`` stratum columns, allocate ``n_of(group)`` rows equally
    over the rest (water-filling) and take the smallest keys. Weight = population / sampled."""
    out, info = [], []
    groups: dict = {}
    for k, n in counts.items():
        groups.setdefault(k[:alloc_over], {})[k[alloc_over:]] = n
    for head, avail in sorted(groups.items(), key=lambda x: str(x[0])):
        alloc = allocate(avail, n_of(head))
        for tail, k in sorted(alloc.items(), key=lambda x: str(x[0])):
            if k == 0:
                continue
            key = head + tail
            f = pl.lit(True)
            for c, v in zip(strata, key):
                f = f & (pl.col(c) == v)
            rows = cand.filter(f).sort("_key").head(k)
            if rows.height != k:
                raise RuntimeError(f"stratum {key}: {rows.height} candidates for {k} rows")
            pop = avail[tail]
            out.append(rows.with_columns(pl.lit(pop / k).alias("weight")))
            info.append({**dict(zip(strata, [int(x) if isinstance(x, (int, np.integer)) else x for x in key])),
                         "population": int(pop), "sampled": int(k)})
    return pl.concat(out), info


REPLAY_COLS = ["game_id", "source", "white_elo", "black_elo", "tempo", "moves_uci", "clocks_s", "time_control_raw",
               "tempo_source", "inc_s", "result", "split"]


def fetch_wide(games_path: Path, chosen: pl.DataFrame, workers: int = 2, max_ply: int = 80) -> pl.DataFrame:
    """The positions-schema columns of the chosen (game_id, ply) rows, rebuilt by replaying only the chosen
    games with the data developer's own row builder (``tnh.data.positions.process_chunk``).

    Reading the wide columns from positions.parquet kept memory that grew with the part of the file read
    (polars and pyarrow alike, about 0.5 GB per 2 % of the file, measured 2026-11-11); replaying is bounded by
    the number of chosen games. ``pos_count`` is not rebuilt here: it comes with the narrow rows."""
    import multiprocessing
    import shutil
    import tempfile

    import pyarrow.parquet as pq

    from tnh.data.positions import process_chunk
    from tnh.paths import DATA_ROOT

    keys = chosen.select("game_id", pl.col("ply").cast(pl.Int64)).unique()
    ids = keys.get_column("game_id").unique().sort()
    tmp = Path(tempfile.mkdtemp(prefix="replay-", dir=DATA_ROOT / "scratch"))
    try:
        g = (pl.scan_parquet(games_path).select(REPLAY_COLS)
             .filter(pl.col("game_id").is_in(ids.implode())).collect().sort("game_id"))
        drop_cache(games_path)
        if g.height != ids.len():
            raise RuntimeError(f"{ids.len() - g.height} chosen games are not in {games_path}")
        gp = tmp / "games.parquet"
        g.write_parquet(gp, row_group_size=1000)
        n_groups = pq.ParquetFile(gp).metadata.num_row_groups
        del g
        parts = tmp / "parts"
        parts.mkdir()
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            for _ in pool.imap_unordered(process_chunk, [(i, str(gp), max_ply, str(parts)) for i in range(n_groups)]):
                pass
        out = []
        for f in sorted(parts.glob("part-*.parquet")):
            w = pl.read_parquet(f).drop("_h").with_columns(pl.col("ply").cast(pl.Int64))
            out.append(w.join(keys, on=["game_id", "ply"], how="semi"))
        w = pl.concat(out)
    finally:
        shutil.rmtree(tmp)
    if w.height != keys.height:
        raise RuntimeError(f"rebuilt {w.height} rows for {keys.height} chosen")
    return w


def subsample(s: pl.DataFrame, n: int, seed: int) -> tuple[pl.DataFrame, list[dict]]:
    """``n`` window rows of a sample, equal allocation over its rating buckets, smallest keys of
    hash(seed, game_id, ply); a smaller ``n`` with the same seed gives a subset. Weights are rescaled so that
    a weighted mean still estimates the population: population rows in the bucket / rows kept."""
    w = s.filter(pl.col("in_window"))
    w = w.with_columns(pl.Series("_key", row_keys(w.get_column("game_id"), w.get_column("ply").cast(pl.Int64), seed)))
    avail = dict(w.group_by("bucket").len().iter_rows())
    pop = dict(w.group_by("bucket").agg(pl.col("weight").sum()).iter_rows())
    alloc = allocate(avail, n)
    out, info = [], []
    for b in sorted(avail):
        k = alloc[b]
        if k == 0:
            continue
        rows = w.filter(pl.col("bucket") == b).sort("_key").head(k)
        out.append(rows.with_columns(pl.lit(pop[b] / k).alias("weight")))
        info.append({"bucket": int(b), "population": float(pop[b]), "sampled": int(k)})
    return pl.concat(out).drop("_key").sort("game_id", "ply"), info


def count_orphans(positions: Path, game_ids: pl.Series) -> int:
    """Game ids in positions.parquet that games.parquet lacks, one row group at a time (a unique over all
    212 million rows at once took more than 6 GB)."""
    import pyarrow.parquet as pq

    known = pl.DataFrame({"game_id": game_ids})
    f = pq.ParquetFile(positions)
    n = 0
    for r in range(f.metadata.num_row_groups):
        ids = pl.from_arrow(f.read_row_group(r, columns=["game_id"])).unique()
        n += ids.join(known, on="game_id", how="anti").height
        if r % 50 == 49:
            drop_cache(positions)
    drop_cache(positions)
    return n


def annotate_main(a):
    def f(p: pl.DataFrame) -> pl.DataFrame:
        exits = [pl.when(pl.col("pos_count") < k).then(pl.col("ply")).min().over("game_id").alias(f"exit_ply_{k}")
                 for k in a.exit_counts]
        first_rare = pl.when(pl.col("pos_count") < a.book_count).then(pl.col("ply")).min().over("game_id")
        p = p.with_columns(
            *exits,
            in_window(p, book_count=a.book_count, max_ply=a.max_ply, max_ply_no_clock=a.max_ply_no_clock,
                      min_clock_s=a.min_clock_s).alias("in_window"),
            (pl.col("ply") >= first_rare).fill_null(False).alias("out_of_book"),
            bucket_of("mover_elo", a.elo_min, a.elo_max).alias("bucket"),
        )
        p = p.with_columns(
            pl.when(pl.col("in_window")).then(pl.lit("window"))
            .when((pl.col("ply") >= a.extra_min_ply) | pl.col("out_of_book")).then(pl.lit("extra"))
            .otherwise(pl.lit(None)).alias("part"))
        return p.filter(pl.col("part").is_not_null()).select(
            "game_id", "ply", "pos_count", "part", "bucket", "in_window", "out_of_book",
            *[f"exit_ply_{k}" for k in a.exit_counts])
    return f


def draw_main(positions: Path, games: pl.DataFrame, a, batch_games: int, log=print, games_path: Path | None = None) -> tuple[pl.DataFrame, list[dict], dict]:
    eg = eligible_games(games, a, [a.split])
    ids = eg.get_column("game_id").to_list()
    if not ids:
        raise SystemExit("no eligible games")
    games_path = games_path or positions.parent / "games.parquet"
    n_for = {"window": a.n, "extra": int(round(a.extra_frac * a.n))}
    counts, cand = candidates(positions, ids, batch_games, annotate_main(a), ["part", "bucket"],
                              lambda k: n_for[k[0]], a.seed, log)
    chosen, strata = choose(cand, counts, ["part", "bucket"], lambda h: n_for[h[0]], 1)
    log(f"chose {chosen.height} rows, peak RSS {peak_gb():.2f} GB")
    wide = fetch_wide(games_path, chosen.select("game_id", "ply"))
    log(f"fetched the wide columns, peak RSS {peak_gb():.2f} GB")
    s = (wide.join(chosen.drop("_key"), on=["game_id", "ply"], how="inner")
         .join(eg, on="game_id", how="left").sort("game_id", "ply"))
    pop = {"games": len(ids), "rows_by_part": {p_: sum(n for k, n in counts.items() if k[0] == p_) for p_ in n_for}}
    return s, strata, pop


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m tnh.eval.sample", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="output: corpus/samples/<name>.parquet")
    p.add_argument("--split", required=True, choices=["train", "val", "test"])
    p.add_argument("--n", type=int, required=True, help="window rows")
    p.add_argument("--extra-frac", type=float, default=0.25, help="extra (out-of-window) rows per window row")
    p.add_argument("--extra-min-ply", type=int, default=20)
    p.add_argument("--exit-counts", type=int, nargs="+",
                   help="book_count values whose exit ply is stored per row (default: --book-count)")
    p.add_argument("--elo-min", type=int, default=1800)
    p.add_argument("--elo-max", type=int, default=2500)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch-games", type=int, default=20_000, help="games read at a time (memory)")
    p.add_argument("--tournament-frac", type=float, default=1.0,
                   help="draw this share of the split's tournaments first (whole tournaments, with the seed)")
    p.add_argument("--positions", type=Path, default=CORPUS / "positions.parquet")
    p.add_argument("--games", type=Path, help="default: games.parquet next to --positions")
    p.add_argument("--out-dir", type=Path, help="default: samples/ next to --positions")
    g = p.add_argument_group("window (tnh.data.window.in_window)")
    g.add_argument("--book-count", type=int, default=5)
    g.add_argument("--max-ply", type=int, default=60)
    g.add_argument("--max-ply-no-clock", type=int, default=40)
    g.add_argument("--min-clock-s", type=int, default=600)
    p.add_argument("--subsample-of", type=Path, help="draw --n window rows from this sample instead of the corpus")
    a = p.parse_args(argv)
    a.exit_counts = a.exit_counts or [a.book_count]
    if a.subsample_of:
        src = pl.read_parquet(a.subsample_of)
        s, info = subsample(src, a.n, a.seed)
        out = a.subsample_of.parent / f"{a.name}.parquet"
        s.write_parquet(out)
        meta = {"subsample_of": str(a.subsample_of), "n": a.n, "seed": a.seed, "rows": s.height,
                "games": s.get_column("game_id").n_unique(), "tournaments": s.get_column("tournament_id").n_unique(),
                "split": a.split, "strata": info,
                **{k: v for k, v in json.loads(a.subsample_of.with_suffix(".json").read_text()).items()
                   if k in ("elo_min", "elo_max", "book_count", "max_ply", "max_ply_no_clock", "min_clock_s",
                            "extra_frac", "positions")}}
        out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
        print(f"wrote {out}: {s.height} window rows, {meta['games']} games, {meta['tournaments']} tournaments")
        return
    games_path = a.games or a.positions.parent / "games.parquet"
    out_dir = a.out_dir or a.positions.parent / "samples"
    if a.elo_min % BUCKET:
        raise SystemExit("--elo-min must be a multiple of 100")

    games = pl.read_parquet(games_path, columns=GAME_COLS)
    print(f"games read, peak RSS {peak_gb():.2f} GB", flush=True)
    orphans = count_orphans(a.positions, games.get_column("game_id"))
    print(f"orphan check done, peak RSS {peak_gb():.2f} GB", flush=True)
    if orphans:
        raise SystemExit(f"{orphans} game_ids in {a.positions} are not in {games_path}: "
                         f"the two files are not from the same build")
    s, strata, pop = draw_main(a.positions, games, a, a.batch_games, games_path=games_path)
    print(f"population: {pop['games']} eligible games (split={a.split}, both ratings {a.elo_min}-{a.elo_max}); "
          f"rows by part {pop['rows_by_part']}")
    for st in strata:
        print(f"  {st['part']:>6} {st['bucket']}: {st['sampled']:>8} of {st['population']:>9}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{a.name}.parquet"
    s.write_parquet(out)
    s.filter(pl.col("in_window")).write_parquet(out_dir / f"{a.name}.window.parquet")
    meta = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    meta.update({"games_file": str(games_path), "rows": s.height, "games": s.get_column("game_id").n_unique(),
                 "tournaments": s.get_column("tournament_id").n_unique(), "strata": strata})
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"peak RSS {peak_gb():.2f} GB")
    print(f"wrote {out}: {s.height} rows, {meta['games']} games, {meta['tournaments']} tournaments")


if __name__ == "__main__":
    main()
