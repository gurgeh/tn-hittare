"""Fixed samples of position rows, the rows features are computed on.

    uv run python -m tnh.eval.sample --name test --split test --n 400000 [--extra-frac 0.25]
    uv run python -m tnh.eval.sample --name fit --split train --n 400000

Writes ``corpus/samples/<name>.parquet`` in the positions schema, plus these columns:

- ``tournament_id``, ``tempo_source``: from the game; the bootstrap clusters on the first.
- ``in_window``: ``tnh.data.window.in_window`` with its defaults (or the options given).
- ``out_of_book``: the row is at or after the game's out-of-book ply.
- ``part``: ``window`` for rows in the window, ``extra`` for rows outside it.
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
    g = np.array([int(s[:16], 16) for s in game_id.to_list()], dtype=np.uint64)
    with np.errstate(over="ignore"):
        x = splitmix64(g ^ splitmix64(np.full(len(g), seed, dtype=np.uint64)))
        return splitmix64(x + ply.to_numpy().astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15))


def allocate(available: dict[int, int], n: int) -> dict[int, int]:
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


def population(positions: pl.LazyFrame, games: pl.DataFrame, a) -> pl.DataFrame:
    """Rows of the eligible games, with in_window computed on the whole game first."""
    g = games.select(GAME_COLS).filter(
        (pl.col("tempo") == "classical") & pl.col("has_clock") & (pl.col("tempo_source") != "assumed")
        & pl.col("white_elo").is_between(a.elo_min, a.elo_max) & pl.col("black_elo").is_between(a.elo_min, a.elo_max)
        & (pl.col("split") == a.split)
    ).drop("split", "white_elo", "black_elo", "tempo", "has_clock")
    p = positions.join(g.lazy(), on="game_id", how="semi").collect().with_columns(pl.col("ply").cast(pl.Int64))
    first_rare = pl.when(pl.col("pos_count") < a.book_count).then(pl.col("ply")).min().over("game_id")
    p = p.with_columns(
        in_window(p, book_count=a.book_count, max_ply=a.max_ply, max_ply_no_clock=a.max_ply_no_clock,
                  min_clock_s=a.min_clock_s).alias("in_window"),
        (pl.col("ply") >= first_rare).fill_null(False).alias("out_of_book"),
    )
    p = p.join(g, on="game_id", how="left")
    return p


def draw(pop: pl.DataFrame, a) -> tuple[pl.DataFrame, list[dict]]:
    pop = pop.with_columns(bucket_of("mover_elo", a.elo_min, a.elo_max).alias("bucket"))
    pop = pop.with_columns(
        pl.when(pl.col("in_window")).then(pl.lit("window"))
        .when((pl.col("ply") >= a.extra_min_ply) | pl.col("out_of_book")).then(pl.lit("extra"))
        .otherwise(pl.lit(None)).alias("part")
    ).filter(pl.col("part").is_not_null())
    pop = pop.with_columns(pl.Series("_key", row_keys(pop.get_column("game_id"), pop.get_column("ply"), a.seed)))
    out, strata = [], []
    for part, n in (("window", a.n), ("extra", int(round(a.extra_frac * a.n)))):
        pp = pop.filter(pl.col("part") == part)
        avail = dict(pp.group_by("bucket").len().iter_rows())
        alloc = allocate(avail, n)
        for b in sorted(avail):
            k = alloc[b]
            if k == 0:
                continue
            rows = pp.filter(pl.col("bucket") == b).sort("_key").head(k)
            out.append(rows.with_columns(pl.lit(avail[b] / k).alias("weight")))
            strata.append({"part": part, "bucket": int(b), "population": int(avail[b]), "sampled": int(k)})
    s = pl.concat(out).drop("_key").sort("game_id", "ply")
    return s, strata


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m tnh.eval.sample", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="output: corpus/samples/<name>.parquet")
    p.add_argument("--split", required=True, choices=["train", "val", "test"])
    p.add_argument("--n", type=int, required=True, help="window rows")
    p.add_argument("--extra-frac", type=float, default=0.25, help="extra (out-of-window) rows per window row")
    p.add_argument("--extra-min-ply", type=int, default=20)
    p.add_argument("--elo-min", type=int, default=1800)
    p.add_argument("--elo-max", type=int, default=2500)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--positions", type=Path, default=CORPUS / "positions.parquet")
    p.add_argument("--games", type=Path, help="default: games.parquet next to --positions")
    p.add_argument("--out-dir", type=Path, help="default: samples/ next to --positions")
    g = p.add_argument_group("window (tnh.data.window.in_window)")
    g.add_argument("--book-count", type=int, default=5)
    g.add_argument("--max-ply", type=int, default=60)
    g.add_argument("--max-ply-no-clock", type=int, default=40)
    g.add_argument("--min-clock-s", type=int, default=600)
    a = p.parse_args(argv)
    games_path = a.games or a.positions.parent / "games.parquet"
    out_dir = a.out_dir or a.positions.parent / "samples"
    if a.elo_min % BUCKET:
        raise SystemExit("--elo-min must be a multiple of 100")

    games = pl.read_parquet(games_path, columns=GAME_COLS)
    pop = population(pl.scan_parquet(a.positions), games, a)
    print(f"population: {pop.height} rows of {pop.get_column('game_id').n_unique()} eligible games "
          f"(split={a.split}, both ratings {a.elo_min}-{a.elo_max}); in window: {pop.get_column('in_window').sum()}")
    s, strata = draw(pop, a)
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
    print(f"wrote {out}: {s.height} rows, {meta['games']} games, {meta['tournaments']} tournaments")


if __name__ == "__main__":
    main()
