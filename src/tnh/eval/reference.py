"""The blitz reference: Maia-3 on Lichess blitz beside Maia-3 on classical games, under one rule.

    uv run python -m tnh.eval.reference --blitz DIR [DIR ...] --classical DIR [DIR ...] \\
        [--blitz-positions corpus/blitz_ref/window.parquet] [--test S] [--fit S] [--results R]

The common window, applied the same way to both kinds of game (no book rule,
since ``pos_count`` means nothing for the blitz games):

- blitz: ply 20 to 59, at least 30 s on the mover's clock (Maia-3's training data
  drops moves after a player has had under 30 s);
- classical: ply 20 to 59, and the clock rule of DESIGN.md (at least 600 s where
  the clock is known, ply < 40 where it is not). The rows come from both parts of
  the test sample (in-book rows at ply 20+ are in the ``extra`` part), weighted.

Each ``DIR`` is one Elo setting (``offset+X``). The table is by 100-point bucket of
the mover's rating: Lichess blitz ratings on one side, FIDE on the other, so a
bucket on one side is not the same players as the same bucket on the other. The
comparison that matters is each kind at its own best setting: per bucket, the
offset with the lowest raw log loss (chosen on the same rows, so slightly
optimistic on both sides alike). Raw log loss, no uniform mixture: there is no
blitz fitting sample, and the mixture would change both sides by little.

Writes ``<results>/reference.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from tnh.eval.blitz import window_expr
from tnh.eval.common import (add_common, boot_counts, boot_means, bucket_of, cluster_codes, est, label, move_scores,
                             parse_variant, provenance, read_sample, variant_name, wmean, write_result)
from tnh.paths import CORPUS

MIN_PLY, MAX_PLY = 20, 60


def classical_rows(s: pl.DataFrame, min_clock_s: int, max_ply_no_clock: int) -> pl.DataFrame:
    clock_ok = (pl.when(pl.col("clock_mover_s").is_null()).then(pl.col("ply") < max_ply_no_clock)
                .otherwise(pl.col("clock_mover_s") >= min_clock_s))
    return s.filter((pl.col("ply") >= MIN_PLY) & (pl.col("ply") < MAX_PLY) & clock_ok)


def side(s: pl.DataFrame, dirs: list[Path], B: int, seed: int) -> dict:
    w = s.get_column("weight").to_numpy()
    cl = cluster_codes(s)
    counts = boot_counts(cl.max() + 1, B, seed)
    b = s.get_column("bucket").to_numpy()
    ub = np.unique(b)
    g = np.searchsorted(ub, b)
    offs, L, T = [], [], []
    for d in dirs:
        v = parse_variant(variant_name(d))
        if v["kind"] != "offset" or v["mover"] != v["opp"]:
            raise SystemExit(f"{d}: the reference takes both-sides offsets only")
        sc, _ = move_scores(s, d)
        offs.append(v["mover"]); L.append(-sc.lp); T.append(sc.top1)
    order = np.argsort(offs)
    offs = [offs[i] for i in order]
    L = np.column_stack([L[i] for i in order])
    T = np.column_stack([T[i] for i in order])
    X = np.column_stack([L, T])
    bm = boot_means(X, w, cl, counts, groups=g, n_groups=len(ub))
    bp = boot_means(X, w, cl, counts)
    V = len(offs)
    pooled = [{"offset": o, "loss": est(wmean(L[:, v], w), bp[:, v]), "top1": est(wmean(T[:, v], w), bp[:, V + v])}
              for v, o in enumerate(offs)]
    rows = []
    for k, bk in enumerate(ub):
        m = g == k
        losses = [wmean(L[m, v], w[m]) for v in range(V)]
        best = int(np.argmin(losses))
        row = {"bucket": int(bk), "rows": int(m.sum()),
               "games": int(s.filter(pl.Series(m)).get_column("game_id").n_unique()),
               "by_offset": [{"offset": o, "loss": est(losses[v], bm[:, k, v]),
                              "top1": est(wmean(T[m, v], w[m]), bm[:, k, V + v])} for v, o in enumerate(offs)],
               "best_offset": offs[best], "best_loss": est(losses[best], bm[:, k, best])}
        if 0.0 in offs:
            z = offs.index(0.0)
            row["loss_at_0"] = est(losses[z], bm[:, k, z])
        rows.append(row)
    return {"offsets": offs, "rows": s.height, "games": s.get_column("game_id").n_unique(),
            "pooled": pooled, "by_bucket": rows, "labels": [label(dirs[i]) for i in order]}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m tnh.eval.reference", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blitz", type=Path, nargs="+", required=True)
    p.add_argument("--classical", type=Path, nargs="+", required=True)
    p.add_argument("--blitz-positions", type=Path, default=CORPUS / "blitz_ref" / "window.parquet")
    p.add_argument("--blitz-elo-min", type=int, default=1800)
    p.add_argument("--min-clock-s", type=int, default=600)
    p.add_argument("--max-ply-no-clock", type=int, default=40)
    add_common(p, fit=False)
    p.add_argument("--name", default="reference", help="result file name (without .json)")
    a = p.parse_args(argv)
    bz = pl.read_parquet(a.blitz_positions).with_columns(pl.col("ply").cast(pl.Int64))
    n0 = bz.height
    bz = bz.filter(window_expr() & (pl.col("mover_elo") >= a.blitz_elo_min) & (pl.col("opp_elo") >= a.blitz_elo_min))
    bz = bz.with_columns(pl.col("game_id").alias("tournament_id"), pl.lit(1.0).alias("weight"),
                         bucket_of("mover_elo").alias("bucket"))
    print(f"blitz: {bz.height} of {n0} rows in the common window")
    cs = classical_rows(read_sample(a.test), a.min_clock_s, a.max_ply_no_clock)
    print(f"classical: {cs.height} test-sample rows in the common window")
    blitz = side(bz, a.blitz, a.boot, a.seed)
    classical = side(cs, a.classical, a.boot, a.seed)
    for name, r in (("blitz", blitz), ("classical", classical)):
        for row in r["by_bucket"]:
            at0 = row.get("loss_at_0", {}).get("est", float("nan"))
            print(f"{name:>9} {row['bucket']}: rows={row['rows']:>7} games={row['games']:>6} at 0 {at0:.4f}; "
                  f"best {row['best_loss']['est']:.4f} at {row['best_offset']:+g}")
    write_result(a.results, a.name, {
        "what": "raw log loss of Maia-3 on Lichess blitz and on classical games, common window",
        "rule": {"blitz": "ply 20-59, clock >= 30 s", "classical": f"ply 20-59, clock >= {a.min_clock_s} s "
                 f"or no clock and ply < {a.max_ply_no_clock}; both sample parts, weighted"},
        "blitz_positions": str(a.blitz_positions), "boot": a.boot,
        "blitz": blitz, "classical": classical,
        "provenance": provenance({"test": a.test}, a.blitz + a.classical)})


if __name__ == "__main__":
    main()
