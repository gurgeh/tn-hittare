"""Where play gets less predictable: log loss by ply and by time left on the clock.

    uv run python -m tnh.eval.breakdown --model DIR [--test S] [--fit S] [--results R]

One move model (the best setting from the sweep). ``eps`` is fitted on the
fitting sample's window rows; the tables are on the test sample.

- By ply: window rows, 10-ply buckets.
- By clock: every out-of-book row before ``--max-ply`` with a known clock, from
  both parts of the sample (``window`` and ``extra``), weighted by the sample
  weights so that each bucket estimates the population mean. Buckets of minutes
  left: under 2, 2-5, 5-10, 10-20, 20-40, over 40. The rows with under 10
  minutes are the ones the window drops.
- Ply x clock: the same rows, because ply and time left move together and a
  late position is not the same kind of position as an early one. Mean entropy
  of the model's distribution and top-1 go beside the log loss, so that a rise in
  log loss can be told apart from positions that simply have more plausible moves.

Writes ``<results>/breakdown.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from tnh.eval.common import (add_common, boot_counts, boot_means, check_splits, cluster_codes, fit_eps, label,
                             mix_logp, move_scores, provenance, read_sample, summarize, window_rows, wmean,
                             write_result)

CLOCK_EDGES_MIN = [0, 2, 5, 10, 20, 40, 10_000]
CLOCK_LABELS = ["<2", "2-5", "5-10", "10-20", "20-40", ">40"]


def table(s: pl.DataFrame, x: np.ndarray, group: np.ndarray, labels: list, B: int, seed: int) -> list[dict]:
    w = s.get_column("weight").to_numpy()
    cl = cluster_codes(s)
    counts = boot_counts(cl.max() + 1, B, seed)
    bm = boot_means(x, w, cl, counts, groups=group, n_groups=len(labels))
    out = []
    for k, lab in enumerate(labels):
        m = group == k
        if not m.any():
            continue
        pt = np.array([wmean(x[m, j], w[m]) for j in range(x.shape[1])])
        row = dict(zip(["mixed", "raw", "top1", "entropy"], summarize(bm[:, k, :], pt)))
        row.update({"group": lab, "rows": int(m.sum()), "weight": float(w[m].sum())})
        out.append(row)
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m tnh.eval.breakdown", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--max-ply", type=int, default=60)
    add_common(p)
    a = p.parse_args(argv)
    full = read_sample(a.test)
    fit = window_rows(read_sample(a.fit))
    check_splits(full, fit)
    sf, _ = move_scores(fit, a.model)
    eps = fit_eps(sf.lp, sf.n, fit.get_column("weight").to_numpy())

    def arrays(s):
        sc, _ = move_scores(s, a.model)
        return np.column_stack([-mix_logp(sc.lp, sc.n, eps), -sc.lp, sc.top1, sc.entropy])

    # by ply, window rows
    win = window_rows(full)
    xw = arrays(win)
    ply_b = (win.get_column("ply").to_numpy() // 10)
    ply_labels = [f"{10 * k}-{10 * k + 9}" for k in range(ply_b.max() + 1)]
    by_ply = table(win, xw, ply_b, ply_labels, a.boot, a.seed)

    # by clock, all out-of-book rows with a clock
    ext = full.filter(pl.col("out_of_book") & (pl.col("ply") < a.max_ply) & pl.col("clock_mover_s").is_not_null())
    xe = arrays(ext)
    mins = ext.get_column("clock_mover_s").to_numpy() / 60
    cb = np.searchsorted(CLOCK_EDGES_MIN, mins, side="right") - 1
    by_clock = table(ext, xe, cb, CLOCK_LABELS, a.boot, a.seed)
    pb = ext.get_column("ply").to_numpy() // 20
    pl_labels = [f"{20 * k}-{20 * k + 19}" for k in range(pb.max() + 1)]
    cross = []
    for k, lab in enumerate(pl_labels):
        m = pb == k
        if not m.any():
            continue
        sub = ext.filter(pl.Series(m))
        for r in table(sub, xe[m], cb[m], CLOCK_LABELS, a.boot, a.seed):
            r["ply"] = lab
            cross.append(r)
    for r in by_clock:
        print(f"clock {r['group']:>6} min: rows={r['rows']:>7} mixed={r['mixed']['est']:.4f} "
              f"[{r['mixed']['lo']:.4f}, {r['mixed']['hi']:.4f}] entropy={r['entropy']['est']:.3f}")
    n_out = ext.filter(~pl.col("in_window")).height
    write_result(a.results, "breakdown", {
        "what": "mixed log loss by ply (window rows) and by clock (out-of-book rows with a clock, weighted)",
        "model": label(a.model), "eps": eps, "boot": a.boot, "max_ply": a.max_ply,
        "rows": {"window": win.height, "clock_rows": ext.height, "clock_rows_outside_window": n_out},
        "by_ply": by_ply, "by_clock": by_clock, "ply_x_clock": cross,
        "provenance": provenance({"test": a.test, "fit": a.fit}, [a.model])})


if __name__ == "__main__":
    main()
