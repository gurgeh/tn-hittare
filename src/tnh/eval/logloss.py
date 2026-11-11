"""Log loss of the played move, raw and mixed with the uniform distribution.

    uv run python -m tnh.eval.logloss --model DIR [DIR ...] [--test S] [--fit S] [--results R]

For each move-model feature directory: ``eps`` is fitted on the window rows of
the fitting sample, then on the window rows of the test sample the raw log
loss, the mixed log loss ``-log((1 - eps) p + eps / n)``, top-1 and the model's
mean entropy, pooled and per 100-point bucket of the mover's rating. Pooled
numbers are weighted by the sample weights; intervals are a bootstrap over
tournaments. Writes ``<results>/logloss.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from tnh.eval.common import (add_common, boot_counts, boot_means, check_splits, cluster_codes, est, fit_eps,
                             label, mix_logp, move_scores, provenance, read_sample, summarize, window_rows, wmean,
                             write_result)


def scored(sample: pl.DataFrame, d: Path, eps: float | None = None):
    """Per-row arrays for one model on one sample (already filtered)."""
    sc, _ = move_scores(sample, d)
    out = {"lp": sc.lp, "n": sc.n, "top1": sc.top1, "entropy": sc.entropy}
    if eps is not None:
        out["mix"] = mix_logp(sc.lp, sc.n, eps)
    return out


def eps_from_fit(fit: pl.DataFrame, d: Path) -> float:
    sc, _ = move_scores(fit, d)
    return fit_eps(sc.lp, sc.n, fit.get_column("weight").to_numpy())


def evaluate(test: pl.DataFrame, fit: pl.DataFrame, d: Path, B: int, seed: int) -> dict:
    eps = eps_from_fit(fit, d)
    s = scored(test, d, eps)
    w = test.get_column("weight").to_numpy()
    cl = cluster_codes(test)
    counts = boot_counts(cl.max() + 1, B, seed)
    x = np.column_stack([-s["lp"], -s["mix"], s["top1"], s["entropy"]])
    point = np.array([wmean(x[:, j], w) for j in range(x.shape[1])])
    boot = boot_means(x, w, cl, counts)
    pooled = dict(zip(["raw", "mixed", "top1", "entropy"], summarize(boot, point)))
    buckets = test.get_column("bucket").to_numpy()
    ub = np.unique(buckets)
    g = np.searchsorted(ub, buckets)
    bb = boot_means(x, w, cl, counts, groups=g, n_groups=len(ub))
    by = []
    for k, b in enumerate(ub):
        m = g == k
        pt = np.array([wmean(x[m, j], w[m]) for j in range(x.shape[1])])
        row = dict(zip(["raw", "mixed", "top1", "entropy"], summarize(bb[:, k, :], pt)))
        row.update({"bucket": int(b), "rows": int(m.sum()), "games": int(test.filter(pl.Series(m))
                                                                           .get_column("game_id").n_unique())})
        by.append(row)
    return {"label": label(d), "dir": str(d), "eps": eps, "pooled": pooled, "by_bucket": by}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m tnh.eval.logloss", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, nargs="+", required=True, help="move-model feature directories")
    add_common(p)
    a = p.parse_args(argv)
    test, fit = window_rows(read_sample(a.test)), window_rows(read_sample(a.fit))
    check_splits(test, fit)
    models = []
    for d in a.model:
        r = evaluate(test, fit, d, a.boot, a.seed)
        pl_ = r["pooled"]
        print(f"{r['label']}: eps={r['eps']:.5f} raw={pl_['raw']['est']:.4f} mixed={pl_['mixed']['est']:.4f} "
              f"[{pl_['mixed']['lo']:.4f}, {pl_['mixed']['hi']:.4f}] top1={pl_['top1']['est']:.4f}")
        models.append(r)
    write_result(a.results, "logloss", {
        "what": "log loss of the played move, window rows; eps fitted on the fitting sample",
        "rows": {"test": test.height, "fit": fit.height},
        "tournaments": {"test": test.get_column("tournament_id").n_unique(),
                        "fit": fit.get_column("tournament_id").n_unique()},
        "boot": a.boot, "provenance": provenance({"test": a.test, "fit": a.fit}, a.model), "models": models})


if __name__ == "__main__":
    main()
