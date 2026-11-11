"""The Elo sweep: which Elo Maia-3 should be given for a player of a given FIDE rating.

    uv run python -m tnh.eval.sweep --variants DIR [DIR ...] [--test S] [--fit S] [--results R]

``--variants`` are Maia-3 feature directories named ``offset+X`` (both ratings
shifted by X) and, optionally, ``offset+X-opp+Y`` (mover and opponent shifted
separately). Fixed-Elo variants are ignored. Every loss here is the mixed log
loss on the window rows, each variant with its own ``eps`` fitted on the
fitting sample.

What it computes:

1. Test log loss per offset, pooled and per 100-point bucket of the mover's rating.
2. The best offset per bucket on the test sample: the minimum of a parabola
   through the best grid point and its two neighbours, with a bootstrap
   interval over tournaments (the whole argmin is redone in every replicate).
   At an edge of the grid the edge is reported and flagged.
3. The map: the same per-bucket best offsets on the *fitting* sample, turned into
   the Elo Maia-3 should get (bucket midpoint + offset), and a weighted straight
   line ``maia_elo = a + b * fide`` through them (weights: square root of the rows
   per bucket; buckets whose minimum is at the edge of the grid are left out),
   bootstrapped over the fitting sample's tournaments.
4. Leakage-free comparisons on the test sample, each against the best single offset
   chosen on the fitting sample: per-bucket offsets chosen on the fitting sample,
   and the linear map, each row taking the grid offset nearest to what the map asks
   for. Differences are paired (the same bootstrap replicates).
5. Where the sweep has mover and opponent offsets varied separately: the pooled
   test log loss over the grid, and the best cell per bucket.

Writes ``<results>/sweep.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from tnh.eval.common import (BUCKET, add_common, boot_counts, boot_means, check_splits, cluster_codes, est, fit_eps,
                             label, mix_logp, move_scores, parse_variant, provenance, read_sample, variant_name,
                             window_rows, wmean, write_result)


def vertex(x: np.ndarray, y: np.ndarray) -> tuple[float, bool]:
    """Argmin of y over the grid x, refined by a parabola through the minimum and its neighbours.
    Returns (offset, at_edge)."""
    if np.all(np.isnan(y)):
        return float("nan"), False
    k = int(np.nanargmin(y))
    if k == 0 or k == len(x) - 1:
        return float(x[k]), True
    xs, ys = x[k - 1:k + 2], y[k - 1:k + 2]
    c = np.polyfit(xs, ys, 2)
    if c[0] <= 0:
        return float(x[k]), False
    v = -c[1] / (2 * c[0])
    return float(np.clip(v, xs[0], xs[2])), False


def losses(sample: pl.DataFrame, fit: pl.DataFrame, dirs: list[Path]) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Mixed log loss per row and variant on the sample and the fitting sample; eps per variant."""
    Lt, Lf, eps = [], [], []
    wf = fit.get_column("weight").to_numpy()
    for d in dirs:
        sf, _ = move_scores(fit, d)
        e = fit_eps(sf.lp, sf.n, wf)
        st, _ = move_scores(sample, d)
        Lt.append(-mix_logp(st.lp, st.n, e))
        Lf.append(-mix_logp(sf.lp, sf.n, e))
        eps.append(e)
        print(f"  {label(d)}: eps={e:.5f} test={wmean(Lt[-1], sample.get_column('weight').to_numpy()):.4f}", flush=True)
    return np.column_stack(Lt), np.column_stack(Lf), eps


def bucket_codes(s: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    b = s.get_column("bucket").to_numpy()
    ub = np.unique(b)
    return ub, np.searchsorted(ub, b)


def best_per_bucket(L: np.ndarray, w: np.ndarray, cl: np.ndarray, counts: np.ndarray, g: np.ndarray, nb: int,
                    offs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Point and bootstrap best offsets per bucket. Returns (point, edge, boot B x nb, mean loss nb x V)."""
    means = np.array([[wmean(L[g == k, v], w[g == k]) for v in range(L.shape[1])] for k in range(nb)])
    pt = np.array([vertex(offs, means[k]) for k in range(nb)])
    bm = boot_means(L, w, cl, counts, groups=g, n_groups=nb)   # B x nb x V
    boot = np.array([[vertex(offs, bm[r, k])[0] for k in range(nb)] for r in range(bm.shape[0])])
    return pt[:, 0], pt[:, 1].astype(bool), boot, means


def fit_map(mid: np.ndarray, best: np.ndarray, rows: np.ndarray, edge: np.ndarray | None = None) -> tuple[float, float]:
    """Weighted line through the per-bucket best Elo. Buckets whose minimum is at the grid's edge are
    censored (the best offset is beyond the grid) and left out."""
    ok = ~np.isnan(best) if edge is None else ~np.isnan(best) & ~edge
    if ok.sum() < 2:
        return float("nan"), float("nan")
    b, a = np.polyfit(mid[ok], mid[ok] + best[ok], 1, w=np.sqrt(rows[ok]))
    return float(a), float(b)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m tnh.eval.sweep", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", type=Path, nargs="+", required=True)
    add_common(p)
    a = p.parse_args(argv)
    test, fit = window_rows(read_sample(a.test)), window_rows(read_sample(a.fit))
    check_splits(test, fit)

    parsed = []
    for d in a.variants:
        v = parse_variant(variant_name(d))
        if v["kind"] != "offset":
            print(f"ignoring {d}: not an offset variant")
            continue
        parsed.append((d, v["mover"], v["opp"]))
    diag = sorted([x for x in parsed if x[1] == x[2]], key=lambda x: x[1])
    grid = [x for x in parsed if x[1] != x[2]]
    if len(diag) < 3:
        raise SystemExit("need at least three both-sides offsets")
    dirs = [x[0] for x in diag]
    offs = np.array([x[1] for x in diag])
    print(f"both-sides offsets: {offs.tolist()}")
    Lt, Lf, eps = losses(test, fit, dirs)

    wt, wf = test.get_column("weight").to_numpy(), fit.get_column("weight").to_numpy()
    ct, cf = cluster_codes(test), cluster_codes(fit)
    Kt, Kf = ct.max() + 1, cf.max() + 1
    counts_t, counts_f = boot_counts(Kt, a.boot, a.seed), boot_counts(Kf, a.boot, a.seed + 1)
    ub_t, gt = bucket_codes(test)
    ub_f, gf = bucket_codes(fit)

    # 1. loss per offset
    pooled_boot = boot_means(Lt, wt, ct, counts_t)
    per_offset = [{"offset": float(o), "eps": eps[v], "dir": str(dirs[v]), "label": label(dirs[v]),
                   "test": est(wmean(Lt[:, v], wt), pooled_boot[:, v]),
                   "fit": wmean(Lf[:, v], wf)} for v, o in enumerate(offs)]
    bb = boot_means(Lt, wt, ct, counts_t, groups=gt, n_groups=len(ub_t))
    per_bucket = []
    for k, b in enumerate(ub_t):
        m = gt == k
        per_bucket.append({"bucket": int(b), "rows": int(m.sum()),
                           "loss": [est(wmean(Lt[m, v], wt[m]), bb[:, k, v]) for v in range(len(offs))]})

    # 2. best offset per bucket, test
    pt, edge, boot, _ = best_per_bucket(Lt, wt, ct, counts_t, gt, len(ub_t), offs)
    lo, hi = np.nanquantile(boot, 0.025, axis=0), np.nanquantile(boot, 0.975, axis=0)
    best_test = [{"bucket": int(b), "best_offset": float(pt[k]), "lo": float(lo[k]), "hi": float(hi[k]),
                  "at_edge": bool(edge[k]), "rows": int((gt == k).sum())} for k, b in enumerate(ub_t)]

    # 3. the map, fitted on the fitting sample
    pf, edge_f, boot_f, _ = best_per_bucket(Lf, wf, cf, counts_f, gf, len(ub_f), offs)
    mid = ub_f + BUCKET / 2
    rows_f = np.array([(gf == k).sum() for k in range(len(ub_f))], dtype=float)
    a0, b0 = fit_map(mid, pf, rows_f, edge_f)
    lo_off, hi_off = offs.min(), offs.max()
    ab_boot = np.array([fit_map(mid, boot_f[r], rows_f, (boot_f[r] <= lo_off) | (boot_f[r] >= hi_off))
                        for r in range(boot_f.shape[0])])
    map_ = {"a": a0, "b": b0, "a_ci": np.nanquantile(ab_boot[:, 0], [0.025, 0.975]).tolist(),
            "b_ci": np.nanquantile(ab_boot[:, 1], [0.025, 0.975]).tolist(),
            "fit_points": [{"bucket": int(b), "best_offset": float(pf[k]), "at_edge": bool(edge_f[k]),
                            "rows": int(rows_f[k])} for k, b in enumerate(ub_f)],
            "formula": "maia_elo = a + b * fide_rating (mover and opponent alike)"}

    # 4. leakage-free comparisons on test
    single = int(np.argmin([wmean(Lf[:, v], wf) for v in range(len(offs))]))
    loss_single = Lt[:, single]
    fit_best_grid = {int(b): int(np.argmin([wmean(Lf[gf == k, v], wf[gf == k]) for v in range(len(offs))]))
                     for k, b in enumerate(ub_f)}
    tb = test.get_column("bucket").to_numpy()
    v_bucket = np.array([fit_best_grid.get(int(b), single) for b in tb])
    loss_bucket = Lt[np.arange(len(Lt)), v_bucket]
    elo = test.get_column("mover_elo").to_numpy().astype(float)
    target = a0 + (b0 - 1) * elo
    v_map = np.abs(target[:, None] - offs[None, :]).argmin(axis=1)
    loss_map = Lt[np.arange(len(Lt)), v_map]
    X = np.column_stack([loss_single, loss_bucket, loss_map, loss_bucket - loss_single, loss_map - loss_single])
    bx = boot_means(X, wt, ct, counts_t)
    ptx = [wmean(X[:, j], wt) for j in range(X.shape[1])]
    compare = {
        "single_offset": float(offs[single]),
        "single": est(ptx[0], bx[:, 0]), "per_bucket": est(ptx[1], bx[:, 1]), "map": est(ptx[2], bx[:, 2]),
        "per_bucket_minus_single": est(ptx[3], bx[:, 3]), "map_minus_single": est(ptx[4], bx[:, 4]),
        "per_bucket_offsets": {str(k): float(offs[v]) for k, v in fit_best_grid.items()},
        "map_offsets_used": {str(float(o)): int((v_map == v).sum()) for v, o in enumerate(offs) if (v_map == v).any()},
    }
    print(f"single offset {offs[single]:+g} (chosen on fit): {ptx[0]:.4f}; per bucket: {ptx[1]:.4f}; "
          f"map: {ptx[2]:.4f}; map - single {ptx[4]:+.4f} [{compare['map_minus_single']['lo']:+.4f}, "
          f"{compare['map_minus_single']['hi']:+.4f}]")

    # 5. mover and opponent separately
    sep = None
    if grid:
        gdirs = [x[0] for x in grid]
        Lg, _, eps_g = losses(test, fit, gdirs)
        cells = [(x[1], x[2], Lt[:, v], eps[v]) for v, x in enumerate(diag)] + \
                [(x[1], x[2], Lg[:, j], eps_g[j]) for j, x in enumerate(grid)]
        M = np.column_stack([c[2] for c in cells])
        bm = boot_means(M, wt, ct, counts_t)
        bmb = boot_means(M, wt, ct, counts_t, groups=gt, n_groups=len(ub_t))
        sep = {"cells": [{"mover": c[0], "opp": c[1], "eps": c[3], "test": est(wmean(M[:, j], wt), bm[:, j])}
                         for j, c in enumerate(cells)],
               "best_per_bucket": []}
        for k, b in enumerate(ub_t):
            m = gt == k
            means = [wmean(M[m, j], wt[m]) for j in range(M.shape[1])]
            j = int(np.argmin(means))
            # how often each cell wins under the bootstrap
            wins = np.bincount(np.nanargmin(bmb[:, k, :], axis=1), minlength=M.shape[1]) / bmb.shape[0]
            sep["best_per_bucket"].append({"bucket": int(b), "mover": cells[j][0], "opp": cells[j][1],
                                           "loss": means[j], "boot_share": float(wins[j])})

    write_result(a.results, "sweep", {
        "what": "Maia-3 Elo sweep, mixed log loss on window rows",
        "rows": {"test": test.height, "fit": fit.height}, "boot": a.boot,
        "offsets": offs.tolist(), "per_offset": per_offset, "per_bucket": per_bucket,
        "best_offset_test": best_test, "map": map_, "compare": compare, "mover_opp": sep,
        "provenance": provenance({"test": a.test, "fit": a.fit}, a.variants)})


if __name__ == "__main__":
    main()
