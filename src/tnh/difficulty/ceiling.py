"""The ceiling: how much do positions differ in practical outcome at all?

No model calls. For strata of positions by their number of games in the filter:

- ``tau``: the between-position SD of White's score left after a baseline of
  the rating terms only (rating difference, mean rating, side to move), by
  moments from the weighted residuals (``analysis.precision_weights``), with a
  95 % interval from a bootstrap over opening families (one-step: weights held
  at the point estimate, coefficients refitted per replicate).
- On a frame with engine values, the same with the engine baseline beside it.
- Empirical Bayes per stratum, normal prior N(0, tau^2): a position's edge over
  the baseline ``r = y - fit`` with sampling variance ``v = sigma2 / n`` shrinks
  to ``r * tau^2 / (tau^2 + v)``, posterior SD ``sqrt(tau^2 v / (tau^2 + v))``.
  Shares of positions with a true edge above 1, 2, 3 points per 100 games
  under that prior, and a tail check: the share of ``|r| / sqrt(tau^2 + v)``
  above 2 and 3 against the normal 4.55 % and 0.27 %. A heavier observed tail
  means the normal prior understates the extremes.
- The list: positions whose 95 % posterior interval excludes zero; a position
  is dropped when a deeper position on its path (its path is a prefix of the
  other's) is also in that pool with the same sign, so a line appears once, at
  its deepest position that still clears the bar. The rest ranked by the size
  of the shrunken edge.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import norm

from tnh.difficulty import analysis as A
from tnh.difficulty.positions import family

RATING_BASE = ["elo_diff", "elo_mean", "stm_white"]
STRATA = [(1000, None), (200, 1000), (100, 200), (50, 100), (20, 50)]


def label(lo: int, hi: int | None) -> str:
    return f"n >= {lo}" if hi is None else f"{lo}-{hi - 1}"


def prepare(positions: pl.DataFrame, family_ply: int) -> tuple[pl.DataFrame, int]:
    df = positions.with_columns(
        y=pl.col("score_white"),
        elo_diff=(pl.col("white_elo") - pl.col("black_elo")) / 100,
        elo_mean=((pl.col("white_elo") + pl.col("black_elo")) / 2 - 2100) / 100,
        stm_white=pl.col("stm_white").cast(pl.Float64),
        family=family(pl.col("moves_uci"), family_ply),
    )
    above = int(df["family"].null_count())
    return df.filter(pl.col("family").is_not_null()), above


def tau_fit(df: pl.DataFrame, base: list[str], boot: int, seed: int) -> dict:
    """tau with its family-bootstrap interval, the residuals and the posterior quantities."""
    y = df["y"].to_numpy().astype(np.float64)
    n = df["n"].to_numpy().astype(np.float64)
    X = A.design(df, base)
    sigma2 = A.per_game_var(df)
    w, tau2 = A.precision_weights(y, n, X, sigma2)
    beta = A.wls(X, y, w)
    r = y - X @ beta
    clusters = (df["family"].rank("dense").to_numpy() - 1).astype(np.int64)
    k = int(clusters.max() + 1)
    counts = A.boot_counts(k, boot, seed)
    p = X.shape[1]
    Ak = np.zeros((k, p, p))
    bk = np.zeros((k, p))
    np.add.at(Ak, clusters, w[:, None, None] * X[:, :, None] * X[:, None, :])
    np.add.at(bk, clusters, (w * y)[:, None] * X)
    ck = np.bincount(clusters, weights=w * y * y, minlength=k)
    dk = np.bincount(clusters, weights=w * sigma2 / n, minlength=k)
    ek = np.bincount(clusters, weights=w, minlength=k)
    AA, bb = np.einsum("bk,kij->bij", counts, Ak), counts @ bk
    cc, dd, ee = counts @ ck, counts @ dk, counts @ ek
    t2 = np.empty(boot)
    for i in range(boot):
        be = np.linalg.solve(AA[i], bb[i])
        rss = cc[i] - 2 * be @ bb[i] + be @ AA[i] @ be
        t2[i] = max(0.0, (rss - dd[i]) / ee[i])
    tb = np.sqrt(t2)
    v = sigma2 / n
    shrink = tau2 / (tau2 + v)
    return {"tau": float(np.sqrt(tau2)), "tau_lo": float(np.quantile(tb, 0.025)), "tau_hi": float(np.quantile(tb, 0.975)),
            "sigma2": sigma2, "positions": df.height, "families": k, "median_n": float(np.median(n)),
            "mean_noise_sd": float(np.sqrt(np.mean(v))), "coef": dict(zip(["const", *base], beta.tolist())),
            "_r": r, "_v": v, "_shrunk": r * shrink, "_post_sd": np.sqrt(tau2 * v / (tau2 + v)), "_tau2": tau2}


def tail_summary(fit: dict) -> dict:
    tau = fit["tau"]
    z = fit["_r"] / np.sqrt(fit["_tau2"] + fit["_v"])
    s = fit["_shrunk"] * 100
    out = {"share_true_edge_above": {str(x): float(2 * norm.sf(x / 100 / tau)) if tau > 0 else 0.0 for x in (1, 2, 3)},
           "shrunk_quantiles": {str(q): float(np.quantile(s, q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)},
           "share_shrunk_above": {str(x): float(np.mean(np.abs(s) > x)) for x in (1, 2, 3)},
           "z_above_2": float(np.mean(np.abs(z) > 2)), "z_above_3": float(np.mean(np.abs(z) > 3)),
           "z_normal_2": 0.0455, "z_normal_3": 0.0027}
    return out


def strata_table(df: pl.DataFrame, boot: int, seed: int) -> tuple[dict, pl.DataFrame]:
    """Per stratum: tau and the tail; and every position with its shrunken edge."""
    out, frames = {}, []
    for lo, hi in STRATA:
        sel = df.filter((pl.col("n") >= lo) & ((pl.col("n") < hi) if hi else pl.lit(True)))
        if sel.height < 50:
            continue
        fit = tau_fit(sel, RATING_BASE, boot, seed)
        out[label(lo, hi)] = {**{k: v for k, v in fit.items() if not k.startswith("_")}, **tail_summary(fit)}
        frames.append(sel.with_columns(stratum=pl.lit(label(lo, hi)), resid=pl.Series(fit["_r"]),
                                       shrunk=pl.Series(fit["_shrunk"]), post_sd=pl.Series(fit["_post_sd"])))
    return out, pl.concat(frames)


def deepest_list(df: pl.DataFrame, top: int = 30) -> tuple[pl.DataFrame, dict]:
    """Positions whose 95 % posterior interval excludes zero, one per line (the deepest), largest edges first."""
    pool = df.filter((pl.col("shrunk").abs() - 1.96 * pl.col("post_sd")) > 0).with_columns(sign=pl.col("shrunk").sign())
    paths = pool["moves_uci"].to_list()
    signs = pool["sign"].to_list()
    by_sign = {s: np.array(sorted(p for p, sg in zip(paths, signs) if sg == s), dtype=object) for s in (-1.0, 1.0)}
    keep = []
    for p, sg in zip(paths, signs):
        lst = by_sign[sg]
        i = np.searchsorted(lst, p + " ")   # a descendant's path starts with p + " "
        keep.append(not (i < len(lst) and lst[i].startswith(p + " ")))
    pool = pool.filter(pl.Series(keep))
    return pool.sort(pl.col("shrunk").abs(), descending=True).head(top), {"pool": len(paths), "after_dedupe": pool.height}
