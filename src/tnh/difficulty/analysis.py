"""The test: does a measure explain the real score beyond the baseline?

Unit of observation: a position in the set. Outcome ``y``: White's score in the
position's games (in the filter), ``n`` games.

- **Baseline**: weighted least squares of ``y`` on the engine's expected score
  for White (root search), its centipawns for White (clipped to +-4 pawns, mate
  = 4), the mean rating difference White - Black, the mean rating, and side to
  move. A flexible baseline is a high bar on purpose.
- **Weights**: ``1 / (sigma2 / n + tau2)``. ``sigma2`` is the per-game variance
  of the score, pooled; ``tau2`` the between-position variance the baseline
  leaves, by moments from its residuals. A position with 2000 games is not
  worth 20 with 100: its own noise is small but the model's error is not.
- **Measures** are signed for White: a loss for the side to move counts for
  White when Black is to move, against White when White is.
- **Held out**: families (the first ``family_ply`` plies of the path) are split
  into ``folds`` folds; every prediction is made by a fit that did not see the
  position's family. ``dmse`` is the drop in weighted squared error from adding
  the measure; ``share`` divides it by what the baseline leaves above the
  binomial noise.
- **Intervals**: 95 % percentile intervals of a bootstrap over families.
  Nested positions share games; positions in one family are resampled together,
  so the overlap is inside a cluster. Families can still share games through
  transpositions; `cross_family_share` says how many.
- **Effect size**: the real score minus the held-out baseline prediction, in
  the top and the bottom tenth of the measure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from tnh.difficulty.positions import family

BASE = ["engine_es_w", "engine_cp_w", "elo_diff", "elo_mean", "stm_white"]
MEASURES = {
    "el_stm": "expected loss of the side to move",
    "hard_stm": "narrow path: 1 - probability of a holding move, side to move",
    "el_next": "expected loss of the other side one ply later",
    "hard_next": "narrow path of the other side one ply later",
    "net2": "two-ply net: loss now minus the other side's loss next",
    "maia_value": "Maia value head, expected score for White",
}
EXTRA = {
    "maia_value_eq": "Maia value head at equal ratings (mapped mean of both sides), expected score for White",
}
REFERENCE = {
    "el_db": "expected loss with the games' own move shares (not a predictor)",
    "hard_db": "1 - share of the games' moves that hold (not a predictor)",
}


def white_frame(positions: pl.DataFrame, measures: pl.DataFrame, family_ply: int,
                extra: tuple[str, ...] = ()) -> tuple[pl.DataFrame, dict]:
    """Positions joined to their measures, every column from White's point of view.
    ``extra``: columns of EXTRA that must be finite too."""
    df = positions.join(measures, on="pos_hash", how="inner")
    info = {"positions": positions.height, "with_measures": df.height}
    sgn = pl.when(pl.col("stm_white")).then(-1.0).otherwise(1.0)   # a loss of the side to move, for White
    es_w = pl.when(pl.col("stm_white")).then(pl.col("engine_es_stm")).otherwise(1 - pl.col("engine_es_stm"))
    cp_w = pl.when(pl.col("stm_white")).then(pl.col("engine_cp_stm")).otherwise(-pl.col("engine_cp_stm"))
    df = df.with_columns(
        y=pl.col("score_white"),
        engine_es_w=es_w,
        engine_cp_w=(cp_w / 100).clip(-4, 4),
        cp_w_raw=cp_w,
        elo_diff=(pl.col("white_elo") - pl.col("black_elo")) / 100,
        elo_logistic=1 / (1 + 10 ** (-(pl.col("white_elo") - pl.col("black_elo")) / 400)),
        elo_diff_sq=((pl.col("white_elo") - pl.col("black_elo")) / 100) ** 2,
        elo_mean=((pl.col("white_elo") + pl.col("black_elo")) / 2 - 2100) / 100,
        mean_elo=(pl.col("white_elo") + pl.col("black_elo")) / 2,
        stm_white=pl.col("stm_white").cast(pl.Float64),
        el_stm=sgn * pl.col("el_stm"),
        hard_stm=sgn * (1 - pl.col("hold_stm")),
        el_next=-sgn * pl.col("el_next"),
        hard_next=-sgn * (1 - pl.col("hold_next")),
        net2=sgn * pl.col("net2"),
        maia_value=pl.when(pl.col("stm_white")).then(pl.col("maia_es_stm")).otherwise(1 - pl.col("maia_es_stm")),
        maia_value_eq=(pl.when(pl.col("stm_white")).then(pl.col("maia_es_stm_eq")).otherwise(1 - pl.col("maia_es_stm_eq"))
                       if "maia_es_stm_eq" in measures.columns else pl.lit(float("nan"))),
        el_db=sgn * pl.col("el_db"),
        hard_db=sgn * (1 - pl.col("hold_db")),
        log_total=pl.col("total_n").log10(),
        family=family(pl.col("moves_uci"), family_ply),
    )
    info["above_family_ply"] = int(df["family"].null_count())
    df = df.filter(pl.col("family").is_not_null())
    cols = [*BASE, *MEASURES, *REFERENCE, *extra]
    finite = pl.all_horizontal([pl.col(c).is_not_null() & pl.col(c).is_finite() for c in cols])
    info["dropped_not_finite"] = int(df.height - df.filter(finite).height)
    df = df.filter(finite)
    info["analysed"] = df.height
    return df, info


# --- fitting ------------------------------------------------------------------------


def design(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return np.column_stack([np.ones(df.height)] + [df[c].to_numpy().astype(np.float64) for c in cols])


def wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    A = X.T @ (w[:, None] * X)
    return np.linalg.solve(A, X.T @ (w * y))


def per_game_var(df: pl.DataFrame) -> float:
    """Pooled variance of one game's score, from the wins, draws and losses."""
    n = df["n"].to_numpy().astype(np.float64)
    s2 = (df["white_wins"].to_numpy() + 0.25 * df["draws"].to_numpy()) / n
    s = df["y"].to_numpy()
    return float(np.sum(n * (s2 - s * s)) / np.sum(n))


def precision_weights(y: np.ndarray, n: np.ndarray, X: np.ndarray, sigma2: float, iters: int = 5) -> tuple[np.ndarray, float]:
    """Weights 1 / (sigma2/n + tau2), tau2 by moments from the baseline's residuals."""
    tau2 = 0.0
    for _ in range(iters):
        w = 1 / (sigma2 / n + tau2)
        r = y - X @ wls(X, y, w)
        tau2 = max(0.0, float(np.sum(w * (r * r - sigma2 / n)) / np.sum(w)))
    return 1 / (sigma2 / n + tau2), tau2


def fold_of(clusters: np.ndarray, folds: int, seed: int) -> np.ndarray:
    k = clusters.max() + 1
    perm = np.random.default_rng(seed).permutation(k)
    return (perm % folds)[clusters]


def oof_predict(X: np.ndarray, y: np.ndarray, w: np.ndarray, fold: np.ndarray) -> np.ndarray:
    out = np.empty_like(y)
    for f in np.unique(fold):
        tr = fold != f
        out[~tr] = X[~tr] @ wls(X[tr], y[tr], w[tr])
    return out


def boot_counts(k: int, B: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(k, np.full(k, 1.0 / k), size=B).astype(np.float64)


def ci(point: float, boot: np.ndarray) -> dict:
    b = boot[np.isfinite(boot)]
    return {"est": float(point), "lo": float(np.quantile(b, 0.025)), "hi": float(np.quantile(b, 0.975)),
            "p_le_0": float(np.mean(b <= 0))}


def boot_ratio(num: np.ndarray, den: np.ndarray, clusters: np.ndarray, counts: np.ndarray, k: int) -> np.ndarray:
    """Replicates of sum(num)/sum(den) with clusters drawn by ``counts``."""
    sn = np.bincount(clusters, weights=num, minlength=k)
    sd = np.bincount(clusters, weights=den, minlength=k)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (counts @ sn) / (counts @ sd)


def boot_wls(X: np.ndarray, y: np.ndarray, w: np.ndarray, clusters: np.ndarray, counts: np.ndarray, k: int) -> np.ndarray:
    """Replicates of the WLS coefficients, B x p."""
    p = X.shape[1]
    A = np.zeros((k, p, p))
    b = np.zeros((k, p))
    np.add.at(A, clusters, w[:, None, None] * X[:, :, None] * X[:, None, :])
    np.add.at(b, clusters, (w * y)[:, None] * X)
    AA = np.einsum("bk,kij->bij", counts, A)
    bb = counts @ b
    out = np.full((len(counts), p), np.nan)
    for i in range(len(counts)):
        try:
            out[i] = np.linalg.solve(AA[i], bb[i])
        except np.linalg.LinAlgError:
            pass
    return out


@dataclass
class Setup:
    folds: int = 10
    boot: int = 1000
    seed: int = 0
    tenth: float = 0.1
    base: list[str] = field(default_factory=lambda: list(BASE))


def tenths(m: np.ndarray, r: np.ndarray, w: np.ndarray, clusters: np.ndarray, counts: np.ndarray, k: int, q: float) -> dict:
    lo, hi = np.quantile(m, [q, 1 - q])
    top, bot = m >= hi, m <= lo
    t = boot_ratio(w * r * top, w * top, clusters, counts, k)
    b = boot_ratio(w * r * bot, w * bot, clusters, counts, k)
    tp = float(np.sum(w * r * top) / np.sum(w * top))
    bp = float(np.sum(w * r * bot) / np.sum(w * bot))
    return {"top": ci(tp, t), "bottom": ci(bp, b), "diff": ci(tp - bp, t - b),
            "cut_top": float(hi), "cut_bottom": float(lo), "n_top": int(top.sum()), "n_bottom": int(bot.sum())}


def slope(m: np.ndarray, r: np.ndarray, w: np.ndarray, clusters: np.ndarray, counts: np.ndarray, k: int) -> dict:
    X = np.column_stack([np.ones(len(m)), m])
    return ci(float(wls(X, r, w)[1]), boot_wls(X, r, w, clusters, counts, k)[:, 1])


def run(df: pl.DataFrame, measures: list[str], s: Setup, splits: dict[str, np.ndarray] | None = None,
        reference: list[str] = ()) -> tuple[dict, pl.DataFrame]:
    """Every number of the test on one frame. ``df`` needs ``y``, ``n``, ``white_wins``,
    ``draws``, ``family``, the base columns and the measure columns.

    Returns the results and the frame with ``w``, the held-out baseline
    prediction ``pred_base`` and per measure ``pred_<m>`` (full-data fit).
    """
    y = df["y"].to_numpy().astype(np.float64)
    n = df["n"].to_numpy().astype(np.float64)
    clusters = (df["family"].rank("dense").to_numpy() - 1).astype(np.int64)
    k = int(clusters.max() + 1)
    counts = boot_counts(k, s.boot, s.seed)
    fold = fold_of(clusters, s.folds, s.seed)
    Xb = design(df, s.base)
    sigma2 = per_game_var(df)
    w, tau2 = precision_weights(y, n, Xb, sigma2)
    noise = float(np.sum(w * sigma2 / n) / np.sum(w))
    pb = oof_predict(Xb, y, w, fold)
    eb = w * (y - pb) ** 2
    mse_b = float(eb.sum() / w.sum())
    rb = y - pb
    res = {"positions": df.height, "families": k, "games_summed": int(n.sum()), "sigma2": sigma2, "tau2": tau2,
           "noise_mse": noise, "mse_base": mse_b, "excess_base": mse_b - noise,
           "base_coef": dict(zip(["const", *s.base], wls(Xb, y, w).tolist())),
           "base_coef_ci": {}, "measures": {}, "folds": s.folds, "boot": s.boot}
    bb = boot_wls(Xb, y, w, clusters, counts, k)
    for j, c in enumerate(["const", *s.base]):
        res["base_coef_ci"][c] = ci(res["base_coef"][c], bb[:, j])
    out = df.with_columns(w=pl.Series(w), pred_base=pl.Series(pb))
    excess_boot = boot_ratio(eb, w, clusters, counts, k) - noise
    for m in [*measures, *reference]:
        X = np.column_stack([Xb, df[m].to_numpy().astype(np.float64)])
        pm = oof_predict(X, y, w, fold)
        d = eb - w * (y - pm) ** 2
        dm = boot_ratio(d, w, clusters, counts, k)
        beta_full = wls(X, y, w)
        bw = boot_wls(X, y, w, clusters, counts, k)
        mv = df[m].to_numpy().astype(np.float64)
        res["measures"][m] = {
            "reference": m in reference,
            "beta": ci(float(beta_full[-1]), bw[:, -1]),
            "sd": float(np.sqrt(np.cov(mv, aweights=w))),
            "beta_per_sd": ci(float(beta_full[-1]) * float(np.sqrt(np.cov(mv, aweights=w))),
                              bw[:, -1] * float(np.sqrt(np.cov(mv, aweights=w)))),
            "dmse": ci(float(d.sum() / w.sum()), dm),
            "share": ci(float(d.sum() / w.sum()) / (mse_b - noise), dm / excess_boot),
            "tenths": tenths(mv, rb, w, clusters, counts, k, s.tenth),
        }
        out = out.with_columns(pl.Series(f"pred_{m}", X @ beta_full), pl.Series(f"oof_{m}", pm))
    if measures:
        X = np.column_stack([Xb] + [df[m].to_numpy().astype(np.float64) for m in measures])
        pm = oof_predict(X, y, w, fold)
        d = eb - w * (y - pm) ** 2
        dm = boot_ratio(d, w, clusters, counts, k)
        res["all_measures"] = {"dmse": ci(float(d.sum() / w.sum()), dm),
                               "share": ci(float(d.sum() / w.sum()) / (mse_b - noise), dm / excess_boot),
                               "coef": dict(zip(measures, wls(X, y, w)[-len(measures):].tolist()))}
        out = out.with_columns(pl.Series("oof_all", pm))
    res["splits"] = {}
    for name, groups in (splits or {}).items():
        res["splits"][name] = {}
        for g in [g for g in np.unique(groups) if g is not None]:
            sel = groups == g
            if sel.sum() < 30:
                continue
            cl = clusters[sel]
            entry = {"positions": int(sel.sum()), "families": int(len(np.unique(cl))), "measures": {}}
            for m in [*measures, *reference]:
                mv = df[m].to_numpy().astype(np.float64)[sel]
                entry["measures"][m] = {"slope": slope(mv, rb[sel], w[sel], cl, counts, k),
                                        "tenths": tenths(mv, rb[sel], w[sel], cl, counts, k, s.tenth)}
            res["splits"][name][str(g)] = entry
    return res, out
