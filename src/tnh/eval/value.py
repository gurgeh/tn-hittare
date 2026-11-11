"""The value head: predicted win, draw and loss against the game's result.

    uv run python -m tnh.eval.value --model DIR [DIR ...] [--test S] [--fit S] [--results R]

Each ``--model`` is a feature directory with a ``wdl`` column: Maia-3's value
head, or an engine's WDL for the position (Stockfish's ``wdl``, the position
along the best line). All are scored on the same rows.

**One position per game.** A game contributes sixty rows with one result; scored
row by row it would count as sixty observations. From each game in the sample,
the window row with the smallest key ``hash(seed, game_id)`` among its window
rows is taken (``--pick random``, default), which spreads the chosen positions
over the whole window; ``--pick first`` takes the game's earliest window row in
the sample instead, the position nearest to the end of preparation. The side to
move is whoever moves in that row, so the result is from the mover's side.

**Recalibration**, fitted on the fitting sample (one row per game), four
parameters: ``logit_c = t * log p_c + b_c + [c = draw] * k * (mean rating - 2200) / 100``
with ``b_loss = 0``. ``t`` sharpens or flattens, ``b_win`` and ``b_draw`` move mass
between the outcomes, and ``k`` lets the draw rate rise with the players'
strength, which a model trained on blitz cannot know about classical play.

Reported on the test sample: three-class log loss, raw and recalibrated,
with intervals over tournaments; calibration tables by 100-point bucket of the
mover's rating: mean predicted draw probability against the real draw rate, and
mean predicted expected score (win + draw / 2) against the real score.

Writes ``<results>/value.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize

from tnh.eval.common import (add_common, attach, boot_counts, boot_means, check_splits, cluster_codes, est, label,
                             provenance, read_sample, window_rows, wmean, write_result)
from tnh.eval.sample import splitmix64

CLASS = {1.0: 0, 0.5: 1, 0.0: 2}


def one_per_game(s: pl.DataFrame, how: str, seed: int) -> pl.DataFrame:
    if how == "first":
        return s.sort("game_id", "ply").group_by("game_id", maintain_order=True).first()
    g = np.array([int(x[:16], 16) for x in s.get_column("game_id").to_list()], dtype=np.uint64)
    key = splitmix64(g ^ splitmix64(np.full(len(g), seed, dtype=np.uint64)) ^ splitmix64(
        s.get_column("ply").to_numpy().astype(np.uint64)))
    return (s.with_columns(pl.Series("_k", key)).sort("game_id", "_k")
            .group_by("game_id", maintain_order=True).first().drop("_k").sort("game_id"))


def wdl_of(s: pl.DataFrame, d: Path) -> np.ndarray:
    j = attach(s, d, ["wdl"])
    n_null = j.get_column("wdl").null_count()
    if n_null:
        raise SystemExit(f"{d}: {n_null} rows without wdl")
    w = np.stack(j.get_column("wdl").to_numpy()).astype(np.float64)
    tot = w.sum(1)
    if np.any(np.abs(tot - 1) > 0.02):
        raise SystemExit(f"{d}: wdl rows do not sum to 1 (range {tot.min():.3f}-{tot.max():.3f})")
    return w / tot[:, None]


def recal_logits(theta: np.ndarray, logp: np.ndarray, r: np.ndarray) -> np.ndarray:
    t, bw, bd, k = theta
    z = t * logp
    z[:, 0] += bw
    z[:, 1] += bd + k * r
    return z - np.logaddexp.reduce(z, axis=1, keepdims=True)


def fit_recal(wdl: np.ndarray, y: np.ndarray, r: np.ndarray, w: np.ndarray) -> np.ndarray:
    logp = np.log(np.clip(wdl, 1e-300, None))   # log of an exact zero would be -inf; mass is never moved
    rows = np.arange(len(y))

    def nll(th):
        return -np.sum(w * recal_logits(th, logp.copy(), r)[rows, y]) / w.sum()

    res = minimize(nll, np.array([1.0, 0.0, 0.0, 0.0]), method="BFGS")
    if not res.success:
        print(f"  recalibration: {res.message}")
    return res.x


def nll_rows(wdl: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = wdl[np.arange(len(y)), y]
    with np.errstate(divide="ignore"):
        return -np.log(p)


def calib_table(s: pl.DataFrame, probs: dict[str, np.ndarray], y: np.ndarray) -> list[dict]:
    b = s.get_column("bucket").to_numpy()
    w = s.get_column("weight").to_numpy()
    score = np.array([1.0, 0.5, 0.0])[y]
    out = []
    for bk in np.unique(b):
        m = b == bk
        row = {"bucket": int(bk), "games": int(m.sum()), "draw_rate": wmean((y[m] == 1).astype(float), w[m]),
               "score": wmean(score[m], w[m])}
        for k, p in probs.items():
            row[f"{k}_p_draw"] = wmean(p[m, 1], w[m])
            row[f"{k}_exp_score"] = wmean(p[m, 0] + 0.5 * p[m, 1], w[m])
        out.append(row)
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m tnh.eval.value", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, nargs="+", required=True)
    p.add_argument("--pick", choices=["random", "first"], default="random")
    add_common(p)
    a = p.parse_args(argv)
    test = one_per_game(window_rows(read_sample(a.test)), a.pick, a.seed)
    fit = one_per_game(window_rows(read_sample(a.fit)), a.pick, a.seed)
    check_splits(test, fit)
    # every game counts once: equal weights within a rating bucket would be the row weights;
    # one row per game is a different design, so the games are weighted equally.
    test = test.with_columns(pl.lit(1.0).alias("weight"))
    fit = fit.with_columns(pl.lit(1.0).alias("weight"))
    yt = np.array([CLASS[x] for x in test.get_column("result_mover").to_list()])
    yf = np.array([CLASS[x] for x in fit.get_column("result_mover").to_list()])
    rt = ((test.get_column("mover_elo") + test.get_column("opp_elo")).to_numpy() / 2 - 2200) / 100
    rf = ((fit.get_column("mover_elo") + fit.get_column("opp_elo")).to_numpy() / 2 - 2200) / 100
    wt = test.get_column("weight").to_numpy()
    ct = cluster_codes(test)
    counts = boot_counts(ct.max() + 1, a.boot, a.seed)
    models, probs = [], {}
    for d in a.model:
        Pt, Pf = wdl_of(test, d), wdl_of(fit, d)
        theta = fit_recal(Pf, yf, rf, fit.get_column("weight").to_numpy())
        Qt = np.exp(recal_logits(theta, np.log(np.clip(Pt, 1e-300, None)), rt))
        raw, rec = nll_rows(Pt, yt), nll_rows(Qt, yt)
        X = np.column_stack([raw, rec, rec - raw])
        bm = boot_means(X, wt, ct, counts)
        name = label(d)
        probs[f"{name} raw"], probs[f"{name} recal"] = Pt, Qt
        m = {"label": name, "dir": str(d),
             "recal": dict(zip(["t", "b_win", "b_draw", "k_draw_per_100"], theta.tolist())),
             "raw": est(wmean(raw, wt), bm[:, 0]), "recalibrated": est(wmean(rec, wt), bm[:, 1]),
             "recal_minus_raw": est(wmean(rec - raw, wt), bm[:, 2]),
             "mean_p_draw_raw": wmean(Pt[:, 1], wt), "mean_p_draw_recal": wmean(Qt[:, 1], wt)}
        print(f"{name}: raw {m['raw']['est']:.4f} recal {m['recalibrated']['est']:.4f}; P(draw) raw "
              f"{m['mean_p_draw_raw']:.3f} recal {m['mean_p_draw_recal']:.3f}, real {np.mean(yt == 1):.3f}; "
              f"theta {np.round(theta, 3).tolist()}")
        models.append(m)
    # a reference with no model: the fitting sample's outcome frequencies
    freq = np.bincount(yf, minlength=3) / len(yf)
    base = nll_rows(np.tile(freq, (len(yt), 1)), yt)
    write_result(a.results, "value", {
        "what": "value-head log loss on the result, one window row per game; recalibration fitted on the fitting sample",
        "pick": a.pick, "games": {"test": test.height, "fit": fit.height},
        "draw_rate": {"test": float(np.mean(yt == 1)), "fit": float(np.mean(yf == 1))},
        "base_rate": {"freq": freq.tolist(), "test_logloss": float(np.mean(base))},
        "ply_quartiles_test": np.percentile(test.get_column("ply").to_numpy(), [25, 50, 75]).tolist(),
        "models": models, "calibration": calib_table(test, probs, yt), "boot": a.boot,
        "provenance": provenance({"test": a.test, "fit": a.fit}, a.model)})


if __name__ == "__main__":
    main()
