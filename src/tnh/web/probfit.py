"""The Prob mix fitted on held-out games: which `tnh.movemix` rule predicts the 2024-26 moves best.

    scripts/tnh-run web-probfit 3 uv run python -m tnh.web.probfit --data PAIRS.parquet --prior PRIOR_DIR \\
        [--prior-column p_mean] [--report OUT] [--write-rule]

`PAIRS.parquet` is a `tnh.web.shrinkfit collect` sample (before / from the split year), the
prior directory is `tnh.web.probprior`'s model-1 probabilities for the same positions.

A position counts when it has games in the test years. Its pre-split games give the counts
`c` (N their sum, as the page's N is); every legal move gets `p = w * c/N + (1 - w) * p_model`.
Scores, weighted by the position's test games:

- **log loss** of the test games' moves, `-sum t_m log p_m / sum t` (the proper score; it
  decides). The database alone gives an infinite log loss wherever a test move had no
  earlier game, so beside it stands `jeffreys`, the database with half a game on every legal
  move, as the database-only estimator with a finite log loss;
- the squared error of the test share, `sum_m (p_m - t_m/T)^2` per position.

Five folds by position, all out of fold; intervals are 2.5 and 97.5 percentiles of a paired
bootstrap over positions. The rule used is the best table form (`tbl_*`) when its paired
difference to the best reaches zero or it loses by less than 1 % of the log loss, else the best.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize, minimize_scalar

from tnh import movemix as mm

N_BINS = [(0, 0), (1, 5), (6, 20), (21, 100), (101, 1000), (1001, 10**12)]
FIT_FORMS = ("nk", "tbl_nk", "tbl_hill", "move")
FLOOR = 1e-15


def table(raw: pl.DataFrame, prior: pl.DataFrame, column: str) -> pl.DataFrame:
    """One row per (position, legal move): c (train games), t (test games), q (the prior)."""
    cnt = raw.group_by("pos", "move", "part").agg(pl.col("n").sum())
    tr = cnt.filter(pl.col("part") == "train").select("pos", "move", pl.col("n").alias("c"))
    te = cnt.filter(pl.col("part") == "test").select("pos", "move", pl.col("n").alias("t"))
    d = (prior.select("pos", "move", pl.col(column).alias("q"))
         .join(tr, on=["pos", "move"], how="left").join(te, on=["pos", "move"], how="left")
         .with_columns(pl.col("c").fill_null(0).cast(pl.Float64), pl.col("t").fill_null(0).cast(pl.Float64)))
    d = d.with_columns(N=pl.col("c").sum().over("pos"), T=pl.col("t").sum().over("pos"), k=pl.len().over("pos"))
    return d.filter(pl.col("T") > 0).sort("pos", "move")


class Data:
    def __init__(self, d: pl.DataFrame) -> None:
        self.pos = d["pos"].to_numpy()
        self.upos, self.inv = np.unique(self.pos, return_inverse=True)
        for c in ("c", "t", "q", "N", "T", "k"):
            setattr(self, c, d[c].to_numpy().astype(float))
        self.share = np.where(self.N > 0, self.c / np.maximum(self.N, 1), 0.0)
        # test games on moves without an earlier game, where the database alone is infinite
        self.unseen = float(np.sum(self.t[(self.c == 0) & (self.N > 0)]) / np.sum(self.t[self.N > 0]))

    def sub(self, mask: np.ndarray) -> "Data":
        o = object.__new__(Data)
        for c in ("pos", "inv", "c", "t", "q", "N", "T", "k", "share"):
            setattr(o, c, getattr(self, c)[mask])
        o.upos, o.inv = np.unique(o.pos, return_inverse=True)
        return o


def probs(form: str, params, d: Data) -> np.ndarray:
    if form == "jeffreys":
        return (d.c + 0.5) / (d.N + 0.5 * d.k)
    w = mm.weight(form, params, d.N, d.c)
    p = w * d.share + (1 - w) * d.q
    if form == "move":
        s = np.bincount(d.inv, weights=p, minlength=len(d.upos))
        p = p / s[d.inv]
    return p


def per_position(form: str, params, d: Data) -> tuple[np.ndarray, np.ndarray]:
    """(log-loss sum, squared-error sum times T) per position."""
    p = probs(form, params, d)
    with np.errstate(divide="ignore"):
        ll = -d.t * np.where(d.t > 0, np.log(np.maximum(p, 0.0) if form == "database" else np.maximum(p, FLOOR)), 0.0)
    se = (p - d.t / d.T) ** 2 * d.T
    return (np.bincount(d.inv, weights=ll, minlength=len(d.upos)),
            np.bincount(d.inv, weights=se, minlength=len(d.upos)))


def logloss(form, params, d: Data) -> float:
    return float(per_position(form, params, d)[0].sum() / d.t.sum())


def fit(form: str, d: Data) -> np.ndarray:
    if form in ("model", "database", "jeffreys"):
        return np.zeros(0)
    if form == "nk":
        r = minimize_scalar(lambda lk: logloss("nk", [lk], d), bounds=(-4, 14), method="bounded")
        return np.array([r.x])
    tbl = [-1.0] + [math.log(0.3)] * 5
    init = {"tbl_nk": tbl + [math.log(100.0)], "tbl_hill": tbl + [0.0, math.log(100.0)], "move": [-3.0, 0.5, 0.0]}[form]
    r = minimize(lambda x: logloss(form, x, d), np.array(init), method="L-BFGS-B")
    return r.x


def compare(d: Data, folds: int = 5, boot: int = 500, seed: int = 11) -> dict:
    fold = (d.upos * np.uint64(0x9E3779B97F4A7C15) >> np.uint64(60)).astype(np.int64) % folds
    rowfold = fold[d.inv]
    forms = ("model", "database", "jeffreys", "nk100") + FIT_FORMS
    ll = {f: np.zeros(len(d.upos)) for f in forms}
    se = {f: np.zeros(len(d.upos)) for f in forms}
    params = {f: [] for f in forms}
    for k in range(folds):
        tr, te = rowfold != k, rowfold == k
        dtr, dte = d.sub(tr), d.sub(te)
        pos_te = np.isin(d.upos, dte.upos)
        for f in forms:
            if f == "nk100":
                prm, form = np.array([math.log(100.0)]), "nk"
            else:
                prm, form = fit(f, dtr), f
            params[f].append([float(x) for x in prm])
            a, b = per_position(form, prm, dte)
            ll[f][pos_te], se[f][pos_te] = a, b
    T = np.bincount(d.inv, weights=d.t, minlength=len(d.upos))
    Npos = np.bincount(d.inv, weights=d.c, minlength=len(d.upos))
    rng = np.random.default_rng(seed)
    counts = np.stack([np.bincount(rng.integers(0, len(d.upos), len(d.upos)), minlength=len(d.upos))
                       for _ in range(boot)])
    groups = {"all": np.ones(len(d.upos), bool)}
    for lo, hi in N_BINS:
        groups[f"N={lo}" if lo == hi else f"N {lo}-{hi}" if hi < 10**12 else f"N>{lo - 1}"] = (Npos >= lo) & (Npos <= hi)
    out = {"positions": int(len(d.upos)), "test_games": float(T.sum()), "unseen_share": d.unseen,
           "params": params, "groups": {}}
    for g, m in groups.items():
        res = {"positions": int(m.sum()), "test_games": float(T[m].sum())}
        bs = {}
        for f in forms:
            tot = T[m].sum()
            point = ll[f][m].sum() / tot if tot else float("nan")
            with np.errstate(invalid="ignore"):
                b = (counts[:, m] @ ll[f][m]) / (counts[:, m] @ T[m]) if tot else np.full(boot, np.nan)
            bs[f] = b
            res[f] = {"ll": float(point), "lo": float(np.nanpercentile(b, 2.5)) if np.isfinite(point) else point,
                      "hi": float(np.nanpercentile(b, 97.5)) if np.isfinite(point) else point,
                      "se": float(se[f][m].sum() / tot) if tot else float("nan")}
        finite = [f for f in forms if np.isfinite(res[f]["ll"])]
        if not finite:                      # no positions in this group
            res["best"], res["vs_best"] = None, {}
            out["groups"][g] = res
            continue
        order = sorted(finite, key=lambda f: res[f]["ll"])
        best = order[0]
        res["best"] = best
        res["vs_best"] = {f: {"diff": res[f]["ll"] - res[best]["ll"],
                              "lo": float(np.nanpercentile(bs[f] - bs[best], 2.5)),
                              "hi": float(np.nanpercentile(bs[f] - bs[best], 97.5))} for f in finite}
        out["groups"][g] = res
    a = out["groups"]["all"]
    tbls = sorted([f for f in ("tbl_nk", "tbl_hill") if np.isfinite(a[f]["ll"])], key=lambda f: a[f]["ll"])
    chosen = a["best"]
    if tbls:
        t = tbls[0]
        v = a["vs_best"][t]
        if v["lo"] <= 0 or v["diff"] < 0.01 * a[a["best"]]["ll"]:
            chosen = t
    out["chosen"] = chosen
    return out


def _commit() -> str:
    r = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


def report_md(res: dict, column: str) -> str:
    G = res["groups"]
    gs = list(G)
    forms = ["model", "jeffreys", "database", "nk100", "nk", "tbl_nk", "tbl_hill", "move"]
    lines = [f"### Prior: model 1 at `{column}`", "",
             f"{res['positions']} positions, {res['test_games']:.0f} test games; {100 * res['unseen_share']:.1f} % of "
             "the test games (positions with earlier games) are on moves with no earlier game.", "",
             "Log loss (nats per test game), out of fold, 95 % interval:", "",
             "| rule | " + " | ".join(f"{g} ({G[g]['positions']})" for g in gs) + " |", "|---|" + "---|" * len(gs)]
    for f in forms:
        cells = []
        for g in gs:
            r = G[g][f]
            cells.append("inf" if not math.isfinite(r["ll"]) else
                         f"{r['ll']:.4f}" + (f" [{r['lo']:.4f}, {r['hi']:.4f}]" if g == "all" else ""))
        lines.append(f"| {f} | " + " | ".join(cells) + " |")
    lines += ["", "Squared error of the test share (per position, weighted by its test games):", "",
              "| rule | " + " | ".join(gs) + " |", "|---|" + "---|" * len(gs)]
    for f in forms:
        lines.append(f"| {f} | " + " | ".join(f"{G[g][f]['se']:.4f}" for g in gs) + " |")
    lines += ["", "Paired difference to the best of each group (log loss):", ""]
    for g in gs:
        b = G[g]["best"]
        if b is None:
            continue
        lines.append(f"- {g}: best {b}; " + "; ".join(
            f"{f} {v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]" for f, v in G[g]["vs_best"].items() if f != b))
    rules = {f: mm.Rule(f if f != "nk100" else "nk", res["params"][f][0]) for f in ("nk", "tbl_nk", "tbl_hill", "move")}
    lines += ["", "Fitted rules (fold 1): " + "; ".join(f"{f}: {r.describe()}" for f, r in rules.items()),
              f"Used: {res['chosen']}.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tnh.web.probfit", description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--prior", type=Path, required=True)
    ap.add_argument("--prior-column", nargs="+", default=["p_mean", "p_2200"],
                    help="the first decides; the others are reported beside it")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--write-rule", action="store_true", help=f"store the rule used in {mm.RULE_FILE}")
    a = ap.parse_args(argv)
    raw = pl.read_parquet(a.data)
    prior = pl.read_parquet(a.prior / "*.parquet")
    meta = json.loads(a.data.with_suffix(".json").read_text()) if a.data.with_suffix(".json").exists() else {}
    mds, nums = [], {}
    for col in a.prior_column:
        t = time.perf_counter()
        d = Data(table(raw, prior, col))
        res = compare(d, boot=a.boot)
        res["seconds"] = round(time.perf_counter() - t, 1)
        nums[col] = res
        mds.append(report_md(res, col))
        print(mds[-1], flush=True)
        if col == a.prior_column[0] and a.write_rule:
            chosen = res["chosen"]
            prm = fit(chosen, d)
            rule = mm.Rule(chosen, [float(x) for x in prm])
            mm.RULE_FILE.write_text(json.dumps({
                "form": rule.form, "params": rule.params, "describe": rule.describe(),
                "provenance": {"fitted": date.today().isoformat(), "commit": _commit(), "sample": meta,
                               "prior": f"{col} from tnh.web.probprior", "positions": res["positions"],
                               "oof_logloss": res["groups"]["all"][chosen], "best": res["groups"]["all"]["best"],
                               "kappa100": res["groups"]["all"]["nk100"],
                               "command": f"python -m tnh.web.probfit --data <collect output> --prior <probprior dir> "
                                          f"--prior-column {col} --write-rule"}}, indent=2) + "\n")
            print(f"wrote {mm.RULE_FILE}: {rule.describe()}")
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.with_suffix(".md").write_text("\n".join(mds))
        a.report.with_suffix(".json").write_text(json.dumps(nums, indent=2, default=float))


if __name__ == "__main__":
    main()
