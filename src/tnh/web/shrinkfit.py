"""Which shrinkage rule predicts held-out games best: a model comparison for the move table.

    scripts/tnh-run web-shrink-collect 3 uv run python -m tnh.web.shrinkfit collect --walks 4000 --out PAIRS.parquet
    scripts/tnh-run web-shrink-fit 3 uv run python -m tnh.web.shrinkfit fit --data PAIRS.parquet [--write-rules]

**collect** draws positions from the edge tables (through `EdgeDB`, classical games by
default) by game-weighted random descent: each walk starts at the initial position and at
each step plays a move with probability proportional to its games, until a position has no
moves in the table or ply 60. Every distinct position visited is in the sample, so common
positions come in once and the walks' tails bring many rare ones. For each position and
move it stores the games, results and rating sums before `--split-year` (train) and from it
(test).

**fit** compares, per quantity, the rules of `tnh.web.shrinkrule`, all of the form
`prediction = prior + w * (move's train mean - prior)`, predicting the move's test mean.

- A pair is a (position, move) with at least one train game (one with both ratings for the
  rating quantities) and at least one test game. Pairs are weighted by their test games
  (for the rating quantities, test games with both ratings), in fitting and in scoring.
- The rating quantities are deltas within a period: the move's mean minus the position's
  mean over all its moves in the same period, so the drift of ratings between periods
  cancels. Their prior `base` is 0.
- Score and Draw are White's score and the draw rate of the move. Without a prior column
  they too are deltas within a period (the move against its position in the same period;
  the draw rate fell between the periods and would otherwise be learnt as part of the
  rule), with `base` 0. With `--prior-file`/`--prior-column` (model 2's value for the
  position after the move, once it exists) they are absolute and `base` is that column.
- Folds: five, by position. Every MSE is out of fold. Intervals: 2.5 and 97.5 percentiles of
  a bootstrap over positions (`--boot` resamples), also for the winner's lead over the
  runner-up, paired.

What the comparison cannot see: moves with no games in the test years are not in it, and
the same players recur across the split, which favours rules that trust a move's own past.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import date
from pathlib import Path

import chess
import numpy as np
import polars as pl
from scipy.optimize import minimize, minimize_scalar

from tnh.web import shrinkrule as sr

QUANTITIES = ("w_elo", "b_elo", "chooser", "gap", "score", "draw")   # and "prob", which needs a prior column
REPORT_BINS = [(1, 1), (2, 5), (6, 20), (21, 10**12)]


# --- collect -----------------------------------------------------------------------


def collect(db, walks: int, seed: int, tempi: list[str] | None, split_year: int, max_ply: int = 60,
            log=print) -> pl.DataFrame:
    from tnh.data.edges import pos_hash
    rng = np.random.default_rng(seed)
    start = pos_hash(chess.Board())
    nodes: dict[int, tuple[np.ndarray, np.ndarray] | None] = {}
    rows = []
    t0 = time.perf_counter()
    for w in range(walks):
        h, ply = start, 0
        while ply < max_ply:
            if h not in nodes:
                p = db.panel(h, tempi=tempi)
                if p["total"] is None or p["moves"].height == 0:
                    nodes[h] = None
                    break
                white = p["total"]["epd"].split()[1] == "w"
                y = p["by_year"].filter(pl.col("year") > 0).with_columns(
                    part=pl.when(pl.col("year") < split_year).then(pl.lit("train")).otherwise(pl.lit("test")),
                    sum_mover=pl.col("mover_elo").fill_null(0) * pl.col("n_rated"),
                    sum_opp=pl.col("opp_elo").fill_null(0) * pl.col("n_rated"))
                g = y.group_by("part", "move").agg(
                    pl.col("n", "white_wins", "draws", "black_wins", "n_rated", "sum_mover", "sum_opp").sum())
                rows.append(g.with_columns(
                    pos=pl.lit(h, pl.UInt64), white_to_move=pl.lit(white),
                    sum_white=pl.col("sum_mover" if white else "sum_opp"),
                    sum_black=pl.col("sum_opp" if white else "sum_mover"),
                ).select("pos", "white_to_move", "part", "move", "n", "white_wins", "draws", "black_wins",
                         "n_rated", "sum_white", "sum_black"))
                m = p["moves"]
                nodes[h] = (m["child_hash"].to_numpy(), m["n"].to_numpy().astype(float))
            node = nodes[h]
            if node is None:
                break
            kids, n = node
            h = int(kids[rng.choice(len(kids), p=n / n.sum())])
            ply += 1
        if (w + 1) % 500 == 0:
            log(f"{w + 1} walks, {len(rows)} positions, {time.perf_counter() - t0:.0f} s")
    return pl.concat(rows)


# --- pairs -------------------------------------------------------------------------


def quantity_frame(raw: pl.DataFrame, q: str, within_period: bool = True) -> pl.DataFrame:
    """Per (pos, part, move): y (the move's mean of `q`), count, the position's count and spread."""
    d = raw.filter(pl.col("n") > 0)
    if q == "prob":
        d = d.with_columns(count=pl.col("n").cast(pl.Float64))
        d = d.with_columns(y=pl.col("count") / pl.col("count").sum().over("pos", "part"), pos_mean=pl.lit(0.0))
    elif q in ("score", "draw"):
        d = d.with_columns(count=pl.col("n").cast(pl.Float64),
                           y=((pl.col("white_wins") + 0.5 * pl.col("draws")) if q == "score" else pl.col("draws"))
                           / pl.col("n"))
        pos = d.group_by("pos", "part").agg(((pl.col("y") * pl.col("count")).sum() / pl.col("count").sum()).alias("pos_mean"))
        d = d.join(pos, on=["pos", "part"])
        if within_period:
            # As the rating quantities: the move against its position in the same period, so
            # a change in the draw rate or White's score between the periods cancels.
            d = d.with_columns(y=pl.col("y") - pl.col("pos_mean"), pos_mean=pl.lit(0.0))
    else:
        d = d.filter(pl.col("n_rated") > 0).with_columns(
            count=pl.col("n_rated").cast(pl.Float64),
            wm=pl.col("sum_white") / pl.col("n_rated"), bm=pl.col("sum_black") / pl.col("n_rated"))
        pos = d.group_by("pos", "part").agg(
            (pl.col("sum_white").sum() / pl.col("n_rated").sum()).alias("pw"),
            (pl.col("sum_black").sum() / pl.col("n_rated").sum()).alias("pb"))
        d = d.join(pos, on=["pos", "part"])
        mover = pl.when(pl.col("white_to_move"))
        y = {"w_elo": pl.col("wm") - pl.col("pw"), "b_elo": pl.col("bm") - pl.col("pb"),
             "chooser": mover.then(pl.col("wm") - pl.col("pw")).otherwise(pl.col("bm") - pl.col("pb")),
             "gap": mover.then((pl.col("wm") - pl.col("bm")) - (pl.col("pw") - pl.col("pb")))
             .otherwise((pl.col("bm") - pl.col("wm")) - (pl.col("pb") - pl.col("pw")))}[q]
        d = d.with_columns(y=y, pos_mean=pl.lit(0.0))
    agg = d.group_by("pos", "part").agg(
        pl.col("count").sum().alias("pos_n"),
        ((pl.col("count") * (pl.col("y") - (pl.col("y") * pl.col("count")).sum() / pl.col("count").sum()) ** 2).sum()
         / pl.col("count").sum()).sqrt().alias("spread"))
    return d.join(agg, on=["pos", "part"]).select("pos", "part", "move", "y", "count", "pos_n", "spread", "pos_mean")


def pairs(raw: pl.DataFrame, q: str, prior: pl.DataFrame | None = None, prior_column: str | None = None) -> pl.DataFrame:
    # With a prior column (model 2, model 1) the figures are absolute; without, Score and
    # Draw are deltas within the period, like the rating quantities.
    f = quantity_frame(raw, q, within_period=prior is None)
    tr = f.filter(pl.col("part") == "train").drop("part")
    te = f.filter(pl.col("part") == "test").select("pos", "move", pl.col("y").alias("y_test"),
                                                  pl.col("count").alias("weight"))
    if q == "prob":
        # Prob: the share's weight depends on the position's games, not the move's, and the
        # pair counts with the position's test games. A move with no train games is in.
        te = f.filter(pl.col("part") == "test").select("pos", "move", pl.col("y").alias("y_test"),
                                                      pl.col("pos_n").alias("weight"))
        p = tr.join(te, on=["pos", "move"], how="right").with_columns(
            pl.col("y").fill_null(0.0), pl.col("count").fill_null(0.0))
        p = p.with_columns(pl.col("pos_n").fill_null(strategy="max").over("pos"), pl.col("spread").fill_null(0.0))
        p = p.with_columns(count=pl.col("pos_n")).filter(pl.col("count") >= 1)
        if prior is None:
            raise SystemExit("prob needs --prior-file and --prior-column (model 1's probability per pos, move)")
    else:
        p = tr.join(te, on=["pos", "move"]).filter((pl.col("count") >= 1) & (pl.col("weight") >= 1))
    p = p.with_columns(base=pl.col("pos_mean"))
    if prior is not None:
        p = p.drop("base").join(prior.select("pos", "move", pl.col(prior_column).alias("base")),
                                on=["pos", "move"], how="inner")
    return p


# --- fitting -----------------------------------------------------------------------


def _arrays(p: pl.DataFrame) -> dict:
    return {c: p[c].to_numpy().astype(float) for c in ("y", "base", "count", "pos_n", "spread", "y_test", "weight")}


def _mse(form, params, a, s) -> float:
    pred = sr.predict(form, params, a["y"], a["base"], a["count"], a["pos_n"], a["spread"], s)
    return float(np.sum(a["weight"] * (pred - a["y_test"]) ** 2) / np.sum(a["weight"]))


def fit_form(form: str, a: dict, s: float) -> np.ndarray:
    if form in ("none", "prior"):
        return np.zeros(0)
    scale = np.sum(a["weight"] * (a["y_test"] - a["base"]) ** 2) / np.sum(a["weight"]) or 1.0
    if form == "nk":
        r = minimize_scalar(lambda lk: _mse(form, [lk], a, s), bounds=(-6, 14), method="bounded")
        return np.array([r.x])
    if form == "bins":
        return bin_weights(a)
    tbl = [-2.0] + [np.log(0.3)] * 5
    inits = {"hill": [0.0, np.log(5)], "logistic": [0.0, 1.0, 0.0, 0.0],
             "share_hill": [0.0, 0.0, 0.0, 0.0, np.log(5)], "share_logistic": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
             "tbl_hill": tbl + [0.0, np.log(5)], "tbl_logistic": tbl + [0.0, 1.0, 0.0, 0.0],
             "share_tbl_hill": [0.0, 0.0, 0.0] + tbl + [0.0, np.log(5)],
             "share_tbl_logistic": [0.0, 0.0, 0.0] + tbl + [0.0, 1.0, 0.0, 0.0]}
    r = minimize(lambda x: _mse(form, x, a, s) / scale, np.array(inits[form]), method="L-BFGS-B")
    return r.x


def bin_weights(a: dict) -> np.ndarray:
    """The least-squares weight per bin of n, clipped to [0, 1]."""
    b = sr.bin_index(a["count"])
    dx, dy = a["y"] - a["base"], a["y_test"] - a["base"]
    w = np.zeros(len(sr.BINS))
    for i in range(len(sr.BINS)):
        m = b == i
        den = np.sum(a["weight"][m] * dx[m] ** 2)
        w[i] = np.clip(np.sum(a["weight"][m] * dx[m] * dy[m]) / den, 0, 1) if den > 0 else 0.0
    return w


def compare(p: pl.DataFrame, forms=sr.FORMS, folds: int = 5, boot: int = 200, seed: int = 11) -> dict:
    a = _arrays(p)
    pos = p["pos"].to_numpy()
    s = float(np.median(a["spread"][a["spread"] > 0])) if np.any(a["spread"] > 0) else 1.0
    fold = (pos * np.uint64(0x9E3779B97F4A7C15) >> np.uint64(60)).astype(np.int64) % folds
    oof = {f: np.zeros(len(pos)) for f in forms}
    fitted = {f: [] for f in forms}
    for k in range(folds):
        tr, te = fold != k, fold == k
        atr = {c: v[tr] for c, v in a.items()}
        for f in forms:
            prm = fit_form(f, atr, s)
            fitted[f].append(prm.tolist())
            oof[f][te] = sr.predict(f, prm, a["y"][te], a["base"][te], a["count"][te], a["pos_n"][te],
                                    a["spread"][te], s)
    # bootstrap over positions
    upos, inv = np.unique(pos, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(upos), size=(boot, len(upos)))
    counts = np.stack([np.bincount(d, minlength=len(upos)) for d in draws])   # boot x positions
    groups = {"all": np.ones(len(pos), bool)}
    for lo, hi in REPORT_BINS:
        groups[f"{lo}" if lo == hi else f"{lo}-{hi}" if hi < 10**12 else f">{lo - 1}"] = (a["count"] >= lo) & (a["count"] <= hi)
    out = {"pairs": len(pos), "positions": len(upos), "spread_scale": s, "groups": {}}
    sq = {f: a["weight"] * (oof[f] - a["y_test"]) ** 2 for f in forms}
    for g, m in groups.items():
        wsum = np.bincount(inv[m], weights=a["weight"][m], minlength=len(upos))
        res = {"pairs": int(m.sum()), "weight": float(a["weight"][m].sum())}
        bs = {}
        for f in forms:
            esum = np.bincount(inv[m], weights=sq[f][m], minlength=len(upos))
            point = esum.sum() / wsum.sum()
            b = (counts @ esum) / (counts @ wsum)
            bs[f] = b
            res[f] = {"mse": float(point), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5))}
        order = sorted(forms, key=lambda f: res[f]["mse"])
        diff = bs[order[1]] - bs[order[0]]
        res["winner"], res["runner_up"] = order[0], order[1]
        res["lead"] = {"mse": res[order[1]]["mse"] - res[order[0]]["mse"],
                       "lo": float(np.percentile(diff, 2.5)), "hi": float(np.percentile(diff, 97.5))}
        # The rule to use: the fewest parameters among the winner and the rules whose paired
        # difference to it has a 95 % interval reaching zero.
        tied = [f for f in order if f == order[0] or np.percentile(bs[f] - bs[order[0]], 2.5) <= 0]
        res["chosen"] = min(tied, key=lambda f: (sr.N_PARAMS[f], res[f]["mse"]))
        res["tied_with_winner"] = tied
        # The owner prefers a rule whose weight for few games is a table in n alone, so one
        # game cannot look like evidence: the best table form is used when its paired
        # difference to the best reaches zero or it loses by less than 1 % of the MSE.
        tbls = [f for f in order if f in sr.TABLE_FORMS]
        if tbls:
            t = tbls[0]
            d = bs[t] - bs[order[0]]
            res["table"] = {"form": t, "mse": res[t]["mse"], "vs_best": res[t]["mse"] - res[order[0]]["mse"],
                            "rel": (res[t]["mse"] - res[order[0]]["mse"]) / res[order[0]]["mse"],
                            "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}
            if res["table"]["lo"] <= 0 or res["table"]["rel"] < 0.01:
                res["chosen"] = t
        out["groups"][g] = res
    out["fold_params"] = fitted
    return out


def curve(rule: sr.Rule, p: pl.DataFrame) -> dict:
    """w at n = 1, 2, 3, 5, 10, 20, 50, 100, 200, other features at their medians."""
    ns = np.array([1, 2, 3, 5, 10, 20, 50, 100, 200], float)
    med_pos = float(np.median(p["pos_n"].to_numpy()))
    med_spread = float(np.median(p["spread"].to_numpy()))
    w = rule.weight(ns, np.full_like(ns, med_pos), np.full_like(ns, med_spread))
    return {int(n): round(float(x), 3) for n, x in zip(ns, w)}


def _commit() -> str:
    r = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


def _table(q: str, res: dict, bins_all: list[float], w_curve: dict) -> str:
    groups = list(res["groups"])
    lines = [f"### {q}", "", f"{res['pairs']} pairs over {res['positions']} positions. Out-of-fold weighted MSE "
             "(weight = test games), 95 % bootstrap interval over positions.", "",
             "| rule | " + " | ".join(f"{g} (pairs {res['groups'][g]['pairs']})" for g in groups) + " |",
             "|---|" + "---|" * len(groups)]
    forms = sorted(sr.FORMS, key=lambda f: res["groups"]["all"][f]["mse"])
    for f in forms:
        cells = []
        for g in groups:
            r = res["groups"][g][f]
            cells.append(f"{r['mse']:.4g} [{r['lo']:.4g}, {r['hi']:.4g}]")
        lines.append(f"| {f} | " + " | ".join(cells) + " |")
    a = res["groups"]["all"]
    lines += ["", f"Winner {a['winner']}, runner-up {a['runner_up']}: lead {a['lead']['mse']:.4g} "
              f"[{a['lead']['lo']:.4g}, {a['lead']['hi']:.4g}]. Not distinguishable from the winner: "
              f"{', '.join(a['tied_with_winner'])}.",
              (f"Best table form {a['table']['form']}: {a['table']['vs_best']:+.4g} against the best "
               f"({100 * a['table']['rel']:+.2f} %) [{a['table']['lo']:.4g}, {a['table']['hi']:.4g}]." if "table" in a else ""),
              f"Used: {a['chosen']}.",
              "Least-squares weight per bin of n (all pairs): " + ", ".join(
                  f"{lo if lo == hi else f'{lo}-{hi}' if hi < 10**12 else f'{lo}+'}: {w:.3f}"
                  for (lo, hi), w in zip(sr.BINS, bins_all)),
              f"The used rule's w(n) at the median position (median N and spread): {w_curve}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tnh.web.shrinkfit", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--table", default="full")
    c.add_argument("--walks", type=int, default=4000)
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--tempi", default="classical", help="comma list, or 'all'")
    c.add_argument("--split-year", type=int, default=2024)
    c.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("fit")
    f.add_argument("--data", type=Path, required=True)
    f.add_argument("--quantities", nargs="*", default=list(QUANTITIES))
    f.add_argument("--prior-file", type=Path, help="parquet with pos, move and the prior column")
    f.add_argument("--prior-column")
    f.add_argument("--boot", type=int, default=200)
    f.add_argument("--report", type=Path, help="write the tables and the numbers here (.md and .json)")
    f.add_argument("--write-rules", action="store_true", help=f"store the winners in {sr.RULES_FILE.name}")
    a = ap.parse_args(argv)

    if a.cmd == "collect":
        from tnh.data.edges import EdgeDB
        tempi = None if a.tempi == "all" else a.tempi.split(",")
        raw = collect(EdgeDB(name=a.table), a.walks, a.seed, tempi, a.split_year)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        meta = {"table": a.table, "walks": a.walks, "seed": a.seed, "tempi": tempi, "split_year": a.split_year,
                "positions": raw["pos"].n_unique(), "rows": raw.height, "date": date.today().isoformat(),
                "commit": _commit()}
        raw.write_parquet(a.out)
        a.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2))
        return

    raw = pl.read_parquet(a.data)
    meta = json.loads(a.data.with_suffix(".json").read_text()) if a.data.with_suffix(".json").exists() else {}
    prior = pl.read_parquet(a.prior_file) if a.prior_file else None
    tables, numbers, rules = [], {}, {}
    for q in a.quantities:
        t = time.perf_counter()
        p = pairs(raw, q, prior if q in ("score", "draw", "prob") else None, a.prior_column)
        res = compare(p, boot=a.boot)
        win = res["groups"]["all"]["chosen"]
        prm = fit_form(win, _arrays(p), res["spread_scale"])
        rule = sr.Rule(win, [float(x) for x in prm], res["spread_scale"], {
            "fitted": date.today().isoformat(), "commit": _commit(), "sample": meta,
            "pairs": res["pairs"],
            "prior": a.prior_column or "0: the move's mean against the position's mean in the same period",
            "collect": ("python -m tnh.web.shrinkfit collect --table {table} --walks {walks} --seed {seed} "
                        "--tempi {t} --split-year {split_year} --out <file>").format(
                            t=",".join(meta["tempi"]) if meta.get("tempi") else "all", **meta) if meta else None,
            "command": "python -m tnh.web.shrinkfit fit --data <collect output>"
                       + (f" --prior-file {a.prior_file} --prior-column {a.prior_column}" if a.prior_file else "")
                       + " --write-rules",
            "oof_mse": res["groups"]["all"][win], "runner_up": res["groups"]["all"]["runner_up"],
            "lead": res["groups"]["all"]["lead"]})
        rules[q] = rule
        bins_all = bin_weights(_arrays(p)).tolist()
        w_curve = curve(rule, p)
        numbers[q] = res | {"bins_all": bins_all, "winner_curve": w_curve, "winner_params": rule.params,
                            "describe": rule.describe(), "seconds": round(time.perf_counter() - t, 1)}
        tables.append(_table(q, res, bins_all, w_curve) + f"Rule: {rule.describe()}\n")
        print(tables[-1], flush=True)
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.with_suffix(".md").write_text("\n".join(tables))
        a.report.with_suffix(".json").write_text(json.dumps(numbers, indent=2))
    if a.write_rules:
        old = json.loads(sr.RULES_FILE.read_text())["rules"] if sr.RULES_FILE.exists() else {}
        old.update({q: {"form": r.form, "params": r.params, "spread_scale": r.spread_scale,
                        "describe": r.describe(), "provenance": r.provenance} for q, r in rules.items()})
        sr.RULES_FILE.write_text(json.dumps({"rules": old}, indent=2) + "\n")
        print(f"wrote {sr.RULES_FILE}")


if __name__ == "__main__":
    main()
