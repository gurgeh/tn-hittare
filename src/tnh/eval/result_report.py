"""The result-model report: ``reports/result_model.md`` from the json files of tnh.eval.result and lookahead.

    uv run python -m tnh.eval.result_report [--results R] [--out reports/result_model.md]

The main table is the speed front: positions per second against test log loss on the look-ahead test
positions (2,000 positions at plies 10-40, one per game), from the cheapest model to the dearest;
a point is on the front when no other point is both faster and better.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from tnh.eval.common import RESULTS, boot_counts, boot_means, cluster_codes, est, read_sample
from tnh.paths import REPORTS

PRED = Path.home() / "out/tn-hittare/scratch/eval/result/predictions.parquet"


def f(e, d=4):
    if e is None:
        return "–"
    return f"{e['est']:.{d}f} [{e['lo']:.{d}f}, {e['hi']:.{d}f}]" if e.get("lo") is not None else f"{e['est']:.{d}f}"


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"


def load(results: Path, name: str):
    p = results / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def front_points(R: Path, rows_file: Path, mover: str = "fast", boot: int = 1000) -> tuple[list[dict], int]:
    """Every point of the speed front evaluated on the same rows: the positions of ``rows_file`` that have every
    input (static predictions, the look-ahead run, Lc0 at 400 and 1600 nodes where those exist)."""
    from tnh.eval.lookahead import OUT
    from tnh.eval.result import SOURCES, design, losses, mlogit_probs, table
    from tnh.paths import FEATURES

    rates = load(R, "result_bench")["rates"]
    res = load(R, "result")["variants"]
    test = read_sample(rows_file).with_columns(pl.col("ply").cast(pl.Int64))
    pred = pl.read_parquet(PRED)
    la = pl.read_parquet(OUT / f"{mover}.parquet").with_columns(pl.col("ply").cast(pl.Int64))
    j = test.join(pred, on=["game_id", "ply"], how="inner").join(la, on=["game_id", "ply"], how="inner", suffix="_la")
    eng = {}
    for nodes in (400, 1600):
        d = FEATURES / "lc0-bt4" / f"nodes{nodes}-wdl-offset+400"
        if d.exists():
            w, ctx, _ = table(j, {**SOURCES, "lc0w": d})
            if not np.isnan(w["lc0w"]).any():
                eng[nodes] = (w, ctx)
    y = np.array([{1.0: 0, 0.5: 1, 0.0: 2}[x] for x in j.get_column("result_mover").to_list()])
    cl = cluster_codes(j)
    counts = boot_counts(cl.max() + 1, boot, 7)
    W = np.ones(len(y))

    def point(name, rate, p, kind, note=""):
        ll = losses(np.asarray(p), y)[0]
        return {"name": name, "rate": rate, "logloss": est(ll.mean(), boot_means(ll, W, cl, counts)[:, 0]),
                "kind": kind, "note": note, "_ll": ll}

    def rate_of(*rs):
        return 1.0 / sum(1.0 / r for r in rs)

    maia1, maia2, pol, s50 = (rates["maia3-79m one rating"], rates["maia3-79m two ratings"],
                              rates["lc0 raw value (policy call)"], rates["lc0 wdl 50 nodes"])
    cost = {"ratings only": float("inf"), "maia0": maia1, "maia600": maia1, "lc0raw": pol, "lc0w": s50,
            "maia0+maia600+lc0raw": rate_of(maia2, pol), "lc0w+lc0raw": rate_of(s50, pol),
            "lc0w+maia0+maia600": rate_of(s50, maia2), "all": rate_of(s50, maia2, pol),
            "maia0+maia600+lc0raw trees": rate_of(maia2, pol), "all trees": rate_of(s50, maia2, pol)}
    label = {"ratings only": "ratings only", "maia0": "Maia value head (+0)", "maia600": "Maia value head (+600)",
             "lc0raw": "Lc0 raw value head", "lc0w": "Lc0 calibrated WDL, 50 nodes",
             "maia0+maia600+lc0raw": "Maia (two) + Lc0 raw (cheap)", "lc0w+lc0raw": "Lc0 WDL 50 + Lc0 raw",
             "lc0w+maia0+maia600": "Lc0 WDL 50 + Maia (two)", "all": "all four (best static)",
             "maia0+maia600+lc0raw trees": "cheap, trees", "all trees": "all four, trees"}
    pts = []
    for k, c in cost.items():
        pts.append(point(label[k] + ("" if "trees" in k else " (logit)"), c, j.get_column(k).to_list(), "static"))
    for nodes, (w, ctx) in eng.items():
        r = rates.get(f"lc0 wdl {nodes} nodes")
        th = np.array(res["lc0w"]["coef"])
        pts.append(point(f"Lc0 calibrated WDL, {nodes} nodes (logit, 50-node coefficients)", r,
                         mlogit_probs(th, design(w, ctx, ["lc0w"])), "engine"))
        th = np.array(res["all"]["coef"])
        pts.append(point(f"all four with Lc0 WDL at {nodes} nodes", rate_of(r, maia2, pol),
                         mlogit_probs(th, design(w, ctx, res["all"]["sources"])), "engine"))
    # look-ahead on top of the best static model: (1 - lambda) best + lambda look-ahead, lambda out of fold
    best = np.array(j.get_column("all").to_list())
    secs = {k: float(j.get_column(f"per_position_{k}").mean()) for k in
            ("exact1_seconds", "exact2_seconds", "mc_seconds", "mc_evals")}
    per_eval = secs["mc_seconds"] / secs["mc_evals"]
    t_best = 1.0 / cost["all"]
    folds = cl % 5
    grid = np.linspace(0, 1, 21)
    for v in [c for c in la.columns if c in ("exact1", "exact2") or c.startswith("mc_")]:
        look = np.array(j.get_column(v).to_list())
        look = look / look.sum(1, keepdims=True)
        sec = secs["exact1_seconds"] if v == "exact1" else (secs["exact1_seconds"] + secs["exact2_seconds"] if v == "exact2"
              else per_eval * int(v.split("_")[1][1:]) * int(v.split("_")[2][1:]))
        oof = np.zeros((len(y), 3))
        for k in range(5):
            tr, te = folds != k, folds == k
            lb = min(grid, key=lambda l: losses((1 - l) * best[tr] + l * look[tr], y[tr])[0].mean())
            oof[te] = (1 - lb) * best[te] + lb * look[te]
        pts.append(point(f"best static + look-ahead {v}", 1.0 / (t_best + sec), oof, "look-ahead",
                         f"look-ahead alone {losses(look, y)[0].mean():.4f}"))
    ref = next(p for p in pts if p["name"].startswith("all four (best static)"))["_ll"]
    for p in pts:
        d = p["_ll"] - ref
        p["minus_best_static"] = est(d.mean(), boot_means(d, W, cl, counts)[:, 0])
        p["front"] = not any(q is not p and q["rate"] >= p["rate"] and q["logloss"]["est"] < p["logloss"]["est"]
                             for q in pts)
    return sorted(pts, key=lambda p: -p["rate"]), len(y)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tnh.eval.result_report", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--out", type=Path, default=REPORTS / "result_model.md")
    ap.add_argument("--fronts", nargs=3, action="append", metavar=("ROWS", "MOVER", "NOTE"),
                    help="a positions file, the look-ahead run on it, and a note; one front table each")
    a = ap.parse_args(argv)
    S = Path.home() / "out/tn-hittare/corpus/samples"
    a.fronts = a.fronts or [[str(S / "lookahead_test600.parquet"), "fast", "every point, Lc0 at 1600 nodes included"]]
    R = a.results
    res, la, eng = load(R, "result"), load(R, "lookahead"), load(R, "lookahead_engine")
    o = []
    w = o.append
    w("# The result model and looking ahead\n\n")
    w(f"Generated by `uv run python -m tnh.eval.result_report` on {time.strftime('%Y-%m-%d %H:%M')} from `{R}`.\n\n")
    w("## The speed front\n\n")
    for rows_file, mover, note in a.fronts:
        pts, n = front_points(R, Path(rows_file), mover)
        w(f"**{n} test positions** (`{Path(rows_file).name}` with the `{mover}` look-ahead run; {note}). Positions per "
          f"second on the laptop GPU, shared with the repertoire builder and another developer's Lc0 run (lower than "
          f"alone), one process, Lc0 batches of 12. Combined costs add the sources' times; a look-ahead point adds the "
          f"measured time of its move-model evaluations to the best static model's (Monte Carlo variants pro rata to "
          f"N x d of the 32 x 16 run). On the front: no other point is both at least as fast and better.\n\n")
        w(table(["model", "positions/s", "log loss", "minus best static", "front", "note"],
                [[p["name"], "free" if p["rate"] == float("inf") else f"{p['rate']:.2f}", f(p["logloss"]),
                  f(p["minus_best_static"]), "yes" if p["front"] else "dominated", p.get("note", "")] for p in pts]) + "\n")
    if res:
        w(f"\n## Part A: the static result model\n\nOne position per game, {res['games']['test']} test games "
          f"({res['test_tournaments']} tournaments), {res['games']['fit']} fitting games; draw rate {res['draw_rate_test']:.3f}. "
          f"Priority subset: plies 10-40, players averaging 2000-2399 ({res['games']['test_priority']} test games). "
          f"Best: `{res['best']}`.\n\n")
        w(table(["variant", "model", "log loss", "minus best", "Brier (expected score)", "priority log loss"],
                [[k, v["model"], f(v["logloss"]), f(v["logloss_minus_best"]), f(v["brier"]), f(v["priority_logloss"])]
                 for k, v in res["variants"].items()]))
        w("\nWhat each source adds beside the others (rise in test log loss when dropped from `all`):\n\n")
        w(table(["source", "rise, all games", "rise, priority subset"],
                [[k, f(v["rise_without"]), f(v["priority_rise_without"])] for k, v in res["adds"].items()]))
        if res.get("offsets"):
            w("\nLc0's `WDLCalibrationElo` offset (each alone with the context, on the games where it exists):\n\n")
            w(table(["source", "games", "log loss", "Lc0 +400 on the same games", "difference"],
                    [[k, v["games"], f"{v['logloss']:.4f}", f"{v['lc0w_same_rows']:.4f}", f(v["minus_lc0w"])]
                     for k, v in res["offsets"].items()]))
        for model, cal in res["calibration"].items():
            for by, rows in cal.items():
                w(f"\nCalibration of `{model}` by {by}:\n\n")
                w(table(["group", "games", "score", "predicted", "draw rate", "predicted draws"],
                        [[r["group"], r["n"], f"{r['score']:.3f}", f"{r['pred_score']:.3f}", f"{r['draw_rate']:.3f}",
                          f"{r['pred_draw']:.3f}"] for r in rows]))
    if eng:
        w("\n## Look-ahead by search\n\n")
        w(table(["variant", "log loss", "minus 50 nodes", "Brier"],
                [[k, f(v["logloss"]), f(v.get("minus_50_nodes")), f(v["brier"])] for k, v in eng.items() if isinstance(v, dict) and "nodes" in v]))
    if la:
        for mover, r in la.items():
            if not isinstance(r, dict) or "variants" not in r:
                continue
            w(f"\n## Look-ahead with the move model (`{mover}`), {r['positions']} positions\n\n")
            w("Leaf: the cheap result model. `shift coef`: least squares of the score on the static expected score and "
              "the shift (look-ahead minus static); 0 = the look-ahead carries nothing, equal to the static coefficient "
              "= it should replace the static value. Blends: (1 - lambda) static + lambda look-ahead, lambda chosen out "
              "of fold (5 folds by tournament).\n\n")
            rows = []
            for k, v in r["variants"].items():
                if k == "static":
                    rows.append([k, f(v["logloss"]), "", f(v["brier"]), "", "", "", ""])
                    continue
                rg = v["regression"]
                rows.append([k, f(v["logloss"]), f(v["logloss_minus_static"]), f(v["brier"]),
                             f"{rg['shift']:+.2f} [{rg['shift_ci'][0]:+.2f}, {rg['shift_ci'][1]:+.2f}] (static {rg['static']:.2f})",
                             f(v["blend"]["oof_minus_static"]), f(v.get("blend_full", {}).get("oof_minus_static")),
                             f"{v['shift']['mean']:+.4f} / {v['shift']['sd']:.4f}"])
            w(table(["variant", "log loss", "minus static", "Brier", "shift coef", "blend with cheap static: minus static",
                     "blend with best static: minus best", "shift mean / sd"], rows))
            w("\nCost per position: " + ", ".join(f"{k} {v:.3f}" for k, v in r["cost"].items()) + "\n")
            for by, rows_ in (r.get("shift_by") or {}).items():
                w(f"\nShift of `{r['variant_for_shift_by']}` by {by}:\n\n")
                w(table(["group", "n", "mean", "sd"], [[x["group"], x["n"], f"{x['mean']:+.4f}", f"{x['sd']:.4f}"] for x in rows_]))
    a.out.write_text("".join(o))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
