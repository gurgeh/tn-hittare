"""reports/step2.md from the result files in reports/step2/."""

from __future__ import annotations

import argparse
import datetime as dt
import json

from tnh.difficulty.positions import OUT, RESULTS
from tnh.paths import REPORTS

NAMES = {
    "el_stm": "expected loss, side to move",
    "hard_stm": "narrow path, side to move",
    "el_next": "expected loss, other side next ply",
    "hard_next": "narrow path, other side next ply",
    "net2": "two-ply net",
    "maia_value": "Maia value head (= its difference from the engine)",
    "maia_value_eq": "Maia value head at equal ratings (H4)",
    "el_db": "REFERENCE: expected loss with the games' own moves",
    "hard_db": "REFERENCE: narrow path with the games' own moves",
}


def f(c: dict, k: int = 3, scale: float = 1.0) -> str:
    return f"{c['est'] * scale:.{k}f} [{c['lo'] * scale:.{k}f}, {c['hi'] * scale:.{k}f}]"


def verdict(res: dict) -> list[str]:
    ms = {m: v for m, v in res["measures"].items() if not v["reference"]}
    pos = [m for m, v in ms.items() if v["dmse"]["lo"] > 0]
    best = max(ms, key=lambda m: ms[m]["dmse"]["est"])
    b = ms[best]
    lines = []
    if pos:
        lines.append(f"**{len(pos)} of {len(ms)} measures lower the held-out error with the whole 95 % interval above "
                     f"zero: {', '.join(NAMES[m] for m in pos)}.**")
    else:
        lines.append(f"**No measure lowers the held-out error with its 95 % interval above zero ({len(ms)} measures "
                     f"tested).**")
    lines.append(f"The largest point estimate is *{NAMES[best]}*: it explains {f(b['share'], 3, 100)} % of what the "
                 f"baseline leaves above the binomial noise. Between the tenth of positions where it favours White most "
                 f"and the tenth where it favours Black most, the real score minus the baseline's prediction differs by "
                 f"{f(b['tenths']['diff'], 3)} points per game.")
    if "all_measures" in res:
        lines.append(f"All measures together: {f(res['all_measures']['share'], 3, 100)} % of the excess.")
    for m in sorted(ms, key=lambda m: -ms[m]["dmse"]["est"]):
        d = ms[m]["dmse"]
        if d["est"] > 0 and d["lo"] <= 0:
            se = (d["hi"] - d["lo"]) / 3.92
            k = (2.8 * se / d["est"]) ** 2
            lines.append(f"If *{NAMES[m]}*'s held-out gain is as large as its point estimate, about {k:.1f} times as "
                         f"many independent families would put its interval above zero four times in five "
                         f"(interval width shrinking as one over the square root of the families).")
            break
    refs = {m: v for m, v in res["measures"].items() if v["reference"]}
    for m, v in refs.items():
        lines.append(f"For scale, the reference *{NAMES[m]}* (the games' own moves, not a predictor) explains "
                     f"{f(v['share'], 3, 100)} %.")
    return lines


def measure_table(res: dict) -> list[str]:
    out = ["| measure | beta | per SD of the measure | held-out dMSE (x1e-5) | share of excess (%) | "
           "top tenth: real - baseline | bottom tenth | top - bottom |",
           "|---|---|---|---|---|---|---|---|"]
    for m, v in res["measures"].items():
        t = v["tenths"]
        out.append(f"| {NAMES[m]} | {f(v['beta'], 3)} | {f(v['beta_per_sd'], 4)} | {f(v['dmse'], 2, 1e5)} | "
                   f"{f(v['share'], 2, 100)} | {f(t['top'], 3)} | {f(t['bottom'], 3)} | {f(t['diff'], 3)} |")
    if "all_measures" in res:
        a = res["all_measures"]
        out.append(f"| all {len([m for m, v in res['measures'].items() if not v['reference']])} together | | | {f(a['dmse'], 2, 1e5)} | {f(a['share'], 2, 100)} | | | |")
    return out


def split_tables(res: dict) -> list[str]:
    out = []
    for name, groups in res["splits"].items():
        out += [f"### By {name}", "",
                "Top tenth minus bottom tenth of the real score minus the held-out baseline, tenths taken within "
                "the group; slope of that residual on the measure.", ""]
        gs = list(groups)
        out.append("| measure | " + " | ".join(f"{g} ({groups[g]['positions']} pos., {groups[g]['families']} fam.)" for g in gs) + " |")
        out.append("|---|" + "---|" * len(gs))
        for m in next(iter(groups.values()))["measures"]:
            cells = []
            for g in gs:
                e = groups[g]["measures"][m]
                cells.append(f"{f(e['tenths']['diff'], 3)}; slope {f(e['slope'], 2)}")
            out.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
        out.append("")
    return out


def position_list(rows: list[dict], measure: str) -> list[str]:
    out = ["| # | line | games | White's score | engine (cp, White) | baseline | with measure | predicted edge | measure | ratings W/B |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        side = "White" if r["edge"] > 0 else "Black"
        out.append(f"| {i} | {r['san']} | {r['n']} | {r['y']:.3f} | {r['cp_w_raw']:+.0f} | {r['pred_base']:.3f} | "
                   f"{r['oof_' + measure]:.3f} | {abs(r['edge']):.3f} for {side} | {r[measure]:+.4f} | "
                   f"{r['white_elo']:.0f}/{r['black_elo']:.0f} |")
    out += ["", "EPDs, in the same order:", "", "```"] + [f"{i:2d} {r['epd']}" for i, r in enumerate(rows, 1)] + ["```"]
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dev_n100")
    a = ap.parse_args(argv)
    an = json.loads((RESULTS / f"analysis_{a.tag}.json").read_text())
    syn_p = RESULTS / "synth.json"
    syn = json.loads(syn_p.read_text()) if syn_p.exists() else None
    runs = sorted((OUT / "runs").glob(f"compute_{a.tag}_*.json"))
    runs = [json.loads(p.read_text()) for p in runs]
    res, si, b = an["all"], an["set_info"], an["budget"]
    L = [f"# Step 2: do difficulty measures explain real results beyond the engine?", "",
         f"Generated by `uv run python -m tnh.difficulty report --tag {a.tag}` on "
         f"{dt.datetime.now():%Y-%m-%d %H:%M} from `{RESULTS}`.", "", "## Answer", ""]
    L += verdict(res) + [""]
    if "leaves" in an:
        L.append("Only the positions where the line ends in the set (no child with enough games; their games do not "
                 "overlap): " + " ".join(verdict(an["leaves"])[:2]))
        L.append("")
    L += ["## What was measured", "",
          f"- Edge tables `{si['name']}` (games dated 2024 or later). Games: tempo `{si['tempo']}`, `elo_band` "
          f"{si['bands'][0]}-{si['bands'][1]} (mean of the two ratings floored to 200, so means 1800-2599).",
          f"- Positions reached by at least N such games: "
          + ", ".join(f"N={k}: {v}" for k, v in si["positions_with_at_least"].items()) + ".",
          f"- The set: N = {si['min_games']}, {si['positions']} positions, {si['leaves']} of them with no child in the "
          f"set. Every position got a path by walking the most played parent (walk failures: {si['walk']}).",
          f"- Analysed: {an['frame']['analysed']} positions ({an['frame']['above_family_ply']} above ply "
          f"{an['family_ply']} have no family and are left out; {an['frame']['dropped_not_finite']} lack a measure; "
          f"model-call gaps {an['measure_counts']}).",
          f"- Games summed over positions: {res['games_summed']:,} (a game counts once per position it reaches).",
          f"- Clusters: {res['families']} opening families (first {an['family_ply']} plies of the path). "
          f"Games arriving from a parent in another family (transpositions across clusters): "
          f"{an['cross_family']['share'] * 100:.1f} %.",
          f"- Nested positions: the main analysis keeps every position and resamples families, so a position and "
          f"its children are always drawn together. The check uses only positions with no child in the set, whose "
          f"games are disjoint.",
          "", "### Models", "",
          f"- Maia-3 **{b['maia_size']}** on CPU (float32, one thread) at `maia_elo = 1580 + 0.558 * fide` for both "
          f"sides, the mean FIDE ratings of the position's games, rounded to 10.",
          f"- Stockfish (one thread): root value MultiPV 1 at {b['root_nodes']:,} nodes; every legal move's value "
          f"by MultiPV at {b['multipv_nodes']:,} nodes for the whole search. Expected score = (W + D/2) from "
          f"Stockfish's WDL.",
          f"- Children for the next-ply measures: the side to move's moves by Maia probability until "
          f"{b['child_mass']:.0%} is covered, at most {b['max_children']}, none below {b['min_child_p']}. "
          f"A move holds when it loses at most {an['eps']} in expected score.",
          ]
    for r in runs:
        L.append(f"- Compute run: {r['positions_done']} positions in {r['wall'] / 60:.0f} min, calls {r['stats']}, "
                 f"seconds {r['seconds']}, peak RSS {r['peak_rss_mb']} MB.")
    L += ["", "### The analysis", "",
          f"- Outcome: White's score in the position's games. Weighted least squares with weights "
          f"1/(sigma2/n + tau2): sigma2 = {res['sigma2']:.4f} per game, tau2 = {res['tau2']:.2e} "
          f"(between positions, after the baseline).",
          f"- Baseline: {', '.join(an['base'])}. Measures are added one at a time, signed so that positive favours "
          f"White.",
          f"- Held out: {res['folds']} folds of families; every prediction from a fit that did not see the family. "
          f"dMSE = baseline error minus error with the measure (positive = the measure helps). Share = dMSE / "
          f"(baseline error - binomial noise); baseline error {res['mse_base']:.3e}, noise {res['noise_mse']:.3e}.",
          f"- Intervals: 95 % percentile, bootstrap over families ({res['boot']} replicates).",
          "", "## Facit first: the planted world", ""]
    if syn:
        L += ["Results generated from engine value + rating difference + beta x measure + a position effect + "
              "game noise, on a real-move opening tree with nested positions, run through the same position-set "
              "and test code.", "",
              "| world | positions | families | planted beta | recovered beta | dMSE (x1e-5) | noise measure's beta |",
              "|---|---|---|---|---|---|---|"]
        for k, v in syn.items():
            L.append(f"| {k} | {v['positions']} | {v['families']} | {v['planted_beta']:g} | {f(v['planted']['beta'])} | "
                     f"{f(v['planted']['dmse'], 2, 1e5)} | {f(v['noise_measure']['beta'])} |")
    else:
        L.append("Not run.")
    L += ["", "## Baseline", "", "| term | coefficient |", "|---|---|"]
    for c, v in res["base_coef_ci"].items():
        L.append(f"| {c} | {f(v, 4)} |")
    L += ["", "## Each measure added to the baseline", "", f"All {res['positions']} positions:", ""]
    L += measure_table(res)
    if "leaves" in an:
        L += ["", f"Only positions with no child in the set ({an['leaves']['positions']} positions, "
                  f"{an['leaves']['families']} families):", ""]
        L += measure_table(an["leaves"])
    rob_p = RESULTS / f"analysis_{a.tag}_elo2.json"
    if rob_p.exists():
        rob = json.loads(rob_p.read_text())
        L += ["", f"Robustness, baseline with non-linear rating terms ({', '.join(rob['base'])}), all positions:", ""]
        L += measure_table(rob["all"])
    mv = an["maia_vs_games"]
    L += ["", "## Maia's difficulty against the games' own", "",
          "Side to move, weighted by games. The games' version uses the in-filter move shares at the position.", "",
          "| | Maia | games | correlation over positions |", "|---|---|---|---|",
          f"| expected loss | {mv['mean_el_maia']:.4f} | {mv['mean_el_games']:.4f} | {mv['corr_el']:.3f} |",
          f"| 1 - hold | {mv['mean_hard_maia']:.3f} | {mv['mean_hard_games']:.3f} | {mv['corr_hard']:.3f} |", ""]
    names = an["measure_corr"]["names"]
    L += ["Correlations between the signed measures:", "", "| | " + " | ".join(names) + " |",
          "|---|" + "---|" * len(names)]
    for n_, row in zip(names, an["measure_corr"]["matrix"]):
        L.append(f"| {n_} | " + " | ".join(f"{x:+.2f}" for x in row) + " |")
    L += ["", "## Splits (all positions)", ""] + split_tables(res)
    li = an["lists"]
    L += ["## For a chess player", "",
          f"Measure: *{NAMES[li['measure']]}* (largest held-out dMSE point estimate), beta {li['beta']:.3f}. "
          f"Predicted edge = beta x (measure - its mean). 'Baseline' is the held-out baseline prediction of White's "
          f"score, 'with measure' the held-out prediction with the measure added.", "",
          f"### The 20 largest predicted edges with the engine within {li['near_cp']:.0f} cp of equal "
          f"({li['near_positions']} such positions)", ""]
    L += position_list(li["edges"], li["measure"])
    L += ["", "### The 20 worst misses", "", "Largest |real - prediction| in units of the expected error "
          "(sqrt(sigma2/n + tau2)).", ""]
    L += position_list(li["misses"], li["measure"])
    p = REPORTS / "step2.md"
    p.write_text("\n".join(L) + "\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
