"""Regenerate the step-1 report from the result files on disk.

    uv run python -m tnh.eval.report [--results reports/step1] [--out reports/step1.md] [--note TEXT]

Reads the json files the analysis commands wrote into ``--results``
(``logloss``, ``sweep``, ``mistakes``, ``logit``, ``stack``, ``breakdown``, ``book``, ``theory``, ``value``,
``reference``; any that is missing is said to be missing), draws the figures
into the same directory, and writes the markdown. Every table says which sample
and which feature variant its numbers came from. Nothing is computed here that
is not in the json files, apart from the figures.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tnh.paths import REPORTS  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUES = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)


def blues(n):
    if n <= 1:
        return [BLUES[5]]
    return [BLUES[round(i * (len(BLUES) - 1) / (n - 1))] for i in range(n)]


def f(e, d=4):
    if e is None:
        return "–"
    if isinstance(e, dict):
        if e.get("est") is None:
            return "–"
        return f"{e['est']:.{d}f} [{e['lo']:.{d}f}, {e['hi']:.{d}f}]"
    return f"{e:.{d}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def sample_line(prov, key):
    s = prov["samples"].get(key)
    if not s:
        return f"{key}: –"
    m = s.get("meta") or {}
    return (f"`{s['path']}` ({m.get('rows', '?')} rows, {m.get('games', '?')} games, "
            f"{m.get('tournaments', '?')} tournaments, split {m.get('split', '?')}, seed {m.get('seed', '?')})")


def src_line(prov):
    return ", ".join(f"`{k}`" for k in prov["features"])


def load(results: Path, name: str):
    p = results / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


# --- sections -------------------------------------------------------------------------

def sec_samples(w, R):
    w("## Samples\n")
    for name in ("logloss", "sweep", "logit"):
        r = R.get(name)
        if r:
            prov = r["provenance"]
            w(f"- Test: {sample_line(prov, 'test')}\n- Fitting: {sample_line(prov, 'fit')}\n")
            m = (prov["samples"].get("test") or {}).get("meta") or {}
            if m:
                w(f"- Drawn by `tnh.eval.sample` from `{m.get('positions')}`: classical games with clocks, "
                  f"`tempo_source` not `assumed`, both ratings {m.get('elo_min')}-{m.get('elo_max')}; window rows "
                  f"plus {m.get('extra_frac')} extra rows per window row from outside the window; stratified on "
                  f"the mover's rating in 100-point buckets. Window: book_count {m.get('book_count')}, max_ply "
                  f"{m.get('max_ply')}, max_ply_no_clock {m.get('max_ply_no_clock')}, min_clock_s "
                  f"{m.get('min_clock_s')}.\n")
                strata = m.get("strata", [])
                if strata:
                    w("\n" + table(["part", "bucket", "sampled", "population"],
                                   [[s["part"], s["bucket"], s["sampled"], s["population"]] for s in strata]))
            break
    w("\nIntervals everywhere are 95 % percentile intervals of a bootstrap over tournaments. Log losses are in "
      "nats per move; pooled numbers are weighted to the population by the sample weights.\n\n")


def sec_logloss(w, r):
    w("## Log loss of the played move\n\n")
    if not r:
        w("_No `logloss.json`._\n\n"); return
    w(f"Test window rows: {r['rows']['test']} ({r['tournaments']['test']} tournaments); `eps` fitted on "
      f"{r['rows']['fit']} fitting rows. Mixed = `-log((1-eps) p + eps/n)`.\n\n")
    w(table(["model / variant", "eps", "raw", "mixed", "top-1", "entropy"],
            [[f"`{m['label']}`", f"{m['eps']:.5f}", f(m["pooled"]["raw"]), f(m["pooled"]["mixed"]),
              f(m["pooled"]["top1"], 3), f(m["pooled"]["entropy"], 3)] for m in r["models"]]))
    w("\nBy bucket of the mover's rating, mixed log loss:\n\n")
    buckets = sorted({b["bucket"] for m in r["models"] for b in m["by_bucket"]})
    rows = []
    for b in buckets:
        row = [b]
        for m in r["models"]:
            x = next((y for y in m["by_bucket"] if y["bucket"] == b), None)
            row.append("–" if x is None else f"{f(x['mixed'], 3)} (n={x['rows']})")
        rows.append(row)
    w(table(["bucket"] + [f"`{m['label']}`" for m in r["models"]], rows) + "\n")


def sec_sweep(w, r, figdir, rel):
    w("## The Elo sweep\n\n")
    if not r:
        w("_No `sweep.json`._\n\n"); return
    offs = r["offsets"]
    w(f"Maia-3 with both ratings shifted by the offset; variants {src_line(r['provenance'])}. Mixed log loss, "
      f"each variant with its own `eps` from the fitting sample. Test: {r['rows']['test']} window rows.\n\n")
    w(table(["offset", "eps", "test", "fitting sample"],
            [[f"{p['offset']:+g}", f"{p['eps']:.5f}", f(p["test"]), f"{p['fit']:.4f}"] for p in r["per_offset"]]))
    w("\nTest log loss by bucket and offset (point estimates; the lowest in each row in bold):\n\n")
    rows = []
    for b in r["per_bucket"]:
        vals = [x["est"] for x in b["loss"]]
        k = int(np.argmin(vals))
        rows.append([b["bucket"], b["rows"]] + [f"**{v:.3f}**" if i == k else f"{v:.3f}" for i, v in enumerate(vals)])
    w(table(["bucket", "rows"] + [f"{o:+g}" for o in offs], rows))
    w("\nBest offset per bucket on the test sample (parabola through the best grid point and its neighbours; "
      "interval from refinding the minimum in every bootstrap replicate):\n\n")
    w(table(["bucket", "rows", "best offset", "95 % interval", "at grid edge"],
            [[b["bucket"], b["rows"], f"{b['best_offset']:+.0f}", f"[{b['lo']:+.0f}, {b['hi']:+.0f}]",
              "yes" if b["at_edge"] else ""] for b in r["best_offset_test"]]))
    m = r["map"]
    w(f"\n**The map**, fitted on the fitting sample's per-bucket best offsets: `maia_elo = {m['a']:.0f} + "
      f"{m['b']:.3f} * fide` (a in [{m['a_ci'][0]:.0f}, {m['a_ci'][1]:.0f}], b in [{m['b_ci'][0]:.3f}, "
      f"{m['b_ci'][1]:.3f}]). Points used:\n\n")
    w(table(["bucket", "rows", "best offset (fitting sample)", "at grid edge"],
            [[p["bucket"], p["rows"], f"{p['best_offset']:+.0f}", "yes" if p["at_edge"] else ""]
             for p in m["fit_points"]]))
    c = r["compare"]
    w(f"\n**One offset or one per rating?** On the test sample, against the best single offset chosen on the "
      f"fitting sample ({c['single_offset']:+g}):\n\n")
    w(table(["rule", "test log loss", "minus single offset"],
            [[f"single offset {c['single_offset']:+g}", f(c["single"]), ""],
             ["per-bucket offset chosen on the fitting sample", f(c["per_bucket"]), f(c["per_bucket_minus_single"])],
             ["linear map, nearest grid offset per row", f(c["map"]), f(c["map_minus_single"])]]))
    if r.get("mover_opp"):
        mo = r["mover_opp"]
        movers = sorted({x["mover"] for x in mo["cells"]})
        opps = sorted({x["opp"] for x in mo["cells"]})
        cell = {(x["mover"], x["opp"]): x["test"]["est"] for x in mo["cells"]}
        w("\n**Mover and opponent offsets separately**, pooled test log loss (rows: mover's offset, columns: "
          "opponent's):\n\n")
        w(table(["mover \\ opp"] + [f"{o:+g}" for o in opps],
                [[f"{mv:+g}"] + [f"{cell[(mv, o)]:.4f}" if (mv, o) in cell else "" for o in opps] for mv in movers]))
        w("\nBest cell per bucket (share of bootstrap replicates in which it stays best):\n\n")
        w(table(["bucket", "mover", "opp", "loss", "boot share"],
                [[b["bucket"], f"{b['mover']:+g}", f"{b['opp']:+g}", f"{b['loss']:.4f}", f"{b['boot_share']:.2f}"]
                 for b in mo["best_per_bucket"]]))
    # figures
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    cols = blues(len(r["per_bucket"]))
    for col, b in zip(cols, r["per_bucket"]):
        v = np.array([x["est"] for x in b["loss"]])
        a1.plot(offs, v - v.min(), color=col, linewidth=2, marker="o", markersize=4, label=str(b["bucket"]))
    a1.set_xlabel("offset added to both ratings"); a1.set_ylabel("log loss above the bucket's minimum (nats)")
    a1.set_title("Test log loss by offset, per rating bucket", color=INK, fontsize=11)
    a1.legend(title="mover's rating", fontsize=8, frameon=False, ncol=2)
    style(a1)
    bt = r["best_offset_test"]
    x = np.array([b["bucket"] + 50 for b in bt])
    y = np.array([b["best_offset"] for b in bt])
    a2.errorbar(x, y, yerr=[y - np.array([b["lo"] for b in bt]), np.array([b["hi"] for b in bt]) - y],
                fmt="o", color=SERIES[0], ecolor=SERIES[0], capsize=3, markersize=6, label="best offset, test")
    xs = np.linspace(x.min() - 50, x.max() + 50, 50)
    a2.plot(xs, m["a"] + (m["b"] - 1) * xs, color=SERIES[1], linewidth=2, label="map from fitting sample")
    a2.set_xlabel("mover's rating (bucket midpoint)"); a2.set_ylabel("offset")
    a2.set_title("Best offset by rating", color=INK, fontsize=11)
    a2.legend(fontsize=8, frameon=False)
    style(a2)
    fig.tight_layout()
    fig.savefig(figdir / "sweep.png", dpi=130)
    plt.close(fig)
    w(f"\n![Elo sweep]({rel}/sweep.png)\n\n")


def cross_ci(x) -> str:
    """The interval of a calibration crossing, only when (nearly) every replicate has one: an interval over
    the replicates that happen to cross would be conditioned on crossing."""
    if x["lo"] is None:
        return ""
    if x["boot_share_inside"] < 0.95:
        return f"(crossing in {100 * x['boot_share_inside']:.0f} % of replicates)"
    return f"[{x['lo']:+.0f}, {x['hi']:+.0f}]"


def sec_shift(w, r, lg, figdir, rel):
    w("## Is classical play a rating-shifted blitz?\n\n")
    if not r:
        w("_No `mistakes.json`._\n\n")
    else:
        t = r["thresholds"]
        w(f"Every legal move classed by Stockfish: shallow `{r['shallow']}`, deep `{r['deep']}`. Expected-score "
          f"loss L against the best move at each budget. Sound: L_deep <= {t['sound']}; visible: a mistake at "
          f"the deep budget with L_shallow > {t['visible']}; deep only: a mistake at the deep budget that the "
          f"shallow budget does not see. Sizes at the deep budget: small up to {t['sizes'][0]}, medium up to "
          f"{t['sizes'][1]}, large above. Test window rows: {r['rows']} kept; dropped "
          f"{r['dropped_missing_value']} where a budget left a move without a value and "
          f"{r['dropped_decided']} decided positions (deep best value outside "
          f"[{1 - t['decided']:.2f}, {t['decided']:.2f}], where the expected score saturates).\n\n")
        w("A pure rating shift calibrates every class at one offset. The table gives, per class, the observed "
          "share of played moves, Maia-3's mass on the class at each offset, and the offset where the two "
          "meet (first crossing, linear between grid points). The interval is shown only when at least 95 % "
          "of the bootstrap replicates cross inside the grid; a class whose mass is not monotone in the offset "
          "can cross twice or not at all, and the figure is then the better read.\n\n")
        for kind in ("visibility", "size"):
            rows = []
            for c, x in r[kind]["pooled"].items():
                co = x["calibrated_offset"]
                ci = cross_ci(x)
                rows.append([c, f"{x['observed']:.4f}"] + [f"{p:.4f}" for p in x["predicted"]] +
                            [f"none ({x['side']})" if co is None else f"{co:+.0f}", ci,
                             f"{x['boot_share_inside']:.2f}"])
            w(table(["class", "observed"] + [f"mass at {o:+g}" for o in r["offsets"]] +
                    ["calibrated at", "95 % interval", "replicates with a crossing"], rows) + "\n")
        w("By bucket of the mover's rating, the offset where each class is calibrated:\n\n")
        names = list(r["visibility"]["pooled"]) + [f"size:{c}" for c in r["size"]["pooled"] if c != "sound"]
        rows = []
        for b, vb, sb in zip(r["buckets"], r["visibility"]["by_bucket"], r["size"]["by_bucket"]):
            row = [b, vb["rows"]]
            for c in names:
                x = sb[c.split(":")[1]] if c.startswith("size:") else vb[c]
                co = x["calibrated_offset"]
                row.append(f"none ({x['side']})" if co is None else f"{co:+.0f} {cross_ci(x)}")
            rows.append(row)
        w(table(["bucket", "rows"] + names, rows) + "\n")
        w("Robustness: the pooled calibrated offset per class under other thresholds (sound = visible threshold, "
          "size limits, decided cut-off). `over`/`under`: the model over- or under-predicts the class at every "
          "offset in the grid.\n\n")
        rows = []
        for x in r["robustness"]:
            def cell(k, c):
                v = x[k][c]
                return x[f"{k}_side"][c] if v is None else f"{v:+.0f}"
            rows.append([x["sound"], f"{x['sizes'][0]:.3f}/{x['sizes'][1]:.3f}", x["decided"], x["rows"]] +
                        [cell("visibility", c) for c in ("sound", "visible", "deep_only")] +
                        [cell("size", c) for c in ("small", "medium", "large")])
        w(table(["sound/visible", "sizes", "decided", "rows", "sound", "visible", "deep only", "small", "medium",
                 "large"], rows) + "\n")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for ax, kind in zip(axes, ("visibility", "size")):
            for i, (c, x) in enumerate(r[kind]["pooled"].items()):
                if x["ratio"] is None:
                    continue
                ax.plot(r["offsets"], x["ratio"], color=SERIES[i], linewidth=2, marker="o", markersize=4, label=c)
            ax.axhline(1.0, color=INK2, linewidth=1)
            ax.set_xlabel("offset added to both ratings")
            ax.set_title(f"Predicted / observed share, by {kind}", color=INK, fontsize=11)
            ax.legend(fontsize=8, frameon=False)
            style(ax)
        axes[0].set_ylabel("Maia-3 mass / share of played moves")
        fig.tight_layout()
        fig.savefig(figdir / "mistakes.png", dpi=130)
        plt.close(fig)
        w(f"![Calibration by kind of mistake]({rel}/mistakes.png)\n\n")
    if lg and all(k in lg["sets"] for k in ("A", "B", "C")):
        w("**Three descriptions in the conditional logit** (fitted on the fitting sample, test log loss, mixed with "
          "a fitted `eps`). A: Maia-3 at the best single offset alone. B: Maia-3 at offset 0 plus the engine loss "
          "terms. C: both Maia-3 settings plus the engine terms. If B is as good as A, the offset was standing in "
          "for calculation.\n\n")
        rows = []
        for k in ("A", "B", "C"):
            s = lg["sets"][k]
            rows.append([k, ", ".join(f"`{c}`" for c in s["columns"]), f(s["test_mixed"]), f(s["test_raw"])])
        w(table(["set", "features", "test, mixed", "test, raw"], rows))
        w("\nCoefficients of B and C are in the conditional-logit section below.\n\n")


def sec_breakdown(w, r, figdir, rel):
    w("## Breakdowns\n\n")
    if not r:
        w("_No `breakdown.json`._\n\n"); return
    w(f"Model `{r['model']}` with `eps` {r['eps']:.5f} from the fitting sample. Test sample.\n\n")
    w("By ply, window rows:\n\n")
    w(table(["ply", "rows", "mixed", "top-1", "entropy"],
            [[x["group"], x["rows"], f(x["mixed"], 3), f(x["top1"], 3), f(x["entropy"], 3)] for x in r["by_ply"]]))
    w(f"\nBy minutes left on the mover's clock: every out-of-book row before ply {r['max_ply']} with a clock, "
      f"{r['rows']['clock_rows']} rows, {r['rows']['clock_rows_outside_window']} of them outside the window, "
      f"weighted to the population:\n\n")
    w(table(["minutes left", "rows", "mixed", "top-1", "entropy"],
            [[x["group"], x["rows"], f(x["mixed"], 3), f(x["top1"], 3), f(x["entropy"], 3)] for x in r["by_clock"]]))
    w("\nPly x clock (mixed log loss, rows), because time left and ply move together:\n\n")
    plys = sorted({x["ply"] for x in r["ply_x_clock"]}, key=lambda s: int(s.split("-")[0]))
    groups = [g["group"] for g in r["by_clock"]]
    cell = {(x["ply"], x["group"]): x for x in r["ply_x_clock"]}
    w(table(["ply \\ minutes"] + groups,
            [[p] + [f"{cell[(p, g)]['mixed']['est']:.3f} ({cell[(p, g)]['rows']})" if (p, g) in cell else ""
                    for g in groups] for p in plys]))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    xs = np.arange(len(r["by_clock"]))
    est = np.array([x["mixed"]["est"] for x in r["by_clock"]])
    lo = np.array([x["mixed"]["lo"] for x in r["by_clock"]])
    hi = np.array([x["mixed"]["hi"] for x in r["by_clock"]])
    ax.errorbar(xs, est, yerr=[est - lo, hi - est], fmt="o-", color=SERIES[0], capsize=3, linewidth=2, markersize=6)
    ax.set_xticks(xs, [x["group"] for x in r["by_clock"]])
    ax.set_xlabel("minutes left on the mover's clock"); ax.set_ylabel("mixed log loss (nats)")
    ax.set_title("Log loss by time left", color=INK, fontsize=11)
    style(ax)
    fig.tight_layout()
    fig.savefig(figdir / "clock.png", dpi=130)
    plt.close(fig)
    w(f"\n![Log loss by clock]({rel}/clock.png)\n\n")


def _off(c):
    o = c["offset"]
    return f"{o['est']:+.0f} [{o['lo']:+.0f}, {o['hi']:+.0f}]" + (" edge" if o["at_edge"] else "")


def sec_book(w, r, figdir, rel):
    w("### Play just out of book\n\n")
    if not r:
        w("_No `book.json`._\n\n"); return
    t = r["thresholds"]
    w(f"Window rows of the test sample by plies since the game left book (`book_count = {r['book_count']}`), "
      f"within ply bands so that rows at the same stage of the game are compared. {r['rows']} rows; dropped "
      f"{r['dropped_missing_value']} with a move without an engine value and {r['dropped_decided']} decided. "
      f"**deep loss**: mean expected-score loss of the played move at `{r['deep']}`; **mistakes**, **visible**, "
      f"**deep only**: share of played moves in those classes (sound: deep loss <= {t['sound']}; visible: also a "
      f"loss > {t['visible']} at `{r['shallow']}`); **obs/Maia**: mistakes, and mean loss, divided by what Maia-3 "
      f"at {r['ref_offset']:+g} expects (Maia-3 memorised openings too, so this does not take out all of a "
      f"position's ease); **offset**: the Maia-3 offset with the lowest log loss in the cell. Cells with fewer than "
      f"{r['min_rows']} rows show only the count. If play just out of book is still better, the first columns are "
      f"lower in the exit and +1 rows than in the later rows of the same band.\n\n")
    hdr = ["plies", "since exit", "rows", "deep loss", "mistakes", "visible", "deep only", "obs/Maia",
           "loss obs/Maia", "offset", "log loss", "top-1"]
    rows = []
    for c in r["cells"]:
        if "offset" not in c:
            rows.append([c["band"], c["group"], c["rows"]] + [""] * (len(hdr) - 3))
            continue
        mist = {"est": 1 - c["sound"]["est"], "lo": 1 - c["sound"]["hi"], "hi": 1 - c["sound"]["lo"]}
        rows.append([c["band"], c["group"], c["rows"], f(c["deep_loss"], 4), f(mist, 3),
                     f"{c['visible']['est']:.3f}", f"{c['deep_only']['est']:.3f}", f(c["ratio_mistake"], 2),
                     f(c["ratio_loss"], 2), _off(c), f"{c['logloss']['est']:.3f}", f"{c['top1']['est']:.3f}"])
    w(table(hdr, rows) + "\n")


def sec_theory(w, r, figdir, rel):
    w("## Where theory ends\n\n")
    if not r:
        w("_No `theory.json`._\n\n"); return
    m = r["sample_meta"] or {}
    w(f"A separate test on its own sample (`{r['sample']}`): Maia-3 alone, no Stockfish, no window rule. Rows "
      f"from classical games with clocks, both players {m.get('elo_min')}-{m.get('elo_max')}, `tempo_source` not "
      f"`assumed`, splits {m.get('splits')}, stratified on ply band x `pos_count` band x rating group with up to "
      f"{m.get('per_cell')} rows per ply x `pos_count` cell; {r['rows']} rows, {r['tournaments']} tournaments. "
      f"Per cell: the offset with the lowest raw log loss (variants {', '.join('`' + v + '`' for v in r['variants'])}"
      f"; parabola vertex, bootstrap interval over tournaments) and the log loss there. Cells with fewer than "
      f"{r['min_rows']} rows show only their count.\n\n"
      "Reading it: describe the curve of the best offset against `pos_count` in each ply band (a row) and rating "
      "group as it is. No shape is assumed: it may rise where players know the position and fall again in the very "
      "most common ones, where a deliberately second-rate move takes the opponent out of theory. The two "
      "continuation columns speak to that without an engine, over the other games that reached the position (the "
      "row's own game taken out, so `pos_count` 1 has none): the share of played moves that are those games' most "
      "common continuation (1/k when k moves tie), and the mean share of those games that chose the played move "
      "(child counts include transpositions). A deliberate sideline shows as a low share in a very common "
      "position. **Where theory ends** is read from the low end only: the smallest `pos_count` band whose best "
      "offset differs from that of the rarest filled band of the same ply band (paired bootstrap difference, 95 % "
      "interval excluding 0).\n\n"
      "Caveats: `pos_count` counts games between players rated 1800 and up in the whole corpus, grandmasters "
      "included, and the game itself (1 means no other game), so it measures what is known to someone, not at the "
      "mover's level; Maia-3 memorised openings from its own training games, so in a common position the model is "
      "also better informed, whatever the player knows; and the pooled tables mix rating groups in proportions that "
      "differ from cell to cell while the best offset depends on rating, so the per-rating tables are the ones to "
      "read where their cells are filled.\n\n")
    pcs, plys = r["pos_count_bands"], r["ply_bands"]

    def grid(cells, val):
        cell = {(c["ply"], c["pos_count"]): c for c in cells}
        out = []
        for pb in plys:
            row = [pb]
            for pc in pcs:
                c = cell.get((pb, pc))
                row.append("" if c is None else (val(c) if "best" in c else f"({c['rows']})"))
            out.append(row)
        return out

    w("**Best offset** [interval] (rows), rows ply band, columns `pos_count`:\n\n")
    w(table(["ply \\ pos_count"] + pcs, grid(r["cells"], lambda c: f"{c['best']['est']:+.0f} [{c['best']['lo']:+.0f}, "
                                                               f"{c['best']['hi']:+.0f}]{' edge' if c['best']['at_edge'] else ''} ({c['rows']})")))
    w("\n**Log loss at the best offset** [interval]:\n\n")
    w(table(["ply \\ pos_count"] + pcs, grid(r["cells"], lambda c: f(c["loss_at_best"], 3))))
    w("\n**Share of played moves that are the most common continuation** [interval]:\n\n")
    w(table(["ply \\ pos_count"] + pcs, grid(r["cells"], lambda c: f(c["most_common_share"], 3))))
    w("\n**Mean share of continuing games that chose the played move** [interval]:\n\n")
    w(table(["ply \\ pos_count"] + pcs, grid(r["cells"], lambda c: f(c["chosen_share"], 3))))
    w("\n**Where theory ends**, from the low end: per ply band, the rarest filled `pos_count` band and the "
      "smallest band whose best offset differs from it (difference [interval]):\n\n")
    rows = []
    for g, ends in r["theory_end"].items():
        for e in ends:
            if e.get("ends_at"):
                x = e["ends_at"]
                ans = f"{x['pos_count']}: {x['est']:+.0f} [{x['lo']:+.0f}, {x['hi']:+.0f}]"
            else:
                ans = e.get("note", "")
            diffs = "; ".join(f"{d['pos_count']} {d['est']:+.0f} [{d['lo']:+.0f}, {d['hi']:+.0f}]"
                              for d in e.get("differences", []))
            rows.append([g, e["ply"], e["rarest"] or "", ans, diffs])
    w(table(["rating group", "plies", "rarest band", "theory ends at", "all differences from the rarest"], rows))
    for g, cells in r["by_rating"].items():
        w(f"\n**Best offset, movers rated {g}**:\n\n")
        w(table(["ply \\ pos_count"] + pcs, grid(cells, lambda c: f"{c['best']['est']:+.0f} [{c['best']['lo']:+.0f}, "
                                                            f"{c['best']['hi']:+.0f}] ({c['rows']})")))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    cols = blues(len(plys))
    for col, pb in zip(cols, plys):
        cs = [c for c in r["cells"] if c["ply"] == pb and "best" in c]
        if not cs:
            continue
        xs = np.array([pcs.index(c["pos_count"]) for c in cs])
        e = np.array([c["best"]["est"] for c in cs])
        lo = np.array([c["best"]["lo"] for c in cs])
        hi = np.array([c["best"]["hi"] for c in cs])
        ax.errorbar(xs, e, yerr=[np.maximum(e - lo, 0), np.maximum(hi - e, 0)], fmt="o-", color=col, capsize=2,
                    linewidth=2, markersize=5, label=f"plies {pb}")
    ax.set_xticks(np.arange(len(pcs)), pcs)
    ax.set_xlabel("pos_count (games reaching the position)"); ax.set_ylabel("best Maia-3 offset")
    ax.set_title("The Elo the play matches, by stage and commonness", color=INK, fontsize=11)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    style(ax)
    fig.tight_layout()
    fig.savefig(figdir / "theory.png", dpi=130)
    plt.close(fig)
    w(f"\n![Where theory ends]({rel}/theory.png)\n\n")


def mistake_tables(w, r, kinds, ref_offsets):
    """Per kind of class: pooled observed share and Maia-3 mass at each offset, and by rating bucket the
    ratio of mass to observed share at the reference offsets."""
    offs = r["offsets"]
    for kind in kinds:
        if kind not in r:
            continue
        rows = []
        for c, x in r[kind]["pooled"].items():
            co = x["calibrated_offset"]
            rows.append([c, f"{x['observed']:.4f}"] + [f"{p_:.4f}" for p_ in x["predicted"]] +
                        [f"none ({x['side']})" if co is None else f"{co:+.0f} {cross_ci(x)}"])
        w(f"**{kind}**, pooled: observed share of played moves and Maia-3's mass on each class:\n\n")
        w(table(["class", "observed"] + [f"mass at {o:+g}" for o in offs] + ["calibrated at"], rows) + "\n")
        refs = [o for o in ref_offsets if o in offs]
        hdr = ["bucket", "rows"] + [f"{c} at {o:+g}" for c in r[kind]["pooled"] for o in refs]
        rows = []
        for b, xb in zip(r["buckets"], r[kind]["by_bucket"]):
            row = [b, xb["rows"]]
            for c in r[kind]["pooled"]:
                for o in refs:
                    ratio = xb[c]["ratio"]
                    row.append("" if ratio is None else f"{ratio[offs.index(o)]:.2f}")
            rows.append(row)
        w(f"**{kind}**, Maia-3 mass / observed share by bucket of the mover's rating (1 = calibrated):\n\n")
        w(table(hdr, rows) + "\n")


def sec_mistakes_extra(w, r, title, note, ref_offsets):
    w(f"{title}\n\n")
    if not r:
        w("_Not run._\n\n"); return
    t = r["thresholds"]
    w(f"{note} Shallow `{r['shallow']}`, deep `{r['deep']}`" + (f", at a glance `{r['glance']}`" if r.get("glance") else "")
      + (f", prior `{r['prior']}`" if r.get("prior") else "") + f". Sound: deep loss <= {t['sound']}. "
      f"{r['rows']} rows kept; dropped {r['dropped_missing_value']} with a move without a value, "
      f"{r['dropped_decided']} decided.\n\n")
    if r.get("glance_def"):
        w(f"At a glance: a deep-budget mistake whose loss Lc0's raw child value already shows "
          f"({r['glance_def'][0]} >= {r['glance_def'][1]} against Lc0's best child); hidden: one it does not. "
          f"In mind: a deep-budget mistake with Lc0 policy prior >= {r.get('prior_p')}.\n\n")
    mistake_tables(w, r, ["visibility", "size", "glance", "prior"], ref_offsets)
    rg = r.get("robustness_glance") or []
    if rg:
        z = r["offsets"].index(0.0) if 0.0 in r["offsets"] else 0
        rows = []
        for x in rg:
            row = [x["glance_def"] and f"{x['glance_def'][0]} {x['glance_def'][1]:g}", x["prior_p"]]
            for kind, cs in (("glance", ["at_a_glance", "hidden"]), ("prior", ["in_mind", "not_in_mind"])):
                for c in cs:
                    v = (x.get(kind) or {}).get(c)
                    row.append("" if not v or v["ratio"] is None else f"{v['ratio'][z]:.2f} (obs {v['observed']:.3f})")
            rows.append(row)
        w("Robustness of the glance and prior classes, Maia-3 mass / observed at offset +0 (pooled):\n\n")
        w(table(["glance definition", "prior p", "at a glance", "hidden", "in mind", "not in mind"], rows) + "\n")


def sec_logit(w, r, title="## The conditional logit"):
    w(f"{title}\n\n")
    if not r:
        w("_Not run._\n\n"); return
    src = r["sources"]
    w("Sources: " + ", ".join(f"`{k}` = `{v}`" for k, v in {**src["logp"], **src["engine"]}.items()) + ".\n\n")
    rf, rt = r["rows"]["fit"], r["rows"]["test"]
    w(f"Rows: fitting {rf['rows']} (dropped {rf['rows_dropped_no_engine_value']} where an engine valued no move), "
      f"test {rt['rows']} (dropped {rt['rows_dropped_no_engine_value']}). Engine moves without a value, fitting "
      f"sample: " + ", ".join(f"{k} {v['moves_without_value']} of {v['moves']}" for k, v in rf.items()
                             if isinstance(v, dict)) +
      f". NaN indicators left out because too few played moves lack a value to fit them (those moves keep the "
      f"imputed worst loss): {', '.join(r['unused_nan_indicators']) or 'none'}.\n\n")
    w(table(["set", "features", "test, mixed", "test, raw", "eps", "Newton iterations"],
            [[k, ", ".join(f"`{c}`" for c in s["columns"]), f(s["test_mixed"]), f(s["test_raw"]),
              f"{s['eps']:.5f}", s["iterations"]] for k, s in r["sets"].items()]))
    w("\nSingle move models, each mixed with its own `eps` (same rows):\n\n")
    w(table(["source", "variant", "eps", "test, mixed"],
            [[k, f"`{s['label']}`", f"{s['eps']:.5f}", f(s["test_mixed"])] for k, s in r["singles"].items()]))
    w("\nEach set minus each single model (paired, negative is better):\n\n")
    w(table(["comparison", "difference"], [[k, f(v)] for k, v in r["set_minus_single"].items()]))
    w(f"\nCoefficients, with intervals from {r['boot_fit']} refits on bootstrap replicates of the fitting "
      f"sample's tournaments:\n\n")
    rows = []
    for k, s in r["sets"].items():
        for i, c in enumerate(s["columns"]):
            ci = f"[{s['beta_lo'][i]:+.3f}, {s['beta_hi'][i]:+.3f}]" if "beta_lo" in s else ""
            rows.append([k, f"`{c}`", f"{s['beta'][i]:+.3f}", ci])
    w(table(["set", "feature", "coefficient", "95 % interval"], rows))
    if r["entered_last"]:
        w(f"\nEntered last: the rise in test log loss when a source is dropped from set `{r['full']}`:\n\n")
        w(table(["source dropped", "rise"], [[k, f(v["test_rise"])] for k, v in r["entered_last"].items()]))
    w("\n")


def sec_stack(w, r, title="## Non-linear stacks: boosted trees and an MLP"):
    w(f"{title}\n\n")
    if not r:
        w("_Not run._\n\n"); return
    o = r["options"]
    rows = r["rows"]
    w(f"A score per legal move from its own features and the position's context, softmax over the legal moves, "
      f"trained on the played move with the weighted log loss: `score = b . x_move + MLP([x_move, x_context])`, two "
      f"hidden layers of {o['hidden']} (SiLU), the skip `b` starting at the linear logit. Fitted on {rows['train']} "
      f"rows of the fitting sample, early stopping on {rows['validation']} rows of held-out tournaments, reported "
      f"on {rows['test']} test window rows; every model mixed with its own `eps`.\n\n"
      f"- Per-move features: {', '.join('`' + c + '`' for c in r['move_features'])}.\n"
      f"- Context features: {', '.join('`' + c + '`' for c in r['context_features'])}.\n"
      f"- The interaction logit is the linear logit plus each engine loss x rating and x log clock, and each log "
      f"probability x rating (standardised rating and clock), so its coefficients can be read.\n\n")
    w("The boosted trees (LightGBM) use the exact group softmax as a custom objective (gradient `w(p - y)`, "
      "hessian `w p(1 - p)` within the position), start from the linear logit's score, and get the per-move "
      "features, the context repeated per move and, in the main variant, explicit products of each engine loss "
      "and each log probability with rating, log clock and ply (a context feature alone cannot be split on at the "
      "root: its gradient sums to zero over whole positions).\n\n")
    order = ["trees", "trees without products", "stack", "interactions", "linear+board", "linear"] + \
        [k for k in r["models"] if k.startswith("maia ")]
    has_t = "trees" in r["models"]
    w(table(["model", "size", "test, mixed", "minus linear logit", "minus MLP"] + (["minus trees"] if has_t else []),
            [[k if k != "stack" else "MLP", r["models"][k].get("params", ""), f(r["models"][k]["test"]),
              f(r["models"][k]["minus_linear"]), f(r["models"][k]["minus_stack"])]
             + ([f(r["models"][k]["minus_trees"])] if has_t else []) for k in order if k in r["models"]]))
    if has_t:
        t = r["models"]["trees"]
        w(f"\nTree settings tried (validation log loss on the held-out tournaments of the fitting sample; chosen: "
          f"{t['settings']['num_leaves']} leaves, rate {t['settings']['learning_rate']}, min "
          f"{t['settings']['min_data_in_leaf']} rows per leaf, {t['trees']} trees):\n\n")
        w(table(["leaves", "rate", "min rows per leaf", "trees", "validation", "seconds"],
                [[x["num_leaves"], x["learning_rate"], x["min_data_in_leaf"], x["trees"], f"{x['validation']:.4f}",
                  x["seconds"]] for x in t["tried"]]))
        w("\nGain importance of the trees (share of total gain, top 15; the linear start is not in it):\n\n")
        w(table(["feature", "share"], [[f"`{n}`", f"{v:.3f}"] for n, v in t["importance"][:15]]))
    if r["ablation"]:
        w("\nAblation: the stack retrained without one group of features. The rise in test log loss is what the "
          "group earns; the cost is what it takes to compute per position, which the repertoire builder pays at "
          "every node.\n\n")
        abl_t = r.get("ablation_trees") or {}
        w(table(["without", "features dropped", "cost per position", "MLP rise"] + (["trees rise"] if abl_t else []),
                [[g, ", ".join(f"`{c}`" for c in v["dropped"]), v["cost"], f(v["rise"])]
                 + ([f(abl_t[g]["rise"]) if g in abl_t else ""] if abl_t else []) for g, v in r["ablation"].items()]))
    for feat, tabs in r["effective_weight"].items():
        w(f"\n**Effective weight on `{feat}`**: the derivative of a move's score with respect to the raw feature, "
          f"averaged over the moves of the positions in each group (MLP by autograd, trees by central differences of "
          f"0.1 standard deviations; intervals over test tournaments for the models as fitted), the interaction "
          f"logit's coefficient at the same contexts, and the linear logit's constant.\n\n")
        for by, rows_ in tabs.items():
            ht = "trees" in rows_[0]
            w(table([("minutes left" if by == "clock" else "mover's rating"), "rows"] + (["trees"] if ht else [])
                    + ["MLP", "interaction logit", "linear logit"],
                    [[x["group"], x["rows"]] + ([f(x["trees"], 2)] if ht else [])
                     + [f(x["stack"], 2), f(x["interactions"], 2), f"{x['linear']:.2f}"] for x in rows_]) + "\n")
    w("\n")


def sec_value(w, r, figdir, rel, title="## The value head", figname="value.png"):
    w(f"{title}\n\n")
    if not r:
        w("_Not run._\n\n"); return
    w(f"One window row per game (`--pick {r['pick']}`; median ply {r['ply_quartiles_test'][1]:.0f}), "
      f"result from the mover's side. Test {r['games']['test']} games, draw rate {r['draw_rate']['test']:.3f}; "
      f"fitting {r['games']['fit']} games, draw rate {r['draw_rate']['fit']:.3f}. A model that only knows the "
      f"fitting sample's outcome frequencies scores {r['base_rate']['test_logloss']:.4f}. `raw + eps`: the WDL mixed "
      f"with a uniform 1/3 by a fitted `eps` (needed where a WDL puts exactly 0 on the outcome that happened). "
      f"Recalibrated: `q = (1-eps) softmax(t log p_c + b_c + [draw] k (mean rating - 2200)/100) + eps/3`, "
      f"`b_loss = 0`, five parameters fitted on the fitting sample.\n\n")
    w(table(["model", "raw", "games with p=0", "raw + eps (eps)", "recalibrated", "recal - (raw + eps)",
             "P(draw) raw", "P(draw) recal", "t, b_win, b_draw, k, eps"],
            [[f"`{m['label']}`", f(m["raw"]) if m["raw"]["est"] != float("inf") else "inf",
              m["games_with_zero_probability_on_the_result"], f"{f(m['raw_plus_eps'])} ({m['eps_only']:.3f})",
              f(m["recalibrated"]), f(m["recal_minus_raw_plus_eps"]),
              f"{m['mean_p_draw_raw']:.3f}", f"{m['mean_p_draw_recal']:.3f}",
              ", ".join(f"{v:.3f}" for v in m["recal"].values())] for m in r["models"]]))
    keys = [k[: -len("_p_draw")] for k in r["calibration"][0] if k.endswith("_p_draw")]
    for what, suffix, real in (("mean predicted draw probability", "_p_draw", "draw_rate"),
                               ("mean predicted expected score (win + draw/2)", "_exp_score", "score")):
        w(f"\nCalibration by bucket of the mover's rating (test), {what}, against the real "
          f"{'draw rate' if real == 'draw_rate' else 'score'}:\n\n")
        w(table(["bucket", "games", "real"] + [f"`{k}`" for k in keys],
                [[c["bucket"], c["games"], f"{c[real]:.3f}"] + [f"{c[k + suffix]:.3f}" for k in keys]
                 for c in r["calibration"]]))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = [c["bucket"] + 50 for c in r["calibration"]]
    ax.plot(x, [c["draw_rate"] for c in r["calibration"]], color=INK, linewidth=2, marker="s", markersize=6,
            label="real draw rate")
    for i, k in enumerate(keys[:7]):
        ax.plot(x, [c[k + "_p_draw"] for c in r["calibration"]], color=SERIES[i], linewidth=2, marker="o",
                markersize=4, label=k, linestyle="--" if k.endswith("raw") else "-")
    ax.set_xlabel("mover's rating (bucket midpoint)"); ax.set_ylabel("draw probability")
    ax.set_title("Predicted against real draw rate", color=INK, fontsize=11)
    ax.legend(fontsize=7, frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(figdir / figname, dpi=130)
    plt.close(fig)
    w(f"\n![Value head calibration]({rel}/{figname})\n\n")


def sec_reference(w, r):
    w("## The blitz reference\n\n")
    if not r:
        w("_No `reference.json`._\n\n"); return
    b, c = r["blitz"], r["classical"]
    w(f"Common window: blitz {r['rule']['blitz']}; classical {r['rule']['classical']}. Raw log loss, no mixture. "
      f"Blitz: {b['games']} Lichess games, {b['rows']} positions, from `{r['blitz_positions']}`, variants "
      f"{', '.join('`' + x + '`' for x in b['labels'])}. Classical: {c['rows']} test-sample rows from "
      f"{c['games']} games, variants {', '.join('`' + x + '`' for x in c['labels'])}. Blitz buckets are Lichess "
      f"blitz ratings, classical buckets FIDE ratings: the same number is not the same strength.\n\n")
    w(table(["side", "offset", "log loss", "top-1"],
            [[name, f"{p['offset']:+g}", f(p["loss"]), f(p["top1"], 3)] for name, s in (("blitz", b), ("classical", c))
             for p in s["pooled"]]))
    w("\nBy bucket of the mover's rating, each side at offset 0 and at its own best offset in the grid:\n\n")
    buckets = sorted({x["bucket"] for x in b["by_bucket"]} | {x["bucket"] for x in c["by_bucket"]})
    bb = {x["bucket"]: x for x in b["by_bucket"]}
    cc = {x["bucket"]: x for x in c["by_bucket"]}
    rows = []
    for k in buckets:
        x, y = bb.get(k), cc.get(k)
        rows.append([k,
                     "" if x is None else f"{x['rows']} / {x['games']}",
                     "" if x is None else f(x.get("loss_at_0"), 3),
                     "" if x is None else f"{f(x['best_loss'], 3)} at {x['best_offset']:+g}",
                     "" if y is None else f"{y['rows']} / {y['games']}",
                     "" if y is None else f(y.get("loss_at_0"), 3),
                     "" if y is None else f"{f(y['best_loss'], 3)} at {y['best_offset']:+g}",
                     "" if x is None or y is None else f"{y['best_loss']['est'] - x['best_loss']['est']:+.3f}"])
    w(table(["bucket", "blitz rows / games", "blitz at 0", "blitz at best", "classical rows / games",
             "classical at 0", "classical at best", "gap at best (classical - blitz)"], rows) + "\n")


def provenance_section(w, R):
    w("## Provenance\n\n")
    for name, r in R.items():
        if not r:
            continue
        w(f"**{name}.json**: samples " + ", ".join(f"{k} `{v['path']}`" for k, v in r["provenance"]["samples"].items())
          + "; features " + ", ".join(f"`{v['path']}`" for v in r["provenance"]["features"].values()) + ".\n\n")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m tnh.eval.report", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=REPORTS / "step1")
    p.add_argument("--out", type=Path, default=REPORTS / "step1.md")
    p.add_argument("--note", help="a paragraph put at the top (e.g. that this is a rehearsal)")
    a = p.parse_args(argv)
    names = ["logloss", "sweep", "mistakes", "mistakes_1m", "mistakes_100k_sub15k", "mistakes_glance", "logit",
             "stack", "logit_sub", "stack_sub", "logit_sub30", "logit_sf10k", "breakdown", "book", "theory", "value", "value_wdl", "reference"]
    R = {n: load(a.results, n) for n in names}
    rel = Path(a.results).resolve().relative_to(Path(a.out).resolve().parent) \
        if Path(a.results).resolve().is_relative_to(Path(a.out).resolve().parent) else Path(a.results).resolve()
    parts: list[str] = []
    w = parts.append
    w("# Step 1: how well existing models predict classical moves\n\n")
    w(f"Generated by `uv run python -m tnh.eval.report` on {time.strftime('%Y-%m-%d %H:%M')} from "
      f"`{a.results}`. Missing results: {', '.join(n for n in names if R[n] is None) or 'none'}.\n\n")
    if a.note:
        w(f"> {a.note}\n\n")
    sec_samples(w, R)
    sec_logloss(w, R["logloss"])
    sec_sweep(w, R["sweep"], a.results, rel)
    sec_shift(w, R["mistakes"], R["logit"], a.results, rel)
    sec_breakdown(w, R["breakdown"], a.results, rel)
    sec_book(w, R["book"], a.results, rel)
    sec_theory(w, R["theory"], a.results, rel)
    best = (R["sweep"] or {}).get("compare", {}).get("single_offset")
    refs = [0.0] + ([best] if best is not None and best != 0 else [])
    if R["mistakes"]:
        w("### Ratios by rating bucket\n\n")
        mistake_tables(w, R["mistakes"], ["visibility", "size"], refs)
    sec_mistakes_extra(w, R["mistakes_1m"], "### The mistake classes with deeper labels",
                       "The same classes on a 15,000-row subsample of the test window rows, with Stockfish at 1,000,000 "
                       "nodes as the deep budget.", refs)
    sec_mistakes_extra(w, R["mistakes_100k_sub15k"], "### The same subsample with the 100,000-node labels",
                       "For comparison on the same rows.", refs)
    sec_mistakes_extra(w, R["mistakes_glance"], "### Mistakes visible at a glance",
                       "On a 100,000-row subsample of the test window rows, with Lc0's raw child values (no search) "
                       "and its policy prior.", refs)
    sec_logit(w, R["logit"])
    sec_stack(w, R["stack"])
    sec_logit(w, R["logit_sub"], "## The conditional logit with Lc0's children and policy, on the 100,000-row subsample")
    sec_stack(w, R["stack_sub"], "## The stacks with Lc0's children and policy, on the 100,000-row subsample")
    sec_logit(w, R["logit_sub30"], "## Lc0's 100-node search against Stockfish 10k and Lc0's children, on the 30,000-row subsample")
    if not R["logit"]:
        sec_logit(w, R["logit_sf10k"], "## The conditional logit with Stockfish 10k only (partial, before 100k)")
    sec_value(w, R["value"], a.results, rel)
    sec_value(w, R["value_wdl"], a.results, rel, "## Lc0 with WDLCalibrationElo", "value_wdl.png")
    sec_reference(w, R["reference"])
    provenance_section(w, R)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(parts))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
