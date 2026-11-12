"""How much does one ply of look-ahead with the move model drift from the static result model, and how much of
the look-ahead should a backed-up value keep?

    uv run python -m tnh.eval.lookahead_drift [--runs fast_exact maia600_exact trees_exact combo-trees_exact]

Reads the exact look-ahead runs of ``tnh.eval.lookahead`` (one and two plies over the moves with probability
at least 0.02, renormalised; leaf the cheap result model) and, per run (= move model):

1. **Drift**: the mean of ``E(exact1) - E(static)`` (the mover's own ply), ``E(exact2) - E(exact1)`` (the
   opponent's reply) and ``E(exact2) - E(static)``, in expected score for the root's side to move, overall and
   by rating band, ply range and book; intervals over tournaments; and the observed score minus each
   prediction's mean expected score (calibration in the large: is the drift a bias?). Negative on the mover's ply = the move
   model throws away more than the static model expects; positive on the opponent's ply = the opponent does.
2. **Blend** ``(1 - kappa) static + kappa exact`` (one ply: the backed-up value
   ``s + kappa (sum_a p(a) s(child_a) - s)``): the kappa with the lowest log loss on the game result, with a
   tournament bootstrap interval (the optimum refound in every replicate), the log loss at 0, the optimum and
   1; and the coefficient on ``E(exact) - E(static)`` when the score is regressed on ``E(static)`` and it.
3. **Signal without the drift**: the same with ``exact - static`` centred per rating band (``static + kappa
   (d - mean_band d)``): a kappa above 0 means the look-ahead carries information beyond its bias.

Writes ``<results>/lookahead_drift.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from tnh.eval.lookahead import OUT, POS_TEST
from tnh.paths import REPORTS

KAPPAS = np.round(np.arange(-1.0, 3.0001, 0.05), 2)
BANDS = {"rating": ([0, 2000, 2200, 2400, 9999], ["1800-1999", "2000-2199", "2200-2399", "2400+"]),
         "ply": ([10, 20, 30, 41], ["10-19", "20-29", "30-40"])}


def es(p: np.ndarray) -> np.ndarray:
    return p[:, 0] + 0.5 * p[:, 1]


def blend(s: np.ndarray, d: np.ndarray, k: float) -> np.ndarray:
    w = np.clip(s + k * d, 1e-6, None)
    return w / w.sum(1, keepdims=True)


def main(argv=None) -> None:
    from tnh.eval.common import boot_counts, boot_means, cluster_codes, est, read_sample, write_result
    from tnh.eval.result import losses

    p = argparse.ArgumentParser(prog="python -m tnh.eval.lookahead_drift", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--positions", type=Path, default=POS_TEST)
    p.add_argument("--runs", nargs="+", default=["fast_exact", "maia600_exact", "trees_exact", "combo-trees_exact"])
    p.add_argument("--results", type=Path, default=REPORTS / "step1")
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args(argv)

    test = read_sample(a.positions).with_columns(pl.col("ply").cast(pl.Int64))
    out = {}
    f_ = lambda e: f"{e['est']:+.4f} [{e['lo']:+.4f}, {e['hi']:+.4f}]"   # noqa: E731
    for run in a.runs:
        f = OUT / f"{run}.parquet"
        if not f.exists():
            print(f"{run}: no {f}")
            continue
        d = pl.read_parquet(f).with_columns(pl.col("ply").cast(pl.Int64))
        j = test.join(d, on=["game_id", "ply"], how="inner").sort("game_id", "ply")
        y = np.array([{1.0: 0, 0.5: 1, 0.0: 2}[x] for x in j.get_column("result_mover").to_list()])
        score = np.array([1.0, 0.5, 0.0])[y]
        S, E1, E2 = (np.array(j.get_column(c).to_list()) for c in ("static", "exact1", "exact2"))
        S, E1, E2 = (x / x.sum(1, keepdims=True) for x in (S, E1, E2))
        cl = cluster_codes(j)
        counts = boot_counts(cl.max() + 1, a.boot, a.seed)
        n = len(y)
        mean = (j.get_column("mover_elo").to_numpy() + j.get_column("opp_elo").to_numpy()) / 2
        ply = j.get_column("ply").to_numpy()
        groups = {"all": np.ones(n, bool)}
        for name, vals in (("rating", mean), ("ply", ply)):
            edges, labels = BANDS[name]
            g = np.searchsorted(edges, vals, side="right") - 1
            for i, lab in enumerate(labels):
                groups[f"{name} {lab}"] = g == i
        oob = j.get_column("out_of_book").to_numpy().astype(bool)
        groups["in book"], groups["out of book"] = ~oob, oob

        # 1. drift
        D = np.column_stack([es(E1) - es(S), es(E2) - es(E1), es(E2) - es(S), E1[:, 1] - S[:, 1], E2[:, 1] - S[:, 1],
                             score - es(S), score - es(E1), score - es(E2)])
        names = ["mover ply (e1 - s)", "opponent ply (e2 - e1)", "two plies (e2 - s)", "draw e1 - s", "draw e2 - s",
                 "score - s", "score - e1", "score - e2"]
        drift = {}
        for gname, m in groups.items():
            if m.sum() < 30:
                continue
            bm = boot_means(D[m], np.ones(m.sum()), cl[m], counts)
            drift[gname] = {"n": int(m.sum()), **{nm: est(D[m, c].mean(), bm[:, c]) for c, nm in enumerate(names)}}

        # 2. and 3. blends
        rband = np.searchsorted(BANDS["rating"][0], mean, side="right") - 1
        blends = {}
        for lab, E in (("exact1", E1), ("exact2", E2)):
            dd = E - S
            dc = dd.copy()
            for b in np.unique(rband):
                dc[rband == b] -= dd[rband == b].mean(0)
            for kind, delta in (("raw", dd), ("centred", dc)):
                L = np.column_stack([losses(blend(S, delta, k), y)[0] for k in KAPPAS])
                pt = L.mean(0)
                bm = boot_means(L, np.ones(n), cl, counts)
                kb = KAPPAS[np.argmin(bm, axis=1)]
                i0, i1, io = int(np.where(KAPPAS == 0)[0][0]), int(np.where(KAPPAS == 1)[0][0]), int(np.argmin(pt))
                diff = np.column_stack([L[:, io] - L[:, i0], L[:, i1] - L[:, i0]])
                bd = boot_means(diff, np.ones(n), cl, counts)
                # score on E(static) and the shift, least squares
                shift = (es(S + delta) - es(S))
                A = np.column_stack([np.ones(n), es(S), shift])
                coef = np.linalg.lstsq(A, score, rcond=None)[0]
                bc = []
                for r in range(min(a.boot, 500)):
                    sw = np.sqrt(counts[r][cl])
                    bc.append(np.linalg.lstsq(A * sw[:, None], score * sw, rcond=None)[0][2])
                blends[f"{lab} {kind}"] = {
                    "kappa": float(KAPPAS[io]), "kappa_ci": np.quantile(kb, [0.025, 0.975]).tolist(),
                    "kappa_at_grid_edge_share": float(np.mean((kb == KAPPAS[0]) | (kb == KAPPAS[-1]))),
                    "logloss_k0": float(pt[i0]), "logloss_opt": float(pt[io]), "logloss_k1": float(pt[i1]),
                    "opt_minus_k0": est(diff[:, 0].mean(), bd[:, 0]), "k1_minus_k0": est(diff[:, 1].mean(), bd[:, 1]),
                    "shift_coef": float(coef[2]), "shift_coef_ci": np.quantile(bc, [0.025, 0.975]).tolist(),
                    "static_coef": float(coef[1])}
        out[run] = {"positions": n, "tournaments": int(cl.max() + 1), "drift": drift, "blends": blends,
                    "cost": {k.replace("per_position_", ""): float(d.get_column(k).mean())
                             for k in d.columns if k.startswith("per_position_")}}
        print(f"\n{run}: {n} positions, {cl.max() + 1} tournaments")
        for g, r in drift.items():
            print(f"  {g:>18} n={r['n']:>4}: " + "; ".join(f"{nm} {f_(r[nm])}" for nm in names[:3] + names[5:]), flush=True)
        for k, r in blends.items():
            print(f"  {k:>16}: kappa {r['kappa']:+.2f} [{r['kappa_ci'][0]:+.2f}, {r['kappa_ci'][1]:+.2f}] "
                  f"(edge {r['kappa_at_grid_edge_share']:.0%}); log loss k0 {r['logloss_k0']:.4f} opt {r['logloss_opt']:.4f} "
                  f"k1 {r['logloss_k1']:.4f}; k1-k0 {f_(r['k1_minus_k0'])}; shift coef {r['shift_coef']:+.2f} "
                  f"[{r['shift_coef_ci'][0]:+.2f}, {r['shift_coef_ci'][1]:+.2f}] (static {r['static_coef']:.2f})", flush=True)
    write_result(a.results, "lookahead_drift", {
        "what": "exact one- and two-ply look-ahead (p >= 0.02, renormalised; cheap leaf) against the static cheap result "
                "model: drift per ply, best blend kappa, and the blend with the drift removed per rating band",
        "kappas": KAPPAS.tolist(), **out})


if __name__ == "__main__":
    main()
