"""How much does one ply of look-ahead with the move model drift from the static result model, and how much of
the look-ahead should a backed-up value keep?

    uv run python -m tnh.eval.lookahead_drift [--runs fast_exact maia600_exact trees_exact combo-trees_exact]
    uv run python -m tnh.eval.lookahead_drift prepared [--run trees_full]   # Prepared (lambda) from the per-child run

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


def _kappa(S, E, y, cl, counts) -> dict:
    """Blend (1 - kappa) S + kappa E: the log-loss optimum with a tournament bootstrap interval, the losses at 0,
    the optimum and 1, and the paired difference of 1 against 0."""
    from tnh.eval.common import boot_means, est
    from tnh.eval.result import losses

    n = len(y)
    L = np.column_stack([losses(blend(S, E - S, k), y)[0] for k in KAPPAS])
    pt = L.mean(0)
    bm = boot_means(L, np.ones(n), cl, counts)
    kb = KAPPAS[np.argmin(bm, axis=1)]
    i0, i1, io = int(np.where(KAPPAS == 0)[0][0]), int(np.where(KAPPAS == 1)[0][0]), int(np.argmin(pt))
    d = L[:, i1] - L[:, i0]
    bd = boot_means(d, np.ones(n), cl, counts)
    return {"kappa": float(KAPPAS[io]), "kappa_ci": np.quantile(kb, [0.025, 0.975]).tolist(),
            "logloss_k0": float(pt[i0]), "logloss_opt": float(pt[io]), "logloss_k1": float(pt[i1]),
            "k1_minus_k0": est(d.mean(), bd[:, 0])}


LAMBDAS = np.round(np.arange(-0.5, 1.5001, 0.01), 2)


def cmd_prepared(argv=None) -> None:
    """``Prepared`` (lambda) fitted on the per-child run (``lookahead run --full-exact``): a share lambda of the
    side to move's probability moved onto its best move (by the static value) makes the one-ply value
    ``e1(lambda) = (1 - lambda) e1 + lambda s(best child)``; lambda* sets the mean of ``e1(lambda) - s`` to 0.

    - ply 1: the root mover's own move (all legal moves, and truncated at p >= 0.02 as the earlier runs);
    - ply 2 alone: the reply at each kept child (all replies), lambda* from the opponent's drift and loss;
    - both plies: lambda at the reply and at the root (best child by its lambda-backed value), lambda* where
      the mean of ``e2(lambda) - s`` crosses 0;
    then the blend ``(1 - kappa) s + kappa e(lambda*)`` against the game result. Writes ``<results>/lookahead_prepared.json``."""
    from tnh.eval.common import boot_counts, boot_means, cluster_codes, est, read_sample, write_result

    p = argparse.ArgumentParser(prog="python -m tnh.eval.lookahead_drift prepared")
    p.add_argument("--positions", type=Path, default=POS_TEST)
    p.add_argument("--run", default="trees_full")
    p.add_argument("--p-min", type=float, default=0.02)
    p.add_argument("--results", type=Path, default=REPORTS / "step1")
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args(argv)

    test = read_sample(a.positions).with_columns(pl.col("ply").cast(pl.Int64))
    d = pl.read_parquet(OUT / f"{a.run}.parquet").with_columns(pl.col("ply").cast(pl.Int64))
    j = test.join(d, on=["game_id", "ply"], how="inner").sort("game_id", "ply")
    n = j.height
    y = np.array([{1.0: 0, 0.5: 1, 0.0: 2}[x] for x in j.get_column("result_mover").to_list()])
    S = np.array(j.get_column("static").to_list())
    S /= S.sum(1, keepdims=True)
    cl = cluster_codes(j)
    counts = boot_counts(cl.max() + 1, a.boot, a.seed)
    mean = (j.get_column("mover_elo").to_numpy() + j.get_column("opp_elo").to_numpy()) / 2
    ply = j.get_column("ply").to_numpy()
    groups = {"all": np.ones(n, bool)}
    for name, vals in (("rating", mean), ("ply", ply)):
        edges, labels = BANDS[name]
        g = np.searchsorted(edges, vals, side="right") - 1
        for i, lab in enumerate(labels):
            groups[f"{name} {lab}"] = g == i
    rband = np.searchsorted(BANDS["rating"][0], mean, side="right") - 1

    cp, cw, ck = j.get_column("c_p").to_list(), j.get_column("c_wdl").to_list(), j.get_column("c_kept").to_list()
    rp, rw = j.get_column("r_p").to_list(), j.get_column("r_wdl").to_list()
    E1f, E1t, B1 = (np.zeros((n, 3)) for _ in range(3))
    tail = np.zeros(n)
    opp_drift, opp_loss = np.zeros(n), np.zeros(n)      # ply 2, p-weighted over the kept children, root's side
    G = len(LAMBDAS)
    E2 = {"full": np.zeros((n, G, 3)), "truncated": np.zeros((n, G, 3))}
    for i in range(n):
        pi = np.array(cp[i], float)
        pi /= pi.sum()
        W = np.array(cw[i], float)
        kept = np.array(ck[i], bool)
        E1f[i] = pi @ W
        pk = pi[kept] / pi[kept].sum()
        E1t[i] = pk @ W[kept]
        tail[i] = 1 - pi[kept].sum()
        B1[i] = W[np.argmax(es(W))]
        # per kept child: reply expectation (full and truncated) and the opponent's best reply (lowest for the root)
        Vk = {"full": W.copy(), "truncated": W.copy()}
        M, Rb = {"full": W[kept].copy(), "truncated": W[kept].copy()}, W[kept].copy()
        for jj, (q, R) in enumerate(zip(rp[i], rw[i])):
            if not len(q):
                continue                                 # the child ends the game: its exact result
            q, R = np.array(q, float), np.array(R, float)
            q /= q.sum()
            M["full"][jj] = q @ R
            qt = np.where(q >= a.p_min, q, 0.0)
            M["truncated"][jj] = (qt / qt.sum()) @ R if qt.sum() > 0 else R[np.argmax(q)]
            Rb[jj] = R[np.argmin(es(R))]
        Wk = W[kept]
        opp_drift[i] = pk @ (es(M["full"]) - es(Wk))
        opp_loss[i] = pk @ (es(Rb) - es(M["full"]))
        for kind in ("full", "truncated"):
            # lambda at the reply: V_c = M + lambda (Rb - M) for the kept children; the others keep their leaf
            Vc = np.repeat(W[None], G, 0)                                       # G x k x 3
            Vc[:, kept] = M[kind][None] + LAMBDAS[:, None, None] * (Rb - M[kind])[None]
            w = pi if kind == "full" else np.where(kept, pi, 0.0) / pi[kept].sum()
            ev = np.einsum("k,gkc->gc", w, Vc)
            cand = Vc if kind == "full" else Vc[:, kept]
            best = cand[np.arange(G), np.argmax(cand[:, :, 0] + 0.5 * cand[:, :, 1], axis=1)]
            E2[kind][i] = ev + LAMBDAS[:, None] * (best - ev)

    f_ = lambda e: f"{e['est']:+.4f} [{e['lo']:+.4f}, {e['hi']:+.4f}]"   # noqa: E731
    res = {"positions": n, "tournaments": int(cl.max() + 1), "tail_mass_mean": float(tail.mean())}

    def ratio(num, den, m):
        """lambda* = -mean(num) / mean(den) on rows m, with a tournament bootstrap interval."""
        bm = boot_means(np.column_stack([num[m], den[m]]), np.ones(m.sum()), cl[m], counts)
        r = -bm[:, 0] / bm[:, 1]
        return {"est": float(-num[m].mean() / den[m].mean()), "lo": float(np.quantile(r, 0.025)),
                "hi": float(np.quantile(r, 0.975))}, est(num[m].mean(), bm[:, 0]), est(den[m].mean(), bm[:, 1])

    one = {}
    for kind, E1 in (("full", E1f), ("truncated", E1t)):
        drift, loss = es(E1) - es(S), es(B1) - es(E1)
        rows = {}
        for gname, m in groups.items():
            lam, dr, lo = ratio(drift, loss, m)
            rows[gname] = {"n": int(m.sum()), "lambda": lam, "drift": dr, "expected_loss": lo,
                           "tail_mass": float(tail[m].mean())}
        one[kind] = rows
    two_alone = {}
    for gname, m in groups.items():
        lam, dr, lo = ratio(opp_drift, opp_loss, m)
        two_alone[gname] = {"n": int(m.sum()), "lambda": lam, "drift": dr, "expected_loss": lo}
    both = {}
    for kind in ("full", "truncated"):
        D = es(E2[kind].reshape(n * G, 3)).reshape(n, G) - es(S)[:, None]
        rows = {}
        for gname, m in groups.items():
            bm = boot_means(D[m], np.ones(m.sum()), cl[m], counts)
            def cross(v):
                s_ = np.sign(v)
                k = np.where(s_[:-1] != s_[1:])[0]
                if not len(k):
                    return float("nan")
                k = k[0]
                return float(LAMBDAS[k] - v[k] * (LAMBDAS[k + 1] - LAMBDAS[k]) / (v[k + 1] - v[k]))
            cb = np.array([cross(v) for v in bm])
            rows[gname] = {"n": int(m.sum()), "lambda": {"est": cross(D[m].mean(0)), "lo": float(np.nanquantile(cb, 0.025)),
                                                          "hi": float(np.nanquantile(cb, 0.975)),
                                                          "no_crossing_share": float(np.isnan(cb).mean())},
                           "drift_at_0": est(D[m, LAMBDAS.tolist().index(0.0)].mean(),
                                             bm[:, LAMBDAS.tolist().index(0.0)])}
        both[kind] = rows
    res.update({"one_ply": one, "reply_ply_alone": two_alone, "both_plies": both})

    # the blends: static against the look-ahead without and with lambda*
    lam_all = one["full"]["all"]["lambda"]["est"]
    lam_band = np.array([one["full"][f"rating {lab}"]["lambda"]["est"] for lab in BANDS["rating"][1]])
    lam_row = lam_band[rband]
    gi = lambda lam: int(np.argmin(np.abs(LAMBDAS - lam)))   # noqa: E731
    e1_lam_band = E1f + lam_row[:, None] * (B1 - E1f)
    blends = {
        "e1 truncated, lambda 0": _kappa(S, E1t, y, cl, counts),
        "e1 full, lambda 0": _kappa(S, E1f, y, cl, counts),
        f"e1 full, lambda* {lam_all:.2f}": _kappa(S, E1f + lam_all * (B1 - E1f), y, cl, counts),
        "e1 full, lambda* per rating band": _kappa(S, e1_lam_band, y, cl, counts),
        "e2 full, lambda 0": _kappa(S, E2["full"][:, gi(0.0)], y, cl, counts),
        f"e2 full, lambda {lam_all:.2f} at both plies": _kappa(S, E2["full"][:, gi(lam_all)], y, cl, counts),
    }
    lam2 = both["full"]["all"]["lambda"]["est"]
    if np.isfinite(lam2):
        blends[f"e2 full, lambda*_2 {lam2:.2f} at both plies"] = _kappa(S, E2["full"][:, gi(lam2)], y, cl, counts)
    res["blends"] = blends
    res["lambda_by_band_used"] = dict(zip(BANDS["rating"][1], lam_band.tolist()))

    print(f"{a.run}: {n} positions, {cl.max() + 1} tournaments; mean tail mass below p {a.p_min}: {tail.mean():.3f}")
    for kind, rows in one.items():
        print(f"\none ply, {kind}: lambda* | drift | expected loss | tail")
        for g, r in rows.items():
            print(f"  {g:>18} n={r['n']:>4}: {r['lambda']['est']:+.2f} [{r['lambda']['lo']:+.2f}, {r['lambda']['hi']:+.2f}] | "
                  f"{f_(r['drift'])} | {f_(r['expected_loss'])} | {r['tail_mass']:.3f}")
    print("\nreply ply alone (opponent, root's side): lambda* | drift | expected loss")
    for g, r in two_alone.items():
        print(f"  {g:>18}: {r['lambda']['est']:+.2f} [{r['lambda']['lo']:+.2f}, {r['lambda']['hi']:+.2f}] | "
              f"{f_(r['drift'])} | {f_(r['expected_loss'])}")
    for kind, rows in both.items():
        print(f"\nboth plies, {kind}: lambda* where mean(e2(lambda) - s) = 0 | drift at lambda 0")
        for g, r in rows.items():
            L = r["lambda"]
            print(f"  {g:>18}: {L['est']:+.2f} [{L['lo']:+.2f}, {L['hi']:+.2f}] (no crossing {L['no_crossing_share']:.0%}) | "
                  f"{f_(r['drift_at_0'])}")
    print("\nblends (1 - kappa) static + kappa look-ahead:")
    for k, r in blends.items():
        print(f"  {k:>36}: kappa {r['kappa']:+.2f} [{r['kappa_ci'][0]:+.2f}, {r['kappa_ci'][1]:+.2f}]; log loss "
              f"k0 {r['logloss_k0']:.4f} opt {r['logloss_opt']:.4f} k1 {r['logloss_k1']:.4f}; k1-k0 {f_(r['k1_minus_k0'])}")
    write_result(a.results, "lookahead_prepared", {
        "what": "Prepared (lambda): share of the side to move's probability moved onto its best move, fitted so the "
                "look-ahead has no mean drift against the static cheap result model; PolicyTrees, cheap leaf",
        "run": a.run, "p_min": a.p_min, **res})


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "prepared":
        cmd_prepared(sys.argv[2:])
    else:
        main()
