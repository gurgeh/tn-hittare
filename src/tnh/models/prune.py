"""Pruned children: evaluate Lc0's raw value only for the moves a cheap signal says matter.

    uv run python -m tnh.models.prune analyse [--fit S] [--test S] [--boot 1000]

Offline, on step 1's feature files. A rule keeps, per position, the moves whose
`max(p_maia+0, p_maia+600, p_lc0policy)` is at least a threshold (`p0.5` ... `p5`: in
percent), or the `k` moves with the largest such probability (`top5`), or both
(`p1|top3`: union). The most probable move is always kept, so every position has at
least one evaluated child.

The child columns with only some children evaluated:

- `lc0ch_eloss`, `lc0ch_cploss`: as in `tnh.eval.logit.engine_columns`, but relative to
  the best **evaluated** child (an unevaluated move cannot be the reference, its value is
  unknown);
- an unevaluated move gets `lc0ch_unev` = 1 and both losses filled (`--fill zero`: 0, the
  indicator's coefficient carries the whole penalty; `--fill worst`: the largest loss
  among the position's evaluated children, as the step 1 code does for an engine's NaN).

With every child kept the columns are exactly step 1's. The logit is refitted per rule on
the fitting subsample's window rows and reported on the test subsample with step 1's
bootstrap seed, like `tnh.models.combo fit`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tnh.models.combo import ENGINE_SOURCES, FIT_SAMPLE, LOGP_SOURCES, TEST_SAMPLE

MATE_CP = 10_000


def parse_rule(rule: str) -> tuple[float | None, int | None]:
    p = k = None
    for part in rule.split("|"):
        if part.startswith("p"):
            p = float(part[1:]) / 100
        elif part.startswith("top"):
            k = int(part[3:])
        elif part == "all":
            return 0.0, None
        else:
            raise ValueError(f"rule {rule!r}: parts are pN (percent) or topK")
    return p, k


def keep_mask(prob: np.ndarray, offsets: np.ndarray, rule: str) -> np.ndarray:
    """Per move: kept by `rule`, from `prob`, the max of the cheap models' probabilities."""
    p, k = parse_rule(rule)
    keep = np.zeros(len(prob), dtype=bool)
    if p is not None:
        keep |= prob >= p
    n = np.diff(offsets)
    row = np.repeat(np.arange(len(n)), n)
    # rank within the position, 0 = most probable
    order = np.lexsort((-prob, row))
    rank = np.empty(len(prob), dtype=np.int64)
    starts = np.repeat(offsets[:-1], n)
    rank[order] = np.arange(len(prob)) - starts
    keep |= rank == 0
    if k is not None:
        keep |= rank < k
    return keep


def pruned_columns(value: np.ndarray, eff_cp: np.ndarray, offsets: np.ndarray, keep: np.ndarray,
                   fill: str) -> dict[str, np.ndarray]:
    """`lc0ch_eloss`, `lc0ch_cploss`, `lc0ch_unev` with only the kept children evaluated."""
    n = np.diff(offsets)
    v = np.where(keep, value, -np.inf)
    best = np.maximum.reduceat(v, offsets[:-1])
    c = np.where(keep, eff_cp, -np.inf)
    cbest = np.maximum.reduceat(c, offsets[:-1])
    eloss = np.repeat(best, n) - value
    closs = np.log1p(np.maximum(np.repeat(cbest, n) - eff_cp, 0.0) / 100.0)
    if fill == "zero":
        eloss = np.where(keep, eloss, 0.0)
        closs = np.where(keep, closs, 0.0)
    elif fill == "worst":
        we = np.maximum.reduceat(np.where(keep, eloss, -np.inf), offsets[:-1])
        wc = np.maximum.reduceat(np.where(keep, closs, -np.inf), offsets[:-1])
        eloss = np.where(keep, eloss, np.repeat(we, n))
        closs = np.where(keep, closs, np.repeat(wc, n))
    else:
        raise ValueError(f"fill {fill!r}")
    return {"lc0ch_eloss": eloss, "lc0ch_cploss": closs, "lc0ch_unev": (~keep).astype(np.float64)}


def _load(sample: Path):
    """Data with m0, mbest, lc0pol and the raw child value and effective centipawns per move."""
    from tnh.eval.common import attach, ragged, read_sample, window_rows
    from tnh.eval.logit import build

    s = window_rows(read_sample(sample))
    D, _ = build(s, LOGP_SOURCES, {})
    j = attach(s, ENGINE_SOURCES["lc0ch"], ["value", "cp", "mate"])
    rv = ragged(j, "value")
    if not np.array_equal(rv.offsets, D.offsets):
        raise ValueError("children and move models disagree on the move lists")
    cp, mate = ragged(j, "cp").values, ragged(j, "mate").values
    if np.isnan(rv.values).any():
        raise ValueError("a child without a value in the feature files")
    eff = np.where(mate != 0, np.sign(mate) * (MATE_CP - 100 * np.abs(mate)), cp)
    prob = np.exp(D.X[:, [D.names.index(c) for c in ("m0", "mbest", "lc0pol")]]).max(1)
    return D, rv.values, eff, prob


def cmd_analyse(a) -> None:
    from tnh.eval.common import boot_counts, boot_means, est, fit_eps, mix_logp, wmean
    from tnh.eval.logit import fit, scores
    from tnh.models.combo import _test_loss

    t0 = time.perf_counter()
    Df, vf, cf, pf = _load(a.fit)
    Dt, vt, ct, pt = _load(a.test)
    print(f"fit rows {len(Df.w)}, test rows {len(Dt.w)}, moves per position {np.mean(Df.n):.1f} "
          f"({time.perf_counter() - t0:.0f} s)", flush=True)
    base = ["m0", "mbest", "lc0pol"]
    full_best_t = None
    out = []
    for rule in a.rules:
        for fill in a.fills:
            kf, kt = keep_mask(pf, Df.offsets, rule), keep_mask(pt, Dt.offsets, rule)
            Xf = Df.subset(base).with_columns(pruned_columns(vf, cf, Df.offsets, kf, fill))
            Xt = Dt.subset(base).with_columns(pruned_columns(vt, ct, Dt.offsets, kt, fill))
            cols = list(Xf.names) if rule != "all" else base + ["lc0ch_eloss", "lc0ch_cploss"]
            Xf, Xt = Xf.subset(cols), Xt.subset(cols)
            beta, it = fit(Xf)
            lp, _ = scores(Xf, beta)
            eps = fit_eps(lp, Df.n, Df.w)
            test = _test_loss(Xt, cols, beta, eps, a.boot)
            lp_t, _ = scores(Xt, beta)
            loss_t = -mix_logp(lp_t, Dt.n, eps)
            if rule == "all":
                loss_all = loss_t
            d = loss_t - loss_all
            bd = boot_means(d[:, None], Dt.w, Dt.clusters, boot_counts(Dt.clusters.max() + 1, a.boot, 7))
            diff = est(wmean(d, Dt.w), bd[:, 0])
            unev_played = ~kt[Dt.y]
            per_pos = np.add.reduceat(kt.astype(np.int64), Dt.offsets[:-1])
            played_out = float(np.average(~kt[Dt.y], weights=Dt.w))
            if full_best_t is None:
                v = vt.copy()
                best = np.maximum.reduceat(v, Dt.offsets[:-1])
                full_best_t = np.repeat(best, Dt.n.astype(np.int64)) == v   # ties: all best moves
            # the engine-best child is outside the kept set when no best-valued move is kept
            best_kept = np.maximum.reduceat((full_best_t & kt).astype(np.int8), Dt.offsets[:-1]) > 0
            r = {"rule": rule, "fill": fill, "columns": cols, "beta": beta.tolist(), "eps": eps, "iterations": it,
                 "children_mean": float(per_pos.mean()), "children_median": float(np.median(per_pos)),
                 "children_p95": float(np.quantile(per_pos, 0.95)), "share_of_children": float(kt.mean()),
                 "played_unevaluated": played_out, "engine_best_unevaluated": float(np.average(~best_kept, weights=Dt.w)),
                 "loss_played_unevaluated": {"mean": float(np.mean(loss_t[unev_played])),
                                             "max": float(np.max(loss_t[unev_played]))} if unev_played.any() else None,
                 "loss_played_evaluated_mean": float(np.mean(loss_t[~unev_played])),
                 "test_mixed": test, "minus_all": diff}
            out.append(r)
            print(f"{rule:>10} {fill:>5}: children mean {r['children_mean']:5.2f} median {r['children_median']:4.0f} "
                  f"p95 {r['children_p95']:4.0f}; played unevaluated {played_out:.3f}, best unevaluated "
                  f"{r['engine_best_unevaluated']:.3f}; test {test['est']:.4f} [{test['lo']:.4f}, {test['hi']:.4f}], "
                  f"minus all {diff['est']:+.4f} [{diff['lo']:+.4f}, {diff['hi']:+.4f}]; loss of unevaluated played "
                  f"{r['loss_played_unevaluated']}; "
                  f"beta {np.round(beta, 3).tolist()}", flush=True)
            if rule == "all":
                break
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"fit": str(a.fit), "test": str(a.test), "boot": a.boot, "rules": out}, indent=1) + "\n")
    print(f"wrote {a.out}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m tnh.models.prune", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("analyse")
    sp.add_argument("--fit", type=Path, default=FIT_SAMPLE)
    sp.add_argument("--test", type=Path, default=TEST_SAMPLE)
    sp.add_argument("--boot", type=int, default=1000)
    sp.add_argument("--rules", nargs="+", default=["all", "p0.5", "p1", "p2", "p5", "top3", "top5", "top8", "top12",
                                                   "p2|top3", "p1|top5"])
    sp.add_argument("--fills", nargs="+", default=["zero", "worst"])
    sp.add_argument("--out", type=Path,
                    default=Path.home() / "out" / "tn-hittare" / "scratch" / "models" / "prune" / "prune.json")
    sp.set_defaults(fn=cmd_analyse)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
