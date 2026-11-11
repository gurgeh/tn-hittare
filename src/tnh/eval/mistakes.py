"""Is classical play a pure rating shift of blitz play? Calibration by kind of mistake.

    uv run python -m tnh.eval.mistakes --variants DIR [DIR ...] --shallow DIR --deep DIR \\
        [--sound 0.03] [--visible 0.03] [--sizes 0.10 0.25] [--decided 0.97] [--test S] [--results R]

Every legal move of a test window row is put in a class by the engine
(expected-score loss against the best move, ``L = v_best - v_move``, one search
per budget, the shallow and deep budgets taken as parameters):

- by visibility: ``sound`` (``L_deep <= sound``); ``visible``, a mistake at the deep
  budget that the shallow budget also sees (``L_shallow > visible``); ``deep_only``,
  a mistake at the deep budget that the shallow budget does not see;
- by size, at the deep budget: ``sound``, ``small`` (up to ``sizes[0]``), ``medium``
  (up to ``sizes[1]``), ``large``.

Then, per Elo offset of Maia-3 and per 100-point bucket of the mover's rating:
the probability mass Maia-3 puts on each class, summed over positions (raw
model probabilities, no uniform mixture), against how often the played move was
in that class. For each class, the offset at which the two meet is found by
linear interpolation between grid offsets, with a bootstrap interval over
tournaments; ``null`` when they do not meet inside the grid (``side`` says
whether the model over- or under-predicts the class at every offset).

A pure rating shift predicts one offset that calibrates every class at once.
If quiet mistakes are what classical players keep making, the deep-only and
small classes are calibrated at a lower offset than the visible ones.

Rows dropped, counted: positions where either budget left a move without a value
(the class of that move is unknown), and decided positions, where the deep
search's best value is above ``decided`` or below ``1 - decided``: there the
expected score saturates, every move is "sound", and the position says nothing
about the kind of mistake. The conclusion is recomputed under a grid of other
thresholds (``robustness`` in the output).

Writes ``<results>/mistakes.json``.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import polars as pl

from tnh.eval.common import (Ragged, add_common, attach, boot_counts, boot_means, cluster_codes, label, parse_variant,
                             ragged, read_sample, provenance, same_moves, variant_name, window_rows, write_result)
from tnh.eval.logit import nanmax_segments

VIS = ["sound", "visible", "deep_only"]
SIZE = ["sound", "small", "medium", "large"]


def classes(Ls: np.ndarray, Ld: np.ndarray, sound: float, visible: float, sizes: tuple[float, float]):
    vis = np.where(Ld <= sound, 0, np.where(Ls > visible, 1, 2))
    size = np.where(Ld <= sound, 0, np.where(Ld <= sizes[0], 1, np.where(Ld <= sizes[1], 2, 3)))
    return vis, size


def crossing(offs: np.ndarray, diff: np.ndarray) -> tuple[float, str]:
    """Offset where diff (pred - obs) crosses zero, by linear interpolation; the first crossing
    from below. ('', offset) when inside, else ('over'|'under', nan)."""
    s = np.sign(diff)
    for i in range(len(offs) - 1):
        if s[i] == 0:
            return float(offs[i]), ""
        if s[i] != s[i + 1]:
            t = diff[i] / (diff[i] - diff[i + 1])
            return float(offs[i] + t * (offs[i + 1] - offs[i])), ""
    if s[-1] == 0:
        return float(offs[-1]), ""
    return float("nan"), "over" if np.all(diff > 0) else "under"


def class_mass(r: Ragged, p: np.ndarray, cls: np.ndarray, n_cls: int) -> np.ndarray:
    """rows x n_cls: probability mass per class."""
    out = np.zeros((len(r.offsets) - 1, n_cls))
    rows = r.row_of
    for c in range(n_cls):
        out[:, c] = np.bincount(rows, weights=np.where(cls == c, p, 0.0), minlength=len(out))
    return out


def analyse(offs, P, y_cls, cls, r, w, cl, counts, g, nb, n_cls, names):
    """P: list over variants of flat probabilities. Returns per-class, per-bucket and pooled results."""
    obs = np.zeros((len(w), n_cls))
    obs[np.arange(len(w)), y_cls] = 1.0
    pred = [class_mass(r, p, cls, n_cls) for p in P]                 # V x rows x C
    X = np.column_stack([obs] + pred)                                 # rows x (C + V*C)
    V = len(P)
    out = {"pooled": {}, "by_bucket": []}

    def block(xm, bm):
        res = {}
        for c in range(n_cls):
            o = xm[c]
            pr = np.array([xm[n_cls + v * n_cls + c] for v in range(V)])
            point, side = crossing(offs, pr - o)
            bcross = []
            for rep in range(bm.shape[0]):
                ob = bm[rep, c]
                prb = np.array([bm[rep, n_cls + v * n_cls + c] for v in range(V)])
                bcross.append(crossing(offs, prb - ob)[0])
            bcross = np.array(bcross)
            res[names[c]] = {"observed": float(o), "predicted": pr.tolist(), "ratio": (pr / o).tolist() if o else None,
                             "calibrated_offset": None if np.isnan(point) else point, "side": side,
                             "lo": float(np.nanquantile(bcross, 0.025)) if np.isfinite(bcross).any() else None,
                             "hi": float(np.nanquantile(bcross, 0.975)) if np.isfinite(bcross).any() else None,
                             "boot_share_inside": float(np.isfinite(bcross).mean())}
        return res

    W = w.sum()
    xm = (w[:, None] * X).sum(0) / W
    bp = boot_means(X, w, cl, counts)
    out["pooled"] = block(xm, bp)
    bb = boot_means(X, w, cl, counts, groups=g, n_groups=nb)
    for k in range(nb):
        m = g == k
        xk = (w[m, None] * X[m]).sum(0) / w[m].sum()
        out["by_bucket"].append({"rows": int(m.sum()), **block(xk, bb[:, k, :])})
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m tnh.eval.mistakes", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", type=Path, nargs="+", required=True, help="Maia-3 offset variants")
    p.add_argument("--shallow", type=Path, required=True, help="engine features, the short search")
    p.add_argument("--deep", type=Path, required=True, help="engine features, the long search")
    p.add_argument("--sound", type=float, default=0.03)
    p.add_argument("--visible", type=float, default=0.03)
    p.add_argument("--sizes", type=float, nargs=2, default=[0.10, 0.25])
    p.add_argument("--decided", type=float, default=0.97)
    p.add_argument("--robust-boot", type=int, default=20, help="replicates per threshold set in the robustness grid (only point estimates are reported)")
    add_common(p, fit=False)
    a = p.parse_args(argv)
    test = window_rows(read_sample(a.test))
    if set(test.get_column("split").unique()) != {"test"}:
        raise SystemExit("not a test sample")

    sh = attach(test, a.shallow, ["value"])
    dp = attach(test, a.deep, ["value"])
    same_moves(sh, dp, "shallow against deep")
    vs = []
    for d in a.variants:
        v = parse_variant(variant_name(d))
        if v["kind"] != "offset" or v["mover"] != v["opp"]:
            print(f"ignoring {d}: not a both-sides offset")
            continue
        j = attach(test, d, ["logp"])
        same_moves(sh, j, f"{d} against the engine")
        vs.append((v["mover"], d, np.exp(ragged(j, "logp").values)))
    vs.sort(key=lambda x: x[0])
    offs = np.array([x[0] for x in vs])

    r = ragged(dp, "value")
    vd, vsh = r.values, ragged(sh, "value").values
    nan_row = np.add.reduceat((np.isnan(vd) | np.isnan(vsh)).astype(int), r.offsets[:-1]) > 0
    bd = nanmax_segments(r, vd)
    bs = nanmax_segments(r, vsh)

    def run(sound, visible, sizes, decided, B, seed):
        dec = (bd > decided) | (bd < 1 - decided)
        keep = ~nan_row & ~dec
        mk = np.repeat(keep, r.lengths)
        lens = r.lengths[keep]
        off = np.zeros(keep.sum() + 1, dtype=np.int64)
        np.cumsum(lens, out=off[1:])
        rk = Ragged(np.empty(0), off)
        Ld = (np.repeat(bd, r.lengths) - vd)[mk]
        Ls = (np.repeat(bs, r.lengths) - vsh)[mk]
        vis, size = classes(Ls, Ld, sound, visible, sizes)
        idx = dp.get_column("_idx").to_numpy()[keep]
        played = off[:-1] + idx
        sub = test.filter(pl.Series(keep))
        w = sub.get_column("weight").to_numpy()
        cl = cluster_codes(sub)
        counts = boot_counts(cl.max() + 1, B, seed)
        b = sub.get_column("bucket").to_numpy()
        ub = np.unique(b)
        g = np.searchsorted(ub, b)
        P = [x[2][mk] for x in vs]
        res = {"rows": int(keep.sum()), "dropped_missing_value": int(nan_row.sum()),
               "dropped_decided": int((dec & ~nan_row).sum()), "buckets": ub.tolist(),
               "visibility": analyse(offs, P, vis[played], vis, rk, w, cl, counts, g, len(ub), 3, VIS),
               "size": analyse(offs, P, size[played], size, rk, w, cl, counts, g, len(ub), 4, SIZE)}
        return res

    main_res = run(a.sound, a.visible, tuple(a.sizes), a.decided, a.boot, a.seed)
    print(f"rows {main_res['rows']}, dropped: {main_res['dropped_missing_value']} with a move without a value, "
          f"{main_res['dropped_decided']} decided")
    for kind in ("visibility", "size"):
        for c, x in main_res[kind]["pooled"].items():
            co = x["calibrated_offset"]
            print(f"  {kind:>10} {c:>9}: observed {x['observed']:.4f}; calibrated at "
                  f"{'none (' + x['side'] + ')' if co is None else f'{co:+.0f}'} "
                  f"[{x['lo']}, {x['hi']}]")
    robust = []
    for sound, decided in itertools.product([0.02, 0.03, 0.05], [0.95, 0.97, 0.99]):
        for scale in (0.67, 1.0, 1.5):
            sizes = (max(sound * 1.5, a.sizes[0] * scale), a.sizes[1] * scale)
            rr = run(sound, sound, sizes, decided, a.robust_boot, a.seed)
            robust.append({"sound": sound, "visible": sound, "sizes": list(sizes), "decided": decided,
                           "rows": rr["rows"],
                           "visibility": {c: x["calibrated_offset"] for c, x in rr["visibility"]["pooled"].items()},
                           "visibility_side": {c: x["side"] for c, x in rr["visibility"]["pooled"].items()},
                           "size": {c: x["calibrated_offset"] for c, x in rr["size"]["pooled"].items()},
                           "size_side": {c: x["side"] for c, x in rr["size"]["pooled"].items()}})
    write_result(a.results, "mistakes", {
        "what": "Maia-3 probability mass per kind of mistake against the played moves, per offset; test window rows",
        "thresholds": {"sound": a.sound, "visible": a.visible, "sizes": a.sizes, "decided": a.decided},
        "shallow": label(a.shallow), "deep": label(a.deep), "offsets": offs.tolist(),
        "variants": [label(x[1]) for x in vs], "boot": a.boot, **main_res, "robustness": robust,
        "provenance": provenance({"test": a.test}, [x[1] for x in vs] + [a.shallow, a.deep])})


if __name__ == "__main__":
    main()
