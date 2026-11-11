"""Is play just out of book still better than play further on? Engine mistakes by plies since leaving book.

    uv run python -m tnh.eval.book --variants DIR [DIR ...] --shallow DIR --deep DIR \\
        [--book-count 5] [--ref-offset 0] [--test S] [--fit S] [--results R]

On the window rows of the main test sample, where Stockfish values exist, by
plies since the game left book under ``book_count`` (the sample's
``exit_ply_<book_count>``: the exit ply, +1, +2, +3-5, +6-10, later), overall and
within ply bands (``--bands``; early positions are easier in themselves, so a
row just out of book is compared with a later one at the same stage of the game):

- **deep loss**: mean expected-score loss of the played move at the deep budget;
  share of played moves per class of ``tnh.eval.mistakes`` (sound / visible / deep
  only; small / medium / large);
- **observed over expected**: the share of mistakes (and visible, deep-only) divided
  by Maia-3's mass on them at ``--ref-offset``, and the mean loss over Maia-3's
  expected loss. Maia-3 has memorised openings from its own games, so this does not
  take out all of a position's ease;
- the Maia-3 offset that minimises the mixed log loss in the cell (free: the sweep
  variants exist on these rows), with log loss and top-1 at ``--ref-offset``.

Which offset fits at which stage and commonness, without Stockfish and on its own
sample, is ``tnh.eval.theory``.

Dropped and counted: rows where an engine budget left a move without a value, and
decided positions (deep best value outside ``[1 - decided, decided]``). Intervals:
bootstrap over tournaments. Writes ``<results>/book.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from tnh.eval.common import (add_common, attach, boot_counts, boot_means, check_splits, cluster_codes,
                             fit_eps, label, mix_logp, parse_variant, provenance, ragged, read_sample, same_moves,
                             variant_name, window_rows, write_result)
from tnh.eval.logit import nanmax_segments
from tnh.eval.mistakes import classes
from tnh.eval.sweep import vertex

REL_LABELS = ["exit", "+1", "+2", "+3-5", "+6-10", ">+10"]
OBS = ["deep_loss", "sound", "visible", "deep_only", "small", "medium", "large"]
PRED = ["p_mistake", "p_visible", "p_deep_only", "p_loss", "logloss", "top1"]


def rel_group(rel: np.ndarray) -> np.ndarray:
    if (rel < 0).any():
        raise ValueError("a window row before its game's exit ply: the sample's exit ply and window disagree")
    return np.select([rel == 0, rel == 1, rel == 2, rel <= 5, rel <= 10], [0, 1, 2, 3, 4], 5)


def parse_bands(xs: list[str]) -> list[tuple[int, int]]:
    out = []
    for x in xs:
        lo, hi = x.split("-")
        out.append((int(lo), int(hi)))
    return out


def summarize_cells(X: np.ndarray, w, cl, counts, codes: np.ndarray, n_cells: int, offs: np.ndarray, V: int,
                    min_rows: int) -> list[dict | None]:
    """Per cell: matched offset with interval, and every observed and predicted quantity with intervals."""
    bm = boot_means(X, w, cl, counts, groups=codes, n_groups=n_cells)        # B x cells x cols
    out = []
    for c in range(n_cells):
        m = codes == c
        n = int(m.sum())
        if n == 0:
            out.append(None)
            continue
        cell = {"rows": n}
        if n < min_rows:
            out.append(cell)
            continue
        pt = (w[m, None] * X[m]).sum(0) / w[m].sum()
        b = bm[:, c, :]
        off, edge = vertex(offs, pt[:V])
        boff = np.array([vertex(offs, b[r, :V])[0] for r in range(b.shape[0])])
        cell["offset"] = {"est": off, "lo": float(np.nanquantile(boff, 0.025)),
                          "hi": float(np.nanquantile(boff, 0.975)), "at_edge": bool(edge)}
        names = OBS + PRED
        for j, k in enumerate(names):
            v = b[:, V + j]
            cell[k] = {"est": float(pt[V + j]), "lo": float(np.nanquantile(v, 0.025)),
                       "hi": float(np.nanquantile(v, 0.975))}
        obs_mist = 1 - pt[V + OBS.index("sound")]
        bobs_mist = 1 - b[:, V + OBS.index("sound")]
        for k, o, bo, pk in (("ratio_mistake", obs_mist, bobs_mist, "p_mistake"),
                             ("ratio_visible", pt[V + 2], b[:, V + 2], "p_visible"),
                             ("ratio_deep_only", pt[V + 3], b[:, V + 3], "p_deep_only"),
                             ("ratio_loss", pt[V + 0], b[:, V + 0], "p_loss")):
            pj = V + len(OBS) + PRED.index(pk)
            r = bo / b[:, pj]
            cell[k] = {"est": float(o / pt[pj]), "lo": float(np.nanquantile(r, 0.025)),
                       "hi": float(np.nanquantile(r, 0.975))}
        out.append(cell)
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m tnh.eval.book", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", type=Path, nargs="+", required=True, help="Maia-3 both-sides offset variants")
    p.add_argument("--shallow", type=Path, required=True)
    p.add_argument("--deep", type=Path, required=True)
    p.add_argument("--ref-offset", type=float, default=0.0, help="the offset measure 3 and log loss are taken at")
    p.add_argument("--book-count", type=int, default=5, help="the exit ply column to count from")
    p.add_argument("--bands", nargs="+", default=["12-19", "20-29", "30-39", "40-59"], help="ply bands, inclusive")
    p.add_argument("--min-rows", type=int, default=100, help="cells with fewer rows show only their row count")
    p.add_argument("--sound", type=float, default=0.03)
    p.add_argument("--visible", type=float, default=0.03)
    p.add_argument("--sizes", type=float, nargs=2, default=[0.10, 0.25])
    p.add_argument("--decided", type=float, default=0.97)
    add_common(p)
    a = p.parse_args(argv)
    s = window_rows(read_sample(a.test))
    fit = window_rows(read_sample(a.fit))
    check_splits(s, fit)
    exit_col = f"exit_ply_{a.book_count}"
    if exit_col not in s.columns:
        raise SystemExit(f"{a.test} has no {exit_col}: draw it again with this version of tnh.eval.sample")

    # engines and classes
    sh = attach(s, a.shallow, ["value"])
    dp = attach(s, a.deep, ["value"])
    same_moves(sh, dp, "shallow against deep")
    r = ragged(dp, "value")
    vd, vs = r.values, ragged(sh, "value").values
    nan_row = np.add.reduceat((np.isnan(vd) | np.isnan(vs)).astype(int), r.offsets[:-1]) > 0
    bd, bs = nanmax_segments(r, vd), nanmax_segments(r, vs)
    decided = ((bd > a.decided) | (bd < 1 - a.decided)) & ~nan_row
    keep = ~nan_row & ~decided
    Ld = np.repeat(bd, r.lengths) - vd
    Ls = np.repeat(bs, r.lengths) - vs
    vis, size = classes(Ls, Ld, a.sound, a.visible, tuple(a.sizes))
    played = r.offsets[:-1] + dp.get_column("_idx").to_numpy()

    # Maia-3 variants
    vs_ = []
    for d in a.variants:
        v = parse_variant(variant_name(d))
        if v["kind"] != "offset" or v["mover"] != v["opp"]:
            print(f"ignoring {d}: not a both-sides offset")
            continue
        vs_.append((v["mover"], d))
    vs_.sort(key=lambda x: x[0])
    offs = np.array([o for o, _ in vs_])
    if a.ref_offset not in offs:
        raise SystemExit(f"--ref-offset {a.ref_offset:+g} is not among the variants {offs.tolist()}")
    wf = fit.get_column("weight").to_numpy()
    L = []
    for o, d in vs_:
        j = attach(fit, d, ["logp"])
        rf = ragged(j, "logp")
        e = fit_eps(rf.values[rf.offsets[:-1] + j.get_column("_idx").to_numpy()], j.get_column("n_legal").to_numpy()
                    .astype(float), wf)
        jt = attach(s, d, ["logp"])
        same_moves(dp, jt, f"{d} against the deep engine")
        rt = ragged(jt, "logp")
        lp = rt.values[played]
        n = s.get_column("n_legal").to_numpy().astype(float)
        L.append(-mix_logp(lp, n, e))
        if o == a.ref_offset:
            pr = np.exp(rt.values)
            rows = r.row_of
            p_mist = np.bincount(rows, weights=np.where(vis != 0, pr, 0.0), minlength=s.height)
            p_vis = np.bincount(rows, weights=np.where(vis == 1, pr, 0.0), minlength=s.height)
            p_deep = np.bincount(rows, weights=np.where(vis == 2, pr, 0.0), minlength=s.height)
            p_loss = np.bincount(rows, weights=np.nan_to_num(pr * Ld), minlength=s.height)
            mx = np.maximum.reduceat(rt.values, rt.offsets[:-1])
            nmax = np.add.reduceat((rt.values == np.repeat(mx, rt.lengths)).astype(float), rt.offsets[:-1])
            top1 = np.where(lp >= mx, 1.0 / nmax, 0.0)
            ref_loss = -mix_logp(lp, n, e)
    V = len(offs)
    obs = np.column_stack([Ld[played], vis[played] == 0, vis[played] == 1, vis[played] == 2,
                           size[played] == 1, size[played] == 2, size[played] == 3]).astype(float)
    X = np.column_stack(L + [obs, p_mist, p_vis, p_deep, p_loss, ref_loss, top1])[keep]
    s = s.filter(pl.Series(keep))
    w = s.get_column("weight").to_numpy()
    cl = cluster_codes(s)
    counts = boot_counts(cl.max() + 1, a.boot, a.seed)
    ply = s.get_column("ply").to_numpy()
    bands = parse_bands(a.bands)
    band = np.full(len(ply), -1)
    for i, (lo, hi) in enumerate(bands):
        band[(ply >= lo) & (ply <= hi)] = i
    in_bands = band >= 0
    band_labels = [f"{lo}-{hi}" for lo, hi in bands] + ["all window rows"]
    nb = len(bands)

    def banded(group: np.ndarray, labels: list[str], mask: np.ndarray | None = None) -> list[dict]:
        """Cells (band, group) for each band and for all the bands together."""
        ok = np.ones(len(group), bool) if mask is None else mask
        G = len(labels)
        res = []
        for bi in range(nb + 1):
            m = ok & ((band == bi) if bi < nb else True)   # the last "band" is every row
            codes = np.where(m, group, -1)
            sel = codes >= 0
            if not sel.any():
                continue
            cells = summarize_cells(X[sel], w[sel], cl[sel], counts, codes[sel], G, offs, V, a.min_rows)
            for gi, c in enumerate(cells):
                if c is not None:
                    res.append({"band": band_labels[bi], "group": labels[gi], **c})
        return res

    ex = s.get_column(exit_col).to_numpy(allow_copy=True).astype(float)
    cells = banded(rel_group((ply - ex).astype(int)), REL_LABELS)
    for c in cells:
        if c["band"] == band_labels[-1] and "offset" in c:
            print(f"plies since exit {c['group']:>6}: rows={c['rows']:>6} offset {c['offset']['est']:+.0f} "
                  f"deep loss {c['deep_loss']['est']:.4f} mistakes {1 - c['sound']['est']:.3f} "
                  f"obs/Maia {c['ratio_mistake']['est']:.2f}")
    write_result(a.results, "book", {
        "what": "engine mistakes, mistakes against Maia-3 and the matched offset by plies since leaving book, "
                "within ply bands; window rows of the test sample",
        "book_count": a.book_count,
        "thresholds": {"sound": a.sound, "visible": a.visible, "sizes": a.sizes, "decided": a.decided},
        "ref_offset": a.ref_offset, "offsets": offs.tolist(), "bands": band_labels, "min_rows": a.min_rows,
        "rows": int(keep.sum()),
        "dropped_missing_value": int(nan_row.sum()), "dropped_decided": int(decided.sum()),
        "shallow": label(a.shallow), "deep": label(a.deep), "variants": [label(d) for _, d in vs_],
        "boot": a.boot, "cells": cells,
        "provenance": provenance({"test": a.test, "fit": a.fit}, [d for _, d in vs_] + [a.shallow, a.deep])})


if __name__ == "__main__":
    main()
