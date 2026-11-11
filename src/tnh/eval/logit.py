"""The conditional logit: a linear score per legal move, softmax over the moves of a position.

    uv run python -m tnh.eval.logit \\
        --logp m0=features/maia3-79m/offset+0 --logp m300=features/maia3-79m/offset+300 \\
        --engine sf10k=features/stockfish/nodes10000-hash16 --engine sf100k=... \\
        --set A=m300 --set B=m0,sf10k,sf100k --set C=m0,m300,sf10k,sf100k \\
        [--full C] [--test S] [--fit S] [--results R] [--boot-fit 100]

Feature sources, each a feature directory:

- ``--logp NAME=DIR``: a move model's log probability, one column ``NAME``.
  Lc0's policy goes in the same way when it arrives.
- ``--engine NAME=DIR``: an engine's per-move values. Columns:
  ``NAME_eloss``, the expected-score loss against the engine's best move;
  ``NAME_cploss``, ``log1p(centipawn loss / 100)``, where mate scores count as
  ``+-(10000 - 100 * moves to mate)`` centipawns (only when the files have ``cp``
  and ``mate``); and ``NAME_nan``, 1 for a move without a value within the budget.
  A missing value is set to the largest loss among the position's valued moves
  (the move is taken to be no better than the worst move the engine did value)
  and flagged by ``NAME_nan``, whose coefficient is fitted. A position where the
  engine valued no move at all is dropped from every fit and every number, and
  counted. ``NAME_nan`` is left out when fewer than ``--min-nan-played`` played moves in the
  fitting sample lack a value: its coefficient cannot be estimated then.

  The expected-score loss saturates in decided positions (+5 and +9 are both
  about 1.0); the squashed centipawn loss does not, which is why both are there.

``--set NAME=a,b,...`` defines a model by its sources (an engine name brings all
its columns, or name single columns). Without ``--set``: every logp source alone
and all sources together. Each set is fitted by maximum likelihood on the
window rows of the fitting sample (weighted by the sample weights) and reported
on the test sample: raw log loss, and mixed with a uniform ``eps`` fitted on the
fitting sample, with intervals over test tournaments and paired differences
against each single move model mixed with its own ``eps``. Coefficients get
intervals from refitting on tournament-bootstrap replicates of the fitting
sample (``--boot-fit``). For ``--full`` (default: the largest set), every source
is also dropped in turn: the rise in test log loss is what that source adds
when entered last.

Writes ``<results>/logit.json`` (or ``--name``).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from tnh.eval.common import (FeatureMismatch, Ragged, add_common, attach, boot_counts, boot_means, check_splits,
                             cluster_codes, est, fit_eps, label, mix_logp, provenance, ragged, read_sample,
                             same_moves, segment_logsumexp, segment_sum, window_rows, wmean, write_result, _columns)

MATE_CP = 10_000


# --- features -----------------------------------------------------------------------

def nanmax_segments(r: Ragged, v: np.ndarray) -> np.ndarray:
    filled = np.where(np.isnan(v), -np.inf, v)
    return np.maximum.reduceat(filled, r.offsets[:-1])


def engine_columns(j: pl.DataFrame, name: str) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    """Loss columns for one engine; returns (columns, rows_without_any_value, counts)."""
    rv = ragged(j, "value")
    v = rv.values
    rows = rv.row_of
    best = nanmax_segments(rv, v)
    no_value = ~np.isfinite(best)
    nan = np.isnan(v)
    cols = {}
    eloss = np.repeat(best, rv.lengths) - v
    cols[f"{name}_eloss"] = eloss
    if "cp" in j.columns and "mate" in j.columns:
        cp = ragged(j, "cp").values
        mate = ragged(j, "mate").values
        eff = np.where(mate != 0, np.sign(mate) * (MATE_CP - 100 * np.abs(mate)), cp)
        eff = np.where(nan, np.nan, eff)   # a move without a value has no score either
        cbest = nanmax_segments(rv, eff)
        closs = np.log1p(np.maximum(np.repeat(cbest, rv.lengths) - eff, 0.0) / 100.0)
        cols[f"{name}_cploss"] = closs
    for c, x in list(cols.items()):
        worst = nanmax_segments(rv, x)
        x = np.where(np.isnan(x), np.repeat(worst, rv.lengths), x)
        cols[c] = x
    cols[f"{name}_nan"] = nan.astype(np.float64)
    counts = {"moves": int(len(v)), "moves_without_value": int(nan.sum()),
              "positions_without_any_value": int(no_value.sum()),
              "positions_with_a_missing_value": int((np.add.reduceat(nan.astype(int), rv.offsets[:-1]) > 0).sum())}
    return cols, no_value, counts


@dataclass
class Data:
    names: list[str]
    X: np.ndarray        # moves x features
    offsets: np.ndarray  # rows + 1
    y: np.ndarray        # flat index of the played move, per row
    n: np.ndarray        # legal moves per row
    w: np.ndarray
    clusters: np.ndarray
    single_lp: dict[str, np.ndarray]   # per logp source: log p of the played move

    @property
    def r(self) -> Ragged:
        return Ragged(np.empty(0), self.offsets)

    def subset(self, cols: list[str]) -> "Data":
        idx = [self.names.index(c) for c in cols]
        return Data(cols, self.X[:, idx], self.offsets, self.y, self.n, self.w, self.clusters, self.single_lp)


def build(sample: pl.DataFrame, logp: dict[str, Path], engines: dict[str, Path]) -> tuple[Data, dict]:
    cols: dict[str, np.ndarray] = {}
    first = None
    drop = np.zeros(sample.height, dtype=bool)
    info = {}
    single = {}
    for name, d in logp.items():
        j = attach(sample, d, ["logp"])
        first = _check(first, j, name)
        r = ragged(j, "logp")
        if not np.all(np.isfinite(r.values)):
            raise FeatureMismatch(f"{d}: non-finite log probabilities")
        cols[name] = r.values
        single[name] = r.values[r.offsets[:-1] + j.get_column("_idx").to_numpy()]
    for name, d in engines.items():
        have = _columns(d)
        j = attach(sample, d, ["value"] + [c for c in ("cp", "mate") if c in have])
        first = _check(first, j, name)
        c, no_value, counts = engine_columns(j, name)
        cols.update(c)
        drop |= no_value
        info[name] = counts
    if first is None:
        raise SystemExit("no feature sources")
    r = ragged(first, "logp" if logp else "value")
    idx = first.get_column("_idx").to_numpy()
    keep = ~drop
    move_keep = np.repeat(keep, r.lengths)
    lens = r.lengths[keep]
    offsets = np.zeros(keep.sum() + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    names = list(cols)
    X = np.column_stack([cols[c][move_keep] for c in names])
    data = Data(names, X, offsets, offsets[:-1] + idx[keep], lens.astype(np.float64),
                sample.get_column("weight").to_numpy()[keep], cluster_codes(sample.filter(pl.Series(keep))),
                {k: v[keep] for k, v in single.items()})
    info["rows"] = int(sample.height)
    info["rows_dropped_no_engine_value"] = int(drop.sum())
    return data, info


def _check(first, j, name):
    if first is not None:
        same_moves(first, j, f"{name} against the first source")
        return first
    return j


# --- the fit --------------------------------------------------------------------------

def scores(D: Data, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row: log probability of the played move, and the flat log probabilities."""
    s = D.X @ beta
    r = D.r
    lse = segment_logsumexp(r, s)
    lp_all = s - np.repeat(lse, r.lengths)
    return lp_all[D.y], lp_all


def fit(D: Data, w: np.ndarray | None = None, beta0: np.ndarray | None = None, max_iter: int = 100,
        tol: float = 1e-10) -> tuple[np.ndarray, int]:
    """Weighted maximum likelihood by Newton's method with step halving. Returns (beta, iterations)."""
    w = D.w if w is None else w
    W = w.sum()
    K = D.X.shape[1]
    beta = np.zeros(K) if beta0 is None else beta0.copy()
    r = D.r

    def ll(b):
        lp, _ = scores(D, b)
        return float(np.sum(w * lp) / W)

    cur = ll(beta)
    for it in range(1, max_iter + 1):
        lp, lp_all = scores(D, beta)
        p = np.exp(lp_all)
        mu = segment_sum(r, D.X * p[:, None])                  # rows x K
        grad = (w[:, None] * (D.X[D.y] - mu)).sum(0) / W
        wm = np.repeat(w, r.lengths) * p
        H = ((D.X * wm[:, None]).T @ D.X - (mu * w[:, None]).T @ mu) / W
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            raise FeatureMismatch(f"singular information matrix for {D.names}: a feature is constant or collinear")
        t = 1.0
        while True:
            new = beta + t * step
            val = ll(new)
            if val >= cur - 1e-15 or t < 1e-6:
                break
            t /= 2
        beta, gain, cur = new, val - cur, val
        if abs(gain) < tol and np.max(np.abs(t * step)) < 1e-7:
            return beta, it
    raise FeatureMismatch(f"no convergence after {max_iter} Newton iterations for {D.names} "
                          f"(a coefficient running off to infinity: a feature separates the played moves)")


# --- main --------------------------------------------------------------------------

def parse_pairs(xs: list[str] | None) -> dict[str, Path]:
    out = {}
    for x in xs or []:
        k, v = x.split("=", 1)
        if k in out:
            raise SystemExit(f"source {k} given twice")
        out[k] = Path(v)
    return out


def expand(spec: str, D: Data, sources: dict[str, list[str]]) -> list[str]:
    cols = []
    for part in spec.split(","):
        if part in sources:
            cols += sources[part]
        elif part in D.names:
            cols.append(part)
        else:
            raise SystemExit(f"unknown feature {part!r}; sources {list(sources)}, columns {D.names}")
    return cols


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m tnh.eval.logit", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logp", action="append", help="NAME=DIR of a move model")
    p.add_argument("--engine", action="append", help="NAME=DIR of an engine")
    p.add_argument("--set", action="append", help="NAME=src,src,... a model to fit")
    p.add_argument("--full", help="the set to drop sources from (default: the largest)")
    p.add_argument("--boot-fit", type=int, default=100, help="refits for the coefficient intervals")
    p.add_argument("--name", default="logit", help="result file name")
    p.add_argument("--min-nan-played", type=int, default=20,
                   help="fit an engine's NaN indicator only when this many played moves lack a value")
    add_common(p)
    a = p.parse_args(argv)
    logp, engines = parse_pairs(a.logp), parse_pairs(a.engine)
    test, fitS = window_rows(read_sample(a.test)), window_rows(read_sample(a.fit))
    check_splits(test, fitS)

    Df, info_f = build(fitS, logp, engines)
    Dt, info_t = build(test, logp, engines)
    # A NaN indicator is fitted only when the played move lacks a value in at least --min-nan-played
    # fitting rows; with fewer its coefficient runs off to -inf (the played move is never one of them).
    # Left out, those moves keep the imputed worst loss.
    unused = [c for c in Df.names if c.endswith("_nan")
              and Df.X[Df.y, Df.names.index(c)].sum() < a.min_nan_played]
    sources = {k: [k] for k in logp}
    for e in engines:
        sources[e] = [c for c in Df.names if c.startswith(f"{e}_") and c not in unused]
    sets = {}
    for s in a.set or []:
        k, v = s.split("=", 1)
        sets[k] = expand(v, Df, sources)
    if not sets:
        sets = {k: [k] for k in logp}
        sets["all"] = [c for c in Df.names if c not in unused]
    full = a.full or max(sets, key=lambda k: len(sets[k]))
    print(f"fit rows {len(Df.w)} (dropped {info_f['rows_dropped_no_engine_value']}), test rows {len(Dt.w)} "
          f"(dropped {info_t['rows_dropped_no_engine_value']}); unused NaN indicators: {unused}")

    Kt = Dt.clusters.max() + 1
    counts_t = boot_counts(Kt, a.boot, a.seed)
    counts_f = boot_counts(Df.clusters.max() + 1, a.boot_fit, a.seed + 1)

    # single move models, each mixed with its own eps
    singles = {}
    for k in logp:
        e = fit_eps(Df.single_lp[k], Df.n, Df.w)
        singles[k] = {"eps": e, "loss": -mix_logp(Dt.single_lp[k], Dt.n, e)}

    def fit_set(cols, boot=True):
        Df_s, Dt_s = Df.subset(cols), Dt.subset(cols)
        beta, it = fit(Df_s)
        lp_f, _ = scores(Df_s, beta)
        e = fit_eps(lp_f, Df.n, Df.w)
        lp_t, _ = scores(Dt_s, beta)
        out = {"columns": cols, "beta": beta.tolist(), "iterations": it, "eps": e,
               "loss_raw": -lp_t, "loss_mixed": -mix_logp(lp_t, Dt.n, e),
               "fit_loglik": float(np.sum(Df.w * lp_f) / Df.w.sum())}
        if boot and a.boot_fit:
            bs = []
            for r in range(a.boot_fit):
                wr = Df.w * counts_f[r][Df.clusters]
                bs.append(fit(Df_s, wr, beta0=beta, max_iter=30)[0])
            bs = np.array(bs)
            out["beta_lo"] = np.quantile(bs, 0.025, axis=0).tolist()
            out["beta_hi"] = np.quantile(bs, 0.975, axis=0).tolist()
        return out

    results = {}
    for k, cols in sets.items():
        r = fit_set(cols)
        results[k] = r
        print(f"set {k}: test raw {wmean(r['loss_raw'], Dt.w):.4f} mixed {wmean(r['loss_mixed'], Dt.w):.4f}; "
              + ", ".join(f"{c}={b:+.3f}" for c, b in zip(cols, r["beta"])), flush=True)

    # entered last: drop each source from the full set
    drops = {}
    for src, cols in sources.items():
        keep = [c for c in sets[full] if c not in cols]
        if len(keep) == len(sets[full]) or not keep:
            continue
        r = fit_set(keep, boot=False)
        drops[src] = r

    # test numbers with intervals; differences are paired
    names = list(results)
    cols_ = [results[k]["loss_mixed"] for k in names] + [results[k]["loss_raw"] for k in names] \
        + [singles[k]["loss"] for k in singles] \
        + [drops[s]["loss_mixed"] - results[full]["loss_mixed"] for s in drops]
    M = np.column_stack(cols_)
    bm = boot_means(M, Dt.w, Dt.clusters, counts_t)
    pt = np.array([wmean(M[:, j], Dt.w) for j in range(M.shape[1])])
    j = 0
    out_sets = {}
    for k in names:
        out_sets[k] = {kk: v for kk, v in results[k].items() if not kk.startswith("loss_")}
        out_sets[k]["test_mixed"] = est(pt[j], bm[:, j]); j += 1
    for k in names:
        out_sets[k]["test_raw"] = est(pt[j], bm[:, j]); j += 1
    out_single = {}
    for k in singles:
        out_single[k] = {"eps": singles[k]["eps"], "test_mixed": est(pt[j], bm[:, j]), "dir": str(logp[k]),
                         "label": label(logp[k])}
        j += 1
    out_drop = {}
    for s in drops:
        out_drop[s] = {"columns": drops[s]["columns"], "beta": drops[s]["beta"],
                       "test_rise": est(pt[j], bm[:, j])}
        j += 1
    # each set against each single model, paired
    vs = {}
    for k in names:
        for s in singles:
            d = results[k]["loss_mixed"] - singles[s]["loss"]
            bd = boot_means(d, Dt.w, Dt.clusters, counts_t)
            vs[f"{k} - {s}"] = est(wmean(d, Dt.w), bd[:, 0])
    for s, r in out_drop.items():
        print(f"drop {s} from {full}: test rises {r['test_rise']['est']:+.4f} "
              f"[{r['test_rise']['lo']:+.4f}, {r['test_rise']['hi']:+.4f}]")
    write_result(a.results, a.name, {
        "what": "conditional logit over the legal moves; fitted on the fitting sample, window rows",
        "sources": {"logp": {k: str(v) for k, v in logp.items()}, "engine": {k: str(v) for k, v in engines.items()}},
        "columns": Df.names, "unused_nan_indicators": unused, "rows": {"fit": info_f, "test": info_t},
        "boot": a.boot, "boot_fit": a.boot_fit, "full": full, "sets": out_sets, "singles": out_single,
        "entered_last": out_drop, "set_minus_single": vs,
        "provenance": provenance({"test": a.test, "fit": a.fit}, list(logp.values()) + list(engines.values()))})


if __name__ == "__main__":
    main()
