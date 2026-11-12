"""The result model: win, draw and loss for the side to move, from the position and both FIDE ratings.

    uv run python -m tnh.eval.result fit [--test S] [--fit S] [--results R] [--boot 1000]
    uv run python -m tnh.eval.result refit            # writes the stored parameters (result_params.json)
    uv run python -m tnh.eval.result check-live --n 200   # the live model against the feature files (GPU)

**Sources**, each a position's win/draw/loss for the side to move from a feature directory's
``wdl`` column (one row per position, keyed on game_id and ply):

- ``lc0w``: Lc0 BT4, 50 nodes, ``WDLCalibrationElo`` = mover + 400 (the frontier value's raw input);
  ``lc0w0`` and ``lc0w800`` the same at + 0 and + 800 where they exist (10,000 + 10,000 games);
- ``lc0raw``: Lc0 BT4's raw value head, one network call (from the policy run);
- ``maia0``, ``maia600``: Maia-3 79M's value head at the players' ratings and at + 600.

Each source enters as two numbers, ``log(w / l)`` and ``log d - (log w + log l) / 2``, with the
probabilities floored at 1e-4 for this purpose only (an exact 0 from a calibrated WDL would
otherwise be an infinite feature; the model's own output is never clipped).

**Context**: mean rating ``(mover + opp) / 2 - 2200`` and rating difference ``mover - opp``, both
per 100 points, ``ply / 40`` and whether White is to move.

**Models**: a multinomial logit (win and draw against loss, each with its own coefficients on every
feature) fitted by maximum likelihood, and LightGBM with the multiclass objective (the proper
three-class log loss), early-stopped on held-out tournaments of the fitting sample, one thread.

**Samples**: one window position per game (``tnh.eval.value.one_per_game``, seed 7): every game of
``test_sub100k`` (``test_value_all``) and 20,000 games of ``fit_sub100k`` (``fit_value20k``). The
result is from the side to move's point of view.

Reported on the test games: three-class log loss and the Brier score of the expected score
``w + d / 2`` against the score, with intervals over tournaments and paired differences; calibration
by rating band, rating difference and ply; what each source adds beside the others. Writes
``<results>/result.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize

from tnh.eval.common import (RESULTS, boot_counts, boot_means, cluster_codes, est, load_features, read_sample,
                             write_result)
from tnh.paths import CORPUS, FEATURES, REPORTS

HERE = Path(__file__).parent
PARAMS = HERE / "result_params.json"
TREES = HERE / "result_trees.txt"
SOURCES = {
    "lc0w": FEATURES / "lc0-bt4" / "nodes50-wdl-offset+400",
    "lc0raw": FEATURES / "lc0-bt4" / "policy",
    "maia0": FEATURES / "maia3-79m" / "offset+0",
    "maia600": FEATURES / "maia3-79m" / "offset+600",
}
EXTRA = {"lc0w0": FEATURES / "lc0-bt4" / "nodes50-wdl-offset+0", "lc0w800": FEATURES / "lc0-bt4" / "nodes50-wdl-offset+800"}
CONTEXT = ["ctx_mean", "ctx_diff", "ctx_ply", "ctx_white"]
FLOOR = 1e-4
CLASS = {1.0: 0, 0.5: 1, 0.0: 2}


# --- features -------------------------------------------------------------------------------

def source_logits(wdl: np.ndarray) -> np.ndarray:
    """rows x 3 (w, d, l) -> rows x 2: log(w/l), log d - (log w + log l)/2."""
    lp = np.log(np.maximum(wdl, FLOOR))
    return np.column_stack([lp[:, 0] - lp[:, 2], lp[:, 1] - 0.5 * (lp[:, 0] + lp[:, 2])])


def context(mover: np.ndarray, opp: np.ndarray, ply: np.ndarray, white: np.ndarray) -> np.ndarray:
    return np.column_stack([((mover + opp) / 2 - 2200) / 100, (mover - opp) / 100, ply / 40, white.astype(float)])


def wdl_column(s: pl.DataFrame, d: Path) -> np.ndarray:
    """The sample's rows' wdl from a feature directory, NaN rows where it has none."""
    f = load_features(d, ["game_id", "ply", "wdl"], s.select("game_id", "ply"))
    j = s.select("game_id", "ply").join(f, on=["game_id", "ply"], how="left", maintain_order="left")
    out = np.full((s.height, 3), np.nan)
    ok = j.get_column("wdl").is_not_null().to_numpy()
    if ok.any():
        w = np.stack(j.filter(pl.Series(ok)).get_column("wdl").to_numpy()).astype(np.float64)
        out[ok] = w / w.sum(1, keepdims=True)
    return out


def table(s: pl.DataFrame, sources: dict[str, Path]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Per source the rows x 3 wdl; the context matrix; the result class (0 win, 1 draw, 2 loss)."""
    s = s.with_columns(pl.col("ply").cast(pl.Int64))
    w = {k: wdl_column(s, d) for k, d in sources.items()}
    ctx = context(s.get_column("mover_elo").to_numpy().astype(float), s.get_column("opp_elo").to_numpy().astype(float),
                  s.get_column("ply").to_numpy().astype(float), s.get_column("white_to_move").to_numpy())
    y = np.array([CLASS[x] for x in s.get_column("result_mover").to_list()])
    return w, ctx, y


def design(w: dict[str, np.ndarray], ctx: np.ndarray, srcs: list[str], use_ctx: bool = True) -> np.ndarray:
    cols = [source_logits(w[k]) for k in srcs] + ([ctx] if use_ctx else [])
    return np.column_stack(cols) if cols else np.zeros((len(ctx), 0))


# --- the multinomial logit -----------------------------------------------------------------------

def mlogit_probs(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    k = X.shape[1] + 1
    W = theta.reshape(2, k)
    Xi = np.column_stack([np.ones(len(X)), X])
    z = np.column_stack([Xi @ W[0], Xi @ W[1], np.zeros(len(X))])
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def fit_mlogit(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    w = np.ones(len(y)) if w is None else w
    Y = np.zeros((len(y), 3))
    Y[np.arange(len(y)), y] = 1
    Xi = np.column_stack([np.ones(len(X)), X])

    def f(theta):
        p = mlogit_probs(theta, X)
        nll = -np.sum(w * np.log(p[np.arange(len(y)), y])) / w.sum()
        g = -((w[:, None] * (Y[:, :2] - p[:, :2])).T @ Xi) / w.sum()
        return nll, g.ravel()

    res = minimize(f, np.zeros(2 * Xi.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    if not res.success:
        raise RuntimeError(f"multinomial logit did not converge: {res.message}")
    return res.x


# --- trees ------------------------------------------------------------------------------------------

def fit_trees(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 1, val_frac: float = 0.2):
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    k = groups.max() + 1
    order = rng.permutation(k)
    sizes = np.bincount(groups, minlength=k)
    held = np.zeros(k, dtype=bool)
    held[order[np.cumsum(sizes[order]) <= val_frac * len(y)]] = True
    hv = held[groups]
    params = {"objective": "multiclass", "num_class": 3, "num_leaves": 15, "learning_rate": 0.03,
              "min_data_in_leaf": 200, "lambda_l2": 1.0, "feature_fraction": 0.9, "num_threads": 1,
              "verbosity": -1, "seed": seed, "deterministic": True}
    dtr = lgb.Dataset(X[~hv], label=y[~hv])
    dva = lgb.Dataset(X[hv], label=y[hv], reference=dtr)
    b = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return b


def trees_probs(b, X: np.ndarray) -> np.ndarray:
    return b.predict(X, num_iteration=b.best_iteration)


# --- evaluation ---------------------------------------------------------------------------------------

def losses(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row: three-class log loss and the squared error of the expected score."""
    score = np.array([1.0, 0.5, 0.0])[y]
    e = p[:, 0] + 0.5 * p[:, 1]
    return -np.log(p[np.arange(len(y)), y]), (e - score) ** 2


def calibration(s: pl.DataFrame, p: np.ndarray, y: np.ndarray, by: str) -> list[dict]:
    mover = s.get_column("mover_elo").to_numpy().astype(float)
    opp = s.get_column("opp_elo").to_numpy().astype(float)
    ply = s.get_column("ply").to_numpy()
    if by == "rating":
        v, edges, labels = (mover + opp) / 2, [0, 2000, 2200, 2400, 9999], ["1800-1999", "2000-2199", "2200-2399", "2400+"]
    elif by == "diff":
        v, edges = mover - opp, [-9999, -200, -100, -25, 25, 100, 200, 9999]
        labels = ["<-200", "-200..-101", "-100..-26", "-25..25", "26..100", "101..200", ">200"]
    else:
        v, edges, labels = ply, [0, 10, 20, 30, 40, 60, 999], ["0-9", "10-19", "20-29", "30-39", "40-59", "60+"]
    g = np.searchsorted(edges, v, side="right") - 1
    score = np.array([1.0, 0.5, 0.0])[y]
    out = []
    for i, lab in enumerate(labels):
        m = g == i
        if not m.any():
            continue
        out.append({"group": lab, "n": int(m.sum()), "score": float(score[m].mean()),
                    "pred_score": float((p[m, 0] + 0.5 * p[m, 1]).mean()),
                    "draw_rate": float((y[m] == 1).mean()), "pred_draw": float(p[m, 1].mean())})
    return out


def priority_mask(s: pl.DataFrame) -> np.ndarray:
    """Plies 10-40 of games between players averaging 2000-2399: the positions that matter most."""
    mean = (s.get_column("mover_elo").to_numpy() + s.get_column("opp_elo").to_numpy()) / 2
    ply = s.get_column("ply").to_numpy()
    return (ply >= 10) & (ply <= 40) & (mean >= 2000) & (mean < 2400)


VARIANTS = [   # name, sources, context, model
    ("ratings only", [], True, "logit"),
    ("maia0", ["maia0"], True, "logit"),
    ("maia600", ["maia600"], True, "logit"),
    ("lc0raw", ["lc0raw"], True, "logit"),
    ("lc0w", ["lc0w"], True, "logit"),
    ("lc0w no context", ["lc0w"], False, "logit"),
    ("maia0+maia600+lc0raw", ["maia0", "maia600", "lc0raw"], True, "logit"),
    ("lc0w+lc0raw", ["lc0w", "lc0raw"], True, "logit"),
    ("lc0w+maia0+maia600", ["lc0w", "maia0", "maia600"], True, "logit"),
    ("all", ["lc0w", "lc0raw", "maia0", "maia600"], True, "logit"),
    ("maia0+maia600+lc0raw trees", ["maia0", "maia600", "lc0raw"], True, "trees"),
    ("all trees", ["lc0w", "lc0raw", "maia0", "maia600"], True, "trees"),
]


def cmd_fit(a) -> None:
    test = read_sample(a.test)
    fit = read_sample(a.fit)
    wt, ct, yt = table(test, SOURCES)
    wf, cf, yf = table(fit, SOURCES)
    for name, w in (("test", wt), ("fit", wf)):
        for k, v in w.items():
            miss = np.isnan(v).any(1).sum()
            if miss:
                raise SystemExit(f"{name}: {miss} rows without {k} ({SOURCES[k]}): run the features first")
    gt, gf = cluster_codes(test), cluster_codes(fit)
    counts = boot_counts(gt.max() + 1, a.boot, a.seed)
    prio = priority_mask(test)
    results, P = {}, {}
    for name, srcs, use_ctx, kind in VARIANTS:
        t0 = time.time()
        Xf, Xt = design(wf, cf, srcs, use_ctx), design(wt, ct, srcs, use_ctx)
        if kind == "logit":
            th = fit_mlogit(Xf, yf)
            p = mlogit_probs(th, Xt)
            extra = {"coef": th.tolist()}
        else:
            b = fit_trees(Xf, yf, gf)
            p = trees_probs(b, Xt)
            extra = {"trees": b.best_iteration}
        P[name] = p
        results[name] = {"sources": srcs, "context": use_ctx, "model": kind, **extra,
                         "seconds_to_fit": round(time.time() - t0, 1)}
        ll, br = losses(p, yt)
        print(f"{name:>28}: log loss {ll.mean():.4f}  brier {br.mean():.4f}  priority {ll[prio].mean():.4f}", flush=True)
    names = list(P)
    best = min(names, key=lambda k: losses(P[k], yt)[0].mean())
    # every variant's prediction per test row, for the look-ahead comparison and the speed front
    pred = test.select("game_id", pl.col("ply").cast(pl.Int64))
    for k, v in P.items():
        pred = pred.with_columns(pl.Series(k, v.tolist(), dtype=pl.List(pl.Float64)))
    a.predictions.parent.mkdir(parents=True, exist_ok=True)
    pred.write_parquet(a.predictions)
    cols, keys = [], []
    for k in names:
        ll, br = losses(P[k], yt)
        llb, brb = losses(P[best], yt)
        cols += [ll, br, ll - llb, br - brb]
        keys.append(k)
    X = np.column_stack(cols)
    w = np.ones(len(yt))
    bm = boot_means(X, w, gt, counts)
    pt = X.mean(0)
    bmp = boot_means(X[prio], w[prio], gt[prio], counts)
    ptp = X[prio].mean(0)
    for i, k in enumerate(keys):
        j = 4 * i
        results[k].update({"logloss": est(pt[j], bm[:, j]), "brier": est(pt[j + 1], bm[:, j + 1]),
                           "logloss_minus_best": est(pt[j + 2], bm[:, j + 2]), "brier_minus_best": est(pt[j + 3], bm[:, j + 3]),
                           "priority_logloss": est(ptp[j], bmp[:, j]),
                           "priority_logloss_minus_best": est(ptp[j + 2], bmp[:, j + 2])})
    # what each source adds beside the others: paired differences against "all"
    adds = {}
    for k in ("maia0", "maia600", "lc0raw", "lc0w"):
        srcs = [s_ for s_ in ["lc0w", "lc0raw", "maia0", "maia600"] if s_ != k]
        th = fit_mlogit(design(wf, cf, srcs), yf)
        p = mlogit_probs(th, design(wt, ct, srcs))
        d = losses(p, yt)[0] - losses(P["all"], yt)[0]
        bd = boot_means(d, w, gt, counts)
        bdp = boot_means(d[prio], w[prio], gt[prio], counts)
        adds[k] = {"rise_without": est(d.mean(), bd[:, 0]), "priority_rise_without": est(d[prio].mean(), bdp[:, 0])}
        print(f"  all without {k}: {d.mean():+.4f} [{np.quantile(bd[:, 0], 0.025):+.4f}, {np.quantile(bd[:, 0], 0.975):+.4f}]")
    # the other Lc0 calibration offsets, where they exist (step 1's 10k + 10k games)
    offsets = {}
    we, _, _ = table(test, EXTRA)
    wfe, _, _ = table(fit, EXTRA)
    for k in EXTRA:
        okt, okf = ~np.isnan(we[k]).any(1), ~np.isnan(wfe[k]).any(1)
        wt2, wf2 = {**wt, k: we[k]}, {**wf, k: wfe[k]}
        rows = {}
        for lab, srcs in ((k, [k]), ("lc0w", ["lc0w"])):
            th = fit_mlogit(design({kk: v[okf] for kk, v in wf2.items()}, cf[okf], srcs), yf[okf])
            p = mlogit_probs(th, design({kk: v[okt] for kk, v in wt2.items()}, ct[okt], srcs))
            rows[lab] = losses(p, yt[okt])[0]
        d = rows[k] - rows["lc0w"]
        bd = boot_means(d, np.ones(okt.sum()), cluster_codes(test.filter(pl.Series(okt))), boot_counts(
            cluster_codes(test.filter(pl.Series(okt))).max() + 1, a.boot, a.seed))
        offsets[k] = {"games": int(okt.sum()), "logloss": float(rows[k].mean()), "lc0w_same_rows": float(rows["lc0w"].mean()),
                      "minus_lc0w": est(d.mean(), bd[:, 0])}
    calib = {k: {by: calibration(test, P[k], yt, by) for by in ("rating", "diff", "ply")} for k in (best, "lc0w")}
    write_result(a.results, a.name, {
        "what": "result model: win/draw/loss for the side to move; one position per game; fitted on fit games, test games",
        "test": str(a.test), "fit": str(a.fit), "games": {"test": len(yt), "fit": len(yf),
                                                          "test_priority": int(prio.sum())},
        "test_tournaments": int(gt.max() + 1), "draw_rate_test": float((yt == 1).mean()),
        "best": best, "variants": results, "adds": adds, "offsets": offsets, "calibration": calib, "boot": a.boot})


# --- the stored model and its live features ------------------------------------------------------------

def cmd_refit(a) -> None:
    """Fit the chosen variant on all fitting games and store it (the test games are not used)."""
    fit = read_sample(a.fit)
    wf, cf, yf = table(fit, SOURCES)
    out = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "fit": str(a.fit), "games": len(yf)}
    for name, srcs in (("full", a.full.split(",")), ("cheap", a.cheap.split(","))):
        th = fit_mlogit(design(wf, cf, srcs), yf)
        out[name] = {"sources": srcs, "context": CONTEXT, "coef": th.tolist(), "model": "logit"}
    out["sources"] = {k: str(v) for k, v in SOURCES.items()}
    out["refit"] = "uv run python -m tnh.eval.result refit"
    PARAMS.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {PARAMS}")


class ResultModel:
    """Win, draw, loss for the side to move given both FIDE ratings, from the stored logit.

    ``variant="full"`` uses Lc0 at 50 nodes with ``WDLCalibrationElo`` (13.6 positions/s, the search);
    ``variant="cheap"`` only network calls (Maia-3 at two ratings and Lc0's raw value), the leaf of the
    look-ahead in ``tnh.eval.lookahead``. Same shape as ``tnh.models.combo.FrontierValue``.
    """

    def __init__(self, variant: str = "full", params: Path = PARAMS, *, lc0_processes: int = 1) -> None:
        p = json.loads(Path(params).read_text())[variant]
        self.sources = p["sources"]
        self.theta = np.array(p["coef"])
        self.name = f"result-{variant}"
        self._maia = self._policy = self._search = None
        self.lc0_processes = lc0_processes

    def source_wdl(self, queries) -> dict[str, np.ndarray]:
        from tnh.models.maia3 import Elo
        out = {}
        if "maia0" in self.sources or "maia600" in self.sources:
            if self._maia is None:
                from tnh.models.maia3 import Maia3
                self._maia = Maia3("79m", batch_size=128)
            m0, m6 = self._maia.predict_many(queries, [Elo.offset(0.0), Elo.offset(600.0)])
            out["maia0"] = np.array([p.wdl for p in m0], dtype=np.float64)
            out["maia600"] = np.array([p.wdl for p in m6], dtype=np.float64)
        if "lc0raw" in self.sources:
            if self._policy is None:
                from tnh.models.lc0 import Lc0Policy
                self._policy = Lc0Policy()
            out["lc0raw"] = np.array([p.wdl for p in self._policy.predict(queries)], dtype=np.float64)
        if "lc0w" in self.sources:
            if self._search is None:
                from tnh.models.lc0 import Lc0Search
                self._search = Lc0Search(50, wdl_elo=Elo.parse("offset+400"), processes=self.lc0_processes)
            out["lc0w"] = np.array([v.wdl for v in self._search.evaluate(queries)], dtype=np.float64)
        return {k: v / v.sum(1, keepdims=True) for k, v in out.items()}

    def from_sources(self, w: dict[str, np.ndarray], queries) -> np.ndarray:
        ctx = context(np.array([q.mover_elo for q in queries], float), np.array([q.opp_elo for q in queries], float),
                      np.array([len(q.moves_uci) for q in queries], float),
                      np.array([q.fen.split()[1] == "w" for q in queries]))
        return mlogit_probs(self.theta, design(w, ctx, self.sources))

    def wdl(self, queries) -> np.ndarray:
        """queries x (win, draw, loss) for the side to move."""
        return self.from_sources(self.source_wdl(queries), queries)

    def expected_score(self, queries) -> np.ndarray:
        w = self.wdl(queries)
        return w[:, 0] + 0.5 * w[:, 1]

    def close(self) -> None:
        if self._search is not None:
            self._search.close()


def cmd_check_live(a) -> None:
    """The live model's sources against the feature files on n test positions."""
    from tnh.models.combo import _queries
    s, qs = _queries(a.test, a.n, 3)
    for variant in ("cheap", "full"):
        m = ResultModel(variant)
        t = time.perf_counter()
        live = m.source_wdl(qs)
        dt = time.perf_counter() - t
        for k, v in live.items():
            ref = wdl_column(s, SOURCES[k])
            print(f"{variant} {k}: max |live - file| = {np.nanmax(np.abs(v - ref)):.5f}")
        print(f"{variant}: {len(qs) / dt:.1f} positions/s")
        m.close()


def cmd_bench(a) -> None:
    """Positions per second of each source on the GPU, one call per batch as the live model makes it."""
    from tnh.models.combo import _queries
    from tnh.models.lc0 import Lc0Policy, Lc0Search
    from tnh.models.maia3 import Elo, Maia3

    _, qs = _queries(a.test, a.n, 4)
    out = {}

    def timed(name, f, q):
        f(q[:24])                                   # warm-up
        t = time.perf_counter()
        f(q)
        out[name] = len(q) / (time.perf_counter() - t)
        print(f"{name}: {out[name]:.1f} positions/s ({len(q)} positions)", flush=True)

    if not a.search_only:
        pol = Lc0Policy()          # loaded first: Lc0's backend needs its memory in one piece
        m = Maia3("79m", batch_size=a.maia_batch)
        timed("maia3-79m one rating", lambda q: m.predict_many(q, [Elo.offset(0.0)]), qs)
        timed("maia3-79m two ratings", lambda q: m.predict_many(q, [Elo.offset(0.0), Elo.offset(600.0)]), qs)
        timed("lc0 raw value (policy call)", pol.predict, qs)
    # the searches run in their own process (an Lc0 engine is about 2 GB of memory beside the networks above)
    for nodes, n in ((50, a.n_search), (400, a.n_search // 2), (1600, a.n_search // 4)):
        if nodes not in a.search_nodes:
            continue
        srch = Lc0Search(nodes, wdl_elo=Elo.parse("offset+400"))
        timed(f"lc0 wdl {nodes} nodes", srch.evaluate, qs[:n])
        srch.close()
    old = (a.results / "result_bench.json")
    if old.exists():
        out = {**json.loads(old.read_text())["rates"], **out}
    write_result(a.results, "result_bench", {"what": "positions per second, laptop GPU shared with the repertoire "
                                             "builder and another developer's Lc0 run, one process", "rates": out,
                                             "positions": a.n, "date": time.strftime("%Y-%m-%dT%H:%M:%S")})


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m tnh.eval.result", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(required=True)
    sp = sub.add_parser("fit")
    sp.add_argument("--test", type=Path, default=CORPUS / "samples" / "test_value_all.parquet")
    sp.add_argument("--fit", type=Path, default=CORPUS / "samples" / "fit_value20k.parquet")
    sp.add_argument("--results", type=Path, default=RESULTS)
    sp.add_argument("--name", default="result")
    sp.add_argument("--predictions", type=Path, default=Path.home() / "out/tn-hittare/scratch/eval/result/predictions.parquet")
    sp.add_argument("--boot", type=int, default=1000)
    sp.add_argument("--seed", type=int, default=7)
    sp.set_defaults(fn=cmd_fit)
    sp = sub.add_parser("refit")
    sp.add_argument("--fit", type=Path, default=CORPUS / "samples" / "fit_value20k.parquet")
    sp.add_argument("--full", default="lc0w,lc0raw,maia0,maia600")
    sp.add_argument("--cheap", default="maia0,maia600,lc0raw")
    sp.set_defaults(fn=cmd_refit)
    sp = sub.add_parser("bench")
    sp.add_argument("--test", type=Path, default=CORPUS / "samples" / "test_value_all.parquet")
    sp.add_argument("--n", type=int, default=1200)
    sp.add_argument("--n-search", type=int, default=240)
    sp.add_argument("--maia-batch", type=int, default=128)
    sp.add_argument("--search-only", action="store_true")
    sp.add_argument("--search-nodes", type=int, nargs="*", default=[])
    sp.add_argument("--results", type=Path, default=RESULTS)
    sp.set_defaults(fn=cmd_bench)
    sp = sub.add_parser("check-live")
    sp.add_argument("--test", type=Path, default=CORPUS / "samples" / "test_value_all.parquet")
    sp.add_argument("--n", type=int, default=200)
    sp.set_defaults(fn=cmd_check_live)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
