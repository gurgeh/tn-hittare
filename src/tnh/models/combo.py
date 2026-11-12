"""Step 1's best combination as a live move model, and Lc0's calibrated WDL as a frontier value.

    # refit the coefficients from the feature files (CPU only, no network runs)
    uv run python -m tnh.models.combo fit [--fit S] [--test S] [--boot 1000]
    # live checks on the GPU
    uv run python -m tnh.models.combo check-live --n 500
    uv run python -m tnh.models.combo throughput --batch-sizes 16 64 256
    uv run python -m tnh.models.combo check-value --n 200

`Combo` is a `MoveModel`: per legal move the features

- `m0`: Maia-3 79M log probability at the players' own ratings (offset +0);
- `mbest`: Maia-3 79M log probability at offset +600 (step 1's best single offset;
  see `maia_best` for the mapped rating);
- `lc0pol`: Lc0 BT4's raw policy (log probability, one node);
- `lc0ch_eloss`, `lc0ch_cploss`: from Lc0 BT4's raw value of every child (no search),
  the expected-score loss against the best child and `log1p(centipawn loss / 100)`,
  computed by `tnh.eval.logit.engine_columns`, the code that made them for the fit;

a linear score `beta . x` with the coefficients in `combo_coef.json`, softmax over the
legal moves, mixed with a uniform `eps`: `p' = (1 - eps) p + eps / n`. That is the
conditional logit of `tnh.eval.logit` (step 1's linear logit had no rating terms), fitted
on the window rows of the fitting subsample's feature files.

`FrontierValue` gives a position's win, draw and loss for the side to move given both
FIDE ratings: Lc0 BT4, 50 nodes, `WDLCalibrationElo` = mover + 400, `Contempt` = mover
minus opponent, then step 1's five-parameter recalibration (`value_recal.json`).
`WithValue(move_model, frontier)` is a `MoveModel` whose `wdl` is the frontier value: the
repertoire builder reads its frontier values from `Prediction.wdl`, so this is the whole
adapter.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from tnh.models.base import Prediction, Query
from tnh.models.prune import keep_mask

MAX_BATCH_DEFAULT = 12   # tnh.models.lc0.MAX_BATCH (not imported here: lc0 loads the bindings lazily)
from tnh.paths import CORPUS, FEATURES, REPORTS

HERE = Path(__file__).parent
COEF_FILE = HERE / "combo_coef.json"
VALUE_FILE = HERE / "value_recal.json"
COLUMNS = ["m0", "mbest", "lc0pol", "lc0ch_eloss", "lc0ch_cploss"]
LOGP_SOURCES = {"m0": FEATURES / "maia3-79m" / "offset+0", "mbest": FEATURES / "maia3-79m" / "offset+600",
                "lc0pol": FEATURES / "lc0-bt4" / "policy"}
ENGINE_SOURCES = {"lc0ch": FEATURES / "lc0-bt4" / "children"}
FIT_SAMPLE = CORPUS / "samples" / "fit_sub100k.parquet"
TEST_SAMPLE = CORPUS / "samples" / "test_sub100k.parquet"
STEP1_SETS = {   # figures in reports/step1/logit_sub.json, to check that this code path reproduces them
    "M+ch": ["m0", "mbest", "lc0ch_eloss", "lc0ch_cploss"],
    "M+pol": ["m0", "mbest", "lc0pol"],
}


def _git_commit() -> str:
    r = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    return r.stdout.strip()


# -- the live model -----------------------------------------------------------------------

def mix(logp: np.ndarray, eps: float) -> np.ndarray:
    n = len(logp)
    return np.log((1 - eps) * np.exp(logp.astype(np.float64)) + eps / n)


def child_columns(values: list) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """`lc0ch_eloss` and `lc0ch_cploss` per position from `MoveValues`, by the fit's own code."""
    from tnh.eval.logit import engine_columns

    j = pl.DataFrame({
        "value": [v.value.astype(np.float32) for v in values],
        "cp": [v.cp.astype(np.float32) for v in values],
        "mate": [v.mate.astype(np.int16) for v in values],
    }, schema={"value": pl.List(pl.Float32), "cp": pl.List(pl.Float32), "mate": pl.List(pl.Int16)})
    cols, no_value, _ = engine_columns(j, "lc0ch")
    if no_value.any():
        raise ValueError(f"{int(no_value.sum())} positions without any child value")
    lens = [len(v.moves) for v in values]
    cuts = np.cumsum([0] + lens)
    e, c = cols["lc0ch_eloss"], cols["lc0ch_cploss"]
    return [e[a:b] for a, b in zip(cuts[:-1], cuts[1:])], [c[a:b] for a, b in zip(cuts[:-1], cuts[1:])]


def pruned_child_columns(values: list, only: list[set[str]], fill: str):
    """`lc0ch_eloss`, `lc0ch_cploss`, `lc0ch_unev` per position with only the kept children
    evaluated, by `tnh.models.prune.pruned_columns`, the code that made them for the fit."""
    from tnh.models.prune import MATE_CP, pruned_columns

    lens = [len(v.moves) for v in values]
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    value = np.concatenate([v.value for v in values]).astype(np.float64) if values else np.zeros(0)
    cp = np.concatenate([v.cp for v in values]).astype(np.float64) if values else np.zeros(0)
    mate = np.concatenate([v.mate for v in values]).astype(np.float64) if values else np.zeros(0)
    keep = np.concatenate([[m in o for m in v.moves] for v, o in zip(values, only)]).astype(bool) \
        if values else np.zeros(0, dtype=bool)
    if np.isnan(value[keep]).any():
        raise ValueError("a kept child without a value")
    eff = np.where(mate != 0, np.sign(mate) * (MATE_CP - 100 * np.abs(mate)), cp)
    cols = pruned_columns(value, eff, offsets, keep, fill)
    cut = lambda x: [x[a:b] for a, b in zip(offsets[:-1], offsets[1:])]   # noqa: E731
    return cut(cols["lc0ch_eloss"]), cut(cols["lc0ch_cploss"]), cut(cols["lc0ch_unev"])


def coef_file(prune: str | None) -> Path:
    return COEF_FILE if prune is None else HERE / f"combo_coef_{prune.replace('|', '_')}.json"


class Combo:
    """Maia-3 (two rating settings) + Lc0 policy + Lc0 child values, combined by the stored logit.

    `wdl` is Lc0's raw value of the position (from the children call); wrap in `WithValue`
    for the calibrated frontier value.
    """

    def __init__(self, coef: Path = COEF_FILE, *, prune: str | None = None, maia_size: str = "79m",
                 maia_best: str | None = None, maia_batch: int = 512) -> None:
        """`prune`: a rule of `tnh.models.prune` (e.g. "p0.5"); its coefficients come from
        `combo_coef_<rule>.json` (refit: `python -m tnh.models.combo fit --prune <rule>`)."""
        from tnh.models.lc0 import Lc0Children, Lc0Policy
        from tnh.models.maia3 import Elo, Maia3

        if prune is not None and coef == COEF_FILE:
            coef = coef_file(prune)
        c = json.loads(Path(coef).read_text())
        self.prune = c.get("prune")
        self.fill = c.get("fill")
        want = COLUMNS if self.prune is None else COLUMNS + ["lc0ch_unev"]
        if c["columns"] != want:
            raise ValueError(f"{coef} has columns {c['columns']}, this model computes {want}")
        self.coef = c
        self.beta = np.array(c["beta"], dtype=np.float64)
        self.eps = float(c["eps"])
        self.elos = [Elo.offset(0.0), Elo.parse(maia_best or c["maia_best"])]
        self.maia = Maia3(maia_size, batch_size=maia_batch)
        self.policy = Lc0Policy()
        self.children = Lc0Children()
        self.name = f"combo-maia3-{maia_size}-lc0" + ("" if self.prune is None else f"-{self.prune}")
        self.seconds: Counter = Counter()

    def features(self, queries: list[Query]) -> tuple[list[list[str]], list[np.ndarray], list[tuple]]:
        """Per query: the legal moves, the moves x 5 feature matrix, Lc0's raw WDL."""
        t = time.perf_counter()
        m0, mb = self.maia.predict_many(queries, self.elos)
        self.seconds["maia"] += time.perf_counter() - t
        t = time.perf_counter()
        pol = self.policy.predict(queries)
        self.seconds["lc0 policy"] += time.perf_counter() - t
        only = None
        if self.prune is not None:
            only = []
            for i in range(len(queries)):
                prob = {}
                for p in (m0[i], mb[i], pol[i]):
                    for m, lp in zip(p.moves, np.exp(p.logp.astype(np.float64))):
                        prob[m] = max(prob.get(m, 0.0), lp)
                ms = list(prob)
                pr = np.array([prob[m] for m in ms])
                keep = keep_mask(pr, np.array([0, len(ms)]), self.prune) if ms else np.zeros(0, dtype=bool)
                only.append({m for m, k in zip(ms, keep) if k})
        t = time.perf_counter()
        ch = self.children.evaluate(queries, only)
        self.seconds["lc0 children"] += time.perf_counter() - t
        t = time.perf_counter()
        if self.prune is None:
            eloss, cploss = child_columns(ch)
        else:
            eloss, cploss, unev = pruned_child_columns(ch, only, self.fill)
        moves, X, wdl = [], [], []
        for i, q in enumerate(queries):
            ms = ch[i].moves
            cols = []
            for p in (m0[i], mb[i], pol[i]):
                idx = {m: k for k, m in enumerate(p.moves)}
                if set(idx) != set(ms):
                    raise ValueError(f"models disagree on the legal moves at {q.fen}")
                cols.append(p.logp[[idx[m] for m in ms]].astype(np.float64))
            cols += [eloss[i], cploss[i]] + ([] if self.prune is None else [unev[i]])
            moves.append(ms)
            X.append(np.column_stack(cols) if ms else np.zeros((0, len(self.beta))))
            wdl.append(ch[i].wdl)
        self.seconds["combine"] += time.perf_counter() - t
        return moves, X, wdl

    def combine(self, X: np.ndarray) -> np.ndarray:
        s = X @ self.beta
        lp = s - np.logaddexp.reduce(s)
        return mix(lp, self.eps).astype(np.float32)

    def predict(self, queries: list[Query]) -> list[Prediction]:
        moves, X, wdl = self.features(queries)
        out = []
        for ms, x, w in zip(moves, X, wdl):
            out.append(Prediction(ms, self.combine(x) if ms else np.zeros(0, dtype=np.float32), w))
        return out


# -- the frontier value -----------------------------------------------------------------------

class FrontierValue:
    """Win, draw, loss for the side to move given both FIDE ratings (step 1's best value head)."""

    def __init__(self, params: Path = VALUE_FILE, *, processes: int = 1) -> None:
        from tnh.models.lc0 import Lc0Search
        from tnh.models.maia3 import Elo

        self.params = json.loads(Path(params).read_text())
        r = self.params["recal"]
        self.theta = np.array([np.log(r["t"]), r["b_win"], r["b_draw"], r["k_draw_per_100"],
                               np.log(r["eps"] / (1 - r["eps"]))])
        self.search = Lc0Search(self.params["nodes"], wdl_elo=Elo.parse(self.params["wdl_elo"]), processes=processes)
        self.seconds: Counter = Counter()

    def raw(self, queries: list[Query]) -> np.ndarray:
        t = time.perf_counter()
        vals = self.search.evaluate(queries)
        self.seconds["lc0 search"] += time.perf_counter() - t
        if any(v.wdl is None for v in vals):
            raise ValueError("a position without a WDL (game over): the frontier value is for live positions")
        w = np.array([v.wdl for v in vals], dtype=np.float64)
        return w / w.sum(1, keepdims=True)

    def recalibrate(self, raw: np.ndarray, queries: list[Query]) -> np.ndarray:
        from tnh.eval.value import _logp, recal_probs

        r = np.array([((q.mover_elo + q.opp_elo) / 2 - 2200) / 100 for q in queries], dtype=np.float64)
        return recal_probs(self.theta, _logp(raw), r)

    def wdl(self, queries: list[Query]) -> np.ndarray:
        """queries x (win, draw, loss) for the mover."""
        return self.recalibrate(self.raw(queries), queries)

    def expected_score(self, queries: list[Query]) -> np.ndarray:
        w = self.wdl(queries)
        return w[:, 0] + 0.5 * w[:, 1]

    def close(self) -> None:
        self.search.close()


class WithValue:
    """A `MoveModel` with another model's moves and the frontier value as its `wdl`."""

    def __init__(self, move_model, frontier: FrontierValue) -> None:
        self.move_model = move_model
        self.frontier = frontier
        self.name = f"{move_model.name}+frontier"

    def predict(self, queries: list[Query]) -> list[Prediction]:
        preds = self.move_model.predict(queries)
        live = [i for i, p in enumerate(preds) if p.moves]
        if live:
            w = self.frontier.wdl([queries[i] for i in live])
            for i, row in zip(live, w):
                preds[i] = Prediction(preds[i].moves, preds[i].logp, tuple(float(x) for x in row))
        return preds


# -- fitting from the feature files -------------------------------------------------------------

def _data(sample: Path):
    from tnh.eval.common import read_sample, window_rows
    from tnh.eval.logit import build

    s = window_rows(read_sample(sample))
    D, info = build(s, LOGP_SOURCES, ENGINE_SOURCES)
    return s, D, info


def _test_loss(D, cols, beta, eps, boot: int, seed: int = 7) -> dict:   # seed 7: step 1's default
    from tnh.eval.common import boot_counts, boot_means, est, mix_logp, wmean
    from tnh.eval.logit import scores

    lp, _ = scores(D.subset(cols), np.asarray(beta))
    loss = -mix_logp(lp, D.n, eps)
    counts = boot_counts(D.clusters.max() + 1, boot, seed)
    bm = boot_means(loss[:, None], D.w, D.clusters, counts)
    return est(wmean(loss, D.w), bm[:, 0])


def cmd_fit_pruned(a) -> None:
    from tnh.eval.common import fit_eps, read_meta
    from tnh.eval.logit import fit, scores
    from tnh.models.prune import _load, pruned_columns

    base = ["m0", "mbest", "lc0pol"]
    cols = COLUMNS + ["lc0ch_unev"]
    parts = {}
    for side, path in (("fit", a.fit), ("test", a.test)):
        D, v, c, pr = _load(path)
        k = keep_mask(pr, D.offsets, a.prune)
        parts[side] = (D.subset(base).with_columns(pruned_columns(v, c, D.offsets, k, a.fill)).subset(cols), k, D)
    Xf, kf, Df = parts["fit"]
    Xt, kt, Dt = parts["test"]
    beta, it = fit(Xf)
    lp, _ = scores(Xf, beta)
    eps = fit_eps(lp, Df.n, Df.w)
    test = _test_loss(Xt, cols, beta, eps, a.boot)
    per_pos = np.add.reduceat(kt.astype(np.int64), Dt.offsets[:-1])
    print(f"prune {a.prune} (fill {a.fill}): beta {np.round(beta, 4).tolist()}, eps {eps:.6f}; children per position "
          f"{per_pos.mean():.2f}; test {test['est']:.4f} [{test['lo']:.4f}, {test['hi']:.4f}]", flush=True)
    out = {
        "what": "conditional logit over the legal moves, Lc0 children evaluated only for the moves kept by the "
                "rule (tnh.models.prune), window rows of the fitting subsample",
        "columns": cols, "beta": beta.tolist(), "eps": eps, "newton_iterations": it, "prune": a.prune, "fill": a.fill,
        "maia_best": "offset+600",
        "sources": {k: {"dir": str(v), "meta": read_meta(v)} for k, v in {**LOGP_SOURCES, **ENGINE_SOURCES}.items()},
        "fit": {"sample": str(a.fit), "window_rows": int(len(Df.w)), "moves": int(Df.X.shape[0]),
                "children_kept": float(kf.mean())},
        "test": {"sample": str(a.test), "window_rows": int(len(Dt.w)), "mixed": test, "boot": a.boot,
                 "children_per_position": float(per_pos.mean())},
        "date": dt.datetime.now().isoformat(timespec="seconds"), "commit": _git_commit(),
        "refit": f"uv run python -m tnh.models.combo fit --prune '{a.prune}' --fill {a.fill}",
    }
    path = a.out if a.out != COEF_FILE else coef_file(a.prune)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}")


def cmd_fit(a) -> None:
    from tnh.eval.common import fit_eps, read_meta
    from tnh.eval.logit import fit, scores

    if a.prune is not None:
        return cmd_fit_pruned(a)
    t0 = time.perf_counter()
    _, Df, info_f = _data(a.fit)
    _, Dt, info_t = _data(a.test)
    print(f"fit rows {len(Df.w)}, test rows {len(Dt.w)} ({time.perf_counter() - t0:.0f} s to load)", flush=True)
    step1 = json.loads((REPORTS / "step1" / "logit_sub.json").read_text())["sets"]

    def one(cols):
        Ds = Df.subset(cols)
        beta, it = fit(Ds)
        lp, _ = scores(Ds, beta)
        eps = fit_eps(lp, Df.n, Df.w)
        return beta, eps, it

    checks = {}
    for name, cols in STEP1_SETS.items():
        beta, eps, _ = one(cols)
        mine = _test_loss(Dt, cols, beta, eps, a.boot)
        ref = step1[name]
        checks[name] = {"columns": cols, "beta": beta.tolist(), "eps": eps, "test_mixed": mine,
                        "step1_beta": ref["beta"], "step1_test_mixed": ref["test_mixed"]}
        print(f"{name}: test {mine['est']:.4f} [{mine['lo']:.4f}, {mine['hi']:.4f}]; step 1 "
              f"{ref['test_mixed']['est']:.4f} [{ref['test_mixed']['lo']:.4f}, {ref['test_mixed']['hi']:.4f}]; "
              f"max |beta - step 1| {np.max(np.abs(beta - np.array(ref['beta']))):.2e}", flush=True)
    beta, eps, it = one(COLUMNS)
    test = _test_loss(Dt, COLUMNS, beta, eps, a.boot)
    print(f"combination {COLUMNS}: beta {np.round(beta, 4).tolist()}, eps {eps:.6f}, {it} Newton iterations; "
          f"test {test['est']:.4f} [{test['lo']:.4f}, {test['hi']:.4f}]", flush=True)
    out = {
        "what": "conditional logit over the legal moves (tnh.eval.logit), window rows of the fitting subsample",
        "columns": COLUMNS, "beta": beta.tolist(), "eps": eps, "newton_iterations": it,
        "maia_best": "offset+600",
        "sources": {k: {"dir": str(v), "meta": read_meta(v)} for k, v in {**LOGP_SOURCES, **ENGINE_SOURCES}.items()},
        "fit": {"sample": str(a.fit), "window_rows": int(len(Df.w)), "moves": int(Df.X.shape[0]),
                "tournaments": int(Df.clusters.max() + 1)},
        "test": {"sample": str(a.test), "window_rows": int(len(Dt.w)), "mixed": test, "boot": a.boot},
        "step1_reproduced": checks,
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "commit": _git_commit(),
        "refit": "uv run python -m tnh.models.combo fit",
    }
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {a.out}")


def cmd_value_params(a) -> None:
    """Copy step 1's recalibration of the chosen value head into the repository, with its source."""
    src = REPORTS / "step1" / "value_wdl.json"
    d = json.loads(src.read_text())
    m = next(x for x in d["models"] if x["label"] == a.label)
    out = {"what": "Lc0 BT4 WDL at 50 nodes with WDLCalibrationElo, then step 1's recalibration "
                   "q = (1-eps) softmax(t log p + b + [draw] k (mean rating - 2200)/100) + eps/3",
           "nodes": 50, "wdl_elo": a.label.split("wdl-")[1], "recal": m["recal"],
           "test_logloss": {"raw": m["raw"], "recalibrated": m["recalibrated"]},
           "games": d["games"], "source": str(src), "feature_dir": m["dir"],
           "date": dt.datetime.now().isoformat(timespec="seconds"), "commit": _git_commit(),
           "refit": "uv run python -m tnh.eval.value (step 1), then uv run python -m tnh.models.combo value-params"}
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {a.out}: {m['recal']}")


# -- live checks ------------------------------------------------------------------------------

def _queries(sample: Path, n: int, seed: int) -> tuple[pl.DataFrame, list[Query]]:
    from tnh.eval.common import read_sample, window_rows
    from tnh.models.run import queries

    s = window_rows(read_sample(sample))
    s = s.sample(n=min(n, s.height), seed=seed).sort("game_id", "ply")
    g = (pl.scan_parquet(CORPUS / "games.parquet").select("game_id", "moves_uci")
         .join(pl.LazyFrame({"game_id": s.get_column("game_id").unique()}), on="game_id", how="semi").collect())
    n0 = s.height
    s = s.join(g, on="game_id", how="inner", maintain_order="left")
    if s.height < n0:
        # games.parquet was rebuilt after the samples were drawn (2026-11-12 07:04): a few
        # sample games are no longer in it. Their rows are dropped and counted; for the rest
        # the replay checks that the moves still reach the sample's FEN.
        print(f"dropped {n0 - s.height} of {n0} sampled rows: their game is not in {CORPUS / 'games.parquet'}")
    return s, queries(s)


def cmd_check_live(a) -> None:
    from tnh.eval.common import attach, ragged
    from tnh.eval.logit import engine_columns

    s, qs = _queries(a.test, a.n, a.seed)
    rule = a.prune[0] if a.prune and a.prune[0] not in (None, "none") else None
    model = Combo(maia_best=a.maia_best, prune=rule)
    t = time.perf_counter()
    moves, X, _ = model.features(qs)
    print(f"{len(qs)} test window positions, live features in {time.perf_counter() - t:.1f} s "
          f"({dict((k, round(v, 1)) for k, v in model.seconds.items())})", flush=True)
    # the same rows from the feature files
    file_cols = {}
    for k, d in LOGP_SOURCES.items():
        j = attach(s, d, ["logp"])
        file_cols[k] = (j.get_column("moves").to_list(), ragged(j, "logp"))
    j = attach(s, ENGINE_SOURCES["lc0ch"], ["value", "cp", "mate"])
    ec, _, _ = engine_columns(j, "lc0ch")
    rch = ragged(j, "value")
    rcp, rmate = ragged(j, "cp"), ragged(j, "mate")
    played = s.get_column("played_uci").to_list()
    worst = Counter()
    ll_live, ll_file, n_pos = [], [], 0
    for i, (ms, x) in enumerate(zip(moves, X)):
        fm = file_cols["m0"][0][i]
        if sorted(fm) != sorted(ms):
            raise ValueError(f"row {i}: feature files and live model disagree on the legal moves")
        order = [fm.index(m) for m in ms]
        F = []
        for k in ("m0", "mbest", "lc0pol"):
            r = file_cols[k][1]
            F.append(r.values[r.offsets[i] : r.offsets[i + 1]][order])
        lo, hi = rch.offsets[i], rch.offsets[i + 1]
        if rule is None:
            F += [ec["lc0ch_eloss"][lo:hi][order], ec["lc0ch_cploss"][lo:hi][order]]
        else:
            from tnh.models.prune import MATE_CP, pruned_columns
            v = rch.values[lo:hi][order].astype(np.float64)
            cpv = rcp.values[lo:hi][order].astype(np.float64)
            mt = rmate.values[lo:hi][order].astype(np.float64)
            eff = np.where(mt != 0, np.sign(mt) * (MATE_CP - 100 * np.abs(mt)), cpv)
            kf = keep_mask(np.exp(np.column_stack(F)).max(1), np.array([0, len(ms)]), rule)
            kl = x[:, 5] == 0
            worst["kept set differs (positions)"] += int(not np.array_equal(kf, kl))
            pc = pruned_columns(v, eff, np.array([0, len(ms)]), kf, model.fill)
            F += [pc["lc0ch_eloss"], pc["lc0ch_cploss"], pc["lc0ch_unev"]]
        F = np.column_stack(F)
        for c, name in enumerate(COLUMNS + ([] if rule is None else ["lc0ch_unev"])):
            if name != "lc0ch_unev":
                worst[name] = max(worst[name], float(np.max(np.abs(F[:, c] - x[:, c]))))
        lp_live, lp_file = model.combine(x), model.combine(F)
        worst["combined logp"] = max(worst["combined logp"], float(np.max(np.abs(lp_live - lp_file))))
        p = ms.index(played[i])
        ll_live.append(-float(lp_live[p]))
        ll_file.append(-float(lp_file[p]))
        n_pos += 1
    w = s.get_column("weight").to_numpy()
    print(f"max |live - feature file| per feature: " + ", ".join(f"{k} {v:.4f}" for k, v in worst.items()))
    print(f"log loss of the played move on these {n_pos} rows (sample-weighted): live {np.average(ll_live, weights=w):.4f}, "
          f"feature files {np.average(ll_file, weights=w):.4f}; unweighted {np.mean(ll_live):.4f} vs {np.mean(ll_file):.4f}")


def cmd_throughput(a) -> None:
    import os

    s, qs = _queries(a.test, a.n, a.seed)
    print(f"load {os.getloadavg()}", flush=True)
    for size in a.maia_sizes:
        model = Combo(maia_size=size)
        model.predict(qs[:16])
        for bs in a.batch_sizes:
            model.seconds.clear()
            t = time.perf_counter()
            for k in range(0, len(qs), bs):
                model.predict(qs[k : k + bs])
            dt_ = time.perf_counter() - t
            parts = ", ".join(f"{k} {v / dt_:.0%}" for k, v in model.seconds.items())
            print(f"maia {size}, batch {bs:>4}: {len(qs) / dt_:6.1f} positions/s end to end ({parts})", flush=True)
        del model


class _GpuSampler:
    """GPU utilisation every 200 ms while a block runs (nvidia-smi), for 'where the time goes'."""

    def __enter__(self):
        self.proc = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
                                      "-lms", "200"], stdout=subprocess.PIPE, text=True)
        return self

    def __exit__(self, *exc):
        self.proc.terminate()
        out, _ = self.proc.communicate(timeout=10)
        self.samples = [int(x) for x in out.split() if x.strip().isdigit()]

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples)) if self.samples else float("nan")


def cmd_front(a) -> None:
    """Log loss against positions per second for the live model's variants, on the same rows."""
    import os

    from tnh.eval.common import boot_counts, boot_means, cluster_codes, est, wmean
    from tnh.models.lc0 import Lc0Children, Lc0Policy
    from tnh.models.maia3 import Elo, Maia3

    s, qs = _queries(a.test, a.n, a.seed)
    played = s.get_column("played_uci").to_list()
    w = s.get_column("weight").to_numpy()
    cl = cluster_codes(s)
    counts = boot_counts(cl.max() + 1, a.boot, 7)
    print(f"{len(qs)} test window positions; load {os.getloadavg()}", flush=True)
    step1 = json.loads((REPORTS / "step1" / "logit_sub.json").read_text())["sets"]
    seconds: dict[str, float] = {}

    def timed(name, f):
        with _GpuSampler() as g:
            t = time.perf_counter()
            r = f()
            seconds[name] = time.perf_counter() - t
        print(f"  {name}: {seconds[name]:.1f} s, {len(qs) / seconds[name]:.1f} positions/s, GPU busy {g.mean:.0f} %",
              flush=True)
        return r

    maia = Maia3("79m", batch_size=512)
    maia.predict_many(qs[:64], [Elo.offset(0.0), Elo.offset(600.0)])
    m0, mb = timed("maia (two settings)", lambda: maia.predict_many(qs, [Elo.offset(0.0), Elo.offset(600.0)]))
    del maia
    pol_model = Lc0Policy()
    pol_model.predict(qs[:24])
    pol = timed("lc0 policy", lambda: pol_model.predict(qs))
    del pol_model

    def aligned(i, ms):
        return [p.logp[[{m: k for k, m in enumerate(p.moves)}[x] for x in ms]].astype(np.float64)
                for p in (m0[i], mb[i], pol[i])]

    def loss_of(lp_rows, eps):
        return np.array([-float(mix(lp, eps)[k]) for lp, k in lp_rows])

    rows = {}
    moves_of = [p.moves for p in m0]
    # Maia only and Maia + policy: step 1's coefficients (logit_sub.json, sets M and M+pol)
    for name, cols in (("maia only (M)", ["m0", "mbest"]), ("maia + lc0 policy (M+pol)", ["m0", "mbest", "lc0pol"])):
        beta, eps = np.array(step1[name.split("(")[1][:-1]]["beta"]), step1[name.split("(")[1][:-1]]["eps"]
        lps = []
        for i in range(len(qs)):
            X = np.column_stack(aligned(i, moves_of[i])[: len(cols)])
            sc = X @ beta
            lps.append((sc - np.logaddexp.reduce(sc), moves_of[i].index(played[i])))
        rows[name] = (loss_of(lps, eps), ["maia (two settings)"] + (["lc0 policy"] if "lc0pol" in cols else []))
    configs = [(r, MAX_BATCH_DEFAULT, False) for r in a.prune] + [(r, b, True) for r in a.prune_big for b in a.big_batches]
    for rule, bs, big in configs:
        rule = None if rule in (None, "none") else rule
        model = Combo.__new__(Combo)
        c = json.loads(coef_file(rule).read_text())
        model.beta, model.eps, model.prune, model.fill = np.array(c["beta"]), float(c["eps"]), c.get("prune"), c.get("fill")
        ch_model = Lc0Children(batch_size=bs, allow_nondeterministic_batch=big)
        only = None
        if rule is not None:
            only = []
            for i in range(len(qs)):
                pr = np.exp(np.column_stack(aligned(i, moves_of[i]))).max(1)
                keep = keep_mask(pr, np.array([0, len(pr)]), rule)
                only.append({m for m, k in zip(moves_of[i], keep) if k})
        ch_model.evaluate(qs[:8], None if only is None else only[:8])
        label = f"+ children {'all' if rule is None else rule}, batch {bs}"
        ch = timed(label, lambda: ch_model.evaluate(qs, only))
        del ch_model
        if rule is None:
            el, cp = child_columns(ch)
            un = None
        else:
            el, cp, un = pruned_child_columns(ch, only, model.fill)
        lps = []
        for i in range(len(qs)):
            ms = ch[i].moves
            base = [p.logp[[{m: k for k, m in enumerate(p.moves)}[x] for x in ms]].astype(np.float64)
                    for p in (m0[i], mb[i], pol[i])]
            X = np.column_stack(base + [el[i], cp[i]] + ([] if un is None else [un[i]]))
            sc = X @ model.beta
            lps.append((sc - np.logaddexp.reduce(sc), ms.index(played[i])))
        rows[f"M+pol {label}"] = (loss_of(lps, model.eps), ["maia (two settings)", "lc0 policy", label])
    ref = rows[next(k for k in rows if "children all, batch" in k and f"batch {MAX_BATCH_DEFAULT}" in k)][0] \
        if any("children all" in k for k in rows) else None
    print(f"\n| model | log loss | minus all children (batch {MAX_BATCH_DEFAULT}) | positions/s |")
    print("|---|---|---|---|")
    for name, (loss, parts) in rows.items():
        bm = boot_means(loss[:, None], w, cl, counts)
        e = est(wmean(loss, w), bm[:, 0])
        d = ""
        if ref is not None:
            bd = boot_means((loss - ref)[:, None], w, cl, counts)
            de = est(wmean(loss - ref, w), bd[:, 0])
            d = f"{de['est']:+.4f} [{de['lo']:+.4f}, {de['hi']:+.4f}]"
        pps = len(qs) / sum(seconds[p] for p in parts)
        print(f"| {name} | {e['est']:.4f} [{e['lo']:.4f}, {e['hi']:.4f}] | {d} | {pps:.1f} |", flush=True)


def cmd_check_value(a) -> None:
    from tnh.eval.common import attach

    s, qs = _queries(a.sample, a.n, a.seed)
    fv = FrontierValue()
    fv.raw(qs[:2])
    t = time.perf_counter()
    raw = fv.raw(qs)
    dt_ = time.perf_counter() - t
    j = attach(s, Path(fv.params["feature_dir"]), ["wdl"])
    file_raw = np.stack(j.get_column("wdl").to_numpy()).astype(np.float64)
    file_raw /= file_raw.sum(1, keepdims=True)
    rec_live, rec_file = fv.recalibrate(raw, qs), fv.recalibrate(file_raw, qs)
    e_live = rec_live[:, 0] + 0.5 * rec_live[:, 1]
    e_file = rec_file[:, 0] + 0.5 * rec_file[:, 1]
    print(f"{len(qs)} positions: {len(qs) / dt_:.1f} positions/s (Lc0, 50 nodes, 1 process); "
          f"max |raw WDL live - feature file| {np.max(np.abs(raw - file_raw)):.4f}; "
          f"max |expected score after recalibration| {np.max(np.abs(e_live - e_file)):.4f}, "
          f"mean {np.mean(np.abs(e_live - e_file)):.5f}")
    fv.close()


def cmd_value_offline(a) -> None:
    """Step 1's recalibrated log loss of the frontier value, from the stored parameters and the feature files."""
    from tnh.eval.common import read_sample, window_rows
    from tnh.eval.value import CLASS, _logp, one_per_game, recal_probs, wdl_of
    from tnh.eval.common import boot_counts, boot_means, cluster_codes, est, wmean

    p = json.loads(VALUE_FILE.read_text())
    s = read_sample(a.sample)
    s = one_per_game(window_rows(s), "random", a.seed)   # as tnh.eval.value: one row per game, equal weights
    w = wdl_of(s, Path(p["feature_dir"]))
    r = p["recal"]
    th = np.array([np.log(r["t"]), r["b_win"], r["b_draw"], r["k_draw_per_100"], np.log(r["eps"] / (1 - r["eps"]))])
    rr = ((s.get_column("mover_elo") + s.get_column("opp_elo")).to_numpy() / 2 - 2200) / 100
    q = recal_probs(th, _logp(w), rr)
    y = np.array([CLASS[x] for x in s.get_column("result_mover").to_list()])
    loss = -np.log(q[np.arange(len(y)), y])
    wt = np.ones(len(y))
    cl = cluster_codes(s)
    bm = boot_means(loss[:, None], wt, cl, boot_counts(cl.max() + 1, 1000, a.seed))
    e = est(wmean(loss, wt), bm[:, 0])
    print(f"{len(y)} games: recalibrated log loss {e['est']:.4f} [{e['lo']:.4f}, {e['hi']:.4f}]; step 1 "
          f"{p['test_logloss']['recalibrated']['est']:.4f} [{p['test_logloss']['recalibrated']['lo']:.4f}, "
          f"{p['test_logloss']['recalibrated']['hi']:.4f}]")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m tnh.models.combo", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("fit")
    sp.add_argument("--fit", type=Path, default=FIT_SAMPLE)
    sp.add_argument("--test", type=Path, default=TEST_SAMPLE)
    sp.add_argument("--boot", type=int, default=1000)
    sp.add_argument("--out", type=Path, default=COEF_FILE)
    sp.add_argument("--prune", help="a rule of tnh.models.prune, e.g. p0.5; writes combo_coef_<rule>.json")
    sp.add_argument("--fill", default="zero", choices=["zero", "worst"])
    sp.set_defaults(fn=cmd_fit)
    sp = sub.add_parser("value-params")
    sp.add_argument("--label", default="lc0-bt4/nodes50-wdl-offset+400")
    sp.add_argument("--out", type=Path, default=VALUE_FILE)
    sp.set_defaults(fn=cmd_value_params)
    sp = sub.add_parser("value-offline")
    sp.add_argument("--sample", type=Path, default=CORPUS / "samples" / "test_value10k.parquet")
    sp.add_argument("--seed", type=int, default=7, help="step 1's default")
    sp.set_defaults(fn=cmd_value_offline)
    for name, fn in (("check-live", cmd_check_live), ("throughput", cmd_throughput)):
        sp = sub.add_parser(name)
        sp.add_argument("--test", type=Path, default=TEST_SAMPLE)
        sp.add_argument("--n", type=int, default=500)
        sp.add_argument("--seed", type=int, default=1)
        sp.add_argument("--maia-best")
        sp.add_argument("--prune", nargs="*", default=[None], help="rules; 'none' for all children")
        sp.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 64, 256])
        sp.add_argument("--maia-sizes", nargs="+", default=["79m"])
        sp.set_defaults(fn=fn)
    sp = sub.add_parser("front")
    sp.add_argument("--test", type=Path, default=TEST_SAMPLE)
    sp.add_argument("--n", type=int, default=2000)
    sp.add_argument("--seed", type=int, default=2)
    sp.add_argument("--boot", type=int, default=1000)
    sp.add_argument("--prune", nargs="+", default=["p2", "p1", "p0.5", "none"])
    sp.add_argument("--prune-big", nargs="*", default=[], help="rules to rerun at the large batch sizes")
    sp.add_argument("--big-batches", type=int, nargs="*", default=[48])
    sp.set_defaults(fn=cmd_front)
    sp = sub.add_parser("check-value")
    sp.add_argument("--sample", type=Path, default=CORPUS / "samples" / "test_value10k.parquet")
    sp.add_argument("--n", type=int, default=200)
    sp.add_argument("--seed", type=int, default=1)
    sp.set_defaults(fn=cmd_check_value)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
