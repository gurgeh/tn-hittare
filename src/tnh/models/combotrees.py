"""The move model on top of Maia and Lc0: a linear logit and boosted trees, per point of the front.

    uv run python -m tnh.models.combotrees fit [--configs ...] [--threads 3] [--out DIR | --overwrite]   # fit, store, report
    uv run python -m tnh.models.combotrees analyse [--threads 3]               # main model: bias, reliability
    uv run python -m tnh.models.combotrees report                              # print the stored tables

A configuration (`CONFIGS`) names what the model sees: Maia-3 at +0 and +600 (79M, 23M or
5M), optionally an Lc0 network's policy (BT4 or the distilled T3), optionally that
network's raw child values, all children or pruned by a rule of `tnh.models.prune`. Per
configuration two models are fitted on the fitting subsample's feature files and stored in
`combo_trees/<config>.json` (+ `.txt` for the trees):

- **linear**: the conditional logit of `tnh.eval.logit` on the base columns (for `+ch-p0.5`
  that is exactly `combo_coef_p0.5.json`);
- **trees**: step 1's form (`tnh.eval.trees`): LightGBM with the exact group softmax as a
  custom objective, starting from the linear logit fitted on the training rows, with
  per-move extras (each model's rank of the move, the board facts of step 1's stack) and
  context that is free at prediction time (both FIDE ratings and their difference, the ply,
  `log n_legal`, the entropy of Maia +600 and of the Lc0 policy), plus explicit products of
  each log probability and child loss with the mover's rating and the ply. No clock, book
  count or Stockfish. 20 % of the fitting tournaments are held out for early stopping.

Reported on the test subsample with step 1's bootstrap (seed 7), paired against linear
`+ch-p0.5` (Combo p0.5). `analyse` adds, for the main configuration, the bias of each model
by kind of move (step 1's classes by the Stockfish labels and classes by Lc0), the expected
loss per move, reliability tables at the thresholds the repertoire search uses, and
the repairs tried.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import chess
import numpy as np

from tnh.paths import FEATURES

HERE = Path(__file__).parent
MODEL_DIR = HERE / "combo_trees"
MAIN = "+ch-p0.5"
CONFIGS = {
    "maia5m": {"maia": "5m"},
    "maia23m": {"maia": "23m"},
    "maia79m": {"maia": "79m"},
    "+pol": {"maia": "79m", "pol": "bt4"},
    "+ch-p2": {"maia": "79m", "pol": "bt4", "ch": "bt4", "rule": "p2"},
    "+ch-p1": {"maia": "79m", "pol": "bt4", "ch": "bt4", "rule": "p1"},
    "+ch-p0.5": {"maia": "79m", "pol": "bt4", "ch": "bt4", "rule": "p0.5"},
    "+ch-all": {"maia": "79m", "pol": "bt4", "ch": "bt4", "rule": "all"},
    "+t3pol": {"maia": "79m", "pol": "t3"},
    "+t3pol+t3ch-p0.5": {"maia": "79m", "pol": "t3", "ch": "t3", "rule": "p0.5"},
}
PRODUCT_COLS = ["m0", "mbest", "lc0pol", "lc0ch_eloss", "lc0ch_cploss"]
PIECES = ["N", "B", "R", "Q", "K"]
RATING_CTX = ["ctx_mover_elo", "ctx_opp_elo", "ctx_elo_diff"]
RATING_BANDS = [(1800, 2000), (2000, 2200), (2200, 2400), (2400, 3000)]
BAND_LABELS = ["1800-1999", "2000-2199", "2200-2399", "2400+"]
PLY_BANDS = [(0, 20), (20, 30), (30, 40), (40, 60)]


def base_columns(cfg: dict) -> list[str]:
    cols = ["m0", "mbest"]
    if cfg.get("pol"):
        cols.append("lc0pol")
    if cfg.get("ch"):
        cols += ["lc0ch_eloss", "lc0ch_cploss"] + ([] if cfg["rule"] == "all" else ["lc0ch_unev"])
    return cols


def sources(cfg: dict) -> dict[str, Path]:
    m = FEATURES / f"maia3-{cfg['maia']}"
    out = {"m0": m / "offset+0", "mbest": m / "offset+600"}
    if cfg.get("pol"):
        out["lc0pol"] = FEATURES / f"lc0-{cfg['pol']}" / "policy"
    if cfg.get("ch"):
        out["lc0ch"] = FEATURES / f"lc0-{cfg['ch']}" / ("children" if cfg["ch"] == "bt4" else f"children-{cfg['rule']}")
    return out


# -- features shared by fitting and the live model ----------------------------------------------

def ranks(x: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Rank of each move within its position by `x`, 0 = largest; ties by move order."""
    n = np.diff(offsets)
    row = np.repeat(np.arange(len(n)), n)
    order = np.lexsort((-x, row))
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x)) - np.repeat(offsets[:-1], n)
    return r


def board_facts(fens: list[str], moves: list[list[str]]) -> dict[str, np.ndarray]:
    out = {k: [] for k in ["capture", "check", "promotion", "castling"] + [f"piece_{p}" for p in PIECES]}
    for fen, ms in zip(fens, moves):
        b = chess.Board(fen)
        for u in ms:
            m = chess.Move.from_uci(u)
            out["capture"].append(b.is_capture(m))
            out["check"].append(b.gives_check(m))
            out["promotion"].append(m.promotion is not None)
            out["castling"].append(b.is_castling(m))
            pt = b.piece_type_at(m.from_square)
            for i, p in enumerate(PIECES):
                out[f"piece_{p}"].append(pt == i + 2)
    return {f"board_{k}": np.asarray(v, dtype=np.float64) for k, v in out.items()}


def entropy(logp: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    p = np.exp(logp)
    return -np.add.reduceat(p * logp, offsets[:-1])


def extra_features(Xb: np.ndarray, base: list[str], offsets: np.ndarray, fens, moves, mover_elo, opp_elo, ply):
    """Per-move extras (ranks, board facts) and per-position context."""
    mv = {f"rank_{c}": ranks(Xb[:, base.index(c)], offsets) for c in ("m0", "mbest", "lc0pol") if c in base}
    mv.update(board_facts(fens, moves))
    ctx = {"ctx_mover_elo": (mover_elo - 2200) / 200, "ctx_opp_elo": (opp_elo - 2200) / 200,
           "ctx_elo_diff": (mover_elo - opp_elo) / 200, "ctx_ply": ply / 40, "ctx_n_legal": np.log(np.diff(offsets)),
           "ctx_entropy_mbest": entropy(Xb[:, base.index("mbest")], offsets)}
    if "lc0pol" in base:
        ctx["ctx_entropy_lc0pol"] = entropy(Xb[:, base.index("lc0pol")], offsets)
    mn, cn = list(mv), list(ctx)
    return (np.column_stack([mv[c] for c in mn]), mn,
            np.column_stack([np.asarray(ctx[c], dtype=np.float64) for c in cn]), cn)


def design(Xb, base, Xe, en, C, cn, offsets, drop_ctx=()):
    from tnh.eval import trees as T

    kc = [i for i, c in enumerate(cn) if c not in drop_ctx]
    return T.design(np.column_stack([Xb, Xe]), base + en, C[:, kc], [cn[i] for i in kc], offsets,
                    [c for c in PRODUCT_COLS if c in base], True)


def child_columns(value, eff, offsets, keep, rule):
    from tnh.models.prune import pruned_columns

    pc = pruned_columns(value, eff, offsets, keep, "zero")
    if rule == "all":
        pc.pop("lc0ch_unev")
    return pc


# -- loading the feature files ----------------------------------------------------------------

def drop_cache(*paths: Path) -> None:
    """Drop the page cache of files already read: tnh-run charges a unit's page cache to its
    memory cap, and the feature directories hold many times the rows a sample needs."""
    import os

    for p in paths:
        files = sorted(Path(p).glob("part-*.parquet")) if Path(p).is_dir() else [Path(p)]
        for f in files:
            fd = os.open(f, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)


class Loaded:
    """One sample under one configuration: rows, base columns, extras, context."""

    def __init__(self, sample: Path, cfg: dict):
        from tnh.eval.common import attach, ragged, read_sample, window_rows
        from tnh.eval.logit import build
        from tnh.models.prune import MATE_CP, keep_mask

        src = sources(cfg)
        s = window_rows(read_sample(sample))
        # rows of games that left games.parquet after the samples were drawn (near copies removed
        # 2026-11-12): no feature run can replay them now, so every configuration drops them alike
        import polars as pl
        from tnh.paths import CORPUS
        n0 = s.height
        s = s.join(pl.scan_parquet(CORPUS / "games.parquet").select("game_id").collect(), on="game_id", how="semi")
        self.dropped_missing_games = n0 - s.height
        D, _ = build(s, {k: v for k, v in src.items() if k != "lc0ch"}, {})
        if cfg.get("ch"):
            j = attach(s, src["lc0ch"], ["value", "cp", "mate"])
            rv = ragged(j, "value")
            if not np.array_equal(rv.offsets, D.offsets):
                raise ValueError("children and move models disagree on the move lists")
            cp, mate = ragged(j, "cp").values, ragged(j, "mate").values
            eff = np.where(mate != 0, np.sign(mate) * (MATE_CP - 100 * np.abs(mate)), cp)
            prob = np.exp(D.X[:, [D.names.index(c) for c in ("m0", "mbest", "lc0pol")]]).max(1)
            keep = keep_mask(prob, D.offsets, cfg["rule"])
            if np.isnan(rv.values[keep]).any():
                raise ValueError(f"{src['lc0ch']}: a kept child without a value (rule or keep sources differ)")
            D = D.with_columns(child_columns(rv.values, eff, D.offsets, keep, cfg["rule"]))
        drop_cache(sample, CORPUS / "games.parquet", *src.values())
        self.base = base_columns(cfg)
        self.D = D.subset(self.base)
        self.s = s
        self.rows = s[D.rows]
        self.Xe, self.en, self.C, self.cn = extra_features(
            self.D.X, self.base, D.offsets, self.rows.get_column("fen").to_list(), D.moves.to_list(),
            self.rows.get_column("mover_elo").to_numpy().astype(float),
            self.rows.get_column("opp_elo").to_numpy().astype(float), self.rows.get_column("ply").to_numpy().astype(float))


# -- fitting ---------------------------------------------------------------------------------

def _git_commit() -> str:
    from tnh.models.combo import _git_commit as g
    return g()


class Fitter:
    """Linear and tree models for one configuration, and their test log probabilities."""

    def __init__(self, cfg_name: str, fit: Path, test: Path, seed: int, threads: int):
        from tnh.eval.stack import Batches, val_split

        self.name, self.cfg = cfg_name, CONFIGS[cfg_name]
        t = time.perf_counter()
        self.F, self.T_ = Loaded(fit, self.cfg), Loaded(test, self.cfg)
        print(f"[{cfg_name}] loaded in {time.perf_counter() - t:.0f} s: fit {len(self.F.D.w)}, test {len(self.T_.D.w)} "
              f"rows (dropped {self.F.dropped_missing_games} and {self.T_.dropped_missing_games} rows of games no "
              f"longer in games.parquet); base {self.F.base}", flush=True)
        self.seed, self.threads = seed, threads
        Df = self.F.D
        held = val_split(Df.clusters, 0.2, seed)
        self.rows_train, self.rows_val = np.flatnonzero(~held), np.flatnonzero(held)
        B = Batches(Df.offsets)
        ylocal = Df.y - Df.offsets[:-1]

        def subrows(rows):
            idx, _, lens = B.moves_of(rows)
            off = np.r_[0, np.cumsum(lens)]
            return idx, off, off[:-1] + ylocal[rows]
        self.subrows = subrows
        self.tr, self.va = subrows(self.rows_train), subrows(self.rows_val)

    def linear(self, extra: dict | None = None, rows=None):
        """Logit on the base columns (+ `extra` columns: name -> (fit values, test values)); all fitting rows."""
        from tnh.eval.logit import fit as fit_logit

        Df, Dt = self.F.D, self.T_.D
        if extra:
            Df = Df.with_columns({k: v[0] for k, v in extra.items()})
            Dt = Dt.with_columns({k: v[1] for k, v in extra.items()})
        w = None if rows is None else Df.w * np.isin(np.arange(len(Df.w)), rows)
        beta, it = fit_logit(Df, w)
        return beta, Df.names, Df.X @ beta, Dt.X @ beta

    def trees(self, settings: list[dict], rounds: int, patience: int, drop_ctx=(), rows_frac: float = 1.0,
              extra: dict | None = None):
        from tnh.eval import trees as T

        b_tr, cols, init_f, init_t = self.linear(extra, self.rows_train)
        F, Tt = self.F, self.T_
        Xbf = F.D.X if not extra else np.column_stack([F.D.X] + [v[0] for v in extra.values()])
        Xbt = Tt.D.X if not extra else np.column_stack([Tt.D.X] + [v[1] for v in extra.values()])
        Xf, names = design(Xbf, cols, F.Xe, F.en, F.C, F.cn, F.D.offsets, drop_ctx)
        Xt, _ = design(Xbt, cols, Tt.Xe, Tt.en, Tt.C, Tt.cn, Tt.D.offsets, drop_ctx)
        tr, trw = self.tr, F.D.w[self.rows_train]
        if rows_frac < 1:
            tours = np.unique(F.D.clusters[self.rows_train])
            keep_t = tours[np.random.default_rng(self.seed + 11).random(len(tours)) < rows_frac]
            sub = self.rows_train[np.isin(F.D.clusters[self.rows_train], keep_t)]
            tr, trw = self.subrows(sub), F.D.w[sub]
        best, tried = None, []
        for prm in settings:
            t1 = time.perf_counter()
            bst, v = T.train(Xf[tr[0]], tr[1], tr[2], trw, init_f[tr[0]], Xf[self.va[0]], self.va[1], self.va[2],
                             F.D.w[self.rows_val], init_f[self.va[0]], prm, rounds, patience, self.seed,
                             threads=self.threads)
            tried.append({**prm, "trees": bst.best_iteration, "validation": v,
                          "seconds": round(time.perf_counter() - t1)})
            print(f"    [{self.name}] {prm['num_leaves']} leaves, rate {prm['learning_rate']}, min leaf "
                  f"{prm['min_data_in_leaf']}: {bst.best_iteration} trees, validation {v:.4f} "
                  f"({tried[-1]['seconds']} s)", flush=True)
            if best is None or v < best[1]:
                if best is not None:
                    best[0].free_dataset()
                best = (bst, v, prm)
            else:
                bst.free_dataset()
        bst, v, prm = best
        out = {"booster": bst, "names": names, "settings": prm, "tried": tried, "init_beta": b_tr, "columns": cols,
               "score_f": init_f + T.raw(bst, Xf), "score_t": init_t + T.raw(bst, Xt)}
        bst.free_dataset()   # the booster kept its training data (free_raw_data=False): GBs per fit
        return out

    def finish(self, score_f, score_t):
        """eps fitted on the fitting rows; test log probabilities of every move (mixed)."""
        from tnh.eval import trees as T
        from tnh.eval.common import fit_eps

        Df, Dt = self.F.D, self.T_.D
        lpf = T.group_logp(score_f, Df.offsets)
        e = fit_eps(lpf[Df.y], Df.n, Df.w)
        lpt = T.group_logp(score_t, Dt.offsets)
        return e, mix_all(lpt, Dt.offsets, e), mix_all(lpf, Df.offsets, e)


def mix_all(lp: np.ndarray, offsets: np.ndarray, eps: float) -> np.ndarray:
    n = np.repeat(np.diff(offsets), np.diff(offsets))
    return np.log((1 - eps) * np.exp(lp) + eps / n)


def store(name: str, cfg: dict, lin_beta, lin_cols, lin_eps, tr: dict, tr_eps: float, test: dict, fit_info: dict,
          model_dir: Path = MODEL_DIR) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    tr["booster"].save_model(str(model_dir / f"{name}.txt"), num_iteration=tr["booster"].best_iteration)
    meta = {"config": name, "spec": cfg, "base_columns": lin_cols,
            "linear": {"beta": np.asarray(lin_beta).tolist(), "eps": lin_eps},
            "trees": {"init_beta": np.asarray(tr["init_beta"]).tolist(), "eps": tr_eps, "settings": tr["settings"],
                      "n_trees": tr["booster"].best_iteration, "design_names": tr["names"], "tried": tr["tried"]},
            "sources": {k: str(v) for k, v in sources(cfg).items()}, "fit": fit_info, "test": test,
            "date": dt.datetime.now().isoformat(timespec="seconds"), "commit": _git_commit(),
            "refit": f"uv run python -m tnh.models.combotrees fit --configs '{name}' --overwrite"}
    (model_dir / f"{name}.json").write_text(json.dumps(meta, indent=1) + "\n")


def est_paired(loss: np.ndarray, ref: np.ndarray, w, clusters, boot: int) -> tuple[dict, dict]:
    from tnh.eval.common import boot_counts, boot_means, est, wmean

    bm = boot_means(np.column_stack([loss, loss - ref]), w, clusters, boot_counts(clusters.max() + 1, boot, 7))
    return est(wmean(loss, w), bm[:, 0]), est(wmean(loss - ref, w), bm[:, 1])


def cmd_fit(a) -> None:
    """Every configuration: linear and trees, stored; test log loss paired against linear Combo p0.5."""
    from tnh.eval import trees as T

    model_dir = a.out.resolve()
    if model_dir == MODEL_DIR.resolve() and not a.overwrite:
        stored = [n for n in a.configs if (MODEL_DIR / f"{n}.json").exists() or (MODEL_DIR / f"{n}.txt").exists()]
        if stored:
            raise SystemExit(f"{MODEL_DIR} already holds {', '.join(stored)}: pass --out DIR to write elsewhere, "
                             "or --overwrite to replace the stored models")
    ref = None
    out = {}
    settings = T.grid(a.leaves, a.rates, a.min_leaf)
    for name in a.configs:
        fz = Fitter(name, a.fit, a.test, a.seed, a.threads)
        beta, cols, sf_, st_ = fz.linear()
        e_lin, lpt_lin, _ = fz.finish(sf_, st_)
        tr = fz.trees(settings, a.rounds, a.patience)
        e_tr, lpt_tr, _ = fz.finish(tr["score_f"], tr["score_t"])
        Dt = fz.T_.D
        losses = {"linear": -lpt_lin[Dt.y], "trees": -lpt_tr[Dt.y]}
        if ref is None:
            if name != MAIN:
                ref_cfg = Fitter(MAIN, a.fit, a.test, a.seed, a.threads)
                b2, _, s2f, s2t = ref_cfg.linear()
                e2, lpt2, _ = ref_cfg.finish(s2f, s2t)
                ref = -lpt2[ref_cfg.T_.D.y]
                del ref_cfg, s2f, s2t, lpt2   # holds a second copy of both samples
            else:
                ref = losses["linear"]
        res = {}
        for k, l in losses.items():
            if len(l) != len(ref):
                raise ValueError("test rows differ between configurations")
            t_, d_ = est_paired(l, ref, Dt.w, Dt.clusters, a.boot)
            res[k] = {"test": t_, "minus_linear_combo_p0.5": d_}
        out[name] = res
        print(f"[{name}] linear {res['linear']['test']['est']:.4f} ({res['linear']['minus_linear_combo_p0.5']['est']:+.4f}), "
              f"trees {res['trees']['test']['est']:.4f} ({res['trees']['minus_linear_combo_p0.5']['est']:+.4f}); "
              f"{tr['booster'].best_iteration} trees", flush=True)
        store(name, fz.cfg, beta, cols, e_lin, tr, e_tr, res,
              {"sample": str(a.fit), "rows": int(len(fz.F.D.w)), "train_rows": int(len(fz.rows_train)),
               "validation_rows": int(len(fz.rows_val)), "test_sample": str(a.test), "test_rows": int(len(Dt.w))},
              model_dir)
        print(f"stored {model_dir / name}.json and .txt", flush=True)
        del fz
    print("\n| configuration | linear | minus linear Combo p0.5 | trees | minus linear Combo p0.5 |\n|---|---|---|---|---|")
    for n, r in out.items():
        f = lambda e: f"{e['est']:.4f} [{e['lo']:.4f}, {e['hi']:.4f}]"  # noqa: E731
        fd = lambda e: f"{e['est']:+.4f} [{e['lo']:+.4f}, {e['hi']:+.4f}]"  # noqa: E731
        print(f"| {n} | {f(r['linear']['test'])} | {fd(r['linear']['minus_linear_combo_p0.5'])} | "
              f"{f(r['trees']['test'])} | {fd(r['trees']['minus_linear_combo_p0.5'])} |")


# -- analysis of the main configuration --------------------------------------------------------

def band_codes(elo: np.ndarray) -> np.ndarray:
    c = np.full(len(elo), -1)
    for i, (lo, hi) in enumerate(RATING_BANDS):
        c[(elo >= lo) & (elo < hi)] = i
    return c


def ratio_table(obs: np.ndarray, masses: dict[str, np.ndarray], w, clusters, bands, ok, boot) -> dict:
    """Mass / observed (both weighted means over rows), with a tournament-bootstrap interval of the ratio."""
    from tnh.eval.common import boot_counts, boot_means, wmean

    res = {}
    for band in ["all"] + BAND_LABELS:
        m = ok if band == "all" else ok & (bands == BAND_LABELS.index(band))
        cl = np.unique(clusters[m], return_inverse=True)[1]
        M = np.column_stack([obs[m]] + [v[m] for v in masses.values()])
        bm = boot_means(M, w[m], cl, boot_counts(cl.max() + 1, boot, 7))
        o = wmean(obs[m], w[m])
        row = {"rows": int(m.sum()), "observed": o}
        for i, (k, v) in enumerate(masses.items()):
            mass = wmean(v[m], w[m])
            r = bm[:, i + 1] / bm[:, 0]
            row[k] = {"mass": mass, "ratio": mass / o, "lo": float(np.quantile(r, 0.025)),
                      "hi": float(np.quantile(r, 0.975))}
        res[band] = row
    return res


def reliability(lp_all: np.ndarray, offsets: np.ndarray, y: np.ndarray, w: np.ndarray, bands: np.ndarray) -> dict:
    """Predicted probability against how often the move is played: all legal moves, and the model's top move."""
    edges = np.array([0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0000001])
    n = np.diff(offsets)
    p = np.exp(lp_all)
    played = np.zeros(len(p))
    played[y] = 1
    wm = np.repeat(w, n)
    top = np.maximum.reduceat(p, offsets[:-1])
    top_idx = offsets[:-1] + np.array([int(np.argmax(p[a:b])) for a, b in zip(offsets[:-1], offsets[1:])])
    top_played = (top_idx == y).astype(float)
    out = {}
    for band in ["all"] + BAND_LABELS:
        rm = np.ones(len(n), bool) if band == "all" else bands == BAND_LABELS.index(band)
        mm = np.repeat(rm, n)
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = mm & (p >= lo) & (p < hi)
            st = rm & (top >= lo) & (top < hi)
            bins.append({"bin": f"{lo:.2f}-{min(hi, 1):.2f}",
                         "moves": int(s.sum()),
                         "pred": float(np.average(p[s], weights=wm[s])) if s.any() else None,
                         "played": float(np.average(played[s], weights=wm[s])) if s.any() else None,
                         "top_positions": int(st.sum()),
                         "top_pred": float(np.average(top[st], weights=w[st])) if st.any() else None,
                         "top_played": float(np.average(top_played[st], weights=w[st])) if st.any() else None})
        thr = {}
        for t in (0.8, 0.9):
            st = rm & (top >= t)
            thr[str(t)] = {"share_of_positions": float(np.average(st[rm], weights=w[rm])),
                           "mean_pred": float(np.average(top[st], weights=w[st])) if st.any() else None,
                           "played": float(np.average(top_played[st], weights=w[st])) if st.any() else None}
        out[band] = {"bins": bins, "threshold": thr}
    return out


def fit_temperature(score: np.ndarray, offsets, y, w, groups=None, n_groups=1) -> np.ndarray:
    """Per group, the T minimising the weighted log loss of softmax(score / T) on the given rows."""
    from scipy.optimize import minimize_scalar
    from tnh.eval import trees as T

    g = np.zeros(len(y), int) if groups is None else groups
    out = np.ones(n_groups)
    n = np.diff(offsets)
    for k in range(n_groups):
        rows = np.flatnonzero(g == k)
        if not len(rows):
            continue
        mv = np.concatenate([np.arange(offsets[r], offsets[r + 1]) for r in rows])
        off = np.r_[0, np.cumsum(n[rows])]
        yy = off[:-1] + (y[rows] - offsets[rows])
        s = score[mv]

        def nll(logt):
            return -np.average(T.group_logp(s / np.exp(logt), off)[yy], weights=w[rows])
        out[k] = float(np.exp(minimize_scalar(nll, bounds=(-1.5, 1.5), method="bounded").x))
    return out


def move_labels(L: "Loaded") -> dict:
    """Per move of a loaded sample: Stockfish 10k and 100k losses, Lc0's full child loss and policy."""
    from tnh.eval.common import attach, ragged
    from tnh.eval.logit import nanmax_segments

    out = {}
    for k, d in (("sf10k", FEATURES / "stockfish" / "nodes10000"), ("sf100k", FEATURES / "stockfish" / "nodes100000")):
        j = attach(L.rows, d, ["value"])
        r = ragged(j, "value")
        best = nanmax_segments(r, r.values)
        out[f"{k}_best"] = best
        out[f"{k}_L"] = np.repeat(best, np.diff(r.offsets)) - r.values
        out[f"{k}_nan_row"] = np.add.reduceat(np.isnan(r.values).astype(int), r.offsets[:-1]) > 0
    j = attach(L.rows, FEATURES / "lc0-bt4" / "children", ["value"])
    drop_cache(FEATURES / "stockfish" / "nodes10000", FEATURES / "stockfish" / "nodes100000",
               FEATURES / "lc0-bt4" / "children")
    r = ragged(j, "value")
    out["lc0_L"] = np.repeat(np.maximum.reduceat(r.values, r.offsets[:-1]), np.diff(r.offsets)) - r.values
    out["lc0pol_p"] = np.exp(L.D.X[:, L.base.index("lc0pol")])
    return out


def move_classes(lab: dict) -> dict:
    Ld, Ls = np.nan_to_num(lab["sf100k_L"], nan=0.0), np.nan_to_num(lab["sf10k_L"], nan=0.0)
    mist = Ld > 0.03
    return {"sound (deep loss <= 0.03)": ~mist, "mistake visible at 10k": mist & (Ls > 0.03),
            "mistake only deep": mist & ~(Ls > 0.03), "large mistake (> 0.25)": Ld > 0.25,
            "looks bad to Lc0's raw child value (> 0.03)": lab["lc0_L"] > 0.03,
            "looks fine to Lc0's raw child value": lab["lc0_L"] <= 0.03,
            "Lc0 policy < 5 %": lab["lc0pol_p"] < 0.05, "Lc0 policy >= 5 %": lab["lc0pol_p"] >= 0.05}


def bias_tables(models: dict, lab: dict, D, rows, boot: int) -> tuple[dict, dict]:
    """Mass / observed per move class and the expected loss per move; the rows used."""
    bands = band_codes(rows.get_column("mover_elo").to_numpy())
    decided = (lab["sf100k_best"] > 0.97) | (lab["sf100k_best"] < 0.03)
    ok = ~decided & ~lab["sf100k_nan_row"] & ~lab["sf10k_nan_row"]
    bias = {}
    for cname, mask in move_classes(lab).items():
        obs = mask[D.y].astype(float)
        masses = {k: np.add.reduceat(np.exp(lp) * mask, D.offsets[:-1]) for k, lp in models.items()}
        bias[cname] = ratio_table(obs, masses, D.w, D.clusters, bands, ok, boot)
    L100 = np.nan_to_num(lab["sf100k_L"], nan=0.0)
    masses = {k: np.add.reduceat(np.exp(lp) * L100, D.offsets[:-1]) for k, lp in models.items()}
    bias["expected loss per move (Stockfish 100k)"] = ratio_table(L100[D.y], masses, D.w, D.clusters, bands, ok, boot)
    return bias, {"rows_in_bias_tables": int(ok.sum()), "rows_dropped_decided": int(decided.sum())}


def cmd_analyse(a) -> None:
    from tnh.eval import trees as T
    from tnh.eval.common import attach

    fz = Fitter(MAIN, a.fit, a.test, a.seed, a.threads)
    Df, Dt = fz.F.D, fz.T_.D
    settings = [json.loads((MODEL_DIR / f"{MAIN}.json").read_text())["trees"]["settings"]] \
        if (MODEL_DIR / f"{MAIN}.json").exists() else T.grid([15], [0.05], [2000])

    labf, labt = move_labels(fz.F), move_labels(fz.T_)

    # the models: Maia +600, linear Combo p0.5, trees; then the debiasing candidates
    models_t, models_f = {}, {}
    models_t["Maia +600"] = T.group_logp(Dt.X[:, fz.T_.base.index("mbest")], Dt.offsets)
    b, cols, sf_, st_ = fz.linear()
    e, lpt, lpf = fz.finish(sf_, st_)
    models_t["linear Combo p0.5"], models_f["linear Combo p0.5"] = lpt, lpf
    scores = {"linear Combo p0.5": (sf_, st_)}
    tr = fz.trees(settings, a.rounds, a.patience)
    e, lpt, lpf = fz.finish(tr["score_f"], tr["score_t"])
    models_t["trees"], models_f["trees"] = lpt, lpf
    scores["trees"] = (tr["score_f"], tr["score_t"])
    del tr

    # (a) debiasing features that are free: Lc0 classes as indicators, and x rating
    def cand(L, lab):
        z = np.repeat(L.C[:, L.cn.index("ctx_mover_elo")], np.diff(L.D.offsets))
        bad = ((L.D.X[:, L.base.index("lc0ch_eloss")] > 0.03) & (L.D.X[:, L.base.index("lc0ch_unev")] == 0))
        low = (np.exp(L.D.X[:, L.base.index("lc0pol")]) < 0.05)
        out = {"ind_lc0bad": bad.astype(float), "ind_lowpol": low.astype(float)}
        out.update({f"{k}*elo": v * z for k, v in list(out.items())})
        for c in ("lc0ch_eloss", "lc0ch_cploss", "mbest", "lc0pol", "lc0ch_unev"):
            out[f"{c}*elo"] = L.D.X[:, L.base.index(c)] * z
        return out
    cf, ct = cand(fz.F, labf), cand(fz.T_, labt)
    extra = {k: (cf[k], ct[k]) for k in cf}
    b, cols_d, sf_, st_ = fz.linear(extra)
    e, lpt, lpf = fz.finish(sf_, st_)
    models_t["linear + class indicators x rating"] = lpt
    trd = fz.trees(settings, a.rounds, a.patience, extra=extra)
    e, lpt, lpf = fz.finish(trd["score_f"], trd["score_t"])
    models_t["trees + class indicators x rating"] = lpt
    # (b) one cheap Stockfish input: 10k nodes, its two loss columns, not in the default model
    from tnh.eval.logit import engine_columns

    def sfcols(L):
        j = attach(L.rows, FEATURES / "stockfish" / "nodes10000", ["value", "cp", "mate"])
        c, _, _ = engine_columns(j, "sf10k")
        return c
    sc_f, sc_t = sfcols(fz.F), sfcols(fz.T_)
    drop_cache(FEATURES / "stockfish" / "nodes10000")
    extra_sf = {k: (sc_f[k], sc_t[k]) for k in ("sf10k_eloss", "sf10k_cploss")}
    b, _, sf_, st_ = fz.linear(extra_sf)
    e, lpt, _ = fz.finish(sf_, st_)
    models_t["linear + Stockfish 10k (not default)"] = lpt
    trs = fz.trees(settings, a.rounds, a.patience, extra=extra_sf)
    e, lpt, _ = fz.finish(trs["score_f"], trs["score_t"])
    models_t["trees + Stockfish 10k (not default)"] = lpt

    # log loss of each, paired against linear Combo p0.5
    ref = -models_t["linear Combo p0.5"][Dt.y]
    loss_tab = {}
    for k, lp in models_t.items():
        if k == "Maia +600":
            from tnh.eval.common import fit_eps
            ef = fit_eps(T.group_logp(Df.X[:, fz.F.base.index("mbest")], Df.offsets)[Df.y], Df.n, Df.w)
            lp = mix_all(lp, Dt.offsets, ef)
        t_, d_ = est_paired(-lp[Dt.y], ref, Dt.w, Dt.clusters, a.boot)
        loss_tab[k] = {"test": t_, "minus_linear": d_}

    # bias by class: mass on the class against how often the played move is in it (raw probabilities)
    bands = band_codes(fz.T_.rows.get_column("mover_elo").to_numpy())
    bias, counts = bias_tables(models_t, labt, Dt, fz.T_.rows, a.boot)

    # reliability, before and after a temperature (global, and per rating band), for linear and trees
    rel = {}
    bands_f = band_codes(fz.F.rows.get_column("mover_elo").to_numpy())
    for k in ("linear Combo p0.5", "trees"):
        rel[k] = reliability(models_t[k], Dt.offsets, Dt.y, Dt.w, bands)
        sfit, stest = scores[k]
        tg = fit_temperature(sfit, Df.offsets, Df.y, Df.w)
        tb = fit_temperature(sfit, Df.offsets, Df.y, Df.w, bands_f, len(RATING_BANDS))
        lp_g = T.group_logp(stest / tg[0], Dt.offsets)
        per_move_t = np.repeat(np.where(bands >= 0, tb[np.maximum(bands, 0)], 1.0), np.diff(Dt.offsets))
        lp_b = T.group_logp(stest / per_move_t, Dt.offsets)
        rel[f"{k}, temperature {tg[0]:.3f}"] = reliability(lp_g, Dt.offsets, Dt.y, Dt.w, bands)
        rel[f"{k}, temperature per band {np.round(tb, 3).tolist()}"] = reliability(lp_b, Dt.offsets, Dt.y, Dt.w, bands)
        for lab_, lp_ in ((f"{k}, temperature {tg[0]:.3f}", lp_g), (f"{k}, temperature per band", lp_b)):
            t_, d_ = est_paired(-lp_[Dt.y], ref, Dt.w, Dt.clusters, a.boot)
            loss_tab[lab_] = {"test": t_, "minus_linear": d_}

    out = {"config": MAIN, "test_rows": int(len(Dt.w)), **counts, "log_loss": loss_tab, "bias": bias,
           "reliability": rel, "debias_features": list(cf), "date": dt.datetime.now().isoformat(timespec="seconds"),
           "commit": _git_commit()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    print_analysis(out)


def stored_logp(name: str, L: "Loaded", kind: str, model_dir: Path = MODEL_DIR) -> np.ndarray:
    """A stored model (linear or trees) on a loaded sample's feature-file rows."""
    import lightgbm as lgb
    from tnh.eval import trees as T

    meta = json.loads((model_dir / f"{name}.json").read_text())
    if meta["base_columns"] != L.base:
        raise ValueError(f"{name}: stored base columns {meta['base_columns']}, loaded {L.base}")
    D = L.D
    if kind == "linear":
        s, eps = D.X @ np.array(meta["linear"]["beta"]), meta["linear"]["eps"]
    else:
        Xd, names = design(D.X, L.base, L.Xe, L.en, L.C, L.cn, D.offsets)
        if names != meta["trees"]["design_names"]:
            raise ValueError(f"{name}: design columns differ from the stored model's")
        booster = lgb.Booster(model_file=str(model_dir / f"{name}.txt"))
        s = D.X @ np.array(meta["trees"]["init_beta"]) + booster.predict(Xd, raw_score=True)
        eps = meta["trees"]["eps"]
    return mix_all(T.group_logp(s, D.offsets), D.offsets, eps)


def cmd_calib(a) -> None:
    """Bias by move class and reliability of stored models on the test sample (no fitting)."""
    from tnh.eval import trees as T

    models, ref, lab, D0, rows0, bands = {}, None, None, None, None, None
    for name in a.configs:
        L = Loaded(a.test, CONFIGS[name])
        key = L.rows.select("game_id", "ply")
        if D0 is None:
            D0, rows0, keys0 = L.D, L.rows, key
            lab = move_labels(L) if "lc0pol" in L.base else None
            models["Maia +600"] = T.group_logp(L.D.X[:, L.base.index("mbest")], L.D.offsets)
        elif not key.equals(keys0) or not np.array_equal(L.D.offsets, D0.offsets):
            raise ValueError(f"{name}: test rows differ from {a.configs[0]}'s")
        for kind in ("linear", "trees"):
            models[f"{kind} {name}"] = stored_logp(name, L, kind)
        if name == MAIN:
            ref = models[f"linear {name}"][D0.y]
        del L
    if lab is None:
        raise ValueError("the first configuration must have the Lc0 policy (the classes use it)")
    loss_tab = {}
    for k, lp in models.items():
        if k == "Maia +600" or ref is None:
            continue
        t_, d_ = est_paired(-lp[D0.y], -ref, D0.w, D0.clusters, a.boot)
        loss_tab[k] = {"test": t_, "minus_linear": d_}
    bias, counts = bias_tables(models, lab, D0, rows0, a.boot)
    bands = band_codes(rows0.get_column("mover_elo").to_numpy())
    rel = {k: reliability(lp, D0.offsets, D0.y, D0.w, bands) for k, lp in models.items() if k != "Maia +600"}
    out = {"configs": a.configs, "test_rows": int(len(D0.w)), **counts, "log_loss": loss_tab, "bias": bias,
           "reliability": rel, "date": dt.datetime.now().isoformat(timespec="seconds"), "commit": _git_commit()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    print_analysis(out)


def print_analysis(o: dict) -> None:
    f = lambda e: f"{e['est']:.4f} [{e['lo']:.4f}, {e['hi']:.4f}]"  # noqa: E731
    fd = lambda e: f"{e['est']:+.4f} [{e['lo']:+.4f}, {e['hi']:+.4f}]"  # noqa: E731
    print("\n| model | test log loss | minus linear Combo p0.5 |\n|---|---|---|")
    for k, v in o["log_loss"].items():
        print(f"| {k} | {f(v['test'])} | {fd(v['minus_linear'])} |")
    for cname, tab in o["bias"].items():
        ms = [k for k in tab["all"] if k not in ("rows", "observed")]
        print(f"\n**{cname}**: mass / observed (1 = unbiased)\n\n| band | rows | observed | " + " | ".join(ms)
              + " |\n|" + "---|" * (len(ms) + 3))
        for band, v in tab.items():
            print(f"| {band} | {v['rows']} | {v['observed']:.4f} | "
                  + " | ".join(f"{v[m]['ratio']:.2f} [{v[m]['lo']:.2f}, {v[m]['hi']:.2f}]" for m in ms) + " |")
    for k, r in o["reliability"].items():
        print(f"\n**reliability, {k}**")
        print("\n| band | bin | moves | predicted | played | top-move positions | top predicted | top played |"
              "\n|---|---|---|---|---|---|---|---|")
        for band, v in r.items():
            for b in v["bins"]:
                if b["bin"] in ("0.70-0.80", "0.80-0.90", "0.90-0.95", "0.95-1.00") or band == "all":
                    fmt = lambda x: "" if x is None else f"{x:.3f}"  # noqa: E731
                    print(f"| {band} | {b['bin']} | {b['moves']} | {fmt(b['pred'])} | {fmt(b['played'])} | "
                          f"{b['top_positions']} | {fmt(b['top_pred'])} | {fmt(b['top_played'])} |")
        print("\n| band | top >= 0.8: share of positions | mean predicted | played | top >= 0.9: share | mean predicted "
              "| played |\n|---|---|---|---|---|---|---|")
        for band, v in r.items():
            t8, t9 = v["threshold"]["0.8"], v["threshold"]["0.9"]
            fmt = lambda x: "" if x is None else f"{x:.3f}"  # noqa: E731
            print(f"| {band} | {t8['share_of_positions']:.3f} | {fmt(t8['mean_pred'])} | {fmt(t8['played'])} | "
                  f"{t9['share_of_positions']:.3f} | {fmt(t9['mean_pred'])} | {fmt(t9['played'])} |")


def cmd_report(a) -> None:
    print_analysis(json.loads(a.out.read_text()))


def main(argv=None) -> None:
    from tnh.models.combo import FIT_SAMPLE, TEST_SAMPLE

    p = argparse.ArgumentParser(prog="python -m tnh.models.combotrees", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("fit", cmd_fit), ("analyse", cmd_analyse)):
        sp = sub.add_parser(name)
        sp.add_argument("--fit", type=Path, default=FIT_SAMPLE)
        sp.add_argument("--test", type=Path, default=TEST_SAMPLE)
        sp.add_argument("--boot", type=int, default=1000)
        sp.add_argument("--seed", type=int, default=7)
        sp.add_argument("--leaves", type=int, nargs="+", default=[15])
        sp.add_argument("--rates", type=float, nargs="+", default=[0.05])
        sp.add_argument("--min-leaf", type=int, nargs="+", default=[2000])
        sp.add_argument("--rounds", type=int, default=2000)
        sp.add_argument("--patience", type=int, default=50)
        sp.add_argument("--threads", type=int, default=1)
        sp.add_argument("--configs", nargs="+", default=[MAIN])
        if name == "fit":
            sp.add_argument("--out", type=Path, default=MODEL_DIR,
                            help="directory for the stored models (default: the repo's, which needs --overwrite)")
            sp.add_argument("--overwrite", action="store_true", help="replace models stored in the repo")
        else:
            sp.add_argument("--out", type=Path,
                            default=Path.home() / "out" / "tn-hittare" / "scratch" / "models" / "combotrees" / "analyse.json")
        sp.set_defaults(fn=fn)
    sp = sub.add_parser("calib", help="bias and reliability of stored models on the test sample")
    sp.add_argument("--test", type=Path, default=TEST_SAMPLE)
    sp.add_argument("--boot", type=int, default=1000)
    sp.add_argument("--configs", nargs="+", default=["+pol", MAIN])
    sp.add_argument("--out", type=Path,
                    default=Path.home() / "out" / "tn-hittare" / "scratch" / "models" / "combotrees" / "calib.json")
    sp.set_defaults(fn=cmd_calib)
    sp = sub.add_parser("report")
    sp.add_argument("--out", type=Path,
                    default=Path.home() / "out" / "tn-hittare" / "scratch" / "models" / "combotrees" / "analyse.json")
    sp.set_defaults(fn=cmd_report)
    a = p.parse_args(argv)
    a.fn(a)




# -- live --------------------------------------------------------------------------------------

class FrontModel:
    """A stored configuration (linear and trees) applied to features computed live."""

    def __init__(self, name: str):
        import lightgbm as lgb

        self.name = name
        self.meta = json.loads((MODEL_DIR / f"{name}.json").read_text())
        self.cfg = self.meta["spec"]
        self.base = self.meta["base_columns"]
        self.booster = lgb.Booster(model_file=str(MODEL_DIR / f"{name}.txt"))

    def features(self, queries, maia: tuple, pol, children, only):
        """Per query moves, and the stacked base matrix, extras and context for the batch.

        `maia` = (m0, mbest) predictions, `pol` Lc0 policy predictions or None, `children`
        MoveValues or None (with `only`, the kept sets, None for all children)."""
        from tnh.models.prune import MATE_CP

        moves = [p.moves for p in maia[0]]
        cols = {c: [] for c in ("m0", "mbest", "lc0pol")}
        for i, ms in enumerate(moves):
            for c, preds in (("m0", maia[0]), ("mbest", maia[1]), ("lc0pol", pol)):
                if preds is None:
                    continue
                idx = {m: k for k, m in enumerate(preds[i].moves)}
                if set(idx) != set(ms):
                    raise ValueError(f"models disagree on the legal moves at {queries[i].fen}")
                cols[c].append(preds[i].logp[[idx[m] for m in ms]].astype(np.float64))
        offsets = np.r_[0, np.cumsum([len(m) for m in moves])].astype(np.int64)
        X = {c: np.concatenate(v) for c, v in cols.items() if v}
        if self.cfg.get("ch"):
            value, cp, mate, keep = [], [], [], []
            for i, ms in enumerate(moves):
                v = children[i]
                idx = {m: k for k, m in enumerate(v.moves)}
                order = [idx[m] for m in ms]
                value.append(v.value[order].astype(np.float64))
                cp.append(v.cp[order].astype(np.float64))
                mate.append(v.mate[order].astype(np.float64))
                keep.append(np.ones(len(ms), bool) if only is None else np.array([m in only[i] for m in ms]))
            value, cp, mate, keep = (np.concatenate(x) for x in (value, cp, mate, keep))
            eff = np.where(mate != 0, np.sign(mate) * (MATE_CP - 100 * np.abs(mate)), cp)
            X.update(child_columns(value, eff, offsets, keep, self.cfg["rule"]))
        Xb = np.column_stack([X[c] for c in self.base])
        fens = [q.fen for q in queries]
        Xe, en, C, cn = extra_features(Xb, self.base, offsets, fens, moves,
                                       np.array([q.mover_elo for q in queries], float),
                                       np.array([q.opp_elo for q in queries], float),
                                       np.array([len(q.moves_uci) for q in queries], float))
        return moves, offsets, Xb, Xe, en, C, cn

    def logp(self, feats, kind: str) -> list[np.ndarray]:
        from tnh.eval import trees as T

        moves, offsets, Xb, Xe, en, C, cn = feats
        if kind == "linear":
            s = Xb @ np.array(self.meta["linear"]["beta"])
            eps = self.meta["linear"]["eps"]
        else:
            Xd, names = design(Xb, self.base, Xe, en, C, cn, offsets)
            if names != self.meta["trees"]["design_names"]:
                raise ValueError("live design columns differ from the stored model's")
            s = Xb @ np.array(self.meta["trees"]["init_beta"]) + self.booster.predict(Xd, raw_score=True)
            eps = self.meta["trees"]["eps"]
        lp = mix_all(T.group_logp(s, offsets), offsets, eps)
        return [lp[a:b].astype(np.float32) for a, b in zip(offsets[:-1], offsets[1:])]


def keep_sets(maia, pol, rule) -> list[set[str]]:
    from tnh.models.prune import keep_mask

    out = []
    for i in range(len(maia[0])):
        prob = {}
        for preds in (maia[0], maia[1], pol):
            for m, x in zip(preds[i].moves, np.exp(preds[i].logp.astype(np.float64))):
                prob[m] = max(prob.get(m, 0.0), x)
        ms = list(prob)
        k = keep_mask(np.array([prob[m] for m in ms]), np.array([0, len(ms)]), rule)
        out.append({m for m, kk in zip(ms, k) if kk})
    return out


def cmd_front(a) -> None:
    """Live: positions per second against test log loss for every stored configuration, linear and trees."""
    import os

    from tnh.eval.common import boot_counts, boot_means, cluster_codes, est, wmean
    from tnh.models.combo import _GpuSampler, _queries
    from tnh.models.lc0 import Lc0Children, Lc0Policy
    from tnh.models.maia3 import Elo, Maia3

    s, qs = _queries(a.test, a.n, a.seed)
    played = s.get_column("played_uci").to_list()
    w = s.get_column("weight").to_numpy()
    cl = cluster_codes(s)
    counts = boot_counts(cl.max() + 1, a.boot, 7)
    print(f"{len(qs)} test window positions; load {os.getloadavg()}", flush=True)
    seconds, busy, out = {}, {}, {}

    def timed(name, f):
        with _GpuSampler() as g:
            t = time.perf_counter()
            r = f()
            seconds[name] = time.perf_counter() - t
        busy[name] = g.mean
        print(f"  {name}: {seconds[name]:.1f} s ({len(qs) / seconds[name]:.0f} positions/s), GPU busy {g.mean:.0f} %",
              flush=True)
        return r

    elos = [Elo.offset(0.0), Elo.offset(600.0)]
    maia = {}
    for size in a.maia_sizes:
        m = Maia3(size, batch_size=512)
        m.predict_many(qs[:64], elos)
        maia[size] = timed(f"maia {size}", lambda m=m: m.predict_many(qs, elos))
        del m
    pols = {}
    for net, file in (("bt4", "BT4-1024x15x32h-swa-6147500.pb.gz"), ("t3", "t3-512x15x16h-distill2-swa-03477500.pb.gz")):
        if net not in a.nets:
            continue
        p = Lc0Policy(file)
        p.predict(qs[:24])
        pols[net] = timed(f"lc0 {net} policy", lambda p=p: p.predict(qs))
        del p
    names = [n for n in a.configs if (MODEL_DIR / f"{n}.json").exists()]
    children = {}
    for n in names:
        cfg = CONFIGS[n]
        if not cfg.get("ch"):
            continue
        key = (cfg["ch"], cfg["rule"])
        if key in children:
            continue
        net_file = "BT4-1024x15x32h-swa-6147500.pb.gz" if cfg["ch"] == "bt4" else "t3-512x15x16h-distill2-swa-03477500.pb.gz"
        only = None if cfg["rule"] == "all" else keep_sets(maia[cfg["maia"]], pols[cfg["pol"]], cfg["rule"])
        ch = Lc0Children(net_file)
        ch.evaluate(qs[:8], None if only is None else only[:8])
        children[key] = (timed(f"lc0 {cfg['ch']} children {cfg['rule']}", lambda ch=ch, only=only: ch.evaluate(qs, only)), only)
        del ch
    losses = {}
    for n in names:
        fm = FrontModel(n)
        cfg = fm.cfg
        parts = [f"maia {cfg['maia']}"] + ([f"lc0 {cfg['pol']} policy"] if cfg.get("pol") else []) \
            + ([f"lc0 {cfg['ch']} children {cfg['rule']}"] if cfg.get("ch") else [])
        ch, only = children.get((cfg.get("ch"), cfg.get("rule")), (None, None))
        t = time.perf_counter()
        feats = fm.features(qs, maia[cfg["maia"]], pols.get(cfg.get("pol")), ch, only)
        t_feat = time.perf_counter() - t
        for kind in ("linear", "trees"):
            t = time.perf_counter()
            lps = fm.logp(feats, kind)
            t_score = time.perf_counter() - t
            loss = np.array([-float(lp[ms.index(pl_)]) for lp, ms, pl_ in zip(lps, feats[0], played)])
            losses[(n, kind)] = loss
            pps = len(qs) / (sum(seconds[p] for p in parts) + t_feat + t_score)
            out[(n, kind)] = {"positions_per_s": pps, "parts": parts, "cpu_features_s": t_feat, "score_s": t_score}
    ref = losses.get((MAIN, "linear"))
    for k, loss in losses.items():
        cols = [loss] + ([loss - ref] if ref is not None else [])
        bm = boot_means(np.column_stack(cols), w, cl, counts)
        out[k]["test"] = est(wmean(loss, w), bm[:, 0])
        if ref is not None:
            out[k]["minus_linear_combo_p0.5"] = est(wmean(loss - ref, w), bm[:, 1])
    # the front: a point is dominated if another is at least as fast and strictly better (or vice versa)
    pts = {k: (v["positions_per_s"], v["test"]["est"]) for k, v in out.items()}
    for k, (sp, ll) in pts.items():
        out[k]["front"] = not any((s2 >= sp and l2 < ll) or (s2 > sp and l2 <= ll) for k2, (s2, l2) in pts.items() if k2 != k)
    print("\n| configuration | model | positions/s | test log loss | minus linear Combo p0.5 | on the front |"
          "\n|---|---|---|---|---|---|")
    for (n, kind), v in sorted(out.items(), key=lambda kv: -kv[1]["positions_per_s"]):
        d = v.get("minus_linear_combo_p0.5")
        print(f"| {n} | {kind} | {v['positions_per_s']:.1f} | {v['test']['est']:.4f} [{v['test']['lo']:.4f}, "
              f"{v['test']['hi']:.4f}] | " + (f"{d['est']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]" if d else "")
              + f" | {'yes' if v['front'] else 'dominated'} |")
    print("\ncomponents: " + ", ".join(f"{k} {len(qs) / v:.0f} pos/s (GPU {busy[k]:.0f} %)" for k, v in seconds.items()))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"rows": len(qs), "load": os.getloadavg(), "components_s": seconds, "gpu_busy": busy,
                                 "points": {f"{n} {k}": v for (n, k), v in out.items()}}, indent=1) + "\n")


def gpu_mib(pid: int) -> int | None:
    """GPU memory this process holds (nvidia-smi), MiB; None where nvidia-smi does not list it."""
    import subprocess
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        p, _, mib = line.partition(",")
        if p.strip() == str(pid):
            return int(mib)
    return None


def live_model(config: str):
    """The live model of a stored configuration: `PolicyTrees` for +pol, `Combo(..., trees=True)` for the main one."""
    from tnh.models.combo import Combo, PolicyTrees
    if config == PolicyTrees.CONFIG:
        return PolicyTrees()
    if config == MAIN:
        return Combo(prune=RULE_MAIN, trees=True)
    raise ValueError(f"no live model for {config!r} (live: {PolicyTrees.CONFIG}, {MAIN})")


def cmd_check_live(a) -> None:
    """A live model against the same rows scored by its stored model from the feature files."""
    import os

    from tnh.models.combo import _GpuSampler, _queries

    s, qs = _queries(a.test, a.n, a.seed)
    L = Loaded(a.test, CONFIGS[a.config])
    key = {(g, p): i for i, (g, p) in enumerate(L.rows.select("game_id", "ply").iter_rows())}
    idx = [key[(g, p)] for g, p in s.select("game_id", "ply").iter_rows()]
    D = L.D
    lp_off = stored_logp(a.config, L, "trees")
    model = live_model(a.config)
    model.predict(qs[:16])          # warm-up: CUDA contexts, kernels
    model.seconds.clear()
    with _GpuSampler() as g:
        t = time.perf_counter()
        preds = model.predict(qs)
        dt_ = time.perf_counter() - t
    mib = gpu_mib(os.getpid())
    worst, worst_p, worst_big, ll_live, ll_off = 0.0, 0.0, 0.0, [], []
    worst_at = None
    moves_all = D.moves.to_list()
    played = s.get_column("played_uci").to_list()
    for q, p, i, pl_ in zip(qs, preds, idx, played):
        if p.wdl is None:
            raise ValueError(f"{model.name} gave no wdl at {q.fen}")
        o = dict(zip(moves_all[i], lp_off[D.offsets[i]:D.offsets[i + 1]]))
        for m, x in zip(p.moves, p.logp):
            d = abs(float(x) - o[m])
            if d > worst:
                worst, worst_at = d, (q.fen, m, float(np.exp(o[m])), float(np.exp(x)))
            worst_p = max(worst_p, abs(float(np.exp(x)) - float(np.exp(o[m]))))
            if max(float(np.exp(x)), float(np.exp(o[m]))) >= 0.01:
                worst_big = max(worst_big, d)
        ll_live.append(-float(p.logp[p.moves.index(pl_)]))
        ll_off.append(-float(o[pl_]))
    w = s.get_column("weight").to_numpy()
    print(f"{model.name}: {len(qs)} test positions in {dt_:.1f} s ({len(qs) / dt_:.1f} positions/s; "
          f"{dict((k, round(v, 1)) for k, v in model.seconds.items())}); GPU busy {g.mean:.0f} %, "
          f"this process {mib} MiB on the GPU; load {os.getloadavg()}")
    print(f"max |log p live - offline| {worst:.4f} (move {worst_at[1]} in {worst_at[0]}: p offline {worst_at[2]:.2e}, "
          f"live {worst_at[3]:.2e}); over moves with p >= 0.01 {worst_big:.4f}; max |p live - p offline| {worst_p:.4f}; "
          f"log loss live {np.average(ll_live, weights=w):.4f}, offline {np.average(ll_off, weights=w):.4f}")


RULE_MAIN = CONFIGS[MAIN]["rule"]


def _front_main(argv=None):
    from tnh.models.combo import TEST_SAMPLE

    p = argparse.ArgumentParser(prog="python -m tnh.models.combotrees front|check-live")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("front")
    sp.add_argument("--test", type=Path, default=TEST_SAMPLE)
    sp.add_argument("--n", type=int, default=2000)
    sp.add_argument("--seed", type=int, default=2)
    sp.add_argument("--boot", type=int, default=1000)
    sp.add_argument("--maia-sizes", nargs="+", default=["5m", "23m", "79m"])
    sp.add_argument("--nets", nargs="+", default=["bt4", "t3"])
    sp.add_argument("--configs", nargs="+", default=list(CONFIGS))
    sp.add_argument("--out", type=Path,
                    default=Path.home() / "out" / "tn-hittare" / "scratch" / "models" / "combotrees" / "front.json")
    sp.set_defaults(fn=cmd_front)
    sp = sub.add_parser("check-live")
    sp.add_argument("--test", type=Path, default=TEST_SAMPLE)
    sp.add_argument("--n", type=int, default=500)
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--config", default=MAIN, help="+pol (PolicyTrees, the default model) or +ch-p0.5 (Combo trees)")
    sp.set_defaults(fn=cmd_check_live)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ("front", "check-live"):
        _front_main()
    else:
        main()
