"""Boosted trees for the stack: LightGBM with the exact group softmax as a custom objective.

One row per legal move, the position as the group. The objective is the weighted
log loss of the played move under a softmax over the group's scores: gradient
``w (p - y)``, hessian ``w p (1 - p)``. No ranking surrogate. The score starts at a
linear logit (``init_score``), so the trees learn the residual. Evaluation uses
``group_loss`` here, not a LightGBM metric.

A feature that is constant within a position has gradients that sum to zero over
any set of whole positions, so the trees can only use it below a split on a
per-move feature. ``design`` therefore adds explicit products: each engine loss
and each log probability times the mover's rating, the log clock and the ply.
"""

from __future__ import annotations

import itertools

import lightgbm as lgb
import numpy as np

PRODUCT_CONTEXT = ["ctx_mover_elo", "ctx_clock_mover", "ctx_ply"]


def segment_lse(s: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    lens = np.diff(offsets)
    m = np.maximum.reduceat(s, offsets[:-1])
    return m + np.log(np.add.reduceat(np.exp(s - np.repeat(m, lens)), offsets[:-1]))


def group_loss(s: np.ndarray, offsets: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted mean negative log probability of the played moves (``y`` are flat indices)."""
    return float(-np.sum(w * (s[y] - segment_lse(s, offsets))) / np.sum(w))


def group_logp(s: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return s - np.repeat(segment_lse(s, offsets), np.diff(offsets))


def objective(offsets: np.ndarray, y: np.ndarray, w: np.ndarray):
    lens = np.diff(offsets)
    played = np.zeros(offsets[-1])
    played[y] = 1.0
    wm = np.repeat(w / w.mean(), lens)

    def fobj(preds, _data):
        p = np.exp(group_logp(preds, offsets))
        grad = wm * (p - played)
        hess = np.maximum(wm * p * (1 - p), 1e-12)
        return grad, hess
    return fobj


def design(Xmove: np.ndarray, move_names: list[str], Xctx_rows: np.ndarray, ctx_names: list[str],
           offsets: np.ndarray, product_cols: list[str], products: bool) -> tuple[np.ndarray, list[str]]:
    """Per-move matrix for the trees, float32, filled in place: move features, context repeated per move,
    and the products (no float64 temporaries of the whole matrix)."""
    lens = np.diff(offsets)
    row_of = np.repeat(np.arange(len(lens)), lens)
    names = list(move_names) + list(ctx_names)
    prods = []
    if products:
        for c, z in itertools.product(product_cols, PRODUCT_CONTEXT):
            if c in move_names and z in ctx_names:
                prods.append((move_names.index(c), ctx_names.index(z)))
                names.append(f"{c}*{z}")
    X = np.empty((len(row_of), len(names)), dtype=np.float32)
    X[:, :Xmove.shape[1]] = Xmove
    for j in range(Xctx_rows.shape[1]):
        X[:, Xmove.shape[1] + j] = Xctx_rows[row_of, j]
    k = Xmove.shape[1] + Xctx_rows.shape[1]
    for i, (c, z) in enumerate(prods):
        X[:, k + i] = Xmove[:, c] * Xctx_rows[row_of, z]
    return X, names


def train(X_tr, off_tr, y_tr, w_tr, init_tr, X_va, off_va, y_va, w_va, init_va, params: dict, rounds: int,
          patience: int, seed: int, threads: int = 0):
    """Train with early stopping on the validation positions. Returns (booster, best validation loss)."""
    p = {"objective": objective(off_tr, y_tr, w_tr), "verbosity": -1, "seed": seed, "deterministic": True,
         "num_threads": threads, "boost_from_average": False, **params}
    dtr = lgb.Dataset(X_tr, init_score=init_tr, free_raw_data=False)
    dva = lgb.Dataset(X_va, init_score=init_va, reference=dtr, free_raw_data=False)

    def feval(preds, _data):
        return "group_logloss", group_loss(preds, off_va, y_va, w_va), False

    booster = lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[dva], feval=feval,
                        callbacks=[lgb.early_stopping(patience, verbose=False)])
    # LightGBM's validation score includes init_score; check it against our own computation
    mine = group_loss(init_va + booster.predict(X_va, raw_score=True, num_iteration=booster.best_iteration),
                      off_va, y_va, w_va)
    theirs = booster.best_score["valid_0"]["group_logloss"]
    if abs(mine - theirs) > 1e-6:
        raise RuntimeError(f"validation loss {theirs} from LightGBM, {mine} recomputed: init_score not applied as assumed")
    return booster, mine


def raw(booster, X) -> np.ndarray:
    return booster.predict(X, raw_score=True, num_iteration=booster.best_iteration)


def grid(leaves: list[int], rates: list[float], min_leaf: list[int]) -> list[dict]:
    # no bagging: it samples moves, not positions
    return [{"num_leaves": l, "learning_rate": r, "min_data_in_leaf": m, "feature_fraction": 0.9, "lambda_l2": 1.0}
            for l, r, m in itertools.product(leaves, rates, min_leaf)]
