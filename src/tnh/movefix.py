"""The move model's correction for overestimated mistakes: a share of the mover's probability moved onto his best move.

    p' = (1 - share) * p + share * [best move],   share = share(elo, ply)

The move model (``tnh.models.default_move_model()``, PolicyTrees) expects players to throw away more than they do:
averaged over its move distribution, the static result model's value of the child is lower than the value of the
move actually played. ``share`` closes that gap. It is applied to the move model's probabilities before the
database mix (``tnh.movemix``), always, independently of any user setting such as preppy's ``Prepared``.

**Best move**, as fitted: the legal move whose child has the highest static value for the mover, the value being
the cheap result model (``tnh.eval.result.ResultModel("cheap")``, expected score ``w + d/2``) of the position after
the move, among all legal moves. An engine that picks its best move otherwise (a search, another value) gets a
somewhat different correction; the share is calibrated for this definition.

**Fit** (``python -m tnh.movefix fit``) on look-ahead runs with every child's sources stored
(``python -m tnh.eval.lookahead run --mover trees --root-sources``): per test position (plies 10-40, one per game,
test games) from the mover's side, the needed correction ``d = s(played child) - e1`` and the available loss
``L = s(best child) - e1``, ``e1 = sum p s(child)`` with p the move model's probabilities; the model is
``d = share(x) L`` with ``share(x) = x . beta`` linear in the features x. ``beta`` solves the moment equations
``sum x (d - L x . beta) = 0``: the corrected drift has mean 0 along every feature (for ``const`` the share is
``sum d / sum L``; for ``band`` the same per band). Least squares through the origin would weight positions by
L^2 and leave a mean drift. Forms are compared by cross-validation over tournaments on the squared corrected drift
per cell (rating band x ply range, weighted by the cell's positions). The leaf is recomputed from the stored
sources with the current cheap model.

**Forms**: ``const``; ``band`` (one share per rating band of the mover, 1800-1999 / 2000-2199 / 2200-2399 /
2400+); ``lin`` (``a + b r + c q``, ``r = (elo - 2200) / 100``, ``q = (ply - 25) / 10``); ``lin_x`` (plus ``r q``).

**Range rules**: elo is held to [1800, 2500] and ply to [10, 40] (the fitted range: no extrapolation), then the
share is clipped to [0, 0.6].

The rule in use is ``movefix.json`` beside this file, with its provenance. Without the file the share is 0.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RULE_FILE = Path(__file__).with_name("movefix.json")
ELO_RANGE = (1800.0, 2500.0)
PLY_RANGE = (10, 40)
SHARE_RANGE = (0.0, 0.6)
BAND_EDGES = [2000.0, 2200.0, 2400.0]
BAND_NAMES = ["1800-1999", "2000-2199", "2200-2399", "2400+"]
FORMS = ("const", "band", "lin", "lin_x")


def features(form: str, elo, ply) -> np.ndarray:
    """rows x parameters, after the range rules on elo and ply."""
    e = np.clip(np.asarray(elo, float), *ELO_RANGE)
    q = np.clip(np.asarray(ply, float), *PLY_RANGE)
    e, q = np.broadcast_arrays(e, q)
    r, z = (e - 2200.0) / 100.0, (q - 25.0) / 10.0
    one = np.ones_like(r)
    if form == "const":
        return one[..., None]
    if form == "band":
        b = np.searchsorted(BAND_EDGES, e, side="right")
        return (b[..., None] == np.arange(4)).astype(float)
    if form == "lin":
        return np.stack([one, r, z], -1)
    if form == "lin_x":
        return np.stack([one, r, z, r * z], -1)
    raise ValueError(f"form {form!r}: one of {FORMS}")


@dataclass
class Rule:
    form: str = "const"
    params: list[float] = field(default_factory=lambda: [0.0])
    provenance: dict = field(default_factory=dict)

    def share(self, elo, ply):
        """The share for a mover of this FIDE rating at this ply (number of half-moves played); scalars or arrays."""
        s = np.clip(features(self.form, elo, ply) @ np.asarray(self.params, float), *SHARE_RANGE)
        return float(s) if np.ndim(s) == 0 else s

    def apply(self, p, best_index, elo, ply) -> np.ndarray:
        """p' = (1 - share) p + share [best]. ``p``: one position's probabilities (n,) or positions x moves
        (padded with zeros); ``best_index``, ``elo``, ``ply``: scalars or one per position."""
        p = np.asarray(p, float)
        s = np.asarray(self.share(elo, ply), float)
        onehot = np.zeros_like(p)
        bi = np.asarray(best_index)
        if p.ndim == 1:
            onehot[int(bi)] = 1.0
        else:
            onehot[np.arange(p.shape[0]), np.broadcast_to(bi, (p.shape[0],))] = 1.0
            s = np.broadcast_to(s, (p.shape[0],))[:, None]
        return (1 - s) * p + s * onehot

    def describe(self) -> str:
        p = self.params
        if self.form == "const":
            return f"share {p[0]:.3f}"
        if self.form == "band":
            return "share by the mover's rating: " + ", ".join(f"{n} {x:.3f}" for n, x in zip(BAND_NAMES, p))
        t = f"{p[0]:.3f} {p[1]:+.4f} (elo-2200)/100 {p[2]:+.4f} (ply-25)/10"
        if self.form == "lin_x":
            t += f" {p[3]:+.4f} r q"
        return f"share = clip({t}, 0, 0.6), elo held to [1800, 2500], ply to [10, 40]"


def load(path: Path = RULE_FILE) -> Rule:
    if not path.exists():
        return Rule(provenance={"fallback": True})
    d = json.loads(path.read_text())
    return Rule(d["form"], d["params"], d.get("provenance", {}))


# --- fitting ---------------------------------------------------------------------------------------------------

def fit_ls(form: str, d: np.ndarray, L: np.ndarray, elo, ply, w=None) -> np.ndarray:
    """The moment fit: sum w x (d - L x . beta) = 0 (mean corrected drift 0 along every feature)."""
    X = features(form, elo, ply)
    w = np.ones(len(d)) if w is None else w
    return np.linalg.solve((X * (w * L)[:, None]).T @ X, X.T @ (w * d))


def cells(elo, ply) -> np.ndarray:
    """Rating band x ply range (10-19, 20-29, 30-40), 12 cells."""
    eb = np.searchsorted(BAND_EDGES, np.asarray(elo, float), side="right")
    pb = np.searchsorted([20, 30], np.asarray(ply, float), side="right")
    return eb * 3 + pb


def cell_drift_sq(resid: np.ndarray, cell: np.ndarray) -> float:
    """Squared mean residual per cell, weighted by the cell's share of the positions."""
    out = 0.0
    for c in np.unique(cell):
        m = cell == c
        out += m.mean() * resid[m].mean() ** 2
    return float(out)


def position_values(runs: list[Path], positions: list[Path], model: str = "cheap"):
    """Per position: e1, s(played child), s(best child) for the mover, recomputed from the stored sources with the
    result model ``model``; plus the frame (game_id, ply, ratings, tournament)."""
    import chess
    import polars as pl

    from tnh.eval.common import read_sample
    from tnh.eval.result import PARAMS, context, design, mlogit_probs

    P = json.loads(PARAMS.read_text())[model]
    th, srcs = np.array(P["coef"]), P["sources"]
    if srcs != ["maia0", "maia600", "lc0raw"]:
        raise ValueError(f"the runs store maia0, maia600, lc0raw; the {model} model needs {srcs}")
    test = pl.concat([read_sample(p) for p in positions], how="diagonal_relaxed").with_columns(pl.col("ply").cast(pl.Int64))
    test = test.unique(["game_id", "ply"], keep="first")
    d = pl.concat([pl.read_parquet(r) for r in runs], how="diagonal_relaxed").with_columns(pl.col("ply").cast(pl.Int64))
    d = d.unique(["game_id", "ply"], keep="first")
    j = test.join(d, on=["game_id", "ply"], how="inner").sort("game_id", "ply")
    e1, sp, sb = [], [], []
    for fen, played, ply, me, op, wtm, cp, cw, cs, ct in j.select(
            "fen", "played_uci", "ply", "mover_elo", "opp_elo", "white_to_move", "c_p", "c_wdl", "c_src",
            "c_terminal").iter_rows():
        legal = [m.uci() for m in chess.Board(fen).legal_moves]       # the run's child order
        if len(legal) != len(cp):
            raise ValueError(f"{fen}: {len(legal)} legal moves, the run has {len(cp)}")
        p = np.array(cp, float)
        p /= p.sum()
        S = np.array(cs, float).reshape(-1, 3, 3)
        live = ~np.array(ct, bool)
        val = np.array([w[0] + 0.5 * w[1] for w in cw], float)             # stored leaf (terminal: the result)
        if live.any():
            w = {k: S[live, i] / S[live, i].sum(1, keepdims=True) for i, k in enumerate(srcs)}
            n = int(live.sum())
            ctx = context(np.full(n, float(op)), np.full(n, float(me)), np.full(n, ply + 1.0), np.full(n, not wtm))
            q = mlogit_probs(th, design(w, ctx, srcs))
            val[live] = 1 - (q[:, 0] + 0.5 * q[:, 1])                     # the child's mover is the opponent
        e1.append(p @ val)
        sp.append(val[legal.index(played)])
        sb.append(val.max())
    return j, np.array(e1), np.array(sp), np.array(sb)


def cmd_fit(a) -> None:
    import polars as pl

    from tnh.eval.common import boot_counts, boot_means, cluster_codes, est

    j, e1, sp, sb = position_values(a.runs, a.positions, a.model)
    d, L = sp - e1, sb - e1
    elo = j.get_column("mover_elo").to_numpy().astype(float)
    ply = j.get_column("ply").to_numpy().astype(float)
    cl = cluster_codes(j)
    n, K = len(d), int(cl.max() + 1)
    counts = boot_counts(K, a.boot, a.seed)
    rng = np.random.default_rng(a.seed)
    fold = rng.permutation(K)[cl] % a.folds
    cell = cells(elo, ply)
    res, oof = {}, {}
    for form in FORMS:
        th = fit_ls(form, d, L, elo, ply)
        pred = np.zeros(n)
        for k in range(a.folds):
            tr, te = fold != k, fold == k
            tk = fit_ls(form, d[tr], L[tr], elo[tr], ply[tr])
            pred[te] = np.clip(features(form, elo[te], ply[te]) @ tk, *SHARE_RANGE) * L[te]
        oof[form] = d - pred
        bt = np.array([fit_ls(form, d, L, elo, ply, counts[b][cl].astype(float)) for b in range(a.boot)])
        res[form] = {"params": th.tolist(), "ci": np.quantile(bt, [0.025, 0.975], axis=0).T.tolist(),
                     "cv_cell_drift_sq": cell_drift_sq(oof[form], cell), "describe": Rule(form, th.tolist()).describe()}
    # the noise floor of the criterion: the same on the residuals of the true rule is about mean(var per cell / n)
    floor = float(sum((cell == c).mean() * oof["const"][cell == c].var() / (cell == c).sum() for c in np.unique(cell)))
    # the choice: the lowest cross-validated cell drift; a richer form must beat the simpler by more than the floor
    chosen = "const"
    for form in ("band", "lin", "lin_x"):
        if res[form]["cv_cell_drift_sq"] < res[chosen]["cv_cell_drift_sq"] - floor:
            chosen = form
    if a.form:
        chosen = a.form
    rule = Rule(chosen, res[chosen]["params"])
    # residual drift per band after the rule (in sample and out of fold), and before
    s = rule.share(elo, ply)
    resid = d - s * L
    groups = {"all": np.ones(n, bool)}
    eb = np.searchsorted(BAND_EDGES, elo, side="right")
    for i, nm in enumerate(BAND_NAMES):
        groups[f"rating {nm}"] = eb == i
    for lo, hi in ((10, 19), (20, 29), (30, 40)):
        groups[f"ply {lo}-{hi}"] = (ply >= lo) & (ply <= hi)
    table = {}
    for g, m in groups.items():
        X = np.column_stack([d[m], resid[m], oof[chosen][m], s[m]])
        bm = boot_means(X, np.ones(m.sum()), cl[m], counts)
        table[g] = {"n": int(m.sum()), "drift_before": est(X[:, 0].mean(), bm[:, 0]),
                    "drift_after": est(X[:, 1].mean(), bm[:, 1]), "drift_after_oof": est(X[:, 2].mean(), bm[:, 2]),
                    "mean_share": float(s[m].mean())}
    f_ = lambda e: f"{e['est']:+.4f} [{e['lo']:+.4f}, {e['hi']:+.4f}]"   # noqa: E731
    print(f"{n} positions, {K} tournaments")
    for form, r in res.items():
        print(f"  {form:>6}: {r['describe']}; CI " + ", ".join(f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in r["ci"])
              + f"; CV cell drift^2 {r['cv_cell_drift_sq'] * 1e6:.2f}e-6")
    print(f"  noise floor of the criterion {floor * 1e6:.2f}e-6")
    print(f"chosen: {chosen}: {rule.describe()}")
    for g, r in table.items():
        print(f"  {g:>16} n={r['n']:>4}: before {f_(r['drift_before'])}; after {f_(r['drift_after'])}; "
              f"out of fold {f_(r['drift_after_oof'])}; mean share {r['mean_share']:.3f}")
    edges = {f"elo {e}, ply {q}": rule.share(e, q) for e in (1800, 2000, 2200, 2400, 2600) for q in (6, 10, 25, 40, 60)}
    print("share at: " + "; ".join(f"{k} {v:.3f}" for k, v in edges.items()))
    out = {"form": chosen, "params": rule.params, "describe": rule.describe(),
           "provenance": {"fitted": time.strftime("%Y-%m-%d"), "commit": _commit(),
                          "runs": [str(r) for r in a.runs], "positions": [str(p) for p in a.positions],
                          "result_model": a.model, "n_positions": n, "tournaments": K, "folds": a.folds,
                          "boot": a.boot, "seed": a.seed, "forms": res, "criterion_noise_floor": floor,
                          "residual_drift": table,
                          "share_at": edges,
                          "best_move": "the legal move whose child has the highest static value (cheap result model, "
                                       "w + d/2) for the mover, among all legal moves",
                          "refit": "uv run python -m tnh.movefix fit"}}
    if a.write:
        RULE_FILE.write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {RULE_FILE}")


def _commit() -> str:
    import subprocess
    r = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip()


def main(argv=None) -> None:
    from tnh.eval.lookahead import OUT, POS_TEST

    p = argparse.ArgumentParser(prog="python -m tnh.movefix", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(required=True)
    sp = sub.add_parser("fit")
    sp.add_argument("--runs", type=Path, nargs="+", default=[OUT / "trees_root.parquet"])
    sp.add_argument("--positions", type=Path, nargs="+", default=[POS_TEST])
    sp.add_argument("--model", default="cheap", help="the result model entry the leaf values are recomputed with")
    sp.add_argument("--form", choices=FORMS, help="store this form instead of the cross-validated choice")
    sp.add_argument("--folds", type=int, default=10)
    sp.add_argument("--boot", type=int, default=1000)
    sp.add_argument("--seed", type=int, default=7)
    sp.add_argument("--write", action="store_true", help=f"store the rule in {RULE_FILE.name}")
    sp.set_defaults(fn=cmd_fit)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
