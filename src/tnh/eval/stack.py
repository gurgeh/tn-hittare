"""A small non-linear model on top of the features, beside the linear logit.

    uv run python -m tnh.eval.stack \\
        --logp m0=features/maia3-79m/offset+0 --logp mbest=features/maia3-79m/offset+300 \\
        --engine sf10k=features/stockfish/nodes10000 --engine sf100k=features/stockfish/nodes100000 \\
        [--deep sf100k] [--board-facts] [--test S] [--fit S] [--games G] [--results R]

**The model.** A score per legal move, softmax over the moves of the position,
trained on the played move with the weighted log loss used everywhere else:

    score(move) = b . x_move + MLP([x_move, x_context])

``x_move`` are the per-move features of ``tnh.eval.logit`` (log probabilities,
engine losses in expected score and squashed centipawns, NaN indicators), plus,
with ``--board-facts``, capture, check, promotion, castling and the piece moved.
The skip starts at the linear logit on the same move features, fitted on the
training rows, so the MLP learns only what the linear logit misses. ``x_context`` is
constant within a position: the mover's and opponent's rating,
ply, plies since leaving book, both clocks (log minutes) and the increment, with
missing indicators, ``log(1 + pos_count)``, ``log(n_legal)``, and the deep
engine's value of the position. A context feature can only matter through its
interaction with the per-move features; the MLP gets both concatenated, so it can
make the weight on a move feature depend on the context. Two hidden layers of
``--hidden`` units (SiLU).
Features are standardised with the fitting sample's means and deviations.

There is a slot for "share of games that chose this move here" (``--move-share``,
a feature directory with a per-move ``share`` column); the position-count table it
needs does not exist yet.

**Fitting.** On the fitting sample's window rows; 20 % of its tournaments are held
out for early stopping (patience ``--patience`` epochs); Adam. Reported on the
test sample's window rows. Each model is also mixed with a uniform ``eps`` fitted
on the whole fitting sample, as everywhere.

**Compared** on the same test rows, with paired tournament-bootstrap intervals:
Maia-3 alone (each ``--logp`` source, mixed with its own ``eps``); the linear logit
on the logit's features (``logit C`` when the sources are m0, mbest and the
engines); the same with the board facts; the **interaction logit**, the linear
logit plus explicit interactions (each engine loss x rating and x log clock, each
log probability x rating), whose coefficients can be read; and the stack.

**Ablation** of the stack by feature group, each retrained without it, with what
the group costs to compute per position (Maia-3 network evaluations, engine nodes).

**Effective weight** of the deep engine's losses: the derivative of the move score
with respect to the raw feature, averaged over the moves of the positions in each
bucket of minutes left and of the mover's rating; for the interaction logit
``b + g_rating * z_rating + g_clock * z_clock``; for the linear logit a constant.

Writes ``<results>/stack.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import chess
import numpy as np
import polars as pl
import torch

from tnh.eval.common import (Ragged, add_common, attach, boot_counts, boot_means, check_splits, est, fit_eps,
                             mix_logp, provenance, read_meta, read_sample, window_rows, wmean, write_result)
from tnh.eval.logit import Data, build, fit as fit_logit, parse_pairs, scores
from tnh.paths import CORPUS

CLOCK_EDGES_MIN = [0, 2, 5, 10, 20, 40, 10_000]
CLOCK_LABELS = ["<2", "2-5", "5-10", "10-20", "20-40", ">40"]
PIECES = ["P", "N", "B", "R", "Q", "K"]   # pawn is the reference: a full one-hot is constant within a position


# --- features ---------------------------------------------------------------------

def board_facts(sample: pl.DataFrame, D: Data) -> dict[str, np.ndarray]:
    """Per move: capture, check, promotion, castling, piece moved (one-hot)."""
    fens = sample.get_column("fen").to_list()
    out = {k: [] for k in ["capture", "check", "promotion", "castling"] + [f"piece_{p}" for p in PIECES if p != "P"]}
    for r, moves in zip(D.rows, D.moves.to_list()):
        b = chess.Board(fens[r])
        for u in moves:
            m = chess.Move.from_uci(u)
            out["capture"].append(b.is_capture(m))
            out["check"].append(b.gives_check(m))
            out["promotion"].append(m.promotion is not None)
            out["castling"].append(b.is_castling(m))
            pt = b.piece_type_at(m.from_square)
            for i, p in enumerate(PIECES):
                if p != "P":
                    out[f"piece_{p}"].append(pt == i + 1)
    return {f"board_{k}": np.asarray(v, dtype=np.float64) for k, v in out.items()}


def context(sample: pl.DataFrame, D: Data, games: pl.DataFrame, deep: Path | None, book_count: int) -> tuple[np.ndarray, list[str], dict[str, list[str]]]:
    """Context features per kept row, and their groups."""
    s = sample[D.rows].join(games, on="game_id", how="left")
    cols, groups = {}, {}

    def add(name, x, group):
        cols[name] = np.asarray(x, dtype=np.float64)
        groups.setdefault(group, []).append(name)

    add("ctx_mover_elo", (s.get_column("mover_elo").to_numpy() - 2200) / 200, "ratings")
    add("ctx_opp_elo", (s.get_column("opp_elo").to_numpy() - 2200) / 200, "ratings")
    ply = s.get_column("ply").to_numpy()
    add("ctx_ply", ply / 40, "stage")
    ex = f"exit_ply_{book_count}"
    if ex in s.columns:
        e = s.get_column(ex).to_numpy(allow_copy=True).astype(float)
        add("ctx_since_book", np.where(np.isnan(e), 0, ply - np.nan_to_num(e)) / 20, "stage")
    for side in ("mover", "opp"):
        c = s.get_column(f"clock_{side}_s").to_numpy(allow_copy=True).astype(float)
        miss = np.isnan(c)
        add(f"ctx_clock_{side}", np.where(miss, 0, np.log1p(np.nan_to_num(c) / 60)), "clock")
        add(f"ctx_clock_{side}_missing", miss, "clock")
    inc = s.get_column("inc_s").to_numpy(allow_copy=True).astype(float)
    add("ctx_inc", np.nan_to_num(inc) / 30, "clock")
    add("ctx_inc_missing", np.isnan(inc), "clock")
    add("ctx_pos_count", np.log1p(s.get_column("pos_count").to_numpy()), "book")
    add("ctx_n_legal", np.log(s.get_column("n_legal").to_numpy()), "stage")
    if deep is not None:
        j = attach(sample[D.rows], deep, ["value"])
        from tnh.eval.common import ragged
        from tnh.eval.logit import nanmax_segments
        r = ragged(j, "value")
        add("ctx_deep_value", nanmax_segments(r, r.values) - 0.5, "deep_engine_value")
    names = list(cols)
    return np.column_stack([cols[c] for c in names]), names, groups


def engine_cost(meta: dict) -> str:
    """What one position's values cost to compute, from the feature directory's meta.json."""
    m, v = meta.get("model", "?"), meta.get("variant", "?")
    if m == "stockfish":
        return f"Stockfish {meta.get('nodes', '?')} nodes"
    if v == "children":
        return f"{m}: one network evaluation per legal move, no search"
    if meta.get("nodes"):
        return f"{m}: {meta['nodes']}-node search"
    return f"{m} {v}"


# --- the network ------------------------------------------------------------------

class Stack(torch.nn.Module):
    def __init__(self, n_move: int, n_ctx: int, hidden: int):
        super().__init__()
        self.skip = torch.nn.Linear(n_move, 1, bias=False)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_move + n_ctx, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1))
        torch.nn.init.zeros_(self.mlp[-1].weight)
        torch.nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, xm: torch.Tensor, xc: torch.Tensor) -> torch.Tensor:
        return (self.skip(xm) + self.mlp(torch.cat([xm, xc], dim=1))).squeeze(1)


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def group_logp(score: torch.Tensor, row: torch.Tensor, n_rows: int) -> torch.Tensor:
    """Log softmax of ``score`` within each row (group of moves)."""
    mx = torch.full((n_rows,), -torch.inf, device=score.device).scatter_reduce(0, row, score.detach(), "amax")
    e = torch.exp(score - mx[row])
    z = torch.zeros(n_rows, device=score.device, dtype=score.dtype).index_add(0, row, e)
    return score - (mx + torch.log(z))[row]


class Batches:
    """Positions in batches: the moves of a set of rows, with each move's row index within the batch."""

    def __init__(self, offsets: np.ndarray):
        self.offsets = offsets
        self.lengths = np.diff(offsets)

    def moves_of(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lens = self.lengths[rows]
        starts = self.offsets[:-1][rows]
        local = np.repeat(np.arange(len(rows)), lens)
        idx = np.repeat(starts - np.r_[0, np.cumsum(lens)[:-1]], lens) + np.arange(lens.sum())
        return idx, local, lens


def train(Xm, Xc, offsets, y, w, rows_train, rows_val, a, device, log=print, init=None) -> tuple[Stack, dict]:
    """``init``: starting weights of the linear skip (standardised units), e.g. a linear logit's."""
    torch.manual_seed(a.seed)
    model = Stack(Xm.shape[1], Xc.shape[1], a.hidden).to(device)
    if init is not None:
        with torch.no_grad():
            model.skip.weight.copy_(torch.as_tensor(init, dtype=torch.float32).reshape(1, -1))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    B = Batches(offsets)
    y_local = y - offsets[:-1]          # index of the played move within its row
    rng = np.random.default_rng(a.seed)

    def batch_loss(rows):
        idx, local, lens = B.moves_of(rows)
        xm = torch.as_tensor(Xm[idx], dtype=torch.float32, device=device)
        xc = torch.as_tensor(Xc[rows][local], dtype=torch.float32, device=device)
        row = torch.as_tensor(local, device=device)
        lp = group_logp(model(xm, xc), row, len(rows))
        first = np.r_[0, np.cumsum(lens)[:-1]]
        played = torch.as_tensor(first + y_local[rows], device=device)
        ww = torch.as_tensor(w[rows], dtype=torch.float32, device=device)
        return -(ww * lp[played]).sum(), ww.sum()

    def evaluate(rows):
        model.eval()
        tot, wt = 0.0, 0.0
        with torch.no_grad():
            for i in range(0, len(rows), a.batch):
                l, s = batch_loss(rows[i:i + a.batch])
                tot += float(l); wt += float(s)
        model.train()
        return tot / wt

    best, best_state, bad, hist = float("inf"), None, 0, []
    for epoch in range(a.max_epochs):
        t0 = time.time()
        perm = rng.permutation(rows_train)
        for i in range(0, len(perm), a.batch):
            l, s = batch_loss(perm[i:i + a.batch])
            opt.zero_grad()
            (l / s).backward()
            opt.step()
        v = evaluate(rows_val)
        hist.append(v)
        log(f"    epoch {epoch}: validation {v:.4f} ({time.time() - t0:.1f} s)")
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    return model, {"epochs": len(hist), "best_validation": best, "validation_history": hist}


def predict(model, Xm, Xc, offsets, device, batch=4096) -> np.ndarray:
    """Flat log probabilities over all moves."""
    B = Batches(offsets)
    out = np.empty(offsets[-1])
    model.eval()
    with torch.no_grad():
        for i in range(0, len(offsets) - 1, batch):
            rows = np.arange(i, min(i + batch, len(offsets) - 1))
            idx, local, _ = B.moves_of(rows)
            xm = torch.as_tensor(Xm[idx], dtype=torch.float32, device=device)
            xc = torch.as_tensor(Xc[rows][local], dtype=torch.float32, device=device)
            lp = group_logp(model(xm, xc), torch.as_tensor(local, device=device), len(rows))
            out[idx] = lp.double().cpu().numpy()
    return out


def effective_weight(model, Xm, Xc, offsets, col: int, scale: float, device, batch=4096) -> np.ndarray:
    """Per row: the mean over its moves of d score / d (raw feature ``col``)."""
    B = Batches(offsets)
    out = np.empty(len(offsets) - 1)
    model.eval()
    for i in range(0, len(offsets) - 1, batch):
        rows = np.arange(i, min(i + batch, len(offsets) - 1))
        idx, local, lens = B.moves_of(rows)
        xm = torch.as_tensor(Xm[idx], dtype=torch.float32, device=device).requires_grad_(True)
        xc = torch.as_tensor(Xc[rows][local], dtype=torch.float32, device=device)
        g, = torch.autograd.grad(model(xm, xc).sum(), xm)
        d = g[:, col].double().cpu().numpy() / scale
        out[rows] = np.bincount(local, weights=d, minlength=len(rows)) / lens
    return out


# --- main ---------------------------------------------------------------------------

def standardise(X: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((X - mean) / sd).astype(np.float32)


def val_split(clusters: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Hold out whole tournaments, in a seeded random order, until at least ``frac`` of the rows are held out."""
    k = clusters.max() + 1
    sizes = np.bincount(clusters, minlength=k)
    order = np.random.default_rng(seed).permutation(k)
    take = np.cumsum(sizes[order]) <= frac * len(clusters)
    take[np.argmin(take) if not take.all() else 0] = True     # the first tournament past the share too
    held = np.zeros(k, dtype=bool)
    held[order[take]] = True
    if held.all():
        raise SystemExit("too few tournaments to hold any out")
    return held[clusters]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m tnh.eval.stack", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logp", action="append", required=True, help="NAME=DIR of a move model")
    p.add_argument("--engine", action="append", help="NAME=DIR of an engine")
    p.add_argument("--deep", help="which engine is the deep one (default: the last --engine)")
    p.add_argument("--move-share", type=Path, help="feature dir with a per-move 'share' column (not built yet)")
    p.add_argument("--board-facts", action="store_true", help="capture, check, promotion, castling, piece moved")
    p.add_argument("--games", type=Path, default=CORPUS / "games.parquet", help="for the increment")
    p.add_argument("--book-count", type=int, default=5)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--batch", type=int, default=1024, help="positions per batch")
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--val-frac", type=float, default=0.2, help="share of fitting rows held out, whole tournaments")
    p.add_argument("--no-ablation", action="store_true")
    p.add_argument("--no-trees", action="store_true", help="skip the boosted trees")
    p.add_argument("--leaves", type=int, nargs="+", default=[15, 63], help="tree tuning grid: num_leaves")
    p.add_argument("--rates", type=float, nargs="+", default=[0.05, 0.15], help="tree tuning grid: learning rate")
    p.add_argument("--min-leaf", type=int, nargs="+", default=[200, 2000], help="tree tuning grid: min rows per leaf")
    p.add_argument("--rounds", type=int, default=2000, help="most trees; early stopping decides")
    p.add_argument("--tree-patience", type=int, default=50)
    p.add_argument("--tune-frac", type=float, default=1.0,
                   help="share of the training tournaments used to choose the tree settings")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_common(p)
    p.add_argument("--name", default="stack", help="result file name (without .json)")
    a = p.parse_args(argv)
    logp, engines = parse_pairs(a.logp), parse_pairs(a.engine)
    deep_name = a.deep or (list(engines)[-1] if engines else None)
    test_s, fit_s = window_rows(read_sample(a.test)), window_rows(read_sample(a.fit))
    check_splits(test_s, fit_s)
    games = pl.read_parquet(a.games, columns=["game_id", "inc_s"])

    Df, info_f = build(fit_s, logp, engines)
    Dt, info_t = build(test_s, logp, engines)
    if a.move_share:
        for D, s in ((Df, fit_s), (Dt, test_s)):
            j = attach(s[D.rows], a.move_share, ["share"])
            from tnh.eval.common import ragged
            D.X = np.column_stack([D.X, ragged(j, "share").values])
            D.names = D.names + ["move_share"]
    if a.board_facts:
        t0 = time.time()
        Df = Df.with_columns(board_facts(fit_s, Df))
        Dt = Dt.with_columns(board_facts(test_s, Dt))
        print(f"board facts: {time.time() - t0:.0f} s")
    deep_dir = engines.get(deep_name) if deep_name else None
    Cf, ctx_names, ctx_groups = context(fit_s, Df, games, deep_dir, a.book_count)
    Ct, _, _ = context(test_s, Dt, games, deep_dir, a.book_count)
    # NaN indicators that never fire on a played move cannot be fitted by the logit (see tnh.eval.logit)
    unused = [c for c in Df.names if c.endswith("_nan") and Df.X[Df.y, Df.names.index(c)].sum() < 20]
    move_names = [c for c in Df.names if c not in unused]
    # feature groups and what they cost per position
    maia_src = [k for k, d in logp.items() if read_meta(d).get("model", "").startswith("maia")]
    groups = {"maia": [c for c in move_names if c in maia_src]} if maia_src else {}
    cost = {"maia": f"{len(maia_src)} Maia-3 network evaluations"} if maia_src else {}
    for k, d in logp.items():
        if k not in maia_src:
            groups[k] = [k]
            cost[k] = f"1 {read_meta(d).get('model', k)} network evaluation"
    for e, d in engines.items():
        groups[e] = [c for c in move_names if c.startswith(f"{e}_")]
        cost[e] = engine_cost(read_meta(d))
    if deep_name:
        cost[deep_name] += " (also the position's value in the context)"
    if a.board_facts:
        groups["board"] = [c for c in move_names if c.startswith("board_")]
        cost["board"] = "free (move generation)"
    for g in ("ratings", "clock", "stage", "book"):
        cost[g] = "free (game data)"
    print(f"per-move features {move_names}; context {ctx_names}")

    # standardisation on the fitting sample
    Xf_all = Df.subset(move_names).X
    mm, ms = Xf_all.mean(0), Xf_all.std(0)
    ms[ms == 0] = 1
    cm, cs = Cf.mean(0), Cf.std(0)
    cs[cs == 0] = 1
    Xmf, Xmt = standardise(Xf_all, mm, ms), standardise(Dt.subset(move_names).X, mm, ms)
    Xcf, Xct = standardise(Cf, cm, cs), standardise(Ct, cm, cs)
    held = val_split(Df.clusters, a.val_frac, a.seed)
    rows_train, rows_val = np.flatnonzero(~held), np.flatnonzero(held)
    print(f"fit rows {len(Df.w)} ({len(rows_train)} train, {len(rows_val)} validation, by tournament); "
          f"test rows {len(Dt.w)}; device {a.device}")

    def run(drop_move: list[str], drop_ctx: list[str], label: str):
        km = [i for i, c in enumerate(move_names) if c not in drop_move]
        kc = [i for i, c in enumerate(ctx_names) if c not in drop_ctx]
        xc_f = Xcf[:, kc] if kc else np.zeros((len(Df.w), 1), np.float32)
        xc_t = Xct[:, kc] if kc else np.zeros((len(Dt.w), 1), np.float32)
        print(f"  stack {label}: {len(km)} move features, {len(kc)} context features")
        # start from the linear logit on the same move features, fitted on the training rows only
        cols = [move_names[i] for i in km]
        b0, _ = fit_logit(Df.subset(cols), Df.w * np.isin(np.arange(len(Df.w)), rows_train))
        model, hist = train(Xmf[:, km], xc_f, Df.offsets, Df.y, Df.w, rows_train, rows_val, a, a.device,
                            init=b0 * ms[km])
        lpf = predict(model, Xmf[:, km], xc_f, Df.offsets, a.device)[Df.y]
        lpt_all = predict(model, Xmt[:, km], xc_t, Dt.offsets, a.device)
        e = fit_eps(lpf, Df.n, Df.w)
        return model, km, kc, hist, e, -mix_logp(lpt_all[Dt.y], Dt.n, e)

    model, km, kc, hist, eps_stack, loss_stack = run([], [], "full")
    results = {"stack": {"loss": loss_stack, "eps": eps_stack, "params": n_params(model), **hist}}

    # linear and interaction logits (unstandardised features, as in tnh.eval.logit)
    base = [c for c in move_names if c in logp or any(c.startswith(f"{e}_") for e in engines)]

    def fit_lin(D_f, D_t, cols, name):
        bf, it = fit_logit(D_f.subset(cols))
        lpf, _ = scores(D_f.subset(cols), bf)
        e = fit_eps(lpf, D_f.n, D_f.w)
        lpt, _ = scores(D_t.subset(cols), bf)
        results[name] = {"loss": -mix_logp(lpt, D_t.n, e), "eps": e, "params": len(cols),
                         "columns": cols, "beta": bf.tolist()}
        return bf

    fit_lin(Df, Dt, base, "linear")
    if a.board_facts:
        fit_lin(Df, Dt, move_names, "linear+board")
    # explicit interactions: engine losses x rating and x log clock, log probabilities x rating
    zr = ctx_names.index("ctx_mover_elo")
    zc = ctx_names.index("ctx_clock_mover")
    inter_f, inter_t, inter_names = {}, {}, []
    for c in base:
        kinds = [("rating", zr)] + ([("clock", zc)] if not c in logp else [])
        if c.endswith("_nan"):
            continue
        for kname, zi in kinds:
            n = f"{c}*{kname}"
            inter_names.append(n)
            inter_f[n] = Df.X[:, Df.names.index(c)] * np.repeat(Xcf[:, zi], np.diff(Df.offsets))
            inter_t[n] = Dt.X[:, Dt.names.index(c)] * np.repeat(Xct[:, zi], np.diff(Dt.offsets))
    Dfi, Dti = Df.with_columns(inter_f), Dt.with_columns(inter_t)
    beta_i = fit_lin(Dfi, Dti, base + inter_names, "interactions")

    singles = {}
    for k in logp:
        e = fit_eps(Df.single_lp[k], Df.n, Df.w)
        singles[k] = {"loss": -mix_logp(Dt.single_lp[k], Dt.n, e), "eps": e}

    # boosted trees
    trees_out, ablation_trees = None, {}
    if not a.no_trees:
        from tnh.eval import trees as T
        prod_cols = [c for c in base if not c.endswith("_nan")]
        Xraw_f, Xraw_t = Df.subset(move_names).X, Dt.subset(move_names).X
        Bf = Batches(Df.offsets)
        ylocal = Df.y - Df.offsets[:-1]
        train_w = Df.w * np.isin(np.arange(len(Df.w)), rows_train)

        def subrows(rows):
            idx, _, lens = Bf.moves_of(rows)
            off = np.r_[0, np.cumsum(lens)]
            return idx, off, off[:-1] + ylocal[rows]

        tr, va = subrows(rows_train), subrows(rows_val)
        rng_t = np.random.default_rng(a.seed + 5)
        tour = np.unique(Df.clusters[rows_train])
        tune_t = tour[rng_t.random(len(tour)) < a.tune_frac] if a.tune_frac < 1 else tour
        rows_tune = rows_train[np.isin(Df.clusters[rows_train], tune_t)]
        tu = subrows(rows_tune)

        def fit_trees(drop_move, drop_ctx, products, settings, tune=False):
            km = [c for c in move_names if c not in drop_move]
            kc = [c for c in ctx_names if c not in drop_ctx]
            kmi = [move_names.index(c) for c in km]
            kci = [ctx_names.index(c) for c in kc]
            b, _ = fit_logit(Df.subset(km), train_w)
            init_f, init_t = Df.subset(km).X @ b, Dt.subset(km).X @ b
            Xf, names = T.design(Xraw_f[:, kmi], km, Cf[:, kci], kc, Df.offsets, prod_cols, products)
            tried, best = [], None
            fit_rows = tu if tune else tr
            fw = Df.w[rows_tune] if tune else Df.w[rows_train]
            for prm in settings:
                t0 = time.time()
                bst, v = T.train(Xf[fit_rows[0]], fit_rows[1], fit_rows[2], fw, init_f[fit_rows[0]],
                                 Xf[va[0]], va[1], va[2], Df.w[rows_val], init_f[va[0]], prm, a.rounds,
                                 a.tree_patience, a.seed)
                tried.append({**prm, "trees": bst.best_iteration, "validation": v, "seconds": round(time.time() - t0)})
                print(f"    trees {prm['num_leaves']} leaves, rate {prm['learning_rate']}, min leaf "
                      f"{prm['min_data_in_leaf']}: {bst.best_iteration} trees, validation {v:.4f}")
                if best is None or v < best[1]:
                    best = (bst, v, prm)
            bst, v, prm = best
            if tune and a.tune_frac < 1:      # refit the chosen setting on all the training rows
                bst, v = T.train(Xf[tr[0]], tr[1], tr[2], Df.w[rows_train], init_f[tr[0]], Xf[va[0]], va[1], va[2],
                                 Df.w[rows_val], init_f[va[0]], prm, a.rounds, a.tree_patience, a.seed)
            Xt, _ = T.design(Xraw_t[:, kmi], km, Ct[:, kci], kc, Dt.offsets, prod_cols, products)
            lpf = T.group_logp(init_f + T.raw(bst, Xf), Df.offsets)[Df.y]
            e = fit_eps(lpf, Df.n, Df.w)
            lpt = T.group_logp(init_t + T.raw(bst, Xt), Dt.offsets)[Dt.y]
            gain = bst.feature_importance("gain")
            imp = sorted(zip(names, (gain / gain.sum()).tolist()), key=lambda x: -x[1])
            return {"loss": -mix_logp(lpt, Dt.n, e), "eps": e, "trees": bst.best_iteration, "settings": prm,
                    "tried": tried, "validation": v, "params": f"{bst.best_iteration} trees x {prm['num_leaves']} leaves",
                    "importance": imp[:25], "_booster": bst, "_km": km, "_kc": kc, "_b": b, "_products": products}

        print("  trees with products, tuning")
        settings = T.grid(a.leaves, a.rates, a.min_leaf)
        tw = fit_trees([], [], True, settings, tune=True)
        chosen = tw["settings"]
        print(f"  trees chosen: {chosen}")
        tn = fit_trees([], [], False, [chosen])
        results["trees"] = tw
        results["trees without products"] = tn
        if not a.no_ablation:
            for g, cols in [(g, c) for g, c in groups.items()] + [(g, None) for g in ("ratings", "clock", "stage", "book")]:
                if cols is not None:
                    dm, dc = cols, (["ctx_deep_value"] if g == deep_name else [])
                else:
                    dm, dc = [], ctx_groups.get(g, [])
                    if not dc:
                        continue
                print(f"  trees without {g}")
                r_ = fit_trees(dm, dc, True, [chosen])
                ablation_trees[g] = {"loss": r_["loss"], "eps": r_["eps"], "trees": r_["trees"], "cost": cost[g],
                                     "dropped": dm + dc}
        trees_out = tw

    # ablation
    ablation = {}
    if not a.no_ablation:
        for g, cols in [(g, c) for g, c in groups.items()] + [(g, None) for g in ("ratings", "clock", "stage", "book")]:
            if cols is not None:
                dm, dc = cols, (["ctx_deep_value"] if g == deep_name else [])
            else:
                dm, dc = [], ctx_groups.get(g, [])
                if not dc:
                    continue
            m_, _, _, h_, e_, l_ = run(dm, dc, f"without {g}")
            ablation[g] = {"loss": l_, "eps": e_, "params": n_params(m_), "epochs": h_["epochs"], "cost": cost[g],
                           "dropped": dm + dc}

    # numbers with intervals; differences are paired against the linear logit and the stack
    wt = Dt.w
    counts = boot_counts(Dt.clusters.max() + 1, a.boot, a.seed)
    named = {**{k: v["loss"] for k, v in results.items()}, **{f"maia {k}": v["loss"] for k, v in singles.items()}}
    out_models = {}
    ref_trees = results["trees"]["loss"] if "trees" in results else None
    for k, l in named.items():
        X = np.column_stack([l, l - results["linear"]["loss"], l - results["stack"]["loss"]]
                            + ([l - ref_trees] if ref_trees is not None else []))
        bm = boot_means(X, wt, Dt.clusters, counts)
        out_models[k] = {"test": est(wmean(l, wt), bm[:, 0]), "minus_linear": est(wmean(X[:, 1], wt), bm[:, 1]),
                         "minus_stack": est(wmean(X[:, 2], wt), bm[:, 2])}
        if ref_trees is not None:
            out_models[k]["minus_trees"] = est(wmean(X[:, 3], wt), bm[:, 3])
        src = results.get(k) or singles.get(k.removeprefix("maia "))
        out_models[k].update({kk: v for kk, v in src.items() if kk != "loss" and not kk.startswith("_")})
    out_ablation_trees = {}
    for g, v in ablation_trees.items():
        d = v["loss"] - results["trees"]["loss"]
        bm = boot_means(np.column_stack([v["loss"], d]), wt, Dt.clusters, counts)
        out_ablation_trees[g] = {"test": est(wmean(v["loss"], wt), bm[:, 0]), "rise": est(wmean(d, wt), bm[:, 1]),
                                 **{kk: vv for kk, vv in v.items() if kk != "loss"}}
    out_ablation = {}
    for g, v in ablation.items():
        d = v["loss"] - results["stack"]["loss"]
        bm = boot_means(np.column_stack([v["loss"], d]), wt, Dt.clusters, counts)
        out_ablation[g] = {"test": est(wmean(v["loss"], wt), bm[:, 0]), "rise": est(wmean(d, wt), bm[:, 1]),
                           **{kk: vv for kk, vv in v.items() if kk != "loss"}}

    # effective weight of the deep engine's losses, by clock and by rating
    eff = {}
    test_rows = test_s[Dt.rows]
    mins = test_rows.get_column("clock_mover_s").to_numpy(allow_copy=True).astype(float) / 60
    cb = np.where(np.isnan(mins), -1, np.searchsorted(CLOCK_EDGES_MIN, np.nan_to_num(mins), side="right") - 1)
    rb = (test_rows.get_column("mover_elo").to_numpy() // 100) * 100
    if deep_name:
        for feat in (f"{deep_name}_eloss", f"{deep_name}_cploss"):
            if feat not in move_names:
                continue
            col = move_names.index(feat)
            per_row = effective_weight(model, Xmt, Xct, Dt.offsets, col, ms[col], a.device)
            bi = base.index(feat)
            lin = results["linear"]["beta"][bi]
            g = {kname: beta_i[(base + inter_names).index(f"{feat}*{kname}")] for kname in ("rating", "clock")}
            inter_row = beta_i[bi] + g["rating"] * Xct[:, zr] + g["clock"] * Xct[:, zc]
            trees_row = None
            if trees_out is not None:
                from tnh.eval import trees as T
                km, kc = trees_out["_km"], trees_out["_kc"]
                kmi = [move_names.index(c) for c in km]
                kci = [ctx_names.index(c) for c in kc]
                Xr = Dt.subset(move_names).X[:, kmi]
                h = 0.1 * ms[col]
                sc = []
                for sign in (1, -1):
                    Xp = Xr.copy()
                    Xp[:, km.index(feat)] += sign * h
                    Xd, _ = T.design(Xp, km, Ct[:, kci], kc, Dt.offsets, prod_cols, trees_out["_products"])
                    sc.append(Xp @ trees_out["_b"] + T.raw(trees_out["_booster"], Xd))
                d = (sc[0] - sc[1]) / (2 * h)
                trees_row = np.add.reduceat(d, Dt.offsets[:-1]) / np.diff(Dt.offsets)
            tabs = {}
            for by, codes, labels in (("clock", cb, CLOCK_LABELS), ("rating", None, None)):
                if by == "rating":
                    ub = np.unique(rb)
                    codes = np.searchsorted(ub, rb)
                    labels = [str(int(x)) for x in ub]
                ok = codes >= 0
                X = np.column_stack([per_row, inter_row] + ([trees_row] if trees_row is not None else []))[ok]
                bm = boot_means(X, wt[ok], Dt.clusters[ok], counts, groups=codes[ok], n_groups=len(labels))
                rows = []
                for i, lab in enumerate(labels):
                    m = codes[ok] == i
                    if not m.any():
                        continue
                    rows.append({"group": lab, "rows": int(m.sum()),
                                 "stack": est(wmean(X[m, 0], wt[ok][m]), bm[:, i, 0]),
                                 "interactions": est(wmean(X[m, 1], wt[ok][m]), bm[:, i, 1]), "linear": lin,
                                 **({"trees": est(wmean(X[m, 2], wt[ok][m]), bm[:, i, 2])} if trees_row is not None
                                    else {})})
                tabs[by] = rows
            eff[feat] = tabs
    for k, v in out_models.items():
        print(f"{k:>16}: test {v['test']['est']:.4f} [{v['test']['lo']:.4f}, {v['test']['hi']:.4f}] "
              f"minus linear {v['minus_linear']['est']:+.4f} [{v['minus_linear']['lo']:+.4f}, {v['minus_linear']['hi']:+.4f}]")
    for g, v in out_ablation_trees.items():
        print(f"  trees without {g:>10}: rise {v['rise']['est']:+.4f} [{v['rise']['lo']:+.4f}, {v['rise']['hi']:+.4f}]")
    for g, v in out_ablation.items():
        print(f"  without {g:>10}: rise {v['rise']['est']:+.4f} [{v['rise']['lo']:+.4f}, {v['rise']['hi']:+.4f}]")
    write_result(a.results, a.name, {
        "what": "a small MLP over per-move and context features, softmax over the legal moves, against the linear "
                "logit, an interaction logit and Maia-3 alone; window rows, fitted on the fitting sample",
        "rows": {"fit": len(Df.w), "train": len(rows_train), "validation": len(rows_val), "test": len(Dt.w),
                 "fit_info": info_f, "test_info": info_t},
        "move_features": move_names, "unused_nan_indicators": unused, "context_features": ctx_names,
        "context_groups": ctx_groups, "groups": groups, "cost": cost, "deep": deep_name,
        "options": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
        "models": out_models, "ablation": out_ablation, "ablation_trees": out_ablation_trees,
        "effective_weight": eff, "boot": a.boot,
        "provenance": provenance({"test": a.test, "fit": a.fit}, list(logp.values()) + list(engines.values()))})


if __name__ == "__main__":
    main()
