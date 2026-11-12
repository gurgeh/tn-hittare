"""Does looking ahead with the move model improve the result prediction?

    uv run python -m tnh.eval.lookahead positions --n 2000      # the test (and fitting) positions
    uv run python -m tnh.eval.lookahead run [--mover fast|combo] [--n-lines 32] [--depth 16] [--limit N]
    uv run python -m tnh.eval.lookahead analyse [--results R]
    uv run python -m tnh.eval.lookahead bias        # draws, calibration and recalibration by depth and mover

For each test position the static result model (``tnh.eval.result``) gives a win/draw/loss for the
side to move. Looking ahead replaces it by the same model's average over continuations drawn from
the move model, for both sides at their own ratings:

- **exact**, one and two plies: every move with probability at least ``--p-min`` (default 0.02) at
  each step, probabilities renormalised over those moves; the leaf values are averaged with those
  weights;
- **Monte Carlo**: ``--n-lines`` lines of ``--depth`` plies sampled from the move model; the estimate
  for N lines of d plies averages the first N lines' values at ply d (d in 2, 4, 8, 16; N in 8, 32).
  A line that ends (mate, stalemate, insufficient material) keeps its exact result.

Every value is a win/draw/loss turned to the root's side to move (win and loss swap at odd plies),
so an average is a proper prediction and gets a log loss like any other.

**The leaf** is the *cheap* result model: Maia-3's value head at two ratings and Lc0's raw value,
both FIDE ratings, ply and colour, in the stored logit (``ResultModel("cheap")``). It costs nothing
beyond the network calls the move model makes at the same node; the full model's 50-node Lc0
search at every leaf would cost 13.6 positions/s per leaf, hundreds of leaves per position.

**The move model** for the lines: ``fast`` is step 1's Maia + Lc0 policy logit (``M+pol``: Maia
at + 0 and + 600 and Lc0's policy, coefficients from ``reports/step1/logit_sub.json``); ``combo`` is
``tnh.models.combo.Combo(prune="p0.5")``, run on a subsample to see whether a better move model
changes the answer.

``run`` writes one row per root position (resumable) to ``<out>/<mover>.parquet``; ``analyse``
compares every variant with the static model (log loss, Brier, paired differences over
tournaments), regresses the score on the static prediction and the shift, and reports the shift's
mean and spread by ply and rating band.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import chess
import numpy as np
import polars as pl

from tnh.models.base import Query
from tnh.paths import CORPUS, REPORTS

OUT = Path.home() / "out" / "tn-hittare" / "scratch" / "eval" / "lookahead"
POS_TEST = CORPUS / "samples" / "lookahead_test.parquet"
POS_FIT = CORPUS / "samples" / "lookahead_fit.parquet"
DEPTHS = [2, 4, 8, 16]
NS = [8, 32]
FLIP = [2, 1, 0]


# --- the evaluator: move probabilities and the leaf value of a batch of nodes -------------------------

class FastMover:
    """Maia-3 at + 0 and + 600 and Lc0's policy in step 1's M+pol logit; the same calls give the leaf."""

    def __init__(self):
        from tnh.eval.result import ResultModel
        from tnh.models.lc0 import Lc0Policy
        from tnh.models.maia3 import Elo, Maia3

        s = json.loads((REPORTS / "step1" / "logit_sub.json").read_text())["sets"]["M+pol"]
        if s["columns"] != ["m0", "mbest", "lc0pol"]:
            raise ValueError(f"unexpected M+pol columns {s['columns']}")
        self.beta, self.eps = np.array(s["beta"]), float(s["eps"])
        self.policy = Lc0Policy()          # first: the GPU is shared, Lc0's backend wants its memory early
        self.maia = Maia3("79m", batch_size=64)
        self.elos = [Elo.offset(0.0), Elo.offset(600.0)]
        self.leaf = ResultModel("cheap")
        self.name = "fast"

    def evaluate(self, queries: list[Query]):
        """Per query: (moves, probabilities) and the leaf wdl for the side to move."""
        m0, m6 = self.maia.predict_many(queries, self.elos)
        pol = self.policy.predict(queries)
        moves, probs = [], []
        for a, b, c in zip(m0, m6, pol):
            ms = a.moves
            ib = {m: k for k, m in enumerate(b.moves)}
            ic = {m: k for k, m in enumerate(c.moves)}
            if set(ib) != set(ms) or set(ic) != set(ms):
                raise ValueError("the models disagree on the legal moves")
            X = np.column_stack([a.logp, b.logp[[ib[m] for m in ms]], c.logp[[ic[m] for m in ms]]]).astype(np.float64)
            s = X @ self.beta
            p = np.exp(s - s.max())
            p /= p.sum()
            p = (1 - self.eps) * p + self.eps / len(p)
            moves.append(ms)
            probs.append(p)
        w = {"maia0": np.array([p.wdl for p in m0], float), "maia600": np.array([p.wdl for p in m6], float),
             "lc0raw": np.array([p.wdl for p in pol], float)}
        w = {k: v / v.sum(1, keepdims=True) for k, v in w.items()}
        leaf = self.leaf.from_sources(w, queries)
        return moves, probs, leaf

    def close(self):
        pass


class ComboMover:
    """``Combo(prune="p0.5")`` for the moves; the leaf from Combo's own Maia and Lc0 policy networks (one more
    call of each: Combo does not hand out the value heads of its internal calls), so no network is loaded twice."""

    def __init__(self):
        from tnh.eval.result import ResultModel
        from tnh.models.combo import Combo
        from tnh.models.maia3 import Elo

        self.combo = Combo(prune="p0.5", maia_batch=64)
        self.elos = [Elo.offset(0.0), Elo.offset(600.0)]
        self.leaf = ResultModel("cheap")
        self.name = "combo"

    def evaluate(self, queries):
        m0, m6 = self.combo.maia.predict_many(queries, self.elos)
        pol = self.combo.policy.predict(queries)
        w = {"maia0": np.array([p.wdl for p in m0], float), "maia600": np.array([p.wdl for p in m6], float),
             "lc0raw": np.array([p.wdl for p in pol], float)}
        w = {k: v / v.sum(1, keepdims=True) for k, v in w.items()}
        leaf = self.leaf.from_sources(w, queries)
        preds = self.combo.predict(queries)
        moves = [p.moves for p in preds]
        probs = []
        for p in preds:
            e = np.exp(p.logp.astype(np.float64))
            probs.append(e / e.sum())
        return moves, probs, leaf

    def close(self):
        pass


class ModelMover:
    """The lines drawn from one of three move models of known bias toward large mistakes, the leaf the cheap
    result model on the same Maia (+0, +600) and Lc0 policy value heads:

    - ``maia600``: Maia-3 79M at + 600 alone (its own probabilities);
    - ``trees``: ``PolicyTrees`` (the default move model's moves), value heads from its own calls;
    - ``combo-trees``: ``Combo(prune="p0.5", trees=True)``, the leaf from one more call of its Maia and policy."""

    def __init__(self, kind: str):
        from tnh.eval.result import ResultModel
        from tnh.models.maia3 import Elo

        self.kind, self.name = kind, kind
        self.elos = [Elo.offset(0.0), Elo.offset(600.0)]
        self.leaf = ResultModel("cheap")
        if kind == "maia600":
            from tnh.models.lc0 import Lc0Policy
            from tnh.models.maia3 import Maia3
            self.policy = Lc0Policy()      # first: the GPU is shared, Lc0's backend wants its memory early
            self.maia = Maia3("79m", batch_size=64)
        elif kind == "trees":
            from tnh.models.combo import PolicyTrees
            self.trees = PolicyTrees(maia_batch=64)
        elif kind == "combo-trees":
            from tnh.models.combo import Combo
            self.combo = Combo(prune="p0.5", trees=True, maia_batch=64)
        else:
            raise ValueError(kind)

    def evaluate(self, queries):
        if self.kind == "maia600":
            m0, m6 = self.maia.predict_many(queries, self.elos)
            pol = self.policy.predict(queries)
            preds = m6
        elif self.kind == "trees":
            preds, (m0, m6, pol) = self.trees.predict_with_calls(queries)
        else:
            m0, m6 = self.combo.maia.predict_many(queries, self.elos)
            pol = self.combo.policy.predict(queries)
            preds = self.combo.predict(queries)
        w = {"maia0": np.array([p.wdl for p in m0], float), "maia600": np.array([p.wdl for p in m6], float),
             "lc0raw": np.array([p.wdl for p in pol], float)}
        w = {k: v / v.sum(1, keepdims=True) for k, v in w.items()}
        leaf = self.leaf.from_sources(w, queries)
        moves, probs = [], []
        for p in preds:
            e = np.exp(p.logp.astype(np.float64))
            moves.append(p.moves)
            probs.append(e / e.sum())
        return moves, probs, leaf

    def close(self):
        pass


# --- nodes --------------------------------------------------------------------------------------

def terminal(board: chess.Board) -> np.ndarray | None:
    """The exact wdl for the side to move if the game is over, else None."""
    if board.is_checkmate():
        return np.array([0.0, 0.0, 1.0])
    if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() \
            or board.is_fivefold_repetition():
        return np.array([0.0, 1.0, 0.0])
    return None


class Node:
    __slots__ = ("board", "history", "mover", "opp", "depth")

    def __init__(self, board, history, mover, opp, depth):
        self.board, self.history, self.mover, self.opp, self.depth = board, history, mover, opp, depth

    def child(self, uci: str) -> "Node":
        b = self.board.copy(stack=False)
        b.push_uci(uci)
        return Node(b, self.history + [uci], self.opp, self.mover, self.depth + 1)

    def query(self) -> Query:
        return Query(self.board.fen(), self.history, int(self.mover), int(self.opp))


def evaluate_nodes(ev, nodes: list[Node], batch: int = 1024):
    """moves, probs and leaf (to the node's side to move) per node; terminal nodes get their exact result."""
    out = [None] * len(nodes)
    live = []
    for i, n in enumerate(nodes):
        t = terminal(n.board)
        if t is None:
            live.append(i)
        else:
            out[i] = ([], np.zeros(0), t)
    for s in range(0, len(live), batch):
        idx = live[s:s + batch]
        moves, probs, leaf = ev.evaluate([nodes[i].query() for i in idx])
        for k, i in enumerate(idx):
            out[i] = (moves[k], probs[k], leaf[k])
    return out


def to_root(w: np.ndarray, depth: int) -> np.ndarray:
    return w if depth % 2 == 0 else w[..., FLIP]


def exact(ev, roots: list[Node], root_eval, p_min: float, stats: dict):
    """One- and two-ply expectations over the moves above p_min (renormalised)."""
    def kids(n, e):
        ms, ps, _ = e
        keep = [(m, p) for m, p in zip(ms, ps) if p >= p_min] or ([(ms[int(np.argmax(ps))], 1.0)] if len(ms) else [])
        z = sum(p for _, p in keep)
        return [(n.child(m), p / z) for m, p in keep]

    level1 = [kids(n, e) for n, e in zip(roots, root_eval)]
    flat1 = [c for ks in level1 for c, _ in ks]
    t = time.perf_counter()
    e1 = evaluate_nodes(ev, flat1)
    stats["exact1_seconds"] += time.perf_counter() - t
    stats["exact1_evals"] += len(flat1)
    level2, flat2 = [], []
    k = 0
    for ks in level1:
        row = []
        for c, p in ks:
            gk = kids(c, e1[k]) if len(e1[k][0]) else []
            row.append(gk)
            flat2 += [g for g, _ in gk]
            k += 1
        level2.append(row)
    t = time.perf_counter()
    e2 = evaluate_nodes(ev, flat2)
    stats["exact2_seconds"] += time.perf_counter() - t
    stats["exact2_evals"] += len(flat2)
    v1, v2 = [], []
    i1 = i2 = 0
    for ks, rows in zip(level1, level2):
        a1, a2 = np.zeros(3), np.zeros(3)
        for (c, p), gk in zip(ks, rows):
            leaf1 = to_root(e1[i1][2], 1)
            a1 += p * leaf1
            if gk:
                acc = np.zeros(3)
                for g, q in gk:
                    acc += q * to_root(e2[i2][2], 2)
                    i2 += 1
                a2 += p * acc
            else:
                a2 += p * leaf1      # the child ends the game: its exact result
            i1 += 1
        if not ks:                    # the root itself is terminal (not expected in the samples)
            a1 = a2 = np.full(3, np.nan)
        v1.append(a1)
        v2.append(a2)
    return np.array(v1), np.array(v2)


def exact_full(ev, roots: list[Node], root_eval, p_min: float, stats: dict) -> list[dict]:
    """Per root: every legal move's probability and leaf wdl (root's side), and for the moves with p >= p_min
    every legal reply's probability and leaf wdl (root's side), so that any backup over two plies (truncated
    or not, expectation or a share on the best reply) can be computed offline. A child that ends the game has
    its exact result and no replies."""
    flat1, owner = [], []
    for i, (n, e) in enumerate(zip(roots, root_eval)):
        for m in e[0]:
            flat1.append(n.child(m))
            owner.append(i)
    t = time.perf_counter()
    e1 = evaluate_nodes(ev, flat1)
    stats["exact1_seconds"] += time.perf_counter() - t
    stats["exact1_evals"] += len(flat1)
    out = [{"c_p": [], "c_wdl": [], "c_kept": [], "r_p": [], "r_wdl": []} for _ in roots]
    flat2, owner2 = [], []
    k = 0
    for i, e in enumerate(root_eval):
        for m, p in zip(e[0], e[1]):
            c, ce = flat1[k], e1[k]
            o = out[i]
            o["c_p"].append(float(p))
            o["c_wdl"].append(to_root(np.asarray(ce[2], float), 1).tolist())
            kept = bool(p >= p_min)
            o["c_kept"].append(kept)
            if kept:
                o["r_p"].append([float(q) for q in ce[1]])
                j = len(o["r_p"]) - 1
                for r in ce[0]:
                    flat2.append(c.child(r))
                    owner2.append((i, j))
            k += 1
    t = time.perf_counter()
    e2 = evaluate_nodes(ev, flat2)
    stats["exact2_seconds"] += time.perf_counter() - t
    stats["exact2_evals"] += len(flat2)
    for o in out:
        o["r_wdl"] = [[] for _ in o["r_p"]]
    for (i, j), ge in zip(owner2, e2):
        out[i]["r_wdl"][j].append(to_root(np.asarray(ge[2], float), 2).tolist())
    return out


def monte_carlo(ev, roots: list[Node], root_eval, n_lines: int, depth: int, rng, stats: dict) -> dict:
    """Values at each depth in DEPTHS, per line: array roots x lines x len(DEPTHS) x 3 (root's side)."""
    R = len(roots)
    vals = np.full((R, n_lines, len(DEPTHS), 3), np.nan)
    cur = [r for r in roots for _ in range(n_lines)]
    cur_eval = [e for e in root_eval for _ in range(n_lines)]
    done = [None] * len(cur)          # exact terminal value at the root's side, once a line ends
    for d in range(1, depth + 1):
        nxt = []
        for j, (n, e) in enumerate(zip(cur, cur_eval)):
            ms, ps, _ = e
            if done[j] is not None or not len(ms):
                nxt.append(n)
                if done[j] is None:
                    done[j] = to_root(e[2], n.depth)
                continue
            nxt.append(n.child(ms[rng.choice(len(ms), p=ps / ps.sum())]))
        live = [j for j in range(len(nxt)) if done[j] is None]
        t = time.perf_counter()
        ev_live = evaluate_nodes(ev, [nxt[j] for j in live])
        stats["mc_seconds"] += time.perf_counter() - t
        stats["mc_evals"] += len(live)
        new_eval = list(cur_eval)
        for k, j in enumerate(live):
            new_eval[j] = ev_live[k]
            if not len(ev_live[k][0]):
                done[j] = to_root(ev_live[k][2], nxt[j].depth)
        cur, cur_eval = nxt, new_eval
        if d in DEPTHS:
            di = DEPTHS.index(d)
            for j in range(len(cur)):
                v = done[j] if done[j] is not None else to_root(cur_eval[j][2], cur[j].depth)
                vals[j // n_lines, j % n_lines, di] = v
    return vals


# --- commands ------------------------------------------------------------------------------------

def cmd_positions(a) -> None:
    """Draw the part B positions: plies 10-40, from the result model's test games and fitting games."""
    from tnh.eval.common import read_sample
    for src, dst, n in ((CORPUS / "samples" / "test_value_all.parquet", POS_TEST, a.n),
                        (CORPUS / "samples" / "fit_value20k.parquet", POS_FIT, a.n_fit)):
        s = read_sample(src).filter(pl.col("ply").is_between(10, 40))
        s = s.sample(n=min(n, s.height), seed=a.seed).sort("game_id", "ply")
        s.write_parquet(dst)
        print(f"{dst}: {s.height} positions (plies 10-40) from {src.name}")


def cmd_run(a) -> None:
    from tnh.models.run import queries as make_queries

    s = pl.read_parquet(a.positions).with_columns(pl.col("ply").cast(pl.Int64))
    if a.limit:
        s = s.head(a.limit)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{a.mover}{a.suffix}.parquet"
    done = pl.read_parquet(out).select("game_id", "ply") if out.exists() else None
    if done is not None:
        s = s.join(done, on=["game_id", "ply"], how="anti")
    g = (pl.scan_parquet(CORPUS / "games.parquet").select("game_id", "moves_uci")
         .join(pl.LazyFrame({"game_id": s.get_column("game_id").unique()}), on="game_id", how="semi").collect())
    s = s.join(g, on="game_id", how="inner", maintain_order="left")
    from tnh.eval.sample import drop_cache, peak_gb
    drop_cache(CORPUS / "games.parquet")       # a capped job's memory counts the page cache it causes
    print(f"{s.height} positions to do ({out})", flush=True)
    ev = FastMover() if a.mover == "fast" else ComboMover() if a.mover == "combo" else ModelMover(a.mover)
    rng = np.random.default_rng(a.seed)
    rows_all = []
    for start in range(0, s.height, a.chunk):
        c = s.slice(start, a.chunk)
        qs = make_queries(c)
        roots = [Node(chess.Board(q.fen), list(q.moves_uci), q.mover_elo, q.opp_elo, 0) for q in qs]
        stats = {k: 0.0 for k in ("root_seconds", "exact1_seconds", "exact2_seconds", "mc_seconds",
                                  "root_evals", "exact1_evals", "exact2_evals", "mc_evals")}
        t = time.perf_counter()
        re = evaluate_nodes(ev, roots)
        stats["root_seconds"] += time.perf_counter() - t
        stats["root_evals"] += len(roots)
        full = exact_full(ev, roots, re, a.p_min, stats) if a.full_exact else None
        v1, v2 = exact(ev, roots, re, a.p_min, stats) if not (a.no_exact or a.full_exact) else (None, None)
        mc = monte_carlo(ev, roots, re, a.n_lines, a.depth, rng, stats)
        rows = {"game_id": c.get_column("game_id").to_list(), "ply": c.get_column("ply").to_list(),
                "static": [list(map(float, e[2])) for e in re]}
        if full is not None:
            for key in ("c_p", "c_wdl", "c_kept", "r_p", "r_wdl"):
                rows[key] = [f[key] for f in full]
        if v1 is not None:
            rows["exact1"] = v1.tolist()
            rows["exact2"] = v2.tolist()
        for di, d in enumerate(DEPTHS):
            if d > a.depth:
                continue
            for n in NS:
                if n > a.n_lines:
                    continue
                rows[f"mc_n{n}_d{d}"] = np.nanmean(mc[:, :n, di], axis=1).tolist()
        df = pl.DataFrame(rows)
        for k, v in stats.items():
            df = df.with_columns(pl.lit(v / len(qs)).alias(f"per_position_{k}"))
        rows_all.append(df)
        prev = pl.read_parquet(out) if out.exists() else None
        (df if prev is None else pl.concat([prev, df], how="vertical_relaxed")).write_parquet(out)
        print(f"{start + c.height}/{s.height}: " + ", ".join(f"{k} {v / len(qs):.3f}" for k, v in stats.items())
              + f"; peak RSS {peak_gb():.2f} GB", flush=True)
    ev.close()


def cmd_analyse(a) -> None:
    from tnh.eval.common import boot_counts, boot_means, cluster_codes, est, read_sample, write_result
    from tnh.eval.result import losses

    test = read_sample(a.positions).with_columns(pl.col("ply").cast(pl.Int64))
    res = {}
    for mover in ("fast", "fast_exact", "combo"):
        f = OUT / f"{mover}.parquet"
        if not f.exists():
            continue
        d = pl.read_parquet(f).with_columns(pl.col("ply").cast(pl.Int64))
        j = test.join(d, on=["game_id", "ply"], how="inner")
        y = np.array([{1.0: 0, 0.5: 1, 0.0: 2}[x] for x in j.get_column("result_mover").to_list()])
        cl = cluster_codes(j)
        counts = boot_counts(cl.max() + 1, a.boot, a.seed)
        static = np.array(j.get_column("static").to_list())
        full = None
        if a.predictions.exists():
            pr = pl.read_parquet(a.predictions).select("game_id", pl.col("ply").cast(pl.Int64), pl.col(a.best).alias("_full"))
            jj = j.select("game_id", "ply").join(pr, on=["game_id", "ply"], how="left", maintain_order="left")
            if jj.get_column("_full").null_count() == 0:
                full = np.array(jj.get_column("_full").to_list())
        variants = [c for c in d.columns if c in ("exact1", "exact2") or c.startswith("mc_")]
        ll_s, br_s = losses(static, y)
        w = np.ones(len(y))
        rows = {"static": {"logloss": est(ll_s.mean(), boot_means(ll_s, w, cl, counts)[:, 0]),
                           "brier": est(br_s.mean(), boot_means(br_s, w, cl, counts)[:, 0])}}
        es = static[:, 0] + 0.5 * static[:, 1]
        score = np.array([1.0, 0.5, 0.0])[y]
        for v in variants:
            p = np.array(j.get_column(v).to_list())
            p = p / p.sum(1, keepdims=True)
            ll, br = losses(p, y)
            X = np.column_stack([ll, br, ll - ll_s, br - br_s])
            bm = boot_means(X, w, cl, counts)
            el = p[:, 0] + 0.5 * p[:, 1]
            shift = el - es
            # score = a + b * static + c * shift (least squares, linear in the expected score)
            A = np.column_stack([np.ones(len(y)), es, shift])
            coef = np.linalg.lstsq(A, score, rcond=None)[0]
            bc = []
            for r in range(min(a.boot, 300)):
                ww = counts[r][cl]
                Aw = A * np.sqrt(ww)[:, None]
                bc.append(np.linalg.lstsq(Aw, score * np.sqrt(ww), rcond=None)[0])
            bc = np.array(bc)
            # the best blend of the two distributions, (1 - lam) static + lam look-ahead, chosen out of fold
            lam_grid = np.linspace(0, 1, 21)
            folds = cl % 5
            oof = np.zeros(len(y))
            lams = []
            for k in range(5):
                tr, te = folds != k, folds == k
                lb = min(lam_grid, key=lambda l: losses((1 - l) * static[tr] + l * p[tr], y[tr])[0].mean())
                lams.append(lb)
                oof[te] = losses((1 - lb) * static[te] + lb * p[te], y[te])[0]
            dob = oof - ll_s
            bo = boot_means(np.column_stack([oof, dob]), w, cl, counts)
            # the same out-of-fold blend on top of the best static model (the full result model)
            blend_full = None
            if full is not None:
                ll_f = losses(full, y)[0]
                oof_f, lams_f = np.zeros(len(y)), []
                for k in range(5):
                    tr, te = folds != k, folds == k
                    lb = min(lam_grid, key=lambda l: losses((1 - l) * full[tr] + l * p[tr], y[tr])[0].mean())
                    lams_f.append(lb)
                    oof_f[te] = losses((1 - lb) * full[te] + lb * p[te], y[te])[0]
                bf = boot_means(np.column_stack([oof_f, oof_f - ll_f]), w, cl, counts)
                blend_full = {"lambdas_by_fold": lams_f, "oof_logloss": est(oof_f.mean(), bf[:, 0]),
                              "oof_minus_static": est((oof_f - ll_f).mean(), bf[:, 1])}
            rows[v] = {"logloss": est(X[:, 0].mean(), bm[:, 0]), "brier": est(X[:, 1].mean(), bm[:, 1]),
                       "logloss_minus_static": est(X[:, 2].mean(), bm[:, 2]),
                       "brier_minus_static": est(X[:, 3].mean(), bm[:, 3]),
                       "regression": {"intercept": coef[0], "static": coef[1], "shift": coef[2],
                                      "static_ci": np.quantile(bc[:, 1], [0.025, 0.975]).tolist(),
                                      "shift_ci": np.quantile(bc[:, 2], [0.025, 0.975]).tolist()},
                       "blend": {"lambdas_by_fold": lams, "oof_logloss": est(oof.mean(), bo[:, 0]),
                                 "oof_minus_static": est(dob.mean(), bo[:, 1])},
                       **({"blend_full": blend_full} if blend_full else {}),
                       "shift": {"mean": float(shift.mean()), "sd": float(shift.std()),
                                 "abs_mean": float(np.abs(shift).mean())}}
        # cost: evaluations and seconds per position
        cost = {k.replace("per_position_", ""): float(d.get_column(k).mean()) for k in d.columns if k.startswith("per_position_")}
        # shift by ply band and rating band for the deepest MC variant
        deep = [v for v in variants if v.startswith("mc_n32")]
        by = {}
        if deep:
            v = deep[-1]
            p = np.array(j.get_column(v).to_list())
            shift = p[:, 0] + 0.5 * p[:, 1] - es
            ply = j.get_column("ply").to_numpy()
            mean = (j.get_column("mover_elo").to_numpy() + j.get_column("opp_elo").to_numpy()) / 2
            for name_, vals, edges, labels in (("ply", ply, [10, 20, 30, 41], ["10-19", "20-29", "30-40"]),
                                               ("rating", mean, [0, 2000, 2200, 2400, 9999],
                                                ["1800-1999", "2000-2199", "2200-2399", "2400+"])):
                g = np.searchsorted(edges, vals, side="right") - 1
                by[name_] = [{"group": lab, "n": int((g == i).sum()), "mean": float(shift[g == i].mean()),
                              "sd": float(shift[g == i].std())} for i, lab in enumerate(labels) if (g == i).any()]
        res[mover] = {"positions": len(y), "variants": rows, "cost": cost, "shift_by": by, "variant_for_shift_by": deep[-1] if deep else None}
        print(f"{mover}: {len(y)} positions; static {ll_s.mean():.4f}")
        for v, r in rows.items():
            if v != "static":
                print(f"  {v:>12}: {r['logloss']['est']:.4f} minus static {r['logloss_minus_static']['est']:+.4f} "
                      f"[{r['logloss_minus_static']['lo']:+.4f}, {r['logloss_minus_static']['hi']:+.4f}] "
                      f"shift coef {r['regression']['shift']:+.2f} [{r['regression']['shift_ci'][0]:+.2f}, {r['regression']['shift_ci'][1]:+.2f}]")
    write_result(a.results, a.name, {"what": "look-ahead with the move model against the static result model", **res})


def cmd_bias(a) -> None:
    """Is the look-ahead's loss with depth the move model's bias toward large mistakes? Per run and variant:
    predicted draws and decisive mass against the observed rates, the calibration slope of the expected
    score (score on predicted expected score, least squares; below 1 = too extreme), and the loss against
    the static model before and after step 1's five-parameter recalibration (``tnh.eval.value.fit_recal``),
    cross-fitted on two halves of the tournaments (each half recalibrated on the other), for the variant
    and the static model alike. Then the movers of the bias run compared on their common positions."""
    from tnh.eval.common import boot_counts, boot_means, cluster_codes, est, read_sample, write_result
    from tnh.eval.result import losses
    from tnh.eval.value import _logp, fit_recal, recal_probs

    test = read_sample(a.positions).with_columns(pl.col("ply").cast(pl.Int64))
    res, per_row = {}, {}

    def recal(p, y, r, fold):
        out = np.zeros_like(p)
        for k in (0, 1):
            tr, te = fold != k, fold == k
            th = fit_recal(p[tr], y[tr], r[tr], np.ones(tr.sum()))
            out[te] = recal_probs(th, _logp(p[te]), r[te])
        return out

    for mover in a.movers:
        f = OUT / f"{mover}.parquet"
        if not f.exists():
            continue
        d = pl.read_parquet(f).with_columns(pl.col("ply").cast(pl.Int64))
        j = test.join(d, on=["game_id", "ply"], how="inner").sort("game_id", "ply")
        y = np.array([{1.0: 0, 0.5: 1, 0.0: 2}[x] for x in j.get_column("result_mover").to_list()])
        score = np.array([1.0, 0.5, 0.0])[y]
        r = ((j.get_column("mover_elo") + j.get_column("opp_elo")).to_numpy() / 2 - 2200) / 100
        cl = cluster_codes(j)
        fold = cl % 2
        counts = boot_counts(cl.max() + 1, a.boot, a.seed)
        w = np.ones(len(y))
        variants = ["static"] + [c for c in d.columns if c in ("exact1", "exact2") or c.startswith("mc_")]
        P = {v: np.array(j.get_column(v).to_list()) for v in variants}
        P = {v: p / p.sum(1, keepdims=True) for v, p in P.items()}
        R = {v: recal(p, y, r, fold) for v, p in P.items()}
        ll_s, llr_s = losses(P["static"], y)[0], losses(R["static"], y)[0]
        rows = {}
        for v in variants:
            p, q = P[v], R[v]
            e = p[:, 0] + 0.5 * p[:, 1]
            ll, llr = losses(p, y)[0], losses(q, y)[0]
            X = np.column_stack([ll - ll_s, llr - llr_s, (ll - ll_s) - (llr - llr_s), p[:, 1], 1 - p[:, 1],
                                 q[:, 1]])
            bm = boot_means(X, w, cl, counts)
            A = np.column_stack([np.ones(len(y)), e])
            slope = np.linalg.lstsq(A, score, rcond=None)[0][1]
            bs = []
            for b in range(min(a.boot, 300)):
                sw = np.sqrt(counts[b][cl])
                bs.append(np.linalg.lstsq(A * sw[:, None], score * sw, rcond=None)[0][1])
            rows[v] = {"logloss": float(ll.mean()), "recal_logloss": float(llr.mean()),
                       "minus_static": est(X[:, 0].mean(), bm[:, 0]),
                       "recal_minus_recal_static": est(X[:, 1].mean(), bm[:, 1]),
                       "removed_by_recal": est(X[:, 2].mean(), bm[:, 2]),
                       "pred_draw": float(p[:, 1].mean()), "pred_decisive": float(1 - p[:, 1].mean()),
                       "recal_pred_draw": float(q[:, 1].mean()),
                       "pred_score": float(e.mean()), "score_slope": float(slope),
                       "score_slope_ci": np.quantile(bs, [0.025, 0.975]).tolist(), "sd_expected_score": float(e.std())}
            per_row[(mover, v)] = (j.select("game_id", "ply"), ll - ll_s, llr - llr_s)
        cost = {k.replace("per_position_", ""): float(d.get_column(k).mean()) for k in d.columns if k.startswith("per_position_")}
        res[mover] = {"positions": len(y), "observed_draw": float((y == 1).mean()), "observed_score": float(score.mean()),
                      "variants": rows, "cost": cost}
        print(f"\n{mover}: {len(y)} positions, observed draws {(y == 1).mean():.3f}, score {score.mean():.3f}")
        print(f"{'variant':>10} | pred draw | slope E | minus static | after recal | removed")
        for v, rr in rows.items():
            f_ = lambda e: f"{e['est']:+.4f} [{e['lo']:+.4f}, {e['hi']:+.4f}]"   # noqa: E731
            print(f"{v:>10} | {rr['pred_draw']:.3f} | {rr['score_slope']:.2f} [{rr['score_slope_ci'][0]:.2f}, "
                  f"{rr['score_slope_ci'][1]:.2f}] | {f_(rr['minus_static'])} | {f_(rr['recal_minus_recal_static'])} | "
                  f"{f_(rr['removed_by_recal'])}", flush=True)
    # the movers of the bias run on their common positions: degradation per mover and paired between movers
    movers = [m for m in ("maia600", "trees", "combo-trees") if m in res]
    cmp = {}
    if len(movers) > 1:
        keys = None
        for m in movers:
            k = per_row[(m, "static")][0]
            keys = k if keys is None else keys.join(k, on=["game_id", "ply"], how="inner")
        sub = test.join(keys, on=["game_id", "ply"], how="inner").sort("game_id", "ply")
        cl = cluster_codes(sub)
        counts = boot_counts(cl.max() + 1, a.boot, a.seed)

        def aligned(m, v, i):
            k, *vals = per_row[(m, v)]
            jj = sub.select("game_id", "ply").join(k.with_columns(pl.Series("_x", vals[i])), on=["game_id", "ply"],
                                                   how="left", maintain_order="left")
            return jj.get_column("_x").to_numpy()

        for v in ("mc_n8_d2", "mc_n8_d4", "mc_n8_d8"):
            cols, names = [], []
            for i, kind in ((0, "raw"), (1, "recal")):
                xs = {m: aligned(m, v, i) for m in movers}
                for m in movers:
                    cols.append(xs[m]); names.append(f"{m} {kind}")
                for m1, m2 in list(zip(movers, movers[1:])) + ([(movers[0], movers[-1])] if len(movers) > 2 else []):
                    cols.append(xs[m2] - xs[m1]); names.append(f"{m2} - {m1} {kind}")
            X = np.column_stack(cols)
            bm = boot_means(X, np.ones(len(X)), cl, counts)
            cmp[v] = {n: est(X[:, c].mean(), bm[:, c]) for c, n in enumerate(names)}
        print(f"\nmovers on {sub.height} common positions (minus static; differences of that):")
        for v, rr in cmp.items():
            for n, e in rr.items():
                print(f"  {v} {n:>28}: {e['est']:+.4f} [{e['lo']:+.4f}, {e['hi']:+.4f}]")
        cmp["positions"] = sub.height
    write_result(a.results, a.name, {"what": "look-ahead degradation against the static model: draws, calibration, "
                                     "and what a cross-fitted five-parameter recalibration removes", "runs": res,
                                     "movers_common_positions": cmp})


def cmd_engine(a) -> None:
    """Look-ahead by search: Lc0's calibrated WDL at 400 and 1600 nodes in place of the 50-node WDL, with the
    coefficients fitted at 50 nodes (``reports/step1/result.json``: the ``lc0w`` and ``all`` variants), on the
    test positions that have them; paired against 50 nodes on the same positions."""
    from tnh.eval.common import boot_counts, boot_means, cluster_codes, est, read_sample, write_result
    from tnh.eval.result import SOURCES, design, losses, mlogit_probs, table
    from tnh.paths import FEATURES

    res = json.loads((a.results / "result.json").read_text())["variants"]
    coef = {"alone": (np.array(res["lc0w"]["coef"]), ["lc0w"]), "full": (np.array(res["all"]["coef"]), res["all"]["sources"])}
    test = read_sample(a.positions).with_columns(pl.col("ply").cast(pl.Int64))
    base, _, _ = table(test, SOURCES)
    out = {}
    for nodes in (400, 1600):
        d = FEATURES / "lc0-bt4" / f"nodes{nodes}-wdl-offset+400"
        if not d.exists():
            continue
        w, ctx, y = table(test, {**SOURCES, "lc0w": d})
        ok = ~np.isnan(w["lc0w"]).any(1)
        sub = test.filter(pl.Series(ok))
        cl = cluster_codes(sub)
        counts = boot_counts(cl.max() + 1, a.boot, a.seed)
        for name, (th, srcs) in coef.items():
            p = mlogit_probs(th, design({k: v[ok] for k, v in w.items()}, ctx[ok], srcs))
            p50 = mlogit_probs(th, design({k: v[ok] for k, v in base.items()}, ctx[ok], srcs))
            ll, br = losses(p, y[ok])
            l50, b50 = losses(p50, y[ok])
            X = np.column_stack([ll, br, ll - l50, br - b50, l50])
            bm = boot_means(X, np.ones(ok.sum()), cl, counts)
            key = f"lc0w{nodes} {name}"
            out[key] = {"nodes": nodes, "positions": int(ok.sum()), "logloss": est(ll.mean(), bm[:, 0]),
                        "brier": est(br.mean(), bm[:, 1]), "minus_50_nodes": est(X[:, 2].mean(), bm[:, 2]),
                        "brier_minus_50_nodes": est(X[:, 3].mean(), bm[:, 3]), "logloss_50_same_rows": est(l50.mean(), bm[:, 4]),
                        "coefficients": "fitted at 50 nodes (no refit)"}
            print(f"{key}: {ll.mean():.4f} against 50 nodes {l50.mean():.4f}, difference {X[:, 2].mean():+.4f} "
                  f"[{np.quantile(bm[:, 2], 0.025):+.4f}, {np.quantile(bm[:, 2], 0.975):+.4f}] on {ok.sum()} positions")
    write_result(a.results, "lookahead_engine", {"what": "Lc0 calibrated WDL at 400 and 1600 nodes in place of 50, "
                                                 "coefficients fitted at 50 nodes; look-ahead test positions", **out})


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m tnh.eval.lookahead", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(required=True)
    sp = sub.add_parser("positions")
    sp.add_argument("--n", type=int, default=2000)
    sp.add_argument("--n-fit", type=int, default=2000)
    sp.add_argument("--seed", type=int, default=11)
    sp.set_defaults(fn=cmd_positions)
    sp = sub.add_parser("run")
    sp.add_argument("--positions", type=Path, default=POS_TEST)
    sp.add_argument("--mover", choices=["fast", "combo", "maia600", "trees", "combo-trees"], default="fast")
    sp.add_argument("--suffix", default="")
    sp.add_argument("--n-lines", type=int, default=32)
    sp.add_argument("--depth", type=int, default=16)
    sp.add_argument("--p-min", type=float, default=0.02)
    sp.add_argument("--no-exact", action="store_true")
    sp.add_argument("--full-exact", action="store_true", help="per child and reply values (exact_full) instead of the averages")
    sp.add_argument("--limit", type=int)
    sp.add_argument("--chunk", type=int, default=50)
    sp.add_argument("--seed", type=int, default=5)
    sp.set_defaults(fn=cmd_run)
    sp = sub.add_parser("bias")
    sp.add_argument("--positions", type=Path, default=POS_TEST)
    sp.add_argument("--movers", nargs="+", default=["fast", "fast_exact", "combo", "maia600", "trees", "combo-trees"])
    sp.add_argument("--results", type=Path, default=REPORTS / "step1")
    sp.add_argument("--name", default="lookahead_bias")
    sp.add_argument("--boot", type=int, default=1000)
    sp.add_argument("--seed", type=int, default=7)
    sp.set_defaults(fn=cmd_bias)
    sp = sub.add_parser("engine")
    sp.add_argument("--positions", type=Path, default=POS_TEST)
    sp.add_argument("--results", type=Path, default=REPORTS / "step1")
    sp.add_argument("--boot", type=int, default=1000)
    sp.add_argument("--seed", type=int, default=7)
    sp.set_defaults(fn=cmd_engine)
    sp = sub.add_parser("analyse")
    sp.add_argument("--positions", type=Path, default=POS_TEST)
    sp.add_argument("--results", type=Path, default=REPORTS / "step1")
    sp.add_argument("--name", default="lookahead")
    sp.add_argument("--predictions", type=Path, default=Path.home() / "out/tn-hittare/scratch/eval/result/predictions.parquet")
    sp.add_argument("--best", default="all", help="the result model's variant to blend on top of")
    sp.add_argument("--boot", type=int, default=1000)
    sp.add_argument("--seed", type=int, default=7)
    sp.set_defaults(fn=cmd_analyse)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
