"""The difficulty measures, from the cached model calls.

Per position P with side to move S and opponent O. ``p`` is Maia-3's move
distribution at the mapped ratings, ``v`` Stockfish's expected score for the
mover after each move (MultiPV, every legal move), ``loss = max(v) - v``.

- ``el_stm``: S's expected loss in the move, ``sum p * loss``.
- ``hold_stm``: the narrow path, the probability on moves with ``loss <= eps``.
- ``el_next``, ``hold_next``: the same for O in the positions after S's move,
  averaged with S's move probabilities over the children that were evaluated
  (renormalised; ``child_cover`` is the probability they cover).
- ``net2``: ``el_stm - el_next``, S's loss now minus O's loss one ply later.
- ``maia_es_stm``: Maia's value head, expected score for S at the mapped ratings.
- ``engine_es_stm``, ``engine_cp_stm``: Stockfish's root search (MultiPV 1, the
  larger budget), for S.
- ``el_db``, ``hold_db``: the same loss and hold with the games' own move shares
  in place of Maia's (only in-filter games). Not a predictor: it uses the moves
  of the games whose results are being explained. It shows how much a perfect
  model of the move at P could explain.

Everything here is from S's point of view; `analysis` turns it into White's.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from tnh.difficulty.compute import Budget, children_of, maia_elo, open_caches
from tnh.data.edges import pos_hash

import chess


def loss_measures(p: np.ndarray, value: np.ndarray, eps: float, best: float | None = None) -> tuple[float, float, int]:
    """(expected loss, probability of holding, moves without a value).

    Moves without an engine value are left out and ``p`` renormalised over the rest.
    ``best`` is the best move's value when ``value`` covers only some of the moves.
    """
    ok = ~np.isnan(value)
    if not ok.any():
        return float("nan"), float("nan"), int((~ok).sum())
    pp = p[ok] / p[ok].sum()
    loss = (np.max(value[ok]) if best is None else best) - value[ok]
    return float(np.sum(pp * loss)), float(np.sum(pp[loss <= eps])), int((~ok).sum())


def _aligned(moves_a: list[str], moves_b: list[str], arr_b) -> np.ndarray:
    idx = {m: i for i, m in enumerate(moves_b)}
    if set(idx) != set(moves_a):
        raise ValueError(f"move lists differ: {sorted(set(moves_a) ^ set(idx))[:5]}")
    a = np.asarray(arr_b, np.float64)
    return np.array([a[idx[m]] for m in moves_a])


def build(positions: pl.DataFrame, b: Budget, *, eps: float = 0.03, db_moves: pl.DataFrame | None = None) -> tuple[pl.DataFrame, dict]:
    """One row per position with every measure above; positions missing a model call are dropped and counted."""
    maia_c, sf_c, root_c = open_caches(b)
    dbm: dict[int, tuple[list[str], np.ndarray]] = {}
    if db_moves is not None:
        for (h,), g in db_moves.group_by(["pos_hash"]):
            dbm[int(h)] = (g["move"].to_list(), g["n"].to_numpy().astype(np.float64))
    out, counts = [], {"missing_maia": 0, "missing_multipv": 0, "missing_root": 0, "children_missing": 0,
                       "moves_without_value": 0, "db_moves_not_legal": 0}
    for r in positions.select("pos_hash", "fen", "white_elo", "black_elo", "stm_white").iter_rows(named=True):
        h = r["pos_hash"]
        w, bl = maia_elo(r["white_elo"]), maia_elo(r["black_elo"])
        se, oe = (w, bl) if r["stm_white"] else (bl, w)
        if not maia_c.has((h, se, oe)):
            counts["missing_maia"] += 1
            continue
        if not sf_c.has((h,)):
            counts["missing_multipv"] += 1
            continue
        if not root_c.has((h,)):
            counts["missing_root"] += 1
            continue
        m, s, rt = maia_c.get((h, se, oe)), sf_c.get((h,)), root_c.get((h,))
        p = np.exp(np.asarray(m["logp"], np.float64))
        p /= p.sum()
        v = _aligned(m["moves"], s["moves"], s["value"])
        el, hold, nn = loss_measures(p, v, eps)
        counts["moves_without_value"] += nn
        # one ply deeper
        board = chess.Board(r["fen"])
        els, holds, ps = [], [], []
        for mv, pc in children_of(m["moves"], p, b):
            bc = board.copy(stack=False)
            bc.push_uci(mv)
            if bc.is_game_over(claim_draw=False):
                continue
            hc = pos_hash(bc)
            if not (maia_c.has((hc, oe, se)) and sf_c.has((hc,))):
                counts["children_missing"] += 1
                continue
            mc, sc = maia_c.get((hc, oe, se)), sf_c.get((hc,))
            pcm = np.exp(np.asarray(mc["logp"], np.float64))
            e2, h2, _ = loss_measures(pcm / pcm.sum(), _aligned(mc["moves"], sc["moves"], sc["value"]), eps)
            els.append(e2)
            holds.append(h2)
            ps.append(pc)
        ps_a = np.asarray(ps)
        el_next = float(np.sum(ps_a * els) / ps_a.sum()) if ps else float("nan")
        hold_next = float(np.sum(ps_a * holds) / ps_a.sum()) if ps else float("nan")
        mw, md, _ml = m["wdl"]
        rw, rd, _rl = rt["wdl"]
        cp = rt["cp"] if rt["mate"] == 0 else float(np.sign(rt["mate"]) * 1000)
        ent = float(-np.sum(p * np.log(np.maximum(p, 1e-12))))
        row = {"pos_hash": h, "el_stm": el, "hold_stm": hold, "el_next": el_next, "hold_next": hold_next,
               "net2": el - el_next, "child_cover": float(ps_a.sum()) if ps else 0.0, "n_children": len(ps),
               "maia_es_stm": mw + 0.5 * md, "engine_es_stm": rw + 0.5 * rd, "engine_cp_stm": cp,
               "engine_depth": rt["depth"], "multipv_best_es_stm": float(np.nanmax(v)),
               "maia_top_p": float(p.max()), "maia_entropy": ent, "n_legal": len(p),
               "maia_self_elo": se, "maia_oppo_elo": oe, "el_db": float("nan"), "hold_db": float("nan"),
               "db_share_valued": float("nan")}
        if h in dbm:
            mv_db, n_db = dbm[h]
            idx = {mm: i for i, mm in enumerate(m["moves"])}
            keep = [i for i, mm in enumerate(mv_db) if mm in idx]
            counts["db_moves_not_legal"] += len(mv_db) - len(keep)
            if keep:
                share = n_db[keep] / n_db[keep].sum()
                vdb = v[[idx[mv_db[i]] for i in keep]]
                e3, h3, _ = loss_measures(share, vdb, eps, best=float(np.nanmax(v)))
                row.update(el_db=e3, hold_db=h3, db_share_valued=float(n_db[keep].sum() / n_db.sum()))
        out.append(row)
    return pl.DataFrame(out), counts
