"""A planted world for step 2: results generated from a known rule, through the real pipeline.

An opening tree of real chess moves, with branching ``branching[d]`` at depth
``d`` and Dirichlet move probabilities. Each game walks from the root to a leaf
and stops there. Every leaf L gets an engine value ``e_L`` and a measure
``m_L``; an inner node gets the probability-weighted mean of its children, so
the rule holds at every depth and nested positions are consistent:

    E[White score of a game ending in L] = a + b * (e_L - 0.5) + c * elo_diff/100 + beta * m_L + u_L

``u_L`` is a leaf effect nobody sees (the between-position variance). A game's
result is drawn with draw probability ``draw`` and the rest split to give that
expected score.

The games go through `tnh.data.edges.tables_from_games`, the position set is
built by `tnh.difficulty.positions.build`, and the test by
`tnh.difficulty.analysis.run`, exactly as on real data. ``noise_measure`` is
built like ``m`` and has no effect: the pipeline must find zero for it.
"""

from __future__ import annotations

import chess
import numpy as np
import polars as pl

from tnh.data.edges import EdgeDB, pos_hash, tables_from_games
from tnh.difficulty import positions as P


def world(*, seed: int = 0, branching: tuple[int, ...] = (8, 6, 3, 2, 2), games: int = 30_000,
          beta: float = 1.0, b: float = 0.5, c: float = 0.014, a: float = 0.53, tau: float = 0.03,
          draw: float = 0.4, m_sd: float = 0.04, e_sd: float = 0.05) -> tuple[pl.DataFrame, dict[int, dict]]:
    """(games frame, planted values per position hash)."""
    rng = np.random.default_rng(seed)
    # the tree: node -> (board, moves, children, probs)
    nodes = [{"board": chess.Board(), "moves": [], "kids": [], "p": None}]
    frontier = [0]
    for k in branching:
        nxt = []
        for i in frontier:
            bd = nodes[i]["board"]
            legal = list(bd.legal_moves)
            pick = rng.choice(len(legal), size=min(k, len(legal)), replace=False)
            for j in pick:
                nb = bd.copy(stack=False)
                nb.push(legal[j])
                nodes.append({"board": nb, "moves": nodes[i]["moves"] + [legal[j].uci()], "kids": [], "p": None})
                nodes[i]["kids"].append(len(nodes) - 1)
                nxt.append(len(nodes) - 1)
            nodes[i]["p"] = rng.dirichlet(np.full(len(nodes[i]["kids"]), 2.0))
        frontier = nxt
    leaves = frontier
    for i in leaves:
        nodes[i]["e"] = 0.5 + rng.normal(0, e_sd)
        nodes[i]["m"] = rng.normal(0, m_sd)
        nodes[i]["z"] = rng.normal(0, m_sd)
        nodes[i]["u"] = rng.normal(0, tau)

    def fill(i: int) -> tuple[float, float, float]:
        nd = nodes[i]
        if not nd["kids"]:
            return nd["e"], nd["m"], nd["z"]
        vals = np.array([fill(j) for j in nd["kids"]])
        nd["e"], nd["m"], nd["z"] = (float(x) for x in nd["p"] @ vals)
        return nd["e"], nd["m"], nd["z"]

    fill(0)
    rows = {"moves_uci": [], "result": [], "year": [], "white_elo": [], "black_elo": [], "tempo": []}
    for _ in range(games):
        i = 0
        while nodes[i]["kids"]:
            i = nodes[i]["kids"][rng.choice(len(nodes[i]["kids"]), p=nodes[i]["p"])]
        we, be = (int(np.clip(rng.normal(2100, 150), 1800, 2499)) for _ in range(2))
        nd = nodes[i]
        s = a + b * (nd["e"] - 0.5) + c * (we - be) / 100 + beta * nd["m"] + nd["u"]
        s = float(np.clip(s, draw / 2 + 0.01, 1 - draw / 2 - 0.01))
        pw = s - draw / 2
        r = rng.choice(3, p=[pw, draw, 1 - pw - draw])
        rows["moves_uci"].append(" ".join(nd["moves"]))
        rows["result"].append(("1-0", "1/2-1/2", "0-1")[r])
        rows["year"].append(2024)
        rows["white_elo"].append(we)
        rows["black_elo"].append(be)
        rows["tempo"].append("classical")
    planted = {pos_hash(nd["board"]): {"e": nd["e"], "m": nd["m"], "z": nd["z"]} for nd in nodes}
    return pl.DataFrame(rows), planted


def pipeline(games: pl.DataFrame, planted: dict[int, dict], *, min_games: int, family_ply: int,
             leaves_only: bool = False, boot: int = 400, seed: int = 0) -> tuple[dict, pl.DataFrame, dict]:
    """The position set and the test on a synthetic world; the measure is ``planted['m']``."""
    from tnh.difficulty import analysis as A
    e, p = tables_from_games(games)
    db = EdgeDB.from_frames(e, p)
    s, _out, info = P.build(db, e.lazy(), min_games=min_games)
    pv = pl.DataFrame({"pos_hash": pl.Series(list(planted), dtype=pl.UInt64),
                       "engine_es_w": [v["e"] for v in planted.values()],
                       "planted": [v["m"] for v in planted.values()],
                       "noise_measure": [v["z"] for v in planted.values()]})
    df = s.join(pv, on="pos_hash", how="inner").with_columns(
        y=pl.col("score_white"),
        elo_diff=(pl.col("white_elo") - pl.col("black_elo")) / 100,
        elo_mean=((pl.col("white_elo") + pl.col("black_elo")) / 2 - 2100) / 100,
        stm_white=pl.col("stm_white").cast(pl.Float64),
        family=P.family(pl.col("moves_uci"), family_ply),
    ).filter(pl.col("family").is_not_null())
    if leaves_only:
        df = df.filter(pl.col("leaf"))
    setup = A.Setup(boot=boot, seed=seed, base=["engine_es_w", "elo_diff", "elo_mean", "stm_white"])
    res, out = A.run(df, ["planted", "noise_measure"], setup)
    return res, out, info
