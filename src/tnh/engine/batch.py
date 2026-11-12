"""Batch mode: the training data for the fitted move-probability update.

Given positions (moves from the start, the mover's and the opponent's rating), each is
searched from scratch with the root's side to move as the opponent (his moves are the
ones whose probabilities the update repairs, and the descent at his node follows his
distribution), for a node budget. One row per root move:

| column | meaning |
|---|---|
| `pos` | index of the position in the input |
| `moves` | the moves from the start, space-separated UCI |
| `mover_elo`, `opp_elo` | the root mover's rating and the other side's |
| `move` | the root move, UCI |
| `p_first` | the move model's probability, alone |
| `p_mixed` | the probability the search used: database counts over the model |
| `value_mover` | the searched value of the move for the mover (expected score) |
| `status` | `expanded`, `stand_in` (no value of its own yet), `frontier` or `finished` |
| `visits` | selections through the move |
| `expanded_below` | expanded positions reachable from the move |
| `root_before`, `root_after` | the root's value for the mover: its own, and searched |
| `nodes` | expansions made in this position's search |

With the move actually played, this is the data the update is fitted on.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from tnh.engine.search import Engine, Settings


def expanded_below(eng: Engine, start: int) -> int:
    g = eng.graph
    seen, stack, n = {start}, [start], 0
    while stack:
        i = stack.pop()
        node = g.nodes[i]
        if node.children is not None and node.moves:
            n += 1
            for c in node.children:
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
    return n


def run_batch(evaluator, positions: list[dict], nodes: int, out: Path, settings: Settings | None = None) -> pl.DataFrame:
    """`positions`: dicts with `moves` (list of UCI), `mover_elo`, `opp_elo`."""
    rows = []
    base = settings or Settings()
    for x, p in enumerate(positions):
        eng = Engine(evaluator, base)
        eng.set_position(list(p["moves"]))
        white = evaluator.white_to_move(eng.pos[eng.graph.root])
        # the root mover is the opponent: I am the other side, with the root mover's ratings swapped in
        eng.set_settings(my_side="black" if white else "white", my_elo=p["opp_elo"], opp_elo=p["mover_elo"])
        eng.search(nodes=1)
        root = eng.graph.nodes[eng.graph.root]
        before = 1.0 - root.value
        r = eng.search(nodes=max(nodes - 1, 0))
        game = eng.game
        q = game.dist[root.id]
        after = 1.0 - eng.V[root.id]
        for j, (m, c) in enumerate(zip(root.moves, root.children)):
            cn = eng.graph.nodes[c]
            status = ("finished" if not cn.moves else "expanded" if cn.children is not None
                      else "stand_in" if cn.info.get("stand_in") else "frontier")
            rows.append({"pos": x, "moves": " ".join(p["moves"]), "mover_elo": p["mover_elo"],
                         "opp_elo": p["opp_elo"], "move": m, "p_first": float(root.info["p_first"][j]),
                         "p_mixed": float(q[j]), "value_mover": 1.0 - float(eng.V.get(c, cn.value)),
                         "status": status, "visits": int(eng.visits[c]), "expanded_below": expanded_below(eng, c),
                         "root_before": before, "root_after": after, "nodes": r.expansions + 1})
    df = pl.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    return df


