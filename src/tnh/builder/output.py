"""What a solution looks like to a person: a table, a PGN and the curve."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass

import chess
import chess.pgn
import numpy as np

from tnh.builder.game import Game, Learned
from tnh.builder.search import Solution, drop_one_gains


@dataclass
class Row:
    key: str
    path: list[str]          # most likely path to it under the repertoire, from the root
    move: str
    reach: float             # probability of reaching the position under the repertoire
    gain: float              # expected score lost if this one position were forgotten
    p_find: float            # how often my model plays the move unaided
    engine: float            # engine's expected score for me after the move (opponent plays best)
    engine_best: float       # engine's expected score after its own best move


def best_paths(game: Game, learned: Learned) -> dict[int, list[str]]:
    """For every node, the most likely path from the root under the repertoire."""
    g = game.graph
    best = {g.root: (1.0, [])}
    for i in game.order:
        if i not in best or not g.nodes[i].expanded:
            continue
        p0, path = best[i]
        n = g.nodes[i]
        pol = game.policy(i, learned)
        for m, c, x in zip(n.moves, n.children, pol):
            if x > 0 and (c not in best or p0 * x > best[c][0]):
                best[c] = (p0 * x, path + [m])
    return {i: p for i, (_, p) in best.items()}


def rows(game: Game, sol: Solution) -> list[Row]:
    g = game.graph
    R = game.reach(sol.learned)
    paths = best_paths(game, sol.learned)
    gains = drop_one_gains(game, sol.learned)
    out = []
    for k, m in sol.learned.items():
        ids = game.by_key[k]
        i = max(ids, key=lambda j: R[j])
        n = g.nodes[i]
        j = n.moves.index(m)
        path = paths.get(i, list(n.path))
        out.append(Row(str(k), path, m, float(sum(R[x] for x in ids)), gains[k], float(game.dist[i][j]),
                       float(n.engine[j]), float(np.nanmax(n.engine))))
    return sorted(out, key=lambda r: (len(r.path), -r.reach))


def _san_line(board: chess.Board | None, moves: list[str]) -> list[str]:
    if board is None:
        return list(moves)
    b = board.copy(stack=False)
    out = []
    for m in moves:
        mv = chess.Move.from_uci(m)
        out.append(b.san(mv))
        b.push(mv)
    return out


def table(rs: list[Row], root_board: chess.Board | None = None) -> str:
    lines = ["| # | line | learn | reach | gain | I find it | engine | engine best |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for x, r in enumerate(rs, 1):
        line = _san_line(root_board, r.path + [r.move])
        lines.append(f"| {x} | {' '.join(line[:-1]) or '(root)'} | **{line[-1]}** | {r.reach:.3f} | "
                     f"{r.gain:+.4f} | {r.p_find:.0%} | {r.engine:.3f} | {r.engine_best:.3f} |")
    return "\n".join(lines)


def pgn(rs: list[Row], root_moves: list[str], headers: dict[str, str] | None = None) -> str:
    """The repertoire as one PGN: the root line, then a variation per learned position
    (its most likely path and the learned move), with comments on the learned moves."""
    game = chess.pgn.Game()
    for k, v in (headers or {}).items():
        game.headers[k] = v
    node = game
    for m in root_moves:
        node = node.add_variation(chess.Move.from_uci(m))
    root = node
    for r in rs:
        node = root
        for m in r.path + [r.move]:
            mv = chess.Move.from_uci(m)
            node = node.variation(mv) if node.has_variation(mv) else node.add_variation(mv)
        c = (f"learn: reach {r.reach:.3f}, gain {r.gain:+.4f}, I find it {r.p_find:.0%}, "
             f"engine {r.engine:.3f} (best {r.engine_best:.3f})")
        node.comment = (node.comment + " " + c).strip()
    out = io.StringIO()
    print(game, file=out, end="\n\n")
    return out.getvalue()


def curve_json(sols: list[Solution]) -> str:
    return json.dumps([{"N": s.N, "value": s.value, "dp_value": s.dp_value, "cost": s.cost,
                        "learned": {str(k): m for k, m in s.learned.items()}} for s in sols], indent=1)


def curve_text(sols: list[Solution]) -> str:
    base = sols[0].value
    lines = ["  N   expected score   gain over N=0   DP's own value   positions used"]
    for s in sols:
        lines.append(f"{s.N:3d}   {s.value:.5f}          {s.value - base:+.5f}        {s.dp_value:.5f}          {s.cost}")
    return "\n".join(lines)

