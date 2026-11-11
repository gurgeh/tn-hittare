"""A final soundness check of the learned moves at a much larger engine budget.

The search judges soundness at the budget it can afford per position (a few
thousand nodes per move). The learned moves are few, so afterwards each is searched
again with `nodes` nodes, as is the best alternative to it, and the move passes if
it is within `eps` of that alternative. A move that fails is banned
(`Params.banned`) and the search is rerun without it; the new choice is checked
in turn, until every learned move passes or `max_rounds` is reached.

One engine process, one thread. Each check is two searches (the move alone, and
every other move), so ten positions at a million nodes is twenty million nodes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import chess
import chess.engine

from tnh.builder.game import Game, Graph, Key, Params
from tnh.builder.search import Solution, solve
from tnh.models.stockfish import STOCKFISH


class DeepChecker:
    """Stockfish through python-chess, at a fixed node budget per search."""

    def __init__(self, nodes: int, *, path: Path = STOCKFISH, hash_mb: int = 64) -> None:
        self.nodes = nodes
        self.engine = chess.engine.SimpleEngine.popen_uci(str(path))
        self.engine.configure({"Threads": 1, "Hash": hash_mb, "UCI_ShowWDL": True})
        self.searches = 0

    def _search(self, board: chess.Board, moves: list[chess.Move]) -> tuple[str, float, int]:
        # a new game object per search makes python-chess send ucinewgame: no hash carried over
        info = self.engine.analyse(board, chess.engine.Limit(nodes=self.nodes), root_moves=moves, game=object())
        self.searches += 1
        if "wdl" not in info or "pv" not in info:
            raise RuntimeError(f"engine gave no WDL or PV at {board.fen()}: {info}")
        return info["pv"][0].uci(), info["wdl"].pov(board.turn).expectation(), int(info.get("depth", 0))

    def check(self, moves_from_start: list[str], move: str, eps: float) -> dict:
        board = chess.Board()
        for m in moves_from_start:
            board.push_uci(m)
        mv = chess.Move.from_uci(move)
        others = [m for m in board.legal_moves if m != mv]
        _, v_move, d_move = self._search(board, [mv])
        if others:
            alt, v_alt, d_alt = self._search(board, others)
        else:
            alt, v_alt, d_alt = "", v_move, d_move
        return {"move": move, "move_value": v_move, "alt_move": alt, "alt_value": v_alt,
                "passes": v_move >= v_alt - eps - 1e-12, "depth": (d_move, d_alt), "nodes": self.nodes}

    def close(self) -> None:
        self.engine.quit()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@dataclass
class Checked:
    sols: list[Solution]
    params: Params                     # with the failed moves banned
    results: dict[tuple[Key, str], dict]
    log: list[str] = field(default_factory=list)
    settled: bool = True               # every learned move at N has passed


def check_repertoire(graph: Graph, params: Params, K: int, N: int,
                     check: Callable[[Key, str], dict], *, max_rounds: int = 5) -> Checked:
    """Solve, check the learned moves at N, ban the failures, solve again, until all pass.
    `check(key, move)` returns a dict with at least `passes`."""
    results: dict[tuple[Key, str], dict] = {}
    banned = set(params.banned)
    log: list[str] = []
    for r in range(max_rounds):
        p = replace(params, banned=frozenset(banned))
        _, sols = solve(Game(graph, p), K)
        S = sols[N].learned
        failed = []
        for k, m in S.items():
            if (k, m) not in results:
                results[(k, m)] = check(k, m)
            if not results[(k, m)]["passes"]:
                failed.append((k, m))
        if not failed:
            log.append(f"round {r}: all {len(S)} learned moves pass")
            return Checked(sols, p, results, log, True)
        for k, m in failed:
            d = results[(k, m)]
            log.append(f"round {r}: {m} at {k} fails: {d['move_value']:.3f} against {d['alt_move']} "
                       f"{d['alt_value']:.3f}; banned, search rerun")
            banned.add((k, m))
    p = replace(params, banned=frozenset(banned))
    _, sols = solve(Game(graph, p), K)
    log.append(f"stopped after {max_rounds} rounds; the learned moves at N={N} are not all checked")
    return Checked(sols, p, results, log, False)
