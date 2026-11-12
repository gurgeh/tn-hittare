"""A synthetic world for the search: a builder graph (planted or random) revealed one
position at a time, with every evaluation counted."""

from __future__ import annotations

from collections import Counter

import numpy as np

from tnh.builder.game import Game, Graph, Params
from tnh.engine.search import Eval


class SynthEvaluator:
    """Positions are node ids of `full`, a builder graph whose root is White to move with me
    as White (`synth.build`'s convention). A position's own value is `info["static"]` (my
    view) when given, else its true value with nothing learned: a value head that is right
    on average. Finished games are the graph's leaves."""

    def __init__(self, full: Graph, *, kappa: float = float("inf")) -> None:
        self.full = full
        self.static = Game(full, Params(kappa=kappa, eps_white=1.0)).values({})
        self.calls: Counter = Counter()
        self.count_calls: Counter = Counter()

    def start(self):
        return self.full.root

    def key(self, pos):
        return self.full.nodes[pos].key

    def ply(self, pos) -> int:
        return self.full.nodes[pos].ply

    def white_to_move(self, pos) -> bool:
        return self.full.nodes[pos].me_to_move

    def moves(self, pos) -> list[str]:
        return list(self.full.nodes[pos].moves)

    def child(self, pos, move: str):
        n = self.full.nodes[pos]
        return n.children[n.moves.index(move)]

    def terminal(self, pos):
        n = self.full.nodes[pos]
        if n.moves:
            return None
        return n.value if n.me_to_move else 1.0 - n.value

    def evaluate(self, items, tempo):
        out = []
        for pos, mover_elo, opp_elo in items:
            self.calls[(self.key(pos), mover_elo, opp_elo, tempo)] += 1
            n = self.full.nodes[pos]
            mine = n.info.get("static", float(self.static[pos]))
            wdl = n.info.get("wdl")                   # the mover's (W, D, L) when a world plants one
            out.append(Eval(list(n.moves), np.asarray(n.p_model, dtype=np.float64),
                            None if n.info.get("policy") is None else np.asarray(n.info["policy"], dtype=np.float64),
                            mine if n.me_to_move else 1.0 - mine, None if wdl is None else tuple(wdl)))
        return out

    def counts(self, pos, moves, mover_elo, db_filter):
        self.count_calls[(self.key(pos), mover_elo, db_filter)] += 1
        c = self.full.nodes[pos].counts
        return None if c is None else np.asarray(c, dtype=np.float64)
