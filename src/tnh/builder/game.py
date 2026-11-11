"""The game the repertoire builder optimises, and its exact value.

The game. In a position I have learned I play the memorised move. Everywhere else
I play like a model of myself. The opponent always plays his distribution: the
database counts laid over a model of him, optionally mixed with a share `lam`
that plays the engine's best move. Where the expanded graph ends, a value
function gives my expected score.

The graph. A node is a (position, ply) pair. Positions reached by transposition
at the same ply are one node, so the tree becomes a DAG; keying on the ply as
well keeps it acyclic (a repetition is a new node, one ply deeper). What is
learned is a *position*: a learned set maps a position key to a move, and that
move is played at every node with that key, whatever the ply or the path. Model
outputs are a function of the position only (cached on the key), so two nodes
with the same key have the same moves and the same distributions.

Everything here is plain expectation over the DAG: `evaluate` is the definition
that the search in `search.py` is checked against.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field

import numpy as np

Key = Hashable
Learned = Mapping[Key, str]          # position key -> the memorised move


@dataclass(eq=False)
class Node:
    """One (position, ply). Arrays are aligned with `moves`.

    - `children is None`: a frontier node, not expanded; its value is `value`.
    - `moves == []`: the game is over; its value is `value`.
    - `p_model`: the mover's model, at the mover's rating.
    - `counts`: database games per move (opponent's nodes), or None.
    - `engine`: the engine's expected score for the mover after each move, NaN where
      it gave none; None where the engine was not asked (soundness then unknown,
      so the position cannot be learned).
    - `value`: my expected score here from the value function (frontier and end).
    """

    id: int
    key: Key
    ply: int
    me_to_move: bool
    moves: list[str]
    p_model: np.ndarray | None = None
    counts: np.ndarray | None = None
    engine: np.ndarray | None = None
    value: float = math.nan
    children: list[int] | None = None
    path: tuple[str, ...] = ()       # the first path that created the node, from the root
    info: dict = field(default_factory=dict)

    @property
    def expanded(self) -> bool:
        return self.children is not None and len(self.moves) > 0


class Graph:
    """Nodes in creation order, one per (key, ply). Edges go from ply t to t + 1."""

    def __init__(self, me_white: bool) -> None:
        self.me_white = me_white
        self.nodes: list[Node] = []
        self.index: dict[tuple[Key, int], int] = {}
        self.root = 0

    def __len__(self) -> int:
        return len(self.nodes)

    def get(self, key: Key, ply: int) -> Node | None:
        i = self.index.get((key, ply))
        return None if i is None else self.nodes[i]

    def add(self, key: Key, ply: int, me_to_move: bool, moves: list[str], **kw) -> Node:
        if (key, ply) in self.index:
            raise ValueError(f"node {(key, ply)} exists")
        node = Node(len(self.nodes), key, ply, me_to_move, list(moves), **kw)
        self.nodes.append(node)
        self.index[(key, ply)] = node.id
        return node

    def link(self, parent: Node, children: list[Node]) -> None:
        if len(children) != len(parent.moves):
            raise ValueError("one child per move")
        for c in children:
            if c.ply != parent.ply + 1 or c.me_to_move == parent.me_to_move:
                raise ValueError(f"child {c.key} at ply {c.ply} cannot follow {parent.key} at ply {parent.ply}")
        parent.children = [c.id for c in children]

    def order(self) -> list[int]:
        """Node ids by ply, root first: a topological order."""
        return sorted(range(len(self.nodes)), key=lambda i: self.nodes[i].ply)

    def reachable(self) -> list[int]:
        """Ids reachable from the root, in topological order."""
        seen = {self.root}
        stack = [self.root]
        while stack:
            n = self.nodes[stack.pop()]
            for c in n.children or ():
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return sorted(seen, key=lambda i: self.nodes[i].ply)

    def child(self, node: Node, move: str) -> Node:
        return self.nodes[node.children[node.moves.index(move)]]


# -- the pieces, each a parameter --------------------------------------------------

def uniform_cost(node: Node) -> int:
    return 1


@dataclass(frozen=True)
class Params:
    """Everything about the game that is not a model output.

    - `kappa`: the model's weight in the opponent blend, in games. `inf` = model
      only. A callable gets the node (and through `node.info` anything else, such
      as ratings) for a rating-dependent kappa later.
    - `lam`: the share of the opponent that plays the engine's best move, per node.
    - `eps_white`, `eps_black`: how far below the engine's best (expected score) a
      move may be and still be learned, for the colour I play.
    - `min_reach`: no budget is spent in nodes whose reach bound is below it.
    - `cost`: the budget units one learned position costs, an int >= 1.
    """

    kappa: float | Callable[[Node], float] = math.inf
    lam: float = 0.0
    eps_white: float = 0.03
    eps_black: float = 0.03
    min_reach: float = 0.0
    cost: Callable[[Node], int] = uniform_cost

    def eps(self, me_white: bool) -> float:
        return self.eps_white if me_white else self.eps_black

    def kappa_at(self, node: Node) -> float:
        return self.kappa(node) if callable(self.kappa) else self.kappa


def blend(counts: np.ndarray | None, p_model: np.ndarray, kappa: float) -> np.ndarray:
    """`(count + kappa * p_model) / (n + kappa)` over the legal moves."""
    if counts is None or math.isinf(kappa):
        return p_model
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.sum()
    if n == 0:
        return p_model
    return (counts + kappa * p_model) / (n + kappa)


def best_index(engine: np.ndarray | None, node: Node) -> int:
    if engine is None or np.all(np.isnan(engine)):
        raise ValueError(f"no engine values at {node.key} (ply {node.ply}): needed for pessimism")
    return int(np.nanargmax(engine))


def opponent_dist(node: Node, params: Params) -> np.ndarray:
    q = blend(node.counts, node.p_model, params.kappa_at(node))
    if params.lam > 0:
        q = (1.0 - params.lam) * q
        q[best_index(node.engine, node)] += params.lam
    return q


def sound(node: Node, eps: float) -> np.ndarray:
    """Indices of the moves within `eps` of the engine's best. Empty when unknown."""
    if node.engine is None or np.all(np.isnan(node.engine)):
        return np.zeros(0, dtype=np.int64)
    best = np.nanmax(node.engine)
    with np.errstate(invalid="ignore"):
        return np.flatnonzero(node.engine >= best - eps - 1e-12)


def dist(node: Node, params: Params) -> np.ndarray:
    """The mover's distribution with nothing learned."""
    return node.p_model if node.me_to_move else opponent_dist(node, params)


class Game:
    """A graph with its parameters: the distributions, soundness and costs, computed once."""

    def __init__(self, graph: Graph, params: Params) -> None:
        self.graph = graph
        self.params = params
        self.order = graph.reachable()
        eps = params.eps(graph.me_white)
        self.dist: dict[int, np.ndarray] = {}
        self.sound: dict[int, np.ndarray] = {}
        self.by_key: dict[Key, list[int]] = {}
        for i in self.order:
            n = graph.nodes[i]
            self.by_key.setdefault(n.key, []).append(i)
            if not n.expanded:
                if math.isnan(n.value):
                    raise ValueError(f"frontier node {n.key} (ply {n.ply}) has no value")
                continue
            d = np.asarray(dist(n, params), dtype=np.float64)
            if d.shape != (len(n.moves),) or abs(d.sum() - 1) > 1e-6 or (d < 0).any():
                raise ValueError(f"bad distribution at {n.key} (ply {n.ply}): {d}")
            self.dist[i] = d
            if n.me_to_move:
                self.sound[i] = sound(n, eps)

    def learnable(self, i: int) -> bool:
        return i in self.sound and len(self.sound[i]) > 0

    def is_sound(self, key: Key, move: str) -> bool:
        for i in self.by_key.get(key, ()):
            if i in self.sound:
                n = self.graph.nodes[i]
                return move in n.moves and n.moves.index(move) in set(self.sound[i].tolist())
        return False

    def cost(self, i: int) -> int:
        c = self.params.cost(self.graph.nodes[i])
        if not (isinstance(c, (int, np.integer)) and c >= 1):
            raise ValueError(f"cost must be an int >= 1, got {c!r}")
        return int(c)

    def key_cost(self, key: Key) -> int:
        return self.cost(self.by_key[key][0])

    def policy(self, i: int, learned: Learned) -> np.ndarray:
        n = self.graph.nodes[i]
        mv = learned.get(n.key) if n.me_to_move else None
        if mv is None:
            return self.dist[i]
        p = np.zeros(len(n.moves))
        p[n.moves.index(mv)] = 1.0
        return p

    # -- exact value ---------------------------------------------------------------

    def values(self, learned: Learned = {}) -> np.ndarray:
        """My expected score at every node, given the learned set. NaN off the reachable part."""
        g = self.graph
        V = np.full(len(g.nodes), np.nan)
        for i in reversed(self.order):
            n = g.nodes[i]
            if not n.expanded:
                V[i] = n.value
            else:
                V[i] = float(self.policy(i, learned) @ V[n.children])
        return V

    def evaluate(self, learned: Learned = {}) -> float:
        self.check(learned)
        return float(self.values(learned)[self.graph.root])

    def reach(self, learned: Learned = {}) -> np.ndarray:
        """Probability of reaching each node in the game, given the learned set."""
        g = self.graph
        R = np.zeros(len(g.nodes))
        R[g.root] = 1.0
        for i in self.order:
            n = g.nodes[i]
            if n.expanded and R[i] > 0:
                np.add.at(R, n.children, R[i] * self.policy(i, learned))
        return R

    def reach_bound(self) -> np.ndarray:
        """An upper bound on the reach of each node over all learned sets: my sound
        moves count as 1 (I could learn them), everything else as its probability.
        Max over parents, so a node's bound is that of its most likely path."""
        g = self.graph
        B = np.zeros(len(g.nodes))
        B[g.root] = 1.0
        for i in self.order:
            n = g.nodes[i]
            if not n.expanded or B[i] == 0:
                continue
            f = self.dist[i].copy()
            if n.me_to_move and self.learnable(i):
                f[self.sound[i]] = 1.0
            np.maximum.at(B, n.children, B[i] * f)
        return B

    def check(self, learned: Learned) -> None:
        for k, mv in learned.items():
            if k not in self.by_key:
                raise ValueError(f"learned position {k} is not in the graph")
            if not self.is_sound(k, mv):
                raise ValueError(f"learned move {mv} at {k} is not sound (or the position is not expanded)")

    def total_cost(self, learned: Learned) -> int:
        return sum(self.key_cost(k) for k in learned)

    # -- marginal changes -------------------------------------------------------------

    def marginals(self, learned: Learned) -> dict[tuple[Key, str | None], float]:
        """The change in my expected score from setting one position's move.

        `(key, move)`: learn `move` at `key` (or switch to it); `(key, None)`: forget
        `key`. Computed as reach x (value after the change - value now) summed over
        the key's nodes, which is exact when no node with the key lies below another
        with the same key (no repetitions). Callers verify with `evaluate`.
        """
        g = self.graph
        V, R = self.values(learned), self.reach(learned)
        out: dict[tuple[Key, str | None], float] = {}
        for i, s in self.sound.items():
            n = g.nodes[i]
            if R[i] == 0 or not len(s) and n.key not in learned:
                continue
            Vc = V[n.children]
            for j in s:
                k = (n.key, n.moves[j])
                out[k] = out.get(k, 0.0) + R[i] * (Vc[j] - V[i])
            if n.key in learned:
                k = (n.key, None)
                out[k] = out.get(k, 0.0) + R[i] * (float(self.dist[i] @ Vc) - V[i])
        return out


def mixture_value(graph: Graph, params: Params, learned: Learned = {}) -> float:
    """Pessimism per opponent rather than per move: with probability `lam` the opponent
    plays the engine's best move everywhere, else his distribution everywhere.

    `Params.lam` mixes per node, which is what the DP can optimise (the value stays a
    sum over subtrees). The two agree at a single critical node and part on long
    lines: per node, the chance that he finds n only-moves in a row is
    prod((1 - lam) p_i + lam), which tends to 0, so the long line keeps most of its
    promise; per opponent it is at least lam. This evaluates a fixed repertoire
    under the stricter reading.
    """
    from dataclasses import replace
    model = Game(graph, replace(params, lam=0.0)).evaluate(learned)
    engine = Game(graph, replace(params, lam=1.0)).evaluate(learned)
    return (1 - params.lam) * model + params.lam * engine
