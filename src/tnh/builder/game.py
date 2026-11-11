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
    - Pessimism has two parts, which model different failures and combine exactly:
      `lam_node`: at every opponent's node, a share `lam_node` of the probability
      moves to the engine's best move. This is model error that is independent from
      node to node; it insures each node separately, so difficulty spread over many
      nodes survives it better than a single trap (the plan's own argument).
      `lam`: a share `lam` of opponents play the engine's best move everywhere (a
      prepared or simply stronger opponent); the rest play the per-node-mixed
      distribution. Against such opponents a long only-move line is worth no more
      than a trap that he also refutes. On a tree this is per-node mixing with a
      node's own share, the posterior probability that the opponent is the engine
      type given his moves on the path there: it starts at `lam`, grows each time he
      plays the best move (`b' = b / (b + (1 - b) q_best)`) and is 0 once he has played
      any other move. My moves scale both types alike, so the posterior does not
      depend on what I learn, and the DP stays exact.
    - `shrink`, `shrink_to`: a frontier value v is replaced by
      `(1 - shrink) * v + shrink * ref`, with ref the parent's value-head value
      (`"parent"`) or the engine's value of the move into the node (`"engine"`).
    - `banned`: (position key, move) pairs that may not be learned, e.g. moves that
      failed a deeper soundness check.
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
    lam_node: float = 0.0
    shrink: float = 0.0
    shrink_to: str = "parent"
    banned: frozenset = frozenset()

    def __post_init__(self) -> None:
        if self.shrink_to not in ("parent", "engine"):
            raise ValueError(f"shrink_to must be 'parent' or 'engine', got {self.shrink_to!r}")
        if not all(0 <= x <= 1 for x in (self.lam, self.lam_node, self.shrink)):
            raise ValueError("lam, lam_node and shrink must lie in [0, 1]")

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


def model_type_dist(node: Node, params: Params) -> np.ndarray:
    """The model-type opponent: counts over the model, with the per-node share `lam_node`
    on the engine's best move."""
    q = blend(node.counts, node.p_model, params.kappa_at(node))
    if params.lam_node > 0:
        q = (1.0 - params.lam_node) * q
        q[best_index(node.engine, node)] += params.lam_node
    return q


def opponent_dist(node: Node, params: Params, beta: float = 0.0) -> np.ndarray:
    """His distribution at a node where a share `beta` of the opponents who reach it are
    the engine type."""
    q = model_type_dist(node, params)
    if beta > 0:
        q = (1.0 - beta) * q
        q[best_index(node.engine, node)] += beta
    return q


def posterior_step(b: float, q_best: float) -> float:
    """The posterior share of the engine type after the opponent plays the best move."""
    return b / (b + (1.0 - b) * q_best) if b > 0 else 0.0


def sound(node: Node, eps: float, banned: frozenset = frozenset()) -> np.ndarray:
    """Indices of the moves within `eps` of the engine's best, less banned ones. Empty when unknown."""
    if node.engine is None or np.all(np.isnan(node.engine)):
        return np.zeros(0, dtype=np.int64)
    best = np.nanmax(node.engine)
    with np.errstate(invalid="ignore"):
        idx = np.flatnonzero(node.engine >= best - eps - 1e-12)
    if banned:
        idx = np.array([j for j in idx if (node.key, node.moves[j]) not in banned], dtype=np.int64)
    return idx


def engine_posterior(graph: Graph, params: Params, order: list[int]) -> dict[int, float]:
    """Per node, the share of the engine type among opponents who reach it. On a tree, exact and independent of my moves. On a DAG the paths into a
    node are combined with my moves counted as 1 (no policy of mine), an approximation."""
    if params.lam == 0:
        return {}
    wA = np.zeros(len(graph.nodes))
    wB = np.zeros(len(graph.nodes))
    wA[graph.root], wB[graph.root] = 1.0 - params.lam, params.lam
    out: dict[int, float] = {}
    for i in order:
        n = graph.nodes[i]
        tot = wA[i] + wB[i]
        out[i] = wB[i] / tot if tot > 0 else 0.0
        if not n.expanded:
            continue
        if n.me_to_move:
            wA[n.children] += wA[i]
            wB[n.children] += wB[i]
        else:
            q = model_type_dist(n, params)
            wA[n.children] += wA[i] * q
            if wB[i] > 0:
                wB[n.children[best_index(n.engine, n)]] += wB[i]
    return out


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
        self.parent: dict[int, tuple[int, int]] = {}      # first parent and the move index from it
        self.beta = engine_posterior(graph, params, self.order)
        self.stats: dict[str, int] = {}
        for i in self.order:
            n = graph.nodes[i]
            self.by_key.setdefault(n.key, []).append(i)
            for j, c in enumerate(n.children or ()):
                self.parent.setdefault(c, (i, j))
            if not n.expanded:
                if math.isnan(n.value):
                    raise ValueError(f"frontier node {n.key} (ply {n.ply}) has no value")
                continue
            d = n.p_model if n.me_to_move else opponent_dist(n, params, self.beta.get(i, 0.0))
            d = np.asarray(d, dtype=np.float64)
            if d.shape != (len(n.moves),) or abs(d.sum() - 1) > 1e-6 or (d < 0).any():
                raise ValueError(f"bad distribution at {n.key} (ply {n.ply}): {d}")
            self.dist[i] = d
            if n.me_to_move:
                self.sound[i] = sound(n, eps, params.banned)
        self.leaf = {i: self._leaf_value(i) for i in self.order if not graph.nodes[i].expanded}

    def _leaf_value(self, i: int) -> float:
        n = self.graph.nodes[i]
        s = self.params.shrink
        if s == 0 or i not in self.parent or not n.moves:
            return n.value
        p, j = self.parent[i]
        pn = self.graph.nodes[p]
        ref = math.nan
        if self.params.shrink_to == "engine" and pn.engine is not None and not math.isnan(pn.engine[j]):
            ref = float(pn.engine[j]) if pn.me_to_move else 1.0 - float(pn.engine[j])
        elif self.params.shrink_to == "engine":
            self.stats["shrink: no engine value into the node, parent's value used"] = \
                self.stats.get("shrink: no engine value into the node, parent's value used", 0) + 1
        if math.isnan(ref):
            ref = pn.value
        if math.isnan(ref):
            raise ValueError(f"no reference value to shrink {n.key} (ply {n.ply}) toward")
        return (1 - s) * n.value + s * ref

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
                V[i] = self.leaf[i]
            else:
                V[i] = float(self.policy(i, learned) @ V[n.children])
        return V

    def evaluate(self, learned: Learned = {}) -> float:
        """The exact expected score. With `lam > 0` this is the mixture of the two opponent
        types (computed from both reaches); on a DAG `values()` uses per-node posteriors
        that are then an approximation. On a tree the two agree."""
        self.check(learned)
        if self.params.lam > 0:
            return self.mixture(learned)
        return float(self.values(learned)[self.graph.root])

    def mixture(self, learned: Learned = {}) -> float:
        g = self.graph
        lam = self.params.lam
        RA = np.zeros(len(g.nodes))
        RB = np.zeros(len(g.nodes))
        RA[g.root] = RB[g.root] = 1.0
        for i in self.order:
            n = g.nodes[i]
            if not n.expanded:
                continue
            if n.me_to_move:
                pol = self.policy(i, learned)
                RA[n.children] += RA[i] * pol
                RB[n.children] += RB[i] * pol
            else:
                RA[n.children] += RA[i] * model_type_dist(n, self.params)
                if RB[i] > 0:
                    RB[n.children[best_index(n.engine, n)]] += RB[i]
        leaves = list(self.leaf)
        v = np.array([self.leaf[i] for i in leaves])
        return float(((1 - lam) * RA[leaves] + lam * RB[leaves]) @ v)

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
    """Per-opponent pessimism computed the slow, obvious way, as a check: the value
    against the model-type opponent and against the engine opponent, mixed by `lam`.

    Per node and per opponent agree at a single critical node and part on long lines:
    per node, the chance that he finds n only-moves in a row is
    prod((1 - lam) p_i + lam), which tends to 0; per opponent it is at least lam.
    """
    from dataclasses import replace
    model = Game(graph, replace(params, lam=0.0)).evaluate(learned)
    engine = Game(graph, replace(params, lam=0.0, lam_node=1.0)).evaluate(learned)
    return (1 - params.lam) * model + params.lam * engine
