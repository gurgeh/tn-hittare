"""Synthetic games: hand-written trees with a planted answer, and random trees and DAGs.

Values in a spec are always *my* expected score, at every node: an engine value at
the opponent's node is converted to his point of view when the node is built.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tnh.builder.game import Graph, Node


@dataclass
class Leaf:
    value: float            # my expected score
    key: str | None = None


@dataclass
class Pos:
    me: bool                                    # I am to move
    moves: list[tuple[str, float, float, object]]   # (move, model probability, engine value for me, child)
    counts: dict[str, int] | None = None
    key: str | None = None
    info: dict = field(default_factory=dict)


def leaf(value: float, key: str | None = None) -> Leaf:
    return Leaf(value, key)


def me(*moves, key: str | None = None, **info) -> Pos:
    return Pos(True, list(moves), None, key, info)


def opp(*moves, counts: dict[str, int] | None = None, key: str | None = None, **info) -> Pos:
    return Pos(False, list(moves), counts, key, info)


def build(spec: Pos, me_white: bool = True) -> Graph:
    """A graph from a nested spec. Nodes with the same `key` at the same ply are one
    node (the first spec seen for it is used), which makes a DAG."""
    g = Graph(me_white)

    def rec(s, ply: int, path: tuple[str, ...], me_to_move: bool) -> Node:
        key = s.key if s.key is not None else "/".join(path) or "root"
        hit = g.get(key, ply)
        if hit is not None:
            return hit
        if isinstance(s, Leaf):
            return g.add(key, ply, me_to_move, [], value=float(s.value), children=None, path=path)
        if s.me != me_to_move:
            raise ValueError(f"{key}: side to move does not alternate")
        mv = [m for m, *_ in s.moves]
        p = np.array([x[1] for x in s.moves], dtype=np.float64)
        e_me = np.array([x[2] for x in s.moves], dtype=np.float64)
        counts = None
        if s.counts is not None:
            counts = np.array([s.counts.get(m, 0) for m in mv], dtype=np.float64)
        node = g.add(key, ply, me_to_move, mv, p_model=p / p.sum(), counts=counts,
                     engine=e_me if me_to_move else 1.0 - e_me, path=path, info=dict(s.info))
        kids = [rec(x[3], ply + 1, path + (x[0],), not me_to_move) for x in s.moves]
        g.link(node, kids)
        return node

    rec(spec, 0, (), spec.me if isinstance(spec, Pos) else True)
    return g


def random_game(rng: np.random.Generator, *, depth: int = 4, branching: tuple[int, int] = (2, 3),
                me_first: bool = True, dag: bool = False, pool: int | None = None,
                alpha: float = 0.6, value_spread: float = 0.4) -> Graph:
    """A random game of `depth` plies.

    - Model distributions are Dirichlet(alpha).
    - Engine values for me: each move's value is the mean of the frontier values below
      it plus noise, so sound and unsound moves both occur for a fixed eps.
    - `dag`: positions at each ply come from a pool of `pool` keys, children are
      drawn from the next ply's pool, so paths meet (transpositions). A key's moves
      and children are drawn once, so it is the same position on every path.
    """
    g = Graph(me_white=True)
    if not dag:
        def rec(ply: int, path: tuple[str, ...], me_to_move: bool) -> Node:
            key = "/".join(path) or "root"
            if ply == depth:
                return g.add(key, ply, me_to_move, [], value=float(rng.uniform(0.5 - value_spread, 0.5 + value_spread)),
                             path=path)
            b = int(rng.integers(branching[0], branching[1] + 1))
            mv = [f"m{j}" for j in range(b)]
            node = g.add(key, ply, me_to_move, mv, p_model=rng.dirichlet([alpha] * b), path=path)
            kids = [rec(ply + 1, path + (m,), not me_to_move) for m in mv]
            g.link(node, kids)
            return node

        rec(0, (), me_first)
    else:
        pool = pool or 3
        layers: list[list[Node]] = []
        for ply in range(depth + 1):
            n_keys = 1 if ply == 0 else pool
            me_to_move = me_first if ply % 2 == 0 else not me_first
            layer = []
            for j in range(n_keys):
                key = "root" if ply == 0 else f"p{ply}.{j}"
                if ply == depth:
                    layer.append(g.add(key, ply, me_to_move, [],
                                       value=float(rng.uniform(0.5 - value_spread, 0.5 + value_spread))))
                else:
                    layer.append(None)
            layers.append(layer)
        for ply in range(depth - 1, -1, -1):
            me_to_move = me_first if ply % 2 == 0 else not me_first
            nxt = layers[ply + 1]
            for j in range(len(layers[ply])):
                key = "root" if ply == 0 else f"p{ply}.{j}"
                b = int(rng.integers(branching[0], min(branching[1], len(nxt)) + 1))
                kids = [nxt[x] for x in rng.choice(len(nxt), size=b, replace=False)]
                mv = [f"to{k.key}" for k in kids]
                node = g.add(key, ply, me_to_move, mv, p_model=rng.dirichlet([alpha] * b))
                g.link(node, kids)
                layers[ply][j] = node
        g.root = layers[0][0].id
    _engine_values(g, rng)
    _paths(g)
    return g


def _engine_values(g: Graph, rng: np.random.Generator, noise: float = 0.05) -> None:
    """A minimax-ish engine: each move's value is the subtree's best-play value plus noise."""
    best: dict[int, float] = {}
    for i in sorted(range(len(g.nodes)), key=lambda i: -g.nodes[i].ply):
        n = g.nodes[i]
        if n.children is None:
            best[i] = n.value
            continue
        vals = np.array([best[c] for c in n.children]) + rng.normal(0, noise, len(n.children))
        vals = np.clip(vals, 0, 1)
        best[i] = float(vals.max() if n.me_to_move else vals.min())
        n.engine = vals if n.me_to_move else 1.0 - vals


def _paths(g: Graph) -> None:
    for i in g.reachable():
        n = g.nodes[i]
        for m, c in zip(n.moves, n.children or ()):
            if not g.nodes[c].path or g.nodes[c].path == ():
                g.nodes[c].path = n.path + (m,)
