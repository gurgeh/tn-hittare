"""Which positions to learn under a price per non-obvious learned move.

The objective is `F(S) = V(S) - mu * C(S)`: my expected score with the learned set S
minus the price times the moves in S that cost (cost 0 for an obvious move, 1
otherwise, `tnh.builder.game.ObviousCost`).

On a tree the decision at one of my positions depends only on its reach r and on
what is best below it, so the net value of a subtree is a function of r:

    W_v(r) = max over decisions below v of  r * V_v - mu * C_v

which is convex and piecewise linear with W(0) = 0, and is computed bottom-up exactly:

- frontier, or a position nothing can be learned in: `r * V0(v)`;
- a chance node (the opponent, or me playing by my model): `sum_a W_a(q_a * r)`;
- learning move a at my position: `W_a(r) - mu * cost(a)`;
- my position: the upper envelope of its options.

One top-down pass then reads off the decisions: at each of my positions, with the
reach the decisions above give it, take the option whose W is largest at that reach.
That is exact on a tree for every mu (checked against enumeration).

On a graph with transpositions a position is learned once and its reach is the sum
over its paths. The bottom-up W is computed on the unrolled graph (each node once), the
top-down pass runs in topological order with reach summed over parents, so each
position's decision sees its total reach; then coordinate ascent on the exact F (the
best single change of one position's decision, verified by an exact evaluation) until
none improves. Guaranteed: F never falls during the ascent, the result is a local
optimum against single-position changes, and on a tree the ascent finds nothing.

A cap of N non-obvious moves is a bisection on mu. It reaches the points of the concave
hull of (cost, value): a cap of N can return fewer than N moves when the hull skips N.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tnh.builder.game import Game, Key, Learned

TOL = 1e-12


# -- convex piecewise-linear functions on [0, 1] -----------------------------------------------

class PL:
    """A convex piecewise-linear function on [0, 1], by its breakpoints."""

    __slots__ = ("x", "y")

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x, self.y = x, y

    @staticmethod
    def line(v: float) -> "PL":
        return PL(np.array([0.0, 1.0]), np.array([0.0, v]))

    def __call__(self, r: float) -> float:
        return float(np.interp(r, self.x, self.y))

    def scaled(self, q: float) -> "PL":
        """r -> f(q r)."""
        if q <= 0:
            return PL(np.array([0.0, 1.0]), np.array([self.y[0], self.y[0]]))
        inner = self.x[(self.x > 0) & (self.x < q)] / q
        x = np.concatenate([[0.0], inner, [1.0]])
        return PL(x, np.interp(x * q, self.x, self.y))

    def shifted(self, c: float) -> "PL":
        return PL(self.x, self.y - c)

    @staticmethod
    def add(fs: list["PL"]) -> "PL":
        x = np.unique(np.concatenate([f.x for f in fs]))
        return PL(x, sum(np.interp(x, f.x, f.y) for f in fs)).simplified()

    def max(self, g: "PL") -> "PL":
        x = np.unique(np.concatenate([self.x, g.x]))
        a, b = np.interp(x, self.x, self.y), np.interp(x, g.x, g.y)
        d = a - b
        cross = np.flatnonzero(d[:-1] * d[1:] < 0)
        if len(cross):
            t = d[cross] / (d[cross] - d[cross + 1])
            xc = x[cross] + t * (x[cross + 1] - x[cross])
            x = np.unique(np.concatenate([x, xc]))
            a, b = np.interp(x, self.x, self.y), np.interp(x, g.x, g.y)
        return PL(x, np.maximum(a, b)).simplified()

    def simplified(self, tol: float = 1e-12) -> "PL":
        if len(self.x) <= 2:
            return self
        dx = np.diff(self.x)
        keep = dx > 1e-15
        x = np.concatenate([[self.x[0]], self.x[1:][keep]])
        y = np.concatenate([[self.y[0]], self.y[1:][keep]])
        if len(x) <= 2:
            return PL(x, y)
        s = np.diff(y) / np.diff(x)
        bend = np.abs(np.diff(s)) > tol
        idx = np.concatenate([[0], np.flatnonzero(bend) + 1, [len(x) - 1]])
        return PL(x[idx], y[idx])


# -- the solve ---------------------------------------------------------------------------------

@dataclass
class Priced:
    mu: float
    learned: dict[Key, str]
    value: float            # V(S): my expected score at the root
    cost: int               # C(S)
    net: float              # V - mu * C
    steps: list[str] = field(default_factory=list)
    W: dict[int, PL] | None = None


def subtree_functions(game: Game, mu: float) -> tuple[dict[int, PL], dict[int, list[tuple[int, PL]]]]:
    """W per node, and per learnable node its learn options (move index, W of the option)."""
    g = game.graph
    E0 = game.values({})
    W: dict[int, PL] = {}
    opts: dict[int, list[tuple[int, PL]]] = {}
    for i in reversed(game.order):
        n = g.nodes[i]
        if not n.expanded:
            W[i] = PL.line(E0[i])
            continue
        d = game.dist[i]
        parts = [W[c].scaled(d[j]) for j, c in enumerate(n.children) if d[j] > 0]
        f = PL.add(parts) if parts else PL.line(0.0)
        if game.learnable(i):
            o = []
            for j in game.sound[i]:
                fa = W[n.children[j]].shifted(mu * game.move_cost(i, int(j)))
                o.append((int(j), fa))
                f = f.max(fa)
            opts[i] = o
        W[i] = f
    return W, opts


def decide(game: Game, W: dict[int, PL], opts: dict[int, list[tuple[int, PL]]]) -> dict[Key, str]:
    """Top-down in topological order; each position decides at its reach summed over parents."""
    g = game.graph
    R = np.zeros(len(g.nodes))
    R[g.root] = 1.0
    learned: dict[Key, str] = {}
    for i in game.order:
        n = g.nodes[i]
        if not n.expanded or R[i] <= 0:
            continue
        pol = game.dist[i]
        if i in opts and n.key not in learned:
            r = R[i] if len(game.by_key[n.key]) == 1 else sum(R[x] for x in game.by_key[n.key])
            unl = PL.add([W[c].scaled(pol[j]) for j, c in enumerate(n.children) if pol[j] > 0]) \
                if np.any(pol > 0) else PL.line(0.0)
            best_v, best_j = unl(r), None
            for j, fa in opts[i]:
                v = fa(r)
                if v > best_v + TOL:
                    best_v, best_j = v, j
            if best_j is not None:
                learned[n.key] = n.moves[best_j]
        if n.me_to_move and n.key in learned:
            pol = np.zeros(len(n.moves))
            pol[n.moves.index(learned[n.key])] = 1.0
        np.add.at(R, n.children, R[i] * pol)
    return learned


def net_value(game: Game, learned: Learned, mu: float) -> tuple[float, float, int]:
    v = game.evaluate(learned)
    c = game.total_cost(learned)
    return v - mu * c, v, c


def ascend(game: Game, learned: dict[Key, str], mu: float, max_steps: int = 10_000) -> tuple[dict[Key, str], list[str]]:
    """Best single change of one position's decision by the exact F, until none improves."""
    cur = {k: m for k, m in learned.items() if k not in game.forced}
    f, _, _ = net_value(game, cur, mu)
    steps = []
    for _ in range(max_steps):
        marg = game.marginals(cur)
        cands = []
        for (k, m), dv in marg.items():
            if k in game.forced:
                continue
            old = game.cost_of(k, cur[k]) if k in cur else 0
            new = game.cost_of(k, m) if m is not None else 0
            if m is not None and cur.get(k) == m:
                continue
            gain = dv - mu * (new - old)
            if gain > TOL:
                cands.append((gain, k, m))
        moved = False
        for gain, k, m in sorted(cands, key=lambda t: -t[0])[:10]:
            trial = dict(cur)
            if m is None:
                del trial[k]
            else:
                trial[k] = m
            ft, _, _ = net_value(game, trial, mu)
            if ft > f + TOL:
                steps.append(f"{'drop' if m is None else 'set'} {k}:{m}")
                cur, f, moved = trial, ft, True
                break
        if not moved:
            break
    return cur, steps


def solve_price(game: Game, mu: float, *, ascent: bool = True, keep_W: bool = False) -> Priced:
    W, opts = subtree_functions(game, mu)
    learned = decide(game, W, opts)
    steps: list[str] = []
    if ascent:
        learned, steps = ascend(game, learned, mu)
    learned = {**learned, **game.forced}
    f, v, c = net_value(game, learned, mu)
    return Priced(mu, learned, v, c, f, steps, W if keep_W else None)


def solve_cap(game: Game, N: int, *, mu_hi: float = 1.0, iters: int = 40, ascent: bool = True) -> Priced:
    """The repertoire at the smallest price that learns at most N non-obvious moves."""
    best = solve_price(game, mu_hi, ascent=ascent)
    if best.cost > N:
        raise ValueError(f"even at price {mu_hi} the repertoire costs {best.cost} > {N}")
    lo_sol = solve_price(game, 0.0, ascent=ascent)
    if lo_sol.cost <= N:
        return lo_sol
    lo, hi = 0.0, mu_hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s = solve_price(game, mid, ascent=ascent)
        if s.cost <= N:
            hi, best = mid, s
        else:
            lo = mid
    return best
