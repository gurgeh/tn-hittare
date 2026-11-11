"""Which positions to learn: the knapsack on the tree, and its repair on a DAG.

The DP. `f_v(k)` is my best expected score in the subtree under node v when at
most k budget units are spent inside it. Bottom-up:

- frontier / end of game / below `min_reach`: `f_v(k) = E0(v)`, the value with
  nothing learned below (plain expectation down to the frontier values);
- opponent's node with distribution q: `f_v = (+)_a q_a f_a`, the max-plus
  convolution over his moves: the best way to split k over the children;
- my node, not learned: the same convolution over my model's moves;
- my node, learned at cost c: `max over sound moves m of f_m(k - c)`;
- my node: the larger of the two, per k.

Every `f` is non-decreasing (k is "at most"), so the convolution of two of them
is too. One pass gives the whole curve `f_root(0..K)`; backtracking from
`f_root(N)` gives the learned set for N. Each array is capped at the budget its
subtree can absorb (the sum of the costs of its learnable nodes), the classical
tree-knapsack bound, so the DP is O(n K) convolution entries on a tree, not O(n K^2).

On a DAG the DP runs on the unrolled tree (each node's curve is computed once
and reused by every parent). That is wrong in two directions: a position shared
by two paths is paid for once per path it is learned in (pessimistic), and the
paths may make different decisions in it (optimistic: the game cannot learn a
position on one path only). `solve` therefore repairs: it takes the union of
the positions learned on any path, keeps the better move where paths disagree,
evaluates the set exactly, and then improves it by local search (add, switch,
drop-and-add) with exact verification. The value reported is always the exact
value of the returned set, never the DP's own figure.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from tnh.builder.game import Game, Key, Learned

TOL = 1e-12


def maxplus(a: np.ndarray, b: np.ndarray, K: int) -> np.ndarray:
    """`out[k] = max_{i + j = k} a[i] + b[j]` for k <= K, with non-decreasing a and b
    that saturate (a[i] = a[-1] for i past the end)."""
    n = min(len(a) + len(b) - 1, K + 1)
    if len(a) < len(b):
        a, b = b, a
    out = np.full(n, -np.inf)
    for j in range(min(len(b), n)):
        m = min(len(a), n - j)
        np.maximum(out[j : j + m], a[:m] + b[j], out=out[j : j + m])
    # entries past len(a) + j are a saturated + b[j]; with a, b non-decreasing the
    # loop above already covers every k < n since len(a) + len(b) - 1 >= n.
    return out


def at(f: np.ndarray, k: int) -> float:
    return float(f[min(k, len(f) - 1)])


@dataclass
class _NodeDP:
    f: np.ndarray
    prefixes: list[np.ndarray] | None = None    # convolution after each child (not learned)
    kids: list[int] | None = None               # child positions in the convolution order
    learn: np.ndarray | None = None             # per k: the learned move index, -1 = not learned
    learn_best: np.ndarray | None = None        # f if learned (for backtracking ties)


class DP:
    """The knapsack DP over a `Game`, up to budget K."""

    def __init__(self, game: Game, K: int) -> None:
        self.game = game
        self.K = K
        g = game.graph
        self.E0 = game.values({})
        bound = game.reach_bound()
        self.active = {i for i in game.order if bound[i] >= game.params.min_reach}
        self.nodes: dict[int, _NodeDP] = {}
        self.convolution_entries = 0
        cap: dict[int, int] = {}
        for i in reversed(game.order):
            n = g.nodes[i]
            if not n.expanded or i not in self.active:
                self.nodes[i] = _NodeDP(np.array([self.E0[i]]))
                cap[i] = 0
                continue
            d = game.dist[i]
            kids = [j for j in range(len(n.moves)) if d[j] > 0 and cap[n.children[j]] > 0]
            c_learn = game.cost(i) if game.learnable(i) else 0
            cap[i] = min(K, sum(cap[n.children[j]] for j in range(len(n.moves))) + c_learn)
            # the not-learned value: E0 from children that hold no budget, convolution over the rest
            base = float(sum(d[j] * self.E0[n.children[j]] for j in range(len(n.moves)) if j not in set(kids)))
            h = np.array([base])
            prefixes = [h]
            for j in kids:
                fc = self.nodes[n.children[j]].f
                self.convolution_entries += len(h) * len(fc)
                h = maxplus(h, d[j] * fc, K)
                prefixes.append(h)
            f = np.full(cap[i] + 1, h[-1])
            f[: len(h)] = h[: cap[i] + 1]
            nd = _NodeDP(f, prefixes, kids)
            if c_learn:
                learn = np.full(cap[i] + 1, -1, dtype=np.int64)
                lb = np.full(cap[i] + 1, -np.inf)
                for j in game.sound[i]:
                    fc = self.nodes[n.children[j]].f
                    for k in range(c_learn, cap[i] + 1):
                        v = at(fc, k - c_learn)
                        if v > lb[k] + TOL:
                            lb[k], learn[k] = v, j
                better = lb > f + TOL
                f = np.where(better, lb, f)
                learn[~better] = -1
                nd.f, nd.learn, nd.learn_best = f, learn, lb
            self.nodes[i] = nd

    def curve(self) -> np.ndarray:
        f = self.nodes[self.game.graph.root].f
        return np.array([at(f, k) for k in range(self.K + 1)])

    def backtrack(self, N: int) -> list[tuple[int, int]]:
        """The (node id, move index) pairs learned for budget N, one per path copy."""
        out: list[tuple[int, int]] = []
        self._back(self.game.graph.root, N, out)
        return out

    def _back(self, i: int, k: int, out: list) -> None:
        nd = self.nodes[i]
        f = nd.f
        k = min(k, len(f) - 1)
        # spend as little as gives the same value
        target = f[k] - TOL
        k = int(np.argmax(f >= target))
        if k == 0 or nd.prefixes is None:
            return
        n = self.game.graph.nodes[i]
        if nd.learn is not None and nd.learn[k] >= 0:
            j = int(nd.learn[k])
            out.append((i, j))
            self._back(n.children[j], k - self.game.cost(i), out)
            return
        d = self.game.dist[i]
        for pos in range(len(nd.kids) - 1, -1, -1):
            j = nd.kids[pos]
            prev, fc = nd.prefixes[pos], self.nodes[n.children[j]].f
            best, best_b = -np.inf, 0
            for b in range(0, min(k, len(fc) - 1) + 1):
                v = at(prev, k - b) + d[j] * fc[b]
                if v > best + TOL:
                    best, best_b = v, b
            if best_b:
                self._back(n.children[j], best_b, out)
            k -= best_b


@dataclass
class Solution:
    N: int
    learned: dict[Key, str]
    value: float                 # exact expected score of `learned`
    dp_value: float              # what the DP claimed for N (equal to `value` on a tree)
    cost: int                    # budget units used
    local_steps: list[str] = field(default_factory=list)


def repair(game: Game, copies: list[tuple[int, int]]) -> dict[Key, str]:
    """One move per position from the DP's per-path decisions."""
    g = game.graph
    moves: dict[Key, set[str]] = {}
    for i, j in copies:
        n = g.nodes[i]
        moves.setdefault(n.key, set()).add(n.moves[j])
    learned = {k: min(ms) for k, ms in moves.items()}
    for k, ms in moves.items():
        if len(ms) > 1:
            best = max(sorted(ms), key=lambda m: game.evaluate({**learned, k: m}))
            learned[k] = best
    return learned


def improve(game: Game, learned: dict[Key, str], N: int, max_steps: int = 1000) -> tuple[dict[Key, str], list[str]]:
    """Local search with exact verification: the best single add or switch that fits,
    else the best drop-then-add, until nothing improves."""
    steps: list[str] = []
    cur = dict(learned)
    val = game.evaluate(cur)
    for _ in range(max_steps):
        used = game.total_cost(cur)
        best = (val + TOL, None)
        marg = game.marginals(cur)
        cands = sorted(((gain, k, m) for (k, m), gain in marg.items() if m is not None and gain > TOL),
                       key=lambda t: -t[0])
        for gain, k, m in cands[:20]:
            if k in cur and cur[k] == m:
                continue
            extra = 0 if k in cur else game.key_cost(k)
            if used + extra > N:
                continue
            trial = {**cur, k: m}
            v = game.evaluate(trial)
            if v > best[0]:
                best = (v, (f"add {k}:{m}" if k not in cur else f"switch {k}:{m}", trial))
        if best[1] is None:
            for d in list(cur):
                base = {k: m for k, m in cur.items() if k != d}
                bused = game.total_cost(base)
                marg = game.marginals(base)
                cands = sorted(((gain, k, m) for (k, m), gain in marg.items()
                                if m is not None and gain > TOL and k != d and k not in base),
                               key=lambda t: -t[0])
                for gain, k, m in cands[:5]:
                    if bused + game.key_cost(k) > N:
                        continue
                    trial = {**base, k: m}
                    v = game.evaluate(trial)
                    if v > best[0]:
                        best = (v, (f"swap {d} -> {k}:{m}", trial))
        if best[1] is None:
            break
        val = best[0]
        steps.append(best[1][0])
        cur = best[1][1]
    return cur, steps


def solve(game: Game, K: int, local_search: bool = True, overshoot: int = 2) -> tuple[DP, list[Solution]]:
    """The DP and a solution for every N in 0..K.

    On a DAG the DP pays for a shared position once per path, so the repaired set of
    a larger budget N' often fits in N. The DP is run to `overshoot * K`, and for each
    N the best repaired set that fits is the start of the local search. On a tree
    every repaired set costs exactly its budget, and this changes nothing.
    """
    tree = is_tree(game)
    dp = DP(game, K if tree else overshoot * K)
    curve = dp.curve()
    starts: list[tuple[float, int, dict[Key, str]]] = []
    for M in range(dp.K + 1):
        learned = repair(game, dp.backtrack(M))
        starts.append((game.evaluate(learned), game.total_cost(learned), learned))
    sols = []
    for N in range(K + 1):
        fits = [(v, M) for M, (v, c, _) in enumerate(starts) if c <= N]
        _, M = max(fits, key=lambda t: (t[0] + (TOL if t[1] == N else 0.0), -t[1]))
        learned = dict(starts[M][2])
        steps: list[str] = [f"start from N'={M}"] if M != N else []
        if local_search:
            learned, more = improve(game, learned, N)
            steps += more
            if not tree:
                # on a DAG, also a plain greedy from nothing; keep the better
                alt, alt_steps = improve(game, {}, N)
                if game.evaluate(alt) > game.evaluate(learned) + TOL:
                    learned, steps = alt, ["greedy start"] + alt_steps
        cost = game.total_cost(learned)
        if cost > N:
            raise AssertionError(f"solution for N={N} costs {cost}")
        sols.append(Solution(N, learned, game.evaluate(learned), float(curve[N]), cost, steps))
    # the exact values must not fall with N: carry a better smaller solution forward
    for N in range(1, K + 1):
        if sols[N].value < sols[N - 1].value - TOL:
            prev = sols[N - 1]
            sols[N] = Solution(N, dict(prev.learned), prev.value, sols[N].dp_value, prev.cost,
                               sols[N].local_steps + [f"kept N={N - 1}"])
    return dp, sols


def candidates(game: Game) -> dict[Key, list[str]]:
    """Every learnable position with its sound moves."""
    out: dict[Key, list[str]] = {}
    for i, s in game.sound.items():
        n = game.graph.nodes[i]
        if len(s):
            out.setdefault(n.key, [n.moves[j] for j in s])
    return out


def brute_force(game: Game, N: int, keys: list[Key] | None = None) -> tuple[float, dict[Key, str]]:
    """The optimum by enumeration: every set of positions within budget N, every sound move."""
    cand = candidates(game)
    if keys is not None:
        cand = {k: v for k, v in cand.items() if k in set(keys)}
    ks = list(cand)
    best = (game.evaluate({}), {})
    for r in range(1, N + 1):
        for subset in itertools.combinations(ks, r):
            if sum(game.key_cost(k) for k in subset) > N:
                continue
            for mvs in itertools.product(*(cand[k] for k in subset)):
                learned = dict(zip(subset, mvs))
                v = game.evaluate(learned)
                if v > best[0] + TOL:
                    best = (v, learned)
    return best


def drop_one_gains(game: Game, learned: Learned) -> dict[Key, float]:
    """For each learned position, the exact loss in expected score if it were forgotten."""
    v = game.evaluate(learned)
    return {k: v - game.evaluate({kk: m for kk, m in learned.items() if kk != k}) for k in learned}


def is_tree(game: Game) -> bool:
    g = game.graph
    parents: dict[int, int] = {}
    keys = set()
    for i in game.order:
        n = g.nodes[i]
        if n.key in keys:
            return False
        keys.add(n.key)
        for c in n.children or ():
            if c in parents:
                return False
            parents[c] = i
    return True

