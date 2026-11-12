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
    free: bool = False                          # f(0) beats E0: something below is learned at no cost


class DP:
    """The knapsack DP over a `Game`, up to budget K.

    Costs are per move and may be 0 (an obvious move). Then a subtree can gain with no
    budget at all, so a child that holds no budget enters with its own f(0), not with
    E0, and backtracking descends wherever f(0) beats E0. Nothing else changes: the
    recursion is exact on a tree for any non-negative integer costs.
    """

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
            fs = [self.nodes[c].f for c in n.children]
            kids = [j for j in range(len(n.moves)) if d[j] > 0 and len(fs[j]) > 1]
            ks = set(kids)
            sound = game.sound.get(i, np.zeros(0, dtype=np.int64)) if game.learnable(i) else []
            costs = {int(j): game.move_cost(i, int(j)) for j in sound}
            cap[i] = min(K, sum(cap[c] for c in n.children) + max(costs.values(), default=0))
            # children that hold no budget enter at their own f(0)
            base = float(sum(d[j] * fs[j][0] for j in range(len(n.moves)) if j not in ks))
            h = np.array([base])
            prefixes = [h]
            for j in kids:
                self.convolution_entries += len(h) * len(fs[j])
                h = maxplus(h, d[j] * fs[j], K)
                prefixes.append(h)
            f = np.full(cap[i] + 1, h[-1])
            f[: len(h)] = h[: cap[i] + 1]
            nd = _NodeDP(f, prefixes, kids)
            if costs:
                learn = np.full(cap[i] + 1, -1, dtype=np.int64)
                lb = np.full(cap[i] + 1, -np.inf)
                for j, c in costs.items():
                    for k in range(c, cap[i] + 1):
                        v = at(fs[j], k - c)
                        if v > lb[k] + TOL:
                            lb[k], learn[k] = v, j
                better = lb > f + TOL
                f = np.where(better, lb, f)
                learn[~better] = -1
                nd.f, nd.learn = f, learn
            nd.free = bool(nd.f[0] > self.E0[i] + TOL)
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
        k = int(np.argmax(f >= f[k] - TOL))      # spend as little as gives the same value
        if nd.prefixes is None or (k == 0 and not nd.free):
            return
        n = self.game.graph.nodes[i]
        if nd.learn is not None and nd.learn[k] >= 0:
            j = int(nd.learn[k])
            out.append((i, j))
            self._back(n.children[j], k - self.game.move_cost(i, j), out)
            return
        d = self.game.dist[i]
        alloc = {}
        for pos in range(len(nd.kids) - 1, -1, -1):
            j = nd.kids[pos]
            prev, fc = nd.prefixes[pos], self.nodes[n.children[j]].f
            best, best_b = -np.inf, 0
            for b in range(0, min(k, len(fc) - 1) + 1):
                v = at(prev, k - b) + d[j] * fc[b]
                if v > best + TOL:
                    best, best_b = v, b
            alloc[j] = best_b
            k -= best_b
        for j, c in enumerate(n.children):
            b = alloc.get(j, 0)
            if d[j] > 0 and (b > 0 or self.nodes[c].free):
                self._back(c, b, out)


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


def improve(game: Game, learned: dict[Key, str], N: int, max_steps: int = 1000,
            swap_limit: int = 12) -> tuple[dict[Key, str], list[str]]:
    """Local search with exact verification: the best single add or switch that fits,
    else (for sets of at most `swap_limit` positions, since each try re-values the
    graph) the best drop-then-add, until nothing improves. Forced positions stay."""
    steps: list[str] = []
    cur = {k: m for k, m in learned.items() if k not in game.forced}
    val = game.evaluate(cur)
    for _ in range(max_steps):
        used = game.total_cost(cur)
        best = (val + TOL, None)
        marg = game.marginals(cur)
        cands = sorted(((gain, k, m) for (k, m), gain in marg.items()
                        if m is not None and gain > TOL and k not in game.forced), key=lambda t: -t[0])
        for gain, k, m in cands[:20]:
            if k in cur and cur[k] == m:
                continue
            extra = game.cost_of(k, m) - (game.cost_of(k, cur[k]) if k in cur else 0)
            if used + extra > N:
                continue
            trial = {**cur, k: m}
            v = game.evaluate(trial)
            if v > best[0]:
                best = (v, (f"add {k}:{m}" if k not in cur else f"switch {k}:{m}", trial))
        if best[1] is None and len(cur) <= swap_limit:
            for d in list(cur):
                base = {k: m for k, m in cur.items() if k != d}
                bused = game.total_cost(base)
                marg = game.marginals(base)
                cands = sorted(((gain, k, m) for (k, m), gain in marg.items()
                                if m is not None and gain > TOL and k != d and k not in base
                                and k not in game.forced), key=lambda t: -t[0])
                for gain, k, m in cands[:5]:
                    if bused + game.cost_of(k, m) > N:
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


def solve(game: Game, K: int, local_search: bool = True, overshoot: int = 2,
          improve_at: set[int] | None = None, greedy_max: int = 12) -> tuple[DP, list[Solution]]:
    """The DP and a solution for every N in 0..K.

    On a DAG the DP pays for a shared position once per path, so the repaired set of
    a larger budget N' often fits in N. The DP is run to `overshoot * K`, and for each
    N the best repaired set that fits is the start of the local search. On a tree
    every repaired set costs exactly its budget, and this changes nothing.

    `improve_at`: the budgets that get the local search (default all); the others keep
    the best repaired set. On a DAG a second local search from nothing runs for budgets up
    to `greedy_max` (it takes a step per position, each re-valuing the graph). Forced moves (`Params.force`) take their cost out of every
    budget; a budget below it has no solution (value -inf). Every solution's `learned`
    includes the forced moves.
    """
    tree = is_tree(game)
    fc = game.forced_cost
    dp = DP(game, max(K - fc, 0) if tree else overshoot * max(K - fc, 0))
    curve = dp.curve()
    starts: list[tuple[float, int, dict[Key, str]]] = []
    for M in range(dp.K + 1):
        learned = repair(game, dp.backtrack(M))
        starts.append((game.evaluate(learned), game.total_cost(learned), learned))
    sols = []
    for N in range(K + 1):
        if N < fc:
            sols.append(Solution(N, dict(game.forced), -np.inf, -np.inf, fc, ["infeasible: forced moves cost more"]))
            continue
        fits = [(v, M) for M, (v, c, _) in enumerate(starts) if c <= N]
        _, M = max(fits, key=lambda t: (t[0] + (TOL if t[1] == N - fc else 0.0), -t[1]))
        learned = dict(starts[M][2])
        steps: list[str] = [f"start from N'={M + fc}"] if M != N - fc else []
        if local_search and (improve_at is None or N in improve_at):
            learned, more = improve(game, learned, N)
            steps += more
            if not tree and N <= greedy_max:
                # on a DAG, also a plain greedy from nothing; keep the better
                alt, alt_steps = improve(game, {}, N)
                if game.evaluate(alt) > game.evaluate(learned) + TOL:
                    learned, steps = alt, ["greedy start"] + alt_steps
        cost = game.total_cost(learned)
        if cost > N:
            raise AssertionError(f"solution for N={N} costs {cost}")
        learned = {**learned, **game.forced}
        sols.append(Solution(N, learned, game.evaluate(learned), float(curve[min(N - fc, dp.K)]), cost, steps))
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
    """The optimum by enumeration: every feasible choice of nothing or a sound move per
    position, depth first with the budget that remains. With zero-cost moves any number
    of positions may be learned, so small games only."""
    cand = candidates(game)
    if keys is not None:
        cand = {k: v for k, v in cand.items() if k in set(keys)}
    ks = [k for k in cand if k not in game.forced]
    costs = {k: {m: game.cost_of(k, m) for m in cand[k]} for k in ks}
    best = [-np.inf, {}]
    chosen: dict[Key, str] = {}

    def rec(x: int, left: int) -> None:
        if x == len(ks):
            v = game.evaluate(chosen)
            if v > best[0] + TOL:
                best[0], best[1] = v, dict(chosen)
            return
        rec(x + 1, left)
        k = ks[x]
        for m, c in costs[k].items():
            if c <= left:
                chosen[k] = m
                rec(x + 1, left - c)
                del chosen[k]

    rec(0, N - game.forced_cost)
    return best[0], {**best[1], **game.forced}


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

