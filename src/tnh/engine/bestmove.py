"""preppy's second mode, `bestmove`: plain expectimax to a reach horizon, deepened step by step.

No price, no learning costs, no PUCT and no visits: the best move at my positions against
the opponent's distribution, to a horizon set by the models alone, so that a value depends
on the models and the settings and not on where the search happened to go.

**Horizon weight rho**, top-down from the root, from the models only:

- rho(root) = 1; with me to move at the root every candidate gets rho 1 (all root moves
  are searched to the same horizon, so their values compare);
- at his positions rho(child) = rho * q_a, q his database-and-model mix without the
  prepared share;
- at my positions rho(child) = rho * (P_a / max_b P_b) ** alpha, P my PUCT prior (the move
  model at my rating blended with Lc0's policy): my top candidate costs nothing, unlikely
  candidates of mine get a shallower horizon. Candidates as in the priced mode: prior at
  least `prior_min`; a candidate whose own value is more than `blunder` below the
  position's own value is dropped once it is known;
- with transpositions the largest rho over the ways in.

**Value at horizon theta.** A position is inside when rho >= theta (and it is not a finished
game). Inside and mine: the max over my candidates of the child's value. Inside and his:
`(1 - Prepared) * sum_a q_a V(child_a) + Prepared * min_a V(child_a)`, the min over the
children's values (solved, not static: there is no price here). Outside the horizon or
finished: the position's own static value (with DrawScore). Every child of an inside
position has been evaluated, so a value at theta is a function of the models and the
settings only: neither extra positions in the graph nor the order of expansion change it.

**Depths.** theta_k = 2 ** (-k / 2). Depth k is complete when every position inside the
horizon and every relevant child of one (my candidates, all his moves) has been
evaluated; its values are then final. Evaluations and the graph are the priced mode's.

**Selectivity** (Settings and UCI; B and C on by default since 2026-11-12, A off). Each keeps a
value at a depth a function of the models, the settings and the earlier completed depths:

- A, `his_anchor` (experimental, off): at an inside position of his only the replies inside the horizon (rho >= theta,
  "searched") are evaluated, and
  `V(n) = (1 - Prepared) * (s(n) + sum_searched q_a (V(child_a) - s(child_a))) + Prepared * min_searched V(child_a)`,
  s the static value, V(n) = s(n) with no searched reply. MistakeFix's best move is the searched
  reply with the highest static value for him (the other replies have no value). Measured on
  c4 (2026-11-12): values +0.04 at depth 12 against full width, since the static value of his
  positions is on average 0.003 better for me than the expectation over his replies' static values.
- B, `widen` = beta > 0 (default 4): at an inside position of mine with weight rho (not the root) candidate a
  only if rho * w_a >= theta / beta, w_a its horizon factor; the top candidate always.
- C, `reduce_margin` > 0 (default 0.03, `reduce_penalty` 0.5): at depth k, a candidate of mine (not at the root) whose value at depth
  k - 1 was more than the margin below the best candidate's there gets its horizon factor times
  `reduce_penalty`.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

import numpy as np

from tnh.builder.game import Node


def theta(k: int) -> float:
    return 2.0 ** (-k / 2)


@dataclass
class Depth:
    k: int
    theta: float
    complete: bool
    root_values: dict[str, float]          # root move -> value
    inside: dict[str, int]                 # root move -> positions inside its horizon
    value: float                           # the root's value
    seconds: float = 0.0
    positions_inside: int = 0
    pass_seconds: float = 0.0
    V: dict[int, float] = field(default_factory=dict)
    mine: dict[int, dict[int, float]] = field(default_factory=dict)   # my inside positions: candidate -> value


class BestMove:
    def __init__(self, engine) -> None:
        self.e = engine
        self.completed: dict[int, Depth] = {}       # for the current root and settings
        self._root = None
        self._key = None
        self._prev: Depth | None = None             # the completed depth before the one in progress (C)
        self._rho_last: dict[int, float] = {}       # the last values pass's horizon (A's searched replies)
        self._th_last = 1.0

    # -- the models' quantities ---------------------------------------------------------------

    def _evaluated(self, n: Node) -> bool:
        return n.children is not None or "result_mover" in n.info or bool(n.info.get("valued"))

    def _finished(self, n: Node) -> bool:
        return "result_mover" in n.info

    def _mix(self, n: Node, p: np.ndarray) -> np.ndarray:
        """His database mix laid over probabilities p (no prepared share)."""
        from tnh.builder.game import blend
        params = self.e.params()
        if params.mix is not None and n.counts is not None and np.sum(n.counts) > 0:
            return np.asarray(params.mix(n.counts, p), dtype=np.float64)
        return blend(n.counts, p, params.kappa_at(n))

    def _q_rho(self, n: Node) -> np.ndarray:
        """For the horizon: his mix over the move model's first probabilities (no correction,
        so the horizon depends on the move model alone)."""
        return self._mix(n, n.info["p_first"])

    def _q(self, n: Node, searched: list[int] | None = None) -> np.ndarray:
        """For the values: his mix over the corrected probabilities, the correction's best move
        being the child with the highest static value for him among all his evaluated
        children (all of them, once the position is inside a completed depth); with A among
        the searched replies only (the most likely move when none is searched)."""
        if not self.e.s.his_anchor:
            return self._mix(n, self.e.fixed(n.id, self.e.best_valued(n.id)))
        if searched is None:
            searched = self._searched(n, self._rho_last, self._th_last)
        g = self.e.graph
        best = max(searched, key=lambda j: 1.0 - g.nodes[n.children[j]].value) if searched else None
        return self._mix(n, self.e.fixed(n.id, best))

    def _searched(self, n: Node, rho: dict[int, float], th: float) -> list[int]:
        """A: his replies inside the horizon and evaluated."""
        g = self.e.graph
        return [j for j, c in enumerate(n.children) if rho.get(c, 0.0) >= th and self._evaluated(g.nodes[c])]

    def _candidates(self, i: int) -> list[int]:
        """My candidate move indices at an evaluated position: the prior gate, then the
        blunder gate where the child's own value is known."""
        e, g = self.e, self.e.graph
        n = g.nodes[i]
        prior = e._prior(n)
        out = []
        for j in range(len(n.moves)):
            if prior[j] < e.s.prior_min:
                continue
            c = g.nodes[n.children[j]]
            if self._evaluated(c) and c.value < n.value - e.s.blunder:
                continue
            out.append(j)
        if not out:
            out = [int(np.argmax(prior))]
        return out

    def _my_moves(self, i: int, at_root: bool, rho_i: float = 1.0, th: float = 0.0) -> list[tuple[int, float]]:
        """(move index, rho factor) for my candidates considered at an inside position with
        weight rho_i at horizon th: the reduction (C) and the widening (B) applied."""
        s, n = self.e.s, self.e.graph.nodes[i]
        prior = self.e._prior(n)
        cand = self._candidates(i)
        if at_root:
            return [(j, 1.0) for j in cand]
        top = max(prior[j] for j in cand)
        f = {j: (prior[j] / top) ** s.alpha for j in cand}
        prev = self._prev.mine.get(i) if (s.reduce_margin > 0 and self._prev is not None) else None
        if prev:
            best = max(prev.values())
            for j, v in prev.items():
                if j in f and v < best - s.reduce_margin:
                    f[j] *= s.reduce_penalty
        out = [(j, w) for j, w in f.items() if s.widen <= 0 or rho_i * w >= th / s.widen]
        return out or [max(f.items(), key=lambda t: (t[1], -t[0]))]

    def _factors(self, i: int, at_root: bool, rho_i: float = 1.0, th: float = 0.0) -> list[tuple[int, float]]:
        """(child id, rho factor) for the children that matter at an inside position."""
        g, n = self.e.graph, self.e.graph.nodes[i]
        if n.me_to_move:
            return [(n.children[j], f) for j, f in self._my_moves(i, at_root, rho_i, th)]
        q = self._q_rho(n)
        return [(c, float(q[j])) for j, c in enumerate(n.children) if q[j] > 0]

    # -- the horizon ---------------------------------------------------------------------------

    def rho(self, th: float) -> tuple[dict[int, float], list[int], list[int]]:
        """rho over the evaluated part of the graph, propagated only from inside positions;
        the positions that still need an evaluation for depth th (largest rho first); and the
        inside positions that are evaluated but have no children yet (created without model
        calls)."""
        g = self.e.graph
        root = g.root
        rho = {root: 1.0}
        heap = [(g.nodes[root].ply, root)]
        need: dict[int, float] = {}
        kids: list[int] = []
        done = set()
        while heap:
            _, i = heapq.heappop(heap)
            if i in done:
                continue
            done.add(i)
            n = g.nodes[i]
            if not self._evaluated(n):
                if rho[i] >= th:
                    need[i] = rho[i]
                continue
            if self._finished(n) or rho[i] < th:
                continue
            if n.children is None:                       # inside, valued only: its children come now
                kids.append(i)
                continue
            anchor = self.e.s.his_anchor and not n.me_to_move
            for c, f in self._factors(i, i == root, rho[i], th):
                r = rho[i] * f
                if r > rho.get(c, -1.0):
                    rho[c] = r
                if not self._evaluated(g.nodes[c]) and (not anchor or r >= th):
                    need[c] = max(need.get(c, 0.0), r)       # children of inside positions are valued
                if c not in done:
                    heapq.heappush(heap, (g.nodes[c].ply, c))
        return rho, [i for i, _ in sorted(need.items(), key=lambda t: -t[1])], kids

    # -- values ----------------------------------------------------------------------------------

    def values(self, rho: dict[int, float], th: float) -> dict[int, float]:
        """One bottom-up pass over the inside set. Positions not evaluated yet (only while a
        depth is in progress) stand in with their parent's own value."""
        g, P = self.e.graph, self.e.s.prepared
        # inside and with children; while a depth is in progress a position that came inside
        # but has no children yet counts with its own static value
        inside = [i for i, r in rho.items() if r >= th and g.nodes[i].children is not None
                  and not self._finished(g.nodes[i])]
        inside.sort(key=lambda i: -g.nodes[i].ply)
        V: dict[int, float] = {}
        self.mine = {}
        self._rho_last, self._th_last = rho, th

        def val(c: int, parent: Node) -> float:
            if c in V:
                return V[c]
            cn = g.nodes[c]
            return cn.value if self._evaluated(cn) else parent.value

        for i in inside:
            n = g.nodes[i]
            if n.me_to_move:
                cv = {j: val(n.children[j], n) for j, _ in self._my_moves(i, i == g.root, rho[i], th)}
                self.mine[i] = cv
                V[i] = max(cv.values())
            elif self.e.s.his_anchor:
                sj = self._searched(n, rho, th)
                if not sj:
                    V[i] = n.value
                    continue
                q = self._q(n, sj)
                vs = np.array([val(n.children[j], n) for j in sj])
                ss = np.array([g.nodes[n.children[j]].value for j in sj])
                V[i] = (1 - P) * (n.value + float(q[sj] @ (vs - ss))) + P * float(vs.min())
            else:
                q = self._q(n)
                vs = np.array([val(c, n) for c in n.children])
                V[i] = (1 - P) * float(q @ vs) + P * float(vs[q > 0].min())
        return V

    def depth_result(self, k: int, th: float, rho: dict[int, float], complete: bool, t0: float) -> Depth:
        g = self.e.graph
        tp = time.perf_counter()
        V = self.values(rho, th)
        pass_s = time.perf_counter() - tp
        root = g.nodes[g.root]
        rv = {}
        for c, _ in self._factors(g.root, True) if root.children is not None and not self._finished(root) else []:
            j = root.children.index(c)
            cn = g.nodes[c]
            v = V.get(c, cn.value)
            rv[root.moves[j]] = v if root.me_to_move else 1 - v
        # positions inside each root move's horizon (by the way in with the largest rho)
        inside = self._inside_per_root_move(rho, th)
        v_root = V.get(g.root, root.value)
        return Depth(k, th, complete, rv, inside, v_root if root.me_to_move else 1 - v_root,
                     time.perf_counter() - t0, sum(1 for r in rho.values() if r >= th), pass_s, V, self.mine)

    def _inside_per_root_move(self, rho, th) -> dict[str, int]:
        g = self.e.graph
        root = g.nodes[g.root]
        if root.children is None or self._finished(root):
            return {}
        counts = {}
        for j, c in enumerate(root.children):
            if rho.get(c, 0.0) < th:
                continue
            seen, stack, cnt = {c}, [c], 0
            while stack:
                i = stack.pop()
                n = g.nodes[i]
                if rho.get(i, 0.0) < th or n.children is None or self._finished(n):
                    continue
                cnt += 1
                for cc in n.children:
                    if cc not in seen and rho.get(cc, 0.0) >= th:
                        seen.add(cc)
                        stack.append(cc)
            counts[root.moves[j]] = cnt
        return counts

    # -- the search -------------------------------------------------------------------------------

    def _reset_if_moved(self) -> None:
        key = (self.e.graph.root, self.e.s)
        if key != self._key:
            self.completed = {}
            self._key = key

    def search(self, *, nodes=None, seconds=None, stop=None, on_info=None, info_every: float = 1.0,
               depth: int | None = None):
        """Deepen until `nodes` more expansions, `seconds`, `stop`, depth `depth` completed, or
        the horizon falls below 1e-9 (a finite tree searched to its end)."""
        e = self.e
        t0 = time.perf_counter()
        start_exp = e.stats["expanded"]
        start_ev = e.stats["evaluations"]
        self._reset_if_moved()
        last_info = t0
        k = max(self.completed) + 1 if self.completed else 0
        self._prev = self.completed.get(k - 1)
        if on_info is not None:                         # provisional lines from what is known
            rho, _, _ = self.rho(theta(k))
            on_info(self.result(t0, start_exp, self.depth_result(k, theta(k), rho, False, t0), provisional=True))
        while True:
            th = theta(k)
            while True:
                rho, need, kids = self.rho(th)
                if kids:                                # no model calls; then the horizon grows from them
                    for s in range(0, len(kids), 256):  # in pieces: one database lookup for thousands of
                        e._materialize(kids[s : s + 256])   # positions took 2 GB (A+B4, depth 14)
                    continue
                if not need:
                    break
                if stop is not None and stop.is_set():
                    break
                if seconds is not None and time.perf_counter() - t0 >= seconds:
                    break
                if nodes is not None and e.stats["evaluations"] - start_ev >= nodes:
                    break
                for s in range(0, len(need), e.s.batch):
                    e._evaluate(need[s : s + e.s.batch])
                    if stop is not None and stop.is_set():
                        break
                    if on_info is not None and time.perf_counter() - last_info >= info_every:
                        r_, _, _ = self.rho(th)
                        on_info(self.result(t0, start_exp, self.depth_result(k, th, r_, False, t0)))
                        last_info = time.perf_counter()
                    if nodes is not None and e.stats["evaluations"] - start_ev >= nodes:
                        break
                    if seconds is not None and time.perf_counter() - t0 >= seconds:
                        break
            if need:                                    # stopped inside a depth
                d = self.depth_result(k, th, rho, False, t0)
                return self.result(t0, start_exp, d)
            d = self.depth_result(k, th, rho, True, t0)
            self.completed[k] = d
            if on_info is not None:
                on_info(self.result(t0, start_exp, d))
                last_info = time.perf_counter()
            k += 1
            self._prev = d
            if (stop is not None and stop.is_set()) or (seconds is not None and time.perf_counter() - t0 >= seconds) \
                    or (nodes is not None and e.stats["evaluations"] - start_ev >= nodes) \
                    or (depth is not None and k > depth) or theta(k) < 1e-9:
                return self.result(t0, start_exp, d)

    # -- output ----------------------------------------------------------------------------------

    def result(self, t0, start_exp, current: Depth, provisional: bool = False):
        from tnh.engine.search import Line, Result
        e, g = self.e, self.e.graph
        final = self.completed[max(self.completed)] if self.completed else None
        show = final if final is not None else current
        root = g.nodes[g.root]
        lines = []
        if show.root_values:
            ranked = sorted(show.root_values.items(), key=lambda t: -t[1]) if root.me_to_move else \
                sorted(show.root_values.items(), key=lambda t: -self._q(root)[root.moves.index(t[0])])
            for m, v in ranked[: e.s.lines]:
                lines.append(self._line(m, v, show, current))
        res = Result(len(g.nodes), e.stats["expanded"] - start_exp, time.perf_counter() - t0, None, lines,
                     dict(e.stats), {"stand_in": 0.0, "frontier": 0.0, "finished": 0.0}, list(e.history),
                     provisional or final is None)
        res.mode = "bestmove"
        res.depth = final.k if final is not None else -1
        res.complete = final is not None
        res.depth_in_progress = current.k
        res.root_value = show.value
        return res

    def _line(self, m: str, v: float, show: Depth, current: Depth):
        from tnh.engine.search import Line
        g = self.e.graph
        V = show.V
        root = g.nodes[g.root]
        moves, probs, flags, reach = [m], [], [], 1.0
        if root.me_to_move:
            probs.append(None); flags.append(True)
        else:
            p = float(self._q(root)[root.moves.index(m)])
            probs.append(p); flags.append(None); reach *= p
        i = root.children[root.moves.index(m)]
        while i in V and len(moves) < 40:
            n = g.nodes[i]
            if n.me_to_move:
                cv = show.mine.get(i)
                if cv:
                    j = max(cv, key=lambda j: (cv[j], -j))
                else:
                    j = max(self._candidates(i), key=lambda j: V.get(n.children[j], g.nodes[n.children[j]].value))
                probs.append(None); flags.append(True)
            else:
                q = self._q(n, [j for j, c in enumerate(n.children) if c in V] if self.e.s.his_anchor else None)
                j = int(np.argmax(q))
                probs.append(float(q[j])); flags.append(None); reach *= float(q[j])
            moves.append(n.moves[j])
            i = n.children[j]
        line = Line(moves, v, None, probs, flags, reach, 0)
        line.inside = show.inside.get(m, 0)
        line.provisional_value = current.root_values.get(m)
        return line


