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
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

import numpy as np

from tnh.builder.game import Node, model_type_dist


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


class BestMove:
    def __init__(self, engine) -> None:
        self.e = engine
        self.completed: dict[int, Depth] = {}       # for the current root and settings
        self._root = None
        self._key = None

    # -- the models' quantities ---------------------------------------------------------------

    def _evaluated(self, n: Node) -> bool:
        return n.children is not None or "result_mover" in n.info or bool(n.info.get("valued"))

    def _finished(self, n: Node) -> bool:
        return "result_mover" in n.info

    def _q(self, n: Node) -> np.ndarray:
        """His mix of database and model, without the prepared share."""
        from dataclasses import replace
        return model_type_dist(n, replace(self.e.params(), lam_node=0.0))

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

    def _factors(self, i: int, at_root: bool) -> list[tuple[int, float]]:
        """(child id, rho factor) for the children that matter at an inside position."""
        g, n = self.e.graph, self.e.graph.nodes[i]
        if n.me_to_move:
            prior = self.e._prior(n)
            cand = self._candidates(i)
            top = max(prior[j] for j in cand)
            a = self.e.s.alpha
            return [(n.children[j], 1.0 if at_root else (prior[j] / top) ** a) for j in cand]
        q = self._q(n)
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
            for c, f in self._factors(i, i == root):
                r = rho[i] * f
                if r > rho.get(c, -1.0):
                    rho[c] = r
                if not self._evaluated(g.nodes[c]):
                    need[c] = max(need.get(c, 0.0), r)       # children of inside positions are valued
                if c not in done:
                    heapq.heappush(heap, (g.nodes[c].ply, c))
        return rho, [i for i, _ in sorted(need.items(), key=lambda t: -t[1])], kids

    # -- values ----------------------------------------------------------------------------------

    def values(self, rho: dict[int, float], th: float) -> dict[int, float]:
        """One bottom-up pass over the inside set. Positions not evaluated yet (only while a
        depth is in progress) stand in with their parent's own value."""
        g, P = self.e.graph, self.e.s.prepared
        inside = [i for i, r in rho.items() if r >= th and self._evaluated(g.nodes[i]) and not self._finished(g.nodes[i])]
        inside.sort(key=lambda i: -g.nodes[i].ply)
        V: dict[int, float] = {}

        def val(c: int, parent: Node) -> float:
            if c in V:
                return V[c]
            cn = g.nodes[c]
            return cn.value if self._evaluated(cn) else parent.value

        for i in inside:
            n = g.nodes[i]
            if n.me_to_move:
                V[i] = max(val(n.children[j], n) for j in self._candidates(i))
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
        for c, _ in self._factors(g.root, True) if self._evaluated(root) and not self._finished(root) else []:
            j = root.children.index(c)
            cn = g.nodes[c]
            v = V.get(c, cn.value)
            rv[root.moves[j]] = v if root.me_to_move else 1 - v
        # positions inside each root move's horizon (by the way in with the largest rho)
        inside = self._inside_per_root_move(rho, th)
        v_root = V.get(g.root, root.value)
        return Depth(k, th, complete, rv, inside, v_root if root.me_to_move else 1 - v_root,
                     time.perf_counter() - t0, sum(1 for r in rho.values() if r >= th), pass_s, V)

    def _inside_per_root_move(self, rho, th) -> dict[str, int]:
        g = self.e.graph
        root = g.nodes[g.root]
        if not self._evaluated(root) or self._finished(root):
            return {}
        counts = {}
        for j, c in enumerate(root.children):
            if rho.get(c, 0.0) < th:
                continue
            seen, stack, cnt = {c}, [c], 0
            while stack:
                i = stack.pop()
                n = g.nodes[i]
                if rho.get(i, 0.0) < th or not self._evaluated(n) or self._finished(n):
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
        if on_info is not None:                         # provisional lines from what is known
            rho, _, _ = self.rho(theta(k))
            on_info(self.result(t0, start_exp, self.depth_result(k, theta(k), rho, False, t0), provisional=True))
        while True:
            th = theta(k)
            while True:
                rho, need, kids = self.rho(th)
                if kids:
                    e._materialize(kids)                # no model calls; then the horizon grows from them
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
                cand = self._candidates(i)
                j = max(cand, key=lambda j: V.get(n.children[j], g.nodes[n.children[j]].value))
                probs.append(None); flags.append(True)
            else:
                q = self._q(n)
                j = int(np.argmax(q))
                probs.append(float(q[j])); flags.append(None); reach *= float(q[j])
            moves.append(n.moves[j])
            i = n.children[j]
        line = Line(moves, v, None, probs, flags, reach, 0)
        line.inside = show.inside.get(m, 0)
        line.provisional_value = current.root_values.get(m)
        return line


