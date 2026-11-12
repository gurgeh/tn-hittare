"""The repertoire search: a selective search that runs as long as it is given and keeps
its graph between positions.

The game is the builder's (`tnh.builder.game`): at the opponent's moves a chance node
with his database counts laid over the move model at his rating; at mine either a
learned move or the move model at my rating; leaves valued by the result model. What
is learned is decided by a price per non-obvious learned move (`tnh.engine.pricing`).

**Growth.** One evaluation per expanded position (moves, the mover's move model, Lc0's
raw policy where the evaluator has it, the position's own value). Leaves are chosen
from the root in batches:

- at my positions, PUCT over my candidates: Q is the child's solved value for me, the
  prior P is the move model at my rating blended with Lc0's raw policy
  (`policy_weight`), U = c_puct * P * sqrt(N) / (1 + N_a); an unvisited candidate gets
  Q = min(best visited Q, the position's own value) - fpu, so an unexplored move never
  looks better than the best explored one;
- at the opponent's positions, the move whose share of the visits lags its probability
  most (`q_a * (N + 1) - N_a`), so visits follow his distribution.

A batch stops at the first collision (a leaf already in the batch). Unexpanded children
stand in with their parent's own value.

**Values.** Not backed up along paths: every `solve_every` expansions the whole graph is
solved exactly (the price solve), and selection reads those values until the next
solve. A new leaf's Q until then is its own value.

**Soundness without a hard limit.** A move can be learned when its child has been
expanded and it passes a safety net that removes plain blunders without valuing every
child first: its prior must be at least `prior_min` (a move that neither the move model
nor Lc0's policy gives any weight is not a candidate), and once valued its child's own
value must not be more than `blunder` below the position's own value. Beyond that the
result model decides, with `prepared` (node-wise share of the opponent's probability
on his best reply, the builder's `lam_node`): his best reply is the expanded child whose
own value is worst for me (the most likely move while none is expanded).

**Settings and the cache.** Evaluations are cached on (position, mover's rating,
opponent's rating, tempo) and database counts on (position, mover's rating, filter).
The graph's structure (positions and moves) does not depend on settings. A change of
ratings, side or tempo leaves the structure but turns every position without a cached
evaluation under the new key back into a frontier, and resets the visit counts; a change
of price, cap, threshold, prepared share, kappa or filter only re-solves (the filter
refetches counts).
"""

from __future__ import annotations

import math
import threading
import time
from collections import Counter
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field, fields, replace
from typing import Any, Protocol

import numpy as np

from tnh.builder.game import Game, Graph, Node, Params, blend
from tnh.engine.pricing import PL, Priced, solve_cap, solve_price, w_at


def identity_update(node: Node, first: np.ndarray, facts: dict) -> np.ndarray:
    """The move-probability update hook (DESIGN, "A fitted update of the move probabilities
    from the search below"): `(node, first probabilities, per-move search facts) ->
    probabilities`. Identity until the fitted model exists."""
    return first


@dataclass(frozen=True)
class FirstObvious:
    """The cost rule on the move model's first probabilities (not on the updated ones):
    0 when my model gives the move at least `threshold`, else 1."""

    threshold: float = 0.85

    def __call__(self, node: Node, j: int) -> int:
        return 0 if node.info["p_first"][j] >= self.threshold else 1


class StubFix:
    """Until `tnh.movefix` exists: the shares measured on real games by rating, flat in ply."""

    def share(self, elo, ply) -> float:
        return float(np.interp(elo, [2000.0, 2200.0, 2400.0], [0.15, 0.30, 0.35]))

    def apply(self, p, best_index, elo, ply) -> np.ndarray:
        s = self.share(elo, ply)
        out = (1 - s) * np.asarray(p, float)
        out[best_index] += s
        return out


@dataclass(frozen=True)
class Settings:
    my_elo: int = 2200
    opp_elo: int = 2200
    my_side: str = "to_move"          # "white", "black" or "to_move" (the side to move at the root)
    tempo: str = "classical"
    price: float = 0.001              # per non-obvious learned move, in expected score
    cap: int = 0                      # at most this many non-obvious learned moves (0 = off)
    obvious: float = 0.85             # a move is obvious when my move model gives it at least this
    prepared: float = 0.0             # share of the opponent's probability on his best reply, per position
    draw_score: float = 0.5           # what a draw is worth to me (0.5 = expected score; the opponent gets 1 minus it)
    mistake_fix: str = "fitted"       # the move model's mistake correction (tnh.movefix): "fitted" or "off"
    mix: str = "fitted"               # the opponent's database mix: "fitted" (tnh.movemix) or "kappa"
    kappa: float = 100.0              # with mix="kappa": the model's weight in games
    db_filter: Hashable = "auto"      # passed to the evaluator's counts; "auto" = the tempo's own
    lines: int = 3
    mode: str = "repertoire"          # "repertoire" (priced, PUCT) or "bestmove" (expectimax to a reach horizon)
    alpha: float = 1.0                # bestmove: my candidates' horizon weight (P_a / max P) ** alpha
    # bestmove selectivity, see tnh.engine.bestmove (root candidates are never widened or reduced)
    his_anchor: bool = False          # A (experimental): his unsearched replies stay at the position's static value
    widen: float = 4.0                # B: beta; my candidate a only if rho * w_a >= theta / beta (0 = all)
    reduce_margin: float = 0.03       # C: a candidate this far below the best at the previous depth ... (0 = off)
    reduce_penalty: float = 0.5       # C: ... gets its horizon weight times this
    c_puct: float = 1.5
    fpu: float = 0.05
    policy_weight: float = 0.5        # Lc0's raw policy in my prior
    prior_min: float = 0.002          # safety net: a candidate needs at least this prior
    explore_uniform: float = 0.0      # share of a uniform prior mixed into PUCT's (tests: reach every move)
    blunder: float = 0.30             # safety net: a learned move's child at most this far below the position
    batch: int = 12
    solve_every: int = 120            # at least this many expansions between full solves
    solve_ratio: float = 4.0          # and expansions since the last solve >= this x its estimated cost
    solve_cost: float = 0.007         # a solve's cost per position in its graph, in expansions
    expand_root: bool = True          # every legal move at the root is expanded once

    EVAL_KEYS = ("my_elo", "opp_elo", "my_side", "tempo")
    FIELD_KEYS = ("db_filter", "draw_score")  # change node fields (counts, values) without new evaluations
    TEMPI = ("classical", "blitz", "fake")


@dataclass
class Eval:
    moves: list[str]
    p: np.ndarray                 # the move model at the mover's rating, over `moves`
    policy: np.ndarray | None     # Lc0's raw policy, or None
    value: float                  # the mover's expected score (the result model): W + D / 2
    wdl: tuple[float, float, float] | None = None   # the mover's win, draw, loss, where the model gives them


class Evaluator(Protocol):
    """Positions and models. `pos` is whatever the evaluator uses for a position."""

    def start(self) -> Any: ...
    def key(self, pos) -> Hashable: ...
    def ply(self, pos) -> int: ...
    def white_to_move(self, pos) -> bool: ...
    def moves(self, pos) -> list[str]: ...
    def child(self, pos, move: str) -> Any: ...
    def terminal(self, pos) -> float | None: ...           # the mover's score if the game is over
    def evaluate(self, items: list[tuple[Any, int, int]], tempo: str) -> list[Eval]: ...
    def counts(self, pos, moves: list[str], mover_elo: int, db_filter) -> np.ndarray | None: ...


@dataclass
class Line:
    moves: list[str]
    value: float                  # expected score for the side to move at the root
    net: float | None             # with me to move: value minus the price of the line's learned moves
    probs: list[float | None]     # per move: the opponent's probability, None for mine
    learned: list[bool | None]    # per move of mine: learned (True) or just likely (False)
    reach: float                  # probability of reaching the end of the line
    nonobvious: int               # non-obvious learned moves in the line


@dataclass
class Result:
    nodes: int
    expansions: int
    seconds: float
    solve: Priced | None
    lines: list[Line]
    stats: dict
    mass: dict                    # where the root value rests: stand-ins, frontier values, finished games
    history: list                 # (expanded positions, root value, root net) at every solve
    provisional: bool = False     # lines from the last solve at another root or other settings


PENDING = ("?",)        # a position's moves before it is first expanded (not read yet)


class Engine:
    def __init__(self, evaluator: Evaluator, settings: Settings | None = None,
                 update: Callable[[Node, np.ndarray, dict], np.ndarray] = identity_update) -> None:
        self.ev = evaluator
        self.s = settings or Settings()
        self.update = update
        self.stats: Counter = Counter()
        self.seconds: Counter = Counter()
        self._evals: dict[tuple, Eval] = {}
        self._counts: dict[tuple, np.ndarray | None] = {}
        self.clear()

    # -- graph and position --------------------------------------------------------------

    def clear(self) -> None:
        self.graph = Graph(me_white=True)
        self.pos: dict[int, Any] = {}
        self.kids: dict[int, list[int]] = {}
        self.visits: Counter = Counter()
        self.V: dict[int, float] = {}
        self.last: Priced | None = None
        self.game: Game | None = None
        self.since_solve = 0
        self.root_moves: list[str] = []
        self.history: list[tuple[int, float, float]] = []
        self.timing: dict = {}
        self.provisional = False
        self.root_results: dict[int, Result] = {}   # the last exact result per root, for provisional lines
        self.transposed: list = []              # (first path's position, another path's) for the history check
        self.eval_since_solve = 0.0
        self.last_solve_seconds = 0.0
        self.last_solve_end = 0.0
        self.cap_mu: float | None = None
        self._node(self.ev.start(), None)
        self.graph.root = 0
        self._refresh()

    def _node(self, pos, parent_value: float | None) -> Node:
        k, ply = self.ev.key(pos), self.ev.ply(pos)
        hit = self.graph.get(k, ply)
        if hit is not None:
            self.stats["transpositions merged"] += 1
            if len(self.transposed) < 400 and self.pos[hit.id] is not pos:
                self.transposed.append((self.pos[hit.id], pos))
            return hit
        n = self.graph.add(k, ply, self.ev.white_to_move(pos) == self.graph.me_white, [], value=math.nan)
        n.moves = PENDING
        self.pos[n.id] = pos
        return n

    def set_position(self, moves: list[str]) -> None:
        """The root after `moves` from the start position. Positions already in the graph are
        reused with what was learned about them."""
        t = time.perf_counter()
        self._set_position(moves)
        self.timing = {"set_position": time.perf_counter() - t}

    def _set_position(self, moves: list[str]) -> None:
        p = self.ev.start()
        node = self.graph.nodes[0]
        for m in moves:
            p = self.ev.child(p, m)
            node = self._node(p, None)
        flips = self.s.my_side == "to_move" and self.ev.white_to_move(self.pos[node.id]) != self.graph.me_white
        self.graph.root = node.id
        self.root_moves = list(moves)
        if flips:
            self._refresh()                   # with Side = to move, a new side to move changes who I am
            return
        # Only the root moves: every node keeps its fields and cached evaluations. The last
        # solve was for another root; it stays as a provisional source of lines until the
        # first solve at this root.
        if math.isnan(node.value):
            node.value = 0.5
            node.info["stand_in"] = True
        self._stale()

    # -- settings -------------------------------------------------------------------------------

    def set_settings(self, **changes) -> dict:
        """Change settings; says what had to be dropped."""
        old = self.s
        new = replace(old, **changes)
        self.s = new
        evals_changed = any(getattr(old, k) != getattr(new, k) for k in Settings.EVAL_KEYS)
        if old.cap != new.cap:
            self.cap_mu = None
        if evals_changed:
            self.visits.clear()
        if evals_changed or any(getattr(old, k) != getattr(new, k) for k in Settings.FIELD_KEYS):
            self._refresh()
            self.stats["refreshes"] += 1
        elif old != new:
            self._stale()                     # a re-solve is enough
        lost = sum(1 for i in self.graph.reachable() if i in self.kids and self.graph.nodes[i].children is None)
        return {"evaluations_keyed_anew": evals_changed, "positions_back_to_frontier": lost}

    def _my_white(self) -> bool:
        if self.s.my_side == "white":
            return True
        if self.s.my_side == "black":
            return False
        return self.ev.white_to_move(self.pos[self.graph.root])

    def _elos(self, pos) -> tuple[int, int]:
        mine = self.ev.white_to_move(pos) == self._my_white()
        return (self.s.my_elo, self.s.opp_elo) if mine else (self.s.opp_elo, self.s.my_elo)

    def _ekey(self, i: int) -> tuple:
        pos = self.pos[i]
        return (self.graph.nodes[i].key, *self._elos(pos), self.s.tempo)

    def _refresh(self) -> None:
        """Fill every position's fields from the caches under the current settings."""
        g = self.graph
        g.me_white = self._my_white()
        need = []
        for n in g.nodes:
            n.me_to_move = self.ev.white_to_move(self.pos[n.id]) == g.me_white
            if n.moves is not PENDING and n.moves and not n.me_to_move:
                e = self._evals.get(self._ekey(n.id))
                if e is not None:
                    need.append(n)
        self._fetch_counts(need)
        for n in g.nodes:
            n.engine, n.counts = None, None
            n.info.pop("stand_in", None)
            n.info.pop("valued", None)
            if "result_mover" in n.info:
                n.value = self._result_value(n)
                n.children, n.p_model = None, None
                continue
            e = None if n.moves is PENDING else self._evals.get(self._ekey(n.id))
            if e is None:
                n.children, n.p_model = None, None
                n.value = math.nan
                continue
            self._apply(n, e)
            n.info["valued"] = True
            n.children = self.kids.get(n.id)
        # stand-ins: a position without its own value takes its parent's
        for i in g.order():
            n = g.nodes[i]
            for c in self.kids.get(i, ()) if n.children is not None else ():
                cn = g.nodes[c]
                if math.isnan(cn.value):
                    cn.value = n.value
                    cn.info["stand_in"] = True
        root = g.nodes[g.root]
        if math.isnan(root.value):
            root.value = 0.5
            root.info["stand_in"] = True
        self.V = {}
        self.last = None
        self.game = None
        self.since_solve = 10 ** 9

    def _stale(self) -> None:
        """The last solve no longer answers for the current root or settings: keep it for
        provisional lines and solve at the next search."""
        self.provisional = self.last is not None
        self.since_solve = 10 ** 9

    def _filter(self) -> Hashable:
        if self.s.db_filter != "auto":
            return self.s.db_filter
        return "blitz" if self.s.tempo == "blitz" else "classical"

    def _my_value(self, n: Node, e: Eval) -> float:
        """My value of a position from its evaluation: my W + draw_score * D. At 0.5 it is the
        expected score computed exactly as before (bit for bit)."""
        ds = self.s.draw_score
        if ds == 0.5 or e.wdl is None:
            return e.value if n.me_to_move else 1.0 - e.value
        w, d, l = e.wdl
        mine_w = w if n.me_to_move else l
        return float(mine_w + ds * d)

    def _result_value(self, n: Node) -> float:
        t = n.info["result_mover"]
        if t == 0.5:
            return 0.5 if self.s.draw_score == 0.5 else self.s.draw_score
        return t if n.me_to_move else 1.0 - t

    def _ckey(self, n: Node) -> tuple:
        return (n.key, self._elos(self.pos[n.id])[0], self._filter())

    def _fetch_counts(self, nodes: list[Node]) -> None:
        """Database counts for the opponent's positions, in one call where the evaluator can."""
        todo = [n for n in nodes if not n.me_to_move and self._ckey(n) not in self._counts]
        if not todo:
            return
        t = time.perf_counter()
        items = [(self.pos[n.id], list(n.moves), self._elos(self.pos[n.id])[0]) for n in todo]
        many = getattr(self.ev, "counts_many", None)
        if many is not None:
            out = many(items, self._filter())
        else:
            out = [self.ev.counts(p, ms, e, self._filter()) for p, ms, e in items]
        for n, c in zip(todo, out):
            self._counts[self._ckey(n)] = c
        self.stats["count lookups"] += len(todo)
        self.stats["count calls"] += 1
        self.seconds["counts"] += time.perf_counter() - t

    def _apply(self, n: Node, e: Eval) -> None:
        idx = {m: j for j, m in enumerate(e.moves)}
        if set(idx) != set(n.moves):
            raise ValueError(f"evaluator's moves differ from the position's at {n.key}")
        order = [idx[m] for m in n.moves]
        p = np.asarray(e.p, dtype=np.float64)[order]
        n.info["p_first"] = p / p.sum()
        n.p_model = n.info["p_first"]
        n.info["policy"] = None if e.policy is None else np.asarray(e.policy, dtype=np.float64)[order]
        n.value = self._my_value(n, e)
        if not n.me_to_move:
            n.counts = self._counts.get(self._ckey(n))

    # -- expansion ------------------------------------------------------------------------------

    def _expand(self, ids: list[int]) -> None:
        """Evaluate the positions and create their children (the priced mode's expansion)."""
        self._materialize(self._evaluate(ids))

    def _evaluate(self, ids: list[int]) -> list[int]:
        """The model calls: each position's moves, its move model and its own value, marked
        `valued`. Its children are not created (see `_materialize`), so a position that is only
        needed for its value costs one graph position, not one per legal move. Returns the
        live (not finished) positions."""
        t0 = time.perf_counter()
        g = self.graph
        live = []
        for i in ids:
            n = g.nodes[i]
            if n.moves is PENDING:
                pos = self.pos[i]
                t = self.ev.terminal(pos)
                if t is not None:
                    n.info["result_mover"] = t
                    n.moves = []
                    n.value = self._result_value(n)
                    n.children = None
                    n.info.pop("stand_in", None)
                    self.stats["finished games"] += 1
                    continue
                n.moves = self.ev.moves(pos)
            if not n.moves:
                continue
            live.append(i)
        todo, seen = [], set()
        for i in live:
            k = self._ekey(i)
            if k in self._evals or k in seen:
                self.stats["evaluation cache hits"] += 1
                continue
            seen.add(k)
            todo.append((i, k))
        self.seconds["expand"] += time.perf_counter() - t0
        if todo:
            t = time.perf_counter()
            out = self.ev.evaluate([(self.pos[i], *self._elos(self.pos[i])) for i, _ in todo], self.s.tempo)
            dt = time.perf_counter() - t
            self.seconds["evaluate"] += dt
            self.eval_since_solve += dt
            self.stats["evaluations"] += len(todo)
            for (i, k), e in zip(todo, out):
                self._evals[k] = e
        for i in live:
            n = g.nodes[i]
            if n.children is None:
                n.info.pop("stand_in", None)
                self._apply(n, self._evals[self._ekey(i)])
                n.info["valued"] = True
        return live

    def _materialize(self, ids: list[int]) -> None:
        """Counts and children for evaluated positions; no model calls."""
        g = self.graph
        todo = [i for i in ids if g.nodes[i].children is None and g.nodes[i].moves and g.nodes[i].moves is not PENDING]
        self._fetch_counts([g.nodes[i] for i in todo])
        t0 = time.perf_counter()
        for i in todo:
            n = g.nodes[i]
            if not n.me_to_move:
                n.counts = self._counts.get(self._ckey(n))
            if i not in self.kids:
                self.kids[i] = [self._node(self.ev.child(self.pos[i], m), n.value).id for m in n.moves]
            for c in self.kids[i]:
                cn = g.nodes[c]
                if cn.children is None and "result_mover" not in cn.info and not cn.info.get("valued") \
                        and (math.isnan(cn.value) or cn.info.get("stand_in")):
                    cn.me_to_move = self.ev.white_to_move(self.pos[c]) == g.me_white
                    cn.value = n.value
                    cn.info["stand_in"] = True
            n.children = self.kids[i]
            self.stats["expanded"] += 1
        self.seconds["expand"] += time.perf_counter() - t0

    # -- the solve ------------------------------------------------------------------------------

    def move_facts(self, i: int) -> dict:
        """What the search has found under each move of a position: the input of the update."""
        g = self.graph
        n = g.nodes[i]
        kids = [g.nodes[c] for c in n.children]
        val = np.array([self.V.get(c.id, c.value) for c in kids])
        mover = val if n.me_to_move else 1.0 - val
        return {"value_mover": mover, "best_minus": mover.max() - mover,
                "stand_in": np.array([bool(c.info.get("stand_in")) for c in kids]),
                "finished": np.array([not c.moves for c in kids]),
                "visits": np.array([self.visits[c.id] for c in kids]),
                "mover_elo": self._elos(self.pos[i])[0], "opp_elo": self._elos(self.pos[i])[1]}

    # -- the mistake correction (tnh.movefix) ----------------------------------------------------

    def fix_rule(self):
        """The fitted rule, or None when off. Without `tnh.movefix` a stub with the shares the
        result developer measured (0.15 at 2000, 0.30 at 2200, 0.35 at 2400), named as such."""
        if self.s.mistake_fix == "off":
            return None
        if self.s.mistake_fix != "fitted":
            raise ValueError(f"mistake_fix must be fitted or off, got {self.s.mistake_fix!r}")
        if getattr(self, "_fix", None) is None:
            try:
                from tnh import movefix
                self._fix, self.fix_source = movefix.load(), "fitted"
            except ImportError:
                self._fix, self.fix_source = StubFix(), "stub"
        return self._fix

    def mover_value(self, parent: Node, child: Node) -> float:
        return child.value if parent.me_to_move else 1.0 - child.value

    def fixed(self, i: int, best: int | None) -> np.ndarray:
        """The node's first probabilities with the mistake correction: a share moved onto its
        best move (`best`; the most likely move when None)."""
        n = self.graph.nodes[i]
        first = n.info["p_first"]
        rule = self.fix_rule()
        if rule is None:
            return first
        b = int(np.argmax(first)) if best is None else best
        p = np.asarray(rule.apply(first, b, self._elos(self.pos[i])[0], n.ply), dtype=np.float64)
        return p / p.sum()

    def best_valued(self, i: int) -> int | None:
        """The child with the highest static value for the mover among the children that have
        a value of their own (None while none has)."""
        g = self.graph
        n = g.nodes[i]
        best, bj = -1.0, None
        for j, c in enumerate(n.children):
            cn = g.nodes[c]
            if cn.info.get("valued") or "result_mover" in cn.info or cn.children is not None:
                v = self.mover_value(n, cn)
                if v > best:
                    best, bj = v, j
        return bj

    def fix_share_root(self) -> float:
        """The share in use at the root's ply for the opponent's rating (for the info line)."""
        rule = self.fix_rule()
        if rule is None:
            return 0.0
        return float(rule.share(self.s.opp_elo, self.graph.nodes[self.graph.root].ply))

    def _prior(self, n: Node) -> np.ndarray:
        p = n.info.get("p_first", n.p_model)
        pol = n.info.get("policy")
        if pol is not None and self.s.policy_weight > 0:
            w = self.s.policy_weight
            p = (1 - w) * p + w * pol / pol.sum()
        if self.s.explore_uniform > 0:
            p = (1 - self.s.explore_uniform) * p + self.s.explore_uniform / len(p)
        return p

    def _prepare(self) -> None:
        """The per-position fields the exact game reads: which of my moves can be learned, and
        the opponent's best reply for the prepared share."""
        g = self.graph
        for i in self.kids:                   # the expanded positions (fields elsewhere are not read)
            n = g.nodes[i]
            if n.children is None or not n.moves:
                continue
            # the mistake correction, then the update hook: mix(counts, update(fix(first))) at
            # his positions, and the same correction where I play by my model
            p = self.fixed(i, self.best_valued(i))
            if self.update is not identity_update:
                p = np.asarray(self.update(n, p, self.move_facts(i)), dtype=np.float64)
                p = p / p.sum()
            n.p_model = p
            if n.me_to_move:
                prior = self._prior(n)
                eng = np.full(len(n.moves), np.nan)
                for j, c in enumerate(n.children):
                    cn = g.nodes[c]
                    if cn.info.get("stand_in"):
                        continue                      # not valued yet: cannot be learned
                    if prior[j] < self.s.prior_min:
                        continue
                    if cn.value < n.value - self.s.blunder:
                        continue
                    eng[j] = 0.5                      # any number: every candidate within eps
                n.engine = eng
            else:
                eng = np.full(len(n.moves), np.nan)
                vals = [(g.nodes[c].value, j) for j, c in enumerate(n.children) if not g.nodes[c].info.get("stand_in")]
                if vals:
                    _, jb = min(vals)
                else:
                    jb = int(np.argmax(n.p_model))
                eng[jb] = 1.0                         # his best reply, by the result model
                n.engine = eng

    def params(self) -> Params:
        return Params(kappa=self.s.kappa, eps_white=1.0, eps_black=1.0, lam_node=self.s.prepared,
                      cost=FirstObvious(self.s.obvious), mix=self._mix())

    def _mix(self):
        if self.s.mix == "kappa":
            return None
        if self.s.mix != "fitted":
            raise ValueError(f"mix must be 'fitted' or 'kappa', got {self.s.mix!r}")
        if getattr(self, "_rule", None) is None:
            from tnh import movemix
            self._rule = movemix.load()
        return self._rule.mix

    def solve(self, final: bool = True) -> Priced:
        """The exact solve of the current graph. With a cap, `final` bisects for the price
        (warm-started at the last one); otherwise the last price is reused."""
        t = time.perf_counter()
        self._prepare()
        game = Game(self.graph, self.params())
        if self.s.cap > 0:
            if final or self.cap_mu is None:
                found = solve_cap(game, self.s.cap, hint=self.cap_mu)
                self.cap_mu = found.mu
                self.stats["cap bisections"] += 1
            res = solve_price(game, self.cap_mu, keep_W=True)
        else:
            res = solve_price(game, self.s.price, keep_W=True)
        Vs = game.values(res.learned)
        self.V = {i: float(Vs[i]) for i in game.order}
        self.R = game.reach(res.learned)
        self.history.append((self.stats["expanded"], res.value, res.net))
        self.game, self.last = game, res
        self.provisional = False
        self.since_solve = 0
        self.eval_since_solve = 0.0
        self.last_solve_end = time.perf_counter()
        self.last_solve_seconds = self.last_solve_end - t
        self.seconds["solve"] += self.last_solve_seconds
        self.stats["solves"] += 1
        return res

    def _due(self) -> bool:
        """Solve when at least `solve_every` expansions have been made and at least
        `solve_ratio` times the last solve's estimated cost, in expansion-equivalents
        (`solve_cost` per position in its graph: on the real models a solve is about 20 s per
        million positions and an expansion about 3 ms). Solving stays about
        1 / (1 + solve_ratio) of the time however large the graph grows, and the schedule is
        a function of the search's own counts, not of the clock, so two identical searches
        are identical whatever else runs on the machine."""
        if self.since_solve < self.s.solve_every:
            return False
        size = len(self.game.order) if self.game is not None else 0
        return self.since_solve >= self.s.solve_ratio * self.s.solve_cost * size

    # -- selection ------------------------------------------------------------------------------

    def _q(self, i: int) -> float:
        return self.V.get(i, self.graph.nodes[i].value)

    def _select(self, pending: Counter) -> list[int] | None:
        g = self.graph
        i = g.root
        path = [i]
        while True:
            n = g.nodes[i]
            if n.children is None or not n.moves:
                return path
            Nv = self.visits[i] + pending[i]
            if n.me_to_move:
                prior = self._prior(n)
                cand = [j for j in range(len(n.moves)) if prior[j] >= self.s.prior_min] or [int(np.argmax(prior))]
                visited = [self._q(n.children[j]) for j in cand if self.visits[n.children[j]] + pending[n.children[j]] > 0]
                fpu = min(max(visited), n.value) - self.s.fpu if visited else n.value - self.s.fpu
                best, bj = -math.inf, cand[0]
                sq = math.sqrt(Nv + 1)
                psum = prior[cand].sum()
                for j in cand:
                    c = n.children[j]
                    Na = self.visits[c] + pending[c]
                    q = self._q(c) if Na > 0 else fpu
                    u = self.s.c_puct * prior[j] / psum * sq / (1 + Na)
                    if q + u > best + 1e-15:
                        best, bj = q + u, j
                i = n.children[bj]
            else:
                if self.game is not None and i in self.game.dist:
                    q = self.game.dist[i]
                else:
                    from tnh.builder.game import model_type_dist
                    # not solved yet: his mix without the prepared share (no best reply known)
                    q = model_type_dist(n, replace(self.params(), lam_node=0.0))
                lag = [q[j] * (Nv + 1) - (self.visits[c] + pending[c]) for j, c in enumerate(n.children)]
                i = n.children[int(np.argmax(lag))]
            if i in path:
                return None
            path.append(i)

    # -- the loop -------------------------------------------------------------------------------

    def search(self, *, nodes: int | None = None, seconds: float | None = None,
               stop: threading.Event | None = None, on_info: Callable[[Result], None] | None = None,
               info_every: float = 1.0, depth: int | None = None) -> Result:
        """Search until `nodes` more expansions, `seconds`, or `stop` is set."""
        if self.s.mode == "bestmove":
            if getattr(self, "bm", None) is None:
                from tnh.engine.bestmove import BestMove
                self.bm = BestMove(self)
            return self.bm.search(nodes=nodes, seconds=seconds, stop=stop, on_info=on_info, info_every=info_every,
                                  depth=depth)
        if self.s.mode != "repertoire":
            raise ValueError(f"mode must be repertoire or bestmove, got {self.s.mode!r}")
        t0 = time.perf_counter()
        start_exp = self.stats["expanded"]
        last_info = t0
        stalled = 0
        tm = self.timing = getattr(self, "timing", {})
        if on_info is not None and self.provisional:
            # lines at once, marked provisional: from the last solve if it covered this root
            # (at my position a move counts only once it has been expanded), else the last
            # result this root had (stepping back to a position searched before)
            r = None
            if self.game is not None and self.graph.root in self.game.dist:
                r = self.result(t0, start_exp)
            if (r is None or not r.lines) and self.graph.root in self.root_results:
                r = replace(self.root_results[self.graph.root], provisional=True, expansions=0, seconds=0.0)
            if r is not None and r.lines:
                on_info(r)
                tm["provisional_info"] = time.perf_counter() - t0
        if self.graph.nodes[self.graph.root].children is None:
            self._expand([self.graph.root])
        self._expand_root()
        tm["root_expansion"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        if self.last is None or self.provisional:
            self.solve(final=False)
        tm["first_solve"] = time.perf_counter() - t1
        if on_info is not None:
            on_info(self.result(t0, start_exp))   # the first exact lines, without waiting a second
            last_info = time.perf_counter()
        while True:
            done = self.stats["expanded"] - start_exp
            if nodes is not None and done >= nodes:
                break
            if seconds is not None and time.perf_counter() - t0 >= seconds:
                break
            if stop is not None and stop.is_set():
                break
            pending: Counter = Counter()
            leaves, paths, ended = [], [], 0
            want = self.s.batch if nodes is None else min(self.s.batch, nodes - done)
            ts = time.perf_counter()
            while len(leaves) < want and ended <= 4 * self.s.batch:
                path = self._select(pending)
                if path is None:
                    self.stats["collisions"] += 1
                    break
                end = self.graph.nodes[path[-1]]
                if not end.moves:
                    # a finished game: count the visit so selection moves on
                    ended += 1
                    self.stats["finished games visited"] += 1
                    for i in path:
                        pending[i] += 1
                    paths.append(path)
                    continue
                if path[-1] in leaves:
                    self.stats["collisions"] += 1
                    break
                leaves.append(path[-1])
                paths.append(path)
                for i in path:
                    pending[i] += 1
            self.seconds["select"] += time.perf_counter() - ts
            if not leaves:
                for path in paths:
                    for i in path:
                        self.visits[i] += 1
                stalled += 1
                if self.since_solve > 0 and (self.since_solve >= self.s.batch or stalled >= 10):
                    self.solve(final=False)   # selection must not run long on stale values
                if stalled >= 1000:
                    # selection keeps ending in finished games or collisions: what is left
                    # unexpanded has no weight in the descent
                    self.stats["stopped: nothing left to expand"] += 1
                    break
                continue
            stalled = 0
            self._expand(leaves)
            for path in paths:
                for i in path:
                    self.visits[i] += 1
            self.since_solve += len(leaves)
            if self._due():
                self.solve(final=False)
            now = time.perf_counter()
            if on_info is not None and now - last_info >= info_every:
                # an info line re-solves only when enough has been evaluated that the solve
                # stays at most about half the time; otherwise it reports the last solve
                if self.since_solve > 0 and self.eval_since_solve >= self.last_solve_seconds:
                    self.solve()
                on_info(self.result(t0, start_exp))
                last_info = time.perf_counter()
        if not (stop is not None and stop.is_set() and self.last is not None):
            self.solve()              # a stop answers at once with the last solve
        r = self.result(t0, start_exp)
        if not r.provisional:
            self.root_results[self.graph.root] = r
            while len(self.root_results) > 64:
                self.root_results.pop(next(iter(self.root_results)))
        return r

    def _expand_root(self) -> None:
        """Every legal move at the root once, so that each has a value of its own."""
        g = self.graph
        root = g.nodes[g.root]
        if not self.s.expand_root or root.children is None:
            return
        todo = [c for c in root.children if g.nodes[c].children is None and "result_mover" not in g.nodes[c].info]
        if todo:
            self._expand(todo)
            self.since_solve += len(todo)

    def _fully_expanded(self) -> bool:
        g = self.graph
        return all(g.nodes[i].children is not None or not g.nodes[i].moves for i in g.reachable())

    def mass(self) -> dict:
        """Probability mass under the current repertoire on what the root value rests on."""
        g, R = self.graph, self.R
        out = {"stand_in": 0.0, "frontier": 0.0, "finished": 0.0}
        for i in self.game.order:
            n = g.nodes[i]
            if n.children is not None and n.moves:
                continue
            if not n.moves:
                out["finished"] += float(R[i])
            elif n.info.get("stand_in"):
                out["stand_in"] += float(R[i])
            else:
                out["frontier"] += float(R[i])
        return out

    def result(self, t0: float, start_exp: int) -> Result:
        return Result(len(self.graph.nodes), self.stats["expanded"] - start_exp, time.perf_counter() - t0,
                      self.last, self.lines(self.s.lines), dict(self.stats), self.mass(), list(self.history),
                      self.provisional)

    # -- lines ----------------------------------------------------------------------------------

    def lines(self, k: int) -> list[Line]:
        if self.last is None:
            self.solve()
        g, game, res = self.graph, self.game, self.last
        root = g.nodes[g.root]
        if root.children is None or not root.moves:
            return []
        out = []
        if root.me_to_move:
            W = res.W or {}
            cands = []
            for j, c in enumerate(root.children):
                if g.nodes[c].info.get("stand_in"):
                    continue
                cost = game.move_cost(g.root, j)
                net = (w_at(W[c], 1.0) if c in W else self.V.get(c, g.nodes[c].value)) - res.mu * cost
                cands.append((net, j))
            for net, j in sorted(cands, key=lambda t: (-t[0], t[1]))[:k]:
                out.append(self._line(j, net))
        else:
            q = game.dist[g.root]
            for j in np.argsort(-q, kind="stable")[:k]:
                out.append(self._line(int(j), None))
        return out

    def _line(self, j0: int, net: float | None) -> Line:
        g, game, res = self.graph, self.game, self.last
        root = g.nodes[g.root]
        moves, probs, learned_flags = [], [], []
        i, j = g.root, j0
        reach, nonobv = 1.0, 0
        seen = set()
        while True:
            n = g.nodes[i]
            m = n.moves[j]
            moves.append(m)
            if n.me_to_move:
                is_l = res.learned.get(n.key) == m
                learned_flags.append(is_l)
                probs.append(None)
                if is_l and game.move_cost(i, j) > 0:
                    nonobv += 1
                reach *= 1.0 if is_l else float(game.dist[i][j]) if i != g.root else 1.0
            else:
                learned_flags.append(None)
                pr = float(game.dist[i][j])
                probs.append(pr)
                reach *= pr
            i = n.children[j]
            seen.add(i)
            cn = g.nodes[i]
            if cn.children is None or not cn.moves or cn.info.get("stand_in") or len(moves) > 40 \
                    or i not in game.dist:            # expanded after the last solve: the line ends here
                break
            if cn.me_to_move:
                lm = res.learned.get(cn.key)
                j = cn.moves.index(lm) if lm is not None else int(np.argmax(game.dist[i]))
            else:
                j = int(np.argmax(game.dist[i]))
        v_child = self.V.get(root.children[j0], g.nodes[root.children[j0]].value)
        value = v_child if root.me_to_move else 1.0 - v_child
        return Line(moves, value, net, probs, learned_flags, reach, nonobv)


def fields_of(s: Settings) -> list[str]:
    return [f.name for f in fields(s)]
