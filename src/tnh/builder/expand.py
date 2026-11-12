"""Growing the game graph lazily against real models.

A model call is one position given to a `MoveModel` (its move distribution and its
value head) or to a `ValueModel` (the engine's value of every legal move). Both
are cached on the position key, so a transposition costs nothing, and counted.

What each node costs:

- created (as a child of an expanded node): one move-model call, which gives its
  frontier value (the value head) and, if it is expanded later, its distribution;
- expanded where budget may be spent (reach bound >= `min_reach`): one engine
  call if I am to move (which moves are sound). Below `min_reach` (the
  expectation margin) my nodes get none and cannot be learned there;
- an opponent's node needs the engine (his best move) only for pessimism: with
  `lam` only on the best-reply spine, the nodes that some path reaches with him
  playing the best move every time; with `lam_node` at every expanded one.

So a run costs about (nodes expanded x legal moves) move-model calls and
(my nodes expanded in the budget zone) engine calls, less what transpositions
save. `Expander.stats` has the exact counts.

Growth is best-first. `grow` expands in order of the reach bound (the most a
node could be reached by any repertoire); `refine` expands the frontier nodes
whose values weigh most in a given solution (their reach under it), which is
how a partly grown graph says where it needs to grow next.

What is guaranteed, and what is not:

- Guaranteed on whatever graph has been grown: the DP is exact on a tree, for any
  kappa, lam, lam_node, costs and pruning (checked against enumeration); the value
  reported for a repertoire is its exact expected score on that graph given the
  frontier values; every learned move is within eps of the engine's best at the
  search's engine budget (and at the deep budget if `deepcheck` ran); every
  model call is cached on the position and counted.
- Not guaranteed: that the graph contains the best repertoire. With `explore` < 1
  a sound move my model rarely plays is grown as if it were rarely reached, and
  `refine` grows only below the current solution, so a line can lose because it
  was looked at less. A position never expanded cannot be learned.
  `grow` also stops at `max_expansions`, so even with `explore` = 1 positions in
  the budget zone may be left unexpanded. `frontier_mass(...)["in_budget_zone"]`
  in the output is the probability under the repertoire of reaching such a
  position: the honest measure of how unfinished a result is.
- Not guaranteed: that the frontier values are unbiased where the search chose.
  The max over my sound moves picks up noise in the value head (the winner's
  curse); `drift` measures it, `Params.shrink` and `solve_until_stable` reduce it.
- On a graph with transpositions the DP is not exact; the repaired solution was
  measured at 99.9 % of the optimum on synthetic graphs (`python -m tnh.builder gap`).
"""

from __future__ import annotations

import heapq
import math
import time
from collections import Counter
from collections.abc import Hashable

import chess
import chess.polyglot
import numpy as np

from tnh.builder.counts import CountSource, NoCounts
from tnh.builder.game import (Game, Graph, Learned, Node, Params, best_index, model_type_dist, opponent_dist,
                             posterior_step)
from tnh.models.base import MoveModel, Query, ValueModel


def pos_key(board: chess.Board) -> int:
    """The same key as `tnh.data.edges.pos_hash`: the polyglot Zobrist hash."""
    return chess.polyglot.zobrist_hash(board)


class Expander:
    def __init__(self, move_model: MoveModel, value_model: ValueModel | None, params: Params, *,
                 me_white: bool, me_elo: int, opp_elo: int, root_moves: list[str] = (),
                 counts: CountSource | None = None, batch: int = 64, explore: float = 1.0,
                 value_fn=None, tail_p: float = 0.0) -> None:
        """`explore`: the reach factor growth gives a sound move at my nodes. 1 is the true
        upper bound (I could learn it). Then every sound move is reachable with
        probability 1, and with five sound moves per position growth fans out fivefold at
        each of my plies; a lower value grows mostly where my model goes and leaves the
        rest to `refine`. Positions whose growth bound is below `min_reach` get no engine
        call and cannot be learned.

        `value_fn`: an object with `wdl(queries) -> (n, 3)` win/draw/loss for the mover,
        e.g. `tnh.models.combo.FrontierValue`. Given, it values every created position
        and the move model is called only when a position is expanded (most positions
        never are). Not given, the move model's `Prediction.wdl` is the value and it is
        called for every created position.

        `tail_p`: children that the mover plays with probability below `tail_p` (and that
        are not sound moves of mine, nor the opponent's engine-best reply where
        pessimism needs it) get no value call: they stand in with their parent's value
        and carry `info["stand_in"]`. `frontier_mass` reports the mass that rests on them.
        """
        self.move_model = move_model
        self.value_model = value_model
        self.value_fn = value_fn
        self.tail_p = tail_p
        self.params = params
        self.me_white = me_white
        self.me_elo, self.opp_elo = me_elo, opp_elo
        self.counts_source = counts or NoCounts()
        self.batch = batch
        self.explore = explore
        self.graph = Graph(me_white)
        self.boards: dict[int, chess.Board] = {}
        self._dist: dict[Hashable, tuple[list[str], np.ndarray]] = {}   # key -> moves, model probabilities
        self._val: dict[Hashable, float] = {}                           # key -> mover's expected score
        self._engine: dict[Hashable, np.ndarray] = {}
        self.bound: dict[int, float] = {}
        self.stats: Counter = Counter()
        self.seconds: Counter = Counter()
        board = chess.Board()
        for m in root_moves:
            board.push_uci(m)
        self.root_moves = tuple(root_moves)
        root = self._make([(board, self.root_moves)], [False], math.nan)[0]
        self.graph.root = root.id
        self.beta: dict[int, float] = {}
        if params.lam > 0:
            self.beta[root.id] = params.lam
        self.bound[root.id] = 1.0

    @classmethod
    def resume(cls, graph: Graph, move_model: MoveModel, value_model: ValueModel | None, params: Params, *,
               me_white: bool, me_elo: int, opp_elo: int, root_moves: list[str],
               counts: CountSource | None = None, batch: int = 64, explore: float = 1.0,
               value_fn=None, tail_p: float = 0.0) -> "Expander":
        """An expander over a saved graph: boards replayed from the paths, the caches
        refilled from the nodes, reach bounds and posteriors recomputed. No model calls
        unless `params` need engine values the saved graph lacks."""
        ex = cls.__new__(cls)
        ex.move_model, ex.value_model, ex.params = move_model, value_model, params
        ex.value_fn, ex.tail_p = value_fn, tail_p
        ex.me_white, ex.me_elo, ex.opp_elo = me_white, me_elo, opp_elo
        ex.counts_source = counts or NoCounts()
        ex.batch, ex.explore = batch, explore
        ex.graph = graph
        ex.root_moves = tuple(root_moves)
        ex.stats, ex.seconds = Counter(), Counter()
        ex.boards, ex._dist, ex._val, ex._engine = {}, {}, {}, {}
        for n in graph.nodes:
            b = chess.Board()
            for m in n.path:
                b.push_uci(m)
            if pos_key(b) != n.key:
                raise ValueError(f"node {n.id}: its path does not lead to its key")
            ex.boards[n.id] = b
            if n.p_model is not None:
                ex._dist.setdefault(n.key, (list(n.moves), n.p_model))
            if not n.info.get("stand_in"):
                ex._val.setdefault(n.key, n.value if n.me_to_move else 1.0 - n.value)
            if n.engine is not None:
                ex._engine.setdefault(n.key, n.engine)
        ex.bound = {graph.root: 1.0}
        ex.beta = {graph.root: params.lam} if params.lam > 0 else {}
        root = graph.nodes[graph.root]
        if (root.me_to_move and ex.in_budget_zone(root)) or ex.needs_best(root):
            ex.ensure_engine(root)
        ex._propagate(root)
        return ex

    # -- model calls ----------------------------------------------------------------

    def _query(self, board: chess.Board, path: tuple[str, ...]) -> Query:
        mine = board.turn == (chess.WHITE if self.me_white else chess.BLACK)
        me, op = (self.me_elo, self.opp_elo) if mine else (self.opp_elo, self.me_elo)
        return Query(board.fen(), list(path), me, op)

    def _mine(self, board: chess.Board) -> bool:
        return board.turn == (chess.WHITE if self.me_white else chess.BLACK)

    def _predict(self, items: list[tuple[Hashable, chess.Board, tuple[str, ...]]]) -> None:
        """Move-model calls for positions not in the cache; with no `value_fn` they also
        give the values."""
        todo, seen = [], set()
        for k, b, path in items:
            if k in self._dist or k in seen:
                self.stats["move model cache hits"] += 1
                continue
            seen.add(k)
            todo.append((k, b, path))
        for s in range(0, len(todo), self.batch):
            chunk = todo[s : s + self.batch]
            t = time.perf_counter()
            preds = self.move_model.predict([self._query(b, p) for _, b, p in chunk])
            self.seconds["move model"] += time.perf_counter() - t
            self.stats["move model calls"] += len(chunk)
            for (k, b, _), pr in zip(chunk, preds):
                p = np.exp(pr.logp.astype(np.float64))
                self._dist[k] = (list(pr.moves), p / p.sum())
                if self.value_fn is None:
                    if pr.wdl is None:
                        raise ValueError(f"{self.move_model.name} gave no value at {b.fen()}")
                    w, d, _l = pr.wdl
                    self._val.setdefault(k, w + 0.5 * d)

    def _values(self, items: list[tuple[Hashable, chess.Board, tuple[str, ...]]]) -> None:
        """The mover's expected score for positions not in the cache."""
        todo, seen = [], set()
        for k, b, path in items:
            if k in self._val or k in seen:
                self.stats["value cache hits"] += 1
                continue
            seen.add(k)
            if b.is_game_over(claim_draw=False):
                self._val[k] = _result_for_mover(b)
                self.stats["game over"] += 1
                continue
            todo.append((k, b, path))
        if self.value_fn is None:
            self._predict(todo)
            return
        for s in range(0, len(todo), self.batch):
            chunk = todo[s : s + self.batch]
            t = time.perf_counter()
            w = np.asarray(self.value_fn.wdl([self._query(b, p) for _, b, p in chunk]), dtype=np.float64)
            self.seconds["value"] += time.perf_counter() - t
            self.stats["value calls"] += len(chunk)
            for (k, _, _), row in zip(chunk, w):
                self._val[k] = float(row[0] + 0.5 * row[1])

    def _set_dist(self, node: Node) -> None:
        """The node's move distribution, aligned with its legal moves."""
        if node.p_model is not None or not node.moves:
            return
        self._predict([(node.key, self.boards[node.id], node.path)])
        moves, p = self._dist[node.key]
        idx = {m: i for i, m in enumerate(moves)}
        if set(idx) != set(node.moves):
            raise ValueError(f"move model and board disagree on the legal moves at {self.boards[node.id].fen()}")
        node.p_model = p[[idx[m] for m in node.moves]]

    def _engine_values(self, node: Node) -> np.ndarray:
        if node.key in self._engine:
            self.stats["engine cache hits"] += 1
            return self._engine[node.key]
        if self.value_model is None:
            raise ValueError("no value model: needed for sound moves and pessimism")
        b = self.boards[node.id]
        t = time.perf_counter()
        mv = self.value_model.evaluate([self._query(b, node.path)])[0]
        self.seconds["engine"] += time.perf_counter() - t
        self.stats["engine calls"] += 1
        idx = {m: i for i, m in enumerate(mv.moves)}
        if set(idx) != set(node.moves):
            raise ValueError(f"engine and board disagree on the legal moves at {b.fen()}")
        v = np.array([mv.value[idx[m]] for m in node.moves], dtype=np.float64)
        self.stats["engine moves without a value"] += int(np.isnan(v).sum())
        self._engine[node.key] = v
        return v

    # -- nodes -------------------------------------------------------------------------

    def _make(self, items: list[tuple[chess.Board, tuple[str, ...]]], stand_in: list[bool],
              parent_value: float) -> list[Node]:
        """Nodes for the boards; `stand_in[i]` children get `parent_value` (mine) and no call.
        A stand-in that is needed for real later is upgraded in place."""
        keyed = [(pos_key(b), b, path) for b, path in items]
        need = [x for x, si in zip(keyed, stand_in) if not si]
        self._values(need)
        if self.value_fn is None:
            self._predict(need)
        out = []
        for (k, b, path), si in zip(keyed, stand_in):
            ply = len(path) - len(self.root_moves)
            mine = self._mine(b)
            node = self.graph.get(k, ply)
            if node is not None:
                self.stats["transpositions merged"] += 1
                if not si and node.info.get("stand_in"):
                    s = self._val[k]
                    node.value = s if mine else 1.0 - s
                    del node.info["stand_in"]
                    self.stats["stand-ins upgraded"] += 1
                out.append(node)
                continue
            moves = [m.uci() for m in b.legal_moves]
            if si:
                node = self.graph.add(k, ply, mine, moves, value=parent_value, path=path, info={"stand_in": True})
                self.stats["stand-ins"] += 1
            else:
                s = self._val[k]
                node = self.graph.add(k, ply, mine, moves, value=s if mine else 1.0 - s, path=path)
            self.boards[node.id] = b
            if k in self._dist and moves:
                self._set_dist(node)
            out.append(node)
        return out

    def in_budget_zone(self, node: Node) -> bool:
        return self.bound.get(node.id, 0.0) >= self.params.min_reach

    def needs_best(self, node: Node) -> bool:
        """The opponent's best move is needed for pessimism: everywhere with `lam_node`,
        and with `lam` only where the engine type can be (posterior > 0)."""
        if node.me_to_move:
            return False
        return self.params.lam_node > 0 or self.beta.get(node.id, 0.0) > 0

    def expand(self, node: Node) -> list[Node]:
        if node.children is not None or not node.moves:
            return []
        self._set_dist(node)
        if not node.me_to_move and node.counts is None:
            t = time.perf_counter()
            node.counts = self.counts_source.counts(node.key, node.moves)
            self.seconds["counts"] += time.perf_counter() - t
            self.stats["count lookups"] += 1
        if (node.me_to_move and self.in_budget_zone(node)) or self.needs_best(node):
            self.ensure_engine(node)
        if node.info.get("stand_in"):
            self._values([(node.key, self.boards[node.id], node.path)])
            s = self._val[node.key]
            node.value = s if node.me_to_move else 1.0 - s
            del node.info["stand_in"]
            self.stats["stand-ins upgraded"] += 1
        b = self.boards[node.id]
        items = []
        for m in node.moves:
            c = b.copy(stack=False)
            c.push_uci(m)
            items.append((c, node.path + (m,)))
        kids = self._make(items, self._stand_ins(node), node.value)
        self.graph.link(node, kids)
        self.stats["expanded"] += 1
        self._propagate(node)
        return kids

    def _stand_ins(self, node: Node) -> list[bool]:
        """Which children get no value call: rarely played, not a sound move of mine, not
        his engine-best reply where pessimism needs it."""
        if self.tail_p <= 0:
            return [False] * len(node.moves)
        p = self._factor_plain(node)
        keep = p >= self.tail_p
        if node.me_to_move and node.engine is not None and not np.all(np.isnan(node.engine)):
            eps = self.params.eps_at(self.me_white, node.id == self.graph.root)
            with np.errstate(invalid="ignore"):
                keep |= node.engine >= np.nanmax(node.engine) - eps - 1e-12
        if not node.me_to_move and self.needs_best(node):
            keep[best_index(node.engine, node)] = True
        return [not k for k in keep]

    def _factor_plain(self, node: Node) -> np.ndarray:
        """The mover's probability of each move, with nothing learned."""
        if node.me_to_move:
            return node.p_model
        return opponent_dist(node, self.params, self.beta.get(node.id, 0.0))

    def ensure_engine(self, node: Node) -> None:
        if node.engine is None and node.moves:
            node.engine = 1.0 * self._engine_values(node)

    def _factor(self, node: Node) -> np.ndarray:
        if node.me_to_move:
            f = node.p_model.copy()
            if node.engine is not None and not np.all(np.isnan(node.engine)):
                eps = self.params.eps_at(self.me_white, node.id == self.graph.root)
                with np.errstate(invalid="ignore"):
                    ok = node.engine >= np.nanmax(node.engine) - eps - 1e-12
                    f[ok] = np.maximum(f[ok], self.explore)
            return f
        return opponent_dist(node, self.params, self.beta.get(node.id, 0.0))

    def _child_beta(self, node: Node) -> np.ndarray:
        b = self.beta.get(node.id, 0.0)
        out = np.zeros(len(node.moves))
        if b == 0:
            return out
        if node.me_to_move:
            return out + b
        q = model_type_dist(node, self.params)
        j = best_index(node.engine, node)
        out[j] = posterior_step(b, q[j])
        return out

    def _propagate(self, node: Node) -> None:
        """Push reach bounds and engine-type posteriors down from `node`. A node that
        enters the budget zone, or the best-reply spine, after it was expanded gets its
        engine call now."""
        stack = [node]
        while stack:
            n = stack.pop()
            if n.children is None:
                continue
            f = self._factor(n) * self.bound[n.id]
            cb = self._child_beta(n)
            for c, x, bb in zip(n.children, f, cb):
                changed = False
                if x > self.bound.get(c, 0.0) + 1e-15:
                    self.bound[c] = float(x)
                    changed = True
                if bb > self.beta.get(c, 0.0) + 1e-15:
                    self.beta[c] = float(bb)
                    changed = True
                cn = self.graph.nodes[c]
                if changed and cn.children is not None:
                    if (cn.me_to_move and self.in_budget_zone(cn)) or self.needs_best(cn):
                        self.ensure_engine(cn)
                    stack.append(cn)

    # -- growth --------------------------------------------------------------------------

    def grow(self, *, expand_reach: float, max_expansions: int, max_calls: int | None = None) -> int:
        """Best-first by reach bound, down to `expand_reach`. Returns the expansions made."""
        heap = [(-self.bound.get(n.id, 0.0), n.id) for n in self.graph.nodes if n.children is None and n.moves]
        heapq.heapify(heap)
        done = 0
        while heap and done < max_expansions:
            nb, i = heapq.heappop(heap)
            n = self.graph.nodes[i]
            if n.children is not None or -nb < self.bound.get(i, 0.0) - 1e-15:
                continue
            if -nb < expand_reach:
                break
            if max_calls is not None and self.stats["move model calls"] + self.stats["engine calls"] >= max_calls:
                self.stats["stopped at max_calls"] += 1
                break
            for c in self.expand(n):
                if c.children is None and c.moves:
                    heapq.heappush(heap, (-self.bound.get(c.id, 0.0), c.id))
            done += 1
        return done

    def refine(self, learned: Learned, *, n: int, min_weight: float = 0.0) -> int:
        """Expand the `n` frontier nodes with the largest weight in the objective: their
        reach under `learned`, or under no repertoire (where my model goes)."""
        imp = frontier_importance(Game(self.graph, self.params), learned)
        done = 0
        for i, w in imp[:n]:
            if w <= min_weight:
                break
            self.expand(self.graph.nodes[i])
            done += 1
        return done

    def board(self, node: Node) -> chess.Board:
        return self.boards[node.id]


def frontier_importance(game: Game, learned: Learned) -> list[tuple[int, float]]:
    """Unexpanded positions by how much their value weighs in the objective: an error of
    d in a frontier value moves my expected score by reach x d. The weight is the larger
    of the reach under `learned` and under no repertoire."""
    R = np.maximum(game.reach(learned), game.reach({}))
    g = game.graph
    out = [(i, float(R[i])) for i in game.order if g.nodes[i].children is None and g.nodes[i].moves]
    return sorted(out, key=lambda t: -t[1])


def _result_for_mover(b: chess.Board) -> float:
    if b.is_checkmate():
        return 0.0
    return 0.5
