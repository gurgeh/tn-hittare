"""What a solution looks like to a person: a table, a PGN, the curve and its steps.

Two gains are reported per learned position, and they answer different questions:

- **loss if forgotten**: the drop in expected score if only this position were
  forgotten and the rest kept. Positions that only work together (one steers
  into the other) each show the pair's whole value, so this column does not add
  up to the total.
- **curve gain**: the rise in the curve at the budget where the position entered
  the repertoire and stayed (see `steps`). When the sets along the curve are
  nested these add up to the total. When several positions enter at one step
  they share it, and the table says so; when a step also drops a position, the
  steps table shows the swap.
"""

from __future__ import annotations

import io
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import chess
import chess.pgn
import numpy as np

from tnh.builder.game import Game, Key, Learned, Node
from tnh.builder.search import Solution, drop_one_gains


# -- the curve -------------------------------------------------------------------------

@dataclass
class Step:
    N: int
    value: float
    marginal: float              # value(N) - value(N - 1)
    entered: list[Key]
    left: list[Key]


def steps(sols: list[Solution]) -> list[Step]:
    """How the learned set changes along the curve. Sets need not be nested: a step can
    swap positions as well as add one."""
    out = []
    for a, b in zip(sols, sols[1:]):
        out.append(Step(b.N, b.value, b.value - a.value,
                        [k for k in b.learned if k not in a.learned],
                        [k for k in a.learned if k not in b.learned]))
    return out


def budget_for_price(sols: list[Solution], mu: float) -> int:
    """The largest N whose marginal gain on the curve is at least `mu` (0 if none)."""
    best = 0
    for s in steps(sols):
        if s.marginal >= mu:
            best = s.N
    return best


def entry_step(sols: list[Solution], N: int) -> dict[Key, Step]:
    """For each position learned at N, the step from which it stays learned up to N."""
    st = steps(sols)
    out: dict[Key, Step] = {}
    for k in sols[N].learned:
        n = N
        while n > 0 and k in sols[n - 1].learned:
            n -= 1
        if n > 0:
            out[k] = st[n - 1]
    return out


def steps_text(sols: list[Solution], name: Callable[[Key], str] = str) -> str:
    lines = ["  N   expected score   marginal    in / out"]
    for s in steps(sols):
        io_ = ", ".join(f"+{name(k)}" for k in s.entered) + "".join(f", -{name(k)}" for k in s.left)
        lines.append(f"{s.N:3d}   {s.value:.5f}          {s.marginal:+.5f}    {io_}")
    return "\n".join(lines)


def curve_json(sols: list[Solution]) -> str:
    return json.dumps([{"N": s.N, "value": s.value, "dp_value": s.dp_value, "cost": s.cost,
                        "learned": {str(k): m for k, m in s.learned.items()}} for s in sols], indent=1)


def curve_text(sols: list[Solution]) -> str:
    base = sols[0].value
    lines = ["  N   expected score   gain over N=0   marginal   DP's own value   positions used"]
    prev = base
    for s in sols:
        lines.append(f"{s.N:3d}   {s.value:.5f}          {s.value - base:+.5f}       {s.value - prev:+.5f}   "
                     f"{s.dp_value:.5f}          {s.cost}")
        prev = s.value
    return "\n".join(lines)


# -- per position -------------------------------------------------------------------------

@dataclass
class Row:
    key: Key
    node: int
    path: list[str]          # most likely path to it under the repertoire, from the root
    move: str
    reach: float             # probability of reaching the position under the repertoire
    loss_if_forgotten: float
    curve_gain: float        # the marginal of the step it entered at (shared if several entered)
    entered_at: int          # that step's N
    shared_with: int         # how many other positions entered at the same step
    p_find: float            # how often my model plays the move unaided
    cost: int                # budget units this move costs
    engine: float            # engine's expected score for me after the move, at the search budget
    engine_best: float       # same, after the engine's best move
    line: list[str] = field(default_factory=list)    # played out: learned moves, his best replies
    line_value: float = math.nan                       # engine's value for me at the line's end
    line_end: str = ""                                 # why the line stops
    deep: dict | None = None                           # the deeper soundness check, when run


def rel_path(node: Node) -> list[str]:
    """The node's creation path from the graph's root (an expander's paths start at the
    initial position, before the root moves)."""
    return list(node.path[len(node.path) - node.ply:]) if node.ply else []


def best_paths(game: Game, learned: Learned) -> dict[int, list[str]]:
    """For every node, the most likely path from the root under the repertoire."""
    g = game.graph
    best = {g.root: (1.0, [])}
    for i in game.order:
        if i not in best or not g.nodes[i].expanded:
            continue
        p0, path = best[i]
        n = g.nodes[i]
        pol = game.policy(i, learned)
        for m, c, x in zip(n.moves, n.children, pol):
            if x > 0 and (c not in best or p0 * x > best[c][0]):
                best[c] = (p0 * x, path + [m])
    return {i: p for i, (_, p) in best.items()}


def _mine(node: Node, j: int) -> float:
    v = float(node.engine[j])
    return v if node.me_to_move else 1.0 - v


def engine_line(game: Game, learned: Learned, start: int,
                ensure: Callable[[Node], None] | None = None) -> tuple[list[str], float, str]:
    """From a learned position: play the learned move, then the opponent's engine-best
    reply, and on while my positions are learned. The value is the engine's, for me,
    of the last move played. `ensure` fetches engine values where they are missing."""
    g = game.graph
    i, line, value = start, [], math.nan
    while True:
        n = g.nodes[i]
        if not n.expanded:
            return line, value, "frontier" if n.moves else "game over"
        if n.me_to_move:
            mv = learned.get(n.key)
            if mv is None:
                return line, value, "not learned"
            j = n.moves.index(mv)
        else:
            if n.engine is None and ensure is not None:
                ensure(n)
            if n.engine is None or np.all(np.isnan(n.engine)):
                return line, value, "no engine value"
            j = int(np.nanargmax(n.engine))
        value = _mine(n, j)
        line.append(n.moves[j])
        i = n.children[j]


def rows(game: Game, sols: list[Solution], N: int, ensure: Callable[[Node], None] | None = None,
         deep: dict[Key, dict] | None = None) -> list[Row]:
    sol = sols[N]
    g = game.graph
    R = game.reach(sol.learned)
    paths = best_paths(game, sol.learned)
    losses = drop_one_gains(game, sol.learned)
    entry = entry_step(sols, N)
    out = []
    for k, m in sol.learned.items():
        ids = game.by_key[k]
        i = max(ids, key=lambda j: R[j])
        n = g.nodes[i]
        j = n.moves.index(m)
        st = entry.get(k)
        line, lv, why = engine_line(game, sol.learned, i, ensure)
        out.append(Row(k, i, paths.get(i, rel_path(n)), m, float(sum(R[x] for x in ids)), losses[k],
                       st.marginal if st else 0.0, st.N if st else 0, (len(st.entered) - 1) if st else 0,
                       float(n.p_model[j]), game.move_cost(i, j), _mine(n, j), float(np.nanmax(n.engine)),
                       line, lv, why, (deep or {}).get(k)))
    return sorted(out, key=lambda r: (r.entered_at, len(r.path)))


def _san_line(board: chess.Board | None, moves: list[str]) -> list[str]:
    if board is None:
        return list(moves)
    b = board.copy(stack=False)
    out = []
    for m in moves:
        mv = chess.Move.from_uci(m)
        out.append(b.san(mv))
        b.push(mv)
    return out


LEGEND = """\
- reach: probability of reaching the position when I play the repertoire.
- loss if forgotten: drop in expected score if only this position were forgotten; complementary positions each show their joint value, so this column does not add up.
- curve gain: rise of the curve at the budget N where the position entered and stayed (`@N`); adds up to the total when the sets along the curve are nested (see the steps). `shared xK` means K positions entered at that step together.
- cost: budget units (with the obviousness rule, 0 when my model plays the move at least the threshold).
- I find it: how often my model plays the move unaided.
- engine: engine's expected score for me after the move, at the search budget (best: after its own best move).
- line: played out from the position, my learned moves and his engine-best replies until a position I have not learned; line value: the engine's value for me at its end."""


def table(rs: list[Row], root_board: chess.Board | None = None, path_board: Callable[[Row], chess.Board] | None = None) -> str:
    deep = any(r.deep for r in rs)
    head = "| # | line to it | learn | cost | reach | loss if forgotten | curve gain | I find it | engine (best) | played out | line value |"
    sep = "|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|"
    if deep:
        head += " deep: move / best other / passes |"
        sep += "---|"
    lines = [head, sep]
    for x, r in enumerate(rs, 1):
        to = _san_line(root_board, r.path + [r.move])
        b = None
        if root_board is not None:
            b = root_board.copy(stack=False)
            for m in r.path:
                b.push_uci(m)
        played = _san_line(b, r.line)
        shared = f" shared x{r.shared_with + 1}" if r.shared_with else ""
        row = (f"| {x} | {' '.join(to[:-1]) or '(root)'} | **{to[-1]}** | {r.cost} | {r.reach:.3f} | {r.loss_if_forgotten:+.4f} | "
               f"{r.curve_gain:+.4f} @{r.entered_at}{shared} | {r.p_find:.0%} | {r.engine:.3f} ({r.engine_best:.3f}) | "
               f"{' '.join(played)} ({r.line_end}) | {r.line_value:.3f} |")
        if deep:
            d = r.deep
            row += (f" {d['move_value']:.3f} / {d['alt_value']:.3f} / {'yes' if d['passes'] else '**no**'} |"
                    if d else " not checked |")
        lines.append(row)
    return "\n".join(lines) + "\n\n" + LEGEND


def pgn(rs: list[Row], root_moves: list[str], headers: dict[str, str] | None = None) -> str:
    """The repertoire as one PGN: the root line, then a variation per learned position
    (its most likely path and the learned move), with comments on the learned moves."""
    game = chess.pgn.Game()
    for k, v in (headers or {}).items():
        game.headers[k] = v
    node = game
    for m in root_moves:
        node = node.add_variation(chess.Move.from_uci(m))
    root = node
    for r in rs:
        node = root
        for m in r.path + [r.move]:
            mv = chess.Move.from_uci(m)
            node = node.variation(mv) if node.has_variation(mv) else node.add_variation(mv)
        c = (f"learn: reach {r.reach:.3f}, loss if forgotten {r.loss_if_forgotten:+.4f}, "
             f"curve gain {r.curve_gain:+.4f} at N={r.entered_at}, I find it {r.p_find:.0%}, "
             f"engine {r.engine:.3f} (best {r.engine_best:.3f}), line value {r.line_value:.3f}")
        if r.deep:
            c += (f", deep {r.deep['move_value']:.3f} vs {r.deep['alt_value']:.3f} "
                  f"({'passes' if r.deep['passes'] else 'FAILS'})")
        node.comment = (node.comment + " " + c).strip()
    out = io.StringIO()
    print(game, file=out, end="\n\n")
    return out.getvalue()


# -- how unfinished a result is ---------------------------------------------------------

def frontier_mass(game: Game, learned: Learned) -> dict[str, float]:
    """Reach-weighted mass of unexpanded positions under the repertoire.

    - `in_budget_zone`: the probability that a game under the repertoire reaches a
      position that was not expanded although budget could have been spent there
      (reach bound >= `min_reach`). Where the search stopped looking, not where it
      chose to stop.
    - `within_2_plies_of_a_learned_move`: the probability that the game reaches a
      frontier value-head read one or two plies after a learned move. The learned
      move's worth rests directly on those reads.
    - `stand_in`: the probability of reaching a position whose value is its parent's
      stand-in (a rare move that got no value call, see `Expander(tail_p=...)`).
    - `total`: all frontier mass (1 minus the mass of finished games).
    """
    g = game.graph
    R = game.reach(learned)
    bound = game.reach_bound()
    zone = game.params.min_reach
    # mass split by plies since the last learned move: 1, 2, more or none
    M = np.zeros((len(g.nodes), 3))
    M[g.root, 2] = 1.0
    for i in game.order:
        n = g.nodes[i]
        if not n.expanded:
            continue
        pol = game.policy(i, learned)
        if n.me_to_move and n.key in learned:
            M[n.children, 0] += M[i].sum() * pol
        else:
            shifted = np.array([0.0, M[i, 0], M[i, 1] + M[i, 2]])
            M[n.children] += np.outer(pol, shifted)
    front = [i for i in game.order if not g.nodes[i].expanded and g.nodes[i].moves]
    return {"total": float(R[front].sum()),
            "in_budget_zone": float(sum(R[i] for i in front if bound[i] >= zone)),
            "within_2_plies_of_a_learned_move": float(M[front, :2].sum()),
            "stand_in": float(sum(R[i] for i in front if g.nodes[i].info.get("stand_in")))}
