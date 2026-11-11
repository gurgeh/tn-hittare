"""Does a repertoire's value hold up as the graph grows below it?

The search chooses among sound moves by values read at the frontier. If those reads
are noisy, the max picks the ones that are too high (the winner's curse) and the
chosen repertoire's value falls as the positions below it are expanded. Two tools:

- `drift`: fix a repertoire and grow the graph below it in rounds (expanding the
  frontier positions with the largest reach under it). Per round: the fixed
  repertoire's exact value, the search's own choice and its value, and the value
  on this round's graph of the previous round's choice. A value function that is a
  martingale under the models' own play gives a flat line for the fixed
  repertoire; a falling line for the search's choices with a flat N=0 line is
  selection, not drift.
- `solve_until_stable`: remedy (a). Solve, grow below the chosen repertoire,
  solve again, until the chosen repertoire's value moves less than `tol` between
  rounds; says when it does not converge.

Remedy (b), shrinking frontier values toward a reference, is `Params.shrink`; the
harness evaluates several parameter variants on the same growth sequence.

Any expander works that has `.graph`, `.params` and `.expand(node)`:
`tnh.builder.expand.Expander` against real models, `RevealExpander` on a synthetic
game whose full tree is known.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from tnh.builder.game import Game, Graph, Learned, Params
from tnh.builder.output import frontier_mass
from tnh.builder.search import solve


@dataclass
class Round:
    round: int
    nodes: int
    expanded: int
    v0: float                      # nothing learned
    fixed: float                   # the fixed repertoire
    chosen: dict                   # the search's choice at N
    chosen_value: float
    prev_chosen_now: float         # the previous round's choice, on this round's graph
    overlap: int                   # positions shared by the choice and the fixed repertoire
    frontier_in_zone: float        # under the fixed repertoire
    variants: dict = field(default_factory=dict)   # name -> (choice, its value under that variant)


def grow_below(expander, learned: Learned, n: int) -> int:
    """Expand the `n` frontier positions with the largest reach under `learned`."""
    game = Game(expander.graph, expander.params)
    R = game.reach(learned)
    g = expander.graph
    front = sorted((i for i in game.order if g.nodes[i].children is None and g.nodes[i].moves),
                   key=lambda i: -R[i])
    done = 0
    for i in front[:n]:
        if R[i] <= 0:
            break
        expander.expand(g.nodes[i])
        done += 1
    return done


def drift(expander, fixed: Learned, N: int, rounds: int, per_round: int,
          variants: dict[str, Params] | None = None) -> list[Round]:
    out: list[Round] = []
    prev = None
    for r in range(rounds + 1):
        game = Game(expander.graph, expander.params)
        _, sols = solve(game, N)
        chosen = sols[N].learned
        var = {}
        for name, p in (variants or {}).items():
            vg = Game(expander.graph, p)
            _, vs = solve(vg, N)
            var[name] = (vs[N].learned, vs[N].value)
        out.append(Round(r, len(game.order), sum(1 for i in game.order if expander.graph.nodes[i].expanded),
                         sols[0].value, game.evaluate(fixed), chosen, sols[N].value,
                         game.evaluate(prev) if prev is not None else math.nan,
                         len(set(chosen) & set(fixed)), frontier_mass(game, fixed)["in_budget_zone"], var))
        prev = chosen
        if r < rounds and grow_below(expander, fixed, per_round) == 0:
            break
    return out


def drift_text(rs: list[Round]) -> str:
    lines = ["round  nodes  expanded   N=0      fixed    search's choice   prev choice now   overlap  frontier in zone"]
    for x in rs:
        lines.append(f"{x.round:5d}  {x.nodes:5d}  {x.expanded:8d}   {x.v0:.4f}   {x.fixed:.4f}   {x.chosen_value:.4f}"
                     f"            {x.prev_chosen_now:.4f}            {x.overlap:3d}      {x.frontier_in_zone:.3f}")
    if rs and rs[0].variants:
        names = list(rs[0].variants)
        lines.append("")
        lines.append("round  " + "  ".join(f"{n:>18s}" for n in names) + "   (each variant's choice, valued by itself)")
        for x in rs:
            lines.append(f"{x.round:5d}  " + "  ".join(f"{x.variants[n][1]:18.4f}" for n in names))
    return "\n".join(lines)


@dataclass
class Stable:
    sols: list
    history: list[tuple[int, float, float, int]]     # round, value of the choice, its value after growth, nodes
    converged: bool
    rounds: int


def solve_until_stable(expander, N: int, *, per_round: int, tol: float, max_rounds: int) -> Stable:
    """Remedy (a): solve, grow below the choice, re-value it, until it moves less than `tol`."""
    history = []
    for r in range(max_rounds):
        game = Game(expander.graph, expander.params)
        _, sols = solve(game, N)
        chosen, v = sols[N].learned, sols[N].value
        if grow_below(expander, chosen, per_round) == 0:
            history.append((r, v, v, len(game.order)))
            return Stable(sols, history, True, r + 1)
        after = Game(expander.graph, expander.params).evaluate(chosen)
        history.append((r, v, after, len(game.order)))
        if abs(after - v) < tol:
            game = Game(expander.graph, expander.params)
            _, sols = solve(game, N)
            return Stable(sols, history, True, r + 1)
    game = Game(expander.graph, expander.params)
    _, sols = solve(game, N)
    return Stable(sols, history, False, max_rounds)


# -- a synthetic game with a known truth -------------------------------------------------

class RevealExpander:
    """A synthetic game whose full tree is known; positions at ply >= `depth` start
    unexpanded, and expanding one reveals its children from the full tree."""

    def __init__(self, full: Graph, params: Params, depth: int) -> None:
        self.full = full
        self.params = params
        self.graph = copy.deepcopy(full)
        self.hidden: dict[int, list[int]] = {}
        for n in self.graph.nodes:
            if n.ply >= depth and n.children is not None:
                self.hidden[n.id] = n.children
                n.children = None
        self.stats = {"expanded": 0}

    def expand(self, node) -> list:
        kids = self.hidden.pop(node.id, None)
        if kids is None:
            return []
        node.children = kids
        self.stats["expanded"] += 1
        return [self.graph.nodes[c] for c in kids]


def noisy_values(g: Graph, params: Params, rng: np.random.Generator, sigma: float) -> None:
    """Every node's `value` becomes its true value with nothing learned (plain expectation
    over the full tree) plus independent noise: a value head that is right on average."""
    true = Game(g, params).values({})
    for n in g.nodes:
        if n.moves and not math.isnan(true[n.id]):
            n.value = float(np.clip(true[n.id] + rng.normal(0, sigma), 0.0, 1.0))


def curse_experiment(seeds: int, sigma: float, variants: dict[str, dict], *, N: int = 4, depth: int = 8,
                     visible: int = 4, eps: float = 0.08, seed0: int = 0) -> dict[str, np.ndarray]:
    """The winner's curse on a synthetic game: the frontier values at ply `visible` are
    the truth plus N(0, sigma). Per variant (Params overrides) and seed: the gain the
    search predicts for its choice and that choice's true gain on the full tree."""
    from dataclasses import replace

    from tnh.builder.synth import random_game
    out: dict[str, list] = {k: [] for k in variants}
    for s in range(seeds):
        rng = np.random.default_rng(seed0 + s)
        full = random_game(rng, depth=depth, branching=(2, 3), alpha=0.5)
        base = Params(eps_white=eps)
        noisy_values(full, base, rng, sigma)
        truth = Game(full, base)
        t0 = truth.evaluate({})
        ex = RevealExpander(full, base, depth=visible)
        for name, kw in variants.items():
            g = Game(ex.graph, replace(base, **kw))
            _, sols = solve(g, N)
            out[name].append((sols[N].value - sols[0].value, truth.evaluate(sols[N].learned) - t0))
    return {k: np.array(v) for k, v in out.items()}
