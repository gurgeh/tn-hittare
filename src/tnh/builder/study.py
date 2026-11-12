"""The questions asked of one root position: which repertoire, what each root move is worth,
and where the difference comes from.

- `grow_under`: best-first growth inside the subtree of one move, so every candidate root
  move gets the same number of expansions and none wins by being looked at more.
- `forced_curves`: per sound root move, the solution with that move forced, at a set of
  budgets (the forced move pays its own cost).
- `theory_profile`: per solution, how much of the game is spent in positions with many
  database games, the opponent's expected loss per move there and elsewhere, and how many
  of my positions on the way have a right move that is not obvious.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from tnh.builder.game import Game, Learned, Node, Params
from tnh.builder.search import Solution, solve


def subtree_of(expander, node: Node) -> Callable[[Node], bool]:
    """Positions whose creation path runs through `node` (a transposition reached first from
    elsewhere belongs to the first path)."""
    pre = node.path
    return lambda n: n.path[: len(pre)] == pre


def grow_under(expander, node: Node, *, max_expansions: int, expand_reach: float = 0.0) -> int:
    """Best-first by reach bound, relative to `node`, among the positions below it."""
    if node.children is None:
        expander.expand(node)
    inside = subtree_of(expander, node)
    b0 = expander.bound.get(node.id, 0.0) or 1.0
    heap = [(-expander.bound.get(n.id, 0.0) / b0, n.id) for n in expander.graph.nodes
            if n.children is None and n.moves and inside(n)]
    heapq.heapify(heap)
    done = 0
    while heap and done < max_expansions:
        nb, i = heapq.heappop(heap)
        n = expander.graph.nodes[i]
        if n.children is not None or -nb < expander.bound.get(i, 0.0) / b0 - 1e-15:
            continue
        if -nb < expand_reach:
            break
        for c in expander.expand(n):
            if c.children is None and c.moves and inside(c):
                heapq.heappush(heap, (-expander.bound.get(c.id, 0.0) / b0, c.id))
        done += 1
    return done


def sound_root_moves(game: Game) -> list[str]:
    g = game.graph
    root = g.nodes[g.root]
    return [root.moves[j] for j in game.sound.get(g.root, [])]


def forced_curves(graph, params: Params, moves: list[str], K: int, budgets: set[int]) -> dict[str, list[Solution]]:
    root = graph.nodes[graph.root]
    out = {}
    for m in moves:
        game = Game(graph, replace(params, force=frozenset({(root.key, m)})))
        _, sols = solve(game, K, improve_at=budgets)
        out[m] = sols
    return out


@dataclass
class Profile:
    plies_in: dict[int, float]        # threshold -> expected number of positions (plies) visited with >= that many games
    opp_loss: dict[str, tuple[float, float]]   # class -> (reach-weighted mean expected loss per move, reach mass)
    opp_positions: dict[str, int]
    my_nonobvious: float              # expected number of my positions where the engine's best move is not obvious
    my_positions: float               # expected number of my positions visited (expanded)


def theory_profile(game: Game, learned: Learned, games: Callable[[Node], int], *,
                   thresholds=(1000, 10000), obvious: float = 0.85, min_reach: float = 1e-4,
                   engine: Callable[[Node], None] | None = None) -> Profile:
    """`games(node)`: games in the full table that reached the position (0 below two).
    `engine(node)`: fetch engine values where missing (the opponent's nodes need them for
    the expected loss; without, they are left out and counted)."""
    g = game.graph
    R = game.reach(learned)
    plies = {t: 0.0 for t in thresholds}
    loss = {"heavy": [0.0, 0.0], "light": [0.0, 0.0], "rare": [0.0, 0.0]}
    npos = {"heavy": 0, "light": 0, "rare": 0, "no engine value": 0}
    nonobv, mine = 0.0, 0.0
    for i in game.order:
        n = g.nodes[i]
        r = R[i]
        if r < min_reach:
            continue
        cnt = games(n)
        for t in thresholds:
            if cnt >= t:
                plies[t] += r
        if not n.expanded:
            continue
        if n.me_to_move:
            mine += r
            if n.engine is not None and not np.all(np.isnan(n.engine)):
                b = int(np.nanargmax(n.engine))
                if n.p_model[b] < obvious:
                    nonobv += r
            continue
        if n.engine is None and engine is not None:
            engine(n)
        if n.engine is None or np.all(np.isnan(n.engine)):
            npos["no engine value"] += 1
            continue
        q = game.dist[i]
        v = np.nan_to_num(n.engine, nan=np.nanmin(n.engine))
        exp_loss = float(q @ (v.max() - v))
        cls = "heavy" if cnt >= thresholds[1] else "light" if cnt >= thresholds[0] else "rare"
        loss[cls][0] += r * exp_loss
        loss[cls][1] += r
        npos[cls] += 1
    opp = {k: ((s / w) if w > 0 else math.nan, w) for k, (s, w) in loss.items()}
    return Profile(plies, opp, npos, nonobv, mine)


def profile_text(rows: list[tuple[str, Profile]], thresholds=(1000, 10000)) -> str:
    head = ("| root move | positions with >= 1,000 games | >= 10,000 | "
            "his loss/move, >= 10,000 games (mass) | 1,000-9,999 (mass) | < 1,000 (mass) | "
            "my positions | of them right move not obvious |")
    lines = [head, "|" + "---|" * 8]
    for name, p in rows:
        o = p.opp_loss
        f = lambda k: f"{o[k][0]:.4f} ({o[k][1]:.2f})" if not math.isnan(o[k][0]) else f"- ({o[k][1]:.2f})"  # noqa: E731
        lines.append(f"| {name} | {p.plies_in[thresholds[0]]:.2f} | {p.plies_in[thresholds[1]]:.2f} | "
                     f"{f('heavy')} | {f('light')} | {f('rare')} | {p.my_positions:.2f} | "
                     f"{p.my_nonobvious:.2f} |")
    return "\n".join(lines)


# -- the command ------------------------------------------------------------------------------

def check_budgets(graph, params: Params, K: int, budgets: set[int], check, *, max_rounds: int = 6):
    """Deep-check every learned move in the solutions at `budgets`, ban failures, solve
    again, until all pass. Returns (sols, params with bans, results, log)."""
    results: dict = {}
    banned = set(params.banned)
    log: list[str] = []
    for r in range(max_rounds):
        p = replace(params, banned=frozenset(banned))
        _, sols = solve(Game(graph, p), K, improve_at=budgets)
        failed = set()
        for N in sorted(budgets):
            for k, m in sols[N].learned.items():
                if (k, m) not in results:
                    results[(k, m)] = check(k, m)
                if not results[(k, m)]["passes"]:
                    failed.add((k, m))
        if not failed:
            log.append(f"deep check round {r}: every learned move at budgets {sorted(budgets)} passes "
                       f"({len(results)} moves checked)")
            return sols, p, results, log
        for k, m in sorted(failed, key=str):
            d = results[(k, m)]
            log.append(f"deep check round {r}: {m} fails ({d['move_value']:.3f} against {d['alt_move']} "
                       f"{d['alt_value']:.3f}); banned")
        banned |= failed
    p = replace(params, banned=frozenset(banned))
    _, sols = solve(Game(graph, p), K, improve_at=budgets)
    log.append(f"deep check stopped after {max_rounds} rounds; the last solutions are not all checked")
    return sols, p, results, log


def run(a) -> None:
    import json
    import pickle
    import time
    from pathlib import Path

    import chess

    from tnh.builder.counts import EdgeDBCounts
    from tnh.builder.deepcheck import DeepChecker
    from tnh.builder.expand import Expander
    from tnh.builder.game import ObviousCost
    from tnh.builder.output import budget_for_price, curve_text, frontier_mass, rel_path, rows, steps_text, table
    from tnh.data.edges import EdgeDB
    from tnh.models.combo import Combo, FrontierValue
    from tnh.models.stockfish import Stockfish

    t0 = time.perf_counter()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    budgets = set(a.budgets)
    K = max(budgets)
    root_moves = a.moves.split()
    me_white = a.me == "white"
    board = chess.Board()
    for m in root_moves:
        board.push_uci(m)

    def san(moves):
        b, o = board.copy(stack=False), []
        for m in moves:
            mv = chess.Move.from_uci(m)
            o.append(b.san(mv))
            b.push(mv)
        return " ".join(o)

    params = Params(kappa=a.kappa, eps_white=a.eps, eps_black=a.eps, min_reach=a.min_reach,
                    cost=ObviousCost(a.obvious))
    db = EdgeDB("full")
    band = None if a.no_band else (a.band_lo, a.band_hi)
    counts = EdgeDBCounts(db, tempi=a.tempo or None, elo_bands=band)
    totals: dict = {}

    def games(node) -> int:
        if node.key not in totals:
            t = db.total(node.key)
            totals[node.key] = int(t["n"]) if t else 0
        return totals[node.key]

    log: list[str] = []
    rep: list[str] = []
    say = lambda s: (log.append(s), print(s, flush=True))   # noqa: E731

    tl = time.perf_counter()
    move_model = Combo(prune=a.prune, maia_size=a.maia_size)
    frontier = FrontierValue()
    say(f"models loaded in {time.perf_counter() - tl:.0f} s")
    with Stockfish(a.sf_nodes, processes=1, per_move=True) as sf:
        kw = dict(me_white=me_white, me_elo=a.elo, opp_elo=a.opp_elo, root_moves=root_moves, counts=counts,
                  explore=a.explore, value_fn=frontier, tail_p=a.tail_p)
        if a.resume:
            with open(a.resume, "rb") as f:
                saved = pickle.load(f)
            ex = Expander.resume(saved["graph"], move_model, sf, params, **kw)
            say(f"resumed {a.resume}: {len(ex.graph)} nodes")
        else:
            ex = Expander(move_model, sf, params, **kw)
        root = ex.graph.nodes[ex.graph.root]
        ex.expand(root)
        game = Game(ex.graph, params)
        sound = sound_root_moves(game)
        say(f"root: {len(root.moves)} legal moves, sound within {a.eps}: {[san([m]) for m in sound]}")
        for m in sound:
            c = ex.graph.child(root, m)
            tg = time.perf_counter()
            n = grow_under(ex, c, max_expansions=a.per_move)
            say(f"grew {san([m])}: {n} expansions, graph {len(ex.graph)} nodes, {time.perf_counter() - tg:.0f} s")
        with open(out / "graph.pkl", "wb") as f:
            pickle.dump({"graph": ex.graph, "root_moves": root_moves, "args": {k: v for k, v in vars(a).items() if k != "fn"}}, f)
        grow_stats = dict(ex.stats)
        grow_seconds = {k: round(v, 1) for k, v in ex.seconds.items()}
        say(f"growth: {grow_stats}, seconds {grow_seconds}")

        # the free choice, deep-checked
        checker = DeepChecker(a.deep_nodes) if a.deep_nodes else None
        cache: dict = {}

        def check(k, m):
            if (k, m) not in cache:
                node = ex.graph.nodes[Game(ex.graph, params).by_key[k][0]]
                cache[(k, m)] = checker.check(list(root_moves) + rel_path(node), m, params.eps(me_white))
            return cache[(k, m)]

        ts = time.perf_counter()
        if checker:
            sols, params_c, results, clog = check_budgets(ex.graph, params, K, budgets, check)
            for line in clog:
                say(line)
        else:
            _, sols = solve(Game(ex.graph, params), K, improve_at=budgets)
            params_c, results = params, {}
        say(f"free solve and deep check: {time.perf_counter() - ts:.0f} s")
        game = Game(ex.graph, params_c)
        root_board = board

        # (b) each sound root move forced
        tf = time.perf_counter()
        forced = forced_curves(ex.graph, params_c, sound, K, budgets)
        say(f"forced curves: {time.perf_counter() - tf:.0f} s")

        # (c) where the difference comes from
        tp = time.perf_counter()
        prof = {}
        for N in a.profile_budgets:
            rows_ = []
            for m in sound:
                fg = Game(ex.graph, replace(params_c, force=frozenset({(root.key, m)})))
                prof_m = theory_profile(fg, forced[m][N].learned, games, obvious=a.obvious,
                                        engine=lambda n: ex.ensure_engine(n) if n.moves else None,
                                        min_reach=a.profile_min_reach)
                rows_.append((san([m]), prof_m))
            prof[N] = rows_
        say(f"theory profiles: {time.perf_counter() - tp:.0f} s, {len(totals)} positions looked up")

        # (d) sensitivity
        sens = []
        for kap in a.kappas:
            for thr in a.thresholds:
                p = replace(params_c, kappa=kap, cost=ObviousCost(thr))
                _, ss = solve(Game(ex.graph, p), K, improve_at=budgets)
                sens.append((kap, thr, {N: (ss[N].value, ss[N].learned.get(root.key, "-"), ss[N].learned) for N in budgets},
                             ss[0].value))
        # side table: pessimism
        side = []
        if a.lam_side:
            for nd in list(ex.graph.nodes):
                if nd.expanded and not nd.me_to_move:
                    ex.ensure_engine(nd)
            for name, p in [("none", params_c), (f"lam {a.lam_side:g}", replace(params_c, lam=a.lam_side)),
                            (f"lam_node {a.lam_side:g}", replace(params_c, lam_node=a.lam_side))]:
                _, ss = solve(Game(ex.graph, p), K, improve_at=budgets)
                side.append((name, {N: (ss[N].value, ss[N].learned.get(root.key, "-"), ss[N].learned) for N in budgets},
                             ss[0].value))
        say(f"sensitivity and side tables done; total engine calls {ex.stats['engine calls']}")

        # rows for the free choice
        free_rows = {N: rows(game, sols, N, ensure=ex.ensure_engine,
                             deep={k: d for (k, m), d in results.items() if sols[N].learned.get(k) == m})
                     for N in sorted(budgets)}
        fm = {N: frontier_mass(game, sols[N].learned) for N in sorted(budgets)}
        if checker:
            checker.close()
        final_stats = dict(ex.stats)
        final_seconds = {k: round(v, 1) for k, v in ex.seconds.items()}
    frontier.close()
    wall = time.perf_counter() - t0

    # -- the report --
    name_of = lambda learned: ", ".join(sorted(san(rel_path(ex.graph.nodes[game.by_key[k][0]]) + [m]).split()[-1] + " in "  # noqa: E731
                                                 + (san(rel_path(ex.graph.nodes[game.by_key[k][0]])) or "root") for k, m in learned.items()))
    rd = ex.graph.nodes[ex.graph.root]
    rep.append(f"# Repertoire builder, round 3: {san([])} {board.fen()}\n")
    rep.append(f"Position after {' '.join(root_moves)}; me {a.me}, {a.elo} against {a.opp_elo}. Move model "
               f"{move_model.name}, frontier value FrontierValue (Lc0 50 nodes, recalibrated), Stockfish "
               f"{a.sf_nodes} nodes per move for soundness, eps {a.eps}, deep check {a.deep_nodes} nodes. "
               f"Opponent: counts from EdgeDB('full'), tempo {a.tempo or 'all'}, rating bands {band or 'all'}, "
               f"over the model with kappa {a.kappa}. Cost: 0 for an obvious move (my model >= {a.obvious}), else 1. "
               f"Pessimism off. Growth: {a.per_move} expansions under each sound fourth move, explore {a.explore}, "
               f"stand-ins below p {a.tail_p}.\n")
    # root move table
    rep.append("## The fourth move: forced, best expected score per budget\n")
    q_root = game.dist[rd.id]
    cnt_root = counts.counts(rd.key, rd.moves)
    hdr = "| 4th move | I play it unaided | engine | games (band) | " + " | ".join(f"N={N}" for N in [0] + sorted(budgets)) + " |"
    rep.append(hdr)
    rep.append("|" + "---|" * (5 + len(budgets)))
    best_free = {N: sols[N].value for N in budgets}
    for m in sound:
        j = rd.moves.index(m)
        vals = [forced[m][N].value for N in [0] + sorted(budgets)]
        fmt = lambda v: "infeasible" if v == -math.inf else f"{v:.4f}"   # noqa: E731
        rep.append(f"| {san([m])} | {rd.p_model[j]:.0%} | {rd.engine[j]:.3f} | {int(cnt_root[j]) if cnt_root is not None else 0} | "
                   + " | ".join(fmt(v) for v in vals) + " |")
    rep.append("| free choice | | | | " + " | ".join(f"{(sols[N].value):.4f}" for N in [0] + sorted(budgets)) + " |\n")
    rep.append("Forced: the fourth move is played whatever, and pays its own cost out of the budget "
               "(1 unless my model plays it at least the threshold), so N=0 is infeasible for a non-obvious move.\n")
    for N in sorted(budgets):
        rep.append(f"## The free choice at N={N}: value {sols[N].value:.4f} (N=0 {sols[0].value:.4f}), "
                   f"{sols[N].cost} non-obvious moves, {len(sols[N].learned)} positions learned\n")
        rep.append(table(free_rows[N], root_board))
        rep.append(f"\nFrontier mass: {', '.join(f'{k} {v:.3f}' for k, v in fm[N].items())}\n")
    rep.append("## The curve (free choice)\n")
    rep.append("```\n" + curve_text(sols) + "\n```\n")
    rep.append("```\n" + steps_text(sols, lambda k: san(rel_path(ex.graph.nodes[game.by_key[k][0]])) or "root") + "\n```\n")
    rep.append("Budget for a price per non-obvious move: " + ", ".join(f"mu {mu:g} -> N={budget_for_price(sols, mu)}"
                                                                     for mu in (0.002, 0.001, 0.0005, 0.0002)) + "\n")
    rep.append("## Where the difference comes from\n")
    for N, rows_ in prof.items():
        rep.append(f"At N={N}, each fourth move forced. Positions: expected number of positions on the way (within the "
                   f"grown graph) with at least that many games in the full table. His loss: the opponent's expected "
                   f"loss per move by Stockfish ({a.sf_nodes} nodes per move), reach-weighted, by how common the "
                   f"position is (mass in brackets). My positions: expected number of my positions on the way; of "
                   f"them, those where Stockfish's best move is not obvious to my model.\n")
        rep.append(profile_text(rows_) + "\n")
    rep.append("## Sensitivity: kappa and the obviousness threshold (free choice)\n")
    rep.append("| kappa | threshold | N=0 | " + " | ".join(f"N={N} (4th move)" for N in sorted(budgets)) + " |")
    rep.append("|" + "---|" * (3 + len(budgets)))
    for kap, thr, d, v0 in sens:
        rep.append(f"| {kap:g} | {thr:g} | {v0:.4f} | " + " | ".join(f"{d[N][0]:.4f} ({san([d[N][1]]) if d[N][1] != '-' else '-'})"
                                                                     for N in sorted(budgets)) + " |")
    base_sets = {N: set(sols[N].learned.items()) for N in budgets}
    rep.append("\nShared positions with the main run (kappa %g, threshold %g):\n" % (a.kappa, a.obvious))
    for kap, thr, d, _ in sens:
        rep.append(f"- kappa {kap:g}, threshold {thr:g}: " + ", ".join(
            f"N={N} {len(set(d[N][2].items()) & base_sets[N])}/{len(d[N][2])}" for N in sorted(budgets)))
    if side:
        rep.append("\n## Side table: pessimism (lam per opponent, lam_node per node)\n")
        rep.append("| pessimism | N=0 | " + " | ".join(f"N={N} (4th move)" for N in sorted(budgets)) + " |")
        rep.append("|" + "---|" * (2 + len(budgets)))
        for name, d, v0 in side:
            rep.append(f"| {name} | {v0:.4f} | " + " | ".join(f"{d[N][0]:.4f} ({san([d[N][1]]) if d[N][1] != '-' else '-'})"
                                                            for N in sorted(budgets)) + " |")
    rep.append("\n## Cost\n")
    rep.append(f"- Graph: {len(ex.graph)} positions, {final_stats.get('expanded', 0)} expanded, "
               f"{final_stats.get('stand-ins', 0)} stand-ins, {final_stats.get('transpositions merged', 0)} transpositions merged.")
    rep.append(f"- Calls during growth: {grow_stats}; seconds {grow_seconds}.")
    rep.append(f"- Calls in all: {final_stats}; seconds {final_seconds}.")
    rep.append(f"- Combo inside: {dict((k, round(v, 1)) for k, v in move_model.seconds.items())}; "
               f"FrontierValue inside: {dict((k, round(v, 1)) for k, v in frontier.seconds.items())}.")
    exp = max(final_stats.get("expanded", 1), 1)
    rep.append(f"- Per expansion: {grow_seconds.get('move model', 0) / exp:.2f} s move model, "
               f"{grow_seconds.get('value', 0) / exp:.2f} s values, {grow_seconds.get('engine', 0) / exp:.2f} s engine; "
               f"per valued position {grow_seconds.get('value', 0) / max(grow_stats.get('value calls', 1), 1):.3f} s.")
    rep.append(f"- Wall time {wall:.0f} s.\n")
    rep.append("## Log\n")
    rep.append("```\n" + "\n".join(log) + "\n```\n")
    text = "\n".join(rep)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(text)
    (out / "run.json").write_text(json.dumps({"args": {k: v for k, v in vars(a).items() if k != "fn"},
                                              "stats": final_stats, "seconds": final_seconds, "wall_s": wall,
                                              "log": log}, indent=1, default=str))
    print(text)
