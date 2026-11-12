"""A solved repertoire graph as files for the web page: `result.json` and one PGN per budget.

    uv run python -m tnh.builder export --graph DIR/graph.pkl --name c4-nf6-g3-e6-bg2-d5

Written to `~/out/tn-hittare/repertoires/<name>/` (`tnh.paths.DATA_ROOT / "repertoires"`).

## `result.json`

```
{
  "meta": {
    "name", "date", "commit",
    "root_fen", "root_moves" (UCI from the start position), "me" ("white" | "black"),
    "me_elo", "opp_elo",
    "params": {"kappa", "obvious_threshold", "eps", "root_eps", "lam", "lam_node", "min_reach",
               "banned": [[pos_hash, uci], ...]},
    "database": {"table", "tempo", "elo_bands"},
    "models": {"move_model", "frontier_value", "engine", "deep_check_nodes"},
    "growth": {...the arguments the graph was grown with...},
    "graph": {"positions", "expanded", "stand_ins"},
    "budgets": [5, 10, 20, 40]
  },
  "solutions": {"<N>": {"value", "cost", "learned": {"<pos_hash>": "<uci>"}}},
  "forced": [{"root_move": uci, "san", "budget": N, "value" (null if infeasible), "cost",
              "learned": {"<pos_hash>": "<uci>"}}],
  "nodes": {
    "<pos_hash>": {
      "epd", "ply" (from the root), "to_move": "me" | "opp",
      "games": games in the full position table that reached it (0 below two),
      "state": "expanded" | "frontier" | "stand_in" | "game_over",
      "value_head": the frontier value's expected score for me at this position
                    (for a stand-in: its parent's, which is what the solve used),
      "frontier_value": the value the solve used when the position is not expanded
                        (equal to value_head unless shrinkage was on), else null,
      "budgets": {"<N>": {"reach", "value" (my expected score from here under that
                  budget's repertoire), "learned" (uci or null)}},
      "moves": [                                   # expanded positions only, every legal move
        {"uci", "san", "child": "<pos_hash>",
         "p_model": the move model's probability at the mover's rating, no database,
         "p_mixed", "db_count": (opponent's positions) the distribution the solve used
                    and the games behind it at his level (null where the position has none),
         "engine", "loss", "sound": (my positions) Stockfish's expected score for me after
                    the move at the search budget, the loss against the best, within eps,
         "obvious", "cost": (my positions) p_model >= the threshold, and the budget it costs,
         "flags": any of "stand_in" (the child got no value call: rarely played),
                  "no_engine_value", "not_exported" (the child is neither expanded nor
                  reached with probability 1e-5 under any budget, so it has no record),
                  "banned" (failed the deep check),
         "value_after": {"<N>": my expected score after the move under that budget's repertoire}}
      ]
    }
  }
}
```

Which positions get a record: every expanded one, and every other one reached with
probability at least 1e-5 under the free choice at some budget or under a forced root
move. A position that occurs at two plies (a repetition or a lost tempo) is one record:
the occurrence with the most reach under the largest budget. Values are for me (the
repertoire's side) everywhere.

## `repertoire_N<budget>.pgn`

One game from the start position: the root moves, then for every learned position its
most likely line from the root and the learned move, as variations. Each learned move
carries `{reach 0.123 value 0.551 cost 1}`: the probability of reaching that position
under the repertoire, my expected score there, and the budget the move costs.
"""

from __future__ import annotations

import datetime as _dt
import io
import json
import math
import subprocess
from collections.abc import Callable
from pathlib import Path

import chess
import chess.pgn
import numpy as np

from tnh.builder.game import Game, Graph, Node
from tnh.builder.output import best_paths, rel_path
from tnh.builder.search import Solution

REACH_MIN = 1e-5


def _f(x) -> float | None:
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else round(x, 6)


def _boards(graph: Graph, root_moves: list[str]) -> Callable[[Node], chess.Board | None]:
    """The board of a node, replayed from the root along its creation path (None when the
    moves are not chess moves, as in a synthetic game)."""
    root = chess.Board()
    for m in root_moves:
        root.push_uci(m)
    cache: dict[int, chess.Board | None] = {}

    def board(n: Node) -> chess.Board | None:
        if n.id not in cache:
            b = root.copy(stack=False)
            try:
                for m in rel_path(n):
                    b.push_uci(m)
            except (ValueError, AssertionError):
                b = None
            cache[n.id] = b
        return cache[n.id]

    return board


def export(game: Game, sols: list[Solution], budgets: list[int], *, root_moves: list[str], meta: dict,
           forced: dict[str, list[Solution]] | None = None, games: Callable[[Node], int] = lambda n: 0,
           obvious: float | None = None, board_of: Callable[[Node], chess.Board | None] | None = None,
           ply_offset: int = 0) -> dict:
    """The `result.json` object for a solved game (see the module docstring). `board_of` and
    `ply_offset` are for graphs whose nodes carry absolute plies and no paths (the engine's)."""
    g = game.graph
    board = board_of or _boards(g, root_moves)
    per = {N: (game.values(sols[N].learned), game.reach(sols[N].learned), sols[N].learned) for N in budgets}
    reach_any = np.zeros(len(g.nodes))
    for _, R, _ in per.values():
        reach_any = np.maximum(reach_any, R)
    for m, fs in (forced or {}).items():
        fg = Game(g, _with_force(game, m))
        for N in budgets:
            if fs[N].value != -math.inf:
                reach_any = np.maximum(reach_any, fg.reach(fs[N].learned))
    big = max(budgets)
    chosen: dict[int, int] = {}                      # pos_hash -> node id
    for i in game.order:
        n = g.nodes[i]
        if not (n.children is not None or reach_any[i] >= REACH_MIN):
            continue
        j = chosen.get(n.key)
        if j is None or per[big][1][i] > per[big][1][j]:
            chosen[n.key] = i
    exported = set(chosen)
    banned = set(game.params.banned)
    nodes = {}
    for key, i in chosen.items():
        n = g.nodes[i]
        b = board(n)
        if n.children is not None and n.moves:
            state = "expanded"
        elif not n.moves:
            state = "game_over"
        elif n.info.get("stand_in"):
            state = "stand_in"
        else:
            state = "frontier"
        rec = {
            "epd": b.epd() if b is not None else None,
            "ply": n.ply - ply_offset,
            "to_move": "me" if n.me_to_move else "opp",
            "games": int(games(n)),
            "state": state,
            "value_head": _f(n.value),
            "frontier_value": _f(game.leaf[i]) if i in game.leaf else None,
            "budgets": {str(N): {"reach": _f(R[i]), "value": _f(V[i]),
                                 "learned": L.get(n.key) if n.me_to_move else None}
                        for N, (V, R, L) in per.items()},
        }
        if state == "expanded":
            rec["moves"] = _moves(game, i, b, per, exported, banned, obvious)
        nodes[str(key)] = rec
    out = {
        "meta": {**meta, "budgets": list(budgets),
                 "graph": {"positions": len(g.nodes), "expanded": sum(1 for n in g.nodes if n.children is not None),
                           "stand_ins": sum(1 for n in g.nodes if n.info.get("stand_in")),
                           "records": len(nodes)}},
        "solutions": {str(N): {"value": _f(sols[N].value), "cost": sols[N].cost,
                               "learned": {str(k): m for k, m in sols[N].learned.items()}} for N in budgets},
        "forced": [],
        "nodes": nodes,
    }
    root = g.nodes[g.root]
    rb = board(root)
    for m, fs in (forced or {}).items():
        for N in budgets:
            out["forced"].append({"root_move": m, "san": rb.san(chess.Move.from_uci(m)) if rb is not None else m,
                                  "budget": N, "value": _f(fs[N].value), "cost": fs[N].cost,
                                  "learned": {str(k): mv for k, mv in fs[N].learned.items()}})
    return out


def _with_force(game: Game, move: str):
    from dataclasses import replace
    root = game.graph.nodes[game.graph.root]
    return replace(game.params, force=frozenset({(root.key, move)}))


def _moves(game: Game, i: int, b: chess.Board | None, per: dict, exported: set, banned: set,
           obvious: float | None) -> list[dict]:
    g = game.graph
    n = g.nodes[i]
    d = game.dist[i]
    sound = set(game.sound.get(i, np.zeros(0, dtype=np.int64)).tolist())
    eng = n.engine
    best = float(np.nanmax(eng)) if eng is not None and not np.all(np.isnan(eng)) else math.nan
    rows = []
    for j, (m, c) in enumerate(zip(n.moves, n.children)):
        cn = g.nodes[c]
        flags = []
        if cn.info.get("stand_in"):
            flags.append("stand_in")
        if cn.key not in exported:
            flags.append("not_exported")
        if (n.key, m) in banned:
            flags.append("banned")
        r = {"uci": m, "san": b.san(chess.Move.from_uci(m)) if b is not None else m, "child": str(cn.key),
             "p_model": _f(n.p_model[j])}
        if n.me_to_move:
            v = math.nan if eng is None else float(eng[j])
            if math.isnan(v):
                flags.append("no_engine_value")
            r.update({"engine": _f(v), "loss": _f(best - v), "sound": j in sound,
                      "obvious": bool(obvious is not None and n.p_model[j] >= obvious),
                      "cost": game.move_cost(i, j)})
        else:
            r.update({"p_mixed": _f(d[j]), "db_count": None if n.counts is None else int(n.counts[j])})
        r["flags"] = flags
        r["value_after"] = {str(N): _f(V[c]) for N, (V, _, _) in per.items()}
        rows.append(r)
    return rows


def pgn_for(game: Game, sol: Solution, root_moves: list[str], headers: dict[str, str] | None = None) -> str:
    """The learned lines of one solution as one PGN with variations."""
    g = game.graph
    V, R = game.values(sol.learned), game.reach(sol.learned)
    paths = best_paths(game, sol.learned)
    pg = chess.pgn.Game()
    for k, v in (headers or {}).items():
        pg.headers[k] = v
    node = pg
    for m in root_moves:
        node = node.add_variation(chess.Move.from_uci(m))
    root = node
    items = []
    for k, mv in sol.learned.items():
        ids = game.by_key[k]
        i = max(ids, key=lambda x: R[x])
        n = g.nodes[i]
        items.append((paths.get(i, rel_path(n)), mv, float(sum(R[x] for x in ids)), float(V[i]),
                      game.move_cost(i, n.moves.index(mv)) if k not in game.forced else game.forced_cost))
    for path, mv, r, v, c in sorted(items, key=lambda t: (len(t[0]), -t[2])):
        node = root
        for m in path + [mv]:
            move = chess.Move.from_uci(m)
            node = node.variation(move) if node.has_variation(move) else node.add_variation(move)
        node.comment = f"reach {r:.3f} value {v:.3f} cost {c}"
    s = io.StringIO()
    print(pg, file=s, end="\n\n")
    return s.getvalue()


def git_commit() -> str:
    here = Path(__file__).resolve().parent
    r = subprocess.run(["git", "-C", str(here), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def write(out_dir: Path, result: dict, pgns: dict[int, str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, separators=(",", ":")))
    for N, text in pgns.items():
        (out_dir / f"repertoire_N{N}.pgn").write_text(text)


def today() -> str:
    return _dt.date.today().isoformat()


def run(a) -> None:
    """The command: reload a study's graph, solve it again exactly as the study did (deep check
    included, which is deterministic), and write the files."""
    import pickle
    import time

    from tnh.builder.deepcheck import DeepChecker
    from tnh.builder.game import ObviousCost, Params
    from tnh.builder.search import solve
    from tnh.builder.study import check_budgets, forced_curves, sound_root_moves
    from tnh.data.edges import EdgeDB
    from tnh.paths import DATA_ROOT

    t0 = time.perf_counter()
    with open(a.graph, "rb") as f:
        saved = pickle.load(f)
    graph, root_moves, args = saved["graph"], saved["root_moves"], saved["args"]
    params = Params(kappa=args["kappa"], eps_white=args["eps"], eps_black=args["eps"], min_reach=args["min_reach"],
                    cost=ObviousCost(args["obvious"]), root_eps=args.get("root_eps"))
    budgets = sorted(args["budgets"])
    K = max(budgets)
    me_white = args["me"] == "white"
    deep_nodes = args["deep_nodes"] if a.deep_nodes is None else a.deep_nodes
    board0 = chess.Board()
    for m in root_moves:
        board0.push_uci(m)
    if deep_nodes:
        checker = DeepChecker(deep_nodes)
        cache: dict = {}
        g0 = Game(graph, params)

        def check(k, m):
            if (k, m) not in cache:
                n = graph.nodes[g0.by_key[k][0]]
                cache[(k, m)] = checker.check(list(root_moves) + rel_path(n), m, params.eps_at(me_white, n.id == graph.root))
            return cache[(k, m)]
        sols, params, _, log = check_budgets(graph, params, K, set(budgets), check)
        checker.close()
    else:
        _, sols = solve(Game(graph, params), K, improve_at=set(budgets))
        log = ["no deep check"]
    for line in log:
        print(line, flush=True)
    game = Game(graph, params)
    sound = sound_root_moves(game)
    forced = forced_curves(graph, params, sound, K, set(budgets))
    db = EdgeDB("full")
    totals: dict = {}

    def games(n) -> int:
        if n.key not in totals:
            t = db.total(n.key)
            totals[n.key] = int(t["n"]) if t else 0
        return totals[n.key]

    meta = {
        "name": a.name, "date": today(), "commit": git_commit(),
        "root_fen": board0.fen(), "root_moves": list(root_moves), "me": args["me"],
        "me_elo": args["elo"], "opp_elo": args["opp_elo"],
        "params": {"kappa": params.kappa, "obvious_threshold": args["obvious"], "eps": args["eps"],
                   "root_eps": params.root_eps, "lam": params.lam, "lam_node": params.lam_node,
                   "min_reach": params.min_reach, "banned": [[str(k), m] for k, m in sorted(params.banned, key=str)]},
        "database": {"table": "EdgeDB('full')", "tempo": args.get("tempo"),
                     "elo_bands": None if args.get("no_band") else [args.get("band_lo"), args.get("band_hi")]},
        "models": {"move_model": f"combo-maia3-{args.get('maia_size')}-lc0-{args.get('prune')}",
                   "frontier_value": "FrontierValue (Lc0 50 nodes, WDLCalibrationElo, recalibrated)",
                   "engine": f"stockfish {args['sf_nodes']} nodes per move", "deep_check_nodes": deep_nodes},
        "growth": {k: args.get(k) for k in ("per_move", "explore", "tail_p", "stable_rounds", "stable_per_round",
                                            "root_candidates")},
    }
    result = export(game, sols, budgets, root_moves=list(root_moves), meta=meta, forced=forced, games=games,
                    obvious=args["obvious"])
    pgns = {N: pgn_for(game, sols[N], list(root_moves),
                       {"Event": f"{a.name}, N={N}", "White": "me" if me_white else "opponent",
                        "Black": "opponent" if me_white else "me",
                        "Annotator": f"tnh.builder, value {sols[N].value:.4f}, cost {sols[N].cost}"})
            for N in budgets}
    out = Path(a.out_root or DATA_ROOT / "repertoires") / a.name
    write(out, result, pgns)
    size = (out / "result.json").stat().st_size
    print(f"wrote {out}: result.json {size / 1e6:.1f} MB, {len(result['nodes'])} records, "
          f"{len(totals)} database lookups; values: "
          + ", ".join(f"N={N} {sols[N].value:.4f}" for N in budgets) + f"; {time.perf_counter() - t0:.0f} s")
