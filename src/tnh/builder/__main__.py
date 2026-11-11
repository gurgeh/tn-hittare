"""The repertoire builder from the command line.

    uv run python -m tnh.builder synth --seed 1 -N 8            # a random synthetic game
    uv run python -m tnh.builder synth --dag --pool 4 -N 8
    uv run python -m tnh.builder gap --seeds 100               # DP against enumeration, trees and DAGs
    uv run python -m tnh.builder chess --moves e2e4 --me black -N 10 --out DIR   # Maia-3 + Stockfish

`chess` runs models; anything longer than a minute goes through scripts/tnh-run.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from tnh.builder.game import Game, Params
from tnh.builder.output import curve_json, curve_text, pgn, rows, table
from tnh.builder.search import DP, brute_force, candidates, improve, is_tree, repair, solve
from tnh.builder.synth import random_game


def _params(a) -> Params:
    return Params(kappa=a.kappa, lam=a.lam, eps_white=a.eps, eps_black=a.eps, min_reach=a.min_reach)


def cmd_synth(a) -> None:
    rng = np.random.default_rng(a.seed)
    g = random_game(rng, depth=a.depth, branching=tuple(a.branching), dag=a.dag, pool=a.pool)
    game = Game(g, _params(a))
    t = time.perf_counter()
    dp, sols = solve(game, a.N)
    dt = time.perf_counter() - t
    print(f"{len(game.order)} nodes, {len(game.sound)} of my positions, tree: {is_tree(game)}, "
          f"solved to N={a.N} in {dt:.2f} s ({dp.convolution_entries} convolution entries)\n")
    print(curve_text(sols))
    s = sols[a.N]
    print(f"\nN={a.N}: learned {s.cost} positions" + (f", local search: {s.local_steps}" if s.local_steps else ""))
    print(table(rows(game, s)))


def _gap_one(game: Game, N_max: int) -> list[dict]:
    e0 = game.evaluate({})
    _, sols = solve(game, N_max)
    dp = DP(game, N_max)
    out = []
    for N in range(1, N_max + 1):
        bf, _ = brute_force(game, N)
        if bf - e0 < 1e-9:
            continue
        raw = repair(game, dp.backtrack(N))
        greedy, _ = improve(game, {}, N)
        share = lambda v: (v - e0) / (bf - e0)
        out.append({"N": N, "dp_claimed": share(sols[N].dp_value), "dp_repaired": share(game.evaluate(raw)),
                    "final": share(sols[N].value), "greedy": share(game.evaluate(greedy))})
    return out


def cmd_gap(a) -> None:
    kinds = [("tree", None)] + [(f"dag pool {p}", p) for p in a.pools]
    print(f"Share of the optimal gain over N=0 (brute force), N=1..{a.N}, {a.seeds} seeds per kind, "
          f"depth {a.tree_depth} for trees, {a.depth} for DAGs.")
    print("dp_claimed: the DP's own value. dp_repaired: its learned set made consistent, evaluated exactly.")
    print("final: after the overshoot starts and local search. greedy: local search from nothing, for comparison.\n")
    print(f"{'kind':14s} {'cases':>5s}  " + "  ".join(f"{c:>22s}" for c in ("dp_claimed", "dp_repaired", "final", "greedy")))
    for name, pool in kinds:
        recs, skipped = [], 0
        for s in range(a.seeds):
            rng = np.random.default_rng(10_000 + s)
            g = random_game(rng, depth=a.depth if pool else a.tree_depth, branching=(2, 3),
                            dag=pool is not None, pool=pool)
            game = Game(g, Params(eps_white=a.eps))
            if len(candidates(game)) > a.max_cand:
                skipped += 1
                continue
            recs += _gap_one(game, a.N)
        if not recs:
            print(f"{name:14s}     0  ({skipped} games skipped: more than {a.max_cand} positions to enumerate)")
            continue
        cols = []
        for c in ("dp_claimed", "dp_repaired", "final", "greedy"):
            x = np.array([r[c] for r in recs])
            cols.append(f"mean {x.mean():.4f} min {x.min():.3f}")
        print(f"{name:14s} {len(recs):5d}  " + "  ".join(f"{c:>22s}" for c in cols)
              + f"   ({skipped} games skipped: more than {a.max_cand} positions to enumerate)", flush=True)


def cmd_chess(a) -> None:
    import chess
    import torch

    from tnh.builder.expand import Expander, frontier_importance
    from tnh.models.maia3 import Elo, Maia3
    from tnh.models.stockfish import Stockfish

    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    maia = Maia3(a.size, Elo.offset(0), device=a.device, local_files_only=True, batch_size=64)
    params = _params(a)
    me_white = a.me == "white"
    with Stockfish(a.sf_nodes, processes=1, per_move=True) as sf:
        ex = Expander(maia, sf, params, me_white=me_white, me_elo=a.elo, opp_elo=a.opp_elo,
                      root_moves=a.moves.split(), explore=a.explore)
        log = []
        n = ex.grow(expand_reach=a.expand_reach, max_expansions=a.max_expansions)
        log.append(f"grow: {n} expansions, {len(ex.graph)} nodes, {dict(ex.stats)}")
        print(log[-1], flush=True)
        for r in range(a.refine_rounds + 1):
            game = Game(ex.graph, params)
            t = time.perf_counter()
            dp, sols = solve(game, a.N)
            log.append(f"solve {r}: {len(game.order)} nodes, value at N={a.N} {sols[a.N].value:.4f} "
                       f"(N=0 {sols[0].value:.4f}), {time.perf_counter() - t:.1f} s")
            print(log[-1], flush=True)
            if r < a.refine_rounds:
                k = ex.refine(sols[a.N].learned, n=a.refine_n)
                log.append(f"refine {r}: {k} expansions, {len(ex.graph)} nodes")
                print(log[-1], flush=True)
        stats = dict(ex.stats)
        seconds = {k: round(v, 1) for k, v in ex.seconds.items()}
        sf_stats = dict(sf.stats)
    board = chess.Board()
    for m in a.moves.split():
        board.push_uci(m)
    sol = sols[a.N]
    rs = rows(game, sol)
    headers = {"Event": f"tnh.builder repertoire, N={a.N}", "White": f"{'me' if me_white else 'opponent'}",
               "Black": f"{'opponent' if me_white else 'me'}",
               "Annotator": f"maia3-{a.size} {a.elo} vs {a.opp_elo}, stockfish {a.sf_nodes} nodes/move, "
                            f"eps {a.eps}, lam {a.lam}"}
    (out / "repertoire.pgn").write_text(pgn(rs, a.moves.split(), headers))
    tab = table(rs, board)
    (out / "table.md").write_text(tab + "\n")
    (out / "curve.json").write_text(curve_json(sols))
    imp = frontier_importance(game, sol.learned)[:15]
    frontier = [{"path": " ".join(ex.graph.nodes[i].path), "weight": w} for i, w in imp]
    wall = time.perf_counter() - t0
    summary = {"args": vars(a), "stats": stats, "seconds": seconds, "stockfish": sf_stats, "wall_s": wall,
               "nodes": len(ex.graph), "log": log, "frontier_top": frontier}
    (out / "run.json").write_text(json.dumps(summary, indent=1, default=str))
    print()
    print(curve_text(sols))
    print()
    print(tab)
    print(f"\nmodel calls: {stats.get('move model calls', 0)} move model, {stats.get('engine calls', 0)} engine; "
          f"seconds {seconds}; wall {wall:.0f} s")
    print("frontier positions that weigh most:", *(f"{f['weight']:.3f} {f['path']}" for f in frontier[:5]), sep="\n  ")
    print(f"written to {out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tnh.builder")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("-N", type=int, default=8)
        p.add_argument("--eps", type=float, default=0.03)
        p.add_argument("--lam", type=float, default=0.0)
        p.add_argument("--kappa", type=float, default=math.inf)
        p.add_argument("--min-reach", type=float, default=0.0)

    p = sub.add_parser("synth")
    common(p)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--branching", type=int, nargs=2, default=(2, 4))
    p.add_argument("--dag", action="store_true")
    p.add_argument("--pool", type=int, default=6)
    p.set_defaults(fn=cmd_synth)

    p = sub.add_parser("gap")
    p.add_argument("-N", type=int, default=3)
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--pools", type=int, nargs="*", default=[3, 5, 8])
    p.add_argument("--max-cand", type=int, default=16)
    p.add_argument("--tree-depth", type=int, default=4)
    p.set_defaults(fn=cmd_gap)

    p = sub.add_parser("chess")
    common(p)
    p.add_argument("--moves", default="e2e4", help="UCI moves from the start to the root")
    p.add_argument("--me", choices=["white", "black"], required=True)
    p.add_argument("--elo", type=int, default=2000)
    p.add_argument("--opp-elo", type=int, default=2000)
    p.add_argument("--size", default="5m")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--sf-nodes", type=int, default=2000, help="Stockfish nodes per legal move")
    p.add_argument("--expand-reach", type=float, default=0.005)
    p.add_argument("--explore", type=float, default=0.25)
    p.add_argument("--max-expansions", type=int, default=300)
    p.add_argument("--refine-rounds", type=int, default=2)
    p.add_argument("--refine-n", type=int, default=40)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_chess)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
