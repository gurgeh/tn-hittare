"""The repertoire builder from the command line.

    uv run python -m tnh.builder synth --seed 1 -N 8            # a random synthetic game
    uv run python -m tnh.builder synth --dag --pool 4 -N 8
    uv run python -m tnh.builder gap --seeds 100               # DP against enumeration, trees and DAGs
    uv run python -m tnh.builder chess --moves e2e4 --me black -N 10 --out DIR   # Maia-3 + Stockfish

`chess` runs models; anything longer than a minute goes through scripts/tnh-run.
"""

from __future__ import annotations

import os

# One thread for BLAS and OpenMP, set before numpy loads: the machine's cores are
# budgeted per process, and a multi-threaded BLAS on long vectors here used four
# cores to run slower than one (2026-11-12).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from tnh.builder.game import Game, Params
from tnh.builder.output import curve_json, curve_text, pgn, rel_path, rows, steps_text, table
from tnh.builder.search import DP, brute_force, candidates, improve, is_tree, repair, solve
from tnh.builder.synth import random_game


def _params(a) -> Params:
    return Params(kappa=a.kappa, lam=a.lam, lam_node=a.lam_node, eps_white=a.eps, eps_black=a.eps,
                  min_reach=a.min_reach, shrink=a.shrink, shrink_to=a.shrink_to)


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
    print(table(rows(game, sols, a.N)))
    print()
    print(steps_text(sols))


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


def cmd_curse(a) -> None:
    from tnh.builder.drift import curse_experiment
    variants = {"no shrink": {}}
    for x in a.shrinks:
        variants[f"parent {x:g}"] = {"shrink": x}
    variants["engine 0.5"] = {"shrink": 0.5, "shrink_to": "engine"}
    variants["until stable"] = {"stable": {"per_round": 10, "tol": 0.002, "max_rounds": 40}}
    variants["stable + parent 0.5"] = {"shrink": 0.5, "stable": {"per_round": 10, "tol": 0.002, "max_rounds": 40}}
    print(f"Winner's curse: full tree depth {a.depth}, frontier at ply {a.visible} read as truth + N(0, sigma), "
          f"N={a.N}, {a.seeds} seeds. Gains over N=0.\n")
    for sigma in a.sigmas:
        res = curse_experiment(a.seeds, sigma, variants, N=a.N, depth=a.depth, visible=a.visible)
        print(f"sigma {sigma:g}")
        print(f"  {'remedy':20s} {'predicted':>10s} {'true':>8s} {'optimism':>9s} {'(se)':>7s} {'expanded':>9s}")
        for name, r in res.items():
            d = r[:, 0] - r[:, 1]
            print(f"  {name:20s} {r[:, 0].mean():+10.4f} {r[:, 1].mean():+8.4f} {d.mean():+9.4f} "
                  f"({d.std() / np.sqrt(len(d)):.4f}) {r[:, 2].mean():9.1f}")
        print()


def _solve_report(game: Game, N: int, label: str, log: list[str]) -> list:
    t = time.perf_counter()
    _, sols = solve(game, N)
    log.append(f"{label}: {len(game.order)} nodes, value at N={N} {sols[N].value:.4f} (N=0 {sols[0].value:.4f}), "
               f"{time.perf_counter() - t:.1f} s")
    print(log[-1], flush=True)
    return sols


def cmd_chess(a) -> None:
    import pickle
    from dataclasses import replace

    import chess
    import torch

    from tnh.builder.deepcheck import DeepChecker, check_repertoire
    from tnh.builder.drift import drift, drift_text
    from tnh.builder.expand import Expander
    from tnh.builder.output import budget_for_price, frontier_mass, steps_text
    from tnh.models.maia3 import Elo, Maia3
    from tnh.models.stockfish import Stockfish

    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    maia = Maia3(a.size, Elo.offset(0), device=a.device, local_files_only=True, batch_size=64)
    params = _params(a)
    me_white = a.me == "white"
    root_moves = a.moves.split()
    board = chess.Board()
    for m in root_moves:
        board.push_uci(m)
    san = lambda path: " ".join(_san(board, list(path)))
    log: list[str] = []
    report: dict = {}
    with Stockfish(a.sf_nodes, processes=1, per_move=True) as sf:
        if a.resume:
            with open(a.resume, "rb") as f:
                saved = pickle.load(f)
            if saved["root_moves"] != root_moves:
                raise ValueError(f"saved graph has root {saved['root_moves']}, not {root_moves}")
            ex = Expander.resume(saved["graph"], maia, sf, params, me_white=me_white, me_elo=a.elo,
                                 opp_elo=a.opp_elo, root_moves=root_moves, explore=a.explore)
            log.append(f"resumed {a.resume}: {len(ex.graph)} nodes; calls to complete it: {dict(ex.stats)}")
            print(log[-1], flush=True)
        else:
            ex = Expander(maia, sf, params, me_white=me_white, me_elo=a.elo, opp_elo=a.opp_elo,
                          root_moves=root_moves, explore=a.explore)
            n = ex.grow(expand_reach=a.expand_reach, max_expansions=a.max_expansions)
            log.append(f"grow: {n} expansions, {len(ex.graph)} nodes, {dict(ex.stats)}")
            print(log[-1], flush=True)
        for r in range(a.refine_rounds + 1):
            sols = _solve_report(Game(ex.graph, params), a.N, f"solve {r}", log)
            if r < a.refine_rounds:
                k = ex.refine(sols[a.N].learned, n=a.refine_n)
                log.append(f"refine {r}: {k} expansions, {len(ex.graph)} nodes")
                print(log[-1], flush=True)

        if a.stable_rounds:
            from tnh.builder.drift import solve_until_stable
            st = solve_until_stable(ex, a.N, per_round=a.drift_per_round, tol=a.stable_tol, max_rounds=a.stable_rounds)
            lines = [f"until stable (tol {a.stable_tol}, {a.drift_per_round} expansions per round): "
                     f"{'converged' if st.converged else 'NOT converged'} after {st.rounds} rounds",
                     "round  nodes   choice's value   after growing below it"]
            lines += [f"{r:5d}  {nn:5d}   {v:.4f}           {after:.4f}" for r, v, after, nn in st.history]
            report["stable"] = "\n".join(lines)
            print(report["stable"], flush=True)
            sols = st.sols

        if a.compare:
            # the same graph under each pessimism; lam_node needs the engine at every opponent's node
            for nd in list(ex.graph.nodes):
                if nd.expanded and not nd.me_to_move:
                    ex.ensure_engine(nd)
            variants = {"lam 0": replace(params, lam=0.0, lam_node=0.0),
                        f"lam {a.compare:g} per opponent": replace(params, lam=a.compare, lam_node=0.0),
                        f"lam {a.compare:g} per node": replace(params, lam=0.0, lam_node=a.compare),
                        "base, shrink 0.5 to parent": replace(params, shrink=0.5)}
            cmp = {}
            for name, p in variants.items():
                g = Game(ex.graph, p)
                _, vs = solve(g, a.N)
                cmp[name] = {"value": vs[a.N].value, "v0": vs[0].value,
                             "learned": {san(rel_path(ex.graph.nodes[g.by_key[k][0]]) + [m]): m
                                         for k, m in vs[a.N].learned.items()},
                             "keys": set(vs[a.N].learned.items())}
            base_keys = cmp[f"lam {a.compare:g} per opponent"]["keys"]
            lines = [f"Same graph ({len(ex.graph)} nodes), N={a.N}: each variant's choice. 'shared' counts "
                     f"(position, move) pairs in common with lam {a.compare:g} per opponent."]
            for name, c in cmp.items():
                lines.append(f"- {name}: value {c['value']:.4f} (N=0 {c['v0']:.4f}), shared {len(c['keys'] & base_keys)}"
                             f"/{len(c['keys'])}: " + "; ".join(sorted(c["learned"])))
            report["compare"] = "\n".join(lines)
            print(report["compare"], flush=True)

        if a.drift_rounds:
            fixed = sols[a.N].learned
            rs = drift(ex, fixed, a.N, a.drift_rounds, a.drift_per_round,
                       variants={"shrink .5 parent": replace(params, shrink=0.5)})
            report["drift"] = drift_text(rs)
            print(report["drift"], flush=True)

        final_game = Game(ex.graph, params)
        if a.deep_nodes:
            with DeepChecker(a.deep_nodes) as dc:
                check = lambda k, m: dc.check(list(root_moves) + rel_path(ex.graph.nodes[final_game.by_key[k][0]]), m,
                                              params.eps(me_white))
                chk = check_repertoire(ex.graph, params, a.N, a.N, check)
            log += chk.log
            print("\n".join(chk.log), flush=True)
            sols = chk.sols
            final_game = Game(ex.graph, chk.params)
            deep = {k: d for (k, m), d in chk.results.items() if sols[a.N].learned.get(k) == m}
            report["deep"] = [{"line": san(rel_path(ex.graph.nodes[final_game.by_key[k][0]]) + [m]), **d}
                              for (k, m), d in chk.results.items()]
        else:
            sols = _solve_report(final_game, a.N, "final solve", log)
            deep = None
        rs = rows(final_game, sols, a.N, ensure=ex.ensure_engine, deep=deep)
        stats = dict(ex.stats)
        seconds = {k: round(v, 1) for k, v in ex.seconds.items()}
        sf_stats = dict(sf.stats)

    sol = sols[a.N]
    names = {k: san(rel_path(ex.graph.nodes[final_game.by_key[k][0]]) + [m]) for s_ in sols for k, m in s_.learned.items()}
    headers = {"Event": f"tnh.builder repertoire, N={a.N}", "White": f"{'me' if me_white else 'opponent'}",
               "Black": f"{'opponent' if me_white else 'me'}",
               "Annotator": f"maia3-{a.size} {a.elo} vs {a.opp_elo}, stockfish {a.sf_nodes} nodes/move, "
                            f"eps {a.eps}, lam {a.lam}, lam_node {a.lam_node}, shrink {a.shrink}"}
    (out / "repertoire.pgn").write_text(pgn(rs, root_moves, headers))
    tab = table(rs, board)
    (out / "table.md").write_text(tab + "\n")
    (out / "curve.json").write_text(curve_json(sols))
    fm = frontier_mass(final_game, sol.learned)
    prices = {mu: budget_for_price(sols, mu) for mu in a.mu}
    with open(out / "graph.pkl", "wb") as f:
        pickle.dump({"graph": ex.graph, "root_moves": root_moves,
                     "args": {k: v for k, v in vars(a).items() if k != "fn"}}, f)
    wall = time.perf_counter() - t0
    summary = {"args": vars(a), "stats": stats, "seconds": seconds, "stockfish": sf_stats, "wall_s": wall,
               "nodes": len(ex.graph), "log": log, "frontier_mass": fm, "budget_for_price": prices,
               **{k: v for k, v in report.items()}}
    (out / "run.json").write_text(json.dumps(summary, indent=1, default=str))
    print()
    print(curve_text(sols))
    print()
    print(steps_text(sols, lambda k: names.get(k, str(k))))
    print("\nbudget for a price per position:", ", ".join(f"mu {mu:g} -> N={n}" for mu, n in prices.items()))
    print()
    print(tab)
    print(f"\nfrontier mass under the repertoire: {fm}")
    print(f"model calls: {stats.get('move model calls', 0)} move model, {stats.get('engine calls', 0)} engine; "
          f"seconds {seconds}; wall {wall:.0f} s")
    print(f"written to {out}")


def _san(board, moves: list[str]) -> list[str]:
    b = board.copy(stack=False)
    out = []
    for m in moves:
        import chess
        mv = chess.Move.from_uci(m)
        out.append(b.san(mv))
        b.push(mv)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tnh.builder")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("-N", type=int, default=8)
        p.add_argument("--eps", type=float, default=0.03)
        p.add_argument("--lam", type=float, default=0.0, help="share of opponents that play the engine's best move")
        p.add_argument("--lam-node", type=float, default=0.0, help="per-node share on the engine's best move")
        p.add_argument("--shrink", type=float, default=0.0)
        p.add_argument("--shrink-to", default="parent", choices=["parent", "engine"])
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

    p = sub.add_parser("curse")
    p.add_argument("-N", type=int, default=4)
    p.add_argument("--seeds", type=int, default=40)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--visible", type=int, default=4)
    p.add_argument("--sigmas", type=float, nargs="*", default=[0.0, 0.04, 0.08])
    p.add_argument("--shrinks", type=float, nargs="*", default=[0.3, 0.5, 0.7, 0.9])
    p.set_defaults(fn=cmd_curse)

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
    p.add_argument("--compare", type=float, default=0.0,
                   help="also solve the grown graph with lam 0, this lam per opponent, per node, and shrink 0.5")
    p.add_argument("--drift-rounds", type=int, default=0)
    p.add_argument("--drift-per-round", type=int, default=40)
    p.add_argument("--resume", help="a graph.pkl from an earlier run; skips growing")
    p.add_argument("--stable-rounds", type=int, default=0, help="remedy (a): grow below the choice until it holds")
    p.add_argument("--stable-tol", type=float, default=0.001)
    p.add_argument("--deep-nodes", type=int, default=0, help="re-check the learned moves at this engine budget")
    p.add_argument("--mu", type=float, nargs="*", default=[0.001, 0.0005, 0.0002])
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_chess)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
