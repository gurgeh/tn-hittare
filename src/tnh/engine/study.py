"""The first real run of the engine: one position, the settings swept on one graph at a few sizes.

    scripts/tnh-run engine-c4 6 uv run python -m tnh.engine.study --report ~/out/tn-hittare/reports/engine_round4_c4.md

Grows the graph in stages (2,000, 10,000, 50,000 expanded positions, as far as the search
time allows), and at each stage solves the same graph again for each price and each cap
(caps from a grid of prices: the smallest price whose repertoire fits). Then the history
check on transpositions the search met, Stockfish's loss for every learned move in the
last stage's repertoires (beside the move, not in the loop), the export for the web page,
and the report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path  # noqa: E402

import chess  # noqa: E402
import numpy as np  # noqa: E402

from tnh.engine.search import Engine, Settings  # noqa: E402

C4 = "c2c4 g8f6 g2g3 e7e6 f1g2 d7d5".split()


def rss_gb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1e6
    return math.nan


def gpu_free_mb() -> int:
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True)
    used, total = (int(x) for x in r.stdout.split(",")[:2])
    return total - used


def san_line(board: chess.Board, moves: list[str]) -> str:
    b, out = board.copy(stack=False), []
    for m in moves:
        mv = chess.Move.from_uci(m)
        out.append(b.san(mv))
        b.push(mv)
    return " ".join(out)


def snapshot(eng: Engine, board: chess.Board, label: str, k: int) -> dict:
    s = eng.last
    root = eng.graph.nodes[eng.graph.root]
    lines = eng.lines(k)
    return {
        "label": label, "price": s.mu, "value": s.value, "net": s.net, "cost": s.cost,
        "learned": len(s.learned), "mass": eng.mass(),
        "root_move": s.learned.get(root.key),
        "lines": [{"san": san_line(board, l.moves), "uci": l.moves, "value": l.value, "net": l.net,
                   "nonobvious": l.nonobvious, "reach": l.reach,
                   "flags": [("L" if f else "M") if p is None else f"{p:.3f}" for p, f in zip(l.probs, l.learned)]}
                  for l in lines],
        "learned_set": {str(kk): m for kk, m in s.learned.items()},
    }


def sweep(eng: Engine, board: chess.Board, prices, caps, grid, k: int) -> dict:
    out = {"prices": [], "caps": []}
    keep = (eng.s.price, eng.s.cap)
    for p in prices:
        eng.set_settings(price=p, cap=0)
        eng.solve()
        out["prices"].append(snapshot(eng, board, f"price {p:g}", k))
    found = []
    for p in grid:
        eng.set_settings(price=p, cap=0)
        eng.solve()
        found.append((p, eng.last.cost, snapshot(eng, board, f"price {p:g}", k)))
    for c in caps:
        fit = [x for x in found if x[1] <= c]
        if not fit:
            out["caps"].append({"label": f"cap {c}", "none": True})
            continue
        p, cost, snap = min(fit, key=lambda t: t[0])
        snap = dict(snap)
        snap["label"] = f"cap {c}"
        snap["cap"] = c
        out["caps"].append(snap)
    eng.set_settings(price=keep[0], cap=keep[1])
    eng.solve()
    return out


def root_table(eng: Engine, board: chess.Board) -> list[dict]:
    g = eng.graph
    root = g.nodes[g.root]
    prior = eng._prior(root)
    pol = root.info.get("policy")
    rows = []
    for j, (m, c) in enumerate(zip(root.moves, root.children)):
        cn = g.nodes[c]
        rows.append({"san": board.san(chess.Move.from_uci(m)), "uci": m, "p_model": float(root.info["p_first"][j]),
                     "policy": None if pol is None else float(pol[j]), "prior": float(prior[j]),
                     "own_value": float(cn.value), "searched": float(eng.V.get(c, cn.value)),
                     "visits": int(eng.visits[c]),
                     "below_prior_min": bool(prior[j] < eng.s.prior_min),
                     "fails_blunder": bool(cn.value < root.value - eng.s.blunder),
                     "learnable": bool(root.engine is not None and not np.isnan(root.engine[j]))})
    return sorted(rows, key=lambda r: -r["searched"])


def history_check(eng: Engine, bundle) -> dict:
    """Maia-3 reads the last eight positions. The cache is keyed on the position: a position
    reached by two paths gets the first path's evaluation. How much does the path matter?"""
    from tnh.models.base import Query
    ev = eng.ev
    pairs = [(a, b) for a, b in eng.transposed if a.path() != b.path()]
    if not pairs:
        return {"pairs": 0}
    qa, qb, q0 = [], [], []
    for a, b in pairs:
        ba = ev.board(a)
        mover_white = ba.turn == chess.WHITE
        me, op = (eng.s.my_elo, eng.s.opp_elo)
        qa.append(Query(ba.fen(), a.path(), me, op))
        qb.append(Query(ba.fen(), b.path(), me, op))
        q0.append(Query(ba.fen(), [], me, op))
        del mover_white
    ea, eb, e0 = bundle.evaluate(qa), bundle.evaluate(qb), bundle.evaluate(q0)

    def tv(x, y):
        ix = {m: i for i, m in enumerate(y.moves)}
        q = np.array([y.p[ix[m]] for m in x.moves])
        return 0.5 * float(np.abs(x.p - q).sum())

    tab = np.array([tv(x, y) for x, y in zip(ea, eb)])
    tan = np.array([tv(x, y) for x, y in zip(ea, e0)])
    vab = np.array([abs(x.value - y.value) for x, y in zip(ea, eb)])
    van = np.array([abs(x.value - y.value) for x, y in zip(ea, e0)])
    q = lambda x: {"mean": float(x.mean()), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)),  # noqa: E731
                   "max": float(x.max())}
    return {"pairs": len(pairs), "tv_two_paths": q(tab), "tv_no_history": q(tan),
            "value_two_paths": q(vab), "value_no_history": q(van)}


def stockfish_losses(eng: Engine, sets: list[dict], nodes: int) -> dict:
    from tnh.builder.output import rel_path  # noqa: F401
    from tnh.builder.deepcheck import DeepChecker
    g, ev = eng.graph, eng.ev
    by_key = {}
    for n in g.nodes:
        if n.children is not None and n.moves:
            by_key.setdefault(str(n.key), n)
    want = {}
    for s in sets:
        for k, m in s["learned_set"].items():
            want[(k, m)] = None
    out = {}
    with DeepChecker(nodes) as dc:
        for (k, m) in want:
            n = by_key[k]
            pos = eng.pos[n.id]
            d = dc.check(pos.path(), m, eps=1.0)
            out[f"{k}:{m}"] = {"move_value": d["move_value"], "best_other": d["alt_move"],
                               "best_other_value": d["alt_value"], "loss": max(0.0, d["alt_value"] - d["move_value"])}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tnh.engine.study")
    ap.add_argument("--stages", type=int, nargs="*", default=[2000, 10000, 30000])
    ap.add_argument("--search-seconds", type=float, default=1200)
    ap.add_argument("--prices", type=float, nargs="*", default=[0.0005, 0.001, 0.002, 0.005])
    ap.add_argument("--caps", type=int, nargs="*", default=[5, 10, 20, 40])
    ap.add_argument("--grid", type=int, default=20)
    ap.add_argument("--lines", type=int, default=8)
    ap.add_argument("--sf-nodes", type=int, default=300_000)
    ap.add_argument("--rss-limit", type=float, default=4.8)
    ap.add_argument("--report", required=True)
    ap.add_argument("--export-name", default="engine-c4-nf6-g3-e6-bg2-d5")
    ap.add_argument("--fake", action="store_true", help="fake models, for a dry run")
    a = ap.parse_args(argv)

    from tnh.data.edges import EdgeDB
    from tnh.engine.chess_eval import ChessEvaluator, FakeBundle
    t0 = time.perf_counter()
    db = EdgeDB("full")
    if a.fake:
        free = 0
        ev = ChessEvaluator({"fake": FakeBundle()}, db)
        tempo = "fake"
    else:
        free = gpu_free_mb()
        if free < 3000:
            raise SystemExit(f"only {free} MB free on the GPU; need 3000")
        ev = ChessEvaluator({}, db)
        tempo = "classical"
    s = Settings(my_elo=2200, opp_elo=2200, my_side="white", tempo=tempo, price=0.001, lines=a.lines)
    eng = Engine(ev, s)
    eng.set_position(C4)
    board = chess.Board()
    for m in C4:
        board.push_uci(m)
    log = {"gpu_free_mb_at_start": free, "stages": []}
    tl = time.perf_counter()
    eng.search(nodes=1)                                   # loads the models
    log["load_seconds"] = time.perf_counter() - tl
    print(f"models loaded in {log['load_seconds']:.0f} s, rss {rss_gb():.2f} GB", flush=True)
    grid = list(np.geomspace(1e-5, 0.05, a.grid))
    searched = 0.0
    prev_sec, prev_exp = dict(eng.seconds), eng.stats["expanded"]
    root_rows = {}
    for target in a.stages:
        left = a.search_seconds - searched
        if left <= 0:
            break
        ts = time.perf_counter()
        import threading
        stop = threading.Event()
        done = threading.Event()

        def watch():
            while not done.wait(2.0):
                if rss_gb() > a.rss_limit:
                    log.setdefault("stopped_for_memory", rss_gb())
                    stop.set()
                    return
        threading.Thread(target=watch, daemon=True).start()
        r = eng.search(nodes=max(target - eng.stats["expanded"], 0), seconds=left, stop=stop)
        done.set()
        searched += time.perf_counter() - ts
        exp = eng.stats["expanded"]
        sec = {k: v - prev_sec.get(k, 0.0) for k, v in eng.seconds.items()}
        per_k = {k: round(1000 * v / max(exp - prev_exp, 1), 2) for k, v in sec.items()}
        prev_sec, prev_exp = dict(eng.seconds), exp
        stage = {"expanded": exp, "positions": len(eng.graph.reachable()), "graph_nodes": len(eng.graph.nodes),
                 "search_seconds": searched, "rss_gb": rss_gb(), "seconds_per_1000_expansions": per_k,
                 "nps": r.expansions / max(r.seconds, 1e-9), "last_solve_seconds": eng.last_solve_seconds,
                 "solves": eng.stats["solves"], "stats": dict(eng.stats)}
        print(f"stage {exp}: {stage['positions']} positions, rss {stage['rss_gb']:.2f} GB, {stage['nps']:.0f}/s, "
              f"per 1000 {per_k}", flush=True)
        tw = time.perf_counter()
        stage["sweep"] = sweep(eng, board, a.prices, a.caps, grid, a.lines)
        stage["sweep_seconds"] = time.perf_counter() - tw
        stage["root"] = root_table(eng, board)
        root_rows[exp] = stage["root"]
        log["stages"].append(stage)
        print(f"  sweep {stage['sweep_seconds']:.0f} s", flush=True)
        if stage["rss_gb"] > a.rss_limit or exp < target:
            break
    log["history"] = list(eng.history)
    tb = time.perf_counter()
    log["history_check"] = history_check(eng, ev.bundle(tempo))
    log["history_check_seconds"] = time.perf_counter() - tb
    print(f"history check {log['history_check']}", flush=True)
    last = log["stages"][-1]
    sets = [x for x in last["sweep"]["prices"] + last["sweep"]["caps"] if not x.get("none")]
    tsf = time.perf_counter()
    log["stockfish"] = stockfish_losses(eng, sets, a.sf_nodes)
    log["stockfish_seconds"] = time.perf_counter() - tsf
    log["n10"] = n10_table(eng, board, last)
    log["export"] = write_export(eng, board, last, a.export_name, db)
    log["wall_seconds"] = time.perf_counter() - t0
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).with_suffix(".json").write_text(json.dumps(log, indent=1, default=str))
    Path(a.report).write_text(report(log, a))
    print(f"wrote {a.report}", flush=True)


def n10_table(eng: Engine, board: chess.Board, stage: dict) -> list[dict]:
    cap10 = next((x for x in stage["sweep"]["caps"] if x.get("cap") == 10 and not x.get("none")), None)
    if cap10 is None:
        return []
    g = eng.graph
    by_key = {}
    for n in g.nodes:
        if n.children is not None and n.moves:
            by_key.setdefault(str(n.key), n)
    rows = []
    for k, m in cap10["learned_set"].items():
        n = by_key[k]
        j = n.moves.index(m)
        b = eng.ev.board(eng.pos[n.id])
        rel = eng.pos[n.id].path()[len(C4):]
        rows.append({"line": san_line(board, rel), "move": b.san(chess.Move.from_uci(m)), "uci": m, "key": k,
                     "p_model": float(n.info["p_first"][j]), "cost": int(n.info["p_first"][j] < eng.s.obvious)})
    return sorted(rows, key=lambda r: (len(r["line"]), r["line"]))


def write_export(eng: Engine, board: chess.Board, stage: dict, name: str, db) -> dict:
    from tnh.builder.export import export, git_commit, pgn_for, today, write
    from tnh.builder.search import Solution
    from tnh.paths import DATA_ROOT
    g = eng.graph
    by_key = {}
    for n in g.nodes:
        if n.children is not None and n.moves:
            by_key.setdefault(str(n.key), n)
    eng.set_settings(price=eng.s.price, cap=0)
    eng.solve()
    game = eng.game
    sols, budgets = {}, []
    for x in stage["sweep"]["caps"]:
        if x.get("none"):
            continue
        learned = {by_key[k].key: m for k, m in x["learned_set"].items()}
        N = x["cap"]
        sols[N] = Solution(N, learned, game.evaluate(learned), x["value"], game.total_cost(learned))
        budgets.append(N)
    for n in g.nodes:                                       # the learnability flags are not engine values
        if n.me_to_move:
            n.engine = None
    totals = {}

    def games(n):
        if n.key not in totals:
            t = db.total(n.key)
            totals[n.key] = int(t["n"]) if t else 0
        return totals[n.key]

    root_ply = g.nodes[g.root].ply
    meta = {"name": name, "date": today(), "commit": git_commit(), "root_fen": board.fen(), "root_moves": C4,
            "me": "white", "me_elo": eng.s.my_elo, "opp_elo": eng.s.opp_elo,
            "params": {"engine": "tnh.engine", "obvious_threshold": eng.s.obvious, "mix": eng.s.mix,
                       "prepared": eng.s.prepared, "prior_min": eng.s.prior_min, "blunder": eng.s.blunder,
                       "budgets_are": "caps on non-obvious learned moves (from a grid of prices)"},
            "database": {"table": "EdgeDB('full')", "filter": "classical, all ratings and years (tnh.movemix's)"},
            "models": {"move_model": "tnh.models.default_move_model() (classical bundle)"}}
    res = export(game, sols, budgets, root_moves=C4, meta=meta, games=games, obvious=eng.s.obvious,
                 board_of=lambda n: eng.ev.board(eng.pos[n.id]), ply_offset=root_ply)
    pgns = {N: pgn_for(game, sols[N], C4, {"Event": f"{name}, cap {N}"}) for N in budgets}
    out = DATA_ROOT / "repertoires" / name
    write(out, res, pgns)
    return {"dir": str(out), "records": len(res["nodes"]), "mb": (out / "result.json").stat().st_size / 1e6}


def report(log: dict, a) -> str:
    L = []
    add = L.append
    add("# preppy, first real run: 1.c4 Nf6 2.g3 e6 3.Bg2 d5, White, 2200 v 2200, classical\n")
    add(f"Models loaded in {log['load_seconds']:.0f} s; GPU had {log['gpu_free_mb_at_start']} MB free at start; "
        f"wall time {log['wall_seconds']:.0f} s.\n")
    for st in log["stages"]:
        add(f"## {st['expanded']} expanded positions ({st['positions']} in the graph)\n")
        add(f"- search time so far {st['search_seconds']:.0f} s, {st['nps']:.0f} expansions/s in this stage, "
            f"host memory {st['rss_gb']:.2f} GB, last solve {st['last_solve_seconds']:.2f} s, solves so far {st['solves']}")
        add(f"- seconds per 1,000 expansions in this stage: {st['seconds_per_1000_expansions']}")
        add(f"- the sweep (each price, the price grid for the caps) took {st['sweep_seconds']:.0f} s\n")
        add("| setting | price | value | net | non-obvious | learned | root move | stand-in / frontier / finished |")
        add("|---|---|---|---|---|---|---|---|")
        for x in st["sweep"]["prices"] + st["sweep"]["caps"]:
            if x.get("none"):
                add(f"| {x['label']} | no price on the grid fits | | | | | | |")
                continue
            m = x["mass"]
            add(f"| {x['label']} | {x['price']:.3g} | {x['value']:.4f} | {x['net']:.4f} | {x['cost']} | {x['learned']} | "
                f"{x['root_move'] or '-'} | {m['stand_in']:.3f} / {m['frontier']:.3f} / {m['finished']:.3f} |")
        add("")
        for x in st["sweep"]["prices"] + st["sweep"]["caps"]:
            if x.get("none"):
                continue
            add(f"**{x['label']}** (non-obvious {x['cost']}{', fewer than the cap' if x.get('cap') and x['cost'] < x['cap'] else ''}):\n")
            add("| # | line (* = learned) | value | net | non-obvious in it | reach at its end |")
            add("|---|---|---|---|---|---|")
            for i, l in enumerate(x["lines"][:5], 1):
                flags = " ".join(f"{mv}{'*' if f == 'L' else ''}" for mv, f in zip(l["san"].split(), l["flags"]))
                add(f"| {i} | {flags} | {l['value']:.4f} | {'-' if l['net'] is None else f"{l['net']:.4f}"} | "
                    f"{l['nonobvious']} | {l['reach']:.3f} |")
            add("")
        add("Root moves by searched value at price 0.001:\n")
        add("| move | move model | Lc0 policy | prior | own value | searched | visits | below prior_min | fails blunder gate |")
        add("|---|---|---|---|---|---|---|---|---|")
        for r in st["root"]:
            add(f"| {r['san']} | {r['p_model']:.3f} | {'-' if r['policy'] is None else f"{r['policy']:.3f}"} | "
                f"{r['prior']:.4f} | {r['own_value']:.4f} | {r['searched']:.4f} | {r['visits']} | "
                f"{'yes' if r['below_prior_min'] else ''} | {'yes' if r['fails_blunder'] else ''} |")
        add("")
    add("## Root value against expanded positions (price 0.001)\n")
    hist = log["history"]
    pick = hist[:: max(1, len(hist) // 25)] + hist[-1:]
    add("| expanded | value | net |")
    add("|---|---|---|")
    for e, v, n in pick:
        add(f"| {e} | {v:.4f} | {n:.4f} |")
    add("\n## Maia's history and the cache key\n")
    add("```\n" + json.dumps(log["history_check"], indent=1) + "\n```\n")
    add("## The N=10 repertoire (cap 10): which learned moves are free\n")
    add("| line to it | learned | move model at 2200 | cost | Stockfish loss |")
    add("|---|---|---|---|---|")
    for r in log["n10"]:
        sf = log["stockfish"].get(f"{r['key']}:{r['uci']}", {})
        add(f"| {r['line'] or '(root)'} | {r['move']} | {r['p_model']:.3f} | {r['cost']} | "
            f"{sf.get('loss', float('nan')):.3f} (best other {sf.get('best_other', '?')}) |")
    add(f"\n## Stockfish on the learned moves ({a.sf_nodes} nodes, one process; {log['stockfish_seconds']:.0f} s)\n")
    losses = [v["loss"] for v in log["stockfish"].values()]
    if losses:
        add(f"{len(losses)} learned moves over all reported settings; loss against Stockfish's best other move: "
            f"mean {np.mean(losses):.3f}, median {np.median(losses):.3f}, max {np.max(losses):.3f}; "
            f"{sum(x > 0.05 for x in losses)} above 0.05.\n")
    add(f"## Export\n\n{log['export']}\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
