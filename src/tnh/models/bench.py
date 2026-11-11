"""Measurements behind the models report: sanity on real games, history, throughput.

Reads a PGN file directly (python-chess), independent of `tnh.data`.

    uv run python -m tnh.models.bench sanity --pgn <file.pgn.zst> --size 23m
    uv run python -m tnh.models.bench history --pgn <file.pgn.zst> --size 23m
    uv run python -m tnh.models.bench throughput --size 23m --batch-sizes 128 256 512 1024
    uv run python -m tnh.models.bench stockfish --pgn <file.pgn.zst> --nodes 1000 10000 100000
"""

from __future__ import annotations

import argparse
import io
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import zstandard

from tnh.models.base import Query

RESULTS = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


@dataclass
class Game:
    event: str
    white_elo: int
    black_elo: int
    result: str
    moves: list[str]


def read_games(path: Path, *, classical_only: bool = True) -> list[Game]:
    """Standard games with both ratings and a decided result. Prints what it drops."""
    with open(path, "rb") as f:
        raw = zstandard.ZstdDecompressor().stream_reader(f).read() if str(path).endswith(".zst") else f.read()
    text = io.StringIO(raw.decode("utf-8", errors="strict"))
    games, dropped, kept_truncated = [], Counter(), 0
    while (g := chess.pgn.read_game(text)) is not None:
        h = g.headers
        if g.errors:
            # Broadcast PGNs end with an illegal king move when the arbiter puts both kings
            # in the centre of a DGT board to signal the result. python-chess stops the
            # mainline there; keep the moves before it.
            kept_truncated += 1
        if h.get("Variant", "Standard") != "Standard" or "FEN" in h:
            dropped["not standard start"] += 1
            continue
        if h.get("Result") not in RESULTS:
            dropped["no result"] += 1
            continue
        if not (h.get("WhiteElo", "").isdigit() and h.get("BlackElo", "").isdigit()):
            dropped["rating missing"] += 1
            continue
        event = h.get("Event", "")
        if classical_only and (any(w in event.lower() for w in ("rapid", "blitz", "tie-break", "armageddon"))
                               or h.get("TimeControl") == "1500+10"):
            dropped["not classical"] += 1
            continue
        moves = [m.uci() for m in g.mainline_moves()]
        if not moves:
            dropped["no moves"] += 1
            continue
        games.append(Game(event, int(h["WhiteElo"]), int(h["BlackElo"]), h["Result"], moves))
    print(f"read {len(games)} games from {path}; dropped {dict(dropped)}; "
          f"{kept_truncated} kept but cut at an illegal move")
    return games


def positions(games: list[Game], *, min_ply: int = 20, max_ply: int = 10_000) -> tuple[list[Query], list[str], list[float]]:
    """Queries for every ply in [min_ply, max_ply), with the played move and the mover's result."""
    queries, played, result = [], [], []
    for g in games:
        b = chess.Board()
        for ply, uci in enumerate(g.moves):
            if min_ply <= ply < max_ply:
                white = b.turn == chess.WHITE
                me, opp = (g.white_elo, g.black_elo) if white else (g.black_elo, g.white_elo)
                queries.append(Query(b.fen(), g.moves[:ply], me, opp))
                played.append(uci)
                r = RESULTS[g.result]
                result.append(r if white else 1.0 - r)
            b.push_uci(uci)
    return queries, played, result


def score(preds, played: list[str], result: list[float]) -> dict:
    """Top-1, mean log loss of the played move (nats), value-head log loss on the result."""
    top1, nll, vll, sign, pdraw = [], [], [], [], []
    for p, mv, r in zip(preds, played, result):
        i = p.moves.index(mv)
        top1.append(int(np.argmax(p.logp)) == i)
        nll.append(-float(p.logp[i]))
        w, d, l = p.wdl
        vll.append(-math.log({1.0: w, 0.5: d, 0.0: l}[r]))
        pdraw.append(d)
        if r != 0.5:
            sign.append((w > l) == (r == 1.0))
    return {
        "n": len(top1),
        "top1": float(np.mean(top1)),
        "logloss": float(np.mean(nll)),
        "value_logloss": float(np.mean(vll)),
        "value_sign": float(np.mean(sign)),
        "mean_p_draw": float(np.mean(pdraw)),
        "draw_rate": float(np.mean([r == 0.5 for r in result])),
    }


def _fmt(d: dict) -> str:
    return (f"n={d['n']} top1={d['top1']:.4f} logloss={d['logloss']:.4f} "
            f"value_logloss={d['value_logloss']:.4f} value_sign={d['value_sign']:.3f} "
            f"P(draw)={d['mean_p_draw']:.3f} (draw rate {d['draw_rate']:.3f})")


def cmd_sanity(a) -> None:
    """Every Elo setting in one pass per size and precision."""
    from tnh.models.maia3 import Elo, Maia3

    games = read_games(a.pgn)
    qs, played, result = positions(games, min_ply=a.min_ply)
    elos = [q.mover_elo for q in qs]
    print(f"{len(qs)} positions at ply >= {a.min_ply}; mover Elo mean {np.mean(elos):.0f}, "
          f"quartiles {np.percentile(elos, [25, 50, 75]).round().tolist()}")
    settings = [Elo.offset(float(x)) for x in a.offsets] + [Elo.fixed(float(x), float(x)) for x in a.fixed]
    for size in a.size:
        for prec in a.precision:
            m = Maia3(size, batch_size=a.batch_size, amp=prec == "fp16")
            t = time.perf_counter()
            results = m.predict_many(qs, settings)
            dt = time.perf_counter() - t
            for e, preds in zip(settings, results):
                print(f"{m.name} {m.variant(e):>16}: {_fmt(score(preds, played, result))}")
            print(f"{m.name} {prec}: {len(settings)} settings x {len(qs)} positions in {dt:.0f} s "
                  f"({len(qs) * len(settings) / dt:.0f} position-settings/s); mean illegal mass "
                  f"{m.illegal_mass_sum / m.n_predicted:.3f}", flush=True)
            del m


def cmd_elo_effect(a) -> None:
    """How the Elo input moves the predictions, both sides set to the same fixed value.

    For each value: fit to the played moves, entropy, distance to the previous value, mean
    P(draw). Then each side alone: the mover's Elo with the opponent's fixed, and back.
    """
    from tnh.models.maia3 import Elo, Maia3

    games = read_games(a.pgn)
    qs, played, result = positions(games, min_ply=a.min_ply)
    if a.limit:
        step = max(1, len(qs) // a.limit)
        qs, played, result = qs[::step], played[::step], result[::step]
    print(f"{len(qs)} positions, mover Elo mean {np.mean([q.mover_elo for q in qs]):.0f}")
    m = Maia3(a.size[0], batch_size=a.batch_size)
    b = a.side_base
    settings = ([Elo.fixed(e, e) for e in a.fixed] + [Elo.fixed(b, b)]
                + [Elo.fixed(e, b) for e in a.side_values] + [Elo.fixed(b, e) for e in a.side_values])
    res = dict(zip((e.name for e in settings), m.predict_many(qs, settings)))

    def tv(x, y):
        return float(np.mean([0.5 * np.abs(np.exp(p.logp) - np.exp(r.logp)).sum() for p, r in zip(x, y)]))

    prev, prev_e = None, None
    for e in a.fixed:
        ps = res[Elo.fixed(e, e).name]
        ent = np.mean([-float(np.sum(np.exp(p.logp) * p.logp)) for p in ps])
        d = "" if prev is None else f" TV to {prev_e}: {tv(ps, prev):.4f}"
        print(f"Elo {e:>5}: {_fmt(score(ps, played, result))} entropy={ent:.3f}{d}", flush=True)
        prev, prev_e = ps, e
    base = res[Elo.fixed(b, b).name]
    for e in a.side_values:
        s_, o_ = res[Elo.fixed(e, b).name], res[Elo.fixed(b, e).name]
        print(f"mover {e} vs opp {b}: TV to both {b} {tv(s_, base):.4f}, "
              f"logloss {score(s_, played, result)['logloss']:.4f} | "
              f"mover {b} vs opp {e}: TV {tv(o_, base):.4f}, logloss {score(o_, played, result)['logloss']:.4f}",
              flush=True)


def cmd_history(a) -> None:
    """Reference history (each frame from its own mover's side) against alternatives."""
    from tnh.models.maia3 import HISTORY, Elo, Encoder, Maia3

    games = read_games(a.pgn)
    qs, played, result = positions(games, min_ply=a.min_ply)

    class NoHistory(Encoder):
        """The current position repeated, as maia3-uci does without --use-uci-history."""
        def encode(self, queries):
            tokens, moves, idxs = super().encode(queries)
            tokens[:] = np.tile(tokens[:, :, -12:], (1, 1, HISTORY))
            return tokens, moves, idxs

    class CurrentPerspective(Encoder):
        """Every frame oriented to the current mover (Lc0 convention), to test the alternative."""
        def encode(self, queries):
            tokens, moves, idxs = super().encode(queries)
            for i in range(HISTORY - 1):
                # frames at odd distance from the current one have the other side to move
                if (HISTORY - 1 - i) % 2 == 1:
                    f = tokens[:, :, 12 * i: 12 * (i + 1)].reshape(-1, 8, 8, 12)[:, ::-1]
                    f = np.concatenate([f[..., 6:], f[..., :6]], axis=-1)
                    tokens[:, :, 12 * i: 12 * (i + 1)] = f.reshape(-1, 64, 12)
            return tokens, moves, idxs

    for size in a.size:
        m = Maia3(size, Elo.offset(0.0), batch_size=a.batch_size)
        for enc in (Encoder, NoHistory, CurrentPerspective):
            m._encoder = enc()
            print(f"{m.name} {enc.__name__:>18}: {_fmt(score(m.predict(qs), played, result))}", flush=True)
        del m


def cmd_throughput(a) -> None:
    import os

    import torch
    from tnh.models.maia3 import Elo, Maia3

    games = read_games(a.pgn)
    qs, _, _ = positions(games, min_ply=0)
    qs = qs[: a.limit]
    print(f"load average before: {os.getloadavg()}")
    sweep = [Elo.offset(float(x)) for x in range(-200, 801, 100)]
    for size in a.size:
        for bs, prec in [(b, p) for p in a.precision for b in a.batch_sizes]:
            m = Maia3(size, Elo.offset(0.0), batch_size=bs, amp=prec == "fp16")
            m.predict(qs[: 2 * bs])  # warm-up
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            m.predict(qs)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
            # GPU-only time on pre-encoded batches, to separate it from Python encoding
            batches = [m._encoder.encode(qs[i: i + bs]) for i in range(0, min(len(qs), 20 * bs), bs)]
            elos = [np.array([Elo.offset(0.0).of(q) for q in qs[i: i + bs]], dtype=np.float32)
                    for i in range(0, min(len(qs), 20 * bs), bs)]
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            for b, e in zip(batches, elos):
                [x.cpu() for x in m._forward(m._upload(b), e)]
            torch.cuda.synchronize()
            gpu = sum(len(b[1]) for b in batches) / (time.perf_counter() - t2)
            peak = torch.cuda.max_memory_allocated() / 2**20
            line = (f"{m.name} {prec} batch {bs:>5}: {len(qs) / dt:8.0f} pos/s end to end, "
                    f"{gpu:8.0f} pos/s GPU only, peak allocated {peak:6.0f} MiB")
            if a.sweep:
                t3 = time.perf_counter()
                m.predict_many(qs, sweep)
                torch.cuda.synchronize()
                line += f"; sweep of {len(sweep)} settings: {len(qs) * len(sweep) / (time.perf_counter() - t3):8.0f} position-settings/s"
            print(line, flush=True)
            del m, batches
            torch.cuda.empty_cache()


def cmd_lc0(a) -> None:
    """Lc0 throughput: policy alone, and searches of several node counts; GPU memory next to Maia-3."""
    import os
    import random
    import subprocess as sp

    import torch
    from tnh.models import lc0
    from tnh.models.maia3 import Elo, Maia3

    def gpu_used() -> int:
        return int(sp.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True).stdout.split()[0])

    games = read_games(a.pgn)
    qs, _, _ = positions(games, min_ply=a.min_ply)
    rng = random.Random(0)
    policy_qs = qs[: a.n_policy]
    search_qs = rng.sample(qs, a.n_search)
    print(f"load {os.getloadavg()}; GPU memory used before: {gpu_used()} MiB")
    maia = Maia3("79m", Elo.offset(0.0)) if a.with_maia else None
    if maia:
        maia.predict(qs[:2048])
        print(f"with Maia-3 79M loaded and warmed up: {gpu_used()} MiB")
    for backend in a.backends:
        pol = lc0.Lc0Policy(backend=backend)
        pol.predict(policy_qs[:100])
        t = time.perf_counter()
        pol.predict(policy_qs)
        dt = time.perf_counter() - t
        print(f"policy {backend}, batch {pol.batch_size}: {len(policy_qs) / dt:.0f} pos/s; GPU memory used {gpu_used()} MiB",
              flush=True)
        del pol
    for nodes in a.nodes:
        for procs in a.processes:
            with lc0.Lc0Search(nodes, processes=procs) as s:
                s.evaluate(search_qs[:procs])
                n = max(len(search_qs) * nodes // max(a.nodes) * procs, 4 * procs)   # roughly equal time per row
                sample = search_qs[:n] if nodes < max(a.nodes) else search_qs
                t = time.perf_counter()
                res = s.evaluate(sample)
                dt = time.perf_counter() - t
                visited = np.mean([np.mean(~np.isnan(r.value)) for r in res])
                print(f"search {nodes:5d} nodes, {procs} processes: {len(sample) / dt:6.1f} pos/s "
                      f"({nodes * len(sample) / dt:.0f} nodes/s) on {len(sample)} positions, share of root moves "
                      f"visited {visited:.3f}; GPU memory used {gpu_used()} MiB", flush=True)
    if maia:
        t = time.perf_counter()
        maia.predict(qs[:20000])
        print(f"Maia-3 79M alongside, alone again: {20000 / (time.perf_counter() - t):.0f} pos/s")


def cmd_lc0_children(a) -> None:
    """Throughput of Lc0Children, pooled across positions or not, and how its move values compare
    with Stockfish's at a deep budget: rank correlation, and the share of Stockfish's mistakes
    (expected-score loss over `--loss`) that the raw network already sees as mistakes."""
    import os
    import random

    from scipy.stats import spearmanr

    from tnh.models import lc0
    from tnh.models.stockfish import Stockfish

    games = read_games(a.pgn)
    qs, _, _ = positions(games, min_ply=a.min_ply)
    sample = random.Random(1).sample(qs, a.n)
    print(f"load {os.getloadavg()}; {len(sample)} positions, mean legal moves "
          f"{np.mean([chess.Board(q.fen).legal_moves.count() for q in sample]):.1f}")
    timing = sample[: a.n_timing]
    for pool in (False, True):
        m = lc0.Lc0Children(pool_across_positions=pool)
        m.evaluate(timing[:5])
        t = time.perf_counter()
        res = m.evaluate(timing)
        dt = time.perf_counter() - t
        print(f"children, pooled across positions {pool}: {len(timing) / dt:.1f} pos/s, "
              f"{m.stats['children evaluated'] / dt:.0f} children/s", flush=True)
        del m
    ch = lc0.Lc0Children().evaluate(sample)
    with Stockfish(a.sf_nodes, processes=a.processes) as sf:
        t = time.perf_counter()
        st = sf.evaluate(sample)
        print(f"stockfish {a.sf_nodes} nodes: {len(sample) / (time.perf_counter() - t):.0f} pos/s", flush=True)
    rhos, skipped = [], 0
    n_mist = seen = n_ok = false_alarm = 0
    edges = [0.05, 0.1, 0.2, 0.5, 1.01]
    by_bucket = [[0, 0] for _ in edges[:-1]]
    for c, s_ in zip(ch, st):
        assert c.moves == s_.moves
        ok = ~np.isnan(c.value) & ~np.isnan(s_.value)
        cv, sv = c.value[ok].astype(np.float64), s_.value[ok].astype(np.float64)
        if len(cv) >= 3 and np.ptp(cv) > 0 and np.ptp(sv) > 0:
            rhos.append(spearmanr(cv, sv).statistic)
        else:
            skipped += 1
        if len(cv) == 0:
            continue
        s_loss = sv.max() - sv
        c_loss = cv.max() - cv
        mist = s_loss > a.loss
        n_mist += int(mist.sum())
        seen += int((mist & (c_loss > a.loss)).sum())
        n_ok += int((~mist).sum())
        for k in range(len(edges) - 1):
            sel = (s_loss > edges[k]) & (s_loss <= edges[k + 1])
            by_bucket[k][0] += int(sel.sum())
            by_bucket[k][1] += int((sel & (c_loss > a.loss)).sum())
        false_alarm += int((~mist & (c_loss > a.loss)).sum())
    print(f"rank correlation (Spearman) of child values with Stockfish {a.sf_nodes}: mean {np.mean(rhos):.3f}, "
          f"median {np.median(rhos):.3f} over {len(rhos)} positions ({skipped} skipped: under 3 moves or constant)")
    print(f"Stockfish mistakes (loss > {a.loss}): {n_mist}; seen by the raw children values (their loss > {a.loss}): "
          f"{seen} = {seen / n_mist:.3f}; moves Stockfish rates fine but the children call mistakes: "
          f"{false_alarm} of {n_ok} = {false_alarm / n_ok:.3f}")
    for k in range(len(edges) - 1):
        n, sn = by_bucket[k]
        print(f"  Stockfish loss in ({edges[k]:.2f}, {min(edges[k + 1], 1):.2f}]: {n:6d} moves, "
              f"seen by the children {sn / max(n, 1):.3f}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m tnh.models.bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, sizes=("23m",)):
        sp.add_argument("--pgn", type=Path, required=True)
        sp.add_argument("--size", nargs="+", default=list(sizes))
        sp.add_argument("--min-ply", type=int, default=20)
        sp.add_argument("--batch-size", type=int, default=512)

    sp = sub.add_parser("sanity"); common(sp)
    sp.add_argument("--offsets", nargs="*", default=[0])
    sp.add_argument("--fixed", nargs="*", default=[])
    sp.add_argument("--precision", nargs="+", default=["fp16"], choices=["fp16", "fp32"])
    sp.set_defaults(fn=cmd_sanity)

    sp = sub.add_parser("elo-effect"); common(sp)
    sp.add_argument("--fixed", nargs="+", type=int,
                    default=[0, 600, 1000, 1500, 1800, 2000, 2200, 2250, 2300, 2400, 2500, 2600, 2700, 2800, 3000,
                             3300, 3600, 4000, 5000])
    sp.add_argument("--side-base", type=int, default=2200)
    sp.add_argument("--side-values", nargs="+", type=int, default=[1600, 2000, 2400, 2800])
    sp.add_argument("--limit", type=int, default=20000)
    sp.set_defaults(fn=cmd_elo_effect)

    sp = sub.add_parser("history"); common(sp)
    sp.set_defaults(fn=cmd_history)

    sp = sub.add_parser("throughput"); common(sp)
    sp.add_argument("--batch-sizes", nargs="+", type=int, default=[128, 256, 512, 1024])
    sp.add_argument("--precision", nargs="+", default=["fp16"], choices=["fp16", "fp32"])
    sp.add_argument("--limit", type=int, default=30000)
    sp.add_argument("--sweep", action="store_true", help="also time the 11 offsets -200..+800 in one pass")
    sp.set_defaults(fn=cmd_throughput)

    sp = sub.add_parser("stockfish")
    sp.add_argument("--pgn", type=Path, required=True)
    sp.add_argument("--nodes", nargs="+", type=int, default=[1000, 10000, 100000])
    sp.add_argument("--nodes-per-move", nargs="*", type=int, default=[])
    sp.add_argument("--no-clear", action="store_true", help="also time without ucinewgame")
    sp.add_argument("--n", type=int, default=300)
    sp.add_argument("--processes", nargs="+", type=int, default=[16])
    sp.add_argument("--hash-mb", type=int, default=16)
    sp.add_argument("--min-ply", type=int, default=20)
    sp.set_defaults(fn=lambda a: cmd_stockfish(a))

    sp = sub.add_parser("lc0")
    sp.add_argument("--pgn", type=Path, required=True)
    sp.add_argument("--min-ply", type=int, default=20)
    sp.add_argument("--n-policy", type=int, default=5000)
    sp.add_argument("--n-search", type=int, default=200)
    sp.add_argument("--nodes", nargs="+", type=int, default=[100, 400, 1600])
    sp.add_argument("--processes", nargs="+", type=int, default=[1])
    sp.add_argument("--backends", nargs="+", default=["cuda-auto"])
    sp.add_argument("--with-maia", action="store_true", help="keep Maia-3 79M loaded on the GPU meanwhile")
    sp.set_defaults(fn=cmd_lc0)

    sp = sub.add_parser("lc0-children")
    sp.add_argument("--pgn", type=Path, required=True)
    sp.add_argument("--min-ply", type=int, default=20)
    sp.add_argument("--n", type=int, default=600)
    sp.add_argument("--n-timing", type=int, default=200)
    sp.add_argument("--sf-nodes", type=int, default=100_000)
    sp.add_argument("--processes", type=int, default=8)
    sp.add_argument("--loss", type=float, default=0.05)
    sp.set_defaults(fn=cmd_lc0_children)

    a = p.parse_args(argv)
    a.fn(a)


def cmd_stockfish(a) -> None:
    """Throughput and stability of the Stockfish value feature across node budgets."""
    import os
    import random

    from scipy.stats import spearmanr

    from tnh.models.stockfish import Stockfish

    games = read_games(a.pgn)
    qs, _, _ = positions(games, min_ply=a.min_ply)
    rng = random.Random(0)
    sample = rng.sample(qs, a.n)
    print(f"{len(sample)} positions sampled at ply >= {a.min_ply}, mean legal moves "
          f"{np.mean([chess.Board(q.fen).legal_moves.count() for q in sample]):.1f}; load {os.getloadavg()}")
    budgets = [(n, False) for n in a.nodes] + [(n, True) for n in a.nodes_per_move]
    results = {}
    for nodes, per_move in budgets:
        for procs in a.processes:
            for det in ([True, False] if a.no_clear else [True]):
                with Stockfish(nodes, processes=procs, hash_mb=a.hash_mb, per_move=per_move, deterministic=det) as sf:
                    sf.evaluate(sample[: procs])   # start-up
                    sf.stats.clear()
                    t = time.perf_counter()
                    res = sf.evaluate(sample)
                    dt = time.perf_counter() - t
                    st = sf.stats
                label = f"{nodes}{'/move' if per_move else ''}"
                print(f"nodes {label:>9} procs {procs:>2} clear {det!s:>5}: {len(sample) / dt:8.1f} pos/s; "
                      f"moves without a value {st['moves without a value'] / st['moves']:.3f}, positions with one "
                      f"{st['positions with a move without a value'] / st['positions']:.3f}; "
                      f"bound scores {st['bound scores'] / st['moves']:.3f}", flush=True)
                if det:
                    results[label] = res
                    v = np.concatenate([r.value for r in res])
                    v = v[~np.isnan(v)]
                    print(f"    values at 0 or 1 (to 0.001): {np.mean((v <= 0.001) | (v >= 0.999)):.3f}; "
                          f"positions where all moves have the same value: "
                          f"{np.mean([len(r.value) > 1 and np.nanmax(r.value) == np.nanmin(r.value) for r in res]):.3f}")
    labels = list(results)

    def compare(x, y):
        rhos, mad, mad_near, top, regret = [], [], [], [], []
        skipped = 0
        for p, r in zip(results[x], results[y]):
            ok = ~np.isnan(p.value) & ~np.isnan(r.value)
            if ok.sum() < 3:
                skipped += 1
                continue
            a_, b_ = p.value[ok].astype(np.float64), r.value[ok].astype(np.float64)
            d = np.abs(a_ - b_)
            mad.append(d.mean())
            near = b_ >= b_.max() - 0.1
            mad_near.append(d[near].mean())
            top.append(int(np.argmax(a_)) == int(np.argmax(b_)))
            regret.append(b_.max() - b_[int(np.argmax(a_))])
            if np.ptp(a_) > 0 and np.ptp(b_) > 0:
                rhos.append(spearmanr(a_, b_).statistic)
            else:
                skipped += 1
        return (f"{x:>9} vs {y:>9}: Spearman {np.mean(rhos):.3f} (median {np.median(rhos):.3f}), "
                f"mean |dE| {np.mean(mad):.4f}, mean |dE| on moves within 0.1 of the best {np.mean(mad_near):.4f}, "
                f"same best move {np.mean(top):.3f}, regret of {x}'s best move valued by {y} {np.mean(regret):.4f}, "
                f"positions skipped {skipped}")

    for i, x in enumerate(labels):
        for y in labels[i + 1:]:
            print(compare(x, y))


if __name__ == "__main__":
    main()
