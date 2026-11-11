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
    top1, nll, vll, legal = [], [], [], []
    for p, mv, r in zip(preds, played, result):
        i = p.moves.index(mv)
        top1.append(int(np.argmax(p.logp)) == i)
        nll.append(-float(p.logp[i]))
        w, d, l = p.wdl
        vll.append(-math.log({1.0: w, 0.5: d, 0.0: l}[r]))
        legal.append(p.legal_mass)
    return {
        "n": len(top1),
        "top1": float(np.mean(top1)),
        "logloss": float(np.mean(nll)),
        "value_logloss": float(np.mean(vll)),
        "illegal_mass": float(1 - np.mean(legal)),
    }


def _fmt(d: dict) -> str:
    return (f"n={d['n']} top1={d['top1']:.4f} logloss={d['logloss']:.4f} "
            f"value_logloss={d['value_logloss']:.4f} illegal_mass={d['illegal_mass']:.2e}")


def cmd_sanity(a) -> None:
    from tnh.models.maia3 import Maia3

    games = read_games(a.pgn)
    qs, played, result = positions(games, min_ply=a.min_ply)
    elos = [q.mover_elo for q in qs]
    print(f"{len(qs)} positions at ply >= {a.min_ply}; mover Elo mean {np.mean(elos):.0f}, "
          f"quartiles {np.percentile(elos, [25, 50, 75]).round().tolist()}")
    settings = [("offset", float(x)) for x in a.offsets] + [("fixed", float(x)) for x in a.fixed]
    for size in a.size:
        for kind, x in settings:
            kw = {"elo_offset": x} if kind == "offset" else {"self_elo": x, "oppo_elo": x}
            m = Maia3(size, batch_size=a.batch_size, **kw)
            t = time.perf_counter()
            preds = m.predict_logp(qs)
            dt = time.perf_counter() - t
            print(f"{m.name} {m.variant:>16}: {_fmt(score(preds, played, result))} ({len(qs) / dt:.0f} pos/s)")
            del m


def cmd_elo_effect(a) -> None:
    """How much the Elo input moves the distribution: KL and top-1 changes between settings."""
    from tnh.models.maia3 import Maia3

    games = read_games(a.pgn)
    qs, played, result = positions(games, min_ply=a.min_ply)
    qs = qs[: a.limit]
    m = Maia3(a.size[0], self_elo=1500, oppo_elo=1500, batch_size=a.batch_size)
    base = {}
    for e in a.fixed:
        m.self_elo = m.oppo_elo = float(e)
        base[e] = m.predict_logp(qs)
    ref = base[a.fixed[0]]
    for e in a.fixed:
        kl = np.mean([float(np.sum(np.exp(p.logp) * (p.logp - r.logp))) for p, r in zip(base[e], ref)])
        changed = np.mean([int(np.argmax(p.logp)) != int(np.argmax(r.logp)) for p, r in zip(base[e], ref)])
        pw = np.mean([p.wdl[0] for p in base[e]])
        pd = np.mean([p.wdl[1] for p in base[e]])
        ent = np.mean([-float(np.sum(np.exp(p.logp) * p.logp)) for p in base[e]])
        print(f"Elo {e:>5}: KL(.||{a.fixed[0]})={kl:.4f} top1 differs={changed:.3f} "
              f"entropy={ent:.3f} mean P(win)={pw:.3f} mean P(draw)={pd:.3f}")
    # self vs opponent: vary one side only
    for s, o in [(2200, 1500), (1500, 2200), (2200, 2200)]:
        m.self_elo, m.oppo_elo = float(s), float(o)
        ps = m.predict_logp(qs)
        pw = np.mean([p.wdl[0] for p in ps]); pd = np.mean([p.wdl[1] for p in ps])
        print(f"self {s} oppo {o}: {_fmt(score(ps, played[: len(qs)], result[: len(qs)]))} "
              f"mean P(win)={pw:.3f} P(draw)={pd:.3f}")


def cmd_history(a) -> None:
    """Reference history (each frame from its own mover's side) against alternatives."""
    from tnh.models.maia3 import HISTORY, Maia3

    games = read_games(a.pgn)
    qs, played, result = positions(games, min_ply=a.min_ply)

    class NoHistory(Maia3):
        """The current position repeated, as maia3-uci does without --use-uci-history."""
        def _encode(self, queries):
            tokens, elos, moves, idxs = super()._encode(queries)
            tokens[:] = np.tile(tokens[:, :, -12:], (1, 1, HISTORY))
            return tokens, elos, moves, idxs

    class CurrentPerspective(Maia3):
        """Every frame oriented to the current mover (Lc0 convention), to test the alternative."""
        def _encode(self, queries):
            tokens, elos, moves, idxs = super()._encode(queries)
            for i in range(HISTORY - 1):
                # frames at odd distance from the current one have the other side to move
                if (HISTORY - 1 - i) % 2 == 1:
                    f = tokens[:, :, 12 * i: 12 * (i + 1)].reshape(-1, 8, 8, 12)[:, ::-1]
                    f = np.concatenate([f[..., 6:], f[..., :6]], axis=-1)
                    tokens[:, :, 12 * i: 12 * (i + 1)] = f.reshape(-1, 64, 12)
            return tokens, elos, moves, idxs

    for size in a.size:
        for cls in (Maia3, NoHistory, CurrentPerspective):
            m = cls(size, elo_offset=0.0, batch_size=a.batch_size)
            print(f"{m.name} {cls.__name__:>18}: {_fmt(score(m.predict_logp(qs), played, result))}")
            del m


def cmd_throughput(a) -> None:
    import torch
    from tnh.models.maia3 import Maia3

    games = read_games(a.pgn)
    qs, _, _ = positions(games, min_ply=0)
    qs = qs[: a.limit]
    for size in a.size:
        for bs in a.batch_sizes:
            m = Maia3(size, elo_offset=0.0, batch_size=bs)
            m.predict_logp(qs[: 2 * bs])  # warm-up
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            m.predict_logp(qs)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
            # GPU-only time on pre-encoded batches, to separate it from Python encoding
            batches = [m._encode(qs[i: i + bs]) for i in range(0, min(len(qs), 20 * bs), bs)]
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            for b in batches:
                [x.cpu() for x in m._launch(b)]
            torch.cuda.synchronize()
            gpu = sum(len(b[2]) for b in batches) / (time.perf_counter() - t2)
            peak = torch.cuda.max_memory_allocated() / 2**20
            print(f"{m.name} batch {bs:>5}: {len(qs) / dt:8.0f} pos/s end to end, "
                  f"{gpu:8.0f} pos/s GPU only, peak allocated {peak:6.0f} MiB")
            del m, batches
            torch.cuda.empty_cache()


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
    sp.set_defaults(fn=cmd_sanity)

    sp = sub.add_parser("elo-effect"); common(sp)
    sp.add_argument("--fixed", nargs="+", type=int, default=[1500, 1900, 2200, 2500, 2800, 3500])
    sp.add_argument("--limit", type=int, default=20000)
    sp.set_defaults(fn=cmd_elo_effect)

    sp = sub.add_parser("history"); common(sp)
    sp.set_defaults(fn=cmd_history)

    sp = sub.add_parser("throughput"); common(sp)
    sp.add_argument("--batch-sizes", nargs="+", type=int, default=[128, 256, 512, 1024])
    sp.add_argument("--limit", type=int, default=30000)
    sp.set_defaults(fn=cmd_throughput)

    sp = sub.add_parser("stockfish")
    sp.add_argument("--pgn", type=Path, required=True)
    sp.add_argument("--nodes", nargs="+", type=int, default=[1000, 10000, 100000])
    sp.add_argument("--n", type=int, default=300)
    sp.add_argument("--processes", nargs="+", type=int, default=[16])
    sp.add_argument("--hash-mb", type=int, default=16)
    sp.add_argument("--min-ply", type=int, default=20)
    sp.set_defaults(fn=lambda a: cmd_stockfish(a))

    a = p.parse_args(argv)
    a.fn(a)


def cmd_stockfish(a) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
