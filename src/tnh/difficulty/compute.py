"""Model calls for step 2, cached on the position hash.

Per position in the set, and per child its side to move is likely to play:

- Maia-3 on CPU, one thread, at the ratings given by step 1's map
  ``maia_elo = 1580 + 0.558 * fide`` for both sides (mean FIDE ratings of the
  position's games; the network ratings are rounded to 10). Move
  probabilities and the value head.
- Stockfish, one process and one thread, MultiPV over every legal move at a
  node budget for the whole search (`tnh.models.stockfish`): the value of each
  move for the side to move.
- Stockfish at the root only, MultiPV 1 at a larger budget: the engine's value
  of the position, which the baseline rests on. The last report that is not a
  bound is used.

The children of a position are its side to move's moves in order of Maia
probability until ``child_mass`` of the probability is covered, at most
``max_children``, none below ``min_child_p``. A child is queried with the
parent's ratings, sides swapped.

Caches live in ``features/positions/<model>/<variant>/part-*.parquet``, one
row per position (and, for Maia, per pair of network ratings), with a
``meta.json`` of the options. A run into a directory whose ``meta.json`` has
other options is refused. Positions are keyed on ``pos_hash``; the move
history Maia sees is the path that first reached the position in this run
(the most played path for positions in the set, the parent's path plus the
move for children), so the same position reached another way reuses that
history.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import numpy as np
import polars as pl

from tnh.data.edges import pos_hash
from tnh.difficulty.positions import OUT, set_paths
from tnh.models.base import Query
from tnh.paths import DATA_ROOT

CACHE = DATA_ROOT / "features" / "positions"
MAP_A, MAP_B = 1580.0, 0.558


def maia_elo(fide: float) -> float:
    """Step 1's map from FIDE rating to the rating Maia-3 is given, rounded to 10."""
    return float(np.round((MAP_A + MAP_B * fide) / 10.0) * 10.0)


@dataclass(frozen=True)
class Budget:
    root_nodes: int = 200_000
    multipv_nodes: int = 50_000
    maia_size: str = "23m"
    child_mass: float = 0.9
    max_children: int = 8
    min_child_p: float = 0.02


# --- the caches --------------------------------------------------------------------


class Cache:
    """Rows keyed on ``key`` columns, in part files, with a meta.json that must match."""

    def __init__(self, d: Path, key: list[str], meta: dict, schema: dict):
        self.d, self.key, self.schema = d, key, schema
        d.mkdir(parents=True, exist_ok=True)
        mp = d / "meta.json"
        if mp.exists():
            old = json.loads(mp.read_text())
            if old != meta:
                raise SystemExit(f"{d}: meta.json has other options: {old} vs {meta}")
        else:
            mp.write_text(json.dumps(meta, indent=1) + "\n")
        parts = sorted(d.glob("part-*.parquet"))
        self.df = (pl.concat([pl.read_parquet(p) for p in parts]).unique(key, keep="first")
                   if parts else pl.DataFrame(schema=schema))
        self.rows: dict[tuple, dict] = {tuple(r[c] for c in key): r for r in self.df.iter_rows(named=True)}
        self.pending: list[dict] = []
        self._k = len(parts)

    def has(self, k: tuple) -> bool:
        return k in self.rows

    def get(self, k: tuple) -> dict:
        return self.rows[k]

    def add(self, row: dict) -> None:
        k = tuple(row[c] for c in self.key)
        if k in self.rows:
            return
        self.rows[k] = row
        self.pending.append(row)

    def flush(self) -> None:
        if not self.pending:
            return
        new = pl.DataFrame(self.pending, schema=self.schema)
        p = self.d / f"part-{self._k:05d}-{os.getpid()}.parquet"
        new.write_parquet(p)
        self._k += 1
        self.df = pl.concat([self.df, new])
        self.pending = []


MAIA_SCHEMA = {"pos_hash": pl.UInt64, "self_elo": pl.Float32, "oppo_elo": pl.Float32,
               "moves": pl.List(pl.String), "logp": pl.List(pl.Float32), "wdl": pl.List(pl.Float32)}
SF_SCHEMA = {"pos_hash": pl.UInt64, "moves": pl.List(pl.String), "value": pl.List(pl.Float32),
             "cp": pl.List(pl.Float32), "mate": pl.List(pl.Int16), "wdl": pl.List(pl.Float32)}
ROOT_SCHEMA = {"pos_hash": pl.UInt64, "best": pl.String, "cp": pl.Float32, "mate": pl.Int16,
               "wdl": pl.List(pl.Float32), "depth": pl.Int16, "used_bound": pl.Boolean}


def open_caches(b: Budget, sf_id: str = "?") -> tuple[Cache, Cache, Cache]:
    maia = Cache(CACHE / f"maia3-{b.maia_size}" / f"fidemap{MAP_A:g}+{MAP_B:g}-cpu-fp32",
                 ["pos_hash", "self_elo", "oppo_elo"],
                 {"size": b.maia_size, "map": [MAP_A, MAP_B], "device": "cpu", "amp": False, "round": 10}, MAIA_SCHEMA)
    sf = Cache(CACHE / "stockfish" / f"nodes{b.multipv_nodes}", ["pos_hash"],
               {"nodes": b.multipv_nodes, "multipv": "all", "hash_mb": 16, "deterministic": True}, SF_SCHEMA)
    root = Cache(CACHE / "stockfish" / f"root-nodes{b.root_nodes}", ["pos_hash"],
                 {"nodes": b.root_nodes, "multipv": 1, "hash_mb": 64, "deterministic": True}, ROOT_SCHEMA)
    return maia, sf, root


# --- the root search ----------------------------------------------------------------


def parse_root(lines: list[str]) -> dict:
    """The last `info ... score` line that is not a bound, else the last one (flagged)."""
    scored = [l for l in lines if l.startswith("info") and " score " in l and " pv " in l]
    if not scored:
        raise RuntimeError(f"no scored line in {lines[-3:]}")
    exact = [l for l in scored if " lowerbound " not in l and " upperbound " not in l]
    line = (exact or scored)[-1]
    t = line.split()
    s = t.index("score")
    w, d, l_ = (int(x) for x in t[t.index("wdl") + 1 : t.index("wdl") + 4])
    kind, val = t[s + 1], int(t[s + 2])
    return {"best": t[t.index("pv") + 1], "cp": float(val) if kind == "cp" else float("nan"),
            "mate": val if kind == "mate" else 0, "wdl": [w / 1000, d / 1000, l_ / 1000],
            "depth": int(t[t.index("depth") + 1]), "used_bound": not exact}


class RootSearch:
    """One Stockfish process, MultiPV 1, cleared before every position."""

    def __init__(self, nodes: int, hash_mb: int = 64):
        from tnh.models.stockfish import STOCKFISH, _Engine
        self.nodes = nodes
        self.e = _Engine(STOCKFISH, hash_mb)
        self.e._send("setoption name MultiPV value 1", "isready")
        self.e._until(b"readyok")

    def __call__(self, moves_uci: list[str]) -> dict:
        self.e._send("ucinewgame", "position startpos moves " + " ".join(moves_uci), f"go nodes {self.nodes}")
        out = self.e._until(b"bestmove").decode().splitlines()
        return parse_root(out)

    def close(self) -> None:
        self.e.close()


# --- the run ------------------------------------------------------------------------


def children_of(moves: list[str], p: np.ndarray, b: Budget) -> list[tuple[str, float]]:
    order = np.argsort(-p, kind="stable")
    out, mass = [], 0.0
    for i in order:
        if mass >= b.child_mass or len(out) >= b.max_children or p[i] < b.min_child_p:
            break
        out.append((moves[i], float(p[i])))
        mass += float(p[i])
    return out


def run(positions: pl.DataFrame, b: Budget, *, chunk: int = 32, flush_every: int = 512,
        max_seconds: float | None = None) -> dict:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from tnh.models.maia3 import Elo, Maia3
    from tnh.models.stockfish import Stockfish

    t0 = time.time()
    maia_c, sf_c, root_c = open_caches(b)
    stats: Counter = Counter()
    secs: Counter = Counter()
    maia = Maia3(b.maia_size, Elo.offset(0), device="cpu", amp=False, batch_size=64, local_files_only=True)
    sf = Stockfish(b.multipv_nodes, processes=1)
    root = RootSearch(b.root_nodes)
    rows = positions.select("pos_hash", "fen", "moves_uci", "white_elo", "black_elo", "stm_white").to_dicts()

    def maia_calls(items: list[tuple[int, str, list[str], float, float]]) -> None:
        todo = [it for it in items if not maia_c.has((it[0], it[3], it[4]))]
        seen, uniq = set(), []
        for it in todo:
            if (it[0], it[3], it[4]) not in seen:
                seen.add((it[0], it[3], it[4]))
                uniq.append(it)
        stats["maia cache hits"] += len(items) - len(uniq)
        if not uniq:
            return
        t = time.perf_counter()
        preds = maia.predict([Query(fen, mv, se, oe) for _, fen, mv, se, oe in uniq])
        secs["maia"] += time.perf_counter() - t
        stats["maia calls"] += len(uniq)
        for (h, _, _, se, oe), pr in zip(uniq, preds):
            maia_c.add({"pos_hash": h, "self_elo": se, "oppo_elo": oe, "moves": pr.moves,
                        "logp": pr.logp.tolist(), "wdl": list(pr.wdl) if pr.wdl else None})

    def sf_calls(items: list[tuple[int, str, list[str]]]) -> None:
        seen, uniq = set(), []
        for h, fen, mv in items:
            if not sf_c.has((h,)) and h not in seen:
                seen.add(h)
                uniq.append((h, fen, mv))
        stats["multipv cache hits"] += len(items) - len(uniq)
        if not uniq:
            return
        t = time.perf_counter()
        vals = sf.evaluate([Query(fen, mv, 0, 0) for _, fen, mv in uniq])
        secs["stockfish multipv"] += time.perf_counter() - t
        stats["multipv calls"] += len(uniq)
        for (h, _, _), v in zip(uniq, vals):
            sf_c.add({"pos_hash": h, "moves": v.moves, "value": v.value.tolist(), "cp": v.cp.tolist(),
                      "mate": v.mate.tolist(), "wdl": list(v.wdl) if v.wdl else None})

    def stop(*_):   # systemctl stop: flush the caches on the way out
        raise KeyboardInterrupt("SIGTERM")

    signal.signal(signal.SIGTERM, stop)
    done = 0
    try:
        for s in range(0, len(rows), chunk):
            if max_seconds is not None and time.time() - t0 > max_seconds:
                stats["stopped at max_seconds"] = 1
                break
            part = rows[s : s + chunk]
            # 1. Maia at the position, ratings of the position's own games
            items = []
            for r in part:
                w, bl = maia_elo(r["white_elo"]), maia_elo(r["black_elo"])
                se, oe = (w, bl) if r["stm_white"] else (bl, w)
                r["_elo"] = (se, oe)
                items.append((r["pos_hash"], r["fen"], r["moves_uci"].split(), se, oe))
            maia_calls(items)
            # 2. the children the side to move is likely to play
            child_items, sf_items = [], []
            for r in part:
                x = maia_c.get((r["pos_hash"], *r["_elo"]))
                p = np.exp(np.asarray(x["logp"], np.float64))
                path = r["moves_uci"].split()
                sf_items.append((r["pos_hash"], r["fen"], path))
                board = chess.Board(r["fen"])
                for mv, _ in children_of(x["moves"], p, b):
                    bc = board.copy(stack=False)
                    bc.push_uci(mv)
                    if bc.is_game_over(claim_draw=False):
                        stats["children game over"] += 1
                        continue
                    h = pos_hash(bc)
                    child_items.append((h, bc.fen(), path + [mv], r["_elo"][1], r["_elo"][0]))
                    sf_items.append((h, bc.fen(), path + [mv]))
                    stats["children"] += 1
            maia_calls(child_items)
            sf_calls(sf_items)
            # 3. the root value at the larger budget
            for r in part:
                if root_c.has((r["pos_hash"],)):
                    stats["root cache hits"] += 1
                    continue
                t = time.perf_counter()
                res = root(r["moves_uci"].split())
                secs["stockfish root"] += time.perf_counter() - t
                stats["root calls"] += 1
                stats["root used a bound"] += int(res["used_bound"])
                root_c.add({"pos_hash": r["pos_hash"], **res})
            done += len(part)
            if sum(len(c.pending) for c in (maia_c, sf_c, root_c)) >= flush_every or s + chunk >= len(rows):
                for c in (maia_c, sf_c, root_c):
                    c.flush()
                el = time.time() - t0
                print(f"[{el:7.0f} s] {done}/{len(rows)} positions; {dict(stats)}; "
                      f"seconds {({k: round(v) for k, v in secs.items()})}", flush=True)
    finally:
        for c in (maia_c, sf_c, root_c):
            c.flush()
        sf.close()
        root.close()
    out = {"positions_done": done, "stats": dict(stats), "seconds": {k: round(v, 1) for k, v in secs.items()},
           "wall": round(time.time() - t0, 1), "budget": asdict(b), "stockfish": sf.engine_id,
           "peak_rss_mb": _peak_mb()}
    print(json.dumps(out, indent=1), flush=True)
    return out


def _peak_mb() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) // 1024
    return -1


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="dev_n100", help="position set (positions_<tag>.parquet)")
    ap.add_argument("--fraction", type=float, default=None, help="a fixed fraction of the positions, by hash")
    ap.add_argument("--max-seconds", type=float, default=None)
    d = Budget()
    ap.add_argument("--root-nodes", type=int, default=d.root_nodes)
    ap.add_argument("--multipv-nodes", type=int, default=d.multipv_nodes)
    ap.add_argument("--maia-size", default=d.maia_size)
    ap.add_argument("--child-mass", type=float, default=d.child_mass)
    ap.add_argument("--max-children", type=int, default=d.max_children)
    ap.add_argument("--min-child-p", type=float, default=d.min_child_p)
    a = ap.parse_args(argv)
    b = Budget(a.root_nodes, a.multipv_nodes, a.maia_size, a.child_mass, a.max_children, a.min_child_p)
    pos = pl.read_parquet(set_paths(a.tag)[0])
    pos = pos.sort("pos_hash")   # hash order is a random order: a partial run is a random subset
    if a.fraction is not None:
        pos = pos.filter((pl.col("pos_hash") % 1000) < a.fraction * 1000)
    print(f"{pos.height} positions from {a.tag}; budget {b}", flush=True)
    res = run(pos, b, max_seconds=a.max_seconds)
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    (OUT / "runs" / f"compute_{a.tag}_{int(time.time())}.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
