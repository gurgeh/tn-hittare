"""Feature runner: a positions file in, feature files out, resumable.

    uv run python -m tnh.models.run --model maia3 --size 23m --elo-offset 0 \\
        --positions ~/out/tn-hittare/corpus/positions.parquet --out <dir> [--limit N]
    uv run python -m tnh.models.run --model maia3 --size 79m --self-elo 2200 --oppo-elo 2200 ...
    uv run python -m tnh.models.run --model stockfish --nodes 10000 --processes 16 ...

The games file (for `moves_uci`) defaults to `games.parquet` next to the positions
file. `--out` defaults to `features/<model>/<variant>/`. Output is written as
`part-NNNNN.parquet`, one per chunk, each renamed into place only when complete, so
an interrupted run loses at most the chunk in progress. A rerun skips every
`(game_id, ply)` already on disk. `meta.json` records the model options, and a rerun
with different options into the same directory is refused.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from tnh.models.base import Query
from tnh.paths import FEATURES


def done_keys(out: Path) -> pl.DataFrame:
    parts = sorted(out.glob("part-*.parquet"))
    if not parts:
        return pl.DataFrame(schema={"game_id": pl.String, "ply": pl.Int64})
    return pl.concat([pl.read_parquet(p, columns=["game_id", "ply"]) for p in parts]).with_columns(
        pl.col("ply").cast(pl.Int64)
    )


def load_todo(positions: Path, games: Path, out: Path, limit: int | None, split: str | None) -> pl.DataFrame:
    pos = pl.read_parquet(positions, columns=["game_id", "ply", "fen", "mover_elo", "opp_elo", "split"])
    n_all = pos.height
    if split is not None:
        pos = pos.filter(pl.col("split") == split)
    pos = pos.with_columns(pl.col("ply").cast(pl.Int64))
    done = done_keys(out)
    todo = pos.join(done, on=["game_id", "ply"], how="anti").sort(["game_id", "ply"])
    print(f"positions: {n_all} in file, {pos.height} selected"
          f"{'' if split is None else f' (split={split})'}, {pos.height - todo.height} already on disk, "
          f"{todo.height} to do", flush=True)
    if limit is not None:
        todo = todo.head(limit)
    ids = todo.get_column("game_id").unique()
    g = (pl.scan_parquet(games).select("game_id", "moves_uci")
         .join(pl.LazyFrame({"game_id": ids}), on="game_id", how="semi").collect())
    missing = ids.len() - g.height
    if missing:
        raise SystemExit(f"{missing} game_ids in {positions} are not in {games}")
    if g.get_column("game_id").is_duplicated().any():
        raise SystemExit(f"duplicate game_id in {games}")
    return todo.join(g, on="game_id", how="left")


def queries(df: pl.DataFrame) -> list[Query]:
    out = []
    split_cache: tuple[str, list[str]] = ("", [])
    for gid, ply, fen, me, opp, moves in df.select("game_id", "ply", "fen", "mover_elo", "opp_elo", "moves_uci").iter_rows():
        if split_cache[0] != gid:
            split_cache = (gid, moves.split())
        out.append(Query(fen, split_cache[1][:ply], int(me), int(opp)))
    return out


def _list_f32(rows: list[np.ndarray]) -> pa.Array:
    offsets = np.zeros(len(rows) + 1, dtype=np.int32)
    np.cumsum([len(r) for r in rows], out=offsets[1:])
    flat = np.concatenate(rows).astype(np.float32) if rows else np.zeros(0, dtype=np.float32)
    return pa.ListArray.from_arrays(pa.array(offsets), pa.array(flat, type=pa.float32()))


def _wdl(rows: list[tuple | None]) -> pa.Array:
    return pa.array([None if w is None else [float(x) for x in w] for w in rows], type=pa.list_(pa.float32()))


def write_part(out: Path, chunk: pl.DataFrame, moves, column: str, values, wdl) -> Path:
    table = pa.table({
        "game_id": pa.array(chunk.get_column("game_id").to_list(), type=pa.string()),
        "ply": pa.array(chunk.get_column("ply").to_numpy(), type=pa.int64()),
        "moves": pa.array(moves, type=pa.list_(pa.string())),
        column: _list_f32(values),
        "wdl": _wdl(wdl),
    })
    existing = [int(p.stem.split("-")[1]) for p in out.glob("part-*.parquet")]
    path = out / f"part-{(max(existing) + 1) if existing else 0:05d}.parquet"
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)
    return path


def check_meta(out: Path, meta: dict) -> None:
    path = out / "meta.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old != meta:
            raise SystemExit(f"{out} holds features made with other options:\n  on disk {old}\n  now     {meta}")
    else:
        out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def build_model(a):
    if a.model == "maia3":
        from tnh.models.maia3 import Maia3

        if a.elo_offset is None and (a.self_elo is None or a.oppo_elo is None):
            raise SystemExit("maia3 needs --elo-offset, or --self-elo and --oppo-elo")
        m = Maia3(a.size, self_elo=a.self_elo, oppo_elo=a.oppo_elo, elo_offset=a.elo_offset,
                  elo_clamp=(a.elo_min, a.elo_max), batch_size=a.batch_size, amp=not a.fp32)
        meta = {"model": m.name, "variant": m.variant, "self_elo": a.self_elo, "oppo_elo": a.oppo_elo,
                "elo_offset": a.elo_offset, "elo_clamp": [a.elo_min, a.elo_max], "amp": m.amp}
        return m, meta
    if a.model == "stockfish":
        from tnh.models.stockfish import Stockfish

        m = Stockfish(a.nodes, processes=a.processes, hash_mb=a.hash_mb, per_move=a.nodes_per_move)
        meta = {"model": m.name, "variant": m.variant, "engine": m.engine_id, "nodes": a.nodes,
                "per_move": a.nodes_per_move, "hash_mb": a.hash_mb}
        return m, meta
    raise SystemExit(f"unknown model {a.model}")


def run(a) -> None:
    model, meta = build_model(a)
    out = a.out or FEATURES / meta["model"] / meta["variant"]
    check_meta(out, meta)
    games = a.games or a.positions.parent / "games.parquet"
    todo = load_todo(a.positions, games, out, a.limit, a.split)
    t0, n = time.perf_counter(), 0
    try:
        for start in range(0, todo.height, a.chunk):
            chunk = todo.slice(start, a.chunk)
            qs = queries(chunk)
            if a.model == "maia3":
                preds = model.predict_logp(qs)
                path = write_part(out, chunk, [p.moves for p in preds], "logp", [p.logp for p in preds],
                                  [p.wdl for p in preds])
            else:
                preds = model.evaluate(qs)
                path = write_part(out, chunk, [p.moves for p in preds], "value", [p.values for p in preds],
                                  [p.wdl for p in preds])
            n += chunk.height
            dt = time.perf_counter() - t0
            print(f"{path.name}: {n}/{todo.height} positions, {n / dt:.0f} pos/s", flush=True)
    finally:
        if a.model == "maia3":
            if model.n_predicted:
                print(f"mean softmax mass on illegal moves before renormalising: "
                      f"{model.illegal_mass_sum / model.n_predicted:.4f} over {model.n_predicted} positions")
        else:
            print(f"stockfish ({model.engine_id}): {dict(model.stats)}")
            model.close()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m tnh.models.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["maia3", "stockfish"])
    p.add_argument("--positions", type=Path, required=True)
    p.add_argument("--games", type=Path, help="default: games.parquet next to --positions")
    p.add_argument("--out", type=Path, help="default: features/<model>/<variant>")
    p.add_argument("--limit", type=int, help="at most this many new positions")
    p.add_argument("--split", choices=["train", "val", "test"], help="only positions from this split")
    p.add_argument("--chunk", type=int, default=50_000, help="positions per part file")
    g = p.add_argument_group("maia3")
    g.add_argument("--size", default="23m", choices=["5m", "23m", "79m"])
    g.add_argument("--elo-offset", type=float)
    g.add_argument("--self-elo", type=float)
    g.add_argument("--oppo-elo", type=float)
    g.add_argument("--elo-min", type=float, default=0.0)
    g.add_argument("--elo-max", type=float, default=5000.0)
    g.add_argument("--batch-size", type=int, default=512)
    g.add_argument("--fp32", action="store_true", help="no autocast (the reference uses float16 autocast)")
    g = p.add_argument_group("stockfish")
    g.add_argument("--nodes", type=int, default=10_000, help="nodes per position for the whole MultiPV search")
    g.add_argument("--nodes-per-move", action="store_true", help="--nodes is per legal move, not per position")
    g.add_argument("--processes", type=int, default=os.cpu_count())
    g.add_argument("--hash-mb", type=int, default=16)
    run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
