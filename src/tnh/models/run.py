"""Feature runner: a positions file in, feature files out, resumable.

    uv run python -m tnh.models.run --model maia3 --size 23m --offsets -200 -100 0 100 200 \
        --positions ~/out/tn-hittare/corpus/positions.parquet [--limit N]
    uv run python -m tnh.models.run --model maia3 --size 79m --elo offset+300-opp+100 fixed2200-2200 ...
    uv run python -m tnh.models.run --model stockfish --nodes 10000 --processes 16 ...
    uv run python -m tnh.models.run --model lc0-policy ...
    uv run python -m tnh.models.run --model lc0-search --nodes 400 [--wdl-elo offset+0] ...

Maia-3 takes any number of Elo settings (`--offsets` for both sides at once,
`--mover-offsets` x `--opp-offsets` for a grid, `--elo` for variant names) and runs
them all on one encoding of each batch, writing one variant directory per setting.

The games file (for `moves_uci`) defaults to `games.parquet` next to the positions
file, else `corpus/games.parquet` (samples live in `corpus/samples/` without one). Feature files go to `<out>/<variant>/`, `--out` defaulting to
`features/<model>/`. Output is written as `part-NNNNN.parquet`, one per chunk, each
renamed into place only when complete, so an interrupted run loses at most the chunk
in progress. A rerun skips every `(game_id, ply)` already on disk in each variant
directory. `meta.json` records the options, and a run into a directory whose
`meta.json` has other options is refused.
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
from tnh import paths
from tnh.paths import FEATURES


def done_keys(out: Path) -> pl.DataFrame:
    parts = sorted(out.glob("part-*.parquet"))
    if not parts:
        return pl.DataFrame(schema={"game_id": pl.String, "ply": pl.Int64})
    return pl.concat([pl.read_parquet(p, columns=["game_id", "ply"]) for p in parts]).with_columns(
        pl.col("ply").cast(pl.Int64)
    )


def load_todo(positions: Path, games: Path, outs: list[Path], limit: int | None, split: str | None) -> pl.DataFrame:
    """Rows still missing in at least one of `outs`, in game order, with `moves_uci`.

    Column `need_<k>` says whether output `k` still lacks the row.
    """
    pos = pl.read_parquet(positions, columns=["game_id", "ply", "fen", "mover_elo", "opp_elo", "split"])
    n_all = pos.height
    if split is not None:
        pos = pos.filter(pl.col("split") == split)
    pos = pos.with_columns(pl.col("ply").cast(pl.Int64))
    for k, out in enumerate(outs):
        done = done_keys(out).with_columns(pl.lit(False).alias(f"need_{k}"))
        pos = pos.join(done, on=["game_id", "ply"], how="left").with_columns(pl.col(f"need_{k}").fill_null(True))
        print(f"{out}: {pos.height - pos[f'need_{k}'].sum()} of {pos.height} selected positions already on disk")
    need = pl.any_horizontal([f"need_{k}" for k in range(len(outs))]) if outs else pl.lit(True)
    todo = pos.filter(need).sort(["game_id", "ply"])
    print(f"positions: {n_all} in file, {pos.height} selected"
          f"{'' if split is None else f' (split={split})'}, {todo.height} missing in at least one output", flush=True)
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
    return todo.join(g, on="game_id", how="left", maintain_order="left")


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


def _list_i16(rows: list[np.ndarray]) -> pa.Array:
    offsets = np.zeros(len(rows) + 1, dtype=np.int32)
    np.cumsum([len(r) for r in rows], out=offsets[1:])
    flat = np.concatenate(rows).astype(np.int16) if rows else np.zeros(0, dtype=np.int16)
    return pa.ListArray.from_arrays(pa.array(offsets), pa.array(flat, type=pa.int16()))


def write_part(out: Path, keys: pl.DataFrame, columns: dict[str, pa.Array]) -> Path:
    table = pa.table({
        "game_id": pa.array(keys.get_column("game_id").to_list(), type=pa.string()),
        "ply": pa.array(keys.get_column("ply").to_numpy(), type=pa.int64()),
        **columns,
    })
    existing = [int(p.stem.split("-")[1]) for p in out.glob("part-*.parquet")]
    path = out / f"part-{(max(existing) + 1) if existing else 0:05d}.parquet"
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)
    return path


def move_model_columns(preds) -> dict[str, pa.Array]:
    return {
        "moves": pa.array([p.moves for p in preds], type=pa.list_(pa.string())),
        "logp": _list_f32([p.logp for p in preds]),
        "wdl": _wdl([p.wdl for p in preds]),
    }


def value_model_columns(vals) -> dict[str, pa.Array]:
    return {
        "moves": pa.array([v.moves for v in vals], type=pa.list_(pa.string())),
        "value": _list_f32([v.value for v in vals]),
        "cp": _list_f32([v.cp for v in vals]),
        "mate": _list_i16([v.mate for v in vals]),
        "wdl": _wdl([v.wdl for v in vals]),
    }


def check_meta(out: Path, meta: dict) -> None:
    path = out / "meta.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old != meta:
            raise SystemExit(f"{out} holds features made with other options:\n  on disk {old}\n  now     {meta}")
    else:
        out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def maia3_settings(a) -> list:
    from tnh.models.maia3 import Elo

    clamp = (a.elo_min, a.elo_max)
    elos = [Elo.parse(n) for n in a.elo or []]
    if any(e.clamp != (0.0, 5000.0) for e in elos) and clamp != (0.0, 5000.0):
        raise SystemExit("give the clamp either in the --elo names or with --elo-min/--elo-max, not both")
    if clamp != (0.0, 5000.0):
        elos = [Elo(**{**e.__dict__, "clamp": clamp}) for e in elos]
    elos += [Elo.offset(x, clamp=clamp) for x in a.offsets or []]
    if bool(a.mover_offsets) != bool(a.opp_offsets):
        raise SystemExit("--mover-offsets and --opp-offsets go together")
    elos += [Elo.offset(m, o, clamp=clamp) for m in a.mover_offsets or [] for o in a.opp_offsets or []]
    seen, unique = set(), []
    for e in elos:
        if e.name not in seen:
            seen.add(e.name)
            unique.append(e)
    if not unique:
        raise SystemExit("maia3 needs at least one Elo setting: --offsets, --mover-offsets/--opp-offsets or --elo")
    return unique


def games_file(positions: Path, games: Path | None) -> Path:
    """`--games`, else games.parquet next to the positions file, else the corpus one."""
    if games is not None:
        if not games.exists():
            raise SystemExit(f"--games {games} does not exist")
        return games
    for candidate in (positions.parent / "games.parquet", paths.CORPUS / "games.parquet"):
        if candidate.exists():
            return candidate
    raise SystemExit(f"no games file: none next to {positions} and none at {paths.CORPUS / 'games.parquet'}; "
                     f"pass --games")


def run(a) -> None:
    games = games_file(a.positions, a.games)
    t0 = time.perf_counter()
    if a.model == "maia3":
        from tnh.models.maia3 import Maia3

        elos = maia3_settings(a)
        model = Maia3(a.size, batch_size=a.batch_size, amp=not a.fp32)
        base = a.out or FEATURES / model.name
        outs = [base / model.variant(e) for e in elos]
        for e, out in zip(elos, outs):
            check_meta(out, {"model": model.name, "variant": model.variant(e), "self_elo": e.self_elo,
                             "oppo_elo": e.oppo_elo, "mover_offset": e.mover_offset, "opp_offset": e.opp_offset,
                             "elo_clamp": list(e.clamp), "amp": model.amp})
        print(f"{model.name}: {len(elos)} settings: {', '.join(e.name for e in elos)}")
    elif a.model == "stockfish":
        from tnh.models.stockfish import Stockfish

        model = Stockfish(a.nodes, processes=a.processes, hash_mb=a.hash_mb, per_move=a.nodes_per_move)
        outs = [(a.out or FEATURES / model.name) / model.variant]
        check_meta(outs[0], {"model": model.name, "variant": model.variant, "engine": model.engine_id,
                             "nodes": a.nodes, "per_move": a.nodes_per_move, "hash_mb": a.hash_mb,
                             "deterministic": model.deterministic})
    elif a.model == "lc0-policy":
        from tnh.models.lc0 import Lc0Policy

        model = Lc0Policy(a.network, batch_size=a.batch_size)
        outs = [(a.out or FEATURES / model.name) / model.variant]
        check_meta(outs[0], {"model": model.name, "variant": model.variant, "network": model.network.name,
                             "backend": model.backend_name})
    elif a.model == "lc0-search":
        from tnh.models.lc0 import Lc0Search
        from tnh.models.maia3 import Elo

        model = Lc0Search(a.nodes, a.network, wdl_elo=Elo.parse(a.wdl_elo) if a.wdl_elo else None,
                          processes=a.lc0_processes)
        outs = [(a.out or FEATURES / model.name) / model.variant]
        check_meta(outs[0], {"model": model.name, "variant": model.variant, "network": model.network.name,
                             "nodes": a.nodes, "wdl_elo": a.wdl_elo, "options": model.options})
    else:
        raise SystemExit(f"unknown model {a.model}")

    todo = load_todo(a.positions, games, outs, a.limit, a.split)
    n = 0
    try:
        for start in range(0, todo.height, a.chunk):
            chunk = todo.slice(start, a.chunk)
            qs = queries(chunk)
            needs = [chunk.get_column(f"need_{k}").to_numpy() for k in range(len(outs))]
            if a.model == "maia3":
                active = [k for k in range(len(outs)) if needs[k].any()]
                results = model.predict_many(qs, [elos[k] for k in active])
                for k, preds in zip(active, results):
                    rows = np.flatnonzero(needs[k])
                    write_part(outs[k], chunk[rows], move_model_columns([preds[i] for i in rows]))
            elif a.model == "lc0-policy":
                write_part(outs[0], chunk, move_model_columns(model.predict(qs)))
            else:
                vals = model.evaluate(qs)
                write_part(outs[0], chunk, value_model_columns(vals))
            n += chunk.height
            dt = time.perf_counter() - t0
            print(f"{n}/{todo.height} positions, {n * len(outs) / dt:.0f} position-settings/s "
                  f"({n / dt:.0f} positions/s)", flush=True)
    finally:
        if a.model == "maia3":
            if model.n_predicted:
                print(f"mean softmax mass on illegal moves before renormalising: "
                      f"{model.illegal_mass_sum / model.n_predicted:.4f} over {model.n_predicted} predictions")
        elif a.model == "stockfish":
            print(f"stockfish ({model.engine_id}): {dict(model.stats)}")
            model.close()
        elif a.model == "lc0-policy":
            print(f"lc0 policy: input transforms used {dict(model.transforms)}")
        else:
            print(f"lc0 search: {dict(model.stats)}")
            model.close()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m tnh.models.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["maia3", "stockfish", "lc0-policy", "lc0-search"])
    p.add_argument("--positions", type=Path, required=True)
    p.add_argument("--games", type=Path,
                   help="default: games.parquet next to --positions, else <corpus>/games.parquet")
    p.add_argument("--out", type=Path, help="parent of the variant directories; default: features/<model>")
    p.add_argument("--limit", type=int, help="at most this many positions not yet done")
    p.add_argument("--split", choices=["train", "val", "test"], help="only positions from this split")
    p.add_argument("--chunk", type=int, default=50_000, help="positions per part file")
    g = p.add_argument_group("maia3")
    g.add_argument("--size", default="23m", choices=["5m", "23m", "79m"])
    g.add_argument("--offsets", type=float, nargs="+", help="offsets added to both ratings, one variant each")
    g.add_argument("--mover-offsets", type=float, nargs="+", help="with --opp-offsets: every combination")
    g.add_argument("--opp-offsets", type=float, nargs="+")
    g.add_argument("--elo", nargs="+", help="variant names: offset+300, offset+300-opp+100, fixed2200-2200")
    g.add_argument("--elo-min", type=float, default=0.0)
    g.add_argument("--elo-max", type=float, default=5000.0)
    g.add_argument("--batch-size", type=int, default=512)
    g.add_argument("--fp32", action="store_true", help="no autocast (the reference uses float16 autocast)")
    g = p.add_argument_group("stockfish and lc0-search")
    g.add_argument("--nodes", type=int, default=10_000, help="nodes per position for the whole search")
    g.add_argument("--nodes-per-move", action="store_true", help="--nodes is per legal move, not per position")
    g.add_argument("--processes", type=int, default=os.cpu_count())
    g.add_argument("--hash-mb", type=int, default=16)
    g = p.add_argument_group("lc0")
    g.add_argument("--network", default="BT4-1024x15x32h-swa-6147500.pb.gz", help="file under <lc0>/networks, or a path")
    g.add_argument("--wdl-elo", help="lc0-search: calibrate to two players, e.g. offset+0 or fixed2200-2000")
    g.add_argument("--lc0-processes", type=int, default=4, help="lc0-search: engine processes sharing the GPU")
    run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
