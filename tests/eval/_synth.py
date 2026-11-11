"""Synthetic positions and feature files where the right answer is known."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


def softmax_rows(z: list[np.ndarray]) -> list[np.ndarray]:
    out = []
    for x in z:
        e = np.exp(x - x.max())
        out.append(e / e.sum())
    return out


def positions(n_rows: int, rng: np.random.Generator, *, n_tournaments: int = 20, split: str = "test",
              min_moves: int = 5, max_moves: int = 30, elo=(1800, 2500)) -> tuple[pl.DataFrame, list[list[str]]]:
    """A sample-shaped frame and each row's move names (m0, m1, ...)."""
    n_moves = rng.integers(min_moves, max_moves + 1, n_rows)
    gids = [f"{ {"test": "e", "train": "a"}.get(split, "c")}{i // 10:015x}" for i in range(n_rows)]
    df = pl.DataFrame({
        "game_id": gids,
        "ply": [20 + i % 10 for i in range(n_rows)],
        "played_uci": ["" for _ in range(n_rows)],
        "n_legal": n_moves.astype(np.int64),
        "mover_elo": rng.integers(elo[0], elo[1], n_rows),
        "opp_elo": rng.integers(elo[0], elo[1], n_rows),
        "tournament_id": [f"{split}-t{(i // 10) % n_tournaments}" for i in range(n_rows)],
        "weight": np.ones(n_rows),
        "split": [split] * n_rows,
        "in_window": [True] * n_rows,
    }).with_columns(((pl.col("mover_elo") // 100) * 100).alias("bucket"))
    moves = [[f"m{j}" for j in range(k)] for k in n_moves]
    return df, moves


def write_features(d: Path, df: pl.DataFrame, moves: list[list[str]], cols: dict[str, list], meta: dict | None = None,
                   wdl: list | None = None, part: int = 0) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    data = {"game_id": pa.array(df.get_column("game_id").to_list()), "ply": pa.array(df.get_column("ply").to_list(), pa.int64()),
            "moves": pa.array(moves, pa.list_(pa.string()))}
    for k, v in cols.items():
        t = pa.int16() if k == "mate" else pa.float32()
        data[k] = pa.array([np.asarray(x).tolist() for x in v], pa.list_(t))
    if wdl is not None:
        data["wdl"] = pa.array([list(map(float, x)) for x in wdl], pa.list_(pa.float32()))
    pq.write_table(pa.table(data), d / f"part-{part:05d}.parquet")
    (d / "meta.json").write_text(json.dumps(meta or {"model": "synth", "variant": d.name}))
    return d


def draw_played(df: pl.DataFrame, moves: list[list[str]], probs: list[np.ndarray], rng) -> pl.DataFrame:
    idx = [int(rng.choice(len(p), p=p)) for p in probs]
    return df.with_columns(pl.Series("played_uci", [m[i] for m, i in zip(moves, idx)]))
