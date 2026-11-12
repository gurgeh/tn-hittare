"""Model 1's probabilities for the positions of a `shrinkfit collect` sample: the prior file
`tnh.web.probfit` fits the Prob mix with.

    scripts/tnh-run web-probprior 4 uv run python -m tnh.web.probprior --data PAIRS.parquet --out PRIOR_DIR

For each position: the game so far (the path the descent first reached it by, from
`<data>.paths.parquet`), and two rating settings: the mean White and Black rating of the
position's pre-split classical games with both ratings (`p_mean`; 2200 for a side when the
position has none, flagged `rated`), and 2200/2200 (`p_2200`). One row per legal move.
Written in parts of `--chunk` positions; a part already on disk is skipped, so a stopped run
resumes.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import chess
import numpy as np
import polars as pl

from tnh.models.base import Query


def position_ratings(raw: pl.DataFrame) -> pl.DataFrame:
    tr = raw.filter(pl.col("part") == "train")
    g = tr.group_by("pos").agg(pl.col("sum_white", "sum_black", "n_rated").sum())
    rated = pl.col("n_rated") > 0
    return g.select("pos", "n_rated",
                    white=pl.when(rated).then(pl.col("sum_white") / pl.col("n_rated")),
                    black=pl.when(rated).then(pl.col("sum_black") / pl.col("n_rated")))


def queries(rows: list[tuple[int, str, float | None, float | None]], fixed: float | None):
    out, keys = [], []
    for pos, path, w, b in rows:
        moves = path.split() if path else []
        board = chess.Board()
        for m in moves:
            board.push_uci(m)
        we = fixed if fixed is not None else (w if w is not None else 2200.0)
        be = fixed if fixed is not None else (b if b is not None else 2200.0)
        mover, opp = (we, be) if board.turn == chess.WHITE else (be, we)
        out.append(Query(board.fen(), moves, int(round(mover)), int(round(opp))))
        keys.append(pos)
    return out, keys


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tnh.web.probprior", description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="a directory for the parts")
    ap.add_argument("--model", default="policy-trees", help="a move-model spec of tnh.web.models")
    ap.add_argument("--chunk", type=int, default=2000)
    a = ap.parse_args(argv)
    from tnh.web.models import load_move_model

    raw = pl.read_parquet(a.data)
    paths = pl.read_parquet(a.data.with_suffix(".paths.parquet"))
    pos = paths.join(position_ratings(raw), on="pos", how="left").sort("pos")
    rows = list(pos.select("pos", "path", "white", "black").iter_rows())
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    model = load_move_model(a.model)
    print(f"model {getattr(model, 'name', a.model)} loaded in {time.perf_counter() - t0:.0f} s; "
          f"{len(rows)} positions, {pos['n_rated'].fill_null(0).eq(0).sum()} without rated pre-split games", flush=True)
    for k in range(0, len(rows), a.chunk):
        part = a.out / f"part-{k // a.chunk:04d}.parquet"
        if part.exists():
            continue
        t = time.perf_counter()
        chunk = rows[k:k + a.chunk]
        cols: dict[str, list] = {"pos": [], "move": []}
        probs = {}
        for label, fixed in (("p_mean", None), ("p_2200", 2200.0)):
            qs, keys = queries(chunk, fixed)
            preds = model.predict(qs)
            probs[label] = [np.exp(p.logp.astype(np.float64)) for p in preds]
            if label == "p_mean":
                for key, p in zip(keys, preds):
                    cols["pos"] += [key] * len(p.moves)
                    cols["move"] += p.moves
        df = pl.DataFrame({"pos": pl.Series(cols["pos"], dtype=pl.UInt64), "move": cols["move"],
                           "p_mean": np.concatenate(probs["p_mean"]), "p_2200": np.concatenate(probs["p_2200"])})
        df.write_parquet(part)
        print(f"part {k // a.chunk}: {len(chunk)} positions, {df.height} moves, "
              f"{len(chunk) * 2 / (time.perf_counter() - t):.0f} queries/s", flush=True)
    print(f"done in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
