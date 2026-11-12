"""The shrinkage weights of the move table, estimated from the edge tables by method of moments.

    scripts/tnh-run web-calibrate 2 uv run python -m tnh.web.calibrate [--table full] [--positions 3000]

The table shrinks a move's database figure toward a prior with weight `kappa` in games:
`(n * db + kappa * prior) / (n + kappa)`. For a normal-normal model the weight that
minimises the error is `kappa = sigma^2 / tau^2`: the variance of one game's figure within
a move over the variance of the moves' true figures around the prior.

Here the prior is the position's own mean under the filters (all moves pooled), for three
figures: White's score, the draw rate and the mean rating of each colour. For each
(position, move) pair the squared deviation of the move's mean from the position's mean has
expectation `tau^2 + sigma^2 * (1/n - 1/N)` (N the position's games), so a weighted least
squares fit of the squared deviations on `1/n - 1/N` gives `sigma^2` as the slope and
`tau^2` as the intercept. For the score and the draw rate `sigma^2` is also known per move
from the results, which gives a second estimate (DerSimonian-Laird) to check the first.

For Score and Draw the table's prior is model 2, not the position mean. Model 2 should sit
closer to the truth than the position mean, so the `kappa_v` estimated here is a lower
bound on the right one: it trusts the database more than it should. Refit with model 2's
values once they exist for a sample.

Positions: the most common ones first, walked from the start position through children
reached by at least `--min-games` games, with the classical filter the page uses by default.
"""

from __future__ import annotations

import argparse
import heapq
import json
import time

import chess
import numpy as np
import polars as pl

from tnh.data.edges import EdgeDB, pos_hash


def walk(db: EdgeDB, n_positions: int, min_games: int, tempi: list[str] | None, split_year: int) -> pl.DataFrame:
    """One row per (position, move, part) over the most common positions; `part` is "all",
    "train" (years before `split_year`) or "test" (from it)."""
    start = pos_hash(chess.Board())
    heap = [(-db.total(start)["n"], start)]
    seen = {start}
    rows = []
    done = 0
    while heap and done < n_positions:
        _, h = heapq.heappop(heap)
        p = db.panel(h, tempi=tempi)
        if p["total"] is None or p["moves"].height == 0:
            continue
        white = p["total"]["epd"].split()[1] == "w"
        m = p["moves"].with_columns(pos=pl.lit(h, pl.UInt64), white_to_move=pl.lit(white))
        cols = ["pos", "white_to_move", "part", "move", "n", "white_wins", "draws", "black_wins",
                "n_rated", "sum_mover", "sum_opp"]
        rows.append(m.with_columns(part=pl.lit("all"), sum_mover=pl.col("mover_elo") * pl.col("n_rated"),
                                   sum_opp=pl.col("opp_elo") * pl.col("n_rated")).select(cols))
        y = p["by_year"].filter(pl.col("year") > 0).with_columns(
            pos=pl.lit(h, pl.UInt64), white_to_move=pl.lit(white),
            part=pl.when(pl.col("year") < split_year).then(pl.lit("train")).otherwise(pl.lit("test")),
            sum_mover=pl.col("mover_elo").fill_null(0) * pl.col("n_rated"),
            sum_opp=pl.col("opp_elo").fill_null(0) * pl.col("n_rated"))
        rows.append(y.group_by("pos", "white_to_move", "part", "move").agg(
            pl.col("n", "white_wins", "draws", "black_wins", "n_rated", "sum_mover", "sum_opp").sum()).select(cols))
        done += 1
        for c, cn in m.select("child_hash", "child_n").iter_rows():
            if cn is not None and cn >= min_games and c not in seen:
                seen.add(c)
                heapq.heappush(heap, (-cn, c))
    return pl.concat(rows)


def regression(dev2: np.ndarray, x: np.ndarray, iterations: int = 5) -> tuple[float, float]:
    """Feasible GLS of dev2 on [1, x]: (intercept tau^2, slope sigma^2).

    A squared deviation has variance proportional to its expectation squared, so each
    round weights the pairs by 1 / (tau^2 + sigma^2 x)^2 from the round before.
    """
    X = np.column_stack([np.ones_like(x), x])
    w = np.ones_like(x)
    coef = np.zeros(2)
    for _ in range(iterations):
        W = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(X * W[:, None], dev2 * W, rcond=None)
        fit = np.maximum(X @ coef, 1e-12)
        w = 1 / fit ** 2
    return float(coef[0]), float(coef[1])


def holdout(df: pl.DataFrame, fig: str, count: str, grid: np.ndarray) -> dict:
    """The kappa that best predicts each move's figure in the test years from the train years,
    shrunk toward the position's train mean; squared error weighted by the test count."""
    tr = df.filter((pl.col("part") == "train") & (pl.col(count) > 0)).drop_nulls(fig)
    te = df.filter((pl.col("part") == "test") & (pl.col(count) > 0)).drop_nulls(fig)
    prior = tr.group_by("pos").agg(((pl.col(fig) * pl.col(count)).sum() / pl.col(count).sum()).alias("prior"))
    j = te.select("pos", "move", pl.col(fig).alias("y"), pl.col(count).alias("w")).join(
        prior, on="pos").join(tr.select("pos", "move", pl.col(fig).alias("m"), pl.col(count).alias("n")),
                              on=["pos", "move"], how="left").with_columns(pl.col("n").fill_null(0), pl.col("m").fill_null(0))
    y, w, pr, m, n = (j[c].to_numpy().astype(float) for c in ("y", "w", "prior", "m", "n"))
    out = {}
    # "all": every move, weighted by its test count (common moves dominate, kappa hardly matters
    # for them); "rare": moves with 1-100 train games, the rows where kappa matters, the error of
    # each pair weighted by its test count, so the noise of y is the same per game.
    for label, sel in (("all", np.ones_like(n, bool)), ("rare", (n >= 1) & (n <= 100))):
        ww, yy, pp, mm, nn = w[sel], y[sel], pr[sel], m[sel], n[sel]
        loss = [float(np.sum(ww * ((nn * mm + k * pp) / (nn + k) - yy) ** 2) / ww.sum()) for k in grid]
        best = int(np.argmin(loss))
        out[label] = {"kappa": float(grid[best]), "loss": loss[best],
                      "curve": {int(grid[i]): round(loss[i], 7) for i in np.searchsorted(grid, [1, 5, 10, 20, 50, 100, 200, 500])},
                      "loss_no_shrink": float(np.sum(ww * (np.where(nn > 0, mm, pp) - yy) ** 2) / ww.sum()),
                      "loss_prior_only": float(np.sum(ww * (pp - yy) ** 2) / ww.sum()), "pairs": int(sel.sum())}
    return out


def rating_variance(tempo: str | None) -> float:
    """Variance of one colour's FIDE rating over rated games: one row group of games.parquet.

    An upper bound on the variance within a move, which the edge tables cannot give (they
    hold sums, not sums of squares)."""
    import pyarrow.parquet as pq
    from tnh.paths import CORPUS
    f = pq.ParquetFile(CORPUS / "games.parquet")
    g = pl.from_arrow(f.read_row_group(f.metadata.num_row_groups - 1, columns=["white_elo", "black_elo", "tempo"]))
    if tempo:
        g = g.filter(pl.col("tempo") == tempo)
    g = g.drop_nulls()
    return float((g["white_elo"].var() + g["black_elo"].var()) / 2)


def dersimonian_laird(df: pl.DataFrame, y: str, var: str) -> float:
    """tau^2 pooled over positions, with per-move variances `var`/n known."""
    d = df.filter(pl.col(var) > 0).with_columns(w=pl.col("n") / pl.col(var))
    g = d.group_by("pos").agg(
        (pl.col("w") * pl.col(y)).sum().alias("wy"), pl.col("w").sum().alias("sw"),
        (pl.col("w") ** 2).sum().alias("sw2"), pl.len().alias("k"))
    d = d.join(g, on="pos").with_columns(mean=pl.col("wy") / pl.col("sw"))
    Q = float((d["w"] * (d[y] - d["mean"]) ** 2).sum())
    dfree = float((g["k"] - 1).sum())
    C = float((g["sw"] - g["sw2"] / g["sw"]).sum())
    return max(0.0, (Q - dfree) / C)


def estimate(df_all: pl.DataFrame, elo_var: float, min_n: int = 2) -> dict:
    df_all = df_all.with_columns(
        mover_elo=pl.when(pl.col("n_rated") > 0).then(pl.col("sum_mover") / pl.col("n_rated")),
        opp_elo=pl.when(pl.col("n_rated") > 0).then(pl.col("sum_opp") / pl.col("n_rated")))
    df_all = df_all.with_columns(
        score=(pl.col("white_wins") + 0.5 * pl.col("draws")) / pl.col("n"),
        draw=pl.col("draws") / pl.col("n"),
        white_elo=pl.when(pl.col("white_to_move")).then(pl.col("mover_elo")).otherwise(pl.col("opp_elo")),
        black_elo=pl.when(pl.col("white_to_move")).then(pl.col("opp_elo")).otherwise(pl.col("mover_elo")),
    ).with_columns(
        score_var=(pl.col("white_wins") + 0.25 * pl.col("draws")) / pl.col("n") - pl.col("score") ** 2,
        draw_var=pl.col("draw") * (1 - pl.col("draw")),
    )
    df = df_all.filter(pl.col("part") == "all")
    grid = np.unique(np.round(np.logspace(0, 3.3, 60)))
    out = {"pairs": df.height, "positions": df["pos"].n_unique()}
    for fig, count, weight_col in (("score", "n", "n"), ("draw", "n", "n"),
                                   ("white_elo", "n_rated", "n_rated"), ("black_elo", "n_rated", "n_rated")):
        d = df.filter(pl.col(count) >= min_n).drop_nulls(fig)
        pos = d.group_by("pos").agg(((pl.col(fig) * pl.col(count)).sum() / pl.col(count).sum()).alias("pm"),
                                     pl.col(count).sum().alias("N"), pl.len().alias("k"))
        d = d.join(pos.filter(pl.col("k") >= 2), on="pos")
        dev2 = ((d[fig] - d["pm"]) ** 2).to_numpy()
        x = (1 / d[count] - 1 / d["N"]).to_numpy()
        tau2, sigma2 = regression(dev2, x)
        r = {"tau2": tau2, "sigma2": sigma2, "kappa": sigma2 / tau2 if tau2 > 0 else float("inf"), "pairs": len(x),
             "holdout": holdout(df_all, fig, count, grid)}
        if fig in ("score", "draw"):
            s2 = float((d[f"{fig}_var"] * d["n"]).sum() / d["n"].sum())
            t2 = dersimonian_laird(d, fig, f"{fig}_var")
        else:
            s2 = elo_var
            t2 = dersimonian_laird(d.with_columns(n=pl.col("n_rated"), v=pl.lit(s2)), fig, "v")
        r |= {"sigma2_within": s2, "tau2_dl": t2, "kappa_dl": s2 / t2 if t2 > 0 else float("inf")}
        out[fig] = r
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tnh.web.calibrate", description=__doc__.split("\n\n")[0])
    ap.add_argument("--table", default="full")
    ap.add_argument("--positions", type=int, default=3000)
    ap.add_argument("--min-games", type=int, default=100)
    ap.add_argument("--tempi", default="classical", help="comma list, or 'all'")
    ap.add_argument("--split-year", type=int, default=2024, help="holdout: train before, test from this year")
    a = ap.parse_args(argv)
    tempi = None if a.tempi == "all" else a.tempi.split(",")
    t = time.perf_counter()
    db = EdgeDB(name=a.table)
    df = walk(db, a.positions, a.min_games, tempi, a.split_year)
    walked = time.perf_counter() - t
    elo_var = rating_variance(tempi[0] if tempi and len(tempi) == 1 else None)
    res = estimate(df, elo_var) | {"table": a.table, "tempi": tempi, "min_games": a.min_games, "walk_s": round(walked, 1)}
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
