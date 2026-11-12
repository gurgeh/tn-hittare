"""The position set for step 2, from the edge tables.

A position is in the set when at least ``min_games`` games in the chosen rating
bands and tempo reached it. Per position:

- ``n``, ``white_wins``, ``draws``, ``black_wins``, ``score_white``: those games
  and their result. They are counted on the edges *into* the position (the move
  that reached it), so games that end in the position count too. A game that
  reaches the position twice by different moves counts twice; in openings that
  is rare. The start position has no edge into it and is never in the set.
- ``white_elo``, ``black_elo``: mean ratings over the games with both ratings.
  On an edge into the position the mover is the side *not* to move here.
- ``total_n``: the position's games in ``pos_index`` (every band, tempo and year
  in the table): the popularity.
- ``moves_uci``, ``ply``: a move sequence that reaches it, walking the most played
  parent upward (``parents`` table, all bands and tempi). Checked by replaying it.
- ``top_child_share``, ``n_children_in_set``, ``leaf``: how the line thins out.
  The games of a position and its children in the set overlap; ``leaf`` marks
  positions with no child in the set, whose games are disjoint from every other
  leaf's (up to transpositions and repetitions).

Clusters for the bootstrap: the opening family is the first ``family_ply`` plies
of ``moves_uci`` (`family`). Positions above that ply are in no family.

Everything is written so that ``name="full"`` works on the full tables: the
edges are scanned lazily with a filter, and parents are read level by level.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import chess
import numpy as np
import polars as pl

from tnh.data.edges import EdgeDB, pos_hash
from tnh.paths import CORPUS, DATA_ROOT, REPORTS

OUT = DATA_ROOT / "scratch" / "difficulty"
RESULTS = REPORTS / "step2"
START = pos_hash(chess.Board())
MAX_WALK = 120


def edges_scan(name: str = "dev", directory: Path | None = None) -> pl.LazyFrame:
    """The edges table of ``EdgeDB(name)`` as a lazy scan (same file naming)."""
    d = directory or CORPUS
    suffix = "" if name == "full" else f"_{name}"
    return pl.scan_parquet(d / f"edges{suffix}.parquet")


def band_filter(bands: tuple[int, int], tempo: str | None, years: tuple[int, int] | None) -> pl.Expr:
    f = pl.col("elo_band").is_between(*bands)
    if tempo is not None:
        f &= pl.col("tempo") == tempo
    if years is not None:
        f &= pl.col("year").is_between(*years)
    return f


_SUMS = ["n", "white_wins", "draws", "black_wins", "sum_mover_elo", "sum_opp_elo", "n_rated"]


def incoming(edges: pl.LazyFrame, filt: pl.Expr) -> pl.DataFrame:
    """Per position: the games under the filter that reached it, summed over the moves into it."""
    return (edges.filter(filt).group_by("child_hash").agg(pl.col(_SUMS).sum())
            .rename({"child_hash": "pos_hash"}).collect())


def outgoing(edges: pl.LazyFrame, filt: pl.Expr, hashes: pl.Series) -> pl.DataFrame:
    """The moves played from the given positions under the filter: pos_hash, move, child_hash, n, results."""
    return (edges.filter(filt & pl.col("pos_hash").is_in(hashes.implode()))
            .group_by("pos_hash", "move", "child_hash")
            .agg(pl.col("n", "white_wins", "draws", "black_wins").sum()).collect())


# --- paths ------------------------------------------------------------------------


def walk_paths(db: EdgeDB, targets: np.ndarray) -> tuple[dict[int, list[str]], dict[str, int]]:
    """A move sequence from the start to each target, along the most played parent.

    Parents are read in batches, one level at a time, and only along the walk.
    A parent already on the walk (a repetition) is skipped for the next best.
    Returns the paths and counts of what failed. Every path is replayed and must
    reach its target's hash.
    """
    parents: dict[int, list[tuple[int, str]]] = {}   # child -> [(parent, move)], most played first

    def load(keys: set[int]) -> None:
        keys = [k for k in keys if k not in parents and k != START]
        if not keys:
            return
        rows = db.parents_many(keys, with_epd=False)   # sorted by child, then games, most first
        for (c,), g in rows.group_by(["child_hash"], maintain_order=True):
            parents[int(c)] = list(zip(g["pos_hash"].to_list(), g["move"].to_list()))
        for k in keys:
            parents.setdefault(int(k), [])

    state = {int(t): ([int(t)], []) for t in targets}   # target -> (visited chain, moves reversed)
    done: dict[int, list[str]] = {}
    stats = {"no_parent": 0, "too_long": 0, "replay_mismatch": 0}
    active = set(state)
    while active:
        load({state[t][0][-1] for t in active})
        nxt = set()
        for t in active:
            chain, rev = state[t]
            while True:
                cur = chain[-1]
                if cur == START:
                    done[t] = rev[::-1]
                    break
                if cur not in parents:
                    nxt.add(t)
                    break
                seen = set(chain)
                choice = next(((p, m) for p, m in parents[cur] if p not in seen), None)
                if choice is None:
                    stats["no_parent"] += 1
                    break
                if len(chain) > MAX_WALK:
                    stats["too_long"] += 1
                    break
                chain.append(choice[0])
                rev.append(choice[1])
        active = nxt
    for t, moves in list(done.items()):
        b = chess.Board()
        for m in moves:
            b.push_uci(m)
        if pos_hash(b) != t:
            stats["replay_mismatch"] += 1
            del done[t]
    return done, stats


# --- the set ------------------------------------------------------------------------


def build(db: EdgeDB, edges: pl.LazyFrame, *, min_games: int, bands: tuple[int, int] = (1800, 2400),
          tempo: str | None = "classical", years: tuple[int, int] | None = None,
          counts_at: tuple[int, ...] = (20, 50, 100, 200, 500, 1000)) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """(positions, their in-filter moves, info). See the module docstring for the columns."""
    t0 = time.time()
    filt = band_filter(bands, tempo, years)
    inc = incoming(edges, filt)
    info = {"min_games": min_games, "bands": list(bands), "tempo": tempo, "years": years,
            "positions_reached_in_filter": inc.height,
            "positions_with_at_least": {str(k): int((inc["n"] >= k).sum()) for k in counts_at}}
    s = inc.filter(pl.col("n") >= min_games)
    hashes = s["pos_hash"]
    tot = db.totals_many(hashes.to_list()).select("pos_hash", "epd", pl.col("n").alias("total_n"))
    missing = s.height - tot.height
    if missing:
        raise RuntimeError(f"{missing} positions with edges into them are not in pos_index")
    s = s.join(tot, on="pos_hash")
    stm_white = s["epd"].str.split(" ").list.get(1) == "w"
    s = s.with_columns(stm_white=stm_white)
    rated = pl.col("n_rated").cast(pl.Float64)
    mover = pl.col("sum_mover_elo") / rated   # the side that moved into the position = not to move here
    opp = pl.col("sum_opp_elo") / rated
    s = s.with_columns(
        score_white=(pl.col("white_wins") + 0.5 * pl.col("draws")) / pl.col("n"),
        white_elo=pl.when(rated > 0).then(pl.when(pl.col("stm_white")).then(opp).otherwise(mover)),
        black_elo=pl.when(rated > 0).then(pl.when(pl.col("stm_white")).then(mover).otherwise(opp)),
    ).drop("sum_mover_elo", "sum_opp_elo")

    out = outgoing(edges, filt, hashes)
    in_set = out.join(s.select(pl.col("pos_hash").alias("child_hash")), on="child_hash", how="semi")
    thin = in_set.group_by("pos_hash").agg(top_child_n=pl.col("n").max(), n_children_in_set=pl.len())
    s = s.join(thin, on="pos_hash", how="left").with_columns(
        pl.col("n_children_in_set").fill_null(0),
        top_child_share=pl.col("top_child_n").fill_null(0) / pl.col("n"),
    ).with_columns(leaf=pl.col("n_children_in_set") == 0).drop("top_child_n")

    paths, walk_stats = walk_paths(db, hashes.to_numpy())
    info["walk"] = walk_stats
    pdf = pl.DataFrame({"pos_hash": pl.Series(list(paths.keys()), dtype=pl.UInt64),
                        "moves_uci": [" ".join(m) for m in paths.values()],
                        "ply": [len(m) for m in paths.values()]})
    before = s.height
    s = s.join(pdf, on="pos_hash", how="inner")
    info["dropped_no_path"] = before - s.height
    s = s.with_columns(fen=pl.col("epd") + " 0 1").sort("total_n", "pos_hash", descending=[True, False])
    info["positions"] = s.height
    info["games_in_filter_reaching_set"] = int(s["n"].sum())
    info["leaves"] = int(s["leaf"].sum())
    info["seconds"] = round(time.time() - t0, 1)
    return s, out, info


def family(moves_uci: pl.Expr, ply: int) -> pl.Expr:
    """The first ``ply`` moves of the path; null for a position above that ply."""
    parts = moves_uci.str.split(" ")
    return pl.when(parts.list.len() >= ply).then(parts.list.head(ply).list.join(" "))


def set_paths(tag: str) -> tuple[Path, Path, Path]:
    return OUT / f"positions_{tag}.parquet", OUT / f"moves_{tag}.parquet", OUT / f"positions_{tag}.json"


def tag_of(name: str, min_games: int, max_games: int | None = None) -> str:
    return f"{name}_n{min_games}" + (f"to{max_games - 1}" if max_games else "")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="dev", help="edge tables: 'full' or e.g. 'dev'")
    ap.add_argument("--min-games", type=int, default=100)
    ap.add_argument("--bands", type=int, nargs=2, default=(1800, 2400))
    ap.add_argument("--tempo", default="classical")
    ap.add_argument("--max-games", type=int, default=None,
                    help="keep only positions with fewer games than this (after the thinning flags are set "
                         "relative to the whole set >= min-games)")
    args = ap.parse_args(argv)
    db = EdgeDB(args.name)
    s, out, info = build(db, edges_scan(args.name), min_games=args.min_games, bands=tuple(args.bands), tempo=args.tempo)
    info["name"] = args.name
    if args.max_games is not None:
        s = s.filter(pl.col("n") < args.max_games)
        out = out.join(s.select("pos_hash"), on="pos_hash", how="semi")
        info["max_games"] = args.max_games
        info["positions"] = s.height
        info["games_in_filter_reaching_set"] = int(s["n"].sum())
        info["leaves"] = int(s["leaf"].sum())
    for k in (2, 4, 6, 8):
        fam = s.select(family(pl.col("moves_uci"), k).alias("f"), "n")
        g = fam.drop_nulls("f").group_by("f").agg(pl.len(), pl.col("n").sum()).sort("len", descending=True)
        info[f"families_ply{k}"] = {"families": g.height, "positions_above": int(fam["f"].null_count()),
                                    "largest_share": round(float(g["len"].max() / g["len"].sum()), 3) if g.height else None,
                                    "largest": g.head(5).to_dicts()}
    OUT.mkdir(parents=True, exist_ok=True)
    p_pos, p_mov, p_info = set_paths(tag_of(args.name, args.min_games, args.max_games))
    s.write_parquet(p_pos)
    out.write_parquet(p_mov)
    p_info.write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))
    print(f"wrote {p_pos} ({s.height} positions), {p_mov}, {p_info}")


if __name__ == "__main__":
    main()
