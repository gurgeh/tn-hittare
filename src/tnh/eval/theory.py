"""Where theory ends: which Maia-3 offset fits at each stage of the game and each level of commonness.

    uv run python -m tnh.eval.theory sample [--name theory] [--per-cell 3000] [--splits train val test]
    uv run python -m tnh.models.run --model maia3 --size 79m --offsets -200 ... 800 \\
        --positions corpus/samples/theory.parquet --games corpus/games.parquet
    uv run python -m tnh.eval.theory analyse --variants DIR [DIR ...] [--sample S] [--results R]

A test of its own, Maia-3 only: no Stockfish, no window rule, its own sample, so
it can be run and rerun without touching the main samples.

**The sample.** Rows of classical games with clocks, both players rated in
``[elo_min, elo_max]``, ``tempo_source`` not ``assumed``, from ``--splits`` (all
three by default: nothing is fitted here, the best offset is only read off), at
plies below the last ply band. Strata: ply band (``--ply-bands``, default 0-7,
8-11, 12-15, 16-19, 20-23, 24-29, 30-39) x ``pos_count`` band (0, 1, 2-4, 5-19,
20-99, 100-999, 1000+) x the mover's rating group (``--rating-groups``, default
1800-2049, 2050-2299, 2300-2500). Each ply x ``pos_count`` cell gets up to
``--per-cell`` rows, shared equally among the rating groups (water-filling);
many cells are thin or empty, since a position at ply 30 is never common and one
at ply 4 never rare. Rows within a stratum: the smallest keys of
``hash(seed, game_id, ply)``, as in ``tnh.eval.sample``. ``weight`` is population
rows per sampled row in the stratum. Written to ``corpus/samples/<name>.parquet``
(positions schema plus ``tournament_id``, ``tempo_source``, ``ply_band``,
``pos_count_band``, ``rating_group``, ``bucket``, ``weight``) and ``<name>.json``.

**The analysis.** Per cell, pooled over rating groups with the weights, and per
rating group: the offset with the lowest raw log loss (the vertex of a parabola
through the best grid point and its neighbours, as in the sweep), with a
bootstrap interval over tournaments, and the log loss at the best grid offset.
Raw log loss: no ``eps`` is fitted here (it came out 0 in the rehearsal). Cells
under ``--min-rows`` rows report only their count. Writes ``<results>/theory.json``.

Reading it: within one ply band, if the best offset rises with ``pos_count``,
people play above their strength in the positions that are known; theory ends at
the count where it has come down to its value in the rare positions of that band.
``pos_count`` counts games of players rated 1800 and up in the whole corpus,
grandmasters included, and the game itself: what is known to someone, not at the
mover's level. Maia-3 memorised openings from its own training games too.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from tnh.eval.common import (BUCKET, RESULTS, boot_counts, boot_means, bucket_of, cluster_codes, label, move_scores,
                             parse_variant, provenance, read_sample, variant_name, write_result)
from tnh.eval.sample import allocate, row_keys
from tnh.eval.sweep import vertex
from tnh.paths import CORPUS

PC_EDGES = [0, 1, 2, 5, 20, 100, 1000]
PC_LABELS = ["0", "1", "2-4", "5-19", "20-99", "100-999", "1000+"]
GAME_COLS = ["game_id", "tournament_id", "tempo", "tempo_source", "has_clock", "split", "white_elo", "black_elo"]


def bands(xs: list[str]) -> list[tuple[int, int]]:
    return [tuple(int(v) for v in x.split("-")) for x in xs]


def band_expr(col: str, bs: list[tuple[int, int]]) -> pl.Expr:
    e = pl.lit(None, dtype=pl.String)
    for lo, hi in reversed(bs):
        e = pl.when(pl.col(col).is_between(lo, hi)).then(pl.lit(f"{lo}-{hi}")).otherwise(e)
    return e


def pc_band_expr() -> pl.Expr:
    e = pl.lit(PC_LABELS[-1])
    for edge, lab in reversed(list(zip(PC_EDGES[1:], PC_LABELS[:-1]))):
        e = pl.when(pl.col("pos_count") < edge).then(pl.lit(lab)).otherwise(e)
    return e


def population(positions: pl.LazyFrame, games: pl.DataFrame, a) -> pl.DataFrame:
    g = games.select(GAME_COLS).filter(
        (pl.col("tempo") == "classical") & pl.col("has_clock") & (pl.col("tempo_source") != "assumed")
        & pl.col("white_elo").is_between(a.elo_min, a.elo_max) & pl.col("black_elo").is_between(a.elo_min, a.elo_max)
        & pl.col("split").is_in(a.splits)
    ).select("game_id", "tournament_id", "tempo_source")
    pb, rg = bands(a.ply_bands), bands(a.rating_groups)
    max_ply = max(hi for _, hi in pb)
    p = (positions.filter(pl.col("ply") <= max_ply).join(g.lazy(), on="game_id", how="inner").collect()
         .with_columns(pl.col("ply").cast(pl.Int64)))
    return p.with_columns(
        band_expr("ply", pb).alias("ply_band"), pc_band_expr().alias("pos_count_band"),
        band_expr("mover_elo", rg).alias("rating_group"), bucket_of("mover_elo", a.elo_min, a.elo_max).alias("bucket"),
    ).filter(pl.col("ply_band").is_not_null() & pl.col("rating_group").is_not_null())


def draw(pop: pl.DataFrame, a) -> tuple[pl.DataFrame, list[dict]]:
    pop = pop.with_columns(pl.Series("_key", row_keys(pop.get_column("game_id"), pop.get_column("ply"), a.seed)))
    avail = {(x, y, z): n for x, y, z, n in pop.group_by("ply_band", "pos_count_band", "rating_group").len().iter_rows()}
    out, strata = [], []
    cells = sorted({(k[0], k[1]) for k in avail})
    for pb, pc in cells:
        groups = {k[2]: n for k, n in avail.items() if k[0] == pb and k[1] == pc}
        alloc = allocate(groups, a.per_cell)
        for rg, k in sorted(alloc.items()):
            if k == 0:
                continue
            rows = (pop.filter((pl.col("ply_band") == pb) & (pl.col("pos_count_band") == pc)
                               & (pl.col("rating_group") == rg)).sort("_key").head(k))
            out.append(rows.with_columns(pl.lit(groups[rg] / k).alias("weight")))
            strata.append({"ply_band": pb, "pos_count_band": pc, "rating_group": rg, "population": int(groups[rg]),
                           "sampled": int(k)})
    return pl.concat(out).drop("_key").sort("game_id", "ply"), strata


def cmd_sample(a) -> None:
    games_path = a.games or a.positions.parent / "games.parquet"
    out_dir = a.out_dir or a.positions.parent / "samples"
    games = pl.read_parquet(games_path, columns=GAME_COLS)
    orphans = (pl.scan_parquet(a.positions).select("game_id").unique()
               .join(games.lazy().select("game_id"), on="game_id", how="anti").select(pl.len()).collect().item())
    if orphans:
        raise SystemExit(f"{orphans} game_ids in {a.positions} are not in {games_path}: not the same build")
    pop = population(pl.scan_parquet(a.positions), games, a)
    if pop.height == 0:
        raise SystemExit("no eligible rows")
    s, strata = draw(pop, a)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{a.name}.parquet"
    s.write_parquet(out)
    meta = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items() if k != "fn"}
    meta.update({"games_file": str(games_path), "rows": s.height, "games": s.get_column("game_id").n_unique(),
                 "tournaments": s.get_column("tournament_id").n_unique(), "strata": strata})
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    cell = s.group_by("ply_band", "pos_count_band").len()
    piv = cell.pivot(on="pos_count_band", index="ply_band", values="len")
    piv = piv.select(["ply_band"] + [c for c in PC_LABELS if c in piv.columns])
    print(f"population {pop.height} rows; sample {s.height} rows, {meta['games']} games, "
          f"{meta['tournaments']} tournaments; rows per cell:")
    print(piv.sort(pl.col("ply_band").str.split("-").list.first().cast(pl.Int64)))
    print(f"wrote {out}")


def cells_of(L: np.ndarray, w, cl, counts, codes, n_cells, offs, min_rows) -> list[dict]:
    bm = boot_means(L, w, cl, counts, groups=codes, n_groups=n_cells)      # B x cells x V
    out = []
    for c in range(n_cells):
        m = codes == c
        n = int(m.sum())
        if n == 0:
            out.append(None)
            continue
        if n < min_rows:
            out.append({"rows": n})
            continue
        means = (w[m, None] * L[m]).sum(0) / w[m].sum()
        best, edge = vertex(offs, means)
        bb = np.array([vertex(offs, bm[r, c])[0] for r in range(bm.shape[0])])
        bl = np.nanmin(bm[:, c, :], axis=1)
        k = int(np.argmin(means))
        out.append({"rows": n,
                    "best": {"est": best, "lo": float(np.nanquantile(bb, 0.025)),
                             "hi": float(np.nanquantile(bb, 0.975)), "at_edge": bool(edge)},
                    "loss_at_best": {"est": float(means[k]), "lo": float(np.nanquantile(bl, 0.025)),
                                     "hi": float(np.nanquantile(bl, 0.975)), "grid_offset": float(offs[k])},
                    "loss_by_offset": means.tolist()})
    return out


def cmd_analyse(a) -> None:
    s = read_sample(a.sample)
    vs = []
    for d in a.variants:
        v = parse_variant(variant_name(d))
        if v["kind"] != "offset" or v["mover"] != v["opp"]:
            print(f"ignoring {d}: not a both-sides offset")
            continue
        vs.append((v["mover"], d))
    vs.sort(key=lambda x: x[0])
    offs = np.array([o for o, _ in vs])
    if len(offs) < 3:
        raise SystemExit("need at least three offsets")
    L = np.column_stack([-move_scores(s, d)[0].lp for _, d in vs])
    w = s.get_column("weight").to_numpy()
    cl = cluster_codes(s)
    counts = boot_counts(cl.max() + 1, a.boot, a.seed)
    meta_p = Path(a.sample).with_suffix(".json")
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else None
    plys = sorted(s.get_column("ply_band").unique().to_list(), key=lambda x: int(x.split("-")[0]))
    pcs = [p for p in PC_LABELS if p in set(s.get_column("pos_count_band").unique().to_list())]
    ip = {p: i for i, p in enumerate(plys)}
    ic = {p: i for i, p in enumerate(pcs)}
    codes = np.array([ip[x] * len(pcs) + ic[y] for x, y in
                      zip(s.get_column("ply_band").to_list(), s.get_column("pos_count_band").to_list())])

    def labelled(cells):
        out = []
        for c, x in enumerate(cells):
            if x is not None:
                out.append({"ply": plys[c // len(pcs)], "pos_count": pcs[c % len(pcs)], **x})
        return out

    pooled = labelled(cells_of(L, w, cl, counts, codes, len(plys) * len(pcs), offs, a.min_rows))
    by_rating = {}
    rg = s.get_column("rating_group").to_numpy()
    for g in sorted(set(rg)):
        m = rg == g
        by_rating[g] = labelled(cells_of(L[m], w[m], cl[m], counts, codes[m], len(plys) * len(pcs), offs, a.min_rows))
    for c in pooled:
        if "best" in c:
            print(f"plies {c['ply']:>6} pos_count {c['pos_count']:>8}: rows={c['rows']:>6} best "
                  f"{c['best']['est']:+.0f} [{c['best']['lo']:+.0f}, {c['best']['hi']:+.0f}] "
                  f"loss {c['loss_at_best']['est']:.3f}")
    write_result(a.results, "theory", {
        "what": "best Maia-3 offset per ply band x pos_count band, raw log loss, own sample, no window rule",
        "sample": str(a.sample), "sample_meta": meta, "rows": s.height,
        "tournaments": s.get_column("tournament_id").n_unique(), "offsets": offs.tolist(),
        "variants": [label(d) for _, d in vs], "min_rows": a.min_rows, "boot": a.boot,
        "ply_bands": plys, "pos_count_bands": pcs, "cells": pooled, "by_rating": by_rating,
        "provenance": provenance({"theory": a.sample}, [d for _, d in vs])})


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m tnh.eval.theory", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(required=True)
    sp = sub.add_parser("sample")
    sp.add_argument("--name", default="theory")
    sp.add_argument("--per-cell", type=int, default=3000, help="rows per ply x pos_count cell")
    sp.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    sp.add_argument("--ply-bands", nargs="+", default=["0-7", "8-11", "12-15", "16-19", "20-23", "24-29", "30-39"])
    sp.add_argument("--rating-groups", nargs="+", default=["1800-2049", "2050-2299", "2300-2500"])
    sp.add_argument("--elo-min", type=int, default=1800)
    sp.add_argument("--elo-max", type=int, default=2500)
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--positions", type=Path, default=CORPUS / "positions.parquet")
    sp.add_argument("--games", type=Path)
    sp.add_argument("--out-dir", type=Path)
    sp.set_defaults(fn=cmd_sample)
    sp = sub.add_parser("analyse")
    sp.add_argument("--variants", type=Path, nargs="+", required=True)
    sp.add_argument("--sample", type=Path, default=CORPUS / "samples" / "theory.parquet")
    sp.add_argument("--min-rows", type=int, default=200)
    sp.add_argument("--results", type=Path, default=RESULTS)
    sp.add_argument("--boot", type=int, default=1000)
    sp.add_argument("--seed", type=int, default=7)
    sp.set_defaults(fn=cmd_analyse)
    a = p.parse_args(argv)
    if a.fn is cmd_sample and a.elo_min % BUCKET:
        raise SystemExit("--elo-min must be a multiple of 100")
    a.fn(a)


if __name__ == "__main__":
    main()
