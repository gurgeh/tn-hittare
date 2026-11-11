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
20-99, 100-999, 1000-9999, 10000-99999, 100000+) x the mover's rating group (``--rating-groups``, default
1800-2049, 2050-2299, 2300-2500). Each ply x ``pos_count`` cell gets up to
``--per-cell`` rows, shared equally among the rating groups (water-filling);
many cells are thin or empty, since a position at ply 30 is never common and one
at ply 4 never rare. Rows within a stratum: the smallest keys of
``hash(seed, game_id, ply)``, as in ``tnh.eval.sample``. ``weight`` is population
rows per sampled row in the stratum. Written to ``corpus/samples/<name>.parquet``
(positions schema plus ``tournament_id``, ``tempo_source``, ``ply_band``,
``pos_count_band``, ``rating_group``, ``bucket``, ``weight`` and the continuation
counts below) and ``<name>.json``.

**Continuation counts**, no engine needed. For each sampled row, every legal move's
resulting position is looked up in ``positions.parquet`` by ``pos_key``, which gives
its ``pos_count`` (0 when no row of the corpus has that position):
``played_child_count`` for the move played, ``max_child_count`` and
``n_max_children`` (how many moves share it), and ``children_count_sum``;
``played_ends_game`` where the next ply of the game has no row, so the played move's
count cannot be looked up (those rows are left out of the two shares);
``max_other_child_count`` and ``n_other_at_max`` for the moves not played. A child
count includes games that reached that position by transposition, which is why the
share below is taken over the children's sum and not over the parent's count.

**The analysis.** Per cell, pooled over rating groups with the weights, and per
rating group: the offset with the lowest raw log loss (the vertex of a parabola
through the best grid point and its neighbours, as in the sweep), with a
bootstrap interval over tournaments, and the log loss at the best grid offset.
Raw log loss: no ``eps`` is fitted here (it came out 0 in the rehearsal). Beside
it, over the *other* games through the position (the row's own game taken out):
the share of played moves that are their most common continuation (``1/k`` when
``k`` moves tie for it) and the mean share of them that chose the played move. A deliberate sideline shows as a low share in a very common position.
Cells under ``--min-rows`` rows report only their count.

**Where theory ends**, per ply band, pooled and per rating group, read from the low
end only: the smallest ``pos_count`` band whose best offset differs from that of the
rarest filled band of the same ply band, where the 95 % interval of the paired
bootstrap difference excludes 0. No shape is assumed: the curve can rise and fall
again in the most common positions. Writes ``<results>/theory.json``.

Reading it:
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

from tnh.eval.sweep import vertex
from tnh.paths import CORPUS

PC_EDGES = [0, 1, 2, 5, 20, 100, 1000, 10_000, 100_000]
PC_LABELS = ["0", "1", "2-4", "5-19", "20-99", "100-999", "1000-9999", "10000-99999", "100000+"]
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


def annotate_theory(a):
    pb, rg = bands(a.ply_bands), bands(a.rating_groups)

    def f(p: pl.DataFrame) -> pl.DataFrame:
        return p.with_columns(
            band_expr("ply", pb).alias("ply_band"), pc_band_expr().alias("pos_count_band"),
            band_expr("mover_elo", rg).alias("rating_group"), bucket_of("mover_elo", a.elo_min, a.elo_max).alias("bucket"),
        ).filter(pl.col("ply_band").is_not_null() & pl.col("rating_group").is_not_null()).select(
            "game_id", "ply", "pos_count", "ply_band", "pos_count_band", "rating_group", "bucket")
    return f


def draw_theory(positions: Path, games: pl.DataFrame, a, batch_games: int, log=print, games_path: Path | None = None):
    """Batched like tnh.eval.sample: narrow rows per batch of games, candidates per stratum, wide rows last."""
    from tnh.eval.sample import candidates, choose, eligible_games, fetch_wide

    eg = eligible_games(games, a, a.splits)
    ids = eg.get_column("game_id").to_list()
    if not ids:
        raise SystemExit("no eligible games")
    strata = ["ply_band", "pos_count_band", "rating_group"]
    counts, cand = candidates(positions, ids, batch_games, annotate_theory(a), strata, lambda k: a.per_cell,
                              a.seed, log)
    chosen, info = choose(cand, counts, strata, lambda h: a.per_cell, 2)
    wide = fetch_wide(games_path or positions.parent / "games.parquet", chosen.select("game_id", "ply"))
    n_plies = games.select("game_id", "n_plies") if "n_plies" in games.columns else None
    s = wide.join(chosen.drop("_key"), on=["game_id", "ply"], how="inner").join(eg, on="game_id", how="left")
    if n_plies is not None:
        s = s.join(n_plies, on="game_id", how="left")
    return s.sort("game_id", "ply"), info, {"games": len(ids)}


def _count_task(args):
    """Worker: the pos_count of the wanted keys found in row groups [lo, hi) of positions.parquet."""
    positions, keys_path, lo, hi = args
    import pyarrow.parquet as pq
    keys = pl.read_parquet(keys_path)
    t = pq.ParquetFile(positions).read_row_groups(list(range(lo, hi)), columns=["pos_key", "pos_count"])
    found = pl.from_arrow(t).join(keys, on="pos_key", how="semi")
    from tnh.eval.sample import drop_cache
    drop_cache(Path(positions))
    return found.group_by("pos_key").agg(pl.col("pos_count").max().cast(pl.Int64).alias("pos_count"),
                                         pl.col("pos_count").min().cast(pl.Int64).alias("_min"))


def lookup_counts(positions: Path, keys: pl.DataFrame, groups_per_task: int = 12, procs: int = 2) -> pl.DataFrame:
    """pos_count per wanted pos_key, read from positions.parquet a few row groups at a time.

    Each slice of row groups is read in a fresh worker process that exits after it: memory used to read
    positions.parquet was found to stay with the process (2026-11-11), so it must not be the main one."""
    import multiprocessing
    import shutil
    import tempfile

    import pyarrow.parquet as pq

    from tnh.paths import DATA_ROOT

    n = pq.ParquetFile(positions).metadata.num_row_groups
    tmp = Path(tempfile.mkdtemp(prefix="lookup-", dir=DATA_ROOT / "scratch"))
    try:
        kp = tmp / "keys.parquet"
        keys.write_parquet(kp)
        tasks = [(str(positions), str(kp), lo, min(lo + groups_per_task, n)) for lo in range(0, n, groups_per_task)]
        with multiprocessing.get_context("spawn").Pool(procs, maxtasksperchild=1) as pool:
            parts = pool.map(_count_task, tasks, chunksize=1)
    finally:
        shutil.rmtree(tmp)
    return pl.concat(parts).group_by("pos_key").agg(pl.col("pos_count").max(), pl.col("_min").min())


def with_continuations(s: pl.DataFrame, positions: Path) -> pl.DataFrame:
    """Add the continuation counts: each legal move's resulting position looked up by pos_key."""
    import chess

    from tnh.data.positions import pos_key

    rows, keys, played = [], [], []
    for i, (fen, uci) in enumerate(s.select("fen", "played_uci").iter_rows()):
        b = chess.Board(fen)
        for m in b.legal_moves:
            b.push(m)
            rows.append(i)
            keys.append(pos_key(b))
            b.pop()
            played.append(m.uci() == uci)
    kids = pl.DataFrame({"_r": rows, "pos_key": keys, "_played": played})
    counts = lookup_counts(positions, kids.select("pos_key").unique())
    counts = counts.with_columns((pl.col("pos_count") != pl.col("_min")).alias("_n")).drop("_min")
    if counts.get_column("_n").any():
        raise SystemExit("a pos_key with two different pos_count values in positions.parquet")
    kids = kids.join(counts.drop("_n"), on="pos_key", how="left").with_columns(pl.col("pos_count").fill_null(0))
    agg = kids.group_by("_r").agg(
        pl.col("pos_count").filter(pl.col("_played")).first().alias("played_child_count"),
        pl.col("pos_count").max().alias("max_child_count"),
        (pl.col("pos_count") == pl.col("pos_count").max()).sum().alias("n_max_children"),
        pl.col("pos_count").sum().alias("children_count_sum"),
        pl.col("pos_count").filter(~pl.col("_played")).max().fill_null(0).alias("max_other_child_count"),
        (pl.col("pos_count").filter(~pl.col("_played"))
         == pl.col("pos_count").filter(~pl.col("_played")).max()).sum().alias("n_other_at_max"),
        pl.col("_played").sum().alias("_np"),
    ).sort("_r")
    if agg.height != s.height or (agg.get_column("_np") != 1).any():
        raise SystemExit("played move not found exactly once among the legal moves")
    found = kids.filter(pl.col("pos_count") > 0).height
    # A move that ends the game leads to a position with no row of its own in positions.parquet, so its
    # count is not there to be found; such rows are flagged and left out of the continuation columns.
    out = s.with_columns(agg.select("played_child_count", "max_child_count", "n_max_children", "children_count_sum",
                                    "max_other_child_count", "n_other_at_max").get_columns())
    out = out.with_columns((pl.col("ply") + 1 >= pl.col("n_plies")).alias("played_ends_game"))
    print(f"continuations: {kids.height} child positions, {found} found in {positions}; "
          f"{out.get_column('played_ends_game').sum()} rows whose move ends the game (or the corpus's max_ply)")
    return out


def cmd_sample(a) -> None:
    games_path = a.games or a.positions.parent / "games.parquet"
    out_dir = a.out_dir or a.positions.parent / "samples"
    games = pl.read_parquet(games_path, columns=GAME_COLS + ["n_plies"])
    s, strata, pop = draw_theory(a.positions, games, a, a.batch_games, games_path=games_path)
    s = with_continuations(s, a.positions)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{a.name}.parquet"
    s.write_parquet(out)
    meta = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items() if k != "fn"}
    meta.update({"games_file": str(games_path), "eligible_games": pop["games"], "rows": s.height,
                 "games": s.get_column("game_id").n_unique(), "tournaments": s.get_column("tournament_id").n_unique(),
                 "strata": strata})
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    cell = s.group_by("ply_band", "pos_count_band").len()
    piv = cell.pivot(on="pos_count_band", index="ply_band", values="len")
    piv = piv.select(["ply_band"] + [c for c in PC_LABELS if c in piv.columns])
    print(f"{pop['games']} eligible games; sample {s.height} rows, {meta['games']} games, "
          f"{meta['tournaments']} tournaments; rows per cell:")
    with pl.Config(tbl_cols=20, tbl_width_chars=200):
        print(piv.sort(pl.col("ply_band").str.split("-").list.first().cast(pl.Int64)))
    print(f"wrote {out}")


def cells_of(L: np.ndarray, extra: np.ndarray, w, cl, counts, codes, n_cells, offs, min_rows):
    """Per cell: best offset (with its bootstrap replicates, for paired differences), loss at the best grid
    offset, and the means of the ``extra`` columns (most-common share, chosen share) over the rows where
    they are defined (not NaN)."""
    V = L.shape[1]
    ok = ~np.isnan(extra).any(axis=1)
    bm = boot_means(L, w, cl, counts, groups=codes, n_groups=n_cells)      # B x cells x V
    be = boot_means(extra[ok], w[ok], cl[ok], counts, groups=codes[ok], n_groups=n_cells)
    out, boots = [], []
    for c in range(n_cells):
        m = codes == c
        n = int(m.sum())
        boots.append(None)
        if n == 0:
            out.append(None)
            continue
        if n < min_rows:
            out.append({"rows": n})
            continue
        means = (w[m, None] * L[m]).sum(0) / w[m].sum()
        me = m & ok
        emeans = (w[me, None] * extra[me]).sum(0) / w[me].sum() if me.any() else np.full(extra.shape[1], np.nan)
        best, edge = vertex(offs, means[:V])
        bb = np.array([vertex(offs, bm[r, c, :V])[0] for r in range(bm.shape[0])])
        boots[-1] = bb
        bl = np.nanmin(bm[:, c, :V], axis=1)
        k = int(np.argmin(means[:V]))
        cell = {"rows": n,
                "best": {"est": best, "lo": float(np.nanquantile(bb, 0.025)),
                         "hi": float(np.nanquantile(bb, 0.975)), "at_edge": bool(edge)},
                "loss_at_best": {"est": float(means[k]), "lo": float(np.nanquantile(bl, 0.025)),
                                 "hi": float(np.nanquantile(bl, 0.975)), "grid_offset": float(offs[k])},
                "loss_by_offset": means[:V].tolist(), "rows_with_continuations": int(me.sum())}
        for j, name in enumerate(EXTRA):
            v = be[:, c, j]
            cell[name] = {"est": float(emeans[j]), "lo": float(np.nanquantile(v, 0.025)),
                          "hi": float(np.nanquantile(v, 0.975))}
        out.append(cell)
    return out, boots


EXTRA = ["most_common_share", "chosen_share"]


def theory_end(cells: list, boots: list, n_pc: int, plys: list[str], pcs: list[str]) -> list[dict]:
    """Per ply band: the smallest pos_count band whose best offset differs from the rarest filled band's
    (paired bootstrap difference, 95 % interval excluding 0)."""
    out = []
    for i, pb in enumerate(plys):
        filled = [j for j in range(n_pc) if boots[i * n_pc + j] is not None]
        if len(filled) < 2:
            out.append({"ply": pb, "rarest": pcs[filled[0]] if filled else None, "ends_at": None,
                        "note": "fewer than two filled pos_count bands"})
            continue
        r0 = filled[0]
        base = boots[i * n_pc + r0]
        rec = {"ply": pb, "rarest": pcs[r0], "ends_at": None, "differences": []}
        for j in filled[1:]:
            d = boots[i * n_pc + j] - base
            pt = cells[i * n_pc + j]["best"]["est"] - cells[i * n_pc + r0]["best"]["est"]
            lo, hi = float(np.nanquantile(d, 0.025)), float(np.nanquantile(d, 0.975))
            rec["differences"].append({"pos_count": pcs[j], "est": pt, "lo": lo, "hi": hi})
            if rec["ends_at"] is None and (lo > 0 or hi < 0):
                rec["ends_at"] = {"pos_count": pcs[j], "est": pt, "lo": lo, "hi": hi}
        if rec["ends_at"] is None:
            rec["note"] = "no filled band differs from the rarest"
        out.append(rec)
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

    # The row's own game is one of the games through the played move's child: take it out, so that the
    # shares are over the *other* games that reached the position (none for pos_count 1).
    ends = s.get_column("played_ends_game").to_numpy()
    own = s.get_column("played_child_count").to_numpy().astype(float) - 1
    if ((own < 0) & ~ends).any():
        raise SystemExit(f"{int(((own < 0) & ~ends).sum())} rows whose next ply exists but whose child has no count")
    others = s.get_column("children_count_sum").to_numpy().astype(float) - 1
    mo = s.get_column("max_other_child_count").to_numpy().astype(float)
    ko = s.get_column("n_other_at_max").to_numpy().astype(float)
    defined = ~ends & (others > 0)
    mc = np.where(own > mo, 1.0, np.where(own == mo, 1.0 / (ko + 1), 0.0))
    chosen = own / np.where(others > 0, others, 1)
    extra = np.column_stack([np.where(defined, mc, np.nan), np.where(defined, chosen, np.nan)])
    print(f"continuation shares: {int(ends.sum())} rows whose move ends the game and "
          f"{int((~ends & ~defined).sum())} with no other game through the position are left out")
    n_cells = len(plys) * len(pcs)

    def labelled(cells):
        out = []
        for c, x in enumerate(cells):
            if x is not None:
                out.append({"ply": plys[c // len(pcs)], "pos_count": pcs[c % len(pcs)], **x})
        return out

    pc_, pb_ = cells_of(L, extra, w, cl, counts, codes, n_cells, offs, a.min_rows)
    pooled, theory_ends = labelled(pc_), {"all": theory_end(pc_, pb_, len(pcs), plys, pcs)}
    by_rating = {}
    rg = s.get_column("rating_group").to_numpy()
    for g in sorted(set(rg)):
        m = rg == g
        c_, b_ = cells_of(L[m], extra[m], w[m], cl[m], counts, codes[m], n_cells, offs, a.min_rows)
        by_rating[g] = labelled(c_)
        theory_ends[g] = theory_end(c_, b_, len(pcs), plys, pcs)
    for c in pooled:
        if "best" in c:
            print(f"plies {c['ply']:>6} pos_count {c['pos_count']:>8}: rows={c['rows']:>6} best "
                  f"{c['best']['est']:+.0f} [{c['best']['lo']:+.0f}, {c['best']['hi']:+.0f}] "
                  f"loss {c['loss_at_best']['est']:.3f}")
    write_result(a.results, "theory", {
        "what": "best Maia-3 offset per ply band x pos_count band, raw log loss, own sample, no window rule",
        "rows_move_ends_game": int(ends.sum()),
        "sample": str(a.sample), "sample_meta": meta, "rows": s.height,
        "tournaments": s.get_column("tournament_id").n_unique(), "offsets": offs.tolist(),
        "variants": [label(d) for _, d in vs], "min_rows": a.min_rows, "boot": a.boot,
        "ply_bands": plys, "pos_count_bands": pcs, "cells": pooled, "by_rating": by_rating, "theory_end": theory_ends,
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
    sp.add_argument("--batch-games", type=int, default=20_000, help="games read at a time (memory)")
    sp.add_argument("--tournament-frac", type=float, default=1.0, help="draw this share of tournaments first")
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
