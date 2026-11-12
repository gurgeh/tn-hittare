"""python -m tnh.difficulty <positions|compute|analyse|synth|report> [options]"""

from __future__ import annotations

import argparse
import json
import sys
import time

import chess
import numpy as np
import polars as pl

from tnh.difficulty import analysis as A
from tnh.difficulty.compute import Budget
from tnh.difficulty.positions import OUT, RESULTS, family, set_paths


def write(name: str, obj: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / f"{name}.json"
    p.write_text(json.dumps(obj, indent=1, default=_default) + "\n")
    print(f"wrote {p}", flush=True)


def _default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def san_line(moves_uci: str) -> str:
    b = chess.Board()
    out = []
    for i, m in enumerate(moves_uci.split()):
        mv = chess.Move.from_uci(m)
        s = b.san(mv)
        out.append(f"{i // 2 + 1}.{s}" if i % 2 == 0 else s)
        b.push(mv)
    return " ".join(out)


def cross_family_share(df: pl.DataFrame, positions: pl.DataFrame, moves: pl.DataFrame, family_ply: int) -> dict:
    """Of the analysed positions' games, the share that arrived from a parent in the set in another family."""
    fam = df.select("pos_hash", "family")
    allfam = positions.select(
        "pos_hash", family(pl.col("moves_uci"), family_ply).alias("pfam"))
    e = (moves.join(fam.rename({"pos_hash": "child_hash", "family": "cfam"}), on="child_hash")
         .join(allfam, on="pos_hash"))
    cross = e.filter(pl.col("pfam").is_not_null() & (pl.col("pfam") != pl.col("cfam")))["n"].sum()
    return {"games_from_other_family": int(cross), "games": int(df["n"].sum()),
            "share": float(cross / df["n"].sum())}


def groups(df: pl.DataFrame) -> dict[str, np.ndarray]:
    med = float(df["total_n"].median())
    pop = np.where(df["total_n"].to_numpy() <= med, f"rare (total <= {med:.0f})", f"common (total > {med:.0f})")
    me = df["mean_elo"].to_numpy()
    rating = np.select([me < 2000, me < 2200], ["mean rating < 2000", "2000-2199"], "2200 and up")
    ply = df["ply"].to_numpy()
    q1, q2 = np.quantile(ply, [1 / 3, 2 / 3])
    plyg = np.select([ply <= q1, ply <= q2], [f"ply <= {q1:.0f}", f"ply {q1 + 1:.0f}-{q2:.0f}"], f"ply > {q2:.0f}")
    stm = np.where(df["stm_white"].to_numpy() > 0.5, "White to move", "Black to move")
    return {"popularity": pop.astype(object), "rating": rating.astype(object), "ply": plyg.astype(object),
            "side to move": stm.astype(object)}


def analyse(a) -> None:
    from tnh.difficulty import measures as M
    t0 = time.time()
    pos_p, mov_p, info_p = set_paths(a.tag)
    positions = pl.read_parquet(pos_p)
    moves = pl.read_parquet(mov_p)
    b = Budget(a.root_nodes, a.multipv_nodes, a.maia_size, a.child_mass, a.max_children, a.min_child_p)
    meas, counts = M.build(positions, b, eps=a.eps, db_moves=moves)
    print(f"measures for {meas.height} of {positions.height} positions; {counts}", flush=True)
    meas.write_parquet(OUT / f"measures_{a.tag}.parquet")
    df, finfo = A.white_frame(positions, meas, a.family_ply)
    setup = A.Setup(folds=a.folds, boot=a.boot, seed=a.seed, base=a.base)
    out_all = {"tag": a.tag, "budget": b.__dict__, "eps": a.eps, "family_ply": a.family_ply,
               "set_info": json.loads(info_p.read_text()), "measure_counts": counts, "frame": finfo,
               "measures_described": A.MEASURES, "reference_described": A.REFERENCE, "base": a.base}
    out_all["cross_family"] = cross_family_share(df, positions, moves, a.family_ply)
    # how Maia's difficulty compares with the games' own (side to move's loss and hold)
    w = df["n"].to_numpy().astype(float)
    raw = df.select("el_stm", "el_db", "hard_stm", "hard_db").to_numpy()
    out_all["maia_vs_games"] = {
        "mean_el_maia": float(np.average(np.abs(raw[:, 0]), weights=w)),
        "mean_el_games": float(np.average(np.abs(raw[:, 1]), weights=w)),
        "mean_hard_maia": float(np.average(np.abs(raw[:, 2]), weights=w)),
        "mean_hard_games": float(np.average(np.abs(raw[:, 3]), weights=w)),
        "corr_el": float(np.corrcoef(np.abs(raw[:, 0]), np.abs(raw[:, 1]))[0, 1]),
        "corr_hard": float(np.corrcoef(np.abs(raw[:, 2]), np.abs(raw[:, 3]))[0, 1]),
        "corr_signed_el": float(np.corrcoef(raw[:, 0], raw[:, 1])[0, 1]),
    }
    cols = [*A.MEASURES, *A.REFERENCE]
    out_all["measure_corr"] = {"names": cols, "matrix": np.corrcoef(df.select(cols).to_numpy().T).tolist()}
    measures = list(A.MEASURES)
    reference = list(A.REFERENCE)
    for variant in a.variants:
        d = df if variant == "all" else df.filter(pl.col("leaf")) if variant == "leaves" else None
        if d is None:
            raise SystemExit(f"unknown variant {variant}")
        res, frame = A.run(d, measures, setup, splits=groups(d), reference=reference)
        out_all[variant] = res
        frame.write_parquet(OUT / f"frame_{a.tag}{a.label}_{variant}.parquet")
        print(f"{variant}: {res['positions']} positions, {res['families']} families; "
              + "; ".join(f"{m} dmse {v['dmse']['est']:.2e} [{v['dmse']['lo']:.2e}, {v['dmse']['hi']:.2e}]"
                          for m, v in res["measures"].items()), flush=True)
    out_all["lists"] = lists(pl.read_parquet(OUT / f"frame_{a.tag}{a.label}_all.parquet"), out_all["all"], a)
    out_all["seconds"] = round(time.time() - t0, 1)
    write(f"analysis_{a.tag}{a.label}", out_all)


def lists(frame: pl.DataFrame, res: dict, a) -> dict:
    """The positions a chess player can judge: the largest predicted edges near equality, and the worst misses."""
    cands = {m: v for m, v in res["measures"].items() if not v["reference"]}
    best = max(cands, key=lambda m: cands[m]["dmse"]["est"])
    beta = res["measures"][best]["beta"]["est"]
    mbar = float(np.average(frame[best].to_numpy(), weights=frame["w"].to_numpy()))
    f = frame.with_columns(edge=beta * (pl.col(best) - mbar),
                           z=(pl.col("y") - pl.col(f"oof_{best}")) / (res["sigma2"] / pl.col("n") + res["tau2"]).sqrt())
    keep = ["epd", "moves_uci", "n", "y", "cp_w_raw", "engine_es_w", "pred_base", f"oof_{best}", "edge", "z",
            best, "total_n", "white_elo", "black_elo", "family"]
    near = f.filter(pl.col("cp_w_raw").abs() <= a.near_cp)
    top = near.sort(pl.col("edge").abs(), descending=True).head(20).select(keep)
    miss = f.sort(pl.col("z").abs(), descending=True).head(20).select(keep)
    conv = lambda df: [{**r, "san": san_line(r["moves_uci"])} for r in df.iter_rows(named=True)]
    return {"measure": best, "beta": beta, "near_cp": a.near_cp, "near_positions": near.height,
            "edges": conv(top), "misses": conv(miss)}


def synth_cmd(a) -> None:
    from tnh.difficulty import synth
    out = {}
    for beta in (1.0, 0.0):
        g, planted = synth.world(seed=a.seed, beta=beta, games=a.games)
        for leaves in (False, True):
            res, _, info = synth.pipeline(g, planted, min_games=a.min_games, family_ply=2, leaves_only=leaves,
                                          boot=a.boot, seed=a.seed)
            key = f"beta{beta:g}_{'leaves' if leaves else 'all'}"
            out[key] = {"planted_beta": beta, "positions": res["positions"], "families": res["families"],
                        "games": a.games, "planted": res["measures"]["planted"],
                        "noise_measure": res["measures"]["noise_measure"], "tau2": res["tau2"]}
            pm = res["measures"]["planted"]["beta"]
            print(f"{key}: beta {pm['est']:.3f} [{pm['lo']:.3f}, {pm['hi']:.3f}], positions {res['positions']}", flush=True)
    write("synth", out)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit(__doc__)
    cmd, rest = argv[0], argv[1:]
    if cmd == "positions":
        from tnh.difficulty.positions import main as m
        return m(rest)
    if cmd == "compute":
        from tnh.difficulty.compute import main as m
        return m(rest)
    if cmd == "report":
        from tnh.difficulty.report import main as m
        return m(rest)
    ap = argparse.ArgumentParser(prog=f"tnh.difficulty {cmd}")
    if cmd == "analyse":
        d = Budget()
        ap.add_argument("--tag", default="dev_n100")
        ap.add_argument("--root-nodes", type=int, default=d.root_nodes)
        ap.add_argument("--multipv-nodes", type=int, default=d.multipv_nodes)
        ap.add_argument("--maia-size", default=d.maia_size)
        ap.add_argument("--child-mass", type=float, default=d.child_mass)
        ap.add_argument("--max-children", type=int, default=d.max_children)
        ap.add_argument("--min-child-p", type=float, default=d.min_child_p)
        ap.add_argument("--eps", type=float, default=0.03, help="a move holds when it loses at most this")
        ap.add_argument("--family-ply", type=int, default=4)
        ap.add_argument("--folds", type=int, default=10)
        ap.add_argument("--boot", type=int, default=1000)
        ap.add_argument("--seed", type=int, default=0)
        ap.add_argument("--near-cp", type=float, default=30)
        ap.add_argument("--variants", nargs="+", default=["all", "leaves"])
        ap.add_argument("--base", nargs="+", default=list(A.BASE), help="baseline columns")
        ap.add_argument("--label", default="", help="suffix of the result file, e.g. _elo2")
        analyse(ap.parse_args(rest))
    elif cmd == "synth":
        ap.add_argument("--games", type=int, default=30_000)
        ap.add_argument("--min-games", type=int, default=30)
        ap.add_argument("--boot", type=int, default=1000)
        ap.add_argument("--seed", type=int, default=1)
        synth_cmd(ap.parse_args(rest))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
