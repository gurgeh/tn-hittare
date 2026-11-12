"""Is the static result model consistent across a move, and how much of the look-ahead's drift is the move model?

    uv run python -m tnh.eval.consistency split      # CPU: the per-child run (lookahead --full-exact) by the played move
    uv run python -m tnh.eval.consistency pairs      # GPU: the cheap model's sources before and after each played move
    uv run python -m tnh.eval.consistency analyse    # CPU: consistency per band, and which input drives it

All values are expected scores ``w + d/2``. Per test position n with the move actually played (child c):

    e1 - s(n) = [e1 - s(c)] + [s(c) - s(n)]

The second bracket is the **static model's consistency**: s(n) and s(c) predict the same game result, the
played child is a real game position, so for a calibrated model its mean is 0. The first bracket is the
**move model's bias in static units**: e1 averages s over the model's move distribution, s(c) is one draw from
the real one. ``lambda_move = -mean(e1 - s(c)) / mean(s(best child) - e1)`` is the share of ``Prepared`` that
cancels the move model's bias alone. ``split`` does this on the per-child run at the root ply and at the reply
(the game's next move at the played child, where the played child was expanded).

``pairs`` and ``analyse`` measure the consistency on every test game (``test_value_all``: one position per
game, the position after the played move as its pair), and decompose the change of the logit's win-vs-loss
log odds across the move into the result model's terms: each source's term, the context terms (intercept,
mean rating, ply, White to move; the rating difference cancels exactly), and each raw source alone.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import chess
import numpy as np
import polars as pl

from tnh.eval.lookahead import OUT, POS_TEST
from tnh.paths import CORPUS, REPORTS

PAIRS = Path.home() / "out" / "tn-hittare" / "scratch" / "eval" / "consistency" / "pairs.parquet"
BANDS = {"rating": ([0, 2000, 2200, 2400, 9999], ["1800-1999", "2000-2199", "2200-2399", "2400+"]),
         "ply": ([0, 10, 20, 30, 41, 999], ["0-9", "10-19", "20-29", "30-40", "41+"])}
SRC = ["maia0", "maia600", "lc0raw"]


def es(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    return p[..., 0] + 0.5 * p[..., 1]


def _groups(mean: np.ndarray, ply: np.ndarray) -> dict[str, np.ndarray]:
    g = {"all": np.ones(len(mean), bool)}
    for name, vals in (("rating", mean), ("ply", ply)):
        edges, labels = BANDS[name]
        b = np.searchsorted(edges, vals, side="right") - 1
        for i, lab in enumerate(labels):
            if (b == i).sum() >= 30:
                g[f"{name} {lab}"] = b == i
    return g


def _games(game_ids) -> pl.DataFrame:
    from tnh.eval.sample import drop_cache
    g = (pl.scan_parquet(CORPUS / "games.parquet").select("game_id", "moves_uci")
         .join(pl.LazyFrame({"game_id": list(set(game_ids))}), on="game_id", how="semi").collect())
    drop_cache(CORPUS / "games.parquet")
    return g


def _means(X: np.ndarray, groups, cl, counts) -> dict:
    """Per group, the column means of X with tournament-bootstrap intervals."""
    from tnh.eval.common import boot_means, est
    out = {}
    for name, m in groups.items():
        bm = boot_means(X[m], np.ones(m.sum()), cl[m], counts)
        out[name] = [est(X[m, c].mean(), bm[:, c]) for c in range(X.shape[1])] + [int(m.sum())]
    return out


def _ratio(num, den, groups, cl, counts) -> dict:
    from tnh.eval.common import boot_means
    out = {}
    for name, m in groups.items():
        bm = boot_means(np.column_stack([num[m], den[m]]), np.ones(m.sum()), cl[m], counts)
        r = -bm[:, 0] / bm[:, 1]
        out[name] = {"est": float(-num[m].mean() / den[m].mean()), "lo": float(np.quantile(r, 0.025)),
                     "hi": float(np.quantile(r, 0.975))}
    return out


f_ = lambda e: f"{e['est']:+.4f} [{e['lo']:+.4f}, {e['hi']:+.4f}]"   # noqa: E731


def cmd_split(a) -> None:
    from tnh.eval.common import boot_counts, cluster_codes, read_sample, write_result

    test = read_sample(a.positions).with_columns(pl.col("ply").cast(pl.Int64))
    d = pl.read_parquet(OUT / f"{a.run}.parquet").with_columns(pl.col("ply").cast(pl.Int64))
    j = test.join(d, on=["game_id", "ply"], how="inner").sort("game_id", "ply")
    j = j.join(_games(j.get_column("game_id").to_list()), on="game_id", how="left", maintain_order="left")
    n = j.height
    S = es(np.array(j.get_column("static").to_list()))
    rows = {k: [] for k in ("e1", "played", "best", "has_reply", "r_e1", "r_played", "r_best", "r_static")}
    for fen, played, ply, moves, cp, cw, ck, rp, rw in j.select(
            "fen", "played_uci", "ply", "moves_uci", "c_p", "c_wdl", "c_kept", "r_p", "r_wdl").iter_rows():
        b = chess.Board(fen)
        legal = [m.uci() for m in b.legal_moves]           # the order of the run's children (Maia's legal-move order)
        if len(legal) != len(cp):
            raise ValueError(f"{fen}: {len(legal)} legal moves, the run has {len(cp)}")
        p = np.array(cp, float)
        p /= p.sum()
        E = es(np.array(cw, float))                        # root mover's side
        k = legal.index(played)
        rows["e1"].append(p @ E)
        rows["played"].append(E[k])
        rows["best"].append(E.max())
        # the reply at the played child, from the child's side to move (the opponent): 1 - root side
        ms = moves.split()
        kept = np.array(ck, bool)
        ok = bool(kept[k]) and ply + 1 < len(ms) and len(rp[int(kept[:k].sum())]) > 0
        rows["has_reply"].append(ok)
        if ok:
            jj = int(kept[:k].sum())
            b.push_uci(played)
            rl = [m.uci() for m in b.legal_moves]
            q = np.array(rp[jj], float)
            q /= q.sum()
            R = 1 - es(np.array(rw[jj], float))
            rows["r_e1"].append(q @ R)
            rows["r_played"].append(R[rl.index(ms[ply + 1])])
            rows["r_best"].append(R.max())
            rows["r_static"].append(1 - E[k])
        else:
            for key in ("r_e1", "r_played", "r_best", "r_static"):
                rows[key].append(np.nan)
    R_ = {k: np.array(v, float) for k, v in rows.items()}
    cl = cluster_codes(j)
    counts = boot_counts(cl.max() + 1, a.boot, a.seed)
    mean = (j.get_column("mover_elo").to_numpy() + j.get_column("opp_elo").to_numpy()) / 2
    ply = j.get_column("ply").to_numpy()
    groups = _groups(mean, ply)
    X = np.column_stack([R_["e1"] - S, R_["e1"] - R_["played"], R_["played"] - S, R_["best"] - R_["e1"]])
    root = _means(X, groups, cl, counts)
    lam_move = _ratio(R_["e1"] - R_["played"], R_["best"] - R_["e1"], groups, cl, counts)
    lam_total = _ratio(R_["e1"] - S, R_["best"] - R_["e1"], groups, cl, counts)
    hr = R_["has_reply"].astype(bool)
    Xr = np.column_stack([R_["r_e1"] - R_["r_static"], R_["r_e1"] - R_["r_played"], R_["r_played"] - R_["r_static"],
                          R_["r_best"] - R_["r_e1"]])[hr]
    gr = _groups(mean[hr], ply[hr])
    clr = cluster_codes(j.filter(pl.Series(hr)))
    cr = boot_counts(clr.max() + 1, a.boot, a.seed)
    reply = _means(Xr, gr, clr, cr)
    lam_move_r = _ratio(Xr[:, 1], Xr[:, 3], gr, clr, cr)
    names = ["e1 - s(n)", "e1 - s(played)", "s(played) - s(n)", "best - e1"]
    out = {"positions": n, "with_reply": int(hr.sum()), "columns": names,
           "root": {g: {"n": v[-1], **dict(zip(names, v[:-1])), "lambda_move": lam_move[g], "lambda_total": lam_total[g]}
                    for g, v in root.items()},
           "reply": {g: {"n": v[-1], **dict(zip(names, v[:-1])), "lambda_move": lam_move_r[g]} for g, v in reply.items()}}
    for part, tab, lm in (("root ply", root, lam_move), ("reply ply", reply, lam_move_r)):
        print(f"\n{part}: " + " | ".join(names) + " | lambda_move")
        for g, v in tab.items():
            print(f"  {g:>16} n={v[-1]:>4}: " + " | ".join(f_(e) for e in v[:-1])
                  + f" | {lm[g]['est']:+.2f} [{lm[g]['lo']:+.2f}, {lm[g]['hi']:+.2f}]")
    write_result(a.results, "consistency_split", {"what": "e1 - s(n) = [e1 - s(played child)] + [s(played child) - s(n)], "
                                                  "PolicyTrees per-child run, cheap leaf, expected scores", **out})


def cmd_pairs(a) -> None:
    """The cheap model's sources at each test position and at the position after the played move."""
    from tnh.eval.common import read_sample
    from tnh.eval.result import ResultModel
    from tnh.eval.sample import peak_gb
    from tnh.models.base import Query
    from tnh.models.run import queries

    s = read_sample(a.sample).with_columns(pl.col("ply").cast(pl.Int64))
    s = s.join(_games(s.get_column("game_id").to_list()), on="game_id", how="inner", maintain_order="left")
    qn = queries(s)
    qc = []
    for q, played in zip(qn, s.get_column("played_uci").to_list()):
        b = chess.Board(q.fen)
        b.push_uci(played)
        qc.append(Query(b.fen(), list(q.moves_uci) + [played], q.opp_elo, q.mover_elo))
    rm = ResultModel("cheap")
    live = [i for i, q in enumerate(qc) if not chess.Board(q.fen).is_game_over()]
    t = time.perf_counter()
    out = {}
    for tag, qs in (("n", qn), ("c", [qc[i] for i in live])):
        acc = {k: [] for k in SRC}
        for k in range(0, len(qs), a.batch):
            w = rm.source_wdl(qs[k:k + a.batch])
            for key in SRC:
                acc[key].append(w[key])
        out[tag] = {key: np.concatenate(v) for key, v in acc.items()}
    dt = time.perf_counter() - t
    keep = np.zeros(len(qn), bool)
    keep[live] = True
    df = s.filter(pl.Series(keep)).select("game_id", "ply", "mover_elo", "opp_elo", "white_to_move", "result_mover",
                                          "tournament_id", "out_of_book")
    for key in SRC:
        df = df.with_columns(pl.Series(f"{key}_n", out["n"][key][keep].tolist()),
                             pl.Series(f"{key}_c", out["c"][key].tolist()))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(a.out)
    print(f"{df.height} pairs ({len(qn) - df.height} games end with the move), {2 * df.height / dt:.0f} positions/s; "
          f"peak RSS {peak_gb():.2f} GB; wrote {a.out}")


def cmd_analyse(a) -> None:
    import json

    from tnh.eval.common import boot_counts, cluster_codes, write_result
    from tnh.eval.result import PARAMS, context, design, mlogit_probs, source_logits

    df = pl.read_parquet(a.pairs)
    n = df.height
    p = json.loads(PARAMS.read_text())["cheap"]
    th = np.array(p["coef"])
    srcs = p["sources"]
    W = th.reshape(2, -1)                   # rows: win, draw (against loss); columns: intercept, 2 per source, context
    me, op = df.get_column("mover_elo").to_numpy().astype(float), df.get_column("opp_elo").to_numpy().astype(float)
    ply = df.get_column("ply").to_numpy().astype(float)
    wtm = df.get_column("white_to_move").to_numpy().astype(bool)
    wn = {k: np.array(df.get_column(f"{k}_n").to_list(), float) for k in SRC}
    wc = {k: np.array(df.get_column(f"{k}_c").to_list(), float) for k in SRC}
    wn = {k: v / v.sum(1, keepdims=True) for k, v in wn.items()}
    wc = {k: v / v.sum(1, keepdims=True) for k, v in wc.items()}
    ctx_n = context(me, op, ply, wtm)
    ctx_c = context(op, me, ply + 1, ~wtm)
    Xn, Xc = design(wn, ctx_n, srcs), design(wc, ctx_c, srcs)
    pn, pc = mlogit_probs(th, Xn), mlogit_probs(th, Xc)
    s_n, s_c = es(pn), 1 - es(pc)                    # both for the mover at n
    # the win-vs-loss log odds for the mover at n: eta_w(n) at n, -eta_w(c) at c (the child's mover is the opponent)
    Xin, Xic = np.column_stack([np.ones(n), Xn]), np.column_stack([np.ones(n), Xc])
    terms = {"intercept": [0]}
    for i, k in enumerate(srcs):
        terms[f"{k} term"] = [1 + 2 * i, 2 + 2 * i]
    base = 1 + 2 * len(srcs)
    for i, k in enumerate(["ctx_mean", "ctx_diff", "ctx_ply", "ctx_white"]):
        terms[k] = [base + i]
    # change across the move of the mover's win-vs-loss log odds, per term: -(W x)(c) - (W x)(n)
    dterm = {k: -(Xic[:, c] @ W[0, c]) - (Xin[:, c] @ W[0, c]) for k, c in terms.items()}
    dlogodds = -(Xic @ W[0]) - (Xin @ W[0])
    # each raw source alone: its own expected score and win-vs-loss log odds, before and after (mover at n)
    raw = {}
    for k in SRC:
        raw[f"{k} raw E"] = (1 - es(wc[k])) - es(wn[k])
        zn, zc = source_logits(wn[k]), source_logits(wc[k])
        raw[f"{k} raw log(w/l)"] = -zc[:, 0] - zn[:, 0]
    y = np.array([{1.0: 0, 0.5: 1, 0.0: 2}[x] for x in df.get_column("result_mover").to_list()])
    score = np.array([1.0, 0.5, 0.0])[y]
    cols = {"s(c) - s(n)": s_c - s_n, "score - s(n)": score - s_n, "score - s(c)": score - s_c,
            "change of log(w/l)": dlogodds, **{f"d {k}": v for k, v in dterm.items()}, **raw}
    X = np.column_stack(list(cols.values()))
    cl = cluster_codes(df)
    counts = boot_counts(cl.max() + 1, a.boot, a.seed)
    mean = (me + op) / 2
    groups = _groups(mean, ply)
    tab = _means(X, groups, cl, counts)
    names = list(cols)
    out = {g: {"n": v[-1], **dict(zip(names, v[:-1]))} for g, v in tab.items()}
    print(f"{n} pairs (test_value_all, one per game); means with tournament-bootstrap intervals")
    for g, v in tab.items():
        print(f"\n{g} (n={v[-1]})")
        for nm, e in zip(names, v[:-1]):
            print(f"  {nm:>24}: {f_(e)}")
    # the white-to-move and ply coefficients themselves
    coef = {"win": dict(zip(["intercept"] + [f"{k}_{t}" for k in srcs for t in ("logodds", "draw")] +
                            ["ctx_mean", "ctx_diff", "ctx_ply", "ctx_white"], W[0].tolist())),
            "draw": dict(zip(["intercept"] + [f"{k}_{t}" for k in srcs for t in ("logodds", "draw")] +
                             ["ctx_mean", "ctx_diff", "ctx_ply", "ctx_white"], W[1].tolist()))}
    print("\ncoefficients (win vs loss):", {k: round(v, 3) for k, v in coef["win"].items()})
    write_result(a.results, "consistency_pairs", {"what": "cheap result model across the played move, all test games: "
                                                  "s(child) - s(n) and its decomposition into the logit's terms and the raw sources",
                                                  "pairs": n, "groups": out, "coefficients": coef})


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m tnh.eval.consistency", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(required=True)
    sp = sub.add_parser("split")
    sp.add_argument("--positions", type=Path, default=POS_TEST)
    sp.add_argument("--run", default="trees_full")
    sp.add_argument("--results", type=Path, default=REPORTS / "step1")
    sp.add_argument("--boot", type=int, default=1000)
    sp.add_argument("--seed", type=int, default=7)
    sp.set_defaults(fn=cmd_split)
    sp = sub.add_parser("pairs")
    sp.add_argument("--sample", type=Path, default=CORPUS / "samples" / "test_value_all.parquet")
    sp.add_argument("--out", type=Path, default=PAIRS)
    sp.add_argument("--batch", type=int, default=256)
    sp.set_defaults(fn=cmd_pairs)
    sp = sub.add_parser("analyse")
    sp.add_argument("--pairs", type=Path, default=PAIRS)
    sp.add_argument("--results", type=Path, default=REPORTS / "step1")
    sp.add_argument("--boot", type=int, default=1000)
    sp.add_argument("--seed", type=int, default=7)
    sp.set_defaults(fn=cmd_analyse)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
