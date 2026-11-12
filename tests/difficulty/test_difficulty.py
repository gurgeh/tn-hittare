"""Step 2: the position set, the measures, the analysis, and the planted world."""

import chess
import numpy as np
import polars as pl
import pytest

from tnh.data.edges import EdgeDB, pos_hash, tables_from_games
from tnh.difficulty import analysis as A
from tnh.difficulty import positions as P
from tnh.difficulty import synth
from tnh.difficulty.compute import parse_root
from tnh.difficulty.measures import loss_measures


def games(rows):
    """rows: (moves, result, white_elo, black_elo, tempo)"""
    return pl.DataFrame({
        "moves_uci": [r[0] for r in rows], "result": [r[1] for r in rows], "year": [2024] * len(rows),
        "white_elo": [r[2] for r in rows], "black_elo": [r[3] for r in rows], "tempo": [r[4] for r in rows],
    })


def board(moves: str) -> chess.Board:
    b = chess.Board()
    for m in moves.split():
        b.push_uci(m)
    return b


# --- measures ---------------------------------------------------------------------


def test_loss_measures_by_hand():
    p = np.array([0.5, 0.3, 0.2])
    v = np.array([0.55, 0.53, 0.30])
    el, hold, nn = loss_measures(p, v, eps=0.03)
    assert el == pytest.approx(0.3 * 0.02 + 0.2 * 0.25)
    assert hold == pytest.approx(0.8)
    assert nn == 0
    # a move without a value is left out and the rest renormalised
    el, hold, nn = loss_measures(p, np.array([0.55, np.nan, 0.30]), eps=0.03)
    assert nn == 1 and el == pytest.approx(0.2 / 0.7 * 0.25) and hold == pytest.approx(0.5 / 0.7)
    # with a best value from outside the subset
    el, _, _ = loss_measures(np.array([1.0]), np.array([0.50]), eps=0.03, best=0.60)
    assert el == pytest.approx(0.10)


def test_parse_root_skips_a_final_bound():
    lines = [
        "info depth 18 seldepth 25 multipv 1 score cp 40 wdl 100 890 10 nodes 1 nps 1 hashfull 0 tbhits 0 time 1 pv e2e4 e7e5",
        "info depth 19 seldepth 26 multipv 1 score cp 55 upperbound wdl 150 845 5 nodes 2 nps 1 hashfull 0 tbhits 0 time 1 pv d2d4",
        "bestmove e2e4 ponder e7e5",
    ]
    r = parse_root(lines)
    assert r["best"] == "e2e4" and r["cp"] == 40 and r["depth"] == 18 and not r["used_bound"]
    assert r["wdl"] == [0.1, 0.89, 0.01]
    r = parse_root(lines[1:])
    assert r["used_bound"] and r["cp"] == 55


# --- the position set ---------------------------------------------------------------


def test_position_set_sides_ratings_and_thinning():
    rows = (
        [("e2e4 e7e5 g1f3", "1-0", 2300, 1900, "classical")] * 3
        + [("e2e4 e7e5 f1c4", "0-1", 2300, 1900, "classical")] * 1
        + [("e2e4 c7c5", "1/2-1/2", 2000, 2200, "classical")] * 2
        + [("d2d4 d7d5", "1-0", 2000, 2000, "classical")] * 2
        + [("e2e4 e7e5 g1f3", "0-1", 1500, 1500, "classical")] * 5    # band 1400: outside the filter
        + [("e2e4 e7e5 g1f3", "0-1", 2300, 1900, "blitz")] * 5         # other tempo
    )
    e, p = tables_from_games(games(rows))
    db = EdgeDB.from_frames(e, p)
    s, out, info = P.build(db, e.lazy(), min_games=2)
    by = {r["pos_hash"]: r for r in s.iter_rows(named=True)}
    e4 = by[pos_hash(board("e2e4"))]                 # Black to move
    assert e4["n"] == 6 and not e4["stm_white"] and e4["ply"] == 1 and e4["moves_uci"] == "e2e4"
    assert e4["white_elo"] == pytest.approx((4 * 2300 + 2 * 2000) / 6)
    assert e4["black_elo"] == pytest.approx((4 * 1900 + 2 * 2200) / 6)
    assert e4["score_white"] == pytest.approx((3 + 0 + 0.5 * 2) / 6)
    assert e4["total_n"] == 16                       # popularity: every band and tempo
    e5 = by[pos_hash(board("e2e4 e7e5"))]            # White to move
    assert e5["stm_white"] and e5["white_elo"] == pytest.approx(2300) and e5["black_elo"] == pytest.approx(1900)
    assert e5["n"] == 4 and e5["score_white"] == pytest.approx(0.75)
    assert e4["n_children_in_set"] == 2 and e4["top_child_share"] == pytest.approx(4 / 6)
    nf3 = by[pos_hash(board("e2e4 e7e5 g1f3"))]
    assert nf3["n"] == 3 and nf3["leaf"] and nf3["ply"] == 3
    assert pos_hash(board("e2e4 e7e5 f1c4")) not in by   # one game in the filter
    assert info["positions"] == s.height


def test_walk_follows_the_most_played_parent():
    # the same position by two move orders; the second is played more often
    rows = ([("g1f3 g8f6 d2d4", "1-0", 2000, 2000, "classical")] * 2
            + [("d2d4 g8f6 g1f3", "1-0", 2000, 2000, "classical")] * 5)
    e, p = tables_from_games(games(rows))
    db = EdgeDB.from_frames(e, p)
    target = pos_hash(board("d2d4 g8f6 g1f3"))
    paths, stats = P.walk_paths(db, np.array([target], np.uint64))
    assert paths[target] == ["d2d4", "g8f6", "g1f3"]
    assert stats == {"no_parent": 0, "too_long": 0, "replay_mismatch": 0}


def test_family_is_the_path_prefix():
    df = pl.DataFrame({"m": ["e2e4 c7c5 g1f3", "e2e4", "d2d4 d7d5"]})
    assert df.select(P.family(pl.col("m"), 2))["m"].to_list() == ["e2e4 c7c5", None, "d2d4 d7d5"]


# --- the white frame: signs ---------------------------------------------------------


def test_white_frame_signs():
    pos = pl.DataFrame({"pos_hash": pl.Series([1, 2], dtype=pl.UInt64), "stm_white": [True, False],
                        "score_white": [0.5, 0.5], "white_elo": [2100.0, 2100.0], "black_elo": [2000.0, 2000.0],
                        "total_n": [100, 100], "moves_uci": ["e2e4 e7e5", "e2e4 e7e5 g1f3"],
                        "n": [100, 100], "white_wins": [30, 30], "draws": [40, 40]})
    mea = pl.DataFrame({"pos_hash": pl.Series([1, 2], dtype=pl.UInt64), "el_stm": [0.1, 0.1], "hold_stm": [0.6, 0.6],
                        "el_next": [0.02, 0.02], "hold_next": [0.9, 0.9], "net2": [0.08, 0.08],
                        "maia_es_stm": [0.6, 0.6], "engine_es_stm": [0.55, 0.55], "engine_cp_stm": [30.0, 30.0],
                        "el_db": [0.05, 0.05], "hold_db": [0.7, 0.7]})
    df, info = A.white_frame(pos, mea, family_ply=2)
    w, b = df.row(0, named=True), df.row(1, named=True)
    assert w["el_stm"] == pytest.approx(-0.1) and b["el_stm"] == pytest.approx(0.1)      # loss of the mover
    assert w["hard_stm"] == pytest.approx(-0.4) and b["hard_stm"] == pytest.approx(0.4)
    assert w["el_next"] == pytest.approx(0.02) and b["el_next"] == pytest.approx(-0.02)  # the other side's loss
    assert w["net2"] == pytest.approx(-0.08) and b["net2"] == pytest.approx(0.08)
    assert w["engine_es_w"] == pytest.approx(0.55) and b["engine_es_w"] == pytest.approx(0.45)
    assert w["engine_cp_w"] == pytest.approx(0.3) and b["engine_cp_w"] == pytest.approx(-0.3)
    assert w["maia_value"] == pytest.approx(0.6) and b["maia_value"] == pytest.approx(0.4)
    assert info["analysed"] == 2


# --- the analysis on plain numbers -----------------------------------------------------


def _frame(rng, k=60, per=20, beta=0.5, n=200):
    fam = np.repeat(np.arange(k), per)
    m = rng.normal(0, 0.05, k * per)
    e = 0.5 + rng.normal(0, 0.05, k * per)
    mu = 0.52 + 0.5 * (e - 0.5) + beta * m
    wins = rng.binomial(n, mu)
    return pl.DataFrame({"y": wins / n, "n": np.full(k * per, n), "white_wins": wins, "draws": np.zeros(k * per, int),
                         "family": [f"f{i}" for i in fam], "engine_es_w": e, "m": m, "z": rng.normal(0, 0.05, k * per)})


def test_run_recovers_a_coefficient_and_zero():
    rng = np.random.default_rng(3)
    df = _frame(rng)
    res, out = A.run(df, ["m", "z"], A.Setup(boot=300, base=["engine_es_w"]))
    bm, bz = res["measures"]["m"]["beta"], res["measures"]["z"]["beta"]
    assert bm["lo"] < 0.5 < bm["hi"] and bm["lo"] > 0
    assert bz["lo"] < 0 < bz["hi"]
    assert res["measures"]["m"]["dmse"]["lo"] > 0
    assert res["measures"]["m"]["tenths"]["diff"]["est"] > 0
    assert "pred_base" in out.columns


# --- the planted world ----------------------------------------------------------------


@pytest.mark.parametrize("beta", [1.0, 0.0])
def test_planted_world(beta):
    g, planted = synth.world(seed=11, beta=beta, games=20_000, branching=(8, 6, 3, 2))
    for leaves in (False, True):
        res, out, info = synth.pipeline(g, planted, min_games=25, family_ply=2, leaves_only=leaves, boot=300)
        b = res["measures"]["planted"]["beta"]
        z = res["measures"]["noise_measure"]["beta"]
        assert b["lo"] < beta < b["hi"], (leaves, b)
        assert z["lo"] < 0 < z["hi"], (leaves, z)
        if beta > 0:
            assert b["lo"] > 0, (leaves, b)
            assert res["measures"]["planted"]["dmse"]["lo"] > 0
        assert info["walk"] == {"no_parent": 0, "too_long": 0, "replay_mismatch": 0}


# --- measures from the caches, by hand -------------------------------------------------


def test_measures_from_hand_made_caches(tmp_path, monkeypatch):
    from tnh.difficulty import compute as C
    from tnh.difficulty import measures as M
    monkeypatch.setattr(C, "CACHE", tmp_path)
    b = C.Budget(root_nodes=10, multipv_nodes=10, maia_size="5m", child_mass=0.9, max_children=8, min_child_p=0.02)
    maia, sf, root = C.open_caches(b)
    bd = board("e2e4 e7e5")                          # White to move
    legal = [m.uci() for m in bd.legal_moves]
    h = pos_hash(bd)
    w_elo, b_elo = 2200.0, 2000.0
    se, oe = C.maia_elo(w_elo), C.maia_elo(b_elo)
    assert (se, oe) == (2810.0, 2700.0)
    p = np.full(len(legal), 1e-9)
    p[legal.index("g1f3")], p[legal.index("f1c4")], p[legal.index("d1h5")] = 0.6, 0.35, 0.05
    p /= p.sum()
    v = np.full(len(legal), 0.40)
    v[legal.index("g1f3")], v[legal.index("f1c4")], v[legal.index("d1h5")] = 0.55, 0.54, 0.45
    maia.add({"pos_hash": h, "self_elo": se, "oppo_elo": oe, "moves": legal, "logp": np.log(p).tolist(),
              "wdl": [0.5, 0.3, 0.2]})
    sf.add({"pos_hash": h, "moves": legal, "value": v.tolist(), "cp": [0.0] * len(legal), "mate": [0] * len(legal),
            "wdl": [0.1, 0.9, 0.0]})
    eq = C.equal_elo(w_elo, b_elo)                   # H4: both sides at the mapped mean
    assert eq == 2750.0
    maia.add({"pos_hash": h, "self_elo": eq, "oppo_elo": eq, "moves": legal, "logp": np.log(p).tolist(),
              "wdl": [0.4, 0.4, 0.2]})
    root.add({"pos_hash": h, "best": "g1f3", "cp": 35.0, "mate": 0, "wdl": [0.12, 0.86, 0.02], "depth": 20,
              "used_bound": False})
    # the children: Black to move, Black's ratings first
    child_el = {}
    for mv, el_c in (("g1f3", 0.02), ("f1c4", 0.10), ("d1h5", 0.30)):
        bc = bd.copy()
        bc.push_uci(mv)
        cl = [m.uci() for m in bc.legal_moves]
        pc = np.full(len(cl), 1.0 / len(cl))
        vc = np.full(len(cl), 0.5)
        vc[0] = 0.5 + el_c * len(cl) / (len(cl) - 1)  # uniform play loses el_c on average
        maia.add({"pos_hash": pos_hash(bc), "self_elo": oe, "oppo_elo": se, "moves": cl, "logp": np.log(pc).tolist(),
                  "wdl": [0.3, 0.4, 0.3]})
        sf.add({"pos_hash": pos_hash(bc), "moves": cl, "value": vc.tolist(), "cp": [0.0] * len(cl),
                "mate": [0] * len(cl), "wdl": None})
        child_el[mv] = el_c
    for c in (maia, sf, root):
        c.flush()
    pos = pl.DataFrame({"pos_hash": pl.Series([h], dtype=pl.UInt64), "fen": [bd.fen()], "white_elo": [w_elo],
                        "black_elo": [b_elo], "stm_white": [True]})
    dbm = pl.DataFrame({"pos_hash": pl.Series([h, h], dtype=pl.UInt64), "move": ["g1f3", "d1h5"], "n": [3, 1]})
    meas, counts = M.build(pos, b, eps=0.03, db_moves=dbm)
    r = meas.row(0, named=True)
    pg, pc_, pq = p[legal.index("g1f3")], p[legal.index("f1c4")], p[legal.index("d1h5")]
    rest = 1 - pg - pc_ - pq
    assert r["el_stm"] == pytest.approx(pc_ * 0.01 + pq * 0.10 + rest * 0.15, rel=1e-5)
    assert r["hold_stm"] == pytest.approx(pg + pc_, rel=1e-5)
    # children by Maia probability until 90 %: g1f3 and f1c4 (d1h5 would pass 0.9)
    assert r["n_children"] == 2
    exp_next = (pg * child_el["g1f3"] + pc_ * child_el["f1c4"]) / (pg + pc_)
    assert r["el_next"] == pytest.approx(exp_next, rel=1e-4)
    assert r["net2"] == pytest.approx(r["el_stm"] - r["el_next"])
    assert r["maia_es_stm"] == pytest.approx(0.65) and r["engine_es_stm"] == pytest.approx(0.55)
    assert r["maia_es_stm_eq"] == pytest.approx(0.6) and counts["missing_equal_value"] == 0
    assert r["el_db"] == pytest.approx(0.25 * 0.10) and r["hold_db"] == pytest.approx(0.75)
    assert counts["missing_maia"] == 0 and counts["children_missing"] == 0
