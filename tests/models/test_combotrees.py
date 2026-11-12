"""The trees on top of Combo: shared features, stored models, live scoring."""

import json

import chess
import numpy as np
import pytest

from tnh.models import combotrees as ct
from tnh.models.base import Prediction, Query


def test_ranks_within_positions():
    x = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
    np.testing.assert_array_equal(ct.ranks(x, np.array([0, 3, 5])), [2, 0, 1, 0, 1])


def test_base_columns_follow_the_configuration():
    assert ct.base_columns(ct.CONFIGS["maia23m"]) == ["m0", "mbest"]
    assert ct.base_columns(ct.CONFIGS["+pol"]) == ["m0", "mbest", "lc0pol"]
    assert ct.base_columns(ct.CONFIGS["+ch-all"])[-1] == "lc0ch_cploss"
    assert ct.base_columns(ct.CONFIGS[ct.MAIN])[-1] == "lc0ch_unev"


def test_board_facts_and_context():
    b = chess.Board()
    f = ct.board_facts([b.fen()], [["e2e4", "g1f3"]])
    assert f["board_piece_N"].tolist() == [0.0, 1.0] and f["board_capture"].tolist() == [0.0, 0.0]
    Xb = np.log(np.array([[0.6, 0.5], [0.4, 0.5]]))
    Xe, en, C, cn = ct.extra_features(Xb, ["m0", "mbest"], np.array([0, 2]), [b.fen()], [["e2e4", "g1f3"]],
                                      np.array([2400.0]), np.array([2200.0]), np.array([0.0]))
    assert C[0, cn.index("ctx_elo_diff")] == pytest.approx(1.0)
    assert C[0, cn.index("ctx_entropy_mbest")] == pytest.approx(np.log(2))


@pytest.mark.parametrize("name", [n for n in ct.CONFIGS if (ct.MODEL_DIR / f"{n}.json").exists()])
def test_stored_models_are_complete(name):
    m = json.loads((ct.MODEL_DIR / f"{name}.json").read_text())
    assert m["base_columns"] == ct.base_columns(ct.CONFIGS[name])
    assert len(m["linear"]["beta"]) == len(m["base_columns"]) and m["commit"]
    assert (ct.MODEL_DIR / f"{name}.txt").exists() and m["trees"]["n_trees"] > 0


def test_front_model_scores_a_distribution():
    if not (ct.MODEL_DIR / f"{ct.MAIN}.json").exists():
        pytest.skip("no stored main model")
    from tnh.models.base import MoveValues

    fm = ct.FrontModel(ct.MAIN)
    b = chess.Board()
    q = Query(b.fen(), [], 2200, 2150)
    ms = [m.uci() for m in b.legal_moves]
    n = len(ms)
    lp = np.log(np.full(n, 1.0 / n, dtype=np.float32))
    pred = Prediction(ms, lp, None)
    vals = MoveValues(ms, np.full(n, 0.5, np.float32), np.zeros(n, np.float32), np.zeros(n, np.int16), None)
    feats = fm.features([q], ([pred], [pred]), [pred], [vals], [set(ms[:5])])
    for kind in ("linear", "trees"):
        out = fm.logp(feats, kind)[0]
        assert abs(np.exp(out.astype(np.float64)).sum() - 1) < 1e-5


@pytest.mark.slow
def test_live_trees_combo():
    from tnh.models import lc0
    from tnh.models.combo import Combo

    if not lc0.binary().exists() or not (ct.MODEL_DIR / f"{ct.MAIN}.json").exists():
        pytest.skip("no Lc0 or no stored model")
    b = chess.Board()
    qs = []
    for mv in ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4"]:
        qs.append(Query(b.fen(), [x.uci() for x in b.move_stack], 2200, 2100))
        b.push_uci(mv)
    lin = Combo(prune="p0.5")
    got = lin.predict(qs)
    del lin            # one model on the GPU at a time: the GPU is shared
    import gc

    import torch
    gc.collect()
    torch.cuda.empty_cache()
    for a, c in zip(got, Combo(prune="p0.5", trees=True).predict(qs)):
        assert sorted(a.moves) == sorted(c.moves)
        assert abs(np.exp(c.logp.astype(np.float64)).sum() - 1) < 1e-5
        pa = dict(zip(a.moves, np.exp(a.logp))); pc = dict(zip(c.moves, np.exp(c.logp)))
        assert 0.5 * sum(abs(pa[m] - pc[m]) for m in pa) < 0.2


@pytest.mark.slow
def test_default_move_model_is_policy_trees():
    from tnh.models import DEFAULT_MOVE_MODEL, default_move_model, lc0
    from tnh.models.combo import Combo

    if not lc0.binary().exists() or not (ct.MODEL_DIR / "+pol.json").exists():
        pytest.skip("no Lc0 or no stored model")
    b = chess.Board()
    qs = []
    for mv in ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4"]:
        qs.append(Query(b.fen(), [x.uci() for x in b.move_stack], 2200, 2100))
        b.push_uci(mv)
    model = default_move_model()
    assert model.name == DEFAULT_MOVE_MODEL
    got = model.predict(qs)
    del model          # one model on the GPU at a time: the GPU is shared
    import gc

    import torch
    gc.collect()
    torch.cuda.empty_cache()
    for a, c in zip(got, Combo(prune="p0.5", trees=True).predict(qs)):
        assert sorted(a.moves) == sorted(c.moves)
        assert abs(np.exp(a.logp.astype(np.float64)).sum() - 1) < 1e-5
        assert a.wdl is not None and abs(sum(a.wdl) - 1) < 1e-5
        pa = dict(zip(a.moves, np.exp(a.logp))); pc = dict(zip(c.moves, np.exp(c.logp)))
        assert 0.5 * sum(abs(pa[m] - pc[m]) for m in pa) < 0.2


def test_fit_refuses_to_overwrite_stored_models():
    with pytest.raises(SystemExit, match="--overwrite"):
        ct.main(["fit", "--configs", ct.MAIN])


def _synthetic(n_pos=400, n_moves=6, seed=0):
    rng = np.random.default_rng(seed)
    offsets = np.arange(0, n_pos * n_moves + 1, n_moves)
    score = rng.normal(size=n_pos * n_moves) * 2
    from tnh.eval import trees as T
    lp = T.group_logp(score, offsets)
    p = np.exp(lp)
    y = np.array([offsets[i] + rng.choice(n_moves, p=p[offsets[i]:offsets[i + 1]]) for i in range(n_pos)])
    return score, lp, offsets, y, np.ones(n_pos), np.arange(n_pos) % 4


def test_reliability_tables_on_a_calibrated_model():
    score, lp, offsets, y, w, bands = _synthetic(4000)
    r = ct.reliability(lp, offsets, y, w, bands)
    assert set(r) == {"all"} | set(ct.BAND_LABELS)
    for b in r["all"]["bins"]:
        if b["moves"] > 500:
            assert abs(b["pred"] - b["played"]) < 0.05
    t = r["all"]["threshold"]["0.8"]
    assert 0 < t["share_of_positions"] < 1


def test_temperature_recovers_a_scaled_score():
    score, lp, offsets, y, w, bands = _synthetic(4000, seed=1)
    t = ct.fit_temperature(score * 2.0, offsets, y, w)        # a model twice too sharp
    assert t[0] == pytest.approx(2.0, rel=0.15)
    tb = ct.fit_temperature(score * 2.0, offsets, y, w, bands, 4)
    assert len(tb) == 4 and np.all(np.abs(tb - 2.0) < 0.6)


def test_ratio_table_is_one_for_the_true_model():
    score, lp, offsets, y, w, bands = _synthetic(4000, seed=2)
    mask = np.zeros(len(lp), bool)
    mask[offsets[:-1]] = True                               # "class" = the first move of each position
    obs = mask[y].astype(float)
    mass = {"true": np.add.reduceat(np.exp(lp) * mask, offsets[:-1])}
    clusters = np.arange(len(w)) // 10
    t = ct.ratio_table(obs, mass, w, clusters, bands, np.ones(len(w), bool), 200)
    assert abs(t["all"]["true"]["ratio"] - 1) < 0.1 and t["all"]["true"]["lo"] < 1 < t["all"]["true"]["hi"]
