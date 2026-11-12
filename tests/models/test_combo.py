"""The combo move model and the frontier value: stored parameters, feature code, and a live run."""

import json

import chess
import numpy as np
import pytest

from tnh.models import combo
from tnh.models.base import MoveValues, Prediction, Query


def test_stored_coefficients_are_complete():
    c = json.loads(combo.COEF_FILE.read_text())
    assert c["columns"] == combo.COLUMNS and len(c["beta"]) == len(combo.COLUMNS)
    assert 0 <= c["eps"] < 0.01 and c["commit"] and c["fit"]["window_rows"] > 0
    for name in combo.STEP1_SETS:   # the stored fit reproduced step 1's figures
        r = c["step1_reproduced"][name]
        assert r["test_mixed"]["est"] == pytest.approx(r["step1_test_mixed"]["est"], abs=1e-9)
    v = json.loads(combo.VALUE_FILE.read_text())
    assert set(v["recal"]) == {"t", "b_win", "b_draw", "k_draw_per_100", "eps"} and v["nodes"] == 50


def test_child_columns_match_the_fit_code():
    mv = MoveValues(["a", "b", "c", "d"], np.array([0.6, 0.5, 1.0, 0.5], dtype=np.float32),
                    np.array([30.0, 0.0, np.nan, 0.0], dtype=np.float32), np.array([0, 0, 1, 0], dtype=np.int16),
                    (0.5, 0.3, 0.2))
    el, cl = combo.child_columns([mv, mv])
    np.testing.assert_allclose(el[0], [0.4, 0.5, 0.0, 0.5], atol=1e-6)
    # mate in 1 counts as 10000 - 100 centipawns
    np.testing.assert_allclose(cl[1], np.log1p(np.array([9900 - 30, 9900, 0, 9900]) / 100), rtol=1e-6)


def test_mix_is_a_distribution():
    lp = np.log(np.array([0.7, 0.2, 0.1]))
    m = combo.mix(lp, 0.01)
    assert np.exp(m).sum() == pytest.approx(1.0)
    assert m[2] > lp[2]


class _Fixed:
    name = "fixed"

    def predict(self, queries):
        return [Prediction(["e2e4", "d2d4"], np.log(np.array([0.6, 0.4], dtype=np.float32)), (0.3, 0.4, 0.3))
                for _ in queries]


class _Frontier:
    def wdl(self, queries):
        return np.array([[0.5, 0.25, 0.25]] * len(queries))


def test_with_value_replaces_only_the_wdl():
    q = Query(chess.STARTING_FEN, [], 2000, 2000)
    p = combo.WithValue(_Fixed(), _Frontier()).predict([q, q])
    assert p[0].moves == ["e2e4", "d2d4"] and p[0].wdl == (0.5, 0.25, 0.25)
    np.testing.assert_allclose(np.exp(p[1].logp), [0.6, 0.4], rtol=1e-6)


@pytest.mark.slow
def test_live_combo_and_frontier():
    from tnh.models import lc0

    if not lc0.binary().exists():
        pytest.skip("no Lc0")
    b = chess.Board()
    moves = ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3"]
    qs = []
    for m in moves:
        qs.append(Query(b.fen(), [x.uci() for x in b.move_stack], 2200, 2100))
        b.push_uci(m)
    model = combo.WithValue(combo.Combo(), combo.FrontierValue())
    preds = model.predict(qs)
    for q, p in zip(qs, preds):
        assert sorted(p.moves) == sorted(m.uci() for m in chess.Board(q.fen).legal_moves)
        assert abs(np.exp(p.logp.astype(np.float64)).sum() - 1) < 1e-5
        assert abs(sum(p.wdl) - 1) < 1e-6
    # the model knows the main line of the Sicilian
    assert preds[2].moves[int(np.argmax(preds[2].logp))] == "g1f3"
    model.frontier.close()


def test_keep_mask_rules():
    from tnh.models.prune import keep_mask

    prob = np.array([0.5, 0.004, 0.2, 0.006, 0.001, 0.001, 0.3])
    off = np.array([0, 4, 7])
    np.testing.assert_array_equal(keep_mask(prob, off, "p0.5"), [1, 0, 1, 1, 0, 0, 1])
    np.testing.assert_array_equal(keep_mask(prob, off, "p50"), [1, 0, 0, 0, 0, 0, 1])   # the top move always stays
    # ties (0.001, 0.001) are broken by move order: the first one is kept
    np.testing.assert_array_equal(keep_mask(prob, off, "top2"), [1, 0, 1, 0, 1, 0, 1])
    np.testing.assert_array_equal(keep_mask(prob, off, "p0.5|top2"), [1, 0, 1, 1, 1, 0, 1])
    assert keep_mask(prob, off, "all").all()


def test_pruned_live_columns_match_the_fit_code():
    from tnh.models.prune import pruned_columns

    a = MoveValues(["a", "b", "c"], np.array([0.6, np.nan, 0.4], dtype=np.float32),
                   np.array([40.0, np.nan, -10.0], dtype=np.float32), np.zeros(3, dtype=np.int16), None)
    el, cl, un = combo.pruned_child_columns([a], [{"a", "c"}], "zero")
    want = pruned_columns(np.array([0.6, np.nan, 0.4]), np.array([40.0, np.nan, -10.0]), np.array([0, 3]),
                          np.array([True, False, True]), "zero")
    np.testing.assert_allclose(el[0], want["lc0ch_eloss"], atol=1e-6)
    np.testing.assert_allclose(cl[0], want["lc0ch_cploss"], atol=1e-6)
    np.testing.assert_array_equal(un[0], [0, 1, 0])
    np.testing.assert_allclose(el[0], [0.0, 0.0, 0.2], atol=1e-6)   # relative to the best evaluated child


@pytest.mark.parametrize("rule", ["p0.5", "p1", "p2"])
def test_pruned_coefficients_stored(rule):
    c = json.loads(combo.coef_file(rule).read_text())
    assert c["prune"] == rule and c["columns"] == combo.COLUMNS + ["lc0ch_unev"] and c["beta"][-1] < 0


@pytest.mark.slow
def test_children_subset_equals_full_evaluation():
    from tnh.models import lc0

    if not lc0.binary().exists():
        pytest.skip("no Lc0")
    b = chess.Board()
    path = []
    for m in ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"]:
        b.push_uci(m)
        path.append(m)
    q = Query(b.fen(), path, 2200, 2200)
    m = lc0.Lc0Children(batch_size=1, pool_across_positions=False)
    full = m.evaluate([q])[0]
    keep = {"e2e3", "d1c2", "g1f3"}
    part = m.evaluate([q], [keep])[0]
    for i, mv in enumerate(full.moves):
        if mv in keep:
            assert part.value[i] == full.value[i] and part.cp[i] == full.cp[i]
        else:
            assert np.isnan(part.value[i]) and np.isnan(part.cp[i]) and part.mate[i] == 0
    assert m.stats["children skipped"] == len(full.moves) - len(keep)


@pytest.mark.slow
def test_live_pruned_combo():
    from tnh.models import lc0

    if not lc0.binary().exists():
        pytest.skip("no Lc0")
    b = chess.Board()
    qs = []
    for mv in ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4"]:
        qs.append(Query(b.fen(), [x.uci() for x in b.move_stack], 2200, 2100))
        b.push_uci(mv)
    full, pruned = combo.Combo(), combo.Combo(prune="p1")
    pf, pp = full.predict(qs), pruned.predict(qs)
    assert pruned.children.stats["children skipped"] > 0
    for a, c in zip(pf, pp):
        assert a.moves == c.moves and abs(np.exp(c.logp.astype(np.float64)).sum() - 1) < 1e-5
        # the most probable move agrees, and the two distributions are close
        assert int(np.argmax(a.logp)) == int(np.argmax(c.logp))
        assert 0.5 * np.abs(np.exp(a.logp) - np.exp(c.logp)).sum() < 0.1
