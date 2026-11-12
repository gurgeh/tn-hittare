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
