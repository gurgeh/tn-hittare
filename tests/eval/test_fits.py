"""Fitted quantities recover what was planted: logit coefficients, the value recalibration,
the sweep's best offset, the calibration crossing of the mistake classes."""

import numpy as np
import polars as pl

from tnh.eval.logit import build, fit, scores
from tnh.eval.mistakes import crossing
from tnh.eval.sweep import vertex
from tnh.eval.value import fit_recal, recal_logits

from _synth import draw_played, positions, softmax_rows, write_features


def test_logit_recovers_planted_coefficients(tmp_path):
    rng = np.random.default_rng(0)
    df, moves = positions(20_000, rng, split="train")
    # a "model" log probability and an engine whose losses matter
    model_lp = [np.log(p) for p in softmax_rows([rng.normal(0, 1.5, len(m)) for m in moves])]
    values = [rng.uniform(0.2, 0.8, len(m)) for m in moves]
    for v in values:
        v[rng.random(len(v)) < 0.15] = np.nan          # some moves without a value
    beta = {"m": 0.8, "sf_eloss": -3.0, "sf_nan": -1.0}
    truth = []
    for lp, v in zip(model_lp, values):
        best = np.nanmax(v)
        eloss = best - v
        nan = np.isnan(v)
        eloss = np.where(nan, np.nanmax(eloss), eloss)
        truth.append(beta["m"] * lp + beta["sf_eloss"] * eloss + beta["sf_nan"] * nan)
    probs = softmax_rows(truth)
    df = draw_played(df, moves, probs, rng)
    dm = write_features(tmp_path / "m" / "offset+0", df, moves, {"logp": model_lp})
    de = write_features(tmp_path / "sf" / "nodes1", df, moves, {"value": values})
    D, info = build(df, {"m": dm}, {"sf": de})
    assert info["rows_dropped_no_engine_value"] == 0
    assert info["sf"]["moves_without_value"] > 0
    b, it = fit(D)
    got = dict(zip(D.names, b))
    for k, v in beta.items():
        assert abs(got[k] - v) < 0.15 * abs(v) + 0.1, (k, got[k], v)
    # the fitted model is at least as good on its own data as the truth
    lp, _ = scores(D, b)
    true_lp = np.array([np.log(p[m.index(x)]) for p, m, x in zip(probs, moves, df.get_column("played_uci"))])
    assert lp.mean() >= true_lp.mean() - 1e-9


def test_logit_weights_act_like_repeated_rows(tmp_path):
    rng = np.random.default_rng(1)
    df, moves = positions(600, rng, split="train")
    lp = [np.log(p) for p in softmax_rows([rng.normal(0, 1, len(m)) for m in moves])]
    df = draw_played(df, moves, [np.exp(x) for x in lp], rng)
    d = write_features(tmp_path / "offset+0", df, moves, {"logp": lp})
    D, _ = build(df, {"m": d}, {})
    w = rng.integers(0, 3, len(D.w)).astype(float)
    b_w, _ = fit(D, w)
    rows = np.repeat(np.arange(df.height), w.astype(int))
    df2 = df[rows].with_columns(pl.Series("ply", np.arange(len(rows))), pl.Series("game_id", [f"r{i:015x}" for i in range(len(rows))]))
    d2 = write_features(tmp_path / "rep" / "offset+0", df2, [moves[i] for i in rows], {"logp": [lp[i] for i in rows]})
    D2, _ = build(df2, {"m": d2}, {})
    b_r, _ = fit(D2)
    assert np.allclose(b_w, b_r, atol=1e-6)


def test_value_recalibration_recovers_planted_parameters():
    rng = np.random.default_rng(2)
    n = 40_000
    base = rng.dirichlet([2, 1, 2], n)
    r = rng.normal(0, 1.5, n)
    theta = np.array([0.7, 0.3, 0.9, 0.25])
    q = np.exp(recal_logits(theta, np.log(base), r))
    y = np.array([rng.choice(3, p=p) for p in q])
    got = fit_recal(base, y, r, np.ones(n))
    assert np.allclose(got, theta, atol=0.08), got


def test_sweep_vertex_and_mistake_crossing():
    x = np.array([-200, 0, 200, 400, 600], float)
    y = (x - 260) ** 2 / 1e5 + 1.2
    v, edge = vertex(x, y)
    assert abs(v - 260) < 1 and not edge
    assert vertex(x, -x)[1]                       # minimum at the edge is flagged
    d = np.array([-0.3, -0.1, 0.1, 0.3, 0.5])     # predicted minus observed
    assert crossing(x, d) == (100.0, "")
    assert crossing(x, np.full(5, 0.2))[1] == "over"
    assert crossing(x, np.full(5, -0.2))[1] == "under"
