"""The move model's correction for overestimated mistakes (tnh.movefix)."""

import json

import numpy as np
import pytest

from tnh import movefix as mf


def test_range_rules_hold_elo_and_ply_at_the_fitted_edges_and_clip_the_share():
    r = mf.Rule("lin", [0.17, 0.05, 0.10])
    assert r.share(2200, 6) == r.share(2200, 10) and r.share(2200, 60) == r.share(2200, 40)
    assert r.share(1600, 25) == r.share(1800, 25) and r.share(2800, 25) == r.share(2500, 25)
    assert mf.Rule("const", [0.9]).share(2200, 25) == 0.6
    assert mf.Rule("const", [-0.2]).share(2200, 25) == 0.0
    s = r.share(np.array([1800, 2200, 2500]), np.array([10, 25, 40]))
    assert s.shape == (3,) and np.all(np.diff(s) > 0)


def test_apply_moves_the_share_onto_the_best_move_one_position_and_a_batch():
    r = mf.Rule("const", [0.25])
    p = np.array([0.5, 0.3, 0.2])
    q = r.apply(p, 1, 2200, 25)
    assert np.allclose(q, [0.375, 0.475, 0.15]) and np.isclose(q.sum(), 1)
    P = np.array([[0.5, 0.3, 0.2], [0.6, 0.4, 0.0]])
    Q = r.apply(P, np.array([1, 0]), np.array([2200, 2000]), np.array([25, 30]))
    assert np.allclose(Q[0], q) and np.allclose(Q[1], [0.7, 0.3, 0.0]) and np.allclose(Q.sum(1), 1)


def test_moment_fit_recovers_a_planted_share():
    rng = np.random.default_rng(0)
    n = 20000
    elo = rng.uniform(1800, 2500, n)
    ply = rng.integers(10, 41, n)
    L = rng.exponential(0.02, n)
    true = mf.Rule("lin", [0.18, 0.05, 0.09])
    d = true.share(elo, ply) * L + rng.normal(0, 0.01, n)
    got = mf.fit_ls("lin", d, L, elo, ply)
    assert np.allclose(got, true.params, atol=0.03), got
    c = mf.fit_ls("const", d, L, elo, ply)[0]
    assert np.isclose(c, d.sum() / L.sum())


def test_load_without_file_is_no_correction(tmp_path):
    r = mf.load(tmp_path / "none.json")
    assert r.share(2200, 25) == 0.0


@pytest.mark.slow
def test_stored_rule_reproduces_from_the_runs():
    d = json.loads(mf.RULE_FILE.read_text())
    prov = d["provenance"]
    runs = [mf.Path(r) for r in prov["runs"]]
    if not all(r.exists() for r in runs):
        pytest.skip("look-ahead runs not present")
    j, e1, sp, sb = mf.position_values(runs, [mf.Path(p) for p in prov["positions"]], prov["result_model"])
    assert len(e1) == prov["n_positions"]
    got = mf.fit_ls(d["form"], sp - e1, sb - e1, j.get_column("mover_elo").to_numpy().astype(float),
                    j.get_column("ply").to_numpy().astype(float))
    assert np.allclose(got, d["params"], atol=1e-9)
