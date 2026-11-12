"""The shrinkage comparison: planted rules are recovered and win, the rule forms, the collector."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from tnh.web import shrinkfit, shrinkrule as sr


def planted(k: float, n_pos: int = 3000, seed: int = 0, share_slope: float = 0.0) -> pl.DataFrame:
    """Pairs from a normal-normal world: the best rule is n / (n + k) with k = sigma^2 / tau^2,
    toward a prior that moves with log share when `share_slope` is set."""
    rng = np.random.default_rng(seed)
    sigma, tau = 1.0, 1.0 / np.sqrt(k)
    rows = []
    for p in range(n_pos):
        m = rng.integers(2, 8)
        n = rng.choice([1, 2, 3, 5, 10, 30, 100], size=m)
        pos_n = n.sum()
        prior = share_slope * np.log(n / pos_n)
        mu = prior + rng.normal(0, tau, m)
        y = mu + rng.normal(0, sigma / np.sqrt(n))
        nt = rng.integers(1, 6, m)
        yt = mu + rng.normal(0, sigma / np.sqrt(nt))
        for j in range(m):
            rows.append((p + 1, j, y[j], float(n[j]), float(pos_n), 0.5, 0.0, yt[j], float(nt[j]), 0.0))
    return pl.DataFrame(rows, schema=["pos", "move", "y", "count", "pos_n", "spread", "pos_mean", "y_test", "weight",
                                      "base"], orient="row").with_columns(pl.col("pos").cast(pl.UInt64))


def test_planted_k_is_recovered_and_wins():
    p = planted(k=4.0)
    a = shrinkfit._arrays(p)
    k = float(np.exp(shrinkfit.fit_form("nk", a, 1.0)[0]))
    assert 3.0 < k < 5.3
    res = shrinkfit.compare(p, forms=("none", "prior", "nk", "bins", "hill"), boot=50)
    g = res["groups"]["all"]
    assert g["nk"]["mse"] < g["none"]["mse"] and g["nk"]["mse"] < g["prior"]["mse"]
    assert g["winner"] in ("nk", "hill", "bins")
    # every MSE is out of fold, with an interval around it
    assert all(g[f]["lo"] <= g[f]["mse"] <= g[f]["hi"] for f in ("none", "prior", "nk"))
    # the binned weights follow n / (n + 4)
    w = shrinkfit.bin_weights(a)
    assert abs(w[0] - 0.2) < 0.08 and abs(w[-2] - 100 / 104) < 0.08   # bin 51-100 holds n = 100


def test_a_planted_share_prior_wins():
    p = planted(k=4.0, share_slope=1.0, seed=1)
    res = shrinkfit.compare(p, forms=("nk", "hill", "share_hill"), boot=30)
    assert res["groups"]["all"]["winner"] == "share_hill"
    prm = shrinkfit.fit_form("share_hill", shrinkfit._arrays(p), 1.0)
    assert abs(prm[1] - 1.0) < 0.2       # the slope on log share


def test_rule_forms():
    n = np.array([1.0, 4.0, 100.0])
    assert np.allclose(sr.weight("nk", [np.log(4)], n, n, n), n / (n + 4))
    assert np.allclose(sr.weight("hill", [0.0, np.log(4)], n, n, n), n / (n + 4))
    assert np.allclose(sr.weight("bins", np.linspace(0.1, 0.9, 9), np.array([1, 3, 1000]), n, n), [0.1, 0.3, 0.9])
    r = sr.Rule("share_hill", [0.0, 10.0, 0.0, 0.0, np.log(4)])
    pr = r.prior(base=np.zeros(1), n=np.array([1.0]), pos_n=np.array([10.0]))
    assert pr[0] == pytest.approx(10 * np.log(0.1))
    assert "ln share" in r.describe() and "n^1.00" in r.describe()
    assert r.predict(np.array([5.0]), np.zeros(1), np.array([4.0]), np.array([4.0]), np.zeros(1))[0] == pytest.approx(2.5)


def test_collect_and_pairs_on_the_fixture():
    from _webfix import tables
    from tnh.data.edges import EdgeDB
    edges, index, parents = tables()
    db = EdgeDB.from_frames(edges, index, parents)
    raw = shrinkfit.collect(db, walks=30, seed=1, tempi=None, split_year=2022, log=lambda *_: None)
    assert set(raw["part"].unique()) == {"train", "test"}
    # every game from the start: train before 2022 (2019, 2020, 2021), test from 2022
    start = raw.filter(pl.col("pos") == raw["pos"][0])
    assert start.filter(pl.col("part") == "train")["n"].sum() + start.filter(pl.col("part") == "test")["n"].sum() == 7
    p = shrinkfit.pairs(raw, "w_elo")
    assert {"y", "y_test", "count", "weight", "base", "pos_n", "spread"} <= set(p.columns)
    assert (p["base"] == 0).all()
    p = shrinkfit.pairs(raw, "score")
    assert p.height >= 1 and ((p["base"] >= 0) & (p["base"] <= 1)).all()
