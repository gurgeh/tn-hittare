"""The Prob mix: the rule module and its fit on a planted world."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from tnh import movemix as mm
from tnh.web import probfit


def planted(alpha: float, n_pos: int = 1500, seed: int = 0) -> pl.DataFrame:
    """Positions whose true move distribution is Dirichlet(alpha * q) around the model's q:
    the best mix is close to N / (N + alpha)."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_pos):
        k = rng.integers(3, 12)
        q = rng.dirichlet(np.ones(k))
        pi = rng.dirichlet(alpha * q + 1e-3)
        N = int(rng.choice([0, 1, 2, 5, 20, 100, 1000]))
        c = rng.multinomial(N, pi) if N else np.zeros(k, int)
        t = rng.multinomial(int(rng.integers(1, 10)), pi)
        for j in range(k):
            rows.append((p + 1, f"m{j}", float(q[j]), float(c[j]), float(t[j])))
    d = pl.DataFrame(rows, schema=["pos", "move", "q", "c", "t"], orient="row").with_columns(pl.col("pos").cast(pl.UInt64))
    return d.with_columns(N=pl.col("c").sum().over("pos"), T=pl.col("t").sum().over("pos"),
                          k=pl.len().over("pos")).filter(pl.col("T") > 0)


def test_rule_mix_and_describe():
    r = mm.Rule("nk", [math.log(100.0)])
    p = r.mix([10, 0, 0], [0.2, 0.3, 0.5])
    w = 10 / 110
    assert np.allclose(p, w * np.array([1, 0, 0]) + (1 - w) * np.array([0.2, 0.3, 0.5]))
    assert np.allclose(r.mix([0, 0, 0], [0.2, 0.3, 0.5]), [0.2, 0.3, 0.5])       # no games: the model
    t = mm.Rule("tbl_nk", [math.log(0.05 / 0.95)] + [math.log(0.02)] * 5 + [math.log(50.0)])
    assert "1: 0.05" in t.describe() and "above 20: N/(N+50)" in t.describe()
    assert mm.weight("tbl_nk", t.params, np.array([1.0, 21.0]))[1] == pytest.approx(21 / 71)
    assert mm.load(mm.RULE_FILE.with_name("nosuch.json")).kappa == pytest.approx(100.0)


def test_planted_alpha_is_recovered_and_the_mix_wins():
    d = probfit.Data(planted(alpha=30.0))
    k = math.exp(probfit.fit("nk", d)[0])
    assert 15 < k < 60
    res = probfit.compare(d, boot=30)
    g = res["groups"]["all"]
    assert g["nk"]["ll"] < g["model"]["ll"] and g["nk"]["ll"] < g["jeffreys"]["ll"]
    assert not math.isfinite(g["database"]["ll"])            # unseen moves: infinite
    assert res["chosen"] in ("nk", "tbl_nk", "tbl_hill", "move")
    # positions without earlier games: every rule is the model there
    g0 = res["groups"]["N=0"]
    assert g0["nk"]["ll"] == pytest.approx(g0["model"]["ll"])
