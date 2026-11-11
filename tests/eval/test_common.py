"""Log loss, the uniform mixture, the bootstrap over tournaments, and joining features."""

import numpy as np
import polars as pl
import pytest

from tnh.eval.common import (FeatureMismatch, attach, boot_counts, boot_means, cluster_codes, fit_eps, mix_logp,
                             move_scores, parse_variant, same_moves, wmean)

from _synth import draw_played, positions, softmax_rows, write_features


def test_exact_model_has_the_log_loss_of_its_entropy(tmp_path):
    rng = np.random.default_rng(0)
    df, moves = positions(40_000, rng)
    probs = softmax_rows([rng.normal(0, 2, len(m)) for m in moves])
    df = draw_played(df, moves, probs, rng)
    d = write_features(tmp_path / "offset+0", df, moves, {"logp": [np.log(p) for p in probs]})
    sc, _ = move_scores(df, d)
    w = np.ones(df.height)
    ll, ent = wmean(-sc.lp, w), wmean(sc.entropy, w)
    se = np.std(-sc.lp - sc.entropy) / np.sqrt(df.height)
    assert abs(ll - ent) < 4 * se
    # a wrong model does worse than the right one, by the KL divergence
    flat = [np.full(len(m), -np.log(len(m))) for m in moves]
    d2 = write_features(tmp_path / "offset+100", df, moves, {"logp": flat})
    sc2, _ = move_scores(df, d2)
    assert wmean(-sc2.lp, w) > ll + 0.1


def test_eps_is_recovered_from_data_made_with_a_known_eps():
    rng = np.random.default_rng(1)
    n_rows, eps = 60_000, 0.05
    n = rng.integers(10, 40, n_rows).astype(float)
    lp_model, lp_played = np.empty(n_rows), np.empty(n_rows)
    for i in range(n_rows):
        k = int(n[i])
        p = softmax_rows([rng.normal(0, 3, k)])[0]
        truth = (1 - eps) * p + eps / k
        j = rng.choice(k, p=truth)
        lp_model[i] = np.log(p[j])
    e = fit_eps(lp_model, n)
    assert abs(e - eps) < 0.006
    # the mixture is the only smoothing: at eps=0 it is the model itself
    assert np.array_equal(mix_logp(lp_model, n, 0.0), lp_model)
    # and a model that is exactly right gets eps = 0 (or nearly)
    lp_exact = np.log(np.full(1000, 0.5))
    assert fit_eps(lp_exact, np.full(1000, 2.0)) < 1e-6


def test_bootstrap_resamples_tournaments_not_rows():
    # two tournaments, all the variation is between them
    x = np.r_[np.zeros(500), np.ones(500)]
    clusters = np.r_[np.zeros(500, int), np.ones(500, int)]
    counts = boot_counts(2, 2000, 3)
    assert counts.shape == (2000, 2) and np.all(counts.sum(1) == 2)
    b = boot_means(x, np.ones(1000), clusters, counts)[:, 0]
    assert set(np.round(b, 6)) <= {0.0, 0.5, 1.0}
    assert 0.2 < np.mean(b == 0.5) < 0.8 and np.mean(b == 0.0) > 0.1   # a row bootstrap would sit at 0.5
    # the replicate mean is the mean over the drawn tournaments' rows
    rng = np.random.default_rng(2)
    x = rng.normal(size=300)
    cl = rng.integers(0, 30, 300)
    w = rng.uniform(0.5, 2, 300)
    counts = boot_counts(30, 5, 4)
    b = boot_means(x, w, cl, counts)[:, 0]
    for r in range(5):
        rw = w * counts[r][cl]
        assert np.isclose(b[r], np.sum(rw * x) / np.sum(rw))


def test_boot_means_by_group():
    rng = np.random.default_rng(5)
    x = rng.normal(size=200)
    cl = rng.integers(0, 10, 200)
    g = rng.integers(0, 3, 200)
    w = np.ones(200)
    counts = boot_counts(10, 4, 1)
    b = boot_means(x, w, cl, counts, groups=g, n_groups=3)
    for r in range(4):
        for k in range(3):
            m = g == k
            rw = counts[r][cl[m]]
            assert np.isclose(b[r, k, 0], np.sum(rw * x[m]) / np.sum(rw))


def test_cluster_codes_are_dense():
    s = pl.DataFrame({"tournament_id": ["b", "a", "b", "z"]})
    assert cluster_codes(s).tolist() == [1, 0, 1, 2]


def test_missing_rows_and_moves_fail_loudly(tmp_path):
    rng = np.random.default_rng(6)
    df, moves = positions(50, rng)
    probs = softmax_rows([rng.normal(size=len(m)) for m in moves])
    df = draw_played(df, moves, probs, rng)
    d = write_features(tmp_path / "a", df.head(40), moves[:40], {"logp": [np.log(p) for p in probs[:40]]})
    with pytest.raises(FeatureMismatch, match="no features"):
        attach(df, d, ["logp"])
    bad = df.with_columns(pl.lit("zz").alias("played_uci"))
    d2 = write_features(tmp_path / "b", df, moves, {"logp": [np.log(p) for p in probs]})
    with pytest.raises(FeatureMismatch, match="played move"):
        attach(bad, d2, ["logp"])
    # same moves in another order
    rev = [m[::-1] for m in moves]
    d3 = write_features(tmp_path / "c", df, rev, {"logp": [np.log(p[::-1]) for p in probs]})
    a, b = attach(df, d2, ["logp"]), attach(df, d3, ["logp"])
    with pytest.raises(FeatureMismatch, match="differ"):
        same_moves(a, b, "test")


def test_variant_names():
    assert parse_variant("offset+300") == {"kind": "offset", "mover": 300.0, "opp": 300.0, "suffix": ""}
    assert parse_variant("offset-200-opp+100")["opp"] == 100.0
    assert parse_variant("fixed2200-2100")["oppo"] == 2100.0
    with pytest.raises(ValueError):
        parse_variant("nodes10000")
