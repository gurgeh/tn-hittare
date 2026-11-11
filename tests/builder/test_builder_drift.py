"""The winner's curse at the frontier, planted: frontier values are the truth plus noise."""

import numpy as np
import pytest

from tnh.builder.drift import (RevealExpander, curse_experiment, drift, drift_text, noisy_values,
                               solve_until_stable)
from tnh.builder.game import Game, Params
from tnh.builder.search import solve
from tnh.builder.synth import random_game


def test_exact_values_have_no_curse_and_shrinking_to_the_parent_is_neutral():
    # Neutral when the frontier's parents are the opponent's (visible depth even, me
    # first): the parent's value is the expectation of its children, so shrinking them
    # toward it changes no value above. Below one of my own positions it does shrink
    # the difference between my moves, which is the point of it.
    res = curse_experiment(8, 0.0, {"none": {}, "parent": {"shrink": 0.6}}, depth=6, visible=4)
    for r in res.values():
        assert r[:, 0] == pytest.approx(r[:, 1], abs=1e-12)
    assert res["none"] == pytest.approx(res["parent"], abs=1e-12)


def test_noisy_values_overstate_the_choice_and_shrinking_helps():
    res = curse_experiment(30, 0.08, {"none": {}, "parent": {"shrink": 0.5}}, depth=7, visible=3)
    opt = {k: (r[:, 0] - r[:, 1]).mean() for k, r in res.items()}
    assert opt["none"] > 0.01                       # the search believes its noise
    assert opt["parent"] < 0.6 * opt["none"]
    assert res["parent"][:, 1].mean() >= res["none"][:, 1].mean() - 1e-3   # and chooses no worse


def noisy_world(seed, sigma=0.08):
    rng = np.random.default_rng(seed)
    full = random_game(rng, depth=7, branching=(2, 3), alpha=0.5)
    params = Params(eps_white=0.08)
    noisy_values(full, params, rng, sigma)
    return full, params


def test_drift_of_a_fixed_repertoire_ends_at_the_truth():
    full, params = noisy_world(1)
    truth = Game(full, params)
    ex = RevealExpander(full, params, depth=3)
    fixed = solve(Game(ex.graph, params), 3)[1][3].learned
    rs = drift(ex, fixed, N=3, rounds=60, per_round=5)
    assert rs[-1].fixed == pytest.approx(truth.evaluate(fixed), abs=1e-12)   # all below it revealed
    assert rs[-1].frontier_in_zone == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(rs[0].prev_chosen_now) and not np.isnan(rs[1].prev_chosen_now)
    assert rs[0].overlap == len(fixed)
    assert "fixed" in drift_text(rs)


def test_drift_can_compare_variants_on_one_growth():
    full, params = noisy_world(2)
    ex = RevealExpander(full, params, depth=3)
    fixed = solve(Game(ex.graph, params), 2)[1][2].learned
    from dataclasses import replace
    rs = drift(ex, fixed, N=2, rounds=3, per_round=4, variants={"shrink .5": replace(params, shrink=0.5)})
    assert all("shrink .5" in r.variants for r in rs)
    assert "valued by itself" in drift_text(rs)


def test_solve_until_stable_converges_and_says_when_it_does_not():
    full, params = noisy_world(3)
    st = solve_until_stable(RevealExpander(full, params, depth=3), 3, per_round=10, tol=1e-3, max_rounds=50)
    assert st.converged
    r, before, after, _ = st.history[-1]
    assert abs(after - before) < 1e-3 or before == after
    st2 = solve_until_stable(RevealExpander(full, params, depth=3), 3, per_round=1, tol=1e-12, max_rounds=2)
    assert not st2.converged and st2.rounds == 2
