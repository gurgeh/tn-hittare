"""The deeper soundness check: a failing learned move is banned and the search rerun."""

import pytest

from tnh.builder.deepcheck import DeepChecker, check_repertoire
from tnh.builder.game import Game, Params
from tnh.builder.search import solve
from tnh.builder.synth import build, leaf, me, opp


def two_choices():
    # At A, "flashy" is sound by the search's engine and wins more; "solid" is the next best.
    A = me(("flashy", 0.1, 0.7, leaf(0.9)), ("solid", 0.1, 0.68, leaf(0.7)), ("usual", 0.8, 0.6, leaf(0.5)), key="A")
    return build(opp(("a", 1.0, 0.5, A), key="root"))


def test_a_move_that_fails_deeper_is_banned_and_replaced():
    g = two_choices()
    params = Params(eps_white=0.05)
    assert solve(Game(g, params), 1)[1][1].learned == {"A": "flashy"}
    deeper = {("A", "flashy"): 0.40, ("A", "solid"): 0.69}    # the deep engine refutes flashy
    calls = []

    def check(key, move):
        calls.append((key, move))
        alt = max(v for (k, m), v in deeper.items() if m != move)
        return {"move_value": deeper[(key, move)], "alt_move": "?", "alt_value": alt,
                "passes": deeper[(key, move)] >= alt - 0.05}

    res = check_repertoire(g, params, K=1, N=1, check=check)
    assert res.settled
    assert res.sols[1].learned == {"A": "solid"}
    assert ("A", "flashy") in res.params.banned
    assert calls == [("A", "flashy"), ("A", "solid")]
    assert any("fails" in line for line in res.log)


def test_unsettled_after_max_rounds_is_reported():
    g = two_choices()
    res = check_repertoire(g, Params(eps_white=0.5), K=1, N=1, max_rounds=1,
                           check=lambda k, m: {"move_value": 0.0, "alt_move": "?", "alt_value": 1.0, "passes": False})
    assert not res.settled
    assert "not all checked" in res.log[-1]


def test_stockfish_deep_check_on_real_positions():
    with DeepChecker(20_000) as dc:
        ok = dc.check(["e2e4"], "d7d6", eps=0.05)
        assert ok["passes"] and 0.3 < ok["move_value"] < 0.6
        bad = dc.check(["e2e4", "e7e5", "g1f3"], "d8h4", eps=0.05)    # 2...Qh4?? 3.Nxh4
        assert not bad["passes"] and bad["move_value"] < 0.1
        assert dc.searches == 4
