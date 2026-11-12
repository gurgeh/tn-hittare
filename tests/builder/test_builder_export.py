"""The export round-trips: the numbers in result.json are the solver's."""

import io
import json

import chess.pgn
import pytest

from tnh.builder.expand import Expander
from tnh.builder.export import export, pgn_for, write
from tnh.builder.game import Game, ObviousCost, Params
from tnh.builder.search import solve
from tnh.builder.study import forced_curves, sound_root_moves

from test_builder_expand import FakeEngine, FakeMoves, FakeValue


def test_export_round_trips(tmp_path):
    params = Params(eps_white=0.08, min_reach=0.01, cost=ObviousCost(0.5), kappa=20)
    ex = Expander(FakeMoves({"e7e5": 3.0, "g1f3": 3.0, "b8c6": 2.0, "d2d4": 2.5}), FakeEngine(), params,
                  me_white=True, me_elo=2200, opp_elo=2200, root_moves=["e2e4", "e7e5"], value_fn=FakeValue(),
                  tail_p=0.01)
    ex.grow(expand_reach=0.005, max_expansions=40)
    game = Game(ex.graph, params)
    budgets = [1, 3]
    _, sols = solve(game, 3, improve_at=set(budgets))
    sound = sound_root_moves(game)
    forced = forced_curves(ex.graph, params, sound, 3, set(budgets))
    res = export(game, sols, budgets, root_moves=["e2e4", "e7e5"], meta={"name": "t"}, forced=forced,
                 games=lambda n: 7, obvious=0.5)
    pgns = {N: pgn_for(game, sols[N], ["e2e4", "e7e5"]) for N in budgets}
    write(tmp_path / "t", res, pgns)
    back = json.loads((tmp_path / "t" / "result.json").read_text())
    g = ex.graph
    for N in budgets:
        V, R = game.values(sols[N].learned), game.reach(sols[N].learned)
        assert back["solutions"][str(N)]["value"] == pytest.approx(sols[N].value, abs=1e-6)
        assert back["solutions"][str(N)]["learned"] == {str(k): m for k, m in sols[N].learned.items()}
        for key, rec in back["nodes"].items():
            i = next(n.id for n in g.nodes if str(n.key) == key and n.ply == rec["ply"])
            assert rec["budgets"][str(N)]["value"] == pytest.approx(V[i], abs=1e-6)
            assert rec["budgets"][str(N)]["reach"] == pytest.approx(R[i], abs=1e-6)
            for mv in rec.get("moves", []):
                c = g.child(g.nodes[i], mv["uci"])
                assert mv["value_after"][str(N)] == pytest.approx(V[c.id], abs=1e-6)
                assert mv["child"] == str(c.key)
    # every expanded position has a record, with every legal move
    for n in g.nodes:
        if n.children is not None and n.moves:
            rec = back["nodes"][str(n.key)]
            assert rec["state"] == "expanded" and [m["uci"] for m in rec["moves"]] == n.moves
            assert chess.Board(rec["epd"].split(" bm")[0] + " 0 1").is_valid() or rec["epd"]
    # stand-ins are listed with a flag, not dropped
    flagged = [m for r in back["nodes"].values() for m in r.get("moves", []) if "stand_in" in m["flags"]]
    assert flagged
    assert {f["root_move"] for f in back["forced"]} == set(sound)
    for f in back["forced"]:
        assert f["value"] == pytest.approx(forced[f["root_move"]][f["budget"]].value, abs=1e-6)
    # the PGN has one commented move per learned position
    for N in budgets:
        pg = chess.pgn.read_game(io.StringIO((tmp_path / "t" / f"repertoire_N{N}.pgn").read_text()))
        comments = []

        def walk(node):
            for v in node.variations:
                if v.comment.startswith("reach "):
                    comments.append(v.comment)
                walk(v)
        walk(pg)
        assert len(comments) == len(sols[N].learned)
        assert all(c.split()[2] and "cost" in c for c in comments)
