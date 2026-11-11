"""Lazy growth on real chess positions with fake models: transpositions, caching,
counted calls, database counts, and the PGN that comes out."""

import io
import zlib

import chess
import chess.pgn
import numpy as np
import polars as pl
import pytest

from tnh.builder.counts import DictCounts, EdgeDBCounts
from tnh.builder.expand import Expander, frontier_importance, pos_key
from tnh.builder.game import Game, Params
from tnh.builder.output import pgn, rows, table
from tnh.builder.search import solve
from tnh.models.base import MoveValues, Prediction


def _h(s: str) -> float:
    return zlib.crc32(s.encode()) / 2**32


class FakeMoves:
    """Logits: a preference per move plus a fixed pseudo-random part; value head from material."""

    name = "fake-moves"

    def __init__(self, prefer: dict[str, float] | None = None) -> None:
        self.prefer = prefer or {}
        self.queries = 0

    def predict(self, queries):
        out = []
        for q in queries:
            self.queries += 1
            b = chess.Board(q.fen)
            moves = [m.uci() for m in b.legal_moves]
            z = np.array([self.prefer.get(m, 0.0) + _h(q.fen + m) for m in moves])
            logp = z - np.log(np.exp(z - z.max()).sum()) - z.max()
            mat = sum(len(b.pieces(pt, b.turn)) * v - len(b.pieces(pt, not b.turn)) * v
                      for pt, v in [(1, 1), (2, 3), (3, 3), (4, 5), (5, 9)])
            s = 1 / (1 + np.exp(-mat / 3))
            out.append(Prediction(moves, logp.astype(np.float32), (0.8 * s, 0.2, 0.8 * (1 - s))))
        return out


class FakeEngine:
    name = "fake-engine"

    def __init__(self) -> None:
        self.queries = 0

    def evaluate(self, queries):
        out = []
        for q in queries:
            self.queries += 1
            b = chess.Board(q.fen)
            moves = [m.uci() for m in b.legal_moves]
            v = np.array([0.5 - 0.2 * _h("e" + q.fen + m) for m in moves], dtype=np.float32)
            out.append(MoveValues(moves, v, np.zeros(len(moves), np.float32), np.zeros(len(moves), np.int16), None))
        return out


QUEENS_PAWN = {"g1f3": 6.0, "d2d4": 6.0, "g8f6": 6.0}


def test_transpositions_are_one_node_and_calls_are_cached():
    mm, eng = FakeMoves(QUEENS_PAWN), FakeEngine()
    # eps 0: only the engine's best counts as reachable by learning, so growth follows
    # the model's own moves (with a wider eps every sound move has reach bound 1)
    ex = Expander(mm, eng, Params(eps_white=0.0, min_reach=0.05), me_white=True, me_elo=2000, opp_elo=2000)
    ex.grow(expand_reach=0.01, max_expansions=60)
    b = chess.Board()
    for m in ("g1f3", "g8f6", "d2d4"):
        b.push_uci(m)
    node = ex.graph.get(pos_key(b), 3)
    assert node is not None
    parents = [n for n in ex.graph.nodes if n.children and node.id in n.children]
    assert len(parents) == 2                      # 1.Nf3 Nf6 2.d4 and 1.d4 Nf6 2.Nf3
    assert ex.stats["transpositions merged"] > 0
    keys = {n.key for n in ex.graph.nodes}
    assert ex.stats["move model calls"] + ex.stats["game over"] == len(keys) == mm.queries
    assert ex.stats["engine calls"] == eng.queries
    # with lam = 0 the engine is asked only at my nodes in the budget zone, once per position
    asked = {n.key for n in ex.graph.nodes if n.engine is not None}
    assert all(n.me_to_move for n in ex.graph.nodes if n.engine is not None)
    assert eng.queries == len(asked)
    assert all(ex.bound[n.id] >= 0.05 for n in ex.graph.nodes if n.engine is not None)


def test_pessimism_needs_the_engine_at_the_opponents_nodes():
    """Per node: at every expanded opponent node. Per opponent: only on the best-reply
    spine, where the engine type can still be."""
    calls = {}
    for mode in ("node", "opponent"):
        eng = FakeEngine()
        params = Params(eps_white=0.05, min_reach=0.05, **({"lam_node": 0.2} if mode == "node" else {"lam": 0.2}))
        ex = Expander(FakeMoves(QUEENS_PAWN), eng, params,
                      me_white=True, me_elo=2000, opp_elo=2000)
        ex.grow(expand_reach=0.01, max_expansions=30)
        opp_expanded = [n for n in ex.graph.nodes if n.expanded and not n.me_to_move]
        if mode == "node":
            assert all(n.engine is not None for n in opp_expanded)
        else:
            game = Game(ex.graph, ex.params)
            spine = [n for n in opp_expanded if game.beta.get(n.id, 0) > 0]
            assert spine and len(spine) < len(opp_expanded)
            assert all(n.engine is not None for n in spine)
        calls[mode] = sum(1 for n in opp_expanded if n.engine is not None)
        Game(ex.graph, ex.params).evaluate({})
    assert calls["opponent"] < calls["node"]


def test_end_to_end_black_after_e4_gives_a_pgn_with_the_learned_moves():
    mm = FakeMoves({"e7e5": 3.0, "g1f3": 3.0, "b8c6": 3.0, "f1b5": 2.0, "c7c5": 2.5})
    ex = Expander(mm, FakeEngine(), Params(eps_black=0.1, min_reach=0.02), me_white=False,
                  me_elo=2000, opp_elo=2000, root_moves=["e2e4"])
    ex.grow(expand_reach=0.005, max_expansions=80)
    game = Game(ex.graph, ex.params)
    _, sols = solve(game, 4)
    sol = sols[4]
    assert 0 < sol.cost <= 4
    assert sols[4].value >= sols[0].value
    rs = rows(game, sols, 4)
    text = pgn(rs, ["e2e4"], {"White": "opponent 2000", "Black": "me 2000"})
    parsed = chess.pgn.read_game(io.StringIO(text))
    found = set()

    def walk(n, board_moves):
        for v in n.variations:
            if v.comment.startswith("learn:"):
                found.add((tuple(board_moves), v.move.uci()))
            walk(v, board_moves + [v.move.uci()])
    walk(parsed, [])
    want = {(("e2e4",) + tuple(r.path), r.move) for r in rs}
    assert found == want
    root_board = chess.Board()
    root_board.push_uci("e2e4")
    assert "| 1 |" in table(rs, root_board)
    imp = frontier_importance(game, sol.learned)
    assert imp and all(not ex.graph.nodes[i].expanded for i, _ in imp)
    assert imp[0][1] >= imp[-1][1]
    # refining where the solution weighs most expands those positions
    n0 = ex.stats["expanded"]
    ex.refine(sol.learned, n=5)
    assert ex.stats["expanded"] == n0 + 5
    assert all(ex.graph.nodes[i].children is not None for i, _ in imp[:5])


def _games(rows_):
    return pl.DataFrame(rows_, schema={"moves_uci": pl.String, "result": pl.String, "year": pl.Int32,
                                       "white_elo": pl.Int32, "black_elo": pl.Int32, "tempo": pl.String},
                        orient="row")


def test_counts_from_the_position_statistics():
    edges = pytest.importorskip("tnh.data.edges")
    games = _games([("e2e4 e7e5 g1f3", "1-0", 2020, 2000, 2000, "classical")] * 25
                   + [("e2e4 c7c5 g1f3", "0-1", 2020, 2000, 2000, "classical")] * 5
                   + [("e2e4 e7e6 d2d4", "0-1", 2020, 1500, 1500, "classical")] * 40)
    db = edges.EdgeDB.from_frames(*edges.tables_from_games(games))
    b = chess.Board()
    b.push_uci("e2e4")
    moves = [m.uci() for m in b.legal_moves]
    at_level = EdgeDBCounts(db, elo_bands=(1800, 2200))
    c = at_level.counts(pos_key(b), moves)
    assert c[moves.index("e7e5")] == 25 and c[moves.index("c7c5")] == 5 and c[moves.index("e7e6")] == 0
    everyone = EdgeDBCounts(db)
    assert everyone.counts(pos_key(b), moves)[moves.index("e7e6")] == 40
    b.push_uci("a7a6")                  # never played: the model alone decides
    assert at_level.counts(pos_key(b), [m.uci() for m in b.legal_moves]) is None
    # and the expander lays them over the model at the opponent's node
    ex = Expander(FakeMoves(), FakeEngine(), Params(kappa=10, eps_white=0.1), me_white=True,
                  me_elo=2000, opp_elo=2000, root_moves=["e2e4"], counts=at_level)
    root = ex.graph.nodes[ex.graph.root]
    assert not root.me_to_move and root.counts[root.moves.index("e7e5")] == 25
    ex.expand(root)
    q = Game(ex.graph, ex.params).dist[root.id]
    assert q[root.moves.index("e7e5")] > 0.6


def test_dict_counts_refuse_illegal_moves():
    with pytest.raises(ValueError, match="not legal"):
        DictCounts({1: {"e2e5": 3}}).counts(1, ["e2e4"])
