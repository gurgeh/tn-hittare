"""Stockfish value feature. Needs the stockfish binary (TNH_STOCKFISH or ~/.local/bin/stockfish)."""

import random

import chess
import numpy as np
import pytest

from tnh.models.base import Query
from tnh.models.stockfish import STOCKFISH, Stockfish, parse_final_multipv

needs_engine = pytest.mark.skipif(not STOCKFISH.exists(), reason=f"no Stockfish at {STOCKFISH}")


def q(fen, moves=()):
    return Query(fen, list(moves), 2000, 2000)


def test_parse_final_multipv_takes_the_last_report():
    lines = [
        "info depth 1 seldepth 2 multipv 1 score cp 30 wdl 60 900 40 nodes 20 nps 1 hashfull 0 tbhits 0 time 1 pv e2e4",
        "info depth 1 seldepth 2 multipv 2 score cp 10 wdl 50 900 50 nodes 20 nps 1 hashfull 0 tbhits 0 time 1 pv d2d4",
        "info depth 2 seldepth 3 multipv 1 score cp 25 wdl 55 905 40 nodes 40 nps 1 hashfull 0 tbhits 0 time 1 pv d2d4 d7d5",
        "info depth 2 seldepth 3 multipv 2 score cp 20 upperbound wdl 50 910 40 nodes 40 nps 1 hashfull 0 tbhits 0 time 1 pv e2e4",
        "info depth 1 seldepth 3 multipv 3 score mate -2 wdl 0 0 1000 nodes 40 nps 1 hashfull 0 tbhits 0 time 1 pv f2f3",
        "bestmove d2d4 ponder d7d5",
    ]
    r = parse_final_multipv(lines)
    assert list(r) == ["d2d4", "e2e4", "f2f3"]
    assert r["d2d4"]["wdl"] == (55, 905, 40) and r["d2d4"]["depth"] == 2
    assert r["e2e4"]["bound"] == "upperbound"
    assert r["f2f3"]["kind"] == "mate" and r["f2f3"]["score"] == -2


@needs_engine
def test_special_positions():
    with Stockfish(20_000, processes=2) as sf:
        mate1, stalemated, one_move, start = sf.evaluate([
            q("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"),   # Qxf7#
            q("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"),
            q("7k/8/8/8/8/8/6q1/7K w - - 0 1"),                                       # only Kxg2
            q(chess.STARTING_FEN),
        ])
        stats = dict(sf.stats)
    assert mate1.values[mate1.moves.index("h5f7")] == 1.0
    assert mate1.wdl == (1.0, 0.0, 0.0)
    assert stalemated.moves == [] and stalemated.wdl is None
    assert one_move.moves == ["h1g2"] and 0.0 <= one_move.values[0] <= 1.0
    assert sorted(start.moves) == sorted(m.uci() for m in chess.Board().legal_moves)
    assert not np.isnan(start.values).any() and not np.isnan(mate1.values).any()
    assert np.all((start.values >= 0) & (start.values <= 1))
    assert stats["no legal moves"] == 1 and stats["one legal move"] == 1


@needs_engine
def test_history_reaches_the_engine():
    """A move that repeats the position a third time is a draw only if the engine saw the game."""
    moves = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]   # f6g8 now repeats a third time
    b = chess.Board()
    for m in moves:
        b.push_uci(m)
    with Stockfish(5_000, processes=1) as sf:
        with_history, without = sf.evaluate([q(b.fen(), moves), q(b.fen())])
    i = with_history.moves.index("f6g8")
    assert with_history.values[i] == pytest.approx(0.5, abs=1e-6)   # draw by repetition: W 0, D 1000, L 0
    assert without.values[without.moves.index("f6g8")] != pytest.approx(0.5, abs=1e-6)   # the start position, about +0.2 for White


@needs_engine
def test_deterministic_across_processes_and_order():
    rng = random.Random(3)
    qs = []
    for _ in range(24):
        b, moves = chess.Board(), []
        for _ in range(rng.randint(10, 60)):
            legal = list(b.legal_moves)
            if not legal:
                break
            m = rng.choice(legal)
            b.push(m)
            moves.append(m.uci())
        qs.append(q(b.fen(), moves))
    with Stockfish(3_000, processes=4) as sf:
        a = sf.evaluate(qs)
        b_ = sf.evaluate(list(reversed(qs)))[::-1]
    with Stockfish(3_000, processes=1) as sf:
        c = sf.evaluate(qs)
    for x, y, z in zip(a, b_, c):
        assert x.moves == y.moves == z.moves
        np.testing.assert_array_equal(x.values, y.values)
        np.testing.assert_array_equal(x.values, z.values)
