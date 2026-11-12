"""Maia-3 input encoding against the reference code. CPU only, no weights needed."""

import random

import chess
import numpy as np
import pytest
from maia3 import uci as ref
from maia3.dataset import tokenize_board
from maia3.utils import get_all_possible_moves, mirror_move

from tnh.models.base import PositionMismatch, Query, Replayer, check_reaches
from tnh.models.maia3 import HISTORY, Elo, Encoder, frame, move_index


def random_games(n, seed=0, max_plies=150):
    rng = random.Random(seed)
    games = []
    for _ in range(n):
        b, moves = chess.Board(), []
        for _ in range(rng.randint(1, max_plies)):
            legal = list(b.legal_moves)
            if not legal:
                break
            # prefer promotions and captures now and then so that they occur
            special = [m for m in legal if m.promotion or b.is_en_passant(m) or b.is_castling(m)]
            m = rng.choice(special) if special and rng.random() < 0.5 else rng.choice(legal)
            b.push(m)
            moves.append(m.uci())
        games.append(moves)
    return games


def test_frame_and_move_index_match_reference():
    vocab = {m: i for i, m in enumerate(get_all_possible_moves())}
    seen = set()
    for moves in random_games(40):
        b = chess.Board()
        for uci in moves:
            assert np.array_equal(frame(b), tokenize_board(b).numpy().astype(np.uint8)), b.fen()
            for m in b.legal_moves:
                key = m.uci() if b.turn == chess.WHITE else mirror_move(m.uci())
                assert vocab[key] == move_index(b, m), (b.fen(), m.uci())
                if m.promotion:
                    seen.add("promotion")
                if b.is_castling(m):
                    seen.add("castling")
                if b.is_en_passant(m):
                    seen.add("en passant")
            b.push_uci(uci)
    assert seen == {"promotion", "castling", "en passant"}


def test_tokens_match_reference_uci_history():
    """Same 8-position window, order, padding and orientation as `maia3-uci --use-uci-history`."""
    cfg = ref.parse_args(["--model", "maia3-5m", "--use-uci-history"])   # resolves the preset, loads nothing
    engine = ref.Maia3UCIEngine(cfg)
    enc = Encoder()
    queries, expected = [], []
    for moves in random_games(15, seed=1):
        b = chess.Board()
        for ply in range(len(moves) + 1):
            engine.cmd_position("position startpos moves " + " ".join(moves[:ply]) if ply else "position startpos")
            expected.append(engine._tokens_from_history(engine.history).numpy()[:, : 12 * HISTORY])
            queries.append(Query(b.fen(), moves[:ply], 1500, 1500))
            if ply < len(moves):
                b.push_uci(moves[ply])
    tokens, _, _ = enc.encode(queries)
    for i, e in enumerate(expected):
        assert np.array_equal(tokens[i], e.astype(np.uint8)), (i, queries[i].fen, len(queries[i].moves_uci))


def test_query_without_moves_repeats_the_position():
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    tokens, moves, _ = Encoder().encode([Query(fen, [], 1500, 1500)])
    f = frame(chess.Board(fen))
    assert np.array_equal(tokens[0], np.concatenate([f] * HISTORY, axis=1))
    assert sorted(moves[0]) == sorted(m.uci() for m in chess.Board(fen).legal_moves)


def test_replayer_checks_the_fen_and_handles_en_passant_conventions():
    moves = ["e2e4", "a7a6", "e4e5", "d7d5"]   # exd6 e.p. is legal now
    b = chess.Board()
    for m in moves:
        b.push_uci(m)
    r = Replayer()
    assert r.boards(Query(b.fen(), moves, 0, 0))[-1].ep_square == chess.D6
    with pytest.raises(PositionMismatch):
        r.boards(Query(b.fen().replace(" d6 ", " - "), moves, 0, 0))
    # after 1.e4 no capture is possible: both conventions are accepted
    b1 = chess.Board()
    b1.push_uci("e2e4")
    check_reaches(b1, b1.fen(en_passant="fen"))
    check_reaches(b1, b1.fen(en_passant="legal"))
    with pytest.raises(PositionMismatch):
        r.boards(Query(chess.STARTING_FEN, ["e2e4"], 0, 0))
    # going back to an earlier ply of the same game, and to another game
    assert len(r.boards(Query(b1.fen(), ["e2e4"], 0, 0))) == 2
    assert len(r.boards(Query(b.fen(), moves, 0, 0))) == 5


def test_elo_settings_names_and_clamp():
    q = Query("", [], 2100, 4900)
    assert Elo.offset(300).of(q) == (2400.0, 5000.0)                 # the network's own clamp
    assert Elo.offset(300, -100).of(q) == (2400.0, 4800.0)
    assert Elo.fixed(1800, 2600).of(q) == (1800.0, 2600.0)
    assert Elo.fixed(3000, 2200, clamp=(600, 2600)).of(q) == (2600.0, 2200.0)
    assert Elo.mapped(1580, 0.558).of(Query("", [], 2200, 1800)) == (1580 + 0.558 * 2200, 1580 + 0.558 * 1800)
    names = {
        "offset+300": Elo.offset(300), "offset-200": Elo.offset(-200), "offset+0": Elo.offset(0),
        "offset+300-opp+100": Elo.offset(300, 100), "offset+0-opp-100": Elo.offset(0, -100),
        "fixed2200-2200": Elo.fixed(2200, 2200), "offset+300-clamp600-2600": Elo.offset(300, clamp=(600, 2600)),
        "map1580+0.558x": Elo.mapped(1580, 0.558),
    }
    for name, e in names.items():
        assert e.name == name and Elo.parse(name) == e
    with pytest.raises(ValueError):
        Elo(self_elo=2000, oppo_elo=2000, mover_offset=0, opp_offset=0)
    with pytest.raises(ValueError):
        Elo(mover_offset=100)
    with pytest.raises(ValueError):
        Elo.parse("offset300")
