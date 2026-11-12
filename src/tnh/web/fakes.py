"""Stand-in models with the interfaces of model 1 and model 2, for tests and for looking at
the page while the GPU is taken. Their numbers mean nothing; the page labels them "fake".

    uv run python -m tnh.web --move-model fake --value-model fake
"""

from __future__ import annotations

import math

import chess
import numpy as np

from tnh.models.base import Prediction, Query, replay

_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
_CENTRE = {chess.D4, chess.E4, chess.D5, chess.E5, chess.C4, chess.F4, chess.C5, chess.F5}


class FakeMoveModel:
    """A `MoveModel`: captures, checks, central squares and developing moves are likelier."""

    name = "fake-move"

    def predict(self, queries: list[Query]) -> list[Prediction]:
        out = []
        for q in queries:
            board = replay(q)[-1]
            moves = list(board.legal_moves)
            w = np.array([self._weight(board, m) for m in moves], dtype=np.float64)
            logp = np.log(w / w.sum()) if len(moves) else np.zeros(0)
            out.append(Prediction([m.uci() for m in moves], logp.astype(np.float32), None))
        return out

    @staticmethod
    def _weight(board: chess.Board, m: chess.Move) -> float:
        w = 1.0
        if board.is_capture(m):
            w += 2.0
        if board.gives_check(m):
            w += 1.0
        if m.to_square in _CENTRE:
            w += 1.5
        piece = board.piece_type_at(m.from_square)
        if piece in (chess.KNIGHT, chess.BISHOP) and chess.square_rank(m.from_square) in (0, 7):
            w += 1.5
        return w ** 2


class FakeValue:
    """`.wdl(queries)`: material for the mover and the rating difference through a logistic,
    with a draw share that is largest at equal chances."""

    name = "fake-value"

    def wdl(self, queries: list[Query]) -> np.ndarray:
        out = np.zeros((len(queries), 3))
        for i, q in enumerate(queries):
            board = chess.Board(q.fen)
            mat = sum(_VALUE[p.piece_type] * (1 if p.color == board.turn else -1) for p in board.piece_map().values())
            e = 1 / (1 + math.exp(-(0.35 * mat + (q.mover_elo - q.opp_elo) / 400 * math.log(10))))
            d = 0.45 * (1 - abs(2 * e - 1))
            win = max(0.0, e - d / 2)
            out[i] = (win, d, max(0.0, 1 - win - d))
        return out

    def expected_score(self, queries: list[Query]) -> np.ndarray:
        w = self.wdl(queries)
        return w[:, 0] + 0.5 * w[:, 1]
