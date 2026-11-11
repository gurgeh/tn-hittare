"""The model interface from docs/DESIGN.md, plus the value-model shape used by engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import chess
import numpy as np


@dataclass
class Query:
    fen: str
    moves_uci: list[str]      # the game so far, from the start position, up to this position
    mover_elo: int
    opp_elo: int


@dataclass
class Prediction:
    moves: list[str]          # every legal move, UCI
    probs: np.ndarray         # float32, same order, sums to 1
    wdl: tuple[float, float, float] | None   # win, draw, loss for the mover


class MoveModel(Protocol):
    name: str

    def predict(self, queries: list[Query]) -> list[Prediction]: ...


@dataclass
class MoveValues:
    """Engine output: expected score for the mover after each legal move.

    Not in DESIGN.md's interface section, which only fixes the feature-file shape
    for engines (`value` in place of `logp`).
    """
    moves: list[str]          # every legal move, UCI
    values: np.ndarray        # float32, same order, expected score in [0, 1] for the mover; NaN if the engine gave none
    wdl: tuple[float, float, float] | None   # win, draw, loss for the mover after the engine's best move


class ValueModel(Protocol):
    name: str

    def evaluate(self, queries: list[Query]) -> list[MoveValues]: ...


class PositionMismatch(ValueError):
    """The moves in a query do not lead to the query's FEN."""


def replay(query: Query) -> list[chess.Board]:
    """Boards from the start of `query.moves_uci` to the query position, inclusive.

    With no moves the query FEN is the only board. With moves, the game starts from the
    standard start position, as DESIGN.md says, and the result must reach the query FEN.
    """
    if not query.moves_uci:
        return [chess.Board(query.fen)]
    board = chess.Board()
    boards = [board.copy(stack=False)]
    for uci in query.moves_uci:
        board.push(_legal(board, uci))
        boards.append(board.copy(stack=False))
    check_reaches(board, query.fen)
    return boards


def _legal(board: chess.Board, uci: str) -> chess.Move:
    move = chess.Move.from_uci(uci)
    if not board.is_legal(move):
        raise PositionMismatch(f"illegal move {uci} in {board.fen()}")
    return move


def position_key(board: chess.Board) -> str:
    """Pieces, side to move, castling and en passant (only when a capture is legal)."""
    return " ".join(board.fen(en_passant="legal").split()[:4])


def check_reaches(board: chess.Board, fen: str) -> None:
    """Same position as `fen`, ignoring move counters and the FEN writer's en passant convention.

    Written out field by field because it runs once per query and parsing the FEN into a
    second board costs more than everything else the Maia-3 encoder does.
    """
    placement, turn, castling, ep = fen.split(" ", 4)[:4]
    ok = (
        turn == ("w" if board.turn else "b")
        and castling == board.castling_xfen()
        and placement == board.board_fen()
    )
    if ok:
        if ep == "-":
            ok = not board.has_legal_en_passant()
        else:
            ok = board.ep_square is not None and ep == chess.square_name(board.ep_square)
    if not ok:
        raise PositionMismatch(f"moves lead to {board.fen()}, query says {fen}")


class Replayer:
    """Replays queries incrementally when they arrive in game order.

    Feature runs ask for every ply of a game in turn. Replaying each from the start
    is quadratic in game length; this keeps the last game's boards and extends them.
    """

    def __init__(self) -> None:
        self._moves: list[str] = []
        self._boards: list[chess.Board] = []

    def boards(self, query: Query) -> list[chess.Board]:
        """Boards from the start to the query position. Callers must not modify the list."""
        moves = query.moves_uci
        if not moves:
            return [chess.Board(query.fen)]
        n, k = len(self._moves), len(moves)
        if self._boards and k <= n and self._moves[:k] == moves:
            boards = self._boards[: k + 1]
        elif self._boards and k > n and moves[:n] == self._moves:
            board = self._boards[-1].copy(stack=False)
            for uci in moves[n:]:
                board.push(_legal(board, uci))
                self._boards.append(board.copy(stack=False))
            self._moves = list(moves)
            boards = self._boards
        else:
            self._boards = replay(Query(chess.STARTING_FEN, [], 0, 0))
            self._moves = []
            return self.boards(query)
        check_reaches(boards[-1], query.fen)
        return boards
