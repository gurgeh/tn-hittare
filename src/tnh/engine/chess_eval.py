"""Chess positions and models for the engine.

A position is a light record (parent, move, key, ply, side to move); its board is
replayed from the nearest ancestor that has one when it is expanded, and kept only for
expanded positions, so a graph of a million mostly unexpanded positions fits in memory.
The query for a model carries the whole game from the start position along the
position's first path (Maia-3 reads the last eight positions).

Tempo is a named bundle of (move model, value, database filter), so a fitted stack can
replace one:

- `classical`: `tnh.models.default_move_model()` (Maia-3 79M at +0 and +600, Lc0 BT4's
  policy, boosted trees; FIDE ratings) for both sides at the mover's rating, the cheap
  result model from the same three network calls as the value, Lc0's raw policy from
  the same policy call for my prior. Database: classical games, all ratings and years,
  the filter `tnh.movemix` was fitted with.
- `blitz`: Maia-3 79M at the given ratings with no offset and no trees (ratings read as
  Lichess blitz ratings), its own value head uncalibrated (it was trained on blitz
  results), no Lc0 policy. Database: blitz and rapid games. Maia on its home ground;
  step 1 measured nothing better for blitz.
- `fake`: deterministic stand-ins for tests (no network).
"""

from __future__ import annotations

import zlib
from collections.abc import Hashable

import chess
import chess.polyglot
import numpy as np

from tnh.engine.search import Eval
from tnh.models.base import Query


class Pos:
    __slots__ = ("parent", "move", "key", "ply", "white", "board")

    def __init__(self, parent, move, key, ply, white, board=None):
        self.parent, self.move, self.key, self.ply, self.white, self.board = parent, move, key, ply, white, board

    def path(self) -> list[str]:
        out, p = [], self
        while p.parent is not None:
            out.append(p.move)
            p = p.parent
        return out[::-1]


DB_FILTERS = {
    "classical": {"tempi": "classical"},
    "blitz": {"tempi": ["blitz", "rapid"]},
    "all": {},
    "off": None,
}


class ChessEvaluator:
    """`bundles`: tempo -> an object with `evaluate(queries) -> list[Eval]` and a `name`;
    a tempo not given is loaded on first use by `bundle_for`. `db` is an EdgeDB or None."""

    def __init__(self, bundles: dict | None = None, db=None, start_fen: str | None = None) -> None:
        self.bundles = dict(bundles or {})
        self.db = db
        b = chess.Board(start_fen) if start_fen else chess.Board()
        self.start_fen = None if start_fen is None else b.fen()
        self._start = Pos(None, None, chess.polyglot.zobrist_hash(b), 0, b.turn == chess.WHITE, b)

    # -- positions --------------------------------------------------------------------------

    def start(self) -> Pos:
        return self._start

    def key(self, pos: Pos) -> Hashable:
        return pos.key

    def ply(self, pos: Pos) -> int:
        return pos.ply

    def white_to_move(self, pos: Pos) -> bool:
        return pos.white

    def board(self, pos: Pos) -> chess.Board:
        if pos.board is not None:
            return pos.board
        stack, p = [], pos
        while p.board is None:
            stack.append(p.move)
            p = p.parent
        b = p.board.copy(stack=False)
        for m in reversed(stack):
            b.push_uci(m)
        return b

    def moves(self, pos: Pos) -> list[str]:
        b = self.board(pos)
        pos.board = b                     # expanded positions keep their board
        return [m.uci() for m in b.legal_moves]

    def child(self, pos: Pos, move: str) -> Pos:
        b = self.board(pos).copy(stack=False)
        b.push_uci(move)
        return Pos(pos, move, chess.polyglot.zobrist_hash(b), pos.ply + 1, not pos.white)

    def terminal(self, pos: Pos) -> float | None:
        b = self.board(pos)
        pos.board = b
        if b.is_checkmate():
            return 0.0
        if b.is_stalemate() or b.is_insufficient_material():
            return 0.5
        return None

    def query(self, pos: Pos, mover_elo: int, opp_elo: int) -> Query:
        b = self.board(pos)
        moves = pos.path()
        if self.start_fen is not None:
            return Query(b.fen(), [], mover_elo, opp_elo)      # a FEN root has no game before it
        return Query(b.fen(), moves, mover_elo, opp_elo)

    # -- models -----------------------------------------------------------------------------

    def bundle(self, tempo: str):
        if tempo not in self.bundles:
            self.bundles[tempo] = bundle_for(tempo)
        return self.bundles[tempo]

    def evaluate(self, items, tempo: str) -> list[Eval]:
        return self.bundle(tempo).evaluate([self.query(p, me, op) for p, me, op in items])

    def counts_many(self, items, db_filter) -> list[np.ndarray | None]:
        filt = DB_FILTERS[db_filter] if isinstance(db_filter, str) else db_filter
        if self.db is None or filt is None:
            return [None] * len(items)
        df = self.db.counts_many([p.key for p, _, _ in items], **filt)
        by: dict[int, dict[str, int]] = {}
        for h, m, n in df.iter_rows():
            by.setdefault(int(h), {})[m] = int(n)
        out = []
        for p, moves, _ in items:
            row = by.get(int(p.key))
            out.append(None if row is None else np.array([row.get(m, 0) for m in moves], dtype=np.float64))
        return out

    def counts(self, pos, moves, mover_elo, db_filter):
        return self.counts_many([(pos, moves, mover_elo)], db_filter)[0]


# -- bundles --------------------------------------------------------------------------------------

class ClassicalBundle:
    """The default move model with Lc0's raw policy and the cheap result value from one pass."""

    def __init__(self, model=None) -> None:
        if model is None:
            from tnh.models import default_move_model
            model = default_move_model()
        self.model = model
        self.name = f"classical:{model.name}"

    def evaluate(self, queries: list[Query]) -> list[Eval]:
        from tnh.eval.result import normalised
        preds, (m0, m6, pol) = self.model.moves.predict_with_calls(queries)
        live = [i for i, p in enumerate(preds) if p.moves]
        vals = {}
        if live:
            w = normalised({"maia0": [m0[i].wdl for i in live], "maia600": [m6[i].wdl for i in live],
                            "lc0raw": [pol[i].wdl for i in live]})
            r = self.model.result.from_sources(w, [queries[i] for i in live])
            vals = {i: float(row[0] + 0.5 * row[1]) for i, row in zip(live, r)}
            wdls = {i: (float(row[0]), float(row[1]), float(row[2])) for i, row in zip(live, r)}
        else:
            wdls = {}
        out = []
        for i, pr in enumerate(preds):
            p = np.exp(pr.logp.astype(np.float64))
            idx = {m: j for j, m in enumerate(pol[i].moves)}
            lp = np.exp(pol[i].logp.astype(np.float64))
            policy = np.array([lp[idx[m]] for m in pr.moves]) if set(idx) == set(pr.moves) else None
            out.append(Eval(list(pr.moves), p / p.sum(), policy, vals.get(i, 0.5), wdls.get(i)))
        return out


class BlitzBundle:
    """Maia-3 79M at the given ratings, no offset, no trees, its own value head."""

    def __init__(self, model=None) -> None:
        if model is None:
            from tnh.models.maia3 import Elo, Maia3
            model = Maia3("79m", Elo.offset(0.0))
        self.model = model
        self.name = f"blitz:{model.name}"

    def evaluate(self, queries: list[Query]) -> list[Eval]:
        out = []
        for pr in self.model.predict(queries):
            p = np.exp(pr.logp.astype(np.float64))
            w, d, l = pr.wdl
            out.append(Eval(list(pr.moves), p / p.sum(), None, float(w + 0.5 * d), (float(w), float(d), float(l))))
        return out


def _h(s: str) -> float:
    return zlib.crc32(s.encode()) / 2 ** 32


class FakeBundle:
    """Deterministic fake models on real chess positions: logits from a hash of the position
    and move plus a preference table, value from material, a separate hashed 'policy'."""

    name = "fake"

    def __init__(self, prefer: dict[str, float] | None = None) -> None:
        self.prefer = prefer or {}
        self.calls = 0

    def evaluate(self, queries: list[Query]) -> list[Eval]:
        out = []
        for q in queries:
            self.calls += 1
            b = chess.Board(q.fen)
            moves = [m.uci() for m in b.legal_moves]
            elo = (q.mover_elo - 2000) / 1000
            z = np.array([self.prefer.get(m, 0.0) + 3 * _h(q.fen + m) + elo * _h("e" + m) for m in moves])
            p = np.exp(z - z.max())
            pol = np.exp(3 * np.array([_h("pol" + q.fen + m) for m in moves]))
            mat = sum(len(b.pieces(pt, b.turn)) * v - len(b.pieces(pt, not b.turn)) * v
                      for pt, v in [(1, 1), (2, 3), (3, 3), (4, 5), (5, 9)])
            val = 1 / (1 + np.exp(-mat / 3)) * 0.9 + 0.05 + 0.02 * (_h("v" + q.fen) - 0.5)
            d = 0.6 * (1 - abs(2 * val - 1)) * (0.3 + 0.7 * _h("d" + q.fen))   # some quiet, some sharp
            w = val - d / 2
            out.append(Eval(moves, p / p.sum(), pol / pol.sum(), float(val), (float(w), float(d), float(1 - w - d))))
        return out


def bundle_for(tempo: str):
    if tempo == "classical":
        return ClassicalBundle()
    if tempo == "blitz":
        return BlitzBundle()
    if tempo == "fake":
        return FakeBundle()
    raise ValueError(f"tempo must be classical, blitz or fake, got {tempo!r}")

