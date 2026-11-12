"""The models behind the move table, each loaded once on first use and kept.

- **Model 1**, the move model: any object with the `MoveModel` interface
  (`.predict(queries) -> list[Prediction]`), by default `tnh.models.default_move_model()`.
- **Model 2**, the result model: any object with `.wdl(queries)` (an array of win, draw,
  loss for the mover per query), by default `tnh.eval.result.ResultModel("cheap")` (network
  calls only); `result-full` adds Lc0 at 50 nodes, `frontier` is step 1's value head.
- Engines (Stockfish, Lc0, preppy) are not here: `tnh.web.engine` runs them.

Models are named by a spec string (`load_move_model`, `load_value_model`): a preset, or
`import:package.module:Name` for any class built without arguments, so a better model drops
in by a flag. Both models take FIDE ratings in `Query.mover_elo` and `Query.opp_elo`.

Results are cached per position, its recent history and both ratings. A model that fails
to load or run raises; the server turns that into an error the page shows.
"""

from __future__ import annotations

import importlib
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Protocol

import chess
import numpy as np

from tnh.models.base import Query, position_key

HISTORY = 8          # positions of history a model may look at; part of the cache key
VALUE_BATCH = 12     # queries per model-2 call unless the model sets `queries_per_call`


class ModelHub(Protocol):
    def move_probs(self, fen: str, moves: list[str], white: int, black: int) -> dict: ...
    def values(self, queries: list[Query], cancelled: Callable[[], bool] = ...) -> list[tuple[float, float, float]]: ...
    def status(self) -> dict: ...
    def close(self) -> None: ...


class ModelsOff(RuntimeError):
    pass


class Superseded(RuntimeError):
    """A newer request for the table came in while this one waited for model 2."""


# --- model specs ------------------------------------------------------------------------

MOVE_MODELS: dict[str, Callable[[], object]] = {
    "policy-trees": lambda: importlib.import_module("tnh.models").default_move_model(),
    "combo-p0.5": lambda: _combo().Combo(prune="p0.5"),
    "combo": lambda: _combo().Combo(),
    "fake": lambda: _fakes().FakeMoveModel(),
}
SHARED = "shared"   # value model spec: the move model object itself, which must have `wdl(queries)`

VALUE_MODELS: dict[str, Callable[[], object]] = {
    "result-cheap": lambda: importlib.import_module("tnh.eval.result").ResultModel("cheap"),
    "result-full": lambda: importlib.import_module("tnh.eval.result").ResultModel("full"),
    "frontier": lambda: _combo().FrontierValue(),
    "fake": lambda: _fakes().FakeValue(),
}


def _combo():
    return importlib.import_module("tnh.models.combo")


def _fakes():
    return importlib.import_module("tnh.web.fakes")


def _load(spec: str, presets: dict[str, Callable[[], object]], kind: str):
    if spec in presets:
        return presets[spec]()
    if spec.startswith("import:"):
        module, _, name = spec[len("import:"):].partition(":")
        return getattr(importlib.import_module(module), name)()
    raise ValueError(f"{kind}: {spec!r} is neither a preset ({', '.join(presets)}) nor import:module:Name")


def load_move_model(spec: str):
    m = _load(spec, MOVE_MODELS, "move model")
    if not callable(getattr(m, "predict", None)):
        raise TypeError(f"move model {spec!r} has no predict(queries)")
    return m


def load_value_model(spec: str):
    m = _load(spec, VALUE_MODELS, "value model")
    if not callable(getattr(m, "wdl", None)):
        raise TypeError(f"value model {spec!r} has no wdl(queries)")
    return m


# --- helpers ---------------------------------------------------------------------------


class _LRU:
    def __init__(self, size: int) -> None:
        self.size = size
        self._d: OrderedDict = OrderedDict()

    def get(self, key):
        v = self._d.get(key)
        if v is not None:
            self._d.move_to_end(key)
        return v

    def put(self, key, value) -> None:
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self.size:
            self._d.popitem(last=False)


def _board(fen: str, moves: list[str]) -> chess.Board:
    """The position shown. With moves, they are played from the start position and must
    reach `fen`, as `tnh.models.base.Query` requires."""
    if not moves:
        return chess.Board(fen)
    b = chess.Board()
    for m in moves:
        b.push_uci(m)
    if position_key(b) != position_key(chess.Board(fen)):
        raise ValueError(f"moves lead to {b.fen()}, not {fen}")
    return b


def query(board: chess.Board, moves: list[str], white: int, black: int) -> Query:
    """The query for the side to move in `board`, with both FIDE ratings."""
    mover, opp = (white, black) if board.turn == chess.WHITE else (black, white)
    return Query(board.fen(), list(moves), mover, opp)


def _key(q: Query) -> tuple:
    return (position_key(chess.Board(q.fen)), tuple(q.moves_uci[-HISTORY:]), len(q.moves_uci) < HISTORY,
            q.mover_elo, q.opp_elo)


def _num(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def terminal_wdl(board: chess.Board) -> tuple[float, float, float] | None:
    """Win, draw, loss for the side to move where the game is over; None where it is not."""
    out = board.outcome(claim_draw=False)
    if out is None:
        return None
    if out.winner is None:
        return (0.0, 1.0, 0.0)
    return (1.0, 0.0, 0.0) if out.winner == board.turn else (0.0, 0.0, 1.0)


# --- the hub -----------------------------------------------------------------------------


class Hub:
    """Model 1 and model 2, created lazily from their specs (or given as objects)."""

    def __init__(self, move_model: str | object = "policy-trees", value_model: str | object = "result-cheap",
                 *, cache_size: int = 20000) -> None:
        self._move_spec, self._value_spec = move_model, value_model
        self._move = None if isinstance(move_model, str) else move_model
        self._value = None if isinstance(value_model, str) else value_model
        self._move_lock, self._value_lock = threading.Lock(), threading.Lock()
        if value_model == SHARED:
            # One object answers both (the default move model fills `wdl` from its own network
            # calls): one copy of the networks on the GPU, so one lock for both kinds of request.
            self._value_lock = self._move_lock
        self._move_cache, self._value_cache = _LRU(cache_size), _LRU(cache_size * 4)
        self.load_s: dict[str, float] = {}
        self.counts = {"value_queries": 0, "value_cached": 0, "value_terminal": 0, "move_queries": 0}

    @staticmethod
    def _name(model, spec) -> str:
        return getattr(model, "name", None) or (spec if isinstance(spec, str) else type(model).__name__)

    def _get_move(self):
        if self._move is None:
            t = time.perf_counter()
            self._move = load_move_model(self._move_spec)
            self.load_s["move_model"] = time.perf_counter() - t
        return self._move

    def _get_value(self):
        if self._value is None:
            t = time.perf_counter()
            if self._value_spec == SHARED:
                model = self._get_move()   # the caller holds the lock the two share
                if not callable(getattr(model, "wdl", None)):
                    raise TypeError(f"value model {SHARED!r}: the move model {self._name(model, self._move_spec)} has no wdl(queries)")
                self._value = model
            else:
                self._value = load_value_model(self._value_spec)
            self.load_s["value_model"] = time.perf_counter() - t
        return self._value

    def move_probs(self, fen: str, moves: list[str], white: int, black: int) -> dict:
        """Model 1 for the side to move: {uci: probability} over the legal moves."""
        board = _board(fen, moves)
        q = query(board, moves, white, black)
        key = _key(q)
        hit = self._move_cache.get(key)
        if hit is not None:
            return {**hit, "cached": True}
        with self._move_lock:
            model = self._get_move()
            t = time.perf_counter()
            pred = model.predict([q])[0]
            ms = (time.perf_counter() - t) * 1000
            self.counts["move_queries"] += 1
        p = np.exp(pred.logp.astype(np.float64))
        out = {"model": self._name(model, self._move_spec), "moves": {m: float(x) for m, x in zip(pred.moves, p)},
               "ms": ms}
        self._move_cache.put(key, out)
        return {**out, "cached": False}

    def values(self, queries: list[Query], cancelled: Callable[[], bool] = lambda: False) -> list[tuple]:
        """Model 2: (win, draw, loss) for the mover of each query. Cached ones are not asked
        again; finished games are scored by their result; the rest go in batches of
        the model's `queries_per_call` (default `VALUE_BATCH`), checking `cancelled` between
        batches (then `Superseded`)."""
        out: list[tuple | None] = [None] * len(queries)
        todo = []
        for i, q in enumerate(queries):
            t = terminal_wdl(chess.Board(q.fen))
            if t is not None:
                out[i] = t
                self.counts["value_terminal"] += 1
                continue
            hit = self._value_cache.get(_key(q))
            if hit is not None:
                out[i] = hit
                self.counts["value_cached"] += 1
            else:
                todo.append(i)
        if todo:
            with self._value_lock:
                model = self._get_value()
                batch = getattr(model, "queries_per_call", VALUE_BATCH)
                for start in range(0, len(todo), batch):
                    if cancelled():
                        raise Superseded(f"{len(todo) - start} of {len(todo)} values not computed")
                    idx = todo[start:start + batch]
                    # Another request may have filled some while this one waited for the lock.
                    need = [i for i in idx if self._value_cache.get(_key(queries[i])) is None]
                    if need:
                        w = np.asarray(model.wdl([queries[i] for i in need]), dtype=np.float64)
                        self.counts["value_queries"] += len(need)
                        for i, row in zip(need, w):
                            self._value_cache.put(_key(queries[i]), tuple(float(x) for x in row))
                    for i in idx:
                        out[i] = self._value_cache.get(_key(queries[i]))
        return out

    def value_name(self) -> str:
        return self._name(self._value, self._value_spec) if self._value is not None else str(self._value_spec)

    def status(self) -> dict:
        return {"move_model": self._name(self._move, self._move_spec) if self._move else str(self._move_spec),
                "value_model": self.value_name(),
                "move_loaded": self._move is not None, "value_loaded": self._value is not None,
                "load_s": self.load_s, "counts": self.counts}

    def close(self) -> None:
        for m in (self._move, self._value):
            if m is not None and callable(getattr(m, "close", None)):
                m.close()


class NoModels:
    """The hub when the server runs with --no-models: every model call is refused."""

    def move_probs(self, fen, moves, white, black):
        raise ModelsOff("models are off (--no-models)")

    def values(self, queries, cancelled=lambda: False):
        raise ModelsOff("models are off (--no-models)")

    def status(self) -> dict:
        return {"disabled": True}

    def close(self) -> None:
        pass
