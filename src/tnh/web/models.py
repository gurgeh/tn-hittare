"""The model panel: Maia-3 and Stockfish, each loaded once on first use and kept.

Both run on the CPU by default. Results are cached per position (and per ratings or node
budget), so stepping back and forth costs nothing after the first visit. A model that
fails to load or run raises; the server turns that into an error the page shows.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol

import chess
import numpy as np

from tnh.models.base import Query, position_key


class ModelHub(Protocol):
    def maia(self, fen: str, moves: list[str], white_elo: int, black_elo: int) -> dict: ...   # FIDE ratings
    def engine(self, fen: str, moves: list[str], nodes: int) -> dict: ...
    def status(self) -> dict: ...


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


# Maia-3 was trained on Lichess blitz ratings. The rating it is given for a FIDE-rated
# player is this linear map, measured in step 1 (2026-11-12) as the best Maia rating by
# FIDE rating on classical games. Applied to both players.
MAIA_INTERCEPT = 1580.0
MAIA_SLOPE = 0.558


def maia_elo(fide: float) -> int:
    """The rating fed to Maia-3 for a player with this FIDE rating."""
    return int(round(MAIA_INTERCEPT + MAIA_SLOPE * fide))


def _num(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


@dataclass
class Hub:
    """Maia-3 at `maia_size` and Stockfish with one process, created lazily."""

    maia_size: str = "5m"
    device: str = "cpu"
    torch_threads: int = 2
    stockfish_processes: int = 1
    cache_size: int = 4096

    def __post_init__(self) -> None:
        self._maia = None
        self._sf = None
        self._maia_lock = threading.Lock()
        self._sf_lock = threading.Lock()
        self._maia_cache = _LRU(self.cache_size)
        self._sf_cache = _LRU(self.cache_size)
        self.load_s: dict[str, float] = {}

    # Maia-3 ------------------------------------------------------------------

    def _get_maia(self):
        if self._maia is None:
            t = time.perf_counter()
            import torch
            from tnh.models.maia3 import Maia3
            torch.set_num_threads(self.torch_threads)
            self._maia = Maia3(self.maia_size, device=self.device, batch_size=1, local_files_only=True)
            self.load_s["maia"] = time.perf_counter() - t
        return self._maia

    def maia(self, fen: str, moves: list[str], white_elo: int, black_elo: int) -> dict:
        """Maia-3 for the side to move. `white_elo` and `black_elo` are FIDE ratings; the
        network gets `maia_elo` of each."""
        from tnh.models.maia3 import Elo
        board = _board(fen, moves)
        mover_fide, opp_fide = (white_elo, black_elo) if board.turn == chess.WHITE else (black_elo, white_elo)
        mover, opp = maia_elo(mover_fide), maia_elo(opp_fide)
        # Maia sees the last 8 positions, so the key is the position plus the last 7 moves.
        key = (position_key(board), tuple(moves[-7:]), len(moves) < 7, mover, opp)
        hit = self._maia_cache.get(key)
        if hit is not None:
            return {**hit, "cached": True}
        with self._maia_lock:
            model = self._get_maia()
            t = time.perf_counter()
            q = Query(board.fen(), list(moves), mover, opp)
            pred = model.predict_many([q], [Elo.fixed(mover, opp)])[0][0]
            ms = (time.perf_counter() - t) * 1000
        p = np.exp(pred.logp.astype(np.float64))
        win, draw, loss = pred.wdl
        out = {
            "model": f"Maia-3 {self.maia_size}, blitz scale",
            "mover_fide": mover_fide, "opp_fide": opp_fide, "mover_maia": mover, "opp_maia": opp,
            "history": len(moves),
            "moves": {m: float(x) for m, x in zip(pred.moves, p)},
            "wdl": [win, draw, loss],
            "ms": ms,
        }
        self._maia_cache.put(key, out)
        return {**out, "cached": False}

    # Stockfish ------------------------------------------------------------------

    def _get_sf(self, nodes: int):
        if self._sf is None:
            t = time.perf_counter()
            from tnh.models.stockfish import Stockfish
            self._sf = Stockfish(nodes, processes=self.stockfish_processes)
            self.load_s["stockfish"] = time.perf_counter() - t
        return self._sf

    def engine(self, fen: str, moves: list[str], nodes: int) -> dict:
        board = _board(fen, moves)
        key = (position_key(board), nodes)
        hit = self._sf_cache.get(key)
        if hit is not None:
            return {**hit, "cached": True}
        with self._sf_lock:
            sf = self._get_sf(nodes)
            sf.nodes = nodes   # read per search; the lock keeps it per call
            t = time.perf_counter()
            mv = sf.evaluate([Query(board.fen(), list(moves), 0, 0)])[0]
            ms = (time.perf_counter() - t) * 1000
        out = {
            "model": f"{sf.engine_id}, {nodes} nodes, MultiPV all",
            "nodes": nodes,
            "moves": {m: {"value": _num(v), "cp": _num(c), "mate": int(k)}
                      for m, v, c, k in zip(mv.moves, mv.value, mv.cp, mv.mate)},
            "wdl": list(mv.wdl) if mv.wdl else None,
            "ms": ms,
        }
        self._sf_cache.put(key, out)
        return {**out, "cached": False}

    def status(self) -> dict:
        return {"maia_loaded": self._maia is not None, "stockfish_loaded": self._sf is not None,
                "maia_size": self.maia_size, "device": self.device, "load_s": self.load_s}

    def close(self) -> None:
        if self._sf is not None:
            self._sf.close()
            self._sf = None


class NoModels:
    """The hub when the server runs with --no-models: every model call is refused."""

    def maia(self, fen, moves, white_elo, black_elo):
        raise ModelsOff("models are off (--no-models)")

    def engine(self, fen, moves, nodes):
        raise ModelsOff("models are off (--no-models)")

    def status(self) -> dict:
        return {"disabled": True}

    def close(self) -> None:
        pass


class ModelsOff(RuntimeError):
    pass
