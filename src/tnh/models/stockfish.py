"""Stockfish as a per-move value feature.

For each position: one search with `MultiPV` over every legal move and `UCI_ShowWDL`,
limited to a fixed number of nodes for the whole search. The value of a move is
the expected score for the mover after it, `(W + D/2) / 1000` from the WDL that
Stockfish prints for the line starting with that move. Mate scores carry WDL 1000/0/0
or 0/0/1000 and need no special case.

Determinism: one thread per engine, `ucinewgame` (which clears the hash and the
history tables) before every position, so a position's values do not depend on what
the same process searched before.

The position is sent as `position startpos moves ...` with the whole game, so the
engine sees repetitions and the fifty-move counter. No tablebases are configured.
"""

from __future__ import annotations

import os
import queue
import subprocess
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
import numpy as np

from tnh.models.base import MoveValues, Query, Replayer

STOCKFISH = Path(os.environ.get("TNH_STOCKFISH", Path.home() / ".local" / "bin" / "stockfish"))
MAX_MULTIPV = 256   # Stockfish's limit; the most legal moves in a chess position is 218


class EngineError(RuntimeError):
    pass


class _Engine:
    """One Stockfish process. Output is read in raw chunks and only the end of each search
    is split into lines: at 10k nodes Stockfish prints ~150 MultiPV lines per position,
    and reading them one by one in Python capped a pool of 16 engines at ~650 positions/s."""

    def __init__(self, path: Path, hash_mb: int) -> None:
        self.proc = subprocess.Popen(
            [str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
        )
        self._fd = self.proc.stdout.fileno()
        self._buf = b""
        self._send("uci")
        lines = self._until(b"uciok").decode().splitlines()
        self.id = next((l[len("id name "):] for l in lines if l.startswith("id name ")), "?")
        self._send(*(f"setoption name {name} value {value}" for name, value in
                     [("Threads", 1), ("Hash", hash_mb), ("UCI_ShowWDL", "true"), ("MultiPV", MAX_MULTIPV)]))
        self._send("isready")
        self._until(b"readyok")

    def _send(self, *lines: str) -> None:
        self.proc.stdin.write(("\n".join(lines) + "\n").encode())

    def _until(self, token: bytes) -> bytes:
        """Everything up to and including the line that starts with `token`."""
        marker = b"\n" + token
        while True:
            i = (b"\n" + self._buf).find(marker)
            if i >= 0:
                j = self._buf.find(b"\n", i)
                if j >= 0:
                    out, self._buf = self._buf[: j + 1], self._buf[j + 1 :]
                    if b"Unknown command" in out or b"info string ERROR" in out:
                        bad = next(l for l in out.decode().splitlines() if "Unknown command" in l or "ERROR" in l)
                        raise EngineError(bad)
                    return out
            chunk = os.read(self._fd, 1 << 16)
            if not chunk:
                raise EngineError(f"engine exited (code {self.proc.wait()}): {self.proc.stderr.read()[-2000:]!r}")
            self._buf += chunk

    def search(self, position: str, nodes: int, clear: bool = True) -> list[str]:
        """The output lines from the last `multipv 1` report to `bestmove`."""
        self._send(*(["ucinewgame"] if clear else []), position, f"go nodes {nodes}")
        out = self._until(b"bestmove")
        k = out.rfind(b" multipv 1 ")
        start = out.rfind(b"\n", 0, k) + 1 if k >= 0 else out.rfind(b"\n", 0, len(out) - 1) + 1
        return out[start:].decode().splitlines()

    def close(self) -> None:
        if self.proc.poll() is None:
            self._send("quit")
            self.proc.wait(timeout=10)


def parse_final_multipv(lines: list[str]) -> dict[str, dict]:
    """The last full MultiPV report before `bestmove`: first move -> score, bound, wdl, depth.

    Stockfish prints all lines of a MultiPV report together, starting with `multipv 1`,
    and prints one more report when the node limit stops the search.
    """
    start = max((i for i, l in enumerate(lines) if l.startswith("info") and " multipv 1 " in l), default=None)
    if start is None:
        return {}
    out = {}
    for line in lines[start:]:
        if not line.startswith("info") or " multipv " not in line:
            continue
        t = line.split()
        if "pv" not in t:
            raise EngineError(f"MultiPV line without a pv: {line}")
        move = t[t.index("pv") + 1]
        s = t.index("score")
        w, d, l = (int(x) for x in t[t.index("wdl") + 1 : t.index("wdl") + 4])
        if w + d + l != 1000:
            raise EngineError(f"WDL does not sum to 1000: {line}")
        out[move] = {
            "kind": t[s + 1],                       # cp or mate
            "score": int(t[s + 2]),
            "bound": t[s + 3] if t[s + 3] in ("lowerbound", "upperbound") else None,
            "wdl": (w, d, l),
            "depth": int(t[t.index("depth") + 1]),
        }
    return out


class Stockfish:
    """Per-move expected score at a fixed node budget, over a pool of one-thread engines."""

    name = "stockfish"

    def __init__(self, nodes: int, *, processes: int | None = None, hash_mb: int = 16,
                 per_move: bool = False, deterministic: bool = True, path: Path = STOCKFISH) -> None:
        """`nodes` is the budget for the whole MultiPV search of a position, or with
        `per_move` the budget per legal move (the search gets `nodes * n_legal`).

        `deterministic=False` skips `ucinewgame` between positions. That saves about 7 ms
        per position (Stockfish 19 clears ~16 MB of history tables), but then a
        position's values depend on what the same process searched before it.
        """
        self.nodes = nodes
        self.per_move = per_move
        self.deterministic = deterministic
        self.hash_mb = hash_mb
        self.processes = processes or os.cpu_count() or 1
        self._engines: queue.Queue[_Engine] = queue.Queue()
        engines = [_Engine(path, hash_mb) for _ in range(self.processes)]
        for e in engines:
            self._engines.put(e)
        self._all = engines
        self.engine_id = engines[0].id
        self._replayer = Replayer()
        self._pool = ThreadPoolExecutor(self.processes)
        self.stats: Counter = Counter()
        self._lock = threading.Lock()

    @property
    def variant(self) -> str:
        v = f"nodes{self.nodes}{'-per-move' if self.per_move else ''}-hash{self.hash_mb}"
        return v if self.deterministic else v + "-noclear"

    def evaluate(self, queries: list[Query]) -> list[MoveValues]:
        jobs = []
        for q in queries:
            board = self._replayer.boards(q)[-1]   # checks that the moves reach the FEN
            legal = [m.uci() for m in board.legal_moves]
            if q.moves_uci:
                position = "position startpos moves " + " ".join(q.moves_uci)
            else:
                position = "position fen " + q.fen
            jobs.append((position, legal))
        return list(self._pool.map(self._one, jobs))

    def _one(self, job) -> MoveValues:
        position, legal = job
        if not legal:
            with self._lock:
                self.stats["no legal moves"] += 1
            return MoveValues([], np.zeros(0, dtype=np.float32), None)
        engine = self._engines.get()
        try:
            lines = engine.search(position, self.nodes * len(legal) if self.per_move else self.nodes,
                                  clear=self.deterministic)
        finally:
            self._engines.put(engine)
        lines_by_move = parse_final_multipv(lines)
        extra = set(lines_by_move) - set(legal)
        if extra:
            raise EngineError(f"engine reported moves that are not legal: {sorted(extra)} after {position}")
        values = np.full(len(legal), np.nan, dtype=np.float32)
        for i, mv in enumerate(legal):
            r = lines_by_move.get(mv)
            if r is not None:
                w, d, _l = r["wdl"]
                values[i] = (w + 0.5 * d) / 1000.0
        best = None
        if lines_by_move:
            first = next(iter(lines_by_move.values()))
            best = tuple(x / 1000.0 for x in first["wdl"])
        with self._lock:
            self.stats["positions"] += 1
            self.stats["moves"] += len(legal)
            self.stats["moves without a value"] += int(np.isnan(values).sum())
            self.stats["positions with a move without a value"] += int(np.isnan(values).any())
            self.stats["bound scores"] += sum(r["bound"] is not None for r in lines_by_move.values())
            self.stats["mate scores"] += sum(r["kind"] == "mate" for r in lines_by_move.values())
            self.stats["one legal move"] += len(legal) == 1
        return MoveValues(legal, values, best)

    def close(self) -> None:
        self._pool.shutdown()
        for e in self._all:
            e.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
