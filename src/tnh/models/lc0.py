"""Leela Chess Zero: raw policy at one node, and values after a small search.

Lc0 is found through `TNH_LC0` if set, else `~/.local/opt/lc0/` (see docs/lc0.md):
`bin/lc0`, the Python bindings under `lib/python3.12/site-packages/lczero/`, and the
network under `networks/`.

- `Lc0Policy` (a `MoveModel`) evaluates positions in batches through the Python
  bindings and returns the raw policy (softmax at temperature 1) over the legal moves,
  with the value head's WDL for the mover.
- `Lc0Search` (a `ValueModel`) runs one search of N nodes per position through the
  UCI engine and returns, for each root move, its Q as an expected score for the mover.
  Moves the search did not visit get NaN. Optionally the search is calibrated to two
  players of given ratings with `WDLCalibrationElo` and `Contempt`, and the position's
  WDL is then the one for those players.

The bindings in Lc0 v0.32.1 map moves to policy indices without the board transform
that `as_input()` applies for canonical-input networks (input format 3 and above):
`GameState.policy_indices()` always uses no transform, while the encoder flips (and in
pawnless positions mirrors or transposes) the board when no castling rights are left.
BT4-1024x15x32h-swa-6147500 has the classical input format (1), so no transform is
ever applied and the bindings are right for it. For a canonical network this module
chooses the transform itself (a port of `ChooseTransform`, src/neural/encoder.cc) and
maps the moves with it; that path is checked against the engine only for BT4, i.e.
with the transform off.
"""

from __future__ import annotations

import glob
import importlib
import math
import os
import subprocess
import sys
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np

from tnh.models.base import MoveValues, Prediction, Query, Replayer

LC0_HOME = Path(os.environ.get("TNH_LC0", Path.home() / ".local" / "opt" / "lc0"))
DEFAULT_NETWORK = "BT4-1024x15x32h-swa-6147500.pb.gz"
# Lc0 v0.32.1's CUDA backend with BT4 gives non-deterministic results for batches of 16
# and more: the same batch evaluated twice differs by up to 0.2 in a policy logit in
# float16 and up to 5 in float32, a race, not rounding. Batches of up to 12 repeat
# exactly (float32 equals batch 1 exactly). Measured with
# scratch/models/scripts/lc0_stable.py; both the bindings and the engine stay at 12.
MAX_BATCH = 12
SHORT_NAMES = {"BT4-1024x15x32h-swa-6147500": "bt4", "t3-512x15x16h-distill2-swa-03477500": "t3"}


def binary(home: Path = LC0_HOME) -> Path:
    return home / "bin" / "lc0"


def network_path(name: str = DEFAULT_NETWORK, home: Path = LC0_HOME) -> Path:
    p = Path(name)
    return p if p.is_absolute() else home / "networks" / name


def short_name(network: Path) -> str:
    stem = network.name.removesuffix(".gz").removesuffix(".pb")
    return SHORT_NAMES.get(stem, stem.lower())


def backends(home: Path = LC0_HOME):
    """The `lczero.backends` extension module from the Lc0 install."""
    dirs = glob.glob(str(home / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"))
    dirs += glob.glob(str(home / "lib" / "*" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"))
    if not dirs:
        raise FileNotFoundError(f"no Lc0 Python bindings for Python {sys.version_info.major}.{sys.version_info.minor} "
                                f"under {home}/lib; see docs/lc0.md")
    if dirs[0] not in sys.path:
        sys.path.insert(0, dirs[0])
    return importlib.import_module("lczero.backends")


# -- the 1858-move policy layout --------------------------------------------------

def _square(sq: int) -> str:
    return "abcdefgh"[sq % 8] + str(sq // 8 + 1)


def _policy_moves() -> list[str]:
    """Lc0's `kMoveStrs`: for each from-square a1..h8, every queen or knight destination in
    square order; then the promotions from rank 7 by from-file, to-file, piece q r b.
    A move to rank 8 without a suffix is the knight promotion."""
    out = []
    for frm in range(64):
        f0, r0 = frm % 8, frm // 8
        for to in range(64):
            df, dr = abs(to % 8 - f0), abs(to // 8 - r0)
            if to != frm and (df == 0 or dr == 0 or df == dr or {df, dr} == {1, 2}):
                out.append(_square(frm) + _square(to))
    for f0 in range(8):
        for f1 in (f0 - 1, f0, f0 + 1):
            if 0 <= f1 < 8:
                out += [f"{'abcdefgh'[f0]}7{'abcdefgh'[f1]}8{p}" for p in "qrb"]
    return out


POLICY_MOVES = _policy_moves()
POLICY_INDEX = {m: i for i, m in enumerate(POLICY_MOVES)}
FLIP, MIRROR, TRANSPOSE = 1, 2, 4


def _reverse_bits_in_bytes(v: int) -> int:
    v = ((v >> 1) & 0x5555555555555555) | ((v & 0x5555555555555555) << 1)
    v = ((v >> 2) & 0x3333333333333333) | ((v & 0x3333333333333333) << 2)
    return ((v >> 4) & 0x0F0F0F0F0F0F0F0F) | ((v & 0x0F0F0F0F0F0F0F0F) << 4)


def _reverse_bytes(v: int) -> int:
    return int.from_bytes(v.to_bytes(8, "little"), "big")


def _transpose(v: int) -> int:
    m = 0xFFFFFFFFFFFFFFFF
    v = ((v & 0xAA00AA00AA00AA00) >> 9) | ((v & 0x0055005500550055) << 9) | (v & 0x55AA55AA55AA55AA)
    v = ((v & 0xCCCC0000CCCC0000) >> 18) | ((v & 0x0000333300003333) << 18) | (v & 0x3333CCCC3333CCCC)
    v = ((v & 0xF0F0F0F000000000) >> 36) | ((v & 0x000000000F0F0F0F) << 36) | (v & 0x0F0F0F0FF0F0F0F0)
    return v & m


def _compare_transposing(bb: int, transform: int) -> int:
    if transform & FLIP:
        bb = _reverse_bits_in_bytes(bb)
    if transform & MIRROR:
        bb = _reverse_bytes(bb)
    alt = _transpose(bb)
    return -1 if bb < alt else (1 if bb > alt else 0)


def _mover_bitboards(board: chess.Board) -> dict[str, int]:
    """Bitboards from the side to move's view, as Lc0 holds them (flipped when Black moves)."""
    us, them = board.turn, not board.turn
    f = (lambda b: chess.flip_vertical(b)) if us == chess.BLACK else (lambda b: b)
    return {
        "ours": f(board.occupied_co[us]), "theirs": f(board.occupied_co[them]),
        "kings": f(board.kings), "queens": f(board.queens), "rooks": f(board.rooks),
        "knights": f(board.knights), "bishops": f(board.bishops), "pawns": f(board.pawns),
        "our_king": f(board.kings & board.occupied_co[us]),
    }


def choose_transform(board: chess.Board) -> int:
    """Port of `ChooseTransform` (Lc0 v0.32.1, src/neural/encoder.cc)."""
    if board.castling_rights:
        return 0
    b = _mover_bitboards(board)
    king = b["our_king"]
    t = 0
    if king & 0x0F0F0F0F0F0F0F0F:
        t |= FLIP
        king = _reverse_bits_in_bytes(king)
    if b["pawns"]:
        return t
    if king & 0xFFFFFFFF00000000:
        t |= MIRROR
        king = _reverse_bytes(king)
    if king & 0xE0C08000:
        t |= TRANSPOSE
    elif king & 0x10204080:
        for key in ("all", "ours", "kings", "queens", "rooks", "knights", "bishops"):
            bb = b["ours"] | b["theirs"] if key == "all" else b[key]
            outcome = _compare_transposing(bb, t)
            if outcome == -1:
                return t
            if outcome == 1:
                return t | TRANSPOSE
    return t


def _transform_square(sq: int, t: int) -> int:
    file, rank = sq % 8, sq // 8
    if t & (MIRROR | TRANSPOSE):
        rank = 7 - rank
    if t & (FLIP | TRANSPOSE):
        file = 7 - file
    return rank * 8 + file


def policy_index(board: chess.Board, move: chess.Move, transform: int) -> int:
    """Port of `MoveToNNIndex`: the move from the mover's side, castling as king takes rook."""
    frm, to = move.from_square, move.to_square
    if board.is_castling(move):
        to = chess.square(7 if chess.square_file(to) > chess.square_file(frm) else 0, chess.square_rank(frm))
    if board.turn == chess.BLACK:
        frm, to = frm ^ 56, to ^ 56
    if transform:
        frm, to = _transform_square(frm, transform), _transform_square(to, transform)
    promo = move.promotion
    suffix = chess.piece_symbol(promo) if promo and promo != chess.KNIGHT else ""
    return POLICY_INDEX[_square(frm) + _square(to) + suffix]


# -- policy at one node --------------------------------------------------------------

class Lc0Policy:
    """Raw policy and value head of an Lc0 network, in batches through the Python bindings."""

    def __init__(self, network: str | Path = DEFAULT_NETWORK, *, backend: str = "cuda-auto",
                 batch_size: int = MAX_BATCH, home: Path = LC0_HOME) -> None:
        if batch_size > MAX_BATCH:
            raise ValueError(f"batch_size above the backend's max_batch {MAX_BATCH}")
        self.network = network_path(str(network), home)
        self._lc = backends(home)
        self._weights = self._lc.Weights(str(self.network))
        # max_batch also bounds the GPU workspace; the default (1024) ran out of memory on
        # the 8 GB card next to other models.
        self._backend = self._lc.Backend(weights=self._weights, backend=backend, options=f"max_batch={MAX_BATCH}")
        self.backend_name = backend
        self.input_format = self._backend.capabilities().input_format()
        self.canonical = self.input_format >= 3   # IsCanonicalFormat in src/neural/encoder.cc
        self.batch_size = batch_size
        self.name = f"lc0-{short_name(self.network)}"
        self.variant = "policy" + ("" if backend == "cuda-auto" else f"-{backend}")
        self._replayer = Replayer()
        self.transforms: Counter = Counter()

    def _state(self, q: Query):
        if q.moves_uci:
            return self._lc.GameState(moves=q.moves_uci)
        return self._lc.GameState(fen=q.fen)

    def predict(self, queries: list[Query]) -> list[Prediction]:
        out: list[Prediction] = []
        for start in range(0, len(queries), self.batch_size):
            batch = queries[start : start + self.batch_size]
            inputs, idxs, moves = [], [], []
            for q in batch:
                board = self._replayer.boards(q)[-1]
                legal = list(board.legal_moves)
                t = choose_transform(board) if self.canonical else 0
                self.transforms[t] += 1
                moves.append([m.uci() for m in legal])
                idxs.append([policy_index(board, m, t) for m in legal])
                inputs.append(self._state(q).as_input(self._backend))
            results = self._backend.evaluate(*inputs)
            for ms, ix, r in zip(moves, idxs, results):
                raw = np.array(r.p_raw(*ix), dtype=np.float64) if ix else np.zeros(0)
                logp = raw - (np.logaddexp.reduce(raw) if len(raw) else 0.0)
                qv, d = float(r.q()), float(r.d())
                out.append(Prediction(ms, logp.astype(np.float32), ((1 + qv - d) / 2, d, (1 - qv - d) / 2)))
        return out


# -- values after a small search -------------------------------------------------------

class EngineError(RuntimeError):
    pass


@dataclass
class EdgeStats:
    move: str
    n: int
    p: float          # policy prior, percent, after the engine's PolicyTemperature
    wl: float | None  # W - L for the mover, None when the move was not visited
    d: float | None
    q: float          # for an unvisited move this is the first-play urgency, not a value
    terminal: bool


def parse_verbose_stats(lines: list[str], board: chess.Board | None = None) -> tuple[list[EdgeStats], EdgeStats | None]:
    """`info string` lines of VerboseMoveStats: one per root move, then the root ("node").

    The engine writes castling as king takes rook (`e1h1`); with `board` the moves are
    returned in standard UCI.
    """
    edges, root = [], None
    for line in lines:
        if not line.startswith("info string "):
            continue
        body = line[len("info string "):]
        if "N:" not in body or "(P:" not in body:
            continue
        move = body.split()[0]

        def field(tag: str) -> str:
            i = body.index(f"({tag}:") + len(tag) + 2
            return body[i : body.index(")", i)].strip()

        wl, d = field("WL"), field("D")
        if board is not None and move != "node":
            move = board.parse_uci(move).uci()
        stats = EdgeStats(
            move=move,
            n=int(body[body.index("N:") + 2 :].split()[0]),
            p=float(field("P").rstrip("%")),
            wl=None if "-.-" in wl else float(wl),
            d=None if "-.-" in d else float(d),
            q=float(field("Q")),
            terminal="(T)" in body,
        )
        if move == "node":
            root = stats
        else:
            edges.append(stats)
    return edges, root


def parse_multipv(lines: list[str], board: chess.Board) -> dict[str, tuple]:
    """The last `info ... multipv k ... score (cp|mate) X ... wdl W D L ... pv MOVE` line per move:
    move -> (kind, score, (W, D, L) permille, k)."""
    out: dict[str, tuple] = {}
    for line in lines:
        if not line.startswith("info") or " multipv " not in line or " pv " not in line:
            continue
        t = line.split()
        move = board.parse_uci(t[t.index("pv") + 1]).uci()
        i = t.index("score")
        w = t.index("wdl")
        out[move] = (t[i + 1], int(t[i + 2]), tuple(int(x) for x in t[w + 1 : w + 4]), int(t[t.index("multipv") + 1]))
    return out


class _Engine:
    def __init__(self, cmd: list[str], options: dict[str, str]) -> None:
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, bufsize=1)
        self.send("uci")
        self.until("uciok")
        for k, v in options.items():
            self.send(f"setoption name {k} value {v}")
        self.send("isready")
        self.until("readyok")

    def send(self, line: str) -> None:
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def until(self, prefix: str) -> list[str]:
        lines = []
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            lines.append(line)
            if line.startswith(prefix):
                return lines
            if line.startswith("error") or "Unknown command" in line or "No such option" in line:
                raise EngineError(line)
        raise EngineError(f"lc0 exited (code {self.proc.poll()}): {self.proc.stderr.read()[-3000:]}")

    def close(self) -> None:
        if self.proc.poll() is None:
            self.send("quit")
            self.proc.wait(timeout=30)


class Lc0Search:
    """Per-move Q after one search of `nodes` nodes from the position, as a `ValueModel`.

    A pool of `processes` engine processes sharing the GPU (a search's own batches are
    small and sequential, so one process leaves the GPU mostly idle). Each process: one
    search thread, no task workers, no NN cache, minibatches of `MAX_BATCH`, no prefetch,
    `SmartPruningFactor 0` so that the search spends all its nodes, and `ucinewgame`
    before every position, so a position's values do not depend on what was searched
    before it in the same process.

    With `wdl_elo` (a `tnh.models.maia3.Elo`) the search is calibrated to the two
    players: `WDLCalibrationElo` = the mover's rating, `Contempt` = mover minus opponent,
    `ContemptMode play` (the engine is the side to move) and `WDLEvalObjectivity 0`,
    so that the reported WDL is the one for those players, not the objective one.
    """

    def __init__(self, nodes: int, network: str | Path = DEFAULT_NETWORK, *, wdl_elo=None, processes: int = 1,
                 backend: str = "cuda-auto", home: Path = LC0_HOME, extra_options: dict[str, str] | None = None) -> None:
        import queue
        from concurrent.futures import ThreadPoolExecutor

        self.nodes = nodes
        self.network = network_path(str(network), home)
        self.wdl_elo = wdl_elo
        self.processes = processes
        self.name = f"lc0-{short_name(self.network)}"
        options = {
            "WeightsFile": str(self.network), "Backend": backend, "BackendOptions": f"max_batch={MAX_BATCH}",
            "MinibatchSize": str(MAX_BATCH), "MaxPrefetch": "0", "Threads": "1", "TaskWorkers": "0",
            "NNCacheSize": "0", "SmartPruningFactor": "0", "VerboseMoveStats": "true", "UCI_ShowWDL": "true",
            "MultiPV": "500", "ScoreType": "Q",
            **(extra_options or {}),
        }
        if wdl_elo is not None:
            options.update({"ContemptMode": "play", "WDLEvalObjectivity": "0"})
        self.options = options
        self._all = [_Engine([str(binary(home))], options) for _ in range(processes)]
        self._free: queue.Queue = queue.Queue()
        for e in self._all:
            e.last_elo = None
            self._free.put(e)
        self._pool = ThreadPoolExecutor(processes)
        self._replayer = Replayer()
        self.stats: Counter = Counter()
        self._lock = threading.Lock()

    @property
    def variant(self) -> str:
        v = f"nodes{self.nodes}"
        return v if self.wdl_elo is None else f"{v}-wdl-{self.wdl_elo.name}"

    def search(self, q: Query, board: chess.Board | None = None) -> tuple[list[EdgeStats], EdgeStats | None, list[str]]:
        if board is None:
            board = self._replayer.boards(q)[-1]
        e = self._free.get()
        try:
            if self.wdl_elo is not None:
                me, opp = self.wdl_elo.of(q)
                if (me, opp) != e.last_elo:
                    e.send(f"setoption name WDLCalibrationElo value {me:g}")
                    e.send(f"setoption name Contempt value {me - opp:g}")
                    e.last_elo = (me, opp)
            e.send("ucinewgame")
            e.send("position startpos moves " + " ".join(q.moves_uci) if q.moves_uci else f"position fen {q.fen}")
            e.send(f"go nodes {self.nodes}")
            lines = e.until("bestmove")
        finally:
            self._free.put(e)
        edges, root = parse_verbose_stats(lines, board)
        return edges, root, lines

    def evaluate(self, queries: list[Query]) -> list[MoveValues]:
        """Q per root move from the MultiPV lines (ScoreType Q: score = 10000 Q, exact), visit
        counts from the verbose stats, whose own Q goes through lc0's approximate logistic."""
        jobs = [(q, self._replayer.boards(q)[-1].copy(stack=False)) for q in queries]
        return list(self._pool.map(self._one, jobs))

    def _one(self, job) -> MoveValues:
        q, board = job
        legal = [m.uci() for m in board.legal_moves]
        n = len(legal)
        value = np.full(n, np.nan, dtype=np.float32)
        cp = np.full(n, np.nan, dtype=np.float32)
        mate = np.zeros(n, dtype=np.int16)
        if not legal:
            with self._lock:
                self.stats["no legal moves"] += 1
            return MoveValues([], value, cp, mate, None)
        edges, root, lines = self.search(q, board)
        visits = {s.move: s.n for s in edges}
        if set(visits) != set(legal):
            raise EngineError(f"lc0 root moves {sorted(visits)} differ from the legal moves {sorted(legal)}")
        pv = parse_multipv(lines, board)
        if set(pv) != set(legal):
            raise EngineError(f"lc0 MultiPV lines cover {len(pv)} of {len(legal)} legal moves")
        for i, mv in enumerate(legal):
            if visits[mv] == 0:
                continue    # the score of an unvisited move is its parent's, not its own
            kind, score, _wdl, _k = pv[mv]
            if kind == "mate":
                mate[i] = score
                value[i] = 1.0 if score > 0 else 0.0
            else:
                qv = score / 10000
                value[i] = (1.0 + qv) / 2.0
                cp[i] = 90.0 * math.tan(1.5637541897 * max(-0.9999, min(0.9999, qv)))
        best = min(pv.values(), key=lambda r: r[3])   # multipv 1
        wdl = tuple(x / 1000 for x in best[2])
        with self._lock:
            self.stats["positions"] += 1
            self.stats["moves"] += n
            self.stats["moves not visited"] += int(np.isnan(value).sum())
        return MoveValues(legal, value, cp, mate, wdl)

    def close(self) -> None:
        self._pool.shutdown()
        for e in self._all:
            e.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# -- every child once, without search ---------------------------------------------------

def _terminal_value(board: chess.Board) -> tuple[float, int] | None:
    """Exact (expected score for the side that just moved, mate) when the game is over
    after the move, else None.

    Handled: checkmate; stalemate; insufficient material (python-chess: K v K, K+minor
    v K, bishops all on one colour); the fifty-move rule (halfmove clock >= 100, which a
    player may claim) and threefold repetition (claimable, counted over the whole game),
    both taken as draws. The network's input carries the halfmove clock and eight
    positions of history, so it cannot see a repetition further back, nor that a draw
    is already claimable.
    """
    if board.is_checkmate():
        return 1.0, 1
    if (board.is_stalemate() or board.is_insufficient_material() or board.halfmove_clock >= 100
            or board.is_repetition(3)):
        return 0.5, 0
    return None


class Lc0Children:
    """The raw network's value of the position after every legal move, as a `ValueModel`.

    Each child is evaluated once, with the game so far plus the move as history. `value`
    is the mover's expected score after the move, 1 minus the child's own expected score
    for its side to move; `cp` by Lc0's formula on the mover's Q; `wdl` the raw value of
    the position itself. Terminal children get their exact value (see `_terminal_value`).
    Children of all queries in a call are pooled and evaluated in calls of `batch_size`
    (at most `MAX_BATCH`), so a position's last partial batch is filled by the next one's.
    """

    def __init__(self, network: str | Path = DEFAULT_NETWORK, *, backend: str = "cuda-auto",
                 batch_size: int = MAX_BATCH, pool_across_positions: bool = True, home: Path = LC0_HOME) -> None:
        if batch_size > MAX_BATCH:
            raise ValueError(f"batch_size above the backend's max_batch {MAX_BATCH}")
        self.network = network_path(str(network), home)
        self._lc = backends(home)
        self._weights = self._lc.Weights(str(self.network))
        self._backend = self._lc.Backend(weights=self._weights, backend=backend, options=f"max_batch={MAX_BATCH}")
        self.backend_name = backend
        self.batch_size = batch_size
        self.pool = pool_across_positions
        self.name = f"lc0-{short_name(self.network)}"
        self.variant = "children" + ("" if backend == "cuda-auto" else f"-{backend}")
        self._replayer = Replayer()
        self.stats: Counter = Counter()

    def _state(self, q: Query, extra: list[str]):
        if q.moves_uci:
            return self._lc.GameState(moves=q.moves_uci + extra)
        return self._lc.GameState(fen=q.fen, moves=extra)

    def _evaluate_inputs(self, inputs: list) -> list[tuple[float, float]]:
        out = []
        for s in range(0, len(inputs), self.batch_size):
            out += [(float(r.q()), float(r.d())) for r in self._backend.evaluate(*inputs[s : s + self.batch_size])]
        return out

    def evaluate(self, queries: list[Query]) -> list[MoveValues]:
        plans = []   # per query: legal moves, terminal values, and the slots of its network inputs
        inputs: list = []
        results: list[MoveValues] = []
        for q in queries:
            game = chess.Board()
            if q.moves_uci:
                for m in q.moves_uci:
                    game.push_uci(m)
            else:
                game = chess.Board(q.fen)
            self._replayer.boards(q)   # checks that the moves reach the FEN
            legal = list(game.legal_moves)
            root_slot = len(inputs)
            inputs.append(self._state(q, []).as_input(self._backend))
            slots, terminal = [], []
            for m in legal:
                game.push(m)
                t = _terminal_value(game)
                game.pop()
                terminal.append(t)
                if t is None:
                    slots.append(len(inputs))
                    inputs.append(self._state(q, [m.uci()]).as_input(self._backend))
                else:
                    slots.append(None)
            plans.append((legal, terminal, slots, root_slot))
            if not self.pool:
                results += self._finish(plans, inputs)
                plans, inputs = [], []
        results += self._finish(plans, inputs)
        return results

    def _finish(self, plans, inputs) -> list[MoveValues]:
        evals = self._evaluate_inputs(inputs)
        out = []
        for legal, terminal, slots, root_slot in plans:
            n = len(legal)
            value = np.full(n, np.nan, dtype=np.float32)
            cp = np.full(n, np.nan, dtype=np.float32)
            mate = np.zeros(n, dtype=np.int16)
            for i, (t, slot) in enumerate(zip(terminal, slots)):
                if t is not None:
                    value[i], mate[i] = t
                    self.stats["terminal children"] += 1
                    if mate[i] == 0:
                        cp[i] = 0.0
                    continue
                q_child, _d = evals[slot]
                q_mover = -q_child
                value[i] = (1.0 + q_mover) / 2.0
                cp[i] = 90.0 * math.tan(1.5637541897 * max(-0.9999, min(0.9999, q_mover)))
            rq, rd = evals[root_slot]
            self.stats["positions"] += 1
            self.stats["children evaluated"] += sum(s is not None for s in slots)
            out.append(MoveValues([m.uci() for m in legal], value, cp, mate, ((1 + rq - rd) / 2, rd, (1 - rq - rd) / 2)))
        return out
