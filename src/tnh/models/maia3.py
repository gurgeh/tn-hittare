"""Maia-3 (CSSLab, May 2026) called directly in batches on the GPU.

The network, its checkpoint and its input encoding are the reference ones from the
`maia3` package (github.com/CSSLab/maia3, pinned in pyproject.toml). Only the
tokenisation is reimplemented, with bitboards instead of a Python loop over squares,
and it is tested square by square against `maia3.dataset.tokenize_board`.

What the reference code does, and so what this wrapper does:

- Input per square: 12 one-hot piece planes for each of the last 8 positions
  (current and 7 earlier), oldest first, current last. Each position is encoded from
  the perspective of *its own* side to move: a position with Black to move is
  mirrored (`Board.mirror()`: ranks flipped, colours swapped). So consecutive history
  frames alternate orientation. When fewer than 8 positions exist, the earliest one is
  repeated in front. No castling rights, en passant, move counters or clocks.
- Ratings: each rating `k` becomes `(k/5000) * e_low + (1 - k/5000) * e_high`, two
  learned 128-dim vectors, after clamping `k` to [0, 5000]. A rating is a continuous
  scalar along one direction in embedding space. There are no rating bins in the
  network; the 100-point bins in the paper only balance the training sample.
- Output: logits over 4352 moves in the mover's frame (64x64 from-to plus 256
  promotions), a value head over (loss, draw, win) for the mover, and a "ponder"
  head that this wrapper does not use.
- Inference runs under `torch.autocast('cuda')` (float16), as `maia3-uci` does.
"""

from __future__ import annotations

import argparse
import math
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F

from maia3.model_registry import resolve_checkpoint_path, resolve_model_spec
from maia3.models import MAIA3Model

from tnh.paths import DATA_ROOT
from tnh.models.base import Prediction, Query, Replayer

WEIGHTS = DATA_ROOT / "weights" / "hf"
SIZES = ("5m", "23m", "79m")
HISTORY = 8
N_MOVES = 4352
ELO_RANGE = (0.0, 5000.0)   # the network's own clamp in MAIA3Model.interpolate_elo
_PROMO = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 3}


def frame(board: chess.Board) -> np.ndarray:
    """(64, 12) uint8 piece planes from the perspective of the side to move.

    Equal to `maia3.dataset.tokenize_board(board)`: planes 0-5 are the mover's
    P N B R Q K, planes 6-11 the opponent's, squares mirrored when Black moves.
    """
    us = board.turn
    them = not us
    bbs = [board.pieces_mask(pt, us) for pt in range(1, 7)] + [board.pieces_mask(pt, them) for pt in range(1, 7)]
    if us == chess.BLACK:
        bbs = [chess.flip_vertical(bb) for bb in bbs]
    bits = np.unpackbits(np.array(bbs, dtype="<u8").view(np.uint8), bitorder="little")
    return bits.reshape(12, 64).T


def move_index(board: chess.Board, move: chess.Move) -> int:
    """Index of `move` in maia3.utils.get_all_possible_moves(), in the mover's frame."""
    frm, to = move.from_square, move.to_square
    if board.turn == chess.BLACK:
        frm, to = frm ^ 56, to ^ 56
    if move.promotion:
        return 4096 + (chess.square_file(frm) * 8 + chess.square_file(to)) * 4 + _PROMO[move.promotion]
    return frm * 64 + to


class Encoder:
    """Queries to network input: (n, 64, 96) uint8 tokens, legal moves and their indices.

    Keeps the boards of the game in progress (see `Replayer`) and their frames, so a
    game queried ply by ply is replayed and tokenised once.
    """

    def __init__(self) -> None:
        self._replayer = Replayer()
        self._frames: dict[int, tuple[chess.Board, np.ndarray]] = {}

    def _frame(self, board: chess.Board) -> np.ndarray:
        hit = self._frames.get(id(board))
        if hit is not None and hit[0] is board:
            return hit[1]
        f = frame(board)
        if len(self._frames) > 4096:
            self._frames.clear()
        self._frames[id(board)] = (board, f)
        return f

    def encode(self, queries: list[Query]) -> tuple[np.ndarray, list[list[str]], list[np.ndarray]]:
        n = len(queries)
        tokens = np.empty((n, 64, 12 * HISTORY), dtype=np.uint8)
        moves: list[list[str]] = []
        idxs: list[np.ndarray] = []
        for i, q in enumerate(queries):
            boards = self._replayer.boards(q)
            frames = [self._frame(b) for b in boards[-HISTORY:]]
            frames = [frames[0]] * (HISTORY - len(frames)) + frames   # repeat the earliest, as the reference does
            tokens[i] = np.concatenate(frames, axis=1)
            board = boards[-1]
            legal = list(board.legal_moves)
            moves.append([m.uci() for m in legal])
            idxs.append(np.fromiter((move_index(board, m) for m in legal), dtype=np.int64, count=len(legal)))
        return tokens, moves, idxs


_NAME_RE = re.compile(
    r"(?:offset(?P<m>[+-]\d+(?:\.\d+)?)(?:-opp(?P<o>[+-]\d+(?:\.\d+)?))?"
    r"|fixed(?P<s>\d+(?:\.\d+)?)-(?P<f>\d+(?:\.\d+)?))"
    r"(?:-clamp(?P<lo>\d+(?:\.\d+)?)-(?P<hi>\d+(?:\.\d+)?))?"
)


@dataclass(frozen=True)
class Elo:
    """The ratings given to the network, one of:

    - fixed: `self_elo` and `oppo_elo`, whatever the players' real ratings;
    - offset: the network gets `mover_elo + mover_offset` and `opp_elo + opp_offset`.

    Both are clamped to `clamp`, by default the network's own [0, 5000]. The name is
    the variant directory: `offset+300` (both sides +300), `offset+300-opp+100`
    (mover +300, opponent +100), `fixed2200-2200`, with `-clampLO-HI` when not default.
    """

    self_elo: float | None = None
    oppo_elo: float | None = None
    mover_offset: float | None = None
    opp_offset: float | None = None
    clamp: tuple[float, float] = ELO_RANGE

    def __post_init__(self) -> None:
        fixed = (self.self_elo is not None, self.oppo_elo is not None)
        offset = (self.mover_offset is not None, self.opp_offset is not None)
        if not ((all(fixed) and not any(offset)) or (all(offset) and not any(fixed))):
            raise ValueError(f"give self_elo and oppo_elo, or mover_offset and opp_offset: {self}")

    @classmethod
    def fixed(cls, self_elo: float, oppo_elo: float, clamp: tuple[float, float] = ELO_RANGE) -> "Elo":
        return cls(self_elo=float(self_elo), oppo_elo=float(oppo_elo), clamp=clamp)

    @classmethod
    def offset(cls, mover: float, opp: float | None = None, clamp: tuple[float, float] = ELO_RANGE) -> "Elo":
        return cls(mover_offset=float(mover), opp_offset=float(mover if opp is None else opp), clamp=clamp)

    @classmethod
    def parse(cls, name: str) -> "Elo":
        m = _NAME_RE.fullmatch(name)
        if m is None:
            raise ValueError(f"not an Elo variant name: {name!r} (e.g. offset+300, offset+300-opp+100, fixed2200-2200)")
        clamp = (float(m["lo"]), float(m["hi"])) if m["lo"] is not None else ELO_RANGE
        if m["m"] is not None:
            return cls.offset(float(m["m"]), None if m["o"] is None else float(m["o"]), clamp)
        return cls.fixed(float(m["s"]), float(m["f"]), clamp)

    @property
    def name(self) -> str:
        if self.mover_offset is not None:
            v = f"offset{self.mover_offset:+g}"
            if self.opp_offset != self.mover_offset:
                v += f"-opp{self.opp_offset:+g}"
        else:
            v = f"fixed{self.self_elo:g}-{self.oppo_elo:g}"
        if self.clamp != ELO_RANGE:
            v += f"-clamp{self.clamp[0]:g}-{self.clamp[1]:g}"
        return v

    def of(self, query: Query) -> tuple[float, float]:
        if self.mover_offset is not None:
            s, o = query.mover_elo + self.mover_offset, query.opp_elo + self.opp_offset
        else:
            s, o = self.self_elo, self.oppo_elo
        lo, hi = self.clamp
        return min(max(s, lo), hi), min(max(o, lo), hi)


class Maia3:
    """Maia-3 as a `MoveModel`, with the ratings it is given set by `elo`.

    `predict_many` runs several `Elo` settings on one encoding of the positions; the
    CPU encoding is the bottleneck for 5M and 23M, so a sweep costs little more than
    one setting until the GPU is saturated.
    """

    def __init__(
        self,
        size: str = "23m",
        elo: Elo | str | None = None,
        *,
        device: str = "cuda",
        amp: bool = True,
        batch_size: int = 512,
        weights_dir: Path = WEIGHTS,
        local_files_only: bool = False,
    ) -> None:
        if size not in SIZES:
            raise ValueError(f"size must be one of {SIZES}, got {size!r}")
        self.size = size
        self.elo = Elo.parse(elo) if isinstance(elo, str) else elo
        self.device = device
        self.amp = amp and device.startswith("cuda")
        self.batch_size = batch_size
        self.name = f"maia3-{size}"
        self.model = _load(size, device, weights_dir, local_files_only)
        self._encoder = Encoder()
        self._gpu = ThreadPoolExecutor(1, thread_name_prefix="maia3-gpu")
        self.n_predicted = 0
        self.illegal_mass_sum = 0.0   # softmax mass outside the legal moves, before renormalising

    def variant(self, elo: Elo | None = None) -> str:
        return (elo or self.elo).name + ("" if self.amp else "-fp32")

    def predict(self, queries: list[Query]) -> list[Prediction]:
        if self.elo is None:
            raise ValueError("this Maia3 has no elo; pass one to the constructor or use predict_many")
        return self.predict_many(queries, [self.elo])[0]

    def predict_many(self, queries: list[Query], elos: list[Elo]) -> list[list[Prediction]]:
        """One list of predictions per `Elo` setting, from a single encoding pass.

        Batches are encoded (python-chess, CPU) on this thread and run on the GPU on a
        worker thread, so encoding the next batch overlaps the network.
        """
        out: list[list[Prediction]] = [[] for _ in elos]
        pending: list[Future] = []

        def drain(f: Future) -> None:
            for k, preds in enumerate(f.result()):
                out[k].extend(preds)

        for start in range(0, len(queries), self.batch_size):
            qs = queries[start : start + self.batch_size]
            batch = self._encoder.encode(qs)
            settings = [np.array([e.of(q) for q in qs], dtype=np.float32).reshape(len(qs), 2) for e in elos]
            pending.append(self._gpu.submit(self._run_batch, batch, settings))
            if len(pending) > 2:
                drain(pending.pop(0))
        for f in pending:
            drain(f)
        return out

    # -- internals ---------------------------------------------------------

    def _encode(self, queries: list[Query], elo: Elo | None = None):
        """Tokens, ratings, moves and move indices for one batch (used by tests and the bench)."""
        tokens, moves, idxs = self._encoder.encode(queries)
        e = elo or self.elo
        elos = np.array([e.of(q) for q in queries], dtype=np.float32).reshape(len(queries), 2)
        return tokens, elos, moves, idxs

    def _put(self, a: np.ndarray) -> torch.Tensor:
        # Pinned memory keeps the copy asynchronous.
        x = torch.from_numpy(a)
        return x.pin_memory().to(self.device, non_blocking=True) if self.device.startswith("cuda") else x.to(self.device)

    @torch.no_grad()
    def _upload(self, batch):
        tokens, moves, idxs = batch
        width = max((len(i) for i in idxs), default=0) or 1
        gather = np.zeros((len(idxs), width), dtype=np.int64)
        valid = np.zeros((len(idxs), width), dtype=bool)
        for r, i in enumerate(idxs):
            gather[r, : len(i)] = i
            valid[r, : len(i)] = True
        return self._put(tokens).float(), self._put(gather), self._put(valid)

    @torch.no_grad()
    def _forward(self, uploaded, elos: np.ndarray):
        t, g, v = uploaded
        e = self._put(elos)
        with torch.autocast("cuda", enabled=self.amp):
            logits_move, logits_value, _ponder = self.model(t, e[:, 0], e[:, 1])
        logits = logits_move.float()
        legal = torch.gather(logits, 1, g).masked_fill(~v, float("-inf"))
        lse_legal = torch.logsumexp(legal, dim=1, keepdim=True)
        logp = legal - lse_legal
        legal_mass = torch.exp(lse_legal[:, 0] - torch.logsumexp(logits, dim=1))
        value = torch.softmax(logits_value.float(), dim=-1)   # (loss, draw, win) for the mover
        return logp, legal_mass, value

    def _collect(self, moves: list[list[str]], launched) -> list[Prediction]:
        logp, legal_mass, value = (x.cpu().numpy() for x in launched)
        out = []
        for r, ms in enumerate(moves):
            loss, draw, win = (float(x) for x in value[r])
            out.append(Prediction(ms, logp[r, : len(ms)].astype(np.float32), (win, draw, loss)))
            if ms:
                self.n_predicted += 1
                self.illegal_mass_sum += 1.0 - float(legal_mass[r])
        return out

    def _run_batch(self, batch, settings: list[np.ndarray]) -> list[list[Prediction]]:
        uploaded = self._upload(batch)
        return [self._collect(batch[1], self._forward(uploaded, elos)) for elos in settings]


class _Maia3Net(MAIA3Model):
    """`MAIA3Model` with the promotion logits computed in one tensor operation.

    The reference forward builds the 256 promotion logits in a Python loop of about
    3000 small GPU operations, which costs ~45 ms of CPU time per batch whatever the
    batch size, and indexes with Python lists, which makes the host wait for the GPU.
    This forward is the reference one line for line except for that block; it does the
    same float16 additions in the same order, and a test checks that the outputs are
    bitwise equal to the reference forward.
    """

    def forward(self, tokens, self_elos, oppo_elos):
        cfg = self.cfg
        if cfg.include_time_info:
            tokens = tokens[:, :, : 12 * cfg.history + 4 - 1]
        else:
            tokens = tokens[:, :, : 12 * cfg.history]
        self_elo_embs = self.interpolate_elo(self_elos).unsqueeze(1).expand(-1, 64, -1)
        oppo_elo_embs = self.interpolate_elo(oppo_elos).unsqueeze(1).expand(-1, 64, -1)
        x = self.token_projection(torch.cat([tokens, self_elo_embs, oppo_elo_embs], dim=-1))
        if hasattr(self, "abs_pe"):
            x = self.abs_pe(x)
        x = self.transformer(x)

        sq_from = self.proj_sq_from(x[:, :64, :])
        sq_to = self.proj_sq_to(x[:, :64, :])
        scores_base = torch.einsum("bid,bjd->bij", sq_from, sq_to) / math.sqrt(cfg.head_hid_dim)
        scores_flat = scores_base.reshape(x.size(0), 64 * 64)

        # rank 7 = squares 48..55, rank 8 = squares 56..63; order (from file, to file, piece)
        promo_biases = self.promo_bias_proj(sq_to[:, 56:64, :]) * math.sqrt(cfg.head_hid_dim)  # (B, 8, 4)
        promotion_logits = (scores_base[:, 48:56, 56:64, None] + promo_biases[:, None, :, :]).reshape(x.size(0), 256)
        logits_move = torch.cat([scores_flat, promotion_logits], dim=1)

        x = self.last_ln(x.mean(dim=1))
        logits_value = self.fc_value(F.relu(self.fc_value_hid(x)))
        logits_ponder = self.fc_ponder(F.relu(self.fc_ponder_hid(x)))
        return logits_move, logits_value, logits_ponder.squeeze(1)


def checkpoint_path(size: str, weights_dir: Path = WEIGHTS, local_files_only: bool = False) -> str:
    spec = resolve_model_spec(f"maia3-{size}")
    return resolve_checkpoint_path(spec, cache_dir=str(weights_dir), local_files_only=local_files_only)


def config(size: str) -> argparse.Namespace:
    return argparse.Namespace(**resolve_model_spec(f"maia3-{size}").config)


def _load(size: str, device: str, weights_dir: Path, local_files_only: bool, *, reference: bool = False) -> MAIA3Model:
    path = checkpoint_path(size, weights_dir, local_files_only)
    model = (MAIA3Model if reference else _Maia3Net)(config(size))
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("model_state_dict", state)
    # Same renaming as maia3.uci.load_model; unlike it, missing or extra keys are an error.
    model.load_state_dict({k.replace("smolgen", "gab"): v for k, v in state.items()}, strict=True)
    return model.to(device).eval()
