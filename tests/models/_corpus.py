"""A tiny games.parquet and positions.parquet in the DESIGN.md schema, from fixture PGNs.

Private to the model tests. The real corpus builder is `tnh.data`; this one exists only
so that the feature runner can be tested before the corpus does. Where DESIGN.md leaves
a detail open, the choice made here is marked GUESS.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from collections import defaultdict
from pathlib import Path

import chess
import chess.pgn
import polars as pl

FIXTURES = Path(__file__).parent / "fixtures"
RESULTS = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


def _sha16(*parts: str) -> str:
    # GUESS: DESIGN says "sha1 over source, players, date and moves" without a byte layout.
    return hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:16]


def _split(tournament_id: str) -> str:
    # GUESS: DESIGN says "a hash of tournament_id, 80/10/10" without naming the hash.
    b = int(hashlib.sha1(tournament_id.encode()).hexdigest()[:8], 16) % 10
    return "train" if b < 8 else ("val" if b == 8 else "test")


def _time_control(tag: str | None, event: str) -> tuple[int | None, int | None, str, str]:
    """base_s, inc_s, tempo, tempo_source."""
    m = re.fullmatch(r"(?:\d+/)?(\d+)(?:\+(\d+))?(?::.*)?", tag or "")
    if m:
        base, inc = int(m[1]), int(m[2] or 0)
        source = "tag"
    else:
        m = re.match(r"\d+/(\d+) min.*?(\d+)\s*s", tag or "")
        if m:   # "40/100 min, 20/50 min, 15 min for the rest, with 30s/move"
            base, inc, source = int(m[1]) * 60, int(m[2]), "tag"
        else:
            low = event.lower()
            tempo = "rapid" if "rapid" in low or "tie-break" in low else ("blitz" if "blitz" in low else "classical")
            return None, None, tempo, "event_name"   # GUESS: an event with no tag and no tempo word is classical
    total = base + 60 * inc
    tempo = "classical" if total >= 3600 else ("rapid" if total > 600 else "blitz")
    return base, inc, tempo, source


def position_key(board: chess.Board) -> str:
    # GUESS: the en passant square counts only when a capture is legal.
    return " ".join(board.fen(en_passant="legal").split()[:4])


def build(out: Path, pgn: Path = FIXTURES / "games.pgn", *, max_ply: int = 80, source: str = "lichess_broadcast") -> tuple[Path, Path]:
    games, keys_per_game = [], []
    with open(pgn) as f:
        while (g := chess.pgn.read_game(f)) is not None:
            h = g.headers
            if h.get("Result") not in RESULTS:
                continue
            board, moves, clocks, keys = g.board(), [], [], []
            for node in g.mainline():
                keys.append(position_key(board))
                moves.append(node.move.uci())
                c = node.clock()
                clocks.append(None if c is None else int(round(c)))
                board.push(node.move)
            keys.append(position_key(board))
            date_raw = h.get("Date", "????.??.??")
            try:
                date = dt.datetime.strptime(date_raw, "%Y.%m.%d").date()
            except ValueError:
                date = None
            base, inc, tempo, tempo_source = _time_control(h.get("TimeControl"), h.get("Event", ""))
            tournament_id = _sha16(source, h.get("Event", ""))   # GUESS: event name per source
            elo = lambda k: int(h[k]) if h.get(k, "").isdigit() else None
            fide = lambda k: int(h[k]) if h.get(k, "").isdigit() else None
            has_clock = sum(c is not None for c in clocks) >= 0.8 * len(clocks)
            games.append({
                "game_id": _sha16(source, h["White"], h["Black"], date_raw, " ".join(moves)),
                "source": source,
                "tournament_id": tournament_id,
                "event": h.get("Event", ""), "site": h.get("Site", ""), "round": h.get("Round", ""),
                "date": date, "year": date.year if date else (int(date_raw[:4]) if date_raw[:4].isdigit() else None),
                "white": h["White"], "black": h["Black"],
                "white_elo": elo("WhiteElo"), "black_elo": elo("BlackElo"),
                "white_fide_id": fide("WhiteFideId"), "black_fide_id": fide("BlackFideId"),
                "white_title": h.get("WhiteTitle"), "black_title": h.get("BlackTitle"),
                "result": h["Result"],
                "time_control_raw": h.get("TimeControl"),
                "tempo": tempo, "tempo_source": tempo_source,
                "base_s": base, "inc_s": inc,
                "has_clock": has_clock,
                "n_plies": len(moves),
                "moves_uci": " ".join(moves),
                "clocks_s": clocks if any(c is not None for c in clocks) else None,
                "split": _split(tournament_id),
            })
            keys_per_game.append(keys)

    pos_count: dict[str, int] = defaultdict(int)
    for g, keys in zip(games, keys_per_game):
        if (g["white_elo"] or 0) >= 1800 and (g["black_elo"] or 0) >= 1800:
            for k in set(keys):
                pos_count[k] += 1

    rows = []
    for g, keys in zip(games, keys_per_game):
        if g["tempo"] != "classical" or g["white_elo"] is None or g["black_elo"] is None:
            continue
        board = chess.Board()
        moves, clocks = g["moves_uci"].split(), g["clocks_s"] or [None] * g["n_plies"]
        for ply, uci in enumerate(moves[:max_ply]):
            white = board.turn == chess.WHITE
            # GUESS: "before the move" = the mover's clock after his previous move (base time
            # before his first), the opponent's clock after the opponent's last move.
            mover_before = clocks[ply - 2] if ply >= 2 else g["base_s"]
            opp_before = clocks[ply - 1] if ply >= 1 else g["base_s"]
            spent = None
            if mover_before is not None and clocks[ply] is not None and g["inc_s"] is not None:
                spent = mover_before - clocks[ply] + g["inc_s"]   # GUESS: wrong at a time-control boundary
            r = RESULTS[g["result"]]
            rows.append({
                "game_id": g["game_id"], "ply": ply, "fen": board.fen(), "played_uci": uci,
                "white_to_move": white,
                "mover_elo": g["white_elo"] if white else g["black_elo"],
                "opp_elo": g["black_elo"] if white else g["white_elo"],
                "clock_mover_s": mover_before, "clock_opp_s": opp_before, "move_time_s": spent,
                "pos_key": keys[ply], "pos_count": pos_count[keys[ply]],
                "n_legal": board.legal_moves.count(),
                "result_mover": r if white else 1.0 - r,
                "split": g["split"],
            })
            board.push_uci(uci)

    out.mkdir(parents=True, exist_ok=True)
    games_path, positions_path = out / "games.parquet", out / "positions.parquet"
    pl.DataFrame(games, infer_schema_length=None, schema_overrides={
        "clocks_s": pl.List(pl.Int64), "date": pl.Date, "base_s": pl.Int64, "inc_s": pl.Int64,
        "white_fide_id": pl.Int64, "black_fide_id": pl.Int64, "time_control_raw": pl.String,
        "white_title": pl.String, "black_title": pl.String}).write_parquet(games_path)
    ints = {c: pl.Int64 for c in ("clock_mover_s", "clock_opp_s", "move_time_s")}
    pl.DataFrame(rows, schema_overrides=ints, infer_schema_length=None).write_parquet(positions_path)
    return games_path, positions_path
