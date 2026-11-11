"""The blitz reference: Maia-3 on Lichess blitz, on the scale it was trained on.

    uv run python -m tnh.eval.blitz fetch [--month 2025-09] [--games 20000] [--min-elo 1800]
    uv run python -m tnh.eval.blitz build [--month 2025-09]

``fetch`` streams the start of one monthly Lichess dump
(``lichess_db_standard_rated_YYYY-MM.pgn.zst``, about 30 GB), decompresses it as
it arrives, keeps rated blitz games between humans where both are rated at
least ``--min-elo``, and closes the connection once it has ``--games`` of them.
Nothing else of the file is stored. The kept games go to
``raw/lichess_blitz_ref/<stem>.first<N>.pgn`` verbatim, with a ``.json`` beside
it that says how many bytes were read, how many games were scanned and why
each one that was not kept was dropped.

Maia-3 was trained on blitz from 2023-01 to 2025-07, so the default month is
2025-09: out of sample with a month to spare. The start of a dump is the first
hours of the month (UTC), which is a sample by time of day, not of the month.

``build`` turns the kept PGN into ``corpus/blitz_ref/games.parquet`` and
``positions.parquet`` in the corpus schemas, and ``corpus/blitz_ref/window.parquet``:
the rows of the common window (ply ``>= 20`` and ``< 60``, at least 30 s on the
mover's clock), which is what the feature runner is given. Differences from
the classical corpus, all deliberate:

- ``pos_count`` is null: the book rule is not part of the common window.
- ``clock_mover_s`` before a side's first move is the base time (Lichess
  clocks are exact; the DGT caveat in DESIGN.md does not apply).
- ``tournament_id`` is the game id: every game is its own bootstrap cluster.
- ``split`` is ``test``; nothing is fitted on these games.
- Ratings are Lichess blitz ratings, not FIDE.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from collections import Counter
from pathlib import Path

import chess
import httpx
import numpy as np
import polars as pl
import zstandard

from tnh.data.parse import GameError, parse_movetext, sha_hex
from tnh.data.pgn import iter_game_texts, split_game
from tnh.paths import CORPUS, RAW

URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
USER_AGENT = "tnh-eval/0.1 (research download)"
RAW_DIR = RAW / "lichess_blitz_ref"
OUT_DIR = CORPUS / "blitz_ref"
RESULT_WHITE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
GOOD_TERMINATIONS = {"Normal", "Time forfeit"}

MIN_PLY = 20
MAX_PLY = 60
MIN_CLOCK_S = 30


class _IterRaw(io.RawIOBase):
    """A readable file over an iterator of byte chunks (the HTTP body)."""

    def __init__(self, chunks):
        self._chunks = chunks
        self._buf = b""
        self.n_bytes = 0

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        while not self._buf:
            chunk = next(self._chunks, None)
            if chunk is None:
                return 0
            self.n_bytes += len(chunk)
            self._buf = chunk
        n = min(len(b), len(self._buf))
        b[:n] = self._buf[:n]
        self._buf = self._buf[n:]
        return n


def drop_reason(h: dict[str, str], min_elo: int) -> str | None:
    """Why a game is not kept, or None. The order is the order of the checks."""
    event = h.get("Event", "")
    if not event.startswith("Rated Blitz"):
        return "not rated blitz"
    if h.get("Variant", "Standard") != "Standard" or "FEN" in h:
        return "not standard"
    if "BOT" in (h.get("WhiteTitle"), h.get("BlackTitle")):
        return "bot"
    if h.get("Result") not in RESULT_WHITE:
        return "no result"
    if h.get("Termination") not in GOOD_TERMINATIONS:
        return f"termination {h.get('Termination')}"
    we, be = h.get("WhiteElo", ""), h.get("BlackElo", "")
    if not (we.isdigit() and be.isdigit()):
        return "rating missing"
    if int(we) < min_elo or int(be) < min_elo:
        return "rating below min"
    tc = h.get("TimeControl", "")
    if "+" not in tc:
        return "no time control"
    return None


def cmd_fetch(a) -> None:
    url = URL.format(month=a.month)
    stem = Path(url).name.removesuffix(".pgn.zst")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"{stem}.first{a.games}.pgn"
    if out.exists():
        raise SystemExit(f"{out} exists; delete it to fetch again")
    tmp = out.with_suffix(".pgn.part")
    dropped, kept, scanned = Counter(), 0, 0
    t0 = time.time()
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60.0, follow_redirects=True) as client:
        with client.stream("GET", url) as r:
            if r.status_code == 429:
                raise SystemExit("HTTP 429: wait a minute and run again")
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            raw = _IterRaw(r.iter_bytes(1 << 16))
            reader = zstandard.ZstdDecompressor().stream_reader(raw, read_across_frames=True)
            text = io.TextIOWrapper(reader, encoding="utf-8", errors="strict", newline="")
            with tmp.open("w", encoding="utf-8") as f:
                for game in iter_game_texts(text):
                    scanned += 1
                    h, _ = split_game(game)
                    why = drop_reason(h, a.min_elo)
                    if why is not None:
                        dropped[why] += 1
                        continue
                    f.write(game if game.endswith("\n\n") else game.rstrip("\n") + "\n\n")
                    kept += 1
                    if kept % 2000 == 0:
                        print(f"kept {kept}/{a.games}, scanned {scanned}, "
                              f"{raw.n_bytes / 1e6:.0f} MB read", flush=True)
                    if kept >= a.games:
                        break
            n_bytes = raw.n_bytes
    tmp.rename(out)
    meta = {"url": url, "content_length": total, "bytes_read": n_bytes, "games_scanned": scanned,
            "games_kept": kept, "min_elo": a.min_elo, "dropped": dict(dropped.most_common()),
            "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0)}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))


def game_rows(gid: str, h: dict[str, str], ucis: list[str], clocks: list[int | None]) -> list[dict]:
    base, inc = (int(x) for x in h["TimeControl"].split("+"))
    we, be = int(h["WhiteElo"]), int(h["BlackElo"])
    rw = RESULT_WHITE[h["Result"]]
    board = chess.Board()
    rows = []
    for ply, uci in enumerate(ucis):
        white = board.turn == chess.WHITE
        before = base if ply < 2 else clocks[ply - 2]
        opp_before = (base if ply < 1 else clocks[ply - 1])
        after = clocks[ply]
        mt = None if before is None or after is None else before - after + inc
        rows.append({
            "game_id": gid, "ply": ply, "fen": board.fen(), "played_uci": uci, "white_to_move": white,
            "mover_elo": we if white else be, "opp_elo": be if white else we,
            "clock_mover_s": before, "clock_opp_s": opp_before,
            "move_time_s": None if mt is None or mt < -2 else max(0, mt),
            "pos_key": board.epd(), "pos_count": None, "n_legal": board.legal_moves.count(),
            "result_mover": rw if white else 1.0 - rw, "split": "test",
        })
        board.push_uci(uci)
    return rows


def cmd_build(a) -> None:
    paths = sorted(RAW_DIR.glob(f"lichess_db_standard_rated_{a.month}.first*.pgn"))
    if len(paths) != 1:
        raise SystemExit(f"expected one kept PGN for {a.month} in {RAW_DIR}, found {paths}")
    path = paths[0]
    games, rows, dropped = [], [], Counter()
    for text in iter_game_texts(path.open(encoding="utf-8", newline="")):
        h, movetext = split_game(text)
        try:
            ucis, clocks, _, cut = parse_movetext(movetext)
        except GameError as e:
            dropped[e.reason] += 1
            continue
        if cut:
            dropped["cut"] += 1
            continue
        if not ucis:
            dropped["no moves"] += 1
            continue
        if sum(c is not None for c in clocks) < 0.8 * len(clocks):
            dropped["clocks missing"] += 1
            continue
        gid = sha_hex(h.get("Site", ""), " ".join(ucis))
        base, inc = (int(x) for x in h["TimeControl"].split("+"))
        games.append({
            "game_id": gid, "source": "lichess_blitz", "tournament_id": gid, "event": h.get("Event", ""),
            "site": h.get("Site", ""), "white": h.get("White", ""), "black": h.get("Black", ""),
            "white_elo": int(h["WhiteElo"]), "black_elo": int(h["BlackElo"]), "result": h["Result"],
            "time_control_raw": h["TimeControl"], "tempo": "blitz", "base_s": base, "inc_s": inc,
            "has_clock": True, "n_plies": len(ucis), "moves_uci": " ".join(ucis), "clocks_s": clocks,
            "split": "test",
        })
        rows.extend(game_rows(gid, h, ucis, clocks))
    print(f"{path.name}: {len(games)} games, {len(rows)} positions; dropped {dict(dropped)}")
    g = pl.DataFrame(games)
    if g.get_column("game_id").is_duplicated().any():
        raise SystemExit("duplicate game_id")
    p = pl.DataFrame(rows, schema_overrides={"pos_count": pl.Int64, "clock_mover_s": pl.Int64,
                                             "clock_opp_s": pl.Int64, "move_time_s": pl.Int64})
    win = p.filter(window_expr())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g.write_parquet(OUT_DIR / "games.parquet")
    p.write_parquet(OUT_DIR / "positions.parquet")
    win.write_parquet(OUT_DIR / "window.parquet")
    elo = np.concatenate([g["white_elo"].to_numpy(), g["black_elo"].to_numpy()])
    print(f"window (ply {MIN_PLY}..{MAX_PLY - 1}, clock >= {MIN_CLOCK_S} s): {win.height} positions from "
          f"{win.get_column('game_id').n_unique()} games; player rating quartiles "
          f"{np.percentile(elo, [25, 50, 75]).tolist()}")
    print(f"wrote {OUT_DIR}")


def window_expr() -> pl.Expr:
    """The common window on the blitz side: ply in [20, 60), at least 30 s on the mover's clock."""
    return (pl.col("ply") >= MIN_PLY) & (pl.col("ply") < MAX_PLY) & (pl.col("clock_mover_s") >= MIN_CLOCK_S)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m tnh.eval.blitz", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(required=True)
    sp = sub.add_parser("fetch")
    sp.add_argument("--month", default="2025-09")
    sp.add_argument("--games", type=int, default=20_000)
    sp.add_argument("--min-elo", type=int, default=1800)
    sp.set_defaults(fn=cmd_fetch)
    sp = sub.add_parser("build")
    sp.add_argument("--month", default="2025-09")
    sp.set_defaults(fn=cmd_build)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
