"""Parse the raw dumps into ``corpus/games.parquet``.

    uv run python -m tnh.data.parse [--sources lichess_broadcast] [--workers N] [--limit-files K]

One row per game, every tempo and rating (schema in docs/DESIGN.md). Each
monthly file is parsed in its own worker process. Then:

1. games are de-duplicated across files and sources on (white surname,
   black surname, year, moves); the copy with the most clock comments wins;
2. tournaments are the connected components of the tournament nodes that the
   games touch (see ``tnh.data.sources``), with the copies of a duplicated
   game joined too;
3. the split is a hash of ``tournament_id``, 80/10/10.

Every dropped game is counted with its reason, printed, and written to
``corpus/parse_stats.json``. ``corpus/games_extra.parquet`` holds per-game
fields that are not part of the contract but are needed for the report and
for checking games by hand (broadcast name, game URL, how tempo was decided).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import multiprocessing
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import chess
import polars as pl

from tnh.data.pgn import iter_file_games, split_game
from tnh.data.sources import SOURCES, Source, event_drop_reason
from tnh.data.timecontrol import decide_tempo, parse_clock
from tnh.paths import CORPUS

RESULTS = ("1-0", "0-1", "1/2-1/2")
STANDARD_VARIANTS = {"standard", "from position"}
START_FEN = chess.STARTING_FEN
KNOWN_TITLES = {"GM", "IM", "FM", "CM", "WGM", "WIM", "WFM", "WCM", "NM", "WNM", "LM", "AGM", "AIM", "AFM", "ACM"}
HAS_CLOCK_SHARE = 0.8

TOKEN_RE = re.compile(r"\{[^}]*\}?|;[^\n]*|\(|\)|\$\d+|[^\s(){};]+")
MOVENUM_RE = re.compile(r"^\d+\.+")
ANNOT_RE = re.compile(r"[!?]+$")
DATE_RE = re.compile(r"^(\d{4})\.(\d{2})\.(\d{2})$")
YEAR_RE = re.compile(r"^(\d{4})\.")

GAMES_SCHEMA = {
    "game_id": pl.String,
    "source": pl.String,
    "tournament_id": pl.String,
    "event": pl.String,
    "site": pl.String,
    "round": pl.String,
    "date": pl.Date,
    "year": pl.Int32,
    "white": pl.String,
    "black": pl.String,
    "white_elo": pl.Int32,
    "black_elo": pl.Int32,
    "white_fide_id": pl.Int64,
    "black_fide_id": pl.Int64,
    "white_title": pl.String,
    "black_title": pl.String,
    "result": pl.String,
    "time_control_raw": pl.String,
    "tempo": pl.String,
    "tempo_source": pl.String,
    "base_s": pl.Int32,
    "inc_s": pl.Int32,
    "has_clock": pl.Boolean,
    "n_plies": pl.Int32,
    "moves_uci": pl.String,
    "clocks_s": pl.List(pl.Int32),
    "split": pl.String,
}


class GameError(Exception):
    """A game that cannot be used; ``reason`` is the count key."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


def parse_movetext(movetext: str, board: chess.Board | None = None) -> tuple[list[str], list[int | None], str | None]:
    """Mainline moves as UCI, the clock after each ply, and the result token.

    Variations are skipped. A comment belongs to the ply before it; the first
    ``%clk`` in the comments after a ply is that ply's clock. Raises
    ``GameError`` on an illegal, ambiguous or unreadable move.
    """
    board = board or chess.Board()
    ucis: list[str] = []
    clocks: list[int | None] = []
    result_token = None
    depth = 0
    for tok in TOKEN_RE.findall(movetext):
        c = tok[0]
        if c == "{":
            if depth == 0 and ucis and clocks[-1] is None:
                clocks[-1] = parse_clock(tok)
            continue
        if c == "(":
            depth += 1
            continue
        if c == ")":
            depth -= 1
            if depth < 0:
                raise GameError("malformed_movetext", "unbalanced )")
            continue
        if depth or c in ";$":
            continue
        if tok in ("1-0", "0-1", "1/2-1/2", "*"):
            result_token = tok
            continue
        san = ANNOT_RE.sub("", MOVENUM_RE.sub("", tok))
        if not san:
            continue
        if san in ("--", "Z0", "0000", "null"):
            raise GameError("null_move", san)
        try:
            move = board.parse_san(san)
        except chess.IllegalMoveError as e:
            raise GameError("illegal_move", str(e)) from None
        except chess.AmbiguousMoveError as e:
            raise GameError("ambiguous_move", str(e)) from None
        except chess.InvalidMoveError as e:
            raise GameError("invalid_san", str(e)) from None
        ucis.append(move.uci())
        clocks.append(None)
        board.push(move)
    if depth:
        raise GameError("malformed_movetext", "unclosed (")
    return ucis, clocks, result_token


def surname(name: str) -> str:
    """Lower-case ASCII letters of the surname: before the comma, else the last word."""
    name = name.strip()
    part = name.split(",")[0] if "," in name else (name.split()[-1] if name.split() else "")
    part = unicodedata.normalize("NFKD", part).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", part.lower())


def _int_or_none(s: str | None, lo: int, hi: int) -> int | None:
    if s is None:
        return None
    s = s.strip()
    if not s.isascii() or not s.isdigit():
        return None
    v = int(s)
    return v if lo <= v <= hi else None


def _title(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip().upper()
    return s if s in KNOWN_TITLES else None


def _valid_date(y: int, m: int, d: int) -> bool:
    try:
        datetime.date(y, m, d)
    except ValueError:  # "2024.59.19" occurs
        return False
    return y >= 1800


def _date(h: dict[str, str]) -> tuple[str | None, int | None]:
    """(ISO date or None, year or None) from Date, falling back to UTCDate."""
    date = None
    year = None
    for tag in ("Date", "UTCDate"):
        v = h.get(tag, "")
        m = DATE_RE.match(v)
        if m and date is None and _valid_date(*map(int, m.groups())):
            date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        y = YEAR_RE.match(v)
        if y and year is None and y.group(1) != "0000":
            year = int(y.group(1))
    return date, year


def sha_hex(*parts: str, n: int = 16) -> str:
    return hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:n]


def parse_game(text: str, source: Source) -> dict:
    """One game's row (plus internal fields prefixed ``_``). Raises GameError to drop it."""
    h, movetext = split_game(text)
    variant = h.get("Variant", "Standard").strip().lower()
    if variant not in STANDARD_VARIANTS:
        raise GameError("variant", variant)
    fen = h.get("FEN")
    if fen is not None and fen.strip() != START_FEN:
        raise GameError("non_standard_start", fen)
    if variant == "from position" and fen is None:
        raise GameError("non_standard_start", "From Position without FEN")

    names = source.event_names(h)
    reason = event_drop_reason(names)
    if reason:
        raise GameError(reason)
    if "BOT" in (h.get("WhiteTitle", "").upper(), h.get("BlackTitle", "").upper()):
        raise GameError("engine", "BOT title")

    result = h.get("Result", "").strip()
    if result not in RESULTS:
        raise GameError("result_not_final" if result == "*" else "result_other", result)

    board = chess.Board()
    ucis, clocks, token = parse_movetext(movetext, board)
    if not ucis:
        raise GameError("no_moves")
    if token is not None and token != result:
        raise GameError("result_token_mismatch", f"{token} vs {result}")
    outcome = board.outcome()
    if outcome is not None and outcome.termination in (chess.Termination.CHECKMATE, chess.Termination.STALEMATE):
        if outcome.result() != result:
            raise GameError("result_contradicts_final_position", f"{outcome.result()} vs {result}")

    n_clk = sum(c is not None for c in clocks)
    tempo = decide_tempo(h.get("TimeControl"), clocks, names)
    white, black = h.get("White", "?"), h.get("Black", "?")
    date, year = _date(h)
    moves_uci = " ".join(ucis)
    return {
        "game_id": sha_hex(source.name, white, black, date or str(year or ""), moves_uci),
        "source": source.name,
        "event": h.get("Event", "?"),
        "site": h.get("Site", "?"),
        "round": h.get("Round", "?"),
        "date": date,
        "year": year,
        "white": white,
        "black": black,
        "white_elo": _int_or_none(h.get("WhiteElo"), 100, 3500),
        "black_elo": _int_or_none(h.get("BlackElo"), 100, 3500),
        "white_fide_id": _int_or_none(h.get("WhiteFideId"), 1, 10**10),
        "black_fide_id": _int_or_none(h.get("BlackFideId"), 1, 10**10),
        "white_title": _title(h.get("WhiteTitle")),
        "black_title": _title(h.get("BlackTitle")),
        "result": result,
        "time_control_raw": h.get("TimeControl"),
        "tempo": tempo.tempo,
        "tempo_source": tempo.source,
        "base_s": tempo.base_s,
        "inc_s": tempo.inc_s,
        "has_clock": n_clk >= HAS_CLOCK_SHARE * len(ucis),
        "n_plies": len(ucis),
        "moves_uci": moves_uci,
        "clocks_s": clocks if n_clk else None,
        "_dedup_key": sha_hex(surname(white), surname(black), str(year or ""), moves_uci, n=20),
        "_clock_share": n_clk / len(ucis),
        "_priority": source.priority,
        "_nodes": source.tournament_nodes(h),
        "_tempo_note": tempo.note,
        "_broadcast_name": h.get("BroadcastName"),
        "_game_url": h.get("GameURL") or h.get("BroadcastURL"),
        "_result_token_missing": token is None,
    }


def parse_texts(texts, source: Source, label: str = "") -> tuple[pl.DataFrame, Counter, dict[str, list[str]]]:
    """Parse game texts into a frame (ROW_SCHEMA), with drop counts and up to three examples per reason."""
    rows = []
    drops: Counter = Counter()
    examples: dict[str, list[str]] = {}
    for i, text in enumerate(texts):
        try:
            row = parse_game(text, source)
        except GameError as e:
            drops[e.reason] += 1
            ex = examples.setdefault(e.reason, [])
            if len(ex) < 3:
                ex.append(f"{label} game {i}: {e}")
            continue
        row["_file"] = label
        rows.append(row)
    df = pl.DataFrame(rows, schema=ROW_SCHEMA).with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
    return df, drops, examples


def parse_file(args: tuple[str, str]) -> tuple[str, pl.DataFrame, Counter, dict[str, list[str]]]:
    """Worker: all games of one file."""
    source_name, path = args
    df, drops, examples = parse_texts(iter_file_games(Path(path)), SOURCES[source_name], Path(path).name)
    return path, df, drops, examples


ROW_SCHEMA = {
    **{c: t for c, t in GAMES_SCHEMA.items() if c not in ("tournament_id", "split")},
    "date": pl.String,
    "_dedup_key": pl.String,
    "_clock_share": pl.Float32,
    "_priority": pl.Int32,
    "_nodes": pl.List(pl.String),
    "_tempo_note": pl.String,
    "_broadcast_name": pl.String,
    "_game_url": pl.String,
    "_result_token_missing": pl.Boolean,
    "_file": pl.String,
}


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        p = self.parent.setdefault(x, x)
        while p != x:
            gp = self.parent.setdefault(p, p)
            self.parent[x] = gp
            x, p = p, gp
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # the smaller key is the root, so the id of a component is its smallest node
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


def split_of(tournament_id: str) -> str:
    h = int(hashlib.sha1(tournament_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def assemble(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """Parsed rows -> (games, extra, number of duplicates removed).

    De-duplication: the copy with the most clock comments wins, then the
    source priority, then the game URL, so that the choice is stable. The
    tournament nodes of all copies are joined; a tournament is a connected
    component, named by its smallest node.
    """
    df = df.sort(["_dedup_key", "_clock_share", "_priority", "_game_url"], descending=[False, True, False, False], nulls_last=True)
    copies = df.group_by("_dedup_key", maintain_order=True).agg(
        pl.col("_nodes").list.explode(keep_nulls=False, empty_as_null=False).alias("_all_nodes"),
        pl.len().alias("_n_copies"),
    )
    n_before = len(df)
    df = df.unique("_dedup_key", keep="first", maintain_order=True).join(copies, on="_dedup_key", how="left")
    n_dup = n_before - len(df)

    uf = UnionFind()
    no_nodes = 0
    for nodes in copies["_all_nodes"].to_list():
        if not nodes:
            no_nodes += 1
            continue
        for other in nodes[1:]:
            uf.union(nodes[0], other)
    if no_nodes:
        raise RuntimeError(f"{no_nodes} games have no tournament node; the source's tournament_nodes is incomplete")
    tid = [uf.find(nodes[0]) for nodes in df["_all_nodes"].to_list()]
    df = df.with_columns(pl.Series("tournament_id", tid, dtype=pl.String))
    df = df.with_columns(pl.col("tournament_id").map_elements(split_of, return_dtype=pl.String).alias("split"))

    if df["game_id"].n_unique() != len(df):
        dup = df.filter(pl.col("game_id").is_duplicated()).select("game_id", "white", "black", "date", "_game_url").head(10)
        raise RuntimeError(f"game_id is not unique after de-duplication:\n{dup}")

    games = df.select([pl.col(c).cast(t) for c, t in GAMES_SCHEMA.items()]).sort("game_id")
    extra = df.select(
        "game_id",
        pl.col("_broadcast_name").alias("broadcast_name"),
        pl.col("_game_url").alias("game_url"),
        pl.col("_file").alias("file"),
        pl.col("_tempo_note").alias("tempo_note"),
        pl.col("_clock_share").alias("clock_share"),
        pl.col("_n_copies").alias("n_copies"),
        pl.col("_result_token_missing").alias("result_token_missing"),
    ).sort("game_id")
    return games, extra, n_dup


def build(sources: list[str], workers: int, limit_files: int | None) -> None:
    t0 = time.time()
    tasks = []
    for name in sources:
        files = SOURCES[name].files()
        if limit_files:
            files = files[:limit_files]
        if not files:
            raise SystemExit(f"no raw files for source {name}; run tnh.data.download first")
        tasks += [(name, str(f)) for f in files]
    tasks.sort(key=lambda t: -Path(t[1]).stat().st_size)  # largest first, for load balance
    print(f"parsing {len(tasks)} files with {workers} workers")

    frames: list[pl.DataFrame] = []
    drops: Counter = Counter()
    examples: dict[str, list[str]] = {}
    # spawn, not fork: forking after polars has started its thread pool can deadlock
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for k, (path, file_df, file_drops, file_ex) in enumerate(pool.imap_unordered(parse_file, tasks), 1):
            frames.append(file_df)
            drops += file_drops
            for r, ex in file_ex.items():
                examples.setdefault(r, []).extend(ex[: 3 - len(examples.get(r, []))])
            if k % 10 == 0 or k == len(tasks):
                print(f"  {k}/{len(tasks)} files, {sum(len(f) for f in frames)} games kept, {sum(drops.values())} dropped, {time.time() - t0:.0f} s")
    df = pl.concat(frames)
    del frames
    n_read = len(df) + sum(drops.values())

    games, extra, n_dup = assemble(df)
    del df
    drops["duplicate"] += n_dup

    CORPUS.mkdir(parents=True, exist_ok=True)
    games.write_parquet(CORPUS / "games.parquet", compression="zstd")
    extra.write_parquet(CORPUS / "games_extra.parquet", compression="zstd")

    stats = {
        "games_read": n_read,
        "games_kept": len(games),
        "dropped": dict(drops.most_common()),
        "drop_examples": examples,
        "tournaments": games["tournament_id"].n_unique(),
        "seconds": round(time.time() - t0),
    }
    (CORPUS / "parse_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(f"read {n_read} games, kept {len(games)} in {stats['tournaments']} tournaments, {time.time() - t0:.0f} s")
    print("dropped:")
    for reason, n in drops.most_common():
        print(f"  {reason:36s} {n:9d}   e.g. {examples.get(reason, [''])[0][:120]}")
    print(games.group_by("tempo").len().sort("len", descending=True))
    print(games.group_by("split").len().sort("split"))
    print(f"wrote {CORPUS / 'games.parquet'}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="lichess_broadcast", help="comma-separated source names")
    ap.add_argument("--workers", type=int, default=None, help="default: all cores")
    ap.add_argument("--limit-files", type=int, default=None, help="only the first K files per source (testing)")
    args = ap.parse_args(argv)
    import os

    build(args.sources.split(","), args.workers or os.cpu_count() or 1, args.limit_files)


if __name__ == "__main__":
    main()
