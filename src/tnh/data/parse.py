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
import io
import itertools
import json
import multiprocessing
import os
import re
import shutil
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import chess
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from tnh.data.pgn import (
    decode,
    iter_file_games,
    iter_game_texts,
    read_bytes,
    split_game,
)
from tnh.data.sources import SOURCES, Source
from tnh.data.timecontrol import WEAK_SOURCES, decide_tempo, parse_clock
from tnh.paths import CORPUS, DATA_ROOT

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


def _is_san_token(tok: str) -> bool:
    return bool(tok) and tok[0] not in "{();$" and tok not in ("1-0", "0-1", "1/2-1/2", "*") \
        and bool(ANNOT_RE.sub("", MOVENUM_RE.sub("", tok)))


CENTRE = {chess.D4, chess.D5, chess.E4, chess.E5}
_SAN_DEST_RE = re.compile(r"([a-h][1-8])[+#]?$")


def _central_king_san(san: str) -> bool:
    """SAN of a king move to d4, d5, e4 or e5 ("Ke4", "Kxd5+")."""
    m = _SAN_DEST_RE.search(san)
    return san.startswith("K") and m is not None and chess.parse_square(m.group(1)) in CENTRE


@dataclass
class Movetext:
    ucis: list[str]
    clocks: list[int | None]
    result_token: str | None
    cut_illegal_king: bool  # cut before an illegal king move to the centre
    stripped: int  # legal king moves to the centre removed from the end (0-2)


def parse_movetext(movetext: str, board: chess.Board | None = None) -> Movetext:
    """Mainline moves as UCI, the clock after each ply and the result token.

    Variations are skipped. A comment belongs to the ply before it; the first
    ``%clk`` in the comments after a ply is that ply's clock. Raises
    ``GameError`` on an illegal, ambiguous or unreadable move, except for the
    arbiter's signal at the end of a broadcast game, when both kings are put
    in the centre of the DGT board to show the result:

    - an illegal king move to d4, d5, e4 or e5 followed by at most one more
      move ends the game there (``cut_illegal_king``);
    - then, while the last move is a king move to one of those squares, legal
      or not, it is removed, at most twice (``stripped``).

    ``board`` is left at the final position after both.
    """
    board = board or chess.Board()
    ucis: list[str] = []
    clocks: list[int | None] = []
    central: list[bool] = []
    result_token = None
    depth = 0
    cut = False
    tokens = TOKEN_RE.findall(movetext)
    for t_i, tok in enumerate(tokens):
        c = tok[0]
        if c == "{":
            if depth == 0 and ucis and clocks[-1] is None and not cut:
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
        if cut:
            continue
        san = ANNOT_RE.sub("", MOVENUM_RE.sub("", tok))
        if not san:
            continue
        if san in ("--", "Z0", "0000", "null"):
            raise GameError("null_move", san)
        try:
            move = board.parse_san(san)
        except chess.IllegalMoveError as e:
            rest = sum(_is_san_token(x) for x in _depth0(tokens[t_i + 1:]))
            if _central_king_san(san) and rest <= 1 and ucis:
                cut = True
                continue
            raise GameError("illegal_move", str(e)) from None
        except chess.AmbiguousMoveError as e:
            raise GameError("ambiguous_move", str(e)) from None
        except chess.InvalidMoveError as e:
            raise GameError("invalid_san", str(e)) from None
        central.append(board.piece_type_at(move.from_square) == chess.KING and move.to_square in CENTRE)
        ucis.append(move.uci())
        clocks.append(None)
        board.push(move)
    if depth:
        raise GameError("malformed_movetext", "unclosed (")
    stripped = 0
    while stripped < 2 and central and central[-1]:
        board.pop()
        ucis.pop()
        clocks.pop()
        central.pop()
        stripped += 1
    return Movetext(ucis, clocks, result_token, cut, stripped)


def _depth0(tokens: list[str]) -> list[str]:
    """The tokens outside variations."""
    out, depth = [], 0
    for t in tokens:
        if t == "(":
            depth += 1
        elif t == ")":
            depth -= 1
        elif depth == 0:
            out.append(t)
    return out


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


def dedup_key(white: str, black: str, year: int | None, moves_uci: str) -> str:
    """The game_id: 16 hex of sha1 over both surnames, the year and the moves. No source in it."""
    return sha_hex(surname(white), surname(black), str(year or ""), moves_uci)


# tags whose presence decides between two copies with equal clocks
TAG_COLUMNS = ("date", "white_elo", "black_elo", "white_fide_id", "black_fide_id", "white_title", "black_title", "time_control_raw")


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
    reason = source.drop_reason(h)
    if reason:
        raise GameError(reason)
    if "BOT" in (h.get("WhiteTitle", "").upper(), h.get("BlackTitle", "").upper()):
        raise GameError("engine", "BOT title")

    result = h.get("Result", "").strip()
    if result not in RESULTS:
        raise GameError("result_not_final" if result == "*" else "result_other", result)

    board = chess.Board()
    mt = parse_movetext(movetext, board)
    ucis, clocks, token = mt.ucis, mt.clocks, mt.result_token
    if not ucis:
        raise GameError("no_moves")
    if token is not None and token != result:
        raise GameError("result_token_mismatch", f"{token} vs {result}")
    outcome = board.outcome()
    if outcome is not None and outcome.termination in (chess.Termination.CHECKMATE, chess.Termination.STALEMATE):
        if outcome.result() != result:
            raise GameError("result_contradicts_final_position", f"{outcome.result()} vs {result}")

    n_clk = sum(c is not None for c in clocks)
    tempo = decide_tempo(h.get("TimeControl"), clocks, names, h.get("EventType"), source.default_tempo)
    white, black = h.get("White", "?"), h.get("Black", "?")
    date, year = _date(h)
    moves_uci = " ".join(ucis)
    key = dedup_key(white, black, year, moves_uci)
    row = {
        "game_id": key,
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
        "_dedup_key": key,
        "_clock_share": n_clk / len(ucis),
        "_priority": source.priority,
        "_nodes": source.tournament_nodes(h),
        "_tempo_note": tempo.note,
        "_broadcast_name": h.get("BroadcastName"),
        "_game_url": h.get("GameURL") or h.get("BroadcastURL"),
        "_result_token_missing": token is None,
        "_round_key": source.round_key(h),
        "_cut_king_ending": mt.cut_illegal_king,
        "_stripped_central_king": mt.stripped,
    }
    row["_n_tags"] = sum(row[c] is not None for c in TAG_COLUMNS)
    return row


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


def parse_file(args: tuple[str, str, int, str | None]) -> tuple[str, pl.DataFrame | int, Counter, dict[str, list[str]], str]:
    """Worker: all games of one file. Also returns the file's encoding (``utf-8`` or ``latin-1``).

    With ``out``, the parsed rows (with ``_id`` for this ``part``) are written
    there and their number returned instead of the frame.
    """
    source_name, path_s, part, out = (*args, 0, None)[:4] if len(args) == 2 else args
    path = Path(path_s)
    if path.suffix == ".zst":
        texts, encoding = iter_file_games(path), "utf-8"
    else:
        text, encoding = decode(read_bytes(path))
        texts = iter_game_texts(io.StringIO(text, newline=""))
    df, drops, examples = parse_texts(texts, SOURCES[source_name], path.name)
    if out is None:
        return path_s, df, drops, examples, encoding
    with_ids(df, part).write_parquet(out, compression="zstd")
    _drop_cache(Path(out))
    return path_s, len(df), drops, examples, encoding


def _drop_cache(path: Path) -> None:
    """Write a file's dirty pages and drop it from the page cache, which tnh-run counts as memory."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


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
    "_round_key": pl.String,
    "_cut_king_ending": pl.Boolean,
    "_stripped_central_king": pl.Int32,
    "_n_tags": pl.Int32,
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


def apply_round_tempo(df: pl.DataFrame) -> pl.DataFrame:
    """``tempo_source = round``: a game whose tempo is only from its name, a default or
    nothing takes the tempo of its broadcast round, when every other game of the round
    with a tempo from its clocks, tag or EventType has that same tempo."""
    strong = ~pl.col("tempo_source").is_in(WEAK_SOURCES)
    rounds = (
        df.select("_round_key", "tempo", "tempo_source")
        .filter(pl.col("_round_key").is_not_null() & strong)
        .group_by("_round_key")
        .agg(pl.col("tempo").unique().alias("_round_tempi"))
        .filter(pl.col("_round_tempi").list.len() == 1)
        .select("_round_key", pl.col("_round_tempi").list.first().alias("_round_tempo"))
    )
    # a narrow frame in the row order of df, so that only three columns of df are replaced
    rt = df.select("_round_key").join(rounds, on="_round_key", how="left", maintain_order="left")["_round_tempo"]
    take = rt.is_not_null() & df["tempo_source"].is_in(WEAK_SOURCES)
    return df.with_columns(
        tempo=pl.when(take).then(rt).otherwise(pl.col("tempo")),
        tempo_source=pl.when(take).then(pl.lit("round")).otherwise(pl.col("tempo_source")),
        _tempo_note=pl.when(take).then(pl.col("_tempo_note") + "_then_round_" + rt).otherwise(pl.col("_tempo_note")),
    )


# The columns the de-duplication plan needs; the wide ones (moves, clocks) stay where they are.
NARROW = ["_id", "_dedup_key", "_clock_share", "_n_tags", "_priority", "_game_url", "_file", "_nodes", "source",
          "_round_key", "tempo", "tempo_source", "_tempo_note", "date", "year", "white", "black", "n_plies",
          "_moves_hash"]
# What the plan decides per kept game, by _id
PLAN_COLUMNS = ["_id", "tournament_id", "split", "_n_copies", "_copy_sources", "_merged_near_copy_ids",
                "tempo", "tempo_source", "_tempo_note"]


def with_ids(df: pl.DataFrame, part: int = 0) -> pl.DataFrame:
    """Add ``_id`` (part << 32 | row) and ``_moves_hash`` to parsed rows."""
    return df.with_columns(
        (pl.int_range(pl.len(), dtype=pl.UInt64) + (part << 32)).alias("_id"),
        pl.col("moves_uci").hash(seed=20261111).alias("_moves_hash"),
    )


def plan(narrow: pl.DataFrame, fetch_moves, merge_near_copies: bool = True) -> tuple[pl.DataFrame, int, dict]:
    """The de-duplication plan, from the narrow columns of every parsed game.

    Returns (one row per kept game with ``PLAN_COLUMNS``, in game_id order;
    the number of exact duplicates removed; info). ``fetch_moves(ids)`` gives
    ``{_id: moves_uci}`` for the few games whose moves are compared as strings.

    - Tempo by broadcast round first (``apply_round_tempo``), over all copies.
    - Exact duplicates (same key): the copy with the most clock comments wins,
      then more tags filled in, then the source priority, then game URL and
      file, so the choice is stable.
    - Near copies (``near_copy_links``): their tournaments are joined; with
      ``merge_near_copies`` the copy with clocks is kept, else the longer one,
      and the removed copies' ids are listed (``_merged_near_copy_ids``). The
      kept game keeps its own ``game_id``.
    - A tournament is a connected component of the tournament nodes of all
      copies, named by its smallest node; the split is a hash of it.
    """
    narrow = apply_round_tempo(narrow)
    keys = narrow.select(
        "_id", "_dedup_key", "_clock_share", "_n_tags", "_priority", "_game_url", "_file", "_nodes", "source",
    ).sort(
        ["_dedup_key", "_clock_share", "_n_tags", "_priority", "_game_url", "_file"],
        descending=[False, True, True, False, False, False],
        nulls_last=True,
    )
    copies = keys.group_by("_dedup_key", maintain_order=True).agg(
        pl.col("_id").first().alias("_id"),
        pl.col("_nodes").list.explode(keep_nulls=False, empty_as_null=False).alias("_all_nodes"),
        pl.len().alias("_n_copies"),
        pl.col("source").unique().sort().alias("_copy_sources"),
    )
    del keys
    n_dup = len(narrow) - len(copies)

    uf = UnionFind()
    no_nodes = 0
    all_nodes = copies["_all_nodes"].to_list()
    for nodes in all_nodes:
        if not nodes:
            no_nodes += 1
            continue
        for other in nodes[1:]:
            uf.union(nodes[0], other)
    if no_nodes:
        raise RuntimeError(f"{no_nodes} games have no tournament node; the source's tournament_nodes is incomplete")

    kept = copies.select("_id", "_n_copies", "_copy_sources").join(
        narrow.drop("_nodes"), on="_id", how="left", maintain_order="left")
    links = near_copy_links(kept, fetch_moves)
    for a, b in links.select("i", "j").iter_rows():
        uf.union(all_nodes[a][0], all_nodes[b][0])
    info = {"near_copy_links": dict(links.group_by("kind").len().iter_rows())}
    tid = pl.Series("tournament_id", [uf.find(nodes[0]) for nodes in all_nodes], dtype=pl.String)
    del all_nodes, uf, copies

    kept = kept.with_columns(
        tid,
        tid.map_elements(split_of, return_dtype=pl.String).alias("split"),
        pl.lit([], dtype=pl.List(pl.String)).alias("_merged_near_copy_ids"),
    )
    if merge_near_copies and links.height:
        kept, removed = merge_near_copy_groups(kept, links)
        info["near_copies_removed"] = removed
    return kept.select(PLAN_COLUMNS), n_dup, info


def apply_plan(wide: pl.DataFrame, planned: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Parsed rows (any subset) -> (games, extra) for those of them that the plan keeps, in plan order."""
    kept = planned.join(wide.drop("tempo", "tempo_source", "_tempo_note"), on="_id", how="inner", maintain_order="left")
    games = kept.select([pl.col(c).cast(t) for c, t in GAMES_SCHEMA.items()])
    extra = kept.select(
        "game_id",
        pl.col("_broadcast_name").alias("broadcast_name"),
        pl.col("_game_url").alias("game_url"),
        pl.col("_file").alias("file"),
        pl.col("_tempo_note").alias("tempo_note"),
        pl.col("_clock_share").alias("clock_share"),
        pl.col("_n_copies").alias("n_copies"),
        pl.col("_copy_sources").alias("copy_sources"),
        pl.col("_result_token_missing").alias("result_token_missing"),
        pl.col("_cut_king_ending").alias("cut_illegal_central_king"),
        pl.col("_stripped_central_king").alias("stripped_central_king_moves"),
        pl.col("_merged_near_copy_ids").alias("merged_near_copy_ids"),
    )
    return games, extra


def assemble(df: pl.DataFrame, merge_near_copies: bool = True) -> tuple[pl.DataFrame, pl.DataFrame, int, dict]:
    """Parsed rows in memory -> (games, extra, number of exact duplicates removed, info).

    ``plan`` and ``apply_plan`` on one frame; ``build`` runs the same two on
    files, so that the wide rows are never all in memory.
    """
    if "_id" not in df.columns:
        df = with_ids(df)
    moves = dict(zip(df["_id"].to_list(), df["moves_uci"].to_list(), strict=True))
    planned, n_dup, info = plan(df.select(NARROW), lambda ids: {i: moves[i] for i in ids}, merge_near_copies)
    games, extra = apply_plan(df, planned)
    if games["game_id"].n_unique() != len(games):
        raise RuntimeError("game_id is not unique after de-duplication")
    return games, extra, n_dup, info


def merge_near_copy_groups(kept: pl.DataFrame, links: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Keep one game of each group of near copies: the one with clocks, else the longest.

    ``links`` holds row numbers of ``kept``. Groups are the connected
    components of the links. Returns the kept frame without the other copies
    (the winner listing their ids and the union of their sources) and counts.
    """
    parent = list(range(kept.height))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in links.select("i", "j").iter_rows():
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    members = sorted({int(x) for pair in links.select("i", "j").iter_rows() for x in pair})
    comp = pl.DataFrame({"_r": members, "_comp": [find(m) for m in members]}, schema={"_r": pl.UInt32, "_comp": pl.UInt32})
    cand = comp.join(
        kept.select(pl.int_range(pl.len(), dtype=pl.UInt32).alias("_r"), pl.col("_dedup_key").alias("game_id"),
                    "_clock_share", "n_plies", "_n_tags", "_priority", "_game_url", "_file", "_copy_sources"),
        on="_r",
    ).sort(["_comp", "_clock_share", "n_plies", "_n_tags", "_priority", "_game_url", "_file"],
           descending=[False, True, True, True, False, False, False], nulls_last=True)
    groups = cand.group_by("_comp", maintain_order=True).agg(
        pl.col("_r").first().alias("_win"),
        pl.col("game_id").slice(1).alias("_merged"),
        pl.col("_r").slice(1).alias("_lose"),
        pl.col("_copy_sources").list.explode().unique().sort().alias("_sources"),
    )
    lose = groups["_lose"].explode().drop_nulls()
    upd = groups.select(pl.col("_win").alias("_r"), "_merged", "_sources")
    kept = kept.with_columns(pl.int_range(pl.len(), dtype=pl.UInt32).alias("_r")).join(
        upd, on="_r", how="left", maintain_order="left")
    kept = kept.with_columns(
        _merged_near_copy_ids=pl.coalesce("_merged", "_merged_near_copy_ids"),
        _copy_sources=pl.coalesce("_sources", "_copy_sources"),
    ).filter(~pl.col("_r").is_in(lose.implode())).drop("_r", "_merged", "_sources")
    removed = {"games_removed": int(lose.len()), "groups": groups.height}
    return kept, removed


NEAR_COPY_MIN_PLIES = 20


def near_copy_links(kept: pl.DataFrame, fetch_moves=None) -> pl.DataFrame:
    """Pairs of kept games (row numbers ``i``, ``j`` of ``kept``) that are very likely the same game.

    - ``same_date_moves``: same date and the same moves, at least 20 plies,
      under different surnames ("Gareev,T" and "Gareyev, Timur"; "Yu Yangyi"
      and "Yangyi Yu"). Compared on a hash of the moves.
    - ``truncated``: same date and surnames, and one move list is a prefix of
      the other, the shorter at least 20 plies (a relay that stopped early, or
      a last move added by hand). Only games that share date and surnames
      with another are fetched (``fetch_moves``) and compared as strings. A
      truncated copy with two different longer continuations is left alone.

    ``kept`` needs ``date``, ``white``, ``black``, ``n_plies`` and either
    ``_moves_hash`` and ``_id`` with ``fetch_moves``, or ``moves_uci``.
    """
    if "_moves_hash" not in kept.columns:
        kept = kept.with_columns(pl.col("moves_uci").hash(seed=20261111).alias("_moves_hash"))
    if fetch_moves is None:
        kept = kept.with_columns(pl.int_range(pl.len(), dtype=pl.UInt64).alias("_id")) if "_id" not in kept.columns else kept
        table = dict(zip(kept["_id"].to_list(), kept["moves_uci"].to_list(), strict=True))
        fetch_moves = lambda ids: {i: table[i] for i in ids}
    k = kept.select(
        pl.int_range(pl.len(), dtype=pl.UInt32).alias("i"), "_id", "date", "_moves_hash", "n_plies",
        pl.col("white").map_elements(surname, return_dtype=pl.String).alias("ws"),
        pl.col("black").map_elements(surname, return_dtype=pl.String).alias("bs"),
    ).filter(pl.col("date").is_not_null() & (pl.col("n_plies") >= NEAR_COPY_MIN_PLIES))
    same = (
        k.group_by("date", "_moves_hash", "n_plies").agg(pl.col("i"))
        .filter(pl.col("i").list.len() > 1)
        .select(pl.col("i").list.first().alias("i"), pl.col("i").list.slice(1).alias("j"))
        .explode("j")
        .with_columns(kind=pl.lit("same_date_moves"))
    )
    groups = k.group_by("date", "ws", "bs").agg("i", "_id").filter(pl.col("i").list.len() > 1)
    moves_of = fetch_moves(groups["_id"].explode().to_list()) if groups.height else {}
    pairs = []
    for ids, gids in groups.select("i", "_id").iter_rows():
        moves = [moves_of[g] for g in gids]
        order = sorted(range(len(ids)), key=lambda x: len(moves[x]))
        for x_pos, x in enumerate(order):
            longer = [y for y in order[x_pos + 1:] if moves[y].startswith(moves[x] + " ")]
            # a truncated copy belongs to a longer game only when the longer copies are one
            # game (each a prefix of the next); two different continuations stay apart
            if all(moves[b].startswith(moves[a] + " ") or moves[b] == moves[a] for a, b in itertools.pairwise(longer)):
                pairs.extend((ids[x], ids[y]) for y in longer)
    trunc = pl.DataFrame(pairs, schema={"i": pl.UInt32, "j": pl.UInt32}, orient="row").with_columns(kind=pl.lit("truncated"))
    return pl.concat([same.select("i", "j", "kind"), trunc])


def king_signal_counts(games: pl.DataFrame, extra: pl.DataFrame) -> dict:
    """Games cut at an illegal central king move, and games with legal ones stripped, by source."""
    x = extra.join(games.select("game_id", "source"), on="game_id")
    return {
        "cut_at_illegal_central_king_move": dict(x.filter(pl.col("cut_illegal_central_king")).group_by("source").len().iter_rows()),
        "stripped_legal_central_king_move": dict(
            x.filter(pl.col("stripped_central_king_moves") > 0).group_by("source").len().iter_rows()),
    }


def build(sources: list[str], workers: int, limit_files: int | None, merge_near_copies: bool = True,
          out_dir: Path | None = None) -> None:
    """Parse every file into a part on disk, plan the de-duplication on the narrow
    columns, then write the kept games part by part. The wide rows (moves,
    clocks) are never all in memory."""
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
    work = DATA_ROOT / "scratch" / "parse" / "parts"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    print(f"parsing {len(tasks)} files with {workers} workers into {work}", flush=True)
    os.environ["POLARS_MAX_THREADS"] = "1"  # read by the spawned workers; the worker count is the CPU budget

    source_of = {path: name for name, path in tasks}
    drops: Counter = Counter()
    drops_by_source: dict[str, Counter] = {name: Counter() for name in sources}
    encodings: dict[str, Counter] = {name: Counter() for name in sources}
    examples: dict[str, list[str]] = {}
    n_parsed = 0
    args = [(name, path, part, str(work / f"part-{part:05d}.parquet")) for part, (name, path) in enumerate(tasks)]
    # spawn, not fork: forking after polars has started its thread pool can deadlock
    with multiprocessing.get_context("spawn").Pool(workers, maxtasksperchild=20) as pool:
        for k, (path, n, file_drops, file_ex, encoding) in enumerate(pool.imap_unordered(parse_file, args), 1):
            n_parsed += n
            drops += file_drops
            drops_by_source[source_of[path]] += file_drops
            encodings[source_of[path]][encoding] += 1
            for r, ex in file_ex.items():
                examples.setdefault(r, []).extend(ex[: 3 - len(examples.get(r, []))])
            if k % 50 == 0 or k == len(tasks):
                print(f"  {k}/{len(tasks)} files, {n_parsed} games kept, {sum(drops.values())} dropped, {time.time() - t0:.0f} s", flush=True)
    n_read = n_parsed + sum(drops.values())
    parts = sorted(work.glob("part-*.parquet"))

    narrow = pl.scan_parquet(parts).select(NARROW).collect()
    for p in parts:
        _drop_cache(p)
    print(f"narrow frame: {narrow.height:,} rows, {narrow.estimated_size() / 1e9:.2f} GB, {time.time() - t0:.0f} s", flush=True)
    before = narrow.group_by("source", "year").len().sort("source", "year")
    kept_before = narrow.group_by("source").len()

    def fetch_moves(ids: list[int]) -> dict[int, str]:
        want = pl.DataFrame({"_id": ids}, schema={"_id": pl.UInt64})
        got = pl.scan_parquet(parts).select("_id", "moves_uci").join(want.lazy(), on="_id", how="semi").collect()
        return dict(zip(got["_id"].to_list(), got["moves_uci"].to_list(), strict=True))

    planned, n_dup, info = plan(narrow, fetch_moves, merge_near_copies)
    del narrow
    print(f"plan: {planned.height:,} games kept, {time.time() - t0:.0f} s", flush=True)
    drops["duplicate"] += n_dup + info.get("near_copies_removed", {}).get("games_removed", 0)

    # write the kept games part by part
    out_dir = out_dir or CORPUS
    out_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, pq.ParquetWriter] = {}
    outs = {"games": out_dir / "games.parquet", "extra": out_dir / "games_extra.parquet"}
    planned = planned.with_columns((pl.col("_id") // (1 << 32)).alias("_part"))
    counts = {"source": Counter(), "tempo": Counter(), "split": Counter(), "cut": Counter(), "strip": Counter()}
    # games.parquet is sorted by game_id (readers rely on it to skip row groups). The kept rows of
    # each part are appended to 16 Arrow streams on the first hex digit of game_id; each stream is
    # then read, sorted and written in turn, so one sixteenth of the games is in memory at a time.
    # Streams rather than a file per part and bucket: that was 512 small files per part.
    bdir = work.parent / "buckets"
    if bdir.exists():
        shutil.rmtree(bdir)
    bdir.mkdir()
    streams: dict[tuple[str, str], tuple] = {}

    def flush_streams() -> None:
        for fh, _ in streams.values():
            fh.flush()
            os.fsync(fh.fileno())
            os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)

    for k, part_path in enumerate(parts, 1):
        part = int(part_path.stem.split("-")[1])
        g, e = apply_plan(pl.read_parquet(part_path), planned.filter(pl.col("_part") == part).drop("_part"))
        part_path.unlink()  # disk: the parts and the streams must not both be whole at once
        b = g["game_id"].str.slice(0, 1)
        for key, df in (("games", g), ("extra", e)):
            df = df.with_columns(_b=b)
            for (bk,), sub in df.partition_by("_b", as_dict=True).items():
                t = sub.drop("_b").to_arrow()
                if (key, bk) not in streams:
                    fh = open(bdir / f"{key}_{bk}.arrows", "wb")  # noqa: SIM115 (closed below)
                    streams[(key, bk)] = (fh, pa.ipc.new_stream(
                        fh, t.schema, options=pa.ipc.IpcWriteOptions(compression="zstd")))
                streams[(key, bk)][1].write_table(t)
        counts["source"].update(g["source"].to_list())
        counts["tempo"].update(g["tempo"].to_list())
        counts["split"].update(g["split"].to_list())
        counts["cut"].update(g.filter(e["cut_illegal_central_king"])["source"].to_list())
        counts["strip"].update(g.filter(e["stripped_central_king_moves"] > 0)["source"].to_list())
        if k % 50 == 0:
            flush_streams()  # their page cache counts against the job's memory under tnh-run
    flush_streams()
    for fh, w in streams.values():
        w.close()
        fh.close()
    for key in ("games", "extra"):
        for bk in "0123456789abcdef":
            path = bdir / f"{key}_{bk}.arrows"
            if not path.exists():
                continue
            with pa.memory_map(str(path)) as src:
                df = pl.from_arrow(pa.ipc.open_stream(src).read_all()).sort("game_id")
            _drop_cache(path)
            t = df.to_arrow()
            del df
            if key not in writers:
                writers[key] = pq.ParquetWriter(str(outs[key]) + ".tmp", t.schema, compression="zstd")
            writers[key].write_table(t, row_group_size=100_000)
            path.unlink()
    for key, w in writers.items():
        w.close()
        Path(str(outs[key]) + ".tmp").replace(outs[key])
        _drop_cache(outs[key])
    shutil.rmtree(work)
    shutil.rmtree(bdir, ignore_errors=True)
    n_kept = planned.height
    for name, n in kept_before.iter_rows():
        # copies of this source's games that lost to a copy from this or another source
        drops_by_source[name]["duplicate"] += n - counts["source"].get(name, 0)

    stats = {
        "games_read": n_read,
        "games_kept": n_kept,
        "dropped": dict(drops.most_common()),
        "dropped_by_source": {name: dict(c.most_common()) for name, c in drops_by_source.items()},
        "read_by_source": {name: sum(c.values()) + kept_before.filter(pl.col("source") == name)["len"].sum() - c["duplicate"]
                           for name, c in drops_by_source.items()},
        "file_encodings": {name: dict(c) for name, c in encodings.items()},
        "drop_examples": examples,
        "tournaments": planned["tournament_id"].n_unique(),
        "games_before_dedup": before.to_dicts(),
        # kept, not dropped: the arbiter's kings-to-the-centre signal at the end of a game
        "cut_at_illegal_central_king_move": dict(counts["cut"]),
        "stripped_legal_central_king_move": dict(counts["strip"]),
        **info,
        "seconds": round(time.time() - t0),
    }
    (out_dir / "parse_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(f"read {n_read} games, kept {n_kept} in {stats['tournaments']} tournaments, {time.time() - t0:.0f} s")
    print("dropped:")
    for reason, n in drops.most_common():
        print(f"  {reason:36s} {n:9d}   e.g. {examples.get(reason, [''])[0][:120]}")
    print("tempo:", dict(counts["tempo"]), "| split:", dict(counts["split"]))
    print(f"wrote {outs['games']}", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default=",".join(SOURCES), help="comma-separated source names (default: all)")
    ap.add_argument("--workers", type=int, default=2, help="parser processes (default 2; see AGENTS.md on the CPU budget)")
    ap.add_argument("--no-merge-near-copies", action="store_true", help="keep near copies as separate games (measurement)")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: corpus/)")
    ap.add_argument("--limit-files", type=int, default=None, help="only the first K files per source (testing)")
    args = ap.parse_args(argv)

    build(args.sources.split(","), args.workers, args.limit_files, not args.no_merge_near_copies, args.out)


if __name__ == "__main__":
    main()
