"""What is specific to one source of games.

A source names its raw files and says, for one game's tags, which tournament
nodes the game belongs to and whether the event is one to drop. Everything
else (move parsing, tempo, de-duplication, the split) is shared, so a second
source (TWIC, ...) is a new ``Source`` here and nothing more.

Tournament nodes are strings. ``tnh.data.parse`` joins every node a game
touches, and joins the nodes of all copies of a de-duplicated game, so a
tournament is a connected component. That makes ids coarse rather than fine:
two broadcasts of the same event, or a renamed broadcast, end up as one.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tnh.paths import RAW

# --- Events that are not over-the-board human chess --------------------------

# Engine events. TCEC and CCC games carry the title BOT in most cases; the
# names catch the rest (e.g. "TCEC Cup 14" rounds without titles).
ENGINE_RE = re.compile(
    r"\btcec\b|\bccc\b|computer chess|\bengine|\bleela knight odds|\bstockfish\b|\blc0\b",
    re.IGNORECASE,
)
SIMUL_RE = re.compile(r"\bs[z]?imul(?:tan|taneous|táneas?|tán|s)?\b|\bszimultán\b", re.IGNORECASE)
# Online play relayed as a broadcast. Only names that say so; Chess.com-sponsored
# OTB events ("FIDE Chess.com Grand Swiss") are kept.
ONLINE_RE = re.compile(
    r"\bonline\b|chess\.com (?:rapid|speed|classic|global|open|titled)|\bcct chess\.com\b"
    r"|\bspeed chess championship\b|\btitled (?:arena|tuesday)\b|\bchess\.com open\b|\bcorporate chess\b",
    re.IGNORECASE,
)


def event_drop_reason(names: list[str]) -> str | None:
    """``engine``, ``simul``, ``online`` or None, from event and broadcast names."""
    text = " || ".join(n for n in names if n)
    if ENGINE_RE.search(text):
        return "engine"
    if SIMUL_RE.search(text):
        return "simul"
    if ONLINE_RE.search(text):
        return "online"
    return None


# --- Lichess broadcasts --------------------------------------------------------

_ROUND_MARK_RE = re.compile(
    r"\s*[-–—:,(]?\s*\b(?:round|rd|game|runde|ronda|r)\s*\.?\s*\d+(?:\s*[-–/]\s*\d+)?\b\)?",
    re.IGNORECASE,
)
_SLUG_RE = re.compile(r"/broadcast/([^/]+)/([^/]+)/([^/]+)$")


def normalize_tour_name(name: str) -> str:
    """The event part of a broadcast name: before the first ``|``, round markers removed.

    "44th Chess Olympiad 2022 | Open | Boards 1-40" -> "44th chess olympiad 2022"
    "2020 Gibraltar Masters - Round 6"            -> "2020 gibraltar masters"
    """
    head = name.split("|")[0]
    head = _ROUND_MARK_RE.sub(" ", head)
    return re.sub(r"\s+", " ", head).strip(" -–:,").lower()


def lichess_broadcast_nodes(h: dict[str, str]) -> list[str]:
    """Tournament nodes of one broadcast game.

    - the normalized broadcast name (the Lichess "tour", with its group);
    - the tour slug from BroadcastURL, which ties a renamed tour together;
    - for the 2020-21 broadcasts with one broadcast per round or per game
      (URL ``/broadcast/<slug>/-/<id>``), the Event tag. In later broadcasts
      the Event tag is often per game ("Round 6: A - B") or generic ("Open"),
      and linking on it joins unrelated events into one component of a fifth
      of the corpus, so it is not used there.
    """
    nodes = []
    name = h.get("BroadcastName", "")
    if name:
        nodes.append("lb/name:" + normalize_tour_name(name))
    m = _SLUG_RE.search(h.get("BroadcastURL", ""))
    if m:
        nodes.append("lb/slug:" + _ROUND_MARK_RE.sub("", m.group(1)).strip("-"))
        event = h.get("Event", "").strip().lower()
        if m.group(2) == "-" and event not in ("", "?", "-"):
            nodes.append("lb/event:" + event)
    return nodes


@dataclass(frozen=True)
class Source:
    name: str
    priority: int  # lower wins a tie between copies of one game with equal clock coverage
    files: Callable[[], list[Path]]
    tournament_nodes: Callable[[dict[str, str]], list[str]]
    event_names: Callable[[dict[str, str]], list[str]]  # names searched for drops and tempo words


def _lichess_broadcast_files() -> list[Path]:
    return sorted((RAW / "lichess_broadcast").glob("lichess_db_broadcast_*.pgn.zst"))


SOURCES: dict[str, Source] = {
    "lichess_broadcast": Source(
        name="lichess_broadcast",
        priority=0,
        files=_lichess_broadcast_files,
        tournament_nodes=lichess_broadcast_nodes,
        event_names=lambda h: [h.get("BroadcastName", ""), h.get("Event", "")],
    ),
}
