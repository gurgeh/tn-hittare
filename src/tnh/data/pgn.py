"""Streaming PGN reading: split a file into games, read the tag pairs.

The move text is parsed later, in a worker process, with python-chess. Here
the file is only cut into one string per game, which is fast, so the parsing
can be spread over all cores.
"""

from __future__ import annotations

import io
import re
from collections.abc import Iterator
from pathlib import Path

import zstandard

TAG_RE = re.compile(r'^\[([A-Za-z0-9_]+)\s+"(.*)"\]\s*$')


def open_text(path: Path) -> io.TextIOBase:
    """Open a .pgn or .pgn.zst as text."""
    if path.suffix == ".zst":
        fh = path.open("rb")
        reader = zstandard.ZstdDecompressor().stream_reader(fh, read_size=1 << 20)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace", newline="")
    return path.open("r", encoding="utf-8", errors="replace", newline="")


def iter_game_texts(lines: Iterator[str] | io.TextIOBase) -> Iterator[str]:
    """Yield the text of each game: its tag lines, a blank line, its move text.

    A game starts at a tag line that follows move text. Lichess writes each
    game's move text on one line, but this does not depend on it.
    """
    buf: list[str] = []
    in_moves = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]") and TAG_RE.match(stripped):
            if in_moves:
                yield "".join(buf)
                buf = []
                in_moves = False
            buf.append(line)
        elif stripped:
            in_moves = True
            buf.append(line)
        else:
            buf.append(line)
    if buf and any(s.strip() for s in buf):
        yield "".join(buf)


def iter_file_games(path: Path) -> Iterator[str]:
    with open_text(path) as f:
        yield from iter_game_texts(f)


def split_game(text: str) -> tuple[dict[str, str], str]:
    """Tag pairs (verbatim values, PGN escapes undone) and the move text."""
    headers: dict[str, str] = {}
    moves: list[str] = []
    for line in text.split("\n"):  # not splitlines(): it also splits on U+0085, found in mis-encoded tags
        s = line.strip()
        m = TAG_RE.match(s) if s.startswith("[") else None
        if m and not moves:
            headers[m.group(1)] = m.group(2).replace('\\"', '"').replace("\\\\", "\\")
        elif s:
            moves.append(s)
    return headers, "\n".join(moves)
