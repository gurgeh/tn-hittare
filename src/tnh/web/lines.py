"""Saved lines: the page's whole move tree, variations and comments, as one PGN file per name.

The files are standard PGN, `Event` = the name, `Date` = the save date, `FEN` and `SetUp`
when the tree does not start from the standard position. A file's name is a slug of the line's
name; the exact name is the `Event` header, and the lines are listed and found by it. Two
names with the same slug are refused, never merged. Writes go to a temporary file in the same
directory and are renamed over the target, so a file is either the old line or the new one.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import os
import re
import tempfile
import unicodedata
from pathlib import Path

import chess
import chess.pgn

from tnh.web.pgn import parse_pgn, tree_pgn

DEFAULT_DIR = Path("~/.local/share/tn-hittare/lines")
MAX_NAME = 100
PREVIEW_PLIES = 8


class LineError(Exception):
    """A refused request: `code` is "invalid", "exists", "collision" or "missing"."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def check_name(name: object) -> str:
    """The name as given, or LineError("invalid"): 1 to MAX_NAME characters, no path separators,
    no leading dot, no control characters, no quotes or backslashes (a PGN tag value cannot hold
    them portably), no leading or trailing spaces."""
    if not isinstance(name, str) or not name:
        raise LineError("invalid", "a line needs a name")
    if len(name) > MAX_NAME:
        raise LineError("invalid", f"a name has at most {MAX_NAME} characters")
    if name != name.strip():
        raise LineError("invalid", "a name cannot start or end with a space")
    if name.startswith("."):
        raise LineError("invalid", "a name cannot start with a dot")
    bad = sorted({c for c in name if c in '/\\"' or unicodedata.category(c) in ("Cc", "Cf", "Zl", "Zp")})
    if bad:
        raise LineError("invalid", "a name cannot hold " + " ".join(repr(c) for c in bad))
    return name


def slug(name: str) -> str:
    """The file stem for a name: ASCII letters and digits joined by dashes ("Sämisch: 5...c5"
    -> "samisch-5-c5"); a name with none of them gets "line-" and a hash of the name."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")[:80].strip("-")
    return s or "line-" + hashlib.sha1(name.encode()).hexdigest()[:10]


def _name_of(path: Path, headers) -> str:
    """The line's name: its Event, or the file's stem when the file has no real Event."""
    ev = headers.get("Event", "?") if headers is not None else "?"
    return path.stem if ev in ("", "?") else ev


def _saved(path: Path) -> str:
    return dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


class LineStore:
    def __init__(self, root: Path):
        self.root = Path(root).expanduser()

    def _path(self, name: str) -> Path:
        return self.root / (slug(name) + ".pgn")

    def _find(self, name: str) -> Path:
        check_name(name)
        path = self._path(name)
        if path.is_file():
            with path.open(encoding="utf-8") as f:
                if _name_of(path, chess.pgn.read_headers(f)) == name:
                    return path
        raise LineError("missing", f"no saved line named {name!r}")

    def list(self) -> list[dict]:
        """Every line: name, saved time, plies in the main line, the main line's first moves.
        A file that cannot be read is listed with its error."""
        out = []
        if not self.root.is_dir():
            return out
        for path in sorted(self.root.glob("*.pgn")):
            entry = {"file": path.name, "saved": _saved(path)}
            try:
                game = chess.pgn.read_game(io.StringIO(path.read_text(encoding="utf-8")))
                if game is None:
                    raise ValueError("no game in the file")
                main = list(game.mainline_moves())
                entry |= {"name": _name_of(path, game.headers), "plies": len(main),
                          "preview": game.board().variation_san(main[:PREVIEW_PLIES]) if main else "",
                          "error": "; ".join(str(e) for e in game.errors[:3]) or None}
            except (OSError, ValueError, UnicodeDecodeError) as e:
                entry |= {"name": path.stem, "plies": None, "preview": "", "error": f"{type(e).__name__}: {e}"}
            out.append(entry)
        out.sort(key=lambda e: e["name"].casefold())
        return out

    def load(self, name: str) -> dict:
        path = self._find(name)
        d = parse_pgn(path.read_text(encoding="utf-8"))
        return d | {"name": name, "saved": _saved(path)}

    def save(self, name: str, tree: dict, overwrite: bool = False, today: dt.date | None = None) -> dict:
        """Writes the tree under `name`. An existing line of that name needs `overwrite`; a line
        of another name with the same file name is refused either way."""
        check_name(name)
        path = self._path(name)
        if path.exists():
            with path.open(encoding="utf-8") as f:
                other = _name_of(path, chess.pgn.read_headers(f))
            if other != name:
                raise LineError("collision", f"{name!r} would be saved as {path.name}, which holds the line "
                                             f"{other!r}; choose another name")
            if not overwrite:
                raise LineError("exists", f"a line named {name!r} exists")
        date = (today or dt.date.today()).strftime("%Y.%m.%d")
        text = tree_pgn(tree, headers={"Event": name, "Date": date})
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            os.unlink(tmp)
            raise
        return {"name": name, "file": path.name, "saved": _saved(path)}

    def delete(self, name: str) -> dict:
        path = self._find(name)
        path.unlink()
        return {"name": name, "file": path.name}
