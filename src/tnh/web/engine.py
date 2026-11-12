"""The engine section: one UCI engine at a time, analysing the page's position until it changes.

`EngineManager` keeps at most one engine process (Stockfish, Lc0 or preppy, the repertoire
engine `tnh.engine.uci`). The page tells it the position, the number of lines and the
engine's settings (`go`); it stops the running search, waits for its `bestmove`, sets up the
new search and sends `go infinite`. The page polls `snapshot` a few times a second.

Info strings are shown only once the search has reported a line (Stockfish's start-up notes
come before), and errors always.

Stale results: every search has a generation number. Output is attributed to the generation
current when it is read, and the generation only moves on after the old search's `bestmove`
has been read and before the new `go` is sent, so a line of the previous position can never
be reported as a line of the new one. The page checks the position as well.

The engine quits when the page selects Off, when another engine is selected, when no page
has asked for `idle_s` seconds, and when the server shuts down. A process that dies or does
not answer is reported as an error with the end of its stderr; nothing is retried.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import chess


class EngineError(RuntimeError):
    pass


@dataclass
class EngineSpec:
    """How to start an engine and what to set on it."""

    name: str
    label: str
    cmd: list[str]
    options: dict[str, str] = field(default_factory=dict)       # set once after `uci`
    settings: tuple[str, ...] = ()                              # options the page may set (whitelist)
    gpu_mb: int = 0                                             # free GPU memory needed to start
    short: str = ""                                             # column header
    # Turns the page's settings and its "me" side into the options to send (Lc0's contempt).
    derive: Callable[[dict, str], dict] | None = None
    # Options whose change must also clear the engine's search tree (`ucinewgame`) before the
    # next search. Lc0 shifts leaf values by its contempt when it evaluates them and shows the
    # score with the inverse shift, so a tree kept from another contempt shows wrong scores.
    # Never preppy's: it keeps its graph across positions and settings on purpose.
    reset_on: tuple[str, ...] = ()
    # Options the engine accepts but does not declare in its `uci` answer (Lc0's hidden ones),
    # with the declaration the page should use.
    hidden: dict[str, dict] = field(default_factory=dict)

    def available(self) -> str | None:
        """None when it can start, else why not."""
        exe = Path(self.cmd[0])
        if exe.is_absolute() and not exe.exists():
            return f"{exe} does not exist"
        return None


def default_specs(*, threads: int = 4, hash_mb: int = 256, lc0_minibatch: int = 12) -> dict[str, EngineSpec]:
    """Stockfish, Lc0 and preppy as the page offers them.

    - Stockfish: the binary `tnh.models.stockfish` uses, `threads` threads (the owner's rule:
      at most 8 engine and worker threads on the machine, so 4 unless he asks), 256 MB hash.
    - Lc0: the install `tnh.models.lc0` uses, the BT4 network, its CUDA backend with batches of
      `lc0_minibatch` (12: larger batches gave a few wrong answers per thousand on this GPU),
      no prefetch, two search threads.
    - preppy: `python -m tnh.engine.uci`, the repertoire engine; the page may set the options in
      `settings`.
    """
    from tnh.models import lc0
    from tnh.models.stockfish import STOCKFISH

    return {
        "stockfish": EngineSpec("stockfish", "Stockfish", [str(STOCKFISH)],
                                {"Threads": str(threads), "Hash": str(hash_mb), "UCI_ShowWDL": "true"}, short="SF"),
        "lc0": EngineSpec("lc0", "Lc0", [str(lc0.binary())],
                          {"WeightsFile": str(lc0.network_path()), "Backend": "cuda-auto",
                           "BackendOptions": f"max_batch={lc0_minibatch}", "MinibatchSize": str(lc0_minibatch),
                           "MaxPrefetch": "0", "Threads": "2", "UCI_ShowWDL": "true",
                           "WDLDrawRateReference": "0.62"},          # BT4's draw rate, per the Lc0 guide
                          ("Contempt", "WDLCalibrationElo", "WDLEvalObjectivity", "WDLContemptAttenuation",
                           "ContemptMode", "MovesLeftThreshold"), gpu_mb=2500, short="Lc0",
                          derive=lc0_contempt, reset_on=LC0_RESET,
                          hidden={"WDLContemptAttenuation": {"name": "WDLContemptAttenuation", "type": "string",
                                                             "default": "1", "vars": []}}),
        "preppy": EngineSpec("preppy", "preppy", [sys.executable, "-m", "tnh.engine.uci"], {},
                             ("MyElo", "OppElo", "Side", "Tempo", "Price", "Cap", "Obvious", "Prepared", "DrawScore",
                              "Mode", "Alpha", "Blunder", "PriorMin", "MistakeFix"),
                             gpu_mb=2500, short="preppy"),
    }


LC0_RESET = ("Contempt", "ContemptMode", "WDLCalibrationElo", "WDLEvalObjectivity", "WDLContemptAttenuation",
             "MovesLeftThreshold")


def _unit(settings: dict, key: str, default: str) -> str:
    raw = str(settings.get(key, "") or "").strip() or default
    try:
        x = float(raw)
    except ValueError as e:
        raise EngineError(f"{key} must be a number from 0 to 1, not {raw!r}") from e
    if not 0 <= x <= 1:
        raise EngineError(f"{key} must be from 0 to 1, not {raw}")
    return raw


def lc0_contempt(settings: dict, me: str) -> dict:
    """Lc0's contempt settings from the page, after the Lc0 team's guide to WDL contempt
    (reference/lc0-sadler-wdl-contempt.md, "Suggested use for opening research"):

    - `Contempt` in Elo (empty or 0 = off), positive when "me" (the side at the bottom of the
      board) is the stronger side. On, `ContemptMode` is `white_side_analysis` or
      `black_side_analysis` for that side, never `play`, which would favour whichever side is
      to move; off, `ContemptMode disable` and an empty `Contempt`.
    - `WDLCalibrationElo` (0 = off), `WDLEvalObjectivity` and `WDLContemptAttenuation` (0-1,
      default 1) pass through.
    - `MovesLeftThreshold` 0.95 when |Contempt| is above 250 (the guide's advice for extreme
      values), else Lc0's default 0.8.

    `WDLDrawRateReference` 0.62 (BT4's draw rate) is set once at start, not here. A change of
    any of these clears Lc0's tree (`EngineSpec.reset_on`): it shifts leaf values by its
    contempt when it evaluates them, so a kept tree shows scores from another setting.
    """
    raw = str(settings.get("Contempt", "") or "").strip()
    try:
        c = float(raw) if raw else 0.0
    except ValueError as e:
        raise EngineError(f"Contempt must be a number of Elo, not {raw!r}") from e
    if me not in ("white", "black"):
        raise EngineError(f"me must be white or black, not {me!r}")
    elo = str(settings.get("WDLCalibrationElo", "") or "").strip() or "0"
    try:
        float(elo)
    except ValueError as e:
        raise EngineError(f"WDLCalibrationElo must be a number, not {elo!r}") from e
    out = {"Contempt": f"{c:g}" if c else "", "ContemptMode": f"{me}_side_analysis" if c else "disable",
           "WDLCalibrationElo": elo,
           "WDLEvalObjectivity": _unit(settings, "WDLEvalObjectivity", "1"),
           "WDLContemptAttenuation": _unit(settings, "WDLContemptAttenuation", "1"),
           "MovesLeftThreshold": "0.95" if abs(c) > 250 else "0.8"}
    return out


def lc0_from_ratings(white: int, black: int, me: str, tempo: str) -> dict:
    """Lc0's contempt settings from the two players, after the guide's recipe: the calibration
    Elo is the rating of "me" plus about 50 per doubling of thinking time from 3'+2" (+200
    for classical, 0 for blitz), and Contempt is the rating difference from my side. Equal
    ratings give Contempt 0: the draw rate of that level, no contempt."""
    if me not in ("white", "black"):
        raise EngineError(f"me must be white or black, not {me!r}")
    mine, theirs = (white, black) if me == "white" else (black, white)
    extra = 0 if tempo == "blitz" else 200
    return {"Contempt": str(int(mine - theirs)), "WDLCalibrationElo": str(int(mine + extra))}


def gpu_free_mb() -> int | None:
    """Free memory of the emptiest GPU, MB; None without nvidia-smi."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        return None
    if r.returncode != 0:
        raise EngineError(f"nvidia-smi failed: {r.stderr.strip()[:200]}")
    return max(int(x) for x in r.stdout.split())


def parse_option(line: str) -> dict | None:
    """`option name N type T [default D] [min a] [max b] [var v]*` as a dict."""
    t = line.split()
    if len(t) < 5 or t[0] != "option" or t[1] != "name" or "type" not in t:
        return None
    i = t.index("type")
    out: dict = {"name": " ".join(t[2:i]), "type": t[i + 1], "vars": []}
    k = i + 2
    keys = ("default", "min", "max", "var")
    while k < len(t):
        key = t[k]
        j = k + 1
        while j < len(t) and t[j] not in keys:
            j += 1
        val = " ".join(t[k + 1:j])
        if key == "var":
            out["vars"].append(val)
        elif key in keys:
            out[key] = val
        k = j
    return out


def parse_info(line: str) -> dict | None:
    """A UCI `info` line with a `pv` (or a `string`) as a dict; None for the rest."""
    t = line.split()
    if not t or t[0] != "info":
        return None
    if len(t) > 1 and t[1] == "string":
        return {"string": line.split(" ", 2)[2] if len(t) > 2 else ""}
    if "pv" not in t or "score" not in t:
        return None
    d: dict = {"multipv": 1}
    i = 1
    while i < len(t):
        k = t[i]
        if k in ("depth", "seldepth", "multipv", "nodes", "nps", "time", "hashfull", "tbhits"):
            d[k] = int(t[i + 1])
            i += 2
        elif k == "score":
            d["kind"], d["value"] = t[i + 1], int(t[i + 2])
            i += 3
            if i < len(t) and t[i] in ("lowerbound", "upperbound"):
                d["bound"] = t[i]
                i += 1
        elif k == "wdl":
            d["wdl"] = [int(x) for x in t[i + 1:i + 4]]
            i += 4
        elif k == "pv":
            d["pv"] = t[i + 1:]
            break
        else:
            i += 1
    return d


class _Process:
    """One engine process: a reader thread sorts its output into control lines and info."""

    def __init__(self, spec: EngineSpec, on_info) -> None:
        self.spec = spec
        try:
            self.proc = subprocess.Popen(spec.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True, bufsize=1)
        except OSError as e:
            raise EngineError(f"{spec.label} did not start: {e}") from e
        self.ctl: queue.Queue = queue.Queue()
        self.on_info = on_info
        self._stderr: list[str] = []
        threading.Thread(target=self._read, daemon=True, name=f"uci-{spec.name}").start()
        threading.Thread(target=self._read_err, daemon=True, name=f"uci-{spec.name}-err").start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("info"):
                self.on_info(line)
            else:
                self.ctl.put(line)
        self.ctl.put(None)          # end of output

    def _read_err(self) -> None:
        for line in self.proc.stderr:
            self._stderr.append(line.rstrip("\n"))
            del self._stderr[:-40]

    def stderr_tail(self) -> str:
        return " | ".join(self._stderr[-5:])

    def send(self, *lines: str) -> None:
        try:
            for ln in lines:
                self.proc.stdin.write(ln + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise EngineError(f"{self.spec.label} is gone ({e}); stderr: {self.stderr_tail()}") from e

    def wait_for(self, prefix: str, timeout: float) -> list[str]:
        """The control lines up to and including the first that starts with `prefix`."""
        seen = []
        end = time.monotonic() + timeout
        while True:
            left = end - time.monotonic()
            if left <= 0:
                raise EngineError(f"{self.spec.label}: no '{prefix}' within {timeout:g} s; stderr: {self.stderr_tail()}")
            try:
                line = self.ctl.get(timeout=left)
            except queue.Empty:
                continue
            if line is None:
                raise EngineError(f"{self.spec.label} exited (code {self.proc.poll()}); stderr: {self.stderr_tail()}")
            seen.append(line)
            if line.startswith(prefix):
                return seen

    def quit(self) -> None:
        if self.proc.poll() is None:
            try:
                self.send("stop", "quit")
                self.proc.wait(timeout=5)
            except (EngineError, subprocess.TimeoutExpired):
                self.proc.kill()
                self.proc.wait(timeout=5)


@dataclass
class _Search:
    gen: int
    fen: str
    moves: list[str]
    board: chess.Board
    multipv: int
    settings: dict


class EngineManager:
    """At most one engine, analysing the latest position the page asked for."""

    def __init__(self, specs: dict[str, EngineSpec], *, idle_s: float = 60.0, start_timeout: float = 300.0,
                 stop_timeout: float = 15.0, gpu_free=gpu_free_mb) -> None:
        self.specs = specs
        self.idle_s = idle_s
        self.start_timeout = start_timeout
        self.stop_timeout = stop_timeout
        self.gpu_free = gpu_free
        self._cmd = threading.Lock()        # one command at a time
        self._data = threading.Lock()       # the results, shared with the reader thread
        self._proc: _Process | None = None
        self._name = "off"
        self._options: dict[str, dict] = {}
        self._applied: dict[str, str] = {}
        self._searching = False
        self._search: _Search | None = None
        self._gen = 0
        self._lines: dict[int, dict] = {}
        self._strings: dict[str, str] = {}
        self._extra: dict[int, dict] = {}      # preppy's per-line info string, by multipv
        self._root: dict | None = None         # preppy's root info string
        self._phase = "off"                    # off, starting (uci, then the models load), running
        self._error: str | None = None
        self._touched = time.monotonic()
        self._closed = False
        threading.Thread(target=self._watch, daemon=True, name="uci-idle").start()

    # -- the engine's output --------------------------------------------------------

    def _on_info(self, line: str) -> None:
        d = parse_info(line)
        if d is None:
            return
        with self._data:
            if "string" in d:
                s = d["string"]
                if s.startswith("preppy multipv="):
                    kv = _kv(s)
                    kv["gen"] = self._gen
                    self._extra[int(kv["multipv"])] = kv
                    return
                if s.startswith("preppy root"):
                    self._root = _kv(s) | {"gen": self._gen}
                # Only what the search says, not start-up notes (Stockfish prints its threads and
                # network at the first go): a string counts once the search has reported a line;
                # errors always do.
                if "error" not in s.lower() and not any(x["gen"] == self._gen for x in self._lines.values()):
                    return
                key = " ".join(s.split()[:2])
                self._strings[key] = s
                return
            d["gen"] = self._gen
            d["at"] = time.monotonic()
            self._lines[d["multipv"]] = d

    # -- commands -----------------------------------------------------------------------

    def _start(self, name: str) -> None:
        spec = self.specs.get(name)
        if spec is None:
            raise EngineError(f"unknown engine {name!r}; known: off, {', '.join(self.specs)}")
        why = spec.available()
        if why:
            raise EngineError(f"{spec.label}: {why}")
        if spec.gpu_mb:
            free = self.gpu_free()
            if free is not None and free < spec.gpu_mb:
                raise EngineError(f"{spec.label} needs about {spec.gpu_mb} MB of GPU memory; {free} MB is free")
        self._phase = "starting"
        p = _Process(spec, self._on_info)
        try:
            p.send("uci")
            lines = p.wait_for("uciok", self.start_timeout)
            self._options = {o["name"]: o for o in (parse_option(x) for x in lines) if o}
            for k, d in spec.hidden.items():
                self._options.setdefault(k, d)
            for k, v in spec.options.items():
                p.send(f"setoption name {k} value {v}")
            p.send("isready")
            _check_accepted(spec, p.wait_for("readyok", self.start_timeout))
        except EngineError:
            p.quit()
            self._phase = "off"
            raise
        self._proc, self._name, self._applied, self._searching = p, name, {}, False
        self._phase = "running"

    def _stop_search(self) -> None:
        if self._proc is not None and self._searching:
            self._proc.send("stop")
            self._proc.wait_for("bestmove", self.stop_timeout)
            self._searching = False

    def _quit(self) -> None:
        if self._proc is not None:
            self._proc.quit()
        self._proc, self._name, self._searching, self._search = None, "off", False, None
        self._phase = "off"
        with self._data:
            self._gen += 1
            self._lines, self._strings, self._extra, self._root = {}, {}, {}, None

    def go(self, name: str, fen: str, moves: list[str], multipv: int, settings: dict | None = None,
           me: str = "white") -> int:
        """Analyse `fen` (reached by `moves` from the start, or from the FEN alone when `moves`
        is empty) with `name`, `multipv` lines and the page's `settings`. Returns the search's
        generation. `name` "off" quits the engine."""
        settings = dict(settings or {})
        with self._cmd:
            self._touched = time.monotonic()
            self._error = None
            if self._closed:
                raise EngineError("the engine manager is closed")
            try:
                if name != self._name:
                    self._quit()
                    if name == "off":
                        return self._gen
                    self._start(name)
                elif name == "off":
                    return self._gen
                board = _board(fen, moves)
                if board.is_game_over():
                    self._stop_search()
                    with self._data:
                        self._gen += 1
                        self._lines, self._strings, self._extra, self._root = {}, {}, {}, None
                        self._search = _Search(self._gen, fen, moves, board, multipv, settings)
                    return self._gen
                self._stop_search()
                spec = self.specs[name]
                if spec.derive is not None:
                    settings = spec.derive(settings, me)
                want = {"MultiPV": str(multipv)} if "MultiPV" in self._options else {}
                for k, v in settings.items():
                    if k in spec.settings and k in self._options:
                        want[k] = str(v)
                changed = {k: v for k, v in want.items() if self._applied.get(k) != v}
                for k, v in changed.items():
                    self._proc.send(f"setoption name {k} value {v}")
                if any(k in spec.reset_on for k in changed):
                    self._proc.send("ucinewgame")
                if changed:
                    self._proc.send("isready")
                    _check_accepted(spec, self._proc.wait_for("readyok", self.start_timeout))
                    self._applied.update(changed)
                with self._data:
                    self._gen += 1
                    self._lines, self._strings, self._extra, self._root = {}, {}, {}, None
                    self._search = _Search(self._gen, fen, moves, board, multipv, settings)
                    gen = self._gen
                pos = "position startpos moves " + " ".join(moves) if moves else f"position fen {fen}"
                self._proc.send(pos.strip(), "go infinite")
                self._searching = True
                return gen
            except EngineError as e:
                self._error = str(e)
                self._quit()
                raise

    def snapshot(self) -> dict:
        """The running search's lines, from White's side, best first."""
        self._touched = time.monotonic()
        with self._data:
            s = self._search
            out = {"engine": self._name, "phase": self._phase, "label": self.specs[self._name].label if self._name in self.specs else "Off",
                   "short": self.specs[self._name].short if self._name in self.specs else "",
                   "gen": self._gen, "error": self._error, "lines": [], "strings": [],
                   "fen": s.fen if s else None, "searching": self._searching,
                   "options": [self._options[k] for k in self.specs[self._name].settings if k in self._options]
                   if self._name in self.specs else [],
                   "applied": {k: v for k, v in self._applied.items()
                               if self._name in self.specs and k in self.specs[self._name].settings}}
            if s is None or s.gen != self._gen:
                return out
            lines = sorted((d for d in self._lines.values() if d["gen"] == s.gen and d["multipv"] <= s.multipv),
                           key=lambda d: d["multipv"])
            strings = list(self._strings.values())
            extra = {k: v for k, v in self._extra.items() if v["gen"] == s.gen}
            root = self._root if self._root and self._root["gen"] == s.gen else None
        flip = s.board.turn == chess.BLACK
        for d in lines:
            sign = -1 if flip else 1
            san, uci = _san_line(s.board, d["pv"])
            w = d.get("wdl")
            out["lines"].append({
                "multipv": d["multipv"], "depth": d.get("depth"), "seldepth": d.get("seldepth"),
                "nodes": d.get("nodes"), "nps": d.get("nps"),
                "cp": sign * d["value"] if d["kind"] == "cp" else None,
                "mate": sign * d["value"] if d["kind"] == "mate" else None,
                "bound": d.get("bound"),
                "wdl": None if w is None else (list(reversed(w)) if flip else w),
                "pv": uci, "san": san,
            })
            x = extra.get(d["multipv"])
            if x is not None:
                out["lines"][-1]["preppy"] = _preppy_line(x, uci, flip)
        out["strings"] = strings
        if root is not None:
            out["root"] = _preppy_root(root, flip)
        return out

    def stop(self) -> None:
        """Quit the engine (the page chose Off, or the server is going down)."""
        with self._cmd:
            self._quit()

    def close(self) -> None:
        self._closed = True
        self.stop()

    def _watch(self) -> None:
        while not self._closed:
            time.sleep(min(5.0, self.idle_s / 4))
            if self._proc is not None and time.monotonic() - self._touched > self.idle_s:
                with self._cmd:
                    if self._proc is not None and time.monotonic() - self._touched > self.idle_s:
                        self._quit()
                        self._error = f"stopped after {self.idle_s:g} s without a page asking"


def _check_accepted(spec: EngineSpec, lines: list[str]) -> None:
    """An engine's complaint about a `setoption` comes back on stdout before `readyok`."""
    bad = [ln for ln in lines if "error" in ln.lower() or "unknown option" in ln.lower() or "no such option" in ln.lower()]
    if bad:
        raise EngineError(f"{spec.label} refused: {' | '.join(bad)[:300]}")


def _board(fen: str, moves: list[str]) -> chess.Board:
    if not moves:
        return chess.Board(fen)
    b = chess.Board()
    for m in moves:
        b.push_uci(m)
    if b.board_fen() != chess.Board(fen).board_fen():
        raise EngineError(f"moves lead to {b.fen()}, not {fen}")
    return b


def _san_line(board: chess.Board, pv: list[str]) -> tuple[list[str], list[str]]:
    b = board.copy(stack=False)
    san, uci = [], []
    for m in pv:
        try:
            mv = chess.Move.from_uci(m)
        except ValueError:
            break
        if not b.is_legal(mv):
            break
        san.append(b.san(mv))
        uci.append(m)
        b.push(mv)
    return san, uci


def _kv(s: str) -> dict:
    """`preppy a=1 b=x ...` as {a: "1", b: "x"}."""
    out = {}
    for tok in s.split()[1:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def _num(x: str | None) -> float | None:
    if x is None or x == "-":
        return None
    try:
        return float(x)
    except ValueError:
        return None


_KNOWN_LINE_KEYS = {"multipv", "value", "net", "nonobvious", "reach", "moves", "provisional", "gen",
                    "mode", "depth", "complete", "inside", "value_in_progress"}


def _int(x: str | None) -> int | None:
    v = _num(x)
    return None if v is None else int(v)


def _flag(x) -> bool:
    return str(x if x is not None else "0").lower() in ("1", "true", "yes")


def _preppy_line(kv: dict, pv: list[str], flip: bool) -> dict:
    """A preppy line's numbers. `value` is the side to move's expected score: `white` turns it
    to White's. Each move's flag: "L" my move learned, "M" my move not learned, else the
    opponent's probability; aligned with `pv` (moves beyond the legal pv are dropped)."""
    v = _num(kv.get("value"))
    order = [it.split(":", 1) for it in kv.get("moves", "").split(",") if ":" in it]
    return {"value": v, "white": None if v is None else (1 - v if flip else v), "net": _num(kv.get("net")),
            "nonobvious": _num(kv.get("nonobvious")), "reach": _num(kv.get("reach")),
            "flags": [f for _m, f in order][:len(pv)],
            # an early line, before preppy's solve for the position is done (shown dimmed)
            "provisional": _flag(kv.get("provisional")),
            # bestmove mode: `value` is final at the last completed `depth` (with complete=0 no
            # depth is complete yet); `value_in_progress` is at the depth being searched
            "mode": kv.get("mode", "repertoire"), "depth": _int(kv.get("depth")),
            "complete": _flag(kv.get("complete")) if "complete" in kv else None,
            "inside": _int(kv.get("inside")),
            "in_progress": _num(kv.get("value_in_progress")),
            "white_in_progress": None if _num(kv.get("value_in_progress")) is None
            else (1 - _num(kv.get("value_in_progress")) if flip else _num(kv.get("value_in_progress"))),
            # keys this page does not know yet, passed on as they came
            "other": {k: v for k, v in kv.items() if k not in _KNOWN_LINE_KEYS}}


def _preppy_root(kv: dict, flip: bool) -> dict:
    v = _num(kv.get("value"))
    return {"value": v, "white": None if v is None else (1 - v if flip else v),
            "mode": kv.get("mode", "repertoire"), "depth": _int(kv.get("depth")),
            "complete": _flag(kv.get("complete")) if "complete" in kv else None,
            "provisional": _flag(kv.get("provisional")), "depth_in_progress": _int(kv.get("depth_in_progress")),
            "fix": kv.get("fix"),                  # MistakeFix in force, when preppy says
            **{k: (kv.get(k) if k == "cap" else _num(kv.get(k)))
               for k in ("net", "price", "cost", "cap", "stand_in", "frontier", "finished", "expanded")}}
