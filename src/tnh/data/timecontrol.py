"""Clock comments, the TimeControl tag, and the tempo of a game.

Tempo follows FIDE: a player who has 60 minutes or more for a 60-move game
plays classical, more than 10 minutes is rapid, the rest is blitz. With
several periods, the 60 moves are counted through them, so
``40/5400+30:1800+30`` gives 5400 + 1800 + 60 * 30 seconds.

In the broadcast dumps the TimeControl tag is missing from most games and is
often free text when present (``90 min/40 moves + 30 min + 30s/move``). The
parser below reads the PGN-standard form, the ``minutes+seconds`` shorthand
and the common free-text forms, and returns None for anything it cannot read
with confidence.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

CLK_RE = re.compile(r"\[%clk\s+(\d+):(\d{1,2}):(\d{1,2}(?:\.\d*)?)\s*\]")

CLASSICAL_S = 3600
RAPID_S = 600


def parse_clock(comment: str) -> int | None:
    """Seconds in the first ``[%clk h:mm:ss]`` of a comment, or None."""
    m = CLK_RE.search(comment)
    if not m:
        return None
    h, mnt, s = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + int(float(s))


@dataclass(frozen=True)
class Period:
    moves: int | None  # moves in this period, None for the rest of the game
    base_s: int
    inc_s: int


@dataclass(frozen=True)
class TimeControl:
    periods: tuple[Period, ...]

    @property
    def base_s(self) -> int:
        return self.periods[0].base_s

    @property
    def inc_s(self) -> int:
        return self.periods[0].inc_s

    def seconds_for(self, n_moves: int = 60) -> int:
        """Time a player has for ``n_moves`` moves, counted through the periods."""
        total = 0
        done = 0
        for p in self.periods:
            if done >= n_moves:
                break
            total += p.base_s
            here = n_moves - done if p.moves is None else min(p.moves, n_moves - done)
            total += here * p.inc_s
            done += here
            if p.moves is None:
                break
        return total

    def bonus_after_move(self, move_number: int) -> int:
        """Time added to the mover's clock after their ``move_number``-th move, beyond the increment.

        This is the base of the next period when that move ends a period.
        """
        done = 0
        for i, p in enumerate(self.periods):
            if p.moves is None:
                return 0
            done += p.moves
            if move_number == done and i + 1 < len(self.periods):
                return self.periods[i + 1].base_s
            if move_number < done:
                return 0
        return 0


def tempo_of_seconds(seconds_for_60: int) -> str:
    if seconds_for_60 >= CLASSICAL_S:
        return "classical"
    if seconds_for_60 > RAPID_S:
        return "rapid"
    return "blitz"


# --- The PGN-standard and shorthand forms -----------------------------------

_STD_FIELD = re.compile(r"^(?:(\d+)/)?(\d+)(?:\+(\d+))?$")
# minutes up to this value are read as minutes in the shorthand "90+30"
_MAX_MINUTES = 179


def _base_seconds(n: int, unit: str | None) -> int:
    if unit == "h":
        return n * 3600
    if unit in ("m", "min"):
        return n * 60
    if unit == "s":
        return n
    # no unit: small numbers are minutes ("90+30", "3+2"), large are seconds ("5400+30")
    return n * 60 if n <= _MAX_MINUTES else n


def parse_standard(tc: str) -> TimeControl | None:
    """``40/5400+30:1800+30``, ``5400+30``, ``5400``, ``90+30``, ``3+2``.

    A bare number up to 180 is read as minutes, larger as seconds: the
    broadcasts mix ``90+30`` and ``5400+30`` for the same control, and no
    broadcast game has a base of 180 seconds or less written in seconds.
    """
    s = tc.strip().replace(" ", "")
    if not s:
        return None
    periods = []
    fields = s.split(":")
    for i, field in enumerate(fields):
        m = _STD_FIELD.match(field)
        if not m:
            return None
        moves, base, inc = m.groups()
        moves_n = int(moves) if moves is not None else None
        if moves_n is not None and moves_n > 100 and int(base) <= 100:
            moves_n, base = int(base), moves  # "5400/40" written for "40/5400"
        if moves_n is None and i < len(fields) - 1:
            return None  # only the last period may be "rest of the game"
        periods.append(Period(moves_n, _base_seconds(int(base), None), int(inc or 0)))
    if periods[0].base_s == 0:
        return None
    return TimeControl(tuple(periods))


# --- Free text ---------------------------------------------------------------

_SEC = r"(?:s\b|sec\b|secs\b|seconds?\b|secondss\b|seg\b|sgs\b|segundos?\b|sekunden\b|sekundi\b|sek\b|\"|''|”|’’|′′|″)"
_MIN = r"(?:minutes?\b|minut[ae]?\b|minutos?\b|minuten\b|mins?\b|mn\b|m\b|'|’|′)"
_INC_RE = re.compile(r"(\d+)\s*\.?\s*" + _SEC)
_HMS_RE = re.compile(r"\b(\d):(\d\d)(?:[:.](\d\d))?\b")
_IN_RE = re.compile(r"\b(\d+)\s*(?:moves?|mov\.?)?\s*in\s*(\d+)")
_SLASH_RE = re.compile(r"(\d+)\s*" + _MIN + r"?\.?\s*/\s*(\d+)")
_MINVAL_RE = re.compile(r"(\d+)\s*\.?\s*" + _MIN)
_HOUR_RE = re.compile(r"(\d+)\s*(?:h\b|hrs?\b|hours?\b|horas?\b|stunden\b)")
_LEAD_RE = re.compile(r"^\s*(?:g(?:ame)?\s*[:/]?\s*)?(\d+)\s*\+\s*(\d+)?")
_FROM_MOVE_RE = re.compile(r"(?:from|starting(?: from)?|desde)\s*(?:the\s*)?(?:move|mov\.?)\s*(?:no\.?\s*)?(?:1|one)\b")
_PER_ROUND_RE = re.compile(r"\br\d|\bround|\brd\.?\s*\d|\brunde|\bronda")
_MOVE_COUNTS = {20, 30, 36, 40, 50, 60}
_NUM_RE = re.compile(r"\d+")
# "10'+2": a number after a plus with no unit, not followed by a minute unit or a slash
_PLUS_INC_RE = re.compile(r"\+\s*(\d{1,2})\b(?!\s*\.?\s*(?:" + _MIN + r"|/|\d))")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return s.replace("‘", "'").replace("`", "'")


def parse_free_text(tc: str) -> TimeControl | None:
    """Base and increment of the first period from free text.

    Returns one period (the first), with its move count when one is given;
    later periods are not read. The base is the earliest of: an ``h:mm``
    time, "40 moves in 100", "90/40" or "40/90", a number with a minute or
    hour unit, or a leading "90+30". The increment is the first number with
    a seconds unit, or the number after a leading "90+". Returns None when no
    base is found, and when the tag gives different controls for different
    rounds ("15 min ... (Round 1 - 4) / 90 min ... (Round 5 - 8)").
    """
    s = _FROM_MOVE_RE.sub(" ", _norm(tc))
    if _PER_ROUND_RE.search(s):
        return None

    inc: int | None = None
    inc_span = (-1, -1)
    m = _INC_RE.search(s)
    if m:
        inc, inc_span = int(m.group(1)), m.span()

    cands: list[tuple[int, int, int | None]] = []  # (position, base seconds, moves)
    if m := _HMS_RE.search(s):
        cands.append((m.start(), int(m.group(1)) * 3600 + int(m.group(2)) * 60, None))
    if m := _IN_RE.search(s):
        a, b = int(m.group(1)), int(m.group(2))
        if a in _MOVE_COUNTS:
            cands.append((m.start(), _base_seconds(b, None), a))
    for m in _SLASH_RE.finditer(s):
        a, b = int(m.group(1)), int(m.group(2))
        if a in _MOVE_COUNTS and b not in _MOVE_COUNTS:
            cands.append((m.start(), _base_seconds(b, None), a))
        elif b in _MOVE_COUNTS:
            cands.append((m.start(), _base_seconds(a, None), b))
        break
    for m in _MINVAL_RE.finditer(s):
        if inc_span[0] <= m.start() < inc_span[1]:
            continue
        cands.append((m.start(), int(m.group(1)) * 60, None))
        break
    if m := _HOUR_RE.search(s):
        cands.append((m.start(), int(m.group(1)) * 3600, None))
    lead = _LEAD_RE.match(s)
    if lead:
        cands.append((lead.start(1), _base_seconds(int(lead.group(1)), None), None))
        if inc is None and lead.group(2) is not None:
            inc = int(lead.group(2))
    if not cands and (m := _NUM_RE.search(s)):
        cands.append((m.start(), _base_seconds(int(m.group(0)), None), None))
    if not cands:
        return None
    pos, base_s, moves = min(cands, key=lambda c: c[0])
    if inc is None and (m := _PLUS_INC_RE.search(s, pos)):
        inc = int(m.group(1))
    if base_s <= 0:
        return None
    return TimeControl((Period(moves, base_s, inc or 0),))


def parse_time_control(tc: str | None) -> tuple[TimeControl | None, str]:
    """(control, how) where how is ``standard``, ``free_text`` or ``unreadable``/``absent``."""
    if tc is None or tc.strip() in ("", "-", "?", "*"):
        return None, "absent"
    std = parse_standard(tc)
    if std is not None:
        return std, "standard"
    free = parse_free_text(tc)
    if free is not None:
        return free, "free_text"
    return None, "unreadable"


# --- From the clocks ----------------------------------------------------------

COMMON_INCREMENTS = (0, 2, 3, 5, 10, 15, 20, 30, 60)
COMMON_BASE_MIN = (3, 5, 10, 15, 20, 25, 30, 45, 50, 60, 75, 90, 100, 110, 120)
MIN_USABLE_CLOCK_S = 30  # a first clock below this means the clock was not running


@dataclass(frozen=True)
class ClockEstimate:
    first_clock_s: int  # the larger of the two clocks after each side's first move
    base_s: int  # rounded to the minute
    inc_s: int  # snapped to a common increment


def estimate_from_clocks(clocks: list[int | None]) -> ClockEstimate | None:
    """Base and increment seen on the clocks, or None when they cannot tell.

    The first clock is read after each side's first move (plies 0 and 1); the
    larger of the two is used, so one player arriving late does not matter.
    The increment is the largest gain between a player's consecutive clocks
    in the first 20 moves, rounded up to a common increment (a fast book move
    gains the increment less a few seconds); a largest gain more than 5 s
    above the next is an arbiter's correction and the next is used. Checked
    against games whose tag and clocks agree: the increment is right in 88 %
    (95 % for +2, 90 % for +10 and +30, 36 % for +3, which reads as +2). Base is the
    first clock minus the increment, snapped to a common base within 5 %, else rounded
    to the minute.
    """
    first = [c for c in clocks[:2] if c is not None]
    if not first or max(first) < MIN_USABLE_CLOCK_S:
        return None
    cmax = max(first)
    gains = sorted(
        clocks[i] - clocks[i - 2]
        for i in range(2, min(len(clocks), 40))
        if clocks[i] is not None and clocks[i - 2] is not None
    )
    raw_inc = 0
    if len(gains) >= 2:
        # the largest gain, unless it stands out from the next (an arbiter's correction)
        raw_inc = gains[-1] if gains[-1] - gains[-2] <= 5 else gains[-2]
    elif gains:
        raw_inc = gains[-1]
    # a gain is the increment minus the time spent, so round up to a common increment
    inc = 0 if raw_inc <= 0 else min([x for x in COMMON_INCREMENTS if x >= raw_inc] or [COMMON_INCREMENTS[-1]])
    minutes = (cmax - inc) / 60
    near = min(COMMON_BASE_MIN, key=lambda b: abs(b - minutes))
    base = near * 60 if abs(near - minutes) <= max(1.0, 0.05 * near) else max(60, round(minutes) * 60)
    return ClockEstimate(cmax, base, inc)


def clock_consistent_with(tc: TimeControl, est: ClockEstimate) -> bool:
    """Do the clocks fit the tagged control? Generous: only clear contradictions fail."""
    expected = tc.base_s + tc.inc_s
    slack = max(180, 0.1 * tc.base_s) + 2 * tc.inc_s
    return abs(est.first_clock_s - expected) <= slack


# --- From the event name --------------------------------------------------------

_BLITZ_WORDS = re.compile(r"blitz|bullet|armageddon|lightning|błyskawic|bleskov", re.IGNORECASE)
_RAPID_WORDS = re.compile(
    r"rapid|rápid|rapide|schnellschach|snabbschack|quickplay|\ballegro\b|\bactive chess\b|szybk",
    re.IGNORECASE,
)
_TIEBREAK_WORDS = re.compile(
    r"tie-?\s?break|play-?\s?off|desempate|stichkampf|barrage|armageddon", re.IGNORECASE
)
_CLASSICAL_WORDS = re.compile(r"\bclassical\b|\bklassisch|\bclásic|\bclassique\b", re.IGNORECASE)


def _tempo_from_text(text: str) -> str | None:
    blitz = bool(_BLITZ_WORDS.search(text))
    rapid = bool(_RAPID_WORDS.search(text))
    if blitz and rapid:
        return None
    if blitz:
        return "blitz"
    if rapid:
        return "rapid"
    if _TIEBREAK_WORDS.search(text):
        return "tiebreak"
    if _CLASSICAL_WORDS.search(text):
        return "classical"
    return None


def tempo_from_names(names: list[str]) -> str | None:
    """``blitz``, ``rapid``, ``classical``, ``tiebreak`` or None from event names.

    Each name is read from its most specific part: "Polish Youth Rapid &
    Blitz 2026 | Blitz | G16" is blitz, from "Blitz". A part with both rapid
    and blitz words ("World Rapid & Blitz") says nothing, and the next part
    is read. The first name that says something decides.
    ``tiebreak`` means "not classical, speed unknown".
    """
    for name in names:
        for part in reversed((name or "").split("|")):
            t = _tempo_from_text(part)
            if t is not None:
                return t
    return None


# --- The ChessBase EventType tag ---------------------------------------------------

_EVENT_TYPE_RE = re.compile(r"\((rapid|blitz|corr)\)|^(rapid|blitz)$", re.IGNORECASE)


def tempo_from_event_type(event_type: str | None) -> str | None:
    """``rapid``, ``blitz`` or None from an EventType tag such as ``swiss (rapid)``.

    ChessBase marks only the faster tempi; a bare ``swiss`` or ``team`` says
    nothing, so it is not read as classical. ``(corr)`` is correspondence and
    is dropped at parse time, not here.
    """
    if not event_type:
        return None
    m = _EVENT_TYPE_RE.search(event_type.strip())
    if not m:
        return None
    t = (m.group(1) or m.group(2)).lower()
    return t if t in ("rapid", "blitz") else None


# --- The decision ---------------------------------------------------------------

# Decisions that the tempo of the broadcast round may replace (see parse.assemble).
WEAK_SOURCES = ("event_name", "assumed", "none")


@dataclass(frozen=True)
class TempoDecision:
    tempo: str  # classical, rapid, blitz, unknown
    source: str  # clock, tag, event_type, event_name, assumed, none ("round" is set later)
    base_s: int | None
    inc_s: int | None
    note: str  # why, for the counts in the report


def decide_tempo(
    tc_raw: str | None,
    clocks: list[int | None],
    names: list[str],
    event_type: str | None = None,
    default: str | None = None,
) -> TempoDecision:
    """Tempo of one game from what the game itself carries. The first that applies wins:

    1. ``clock``: the clocks, where the tag is missing, unreadable or clearly
       contradicted by them (base and increment estimated from the clocks).
    2. ``tag``: a readable TimeControl tag.
    3. ``event_type``: an EventType tag marked rapid or blitz (TWIC).
    4. (``round``: set afterwards in ``parse.assemble`` over all games of a
       broadcast round; it replaces 5-7.)
    5. ``event_name``: tempo words in the event or broadcast name. A name that
       says tiebreak or playoff gives ``unknown``, never the default.
    6. ``assumed``: the source's default, if it has one (TWIC: classical).
    7. ``none``: ``unknown``.
    """
    tc, how = parse_time_control(tc_raw)
    est = estimate_from_clocks(clocks)

    if tc is not None:
        tag_tempo = tempo_of_seconds(tc.seconds_for(60))
        if est is None:
            return TempoDecision(tag_tempo, "tag", tc.base_s, tc.inc_s, f"tag_{how}_no_clock")
        if clock_consistent_with(tc, est):
            return TempoDecision(tag_tempo, "tag", tc.base_s, tc.inc_s, f"tag_{how}_clock_agrees")
        clock_tempo = tempo_of_seconds(est.base_s + 60 * est.inc_s)
        return TempoDecision(clock_tempo, "clock", est.base_s, est.inc_s, f"tag_{how}_overruled_by_clock")
    if est is not None:
        clock_tempo = tempo_of_seconds(est.base_s + 60 * est.inc_s)
        return TempoDecision(clock_tempo, "clock", est.base_s, est.inc_s, f"tag_{how}_clock")
    et = tempo_from_event_type(event_type)
    if et is not None:
        return TempoDecision(et, "event_type", None, None, f"tag_{how}_event_type_{et}")
    named = tempo_from_names(names)
    if named in ("rapid", "blitz", "classical"):
        return TempoDecision(named, "event_name", None, None, f"tag_{how}_name_{named}")
    if named == "tiebreak":
        return TempoDecision("unknown", "none", None, None, f"tag_{how}_name_tiebreak")
    if default is not None:
        return TempoDecision(default, "assumed", None, None, f"tag_{how}_assumed_{default}")
    return TempoDecision("unknown", "none", None, None, f"tag_{how}_nothing")
