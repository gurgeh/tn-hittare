"""The database panel: the interface's seam in front of `tnh.data.edges.EdgeDB`, and what it shows.

The page reads position statistics only through `EdgeDB`, opened by table name (`full` or
`dev`) once per server process: `open_stats` picks the table, `report` turns
`EdgeDB.panel` and `EdgeDB.parents` into the panel's JSON. Tests pass an
`EdgeDB.from_frames` over tables built by `tables_from_games`.

Counting, as in docs/DESIGN.md: a position's count is the games that reached it, each
game once (`EdgeDB.total`, `child_n`); a move's count is the games that played it from
there. Filters apply to moves only: pos_index has no breakdown by year, band or tempo.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import NormalDist

import chess
import polars as pl

from tnh.data.edges import EdgeDB, pos_hash

TEMPI = ("classical", "rapid", "blitz", "unknown")
YEAR_MAX = 32767
TABLE_ENV = "TNH_WEB_TABLE"
_POOL = ThreadPoolExecutor(4, thread_name_prefix="stats")


# --- which table -------------------------------------------------------------------


def table_files(name: str, directory: Path) -> list[Path]:
    """The files `EdgeDB(name=...)` opens. Only used to see whether a table exists."""
    suffix = "" if name == "full" else f"_{name}"
    return [directory / f"{t}{suffix}.parquet" for t in ("edges", "pos_index", "parents")]


def choose_table(flag: str | None, directory: Path) -> tuple[str, str]:
    """(table name, why): the flag, else $TNH_WEB_TABLE, else `full` when its files exist,
    else `dev`."""
    if flag:
        return flag, "--table"
    if os.environ.get(TABLE_ENV):
        return os.environ[TABLE_ENV], f"${TABLE_ENV}"
    if all(p.exists() for p in table_files("full", directory)):
        return "full", "its files exist"
    return "dev", "no full table yet"


def open_stats(flag: str | None = None, directory: Path | None = None) -> "Stats":
    from tnh.paths import CORPUS
    directory = directory or CORPUS
    name, why = choose_table(flag, directory)
    missing = [str(p) for p in table_files(name, directory) if not p.exists()]
    if missing:
        raise SystemExit(f"table {name!r} ({why}) is missing: {', '.join(missing)}")
    db = EdgeDB(name=name, directory=directory)
    built = max(p.stat().st_mtime for p in table_files(name, directory))
    return Stats(db, name, why=why, built=datetime.fromtimestamp(built).isoformat(timespec="minutes"))


class Stats:
    """An open `EdgeDB` with the facts about it that the page shows."""

    def __init__(self, db: EdgeDB, name: str, *, why: str = "", built: str | None = None) -> None:
        self.db = db
        self.name = name
        summary = db.summary()
        self._info = {
            "table": name, "why": why, "built": built,
            "games": summary["games"], "year_min": summary["first_year"], "year_max": summary["last_year"],
        }

    def info(self) -> dict:
        return self._info


# --- filters ---------------------------------------------------------------------


@dataclass(frozen=True)
class Filters:
    """Which games count. `None` is no limit.

    Years are inclusive. The rating filter is on `elo_band`, the mean rating floored to
    200: `elo_min` and `elo_max` are band edges, a game counts when
    `elo_min <= band < elo_max`. Any rating or year limit leaves out the games where the
    rating or the year is unknown (band 0, year 0).
    """

    year_min: int | None = None
    year_max: int | None = None
    elo_min: int | None = None
    elo_max: int | None = None
    tempi: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.tempi is not None:
            bad = set(self.tempi) - set(TEMPI)
            if bad:
                raise ValueError(f"unknown tempo {sorted(bad)}; known: {TEMPI}")

    @property
    def active(self) -> bool:
        return bool(self.edgedb())

    def edgedb(self) -> dict:
        """The filters as `EdgeDB` keyword arguments."""
        kw: dict = {}
        if self.year_min is not None or self.year_max is not None:
            kw["years"] = (max(1, self.year_min or 1), self.year_max if self.year_max is not None else YEAR_MAX)
        if self.elo_min is not None or self.elo_max is not None:
            kw["elo_bands"] = (max(1, self.elo_min or 1), self.elo_max - 1 if self.elo_max is not None else YEAR_MAX)
        if self.tempi is not None:
            kw["tempi"] = list(self.tempi)
        return kw


# --- trends ------------------------------------------------------------------------


def two_proportion(x1: int, n1: int, x0: int, n0: int) -> tuple[float, float]:
    """Two-sided two-proportion z-test with the pooled proportion: (z, p)."""
    p = (x1 + x0) / (n1 + n0)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n0))
    if se == 0:
        return 0.0, 1.0
    z = (x1 / n1 - x0 / n0) / se
    return z, 2 * (1 - NormalDist().cdf(abs(z)))


@dataclass(frozen=True)
class TrendRule:
    """When a move's share counts as trending.

    The span of years with data under the filters is split into a recent part, the last
    `recent_years` but at most half the span (so the 2024+ table compares 2026 with
    2024-2025), and everything before. For each move a two-proportion z-test compares its
    share of the position's games in the two parts. A move is flagged when the two-sided
    p-value is below `alpha / k`, k being the number of moves tested (Bonferroni). Both
    parts need `min_games` games in the position, the move `min_move_games`. Games are
    treated as independent; they are not (one player repeating a line through a
    tournament), so the test flags too often.
    """

    recent_years: int = 3
    alpha: float = 0.01
    min_games: int = 30
    min_move_games: int = 5

    def recent(self, first: int, last: int) -> int:
        return max(1, min(self.recent_years, (last - first + 1) // 2))


def _trends(rows: list[dict], years: list[int], year_totals: dict[int, int], rule: TrendRule) -> dict | None:
    if len(years) < 2:
        return None
    recent_cut = years[-1] - rule.recent(years[0], years[-1]) + 1
    n_recent = sum(n for y, n in year_totals.items() if y >= recent_cut)
    n_early = sum(n for y, n in year_totals.items() if y < recent_cut)
    meta = {
        "recent_from": recent_cut, "n_recent": n_recent, "n_early": n_early,
        "tested": n_recent >= rule.min_games and n_early >= rule.min_games,
        "test": (f"Two-proportion z-test of the share in {recent_cut}-{years[-1]} against "
                 f"{years[0]}-{recent_cut - 1}, flagged at p < {rule.alpha:g}/k (Bonferroni over the k moves "
                 f"tested). Needs {rule.min_games}+ games in each period and {rule.min_move_games}+ of the move. "
                 "Treats games as independent, so it flags too often"),
    }
    if meta["tested"]:
        tested = [m for m in rows if m["n"] >= rule.min_move_games]
        k = max(len(tested), 1)
        for m in tested:
            x1 = sum(n for y, n in zip(years, m["series"]) if y >= recent_cut)
            x0 = sum(n for y, n in zip(years, m["series"]) if y < recent_cut)
            z, p = two_proportion(x1, n_recent, x0, n_early)
            m["trend"] = {"early": x0 / n_early, "recent": x1 / n_recent, "z": z, "p": p,
                          "flag": (1 if z > 0 else -1) if p < rule.alpha / k else 0}
    return meta


# --- the panel ---------------------------------------------------------------------


def _num(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


def _san(epd: str | None, uci: str) -> tuple[str, str | None]:
    if epd is None:
        return uci, None
    b = chess.Board()
    b.set_epd(epd)
    m = chess.Move.from_uci(uci)
    return (b.san(m) if b.is_legal(m) else uci), b.fen()


def report(stats: Stats, board: chess.Board, f: Filters, rule: TrendRule = TrendRule()) -> dict:
    """Everything the database panel shows for one position.

    `status` says what the numbers are:
    - `none`: fewer than two games in the table reach the position, so it is not in it;
    - `ended`: games reach it but none played a move from it (they ended there or at ply 80);
    - `filtered`: games played moves from it, but none under the filters;
    - `ok`.
    """
    h = pos_hash(board)
    kw = f.edgedb()
    db = stats.db
    fut_panel = _POOL.submit(db.panel, h, **kw)
    fut_parents = _POOL.submit(db.parents, h, **kw)
    panel, parents = fut_panel.result(), fut_parents.result()
    total, moves, by_year = panel["total"], panel["moves"], panel["by_year"]

    years: list[int] = []
    year_totals: dict[int, int] = {}
    move_year: dict[tuple[str, int], int] = {}
    yrows = by_year.filter(pl.col("year") > 0) if by_year.height else by_year
    if yrows.height:
        yt = yrows.group_by("year").agg(pl.col("n").sum()).sort("year")
        years = yt["year"].to_list()
        year_totals = dict(zip(years, yt["n"].to_list()))
        move_year = {(m, y): n for m, y, n in yrows.select("move", "year", "n").iter_rows()}

    rows = []
    for r in moves.iter_rows(named=True):
        mv = chess.Move.from_uci(r["move"])
        rows.append({
            "uci": r["move"], "san": board.san(mv) if board.is_legal(mv) else r["move"], "n": r["n"],
            "share": r["share"], "score": r["score"],
            "draw": r["draws"] / r["n"] if r["n"] else None,
            "n_rated": r["n_rated"],
            "elo": _num(r["mover_elo"]), "opp_elo": _num(r["opp_elo"]), "perf": _num(r["performance"]),
            "last_year": r["last_year"],
            "reach": r["child_n"],
            "series": [move_year.get((r["move"], y), 0) for y in years],
        })
    trend = _trends(rows, years, year_totals, rule)

    move_orders = []
    for epd, mv, n in parents.select("epd", "move", "n").iter_rows():
        san, fen = _san(epd, mv)
        move_orders.append({"san": san, "uci": mv, "n": n, "parent_fen": fen})

    played = int(moves["n"].sum()) if moves.height else 0
    if total is None:
        status = "none"
    elif played:
        status = "ok"
    elif f.active and db.moves(h).height:
        status = "filtered"
    else:
        status = "ended"
    white = board.turn == chess.WHITE
    tw, td, tb = (int(moves[c].sum()) if moves.height else 0 for c in ("white_wins", "draws", "black_wins"))
    return {
        "hash": f"{h:016x}",
        "white_to_move": white,
        "status": status,
        "total": {
            "reach": total["n"] if total else None,       # games that reached it, all years/bands/tempi
            "played": played if total else None,          # sum of the moves' games under the filters
            "white_wins": tw, "draws": td, "black_wins": tb,
            "score": ((tw if white else tb) + 0.5 * td) / played if played else None,
        },
        "moves": rows,
        "years": years,
        "year_totals": [year_totals[y] for y in years],
        "trend": trend,
        "move_orders": move_orders,
    }
