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


# --- shrinkage ---------------------------------------------------------------------


@dataclass(frozen=True)
class Shrink:
    """The weights, in games, with which the move table shrinks the database toward a prior.

    - `kappa`: move probability, `(count + kappa * p_model) / (n + kappa)`, as the
      repertoire builder mixes the opponent's moves. 100 is the builder study's default;
      nothing has been fitted for it yet (it needs model 1 on a sample).
    - `kappa_v`: White's score and the draw rate, `(n * db + kappa_v * model2) / (n + kappa_v)`.
      `python -m tnh.web.calibrate` on the 3,000 most common positions of the full table,
      classical games, with the position's own mean as the prior: predicting each move's
      2024-26 figure from the years before, the error is lowest at 10-22 (rare moves 10,
      all moves 22; flat between). Method of moments (DerSimonian-Laird) says 105-116, but
      it assumes one between-move variance for all moves and rare moves vary more. Model 2
      should deserve more weight than the position mean, so 20, at the top of the
      out-of-sample range. Refit when model 2's values exist for a sample.
    - `k_elo`: W-Elo and B-Elo, `n_rated / (n_rated + k_elo) * (move mean - position mean)`.
      Method of moments gives 21-22 (within-move variance 83,692, SD 289, from one row group
      of games.parquet, an upper bound since edges hold no sums of squares; between-move
      variance 3,760-3,963 by DerSimonian-Laird). Out of sample it is wrong: on rare moves
      k = 22 raises the squared error of the 2024-26 mean from 35,746 to 52,279, and the
      error falls all the way to k = 1. Players repeat their moves, so a move's own mean
      predicts the next player well. 1.
    """

    kappa: float = 100.0
    kappa_v: float = 20.0
    k_elo: float = 1.0

    def mix(self, n: float, db: float | None, model: float | None, kappa: float) -> float | None:
        if model is None:
            return db if n else None
        if not n or db is None:
            return model
        return (n * db + kappa * model) / (n + kappa)


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


def _white_black(r: dict, white_to_move: bool) -> tuple[float | None, float | None]:
    """A move row's mean ratings of White and Black (its mover and opponent are given)."""
    m, o = _num(r["mover_elo"]), _num(r["opp_elo"])
    return (m, o) if white_to_move else (o, m)


def report(stats: Stats, board: chess.Board, f: Filters, rule: TrendRule = TrendRule(),
           shrink: Shrink = Shrink()) -> dict:
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

    white = board.turn == chess.WHITE
    # The position's mean rating of each colour over the moves' rated games under the filters.
    rated = sum(r["n_rated"] for r in moves.iter_rows(named=True)) if moves.height else 0
    pos_w = pos_b = None
    if rated:
        pos_w = sum(r["n_rated"] * _white_black(r, white)[0] for r in moves.iter_rows(named=True) if r["n_rated"]) / rated
        pos_b = sum(r["n_rated"] * _white_black(r, white)[1] for r in moves.iter_rows(named=True) if r["n_rated"]) / rated
    pos_m, pos_o = (pos_w, pos_b) if white else (pos_b, pos_w)
    pos_gap = pos_m - pos_o if rated else None
    rows = []
    for r in moves.iter_rows(named=True):
        mv = chess.Move.from_uci(r["move"])
        w_elo, b_elo = _white_black(r, white)
        nr = r["n_rated"] or 0
        weight = nr / (nr + shrink.k_elo) if nr else 0.0
        m_elo, o_elo = _num(r["mover_elo"]), _num(r["opp_elo"])
        rows.append({
            "white_elo": w_elo, "black_elo": b_elo,
            "w_elo_delta": weight * (w_elo - pos_w) if nr else None,
            "b_elo_delta": weight * (b_elo - pos_b) if nr else None,
            # The alternative form: the chooser's delta, and the move's rating gap (chooser
            # minus opponent) against the position's gap, both shrunk the same way.
            "mover_elo_delta": weight * (m_elo - pos_m) if nr else None,
            "gap": m_elo - o_elo if nr else None,
            "gap_delta": weight * ((m_elo - o_elo) - pos_gap) if nr else None,
            "white_wins": r["white_wins"], "draws": r["draws"], "black_wins": r["black_wins"],
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
            "white_score": (tw + 0.5 * td) / played if played else None,
            "draw": td / played if played else None,
            "white_elo": pos_w, "black_elo": pos_b, "n_rated": rated, "gap": pos_gap,
        },
        "shrink": {"kappa": shrink.kappa, "kappa_v": shrink.kappa_v, "k_elo": shrink.k_elo},
        "moves": rows,
        "years": years,
        "year_totals": [year_totals[y] for y in years],
        "trend": trend,
        "move_orders": move_orders,
    }


# --- the model columns -----------------------------------------------------------------


def _white_view(wdl_mover: tuple | None, white_to_move: bool) -> tuple[float, float] | None:
    """(White's expected score, draw probability) from a (win, draw, loss) for the mover."""
    if wdl_mover is None:
        return None
    w, d, l = wdl_mover
    return ((w if white_to_move else l) + 0.5 * d, d)


def model_table(rep: dict, probs: dict[str, float], position_wdl: tuple | None,
                child_wdl: dict[str, tuple | None], child_white_to_move: bool, shrink: Shrink) -> dict:
    """The model columns of the move table from a `report`, model 1's probabilities for the
    side to move, and model 2's (win, draw, loss) for the mover of the position and of each
    child. Everything is from White's side; the page chooses how to show it.

    - `p_mix = (count + kappa * p_model) / (n + kappa)`, n the moves' games under the filters;
    - `score_mix`, `draw_mix` = `(n * db + kappa_v * model) / (n + kappa_v)`, n the move's games.
    """
    db = {m["uci"]: m for m in rep["moves"]}
    n = (rep["total"]["played"] or 0) if rep["status"] == "ok" else 0
    out = {}
    for uci in set(probs) | set(db) | set(child_wdl):
        p = probs.get(uci)
        m = db.get(uci)
        count = m["n"] if m else 0
        model = _white_view(child_wdl.get(uci), child_white_to_move)
        db_score = (m["white_wins"] + 0.5 * m["draws"]) / m["n"] if m and m["n"] else None
        db_draw = m["draws"] / m["n"] if m and m["n"] else None
        p_mix = (count + shrink.kappa * p) / (n + shrink.kappa) if p is not None else None
        out[uci] = {
            "p_model": p, "p_mix": p_mix, "p_db": count / n if n else None,
            "delta": p_mix - p if p is not None else None,
            "score_model": model[0] if model else None, "draw_model": model[1] if model else None,
            "score_db": db_score, "draw_db": db_draw, "n": count,
            "score_mix": shrink.mix(count, db_score, model[0] if model else None, shrink.kappa_v),
            "draw_mix": shrink.mix(count, db_draw, model[1] if model else None, shrink.kappa_v),
        }
    pos_model = _white_view(position_wdl, rep["white_to_move"])
    t = rep["total"]
    position = {
        "score_model": pos_model[0] if pos_model else None, "draw_model": pos_model[1] if pos_model else None,
        "wdl_white": None if position_wdl is None else (
            list(position_wdl) if rep["white_to_move"] else [position_wdl[2], position_wdl[1], position_wdl[0]]),
        "score_db": t["white_score"], "draw_db": t["draw"], "n": n,
        "score_mix": shrink.mix(n, t["white_score"], pos_model[0] if pos_model else None, shrink.kappa_v),
        "draw_mix": shrink.mix(n, t["draw"], pos_model[1] if pos_model else None, shrink.kappa_v),
    }
    return {"moves": out, "position": position}
