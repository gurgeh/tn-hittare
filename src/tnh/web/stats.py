"""The database panel: the interface's seam in front of `tnh.data.edges.EdgeDB`, and the table it shows.

`Stats` wraps an `EdgeDB` (the parquet tables, or `EdgeDB.from_frames` on the output of
`tables_from_games` for fixtures). The page only ever talks to `Stats`, so a change in the
read API is absorbed here.

One thing `EdgeDB` cannot answer is which positions lead into a position: that needs a
lookup on `child_hash`, which the tables are not sorted by. `Stats` answers it only when
it was given the frames in memory (fixtures); otherwise `parents` is None and the panel
says the source cannot tell.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import chess
import polars as pl

from tnh.data.edges import EdgeDB, pos_hash

TEMPI = ("classical", "rapid", "blitz", "unknown")
YEAR_MAX = 32767


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

    def polars(self) -> pl.Expr:
        """The same filter over raw edges rows, for `parents` on in-memory frames."""
        e = pl.lit(True)
        kw = self.edgedb()
        for col, key in (("year", "years"), ("elo_band", "elo_bands")):
            if key in kw:
                e &= pl.col(col).is_between(*kw[key])
        if "tempi" in kw:
            e &= pl.col("tempo").is_in(kw["tempi"])
        return e


class Stats:
    def __init__(self, db: EdgeDB, name: str, frames: tuple[pl.DataFrame, pl.DataFrame] | None = None) -> None:
        self.db = db
        self.name = name
        self._by_child = None
        self._epd: dict[int, str] = {}
        self._info = {"source": name}
        if frames is not None:
            edges, index = frames
            self._by_child = edges.select("child_hash", "pos_hash", "move", "year", "elo_band", "tempo", "n").sort("child_hash")
            self._epd = dict(zip(index["pos_hash"].to_list(), index["epd"].to_list()))
            years = edges.filter(pl.col("year") > 0)["year"]
            self._info |= {"positions": index.height, "edge_rows": edges.height,
                           "year_min": int(years.min()) if years.len() else None,
                           "year_max": int(years.max()) if years.len() else None}

    @classmethod
    def from_frames(cls, edges: pl.DataFrame, pos_index: pl.DataFrame, name: str) -> "Stats":
        return cls(EdgeDB.from_frames(edges, pos_index), name, (edges, pos_index))

    def info(self) -> dict:
        return self._info

    def parents(self, h: int, f: Filters) -> pl.DataFrame | None:
        """Edges into the position: `parent_hash`, `parent_epd`, `move`, `n`. None when unknown."""
        if self._by_child is None:
            return None
        s = self._by_child["child_hash"]
        lo, hi = s.search_sorted(h, side="left"), s.search_sorted(h, side="right")
        rows = self._by_child.slice(lo, hi - lo).filter(f.polars())
        out = rows.group_by("pos_hash", "move").agg(pl.col("n").sum()).rename({"pos_hash": "parent_hash"})
        return out.with_columns(
            parent_epd=pl.Series([self._epd.get(p) for p in out["parent_hash"].to_list()], dtype=pl.String)
        ).sort("n", descending=True)


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

    The years with data under the filters are split into the last `recent_years`
    (counted back from the latest year with a game) and everything before. For each move
    a two-proportion z-test compares its share of the position's games in the two
    periods. A move is flagged when the two-sided p-value is below `alpha / k`, k being
    the number of moves tested in the position (Bonferroni). Both periods need at least
    `min_games` games in the position, and the move `min_move_games`. Games are treated as
    independent; they are not (one player repeating a line through a tournament), so the
    test is anti-conservative and the flag is a pointer, not a verdict.
    """

    recent_years: int = 3
    alpha: float = 0.01
    min_games: int = 30
    min_move_games: int = 5


def _num(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


def report(stats: Stats, board: chess.Board, f: Filters, rule: TrendRule = TrendRule()) -> dict:
    """Everything the database panel shows for one position."""
    h = pos_hash(board)
    white = board.turn == chess.WHITE
    kw = f.edgedb()
    db = stats.db
    total_row = db.total(h)
    moves = db.moves(h, **kw)
    by_year = db.moves_by_year(h, **kw).filter(pl.col("year") > 0) if moves.height else None
    kids = db.children(h, **kw) if moves.height else None
    reach = dict(zip(kids["move"].to_list(), kids["child_n"].to_list())) if kids is not None else {}

    total_n = int(moves["n"].sum()) if moves.height else 0
    tw, td, tb = (int(moves[c].sum()) if moves.height else 0 for c in ("white_wins", "draws", "black_wins"))

    years: list[int] = []
    year_totals: dict[int, int] = {}
    move_year: dict[tuple[str, int], int] = {}
    if by_year is not None and by_year.height:
        yt = by_year.group_by("year").agg(pl.col("n").sum()).sort("year")
        years = yt["year"].to_list()
        year_totals = dict(zip(years, yt["n"].to_list()))
        move_year = {(m, y): n for m, y, n in by_year.select("move", "year", "n").iter_rows()}

    trend_meta = None
    recent_cut = None
    if years:
        recent_cut = max(years) - rule.recent_years + 1
        n_recent = sum(n for y, n in year_totals.items() if y >= recent_cut)
        n_early = sum(n for y, n in year_totals.items() if y < recent_cut)
        trend_meta = {
            "years": [years[0], years[-1]], "recent_from": recent_cut, "n_recent": n_recent, "n_early": n_early,
            "tested": n_recent >= rule.min_games and n_early >= rule.min_games,
            "test": (f"Two-proportion z-test of the share in {recent_cut}-{years[-1]} against "
                     f"{years[0]}-{recent_cut - 1}, flagged at p < {rule.alpha:g}/k (Bonferroni over the k moves "
                     f"tested). Needs {rule.min_games}+ games in each period and {rule.min_move_games}+ of the move. "
                     "Treats games as independent, so it flags too often"),
        }

    rows = []
    for r in moves.iter_rows(named=True):
        mv = chess.Move.from_uci(r["move"])
        rows.append({
            "uci": r["move"], "san": board.san(mv) if board.is_legal(mv) else r["move"], "n": r["n"],
            "share": r["share"], "score": r["score"],
            "draw": r["draws"] / r["n"] if r["n"] else None,
            "elo": _num(r["mover_elo"]), "opp_elo": _num(r["opp_elo"]), "perf": _num(r["performance"]),
            "last_year": r["last_year"],
            "reach": reach.get(r["move"]),
            "series": [move_year.get((r["move"], y), 0) for y in years],
        })

    if trend_meta and trend_meta["tested"]:
        tested = [m for m in rows if m["n"] >= rule.min_move_games]
        k = max(len(tested), 1)
        for m in tested:
            x1 = sum(n for y, n in zip(years, m["series"]) if y >= recent_cut)
            x0 = sum(n for y, n in zip(years, m["series"]) if y < recent_cut)
            z, p = two_proportion(x1, trend_meta["n_recent"], x0, trend_meta["n_early"])
            m["trend"] = {"early": x0 / trend_meta["n_early"], "recent": x1 / trend_meta["n_recent"],
                          "z": z, "p": p, "flag": (1 if z > 0 else -1) if p < rule.alpha / k else 0}

    move_orders = None
    parents = stats.parents(h, f)
    if parents is not None:
        move_orders = []
        for epd, mv, n in parents.select("parent_epd", "move", "n").iter_rows():
            san, fen = mv, None
            if epd is not None:
                pb = chess.Board()
                pb.set_epd(epd)
                m = chess.Move.from_uci(mv)
                san = pb.san(m) if pb.is_legal(m) else mv
                fen = pb.fen()
            move_orders.append({"san": san, "uci": mv, "n": n, "parent_fen": fen})

    return {
        "hash": f"{h:016x}",
        "white_to_move": white,
        "total": {
            "n": total_n, "white_wins": tw, "draws": td, "black_wins": tb,
            "score": ((tw if white else tb) + 0.5 * td) / total_n if total_n else None,
            "reach": total_row["n"] if total_row else None,
        },
        "moves": rows,
        "years": years,
        "year_totals": [year_totals[y] for y in years],
        "trend": trend_meta,
        "move_orders": move_orders,
    }
