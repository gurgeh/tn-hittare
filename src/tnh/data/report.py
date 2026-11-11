"""Step 0 report: ``reports/step0_stats.md``.

    uv run python -m tnh.data.report

Reads ``corpus/games.parquet``, ``corpus/games_extra.parquet``,
``corpus/positions.parquet``, ``corpus/book_exit.parquet`` and
``corpus/parse_stats.json`` and writes the tables the step-0 decisions rest
on, per source and in total. Everything in it is computed here except the
paragraphs on the hand spot-checks, which record what was found when the
listed games were looked at (the samples are drawn with a fixed seed, so the
same games are listed on every run of the same corpus).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import polars as pl

from tnh.data.parse import surname
from tnh.data.window import in_window
from tnh.paths import CORPUS, DATA_ROOT, REPORTS

BANDS = {"1800-2500": (1800, 2500), "2100-2300": (2100, 2300)}
PLY_BUCKETS = [(30, 39), (40, 49), (50, 59), (60, 69), (70, 79)]
BOOK_COUNTS = [2, 5, 20, 100]
TEMPI = ["classical", "rapid", "blitz", "unknown"]
SEED = 20261111
LB, TWIC = "lichess_broadcast", "twic"

SPOT_CHECK_NOTE = """\
What was found when the three samples below were looked at by hand (corpus
built 2026-11-11 with both sources; the samples use a fixed seed):

- **Broadcast `unknown` (20 games).** After the round rule, what is left is
  team leagues and opens relayed without clocks or tags, where no game of the
  round has a clock either: the Slovak Extraliga (three), the Women's and 2.
  Women's Bundesliga (three), Oberliga Württemberg (two), the Swiss and Israeli
  team leagues, the European Club Cup, Manaus Open (two), Buenos Aires Masters,
  the Pfälzischer Kongress (two), First Saturday Budapest, the Saxony seniors,
  ITT de la Bandera and a Chinese master-candidate tournament. All twenty are
  classical events by their nature; none is a rapid or blitz game.
- **Broadcast `classical` (20 games).** All classical: eighteen at 90+30 (by
  tag or by clocks that read 5 218-5 459 s after the first move), the
  German U8 championship at 60+10 (classical by the FIDE rule, 60 min + 60 x
  10 s), and one game classical by its round (Odense championship).
- **TWIC `assumed` classical (20 games).** Nineteen are from events that are
  classical: opens (San Sebastian, Caioba, Groningen, Glassy Stream, Sozina,
  Wroclaw, Eurasia Cup, AXPG Brazil, Politiken Cup, Patras, Delhi), national
  and youth championships (Russian women, Yugoslav youth, Italian), the Czech
  Extraliga, NATO championship, Biel MTO, Gausdal and the World Seniors. The
  twentieth, the 1st Thanksgiving Open in Stillwater (USA, 2009), is a US
  weekend Swiss whose time control I cannot tell; such events often play G/60
  or faster in early rounds. So about 19 of 20 are really classical. What
  `assumed` cannot catch is a rapid game in an event whose name does not say
  so: knockout tiebreaks are caught by the round number (`n.3` on), other
  play-offs only by name.
"""


KING_STRIP_FINDINGS = """\
Looked at by hand: the early strips are real moves, not the arbiter's
signal: a king walking up the board in a queenless middlegame or an early
endgame after mass exchanges (Polgar-Mamedyarov 2002, Shirov-Cazacu 2019), a
king hunted into the centre before resigning (Vaibhav Nema-Lalit Rajoriya,
19 plies), and draws by repetition where the repeating move is a king step
(Yuffa-Narayanan, Kd5-e6). The cost is the last move of about 1 % of the
stripped games, at the end of the game, in positions that are either won
or repeated. The share before ply 60 is 1.1 %, so this is not a noticeable
share; the sample is listed because it was asked for.

TWIC's rate of legal strips (2 % of its games) is the base rate of games that
simply end with a king move to the centre; the broadcasts' higher rate (3.5 %)
is that plus the arbiter's signal where it happened to be legal.
"""


def band_expr(white: str, black: str, lo: int, hi: int) -> pl.Expr:
    return pl.col(white).is_between(lo, hi) & pl.col(black).is_between(lo, hi)


def md_table(df: pl.DataFrame, floatfmt: str = "{:.1%}") -> str:
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---:" if df[c].dtype.is_numeric() else "---" for c in cols) + "|"]
    for row in df.iter_rows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(floatfmt.format(v))
            elif isinstance(v, int):
                cells.append(f"{v:,}".replace(",", " "))
            else:
                cells.append("" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def du(path: Path) -> str:
    out = subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True, check=True).stdout
    return out.split()[0]


def pivot_tempo(df: pl.DataFrame, index: str) -> pl.DataFrame:
    t = df.group_by(index, "tempo").len().pivot(on="tempo", index=index, values="len").sort(index, nulls_last=True)
    t = t.with_columns(pl.exclude(index).fill_null(0))
    return t.select(index, *[c for c in TEMPI if c in t.columns])


def year_str(t: pl.DataFrame) -> pl.DataFrame:
    return t.with_columns(pl.col("year").cast(pl.String).fill_null("no date"))


def gid_int() -> pl.Expr:
    """game_id as an integer (60 bits of it), to keep 100+ M position rows small in memory."""
    return pl.col("game_id").str.slice(0, 15).str.to_integer(base=16)


def main() -> None:
    # the first two clocks are all the report shows; the full lists would double the memory
    games = pl.scan_parquet(CORPUS / "games.parquet").with_columns(pl.col("clocks_s").list.head(2)).collect()
    extra = pl.read_parquet(CORPUS / "games_extra.parquet")
    stats = json.loads((CORPUS / "parse_stats.json").read_text())
    g = games.join(extra, on="game_id")
    both = pl.col("white_elo").is_not_null() & pl.col("black_elo").is_not_null()
    g = g.with_columns(
        band=pl.when(~both).then(pl.lit("rating missing"))
        .when(band_expr("white_elo", "black_elo", 2100, 2300)).then(pl.lit("2100-2300"))
        .when(band_expr("white_elo", "black_elo", 1800, 2500)).then(pl.lit("1800-2500, not 2100-2300"))
        .otherwise(pl.lit("other ratings")),
        in_1800_2500=both & band_expr("white_elo", "black_elo", 1800, 2500),
        in_2100_2300=both & band_expr("white_elo", "black_elo", 2100, 2300),
    )
    sources = sorted(g["source"].unique())
    out: list[str] = []
    w = out.append
    w("# Step 0: the corpus of OTB games\n")
    w("Generated by `uv run python -m tnh.data.report` from `corpus/` under "
      f"`{DATA_ROOT}`. Sources: `lichess_broadcast` (the 81 monthly Lichess broadcast dumps 2020-01 to 2026-09) "
      "and `twic` (The Week in Chess, issues 210 to 1670, November 1998 to 2026).\n")

    # --- headline ---------------------------------------------------------------
    w("## Headline\n")
    w(f"- Games read: {stats['games_read']:,}; kept: {stats['games_kept']:,} in {stats['tournaments']:,} tournaments.")
    rows = []
    for src in [*sources, "all"]:
        s = g if src == "all" else g.filter(pl.col("source") == src)
        cl = s.filter(pl.col("tempo") == "classical")
        rows.append({
            "source (of the kept copy)": src,
            "read": stats["read_by_source"].get(src, stats["games_read"]) if src != "all" else stats["games_read"],
            "kept": s.height,
            "classical": cl.height,
            "classical 1800-2500": cl.filter(pl.col("in_1800_2500")).height,
            "of which with clocks": cl.filter(pl.col("in_1800_2500") & pl.col("has_clock")).height,
            "classical 2100-2300": cl.filter(pl.col("in_2100_2300")).height,
            "of which with clocks ": cl.filter(pl.col("in_2100_2300") & pl.col("has_clock")).height,
            "classical 1800-2500, assumed": cl.filter(pl.col("in_1800_2500") & (pl.col("tempo_source") == "assumed")).height,
        })
    w("")
    w(md_table(pl.DataFrame(rows)) + "\n")
    w("A game found in both sources is counted under the source of the copy that was kept: the broadcast "
      "copy when it has clocks. `assumed` is TWIC's default of classical for games with no sign of rapid or blitz; "
      "anything measured on classical games can leave those out.\n")

    # --- by source and year, before and after de-duplication ---------------------
    w("## Games by source and year, before and after de-duplication\n")
    before = pl.DataFrame(stats["games_before_dedup"]).rename({"len": "before"})
    after = g.group_by("source", "year").len().rename({"len": "after"})
    t = before.join(after, on=["source", "year"], how="full", coalesce=True, nulls_equal=True)
    t = t.with_columns(pl.col("before", "after").fill_null(0))
    t = t.pivot(on="source", index="year", values=["before", "after"]).sort("year", nulls_last=True)
    t = t.with_columns(pl.exclude("year").fill_null(0))
    order = ["year"] + [f"{k}_{s}" for s in sources for k in ("before", "after")]
    w("`before`: games parsed and kept by each source's rules; `after`: games whose kept copy is from that source. "
      "Within TWIC the same game often appears in two issues; across sources the broadcast copy with clocks wins.\n")
    w(md_table(year_str(t.select([c for c in order if c in t.columns]))) + "\n")

    # --- tempo, year, band, per source -------------------------------------------
    w("## Games by tempo, year and rating band\n")
    for src in sources:
        s = g.filter(pl.col("source") == src)
        w(f"### {src}\n")
        w(md_table(year_str(pivot_tempo(s, "year"))) + "\n")
        w(md_table(pivot_tempo(s, "band")) + "\n")
    w("Rating bands are exclusive rows here; \"1800-2500\" elsewhere includes 2100-2300. Ratings are the PGN "
      "`WhiteElo`/`BlackElo` as given; for rapid and blitz events they may be rapid or blitz ratings.\n")

    # --- clocks ------------------------------------------------------------------
    w("## Clocks\n")
    w("`has_clock`: at least 80 % of the moves carry `%clk`.\n")
    t = g.group_by("source", "band", "tempo").agg(games=pl.len(), with_clock=pl.col("has_clock").sum(), share=pl.col("has_clock").mean())
    w(md_table(t.sort("source", "band", "tempo")) + "\n")

    # --- tempo source -------------------------------------------------------------
    w("## How tempo was decided\n")
    w("Order (DESIGN.md): `clock` where the clocks contradict the TimeControl tag or there is no readable tag; `tag`; "
      "`event_type` (TWIC's ChessBase EventType tag, marked `(rapid)`, `(blitz)` or `rapid`/`blitz`; not yet in DESIGN.md); "
      "`round` (the other games of the broadcast round with a tempo from the first three agree); `event_name` (tempo "
      "words in the names; \"tiebreak\", \"playoff\" and TWIC knockout tiebreaks, rounds `n.3` on, give `unknown`); "
      "`assumed` (TWIC: classical); `none` (`unknown`).\n")
    t = g.group_by("source", "tempo_source", "tempo").len().sort("source", "tempo_source", "tempo")
    w(md_table(t) + "\n")
    t = g.group_by("source", "tempo_note").len().sort("source", "len", descending=[False, True]).rename({"tempo_note": "detail", "len": "games"})
    w("In detail (`tag_<how the tag was read>_<what happened>`):\n")
    w(md_table(t) + "\n")
    w(f"**Unknown: {g.filter(pl.col('tempo') == 'unknown').height:,} games**, "
      f"{g.filter((pl.col('tempo') == 'unknown') & pl.col('in_1800_2500')).height:,} of them with both players 1800-2500.\n")

    # --- spot checks --------------------------------------------------------------
    w("### Spot checks\n")
    w(SPOT_CHECK_NOTE)
    samples = [(LB, "unknown", None, SEED), (LB, "classical", None, SEED + 1), (TWIC, "classical", "assumed", SEED + 2)]
    for src, tempo, tsrc, seed in samples:
        s = g.filter((pl.col("source") == src) & (pl.col("tempo") == tempo))
        if tsrc:
            s = s.filter(pl.col("tempo_source") == tsrc)
        if s.height == 0:
            continue
        s = s.sample(min(20, s.height), seed=seed)
        cols = [pl.col("event").str.slice(0, 50), pl.col("site").str.slice(0, 25), "date", "round", "white_elo", "black_elo"]
        if src == LB:
            cols = [
                pl.col("broadcast_name").str.slice(0, 55),
                pl.col("time_control_raw").str.slice(0, 30).alias("TimeControl"),
                "tempo_source", "base_s", "inc_s",
                pl.col("clocks_s").list.head(2).cast(pl.List(pl.String)).list.join(", ").alias("first clocks"),
                "white_elo", "game_url",
            ]
        w(f"\n{src}, {tempo}{' (' + tsrc + ')' if tsrc else ''}, {s.height} games (seed {seed}):\n")
        w(md_table(s.select(cols)) + "\n")

    # --- cross-source duplicates ---------------------------------------------------
    w("## TWIC and the broadcasts: de-duplication\n")
    tw = g.filter(pl.col("copy_sources").list.contains(TWIC) & (pl.col("year") >= 2020))
    matched = tw.filter(pl.col("copy_sources").list.contains(LB))
    w(f"TWIC games dated 2020 or later (after TWIC's own duplicates are merged): {tw.height:,}. "
      f"Matched to a broadcast game on surnames, year and moves: **{matched.height:,}** "
      f"({matched.height / max(tw.height, 1):.1%}). By year:\n")
    t = tw.group_by("year").agg(twic_games=pl.len(), matched=pl.col("copy_sources").list.contains(LB).sum()).with_columns(
        share=pl.col("matched") / pl.col("twic_games")).sort("year")
    w(md_table(t) + "\n")
    w(near_duplicates(g))

    # --- tournaments ------------------------------------------------------------------
    w("## Tournaments and the split\n")
    span = games.group_by("tournament_id").agg(pl.col("split").n_unique().alias("n")).filter(pl.col("n") > 1).height
    tsrc = g.group_by("tournament_id").agg(
        has_lb=pl.col("copy_sources").list.contains(LB).any(),
        has_twic=pl.col("copy_sources").list.contains(TWIC).any(),
        # TWIC events are named only on games whose kept copy is TWIC's; a matched game carries the broadcast's tags
        twic_events=pl.concat_str("event", "site", pl.col("year").cast(pl.String), separator="|", ignore_nulls=True)
        .filter(pl.col("source") == TWIC).n_unique(),
        n=pl.len(),
    )
    joined = tsrc.filter(pl.col("has_lb") & pl.col("has_twic"))
    w(f"- Tournaments: {games['tournament_id'].n_unique():,}; tournaments whose games span two splits: **{span}** "
      "(the split is a function of `tournament_id`, so this checks that no game was given another split).")
    w(f"- Tournaments holding both broadcast and TWIC games (a TWIC event joined to a broadcast tournament through a "
      f"shared game): **{joined.height:,}**, holding {joined['n'].sum():,} games and "
      f"{joined['twic_events'].sum():,} distinct TWIC events (event, site, year, counted on the games kept from TWIC).")
    tour = games.group_by("tournament_id").len().sort("len", descending=True)
    w(f"- The largest tournament has {tour['len'][0]:,} games ({tour['len'][0] / games.height:.2%} of the corpus): "
      f"`{tour['tournament_id'][0]}`. Split of all games: "
      + ", ".join(f"{s} {n:,}" for s, n in games.group_by("split").len().sort("split").iter_rows()) + ".\n")
    w(md_table(tour.head(10).rename({"len": "games"})) + "\n")

    # --- window --------------------------------------------------------------------
    pos_path = CORPUS / "positions.parquet"
    n_pos = pl.scan_parquet(pos_path).select(pl.len()).collect().item()
    w("## Rows in the window\n")
    w(f"`positions.parquet`: {n_pos:,} rows (classical games with both ratings, plies below 80). "
      "A row is in a band when both the mover and the opponent are. Defaults: "
      "`book_count=5, max_ply=60, max_ply_no_clock=40, min_clock_s=600`.\n")
    src_of = games.select(gid_int().alias("gid"), "source")
    win_rows = []
    # one (source, band, split) at a time: the band and the split are the same on every row of a
    # game, so filtering on them first keeps whole games and keeps memory down
    for src in sources:
        ids = src_of.filter(pl.col("source") == src).select("gid")
        for bname, (lo, hi) in BANDS.items():
            counts: dict[tuple[int, str], tuple[int, int]] = {}
            for split in ("train", "val", "test"):
                pos = (
                    pl.scan_parquet(pos_path)
                    .filter((pl.col("split") == split) & band_expr("mover_elo", "opp_elo", lo, hi))
                    .select(gid_int().alias("gid"), "ply", "pos_count", "clock_mover_s")
                    .join(ids.lazy(), on="gid", how="semi")
                    .rename({"gid": "game_id"})
                    .collect()
                )
                for bc in BOOK_COUNTS:
                    t = pos.filter(in_window(pos, book_count=bc)).select(rows=pl.len(), games=pl.col("game_id").n_unique())
                    counts[(bc, split)] = (t["rows"][0], t["games"][0])
                del pos
            for bc in BOOK_COUNTS:
                win_rows.append({"source": src, "band": bname, "book_count": bc,
                                 **{f"{sp} rows": counts[(bc, sp)][0] for sp in ("train", "val", "test")},
                                 "test games": counts[(bc, "test")][1],
                                 "all rows": sum(counts[(bc, sp)][0] for sp in ("train", "val", "test"))})
    win = pl.DataFrame(win_rows)
    tot = win.group_by("band", "book_count").agg(pl.exclude("source").sum()).with_columns(source=pl.lit("all"))
    win = pl.concat([win, tot.select(win.columns)]).sort("band", "book_count", "source")
    w(md_table(win.filter(pl.col("book_count") == 5).drop("book_count")) + "\n")
    w("### Sensitivity to book_count\n")
    w("All sources together, `book_count` 2, 5, 20, 100, the other thresholds at their defaults:\n")
    w(md_table(win.filter(pl.col("source") == "all").drop("source")) + "\n")

    # --- where games leave book, before and after TWIC ------------------------------
    w("## Where games leave book, before and after TWIC\n")
    be = pl.read_parquet(CORPUS / "book_exit.parquet").join(
        g.select("game_id", "source", "in_1800_2500", "in_2100_2300"), on="game_id")
    rows = []
    for src in sources:
        for bname in BANDS:
            s = be.filter((pl.col("source") == src) & pl.col("in_" + bname.replace("-", "_")))
            for bc in BOOK_COUNTS:
                own = f"exit{bc}_only_{src}"
                rows.append({
                    "games of": src, "band": bname, "book_count": bc, "games": s.height,
                    f"median ply, counted over {src} alone": s[own].median() if own in s.columns else None,
                    "median ply, counted over all sources": s[f"exit{bc}_all"].median(),
                    "never out of book before ply 80 (all sources)": s[f"exit{bc}_all"].null_count(),
                })
    t = pl.DataFrame(rows)
    w("`pos_count` counts games with both players rated 1800 or more, any tempo, any split. Before TWIC it "
      "counted broadcast games only; the column \"counted over lichess_broadcast alone\" for broadcast games is "
      "the step-0 number.\n")
    w(md_table(t.filter(pl.col("book_count") == 5).drop("book_count"), "{:.0f}") + "\n")
    w(md_table(t.filter(pl.col("band") == "1800-2500").drop("band"), "{:.0f}") + "\n")

    # --- time trouble (broadcasts, clocks) --------------------------------------------
    time_trouble(games, pos_path, w)

    # --- drops -----------------------------------------------------------------------
    w("## What was dropped, and why\n")
    w(f"Read {stats['games_read']:,} games, kept {stats['games_kept']:,}.\n")
    expl = {
        "result_not_final": "Result `*`: unfinished when the dump was taken, or never updated",
        "online": "Online play: broadcast names with \"Online\", Chess.com events, corporate leagues; TWIC Site ending in `INT` (`chess.com INT`, `ICC INT`)",
        "no_moves": "No moves: forfeits, byes, pairings without a game",
        "duplicate": "Another copy of the same game (surnames, year, moves) was kept: the one with clocks, then more tags",
        "engine": "Engine games: TCEC and computer events, or a BOT title",
        "variant": "Variant tag not Standard; TWIC: Chess960, Fischer Random and freestyle events by name",
        "non_standard_start": "FEN tag with a non-standard start",
        "illegal_move": "An illegal move in the move text, other than a final king move (see below); the whole game is dropped",
        "result_contradicts_final_position": "The final position is mate or stalemate and the result says otherwise",
        "simul": "Simultaneous exhibitions (names; TWIC EventType `simul`)",
        "correspondence": "TWIC EventType `(corr)`",
        "result_other": "Result neither `1-0`, `0-1`, `1/2-1/2` nor `*`",
        "ambiguous_move": "An ambiguous SAN move",
        "invalid_san": "Unreadable SAN",
        "null_move": "A null move (`--`)",
        "result_token_mismatch": "The result at the end of the move text differs from the Result tag",
        "malformed_movetext": "Unbalanced variation brackets",
    }
    reasons = sorted({r for d in stats["dropped_by_source"].values() for r in d}, key=lambda r: -stats["dropped"].get(r, 0))
    t = pl.DataFrame({
        "reason": reasons,
        **{src: [stats["dropped_by_source"][src].get(r, 0) for r in reasons] for src in stats["dropped_by_source"]},
        "meaning": [expl.get(r, "") for r in reasons],
    })
    w(md_table(t) + "\n")
    king_signal(g, stats, w)
    enc = stats.get("file_encodings", {})
    if enc:
        w("Files by encoding: " + "; ".join(f"{s}: " + ", ".join(f"{e} {n}" for e, n in d.items()) for s, d in enc.items()) + ".\n")
    w("Kept, not dropped: team events, whose tournament id is the competition; tiebreak and armageddon games "
      "inside classical events, which are rapid or blitz by their tag, clocks or name, or `unknown`; and games "
      "without ratings, which stay in `games.parquet` but give no positions.\n")

    # --- disk -----------------------------------------------------------------------
    w("## Disk\n")
    for sub in ("raw/lichess_broadcast", "raw/twic", "raw/twic_early", "corpus"):
        p = DATA_ROOT / sub
        if p.exists():
            w(f"- `{sub}/`: {du(p)}")
    w(f"- all of `{DATA_ROOT}`: {du(DATA_ROOT)}")
    for f in ("games.parquet", "games_extra.parquet", "positions.parquet", "book_exit.parquet"):
        p = CORPUS / f
        w(f"- `corpus/{f}`: {p.stat().st_size / 1e9:.2f} GB")
    w(f"- free on the volume: {shutil.disk_usage(DATA_ROOT).free / 1e9:.0f} GB")

    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "step0_stats.md"
    path.write_text("\n".join(out) + "\n")
    print(f"wrote {path}")


def king_signal(g: pl.DataFrame, stats: dict, w) -> None:
    """The arbiter's kings-to-the-centre signal at the end of a game: counts, years, plies."""
    w("### The kings put in the centre\n")
    w("At the end of a broadcast game the arbiter often puts both kings in the centre of the DGT board to show the "
      "result. Rules, for every source: an illegal king move to d4, d5, e4 or e5 followed by at most one more move "
      "ends the game there; then, while the last move is a king move to one of those squares, it is removed, at most "
      "twice. The game is kept with its result. Any other illegal move drops the game.\n")
    cut = pl.col("cut_illegal_central_king")
    strip = pl.col("stripped_central_king_moves") > 0
    t = g.group_by("source", "year").agg(
        cut_at_illegal_central_king=cut.sum(),
        legal_central_king_stripped=strip.sum(),
        of_which_two_stripped=(pl.col("stripped_central_king_moves") == 2).sum(),
    ).filter((pl.col("cut_at_illegal_central_king") + pl.col("legal_central_king_stripped")) > 0).sort("source", "year")
    w(md_table(year_str(t)) + "\n")
    ill = {src: stats["dropped_by_source"][src].get("illegal_move", 0) for src in stats["dropped_by_source"]}
    n_cut_lb = g.filter(cut & (pl.col("source") == LB)).height
    w("Games still dropped for an illegal move: " + ", ".join(f"{s} {n:,}" for s, n in ill.items()) + ". "
      f"Step 0 dropped 4,626 broadcast games for an illegal move; {n_cut_lb:,} broadcast games are now cut at an "
      f"illegal central king move and kept, and {ill.get(LB, 0):,} still dropped; the other "
      f"{4626 - n_cut_lb - ill.get(LB, 0):,} of the 4,626 were not traced one by one "
      "(de-duplication now chooses among more copies, so some of them may be kept from TWIC). "
      "TWIC has no illegal central king moves: its games are typed or cleaned before publication. The cuts stop "
      "after 2023, which suggests the relay began to reject illegal moves then; from then on the signal shows only "
      "as a legal king move, and the legal strips in the broadcasts rise accordingly.\n")
    s = g.filter(strip & ~cut).with_columns(ply=pl.col("n_plies"))  # the ply of the (first) stripped move
    if s.height:
        bins = [0, 40, 60, 80, 100, 120, 10_000]
        labels = ["0-39", "40-59", "60-79", "80-99", "100-119", "120+"]
        bucket = pl.lit(labels[-1])
        for edge, label in reversed(list(zip(bins[1:-1], labels[:-1], strict=True))):
            bucket = pl.when(pl.col("ply") < edge).then(pl.lit(label)).otherwise(bucket)
        d = s.with_columns(bucket=bucket).group_by("source", "bucket").len()
        d = d.pivot(on="bucket", index="source", values="len").fill_null(0)
        w("Legal central king moves stripped (and no illegal one cut), by the ply of the stripped move:\n")
        w(md_table(d.select("source", *[c for c in labels if c in d.columns])) + "\n")
        early = s.filter(pl.col("ply") < 60)
        if early.height:
            ex = early.sample(min(20, early.height), seed=SEED).select(
                "source", pl.col("event").str.slice(0, 40), "date", "white", "black", "result", "ply",
                pl.col("moves_uci").str.split(" ").list.tail(4).list.join(" ").alias("last moves kept"), "game_url")
            w(f"{early.height:,} of them before ply 60 ({early.height / s.height:.1%}); a sample of {ex.height}:\n")
            w(md_table(ex) + "\n")
            w(KING_STRIP_FINDINGS)


def near_duplicates(g: pl.DataFrame) -> str:
    """TWIC games from 2020 on that did not match a broadcast game but plausibly are one."""
    lines = []
    cols = ["game_id", "white", "black", "date", "year", "moves_uci", "n_plies", "tournament_id", "split", "event", "round"]
    tw = g.filter((pl.col("copy_sources") == [TWIC]) & (pl.col("year") >= 2020)).select(*cols, "source")
    lb = g.filter(pl.col("copy_sources") == [LB]).select(*cols, "broadcast_name")
    lb = lb.filter(pl.col("year") >= 2020)

    def add_names(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            ws=pl.col("white").map_elements(surname, return_dtype=pl.String),
            bs=pl.col("black").map_elements(surname, return_dtype=pl.String),
        )

    tw, lb = add_names(tw), add_names(lb)
    # 1. same date and surnames, different moves
    pair = tw.join(lb, on=["date", "ws", "bs"], suffix="_lb").filter(pl.col("date").is_not_null())
    a, b = pl.col("moves_uci"), pl.col("moves_uci_lb")
    pair = pair.with_columns(
        kind=pl.when(a.str.starts_with(b) | b.str.starts_with(a)).then(pl.lit("one is a truncated copy of the other"))
        .when(a.str.split(" ").list.head(20).list.join(" ") == b.str.split(" ").list.head(20).list.join(" "))
        .then(pl.lit("same first 20 plies, different later"))
        .otherwise(pl.lit("different before ply 20 (another game the same day)")),
        same_split=pl.col("split") == pl.col("split_lb"),
    )
    t = pair.group_by("kind").agg(pairs=pl.len(), twic_games=pl.col("game_id").n_unique(),
                                  in_different_splits=(~pl.col("same_split")).sum()).sort("pairs", descending=True)
    lines.append(f"Unmatched TWIC games from 2020 on: {tw.height:,}. Of these, {pair['game_id'].n_unique():,} have a "
                 "broadcast game on the same date between players with the same surnames (white and black in the same "
                 "order) but with different moves:\n")
    lines.append(md_table(t) + "\n")
    # 2. same date and moves, surnames differ
    pm = tw.join(lb, on=["date", "moves_uci"], suffix="_lb").filter(pl.col("date").is_not_null())
    pm = pm.filter(pl.col("n_plies") >= 20)
    lines.append(f"Same date and the same moves (at least 20 plies) but surnames that the key reads differently: "
                 f"**{pm['game_id'].n_unique():,}** TWIC games, "
                 f"{(pm['split'] != pm['split_lb']).sum():,} of the pairs in different splits.\n")
    if pm.height:
        s = pm.sample(min(10, pm.height), seed=SEED).select("white", "black", "white_lb", "black_lb", "date")
        lines.append(md_table(s) + "\n")
    # 3. same surnames and moves, different year (the key's year)
    py = tw.join(add_names(g.filter(pl.col("copy_sources") == [LB]).select(*cols)), on=["ws", "bs", "moves_uci"], suffix="_lb")
    py = py.filter((pl.col("year") != pl.col("year_lb")) & (pl.col("n_plies") >= 20))
    lines.append(f"Same surnames and moves (at least 20 plies), different year: **{py['game_id'].n_unique():,}** TWIC games.\n")
    near = pair.filter(pl.col("kind") != "different before ply 20 (another game the same day)")
    if near.height:
        s = near.sample(min(10, near.height), seed=SEED).select(
            "white", "black", "date", "n_plies", "n_plies_lb", "kind", pl.col("event").str.slice(0, 30),
            pl.col("broadcast_name").str.slice(0, 40))
        lines.append("A sample of truncated or late-diverging pairs:\n")
        lines.append(md_table(s) + "\n")
    return "\n".join(lines)


def time_trouble(games: pl.DataFrame, pos_path: Path, w) -> None:
    w("## Time trouble by ply\n")
    w("Classical games with clocks (`has_clock`, broadcasts only; TWIC has no clocks), both players 1800-2500. "
      "The clock is the mover's, before the move. Ply 50-59 is moves 26-30. Controls are `base_s+inc_s` of the first "
      "period (from the tag, or estimated from the clocks).\n")
    gc = games.filter((pl.col("tempo") == "classical") & pl.col("has_clock")).select(
        gid_int().alias("gid"), control=(pl.col("base_s") // 60).cast(pl.String) + "+" + pl.col("inc_s").cast(pl.String))
    top = gc.group_by("control").len().sort("len", descending=True).head(6)["control"].to_list()
    gc = gc.with_columns(control=pl.when(pl.col("control").is_in(top)).then(pl.col("control")).otherwise(pl.lit("other")))
    pt = (
        pl.scan_parquet(pos_path)
        .select(gid_int().alias("gid"), "ply", "clock_mover_s", "mover_elo", "opp_elo")
        .filter(band_expr("mover_elo", "opp_elo", 1800, 2500) & pl.col("clock_mover_s").is_not_null() & (pl.col("ply") >= 30))
        .join(gc.lazy(), on="gid", how="inner")
        .with_columns(bucket=(pl.col("ply") // 10 * 10))
        .collect()
    )
    t = pt.group_by("control", "bucket").agg(moves=pl.len(), lt10=(pl.col("clock_mover_s") < 600).mean(),
                                              lt5=(pl.col("clock_mover_s") < 300).mean())
    for label, col, lim in (("< 10 minutes", "lt10", 600), ("< 5 minutes", "lt5", 300)):
        p = t.with_columns(bucket=pl.format("ply {}-{}", pl.col("bucket"), pl.col("bucket") + 9)).pivot(on="bucket", index="control", values=col)
        allc = pt.group_by("bucket").agg(v=(pl.col("clock_mover_s") < lim).mean())
        allrow = {"control": "all"} | {f"ply {b}-{b + 9}": v for b, v in allc.iter_rows()}
        p = pl.concat([p, pl.DataFrame([allrow])], how="diagonal")
        order = ["control"] + [f"ply {a}-{b}" for a, b in PLY_BUCKETS]
        counts = t.group_by("control").agg(moves=pl.col("moves").sum())
        p = p.join(counts, on="control", how="left").select(*order, "moves")
        w(f"Share of moves made with **{label}** on the mover's clock:\n")
        w(md_table(p.sort("moves", descending=True, nulls_last=True)) + "\n")
    gp = pt.group_by("gid", (pl.col("ply") % 2).alias("side")).agg(
        first_lt10=pl.col("ply").filter(pl.col("clock_mover_s") < 600).min(),
        first_lt5=pl.col("ply").filter(pl.col("clock_mover_s") < 300).min())
    n = gp.height
    t = pl.DataFrame({
        "before ply": [40, 50, 60, 70, 80],
        "player-games below 10 min": [gp.filter(pl.col("first_lt10") < c).height / n for c in (40, 50, 60, 70, 80)],
        "player-games below 5 min": [gp.filter(pl.col("first_lt5") < c).height / n for c in (40, 50, 60, 70, 80)],
    })
    w(f"Share of player-games ({n:,}) where the mover's clock went below the limit at some ply from 30 up to the given ply:\n")
    w(md_table(t) + "\n")


if __name__ == "__main__":
    main()
