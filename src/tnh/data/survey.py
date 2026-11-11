"""Tag-frequency table over a sample of the raw dumps.

    uv run python -m tnh.data.survey [--every 20] [--top 25]

Reads every N-th game of every monthly file (headers only, no move parsing)
and writes ``reports/tag_survey_<source>.md``: how often each tag is present, and the
most common values of the tags that decisions are based on. This is the
evidence behind the choices in ``tnh.data.parse``.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict

from tnh.data.pgn import iter_file_games, split_game
from tnh.data.sources import SOURCES
from tnh.paths import REPORTS

VALUE_TAGS = [
    "EventType", "EventRounds", "EventCategory", "EventDate", "TimeControl", "Variant", "Result", "WhiteTitle", "Round", "Site", "Event",
    "BroadcastName", "Termination", "Annotator", "Board", "WhiteTeam",
]


def url_shape(url: str) -> str:
    """BroadcastURL with slugs replaced by placeholders, to see its structure."""
    m = re.match(r"https://lichess\.org/broadcast/([^/]+)/([^/]+)/([^/]+)$", url)
    if not m:
        return "other: " + url[:60]
    return "/broadcast/<tour>/" + ("-" if m.group(2) == "-" else "<round>") + "/<id>"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--source", default="lichess_broadcast", choices=sorted(SOURCES))
    args = ap.parse_args(argv)

    files = SOURCES[args.source].files()
    presence: Counter[str] = Counter()
    values: dict[str, Counter[str]] = defaultdict(Counter)
    url_shapes_by_year: dict[str, Counter[str]] = defaultdict(Counter)
    tag_years: dict[str, Counter[str]] = defaultdict(Counter)
    games_by_year: Counter[str] = Counter()
    n = 0
    total = 0
    for path in files:
        for i, text in enumerate(iter_file_games(path)):
            total += 1
            if i % args.every:
                continue
            headers, _ = split_game(text)
            year = headers.get("Date", "????")[:4]
            n += 1
            presence.update(headers.keys())
            for t in VALUE_TAGS:
                values[t][headers.get(t, "<absent>")] += 1
            if "BroadcastURL" in headers:
                url_shapes_by_year[year][url_shape(headers["BroadcastURL"])] += 1
            tag_years[year].update(headers.keys())
            games_by_year[year] += 1
            for side in ("White", "Black"):
                elo = headers.get(f"{side}Elo")
                values["Elo form"][
                    "<absent>" if elo is None else ("digits" if elo.isdigit() else repr(elo)[:20])
                ] += 1
                fid = headers.get(f"{side}FideId")
                values["FideId form"][
                    "<absent>" if fid is None else ("digits" if fid.isdigit() else repr(fid)[:20])
                ] += 1

    lines = [
        f"# Tag survey: {args.source}",
        "",
        f"Source `{args.source}`. Every {args.every}th game of {len(files)} files: {n} of {total} games.",
        f"Regenerate with `uv run python -m tnh.data.survey --source {args.source} --every {args.every}`.",
        "",
        "## Tag presence",
        "",
        "| tag | games | share |",
        "|---|---:|---:|",
    ]
    for tag, c in presence.most_common():
        lines.append(f"| {tag} | {c} | {c / n:.1%} |")
    key_tags = ["TimeControl", "EventType", "WhiteElo", "WhiteFideId", "WhiteTitle", "EventDate", "Round"]
    lines += ["", "## Key tags by year (share of games)", "", "| year | games | " + " | ".join(key_tags) + " |",
              "|---|---:|" + "---:|" * len(key_tags)]
    for year in sorted(games_by_year):
        cells = [f"{tag_years[year][t] / games_by_year[year]:.0%}" for t in key_tags]
        lines.append(f"| {year} | {games_by_year[year]} | " + " | ".join(cells) + " |")
    lines += ["", "## BroadcastURL shape by year", ""]
    for year in sorted(url_shapes_by_year):
        shapes = ", ".join(f"`{s}` {c}" for s, c in url_shapes_by_year[year].most_common(4))
        lines.append(f"- {year}: {shapes}")
    for t in [*VALUE_TAGS, "Elo form", "FideId form"]:
        cnt = values[t]
        lines += ["", f"## {t}", "", f"{len(cnt)} distinct values.", "", "| value | games |", "|---|---:|"]
        for v, c in cnt.most_common(args.top):
            lines.append(f"| `{v[:80]}` | {c} |")
    out = REPORTS / f"tag_survey_{args.source}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
