"""Tag-frequency table over a sample of the raw dumps.

    uv run python -m tnh.data.survey [--every 20] [--top 25]

Reads every N-th game of every monthly file (headers only, no move parsing)
and writes ``reports/tag_survey.md``: how often each tag is present, and the
most common values of the tags that decisions are based on. This is the
evidence behind the choices in ``tnh.data.parse``.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict

from tnh.data.download import raw_dir
from tnh.data.pgn import iter_file_games, split_game
from tnh.paths import REPORTS

VALUE_TAGS = [
    "TimeControl", "Variant", "Result", "WhiteTitle", "Round", "Site", "Event",
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
    args = ap.parse_args(argv)

    files = sorted(raw_dir().glob("lichess_db_broadcast_*.pgn.zst"))
    presence: Counter[str] = Counter()
    values: dict[str, Counter[str]] = defaultdict(Counter)
    url_shapes_by_year: dict[str, Counter[str]] = defaultdict(Counter)
    n = 0
    total = 0
    for path in files:
        year = path.name.split("_")[-1][:4]
        for i, text in enumerate(iter_file_games(path)):
            total += 1
            if i % args.every:
                continue
            headers, _ = split_game(text)
            n += 1
            presence.update(headers.keys())
            for t in VALUE_TAGS:
                values[t][headers.get(t, "<absent>")] += 1
            url_shapes_by_year[year][url_shape(headers.get("BroadcastURL", ""))] += 1
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
        "# Tag survey of the Lichess broadcast dumps",
        "",
        f"Every {args.every}th game of {len(files)} monthly files: {n} of {total} games.",
        "Regenerate with `uv run python -m tnh.data.survey`.",
        "",
        "## Tag presence",
        "",
        "| tag | games | share |",
        "|---|---:|---:|",
    ]
    for tag, c in presence.most_common():
        lines.append(f"| {tag} | {c} | {c / n:.1%} |")
    lines += ["", "## BroadcastURL shape by year", ""]
    for year in sorted(url_shapes_by_year):
        shapes = ", ".join(f"`{s}` {c}" for s, c in url_shapes_by_year[year].most_common(4))
        lines.append(f"- {year}: {shapes}")
    for t in [*VALUE_TAGS, "Elo form", "FideId form"]:
        cnt = values[t]
        lines += ["", f"## {t}", "", f"{len(cnt)} distinct values.", "", "| value | games |", "|---|---:|"]
        for v, c in cnt.most_common(args.top):
            lines.append(f"| `{v[:80]}` | {c} |")
    out = REPORTS / "tag_survey.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
