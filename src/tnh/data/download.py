"""Download the Lichess broadcast dumps.

    uv run python -m tnh.data.download [--first 2020-01] [--last 2026-09]

Monthly files from https://database.lichess.org/broadcast/ go to
``raw/lichess_broadcast/``, exactly as fetched. The server publishes
``sha256sums.txt``; a file already on disk is skipped only when its hash
matches, so a truncated download is fetched again. A new download is written to
``<name>.part`` and renamed after its hash is verified.

Polite fetching: one request at a time, a pause between files, a generic
User-Agent, and a one-minute back-off on HTTP 429. Any other HTTP error or a
hash mismatch stops the run.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import time
from pathlib import Path

import httpx

from tnh.paths import RAW

BASE_URL = "https://database.lichess.org/broadcast/"
SOURCE = "lichess_broadcast"
USER_AGENT = "tnh-corpus/0.1 (research download)"
PAUSE_S = 2.0
BACKOFF_429_S = 60.0
MAX_429 = 5
NAME_RE = re.compile(r"^lichess_db_broadcast_(\d{4}-\d{2})\.pgn\.zst$")


def raw_dir() -> Path:
    return RAW / SOURCE


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def get(client: httpx.Client, url: str) -> httpx.Response:
    """GET with a one-minute back-off on 429. Every other error is raised."""
    for attempt in range(MAX_429 + 1):
        r = client.get(url)
        if r.status_code == 429:
            print(f"HTTP 429 on {url}; waiting {BACKOFF_429_S:.0f} s (attempt {attempt + 1})")
            time.sleep(BACKOFF_429_S)
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"HTTP 429 {MAX_429 + 1} times in a row on {url}")


def parse_sums(text: str) -> dict[str, str]:
    """``sha256sums.txt`` -> {month: sha256} for the monthly .pgn.zst files."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, name = line.split()
        m = NAME_RE.match(name.lstrip("*"))
        if m:
            out[m.group(1)] = digest
    return out


def download_file(client: httpx.Client, url: str, dest: Path, digest: str) -> int:
    part = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    n = 0
    for attempt in range(MAX_429 + 1):
        with client.stream("GET", url) as r:
            if r.status_code == 429:
                print(f"HTTP 429 on {url}; waiting {BACKOFF_429_S:.0f} s (attempt {attempt + 1})")
                time.sleep(BACKOFF_429_S)
                continue
            r.raise_for_status()
            with part.open("wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
                    h.update(chunk)
                    n += len(chunk)
            break
    else:
        raise RuntimeError(f"HTTP 429 {MAX_429 + 1} times in a row on {url}")
    if h.hexdigest() != digest:
        raise RuntimeError(
            f"sha256 mismatch for {dest.name}: got {h.hexdigest()}, expected {digest}. "
            f"Left the partial file at {part}"
        )
    part.rename(dest)
    return n


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first", default="2020-01", help="first month, YYYY-MM")
    ap.add_argument("--last", default="2026-09", help="last month, YYYY-MM")
    args = ap.parse_args(argv)

    out = raw_dir()
    out.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=120.0, follow_redirects=True) as client:
        sums_text = get(client, BASE_URL + "sha256sums.txt").text
        (out / "sha256sums.txt").write_text(sums_text)
        sums = {m: d for m, d in parse_sums(sums_text).items() if args.first <= m <= args.last}
        print(f"{len(sums)} monthly files listed in sha256sums.txt between {args.first} and {args.last}")

        skipped = fetched = 0
        nbytes = 0
        for month in sorted(sums):
            name = f"lichess_db_broadcast_{month}.pgn.zst"
            dest = out / name
            if dest.exists():
                if sha256_file(dest) == sums[month]:
                    skipped += 1
                    continue
                print(f"{name}: on disk but the hash does not match; fetching again")
            time.sleep(PAUSE_S)
            n = download_file(client, BASE_URL + name, dest, sums[month])
            nbytes += n
            fetched += 1
            print(f"{name}: {n / 1e6:.1f} MB")
    print(f"done: {fetched} fetched ({nbytes / 1e6:.0f} MB), {skipped} already present and verified")


if __name__ == "__main__":
    main()
