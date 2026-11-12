"""A tiny games.parquet and positions.parquet from the fixture PGN, built by `tnh.data`.

Private to the model tests. Calls the real corpus builder (parse, assemble, positions)
so that the fixture corpus cannot drift from the contract. Nothing is reimplemented here.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

from tnh.data import positions
from tnh.data.parse import assemble, parse_texts
from tnh.data.pgn import iter_file_games
from tnh.data.sources import SOURCES

FIXTURES = Path(__file__).parent / "fixtures"


@contextlib.contextmanager
def _corpus_dir(out: Path):
    """positions.build reads and writes under its module's CORPUS and keeps its work files under
    DATA_ROOT; point both at `out` for the call, so a test never touches the real data directory.
    (On 2026-11-12 a run of these tests deleted the work files of a full build.)"""
    old = positions.CORPUS, positions.DATA_ROOT
    positions.CORPUS = positions.DATA_ROOT = out
    try:
        yield
    finally:
        positions.CORPUS, positions.DATA_ROOT = old


def build(out: Path, pgn: Path = FIXTURES / "games.pgn", *, max_ply: int = 80,
          source: str = "lichess_broadcast") -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    df, drops, examples = parse_texts(iter_file_games(pgn), SOURCES[source], label=pgn.name)
    if drops:
        print(f"fixture corpus: dropped {dict(drops)}: {examples}")
    games, _extra, _n_dup, _info = assemble(df)
    games_path = out / "games.parquet"
    games.write_parquet(games_path)
    with _corpus_dir(out):
        positions.build(max_ply=max_ply, workers=1, chunk=1000)
    return games_path, out / "positions.parquet"
