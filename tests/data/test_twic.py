"""TWIC as a second source: reading, drops, tempo, tournaments, and merging with broadcasts."""

import zipfile
from pathlib import Path

import polars as pl
import pytest

from tnh.data.parse import assemble, parse_texts
from tnh.data.pgn import iter_file_games
from tnh.data.sources import SOURCES

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def merged():
    lb, lb_drops, _ = parse_texts(iter_file_games(FIX / "broadcasts.pgn"), SOURCES["lichess_broadcast"], "lb")
    tw, tw_drops, _ = parse_texts(iter_file_games(FIX / "twic_sample.pgn"), SOURCES["twic"], "twic")
    games, extra, n_dup = assemble(pl.concat([lb, tw]))
    return games.join(extra, on="game_id"), tw_drops, n_dup


def row(df, white_prefix):
    r = df.filter(pl.col("white").str.starts_with(white_prefix))
    assert len(r) == 1, white_prefix
    return r.row(0, named=True)


def test_latin1_and_zip(tmp_path):
    texts = list(iter_file_games(FIX / "twic_sample.pgn"))
    assert '[White "Müller,K"]' in texts[1]
    z = tmp_path / "twic9999g.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(FIX / "twic_sample.pgn", "twic9999.pgn")
    assert list(iter_file_games(z)) == texts


def test_twic_drops(merged):
    _, drops, _ = merged
    assert drops["online"] == 1  # Site "chess.com INT"
    assert drops["correspondence"] == 1


def test_twic_tempo(merged):
    df, _, _ = merged
    assert (row(df, "Sigma")["tempo"], row(df, "Sigma")["tempo_source"]) == ("blitz", "event_name")
    assert (row(df, "Upsilon")["tempo"], row(df, "Upsilon")["tempo_source"]) == ("rapid", "event_type")
    m = row(df, "Müller")
    assert (m["tempo"], m["tempo_source"], m["has_clock"]) == ("classical", "assumed", False)


def test_twic_event_across_new_year_is_one_tournament(merged):
    df, _, _ = merged
    a, b = row(df, "Müller"), row(df, "Ørsted")
    assert a["year"] == 2024 and b["year"] == 2025
    assert a["tournament_id"] == b["tournament_id"] and a["split"] == b["split"]


def test_twic_copy_of_a_broadcast_game_is_merged(merged):
    df, _, n_dup = merged
    copies = df.filter(pl.col("moves_uci") == "e2e4 e7e6 d2d4 d7d5 b1c3 f8b4")
    assert len(copies) == 1
    g = copies.row(0, named=True)
    assert g["source"] == "lichess_broadcast" and g["has_clock"]  # the copy with clocks wins
    assert g["n_copies"] == 3 and g["copy_sources"] == ["lichess_broadcast", "twic"]
    assert n_dup == 2
    # the broadcast tournaments and the TWIC event are one component
    assert g["tournament_id"] == "lb/name:club relay 2025"
