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
    games, extra, n_dup, _ = assemble(pl.concat([lb, tw]))
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


def test_ko_tiebreaks_are_not_assumed_classical():
    from tnh.data.sources import twic_names
    from tnh.data.timecontrol import decide_tempo

    def tempo(event, rnd, et=""):
        h = {"Event": event, "Round": rnd, "EventType": et}
        return decide_tempo(None, [None] * 4, twic_names(h), et, "classical").tempo

    assert tempo("FIDE World Cup 2017", "3.2") == "classical"
    assert tempo("FIDE World Cup 2017", "3.3") == "unknown"
    assert tempo("ch-RUS KO", "1.5") == "unknown"
    assert tempo("Some Cup", "2.4", et="k.o.") == "unknown"
    assert tempo("Politiken Cup", "7.4") == "classical"  # not a knockout: board 4 of round 7
    assert tempo("Amber Blindfold", "3") == "rapid"


def test_engine_names():
    from tnh.data.sources import event_drop_reason

    assert event_drop_reason(["Computer Chess Match Tournament"]) == "engine"
    assert event_drop_reason(["Duras IM Tespo Engineering"]) is None
    assert event_drop_reason(["87th Marshall CCC"]) is None


MOVES22 = "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e5 7. Nb3 Be6 8. f3 Be7 9. Qd2 O-O 10. O-O-O Nbd7 11. g4 b5"


def pgn(event, site, white, black, moves, date="2023.12.27", extra=""):
    return (f'[Event "{event}"]\n[Site "{site}"]\n[Date "{date}"]\n[Round "1"]\n[White "{white}"]\n'
            f'[Black "{black}"]\n[Result "1-0"]\n{extra}\n{moves} 1-0\n')


def test_near_copies_stay_two_games_but_share_a_tournament():
    lb_texts = [
        pgn("Rapid Cup", "?", "Gareyev, Timur", "Nepomniachtchi, Ian", MOVES22,
            extra='[BroadcastName "Some Broadcast 2023"]\n[BroadcastURL "https://lichess.org/broadcast/some/round-1/ZZZZZZZZ"]\n'),
        pgn("Other", "?", "Alpha, Anna", "Beta, Bo", MOVES22 + " 12. g5", date="2023.12.28",
            extra='[BroadcastName "Another Broadcast 2023"]\n[BroadcastURL "https://lichess.org/broadcast/another/round-1/YYYYYYYY"]\n'),
    ]
    tw_texts = [
        pgn("Classic A", "London ENG", "Gareev,T", "Nepomniachtchi,I", MOVES22, extra='[EventDate "2023.12.27"]\n'),
        pgn("Classic B", "Paris FRA", "Alpha,A", "Beta,B", MOVES22, date="2023.12.28", extra='[EventDate "2023.12.27"]\n'),
    ]
    lb, _, _ = parse_texts(lb_texts, SOURCES["lichess_broadcast"], "lb")
    tw, _, _ = parse_texts(tw_texts, SOURCES["twic"], "twic")
    games, extra, n_dup, info = assemble(pl.concat([lb, tw]))
    assert n_dup == 0 and len(games) == 4
    assert info["near_copy_links"] == {"same_date_moves": 1, "truncated": 1}
    t = dict(games.select(pl.col("white"), "tournament_id").iter_rows())
    assert t["Gareev,T"] == t["Gareyev, Timur"]
    assert t["Alpha,A"] == t["Alpha, Anna"]
    assert t["Gareev,T"] != t["Alpha,A"]
