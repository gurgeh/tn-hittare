"""Parsing, de-duplication and the tournament split, on a hand-made PGN fixture."""

from pathlib import Path

import pytest

from tnh.data.parse import GameError, assemble, parse_movetext, parse_texts, surname
from tnh.data.pgn import iter_file_games, split_game
from tnh.data.sources import SOURCES, normalize_tour_name

FIXTURE = Path(__file__).parent / "fixtures" / "broadcasts.pgn"
SRC = SOURCES["lichess_broadcast"]


@pytest.fixture(scope="module")
def parsed():
    df, drops, examples = parse_texts(iter_file_games(FIXTURE), SRC, "fixture")
    games, extra, n_dup, _ = assemble(df)
    return df, drops, games, extra, n_dup


def by_url(games, extra, suffix):
    gid = extra.filter(extra["game_url"].str.ends_with(suffix))["game_id"]
    assert len(gid) == 1, suffix
    return games.filter(games["game_id"] == gid[0]).row(0, named=True)


def test_split_game_keeps_tags_verbatim_and_handles_nel():
    # U+0085 inside a tag value (mis-encoded UTF-8 in real dumps) must not split the line
    text = '[Event "a\x85b"]\n[Result "1-0"]\n\n1. e4 1-0\n'
    h, moves = split_game(text)
    assert h == {"Event": "a\x85b", "Result": "1-0"}
    assert moves == "1. e4 1-0"


def test_movetext_clocks_variations_and_annotations():
    mt = (
        "1. e4 { [%eval 0.2] [%clk 1:30:43] } 1... e5 { [%clk 1:30:42] } "
        "2. Nf3?! { [%eval 0.1] } { Inaccuracy. } { [%clk 1:29:00] } (2. f4 exf4 { [%clk 0:00:01] }) "
        "2... Nc6 3. Bb5 { [%clk 1:28:00.5] } 1-0"
    )
    ucis, clocks, token, cut = parse_movetext(mt)
    assert ucis == ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]
    assert clocks == [5443, 5442, 5340, None, 5280]
    assert token == "1-0" and not cut


def test_movetext_illegal_move_raises():
    with pytest.raises(GameError) as e:
        parse_movetext("1. e4 e5 2. Nf6 Nc6 1-0")
    assert e.value.reason == "illegal_move"
    # an illegal king move with two more moves after it is not the kings-to-the-centre ending
    with pytest.raises(GameError):
        parse_movetext("1. e4 e5 2. Ke3 Nc6 3. d4 1-0")


def test_final_illegal_king_move_cuts_the_game():
    # the arbiter puts both kings in the centre: Ke4 (illegal) and Ke5 follow the last real move
    ucis, clocks, token, cut = parse_movetext("1. e4 { [%clk 1:30:00] } e5 2. Nf3 Nc6 3. Ke4 { [%clk 1:00:00] } Ke5 1-0")
    assert ucis == ["e2e4", "e7e5", "g1f3", "b8c6"] and cut and token == "1-0"
    assert clocks == [5400, None, None, None]
    ucis, _, _, cut = parse_movetext("1. e4 e5 2. Nf3 Nc6 3. Ke4 1/2-1/2")
    assert len(ucis) == 4 and cut


def test_drops_are_counted(parsed):
    _, drops, games, _, n_dup = parsed
    assert drops["variant"] == 1
    assert drops["illegal_move"] == 1  # "Nf6" in game g0000012
    assert drops["result_not_final"] == 1
    assert n_dup == 1
    assert len(games) == 8  # g0000009 is kept, cut before its final "Ke3"


def test_dedup_keeps_the_copy_with_clocks(parsed):
    _, _, games, extra, _ = parsed
    row = by_url(games, extra, "g0000007")
    assert row["has_clock"] is True
    assert row["white"] == "Bo Beta"
    assert extra.filter(extra["game_id"] == row["game_id"])["n_copies"][0] == 2
    assert extra.filter(extra["game_url"].str.ends_with("g0000006")).is_empty()


def test_duplicate_joins_the_tournaments_of_its_copies(parsed):
    df, _, games, extra, _ = parsed
    # the losing copy's broadcast ("Club Relay 2025") has no other game; its node
    # must have been joined to the winning copy's, so the kept game's id is the
    # smaller of the two names
    row = by_url(games, extra, "g0000007")
    assert row["tournament_id"] == "lb/name:club relay 2025"


def test_tournament_is_constant_within_an_event(parsed):
    _, _, games, extra, _ = parsed
    open_r1 = by_url(games, extra, "g0000001")
    open_r2 = by_url(games, extra, "g0000002")
    women = by_url(games, extra, "g0000003")
    assert open_r1["tournament_id"] == open_r2["tournament_id"] == women["tournament_id"]
    assert open_r1["split"] == open_r2["split"] == women["split"]
    # 2020-style broadcasts, one per round, joined through the Event tag
    old1 = by_url(games, extra, "g0000004")
    old2 = by_url(games, extra, "g0000005")
    assert old1["tournament_id"] == old2["tournament_id"]
    assert old1["split"] == old2["split"]
    assert old1["tournament_id"] != open_r1["tournament_id"]


def test_game_fields(parsed):
    _, _, games, extra, _ = parsed
    g = by_url(games, extra, "g0000001")
    assert g["white_elo"] == 2210 and g["black_elo"] == 2105
    assert g["white_fide_id"] == 1001 and g["white_title"] == "FM" and g["black_title"] is None
    assert g["tempo"] == "classical" and g["tempo_source"] == "tag"
    assert (g["base_s"], g["inc_s"]) == (5400, 30)
    assert g["has_clock"] is True and g["n_plies"] == 10
    assert g["clocks_s"][:2] == [5455, 5450]
    assert str(g["date"]) == "2025-03-01" and g["year"] == 2025
    assert len(g["game_id"]) == 16
    # no clock, no tag, nothing in the name
    g2 = by_url(games, extra, "g0000004")
    assert g2["tempo"] == "unknown" and g2["tempo_source"] == "none" and g2["clocks_s"] is None
    # no tag, 15-minute clocks
    w = by_url(games, extra, "g0000003")
    assert w["tempo"] == "rapid" and w["tempo_source"] == "clock" and w["base_s"] == 900


def test_surname_and_tour_name():
    assert surname("Goryachkina, Aleksandra") == "goryachkina"
    assert surname("Magnus Carlsen") == "carlsen"
    assert surname("Ødegaard, Ø") == "degaard"
    assert normalize_tour_name("44th Chess Olympiad 2022 | Open | Boards 1-40") == "44th chess olympiad 2022"
    assert normalize_tour_name("2020 Gibraltar Masters - Round 6") == "2020 gibraltar masters"
    assert normalize_tour_name("Kragerø Resort Chess 2020 - Game 12") == "kragerø resort chess 2020"


def test_round_tempo(parsed):
    _, _, games, extra, _ = parsed
    g = by_url(games, extra, "g0000011")  # no tag, no clock; round 1 of the Open is classical by tag
    assert (g["tempo"], g["tempo_source"]) == ("classical", "round")
    assert g["base_s"] is None
    # a round where nothing is known stays unknown
    assert by_url(games, extra, "g0000004")["tempo_source"] == "none"


def test_game_id_is_the_dedup_key_without_source(parsed):
    from tnh.data.parse import dedup_key

    _, _, games, extra, _ = parsed
    g = by_url(games, extra, "g0000007")
    assert g["game_id"] == dedup_key("Beta, Bo", "Theta, Tor", 2025, g["moves_uci"])
    assert g["game_id"] == dedup_key("Beta,B", "Theta,T", 2025, g["moves_uci"])


def test_kings_to_the_centre_game_is_kept_and_marked(parsed):
    _, _, games, extra, _ = parsed
    g = by_url(games, extra, "g0000009")
    assert g["moves_uci"] == "e2e4 e7e5" and g["result"] == "1-0"
    assert extra.filter(extra["game_url"].str.ends_with("g0000009"))["cut_king_ending"][0] is True
