"""Clock comments, TimeControl tags and the tempo decision."""

import pytest

from tnh.data.timecontrol import (
    decide_tempo,
    estimate_from_clocks,
    parse_clock,
    parse_free_text,
    parse_standard,
    parse_time_control,
    tempo_from_names,
    tempo_of_seconds,
)


def tempo(tc: str) -> str:
    c, _ = parse_time_control(tc)
    assert c is not None, tc
    return tempo_of_seconds(c.seconds_for(60))


def test_parse_clock():
    assert parse_clock("[%clk 1:29:45]") == 5385
    assert parse_clock("[%eval 0.3] [%clk 0:00:07.4]") == 7
    assert parse_clock("[%clk 12:00:00]") == 43200
    assert parse_clock("Inaccuracy. Best move was Be4.") is None


def test_multi_period_control():
    c = parse_standard("40/5400+30:1800+30")
    assert (c.base_s, c.inc_s) == (5400, 30)
    assert c.seconds_for(60) == 5400 + 1800 + 60 * 30
    assert c.bonus_after_move(40) == 1800
    assert c.bonus_after_move(39) == 0 and c.bonus_after_move(41) == 0
    assert tempo("40/5400+30:1800+30") == "classical"
    # "5400/40" written the wrong way round
    assert parse_standard("5400/40+30:1800").base_s == 5400


def test_bare_seconds_and_minutes_shorthand():
    assert (parse_standard("5400+30").base_s, parse_standard("5400+30").inc_s) == (5400, 30)
    assert parse_standard("90+30").base_s == 5400
    assert parse_standard("3+2").base_s == 180
    assert parse_standard("180+2").base_s == 180  # seconds, not 180 minutes
    assert parse_standard("600").base_s == 600
    assert tempo("5400+30") == "classical"
    assert tempo("90+30") == "classical"
    assert tempo("60+0") == "classical"  # exactly 60 minutes
    assert tempo("45+15") == "classical"  # 45 min + 60 * 15 s = 60 min
    assert tempo("45+10") == "rapid"
    assert tempo("15+10") == "rapid"
    assert tempo("10+2") == "rapid"
    assert tempo("600") == "blitz"  # 10 minutes flat is not more than 10
    assert tempo("3+2") == "blitz"


@pytest.mark.parametrize(
    "text,base,inc,expected",
    [
        ("90 minutes for 40 moves + 30 minutes for the rest of the game + 30 seconds per move from move one.", 5400, 30, "classical"),
        ("100 minutes/40 moves + 40 minutes/rest with 30 seconds increment from move 1", 6000, 30, "classical"),
        ("40 moves in 100 f.b. 50 mins with 30 secs per move", 6000, 30, "classical"),
        ("90/40 + 30 min with 30 sec increment", 5400, 30, "classical"),
        ("90m+30s", 5400, 30, "classical"),
        ("1:30:30", 5400, 0, "classical"),
        ("G: 15 minutes + 10 seconds added per move", 900, 10, "rapid"),
        ("10'+2", 600, 2, "rapid"),
        ("3 min + 2 sec per move", 180, 2, "blitz"),
    ],
)
def test_free_text(text, base, inc, expected):
    c = parse_free_text(text)
    assert (c.base_s, c.inc_s) == (base, inc)
    assert tempo_of_seconds(c.seconds_for(60)) == expected


def test_free_text_with_controls_per_round_is_unreadable():
    tc = "15 minutes + 5 seconds per move from move one (Round 1 - 4) / 90 minutes + 30 seconds per move (Round 5 - 8)"
    assert parse_time_control(tc) == (None, "unreadable")
    assert parse_time_control(None) == (None, "absent")
    assert parse_time_control("-") == (None, "absent")


def test_estimate_from_clocks():
    # 90+30 with the increment given before move 1, a few quick book moves
    clocks = [5455, 5450, 5483, 5478, 5510, 5300, 5535, 5200]
    est = estimate_from_clocks(clocks)
    assert (est.base_s, est.inc_s) == (5400, 30)
    assert estimate_from_clocks([None, None, 100]) is None
    assert estimate_from_clocks([0, 0, 0, 0]) is None  # clock not running


def test_names():
    assert tempo_from_names(["Norway Chess 2025 | Armageddon"]) == "blitz"
    assert tempo_from_names(["British Rapidplay Championship"]) == "rapid"
    assert tempo_from_names(["World Rapid & Blitz 2024"]) is None
    assert tempo_from_names(["Polish Youth Rapid & Blitz Championships 2026 | Blitz | G16"]) == "blitz"
    assert tempo_from_names(["World Rapid & Blitz 2024 | Rapid Open"]) == "rapid"
    assert tempo_from_names(["Some Open", "Tiebreaks"]) == "tiebreak"
    assert tempo_from_names(["Tata Steel Masters"]) is None


def test_decide_tempo_order():
    classical_clocks = [5455, 5450, 5480, 5470, 5500, 5300]
    blitz_clocks = [183, 182, 185, 184, 186, 180, 187, 178]
    # tag, clocks agree
    d = decide_tempo("90+30", classical_clocks, [])
    assert (d.tempo, d.source) == ("classical", "tag")
    # tag says classical, clocks show 3 minutes: the clocks win
    d = decide_tempo("90+30", blitz_clocks, [])
    assert (d.tempo, d.source, d.base_s, d.inc_s) == ("blitz", "clock", 180, 2)
    # no tag: clocks
    d = decide_tempo(None, classical_clocks, ["Blitz Open"])
    assert (d.tempo, d.source) == ("classical", "clock")
    # no tag, no clocks: name
    d = decide_tempo(None, [None] * 6, ["Blitz Open"])
    assert (d.tempo, d.source) == ("blitz", "event_name")
    # tag classical, name blitz, no clock to settle it
    d = decide_tempo("90+30", [None] * 6, ["Blitz Open"])
    assert (d.tempo, d.source) == ("unknown", "none")
    d = decide_tempo(None, [None] * 6, ["Open"])
    assert (d.tempo, d.source) == ("unknown", "none")
