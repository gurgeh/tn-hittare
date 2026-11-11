"""The three window rules."""

import polars as pl

from tnh.data.window import in_window


def frame(rows):
    return pl.DataFrame(rows, schema={"game_id": pl.String, "ply": pl.Int32, "pos_count": pl.Int32, "clock_mover_s": pl.Int32}, orient="row")


def kept(df, **kw):
    return df.filter(in_window(df, **kw)).select("game_id", "ply").rows()


def test_move_out_of_book_is_excluded_and_next_is_included():
    # ply 3 is the first position with pos_count < 5: the move played at ply 2
    # led out of book and is excluded; ply 3 is the first row that counts.
    df = frame([("g", 0, 1000, None), ("g", 1, 300, None), ("g", 2, 40, None), ("g", 3, 4, None), ("g", 4, 1, None), ("g", 5, 1, None)])
    assert kept(df) == [("g", 3), ("g", 4), ("g", 5)]


def test_out_of_book_is_for_good_even_after_transposing_back():
    df = frame([("g", 0, 1000, None), ("g", 1, 2, None), ("g", 2, 50, None), ("g", 3, 1, None)])
    assert kept(df) == [("g", 1), ("g", 2), ("g", 3)]


def test_book_count_threshold_and_never_out_of_book():
    df = frame([("a", 0, 100, None), ("a", 1, 10, None), ("a", 2, 3, None), ("b", 0, 100, None), ("b", 1, 50, None)])
    assert kept(df, book_count=5) == [("a", 2)]
    assert kept(df, book_count=20) == [("a", 1), ("a", 2)]
    assert kept(df, book_count=2) == []


def test_max_ply():
    df = frame([("g", p, 1 if p >= 2 else 100, 3000) for p in range(70)])
    rows = kept(df, max_ply=60)
    assert rows[0] == ("g", 2) and rows[-1] == ("g", 59)


def test_clock_rule_and_missing_clocks():
    df = frame([("g", 0, 100, 5400), ("g", 1, 1, 5400), ("g", 2, 1, 700), ("g", 3, 1, 599), ("g", 4, 1, 600), ("g", 5, 1, None)])
    assert kept(df, min_clock_s=600) == [("g", 1), ("g", 2), ("g", 4), ("g", 5)]


def test_rows_without_clock_stop_at_max_ply_no_clock():
    # no clock at all: plies 2..39 count, 40 does not
    df = frame([("n", p, 100 if p < 2 else 1, None) for p in range(60)])
    rows = kept(df)
    assert rows[0] == ("n", 2) and rows[-1] == ("n", 39)
    # with a clock, the same plies run on to max_ply as long as the clock is >= min_clock_s
    df = frame([("c", p, 100 if p < 2 else 1, 3000) for p in range(70)])
    rows = kept(df)
    assert rows[0] == ("c", 2) and rows[-1] == ("c", 59)
    # mixed: a missing reading at ply 45 is excluded, readings around it count
    df = frame([("m", p, 1, None if p == 45 else 3000) for p in range(50)])
    assert ("m", 45) not in kept(df) and ("m", 44) in kept(df) and ("m", 46) in kept(df)
    assert kept(df, max_ply_no_clock=60)[-1] == ("m", 49)


def test_works_on_lazy_frames():
    df = frame([("g", 0, 100, None), ("g", 1, 1, None)]).lazy()
    assert df.filter(in_window(df)).collect().height == 1
