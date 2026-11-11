"""The sample is stratified on the mover's rating, weighted, reproducible, and the same whatever the batch size."""

from types import SimpleNamespace

import numpy as np
import polars as pl

from tnh.eval.sample import allocate, draw_main, row_keys


def corpus(tmp_path, rng, n_games=400, n_ply=60):
    """games.parquet with real random games and the positions.parquet the data developer's row builder makes
    of them (sorted by game_id), with a synthetic pos_count: 100 - 10 * ply, floored at 0."""
    import chess
    from tnh.data.positions import process_chunk

    games = []
    for g in range(n_games):
        gid = f"{(g * 2654435761) % (1 << 64):016x}"
        while True:
            b, moves = chess.Board(), []
            for _ in range(n_ply):
                ms = list(b.legal_moves)
                if not ms:
                    break
                m = ms[int(rng.integers(len(ms)))]
                moves.append(m.uci())
                b.push(m)
            if len(moves) == n_ply and not b.is_game_over():
                break
        we, be = int(rng.integers(1700, 2700)), int(rng.integers(1700, 2700))
        games.append({"game_id": gid, "source": "lichess_broadcast", "tournament_id": f"t{g % 25}",
                      "tempo": "classical", "tempo_source": "assumed" if g % 17 == 0 else "tag",
                      "has_clock": g % 13 != 0, "split": "test" if g % 3 == 0 else "train", "white_elo": we,
                      "black_elo": be, "n_plies": n_ply, "moves_uci": " ".join(moves),
                      "clocks_s": [5400 - 100 * (p + 2) for p in range(n_ply)], "time_control_raw": None,
                      "inc_s": 30, "result": "1/2-1/2"})
    games = pl.DataFrame(games).sort("game_id")
    gp = tmp_path / "games.parquet"
    games.write_parquet(gp, row_group_size=100)
    parts = tmp_path / "parts"
    parts.mkdir()
    import pyarrow.parquet as pq
    for i in range(pq.ParquetFile(gp).metadata.num_row_groups):
        process_chunk((i, str(gp), 80, str(parts)))
    pos = (pl.concat([pl.read_parquet(f) for f in sorted(parts.glob("*.parquet"))]).drop("_h")
           .with_columns(pl.max_horizontal(100 - 10 * pl.col("ply"), pl.lit(0)).cast(pl.Int32).alias("pos_count"))
           .sort("game_id", "ply"))
    path = tmp_path / "positions.parquet"
    pos.write_parquet(path, row_group_size=500)
    return path, games


def args(**kw):
    a = dict(split="test", n=700, extra_frac=0.25, exit_counts=[2, 5, 20, 100], extra_min_ply=20, elo_min=1800,
             elo_max=2500, seed=1, book_count=5, max_ply=60, max_ply_no_clock=40, min_clock_s=600)
    a.update(kw)
    return SimpleNamespace(**a)


def quiet(*_):
    pass


def test_filters_games_and_flags_the_window(tmp_path):
    path, games = corpus(tmp_path, np.random.default_rng(0))
    s, _, _ = draw_main(path, games, args(n=100_000), 50, quiet)     # everything in the window and extra parts
    g = games.filter(pl.col("game_id").is_in(s.get_column("game_id").unique().to_list()))
    assert (g.get_column("split") == "test").all() and g.get_column("has_clock").all()
    assert (g.get_column("tempo_source") != "assumed").all()
    assert g.select(pl.col("white_elo").is_between(1800, 2500).all()).item()
    # out of book from ply 10 (pos_count 0 < 5 at ply 10), the clock drops below 600 at ply 49
    w = s.filter(pl.col("in_window"))
    assert w.get_column("ply").min() == 10 and w.get_column("ply").max() == 48
    # the rebuilt wide columns equal the stored ones
    stored = pl.read_parquet(path).with_columns(pl.col("ply").cast(pl.Int64))
    j = s.select("game_id", "ply", "fen", "pos_key", "played_uci", "n_legal", "clock_opp_s", "pos_count").join(
        stored, on=["game_id", "ply"], how="left", suffix="_s")
    for c in ("fen", "pos_key", "played_uci", "n_legal", "clock_opp_s", "pos_count"):
        assert (j.get_column(c) == j.get_column(f"{c}_s")).all(), c
    for k, ply in ((2, 10), (5, 10), (20, 9), (100, 1)):
        assert (s.get_column(f"exit_ply_{k}") == ply).all()


def test_batching_does_not_change_the_sample(tmp_path):
    # several batches of games against one: the same rows, weights and strata
    path, games = corpus(tmp_path, np.random.default_rng(1), n_games=600)
    one, st1, _ = draw_main(path, games, args(), 10_000, quiet)
    many, st2, _ = draw_main(path, games, args(), 17, quiet)
    assert one.equals(many) and st1 == st2
    big, _, _ = draw_main(path, games, args(split="train", n=3000), 10_000, quiet)
    big2, _, _ = draw_main(path, games, args(split="train", n=3000), 23, quiet)
    assert big.equals(big2)


def test_stratified_reproducible_and_nested(tmp_path):
    path, games = corpus(tmp_path, np.random.default_rng(2))
    s1, st1, _ = draw_main(path, games, args(), 40, quiet)
    win = s1.filter(pl.col("part") == "window")
    counts = win.group_by("bucket").len().sort("bucket")
    assert counts.get_column("bucket").to_list() == list(range(1800, 2500, 100))   # 2500 folds into 2400
    c = counts.get_column("len").to_numpy()
    assert c.max() - c.min() <= 1 and c.sum() == 700
    for st in st1:
        rows = s1.filter((pl.col("part") == st["part"]) & (pl.col("bucket") == st["bucket"]))
        assert rows.height == st["sampled"]
        assert np.allclose(rows.get_column("weight").to_numpy(), st["population"] / st["sampled"])
    assert s1.filter(pl.col("part") == "extra").height == 175
    assert not s1.filter(pl.col("part") == "extra").get_column("in_window").any()
    big, _, _ = draw_main(path, games, args(n=1000), 40, quiet)
    k1 = set(map(tuple, win.select("game_id", "ply").rows()))
    kb = set(map(tuple, big.filter(pl.col("part") == "window").select("game_id", "ply").rows()))
    assert k1 <= kb
    capped, st, _ = draw_main(path, games, args(n=5000), 40, quiet)
    win_st = [x for x in st if x["part"] == "window"]
    assert sum(x["sampled"] for x in win_st) == min(5000, sum(x["population"] for x in win_st))
    assert any(x["sampled"] == x["population"] for x in win_st)
    other, _, _ = draw_main(path, games, args(seed=2), 40, quiet)
    assert not other.select("game_id", "ply").equals(s1.select("game_id", "ply"))


def test_row_keys_match_the_scalar_definition():
    gids = pl.Series(["ffffffffffffffff", "0123456789abcdef", "00000000000000ff"])
    ply = pl.Series([0, 7, 79])
    from tnh.eval.sample import splitmix64
    g = np.array([int(s, 16) for s in gids.to_list()], dtype=np.uint64)
    with np.errstate(over="ignore"):
        x = splitmix64(g ^ splitmix64(np.full(3, 5, dtype=np.uint64)))
        ref = splitmix64(x + ply.to_numpy().astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15))
    assert (row_keys(gids, ply, 5) == ref).all()


def test_allocate_water_fills():
    assert allocate({1: 10, 2: 100, 3: 100}, 90) == {1: 10, 2: 40, 3: 40}
    assert allocate({1: 5, 2: 5}, 100) == {1: 5, 2: 5}
    a = allocate({1: 50, 2: 50, 3: 50}, 100)
    assert sum(a.values()) == 100 and max(a.values()) - min(a.values()) <= 1


def test_theory_sample_is_stratified_on_ply_and_pos_count(tmp_path):
    from tnh.eval import theory
    path, games = corpus(tmp_path, np.random.default_rng(5))
    a = SimpleNamespace(splits=["train", "test"], ply_bands=["0-7", "8-11", "12-15"],
                        rating_groups=["1800-2149", "2150-2500"], elo_min=1800, elo_max=2500, per_cell=60, seed=1)
    s, strata, _ = theory.draw_theory(path, games, a, 30, quiet)
    s2, _, _ = theory.draw_theory(path, games, a, 10_000, quiet)
    assert s.equals(s2)
    assert s.get_column("ply").max() <= 15
    per_cell = s.group_by("ply_band", "pos_count_band").len()
    for pb, pc, n in per_cell.iter_rows():
        avail = sum(x["population"] for x in strata if x["ply_band"] == pb and x["pos_count_band"] == pc)
        assert n == min(60, avail)
    assert set(s.filter(pl.col("ply") == 0).get_column("pos_count_band")) == {"100-999"}
    assert set(s.filter(pl.col("ply") >= 10).get_column("pos_count_band")) == {"0"}
    for st in strata:
        rows = s.filter((pl.col("ply_band") == st["ply_band"]) & (pl.col("pos_count_band") == st["pos_count_band"])
                        & (pl.col("rating_group") == st["rating_group"]))
        assert rows.height == st["sampled"]
        assert np.allclose(rows.get_column("weight").to_numpy(), st["population"] / st["sampled"])


def test_continuation_counts(tmp_path):
    import chess
    from tnh.data.positions import pos_key
    from tnh.eval.theory import with_continuations

    start = chess.Board()
    counts = {}
    for uci, n in (("e2e4", 500), ("d2d4", 500), ("c2c4", 100), ("g1f3", 50)):
        b = start.copy(); b.push_uci(uci); counts[pos_key(b)] = n
    pd = pl.DataFrame({"pos_key": list(counts), "pos_count": list(counts.values())})
    pd.write_parquet(tmp_path / "positions.parquet")
    s = pl.DataFrame({"fen": [start.fen()] * 3, "played_uci": ["e2e4", "c2c4", "a2a3"],
                      "game_id": ["a", "b", "c"], "ply": [0, 0, 0], "n_plies": [40, 40, 1]})
    out = with_continuations(s, tmp_path / "positions.parquet")
    assert out.get_column("played_child_count").to_list() == [500, 100, 0]
    assert out.get_column("max_child_count").to_list() == [500] * 3
    assert out.get_column("n_max_children").to_list() == [2] * 3     # e4 and d4 tie
    assert out.get_column("children_count_sum").to_list() == [1150] * 3
    assert out.get_column("played_ends_game").to_list() == [False, False, True]   # game c has one ply
    assert out.get_column("max_other_child_count").to_list() == [500, 500, 500]
    assert out.get_column("n_other_at_max").to_list() == [1, 2, 2]
