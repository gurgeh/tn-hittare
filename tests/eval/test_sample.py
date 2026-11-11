"""The sample is stratified on the mover's rating, weighted, and reproducible."""

from types import SimpleNamespace

import numpy as np
import polars as pl

from tnh.eval.sample import allocate, draw, population


def corpus(rng, n_games=400):
    games, rows = [], []
    for g in range(n_games):
        gid = f"{g:016x}"
        we, be = int(rng.integers(1700, 2700)), int(rng.integers(1700, 2700))
        split = "test" if g % 3 == 0 else "train"
        src = "assumed" if g % 17 == 0 else "tag"
        games.append({"game_id": gid, "tournament_id": f"t{g % 25}", "tempo": "classical", "tempo_source": src,
                      "has_clock": g % 13 != 0, "split": split, "white_elo": we, "black_elo": be})
        for ply in range(60):
            rows.append({"game_id": gid, "ply": ply, "mover_elo": we if ply % 2 == 0 else be,
                         "opp_elo": be if ply % 2 == 0 else we, "pos_count": max(0, 100 - 10 * ply),
                         "clock_mover_s": None if ply < 2 else 5400 - 100 * ply, "split": split})
    return pl.DataFrame(rows).lazy(), pl.DataFrame(games)


def args(**kw):
    a = dict(split="test", n=700, extra_frac=0.25, extra_min_ply=20, elo_min=1800, elo_max=2500, seed=1,
             book_count=5, max_ply=60, max_ply_no_clock=40, min_clock_s=600)
    a.update(kw)
    return SimpleNamespace(**a)


def test_population_filters_games_and_flags_the_window():
    rng = np.random.default_rng(0)
    pos, games = corpus(rng)
    pop = population(pos, games, args())
    g = games.filter(pl.col("game_id").is_in(pop.get_column("game_id").unique().to_list()))
    assert (g.get_column("split") == "test").all() and g.get_column("has_clock").all()
    assert (g.get_column("tempo_source") != "assumed").all()
    assert g.select(pl.col("white_elo").is_between(1800, 2500).all()).item()
    # out of book from ply 10 (pos_count 0 < 5 at ply 10), the clock drops below 600 at ply 49
    w = pop.filter(pl.col("in_window"))
    assert w.get_column("ply").min() == 10 and w.get_column("ply").max() == 48


def test_stratified_reproducible_and_nested():
    rng = np.random.default_rng(1)
    pos, games = corpus(rng)
    pop = population(pos, games, args())
    s1, st1 = draw(pop, args())
    s2, _ = draw(pop.sample(fraction=1.0, shuffle=True, seed=3), args())
    assert s1.equals(s2)                                   # row order of the input does not matter
    win = s1.filter(pl.col("part") == "window")
    counts = win.group_by("bucket").len().sort("bucket")
    assert counts.get_column("bucket").to_list() == list(range(1800, 2500, 100))   # 2500 folds into 2400
    c = counts.get_column("len").to_numpy()
    assert c.max() - c.min() <= 1 and c.sum() == 700
    # weights: population rows per sampled row in each stratum
    for st in st1:
        rows = s1.filter((pl.col("part") == st["part"]) & (pl.col("bucket") == st["bucket"]))
        assert rows.height == st["sampled"]
        assert np.allclose(rows.get_column("weight").to_numpy(), st["population"] / st["sampled"])
    assert s1.filter(pl.col("part") == "extra").height == 175
    assert not s1.filter(pl.col("part") == "extra").get_column("in_window").any()
    # a larger n with the same seed is a superset; another seed is another sample
    big, _ = draw(pop, args(n=1000))
    k1 = set(map(tuple, win.select("game_id", "ply").rows()))
    kb = set(map(tuple, big.filter(pl.col("part") == "window").select("game_id", "ply").rows()))
    assert k1 <= kb
    # a bucket with too few rows gives all it has, the others share the rest
    capped, st = draw(pop, args(n=5000))
    win_st = [x for x in st if x["part"] == "window"]
    assert sum(x["sampled"] for x in win_st) == min(5000, sum(x["population"] for x in win_st))
    assert any(x["sampled"] == x["population"] for x in win_st)
    other, _ = draw(pop, args(seed=2))
    assert not other.select("game_id", "ply").equals(s1.select("game_id", "ply"))


def test_allocate_water_fills():
    assert allocate({1: 10, 2: 100, 3: 100}, 90) == {1: 10, 2: 40, 3: 40}
    assert allocate({1: 5, 2: 5}, 100) == {1: 5, 2: 5}
    a = allocate({1: 50, 2: 50, 3: 50}, 100)
    assert sum(a.values()) == 100 and max(a.values()) - min(a.values()) <= 1
