"""Every analysis command end to end on synthetic samples, then the report.

The synthetic "Maia" at offset o has logits z * (1 + o / 400); the moves are drawn
from offset +100, so the sweep's best offset must come out near +100.
"""

import json

import numpy as np
import polars as pl

from tnh.eval import book, breakdown, logit, logloss, mistakes, report, sweep, theory, value

from _synth import positions, softmax_rows, write_features

OFFSETS = [-100, 0, 100, 200, 300]


def make(tmp, split, n, rng, offset_true=100):
    part = 0 if split == "test" else 1
    df, moves = positions(n, rng, split=split, n_tournaments=15)
    z = [rng.normal(0, 1.2, len(m)) for m in moves]
    values = [np.clip(0.5 + 0.1 * x + rng.normal(0, 0.02, len(x)), 0, 1) for x in z]
    truth = softmax_rows([zz * (1 + offset_true / 400) for zz in z])
    idx = [int(rng.choice(len(p), p=p)) for p in truth]
    res = rng.choice([0.0, 0.5, 1.0], n, p=[0.3, 0.4, 0.3])
    df = df.with_columns(
        pl.Series("played_uci", [m[i] for m, i in zip(moves, idx)]),
        pl.Series("clock_mover_s", rng.integers(30, 5400, n)),
        pl.Series("out_of_book", rng.random(n) < 0.8),
        pl.Series("pos_count", rng.choice([0, 1, 3, 10, 50, 500, 5000], n)),
        pl.Series("exit_ply_5", rng.integers(5, 21, n)),
        pl.Series("ply_band", rng.choice(["0-7", "12-15", "24-29"], n)),
        pl.Series("pos_count_band", rng.choice(["1", "5-19", "1000-9999"], n)),
        pl.Series("rating_group", rng.choice(["1800-2149", "2150-2500"], n)),
        pl.Series("max_child_count", rng.integers(5, 50, n)),
        pl.Series("n_max_children", np.ones(n, dtype=np.int64)),
        pl.lit(False).alias("played_ends_game"),
        pl.lit(0, dtype=pl.Int64).alias("n_other_at_max"),
        pl.Series("children_count_sum", rng.integers(60, 100, n)),
        pl.lit("window").alias("part"),
        pl.Series("result_mover", res),
    )
    df = df.with_columns(pl.when(pl.Series(rng.random(n) < 0.5)).then(pl.col("max_child_count"))
                         .otherwise(pl.col("max_child_count") // 2).alias("played_child_count"))
    df = df.with_columns(pl.when(pl.col("played_child_count") == pl.col("max_child_count"))
                         .then(pl.col("max_child_count") // 2).otherwise(pl.col("max_child_count") - 1)
                         .alias("max_other_child_count"))
    # some rows outside the window, as a real sample has
    df = df.with_columns((pl.col("clock_mover_s") >= 600).alias("in_window"))
    wdl = rng.dirichlet([2, 2, 2], n)
    # each offset also gets its own small perturbation, as the real network's outputs are not exactly
    # a rescaling of one another (exact collinearity would leave the logit unidentified)
    pert = {o: [rng.normal(0, 0.05, len(zz)) for zz in z] for o in OFFSETS if o != offset_true}
    for o in OFFSETS:
        lp = [np.log(p) for p in softmax_rows([zz * (1 + o / 400) + (pert[o][i] if o in pert else 0)
                                               for i, zz in enumerate(z)])]
        write_features(tmp / "maia" / f"offset{o:+d}", df, moves, {"logp": lp}, wdl=wdl, part=part,
                       meta={"model": "maia-synth", "variant": f"offset{o:+d}"})
    for nodes, noise in ((10, 0.05), (100, 0.01)):
        v = [np.clip(x + rng.normal(0, noise, len(x)), 0, 1) for x in values]
        cp = [400 * (x - 0.5) for x in v]
        mate = [np.zeros(len(x), dtype=np.int16) for x in v]
        write_features(tmp / "sf" / f"nodes{nodes}", df, moves, {"value": v, "cp": cp, "mate": mate}, wdl=wdl, part=part,
                       meta={"model": "sf-synth", "variant": f"nodes{nodes}"})
    path = tmp / "samples" / f"{split}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


def test_all_commands_and_the_report(tmp_path):
    rng = np.random.default_rng(0)
    feats = tmp_path / "f"
    test = make(feats, "test", 3000, rng)
    fit = make(feats, "train", 3000, rng)   # same feature dirs, other game ids
    R = tmp_path / "results"
    C = ["--test", str(test), "--fit", str(fit), "--results", str(R), "--boot", "50"]
    M = [str(feats / "maia" / f"offset{o:+d}") for o in OFFSETS]
    logloss.main(["--model", M[1], M[2]] + C)
    sweep.main(["--variants", *M] + C)
    sw = json.loads((R / "sweep.json").read_text())
    pooled = [p["test"]["est"] for p in sw["per_offset"]]
    assert OFFSETS[int(np.argmin(pooled))] == 100
    assert sw["compare"]["single_offset"] == 100
    mistakes.main(["--variants", *M, "--shallow", str(feats / "sf/nodes10"), "--deep", str(feats / "sf/nodes100"),
                   "--test", str(test), "--results", str(R), "--boot", "20", "--robust-boot", "2"])
    mistakes.main(["--variants", *M, "--shallow", str(feats / "sf/nodes10"), "--deep", str(feats / "sf/nodes100"),
                   "--glance", str(feats / "sf/nodes10"), "--prior", M[1], "--name", "mistakes_glance",
                   "--test", str(test), "--results", str(R), "--boot", "10", "--robust-boot", "2"])
    mg = json.loads((R / "mistakes_glance.json").read_text())
    assert set(mg["glance"]["pooled"]) == {"sound", "at_a_glance", "hidden"} and len(mg["robustness_glance"]) == 15
    logit.main(["--logp", f"m0={M[1]}", "--logp", f"mbest={M[2]}", "--engine", f"sf10={feats / 'sf/nodes10'}",
                "--engine", f"sf100={feats / 'sf/nodes100'}", "--set", "A=mbest", "--set", "B=m0,sf10,sf100",
                "--set", "C=m0,mbest,sf10,sf100", "--full", "C", "--boot-fit", "5"] + C)
    lg = json.loads((R / "logit.json").read_text())
    # the logit on m0 alone would recover the temperature 1.25; here A is mbest, which is exact: beta ~ 1
    assert abs(lg["sets"]["A"]["beta"][0] - 1.0) < 0.15
    breakdown.main(["--model", M[2]] + C)
    book.main(["--variants", *M, "--shallow", str(feats / "sf/nodes10"), "--deep", str(feats / "sf/nodes100"),
               "--min-rows", "20"] + C)
    bk = json.loads((R / "book.json").read_text())
    # moves are drawn from offset +100 everywhere, so every well-filled cell matches about +100
    allc = [c for c in bk["cells"] if c["band"] == "all window rows" and "offset" in c and c["rows"] > 300]
    assert allc and all(abs(c["offset"]["est"] - 100) < 60 for c in allc)
    theory.main(["analyse", "--variants", *M, "--sample", str(test), "--results", str(R), "--boot", "30",
                 "--min-rows", "50"])
    th = json.loads((R / "theory.json").read_text())
    assert all(abs(c["best"]["est"] - 100) < 80 for c in th["cells"] if "best" in c and c["rows"] > 300)
    assert all(abs(c["most_common_share"]["est"] - 0.5) < 0.1 for c in th["cells"] if "best" in c and c["rows"] > 300)
    assert "all" in th["theory_end"]
    value.main(["--model", M[1], str(feats / "sf/nodes100")] + C)
    out = tmp_path / "step1.md"
    report.main(["--results", str(R), "--out", str(out), "--note", "synthetic"])
    text = out.read_text()
    for h in ("## The Elo sweep", "## Is classical play a rating-shifted blitz?", "## The conditional logit",
              "## The value head", "## Breakdowns", "### Play just out of book", "## Where theory ends"):
        assert h in text
    assert (R / "sweep.png").exists() and (R / "mistakes.png").exists() and (R / "theory.png").exists()
