"""Every analysis command end to end on synthetic samples, then the report.

The synthetic "Maia" at offset o has logits z * (1 + o / 400); the moves are drawn
from offset +100, so the sweep's best offset must come out near +100.
"""

import json

import numpy as np
import polars as pl

from tnh.eval import breakdown, logit, logloss, mistakes, report, sweep, value

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
        pl.lit(True).alias("out_of_book"),
        pl.Series("result_mover", res),
    )
    # some rows outside the window, as a real sample has
    df = df.with_columns((pl.col("clock_mover_s") >= 600).alias("in_window"))
    wdl = rng.dirichlet([2, 2, 2], n)
    for o in OFFSETS:
        lp = [np.log(p) for p in softmax_rows([zz * (1 + o / 400) for zz in z])]
        write_features(tmp / "maia" / f"offset{o:+d}", df, moves, {"logp": lp}, wdl=wdl, part=part,
                       meta={"model": "maia-synth", "variant": f"offset{o:+d}"})
    for nodes, noise in ((10, 0.05), (100, 0.0)):
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
    logit.main(["--logp", f"m0={M[1]}", "--logp", f"mbest={M[2]}", "--engine", f"sf10={feats / 'sf/nodes10'}",
                "--engine", f"sf100={feats / 'sf/nodes100'}", "--set", "A=mbest", "--set", "B=m0,sf10,sf100",
                "--set", "C=m0,mbest,sf10,sf100", "--full", "C", "--boot-fit", "5"] + C)
    lg = json.loads((R / "logit.json").read_text())
    # the logit on m0 alone would recover the temperature 1.25; here A is mbest, which is exact: beta ~ 1
    assert abs(lg["sets"]["A"]["beta"][0] - 1.0) < 0.15
    breakdown.main(["--model", M[2]] + C)
    value.main(["--model", M[1], str(feats / "sf/nodes100")] + C)
    out = tmp_path / "step1.md"
    report.main(["--results", str(R), "--out", str(out), "--note", "synthetic"])
    text = out.read_text()
    for h in ("## The Elo sweep", "## Is classical play a rating-shifted blitz?", "## The conditional logit",
              "## The value head", "## Breakdowns"):
        assert h in text
    assert (R / "sweep.png").exists() and (R / "mistakes.png").exists()
