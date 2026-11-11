"""Shared pieces of the step-1 evaluation: feature loading and joining, the uniform
mixture, weighted means, and the bootstrap over tournaments.

Conventions used everywhere in ``tnh.eval``:

- Log losses are in nats, means are weighted by the sample's ``weight`` column
  (population rows per sampled row in the row's stratum), so a pooled number
  estimates the mean over the population the sample was drawn from.
- Intervals are percentile intervals of a bootstrap that resamples tournaments
  (``tournament_id``) with replacement, never rows.
- A feature file that does not cover the sample, a played move that is not in a
  feature row's ``moves``, or two feature files whose ``moves`` differ raises
  ``FeatureMismatch``. Nothing is dropped silently.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from tnh.paths import REPORTS

BUCKET = 100
RESULTS = REPORTS / "step1"
KEY = ["game_id", "ply"]


class FeatureMismatch(ValueError):
    """Features and positions do not line up."""


# --- samples -----------------------------------------------------------------------

SAMPLE_EXTRA = ("tournament_id", "tempo_source", "in_window", "out_of_book", "part", "bucket", "weight")


def read_sample(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    missing = [c for c in ("game_id", "ply", "played_uci", "mover_elo", "opp_elo", "tournament_id", "weight")
               if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is not a tnh.eval sample: no columns {missing}")
    if df.select(pl.struct(KEY).is_duplicated().any()).item():
        raise FeatureMismatch(f"{path}: duplicate (game_id, ply)")
    return df.with_columns(pl.col("ply").cast(pl.Int64))


def bucket_of(col: str = "mover_elo", lo: int | None = None, hi: int | None = None) -> pl.Expr:
    """100-point bucket, the lower edge. With ``hi`` the top value folds into the bucket below it
    (a band given as 1800-2500 inclusive has seven buckets, 2500 in the 2400 one)."""
    b = (pl.col(col) // BUCKET) * BUCKET
    if hi is not None and hi % BUCKET == 0:
        b = pl.when(pl.col(col) >= hi).then(hi - BUCKET).otherwise(b)
    if lo is not None:
        b = pl.max_horizontal(b, pl.lit(lo // BUCKET * BUCKET))
    return b.cast(pl.Int64)


# --- feature files -----------------------------------------------------------------

def read_meta(d: Path) -> dict:
    p = Path(d) / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_features(d: Path, columns: list[str] | None = None) -> pl.DataFrame:
    parts = sorted(Path(d).glob("part-*.parquet"))
    if not parts:
        raise FeatureMismatch(f"no part-*.parquet in {d}")
    df = pl.concat([pl.read_parquet(p, columns=columns) for p in parts], how="vertical_relaxed")
    df = df.with_columns(pl.col("ply").cast(pl.Int64))
    n_dup = df.height - df.select(pl.struct(KEY).n_unique()).item()
    if n_dup:
        raise FeatureMismatch(f"{d}: {n_dup} duplicate (game_id, ply) rows")
    return df


def attach(sample: pl.DataFrame, d: Path, columns: list[str]) -> pl.DataFrame:
    """The feature rows of ``d`` for every sample row, in the sample's order, with ``_idx``,
    the index of the played move in ``moves``. Raises if any sample row is not covered or
    the played move is not among the moves."""
    feats = load_features(d, KEY + ["moves"] + columns)
    j = sample.select(KEY + ["played_uci", "n_legal"]).join(feats, on=KEY, how="left", maintain_order="left")
    n_missing = j.get_column("moves").null_count()
    if n_missing:
        raise FeatureMismatch(f"{d}: {n_missing} of {sample.height} sample rows have no features "
                              f"(run the feature runner on this sample first)")
    n_moves = j.get_column("moves").list.len()
    bad = (n_moves != j.get_column("n_legal")).sum()
    if bad:
        raise FeatureMismatch(f"{d}: {bad} rows where len(moves) != n_legal")
    for c in columns:
        if c != "wdl" and j.schema[c] == pl.List:
            k = (j.get_column(c).list.len() != n_moves).sum()
            if k:
                raise FeatureMismatch(f"{d}: {k} rows where len({c}) != len(moves)")
    idx = played_index(j)
    return j.with_columns(pl.Series("_idx", idx))


def played_index(df: pl.DataFrame) -> np.ndarray:
    """Position of ``played_uci`` in ``moves`` per row; raises unless it is there exactly once."""
    ex = (df.select(pl.int_range(pl.len()).alias("_r"), "moves", "played_uci").explode("moves")
          .with_columns(pl.int_range(pl.len()).over("_r").alias("_j"))
          .filter(pl.col("moves") == pl.col("played_uci"))
          .group_by("_r").agg(pl.col("_j").first(), pl.len().alias("_n")))
    if ex.height != df.height or (ex.get_column("_n") != 1).any():
        bad = df.height - ex.filter(pl.col("_n") == 1).height
        raise FeatureMismatch(f"played move not found exactly once in moves in {bad} rows")
    return ex.sort("_r").get_column("_j").to_numpy().astype(np.int64)


def same_moves(a: pl.DataFrame, b: pl.DataFrame, what: str = "") -> None:
    """Two attached frames (same sample, same order) must list the same moves in the same order."""
    if a.height != b.height or not (a.get_column("game_id") == b.get_column("game_id")).all() \
            or not (a.get_column("ply") == b.get_column("ply")).all():
        raise FeatureMismatch(f"{what}: rows not aligned")
    diff = (a.get_column("moves") != b.get_column("moves")).sum()
    if diff:
        raise FeatureMismatch(f"{what}: move lists differ in {diff} of {a.height} rows")


@dataclass
class Ragged:
    """Per-move values of many positions, flat, with ``offsets`` (length rows + 1)."""
    values: np.ndarray
    offsets: np.ndarray

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    @property
    def row_of(self) -> np.ndarray:
        return np.repeat(np.arange(len(self.offsets) - 1), self.lengths)


def ragged(df: pl.DataFrame, col: str, dtype=np.float64) -> Ragged:
    lens = df.get_column(col).list.len().to_numpy()
    offsets = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    vals = df.get_column(col).explode().to_numpy()
    if vals.dtype == object:
        vals = np.array([np.nan if v is None else v for v in vals])
    vals = vals.astype(dtype)
    if len(vals) != offsets[-1]:
        raise FeatureMismatch(f"{col}: flattened length {len(vals)} != {offsets[-1]}")
    return Ragged(vals, offsets)


def segment_max(r: Ragged, values: np.ndarray | None = None) -> np.ndarray:
    v = r.values if values is None else values
    return np.maximum.reduceat(v, r.offsets[:-1])


def segment_sum(r: Ragged, values: np.ndarray) -> np.ndarray:
    return np.add.reduceat(values, r.offsets[:-1], axis=0)


def segment_logsumexp(r: Ragged, values: np.ndarray) -> np.ndarray:
    m = np.maximum.reduceat(values, r.offsets[:-1])
    return m + np.log(np.add.reduceat(np.exp(values - np.repeat(m, r.lengths)), r.offsets[:-1]))


@dataclass
class MoveScores:
    """What a move model says about the played move, per sample row."""
    lp: np.ndarray        # log probability of the played move
    n: np.ndarray         # legal moves
    top1: np.ndarray      # 1 where the played move has the highest probability (1/k when k moves tie)
    entropy: np.ndarray   # of the model's distribution, nats
    wdl: np.ndarray | None  # rows x 3, or None


def move_scores(sample: pl.DataFrame, d: Path) -> tuple[MoveScores, pl.DataFrame]:
    j = attach(sample, d, ["logp", "wdl"] if "wdl" in _columns(d) else ["logp"])
    r = ragged(j, "logp")
    idx = j.get_column("_idx").to_numpy()
    lp = r.values[r.offsets[:-1] + idx]
    mx = segment_max(r)
    p = np.exp(r.values)
    ent = -segment_sum(r, np.where(p > 0, p * r.values, 0.0))
    wdl = None
    if "wdl" in j.columns and j.get_column("wdl").null_count() == 0:
        wdl = np.stack(j.get_column("wdl").to_numpy()).astype(np.float64)
    # top-1 with ties shared: a played move tied with k-1 others for the highest probability scores 1/k
    n_max = segment_sum(r, (r.values == np.repeat(mx, r.lengths)).astype(np.float64))
    top1 = np.where(lp >= mx, 1.0 / n_max, 0.0)
    return MoveScores(lp, j.get_column("n_legal").to_numpy().astype(np.float64), top1, ent, wdl), j


def _columns(d: Path) -> list[str]:
    import pyarrow.parquet as pq
    first = sorted(Path(d).glob("part-*.parquet"))
    if not first:
        raise FeatureMismatch(f"no part-*.parquet in {d}")
    return pq.read_schema(first[0]).names


# --- the uniform mixture --------------------------------------------------------------

def mix_logp(lp: np.ndarray, n: np.ndarray, eps: float) -> np.ndarray:
    """log((1 - eps) p + eps / n), computed in log space."""
    if eps <= 0:
        return lp
    return np.logaddexp(np.log1p(-eps) + lp, np.log(eps) - np.log(n))


def fit_eps(lp: np.ndarray, n: np.ndarray, w: np.ndarray | None = None, tol: float = 1e-10) -> float:
    """Maximum-likelihood ``eps`` in [0, 1). The loss is convex in eps; bisection on its derivative."""
    w = np.ones_like(lp) if w is None else w
    p = np.exp(lp)
    u = 1.0 / n

    def deriv(e: float) -> float:
        with np.errstate(divide="ignore", invalid="ignore"):   # p = 0 at e = 0: the slope is -inf, eps > 0
            return -float(np.sum(w * (u - p) / ((1 - e) * p + e * u)) / np.sum(w))

    if deriv(0.0) >= 0:
        return 0.0
    lo, hi = 0.0, 0.999
    if deriv(hi) <= 0:
        return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if deriv(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --- weighted means and the bootstrap over tournaments ---------------------------

def wmean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * x) / np.sum(w))


def cluster_codes(sample: pl.DataFrame, col: str = "tournament_id") -> np.ndarray:
    """Dense codes 0..K-1 of the cluster column."""
    return (sample.get_column(col).rank("dense") - 1).to_numpy().astype(np.int64)


def boot_counts(n_clusters: int, B: int, seed: int) -> np.ndarray:
    """B x K: how many times each cluster is drawn in each replicate."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=B).astype(np.float64)


def boot_means(x: np.ndarray, w: np.ndarray, clusters: np.ndarray, counts: np.ndarray,
               groups: np.ndarray | None = None, n_groups: int = 1) -> np.ndarray:
    """Weighted means under the cluster bootstrap.

    ``x`` is rows or rows x k. Returns B x k (no groups) or B x n_groups x k.
    Groups (e.g. rating buckets) are given as codes 0..n_groups-1.
    """
    x2 = x.reshape(len(x), -1)
    K = counts.shape[1]
    g = np.zeros(len(x), dtype=np.int64) if groups is None else groups
    cell = clusters * n_groups + g
    k = x2.shape[1]
    sums = np.zeros((K * n_groups, k))
    for j in range(k):
        sums[:, j] = np.bincount(cell, weights=w * x2[:, j], minlength=K * n_groups)
    wsum = np.bincount(cell, weights=w, minlength=K * n_groups)
    sums = sums.reshape(K, n_groups, k)
    wsum = wsum.reshape(K, n_groups)
    num = np.einsum("bk,kgj->bgj", counts, sums)
    den = counts @ wsum
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den[:, :, None]
    return out[:, 0, :] if groups is None else out


def interval(boot: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    a = (1 - level) / 2
    return np.nanquantile(boot, a, axis=0), np.nanquantile(boot, 1 - a, axis=0)


# --- variants ---------------------------------------------------------------------

VARIANT_RE = re.compile(r"^(?:offset(?P<m>[+-]\d+(?:\.\d+)?)(?:-opp(?P<o>[+-]\d+(?:\.\d+)?))?"
                        r"|fixed(?P<s>\d+(?:\.\d+)?)-(?P<x>\d+(?:\.\d+)?))")


def parse_variant(name: str) -> dict:
    """``offset+300`` -> mover and opponent offset 300; ``offset+300-opp+100``; ``fixed2200-2200``.
    Anything after the Elo part (a suffix) is kept in ``suffix``."""
    m = VARIANT_RE.match(name)
    if not m:
        raise ValueError(f"not an Elo variant name: {name!r}")
    suffix = name[m.end():]
    if m["m"] is not None:
        mo = float(m["m"])
        return {"kind": "offset", "mover": mo, "opp": mo if m["o"] is None else float(m["o"]), "suffix": suffix}
    return {"kind": "fixed", "self": float(m["s"]), "oppo": float(m["x"]), "suffix": suffix}


def variant_name(d: Path) -> str:
    meta = read_meta(d)
    return meta.get("variant", Path(d).name)


def model_name(d: Path) -> str:
    meta = read_meta(d)
    return meta.get("model", Path(d).parent.name)


def label(d: Path) -> str:
    return f"{model_name(d)}/{variant_name(d)}"


# --- results -----------------------------------------------------------------------

def write_result(results: Path, name: str, obj: dict) -> Path:
    results.mkdir(parents=True, exist_ok=True)
    p = results / f"{name}.json"
    p.write_text(json.dumps(obj, indent=1, default=_json_default) + "\n")
    print(f"wrote {p}")
    return p


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def provenance(sample_paths: dict[str, Path], feature_dirs: list[Path]) -> dict:
    """What a result was computed from: the samples (with their own json) and the feature dirs' meta."""
    out = {"samples": {}, "features": {}}
    for k, p in sample_paths.items():
        meta_p = Path(p).with_suffix(".json")
        out["samples"][k] = {"path": str(p), "meta": json.loads(meta_p.read_text()) if meta_p.exists() else None}
    for d in feature_dirs:
        out["features"][label(d)] = {"path": str(d), "meta": read_meta(d)}
    return out


# --- command-line pieces shared by the analysis commands ----------------------------

def add_common(p, *, fit: bool = True) -> None:
    from tnh.paths import CORPUS
    p.add_argument("--test", type=Path, default=CORPUS / "samples" / "test.parquet", help="test sample")
    if fit:
        p.add_argument("--fit", type=Path, default=CORPUS / "samples" / "fit.parquet", help="fitting sample (train)")
    p.add_argument("--results", type=Path, default=RESULTS, help="where the result json goes")
    p.add_argument("--boot", type=int, default=1000, help="bootstrap replicates")
    p.add_argument("--seed", type=int, default=7)


def check_splits(test: pl.DataFrame, fit: pl.DataFrame | None) -> None:
    """Leakage guard: the test sample is all `test`, the fitting sample all `train`, no shared tournament."""
    if "split" in test.columns and set(test.get_column("split").unique()) != {"test"}:
        raise SystemExit(f"test sample has splits {set(test.get_column('split').unique())}")
    if fit is None:
        return
    if "split" in fit.columns and set(fit.get_column("split").unique()) != {"train"}:
        raise SystemExit(f"fitting sample has splits {set(fit.get_column('split').unique())}")
    shared = set(test.get_column("tournament_id").unique()) & set(fit.get_column("tournament_id").unique())
    if shared:
        raise SystemExit(f"{len(shared)} tournaments are in both samples")


def window_rows(s: pl.DataFrame) -> pl.DataFrame:
    return s.filter(pl.col("in_window")) if "in_window" in s.columns else s


def summarize(boot: np.ndarray, point: np.ndarray) -> list[dict]:
    lo, hi = interval(boot)
    return [{"est": float(p), "lo": float(a), "hi": float(b)} for p, a, b in zip(np.atleast_1d(point), lo, hi)]


def est(point: float, boot: np.ndarray) -> dict:
    lo, hi = interval(boot)
    return {"est": float(point), "lo": float(lo), "hi": float(hi)}
