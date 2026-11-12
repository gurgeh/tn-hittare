"""Shrinkage rules for the move table: `prediction = prior + w * (move's mean - prior)`.

One definition for the fitting tool (`tnh.web.shrinkfit`) and for the server. A rule is a
form name and a parameter vector; `predict` evaluates it over arrays of features:

- `x`: the move's own mean (for a delta, the move's delta against the position);
- `base`: the prior before any adjustment: the position's mean for Score and Draw (model 2
  once it is available), 0 for the rating deltas;
- `n`: the move's count (games, or games with both ratings for the rating quantities);
- `pos_n`: the position's count; `share = n / pos_n` is not a separate feature;
- `spread`: the count-weighted standard deviation of the quantity among the position's moves.

Forms (all `w` in [0, 1]):

| form | prior | w |
|---|---|---|
| none | base | 1 |
| prior | base | 0 |
| nk | base | n / (n + k) |
| bins | base | one weight per bin of n (`BINS`) |
| hill | base | n^a / (n^a + k) |
| logistic | base | sigmoid(b0 + b1 log n + b2 log pos_n + b3 log(1 + spread / s)) |
| share_hill | base + c0 + c1 log share + c2 log pos_n | n^a / (n^a + k) |
| share_logistic | base + c0 + c1 log share + c2 log pos_n | the logistic above |

`s` (the spread scale) is stored in the rule beside the parameters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BINS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 10**12)]
FORMS = ("none", "prior", "nk", "bins", "hill", "logistic", "share_hill", "share_logistic")
N_PARAMS = {"none": 0, "prior": 0, "nk": 1, "bins": len(BINS), "hill": 2, "logistic": 4,
            "share_hill": 5, "share_logistic": 7}
RULES_FILE = Path(__file__).parent / "shrink_rules.json"


def bin_index(n: np.ndarray) -> np.ndarray:
    n = np.asarray(n)
    idx = np.zeros(n.shape, dtype=np.int64)
    for i, (lo, _hi) in enumerate(BINS):
        idx[n >= lo] = i
    return idx


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def _logs(n, pos_n, spread, s):
    n = np.maximum(np.asarray(n, float), 1e-9)
    pos_n = np.maximum(np.asarray(pos_n, float), 1.0)
    return np.log(n), np.log(pos_n), np.log1p(np.asarray(spread, float) / s)


def weight(form: str, p: np.ndarray, n, pos_n, spread, s: float = 1.0) -> np.ndarray:
    n = np.asarray(n, float)
    if form == "none":
        return np.ones_like(n)
    if form == "prior":
        return np.zeros_like(n)
    if form == "nk":
        return n / (n + np.exp(p[0]))
    if form == "bins":
        return np.clip(np.asarray(p)[bin_index(n)], 0, 1)
    if form in ("hill", "share_hill"):
        a, k = np.exp(p[-2]), np.exp(p[-1])
        na = np.maximum(n, 1e-9) ** a
        return na / (na + k)
    if form in ("logistic", "share_logistic"):
        b = p[-4:]
        ln, lp, ls = _logs(n, pos_n, spread, s)
        return _sigmoid(b[0] + b[1] * ln + b[2] * lp + b[3] * ls)
    raise ValueError(f"unknown form {form!r}")


def prior(form: str, p: np.ndarray, base, n, pos_n) -> np.ndarray:
    base = np.asarray(base, float)
    if form.startswith("share_"):
        n = np.maximum(np.asarray(n, float), 1e-9)
        pos_n = np.maximum(np.asarray(pos_n, float), 1.0)
        return base + p[0] + p[1] * np.log(n / pos_n) + p[2] * np.log(pos_n)
    return base


def predict(form: str, p, x, base, n, pos_n, spread, s: float = 1.0) -> np.ndarray:
    pr = prior(form, p, base, n, pos_n)
    return pr + weight(form, p, n, pos_n, spread, s) * (np.asarray(x, float) - pr)


@dataclass
class Rule:
    """A fitted rule for one quantity, as stored in `shrink_rules.json`."""

    form: str
    params: list[float]
    spread_scale: float = 1.0
    provenance: dict = field(default_factory=dict)

    def predict(self, x, base, n, pos_n, spread):
        return predict(self.form, np.asarray(self.params, float), x, base, n, pos_n, spread, self.spread_scale)

    def weight(self, n, pos_n, spread):
        return weight(self.form, np.asarray(self.params, float), n, pos_n, spread, self.spread_scale)

    def prior(self, base, n, pos_n):
        return prior(self.form, np.asarray(self.params, float), base, n, pos_n)

    def describe(self) -> str:
        """One line for a tooltip."""
        p = self.params
        if self.form == "none":
            return "no shrinkage"
        if self.form == "prior":
            return "the prior alone"
        if self.form == "nk":
            return f"w = n/(n+{np.exp(p[0]):.3g})"
        if self.form == "bins":
            return "w by games: " + ", ".join(f"{lo if lo == hi else f'{lo}-{hi}' if hi < 10**12 else f'{lo}+'}: {w:.2f}"
                                               for (lo, hi), w in zip(BINS, p))
        w = (f"w = n^{np.exp(p[-2]):.2f}/(n^{np.exp(p[-2]):.2f}+{np.exp(p[-1]):.3g})" if self.form.endswith("hill")
             else "w = logistic({:.2f} {:+.2f} ln n {:+.2f} ln N {:+.2f} ln(1+spread/{:.3g}))".format(*p[-4:], self.spread_scale))
        if self.form.startswith("share_"):
            return f"prior = base {p[0]:+.3g} {p[1]:+.3g} ln share {p[2]:+.3g} ln N; {w}"
        return w


def load_rules(path: Path = RULES_FILE) -> dict[str, Rule]:
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    return {q: Rule(r["form"], r["params"], r.get("spread_scale", 1.0), r.get("provenance", {}))
            for q, r in d["rules"].items()}
