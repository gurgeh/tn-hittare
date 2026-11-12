"""How the database's move counts and a move model are mixed into one probability per move.

    p = w * share + (1 - w) * p_model,   share = count / N,   N = the position's games

The interface (`tnh.web`) shows it as Prob and the repertoire builder uses the same mix for the
opponent's moves; this module depends on nothing in either, so both can import it. The rule
in use is in `movemix.json` beside this file, written by `python -m tnh.web.probfit fit
--write-rule` with its provenance. Without the file the rule is `N / (N + 100)`.

Forms (`w` in [0, 1]):

| form | w |
|---|---|
| model | 0 |
| database | 1 |
| nk | N / (N + kappa) |
| tbl_nk | one weight per bin of N up to 20 games (`BINS`, rising), then N / (N + kappa) |
| tbl_hill | the table, then N^a / (N^a + k) |
| move | sigmoid(b0 + b1 ln(1 + N) + b2 ln(1 + count)) per move, the mix renormalised |

A position with no games (N = 0) gets the model alone in every form.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BINS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20)]
TABLE_MAX = 20
FORMS = ("model", "database", "nk", "tbl_nk", "tbl_hill", "move")
RULE_FILE = Path(__file__).with_name("movemix.json")


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def table_weights(t) -> np.ndarray:
    """Six rising weights from parameters: a logit for N = 1, then non-negative steps."""
    t = np.asarray(t, float)
    return _sigmoid(np.cumsum(np.concatenate([t[:1], np.exp(t[1:6])])))


def _bin(N: np.ndarray) -> np.ndarray:
    idx = np.zeros(N.shape, dtype=np.int64)
    for i, (lo, _hi) in enumerate(BINS):
        idx[N >= lo] = i
    return idx


def weight(form: str, params, N, count=None) -> np.ndarray:
    """w per row; `N` the position's games, `count` the move's (only the `move` form uses it)."""
    N = np.asarray(N, float)
    p = np.asarray(params, float)
    if form == "model":
        w = np.zeros_like(N)
    elif form == "database":
        w = np.ones_like(N)
    elif form == "nk":
        w = N / (N + math.exp(p[0]))
    elif form in ("tbl_nk", "tbl_hill"):
        small = table_weights(p[:6])[_bin(N)]
        if form == "tbl_nk":
            tail = N / (N + math.exp(p[6]))
        else:
            a, k = math.exp(p[6]), math.exp(p[7])
            na = np.maximum(N, 1e-9) ** a
            tail = na / (na + k)
        w = np.where(N <= TABLE_MAX, small, tail)
    elif form == "move":
        c = np.zeros_like(N) if count is None else np.asarray(count, float)
        w = _sigmoid(p[0] + p[1] * np.log1p(N) + p[2] * np.log1p(c))
    else:
        raise ValueError(f"unknown form {form!r}")
    return np.where(N > 0, w, 0.0)


@dataclass
class Rule:
    form: str = "nk"
    params: list[float] = field(default_factory=lambda: [math.log(100.0)])
    provenance: dict = field(default_factory=dict)

    def mix(self, counts, p_model) -> np.ndarray:
        """The mixed probabilities over one position's legal moves (arrays in the same order)."""
        c = np.asarray(counts, float)
        q = np.asarray(p_model, float)
        N = float(c.sum())
        share = c / N if N > 0 else np.zeros_like(c)
        w = weight(self.form, self.params, np.full_like(c, N), c)
        p = w * share + (1 - w) * q
        s = p.sum()
        return p / s if s > 0 else q

    def describe(self) -> str:
        p = self.params
        if self.form == "model":
            return "the model alone"
        if self.form == "database":
            return "the database alone"
        if self.form == "nk":
            return f"w = N/(N+{math.exp(p[0]):.3g})"
        if self.form == "move":
            return f"w = logistic({p[0]:.2f} {p[1]:+.2f} ln(1+N) {p[2]:+.2f} ln(1+games of the move))"
        w = table_weights(p[:6])
        t = ", ".join(f"{lo if lo == hi else f'{lo}-{hi}'}: {x:.2f}" for (lo, hi), x in zip(BINS, w))
        tail = (f"N/(N+{math.exp(p[6]):.3g})" if self.form == "tbl_nk"
                else f"N^{math.exp(p[6]):.2f}/(N^{math.exp(p[6]):.2f}+{math.exp(p[7]):.3g})")
        return f"w by the position's games: {t}; above {TABLE_MAX}: {tail}"

    @property
    def kappa(self) -> float | None:
        """The equivalent kappa where the rule is N / (N + kappa) (above the table for tbl_nk)."""
        return math.exp(self.params[-1]) if self.form in ("nk", "tbl_nk") else None


def load(path: Path = RULE_FILE) -> Rule:
    if not path.exists():
        return Rule(provenance={"fallback": True})
    d = json.loads(path.read_text())
    return Rule(d["form"], d["params"], d.get("provenance", {}))
