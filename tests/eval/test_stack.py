"""The stack finds an interaction between a context feature and a per-move feature that the linear
logit cannot represent; the group softmax is exact."""

from types import SimpleNamespace

import numpy as np
import torch

from tnh.eval.logit import Data, fit, scores
from tnh.eval.stack import Batches, effective_weight, group_logp, predict, train


def synth(n_rows, rng):
    lens = rng.integers(5, 25, n_rows)
    offsets = np.r_[0, np.cumsum(lens)]
    z = rng.normal(size=n_rows)                    # context, constant within a position
    x = rng.normal(size=offsets[-1])               # per-move feature
    slope = 1.0 + 1.5 * z                          # its true weight depends on the context
    s = np.repeat(slope, lens) * x
    y = np.empty(n_rows, dtype=np.int64)
    for i in range(n_rows):
        a, b = offsets[i], offsets[i + 1]
        p = np.exp(s[a:b] - s[a:b].max())
        y[i] = a + rng.choice(b - a, p=p / p.sum())
    return x[:, None], z[:, None], offsets, y, slope


def test_group_softmax_is_exact():
    s = torch.tensor([1.0, 2.0, 3.0, -1.0, 0.5])
    row = torch.tensor([0, 0, 0, 1, 1])
    lp = group_logp(s, row, 2)
    assert torch.allclose(lp[:3], torch.log_softmax(s[:3], 0)) and torch.allclose(lp[3:], torch.log_softmax(s[3:], 0))
    b = Batches(np.array([0, 3, 5, 9]))
    idx, local, lens = b.moves_of(np.array([2, 0]))
    assert idx.tolist() == [5, 6, 7, 8, 0, 1, 2] and local.tolist() == [0, 0, 0, 0, 1, 1, 1]


def test_stack_finds_a_context_interaction_the_logit_misses():
    rng = np.random.default_rng(0)
    xm, xc, off, y, slope = synth(6000, rng)
    xt, zt, offt, yt, slope_t = synth(3000, rng)
    n = np.diff(off).astype(float)
    w = np.ones(len(y))
    clusters = np.arange(len(y)) % 30
    # the linear logit: x alone (a context feature has no effect inside a softmax)
    D = Data(["x"], xm.astype(float), off, y, n, w, clusters, {})
    Dt = Data(["x"], xt.astype(float), offt, yt, np.diff(offt).astype(float), np.ones(len(yt)), clusters[:len(yt)], {})
    b, _ = fit(D)
    lin = -scores(Dt, b)[0].mean()
    a = SimpleNamespace(seed=0, hidden=16, lr=5e-3, weight_decay=0.0, batch=256, max_epochs=30, patience=3)
    rows = np.arange(len(y))
    model, hist = train(xm.astype(np.float32), xc.astype(np.float32), off, y, w, rows[:5000], rows[5000:], a, "cpu",
                        log=lambda *_: None)
    lp = predict(model, xt.astype(np.float32), zt.astype(np.float32), offt, "cpu")[yt]
    stack = -lp.mean()
    true = -np.mean([np.log(np.exp(slope_t[i] * xt[offt[i]:offt[i + 1], 0])[yt[i] - offt[i]]
                            / np.exp(slope_t[i] * xt[offt[i]:offt[i + 1], 0]).sum()) for i in range(len(yt))])
    assert stack < lin - 0.05, (stack, lin)
    assert stack < true + 0.03, (stack, true)
    # the effective weight on x follows the planted 1 + 1.5 z
    ew = effective_weight(model, xt.astype(np.float32), zt.astype(np.float32), offt, 0, 1.0, "cpu")
    assert np.corrcoef(ew, slope_t)[0, 1] > 0.9
