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


def test_tree_objective_is_the_gradient_of_the_group_log_loss():
    from tnh.eval.trees import group_loss, objective
    rng = np.random.default_rng(3)
    off = np.array([0, 3, 7, 9])
    y = np.array([1, 5, 7])
    w = np.array([1.0, 2.0, 0.5])
    s = rng.normal(size=9)
    g, h = objective(off, y, w)(s, None)
    eps = 1e-6
    num = np.array([(group_loss(s + eps * np.eye(9)[j], off, y, w) - group_loss(s - eps * np.eye(9)[j], off, y, w))
                    / (2 * eps) for j in range(9)])
    # the objective's gradient is the summed loss with weights normalised to mean 1
    assert np.allclose(g / len(y), num, atol=1e-6)
    assert (h > 0).all()


def test_trees_recover_a_context_dependent_coefficient():
    from tnh.eval import trees as T
    rng = np.random.default_rng(1)
    xm, xc, off, y, slope = synth(8000, rng)
    xt, zt, offt, yt, slope_t = synth(3000, rng)
    w = np.ones(len(y))
    D = Data(["x"], xm.astype(float), off, y, np.diff(off).astype(float), w, np.arange(len(y)) % 30, {})
    b, _ = fit(D)
    Dt = Data(["x"], xt.astype(float), offt, yt, np.diff(offt).astype(float), np.ones(len(yt)), np.arange(len(yt)) % 30, {})
    lin = -scores(Dt, b)[0].mean()
    names, ctx = ["x"], ["ctx_mover_elo"]
    X, _ = T.design(xm, names, xc, ctx, off, ["x"], True)
    Xt, _ = T.design(xt, names, zt, ctx, offt, ["x"], True)
    rows = np.arange(len(y))
    from tnh.eval.stack import Batches
    B = Batches(off)

    def sub(r):
        idx, _, lens = B.moves_of(r)
        o = np.r_[0, np.cumsum(lens)]
        return idx, o, o[:-1] + (y - off[:-1])[r]

    tr, va = sub(rows[:6500]), sub(rows[6500:])
    init = xm[:, 0] * b[0]
    prm = T.grid([15], [0.1], [50])[0]
    bst, v = T.train(X[tr[0]], tr[1], tr[2], w[:6500], init[tr[0]], X[va[0]], va[1], va[2], w[6500:], init[va[0]],
                     prm, 500, 30, 0, threads=2)
    st = xt[:, 0] * b[0] + T.raw(bst, Xt)
    trees = T.group_loss(st, offt, yt, np.ones(len(yt)))
    assert trees < lin - 0.05, (trees, lin)
    # effective weight by central differences follows the planted 1 + 1.5 z
    h = 0.1
    sc = []
    for sgn in (1, -1):
        xp = xt + sgn * h
        Xp, _ = T.design(xp, names, zt, ctx, offt, ["x"], True)
        sc.append(xp[:, 0] * b[0] + T.raw(bst, Xp))
    d = (sc[0] - sc[1]) / (2 * h)
    ew = np.add.reduceat(d, offt[:-1]) / np.diff(offt)
    assert np.corrcoef(ew, slope_t)[0, 1] > 0.8
