"""Maia-3 wrapper against the reference implementation. Needs the GPU and the weights.

Two references:
- in process: `maia3.uci.Maia3UCIEngine.score_moves`, the code behind `maia3-uci`, asked for
  every legal move so that the full distribution and the per-move WDL are compared;
- out of process: the `maia3-uci` engine itself over stdin/stdout, which prints the top 20
  moves in order with a WDL per move in permille, but no probabilities.

`maia3-uci` never prints the WDL of the position itself. Its WDL for a candidate move is
the value head on the position after the move, from the other side's view and with the
two ratings swapped, inverted back. The wrapper's own value head is compared on exactly
those child positions.
"""

import subprocess
import sys

import chess
import chess.pgn
import numpy as np
import pytest

from tnh.models.base import Query

pytestmark = pytest.mark.slow

FIXTURE = __import__("pathlib").Path(__file__).parent / "fixtures" / "games.pgn"
ELO_PAIRS = [(1100, 1500), (1900, 1900), (2500, 2200), (2900, 2600), (600, 2600), (1850, 2450)]


def fixture_positions():
    """Every position before a move in the fixture games, with history and cycling ratings."""
    out = []
    with open(FIXTURE) as f:
        while (g := chess.pgn.read_game(f)) is not None:
            b, moves = chess.Board(), []
            for m in g.mainline_moves():
                me, opp = ELO_PAIRS[len(out) % len(ELO_PAIRS)]
                out.append(Query(b.fen(), list(moves), me, opp))
                b.push(m)
                moves.append(m.uci())
    return out


def child(q: Query, uci: str) -> Query:
    b = chess.Board(q.fen)
    b.push_uci(uci)
    return Query(b.fen(), q.moves_uci + [uci], q.opp_elo, q.mover_elo)


def permille(wdl):
    from maia3.uci import _probabilities_to_permille
    return _probabilities_to_permille(wdl)


# float32 must agree to float32 noise; that, and the forward being bitwise equal to the
# reference forward on the same batch (test below), is the proof that the wrapper is the
# reference. Under float16 autocast, the mode maia3-uci runs in, cuBLAS picks different
# kernels for a batch of 64 than for the reference's batch of 1, and a logit (magnitude
# 10-30, float16 spacing 1/64 or 1/32 there) can round a step or two the other way. The
# float16 bounds are set just above what that rounding produced on these positions
# (worst: 79M, |dlog p| 0.031, WDL 3 permille) and catch any real encoding error, which
# moves log probabilities by whole units.
TOL = {
    False: {"dp": 1e-5, "dlogp": 1e-4, "order": 1e-5, "wdl": 1},
    True: {"dp": 2e-2, "dlogp": 0.08, "order": 0.07, "wdl": 4},
}


@pytest.fixture(scope="module", params=["5m", "23m", "79m"])
def size(request):
    return request.param


@pytest.fixture(scope="module", params=[False, True], ids=["fp32", "fp16"])
def amp(request):
    return request.param


@pytest.fixture(scope="module")
def ours(size, amp):
    from tnh.models.maia3 import Elo, Maia3
    qs = fixture_positions()
    m = Maia3(size, Elo.offset(0.0), batch_size=64, amp=amp)   # offset 0: the network gets the query's own ratings
    preds = m.predict(qs)
    kids = [child(q, mv) for q, p in zip(qs, preds) for mv in p.moves]
    kid_preds = iter(m.predict(kids))
    kid_wdl = [{mv: next(kid_preds).wdl for mv in p.moves} for p in preds]
    return qs, preds, kid_wdl


def test_in_process_reference(size, amp, ours):
    from maia3 import uci as ref
    from tnh.models.maia3 import checkpoint_path

    qs, preds, kid_wdl = ours
    cfg = ref.parse_args(["--model", f"maia3-{size}", "--use-uci-history", "--checkpoint-path", checkpoint_path(size)]
                         + ([] if amp else ["--no-use-amp"]))
    eng = ref.Maia3UCIEngine(cfg)
    eng.ensure_model_loaded()
    eng.multipv = 1000   # every legal move
    max_dp, max_dlogp, max_dwdl, n_moves, sum_dlogp = 0.0, 0.0, 0, 0, 0.0
    for q, p, kw in zip(qs, preds, kid_wdl):
        eng.self_elo, eng.oppo_elo = q.mover_elo, q.opp_elo
        eng.cmd_position("position startpos moves " + " ".join(q.moves_uci) if q.moves_uci else "position startpos")
        _, top = eng.score_moves()
        ref_p = {t["move"].uci(): t["policy"] for t in top}
        assert set(ref_p) == set(p.moves)
        mine = dict(zip(p.moves, np.exp(p.logp.astype(np.float64))))
        for mv, rp in ref_p.items():
            max_dp = max(max_dp, abs(mine[mv] - rp))
            if rp > 1e-4:
                max_dlogp = max(max_dlogp, abs(np.log(mine[mv]) - np.log(rp)))
            sum_dlogp += abs(np.log(mine[mv]) - np.log(rp))
            w, d, l = kw[mv]
            mine_wdl = permille((w, d, l))[::-1]   # child's view, inverted as the reference does
            ref_wdl = next(t["wdl"] for t in top if t["move"].uci() == mv)
            max_dwdl = max(max_dwdl, max(abs(a - b) for a, b in zip(mine_wdl, ref_wdl)))
        n_moves += len(ref_p)
    tol = TOL[amp]
    print(f"\n{size} {'fp16' if amp else 'fp32'} in process: {len(qs)} positions, {n_moves} moves: "
          f"max |dp| {max_dp:.2e}, max |dlog p| (p>1e-4) {max_dlogp:.2e}, mean |dlog p| {sum_dlogp / n_moves:.2e}, "
          f"max WDL difference {max_dwdl} permille")
    assert max_dp < tol["dp"]
    assert max_dlogp < tol["dlogp"]
    assert max_dwdl <= tol["wdl"]


def test_uci_engine(size, amp, ours):
    from tnh.models.maia3 import checkpoint_path

    qs, preds, kid_wdl = ours
    cmd = [sys.executable, "-m", "maia3.uci", "--model", f"maia3-{size}", "--use-uci-history",
           "--checkpoint-path", checkpoint_path(size), "--multipv", "20"] + ([] if amp else ["--no-use-amp"])
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def send(line):
        proc.stdin.write(line + "\n")
        proc.stdin.flush()

    def until(prefix):
        lines = []
        for line in proc.stdout:
            lines.append(line.strip())
            if line.startswith(prefix):
                return lines
        raise RuntimeError("maia3-uci exited")

    send("uci"); until("uciok")
    send("isready"); until("readyok")
    order_violations, n_pairs, max_dwdl, top_set_diff = 0, 0, 0, 0
    worst_violation = 0.0
    for q, p, kw in zip(qs, preds, kid_wdl):
        send(f"setoption name SelfElo value {q.mover_elo}")
        send(f"setoption name OppoElo value {q.opp_elo}")
        send("position startpos moves " + " ".join(q.moves_uci) if q.moves_uci else "position startpos")
        send("go nodes 1")
        lines = [l for l in until("bestmove") if l.startswith("info") and " multipv " in l]
        ref_moves, ref_wdl = [], {}
        for l in lines:
            t = l.split()
            mv = t[t.index("pv") + 1]
            ref_moves.append(mv)
            i = t.index("wdl")
            ref_wdl[mv] = tuple(int(x) for x in t[i + 1 : i + 4])
        mine = dict(zip(p.moves, np.exp(p.logp.astype(np.float64))))
        # the reference order must be our order, up to ties
        for a, b in zip(ref_moves, ref_moves[1:]):
            n_pairs += 1
            if mine[a] < mine[b]:
                order_violations += 1
                worst_violation = max(worst_violation, (mine[b] - mine[a]) / mine[b])
        k = len(ref_moves)
        ours_top = sorted(p.moves, key=lambda mv: -mine[mv])[:k]
        kth = mine[ours_top[-1]]
        top_set_diff += sum(1 for mv in set(ref_moves) ^ set(ours_top) if abs(mine[mv] - kth) > TOL[amp]["order"] * kth)
        for mv in ref_moves:
            w, d, l = kw[mv]
            max_dwdl = max(max_dwdl, max(abs(a - b) for a, b in zip(permille((w, d, l))[::-1], ref_wdl[mv])))
    send("quit")
    proc.wait(timeout=30)
    print(f"\n{size} {'fp16' if amp else 'fp32'} maia3-uci: {len(qs)} positions, {n_pairs} adjacent pairs in its top-20, "
          f"{order_violations} out of order (largest relative prob gap {worst_violation:.1e}), "
          f"{top_set_diff} top-20 membership differences beyond tolerance, max WDL difference {max_dwdl} permille")
    assert worst_violation < TOL[amp]["order"]
    assert top_set_diff == 0
    assert max_dwdl <= TOL[amp]["wdl"]


def test_fast_forward_is_bitwise_reference(size, amp):
    import torch
    from tnh.models.maia3 import WEIGHTS, Elo, Maia3, _load

    m = Maia3(size, Elo.offset(0.0))
    reference = _load(size, "cuda", WEIGHTS, True, reference=True)
    qs = fixture_positions()[:300]
    tokens, elos, _, _ = m._encode(qs)
    t = torch.from_numpy(tokens).cuda().float()
    e = torch.from_numpy(elos).cuda()
    with torch.no_grad(), torch.autocast("cuda", enabled=amp):
        a, b = m.model(t, e[:, 0], e[:, 1]), reference(t, e[:, 0], e[:, 1])
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_elo_changes_predictions(size):
    from tnh.models.maia3 import Elo, Maia3

    qs = fixture_positions()[20:220]
    lo, hi = Maia3(size).predict_many(qs, [Elo.fixed(1200, 1200), Elo.fixed(2500, 2500)])
    tv = np.mean([0.5 * np.abs(np.exp(a.logp) - np.exp(b.logp)).sum() for a, b in zip(lo, hi)])
    print(f"\n{size}: mean total variation between Elo 1200 and 2500: {tv:.3f}")
    assert tv > 0.05


def test_predict_many_equals_one_setting_at_a_time(size):
    from tnh.models.maia3 import Elo, Maia3

    qs = fixture_positions()[:300]
    settings = [Elo.offset(0), Elo.offset(300, -100), Elo.fixed(2200, 2600)]
    m = Maia3(size, batch_size=64)
    many = m.predict_many(qs, settings)
    for e, preds in zip(settings, many):
        single = Maia3(size, e, batch_size=64).predict(qs)
        for a, b in zip(preds, single):
            assert a.moves == b.moves and a.wdl == b.wdl
            np.testing.assert_array_equal(a.logp, b.logp)
