"""Lc0 wrapper against the Lc0 engine. Needs the install from docs/lc0.md; the GPU tests are slow."""

import random

import chess
import chess.pgn
import numpy as np
import pytest

from tnh.models import lc0
from tnh.models.base import Query

installed = pytest.mark.skipif(not lc0.binary().exists(), reason=f"no Lc0 at {lc0.LC0_HOME} (docs/lc0.md)")
FIXTURE = __import__("pathlib").Path(__file__).parent / "fixtures" / "games.pgn"

def pawnless_positions(n_per_transform=3, seed=0) -> list[str]:
    """Legal pawnless positions, not game over, a few for each transform Lc0 would choose."""
    rng = random.Random(seed)
    by_t: dict[int, list[str]] = {}
    pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    while sum(min(len(v), n_per_transform) for v in by_t.values()) < 8 * n_per_transform:
        b = chess.Board(None)
        squares = rng.sample(range(64), 4)
        b.set_piece_at(squares[0], chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(squares[1], chess.Piece(chess.KING, chess.BLACK))
        b.set_piece_at(squares[2], chess.Piece(rng.choice(pieces), chess.WHITE))
        b.set_piece_at(squares[3], chess.Piece(rng.choice(pieces), rng.choice([chess.WHITE, chess.BLACK])))
        b.turn = rng.choice([chess.WHITE, chess.BLACK])
        if not b.is_valid() or b.is_game_over():
            continue
        t = lc0.choose_transform(b)
        if len(by_t.setdefault(t, [])) < n_per_transform:
            by_t[t].append(b.fen())
    return [f for v in by_t.values() for f in v]


PAWNLESS = pawnless_positions()


def random_positions(n, seed=0):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b, moves = chess.Board(), []
        for _ in range(rng.randint(0, 150)):
            legal = list(b.legal_moves)
            if not legal:
                break
            sp = [m for m in legal if m.promotion or b.is_castling(m) or b.is_en_passant(m) or b.is_capture(m)]
            m = rng.choice(sp) if sp and rng.random() < 0.4 else rng.choice(legal)
            b.push(m)
            moves.append(m.uci())
        if not b.is_game_over():
            out.append(Query(b.fen(), moves, 2000, 2000))
    return out


def fixture_positions(step=3):
    out = []
    with open(FIXTURE) as f:
        while (g := chess.pgn.read_game(f)) is not None:
            b, moves = chess.Board(), []
            for i, m in enumerate(g.mainline_moves()):
                if i % step == 0:
                    out.append(Query(b.fen(), list(moves), 2400, 2400))
                b.push(m)
                moves.append(m.uci())
    return out


def all_positions():
    return fixture_positions() + random_positions(150) + [Query(f, [], 2000, 2000) for f in PAWNLESS]


@installed
def test_policy_table_matches_the_bindings_without_transform():
    lcb = lc0.backends()
    n = 0
    for q in random_positions(200, seed=1):
        b = chess.Board(q.fen)
        gs = lcb.GameState(moves=q.moves_uci) if q.moves_uci else lcb.GameState(fen=q.fen)
        mine = {m.uci(): lc0.policy_index(b, m, 0) for m in b.legal_moves}
        assert sorted(gs.moves()) == sorted(mine)
        for mv, ix in zip(gs.moves(), gs.policy_indices()):
            assert mine[mv] == ix, (q.fen, mv)
            n += 1
    assert n > 3000


def test_transforms_cover_every_kind():
    seen = {lc0.choose_transform(chess.Board(f)) for f in PAWNLESS}
    assert seen == set(range(8))


def engine_policy(positions, nodes=1):
    """The engine's own P (percent, 2 decimals) per move, and its Q for the position (ScoreType Q:
    score = 10000 Q, exact to 1e-4) at one node, policy temperature 1."""
    e = lc0._Engine([str(lc0.binary())], {
        "WeightsFile": str(lc0.network_path()), "Backend": "cuda-auto", "Threads": "1",
        "BackendOptions": f"max_batch={lc0.MAX_BATCH}", "VerboseMoveStats": "true", "PolicyTemperature": "1.0",
        "SmartPruningFactor": "0", "ScoreType": "Q"})
    out = []
    try:
        for q in positions:
            board = chess.Board()
            for m in q.moves_uci:
                board.push_uci(m)
            if not q.moves_uci:
                board = chess.Board(q.fen)
            e.send("ucinewgame")
            e.send("position startpos moves " + " ".join(q.moves_uci) if q.moves_uci else f"position fen {q.fen}")
            e.send(f"go nodes {nodes}")
            lines = e.until("bestmove")
            edges, root = lc0.parse_verbose_stats(lines, board)
            score = [l.split() for l in lines if l.startswith("info") and " score cp " in l][-1]
            out.append(({s.move: s.p for s in edges}, int(score[score.index("cp") + 1]) / 10000, root))
    finally:
        e.close()
    return out


@installed
@pytest.mark.slow
@pytest.mark.parametrize("batch_size", [1, lc0.MAX_BATCH])
def test_policy_agrees_with_engine(batch_size):
    positions = all_positions()
    model = lc0.Lc0Policy(batch_size=batch_size)
    preds = model.predict(positions)
    del model
    ref = engine_policy(positions)
    dps, dqs, dds = [], [], []
    for q, p, (ep, eq, root) in zip(positions, preds, ref):
        assert set(ep) == set(p.moves)
        mine = dict(zip(p.moves, 100 * np.exp(p.logp.astype(np.float64))))
        dps += [abs(mine[mv] - pct) for mv, pct in ep.items()]
        w, d, l = p.wdl
        dqs.append(abs((w - l) - eq))
        dds.append(abs(d - root.d))
    print(f"\nbatch {batch_size}: {len(positions)} positions, {len(dps)} moves: |P difference| in percentage "
          f"points max {max(dps):.4f}, mean {np.mean(dps):.5f}; |Q difference| max {max(dqs):.5f}; "
          f"|D difference| max {max(dds):.4f}")
    # The engine's P comes from FastExp, an approximation, and its displayed D goes through
    # FastLog/FastLogistic (WDLRescale); its Q (ScoreType Q) does not.
    # batch 12 in float16: kernels differ from batch 1 by up to ~0.11 in a logit (float32 is
    # bitwise equal), measured 0.88 percentage points at most
    assert max(dps) < (0.1 if batch_size == 1 else 1.0)
    assert max(dqs) < (1e-4 + 1e-6 if batch_size == 1 else 0.01)
    assert max(dds) < (0.005 if batch_size == 1 else 0.01)


@installed
@pytest.mark.slow
def test_search_values_match_the_engines_own_output_and_repeat():
    """Values from Lc0Search against the verbose stats of the same search (whose Q is rounded
    through lc0's FastLog/FastLogistic) and against a second run."""
    positions = fixture_positions(step=9)[:40] + [Query(f, [], 2000, 2000) for f in PAWNLESS[:4]]
    with lc0.Lc0Search(200) as s:
        a = s.evaluate(positions)
        verbose = [s.search(q)[0] for q in positions]
        b = s.evaluate(positions)
    n_visited, dq = 0, []
    for x, edges in zip(a, verbose):
        by = {e.move: e for e in edges}
        for mv, v in zip(x.moves, x.value):
            e = by[mv]
            assert np.isnan(v) == (e.n == 0)
            if e.n > 0 and not e.terminal:
                n_visited += 1
                dq.append(abs((2 * v - 1) - e.q))
    print(f"\n{len(positions)} positions, {sum(len(x.moves) for x in a)} root moves, {n_visited} visited: "
          f"|Q - verbose Q| max {max(dq):.5f}, mean {np.mean(dq):.5f}")
    assert n_visited > 100 and max(dq) < 0.005
    same = all(np.array_equal(x.value, y.value, equal_nan=True) for x, y in zip(a, b))
    print(f"second run identical: {same}")
    assert same


@installed
@pytest.mark.slow
def test_calibrated_wdl_moves_with_rating():
    from tnh.models.maia3 import Elo

    moves = ["e2e4", "e7e5", "g1f3", "b8c6"]
    board = chess.Board()
    for m in moves:
        board.push_uci(m)
    q = Query(board.fen(), moves, 1800, 1800)
    wdl = {}
    for name in ("fixed1200-1200", "fixed2000-2000", "fixed2800-2800", "fixed2400-2000"):
        with lc0.Lc0Search(1, wdl_elo=Elo.parse(name)) as s:
            wdl[name] = s.evaluate([q])[0].wdl
    with lc0.Lc0Search(1) as s:
        raw = s.evaluate([q])[0].wdl
    print("\n" + "\n".join(f"{k}: W {w:.3f} D {d:.3f} L {l:.3f}" for k, (w, d, l) in {**wdl, "raw": raw}.items()))
    # weaker players draw less; a rating advantage raises the mover's expected score
    assert wdl["fixed1200-1200"][1] < wdl["fixed2000-2000"][1] < wdl["fixed2800-2800"][1]
    e = {k: w + d / 2 for k, (w, d, l) in wdl.items()}
    assert e["fixed2400-2000"] > e["fixed2000-2000"] + 0.05


@installed
@pytest.mark.slow
def test_search_values_do_not_depend_on_the_number_of_processes():
    positions = fixture_positions(step=11)[:24]
    with lc0.Lc0Search(100, processes=1) as s:
        a = s.evaluate(positions)
    with lc0.Lc0Search(100, processes=3) as s:
        b = s.evaluate(positions)
    worst = max(float(np.nanmax(np.abs(x.value - y.value))) if np.any(~np.isnan(x.value)) else 0.0
                for x, y in zip(a, b))
    same_visits = all(np.array_equal(np.isnan(x.value), np.isnan(y.value)) for x, y in zip(a, b))
    print(f"\n1 vs 3 processes: same visited moves {same_visits}, max |value difference| {worst:.5f}")
    assert same_visits and worst == 0.0
