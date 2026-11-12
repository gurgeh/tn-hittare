"""preppy: the repertoire engine over UCI.

    uv run python -m tnh.engine.uci              # classical and blitz models, the full database
    uv run python -m tnh.engine.uci --fake       # deterministic fake models, no database (tests)

This docstring is the contract for a GUI (the web page's engine section).

## Behaviour

- `uci` answers at once; the models load once, at the first `isready` or `go`.
- `position startpos [moves ...]` or `position fen F [moves ...]` sets the root. The graph
  and every cache are kept: stepping into a child or back reuses what was searched. Only
  `ucinewgame` clears them (and a `position fen` with a new starting FEN).
- `go infinite`, `go nodes N` (N more expanded positions), `go movetime MS`; any other
  `go` (clock times) searches until `stop`. `stop` answers with `bestmove` at once, from
  the last exact solve (it does not wait for a new one).
- `setoption` between searches. A `position`, `setoption`, `go` or `ucinewgame` that
  arrives while a search runs stops it first (its `bestmove` is sent).
- `quit`, or the end of stdin, stops any search and exits.
- An error in a command is reported as `info string preppy error=...` and the engine
  goes on; nothing is swallowed silently.

## Options

| option | type | default | range / values | meaning |
|---|---|---|---|---|
| MyElo | spin | 2200 | 400-3200 | my rating (FIDE for classical, Lichess blitz for blitz) |
| OppElo | spin | 2200 | 400-3200 | the opponent's |
| Side | combo | tomove | white, black, tomove | which side I am; tomove = the side to move at the root |
| Tempo | combo | classical | classical, blitz (+ fake with --fake) | the model bundle: classical = the default move model (Maia-3 79M +0/+600, Lc0 policy, trees) with the cheap result model, classical games in the database; blitz = plain Maia-3 79M at the given ratings with its own value head, blitz and rapid games |
| Price | string (float) | 0.001 | >= 0 | price per non-obvious learned move, in expected score |
| Cap | spin | 0 | 0-1000 | at most this many non-obvious learned moves (0 = off; then Price decides). A cap can return fewer: the price reaches only the concave hull of value against cost |
| Obvious | string (float) | 0.85 | 0-1 | a move is obvious (costs 0) when my move model alone gives it at least this |
| Prepared | string (float) | 0 | 0-1 | share of the opponent's probability on his best reply, at every position |
| DrawScore | string (float) | 0.5 | 0-1 | what a draw is worth to me: a position's value for me is W + DrawScore x D from the result model (the opponent gets 1 minus it), a drawn finished game is worth DrawScore; the opponent's move probabilities do not change. 0.5 is the plain expected score. The real contempt is the rating difference (MyElo against OppElo), which the models play out; this only re-weights draws |
| Mix | combo | fitted | fitted, kappa | the opponent's database mix: the fitted rule (tnh.movemix) or (count + kappa p) / (n + kappa) |
| Kappa | string (float) | 100 | > 0 | with Mix = kappa |
| DbFilter | combo | auto | auto, classical, blitz, all, off | the database games counted; auto = the tempo's own |
| MultiPV | spin | 3 | 1-20 | lines shown |
| CPuct | string (float) | 1.5 | > 0 | PUCT exploration at my moves |
| PolicyWeight | string (float) | 0.5 | 0-1 | Lc0's raw policy in my prior (the rest is my move model) |
| PriorMin | string (float) | 0.002 | 0-1 | safety net: a move with a smaller prior is not a candidate |
| Blunder | string (float) | 0.15 | 0-1 | safety net: a learned move's position may be at most this much worse than the one before |
| Batch | spin | 12 | 1-512 | positions evaluated per batch |
| SolveRatio | string (float) | 4 | >= 0 | solve again when the time since the last solve is this many times its duration (solving stays about 1/(1+ratio) of the time) |

## Output

Every value is my expected score in the repertoire game (`tnh.engine.search`), turned into
the side to move's at the root. **Score mapping:** `score cp X` with
`X = round(400 * log10(E / (1 - E)))` for that expected score E, clipped to +-2000
(0 = even, +200 = 0.76, +400 = 0.91). No `wdl`: the search keeps expected scores only.

A round of info goes out at once when a search starts from a root the last solve
covered (provisional), again right after the first solve at the new root, and then at
most about once per second. Per line, in order of `multipv`:

    info depth D multipv K score cp X nodes N nps S time MS pv m1 m2 ...
    info string preppy multipv=K value=V net=NET nonobvious=C reach=R provisional=0|1 moves=m1:F1,m2:F2,...

and once per round:

    info string preppy root value=V net=NET price=P cost=C cap=X stand_in=S frontier=F finished=G expanded=E provisional=0|1

- `value`: the line's expected score for the side to move at the root (6 decimals).
- `net` (me to move): value minus price times the non-obvious moves the line's repertoire
  learns; `-` when the opponent is to move.
- `nonobvious`: non-obvious learned moves on the line. `reach`: probability of reaching
  the line's end under the repertoire.
- Each move's flag `F`: `L` my move, learned; `M` my move, not learned (my model's most
  likely); a number: the opponent's probability of the move.
- With me to move the lines are my candidate moves by net value; with the opponent to
  move, his most likely replies.
- `provisional=1`: the round comes from the last solve made at another root or with
  other settings (sent at once after `position` or `setoption`, so the page has lines
  while the new root is solved); the next round with `provisional=0` replaces it. Only
  `provisional=0` values are exact for the current root and settings.
- `root`: the repertoire's value and net value, the price in force (with a cap, the one
  found), the non-obvious moves it learns, `cap=-` without a cap, the probability mass
  that ends on stand-ins (positions not yet expanded, valued by their parent),
  frontier positions and finished games, and the expanded positions so far.
- `nodes` in the `info` line is the number of expanded positions in the graph.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_v, "1")

import chess  # noqa: E402

from tnh.engine.api import Handle  # noqa: E402
from tnh.engine.search import Engine, Result, Settings  # noqa: E402

NAME = "preppy"

# UCI option -> (Settings field, type, UCI declaration)
OPTIONS = {
    "MyElo": ("my_elo", int, "type spin default 2200 min 400 max 3200"),
    "OppElo": ("opp_elo", int, "type spin default 2200 min 400 max 3200"),
    "Side": ("my_side", str, "type combo default tomove var white var black var tomove"),
    "Tempo": ("tempo", str, "type combo default {tempo} var classical var blitz{fake}"),
    "Price": ("price", float, "type string default 0.001"),
    "Cap": ("cap", int, "type spin default 0 min 0 max 1000"),
    "Obvious": ("obvious", float, "type string default 0.85"),
    "Prepared": ("prepared", float, "type string default 0"),
    "DrawScore": ("draw_score", float, "type string default 0.5"),
    "Mix": ("mix", str, "type combo default fitted var fitted var kappa"),
    "Kappa": ("kappa", float, "type string default 100"),
    "DbFilter": ("db_filter", str, "type combo default auto var auto var classical var blitz var all var off"),
    "MultiPV": ("lines", int, "type spin default 3 min 1 max 20"),
    "CPuct": ("c_puct", float, "type string default 1.5"),
    "PolicyWeight": ("policy_weight", float, "type string default 0.5"),
    "PriorMin": ("prior_min", float, "type string default 0.002"),
    "Blunder": ("blunder", float, "type string default 0.15"),
    "Batch": ("batch", int, "type spin default 12 min 1 max 512"),
    "SolveRatio": ("solve_ratio", float, "type string default 4"),
}


def cp_of(e: float) -> int:
    e = min(max(e, 1e-3), 1 - 1e-3)
    return int(round(max(-2000.0, min(2000.0, 400 * math.log10(e / (1 - e))))))


class UCI:
    def __init__(self, engine: Engine, out=sys.stdout, make_evaluator=None, fake: bool = False) -> None:
        self.engine = engine
        self.handle = Handle(engine)
        self.out = out
        self.make_evaluator = make_evaluator
        self.lock = threading.Lock()
        self.fake = fake
        self.loaded = False
        self.trace: list[dict] = []          # per go: where the time went until the first lines
        self._t_position = None
        self._stop_wait = 0.0

    def send(self, line: str) -> None:
        with self.lock:
            self.out.write(line + "\n")
            self.out.flush()

    # -- reporting ---------------------------------------------------------------------------

    def report(self, r: Result | None) -> None:
        if r is None or r.solve is None:
            return
        if self.trace and r.lines:
            tr = self.trace[-1]
            key = "first_provisional" if r.provisional else "first_info"
            if key not in tr:
                tr[key] = time.perf_counter() - tr["t_position"]
                tr.update({k: v for k, v in self.engine.timing.items()})
        eng = self.engine
        nodes = eng.stats["expanded"]
        ms = max(int(r.seconds * 1000), 1)
        nps = int(r.expansions / max(r.seconds, 1e-9))
        for k, line in enumerate(r.lines, 1):
            self.send(f"info depth {len(line.moves)} multipv {k} score cp {cp_of(line.value)} nodes {nodes} "
                      f"nps {nps} time {ms} pv {' '.join(line.moves)}")
            flags = ",".join(f"{m}:{'L' if l else 'M'}" if p is None else f"{m}:{p:.4f}"
                             for m, p, l in zip(line.moves, line.probs, line.learned))
            net = "-" if line.net is None else f"{line.net:.6f}"
            self.send(f"info string preppy multipv={k} value={line.value:.6f} net={net} "
                      f"nonobvious={line.nonobvious} reach={line.reach:.6f} provisional={int(r.provisional)} "
                      f"moves={flags}")
        s, m = r.solve, r.mass
        root = eng.graph.nodes[eng.graph.root]
        v = s.value if root.me_to_move else 1 - s.value
        cap = "-" if eng.s.cap == 0 else str(eng.s.cap)
        self.send(f"info string preppy root value={v:.6f} net={s.net:.6f} price={s.mu:.6g} cost={s.cost} cap={cap} "
                  f"stand_in={m['stand_in']:.4f} frontier={m['frontier']:.4f} finished={m['finished']:.4f} "
                  f"expanded={nodes} provisional={int(r.provisional)}")

    def bestmove(self, r: Result | None) -> None:
        if r is not None and r.lines:
            self.send(f"bestmove {r.lines[0].moves[0]}")
            return
        b = self.engine.ev.board(self.engine.pos[self.engine.graph.root])
        legal = list(b.legal_moves)
        self.send(f"bestmove {legal[0].uci() if legal else '0000'}")

    # -- commands ------------------------------------------------------------------------------

    def idle(self) -> None:
        """Stop a running search first (its bestmove is sent by the search thread)."""
        if self.handle.busy:
            t = time.perf_counter()
            self.handle.stop()
            self._stop_wait = time.perf_counter() - t

    def load(self) -> None:
        if not self.loaded:
            self.engine.ev.bundle(self.engine.s.tempo)
            self.loaded = True

    def cmd_uci(self) -> None:
        self.send(f"id name {NAME}")
        self.send("id author tn-hittare")
        for name, (_, _, decl) in OPTIONS.items():
            self.send(f"option name {name} " + decl.format(fake=" var fake" if self.fake else "",
                                                         tempo=self.engine.s.tempo))
        self.send("uciok")

    def cmd_setoption(self, args: list[str]) -> None:
        self.idle()
        if "name" not in args:
            return
        i = args.index("name")
        j = args.index("value") if "value" in args else len(args)
        name = " ".join(args[i + 1 : j])
        value = " ".join(args[j + 1 :]) if j < len(args) else ""
        if name not in OPTIONS:
            self.send(f"info string preppy error=unknown_option:{name}")
            return
        field, typ, _ = OPTIONS[name]
        if field == "my_side" and value == "tomove":
            value = "to_move"
        self.engine.set_settings(**{field: typ(value)})
        if field == "tempo":
            self.loaded = False

    def cmd_position(self, args: list[str]) -> None:
        self._stop_wait = 0.0
        t_pos = time.perf_counter()
        self.idle()
        self.trace.append({"t_position": t_pos, "moves": len(args), "stop_wait": self._stop_wait})
        moves = args[args.index("moves") + 1 :] if "moves" in args else []
        fen = None
        if args and args[0] == "fen":
            end = args.index("moves") if "moves" in args else len(args)
            fen = chess.Board(" ".join(args[1:end])).fen()
            if fen == chess.Board().fen():
                fen = None
        cur = getattr(self.engine.ev, "start_fen", None)
        if fen != cur:
            if self.make_evaluator is None:
                raise ValueError("this engine starts from the standard position only")
            old = self.engine.ev
            self.engine.ev = self.make_evaluator(fen, getattr(old, "bundles", None))
            self.engine.clear()
        self.engine.set_position(moves)

    def cmd_go(self, args: list[str]) -> None:
        self.idle()
        self.load()
        nodes = seconds = None
        if "nodes" in args:
            nodes = int(args[args.index("nodes") + 1])
        if "movetime" in args:
            seconds = int(args[args.index("movetime") + 1]) / 1000
        self.handle.start(nodes=nodes, seconds=seconds, on_info=self.report,
                          on_done=lambda r: (self.report(r), self.bestmove(r)))

    def dispatch(self, cmd: str, args: list[str]) -> bool:
        if cmd == "uci":
            self.cmd_uci()
        elif cmd == "isready":
            if not self.handle.busy:
                self.load()
            self.send("readyok")
        elif cmd == "setoption":
            self.cmd_setoption(args)
        elif cmd == "ucinewgame":
            self.idle()
            self.engine.clear()
        elif cmd == "position":
            self.cmd_position(args)
        elif cmd == "go":
            self.cmd_go(args)
        elif cmd == "stop":
            self.idle()
        elif cmd == "quit":
            self.idle()
            return False
        else:
            self.send(f"info string preppy error=unknown_command:{cmd}")
        return True

    def run(self, inp=sys.stdin) -> None:
        for raw in inp:
            parts = raw.split()
            if not parts:
                continue
            try:
                if not self.dispatch(parts[0], parts[1:]):
                    return
            except Exception as e:                      # shown to the GUI, and the engine goes on
                msg = f"{type(e).__name__}:{e}".replace("\n", " ")
                self.send(f"info string preppy error={msg}")
        self.idle()                                     # stdin closed


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tnh.engine.uci")
    ap.add_argument("--fake", action="store_true", help="deterministic fake models and no database (tests)")
    ap.add_argument("--no-db", action="store_true")
    a = ap.parse_args(argv)
    from tnh.engine.chess_eval import ChessEvaluator, FakeBundle

    db = None
    if not (a.fake or a.no_db):
        from tnh.data.edges import EdgeDB
        db = EdgeDB("full")
    base = {"fake": FakeBundle()} if a.fake else {}

    def make(fen, bundles=None):
        return ChessEvaluator(bundles if bundles is not None else base, db, start_fen=fen)

    settings = Settings(tempo="fake") if a.fake else Settings()
    eng = Engine(make(None), settings)
    UCI(eng, make_evaluator=make, fake=a.fake).run()


if __name__ == "__main__":
    main()
