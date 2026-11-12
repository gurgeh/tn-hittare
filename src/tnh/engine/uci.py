"""The repertoire engine over UCI.

    uv run python -m tnh.engine.uci              # classical and blitz models, the full database
    uv run python -m tnh.engine.uci --fake       # deterministic fake models, no database (tests)

Commands: `uci`, `isready`, `setoption` (while idle), `ucinewgame` (clears the graph;
otherwise it is kept between positions and stepping into a child or back reuses it),
`position startpos|fen ... [moves ...]`, `go [nodes N | movetime MS | infinite]`
(anything else, such as clock times, searches until `stop`), `stop`, `quit`.

**What the score means.** Every value is my expected score in the repertoire game (see
`tnh.engine.search`), turned into the side to move's at the root. `score cp` is
`400 * log10(E / (1 - E))` for that expected score E (clipped to +-2000): 0 is an even
game, +200 is 0.76. With me to move the lines are my candidate moves by net value (value
minus the price of the non-obvious moves the line's repertoire learns); with the
opponent to move they are his most likely replies.

**`info string tnh ...`** gives the exact numbers, one line per pv and one per search:

    info string tnh line K value V net N nonobvious C reach R moves m1:F1 m2:F2 ...
    info string tnh root value V net N price P cost C cap X stand_in S frontier F finished G expanded E

where each move's flag F is `L` (my move, learned), `M` (my move, not learned: the move
model's most likely) or the opponent's probability of the move (e.g. `0.412`).
`cap` is `-` without a cap; with one, `cost` may be below it (the price only reaches the
points of the concave hull of value against cost).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_v, "1")

import chess  # noqa: E402

from tnh.engine.api import Handle  # noqa: E402
from tnh.engine.search import Engine, Result, Settings  # noqa: E402

NAME = "tnh-repertoire"

# UCI option -> (Settings field, type, UCI declaration)
OPTIONS = {
    "MyElo": ("my_elo", int, "type spin default 2200 min 400 max 3200"),
    "OppElo": ("opp_elo", int, "type spin default 2200 min 400 max 3200"),
    "Side": ("my_side", str, "type combo default tomove var white var black var tomove"),
    "Tempo": ("tempo", str, "type combo default classical var classical var blitz{fake}"),
    "Price": ("price", float, "type string default 0.001"),
    "Cap": ("cap", int, "type spin default 0 min 0 max 1000"),
    "Obvious": ("obvious", float, "type string default 0.85"),
    "Prepared": ("prepared", float, "type string default 0"),
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
    def __init__(self, engine: Engine, out=sys.stdout, make_evaluator=None) -> None:
        self.engine = engine
        self.handle = Handle(engine)
        self.out = out
        self.make_evaluator = make_evaluator
        self.lock = threading.Lock()
        self.t0 = 0.0
        self.exp0 = 0
        self.fake = False

    def send(self, line: str) -> None:
        with self.lock:
            self.out.write(line + "\n")
            self.out.flush()

    # -- reporting ---------------------------------------------------------------------------

    def report(self, r: Result) -> None:
        if r is None:
            return
        eng = self.engine
        nodes = eng.stats["expanded"]
        ms = max(int(r.seconds * 1000), 1)
        nps = int(r.expansions / max(r.seconds, 1e-9))
        for k, line in enumerate(r.lines, 1):
            pv = " ".join(line.moves)
            self.send(f"info depth {len(line.moves)} multipv {k} score cp {cp_of(line.value)} nodes {nodes} "
                      f"nps {nps} time {ms} pv {pv}")
            flags = []
            for m, p, l in zip(line.moves, line.probs, line.learned):
                flags.append(f"{m}:{'L' if l else 'M'}" if p is None else f"{m}:{p:.4f}")
            net = "-" if line.net is None else f"{line.net:.5f}"
            self.send(f"info string tnh line {k} value {line.value:.5f} net {net} nonobvious {line.nonobvious} "
                      f"reach {line.reach:.5f} moves {' '.join(flags)}")
        s = r.solve
        m = r.mass
        root = eng.graph.nodes[eng.graph.root]
        v = s.value if root.me_to_move else 1 - s.value
        cap = "-" if eng.s.cap == 0 else str(eng.s.cap)
        self.send(f"info string tnh root value {v:.5f} net {s.net:.5f} price {s.mu:.6g} cost {s.cost} cap {cap} "
                  f"stand_in {m['stand_in']:.4f} frontier {m['frontier']:.4f} finished {m['finished']:.4f} "
                  f"expanded {nodes}")

    def bestmove(self, r: Result | None) -> None:
        if r is not None and r.lines:
            self.send(f"bestmove {r.lines[0].moves[0]}")
            return
        root = self.engine.graph.nodes[self.engine.graph.root]
        b = self.engine.ev.board(self.engine.pos[root.id])
        legal = list(b.legal_moves)
        self.send(f"bestmove {legal[0].uci() if legal else '0000'}")

    # -- commands ------------------------------------------------------------------------------

    def cmd_uci(self) -> None:
        self.send(f"id name {NAME}")
        self.send("id author tn-hittare")
        for name, (_, _, decl) in OPTIONS.items():
            self.send(f"option name {name} {decl.format(fake=' var fake' if self.fake else '')}")
        self.send("uciok")

    def cmd_setoption(self, args: list[str]) -> None:
        self.handle.require_idle()
        if "name" not in args:
            return
        i = args.index("name")
        j = args.index("value") if "value" in args else len(args)
        name = " ".join(args[i + 1 : j])
        value = " ".join(args[j + 1 :]) if j < len(args) else ""
        if name not in OPTIONS:
            self.send(f"info string unknown option {name}")
            return
        field, typ, _ = OPTIONS[name]
        if field == "my_side" and value == "tomove":
            value = "to_move"
        self.engine.set_settings(**{field: typ(value)})

    def cmd_position(self, args: list[str]) -> None:
        self.handle.require_idle()
        moves = args[args.index("moves") + 1 :] if "moves" in args else []
        if args and args[0] == "fen":
            end = args.index("moves") if "moves" in args else len(args)
            fen = chess.Board(" ".join(args[1:end])).fen()
            if fen != chess.Board().fen():
                if self.make_evaluator is None:
                    raise ValueError("this engine starts from the standard position only")
                cur = getattr(self.engine.ev, "start_fen", None)
                if cur != fen:
                    self.engine.ev = self.make_evaluator(fen)
                    self.engine.clear()
            elif getattr(self.engine.ev, "start_fen", None) is not None:
                self.engine.ev = self.make_evaluator(None)
                self.engine.clear()
        elif getattr(self.engine.ev, "start_fen", None) is not None and self.make_evaluator is not None:
            self.engine.ev = self.make_evaluator(None)
            self.engine.clear()
        self.engine.set_position(moves)

    def cmd_go(self, args: list[str]) -> None:
        self.handle.require_idle()
        nodes = seconds = None
        if "nodes" in args:
            nodes = int(args[args.index("nodes") + 1])
        if "movetime" in args:
            seconds = int(args[args.index("movetime") + 1]) / 1000
        self.handle.start(nodes=nodes, seconds=seconds, on_info=self.report,
                          on_done=lambda r: (self.report(r), self.bestmove(r)))

    def run(self, inp=sys.stdin) -> None:
        for raw in inp:
            parts = raw.split()
            if not parts:
                continue
            cmd, args = parts[0], parts[1:]
            if cmd == "uci":
                self.cmd_uci()
            elif cmd == "isready":
                self.send("readyok")
            elif cmd == "setoption":
                self.cmd_setoption(args)
            elif cmd == "ucinewgame":
                self.handle.require_idle()
                self.engine.clear()
            elif cmd == "position":
                self.cmd_position(args)
            elif cmd == "go":
                self.cmd_go(args)
            elif cmd == "stop":
                self.handle.stop()
            elif cmd == "quit":
                if self.handle.busy:
                    self.handle.stop()
                return
            else:
                self.send(f"info string unknown command {cmd}")


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
    bundles = {"fake": FakeBundle()} if a.fake else {}

    def make(fen):
        return ChessEvaluator(bundles, db, start_fen=fen)

    settings = Settings(tempo="fake") if a.fake else Settings()
    eng = Engine(make(None), settings)
    u = UCI(eng, make_evaluator=make)
    u.fake = a.fake
    u.run()


if __name__ == "__main__":
    main()
