"""The repertoire builder's results as a layer over the board: a stopgap until the builder runs
as an engine beside Stockfish (docs/DESIGN.md, "Where the builder is going").

Reads `<dir>/<name>/result.json` and `repertoire_N<budget>.pgn` as `tnh.builder.export`
writes them, one run at a time on first use, and keeps them in memory. Answers per (run,
budget, position). Values in the export are for the repertoire's side ("me"); everything
here is turned to White's view, like the rest of the page: `white(v)`.
"""

from __future__ import annotations

import io
import json
import threading
from pathlib import Path

import chess
import chess.pgn

from tnh.data.edges import pos_hash


class RunMissing(KeyError):
    pass


class Run:
    def __init__(self, d: Path) -> None:
        self.dir = d
        r = json.loads((d / "result.json").read_text())
        self.meta = r["meta"]
        self.solutions = r["solutions"]
        self.forced = r["forced"]
        self.nodes = r["nodes"]
        self.me_white = self.meta["me"] == "white"
        self.budgets = [int(b) for b in self.meta["budgets"]]
        self.lines = {N: self._lines(N) for N in self.budgets}

    def white(self, v: float | None) -> float | None:
        return None if v is None else (v if self.me_white else 1 - v)

    def _lines(self, N: int) -> list[dict]:
        """The learned lines of budget N from its PGN: for each learned move, the most likely
        line from the start that reaches it (as the export chose it), reach, value, cost."""
        p = self.dir / f"repertoire_N{N}.pgn"
        if not p.exists():
            return []
        game = chess.pgn.read_game(io.StringIO(p.read_text()))
        out = []

        def walk(node, moves, sans):
            for v in node.variations:
                m, s = moves + [v.move.uci()], sans + [node.board().san(v.move)]
                c = v.comment.split()
                if len(c) >= 6 and c[0] == "reach" and c[2] == "value" and c[4] == "cost":
                    out.append({"moves": m, "san": sans_line(s, len(self.meta["root_moves"])),
                                "reach": float(c[1]), "value": self.white(float(c[3])), "cost": int(c[5])})
                walk(v, m, s)

        walk(game, [], [])
        return sorted(out, key=lambda x: -x["reach"])

    def summary(self) -> dict:
        root = self.nodes.get(str(pos_hash(chess.Board(self.meta["root_fen"]))))
        forced: dict[str, dict] = {}
        for f in self.forced:
            e = forced.setdefault(f["root_move"], {"uci": f["root_move"], "san": f["san"], "values": {}, "costs": {}})
            e["values"][str(f["budget"])] = self.white(f["value"])
            e["costs"][str(f["budget"])] = f["cost"]
        return {
            "name": self.meta["name"], "date": self.meta.get("date"), "commit": self.meta.get("commit"),
            "me": self.meta["me"], "me_elo": self.meta.get("me_elo"), "opp_elo": self.meta.get("opp_elo"),
            "root_fen": self.meta["root_fen"], "root_moves": self.meta["root_moves"],
            "budgets": self.budgets, "params": self.meta.get("params"), "database": self.meta.get("database"),
            "models": self.meta.get("models"), "graph": self.meta.get("graph"),
            "solutions": {N: {"value": self.white(s["value"]), "cost": s["cost"], "learned": len(s["learned"])}
                          for N, s in self.solutions.items()},
            "root_values": {N: self.white(b["value"]) for N, b in root["budgets"].items()} if root else None,
            "forced": list(forced.values()),
            "lines": {str(N): ls for N, ls in self.lines.items()},
        }

    def position(self, board: chess.Board, N: int) -> dict:
        if N not in self.budgets:
            raise ValueError(f"budget {N} not in this run ({self.budgets})")
        rec = self.nodes.get(str(pos_hash(board)))
        if rec is None:
            return {"in_graph": False}
        b = rec["budgets"][str(N)]
        me_to_move = rec["to_move"] == "me"
        out = {
            "in_graph": True, "state": rec["state"], "ply": rec["ply"], "games": rec["games"],
            "to_move": rec["to_move"], "me_to_move": me_to_move,
            "reach": b["reach"], "value": self.white(b["value"]), "learned": b["learned"],
            "value_head": self.white(rec["value_head"]), "frontier_value": self.white(rec["frontier_value"]),
            "moves": {},
        }
        for m in rec.get("moves", []):
            e = {"value_after": self.white(m["value_after"][str(N)]), "p_model": m["p_model"], "flags": m["flags"],
                 "child_in_run": "not_exported" not in m["flags"]}
            if me_to_move:
                e |= {"engine": self.white(m["engine"]), "loss": m["loss"], "sound": m["sound"],
                      "obvious": m["obvious"], "cost": m["cost"]}
            else:
                e |= {"p_mixed": m["p_mixed"], "db_count": m["db_count"]}
            out["moves"][m["uci"]] = e
        return out


def sans_line(sans: list[str], skip: int) -> str:
    """A SAN line with move numbers, from ply `skip` on (the run's root moves are left out)."""
    parts = []
    for i, s in enumerate(sans):
        if i < skip:
            continue
        num = i // 2 + 1
        if i % 2 == 0:
            parts.append(f"{num}.{s}")
        elif i == skip:
            parts.append(f"{num}...{s}")
        else:
            parts.append(s)
    return " ".join(parts)


class Store:
    """The runs under a directory, each loaded on first use and kept."""

    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()

    def names(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        return sorted(p.parent.name for p in self.dir.glob("*/result.json"))

    def get(self, name: str) -> Run:
        if name not in self.names():
            raise RunMissing(f"no run {name!r} in {self.dir}")
        with self._lock:
            if name not in self._runs:
                self._runs[name] = Run(self.dir / name)
            return self._runs[name]
