"""A fake UCI engine for the engine-section tests. Declares preppy's options.

While searching (`go infinite`) it prints, every 20 ms, one `info` line per pv: the legal
moves of the position in order, score `cp 1000 * P + 10 * K` where P counts the `position`
commands so far (so a line betrays which position it belongs to) and K is the pv. On `stop`
it prints one more round for the old position before `bestmove`, as a real engine may.
`--slow-start S` waits before `uciok`; `--die-on-go` exits on the first `go`.
"""

import argparse
import sys
import threading
import time

import chess

OPTIONS = [
    "option name MultiPV type spin default 1 min 1 max 20",
    "option name MyElo type spin default 2200 min 400 max 3200",
    "option name OppElo type spin default 2200 min 400 max 3200",
    "option name Side type combo default tomove var white var black var tomove",
    "option name Tempo type combo default classical var classical var blitz",
    "option name Price type string default 0.001",
    "option name Cap type spin default 0 min 0 max 1000",
    "option name Obvious type string default 0.85",
    "option name Prepared type string default 0",
    "option name Hash type spin default 16 min 1 max 1024",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slow-start", type=float, default=0.0)
    ap.add_argument("--die-on-go", action="store_true")
    a = ap.parse_args()
    out_lock = threading.Lock()

    def say(s):
        with out_lock:
            sys.stdout.write(s + "\n")
            sys.stdout.flush()

    state = {"board": chess.Board(), "pos": 0, "multipv": 1, "options": {}, "stop": None, "thread": None}

    def rounds(board, pos, mpv, k_round):
        legal = [m.uci() for m in board.legal_moves][:mpv]
        for k, m in enumerate(legal, 1):
            say(f"info depth {k_round} seldepth {k_round} multipv {k} score cp {1000 * pos + 10 * k} nodes {100 * k_round} "
                f"nps 1000 pv {m}")
        say(f"info string tnh root value 0.5 pos {pos} depth {k_round}")

    def search(stop, board, pos, mpv):
        k = 1
        while not stop.is_set():
            rounds(board, pos, mpv, k)
            k += 1
            time.sleep(0.02)
        rounds(board, pos, mpv, k)          # one more round after stop, for the old position
        legal = list(board.legal_moves)
        say(f"bestmove {legal[0].uci() if legal else '0000'}")

    for line in sys.stdin:
        t = line.split()
        if not t:
            continue
        cmd = t[0]
        if cmd == "uci":
            time.sleep(a.slow_start)
            say("id name fake-uci")
            for o in OPTIONS:
                say(o)
            say("uciok")
        elif cmd == "isready":
            say("readyok")
        elif cmd == "setoption":
            name = t[t.index("name") + 1:t.index("value")]
            value = " ".join(t[t.index("value") + 1:])
            state["options"][" ".join(name)] = value
            say(f"info string set {' '.join(name)} {value}")
            if name == ["MultiPV"]:
                state["multipv"] = int(value)
        elif cmd == "position":
            if t[1] == "startpos":
                b = chess.Board()
                rest = t[2:]
            else:
                i = t.index("moves") if "moves" in t else len(t)
                b = chess.Board(" ".join(t[2:i]))
                rest = t[i:]
            for m in rest[1:] if rest[:1] == ["moves"] else []:
                b.push_uci(m)
            state["board"] = b
            state["pos"] += 1
        elif cmd == "go":
            if a.die_on_go:
                sys.stderr.write("fake engine: dying on go\n")
                sys.exit(3)
            stop = threading.Event()
            th = threading.Thread(target=search, args=(stop, state["board"].copy(), state["pos"], state["multipv"]))
            state["stop"], state["thread"] = stop, th
            th.start()
        elif cmd == "stop":
            if state["stop"] is not None:
                state["stop"].set()
                state["thread"].join()
                state["stop"] = None
        elif cmd == "quit":
            if state["stop"] is not None:
                state["stop"].set()
                state["thread"].join()
            return


if __name__ == "__main__":
    main()
