"""The HTTP side: a JSON API over `stats` and `models`, and the static page."""

from __future__ import annotations

import contextlib
import io
import os
import re
import threading
import time
from pathlib import Path

import chess
import chess.pgn
import chess.svg
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from tnh.web.engine import EngineError, EngineManager
from tnh.web.models import ModelHub, ModelsOff, Superseded, _LRU, query
from tnh.web.models import _board as _models_board
from tnh.web.stats import TEMPI, Filters, Shrink, Stats, TrendRule, model_table, report

# $TNH_WEB_STATIC serves another copy of the page, so it can be changed without touching the
# one a running server serves from the working tree.
STATIC = Path(os.environ.get("TNH_WEB_STATIC", Path(__file__).parent / "static"))
API_VERSION = 3      # the page checks it: an older server lacks /api/model and the Elo deltas
P_LIST = 0.01        # model 1's moves at or above this probability are listed beside the database's


def _filters(y0: int | None, y1: int | None, e0: int | None, e1: int | None, tempo: str | None) -> Filters:
    tempi = None
    if tempo:
        tempi = tuple(t for t in tempo.split(",") if t)
        if set(tempi) == set(TEMPI):
            tempi = None
    try:
        return Filters(y0, y1, e0, e1, tempi)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


def _board(fen: str) -> chess.Board:
    try:
        board = chess.Board(fen)
    except ValueError as e:
        raise HTTPException(400, f"bad FEN: {e}") from e
    if not board.is_valid():
        raise HTTPException(400, f"invalid position: {board.status()!r}")
    return board


def _moves(moves: str) -> list[str]:
    return [m for m in moves.replace(",", " ").split() if m]


def _pgn_tree(text: str) -> dict:
    """A PGN (first game) as {root_fen, tree}: nodes {uci, san, children} with variations."""
    game = chess.pgn.read_game(io.StringIO(text))
    if game is None:
        raise HTTPException(400, "no game in the PGN")
    if game.errors:
        raise HTTPException(400, "PGN errors: " + "; ".join(str(e) for e in game.errors[:5]))

    def walk(node: chess.pgn.GameNode) -> list[dict]:
        out = []
        for child in node.variations:
            out.append({"uci": child.move.uci(), "san": child.san(), "children": walk(child)})
        return out

    root = game.board()
    return {"root_fen": root.fen(), "tree": walk(game), "headers": dict(game.headers)}


def piece_sprite(tile: int = 40) -> str:
    """The standard piece drawings (Colin M.L. Burnett's, as on Lichess and Wikipedia) as one sprite
    for cm-chessboard: a group per piece with the ids wk, wq, ... bp, on tiles of `tile` units.

    Taken at run time from python-chess (`chess.svg.PIECES`, drawn on 45 units), so the
    drawings are not copied into this repository.
    """
    parts = []
    for symbol, body in chess.svg.PIECES.items():
        name = ("w" if symbol.isupper() else "b") + symbol.lower()
        body = re.sub(r'^<g id="[^"]*"', "<g", body)
        parts.append(f'<g id="{name}" transform="scale({tile / chess.svg.SQUARE_SIZE:.6f})">{body}</g>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{tile}" height="{tile}" viewBox="0 0 {tile} {tile}">\n'
            + "\n".join(parts) + "\n</svg>\n")


def _board_with(fen: str, moves: list[str]) -> chess.Board:
    """The board after `moves` from the start (with its history), or the FEN alone."""
    try:
        return _models_board(fen, moves)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


def create_app(stats: Stats, hub: ModelHub, *, trend_rule: TrendRule = TrendRule(),
               shrink: Shrink = Shrink(), engines: EngineManager | None = None) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(_app):
        yield
        if engines is not None:          # the engine process goes with the server
            engines.close()

    app = FastAPI(title="tn-hittare", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    sprite = piece_sprite()

    @app.get("/pieces.svg")
    def pieces():
        return Response(sprite, media_type="image/svg+xml", headers={"Cache-Control": "no-cache"})

    @app.get("/api/info")
    def info():
        return {"api": API_VERSION, "stats": stats.info(), "models": hub.status(), "tempi": list(TEMPI),
                "p_list": P_LIST,
                "shrink": shrink.describe(),
                "trend": {"recent_years": trend_rule.recent_years, "alpha": trend_rule.alpha,
                          "min_games": trend_rule.min_games}}

    reports = _LRU(2000)
    gen, gen_lock = {"n": 0}, threading.Lock()

    def cached_report(board: chess.Board, f: Filters) -> dict:
        key = (board.epd(), f)
        hit = reports.get(key)
        if hit is None:
            hit = report(stats, board, f, trend_rule, shrink)
            reports.put(key, hit)
        return hit

    @app.get("/api/stats")
    def position_stats(fen: str, y0: int | None = None, y1: int | None = None, e0: int | None = None,
                       e1: int | None = None, tempo: str | None = None):
        board = _board(fen)
        f = _filters(y0, y1, e0, e1, tempo)
        t = time.perf_counter()
        out = dict(cached_report(board, f))
        out["ms"] = (time.perf_counter() - t) * 1000
        return out

    @app.get("/api/model")
    def model_columns(fen: str, moves: str = "", white: int = Query(2200, ge=0, le=4000),
                      black: int = Query(2200, ge=0, le=4000), y0: int | None = None, y1: int | None = None,
                      e0: int | None = None, e1: int | None = None, tempo: str | None = None):
        """The model columns of the move table: model 1 for the side to move, model 2 for the
        position and each listed move (the database's moves under the filters and model 1's
        moves of at least `P_LIST`), mixed with the database. A newer call supersedes this one
        between model-2 batches (409); what was computed stays cached."""
        board = _board(fen)
        mv = _moves(moves)
        f = _filters(y0, y1, e0, e1, tempo)
        with gen_lock:
            gen["n"] += 1
            mine = gen["n"]
        t0 = time.perf_counter()
        rep = cached_report(board, f)
        if board.is_game_over():
            raise HTTPException(400, "game over in this position")
        try:
            probs = hub.move_probs(fen, mv, white, black)
            t1 = time.perf_counter()
            listed = sorted({m["uci"] for m in rep["moves"]} | {u for u, p in probs["moves"].items() if p >= P_LIST})
            mboard = _board_with(fen, mv)
            queries = [query(mboard, mv, white, black)]
            for u in listed:
                child = mboard.copy(stack=False)
                child.push_uci(u)
                queries.append(query(child, mv + [u] if mv else [], white, black))
            before = dict(hub.status().get("counts", {}))
            vals = hub.values(queries, cancelled=lambda: gen["n"] != mine)
            after = hub.status().get("counts", {})
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except Superseded as e:
            raise HTTPException(409, f"superseded by a newer request: {e}") from e
        t2 = time.perf_counter()
        out = model_table(rep, probs["moves"], vals[0], dict(zip(listed, vals[1:])),
                          board.turn == chess.BLACK, shrink)
        out |= {
            "move_model": probs["model"], "value_model": hub.status().get("value_model"),
            "white": white, "black": black, "listed": listed, "shrink": rep["shrink"],
            "cost": {"move_model_ms": probs["ms"] if not probs["cached"] else 0.0, "move_model_cached": probs["cached"],
                     "value_queries": len(queries),
                     "value_computed": after.get("value_queries", 0) - before.get("value_queries", 0),
                     "value_ms": (t2 - t1) * 1000, "total_ms": (t2 - t0) * 1000},
        }
        return out

    @app.get("/api/uci/engines")
    def uci_engines():
        """The engines the engine section offers, and whether each can start."""
        if engines is None:
            return {"engines": []}
        return {"engines": [{"name": s.name, "label": s.label, "short": s.short, "settings": list(s.settings),
                             "unavailable": s.available()} for s in engines.specs.values()]}

    @app.post("/api/uci/go")
    async def uci_go(request: Request):
        """Analyse a position: {engine, fen, moves, multipv, settings, me}; engine "off" quits it.
        `me` is the side at the bottom of the board (Lc0's contempt side)."""
        if engines is None:
            raise HTTPException(404, "no engines on this server")
        body = await request.json()
        name = body.get("engine", "off")
        fen = body.get("fen") or chess.STARTING_FEN
        _board(fen)
        moves = [m for m in body.get("moves", []) if m]
        multipv = int(body.get("multipv", 3))
        if not 1 <= multipv <= 20:
            raise HTTPException(400, "multipv must be 1-20")
        try:
            gen = await run_in_threadpool(engines.go, name, fen, moves, multipv, body.get("settings") or {},
                                          body.get("me", "white"))
        except EngineError as e:
            raise HTTPException(503, str(e)) from e
        return {"gen": gen}

    @app.get("/api/uci/state")
    def uci_state():
        if engines is None:
            return {"engine": "off", "lines": [], "strings": [], "error": None, "gen": 0, "fen": None}
        return engines.snapshot()

    @app.post("/api/pgn")
    async def pgn(request: Request):
        return _pgn_tree((await request.body()).decode("utf-8", errors="replace"))

    @app.exception_handler(ModelsOff)
    async def models_off(_request, exc: ModelsOff):
        return JSONResponse({"detail": str(exc)}, status_code=503)

    @app.exception_handler(Exception)
    async def unhandled(_request, exc: Exception):
        # Shown in the page's status line; the traceback is in the server log.
        return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)

    return app
