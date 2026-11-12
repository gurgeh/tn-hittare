"""The HTTP side: a JSON API over `stats` and `models`, and the static page."""

from __future__ import annotations

import io
import re
import time
from pathlib import Path

import chess
import chess.pgn
import chess.svg
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from tnh.web.models import ModelHub, ModelsOff
from tnh.web.stats import TEMPI, Filters, Stats, TrendRule, report

STATIC = Path(__file__).parent / "static"
MAX_ENGINE_NODES = 5_000_000


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


def create_app(stats: Stats, hub: ModelHub, *, trend_rule: TrendRule = TrendRule()) -> FastAPI:
    app = FastAPI(title="tn-hittare", docs_url=None, redoc_url=None, openapi_url=None)
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
        return {"stats": stats.info(), "models": hub.status(), "tempi": list(TEMPI),
                "trend": {"recent_years": trend_rule.recent_years, "alpha": trend_rule.alpha,
                          "min_games": trend_rule.min_games}}

    @app.get("/api/stats")
    def position_stats(fen: str, y0: int | None = None, y1: int | None = None, e0: int | None = None,
                       e1: int | None = None, tempo: str | None = None):
        board = _board(fen)
        f = _filters(y0, y1, e0, e1, tempo)
        t = time.perf_counter()
        out = report(stats, board, f, trend_rule)
        out["ms"] = (time.perf_counter() - t) * 1000
        return out

    @app.get("/api/maia")
    def maia(fen: str, moves: str = "", white: int = Query(2200, ge=0, le=4000),
             black: int = Query(2200, ge=0, le=4000)):
        _board(fen)
        try:
            return hub.maia(fen, _moves(moves), white, black)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/api/engine")
    def engine(fen: str, moves: str = "", nodes: int = Query(200_000, ge=1000, le=MAX_ENGINE_NODES)):
        board = _board(fen)
        if board.is_game_over():
            raise HTTPException(400, "game over in this position")
        try:
            return hub.engine(fen, _moves(moves), nodes)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

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
