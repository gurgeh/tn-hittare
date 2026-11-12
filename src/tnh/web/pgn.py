"""The page's move tree and PGN text: parse a PGN into the tree, write the tree as PGN."""

from __future__ import annotations

import io

import chess
import chess.pgn


def parse_pgn(text: str) -> dict:
    """A PGN (first game) as {root_fen, comment, tree, headers}: nodes {uci, san, comment, pre,
    children} with variations. `comment` is the text after the move, `pre` the comment that opens
    a variation before its first move, the top-level `comment` the one before the first move.
    Comments are kept as text: NAGs and [%...] commands inside them are not interpreted."""
    game = chess.pgn.read_game(io.StringIO(text))
    if game is None:
        raise ValueError("no game in the PGN")
    if game.errors:
        raise ValueError("PGN errors: " + "; ".join(str(e) for e in game.errors[:5]))

    def walk(node: chess.pgn.GameNode) -> list[dict]:
        out = []
        for child in node.variations:
            out.append({"uci": child.move.uci(), "san": child.san(), "comment": child.comment,
                        "pre": child.starting_comment, "children": walk(child)})
        return out

    root = game.board()
    return {"root_fen": root.fen(), "comment": game.comment, "tree": walk(game), "headers": dict(game.headers)}


def pgn_comment(text: str) -> str:
    """A comment as PGN can carry it: a closing brace would end the comment early, so it
    becomes a closing parenthesis."""
    return text.replace("}", ")").strip()


def tree_pgn(body: dict, headers: dict[str, str] | None = None) -> str:
    """The PGN of a move tree in the form `parse_pgn` returns ({root_fen, comment, headers,
    tree}; `san` is not needed), with its variations and comments. Raises ValueError on a bad
    FEN, a bad node or an illegal move. `headers` are set over the tree's own."""
    if not isinstance(body, dict):
        raise ValueError("expected a JSON object")
    fen = body.get("root_fen") or chess.STARTING_FEN
    try:
        board = chess.Board(fen)
    except ValueError as e:
        raise ValueError(f"bad FEN: {e}") from e
    game = chess.pgn.Game()
    for k, v in (body.get("headers") or {}).items():
        if k not in ("FEN", "SetUp"):
            game.headers[str(k)] = str(v)
    for k, v in (headers or {}).items():
        game.headers[k] = v
    if board.fen() != chess.STARTING_FEN:
        game.setup(board)
    game.comment = pgn_comment(str(body.get("comment") or ""))

    def add(node: chess.pgn.GameNode, board: chess.Board, items) -> None:
        if not isinstance(items, list):
            raise ValueError("children must be a list")
        for it in items:
            if not isinstance(it, dict) or not isinstance(it.get("uci"), str):
                raise ValueError(f"bad node: {it!r}"[:200])
            try:
                move = chess.Move.from_uci(it["uci"])
            except ValueError as e:
                raise ValueError(f"bad move {it['uci']!r}") from e
            if not board.is_legal(move):
                raise ValueError(f"illegal move {it['uci']} in {board.fen()}")
            child = node.add_variation(move, comment=pgn_comment(str(it.get("comment") or "")),
                                       starting_comment=pgn_comment(str(it.get("pre") or "")))
            after = board.copy(stack=False)
            after.push(move)
            add(child, after, it.get("children") or [])

    add(game, board, body.get("tree") or [])
    return game.accept(chess.pgn.StringExporter(headers=True, variations=True, comments=True, columns=80))
