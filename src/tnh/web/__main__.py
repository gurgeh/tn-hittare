"""Serve the interface on 127.0.0.1.

    uv run python -m tnh.web [--port 8765] [--data auto|dev|full|fixture|games:PATH] [--maia-size 5m]

Data, in order for `auto`: the development tables `corpus/edges_dev.parquet` and
`corpus/pos_index_dev.parquet`, read by `tnh.data.edges.EdgeDB`; otherwise
a fixture built in memory by `tnh.data.edges.tables_from_games` from a slice of
`games.parquet` (one row group, the first `--fixture-games` rows), cached under
`scratch/web/`. `games:PATH` builds the tables in memory from a parquet file of games in
the games.parquet schema (tests use it). The page says which one it is showing.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time

import polars as pl
import pyarrow.parquet as pq
import uvicorn

from tnh.paths import CORPUS, DATA_ROOT
from tnh.web.app import create_app
from tnh.web.models import Hub
from tnh.web.stats import Stats

HOST = "127.0.0.1"
SCRATCH = DATA_ROOT / "scratch" / "web"


def fixture_stats(n_games: int, row_group: int = -1) -> Stats:
    """Edges and pos_index for `n_games` games of one row group of games.parquet, cached."""
    from tnh.data.edges import GAME_COLUMNS, tables_from_games

    edges_path = SCRATCH / f"fixture_{n_games}_edges.parquet"
    index_path = SCRATCH / f"fixture_{n_games}_pos_index.parquet"
    name = f"fixture: {n_games} games from games.parquet, built by tables_from_games"
    if edges_path.exists() and index_path.exists():
        return Stats.from_frames(pl.read_parquet(edges_path), pl.read_parquet(index_path), name)
    t = time.perf_counter()
    f = pq.ParquetFile(CORPUS / "games.parquet")
    rg = row_group % f.metadata.num_row_groups
    games = pl.from_arrow(f.read_row_group(rg, columns=GAME_COLUMNS)).head(n_games)
    edges, index = tables_from_games(games)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    edges.write_parquet(edges_path)
    index.write_parquet(index_path)
    print(f"fixture: {games.height} games -> {edges.height} edge rows, {index.height} positions "
          f"in {time.perf_counter() - t:.1f}s, cached in {SCRATCH}", file=sys.stderr)
    return Stats.from_frames(edges, index, name)


def open_stats(data: str, fixture_games: int):
    if data in ("auto", "dev"):
        from tnh.data.edges import EdgeDB
        dev = CORPUS / "edges_dev.parquet"
        if dev.exists() and (CORPUS / "pos_index_dev.parquet").exists():
            return Stats(EdgeDB(name="dev"), f"edges_dev ({dev})")
        if data == "dev":
            raise SystemExit(f"--data dev: {dev} does not exist yet")
        print(f"auto: {dev} does not exist yet; using the fixture", file=sys.stderr)
    if data.startswith("games:"):
        from tnh.data.edges import tables_from_games
        path = data.split(":", 1)[1]
        games = pl.read_parquet(path)
        edges, index = tables_from_games(games)
        return Stats.from_frames(edges, index, f"games: {games.height} games from {path}")
    if data == "full":
        from tnh.data.edges import EdgeDB
        return Stats(EdgeDB(name="full"), "edges (full corpus)")
    if data in ("auto", "fixture"):
        return fixture_stats(fixture_games)
    raise SystemExit(f"--data: auto, dev, full, fixture or games:PATH, got {data!r}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tnh.web", description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--data", default="auto", help="auto, dev, full, fixture or games:PATH")
    ap.add_argument("--fixture-games", type=int, default=30000)
    ap.add_argument("--maia-size", default="5m", choices=["5m", "23m", "79m"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--torch-threads", type=int, default=2)
    args = ap.parse_args(argv)

    with socket.socket() as s:
        try:
            s.bind((HOST, args.port))
        except OSError as e:
            raise SystemExit(f"port {args.port} on {HOST} is taken ({e}); pass --port") from e

    stats = open_stats(args.data, args.fixture_games)
    hub = Hub(maia_size=args.maia_size, device=args.device, torch_threads=args.torch_threads)
    app = create_app(stats, hub)
    print(f"tn-hittare: http://{HOST}:{args.port}/  ({stats.info()['source']})", file=sys.stderr)
    try:
        uvicorn.run(app, host=HOST, port=args.port, log_level="warning")
    finally:
        hub.close()


if __name__ == "__main__":
    main()
