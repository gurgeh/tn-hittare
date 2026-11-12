"""Serve the interface on 127.0.0.1.

    uv run python -m tnh.web [--port 8765] [--table full|dev] [--move-model combo-p0.5] [--value-model frontier]

The position statistics come from `tnh.data.edges.EdgeDB` by table name: `--table`, else
$TNH_WEB_TABLE, else `full` when `corpus/edges.parquet`, `pos_index.parquet` and
`parents.parquet` exist, else `dev`. The table is opened once, when the server starts;
restart it after the tables are rebuilt. The page shows which table it reads and its
game count.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

import uvicorn

from tnh.paths import DATA_ROOT
from tnh.web.app import create_app
from tnh.web.models import Hub, NoModels
from tnh.web.repertoire import Store
from tnh.web.stats import open_stats

HOST = "127.0.0.1"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tnh.web", description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--table", default=None, help="EdgeDB table name: full or dev (default: full if it exists)")
    ap.add_argument("--move-model", default="combo-p0.5",
                    help="model 1: combo-p0.5, combo, fake, or import:module:Name (any MoveModel)")
    ap.add_argument("--value-model", default="frontier",
                    help="model 2: frontier, fake, or import:module:Name (any object with .wdl(queries))")
    ap.add_argument("--no-models", action="store_true", help="the database alone: no models and no engine")
    ap.add_argument("--repertoires", type=Path, default=DATA_ROOT / "repertoires",
                    help="directory of the repertoire builder's runs (tnh.builder.export), shown as a layer")
    args = ap.parse_args(argv)

    with socket.socket() as s:
        # As uvicorn binds: without SO_REUSEADDR a connection of the server just stopped, still in
        # TIME-WAIT for a minute, made this check refuse a restart on a port that was free.
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((HOST, args.port))
        except OSError as e:
            raise SystemExit(f"port {args.port} on {HOST} is taken ({e}); pass --port") from e

    t = time.perf_counter()
    stats = open_stats(args.table)
    info = stats.info()
    print(f"table {info['table']} ({info['why']}): {info['games']} games, years {info['year_min']}-{info['year_max']}, "
          f"opened in {time.perf_counter() - t:.2f}s", file=sys.stderr)
    hub = NoModels() if args.no_models else Hub(args.move_model, args.value_model)
    app = create_app(stats, hub, runs=Store(args.repertoires))
    print(f"tn-hittare: http://{HOST}:{args.port}/", file=sys.stderr)
    try:
        uvicorn.run(app, host=HOST, port=args.port, log_level="warning")
    finally:
        hub.close()


if __name__ == "__main__":
    main()
