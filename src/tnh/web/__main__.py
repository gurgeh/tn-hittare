"""Serve the interface on 127.0.0.1.

    uv run python -m tnh.web [--port 8765] [--table full|dev] [--maia-size 5m]

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

import uvicorn

from tnh.web.app import create_app
from tnh.web.models import Hub, NoModels
from tnh.web.stats import open_stats

HOST = "127.0.0.1"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tnh.web", description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--table", default=None, help="EdgeDB table name: full or dev (default: full if it exists)")
    ap.add_argument("--maia-size", default="5m", choices=["5m", "23m", "79m"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--no-models", action="store_true", help="no Maia and no engine (for a busy machine)")
    args = ap.parse_args(argv)

    with socket.socket() as s:
        try:
            s.bind((HOST, args.port))
        except OSError as e:
            raise SystemExit(f"port {args.port} on {HOST} is taken ({e}); pass --port") from e

    t = time.perf_counter()
    stats = open_stats(args.table)
    info = stats.info()
    print(f"table {info['table']} ({info['why']}): {info['games']} games, years {info['year_min']}-{info['year_max']}, "
          f"opened in {time.perf_counter() - t:.2f}s", file=sys.stderr)
    hub = NoModels() if args.no_models else Hub(maia_size=args.maia_size, device=args.device, torch_threads=args.torch_threads)
    app = create_app(stats, hub)
    print(f"tn-hittare: http://{HOST}:{args.port}/", file=sys.stderr)
    try:
        uvicorn.run(app, host=HOST, port=args.port, log_level="warning")
    finally:
        hub.close()


if __name__ == "__main__":
    main()
