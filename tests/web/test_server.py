"""`python -m tnh.web` starts, serves the page and its static files, and listens on 127.0.0.1 only."""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from _webfix import write_tables
from tnh.paths import DATA_ROOT

STATIC = Path(__file__).parents[2] / "src" / "tnh" / "web" / "static"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def listening(port: int) -> list[str]:
    """Local addresses of the sockets listening on `port`, from /proc/net/tcp{,6}."""
    out = []
    for name in ("tcp", "tcp6"):
        p = Path("/proc/net") / name
        if not p.exists():
            continue
        for line in p.read_text().splitlines()[1:]:
            f = line.split()
            addr, st = f[1], f[3]
            host, prt = addr.rsplit(":", 1)
            if st == "0A" and int(prt, 16) == port:
                out.append(host)
    return out


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """The server on the fixture tables written as `corpus/*_dev.parquet` under a temporary TNH_DATA."""
    data = tmp_path_factory.mktemp("web")
    write_tables(data / "corpus", "dev")
    (data / "weights").symlink_to(DATA_ROOT / "weights")   # Maia's weights live under TNH_DATA
    port = free_port()
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TNH_DATA": str(data)}
    env.pop("TNH_WEB_TABLE", None)
    proc = subprocess.Popen(
        [sys.executable, "-m", "tnh.web", "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 60
    while True:
        if proc.poll() is not None:
            pytest.fail(f"server exited with {proc.returncode}: {proc.stdout.read()}")
        try:
            httpx.get(base + "/api/info", timeout=1)
            break
        except httpx.TransportError:
            if time.monotonic() > deadline:
                proc.kill()
                pytest.fail("server did not answer within 60 s")
            time.sleep(0.2)
    yield base, port
    proc.terminate()
    proc.wait(timeout=10)


def test_serves_page_and_static(server):
    base, _ = server
    r = httpx.get(base + "/")
    assert r.status_code == 200 and '<div id="board">' in r.text
    for src in re.findall(r'(?:href|src)="(/static/[^"]+)"', r.text):
        assert httpx.get(base + src).status_code == 200, src
    for path in ["app.js", "tree.js", "style.css", "markers.svg", "vendor/chessnut/pieces.svg",
                 "vendor/chess.js/chess.js", "vendor/cm-chessboard/src/Chessboard.js"]:
        r = httpx.get(f"{base}/static/{path}")
        assert r.status_code == 200, path
    assert "javascript" in httpx.get(base + "/static/app.js").headers["content-type"]


def test_every_module_import_resolves(server):
    """Each relative ES-module import in the static tree points at a file that is served."""
    base, _ = server
    for js in (p for p in STATIC.rglob("*.js") if p.is_file()):
        for target in re.findall(r'^\s*import\s[^"\']*["\'](\.[^"\']+)["\']', js.read_text(), re.M):
            rel = (js.parent / target).resolve().relative_to(STATIC)
            assert httpx.get(f"{base}/static/{rel}").status_code == 200, f"{js.name} -> {target}"


def test_api_through_the_server(server):
    base, _ = server
    d = httpx.get(base + "/api/stats", params={"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}).json()
    assert d["total"]["played"] == 8
    info = httpx.get(base + "/api/info").json()["stats"]
    assert (info["table"], info["games"], info["year_min"], info["year_max"]) == ("dev", 8, 2019, 2024)
    assert info["why"] == "no full table yet"


def test_binds_to_localhost_only(server):
    _, port = server
    addrs = listening(port)
    assert addrs, "no listening socket found in /proc/net/tcp"
    assert set(addrs) == {"0100007F"}, addrs   # 127.0.0.1, little-endian hex


def test_refuses_a_taken_port(server):
    _, port = server
    r = subprocess.run([sys.executable, "-m", "tnh.web", "--port", str(port)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "taken" in r.stderr


def test_table_choice(tmp_path, monkeypatch):
    from tnh.web.stats import choose_table, open_stats
    monkeypatch.delenv("TNH_WEB_TABLE", raising=False)
    write_tables(tmp_path, "dev")
    assert choose_table(None, tmp_path) == ("dev", "no full table yet")
    write_tables(tmp_path, "full")
    assert choose_table(None, tmp_path) == ("full", "its files exist")
    assert choose_table("dev", tmp_path) == ("dev", "--table")
    monkeypatch.setenv("TNH_WEB_TABLE", "dev")
    assert choose_table(None, tmp_path)[0] == "dev"
    assert open_stats(None, tmp_path).info()["table"] == "dev"
    with pytest.raises(SystemExit, match="missing"):
        open_stats("nosuch", tmp_path)


def test_counts_of_positions_that_straddle_row_groups(tmp_path):
    """The page's move counts equal a plain filter of the edges table, for every position,
    including those whose rows the four-row groups split (EdgeDB read only the last
    group of such a position before 0373543)."""
    import chess
    import polars as pl
    from _webfix import tables
    from tnh.web.stats import Filters, open_stats, report

    write_tables(tmp_path, "dev")
    edges, index, _ = tables()
    edges = edges.with_row_index("i")
    spans = edges.group_by("pos_hash").agg((pl.col("i").min() // 4).alias("g0"), (pl.col("i").max() // 4).alias("g1"))
    straddling = set(spans.filter(pl.col("g0") != pl.col("g1"))["pos_hash"].to_list())
    assert straddling, "the fixture should have positions split across row groups"

    st = open_stats("dev", tmp_path)
    for h, epd in index.select("pos_hash", "epd").iter_rows():
        board = chess.Board()
        board.set_epd(epd)
        want = dict(edges.filter(pl.col("pos_hash") == h).group_by("move").agg(pl.col("n").sum()).iter_rows())
        got = {m["uci"]: m["n"] for m in report(st, board, Filters())["moves"]}
        assert got == want, (epd, h in straddling)
