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

from _webfix import games

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
    path = tmp_path_factory.mktemp("web") / "games.parquet"
    games().write_parquet(path)
    port = free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "tnh.web", "--port", str(port), "--data", f"games:{path}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env={**os.environ, "PYTHONUNBUFFERED": "1"},
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
    assert d["total"]["n"] == 8
    assert httpx.get(base + "/api/info").json()["stats"]["source"].startswith("games: 8 games")


def test_binds_to_localhost_only(server):
    _, port = server
    addrs = listening(port)
    assert addrs, "no listening socket found in /proc/net/tcp"
    assert set(addrs) == {"0100007F"}, addrs   # 127.0.0.1, little-endian hex


def test_refuses_a_taken_port(server):
    _, port = server
    r = subprocess.run([sys.executable, "-m", "tnh.web", "--port", str(port), "--data", "games:/nonexistent"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "taken" in r.stderr
