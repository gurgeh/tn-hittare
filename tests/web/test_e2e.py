"""The page in a headless browser: load, play moves by drag, click and keys, filter, paste.

Marked slow: it needs the system Python's Playwright and its headless Chromium, which are
not project dependencies. The browser runs with DISPLAY and WAYLAND_DISPLAY removed, so it
cannot open a window.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from test_server import server  # noqa: F401  (the fixture)

DRIVER = Path(__file__).with_name("e2e_browser.py")


def _system_playwright() -> str | None:
    py = shutil.which("python3", path="/usr/bin:/bin")
    if py is None:
        return None
    r = subprocess.run([py, "-c", "import playwright.sync_api"], capture_output=True)
    return py if r.returncode == 0 else None


@pytest.mark.slow
def test_page_in_headless_browser(server):  # noqa: F811
    py = _system_playwright()
    if py is None:
        pytest.skip("no Playwright in the system Python")
    base, _ = server
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    # $TNH_E2E_SHOTS: a directory for the driver's screenshots (never the repository)
    shots = [os.environ["TNH_E2E_SHOTS"]] if os.environ.get("TNH_E2E_SHOTS") else []
    r = subprocess.run([py, str(DRIVER), base + "/", *shots], capture_output=True, text=True, timeout=180, env=env,
                       cwd=DRIVER.parent)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "e2e ok" in r.stdout
