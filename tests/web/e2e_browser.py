"""End-to-end check of the page in a headless Chromium, driven by Playwright.

Run by `test_e2e.py` with the system Python (Playwright is not a project dependency) and
without DISPLAY or WAYLAND_DISPLAY, so no window can open:

    env -u DISPLAY -u WAYLAND_DISPLAY python3 tests/web/e2e_browser.py http://127.0.0.1:PORT/ [SHOT_DIR]

Expects the server to serve the eight games of `_webfix.GAMES`. Exits non-zero on the
first failed check.
"""

from __future__ import annotations

import sys

from playwright.sync_api import expect, sync_playwright

BASE = sys.argv[1]
SHOTS = sys.argv[2] if len(sys.argv) > 2 else None


def counts(page) -> dict[str, int]:
    out = {}
    for tr in page.locator("#tbl tbody tr[data-uci]:not(.nodb)").all():
        n = tr.locator("td").nth(1).inner_text().strip()
        if n:
            out[tr.get_attribute("data-uci")] = int(n)
    return out


def settle(page) -> None:
    page.wait_for_function("window.tnh && window.tnh.state.stats && window.tnh.state.maia")


def square_center(page, sq: str) -> tuple[float, float]:
    box = page.locator(f"#board rect[data-square='{sq}']").bounding_box()
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


def main() -> None:
    errors: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

        page.goto(BASE)
        settle(page)
        assert counts(page) == {"e2e4": 4, "d2d4": 2, "c2c4": 2}, counts(page)

        # drag e2-e4 on the board
        page.mouse.move(*square_center(page, "e2"))
        page.mouse.down()
        page.mouse.move(*square_center(page, "e4"), steps=8)
        page.mouse.up()
        page.wait_for_function("location.hash.includes('m=e2e4')")
        settle(page)
        assert counts(page) == {"e7e5": 3, "c7c5": 1}, counts(page)
        expect(page.locator("#moves .mv.cur")).to_have_text("1. e4")

        # click a row to play it, then keys back and forward
        page.locator("#tbl tbody tr[data-uci='e7e5']").click()
        settle(page)
        assert counts(page) == {"g1f3": 2, "f1c4": 1}, counts(page)
        page.keyboard.press("ArrowLeft")
        page.keyboard.press("ArrowLeft")
        settle(page)
        assert counts(page) == {"e2e4": 4, "d2d4": 2, "c2c4": 2}, counts(page)
        page.keyboard.press("End")
        page.wait_for_function("location.hash.includes('m=e2e4.e7e5')")

        # a filter applies to the table
        page.keyboard.press("Home")
        settle(page)
        for t in ("rapid", "blitz", "unknown"):
            page.locator(f"#tempi input[value='{t}']").uncheck()
        page.wait_for_function("location.hash.includes('t=classical')")
        settle(page)
        assert counts(page) == {"e2e4": 2, "d2d4": 1, "c2c4": 2}, counts(page)
        page.locator("#tempi input[value='rapid']").check()
        page.locator("#tempi input[value='blitz']").check()
        page.locator("#tempi input[value='unknown']").check()
        settle(page)

        # the URL carries the position: reload lands on it
        page.goto(BASE + "#m=d2d4.g8f6.c2c4")
        page.reload()
        settle(page)
        assert counts(page) == {"e7e6": 3, "g7g6": 1}, counts(page)
        expect(page.locator("#orders")).to_contain_text("2 different last moves")

        # paste a PGN with a variation
        page.locator("#paste").fill("1. c4 Nf6 (1... e5) 2. d4 *")
        page.locator("#paste").press("Enter")
        page.wait_for_function("location.hash.includes('m=c2c4.g8f6.d2d4')")
        expect(page.locator("#moves .var")).to_contain_text("e5")
        settle(page)
        assert counts(page) == {"e7e6": 3, "g7g6": 1}, counts(page)

        # paste a FEN: unknown position, Maia warns that it has no history
        page.locator("#paste").fill("8/8/8/4k3/8/8/4K3/7R w - - 0 1")
        page.locator("#paste").press("Enter")
        page.wait_for_function("location.hash.includes('fen=')")
        settle(page)
        assert counts(page) == {}, counts(page)
        expect(page.locator("#totals")).to_contain_text("No games.")
        expect(page.locator("#models")).to_contain_text("no move history")
        expect(page.locator("#models")).to_contain_text("2200→2808")
        if SHOTS:
            page.screenshot(path=f"{SHOTS}/e2e_fen.png")
        browser.close()
    if errors:
        raise SystemExit("console errors: " + "; ".join(errors))
    print("e2e ok")


if __name__ == "__main__":
    main()
