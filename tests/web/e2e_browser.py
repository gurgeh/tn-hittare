"""End-to-end check of the page in a headless Chromium, driven by Playwright.

Run by `test_e2e.py` with the system Python (Playwright is not a project dependency) and
without DISPLAY or WAYLAND_DISPLAY, so no window can open:

    env -u DISPLAY -u WAYLAND_DISPLAY python3 tests/web/e2e_browser.py http://127.0.0.1:PORT/ [SHOT_DIR]

Expects the server to serve the eight games of `_webfix.GAMES` with the fake models
(`--move-model fake --value-model fake`). Exits non-zero on the first failed check.
"""

from __future__ import annotations

import re
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
    page.wait_for_function("window.tnh && window.tnh.state.stats && window.tnh.state.model")


def square_center(page, sq: str) -> tuple[float, float]:
    box = page.locator(f"#board rect[data-square='{sq}']").bounding_box()
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


COMMENTS_JS = """() => {
    const out = {}
    const walk = (n, path) => {
        if (n.comment) out[path.join('.')] = n.comment
        if (n.pre) out['pre:' + path.join('.')] = n.pre
        n.children.forEach(c => walk(c, [...path, c.uci]))
    }
    walk(window.tnh.state.tree.root, [])
    return out
}"""


def norm(text: str) -> str:
    return " ".join(text.split())


def comments(page) -> dict[str, str]:
    """The tree's comments by the UCI path of their move ("" = before the first move)."""
    return page.evaluate(COMMENTS_JS)


def main_move(page, label: str):
    """A move of the main line in the move list, by its text."""
    return page.locator("#moves > .mv").filter(has_text=re.compile("^" + re.escape(label) + "$"))


def var_move(page, label: str):
    return page.locator("#moves .var > .mv").filter(has_text=re.compile("^" + re.escape(label) + "$"))


def write_comment(page, text: str) -> None:
    """c, type, Enter: the comment of the move shown gets `text` appended."""
    page.keyboard.press("c")
    expect(page.locator("#comment")).to_be_focused()
    page.keyboard.type(text)
    page.keyboard.press("Enter")
    page.wait_for_function("document.activeElement && document.activeElement.id === 'board'")


def check_comments(page) -> None:
    """Comments: set, edit, cancel, remove, in the main line and a variation; the PGN; paste
    round trip; del and promote; typing in the field fires no shortcut."""
    page.locator("#paste").fill("1. e4 e5 (1... c5 2. Nf3) 2. Nf3 Nc6 *")
    page.locator("#paste").press("Enter")
    page.wait_for_function("location.hash.includes('m=e2e4.e7e5.g1f3.b8c6')")
    expect(page.locator("#moves .cmt")).to_have_count(0)

    # set, then edit (c puts the caret at the end), on a main-line move
    main_move(page, "1. e4").click()
    page.wait_for_function("location.hash.includes('m=e2e4&')")
    expect(page.locator("#comment")).to_have_value("")
    write_comment(page, "king pawn")
    expect(page.locator("#moves .cmt")).to_have_text(["{king pawn}"])
    write_comment(page, " opening")
    assert comments(page) == {"e2e4": "king pawn opening"}, comments(page)
    expect(page.locator("#comment")).to_have_value("king pawn opening")
    # Escape cancels
    page.keyboard.press("c")
    page.locator("#comment").fill("thrown away")
    page.keyboard.press("Escape")
    page.wait_for_function("document.activeElement && document.activeElement.id === 'board'")
    assert comments(page) == {"e2e4": "king pawn opening"}, comments(page)
    expect(page.locator("#comment")).to_have_value("king pawn opening")

    # typing in the field fires none of the page's shortcuts
    hash0 = page.evaluate("location.hash")
    page.keyboard.press("c")
    page.keyboard.type("fea p/jkc")
    for k in ("ArrowLeft", "ArrowRight", "Home", "End", "ArrowDown", "Delete"):
        page.keyboard.press(k)
    expect(page.locator("#comment")).to_be_focused()
    expect(page.locator("#comment")).to_have_value("king pawn openingfea p/jkc")
    st = page.evaluate("({o: window.tnh.state.orientation, e: window.tnh.state.eng.name, a: window.tnh.state.showAll, "
                       "h: location.hash, c: window.tnh.state.cursor, t: window.tnh.state.tree.current.uci})")
    assert st == {"o": "w", "e": "off", "a": False, "h": hash0, "c": None, "t": "e2e4"}, st
    page.keyboard.press("Escape")
    assert comments(page) == {"e2e4": "king pawn opening"}, comments(page)

    # a comment in the variation, with a closing brace
    var_move(page, "1... c5").click()
    page.wait_for_function("location.hash.includes('m=e2e4.c7c5&')")
    write_comment(page, "Sicilian } brace")
    expect(page.locator("#moves .var .cmt")).to_have_text(["{Sicilian } brace}"])
    # text typed and left by a click on another move is saved, and the click still lands
    page.keyboard.press("c")
    page.keyboard.type(" defence")
    main_move(page, "2. Nf3").click()
    page.wait_for_function("location.hash.includes('m=e2e4.e7e5.g1f3&')")
    assert comments(page) == {"e2e4": "king pawn opening", "e2e4.c7c5": "Sicilian } brace defence"}, comments(page)
    expect(page.locator("#comment")).to_have_value("")
    # a click on a comment goes to its move
    page.locator("#moves .cmt").first.click()
    page.wait_for_function("location.hash.includes('m=e2e4&')")
    # before the first move
    page.keyboard.press("Home")
    write_comment(page, "start")
    expect(page.locator("#moves > *").first).to_have_text("{start}")
    # a move after a comment gets its number, as in PGN
    expect(main_move(page, "1... e5")).to_have_count(1)

    # the PGN: main line and variation, the brace replaced; the copy button gives the same text
    pgn = page.evaluate("window.tnh.pgn()")
    assert "{ start } 1. e4 { king pawn opening } 1... e5 ( 1... c5 { Sicilian ) brace defence } 2. Nf3 ) 2. Nf3 Nc6 *" \
        in norm(pgn), pgn
    page.locator("#copypgn").click()
    expect(page.locator("#status")).to_contain_text("PGN copied")
    assert page.evaluate("navigator.clipboard.readText()") == pgn

    # paste round trip, with NAGs and [%...] commands in comments, and a comment opening a variation
    extra = pgn.replace("Nc6 *", "Nc6 { $1 [%clk 0:01:00] [%cal Ge2e4] } ( { opens } 2... d6 { Philidor } ) *")
    page.locator("#paste").fill(extra)
    page.locator("#paste").press("Enter")
    page.wait_for_function("location.hash.includes('m=e2e4.e7e5.g1f3.b8c6&')")
    want = {"": "start", "e2e4": "king pawn opening", "e2e4.c7c5": "Sicilian ) brace defence",
            "e2e4.e7e5.g1f3.b8c6": "$1 [%clk 0:01:00] [%cal Ge2e4]", "pre:e2e4.e7e5.g1f3.d7d6": "opens",
            "e2e4.e7e5.g1f3.d7d6": "Philidor"}
    assert comments(page) == want, comments(page)
    expect(page.locator("#moves .var .cmt.pre")).to_have_text(["{opens}"])
    again = page.evaluate("window.tnh.pgn()")
    assert norm(again) == norm(extra), (again, extra)     # the same PGN, up to line breaks
    if SHOTS:
        page.locator("#paste").fill(again)        # the PGN shown beside the move list it came from
        page.evaluate("document.getElementById('paste').rows = 9")
        page.screenshot(path=f"{SHOTS}/e2e_comments.png")
        page.locator("#paste").fill("")
        page.evaluate("document.getElementById('paste').rows = 2")

    # reload: the tree and its comments come back, at the URL's position
    page.reload()
    settle(page)
    assert comments(page) == want, comments(page)
    assert page.evaluate("window.tnh.state.tree.current.uci") == "b8c6"
    expect(page.locator("#comment")).to_have_value("$1 [%clk 0:01:00] [%cal Ge2e4]")
    assert norm(page.evaluate("window.tnh.pgn()")) == norm(extra)

    # remove: an empty comment
    main_move(page, "1. e4").click()
    page.keyboard.press("c")
    page.locator("#comment").fill("")
    page.keyboard.press("Enter")
    assert "e2e4" not in comments(page)
    expect(page.locator("#moves .cmt", has_text="king pawn")).to_have_count(0)
    page.keyboard.press("c")                  # put it back for what follows
    page.keyboard.type("king pawn opening")
    page.keyboard.press("Enter")

    # promote and del keep the comments of the moves that remain
    var_move(page, "1... c5").click()
    page.keyboard.press("p")
    expect(main_move(page, "1... c5")).to_have_count(1)
    assert comments(page) == want, comments(page)
    assert "1... c5 { Sicilian ) brace defence } ( 1... e5" in norm(page.evaluate("window.tnh.pgn()"))
    page.keyboard.press("Delete")             # c5 and its Nf3 go; the rest stay with their comments
    page.wait_for_function("location.hash.includes('m=e2e4&')")
    gone = {k: v for k, v in want.items() if "c7c5" not in k}
    assert comments(page) == gone, comments(page)
    page.reload()
    settle(page)
    assert comments(page) == gone, comments(page)


class Dialogs:
    """Answers the page's confirm() dialogs from a queue and keeps what they asked."""

    def __init__(self, page):
        self.answers: list[bool] = []
        self.asked: list[str] = []
        page.on("dialog", self.on)

    def on(self, d) -> None:
        self.asked.append(d.message)
        (d.accept if self.answers and self.answers.pop(0) else d.dismiss)()

    def wait(self, page, n: int) -> None:
        for _ in range(100):
            if len(self.asked) >= n:
                return
            page.wait_for_timeout(50)
        raise AssertionError(f"expected {n} dialogs, saw {self.asked}")


def saved_lines(page) -> list[dict]:
    return page.evaluate("fetch('/api/lines').then(r => r.json()).then(d => d.lines)")


def check_lines(page, dialogs: Dialogs) -> None:
    """Saved lines: save, reload, overwrite prompt, load with the unsaved-changes prompt, delete."""
    state = page.locator("#linestate")
    expect(state).to_have_text("not saved as a line")
    expect(page.locator("#linelist option")).to_have_text(["no saved lines"])
    before = comments(page)
    page.locator("#linename").fill("Italian: main")
    page.locator("#linesave").click()
    expect(page.locator("#status")).to_contain_text('saved "Italian: main" as italian-main.pgn')
    expect(state).to_have_text('line "Italian: main": saved')
    [line] = saved_lines(page)
    assert line["name"] == "Italian: main" and line["preview"].startswith("1. e4 e5 2. Nf3 Nc6"), line
    expect(page.locator("#linelist")).to_have_value("Italian: main")

    # a change makes it unsaved; undoing the change makes it saved again; a reload keeps both
    page.keyboard.press("End")
    page.wait_for_function("window.tnh.state.tree.current.uci === 'b8c6'")
    page.evaluate("window.tnh.play('f1c4')")
    expect(state).to_have_text('line "Italian: main": unsaved changes')
    page.keyboard.press("Delete")
    expect(state).to_have_text('line "Italian: main": saved')
    page.evaluate("window.tnh.play('f1c4')")
    page.reload()
    settle(page)
    expect(state).to_have_text('line "Italian: main": unsaved changes')
    expect(page.locator("#linename")).to_have_value("Italian: main")
    assert comments(page) == before, comments(page)

    # saving over the name asks first: no, then yes
    dialogs.answers[:] = [False]
    page.locator("#linesave").click()
    expect(page.locator("#status")).to_contain_text('not saved: "Italian: main" exists')
    assert "Overwrite" in dialogs.asked[-1], dialogs.asked
    expect(state).to_have_text('line "Italian: main": unsaved changes')
    dialogs.answers[:] = [True]
    page.locator("#linesave").click()
    expect(state).to_have_text('line "Italian: main": saved')
    assert len(dialogs.asked) == 2
    [line] = saved_lines(page)
    assert line["preview"].startswith("1. e4 e5 2. Nf3 Nc6 3. Bc4"), line

    # a pasted PGN is a new, unsaved tree; save it under another name
    page.locator("#paste").fill("1. d4 d5 2. c4 {the gambit} *")
    page.locator("#paste").press("Enter")
    expect(state).to_have_text("not saved as a line")
    expect(page.locator("#linename")).to_have_value("")
    page.locator("#linename").fill("QGD")
    page.locator("#linename").press("Enter")
    expect(state).to_have_text('line "QGD": saved')
    assert [x["name"] for x in saved_lines(page)] == ["Italian: main", "QGD"]

    # loading another line over unsaved changes asks first: no keeps the tree, yes replaces it
    page.evaluate("window.tnh.play('e7e6')")
    page.locator("#linelist").select_option("Italian: main")
    dialogs.answers[:] = [False]
    n = len(dialogs.asked)
    page.locator("#lineload").click()
    dialogs.wait(page, n + 1)
    expect(state).to_have_text('line "QGD": unsaved changes')
    assert "unsaved changes" in dialogs.asked[-1] and "Italian: main" in dialogs.asked[-1], dialogs.asked
    assert page.evaluate("window.tnh.state.tree.current.uci") == "e7e6"
    dialogs.answers[:] = [True]
    page.locator("#lineload").click()
    expect(state).to_have_text('line "Italian: main": saved')
    expect(page.locator("#linename")).to_have_value("Italian: main")
    assert comments(page) == before, comments(page)
    assert page.evaluate("window.tnh.state.tree.current === window.tnh.state.tree.root")
    # the board follows (its animations run one after another)
    page.wait_for_function("document.querySelectorAll('#board g[data-square]').length === 32 && "
                           "document.querySelector(\"#board g[data-square='e2']\") && "
                           "!document.querySelector(\"#board g[data-square='d4']\")")
    if SHOTS:
        page.screenshot(path=f"{SHOTS}/e2e_lines.png")
    # loading a line with no unsaved changes asks nothing
    n = len(dialogs.asked)
    page.locator("#lineload").click()
    expect(page.locator("#status")).to_contain_text('loaded "Italian: main"')
    assert len(dialogs.asked) == n

    # delete asks first
    page.locator("#linelist").select_option("QGD")
    dialogs.answers[:] = [False]
    n = len(dialogs.asked)
    page.locator("#linedel").click()
    dialogs.wait(page, n + 1)
    assert "Delete" in dialogs.asked[-1]
    assert len(saved_lines(page)) == 2
    dialogs.answers[:] = [True]
    page.locator("#linedel").click()
    expect(page.locator("#status")).to_contain_text('deleted "QGD"')
    assert [x["name"] for x in saved_lines(page)] == ["Italian: main"]
    expect(page.locator("#linelist option")).to_have_count(2)

    # a reload keeps the loaded line, saved
    page.reload()
    settle(page)
    expect(state).to_have_text('line "Italian: main": saved')
    expect(page.locator("#linelist")).to_have_value("Italian: main")


def main() -> None:
    errors: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1366, "height": 768},
                                  permissions=["clipboard-read", "clipboard-write"])
        page = ctx.new_page()
        dialogs = Dialogs(page)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

        page.goto(BASE + "#t=all")
        settle(page)
        assert counts(page) == {"e2e4": 4, "d2d4": 2, "c2c4": 2}, counts(page)
        # the model columns are filled: no cell still waits for the models
        expect(page.locator("#tbl td.pending")).to_have_count(0)
        expect(page.locator("#models")).to_contain_text("fake-move")
        expect(page.locator("#tbl thead")).to_contain_text("W-Elo")
        # the other form of the rating columns, one switch away
        page.evaluate("window.tnh.CHOICES.elo = 'chooser'; window.tnh.render()")
        expect(page.locator("#tbl thead")).to_contain_text("Δ gap")
        page.evaluate("window.tnh.CHOICES.elo = 'colours'; window.tnh.render()")
        # the ratings go to the models and the URL
        page.locator("#white").fill("2400")
        page.locator("#white").press("Enter")
        page.locator("#white").blur()
        page.wait_for_function("location.hash.includes('w=2400') && window.tnh.state.model && window.tnh.state.model.white === 2400")

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
        page.wait_for_function("!location.hash.includes('t=')")   # classical is the default
        settle(page)
        assert counts(page) == {"e2e4": 2, "d2d4": 1, "c2c4": 2}, counts(page)
        page.locator("#tempi input[value='rapid']").check()
        page.locator("#tempi input[value='blitz']").check()
        page.locator("#tempi input[value='unknown']").check()
        settle(page)

        # the URL carries the position: reload lands on it
        page.goto(BASE + "#m=d2d4.g8f6.c2c4&t=all")
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

        check_comments(page)
        check_lines(page, dialogs)

        # the controls that are gone, and the engine section that replaces them
        for gone in ("#nodes", "#run", "#repline", "#replines", "#stubs"):
            expect(page.locator(gone)).to_have_count(0)
        assert page.locator("#eng option").all_inner_texts()[:1] == ["Off"]
        assert len(page.locator("#eng option").all_inner_texts()) == 4
        expect(page.locator("#englines")).to_be_hidden()

        # an engine update must not rebuild the table or replace an engine line's element
        page.goto(BASE + "#t=all")
        page.reload()
        settle(page)
        page.locator("#eng").select_option("preppy")
        page.wait_for_function("window.tnh.state.eng.snap && window.tnh.state.eng.snap.lines.length >= 2", timeout=60000)
        page.wait_for_function("document.querySelector('#tbl thead th[data-k=eng]') !== null")
        page.locator("#tbl tbody tr[data-uci]").first.hover()
        page.evaluate("""() => { window.__row = document.querySelector('#tbl tbody tr[data-uci]');
                                window.__line = document.querySelector('#englines .eline');
                                window.__builds = window.tnh.tableBuilds();
                                window.__polls = 0;
                                const f = window.fetch; window.fetch = (...a) => { if (String(a[0]).includes('/api/uci/state')) window.__polls++; return f(...a) } }""")
        page.wait_for_function("window.__polls >= 6", timeout=20000)      # several engine updates
        assert page.evaluate("document.contains(window.__row) && document.contains(window.__line)")
        assert page.evaluate("window.tnh.tableBuilds()") == page.evaluate("window.__builds")
        assert page.evaluate("window.__row.matches(':hover')")
        assert page.locator("#tbl tbody td.engcell").count() >= 1

        # ... nor rebuild the move list, nor take the focus from the comment field while typing
        page.keyboard.press("Home")
        page.wait_for_function("window.tnh.state.eng.snap && window.tnh.state.eng.snap.lines.length >= 1", timeout=60000)
        page.evaluate("""() => { window.__mvs = [...document.querySelectorAll('#moves *')];
                                window.__builds = window.tnh.tableBuilds(); window.__polls = 0 }""")
        page.keyboard.press("c")
        page.keyboard.type(" while the engine runs", delay=80)       # spans several polls
        page.wait_for_function("window.__polls >= 6", timeout=20000)
        expect(page.locator("#comment")).to_be_focused()
        expect(page.locator("#comment")).to_have_value("start while the engine runs")
        assert page.evaluate("window.__mvs.length > 0 && window.__mvs.every(e => document.contains(e)) && "
                             "document.querySelectorAll('#moves *').length === window.__mvs.length")
        assert page.evaluate("window.tnh.tableBuilds()") == page.evaluate("window.__builds")
        page.keyboard.press("Escape")
        assert comments(page)[""] == "start"

        # preppy's bestmove mode: Mode is a select of its declared values; Price, Cap and Obvious
        # are greyed out; lines show the depth and what is inside the horizon; still no rebuild
        mode = page.locator("#engset select[data-opt='Mode']")
        assert sorted(mode.locator("option").all_inner_texts()) == ["bestmove", "repertoire"]
        expect(page.locator("#engset input[data-opt='Price']")).to_be_enabled()
        expect(page.locator("#engset input[data-opt='Alpha']")).to_be_disabled()          # repertoire mode
        expect(page.locator("#engset [data-group='bestmove']")).to_have_class(re.compile("off"))
        for k in ("Blunder", "PriorMin"):
            expect(page.locator(f"#engset input[data-opt='{k}']")).to_be_enabled()
        # the settings for both modes come first, then the repertoire group, then the bestmove one
        order = page.evaluate("[...document.querySelectorAll('#engset [data-opt]')].map(e => e.dataset.opt)")
        assert order.index("Mode") < order.index("Price") < order.index("Alpha"), order
        assert order.index("Blunder") < order.index("Price") and order.index("PriorMin") < order.index("Price"), order
        mode.select_option("bestmove")
        for k in ("Price", "Cap", "Obvious"):
            expect(page.locator(f"#engset input[data-opt='{k}']")).to_be_disabled()
        expect(page.locator("#engset input[data-opt='Alpha']")).to_be_enabled()
        page.wait_for_function("window.tnh.state.eng.snap && window.tnh.state.eng.snap.root && "
                               "window.tnh.state.eng.snap.root.mode === 'bestmove' && window.tnh.state.eng.snap.lines.length >= 2",
                               timeout=60000)
        expect(page.locator("#engstat")).to_contain_text("expanded")
        expect(page.locator("#englines .eprog").first).to_contain_text("inside")
        expect(page.locator("#engstrings")).not_to_contain_text("non-obvious")
        page.evaluate("() => { window.__builds = window.tnh.tableBuilds(); window.__row = document.querySelector('#tbl tbody tr[data-uci]'); window.__polls = 0 }")
        page.wait_for_function("window.__polls >= 4", timeout=20000)
        assert page.evaluate("document.contains(window.__row) && window.tnh.tableBuilds() === window.__builds")
        mode.select_option("repertoire")
        expect(page.locator("#engset input[data-opt='Price']")).to_be_enabled()
        expect(page.locator("#engset input[data-opt='Alpha']")).to_be_disabled()

        page.locator("#eng").select_option("off")
        page.wait_for_function("document.querySelector('#tbl thead th[data-k=eng]') === null")

        # paste a FEN: unknown position, Maia warns that it has no history
        page.locator("#paste").fill("8/8/8/4k3/8/8/4K3/7R w - - 0 1")
        page.locator("#paste").press("Enter")
        page.wait_for_function("location.hash.includes('fen=')")
        settle(page)
        assert counts(page) == {}, counts(page)
        expect(page.locator("#totals")).to_contain_text("No games.")
        expect(page.locator("#models")).to_contain_text("no move history")
        expect(page.locator("#models")).to_contain_text("fake-move")
        # a loaded FEN replaced the tree and its comments, also after a reload
        expect(page.locator("#moves .cmt")).to_have_count(0)
        page.reload()
        settle(page)
        assert comments(page) == {}, comments(page)
        assert page.evaluate("window.tnh.state.tree.root.fen").startswith("8/8/8/4k3")
        if SHOTS:
            page.screenshot(path=f"{SHOTS}/e2e_fen.png")
        browser.close()
    if errors:
        raise SystemExit("console errors: " + "; ".join(errors))
    print("e2e ok")


if __name__ == "__main__":
    main()
