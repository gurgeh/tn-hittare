# tn-hittare

A model of how humans play classical chess, and a repertoire builder on top of
it. The owner is a 2200 FIDE player and a senior programmer.

- Goal, reasoning and plan: `~/Documents/hugin/memory/digitalt/tn-hittare.md`
  (Swedish). Read it before anything else.
- The contract between modules: [`docs/DESIGN.md`](docs/DESIGN.md).

## Working here

- Python 3.12 through `uv`. Run everything as `uv run ...`. Tests: `uv run pytest`.
- Code in `src/tnh/`, tests in `tests/`. Tests that need network, a GPU or a
  built corpus are marked `slow` and are skipped by default.
- Data never goes in the repository. It goes under `~/out/tn-hittare/` (see
  `src/tnh/paths.py`). Never write to `~/Downloads` or to the vault.
- Disk is limited. Keep everything under `~/out/tn-hittare/` below 20 GB, and
  say in your report how much you used.
- The GPU is an RTX 4070 Laptop with 8 GB. Other programs use about 1 GB of it.
- Nothing may open a window on the real display. No GUI programs.
- No `sudo` and no `pkexec`. If a system package is missing, stop and say so.
- Fetch politely: one request at a time, a pause between requests, a generic
  `User-Agent` with no personal details, and back off for a minute on HTTP 429.
- Errors are shown, not hidden. No blanket `try/except`, no silent retries, no
  fallbacks that make a failure look like success. Count what is dropped and
  why, and print the counts.
- Commit small, with a message that says what and why. Add only the paths you
  own (`git add <path>`, never `git add -A`). Another agent may be committing
  in the same tree at the same time.
- Do not edit `docs/DESIGN.md` on your own. If the contract has to change,
  say what and why in your report.

## Reports

When you finish, report what is verified and how, what is not verified, every
number with the command that produced it, and what you would do next. A result
that surprised you goes first.
