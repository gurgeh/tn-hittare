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
- **Every job that reads the corpus or runs longer than a minute goes through
  `scripts/tnh-run NAME MEM_GB CMD...`**, which runs it as its own systemd
  service with a hard memory cap. The machine has 30 GB and is the owner's
  laptop. On 2026-11-11 a sampler reached 25 GB inside the terminal's cgroup
  and systemd-oomd killed the whole terminal tab, the agents included. Under
  `tnh-run` a job that passes its cap is stopped alone, with exit code 99 and
  a line in its log. A stop at the cap is a reason to stop and think, not to
  run again: find where the memory goes by instrumenting a run that fits.
  Rules of thumb: give a job the cap you expect it to need plus a margin, at
  most 12 GB for one job, and never let the caps of the jobs running at the
  same time add up to more than 20 GB. Measure before you scale: run on a
  tenth of the data first and read the "memory peak" line that systemd prints
  (`journalctl --user -u tnh-NAME`).
- **Engine processes: never more than the machine's 8 physical cores, counted
  over everything running at the same time.** Hyperthreads add nothing to a
  chess engine; the benchmark here showed the same throughput at 16 processes
  as at 8. Leave cores for the other jobs: 6 Stockfish processes when
  something else CPU-bound runs beside it. The same limit holds for worker
  pools of any kind.
- `positions.parquet` has 212 million rows. Never collect it, or a join
  against it, whole. Scan it lazily, select the columns you need, filter
  before you join, and work through it in batches of games.
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
