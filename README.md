# tn-hittare

A model of how humans play classical chess, and a repertoire builder on top of
it: openings that are objectively sound, unusual, and hard to meet over the
board, within a budget of positions to memorise.

The goal and the plan are in `~/Documents/hugin/memory/digitalt/tn-hittare.md`.
The module contract is in [`docs/DESIGN.md`](docs/DESIGN.md).

```bash
uv sync
uv run pytest
```

Data lives under `~/out/tn-hittare/`, never in the repository.
