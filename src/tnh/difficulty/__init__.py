"""Step 2: do difficulty measures from a model of human play explain real results
beyond the engine evaluation and the rating difference?

- `positions`: the position set from the edge tables (games, real score, ratings,
  the most played path, opening family, how the line thins out).
- `compute`: model calls (Stockfish root and per-move values, Maia-3 on CPU),
  cached on the position hash.
- `measures`: the difficulty measures from the cached calls.
- `analysis`: baseline, measures added one at a time, held-out families,
  bootstrap over families.
- `synth`: a planted world where the answer is known.
- `report`: reports/step2.md from the result files.

Run as `python -m tnh.difficulty <command>`.
"""
