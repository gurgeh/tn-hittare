"""The repertoire builder: which positions to learn, and which move in each.

- `game`: the graph, the parameters (blend, pessimism, soundness, cost) and the exact value.
- `search`: the knapsack DP, its repair on transpositions, local search, brute force.
- `expand`: growing the graph lazily against a `MoveModel` and a `ValueModel`, with counted calls.
- `counts`: where the opponent's database counts come from.
- `output`: table, PGN and curve.
- `synth`: synthetic games for tests.

Command line: `python -m tnh.builder --help`.
"""
