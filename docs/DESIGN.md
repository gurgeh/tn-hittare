# Design

The goal, the reasoning and the full plan are in the vault note
`~/Documents/hugin/memory/digitalt/tn-hittare.md` (Swedish). Read it first.
This file is the contract between the modules: what each one produces and in
which shape. Change it in the same commit as the code that changes the shape.

## What is being built now

Steps 0 and 1 of the plan. Nothing is trained yet.

- **Step 0, the corpus.** Classical over-the-board games with ratings, split
  by tournament, with clock times where they exist.
- **Step 1, measurement.** Log loss of existing models on the played moves of
  that corpus, and log loss of their value heads on the game results.

## Data layout

Everything large lives under `tnh.paths.DATA_ROOT` (`~/out/tn-hittare`,
override with `TNH_DATA`). It can all be regenerated from the sources.

```
raw/<source>/...                  downloads, exactly as fetched
corpus/games.parquet
corpus/positions.parquet
features/<model>/<variant>/*.parquet
reports/*.md
```

## `corpus/games.parquet`

One row per game, every tempo, every rating. Filtering happens later.

| column | type | meaning |
|---|---|---|
| `game_id` | str | 16 hex chars of sha1 over source, players, date and moves. Stable across rebuilds |
| `source` | str | `lichess_broadcast`, `twic`, ... |
| `tournament_id` | str | The unit the split is made on. One id per event, not per round |
| `event`, `site`, `round` | str | PGN tags, verbatim |
| `date` | date, nullable | Null when the PGN date is partial |
| `year` | int, nullable | |
| `white`, `black` | str | |
| `white_elo`, `black_elo` | int, nullable | As given in the PGN |
| `white_fide_id`, `black_fide_id` | int, nullable | |
| `white_title`, `black_title` | str, nullable | |
| `result` | str | `1-0`, `0-1`, `1/2-1/2`. Games with any other result are dropped |
| `time_control_raw` | str, nullable | The `TimeControl` tag, verbatim |
| `tempo` | str | `classical`, `rapid`, `blitz`, `unknown` |
| `tempo_source` | str | How `tempo` was decided: `tag`, `clock`, `event_name`, `none` |
| `base_s`, `inc_s` | int, nullable | Base time and increment of the first time-control period |
| `has_clock` | bool | True when at least 80 % of the moves carry `%clk` |
| `n_plies` | int | |
| `moves_uci` | str | The whole game, space-separated UCI |
| `clocks_s` | list[int], nullable | Seconds left on the mover's clock after each ply. Null entries where missing |
| `split` | str | `train`, `val`, `test`. A hash of `tournament_id`, 80/10/10 |

Tempo follows FIDE: 60 minutes or more per player for a 60-move game
(`base_s + 60 * inc_s >= 3600`) is classical, more than 10 minutes is rapid,
the rest is blitz.

The same game arriving from two sources is stored once. The copy with clock
times wins.

## `corpus/positions.parquet`

One row per move to be predicted. Built from the games where `tempo` is
`classical` and both ratings are known, for plies below `max_ply` (default 80,
which is move 40).

| column | type | meaning |
|---|---|---|
| `game_id` | str | |
| `ply` | int | 0 for White's first move |
| `fen` | str | The position before the move, full FEN |
| `played_uci` | str | The move that was played |
| `white_to_move` | bool | |
| `mover_elo`, `opp_elo` | int | |
| `clock_mover_s`, `clock_opp_s` | int, nullable | Before the move |
| `move_time_s` | int, nullable | Time spent on this move. Never a model input |
| `pos_key` | str | Board, side to move, castling rights and en passant square. No move counters |
| `pos_count` | int | Number of games in `games.parquet` with both ratings at least 1800 that reach `pos_key`, any tempo |
| `n_legal` | int | |
| `result_mover` | float | 1, 0.5 or 0 from the mover's side |
| `split` | str | Copied from the game |

History is not stored per row. A model that needs earlier positions replays
`moves_uci` from the game up to `ply`.

## The window

Which rows count is decided at evaluation time, not when the corpus is built,
so that the thresholds can be changed without a rebuild. `tnh.data.window`
holds one function:

```python
def in_window(positions, *, book_count=5, max_ply=60, min_clock_s=600) -> polars.Expr
```

- **Out of book.** A game is out of book from the first ply where
  `pos_count < book_count`. The row at that ply is the first one that counts.
  The move that led there is excluded, since it is rare by construction.
- **Not too late.** `ply < max_ply`.
- **Not in time trouble.** Where the clock is known, `clock_mover_s >=
  min_clock_s`. Rows without a clock rely on `max_ply` alone.

## Model interface

`tnh.models.base`:

```python
@dataclass
class Query:
    fen: str
    moves_uci: list[str]      # the game so far, from the start position, up to this position
    mover_elo: int
    opp_elo: int

@dataclass
class Prediction:
    moves: list[str]          # every legal move, UCI
    probs: np.ndarray         # float32, same order, sums to 1
    wdl: tuple[float, float, float] | None   # win, draw, loss for the mover

class MoveModel(Protocol):
    name: str
    def predict(self, queries: list[Query]) -> list[Prediction]: ...
```

A model's own Elo input is a parameter of the model object, separate from
`mover_elo` in the query, so that the Elo fed to the network can be swept
while the player's real rating stays in the data.

## Feature files

`features/<model>/<variant>/part-*.parquet`, one row per `(game_id, ply)`:

| column | type |
|---|---|
| `game_id`, `ply` | as in positions |
| `moves` | list[str], every legal move |
| `logp` | list[float32], log probability per move, same order |
| `wdl` | list[float32] of length 3, nullable |

Engine features use the same shape with `value` (expected score for the mover
after each move) in place of `logp`.

A feature run is resumable: it skips `(game_id, ply)` pairs that are already
on disk.

## Evaluation (step 1, not started)

- Log loss of the played move, with the model mixed with a uniform
  distribution over the legal moves, `p' = (1 - eps) * p + eps / n`. `eps` is
  fitted. This keeps the score proper and stops single impossible moves from
  dominating.
- Maia-3's Elo input swept from 1800 to 2600.
- A conditional logit over the legal moves with model log probabilities and
  engine values as features.
- Log loss of the value head on `result_mover`.
- Fit on `train`, report on `test`. The split is by tournament.
