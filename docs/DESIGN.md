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
corpus/games_extra.parquet        per-source detail behind the parsing decisions; not an input to anything downstream
corpus/parse_stats.json           what was dropped and why
corpus/positions.parquet
corpus/samples/<name>.parquet     fixed subsets of positions.parquet with extra columns; what features are computed on
corpus/samples/<name>.window.parquet   the same sample, window rows only
corpus/blitz_ref/                 the blitz reference in the corpus schema
raw/lichess_blitz_ref/            the start of one Lichess monthly dump, rated blitz, both players 1800 or more
weights/                          downloaded model weights
scratch/<module>/                 a developer's own experiments; nothing downstream reads it
features/<model>/<variant>/*.parquet
reports/*.md
reports/step1/                    the json and figures behind step1.md
```

## `corpus/games.parquet`

One row per game: standard chess between humans over the board, every tempo,
every rating. Variants, non-standard starting positions, engine games, simuls
and online events are dropped at parse time, as are games without a result,
without moves or with an illegal move. `parse_stats.json` counts each reason.
Filtering on tempo and rating happens later.

| column | type | meaning |
|---|---|---|
| `game_id` | str | 16 hex chars of sha1 over the de-duplication key: both surnames, the year and the moves. The source is not part of it, so the id is the same whichever copy of a game is kept |
| `source` | str | `lichess_broadcast`, `twic`, ... |
| `tournament_id` | str | The unit the split is made on. One id per event, not per round |
| `event`, `site`, `round` | str | PGN tags, verbatim |
| `date` | date, nullable | The `Date` tag, else `UTCDate`. Null when partial |
| `year` | int, nullable | |
| `white`, `black` | str | |
| `white_elo`, `black_elo` | int, nullable | As given in the PGN |
| `white_fide_id`, `black_fide_id` | int, nullable | |
| `white_title`, `black_title` | str, nullable | |
| `result` | str | `1-0`, `0-1`, `1/2-1/2`. Games with any other result are dropped |
| `time_control_raw` | str, nullable | The `TimeControl` tag, verbatim |
| `tempo` | str | `classical`, `rapid`, `blitz`, `unknown` |
| `tempo_source` | str | How `tempo` was decided: `tag`, `clock`, `round`, `event_name`, `assumed`, `none` |
| `base_s`, `inc_s` | int, nullable | Base time and increment of the first time-control period. Estimates when `tempo_source` is `clock` |
| `has_clock` | bool | True when at least 80 % of the moves carry `%clk` |
| `n_plies` | int | |
| `moves_uci` | str | The whole game, space-separated UCI |
| `clocks_s` | list[int], nullable | Seconds left on the mover's clock after each ply. Null entries where missing |
| `split` | str | `train`, `val`, `test`. A hash of `tournament_id`, 80/10/10 |

Tempo follows FIDE: 60 minutes or more per player for a 60-move game
(`base_s + 60 * inc_s >= 3600`) is classical, more than 10 minutes is rapid,
the rest is blitz.

How `tempo` is decided, in order. The first that applies wins.

1. `clock`: the clock readings, where they contradict the tag or the tag is
   missing or unreadable.
2. `tag`: the `TimeControl` tag.
3. `round`: every other game with a known tempo in the same broadcast round
   has the same tempo.
4. `event_name`: the words rapid, blitz and the like in the event name.
5. `assumed`: the source's default. TWIC carries no time control and no
   clocks; a TWIC game with no sign of rapid or blitz is taken as classical.
   Anything measured on classical games must be able to leave these out.
6. `none`: `tempo` is `unknown`.

The same game arriving from two sources is stored once. The copy with clock
times wins, then the copy with more tags filled in. Two copies of one game
put their tournaments in the same split.

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
| `clock_mover_s`, `clock_opp_s` | int, nullable | Before the move. Null before a side's first move: many DGT clocks add the increment before move 1, so the base time is not the reading |
| `move_time_s` | int, nullable | Time spent on this move. Null where it cannot be computed, including where it comes out below -2 s (an unknown bonus at a time control). Never a model input |
| `pos_key` | str | Board, side to move, castling rights and en passant square. No move counters |
| `pos_count` | int | Number of games in `games.parquet` with both ratings at least 1800 that reach `pos_key`, any tempo, any split. 0 when only games with a lower-rated player reach it |
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
def in_window(positions, *, book_count=5, max_ply=60, max_ply_no_clock=40, min_clock_s=600) -> polars.Expr
```

Apply it before any other row filter. The out-of-book point is computed per
game over all of the game's rows.

- **Out of book.** A game is out of book from the first ply where
  `pos_count < book_count`. The row at that ply is the first one that counts.
  The move that led there is excluded, since it is rare by construction.
- **Not too late.** `ply < max_ply`.
- **Not in time trouble.** Where the clock is known, `clock_mover_s >=
  min_clock_s`. Rows without a clock must have `ply < max_ply_no_clock`.

`max_ply_no_clock` is 40 because of what step 0 measured on classical games
between players rated 1800 to 2500: 1.7 % of the moves at plies 30-39 are
made with under ten minutes left, 9.3 % at plies 40-49 and 25.1 % at plies
50-59.

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
    logp: np.ndarray          # float32, same order, log probability over the legal moves
    wdl: tuple[float, float, float] | None   # win, draw, loss for the mover

@dataclass
class MoveValues:
    moves: list[str]          # every legal move, UCI
    value: np.ndarray         # float32, expected score for the mover after the move; NaN = no value within the budget
    cp: np.ndarray            # float32, centipawns for the mover after the move; NaN where the score is a mate or missing
    mate: np.ndarray          # int16, moves to mate, positive when the mover mates; 0 where the score is not a mate
    wdl: tuple[float, float, float] | None   # the position itself, along the best line, for the mover

class MoveModel(Protocol):
    name: str
    def predict(self, queries: list[Query]) -> list[Prediction]: ...

class ValueModel(Protocol):
    name: str
    def evaluate(self, queries: list[Query]) -> list[MoveValues]: ...
```

Log probabilities, not probabilities: float32 probabilities lose the small
ones, and those are the ones that dominate a log loss.

A model's own Elo input is a parameter of the model object, separate from
`mover_elo` in the query, so that the Elo fed to the network can be swept
while the player's real rating stays in the data. Maia-3 takes the two
ratings as continuous values. It was trained on Lichess blitz ratings; the
ratings in the corpus are FIDE ratings, and the difference is what the sweep
measures.

## Feature files

`features/<model>/<variant>/part-*.parquet`, one row per `(game_id, ply)`,
and a `meta.json` with the options the run was made with. A run into a
directory whose `meta.json` has other options is refused.

Move models:

| column | type |
|---|---|
| `game_id`, `ply` | as in positions |
| `moves` | list[str], every legal move |
| `logp` | list[float32], log probability per move, same order |
| `wdl` | list[float32] of length 3, nullable |

Value models: `moves`, `value`, `cp`, `mate` and `wdl` as in `MoveValues`.

A feature run is resumable: it skips `(game_id, ply)` pairs that are already
on disk.

Variant names say what was varied: `offset+300`, `fixed2200-2200`,
`nodes10000`, with suffixes for anything else that is not the default.

## Evaluation (step 1)

`tnh.eval`. Everything is fitted on `train` and reported on `test`. The split
is by tournament. Rows from one game are not independent, so every interval
is a bootstrap over tournaments.

- **The sample.** Features are computed on fixed samples, not on all of
  `positions.parquet`. A sample is drawn after `in_window`, from classical
  games with clocks whose `tempo_source` is not `assumed`, with both ratings
  in a band (inclusive; the top edge belongs to the last bucket). It is
  stratified on the mover's rating in 100-point buckets with equal
  allocation, a short bucket's surplus going to the others. A sample has the
  positions schema plus `tournament_id`, `tempo_source`, `in_window`,
  `out_of_book`, `part`, `bucket` and `weight`. `weight` is population rows
  per sampled row, and every pooled number is weighted with it. A sample also
  holds a smaller set of rows from outside the window, flagged by
  `in_window`, for the clock breakdown and for the window shared with blitz.
- **Log loss of the played move**, with the model mixed with a uniform
  distribution over the legal moves, `p' = (1 - eps) * p + eps / n`. `eps` is
  fitted. This keeps the score proper and stops single impossible moves from
  dominating.
- **The Elo sweep.** Maia-3 with both ratings offset by the same amount, from
  -200 to +800. The first measurement on elite games had its minimum at +300
  and broke down above about 2800 on Maia's scale. The result wanted is the
  best offset as a function of the player's FIDE rating. The mover's and the
  opponent's offset are also varied separately.
- **Breakdowns** of the log loss: by the mover's rating, by ply, and by the
  time left on the clock. The last one shows where play degrades, which is
  what `min_clock_s` should be set from.
- **A conditional logit** over the legal moves with model log probabilities
  and engine values as features.
- **The value head.** Log loss of the predicted win, draw and loss on
  `result_mover`, one position per game. Raw, and after a recalibration
  fitted on `train`: the first measurement gave a mean draw probability of
  0.16 where 43 % of the games were drawn. The recalibration has five
  parameters: a temperature, a bias for win and one for draw, a draw term
  that grows with rating, and the weight of a uniform mixture. An engine's
  win, draw and loss come in permille and are often exactly 0 for an outcome
  that then happens, so its raw log loss is infinite and is reported as such.
- **The reference.** The same log loss for Maia-3 on Lichess blitz games at
  the players' own ratings, from a month after its training data ends
  (July 2025). The gap between that and the classical figure is what a model
  made for classical play could gain.
