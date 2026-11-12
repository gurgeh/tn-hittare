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
corpus/edges.parquet               position statistics, see below
corpus/pos_index.parquet
corpus/book_exit.parquet           per game, the ply at which it leaves book for book_count 2, 5, 20 and 100
corpus/samples/<name>.parquet     fixed subsets of positions.parquet with extra columns; what features are computed on
corpus/samples/<name>.window.parquet   the same sample, window rows only
corpus/blitz_ref/                 the blitz reference in the corpus schema
raw/lichess_blitz_ref/            the start of one Lichess monthly dump, rated blitz, both players 1800 or more
weights/                          downloaded model weights
scratch/<module>/                 a developer's own experiments; nothing downstream reads it
features/<model>/<variant>/*.parquet
features/positions/<model>/<variant>/*.parquet   keyed on the position, not on a game row; see Feature files
reports/*.md
reports/step1/                    the json and figures behind step1.md
reports/step2/                    the json behind step2.md
```

## `corpus/games.parquet`

One row per game: standard chess between humans over the board, every tempo,
every rating. Variants, non-standard starting positions, engine games, simuls
and online events are dropped at parse time, as are games without a result,
without moves or with an illegal move. `parse_stats.json` counts each reason.

A final king move to d4, d5, e4 or e5 is removed, legal or not, at most once
per side: it is the arbiter placing the kings in the centre of a DGT board to
signal the result. The game is kept with its result. An illegal move of any
other kind drops the game.
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
| `tempo_source` | str | How `tempo` was decided: `clock`, `tag`, `event_type`, `round`, `event_name`, `assumed`, `none` |
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
3. `event_type`: a ChessBase `EventType` tag that says rapid or blitz. A
   bare `swiss` or `tourn` says nothing.
4. `round`: every other game with a known tempo in the same broadcast round
   has the same tempo.
5. `event_name`: the words rapid, blitz and the like in the event name.
6. `assumed`: the source's default. TWIC rarely carries a time control and
   never clocks; a TWIC game with no sign of rapid or blitz is taken as
   classical. Anything measured on classical games must be able to leave
   these out.
7. `none`: `tempo` is `unknown`.

Never `assumed`, always `unknown` unless something above decides: a game
whose name says tiebreak or playoff, and in a knockout event any game in
sub-round 3 or later (`3.3`), since those are the rapid and blitz tiebreaks
played under the event's own name.

Online events are dropped. In TWIC they are marked by a `Site` ending in
` INT`.

The same game arriving from two sources is stored once. The copy with clock
times wins, then the copy with more tags filled in. Two copies of one game
put their tournaments in the same split. So do near copies, which the key
does not catch: the same date and moves under differently spelt names, and a
truncated copy of a longer game. Those are merged in a second pass: games of
20 plies or more with the same date and the same moves, and truncated copies,
are one game. The copy with clocks is kept, else the longer one, and it keeps
its own `game_id`; the removed copies' ids go to
`games_extra.merged_near_copy_ids` and are counted in `parse_stats.json` as
`near_copies_removed`. A truncated copy with two different longer
continuations is left alone.

`games.parquet` is sorted by `game_id`, and `positions.parquet` by
`(game_id, ply)`. The sampler relies on both.

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

## Position statistics: `corpus/edges.parquet`

What the database view, the trends, the opponent prior in the repertoire
builder and the "share of games that chose this move" feature all read. Built
by `tnh.data.edges` from `games.parquet`: every kept game, every tempo, plies
below 80.

A position is identified by `pos_hash`, the 64-bit Zobrist hash that
`chess.polyglot.zobrist_hash` gives: pieces, side to move, castling rights,
and the en passant square when a pawn stands ready to capture there, whether
or not the capture is legal. (`positions.pos_key` uses the legal-capture
rule; the two differ only when that pawn is pinned.) Transpositions share a
hash.

One row per (position, move, year, rating band, tempo):

| column | type | meaning |
|---|---|---|
| `pos_hash` | u64 | The position before the move |
| `move` | str | UCI |
| `child_hash` | u64 | The position after the move |
| `year` | i16 | 0 when unknown |
| `elo_band` | i16 | Mean rating of the two players, floored to 200; 0 when a rating is missing |
| `tempo` | str | As in games |
| `n` | u32 | Games |
| `white_wins`, `draws`, `black_wins` | u32 | Results of those games |
| `sum_mover_elo`, `sum_opp_elo` | u64 | Over the games where both ratings are known |
| `n_rated` | u32 | How many those are |

A row is kept only when its position was reached by at least two games; the
rest of a game past that point is in `games.parquet`.

`corpus/pos_index.parquet`, one row per position with at least two games:
`pos_hash`, `epd` (of the first game in `game_id` order to reach it), `n`,
and the results. It is the lookup for a position's total and for the count
of any child.

`corpus/parents.parquet`, derived from edges: `child_hash`, `pos_hash`,
`move`, `n`, over all years, bands and tempos. It answers "which positions
lead here".

Counting. `pos_index.n` is the number of games that reach the position at a
ply up to 80 (the position after ply 79 included), each game once however
often it returns, including games that end there. `edges.n` is the number of
games that played the move from the position, each game once per (position,
move). A game that returns to a position and plays another move counts once
for each move, so a position's moves can add up to more than the games that
went on from it.

Layout. `edges` and `pos_index` are sorted by `pos_hash`, `parents` by
`child_hash`, in small row groups (32 k rows for edges, 8 k for the others).
`EdgeDB` reads the row groups' min and max once and then only the groups and
columns a lookup needs; a lookup costs 10 to 30 ms.

`tnh.data.edges.EdgeDB(name="full" | "dev")` is the reader: `total`, `moves`
(with `n_rated` and `last_year`), `moves_by_year`, `children`, `panel` (the
database view of a position in one scan) and `parents`. The `dev` tables
(`edges_dev.parquet` and so on) hold the games dated 2024 or later.
`edges.performance` is the one definition of rating performance:
`opp - 400 * log10(1 / score - 1)`, the difference capped at 800.

Hash collisions: with a few hundred million positions the chance of one
anywhere is about one in a thousand. Accepted.

## The repertoire builder: `tnh.builder`

The game it optimises: I play the memorised move in the positions I have
learned and like a model of myself everywhere else; the opponent always plays
like a model of the opponent. The task is to choose which N positions to
learn, and which move in each, to maximise my expected score.

- **Opponent's move distribution:** the database counts for the position
  laid over the model, `(count + kappa * p_model) / (n + kappa)`. `kappa` is
  a parameter, later a function of rating.
- **My move distribution** where nothing is learned: a `MoveModel` at my
  rating.
- **Sound moves:** in a learned position only moves within `eps` of the
  engine's best may be chosen, by a `ValueModel`.
- **Value where the budget runs out:** the expected score from a value
  function of the position and both ratings, read after a few plies of plain
  expectation over both models.
- **Search:** a knapsack on the tree by dynamic programming, the value of a
  subtree given k positions to spend in it. Exact on a tree; on a graph with
  transpositions the DP is repaired and every reported value is the exact
  expected score of the reported set on that graph.
- **Graph:** grown lazily against the models, best-first on a reach bound
  (`grow`), then below the current solution (`refine`). A node is a position
  key, `tnh.data.edges.pos_hash`, so transpositions share a node and a model
  call. Reach pruning does not bound the work: a sound move of mine has reach
  1 because I could learn it. Growth is therefore a heuristic, and
  `frontier_mass` reports how much of the result rests on positions that were
  never expanded.
- **Pessimism**, two parameters that model different failures and combine
  exactly:
  - `lam`: a share of opponents play the engine's best move everywhere (a
    prepared or stronger opponent). On a tree this is per-node mixing with
    the posterior that he is that type given his moves so far.
  - `lam_node`: at every opponent node a share of the probability moves to
    the engine's best move (model error, independent between nodes). This
    one favours difficulty spread over many nodes above a single trap.
  Both are set by the backtest (step 4), not by taste.
- **Bias at the frontier:** the maximum over my sound moves picks up noise in
  the frontier values, and a line that has been grown loses optimism that an
  ungrown line keeps. `solve_until_stable` grows below the chosen set until
  the same set is chosen twice and says so when it does not converge;
  `Params.shrink` pulls a frontier value toward its parent's.
- **Deep soundness:** the learned moves are few, so each is checked again
  with a deep engine search inside the loop (`tnh.builder.deepcheck`); a move
  that fails is banned and the search reruns.
- **Budget:** N positions, or a price `mu` per position (`budget_for_price`);
  the curve gives both.
- **What a position costs** (decided by the user 2026-11-12, not built): 0
  when the move learned there is obvious and 1 when it is not. Obvious means
  that the move model at my own rating gives the move at least a threshold,
  somewhere in 0.8-0.9. The threshold is a parameter and the output reports
  how the chosen set changes across that range. Until this is built every
  learned position costs 1.
- **Output:** the learned positions with their move, reach probability, the
  loss if that position alone were forgotten and the rise of the curve where
  it entered; the curve of expected score against N; each line played out
  with the engine's best replies and the engine value at its end; the frontier
  mass; the swing of the chosen set's value between rounds.
- **CPU:** one BLAS/OpenMP thread. The builder is one process on one core.

**Where the builder is going** (the user, 2026-11-12; not built):

- **No hard limit on sound moves.** `eps` = 0.03 shut out moves he considers
  playable (4.b3 and 4.cxd5 after 1.c4 Nf6 2.g3 e6 3.Bg2 d5). In its place:
  the result model, which has the engine's value among its inputs; the share
  of prepared opponents (`lam`); a wide safety net against plain blunders; and
  the engine loss reported for every chosen move.
- **A selective search in the manner of Lc0**, written here and not as a
  change to Lc0's code: PUCT over my candidate moves with a prior from the
  move model at my rating and Lc0's policy, descent by probability at the
  opponent's moves, leaves valued by the result model in batches. It runs as
  long as it is given and its answer only gets finer.
- **A fitted update of the move probabilities from the search below** (the
  user, 2026-11-12; after the first engine exists). The default move model
  cannot see which of the plausible moves are bad: it gives large mistakes
  1.43 times their real frequency (1.62 at 2200-2399) and 1.26 times the real
  expected loss per move, and a leaf's value does not change how often the
  opponent is thought to play into it. The repair is a model, fitted on real
  games, that takes a node's first probabilities and what the search has
  found under each move (the searched value against the best, how much search
  it rests on, the ratings) and returns the probabilities to use. Step 1's
  child-value columns are the special case of a search of depth zero (1.158
  to 1.134 nats; the ratios above fall to 1.18 and 1.12). The data is the
  engine run at a small budget on sampled game positions with the move played
  as the label; the engine logs what that needs. Until it exists the search
  uses the first probabilities as they are, and the root value is logged
  against the number of expanded nodes so that the drift is visible.
- **A price per non-obvious learned move** in place of a fixed N as the
  default: a move is learned where reach times gain exceeds the price, so the
  depth sets itself.
- **Used as an engine, not as a report.** In the interface it is chosen
  beside Stockfish and Lc0, shows the best few lines from the current
  position and updates them while it searches. That needs a search that keeps
  its graph between positions (a cache keyed on position, ratings and
  parameters, reused when the user steps into a child) and could be wrapped
  in UCI, with ratings and parameters as options.
- The first real run (round 3) and its export to the interface are a stopgap
  for looking at results until then.

Every model enters through the interfaces above, so the builder can be tested
on synthetic models where the best repertoire is planted and known.

## The interface: `tnh.web`

A local web application on 127.0.0.1. A board, and beside it, for the
position shown:

- the database view from `edges.parquet`: each move with its games, score,
  ratings and performance, filtered on years, rating band and tempo;
- trending moves: the change in a move's share over the years;
- what the models say: Maia-3's move probabilities and win, draw and loss at
  two chosen ratings, and the engine's evaluation;
- later, the statistics under best play with shrinkage, and the repertoire
  button, which calls `tnh.builder` from the position shown.

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

**Position-keyed feature files.** `features/positions/<model>/<variant>/part-*.parquet`
hold one row per `pos_hash` (for Maia, per `pos_hash` and pair of network
ratings), with the same `meta.json` rule. They are for work whose unit is a
position in the edge tables and not a row of a game: step 2 and the builder.
The same columns as above with `pos_hash` in place of `(game_id, ply)`.
Maia sees a move history, and a position has many: the history is the path
that first reached the position in the run that filled the cache (the most
played path for a position in a set, the parent's path plus the move for a
child). A position reached another way reuses that row. Whether this matters
has not been measured.

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
  `out_of_book`, `exit_ply_<book_count>`, `part`, `bucket` and `weight`. `weight` is population rows
  per sampled row, and every pooled number is weighted with it. A sample also
  holds a smaller set of rows from outside the window, flagged by
  `in_window`, for the clock breakdown and for the window shared with blitz.
- **Log loss of the played move**, with the model mixed with a uniform
  distribution over the legal moves, `p' = (1 - eps) * p + eps / n`. `eps` is
  fitted. This keeps the score proper and stops single impossible moves from
  dominating.
- **Where theory ends**, a separate test with its own sample
  (`tnh.eval.theory`), Maia-3 only. Rows are stratified on ply band, band of
  `pos_count` and the mover's rating group, with no window rule. Per cell,
  the offset with the lowest log loss. Memory makes people play above their
  strength, so within one ply band and one rating group the best offset
  should rise with `pos_count`; theory ends at the count where it has come
  down to its value in rare positions. This is what `book_count` should be
  set from. **Measured 2026-11-12: it does not rise.** The best offset is flat
  from `pos_count` 1 to 1,000-9,999 and lower above 10,000. `book_count`
  therefore has no measured basis and stays at 5 by convention.
- **The Elo sweep.** Maia-3 with both ratings offset by the same amount, from
  -200 to +800. The first measurement on elite games had its minimum at +300
  and broke down above about 2800 on Maia's scale. The result wanted is the
  best offset as a function of the player's FIDE rating. The mover's and the
  opponent's offset are also varied separately. **Measured 2026-11-12** on
  Maia-3 79m, classical 1800-2500: `maia_elo = 1580 + 0.558 * fide` for both
  players (best single offset +600; +724 at 1800 falling to +479 at 2400;
  the mover's and the opponent's offset are the same). Everything that calls
  Maia for a FIDE-rated player uses this map until a better one is measured.
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


**What step 1 settled for the layers above** (`reports/step1.md`, 2026-11-12):

- The move model to build on is Maia-3 at offset 0 and at the mapped rating,
  Lc0's policy and Lc0's raw value of every child, combined by a conditional
  logit or boosted trees: 1.14-1.15 nats per move against 1.29 for Maia at its
  best offset. All of it runs on the GPU. Stockfish at 100,000 nodes adds
  0.01-0.02 at about ten times the cost; Stockfish at 10,000 nodes adds
  nothing beside it.
- The best value head measured is Lc0 with `WDLCalibrationElo` at +400 and 50
  nodes (0.848 raw, 0.835 recalibrated, on its own 10,000 games). Maia-3's
  value head needs the five-parameter recalibration (1.221 raw, 0.888 after).

**The live model** (`tnh.models.combo`, 2026-11-12):

- `Combo()` is that combination as a `MoveModel`: Maia-3 79M at offset 0 and
  at +600, Lc0's policy and two columns from Lc0's raw child values, in the
  linear conditional logit with a uniform mixture. No rating terms and no
  trees. Coefficients are in `src/tnh/models/combo_coef.json` with their
  provenance; `python -m tnh.models.combo fit` refits them from the feature
  files. Test figure on `test_sub100k`: 1.1475 [1.1389, 1.1562].
- The second Maia setting is offset +600, not the mapped rating: step 1's
  feature files hold +600, so that is what the coefficients were fitted on.
  `Combo(maia_best="map1580+0.558x")` swaps the mapped rating in under the
  same coefficients (1.1606 against 1.1617 on 500 rows, not distinguishable).
  `Elo.parse("map<a>+<b>x")` is the mapped variant.
- `FrontierValue()` gives win/draw/loss for the mover from both FIDE ratings:
  Lc0 at 50 nodes, `WDLCalibrationElo` = mover + 400, `Contempt` = mover -
  opponent, then the stored recalibration (`value_recal.json`).
- `WithValue(Combo(), FrontierValue())` is a `MoveModel` whose `wdl` is the
  frontier value. The builder reads frontier values from `Prediction.wdl`, so
  this is passed as its `move_model` and nothing else changes.
- Cost on the laptop GPU: `Combo` 32 positions per second whatever the batch
  size, 88-91 % of it in Lc0's child values (one network call per legal
  move); `FrontierValue` 13.6 positions per second with one Lc0 process.
  Maia-3 23M in place of 79M does not change the total.
- **Pruned children**, `Combo(prune="p0.5")`: a child is evaluated only where
  the largest of Maia +0, Maia +600 and Lc0's policy gives the move at least
  0.5 %; the most probable move is always kept. The child columns are then
  relative to the best evaluated child, and an unevaluated move gets zero
  losses and the indicator `lc0ch_unev` with its own coefficient. 10.7
  children per position in place of 36, 82 positions per second in place of
  32, and the same log loss (1.1476 against 1.1475, difference +0.0000
  [-0.0003, +0.0004] on `test_sub100k`). `p1` and `p2` cost 0.0008 and 0.0020.
  Coefficients per rule in `combo_coef_<rule>.json`. The default is still all
  children.
- **Boosted trees on top**, `tnh.models.combotrees` (LightGBM, 15 leaves,
  rate 0.05, at least 2000 rows per leaf, early stopping on held-out
  tournaments of the fitting sample; the linear logit's inputs plus both
  ratings, ply, the number of legal moves and each model's rank of the move).
  Offline on `test_sub100k`, linear / trees: Maia 5M 1.433 / 1.428, 23M 1.332
  / 1.327, 79M 1.282 / 1.277; + Lc0 policy 1.172 / 1.158; + children at p2
  1.150 / 1.136, p1 1.148 / 1.135, p0.5 1.148 / 1.134. The trees cost almost
  no time. The top move's probability is reliable as it stands (0.850
  predicted, 0.849 played in the 0.8-0.9 bin) and the classes Lc0 can see are
  unbiased in every rating band; what remains is mistakes only Stockfish
  sees, overweighted by 1.18 overall and 1.34 at 2400 and above for large
  ones (Maia +600: 2.17 and 3.20). A T3 network in place of BT4 is worse at
  every step.
- **The default result model** (2026-11-12, `tnh.eval.result.ResultModel`): a
  multinomial logit on value heads with both ratings, ply and side to move.
  `"cheap"`: Maia-3's value head at +0 and +600 and Lc0's raw value head,
  0.8346 nats on 28,613 test games at about 350 positions per second.
  `"full"` adds Lc0 at 50 nodes with `WDLCalibrationElo` +400: 0.8276 at 16
  per second. Ratings alone 0.9675, Lc0's raw value alone 0.8410. Trees lose
  to the logit. The cheap one is the default: it needs exactly the network
  calls the default move model makes, so a node's value comes with its move
  probabilities. Looking ahead with the move model does not improve it: one
  ply of exact expectation blended in gives +0.0001 [-0.0012, +0.0016], and
  Monte Carlo lines are worse the deeper they go (+0.04 at 16 plies).
  Report: `reports/result_model.md`.
- **The default move model** (the user, 2026-11-12): Maia-3 79M at +0 and
  +600 and Lc0's policy under the trees, without child values (1.158). It is
  about four times as fast as with children at p0.5, and in the search the
  likely children are expanded anyway, so their values come in that way.
  `Combo(prune="p0.5", trees=True)` (1.134) is the accurate option.
- **Batch size stays at 12.** Lc0 0.32.1's own benchmark on BT4 peaks at
  1,720 evaluations per second at batch 48 against 1,461 at 12, and above 12
  a few positions per thousand come out wrong (up to 6 in a policy logit) and
  differ between runs. The GPU is 93 % busy in the child calls, so it is the
  network that limits, not the batching.

## Difficulty against results (step 2): `tnh.difficulty`

Question: does a difficulty measure from the move model explain White's real
score in a position beyond the engine value and the ratings? Commands:
`positions`, `compute`, `analyse`, `ceiling`, `synth`, `report`
(`python -m tnh.difficulty <command>`). Report: `reports/step2.md`.

- **Unit**: a position of the edge tables with at least `min_games` games in a
  rating band and tempo (default classical, 1800-2400), optionally below
  `max_games`. A set is named `<table>_n<min>` or `<table>_n<min>to<max-1>`.
  Per position: its games, White's score, mean ratings, the most played path
  (checked by replay), and its opening family, the first 4 plies of that path.
- **Model calls**, cached on `pos_hash`: Maia-3 on CPU at the mapped ratings,
  Stockfish MultiPV over every legal move (50,000 nodes), Stockfish at the
  root (200,000 nodes). Children are the side to move's moves by Maia
  probability until 0.9 is covered, at most 8, none below 0.02.
- **Measures**, from the side to move's view, signed for White in the
  analysis: expected loss (`el_stm`), narrow path (`hold_stm`), the same for
  the other side one ply on, the two-ply net, Maia's value head, and the value
  head with equal ratings. `el_db` and `hold_db` use the games' own move
  shares and are a reference, not a predictor.
- **Baseline**: weighted least squares of White's score on the engine's
  expected score and centipawns for White, the rating difference, the mean
  rating and the side to move. Weights `1 / (sigma2 / n + tau2)`.
- **The test**: families are split into 10 folds, every prediction comes from
  a fit that did not see the position's family, and every interval is a
  bootstrap over families (1,000 draws). Reported per measure: the share of
  what the baseline leaves above the binomial noise, and the real score minus
  the held-out baseline in the top tenth of the measure minus the bottom tenth.
- **The ceiling** (`ceiling`): the between-position SD of White's score after
  the rating terms only, per stratum of games, with empirical-Bayes shrunken
  edges and a list where each line appears once.
- **A planted world** (`synth`) with a known answer is run before the real one.
- **Rules for a confirmation run**: hypotheses and the statistic are written
  down before the set is computed, the pipeline and budgets are the earlier
  run's, and the report says confirmed, not confirmed or contradicted per
  hypothesis. Positions that are new can still share games with an earlier
  set through their ancestors; the report must say so.

Measured 2026-11-12 on the 2024+ table: between-position SD 0.010 (at least
100 games) and 0.014 (50-99 games) after engine and ratings; expected loss of
the side to move is the only measure that held on new positions, 0.008 between
the extreme tenths.

Step 2 reads the tables through the public batched readers `parents_many` and
`totals_many`.
