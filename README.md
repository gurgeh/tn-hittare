# tn-hittare

A model of how humans play classical chess, and a repertoire builder on top of
it.

An engine says which moves are sound. It does not say which of three equal
moves leaves the opponent with the harder game. The aim here is an opening
repertoire that is objectively sound, unusual, and hard to meet over the
board, within a budget of positions to memorise. Not traps: positions where
the opponent has to find many right moves in a row.

"TN" is a theoretical novelty. "Hittare" is Swedish for finder.

## Plan

0. **A corpus** of classical over-the-board games with ratings and clock
   times, split by tournament. First source: the
   [Lichess broadcast dumps](https://database.lichess.org/#broadcasts).
1. **Measure existing models** on it: log loss of
   [Maia-3](https://github.com/CSSLab/maia3) on the moves that were played,
   with its Elo input swept, then a small mixture with engine features, and
   the log loss of the value head on the results. Maia-3 was trained on
   online blitz, so this shows how far it is from classical play.
2. **Test difficulty measures against results.** Does the gap between the
   human expected score and the engine's explain how positions actually score?
3. **The repertoire builder.** I play the memorised move in N chosen
   positions and like the model everywhere else; the opponent always plays
   like the model. Choose the N positions that maximise the expected score.
4. **Backtest** the selection on later games.
5. **Fine-tune** with time control and clock as inputs, if step 1 says it is
   worth it.

Steps 0 and 1 are being built. The contract between the modules is in
[`docs/DESIGN.md`](docs/DESIGN.md).

## Running

```bash
uv sync
uv run pytest
```

Data lives under `~/out/tn-hittare/` (override with `TNH_DATA`), never in the
repository.

## License

MIT. The Lichess broadcast games are CC BY-SA 4.0 and Maia-3 has its own
license; neither is redistributed here beyond a few games used as test
fixtures.
