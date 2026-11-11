"""Step 1: how well existing models predict the moves humans play in classical chess.

Each command reads files and writes files, so the steps can be rerun separately.
In order:

    tnh.eval.sample      fixed, stratified samples of position rows (test from `test`, fit from `train`)
    tnh.models.run       features on both samples (owned by tnh.models)
    tnh.eval.logloss     log loss raw and with the fitted uniform mixture
    tnh.eval.sweep       the Elo sweep: best offset per rating bucket, the FIDE -> Maia map
    tnh.eval.mistakes    calibration by kind of mistake: is classical a rating-shifted blitz?
    tnh.eval.logit       the conditional logit over the legal moves
    tnh.eval.breakdown   log loss by ply and by time left
    tnh.eval.value       the value head on the game results
    tnh.eval.blitz       fetch and build the Lichess blitz reference positions
    tnh.eval.reference   Maia-3 on blitz beside classical under one window
    tnh.eval.report      reports/step1.md from the result files

Shared conventions are in ``tnh.eval.common``.
"""
