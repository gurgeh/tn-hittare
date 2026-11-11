"""The feature runner end to end on a fixture corpus in the DESIGN.md schema."""

import chess
import numpy as np
import polars as pl
import pytest

from _corpus import build
from tnh.models import run
from tnh.models.stockfish import STOCKFISH

GAMES_COLUMNS = ["game_id", "source", "tournament_id", "event", "site", "round", "date", "year", "white", "black",
                 "white_elo", "black_elo", "white_fide_id", "black_fide_id", "white_title", "black_title", "result",
                 "time_control_raw", "tempo", "tempo_source", "base_s", "inc_s", "has_clock", "n_plies", "moves_uci",
                 "clocks_s", "split"]
POSITIONS_COLUMNS = ["game_id", "ply", "fen", "played_uci", "white_to_move", "mover_elo", "opp_elo", "clock_mover_s",
                     "clock_opp_s", "move_time_s", "pos_key", "pos_count", "n_legal", "result_mover", "split"]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    games, positions = build(tmp_path_factory.mktemp("corpus"))
    return games, positions


def test_fixture_corpus_follows_the_schema(corpus):
    games, positions = (pl.read_parquet(p) for p in corpus)
    assert games.columns == GAMES_COLUMNS
    assert positions.columns == POSITIONS_COLUMNS
    assert games.height == 5
    assert set(games["tempo"]) == {"classical", "rapid"}
    assert positions["game_id"].n_unique() == 4      # the rapid game has no positions
    assert positions["ply"].max() == 79
    assert positions.filter(pl.col("ply") == 0)["pos_count"].to_list() == [5, 5, 5, 5]


def read_features(out):
    return pl.read_parquet(sorted(out.glob("part-*.parquet")))


def check_against_positions(feat, positions, column):
    pos = pl.read_parquet(positions)
    assert feat.select("game_id", "ply").is_duplicated().sum() == 0
    joined = pos.join(feat, on=["game_id", "ply"], how="inner")
    assert joined.height == pos.height == feat.height
    for fen, moves, vals, n_legal, played in joined.select("fen", "moves", column, "n_legal", "played_uci").iter_rows():
        assert sorted(moves) == sorted(m.uci() for m in chess.Board(fen).legal_moves)
        assert len(vals) == len(moves) == n_legal and played in moves
    return joined


@pytest.mark.skipif(not STOCKFISH.exists(), reason=f"no Stockfish at {STOCKFISH}")
def test_stockfish_run_is_resumable(corpus, tmp_path):
    _, positions = corpus
    out = tmp_path / "sf"
    base = ["--model", "stockfish", "--nodes", "3000", "--processes", "4", "--positions", str(positions),
            "--out", str(out), "--chunk", "25"]
    run.main(base + ["--limit", "60"])
    assert len(list(out.glob("part-*.parquet"))) == 3
    assert read_features(out).height == 60
    run.main(base)
    first = read_features(out)
    joined = check_against_positions(first, positions, "value")
    vals = np.concatenate([np.array(v, dtype=np.float64) for v in joined["value"].to_list()])
    # NaN where the node budget ran out before the engine reached the move (counted in the run's stats)
    assert np.all(np.isnan(vals) | ((vals >= 0) & (vals <= 1)))
    assert np.isnan(vals).mean() < 0.2
    assert all(w is None or (len(w) == 3 and abs(sum(w) - 1) < 1e-5) for w in joined["wdl"].to_list())
    n_parts = len(list(out.glob("part-*.parquet")))
    run.main(base)                                 # nothing left: no new part
    assert len(list(out.glob("part-*.parquet"))) == n_parts
    with pytest.raises(SystemExit, match="other options"):
        run.main(["--model", "stockfish", "--nodes", "4000", "--processes", "2", "--positions", str(positions),
                  "--out", str(out)])


@pytest.mark.slow
def test_maia3_run(corpus, tmp_path):
    from tnh.models.maia3 import Maia3
    from tnh.models.run import load_todo, queries

    games, positions = corpus
    out = tmp_path / "maia"
    args = ["--model", "maia3", "--size", "5m", "--elo-offset", "100", "--positions", str(positions),
            "--out", str(out), "--chunk", "100"]
    run.main(args + ["--limit", "150"])
    run.main(args)
    feat = read_features(out)
    joined = check_against_positions(feat, positions, "logp")
    for logp in joined["logp"].to_list():
        assert abs(np.exp(np.array(logp, dtype=np.float64)).sum() - 1) < 1e-4
    # same numbers as calling the model directly
    todo = load_todo(positions, games, tmp_path / "empty", None, None)
    direct = Maia3("5m", elo_offset=100).predict_logp(queries(todo))
    by_key = {(g, p): lp for g, p, lp in feat.select("game_id", "ply", "logp").iter_rows()}
    for (g, p), d in zip(todo.select("game_id", "ply").iter_rows(), direct):
        np.testing.assert_allclose(by_key[(g, p)], d.logp, atol=0.07)   # float16 batch noise, see test_maia3_reference
