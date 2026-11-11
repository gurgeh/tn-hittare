"""The feature runner end to end on a fixture corpus in the DESIGN.md schema."""

import chess
import numpy as np
import polars as pl
import pytest

from _corpus import build
from tnh.models import run
from tnh.models.stockfish import STOCKFISH


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    games, positions = build(tmp_path_factory.mktemp("corpus"))
    return games, positions


def test_fixture_corpus_is_built_by_tnh_data(corpus):
    from tnh.data.parse import GAMES_SCHEMA
    from tnh.data.positions import POSITIONS_SCHEMA

    games, positions = (pl.read_parquet(p) for p in corpus)
    assert games.schema == pl.Schema(GAMES_SCHEMA)
    assert positions.schema == pl.Schema(POSITIONS_SCHEMA)
    assert games.height == 5
    assert set(games["tempo"]) == {"classical", "rapid"}
    assert positions["game_id"].n_unique() == 4      # the rapid game has no positions
    assert positions["ply"].max() == 79


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
    base = ["--model", "stockfish", "--nodes", "3000", "--processes", "4", "--positions", str(positions),
            "--out", str(tmp_path), "--chunk", "25"]
    out = tmp_path / "nodes3000"
    run.main(base + ["--limit", "60"])
    assert len(list(out.glob("part-*.parquet"))) == 3
    assert read_features(out).height == 60
    run.main(base)
    feat = read_features(out)
    assert feat.schema == pl.Schema({"game_id": pl.String, "ply": pl.Int64, "moves": pl.List(pl.String),
                                     "value": pl.List(pl.Float32), "cp": pl.List(pl.Float32),
                                     "mate": pl.List(pl.Int16), "wdl": pl.List(pl.Float32)})
    joined = check_against_positions(feat, positions, "value")
    n_mates = 0
    for value, cp, mate in joined.select("value", "cp", "mate").iter_rows():
        v, c, m = np.array(value, dtype=np.float64), np.array(cp, dtype=np.float64), np.array(mate)
        assert len(v) == len(c) == len(m)
        # NaN value where the budget ran out before the engine reached the move
        assert np.all(np.isnan(v) | ((v >= 0) & (v <= 1)))
        assert np.array_equal(np.isnan(c), np.isnan(v) | (m != 0))   # cp only for non-mate scores
        assert np.all(v[m > 0] == 1.0) and np.all(v[m < 0] == 0.0)
        assert np.all(m[np.isnan(v)] == 0)
        n_mates += int((m != 0).sum())
    assert n_mates > 0                               # the fixture games contain mates
    vals = np.concatenate([np.array(v, dtype=np.float64) for v in joined["value"].to_list()])
    assert np.isnan(vals).mean() < 0.2
    assert all(w is None or (len(w) == 3 and abs(sum(w) - 1) < 1e-5) for w in joined["wdl"].to_list())
    n_parts = len(list(out.glob("part-*.parquet")))
    run.main(base)                                 # nothing left: no new part
    assert len(list(out.glob("part-*.parquet"))) == n_parts
    (out / "meta.json").write_text((out / "meta.json").read_text().replace('"hash_mb": 16', '"hash_mb": 32'))
    with pytest.raises(SystemExit, match="other options"):
        run.main(base)


@pytest.mark.slow
def test_maia3_run_several_settings_one_pass(corpus, tmp_path):
    from tnh.models.maia3 import Elo, Maia3
    from tnh.models.run import load_todo, queries

    games, positions = corpus
    common = ["--model", "maia3", "--size", "5m", "--positions", str(positions), "--out", str(tmp_path),
              "--chunk", "100"]
    run.main(common + ["--offsets", "100", "--limit", "150"])          # one setting, partly done
    run.main(common + ["--offsets", "0", "100", "--elo", "offset+100-opp-100", "fixed2200-2200"])
    names = ["offset+0", "offset+100", "offset+100-opp-100", "fixed2200-2200"]
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_dir()) == sorted(names)
    todo = load_todo(positions, games, [], None, None)
    direct = Maia3("5m").predict_many(queries(todo), [Elo.parse(n) for n in names])
    for name, preds in zip(names, direct):
        feat = read_features(tmp_path / name)
        assert feat.schema == pl.Schema({"game_id": pl.String, "ply": pl.Int64, "moves": pl.List(pl.String),
                                         "logp": pl.List(pl.Float32), "wdl": pl.List(pl.Float32)})
        joined = check_against_positions(feat, positions, "logp")
        for logp in joined["logp"].to_list():
            assert abs(np.exp(np.array(logp, dtype=np.float64)).sum() - 1) < 1e-4
        by_key = {(g, p): lp for g, p, lp in feat.select("game_id", "ply", "logp").iter_rows()}
        for (g, p), d in zip(todo.select("game_id", "ply").iter_rows(), preds):
            np.testing.assert_allclose(by_key[(g, p)], d.logp, atol=0.07)   # float16 batch noise
    # the settings differ from each other
    a, b = (read_features(tmp_path / n).sort("game_id", "ply")["logp"].to_list() for n in ("offset+0", "fixed2200-2200"))
    assert max(abs(x - y) for u, v in zip(a, b) for x, y in zip(u, v)) > 0.1


@pytest.mark.slow
def test_lc0_run_both_kinds(corpus, tmp_path):
    from tnh.models import lc0

    if not lc0.binary().exists():
        pytest.skip(f"no Lc0 at {lc0.LC0_HOME}")
    _, positions = corpus
    common = ["--positions", str(positions), "--out", str(tmp_path), "--chunk", "40", "--limit", "80"]
    run.main(["--model", "lc0-policy"] + common)
    run.main(["--model", "lc0-search", "--nodes", "50", "--lc0-processes", "2"] + common)
    pol = read_features(tmp_path / "policy")
    assert pol.schema == pl.Schema({"game_id": pl.String, "ply": pl.Int64, "moves": pl.List(pl.String),
                                    "logp": pl.List(pl.Float32), "wdl": pl.List(pl.Float32)})
    assert pol.height == 80
    for logp in pol["logp"].to_list():
        assert abs(np.exp(np.array(logp, dtype=np.float64)).sum() - 1) < 1e-4
    val = read_features(tmp_path / "nodes50")
    assert val.schema == pl.Schema({"game_id": pl.String, "ply": pl.Int64, "moves": pl.List(pl.String),
                                    "value": pl.List(pl.Float32), "cp": pl.List(pl.Float32),
                                    "mate": pl.List(pl.Int16), "wdl": pl.List(pl.Float32)})
    assert val.height == 80
    visited = np.concatenate([~np.isnan(np.array(v, dtype=np.float64)) for v in val["value"].to_list()])
    assert 0 < visited.mean() < 1          # 50 nodes: some root moves are never visited
    assert sorted(pol["game_id"] + pol["ply"].cast(pl.String)) == sorted(val["game_id"] + val["ply"].cast(pl.String))
    run.main(["--model", "lc0-children"] + common)
    ch = read_features(tmp_path / "children")
    assert ch.schema == val.schema and ch.height == 80
    vals = np.concatenate([np.array(v, dtype=np.float64) for v in ch["value"].to_list()])
    assert not np.isnan(vals).any() and np.all((vals >= 0) & (vals <= 1))   # every child has a value


def test_games_file_default(tmp_path, monkeypatch):
    from tnh.models.run import games_file

    corpus_dir, samples = tmp_path / "corpus", tmp_path / "corpus" / "samples"
    samples.mkdir(parents=True)
    monkeypatch.setattr(run.paths, "CORPUS", corpus_dir)
    pos = samples / "test.parquet"
    with pytest.raises(SystemExit, match="no games file"):
        games_file(pos, None)
    (corpus_dir / "games.parquet").write_bytes(b"")
    assert games_file(pos, None) == corpus_dir / "games.parquet"      # falls back to the corpus
    (samples / "games.parquet").write_bytes(b"")
    assert games_file(pos, None) == samples / "games.parquet"          # beside the positions wins
    assert games_file(pos, corpus_dir / "games.parquet") == corpus_dir / "games.parquet"
    with pytest.raises(SystemExit, match="does not exist"):
        games_file(pos, tmp_path / "nope.parquet")
