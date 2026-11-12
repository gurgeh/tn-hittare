"""The real models behind the model panel: Maia-3 5m and one Stockfish process on the CPU."""

from __future__ import annotations

import chess
import pytest

from tnh.web.models import Hub


@pytest.mark.slow
def test_hub_maia_and_engine():
    hub = Hub(maia_size="5m", device="cpu", torch_threads=1)
    try:
        b = chess.Board()
        b.push_uci("e2e4")
        m = hub.maia(b.fen(), ["e2e4"], 2200, 1800)
        assert (m["mover_fide"], m["opp_fide"]) == (1800, 2200) and m["mover_maia"] == 2584
        assert sum(m["moves"].values()) == pytest.approx(1, abs=1e-4)
        assert len(m["moves"]) == 20 and m["cached"] is False
        assert hub.maia(b.fen(), ["e2e4"], 2200, 1800)["cached"] is True
        # other ratings are another cache entry
        assert hub.maia(b.fen(), ["e2e4"], 2200, 2400)["cached"] is False

        e = hub.engine(b.fen(), ["e2e4"], 20_000)
        assert set(e["moves"]) == {mv.uci() for mv in b.legal_moves}
        assert all(0 <= v["value"] <= 1 for v in e["moves"].values() if v["value"] is not None)
        assert hub.engine(b.fen(), ["e2e4"], 20_000)["cached"] is True
    finally:
        hub.close()
