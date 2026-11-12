"""The real models behind the move table: model 1 (Combo, pruned) and model 2 (FrontierValue).

Slow and needs the GPU: Maia-3 79M and Lc0. Not run in round 3 (the GPU was taken).
"""

from __future__ import annotations

import chess
import pytest

from tnh.web.models import Hub, query


@pytest.mark.slow
def test_hub_with_the_live_models():
    hub = Hub("combo-p0.5", "frontier")
    try:
        b = chess.Board()
        b.push_uci("e2e4")
        m = hub.move_probs(b.fen(), ["e2e4"], 2200, 1800)
        assert len(m["moves"]) == 20 and sum(m["moves"].values()) == pytest.approx(1, abs=1e-4)
        assert hub.move_probs(b.fen(), ["e2e4"], 2200, 1800)["cached"] is True
        qs = []
        for mv in ("e7e5", "c7c5", "e7e6"):
            c = b.copy()
            c.push_uci(mv)
            qs.append(query(c, ["e2e4", mv], 2200, 1800))
        w = hub.values(qs)
        assert all(abs(sum(x) - 1) < 1e-6 for x in w)
        assert hub.counts["value_queries"] == 3
        hub.values(qs)
        assert hub.counts["value_queries"] == 3
    finally:
        hub.close()
