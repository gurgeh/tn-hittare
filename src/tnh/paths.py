"""Where data lives. Nothing large is ever stored in the repository."""

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("TNH_DATA", Path.home() / "out" / "tn-hittare"))

RAW = DATA_ROOT / "raw"            # downloads, exactly as fetched; one directory per source
CORPUS = DATA_ROOT / "corpus"      # games.parquet, positions.parquet
FEATURES = DATA_ROOT / "features"  # per-model outputs, keyed on (game_id, ply)
REPORTS = DATA_ROOT / "reports"    # markdown reports and their figures
