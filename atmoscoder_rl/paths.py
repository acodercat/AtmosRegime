"""Where things live in this repository. Every module takes its paths from here."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"                         # the vendored dataset (data/README.md)
SPLITS_PATH = REPO_ROOT / "splits" / "splits.json"    # the frozen partition
EXPERIMENTS_DIR = REPO_ROOT / "experiments"           # run files, gitignored (runs.py)
MODELS_TOML = REPO_ROOT / "models.toml"               # model configs (evaluation/client.py)
