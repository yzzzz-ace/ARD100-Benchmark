"""Portable path resolution for the public benchmark package."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def env_path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else fallback.resolve()


DATA_ROOT = env_path("ARD100_DATASET_ROOT", REPO_ROOT / "data" / "ARD100_prepared")
OFFICIAL_ROOT = env_path("ARD100_OFFICIAL_ROOT", REPO_ROOT / "data" / "ARD100_official_split")
YOLOMG_CODE_ROOT = env_path("YOLOMG_CODE_ROOT", REPO_ROOT / "external" / "YOLOMG")
OUTPUT_ROOT = env_path("ARD100_OUTPUT_ROOT", REPO_ROOT / "outputs")
WEIGHTS_ROOT = env_path("ARD100_WEIGHTS_ROOT", REPO_ROOT / "weights")
INITIAL_WEIGHTS_ROOT = env_path("ARD100_INITIAL_WEIGHTS_ROOT", REPO_ROOT / "initial_weights")

