from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "data_sources.yaml"


def load_data_sources(config_path: str | Path | None = None) -> Dict[str, Dict[str, Any]]:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Data source config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise TypeError(f"Expected a mapping in the config file: {path}")

    return config
