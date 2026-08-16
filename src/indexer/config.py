from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise TypeError(f"Expected a mapping in the config file: {path}")

    return config


def load_data_sources(config_path: str | Path | None = None) -> Dict[str, Dict[str, Any]]:
    return load_config(config_path).get("data_sources", {})


def load_inference_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    return load_config(config_path).get("inference", {})
