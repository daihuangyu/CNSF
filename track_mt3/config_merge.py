from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

from .config import ExperimentConfig, _construct


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_merged_config(*paths: str | Path) -> ExperimentConfig:
    values: Dict[str, Any] = {}
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = deep_merge(values, yaml.safe_load(handle) or {})
    return _construct(ExperimentConfig, values)

