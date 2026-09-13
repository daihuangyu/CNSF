from __future__ import annotations

from pathlib import Path

import yaml


DEFAULT_OPERATING_POINTS = Path("configs/evaluation/operating_points.yaml")


def method_key(method: str) -> str:
    return method.strip().lower().replace("-", "_")


def load_operating_point(path: str | Path, method: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    key = method_key(method)
    if not isinstance(document, dict) or key not in document:
        raise KeyError(f"operating point {key!r} is not defined in {path}")
    point = document[key]
    if not isinstance(point, dict):
        raise TypeError(f"operating point {key!r} must be a mapping")
    return point
