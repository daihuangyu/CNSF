#!/usr/bin/env python3
"""Fresh-init launcher for the exact-12k Track-MT3-CM control."""
from __future__ import annotations

import os
from pathlib import Path

import _bootstrap  # noqa: F401

from track_mt3.config_merge import load_merged_config
from track_mt3.models import TrackMT3
from track_mt3.training import Trainer


BASE_CONFIG = "configs/paper.yaml"
V9_CONFIG = "configs/training/track_mt3.yaml"
CONTROL_CONFIG = "configs/training/track_mt3_cm_exact12k.yaml"
EXPECTED_PARAMETERS = 8_484_460


def load_control_config():
    return load_merged_config(BASE_CONFIG, V9_CONFIG, CONTROL_CONFIG)


def validate_control(config) -> int:
    expected = {
        "encoder_layers": 3,
        "decoder_layers": 3,
        "feedforward_dim": 1472,
        "window_size": 20,
    }
    observed = {name: getattr(config.model, name) for name in expected}
    if observed != expected:
        raise ValueError(f"invalid Track-MT3-CM model config: {observed}; expected {expected}")
    if config.training.seed != 1919 or config.training.updates != 12_000:
        raise ValueError("Track-MT3-CM requires seed=1919 and updates=12000")
    if config.training.early_stopping_patience != 0:
        raise ValueError("early stopping must be disabled for an exact-12k run")
    model = TrackMT3(config)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != EXPECTED_PARAMETERS:
        raise ValueError(
            f"Track-MT3-CM parameter drift: got {parameters:,}, "
            f"expected {EXPECTED_PARAMETERS:,}"
        )
    return parameters


def main() -> None:
    config = load_control_config()
    parameters = validate_control(config)
    output_dir = Path(config.training.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"fresh-init control refuses non-empty output directory: {output_dir}"
        )
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            "Track-MT3-CM preflight: "
            f"parameters={parameters:,}, window={config.model.window_size}, "
            f"seed={config.training.seed}, exact_updates={config.training.updates}",
            flush=True,
        )
    Trainer(config).fit()


if __name__ == "__main__":
    main()
