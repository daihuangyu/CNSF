#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401

from track_mt3.config import resolve_device
from track_mt3.config_merge import load_merged_config
from track_mt3.evaluation import evaluate_model
from track_mt3.models import TrackMT3


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Track-MT3")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="evaluation.npz")
    arguments = parser.parse_args()
    config = load_merged_config(arguments.config, *arguments.overlay)
    device = torch.device(resolve_device(config.training.device))
    model = TrackMT3(config).to(device)
    checkpoint = torch.load(arguments.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    result = evaluate_model(model, config)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        gospa=result.gospa,
        pro_gospa=result.pro_gospa,
        localization=result.localization,
        missed=result.missed,
        false=result.false,
    )
    print(json.dumps(result.summary(), indent=2))


if __name__ == "__main__":
    main()
