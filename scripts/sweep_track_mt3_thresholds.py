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
from track_mt3.evaluation import evaluate_model, load_trajectory
from track_mt3.models import TrackMT3


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper Fig. 13 confidence-threshold sweep")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/threshold_sweep.json")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--minimum", type=float, default=0.05)
    parser.add_argument("--maximum", type=float, default=0.9)
    parser.add_argument("--increment", type=float, default=0.05)
    parser.add_argument("--scene", action="append", choices=("scenario1", "scenario2", "scenario3"))
    parser.add_argument("--max-runs", type=int)
    arguments = parser.parse_args()
    config = load_merged_config(arguments.config, *arguments.overlay)
    device = torch.device(resolve_device(config.training.device))
    checkpoint = torch.load(arguments.checkpoint, map_location=device, weights_only=False)
    values = np.round(
        np.arange(arguments.minimum, arguments.maximum + arguments.increment / 2, arguments.increment),
        4,
    )
    records = []
    defaults = (
        config.model.detection_threshold,
        config.model.tracking_threshold,
        config.evaluation.existence_threshold,
    )
    scene_trajectories = {"generated": None}
    if arguments.dataset_dir:
        dataset_dir = Path(arguments.dataset_dir)
        scenes = arguments.scene or ["scenario1", "scenario2", "scenario3"]
        scene_trajectories = {
            scene: [load_trajectory(path) for path in sorted((dataset_dir / scene).glob("run_*.npz"))]
            for scene in scenes
        }
        if arguments.max_runs is not None:
            scene_trajectories = {
                scene: trajectories[: arguments.max_runs]
                for scene, trajectories in scene_trajectories.items()
            }
    model = TrackMT3(config).to(device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    for kind in ("existence", "detection", "tracking"):
        for threshold in values:
            (
                config.model.detection_threshold,
                config.model.tracking_threshold,
                config.evaluation.existence_threshold,
            ) = defaults
            if kind == "existence":
                config.evaluation.existence_threshold = float(threshold)
            else:
                setattr(config.model, f"{kind}_threshold", float(threshold))
            for scene, trajectories in scene_trajectories.items():
                result = evaluate_model(model, config, trajectories=trajectories)
                records.append(
                    {
                        "threshold_type": kind,
                        "threshold": float(threshold),
                        "scene": scene,
                        **result.detailed_summary(),
                    }
                )
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    print(f"wrote {len(records)} sweep points to {output}")


if __name__ == "__main__":
    main()
