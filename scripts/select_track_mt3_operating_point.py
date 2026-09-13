#!/usr/bin/env python3
"""Select one global Track-MT3 operating point on an explicit validation set."""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401

from track_mt3.config import resolve_device
from track_mt3.config_merge import load_merged_config
from track_mt3.data.simulator import MultiTargetSimulator
from track_mt3.evaluation import evaluate_model, load_trajectory
from track_mt3.models import TrackMT3


SCENES = ("scenario1", "scenario2", "scenario3")


def comma_floats(text: str) -> list[float]:
    values = [float(item) for item in text.split(",")]
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise argparse.ArgumentTypeError("thresholds must be a comma-separated list in [0,1]")
    return values


def load_split(directory: Path, max_runs: int | None) -> dict[str, list]:
    datasets = {}
    for scene in SCENES:
        paths = sorted((directory / scene).glob("run_*.npz"))
        if max_runs is not None:
            paths = paths[:max_runs]
        if not paths:
            raise FileNotFoundError(f"no validation trajectories found in {directory / scene}")
        datasets[scene] = [load_trajectory(path) for path in paths]
    return datasets


def generate_split(
    config_paths: list[str | Path], seed: int, runs: int
) -> dict[str, list]:
    datasets = {}
    for scene in SCENES:
        scene_number = int(scene[-1])
        config = load_merged_config(
            *config_paths, f"configs/experiments/scenario{scene_number}.yaml"
        )
        datasets[scene] = [
            MultiTargetSimulator(config.simulation, seed + run).simulate(
                config.evaluation.trajectory_steps
            )
            for run in range(runs)
        ]
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-step", type=int)
    split = parser.add_mutually_exclusive_group(required=True)
    split.add_argument("--validation-dir", type=Path)
    split.add_argument("--validation-seed", type=int)
    parser.add_argument("--validation-runs", type=int, default=10)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument(
        "--existence-thresholds",
        type=comma_floats,
        default=comma_floats("0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90"),
        help="also controls the QTM detection-query threshold",
    )
    parser.add_argument(
        "--tracking-thresholds",
        type=comma_floats,
        default=comma_floats("0.70,0.75,0.80,0.85,0.90,0.95"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_paths = [args.config, *args.overlay]
    config = load_merged_config(*config_paths)
    device = torch.device(resolve_device(config.training.device))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    checkpoint_step = int(checkpoint.get("training_state", {}).get("step", -1))
    if args.expected_step is not None and checkpoint_step != args.expected_step:
        raise ValueError(
            f"expected step {args.expected_step}, got {checkpoint_step}"
        )
    model = TrackMT3(config).to(device).eval()
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    if args.validation_dir is not None:
        datasets = load_split(args.validation_dir, args.max_runs)
        selection_split = str(args.validation_dir)
    else:
        if args.max_runs is not None:
            raise ValueError("--max-runs applies only to --validation-dir")
        datasets = generate_split(
            config_paths, args.validation_seed, args.validation_runs
        )
        selection_split = (
            f"generated scenarios, seeds "
            f"{args.validation_seed}..{args.validation_seed + args.validation_runs - 1}"
        )

    records = []
    for existence, tracking in product(
        args.existence_thresholds, args.tracking_thresholds
    ):
        # The published two-knob Track-MT3 operating point ties the output
        # existence threshold to the QTM detection-query threshold. Tracking
        # query retention remains the second independent knob.
        config.evaluation.existence_threshold = existence
        config.model.detection_threshold = existence
        config.model.tracking_threshold = tracking
        scenes = {
            scene: evaluate_model(model, config, trajectories=trajectories).detailed_summary()
            for scene, trajectories in datasets.items()
        }
        record = {
            "existence_threshold": existence,
            "qtm_detection_threshold": existence,
            "qtm_tracking_threshold": tracking,
            "mean_gospa": float(np.mean([row["gospa"] for row in scenes.values()])),
            "mean_pro_gospa": float(
                np.mean([row["pro_gospa"] for row in scenes.values()])
            ),
            "scenes": scenes,
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    # Deterministic tie breaking prefers the larger thresholds, so a numerical
    # tie cannot silently favor a more permissive tracker.
    best = min(
        records,
        key=lambda row: (
            row["mean_gospa"],
            -row["existence_threshold"],
            -row["qtm_tracking_threshold"],
        ),
    )
    result = {
        "selection_split": selection_split,
        "selection_rule": "minimum unweighted three-scene mean GOSPA",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_step,
        "runs_per_scene": {scene: len(rows) for scene, rows in datasets.items()},
        "best_global_operating_point": best,
        "grid": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"selected global operating point: {json.dumps(best)}", flush=True)


if __name__ == "__main__":
    main()
