#!/usr/bin/env python3
"""Evaluate CNSF T-GOSPA with tracker-native runtime identities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

import _bootstrap  # noqa: F401

from track_mt3.evaluation import load_operating_point, load_trajectory
from track_mt3.metrics import trajectory_gospa
from track_mt3.models_v17 import EndToEndRecursiveTracker, V17BConfig
from track_mt3.models_v17.batching import pad_current_frames


SCENES = ("scenario1", "scenario2", "scenario3")


@torch.inference_mode()
def evaluate_scene(model, trajectories, arguments) -> list[dict[str, float]]:
    output_positions = [[] for _ in trajectories]
    output_identities = [[] for _ in trajectories]
    state = None
    for frame_index in range(len(trajectories[0])):
        measurements, mask, _, frame_time, _, _ = pad_current_frames(
            [trajectory[frame_index] for trajectory in trajectories], model.device
        )
        output, state = model.forward_inference_step_fast(
            measurements,
            mask,
            frame_time,
            state,
            birth_candidate_threshold=arguments.candidate_threshold,
            confirmation_hits=arguments.confirmation_hits,
            retention_threshold=arguments.retention_threshold,
            survival_warmup_frames=arguments.survival_warmup_frames,
        )
        for row in range(len(trajectories)):
            probabilities = output.existing_logits[row].sigmoid()
            selected = (
                output.existing_mask[row]
                & state.confirmed_mask[row]
                & (probabilities >= arguments.existence_threshold)
            )
            output_positions[row].append(
                output.posterior_mean[row, selected, :2].detach().cpu().numpy()
            )
            output_identities[row].append(
                output.existing_runtime_ids[row, selected].detach().cpu().numpy()
            )

    records = []
    start = arguments.first_scored_frame
    for trajectory, positions, identities in zip(
        trajectories, output_positions, output_identities
    ):
        truth_positions = [
            np.asarray(frame.states[:, :2], dtype=np.float64)
            for frame in trajectory[start:]
        ]
        truth_identities = [
            np.asarray(frame.target_ids, dtype=np.int64)
            for frame in trajectory[start:]
        ]
        value = trajectory_gospa(
            truth_positions,
            truth_identities,
            positions[start:],
            identities[start:],
            c=2.0,
            p=1.0,
            gamma=1.0,
        )
        scored_frames = len(truth_positions)
        records.append(
            {
                "tgospa": value.total / scored_frames,
                "localization": value.localization / scored_frames,
                "missed": value.missed / scored_frames,
                "false": value.false / scored_frames,
                "switch": value.switch / scored_frames,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/training/cnsf_exact12k.yaml",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--operating-points",
        default="configs/evaluation/operating_points.yaml",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate-threshold", type=float)
    parser.add_argument("--existence-threshold", type=float)
    parser.add_argument("--confirmation-hits", type=int)
    parser.add_argument("--retention-threshold", type=float)
    parser.add_argument("--survival-warmup-frames", type=int)
    parser.add_argument("--first-scored-frame", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    point = load_operating_point(arguments.operating_points, "cnsf")
    if arguments.candidate_threshold is None:
        arguments.candidate_threshold = float(point["candidate"])
    if arguments.existence_threshold is None:
        arguments.existence_threshold = float(point["existence"])
    if arguments.confirmation_hits is None:
        arguments.confirmation_hits = int(point["confirmation_hits"])
    if arguments.retention_threshold is None:
        arguments.retention_threshold = float(point["retention"])
    if arguments.survival_warmup_frames is None:
        arguments.survival_warmup_frames = int(point["survival_warmup_frames"])

    config = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    device = torch.device(arguments.device)
    model = EndToEndRecursiveTracker(V17BConfig(**config["model"])).to(device)
    checkpoint = torch.load(arguments.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    scenes = {}
    for scene in SCENES:
        paths = sorted((arguments.dataset_dir / scene).glob("run_*.npz"))[
            : arguments.runs
        ]
        if len(paths) != arguments.runs:
            raise FileNotFoundError(
                f"expected {arguments.runs} trajectories in "
                f"{arguments.dataset_dir / scene}"
            )
        trajectories = [load_trajectory(path) for path in paths]
        records = evaluate_scene(model, trajectories, arguments)
        scenes[scene] = {
            "tgospa": float(np.mean([record["tgospa"] for record in records])),
            "per_run": records,
        }
        print(f"{scene}: T-GOSPA={scenes[scene]['tgospa']:.6f}", flush=True)

    result = {
        "method": "CNSF",
        "checkpoint": str(arguments.checkpoint),
        "runs_per_scene": arguments.runs,
        "first_scored_frame": arguments.first_scored_frame,
        "metric": {"p": 1.0, "c": 2.0, "gamma": 1.0},
        "operating_point": {
            "candidate": arguments.candidate_threshold,
            "existence": arguments.existence_threshold,
            "confirmation_hits": arguments.confirmation_hits,
            "retention": arguments.retention_threshold,
            "survival_warmup_frames": arguments.survival_warmup_frames,
        },
        "scenes": scenes,
        "mean_tgospa": float(
            np.mean([scene["tgospa"] for scene in scenes.values()])
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
