#!/usr/bin/env python3
"""Evaluate the frozen Track-MT3-CM checkpoint for the paper table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401

from track_mt3.config import resolve_device
from track_mt3.config_merge import load_merged_config
from track_mt3.data.window import build_sliding_windows
from track_mt3.evaluation import evaluate_model, load_trajectory
from track_mt3.metrics.tgospa import trajectory_gospa
from track_mt3.models import TrackMT3


SCENES = ("scenario1", "scenario2", "scenario3")


@torch.inference_mode()
def native_tgospa(model, config, trajectories, first_scored_frame: int) -> list[dict]:
    records = []
    for frames in trajectories:
        tracks = model.empty_tracks()
        next_track_id = 0
        next_ephemeral_id = -1
        positions = [np.empty((0, 2)) for _ in range(config.model.window_size - 1)]
        identities = [
            np.empty((0,), dtype=np.int64)
            for _ in range(config.model.window_size - 1)
        ]
        for window in build_sliding_windows(frames, config.model.window_size):
            prediction = model.forward_window(window.to(model.device), tracks)
            probabilities = prediction.probabilities.reshape(-1)
            keep = probabilities > config.evaluation.existence_threshold
            query_ids = torch.full(
                (len(prediction.positions),), -1, dtype=torch.long, device=model.device
            )
            query_ids[: len(tracks)] = tracks.track_ids
            next_tracks, selected = model.qtm(
                prediction, tracks, query_track_ids=query_ids
            )
            new_mask = next_tracks.track_ids < 0
            new_count = int(new_mask.sum())
            if new_count:
                next_tracks.track_ids[new_mask] = torch.arange(
                    next_track_id,
                    next_track_id + new_count,
                    dtype=torch.long,
                    device=model.device,
                )
                next_track_id += new_count

            output_ids = torch.full(
                (len(prediction.positions),), -1, dtype=torch.long, device=model.device
            )
            output_ids[selected] = next_tracks.track_ids
            ephemeral = keep & (output_ids < 0)
            ephemeral_count = int(ephemeral.sum())
            if ephemeral_count:
                output_ids[ephemeral] = torch.arange(
                    next_ephemeral_id,
                    next_ephemeral_id - ephemeral_count,
                    step=-1,
                    dtype=torch.long,
                    device=model.device,
                )
                next_ephemeral_id -= ephemeral_count
            positions.append(prediction.positions[keep].detach().cpu().numpy())
            identities.append(output_ids[keep].detach().cpu().numpy())
            tracks = next_tracks

        truth_positions = [
            np.asarray(frame.states[:, :2], dtype=np.float64)
            for frame in frames[first_scored_frame:]
        ]
        truth_ids = [
            np.asarray(frame.target_ids, dtype=np.int64)
            for frame in frames[first_scored_frame:]
        ]
        value = trajectory_gospa(
            truth_positions,
            truth_ids,
            positions[first_scored_frame:],
            identities[first_scored_frame:],
            c=2.0,
            p=1.0,
            gamma=1.0,
        )
        frames_scored = len(truth_positions)
        records.append(
            {
                "tgospa": value.total / frames_scored,
                "localization": value.localization / frames_scored,
                "missed": value.missed / frames_scored,
                "false": value.false / frames_scored,
                "switch": value.switch / frames_scored,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method", default="Track-MT3-CM")
    parser.add_argument("--expected-step", type=int, default=12_000)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--existence-threshold", type=float, required=True)
    parser.add_argument("--tracking-threshold", type=float, required=True)
    parser.add_argument("--first-scored-frame", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_merged_config(args.config, *args.overlay)
    config.evaluation.existence_threshold = args.existence_threshold
    config.model.detection_threshold = args.existence_threshold
    config.model.tracking_threshold = args.tracking_threshold
    device = torch.device(resolve_device(config.training.device))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    checkpoint_step = int(checkpoint.get("training_state", {}).get("step", -1))
    if checkpoint_step != args.expected_step:
        raise ValueError(
            f"expected step {args.expected_step}, got {checkpoint_step}"
        )
    model = TrackMT3(config).to(device).eval()
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)

    scenes = {}
    for scene in SCENES:
        paths = sorted((args.dataset_dir / scene).glob("run_*.npz"))[: args.runs]
        if len(paths) != args.runs:
            raise FileNotFoundError(
                f"expected {args.runs} trajectories in {args.dataset_dir / scene}"
            )
        trajectories = [load_trajectory(path) for path in paths]
        frame_metrics = evaluate_model(model, config, trajectories=trajectories)
        trajectory_metrics = native_tgospa(
            model, config, trajectories, args.first_scored_frame
        )
        scenes[scene] = {
            "gospa": float(frame_metrics.gospa.mean()),
            "pro_gospa": float(frame_metrics.pro_gospa.mean()),
            "tgospa": float(np.mean([row["tgospa"] for row in trajectory_metrics])),
            "tgospa_per_run": trajectory_metrics,
        }
        print(f"{scene}: {json.dumps(scenes[scene])}", flush=True)

    summary = {
        "method": args.method,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_step,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "operating_point": {
            "existence": args.existence_threshold,
            "qtm_detection": args.existence_threshold,
            "qtm_tracking": args.tracking_threshold,
        },
        "runs_per_scene": args.runs,
        "first_scored_frame": args.first_scored_frame,
        "scenes": scenes,
        "mean_gospa": float(np.mean([row["gospa"] for row in scenes.values()])),
        "mean_pro_gospa": float(
            np.mean([row["pro_gospa"] for row in scenes.values()])
        ),
        "mean_tgospa": float(np.mean([row["tgospa"] for row in scenes.values()])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
