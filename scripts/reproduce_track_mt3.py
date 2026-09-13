#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401

from track_mt3.config import resolve_device
from track_mt3.config_merge import load_merged_config
from track_mt3.evaluation import evaluate_model, load_trajectory
from track_mt3.models import TrackMT3


EXPECTED_CHECKPOINT_SHA256 = (
    "587c97166f7cbd3f1225efbfedf007cbeb1cbb34201e552ad00f667249a36e07"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the frozen Track-MT3 evaluation")
    parser.add_argument("--checkpoint", default="outputs/track_mt3/checkpoints/best.pt")
    parser.add_argument("--dataset-dir", default="datasets/track_mt3_paper/evaluation")
    parser.add_argument("--output", default="outputs/track_mt3/reproduced_full50.json")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--skip-checksum", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint_sha256 = sha256(checkpoint_path)
    if not args.skip_checksum and checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"unexpected Track-MT3 checkpoint checksum: {checkpoint_sha256}; "
            f"expected {EXPECTED_CHECKPOINT_SHA256}"
        )

    config = load_merged_config("configs/paper.yaml", "configs/training/track_mt3.yaml")
    if config.model.track_confirmation_enabled:
        raise ValueError("Track-MT3 reproduction requires track_confirmation_enabled=false")
    device = torch.device(resolve_device(config.training.device))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    step = int(checkpoint.get("training_state", {}).get("step", -1))
    model = TrackMT3(config).to(device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)

    scenes = {}
    dataset_dir = Path(args.dataset_dir)
    for scene in ("scenario1", "scenario2", "scenario3"):
        paths = sorted((dataset_dir / scene).glob("run_*.npz"))
        if args.max_runs is not None:
            paths = paths[: args.max_runs]
        if not paths:
            raise FileNotFoundError(f"no trajectories found in {dataset_dir / scene}")
        result = evaluate_model(model, config, trajectories=[load_trajectory(p) for p in paths])
        scenes[scene] = result.detailed_summary()

    record = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": step,
        "protocol": "fixed full-50" if args.max_runs is None else f"fixed first-{args.max_runs}",
        "thresholds": {
            "detection": config.model.detection_threshold,
            "tracking": config.model.tracking_threshold,
            "existence": config.evaluation.existence_threshold,
            "track_confirmation_enabled": config.model.track_confirmation_enabled,
        },
        "mean_gospa": float(np.mean([row["gospa"] for row in scenes.values()])),
        "mean_pro_gospa": float(np.mean([row["pro_gospa"] for row in scenes.values()])),
        "scenes": scenes,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
