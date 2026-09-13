#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from track_mt3.config_merge import load_merged_config
from track_mt3.data import Frame, MultiTargetSimulator


PAPER_TABLE_1 = {
    "total_measurements": 401_651_991,
    "target_measurements": 81_664_937,
    "clutter_measurements": 319_987_054,
    "average_measurements_per_batch": 8_034,
    "average_measurements_per_window": 252,
}


def frame_offsets(lengths: list[int]) -> np.ndarray:
    return np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths, dtype=np.int64)))


def save_trajectory(path: Path, frames: list[Frame]) -> None:
    state_lengths = [len(frame.states) for frame in frames]
    measurement_lengths = [len(frame.measurements) for frame in frames]
    states = np.concatenate([frame.states for frame in frames], axis=0)
    target_ids = np.concatenate([frame.target_ids for frame in frames], axis=0)
    measurements = np.concatenate([frame.measurements for frame in frames], axis=0)
    measurement_ids = np.concatenate([frame.measurement_ids for frame in frames], axis=0)
    np.savez_compressed(
        path,
        frame_indices=np.asarray([frame.index for frame in frames], dtype=np.int64),
        times=np.asarray([frame.time for frame in frames], dtype=np.float64),
        state_offsets=frame_offsets(state_lengths),
        states=states,
        target_ids=target_ids,
        measurement_offsets=frame_offsets(measurement_lengths),
        measurements=measurements,
        measurement_ids=measurement_ids,
    )


def audit_training_stream(config, samples: int) -> dict[str, float | int]:
    seed_sequence = np.random.SeedSequence(config.training.seed)
    target_measurements = 0
    clutter_measurements = 0
    for child_seed in seed_sequence.spawn(samples):
        frames = MultiTargetSimulator(config.simulation, child_seed).simulate(config.model.window_size)
        for frame in frames:
            clutter = int((frame.measurement_ids < 0).sum())
            clutter_measurements += clutter
            target_measurements += len(frame.measurement_ids) - clutter
    total = target_measurements + clutter_measurements
    return {
        "audited_windows": samples,
        "total_measurements": total,
        "target_measurements": target_measurements,
        "clutter_measurements": clutter_measurements,
        "average_measurements_per_window": total / samples,
        "projected_average_measurements_per_batch_32": total / samples * 32,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic Track-MT3 simulation data")
    parser.add_argument("--base-config", default="configs/paper.yaml")
    parser.add_argument("--output-dir", default="datasets/track_mt3_paper")
    parser.add_argument("--audit-windows", type=int, default=10_000)
    parser.add_argument("--evaluation-seed", type=int, default=10_000)
    arguments = parser.parse_args()

    output_dir = Path(arguments.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty dataset directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_merged_config(arguments.base_config)
    manifest = {
        "paper_table_1": PAPER_TABLE_1,
        "training": {
            "storage": "online_random_simulation",
            "seed": base_config.training.seed,
            "updates": base_config.training.updates,
            "reported_batch_size": base_config.training.batch_size,
            "window_steps": base_config.model.window_size,
            "independent_windows_over_training": (
                base_config.training.updates * base_config.training.batch_size
            ),
            "parameters": base_config.to_dict()["simulation"],
            "audit": audit_training_stream(base_config, arguments.audit_windows),
        },
        "evaluation": {},
    }

    for scene in (1, 2, 3):
        config = load_merged_config(
            arguments.base_config,
            f"configs/experiments/scenario{scene}.yaml",
        )
        scene_dir = output_dir / "evaluation" / f"scenario{scene}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        seeds = []
        for run in range(config.evaluation.monte_carlo_runs):
            seed = arguments.evaluation_seed + run
            frames = MultiTargetSimulator(config.simulation, seed).simulate(
                config.evaluation.trajectory_steps
            )
            save_trajectory(scene_dir / f"run_{run:03d}.npz", frames)
            seeds.append(seed)
        manifest["evaluation"][f"scenario{scene}"] = {
            "runs": config.evaluation.monte_carlo_runs,
            "trajectory_steps": config.evaluation.trajectory_steps,
            "seeds": seeds,
            "parameters": config.to_dict()["simulation"],
        }

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
