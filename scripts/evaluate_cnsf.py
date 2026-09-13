#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product
import json
from pathlib import Path

import numpy as np
import torch
import yaml

import _bootstrap  # noqa: F401

from track_mt3.evaluation import load_trajectory
from track_mt3.metrics import gospa, pro_gospa
from track_mt3.models_v17 import (
    EndToEndRecursiveTracker,
    JointAssociationTracker,
    SeparatedLifecycleTracker,
    V17BConfig,
)
from track_mt3.models_v17.batching import pad_current_frames


@torch.no_grad()
def evaluate_scene(
    model,
    trajectories,
    *,
    candidate_threshold: float,
    output_threshold: float,
    existence_threshold: float,
    confirmation_hits: int,
    first_scored_frame: int,
    death_state_mode: str,
    retention_threshold: float,
    survival_warmup_frames: int,
    confirmed_association_bias: float,
    truth_free_model_input: bool,
    fast_inference: bool = False,
) -> dict[str, dict[str, float]]:
    model.eval()
    state = None
    modes = {
        "confirmed": {name: [] for name in ("gospa", "pro_gospa", "localization", "missed", "false")},
        "immediate": {name: [] for name in ("gospa", "pro_gospa", "localization", "missed", "false")},
    }
    time_bins: dict[str, dict[str, dict[str, list[float]]]] = {}
    false_active = []
    tentative_active = []
    confirmed_active = []
    active_counts = []
    for time_index in range(len(trajectories[0])):
        batch = pad_current_frames(
            [trajectory[time_index] for trajectory in trajectories], model.device
        )
        measurements, mask, measurement_ids, frame_time, truth_states, truth_ids = batch
        if truth_free_model_input:
            inference_step = (
                model.forward_inference_step_fast
                if fast_inference
                else model.forward_inference_step
            )
            output, state = inference_step(
                measurements,
                mask,
                frame_time,
                state,
                birth_candidate_threshold=candidate_threshold,
                confirmation_hits=confirmation_hits,
                retention_threshold=retention_threshold,
                survival_warmup_frames=survival_warmup_frames,
            )
        else:
            output, state = model.forward_step(
                measurements,
                mask,
                measurement_ids,
                frame_time,
                truth_ids,
                state,
                association_oracle_probability=0.0,
                association_update_mode="moment",
                birth_state_mode="predicted",
                birth_candidate_threshold=candidate_threshold,
                confirmation_hits=confirmation_hits,
                death_state_mode=death_state_mode,
                retention_threshold=retention_threshold,
                survival_warmup_frames=survival_warmup_frames,
                confirmed_association_bias=confirmed_association_bias,
            )
        if time_index < first_scored_frame:
            continue
        bin_start = first_scored_frame + (
            (time_index - first_scored_frame) // 20
        ) * 20
        bin_end = min(bin_start + 19, len(trajectories[0]) - 1)
        bin_name = f"{bin_start:02d}-{bin_end:02d}"
        bin_metrics = time_bins.setdefault(
            bin_name,
            {
                mode: {
                    name: []
                    for name in (
                        "gospa",
                        "pro_gospa",
                        "localization",
                        "missed",
                        "false",
                    )
                }
                for mode in modes
            },
        )
        for row, targets in enumerate(truth_states):
            existing_probability = output.existing_logits[row].sigmoid()
            confirmed_existing = (
                output.existing_mask[row]
                & state.confirmed_mask[row]
                & (existing_probability >= existence_threshold)
            )
            immediate_birth = output.birth_mask[row] & (
                output.birth_logits[row].sigmoid() >= output_threshold
            )
            target_positions = targets[:, :2].detach().cpu().numpy()
            for mode, include_births in (("confirmed", False), ("immediate", True)):
                positions = output.posterior_mean[row, confirmed_existing, :2]
                probabilities = existing_probability[confirmed_existing]
                if include_births:
                    positions = torch.cat(
                        (positions, output.birth_positions[row, immediate_birth])
                    )
                    probabilities = torch.cat(
                        (probabilities, output.birth_logits[row, immediate_birth].sigmoid())
                    )
                position_array = positions.detach().cpu().numpy()
                probability_array = probabilities.detach().cpu().numpy()
                result = gospa(position_array, target_positions)
                modes[mode]["gospa"].append(result.value)
                pro_value = pro_gospa(
                    position_array, probability_array, target_positions
                )
                modes[mode]["pro_gospa"].append(pro_value)
                modes[mode]["localization"].append(result.localization)
                modes[mode]["missed"].append(result.missed)
                modes[mode]["false"].append(result.false)
                bin_metrics[mode]["gospa"].append(result.value)
                bin_metrics[mode]["pro_gospa"].append(pro_value)
                bin_metrics[mode]["localization"].append(result.localization)
                bin_metrics[mode]["missed"].append(result.missed)
                bin_metrics[mode]["false"].append(result.false)
            active = state.active_mask[row]
            active_counts.append(float(active.sum()))
            if not truth_free_model_input:
                false_active.append(
                    float((active & (state.supervision_ids[row] < 0)).sum())
                )
            tentative_active.append(
                float((active & ~state.confirmed_mask[row]).sum())
            )
            confirmed_active.append(float((active & state.confirmed_mask[row]).sum()))
    return {
        **{
            mode: {name: float(np.mean(values)) for name, values in metrics.items()}
            for mode, metrics in modes.items()
        },
        "state": {
            "false_active": (
                float(np.mean(false_active)) if false_active else None
            ),
            "tentative_active": float(np.mean(tentative_active)),
            "confirmed_active": float(np.mean(confirmed_active)),
            "active_max": float(np.max(active_counts)),
            "slot_saturation_fraction": float(
                np.mean(np.asarray(active_counts) >= model.config.max_tracks)
            ),
        },
        "time_bins": {
            bin_name: {
                mode: {
                    name: float(np.mean(values))
                    for name, values in metrics.items()
                }
                for mode, metrics in bin_modes.items()
            }
            for bin_name, bin_modes in time_bins.items()
        },
    }


def _comma_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",")]


def _comma_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate learned tentative births while death remains oracle"
    )
    parser.add_argument(
        "--config", default="configs/training/cnsf_exact12k.yaml"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--scenes",
        default="scenario1,scenario2,scenario3",
        help="comma-separated evaluation scenes",
    )
    parser.add_argument("--candidate-thresholds", default="0.35,0.5,0.65")
    parser.add_argument("--output-thresholds", default="0.68")
    parser.add_argument("--existence-thresholds", default="0.5")
    parser.add_argument("--confirmation-hits", default="1,2")
    parser.add_argument("--death-state-modes", default="oracle_pre")
    parser.add_argument("--retention-thresholds", default="0.5")
    parser.add_argument("--survival-warmup-frames", type=int, default=0)
    parser.add_argument("--confirmed-association-biases", default="0.0")
    parser.add_argument("--truth-free-model-input", action="store_true")
    parser.add_argument("--fast-inference", action="store_true")
    parser.add_argument("--inference-sinkhorn-iterations", type=int)
    parser.add_argument("--first-scored-frame", type=int, default=20)
    parser.add_argument(
        "--output", default="outputs/cnsf/evaluation.json"
    )
    arguments = parser.parse_args()
    if arguments.fast_inference and not arguments.truth_free_model_input:
        raise ValueError("--fast-inference requires --truth-free-model-input")
    with Path(arguments.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage = config["training"].get("stage")
    if stage == "end_to_end":
        model_class = EndToEndRecursiveTracker
    elif stage == "survival":
        model_class = SeparatedLifecycleTracker
    else:
        model_class = JointAssociationTracker
    model_config = V17BConfig(**config["model"])
    if arguments.inference_sinkhorn_iterations is not None:
        model_config = replace(
            model_config,
            association_sinkhorn_iterations=arguments.inference_sinkhorn_iterations,
        )
    model = model_class(model_config).to(device)
    checkpoint = torch.load(arguments.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    dataset_dir = Path(config["training"]["evaluation_dataset_dir"])
    selected_scenes = arguments.scenes.split(",")
    allowed_scenes = {"scenario1", "scenario2", "scenario3"}
    if not selected_scenes or any(scene not in allowed_scenes for scene in selected_scenes):
        raise ValueError(f"--scenes must contain only {sorted(allowed_scenes)}")
    datasets = {
        scene: [
            load_trajectory(path)
            for path in sorted((dataset_dir / scene).glob("run_*.npz"))[: arguments.runs]
        ]
        for scene in selected_scenes
    }
    records = []
    if arguments.truth_free_model_input:
        if arguments.death_state_modes != "predicted":
            raise ValueError("truth-free evaluation requires predicted death")
        if any(
            value != 0.0
            for value in _comma_floats(arguments.confirmed_association_biases)
        ):
            raise ValueError("truth-free evaluation forbids manual association bias")
    settings = product(
        _comma_floats(arguments.candidate_thresholds),
        _comma_floats(arguments.output_thresholds),
        _comma_floats(arguments.existence_thresholds),
        _comma_ints(arguments.confirmation_hits),
        arguments.death_state_modes.split(","),
        _comma_floats(arguments.confirmed_association_biases),
        _comma_floats(arguments.retention_thresholds),
    )
    for (
        candidate_threshold,
        output_threshold,
        existence_threshold,
        confirmation_hits,
        death_state_mode,
        confirmed_association_bias,
        retention_threshold,
    ) in settings:
        scenes = {
            scene: evaluate_scene(
                model,
                trajectories,
                candidate_threshold=candidate_threshold,
                output_threshold=output_threshold,
                existence_threshold=existence_threshold,
                confirmation_hits=confirmation_hits,
                first_scored_frame=arguments.first_scored_frame,
                death_state_mode=death_state_mode,
                retention_threshold=retention_threshold,
                survival_warmup_frames=arguments.survival_warmup_frames,
                confirmed_association_bias=confirmed_association_bias,
                truth_free_model_input=arguments.truth_free_model_input,
                fast_inference=arguments.fast_inference,
            )
            for scene, trajectories in datasets.items()
        }
        record = {
            "checkpoint_step": int(checkpoint["step"]),
            "candidate_threshold": candidate_threshold,
            "output_threshold": output_threshold,
            "existence_threshold": existence_threshold,
            "confirmation_hits": confirmation_hits,
            "death_state_mode": death_state_mode,
            "retention_threshold": retention_threshold,
            "survival_warmup_frames": arguments.survival_warmup_frames,
            "confirmed_association_bias": confirmed_association_bias,
            "truth_free_model_input": arguments.truth_free_model_input,
            "fast_inference": arguments.fast_inference,
            "inference_sinkhorn_iterations": model_config.association_sinkhorn_iterations,
            "mean_confirmed_gospa": float(
                np.mean(
                    [value["confirmed"]["gospa"] for value in scenes.values()]
                )
            ),
            "mean_confirmed_pro_gospa": float(
                np.mean(
                    [
                        value["confirmed"]["pro_gospa"]
                        for value in scenes.values()
                    ]
                )
            ),
            "mean_immediate_gospa": float(
                np.mean(
                    [value["immediate"]["gospa"] for value in scenes.values()]
                )
            ),
            "mean_immediate_pro_gospa": float(
                np.mean(
                    [
                        value["immediate"]["pro_gospa"]
                        for value in scenes.values()
                    ]
                )
            ),
            "scenes": scenes,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
