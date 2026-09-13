#!/usr/bin/env python3
"""Measure unified batch-one, frame-by-frame tracker inference latency."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from track_mt3.config_merge import load_merged_config
from track_mt3.data.window import build_sliding_windows
from track_mt3.evaluation import load_operating_point, load_trajectory
from track_mt3.models import TrackMT3
from track_mt3.models_v17 import EndToEndRecursiveTracker, V17BConfig
from track_mt3.models_v17.batching import pad_current_frames
from track_mt3.tracking import OnlineTracker


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summary(samples: list[float]) -> dict[str, float | int]:
    milliseconds = np.asarray(samples) * 1000.0
    return {
        "mean_ms_per_frame": float(milliseconds.mean()),
        "std_ms_per_frame": float(milliseconds.std(ddof=0)),
        "p50_ms_per_frame": float(np.percentile(milliseconds, 50)),
        "p95_ms_per_frame": float(np.percentile(milliseconds, 95)),
        "measured_frames": len(samples),
    }


@torch.inference_mode()
def benchmark_track_mt3(
    device: torch.device,
    frames: list,
    repeats: int,
    config_path: Path | None,
    checkpoint_path: Path,
    first_scored_frame: int,
) -> dict:
    config_paths = ["configs/paper.yaml", "configs/training/track_mt3.yaml"]
    if config_path is not None:
        config_paths.append(config_path)
    config = load_merged_config(*config_paths)
    model = TrackMT3(config).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    windows = [
        window.to(device)
        for window in build_sliding_windows(frames, config.model.window_size)
    ]
    samples: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(repeats):
        tracker = OnlineTracker(model)
        for window in windows:
            synchronize(device)
            started = time.perf_counter()
            tracker.step(window)
            synchronize(device)
            # A full window ending at frame 20 corresponds to the first paper
            # scoring frame. Using the window-list index here would silently
            # time frames 39--99 and give Track-MT3 fewer samples than CNSF.
            if window.end_index >= first_scored_frame:
                samples.append(time.perf_counter() - started)
    result = summary(samples)
    result.update(
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        config_overlay=str(config_path) if config_path is not None else None,
        historical_measurement_frames=config.model.window_size,
        peak_cuda_memory_mb=(
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else None
        ),
    )
    return result


@torch.inference_mode()
def benchmark_cnsf(
    device: torch.device,
    frames: list,
    repeats: int,
    config_path: Path,
    checkpoint_path: Path,
    compile_association: bool,
    first_scored_frame: int,
    candidate_threshold: float,
    existence_threshold: float,
    retention_threshold: float,
    confirmation_hits: int,
    survival_warmup_frames: int,
) -> dict:
    config = yaml.safe_load(config_path.read_text())
    model_config = V17BConfig(**config["model"])
    model = EndToEndRecursiveTracker(model_config).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if compile_association:
        model.association = torch.compile(
            model.association, mode="reduce-overhead", dynamic=True
        )
    batches = [pad_current_frames([frame], device) for frame in frames]
    samples: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(repeats):
        state = None
        for index, batch in enumerate(batches):
            measurements, mask, _, frame_time, _, _ = batch
            synchronize(device)
            started = time.perf_counter()
            output, state = model.forward_inference_step_fast(
                measurements,
                mask,
                frame_time,
                state,
                birth_candidate_threshold=candidate_threshold,
                confirmation_hits=confirmation_hits,
                retention_threshold=retention_threshold,
                survival_warmup_frames=survival_warmup_frames,
            )
            probability = output.existing_logits[0].sigmoid()
            keep = (
                output.existing_mask[0]
                & state.confirmed_mask[0]
                & (probability >= existence_threshold)
            )
            output.posterior_mean[0, keep, :2].cpu()
            synchronize(device)
            if index >= first_scored_frame:
                samples.append(time.perf_counter() - started)
    result = summary(samples)
    result.update(
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        historical_measurement_frames=1,
        sinkhorn_iterations=model_config.association_sinkhorn_iterations,
        compiled_association=compile_association,
        peak_cuda_memory_mb=(
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else None
        ),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--models", default="track_mt3,cnsf",
        help="comma-separated track_mt3,cnsf (Track-MT3-CM uses Track-MT3 options)",
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--trajectory",
        default="datasets/track_mt3_paper/evaluation/scenario3/run_000.npz",
    )
    parser.add_argument(
        "--track-mt3-checkpoint", default="outputs/track_mt3/checkpoints/best.pt"
    )
    parser.add_argument(
        "--track-mt3-config",
        type=Path,
        help="optional overlay applied after the Track-MT3 configuration",
    )
    parser.add_argument(
        "--cnsf-config",
        default="configs/training/cnsf_exact12k.yaml",
    )
    parser.add_argument(
        "--cnsf-checkpoint",
        default="outputs/cnsf_exact12k/checkpoints/step_012000.pt",
    )
    parser.add_argument("--compile-association", action="store_true")
    parser.add_argument(
        "--operating-points",
        default="configs/evaluation/operating_points.yaml",
    )
    parser.add_argument("--first-scored-frame", type=int, default=20)
    parser.add_argument("--candidate-threshold", type=float)
    parser.add_argument("--existence-threshold", type=float)
    parser.add_argument("--retention-threshold", type=float)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    cnsf_point = load_operating_point(arguments.operating_points, "cnsf")
    candidate_threshold = (
        arguments.candidate_threshold
        if arguments.candidate_threshold is not None
        else float(cnsf_point["candidate"])
    )
    existence_threshold = (
        arguments.existence_threshold
        if arguments.existence_threshold is not None
        else float(cnsf_point["existence"])
    )
    retention_threshold = (
        arguments.retention_threshold
        if arguments.retention_threshold is not None
        else float(cnsf_point["retention"])
    )
    confirmation_hits = int(cnsf_point["confirmation_hits"])
    survival_warmup_frames = int(cnsf_point["survival_warmup_frames"])

    selected = [name.strip() for name in arguments.models.split(",")]
    allowed = {"track_mt3", "cnsf"}
    if not selected or any(name not in allowed for name in selected):
        raise ValueError(f"--models must contain only {sorted(allowed)}")
    torch.set_num_threads(arguments.threads)
    torch.set_num_interop_threads(1)
    device = torch.device(arguments.device)
    frames = load_trajectory(arguments.trajectory)
    record: dict = {
        "device": str(device),
        "torch_threads": arguments.threads,
        "batch_size": 1,
        "trajectory": arguments.trajectory,
        "repeats": arguments.repeats,
        "first_scored_frame": arguments.first_scored_frame,
        "timing_boundary": "in-memory model input to final thresholded track outputs",
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    if "track_mt3" in selected:
        record["track_mt3"] = benchmark_track_mt3(
            device,
            frames,
            arguments.repeats,
            arguments.track_mt3_config,
            Path(arguments.track_mt3_checkpoint),
            arguments.first_scored_frame,
        )
    if "cnsf" in selected:
        record["cnsf"] = benchmark_cnsf(
            device,
            frames,
            arguments.repeats,
            Path(arguments.cnsf_config),
            Path(arguments.cnsf_checkpoint),
            arguments.compile_association,
            arguments.first_scored_frame,
            candidate_threshold,
            existence_threshold,
            retention_threshold,
            confirmation_hits,
            survival_warmup_frames,
        )
    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
