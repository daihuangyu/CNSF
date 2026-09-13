from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch

from track_mt3.config import ExperimentConfig
from track_mt3.data.simulator import Frame, MultiTargetSimulator
from track_mt3.data.window import build_sliding_windows
from track_mt3.metrics import gospa, pro_gospa
from track_mt3.tracking import OnlineTracker


@dataclass
class EvaluationResult:
    gospa: np.ndarray
    pro_gospa: np.ndarray
    localization: np.ndarray
    missed: np.ndarray
    false: np.ndarray
    mean_probability: np.ndarray
    max_probability: np.ndarray
    predicted_count: np.ndarray
    target_count: np.ndarray
    matched_count: np.ndarray

    def summary(self) -> dict[str, float]:
        return {
            "gospa": float(self.gospa.mean()),
            "pro_gospa": float(self.pro_gospa.mean()),
            "localization": float(self.localization.mean()),
            "missed": float(self.missed.mean()),
            "false": float(self.false.mean()),
            "mean_probability": float(self.mean_probability.mean()),
            "max_probability": float(self.max_probability.mean()),
            "predicted_count": float(self.predicted_count.mean()),
            "target_count": float(self.target_count.mean()),
            "matched_count": float(self.matched_count.mean()),
        }

    def detailed_summary(self) -> dict[str, float]:
        summary = self.summary()
        for name in ("gospa", "pro_gospa", "localization", "missed", "false"):
            values = getattr(self, name)
            summary[f"{name}_std"] = float(values.std())
            summary[f"{name}_p50"] = float(np.percentile(values, 50))
            summary[f"{name}_p90"] = float(np.percentile(values, 90))
        return summary


def load_trajectory(path: str | Path) -> list[Frame]:
    data = np.load(path)
    frames = []
    has_ages = "target_ages" in data
    for index, (frame_index, frame_time) in enumerate(
        zip(data["frame_indices"], data["times"])
    ):
        state_start, state_end = data["state_offsets"][index : index + 2]
        measurement_start, measurement_end = data["measurement_offsets"][index : index + 2]
        n_targets = state_end - state_start
        frames.append(
            Frame(
                index=int(frame_index),
                time=float(frame_time),
                states=data["states"][state_start:state_end],
                target_ids=data["target_ids"][state_start:state_end],
                target_ages=data["target_ages"][state_start:state_end] if has_ages else np.zeros(n_targets, dtype=np.int64),
                measurements=data["measurements"][measurement_start:measurement_end],
                measurement_ids=data["measurement_ids"][measurement_start:measurement_end],
            )
        )
    return frames


@torch.no_grad()
def evaluate_model(
    model,
    config: ExperimentConfig,
    *,
    seed: int = 10_000,
    trajectories: Sequence[Sequence[Frame]] | None = None,
) -> EvaluationResult:
    model.eval()
    all_gospa: List[List[float]] = []
    all_pro: List[List[float]] = []
    all_localization: List[List[float]] = []
    all_missed: List[List[float]] = []
    all_false: List[List[float]] = []
    all_mean_probability: List[List[float]] = []
    all_max_probability: List[List[float]] = []
    all_predicted_count: List[List[float]] = []
    all_target_count: List[List[float]] = []
    all_matched_count: List[List[float]] = []
    if trajectories is None:
        trajectories = [
            MultiTargetSimulator(config.simulation, seed + run).simulate(
                config.evaluation.trajectory_steps
            )
            for run in range(config.evaluation.monte_carlo_runs)
        ]
    batch_size = max(1, config.evaluation.batch_size)
    for batch_start in range(0, len(trajectories), batch_size):
        trajectory_batch = trajectories[batch_start : batch_start + batch_size]
        window_batches = [
            build_sliding_windows(frames, config.model.window_size)
            for frames in trajectory_batch
        ]
        if not window_batches or any(
            len(windows) != len(window_batches[0]) for windows in window_batches
        ):
            raise ValueError("evaluation trajectories must have equal lengths")
        tracks = [model.empty_tracks() for _ in window_batches]
        next_track_ids = [0 for _ in window_batches]
        batch_metrics = [
            {
                "gospa": [], "pro": [], "localization": [], "missed": [], "false": [],
                "mean_probability": [], "max_probability": [], "predicted_count": [],
                "target_count": [], "matched_count": [],
            }
            for _ in window_batches
        ]
        for index in range(len(window_batches[0])):
            windows = [run[index] for run in window_batches]
            predictions = model.forward_windows(windows, tracks)
            next_tracks = []
            for run_index, (window, prediction, track) in enumerate(
                zip(windows, predictions, tracks)
            ):
                if config.model.propagate_tracks:
                    query_track_ids = torch.full(
                        (len(prediction.positions),),
                        -1,
                        dtype=torch.long,
                        device=model.device,
                    )
                    query_track_ids[: len(track)] = track.track_ids
                    next_track, _ = model.qtm(
                        prediction, track, query_track_ids=query_track_ids
                    )
                    new_mask = next_track.track_ids < 0
                    new_count = int(new_mask.sum())
                    if new_count:
                        start = next_track_ids[run_index]
                        next_track.track_ids[new_mask] = torch.arange(
                            start,
                            start + new_count,
                            dtype=torch.long,
                            device=model.device,
                        )
                        next_track_ids[run_index] += new_count
                    next_tracks.append(next_track)
                if index < config.evaluation.warmup_windows:
                    continue
                probabilities = prediction.probabilities.squeeze(-1).cpu().numpy()
                positions = prediction.positions.cpu().numpy()
                targets = window.target_positions.numpy()
                valid = probabilities > config.evaluation.existence_threshold
                result = gospa(
                    positions[valid],
                    targets,
                    cutoff=config.evaluation.gospa_cutoff,
                    order=config.evaluation.gospa_order,
                    alpha=config.evaluation.gospa_alpha,
                )
                metrics = batch_metrics[run_index]
                metrics["gospa"].append(result.value)
                metrics["localization"].append(result.localization)
                metrics["missed"].append(result.missed)
                metrics["false"].append(result.false)
                metrics["mean_probability"].append(float(probabilities.mean()))
                metrics["max_probability"].append(float(probabilities.max()))
                metrics["predicted_count"].append(float(valid.sum()))
                metrics["target_count"].append(float(len(targets)))
                metrics["matched_count"].append(float(len(result.matched_pairs)))
                metrics["pro"].append(
                    pro_gospa(
                        positions,
                        probabilities,
                        targets,
                        cutoff=config.evaluation.gospa_cutoff,
                        order=config.evaluation.gospa_order,
                        alpha=config.evaluation.gospa_alpha,
                    )
                )
            if config.model.propagate_tracks:
                tracks = next_tracks
        for metrics in batch_metrics:
            all_gospa.append(metrics["gospa"])
            all_pro.append(metrics["pro"])
            all_localization.append(metrics["localization"])
            all_missed.append(metrics["missed"])
            all_false.append(metrics["false"])
            all_mean_probability.append(metrics["mean_probability"])
            all_max_probability.append(metrics["max_probability"])
            all_predicted_count.append(metrics["predicted_count"])
            all_target_count.append(metrics["target_count"])
            all_matched_count.append(metrics["matched_count"])
    return EvaluationResult(
        gospa=np.asarray(all_gospa),
        pro_gospa=np.asarray(all_pro),
        localization=np.asarray(all_localization),
        missed=np.asarray(all_missed),
        false=np.asarray(all_false),
        mean_probability=np.asarray(all_mean_probability),
        max_probability=np.asarray(all_max_probability),
        predicted_count=np.asarray(all_predicted_count),
        target_count=np.asarray(all_target_count),
        matched_count=np.asarray(all_matched_count),
    )
