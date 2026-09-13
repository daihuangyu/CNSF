from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from track_mt3.data.simulator import Frame
from track_mt3.tracking.online_tracker import TrackingOutput


def plot_metric_curves(
    curves: Mapping[str, np.ndarray], output: str | Path, *, ylabel: str, dt: float = 0.1
) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    for name, values in curves.items():
        values = np.asarray(values)
        mean = values.mean(axis=0) if values.ndim == 2 else values
        x = np.arange(1, len(mean) + 1) * dt
        axis.plot(x, mean, label=name)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_trajectories(
    frames: Sequence[Frame], outputs: Sequence[TrackingOutput], output: str | Path
) -> None:
    figure, axis = plt.subplots(figsize=(7, 7))
    true_tracks: dict[int, list[np.ndarray]] = {}
    for frame in frames:
        for target_id, state in zip(frame.target_ids, frame.states):
            true_tracks.setdefault(int(target_id), []).append(state[:2])
    for target_id, values in true_tracks.items():
        points = np.asarray(values)
        axis.plot(points[:, 0], points[:, 1], "o-", markersize=2, alpha=0.7, label=f"true {target_id}")
    predicted_tracks: dict[int, list[np.ndarray]] = {}
    for result in outputs:
        for track_id, position in zip(result.track_ids, result.positions):
            predicted_tracks.setdefault(int(track_id), []).append(position.numpy())
    for track_id, values in predicted_tracks.items():
        points = np.asarray(values)
        axis.plot(points[:, 0], points[:, 1], "x--", markersize=3, label=f"pred {track_id}")
    axis.set(xlabel="X (m)", ylabel="Y (m)", xlim=(-10, 10), ylim=(-10, 10))
    axis.grid(alpha=0.25)
    if len(true_tracks) + len(predicted_tracks) <= 16:
        axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_target_counts(
    frames: Sequence[Frame], outputs: Sequence[TrackingOutput], output: str | Path, *, window_size: int
) -> None:
    evaluated_frames = frames[window_size - 1 : window_size - 1 + len(outputs)]
    times = np.asarray([frame.time for frame in evaluated_frames])
    true_counts = np.asarray([len(frame.target_ids) for frame in evaluated_frames])
    predicted_counts = np.asarray([len(item.track_ids) for item in outputs])
    figure, axis = plt.subplots(figsize=(7, 3.5))
    axis.step(times, true_counts, where="post", label="True")
    axis.step(times, predicted_counts, where="post", label="Track-MT3")
    axis.set(xlabel="Time (s)", ylabel="Number of targets")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
