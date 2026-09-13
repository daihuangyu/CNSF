from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch

from .simulator import Frame


@dataclass(frozen=True)
class MeasurementWindow:
    end_index: int
    measurements: torch.Tensor
    time_indices: torch.Tensor
    measurement_ids: torch.Tensor
    target_positions: torch.Tensor
    target_ids: torch.Tensor
    target_ages: torch.Tensor = torch.empty(0, dtype=torch.long)

    def to(self, device: torch.device | str) -> "MeasurementWindow":
        return MeasurementWindow(
            end_index=self.end_index,
            measurements=self.measurements.to(device),
            time_indices=self.time_indices.to(device),
            measurement_ids=self.measurement_ids.to(device),
            target_positions=self.target_positions.to(device),
            target_ids=self.target_ids.to(device),
            target_ages=self.target_ages.to(device) if len(self.target_ages) else torch.empty(0, dtype=torch.long, device=device),
        )


def build_sliding_windows(
    frames: Sequence[Frame], window_size: int, *, include_partial: bool = False
) -> List[MeasurementWindow]:
    """Sliding windows over ``frames``, each labelled by its last frame.

    Time offsets are right-aligned: the current frame is always
    ``window_size - 1``. For a full window this is identical to left-aligned
    numbering, so weights trained on full windows stay valid. With
    ``include_partial`` the sequence start also yields windows, covering fewer
    than ``window_size`` frames and therefore using only the highest offsets;
    this is what lets the tracker emit estimates from the very first frame
    instead of discarding the first ``window_size - 1`` of them.
    """
    if window_size < 1:
        raise ValueError("window_size must be positive")
    first_end = 0 if include_partial else window_size - 1
    windows: List[MeasurementWindow] = []
    for end in range(first_end, len(frames)):
        start = max(end - window_size + 1, 0)
        current = frames[start : end + 1]
        offsets = range(window_size - len(current), window_size)
        measurement_parts = [frame.measurements for frame in current]
        id_parts = [frame.measurement_ids for frame in current]
        time_parts = [np.full(len(frame.measurements), offset, dtype=np.int64) for offset, frame in zip(offsets, current)]
        measurements = np.concatenate(measurement_parts, axis=0) if any(len(x) for x in measurement_parts) else np.empty((0, 2), np.float32)
        measurement_ids = np.concatenate(id_parts) if any(len(x) for x in id_parts) else np.empty((0,), np.int64)
        time_indices = np.concatenate(time_parts) if any(len(x) for x in time_parts) else np.empty((0,), np.int64)
        windows.append(
            MeasurementWindow(
                end_index=frames[end].index,
                measurements=torch.from_numpy(measurements),
                time_indices=torch.from_numpy(time_indices),
                measurement_ids=torch.from_numpy(measurement_ids),
                target_positions=torch.from_numpy(frames[end].states[:, :2]),
                target_ids=torch.from_numpy(frames[end].target_ids),
                target_ages=torch.from_numpy(frames[end].target_ages) if len(frames[end].target_ages) else torch.empty(0, dtype=torch.long),
            )
        )
    return windows


def pad_measurement_windows(
    windows: Sequence[MeasurementWindow],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not windows:
        raise ValueError("windows must not be empty")
    max_measurements = max(max(len(window.measurements), 1) for window in windows)
    batch = len(windows)
    measurements = torch.zeros((batch, max_measurements, 2), dtype=torch.float32)
    times = torch.zeros((batch, max_measurements), dtype=torch.long)
    padding_mask = torch.ones((batch, max_measurements), dtype=torch.bool)
    # -2 marks padding so it stays distinct from clutter (-1) and from any target
    # id, which keeps the contrastive objective from pairing padded slots.
    measurement_ids = torch.full((batch, max_measurements), -2, dtype=torch.long)
    for index, window in enumerate(windows):
        count = len(window.measurements)
        if count:
            measurements[index, :count] = window.measurements
            times[index, :count] = window.time_indices
            padding_mask[index, :count] = False
            measurement_ids[index, :count] = window.measurement_ids
        else:
            # A fully masked row makes attention softmax produce NaN, so an empty
            # window keeps one zero-valued slot marked as valid instead. Its id
            # stays -2 so the contrastive objective still ignores it.
            padding_mask[index, 0] = False
    return measurements, times, padding_mask, measurement_ids

