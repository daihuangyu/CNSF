from __future__ import annotations

from collections.abc import Sequence

import torch

from track_mt3.data.simulator import Frame


def pad_current_frames(
    frames: Sequence[Frame], device: torch.device | str
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[torch.Tensor],
    list[torch.Tensor],
]:
    """Batch physical frames without constructing a historical window."""

    if not frames:
        raise ValueError("frames must not be empty")
    width = max(1, max(len(frame.measurements) for frame in frames))
    batch = len(frames)
    measurements = torch.zeros(batch, width, 2, dtype=torch.float32, device=device)
    padding_mask = torch.ones(batch, width, dtype=torch.bool, device=device)
    measurement_ids = torch.full(
        (batch, width), -2, dtype=torch.long, device=device
    )
    for index, frame in enumerate(frames):
        count = len(frame.measurements)
        if count:
            measurements[index, :count] = torch.as_tensor(
                frame.measurements, dtype=torch.float32, device=device
            )
            padding_mask[index, :count] = False
            measurement_ids[index, :count] = torch.as_tensor(
                frame.measurement_ids, dtype=torch.long, device=device
            )
    frame_time = torch.as_tensor(
        [frame.time for frame in frames], dtype=torch.float32, device=device
    )
    truth_states = [
        torch.as_tensor(frame.states, dtype=torch.float32, device=device)
        for frame in frames
    ]
    truth_ids = [
        torch.as_tensor(frame.target_ids, dtype=torch.long, device=device)
        for frame in frames
    ]
    return (
        measurements,
        padding_mask,
        measurement_ids,
        frame_time,
        truth_states,
        truth_ids,
    )
