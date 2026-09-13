from __future__ import annotations

from dataclasses import dataclass

import torch

from track_mt3.data.window import MeasurementWindow
from track_mt3.models.outputs import FramePrediction

from .track_state import TrackState


@dataclass(frozen=True)
class TrackingOutput:
    frame_index: int
    track_ids: torch.Tensor
    positions: torch.Tensor
    probabilities: torch.Tensor
    ages: torch.Tensor
    prediction: FramePrediction


class OnlineTracker:
    def __init__(self, model):
        self.model = model
        self.next_track_id = 0
        self.tracks = model.empty_tracks()

    def reset(self) -> None:
        self.next_track_id = 0
        self.tracks = self.model.empty_tracks()

    @torch.no_grad()
    def step(self, window: MeasurementWindow) -> TrackingOutput:
        self.model.eval()
        prediction = self.model.forward_window(window, self.tracks)
        query_track_ids = torch.full(
            (len(prediction.positions),), -1, dtype=torch.long, device=self.model.device
        )
        if len(self.tracks):
            query_track_ids[: len(self.tracks)] = self.tracks.track_ids
        next_tracks, selected = self.model.qtm(
            prediction,
            self.tracks,
            query_track_ids=query_track_ids,
        )
        new_mask = next_tracks.track_ids < 0
        new_count = int(new_mask.sum())
        if new_count:
            next_tracks.track_ids[new_mask] = torch.arange(
                self.next_track_id,
                self.next_track_id + new_count,
                dtype=torch.long,
                device=self.model.device,
            )
            self.next_track_id += new_count
        self.tracks = next_tracks
        return TrackingOutput(
            frame_index=window.end_index,
            track_ids=next_tracks.track_ids.detach().cpu(),
            positions=prediction.positions[selected].detach().cpu(),
            probabilities=prediction.probabilities[selected].detach().cpu(),
            ages=next_tracks.ages.detach().cpu(),
            prediction=prediction,
        )

