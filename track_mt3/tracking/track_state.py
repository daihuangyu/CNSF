from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TrackState:
    features: torch.Tensor
    references: torch.Tensor
    target_ids: torch.Tensor
    track_ids: torch.Tensor
    ages: torch.Tensor
    ssm_states: torch.Tensor | None = None
    ssm_slot_map: dict | None = None
    memory: torch.Tensor | None = None
    measurement_cache: torch.Tensor | None = None
    measurement_cache_mask: torch.Tensor | None = None
    measurement_cache_positions: torch.Tensor | None = None
    measurement_cache_ids: torch.Tensor | None = None

    @classmethod
    def empty(
        cls,
        hidden_dim: int,
        device: torch.device | str,
        *,
        max_tracks: int = 0,
        track_d_inner: int = 0,
        track_d_state: int = 0,
        memory_size: int = 0,
        cache_frames: int = 0,
        cache_tokens_per_frame: int = 0,
    ) -> "TrackState":
        if max_tracks > 0 and track_d_inner > 0 and track_d_state > 0:
            ssm_states = torch.zeros((max_tracks, track_d_inner, track_d_state), device=device)
            ssm_slot_map: dict = {}
        else:
            ssm_states = None
            ssm_slot_map = None
        memory = torch.zeros((1, memory_size, hidden_dim), device=device) if memory_size > 0 else None
        cache_shape = (1, cache_frames, cache_tokens_per_frame)
        measurement_cache = (
            torch.zeros((*cache_shape, hidden_dim), device=device)
            if cache_frames > 0 and cache_tokens_per_frame > 0 else None
        )
        measurement_cache_mask = (
            torch.ones(cache_shape, dtype=torch.bool, device=device)
            if measurement_cache is not None else None
        )
        measurement_cache_positions = (
            torch.zeros((*cache_shape, hidden_dim), device=device)
            if measurement_cache is not None else None
        )
        measurement_cache_ids = (
            torch.full(cache_shape, -2, dtype=torch.long, device=device)
            if measurement_cache is not None else None
        )
        return cls(
            features=torch.empty((0, hidden_dim), device=device),
            references=torch.empty((0, 2), device=device),
            target_ids=torch.empty((0,), dtype=torch.long, device=device),
            track_ids=torch.empty((0,), dtype=torch.long, device=device),
            ages=torch.empty((0,), dtype=torch.long, device=device),
            ssm_states=ssm_states,
            ssm_slot_map=ssm_slot_map,
            memory=memory,
            measurement_cache=measurement_cache,
            measurement_cache_mask=measurement_cache_mask,
            measurement_cache_positions=measurement_cache_positions,
            measurement_cache_ids=measurement_cache_ids,
        )

    def __len__(self) -> int:
        return self.features.shape[0]

    def to(self, device: torch.device | str) -> "TrackState":
        return TrackState(
            features=self.features.to(device),
            references=self.references.to(device),
            target_ids=self.target_ids.to(device),
            track_ids=self.track_ids.to(device),
            ages=self.ages.to(device),
            ssm_states=self.ssm_states.to(device) if self.ssm_states is not None else None,
            ssm_slot_map=self.ssm_slot_map,
            memory=self.memory.to(device) if self.memory is not None else None,
            measurement_cache=self.measurement_cache.to(device) if self.measurement_cache is not None else None,
            measurement_cache_mask=self.measurement_cache_mask.to(device) if self.measurement_cache_mask is not None else None,
            measurement_cache_positions=self.measurement_cache_positions.to(device) if self.measurement_cache_positions is not None else None,
            measurement_cache_ids=self.measurement_cache_ids.to(device) if self.measurement_cache_ids is not None else None,
        )

    def validate(self) -> None:
        size = len(self)
        if self.references.shape != (size, 2):
            raise ValueError("track references must have shape [N, 2]")
        if any(value.shape != (size,) for value in (self.target_ids, self.track_ids, self.ages)):
            raise ValueError("track metadata lengths do not match features")
