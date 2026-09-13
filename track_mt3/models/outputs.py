from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class LayerPrediction:
    normalized_positions: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor


@dataclass
class ProposalOutputs:
    """Batched encoder scores over every measurement, for the two-stage path.

    There is one entry per padded measurement slot rather than per query, so the
    auxiliary proposal loss has to slice each batch row down to its true length
    before matching against targets.
    """

    logits: torch.Tensor
    normalized_positions: torch.Tensor
    positions: torch.Tensor
    padding_mask: torch.Tensor

    def row(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unpadded (normalized_positions, positions, logits) of one batch entry."""
        keep = ~self.padding_mask[index]
        return (
            self.normalized_positions[index][keep],
            self.positions[index][keep],
            self.logits[index][keep],
        )


@dataclass
class FramePrediction:
    normalized_positions: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor
    hidden: torch.Tensor
    num_track_queries: int
    auxiliary: Tuple[LayerPrediction, ...] = ()
    attention_maps: torch.Tensor | None = None

    @property
    def probabilities(self) -> torch.Tensor:
        return self.logits.sigmoid()

    @property
    def num_detection_queries(self) -> int:
        return self.positions.shape[-2] - self.num_track_queries

