from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from track_mt3.config import LossConfig
from track_mt3.models.outputs import FramePrediction
from track_mt3.tracking.track_state import TrackState


@dataclass(frozen=True)
class CTAResult:
    prediction_indices: torch.Tensor
    target_indices: torch.Tensor
    query_target_ids: torch.Tensor
    inherited_count: int
    newborn_count: int
    newborn_prediction_indices: torch.Tensor = torch.empty(0, dtype=torch.long)

    def assignment_matrix(self, num_predictions: int, num_targets: int) -> torch.Tensor:
        matrix = torch.zeros(
            (num_predictions, num_targets), dtype=torch.bool, device=self.prediction_indices.device
        )
        if len(self.prediction_indices):
            matrix[self.prediction_indices, self.target_indices] = True
        return matrix


@dataclass(frozen=True)
class InheritedMatches:
    """Track-query part of the alignment, identical for every decoder layer.

    Only the newborn Hungarian step depends on the layer's own predictions, so this
    part is computed once per window and reused by the auxiliary layers.
    """

    prediction_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    query_target_ids: tuple[int, ...]
    newborn_target_indices: tuple[int, ...]


class CrossFrameTargetAlignment:
    """Equations (32)--(37): inherited track matches plus newborn matching."""

    def __init__(self, config: LossConfig):
        self.config = config

    def inherited_matches(
        self, tracks: TrackState, target_ids: torch.Tensor, num_predictions: int
    ) -> InheritedMatches:
        target_id_values = [int(value) for value in target_ids.tolist()]
        track_id_values = [int(value) for value in tracks.target_ids.tolist()]
        query_target_id_values = [-1] * num_predictions
        prediction_indices: list[int] = []
        target_indices: list[int] = []
        id_to_target = {target_id: index for index, target_id in enumerate(target_id_values)}
        matched_ids: set[int] = set()
        for query_index, target_id in enumerate(track_id_values):
            if target_id >= 0 and target_id in id_to_target and target_id not in matched_ids:
                prediction_indices.append(query_index)
                target_indices.append(id_to_target[target_id])
                matched_ids.add(target_id)
                query_target_id_values[query_index] = target_id
        return InheritedMatches(
            prediction_indices=tuple(prediction_indices),
            target_indices=tuple(target_indices),
            query_target_ids=tuple(query_target_id_values),
            newborn_target_indices=tuple(
                index
                for index, target_id in enumerate(target_id_values)
                if target_id not in matched_ids
            ),
        )

    @torch.no_grad()
    def __call__(
        self,
        prediction: FramePrediction,
        tracks: TrackState,
        target_positions: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        inherited: InheritedMatches | None = None,
        include_query_target_ids: bool = True,
    ) -> CTAResult:
        device = prediction.positions.device
        target_positions = target_positions.to(device)
        num_predictions = len(prediction.positions)
        # All bookkeeping below runs on Python lists. Doing it on device tensors costs
        # one host-device synchronisation per element, and this function is called
        # batch x windows x decoder_layers times per optimizer step.
        if inherited is None:
            inherited = self.inherited_matches(tracks, target_ids, num_predictions)
        query_target_id_values = list(inherited.query_target_ids)
        newborn_target_indices = list(inherited.newborn_target_indices)
        target_id_values: list[int] | None = None
        detection_start = prediction.num_track_queries
        newborn_predictions: list[int] = []
        newborn_targets: list[int] = []
        if detection_start < num_predictions and newborn_target_indices:
            newborn_index_tensor = torch.as_tensor(
                newborn_target_indices, dtype=torch.long, device=device
            )
            detection_positions = prediction.positions[detection_start:]
            detection_logits = prediction.logits[detection_start:]
            state_cost = torch.cdist(
                detection_positions, target_positions[newborn_index_tensor], p=1
            )
            object_cost = F.softplus(-detection_logits).expand(-1, len(newborn_target_indices))
            cost = (
                self.config.matching_state_weight * state_cost
                + self.config.matching_class_weight * object_cost
            )
            rows, columns = linear_sum_assignment(cost.cpu().numpy())
            for row, column in zip(rows.tolist(), columns.tolist()):
                prediction_index = detection_start + int(row)
                target_index = newborn_target_indices[column]
                newborn_predictions.append(prediction_index)
                newborn_targets.append(target_index)
                if include_query_target_ids:
                    if target_id_values is None:
                        target_id_values = [int(value) for value in target_ids.tolist()]
                    query_target_id_values[prediction_index] = target_id_values[target_index]

        prediction_indices = torch.tensor(
            list(inherited.prediction_indices) + newborn_predictions,
            dtype=torch.long,
            device=device,
        )
        matched_target_indices = torch.tensor(
            list(inherited.target_indices) + newborn_targets, dtype=torch.long, device=device
        )
        # Auxiliary layers only consume the matched index pairs; QTM and teacher forcing
        # read query_target_ids from the final layer alone, so skipping this tensor for
        # the auxiliary layers removes five device allocations per window.
        query_target_ids = (
            torch.tensor(query_target_id_values, dtype=torch.long, device=device)
            if include_query_target_ids
            else torch.empty(0, dtype=torch.long, device=device)
        )
        newborn_prediction_indices = (
            torch.tensor(newborn_predictions, dtype=torch.long, device=device)
            if newborn_predictions
            else torch.empty(0, dtype=torch.long, device=device)
        )
        return CTAResult(
            prediction_indices=prediction_indices,
            target_indices=matched_target_indices,
            query_target_ids=query_target_ids,
            inherited_count=len(inherited.prediction_indices),
            newborn_count=len(newborn_predictions),
            newborn_prediction_indices=newborn_prediction_indices,
        )

