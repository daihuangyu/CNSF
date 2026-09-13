from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from track_mt3.config import LossConfig
from track_mt3.matching.cta import CTAResult
from track_mt3.models.outputs import FramePrediction


@dataclass
class FrameLoss:
    total: torch.Tensor
    localization: torch.Tensor
    confidence: torch.Tensor
    auxiliary: torch.Tensor


class CollectiveAverageCriterion:
    """Collective average loss in equations (44)--(47)."""

    def __init__(self, config: LossConfig):
        self.config = config

    def _layer_scale(self, index: int, count: int) -> float:
        """Relative weight of auxiliary decoder layer ``index`` out of ``count``."""
        if self.config.auxiliary_layer_weighting == "uniform" or count <= 1:
            return 1.0
        if self.config.auxiliary_layer_weighting != "progressive":
            raise ValueError(
                f"unknown auxiliary_layer_weighting {self.config.auxiliary_layer_weighting}"
            )
        weights = [(position + 1) / count for position in range(count)]
        return weights[index] * count / sum(weights)

    @staticmethod
    def _newborn_weights(
        assignment: CTAResult,
        config: LossConfig,
        device: torch.device,
        target_ages: torch.Tensor | None = None,
        target_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if config.newborn_weight >= 1.0 and config.age_ramp <= 0:
            return None
        if not len(assignment.prediction_indices):
            return None
        newborn_mask = torch.zeros_like(assignment.prediction_indices, dtype=torch.bool)
        if len(assignment.newborn_prediction_indices):
            for idx in assignment.newborn_prediction_indices.tolist():
                match = (assignment.prediction_indices == idx).nonzero(as_tuple=False)
                if len(match):
                    newborn_mask[match[0, 0]] = True
        weights = torch.ones(len(assignment.prediction_indices), dtype=torch.float32, device=device)
        if config.newborn_weight < 1.0:
            weights[newborn_mask] = config.newborn_weight
        if config.age_ramp > 0 and target_ages is not None and target_indices is not None:
            ages = target_ages[target_indices.cpu()].to(device).float()
            ramp = torch.clamp(
                config.newborn_weight + (1.0 - config.newborn_weight) * ages / config.age_ramp,
                min=config.newborn_weight,
                max=1.0,
            )
            weights[newborn_mask] = ramp[newborn_mask]
        return weights

    def _layer_loss(
        self,
        positions: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        assignment: CTAResult,
        target_ages: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean_reduction = self.config.reduction == "mean"
        normalizer = max(float(len(targets)), self.config.empty_target_normalizer)
        matched = len(assignment.prediction_indices)
        needs_per_pair = (
            matched
            and (self.config.newborn_weight < 1.0 or self.config.age_ramp > 0)
        )
        if matched:
            pair_weights = self._newborn_weights(
                assignment, self.config, positions.device, target_ages,
                assignment.target_indices if needs_per_pair else None,
            )
            if pair_weights is not None:
                per_pair = F.l1_loss(
                    positions[assignment.prediction_indices],
                    targets[assignment.target_indices],
                    reduction="none",
                ).mean(dim=-1)
                localization = (per_pair * pair_weights).mean() if mean_reduction else (per_pair * pair_weights).sum() / normalizer
            else:
                reduction = "mean" if mean_reduction else "sum"
                localization = F.l1_loss(
                    positions[assignment.prediction_indices],
                    targets[assignment.target_indices],
                    reduction=reduction,
                )
                if not mean_reduction:
                    localization = localization / normalizer
        else:
            localization = positions.sum() * 0.0
        if matched and (self.config.newborn_weight < 1.0 or self.config.age_ramp > 0):
            labels = torch.zeros_like(logits)
            labels[assignment.prediction_indices] = 1.0
            nb_pred_idx = assignment.newborn_prediction_indices
            inherited_pos_weight = self.config.confidence_positive_weight
            newborn_pos_weight = inherited_pos_weight * self.config.newborn_weight
            pos_weight = torch.full_like(logits, inherited_pos_weight)
            if len(nb_pred_idx):
                pos_weight[nb_pred_idx] = newborn_pos_weight
            if mean_reduction:
                confidence = F.binary_cross_entropy_with_logits(
                    logits, labels, pos_weight=pos_weight, reduction="mean",
                )
            else:
                confidence = F.binary_cross_entropy_with_logits(
                    logits, labels, pos_weight=pos_weight, reduction="sum",
                ) / normalizer
        else:
            labels = torch.zeros_like(logits)
            if matched:
                labels[assignment.prediction_indices] = 1.0
            positive_weight = torch.as_tensor(
                self.config.confidence_positive_weight,
                dtype=logits.dtype,
                device=logits.device,
            )
            reduction = "mean" if mean_reduction else "sum"
            confidence = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=positive_weight, reduction=reduction,
            )
            if not mean_reduction:
                confidence = confidence / normalizer
        return localization, confidence

    def frame_loss(
        self,
        prediction: FramePrediction,
        targets: torch.Tensor,
        assignment: CTAResult,
        auxiliary_assignments: Sequence[CTAResult] | None = None,
        target_ages: torch.Tensor | None = None,
    ) -> FrameLoss:
        targets = targets.to(prediction.positions.device)
        if target_ages is not None:
            target_ages = target_ages.to(prediction.positions.device)
        localization, confidence = self._layer_loss(
            prediction.positions, prediction.logits, targets, assignment, target_ages
        )
        auxiliary = prediction.positions.sum() * 0.0
        if auxiliary_assignments is None:
            auxiliary_assignments = [assignment] * len(prediction.auxiliary)
        if len(auxiliary_assignments) != len(prediction.auxiliary):
            raise ValueError("one assignment is required per auxiliary layer")
        for index, (layer, layer_assignment) in enumerate(
            zip(prediction.auxiliary, auxiliary_assignments)
        ):
            layer_localization, layer_confidence = self._layer_loss(
                layer.positions, layer.logits, targets, layer_assignment, target_ages
            )
            auxiliary = auxiliary + self._layer_scale(index, len(prediction.auxiliary)) * (
                self.config.localization_weight * layer_localization
                + self.config.confidence_weight * layer_confidence
            )
        total = (
            self.config.localization_weight * localization
            + self.config.confidence_weight * confidence
            + self.config.auxiliary_weight * auxiliary
        )
        return FrameLoss(total=total, localization=localization, confidence=confidence, auxiliary=auxiliary)

    def collective_average(self, frame_losses: Sequence[FrameLoss]) -> FrameLoss:
        if not frame_losses:
            raise ValueError("CAL requires at least one frame")

        def average(field: str) -> torch.Tensor:
            return torch.stack([getattr(loss, field) for loss in frame_losses]).mean()

        return FrameLoss(
            total=average("total"),
            localization=average("localization"),
            confidence=average("confidence"),
            auxiliary=average("auxiliary"),
        )
