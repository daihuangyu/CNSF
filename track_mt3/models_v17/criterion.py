from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from scipy.optimize import linear_sum_assignment

from .outputs import RecursiveFrameOutput


@dataclass(frozen=True)
class V17AFrameLoss:
    total: torch.Tensor
    set_risk: torch.Tensor
    localization: torch.Tensor
    confidence: torch.Tensor
    prior_nll: torch.Tensor
    posterior_nll: torch.Tensor
    prior_position_error: torch.Tensor
    posterior_position_error: torch.Tensor
    prior_position_anchor: torch.Tensor
    prior_velocity_anchor: torch.Tensor
    posterior_velocity_anchor: torch.Tensor
    matched_existing: int
    matched_births: int


def gaussian_nll(
    value: torch.Tensor, mean: torch.Tensor, covariance: torch.Tensor
) -> torch.Tensor:
    if not len(value):
        return mean.sum() * 0.0
    cholesky = torch.linalg.cholesky(covariance)
    residual = (value - mean).unsqueeze(-1)
    solved = torch.cholesky_solve(residual, cholesky)
    mahalanobis = (residual.transpose(-1, -2) @ solved).flatten()
    log_determinant = 2.0 * torch.log(
        cholesky.diagonal(dim1=-2, dim2=-1)
    ).sum(-1)
    return 0.5 * (mahalanobis + log_determinant).mean()


class V17AOracleCriterion:
    """CTA-constrained Hungarian set loss plus explicit filter supervision."""

    def __init__(
        self,
        *,
        localization_weight: float = 1.0,
        confidence_weight: float = 1.0,
        prior_nll_weight: float = 0.1,
        posterior_nll_weight: float = 0.1,
        matching_class_weight: float = 1.0,
        prior_position_weight: float = 0.0,
        prior_velocity_weight: float = 0.0,
        posterior_velocity_weight: float = 0.0,
        position_anchor_beta: float = 0.25,
        velocity_anchor_beta: float = 0.5,
        set_risk_cutoff: float = 2.0,
    ):
        self.localization_weight = localization_weight
        self.confidence_weight = confidence_weight
        self.prior_nll_weight = prior_nll_weight
        self.posterior_nll_weight = posterior_nll_weight
        self.matching_class_weight = matching_class_weight
        self.prior_position_weight = prior_position_weight
        self.prior_velocity_weight = prior_velocity_weight
        self.posterior_velocity_weight = posterior_velocity_weight
        self.position_anchor_beta = position_anchor_beta
        self.velocity_anchor_beta = velocity_anchor_beta
        self.set_risk_cutoff = set_risk_cutoff
        if position_anchor_beta <= 0.0 or velocity_anchor_beta <= 0.0:
            raise ValueError("kinematic anchor beta values must be positive")
        if set_risk_cutoff <= 0.0:
            raise ValueError("set_risk_cutoff must be positive")

    def __call__(
        self,
        output: RecursiveFrameOutput,
        truth_states: Sequence[torch.Tensor],
        truth_ids: Sequence[torch.Tensor],
    ) -> V17AFrameLoss:
        anchor = output.posterior_mean.sum() * 0.0
        device = output.posterior_mean.device
        batch = output.posterior_mean.shape[0]
        states = [value.to(device) for value in truth_states]
        identifiers = [value.to(device) for value in truth_ids]
        padded_states = pad_sequence(states, batch_first=True)
        padded_ids = pad_sequence(
            identifiers, batch_first=True, padding_value=-2
        )
        target_count = padded_states.shape[1]
        lengths = torch.as_tensor(
            [len(value) for value in states], device=device
        )
        target_valid = (
            torch.arange(target_count, device=device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )

        if target_count:
            inherited = (
                output.existing_mask.unsqueeze(-1)
                & target_valid.unsqueeze(1)
                & (
                    output.existing_owner_ids.unsqueeze(-1)
                    == padded_ids.unsqueeze(1)
                )
            )
            existing_target_index = inherited.to(torch.int64).argmax(dim=-1)
            existing_matched_mask = inherited.any(dim=-1)
            owned_targets = inherited.any(dim=1)
        else:
            existing_target_index = torch.zeros_like(
                output.existing_owner_ids
            )
            existing_matched_mask = torch.zeros_like(output.existing_mask)
            owned_targets = target_valid
        remaining_targets = target_valid & ~owned_targets

        existing_batch, existing_slot = torch.nonzero(
            existing_matched_mask, as_tuple=True
        )
        if existing_batch.numel():
            inherited_target = padded_states[
                existing_batch,
                existing_target_index[existing_batch, existing_slot],
            ]
            existing_localization = (
                output.posterior_mean[existing_batch, existing_slot, :2]
                - inherited_target[:, :2]
            ).abs().mean(-1)
            prior_values = inherited_target
            prior_means = output.prior_mean[existing_batch, existing_slot]
            prior_covariances = output.prior_covariance[
                existing_batch, existing_slot
            ]
            posterior_means = output.posterior_mean[
                existing_batch, existing_slot
            ]
            posterior_covariances = output.posterior_covariance[
                existing_batch, existing_slot
            ]
            prior_errors = torch.linalg.vector_norm(
                prior_means[:, :2] - prior_values[:, :2], dim=-1
            )
            posterior_errors = torch.linalg.vector_norm(
                posterior_means[:, :2] - prior_values[:, :2], dim=-1
            )
            prior_position_anchor = F.smooth_l1_loss(
                prior_means[:, :2],
                prior_values[:, :2],
                reduction="mean",
                beta=self.position_anchor_beta,
            )
            prior_velocity_anchor = F.smooth_l1_loss(
                prior_means[:, 2:],
                prior_values[:, 2:],
                reduction="mean",
                beta=self.velocity_anchor_beta,
            )
            posterior_velocity_anchor = F.smooth_l1_loss(
                posterior_means[:, 2:],
                prior_values[:, 2:],
                reduction="mean",
                beta=self.velocity_anchor_beta,
            )
        else:
            existing_localization = output.posterior_mean.new_empty(0)
            prior_position_anchor = anchor
            prior_velocity_anchor = anchor
            posterior_velocity_anchor = anchor

        birth_matched_mask = torch.zeros_like(output.birth_mask)
        birth_target_index = torch.zeros_like(
            output.birth_measurement_ids
        )
        if target_count and output.birth_positions.shape[1]:
            state_cost = torch.cdist(
                output.birth_positions,
                padded_states[..., :2],
                p=1,
            )
            object_cost = F.softplus(-output.birth_logits).unsqueeze(-1)
            cost_cpu = (
                state_cost + self.matching_class_weight * object_cost
            ).detach().cpu().numpy()
            birth_mask_cpu = output.birth_mask.detach().cpu().numpy()
            remaining_cpu = remaining_targets.detach().cpu().numpy()
            matched_batches: list[int] = []
            matched_measurements: list[int] = []
            matched_targets: list[int] = []
            for batch_index in range(batch):
                birth_indices = np.flatnonzero(birth_mask_cpu[batch_index])
                target_indices = np.flatnonzero(remaining_cpu[batch_index])
                if not len(birth_indices) or not len(target_indices):
                    continue
                rows, columns = linear_sum_assignment(
                    cost_cpu[batch_index][np.ix_(birth_indices, target_indices)]
                )
                matched_batches.extend([batch_index] * len(rows))
                matched_measurements.extend(birth_indices[rows].tolist())
                matched_targets.extend(target_indices[columns].tolist())
            if matched_batches:
                matched_batch_tensor = torch.as_tensor(
                    matched_batches, device=device, dtype=torch.long
                )
                matched_measurement_tensor = torch.as_tensor(
                    matched_measurements, device=device, dtype=torch.long
                )
                matched_target_tensor = torch.as_tensor(
                    matched_targets, device=device, dtype=torch.long
                )
                birth_matched_mask[
                    matched_batch_tensor, matched_measurement_tensor
                ] = True
                birth_target_index[
                    matched_batch_tensor, matched_measurement_tensor
                ] = matched_target_tensor

        birth_batch, birth_measurement = torch.nonzero(
            birth_matched_mask, as_tuple=True
        )
        if birth_batch.numel():
            matched_birth_target = padded_states[
                birth_batch,
                birth_target_index[birth_batch, birth_measurement],
                :2,
            ]
            birth_localization = (
                output.birth_positions[birth_batch, birth_measurement]
                - matched_birth_target
            ).abs().mean(-1)
        else:
            birth_localization = output.birth_positions.new_empty(0)
        localization_values = torch.cat(
            (existing_localization, birth_localization)
        )
        localization = (
            localization_values.mean() if localization_values.numel() else anchor
        )

        existing_confidence = F.binary_cross_entropy_with_logits(
            output.existing_logits,
            existing_matched_mask.to(output.existing_logits.dtype),
            reduction="none",
        ).masked_fill(~output.existing_mask, 0.0)
        birth_confidence = F.binary_cross_entropy_with_logits(
            output.birth_logits,
            birth_matched_mask.to(output.birth_logits.dtype),
            reduction="none",
        ).masked_fill(~output.birth_mask, 0.0)
        confidence_count = (
            output.existing_mask.sum(-1) + output.birth_mask.sum(-1)
        )
        confidence_row = (
            existing_confidence.sum(-1) + birth_confidence.sum(-1)
        ) / confidence_count.clamp_min(1)
        confidence_rows = confidence_count > 0
        confidence = (
            confidence_row[confidence_rows].mean()
            if torch.any(confidence_rows)
            else anchor
        )

        # Differentiable order-1 Pro-GOSPA risk under the same CTA-constrained
        # assignment used by the tracking loss.  A matched Bernoulli contributes
        # p * min(distance, c) + (1-p) * c/2, an unmatched Bernoulli p*c/2,
        # and an unmatched target c/2.  The discrete assignment is deliberately
        # detached; gradients act on positions and existence probabilities.
        cutoff = float(self.set_risk_cutoff)
        half_cutoff = cutoff / 2.0
        existing_probability = output.existing_logits.sigmoid()
        birth_probability = output.birth_logits.sigmoid()
        set_risk_row = (
            existing_probability.masked_fill(~output.existing_mask, 0.0).sum(-1)
            + birth_probability.masked_fill(~output.birth_mask, 0.0).sum(-1)
        ) * half_cutoff
        if existing_batch.numel():
            matched_probability = existing_probability[
                existing_batch, existing_slot
            ]
            matched_distance = torch.linalg.vector_norm(
                output.posterior_mean[existing_batch, existing_slot, :2]
                - inherited_target[:, :2],
                dim=-1,
            ).clamp_max(cutoff)
            matched_cost = (
                matched_probability * matched_distance
                + (1.0 - matched_probability) * half_cutoff
            )
            set_risk_row = set_risk_row.index_add(
                0,
                existing_batch,
                matched_cost - matched_probability * half_cutoff,
            )
        if birth_batch.numel():
            matched_probability = birth_probability[birth_batch, birth_measurement]
            matched_distance = torch.linalg.vector_norm(
                output.birth_positions[birth_batch, birth_measurement]
                - matched_birth_target,
                dim=-1,
            ).clamp_max(cutoff)
            matched_cost = (
                matched_probability * matched_distance
                + (1.0 - matched_probability) * half_cutoff
            )
            set_risk_row = set_risk_row.index_add(
                0,
                birth_batch,
                matched_cost - matched_probability * half_cutoff,
            )
        matched_existing_per_row = existing_matched_mask.sum(-1)
        matched_births_per_row = birth_matched_mask.sum(-1)
        missed_targets = (
            lengths - matched_existing_per_row - matched_births_per_row
        ).clamp_min(0)
        set_risk = (set_risk_row + missed_targets * half_cutoff).mean()

        if existing_batch.numel():
            prior_nll = gaussian_nll(
                prior_values,
                prior_means,
                prior_covariances,
            )
            posterior_nll = gaussian_nll(
                prior_values,
                posterior_means,
                posterior_covariances,
            )
            prior_position_error = prior_errors.mean()
            posterior_position_error = posterior_errors.mean()
        else:
            prior_nll = posterior_nll = anchor
            prior_position_error = posterior_position_error = anchor.detach()
        total = (
            self.localization_weight * localization
            + self.confidence_weight * confidence
            + self.prior_nll_weight * prior_nll
            + self.posterior_nll_weight * posterior_nll
            + self.prior_position_weight * prior_position_anchor
            + self.prior_velocity_weight * prior_velocity_anchor
            + self.posterior_velocity_weight * posterior_velocity_anchor
        )
        return V17AFrameLoss(
            total=total,
            set_risk=set_risk,
            localization=localization.detach(),
            confidence=confidence.detach(),
            prior_nll=prior_nll.detach(),
            posterior_nll=posterior_nll.detach(),
            prior_position_error=prior_position_error.detach(),
            posterior_position_error=posterior_position_error.detach(),
            prior_position_anchor=prior_position_anchor.detach(),
            prior_velocity_anchor=prior_velocity_anchor.detach(),
            posterior_velocity_anchor=posterior_velocity_anchor.detach(),
            matched_existing=int(existing_matched_mask.sum()),
            matched_births=int(birth_matched_mask.sum()),
        )
