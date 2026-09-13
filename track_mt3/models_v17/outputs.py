from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RecursiveFrameOutput:
    """All causal intermediates needed by loss functions and diagnostics."""

    prior_mean: torch.Tensor
    prior_covariance: torch.Tensor
    posterior_mean: torch.Tensor
    posterior_covariance: torch.Tensor
    existing_logits: torch.Tensor
    survival_logits: torch.Tensor
    survival_prior_logits: torch.Tensor
    existing_mask: torch.Tensor
    existing_confirmed_mask: torch.Tensor
    existing_owner_ids: torch.Tensor
    existing_runtime_ids: torch.Tensor
    association_mask: torch.Tensor
    birth_positions: torch.Tensor
    birth_logits: torch.Tensor
    birth_mask: torch.Tensor
    birth_measurement_ids: torch.Tensor
    measurement_embeddings: torch.Tensor
    measurement_padding_mask: torch.Tensor
    measurement_objectness_logits: torch.Tensor
    association_strength: torch.Tensor | None = None
    predicted_pair_probabilities: torch.Tensor | None = None
    predicted_miss_probabilities: torch.Tensor | None = None
    predicted_unclaimed_probabilities: torch.Tensor | None = None
    predicted_death_probabilities: torch.Tensor | None = None
    predicted_pair_log_probabilities: torch.Tensor | None = None
    predicted_miss_log_probabilities: torch.Tensor | None = None
    predicted_unclaimed_log_probabilities: torch.Tensor | None = None
    predicted_death_log_probabilities: torch.Tensor | None = None
    association_marginal_error: torch.Tensor | None = None
    association_capacity_violation: torch.Tensor | None = None
    oracle_association_used: torch.Tensor | None = None
    association_track_bias: torch.Tensor | None = None
    predicted_undetected_mean: torch.Tensor | None = None
    birth_target_log_intensity: torch.Tensor | None = None
    birth_clutter_log_intensity: torch.Tensor | None = None
    association_hypothesis_weights: torch.Tensor | None = None
    alternative_posterior_mean: torch.Tensor | None = None
    birth_unclaimed_probabilities: torch.Tensor | None = None

    def newborn_probabilities(self, *, detach_unclaimed: bool = False) -> torch.Tensor:
        """Factor newborn evidence into target-likeness and residual mass.

        The objectness head separates target-generated measurements from clutter.
        The association transport separately decides whether an existing track
        already explains each measurement.  Their product is therefore the
        causal birth posterior used by the learned lifecycle.
        """

        if self.birth_target_log_intensity is not None:
            return self.birth_logits.sigmoid()
        objectness = self.measurement_objectness_logits.sigmoid()
        if self.predicted_unclaimed_probabilities is None:
            return objectness
        unclaimed = self.predicted_unclaimed_probabilities
        if detach_unclaimed:
            unclaimed = unclaimed.detach()
        return objectness * unclaimed

    def row_predictions(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        existing = self.existing_mask[index]
        births = self.birth_mask[index]
        positions = torch.cat(
            (
                self.posterior_mean[index, existing, :2],
                self.birth_positions[index, births],
            )
        )
        logits = torch.cat(
            (self.existing_logits[index, existing], self.birth_logits[index, births])
        )
        return positions, logits
