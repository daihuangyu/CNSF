from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .outputs import RecursiveFrameOutput


@dataclass(frozen=True)
class V17DeathLoss:
    total: torch.Tensor
    alive_probability: torch.Tensor
    dead_probability: torch.Tensor
    alive_tracks: int
    dead_tracks: int
    cardinality: torch.Tensor | None = None
    predicted_cardinality: torch.Tensor | None = None
    target_cardinality: torch.Tensor | None = None


@dataclass(frozen=True)
class V18CBernoulliLoss:
    """Proper posterior and one-step survival scoring for v18-C."""

    posterior: torch.Tensor
    survival_transition: torch.Tensor
    cardinality: torch.Tensor
    alive_probability: torch.Tensor
    dead_probability: torch.Tensor
    missed_alive_probability: torch.Tensor
    survival_alive_probability: torch.Tensor
    survival_dead_probability: torch.Tensor
    posterior_tracks: int
    transition_tracks: int
    predicted_cardinality: torch.Tensor
    target_cardinality: torch.Tensor


def negative_exposed_death_mean(
    losses: Sequence[V17DeathLoss], anchor: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Average only frames containing a real lifecycle negative.

    Positive-only frames are useful diagnostics, but optimizing them before
    predicted births expose false/dead tracks makes the survival head learn
    the degenerate always-alive solution.
    """

    exposed = [loss.total for loss in losses if loss.dead_tracks > 0]
    if not exposed:
        return anchor * 0.0, 0
    return torch.stack(exposed).mean(), len(exposed)


class V17DeathCriterion:
    """Balanced existence loss after dead/false tracks have been exposed."""

    def __init__(
        self,
        *,
        logit_field: str = "existing_logits",
        hard_negative_fraction: float = 0.0,
        hard_negative_weight: float = 0.0,
        natural_frequency: bool = False,
    ):
        if logit_field not in {"existing_logits", "survival_logits"}:
            raise ValueError("invalid death logit field")
        if not 0.0 <= hard_negative_fraction <= 1.0:
            raise ValueError("hard_negative_fraction must be in [0,1]")
        if hard_negative_weight < 0.0:
            raise ValueError("hard_negative_weight must be non-negative")
        self.logit_field = logit_field
        self.hard_negative_fraction = hard_negative_fraction
        self.hard_negative_weight = hard_negative_weight
        self.natural_frequency = natural_frequency

    def __call__(
        self,
        output: RecursiveFrameOutput,
        truth_ids: Sequence[torch.Tensor],
    ) -> V17DeathLoss:
        anchor = output.existing_logits.sum() * 0.0
        active = output.existing_mask
        predicted_cardinality_per_row = (
            output.existing_logits.sigmoid() * active
        ).sum(-1) + (
            output.birth_logits.sigmoid() * output.birth_mask
        ).sum(-1)
        target_cardinality_per_row = torch.tensor(
            [len(identifiers) for identifiers in truth_ids],
            device=output.existing_logits.device,
            dtype=output.existing_logits.dtype,
        )
        cardinality = F.smooth_l1_loss(
            predicted_cardinality_per_row,
            target_cardinality_per_row,
        )
        if not torch.any(active):
            return V17DeathLoss(
                total=anchor,
                cardinality=cardinality,
                alive_probability=anchor.detach(),
                dead_probability=anchor.detach(),
                alive_tracks=0,
                dead_tracks=0,
                predicted_cardinality=predicted_cardinality_per_row.mean().detach(),
                target_cardinality=target_cardinality_per_row.mean().detach(),
            )
        padded_truth_ids = pad_sequence(
            [identifiers.to(output.existing_owner_ids.device) for identifiers in truth_ids],
            batch_first=True,
            padding_value=-2,
        )
        alive = (
            output.existing_owner_ids.unsqueeze(-1)
            == padded_truth_ids.unsqueeze(1)
        ).any(-1)
        logit_tensor = getattr(output, self.logit_field)[active]
        label_tensor = alive[active]
        if self.natural_frequency:
            total = F.binary_cross_entropy_with_logits(
                logit_tensor,
                label_tensor.to(logit_tensor.dtype),
            )
        else:
            terms = []
            if torch.any(label_tensor):
                terms.append(F.softplus(-logit_tensor[label_tensor]).mean())
            if torch.any(~label_tensor):
                negative_losses = F.softplus(logit_tensor[~label_tensor])
                negative_term = negative_losses.mean()
                if (
                    self.hard_negative_fraction > 0.0
                    and self.hard_negative_weight > 0.0
                ):
                    hard_count = max(
                        1,
                        int(round(len(negative_losses) * self.hard_negative_fraction)),
                    )
                    hard_term = negative_losses.topk(hard_count).values.mean()
                    negative_term = (
                        negative_term + self.hard_negative_weight * hard_term
                    )
                terms.append(negative_term)
            total = torch.stack(terms).mean()
        probabilities = logit_tensor.sigmoid()
        alive_probability = (
            probabilities[label_tensor].mean().detach()
            if torch.any(label_tensor)
            else anchor.detach()
        )
        dead_probability = (
            probabilities[~label_tensor].mean().detach()
            if torch.any(~label_tensor)
            else anchor.detach()
        )
        return V17DeathLoss(
            total=total,
            cardinality=cardinality,
            alive_probability=alive_probability,
            dead_probability=dead_probability,
            alive_tracks=int(label_tensor.sum()),
            dead_tracks=int((~label_tensor).sum()),
            predicted_cardinality=predicted_cardinality_per_row.mean().detach(),
            target_cardinality=target_cardinality_per_row.mean().detach(),
        )


class V18CBernoulliCriterion:
    """Calibrate one posterior and a separate causal survival transition.

    Posterior labels use every active slot in its natural class frequency.
    This is a proper scoring rule, unlike the balanced classification loss
    used by v18-B.  Live-but-undetected tracks may receive an additional
    persistence weight.  The survival prior is trained only on tracks that
    truly existed in the previous frame, with the current truth deciding
    whether that one-step survival transition succeeded.
    """

    def __init__(self, *, missed_alive_weight: float = 1.0):
        if missed_alive_weight < 0.0:
            raise ValueError("missed_alive_weight must be non-negative")
        self.missed_alive_weight = missed_alive_weight

    def __call__(
        self,
        output: RecursiveFrameOutput,
        truth_ids: Sequence[torch.Tensor],
        previous_truth_ids: Sequence[torch.Tensor] | None,
    ) -> V18CBernoulliLoss:
        anchor = output.existing_logits.sum() * 0.0
        active = output.existing_mask
        padded_truth = pad_sequence(
            [identifiers.to(output.existing_owner_ids.device) for identifiers in truth_ids],
            batch_first=True,
            padding_value=-2,
        )
        alive = (
            output.existing_owner_ids.unsqueeze(-1) == padded_truth.unsqueeze(1)
        ).any(-1)
        detected = (
            output.existing_owner_ids.unsqueeze(-1)
            == output.birth_measurement_ids.unsqueeze(1)
        ).any(-1)
        missed_alive = active & alive & ~detected

        if torch.any(active):
            posterior_terms = F.binary_cross_entropy_with_logits(
                output.existing_logits[active],
                alive[active].to(output.existing_logits.dtype),
                reduction="none",
            )
            posterior_weights = 1.0 + self.missed_alive_weight * missed_alive[active].to(
                posterior_terms.dtype
            )
            posterior = (posterior_terms * posterior_weights).sum() / posterior_weights.sum()
            posterior_probabilities = output.existing_logits[active].sigmoid()
            posterior_labels = alive[active]
            alive_probability = (
                posterior_probabilities[posterior_labels].mean().detach()
                if torch.any(posterior_labels)
                else anchor.detach()
            )
            dead_probability = (
                posterior_probabilities[~posterior_labels].mean().detach()
                if torch.any(~posterior_labels)
                else anchor.detach()
            )
        else:
            posterior = anchor
            alive_probability = anchor.detach()
            dead_probability = anchor.detach()

        missed_alive_probability = (
            output.existing_logits[missed_alive].sigmoid().mean().detach()
            if torch.any(missed_alive)
            else anchor.detach()
        )
        predicted_cardinality_per_row = (
            output.existing_logits.sigmoid() * output.existing_mask
        ).sum(-1) + (
            output.birth_logits.sigmoid() * output.birth_mask
        ).sum(-1)
        target_cardinality_per_row = torch.tensor(
            [len(identifiers) for identifiers in truth_ids],
            device=output.existing_logits.device,
            dtype=output.existing_logits.dtype,
        )
        cardinality = F.smooth_l1_loss(
            predicted_cardinality_per_row,
            target_cardinality_per_row,
        )
        transition = torch.zeros_like(active)
        survived = torch.zeros_like(active)
        if previous_truth_ids is not None:
            padded_previous_truth = pad_sequence(
                [
                    identifiers.to(output.existing_owner_ids.device)
                    for identifiers in previous_truth_ids
                ],
                batch_first=True,
                padding_value=-2,
            )
            existed_previously = (
                output.existing_owner_ids.unsqueeze(-1)
                == padded_previous_truth.unsqueeze(1)
            ).any(-1)
            transition = active & existed_previously
            survived = transition & alive
        if torch.any(transition):
            survival_transition = F.binary_cross_entropy_with_logits(
                output.survival_prior_logits[transition],
                survived[transition].to(output.survival_prior_logits.dtype),
            )
            survival_probabilities = output.survival_prior_logits[transition].sigmoid()
            transition_labels = survived[transition]
            survival_alive_probability = (
                survival_probabilities[transition_labels].mean().detach()
                if torch.any(transition_labels)
                else anchor.detach()
            )
            survival_dead_probability = (
                survival_probabilities[~transition_labels].mean().detach()
                if torch.any(~transition_labels)
                else anchor.detach()
            )
        else:
            survival_transition = anchor
            survival_alive_probability = anchor.detach()
            survival_dead_probability = anchor.detach()

        return V18CBernoulliLoss(
            posterior=posterior,
            survival_transition=survival_transition,
            cardinality=cardinality,
            alive_probability=alive_probability,
            dead_probability=dead_probability,
            missed_alive_probability=missed_alive_probability,
            survival_alive_probability=survival_alive_probability,
            survival_dead_probability=survival_dead_probability,
            posterior_tracks=int(active.sum()),
            transition_tracks=int(transition.sum()),
            predicted_cardinality=predicted_cardinality_per_row.mean().detach(),
            target_cardinality=target_cardinality_per_row.mean().detach(),
        )
