from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence

from .outputs import RecursiveFrameOutput


@dataclass(frozen=True)
class AssociationFrameLoss:
    total: torch.Tensor
    pair_nll: torch.Tensor
    miss_nll: torch.Tensor
    unclaimed_nll: torch.Tensor
    death_nll: torch.Tensor
    top1_accuracy: torch.Tensor
    miss_precision: torch.Tensor
    miss_recall: torch.Tensor
    unclaimed_accuracy: torch.Tensor
    death_accuracy: torch.Tensor
    true_event_probability: torch.Tensor
    marginal_error: torch.Tensor
    capacity_violation: torch.Tensor
    pair_events: int
    miss_events: int
    unclaimed_events: int
    death_events: int
    supervised_events: int


class V17BAssociationCriterion:
    """Direct transport supervision from simulator-only identity labels."""

    def __init__(
        self,
        *,
        balance_event_types: bool = False,
        pair_weight: float = 1.0,
        miss_weight: float = 1.0,
        unclaimed_weight: float = 1.0,
        death_weight: float = 1.0,
    ):
        weights = (pair_weight, miss_weight, unclaimed_weight, death_weight)
        if any(weight < 0.0 for weight in weights):
            raise ValueError("association event weights must be non-negative")
        if not any(weight > 0.0 for weight in weights):
            raise ValueError("at least one association event weight must be positive")
        self.balance_event_types = balance_event_types
        self.pair_weight = pair_weight
        self.miss_weight = miss_weight
        self.unclaimed_weight = unclaimed_weight
        self.death_weight = death_weight

    def __call__(
        self,
        output: RecursiveFrameOutput,
        truth_ids: Sequence[torch.Tensor] | None = None,
    ) -> AssociationFrameLoss:
        pair_log = output.predicted_pair_log_probabilities
        miss_log = output.predicted_miss_log_probabilities
        unclaimed_log = output.predicted_unclaimed_log_probabilities
        pair_probability = output.predicted_pair_probabilities
        miss_probability = output.predicted_miss_probabilities
        unclaimed_probability = output.predicted_unclaimed_probabilities
        death_log = output.predicted_death_log_probabilities
        death_probability = output.predicted_death_probabilities
        if any(
            value is None
            for value in (
                pair_log,
                miss_log,
                unclaimed_log,
                pair_probability,
                miss_probability,
                unclaimed_probability,
            )
        ):
            raise ValueError("association outputs are required for the G2 loss")

        active = output.existing_mask
        valid = ~output.measurement_padding_mask
        owners = output.existing_owner_ids
        measurement_ids = output.birth_measurement_ids

        if death_probability is not None:
            if death_log is None or truth_ids is None:
                raise ValueError(
                    "joint lifecycle association requires death logs and truth_ids"
                )
            padded_truth_ids = pad_sequence(
                [identifiers.to(owners.device) for identifiers in truth_ids],
                batch_first=True,
                padding_value=-2,
            )
            alive = (owners.unsqueeze(-1) == padded_truth_ids.unsqueeze(1)).any(-1)
        else:
            alive = active

        # Simulator IDs construct labels only.  Keep all matching on-device and
        # preserve the legacy rule that duplicate IDs select their first event.
        matches = (
            active.unsqueeze(-1)
            & alive.unsqueeze(-1)
            & valid.unsqueeze(1)
            & (owners >= 0).unsqueeze(-1)
            & (owners.unsqueeze(-1) == measurement_ids.unsqueeze(1))
        )
        pair_target = matches & (matches.to(torch.int64).cumsum(dim=-1) == 1)
        has_pair = pair_target.any(dim=-1)
        miss_target = active & alive & ~has_pair
        death_target = (
            active & ~alive if death_probability is not None else active & False
        )
        claimed = matches.any(dim=1)
        unclaimed_target = valid & ~claimed

        pair_terms = -pair_log[pair_target]
        miss_terms = -miss_log[miss_target]
        unclaimed_terms = -unclaimed_log[unclaimed_target]
        death_terms = (
            -death_log[death_target]
            if death_log is not None
            else output.posterior_mean.new_empty(0)
        )
        terms = torch.cat((pair_terms, miss_terms, unclaimed_terms, death_terms))
        probabilities = torch.cat(
            (
                pair_probability[pair_target],
                miss_probability[miss_target],
                unclaimed_probability[unclaimed_target],
                (
                    death_probability[death_target]
                    if death_probability is not None
                    else output.posterior_mean.new_empty(0)
                ),
            )
        )

        masked_pair = pair_probability.masked_fill(~valid.unsqueeze(1), -1.0)
        best_pair_probability, predicted_measurement = masked_pair.max(dim=-1)
        if death_probability is None:
            predicted_death_mask = active & False
            predicted_miss_mask = active & (miss_probability > best_pair_probability)
        else:
            predicted_death_mask = (
                active
                & (death_probability > best_pair_probability)
                & (death_probability > miss_probability)
            )
            predicted_miss_mask = (
                active
                & ~predicted_death_mask
                & (miss_probability > best_pair_probability)
            )
        actual_measurement = pair_target.to(torch.int64).argmax(dim=-1)
        track_correct_mask = active & torch.where(
            death_target,
            predicted_death_mask,
            torch.where(
                miss_target,
                predicted_miss_mask,
                (~predicted_miss_mask)
                & (~predicted_death_mask)
                & (predicted_measurement == actual_measurement),
            ),
        )
        miss_true_positive_mask = predicted_miss_mask & miss_target

        best_track_pair = (
            pair_probability.masked_fill(~active.unsqueeze(-1), 0.0).max(dim=1).values
        )
        predicted_unclaimed = unclaimed_probability >= best_track_pair
        unclaimed_correct_mask = unclaimed_target & predicted_unclaimed

        anchor = output.posterior_mean.sum() * 0.0
        category_terms = (pair_terms, miss_terms, unclaimed_terms, death_terms)
        category_weights = (
            self.pair_weight,
            self.miss_weight,
            self.unclaimed_weight,
            self.death_weight,
        )
        category_means = tuple(
            values.mean() if values.numel() else anchor for values in category_terms
        )
        if self.balance_event_types:
            present = [
                (weight, mean)
                for weight, mean, values in zip(
                    category_weights, category_means, category_terms
                )
                if weight > 0.0 and values.numel()
            ]
            total = (
                sum(weight * mean for weight, mean in present)
                / sum(weight for weight, _ in present)
                if present
                else anchor
            )
        else:
            total = terms.mean() if terms.numel() else anchor
        true_probability = (
            probabilities.mean().detach() if probabilities.numel() else anchor.detach()
        )
        device = output.posterior_mean.device

        def scalar(value: float) -> torch.Tensor:
            return torch.tensor(float(value), device=device)

        track_count = int(active.sum())
        track_correct = int(track_correct_mask.sum())
        miss_predicted = int(predicted_miss_mask.sum())
        miss_actual = int(miss_target.sum())
        miss_true_positive = int(miss_true_positive_mask.sum())
        unclaimed_count = int(unclaimed_target.sum())
        unclaimed_correct = int(unclaimed_correct_mask.sum())
        death_count = int(death_target.sum())
        death_correct = int((predicted_death_mask & death_target).sum())
        pair_events = int(pair_target.sum())
        miss_events = miss_actual
        unclaimed_events = unclaimed_count
        marginal_error = (
            output.association_marginal_error.mean().detach()
            if output.association_marginal_error is not None
            else anchor.detach()
        )
        capacity_violation = (
            output.association_capacity_violation.mean().detach()
            if output.association_capacity_violation is not None
            else anchor.detach()
        )
        return AssociationFrameLoss(
            total=total,
            pair_nll=category_means[0].detach(),
            miss_nll=category_means[1].detach(),
            unclaimed_nll=category_means[2].detach(),
            death_nll=category_means[3].detach(),
            top1_accuracy=scalar(track_correct / max(track_count, 1)),
            miss_precision=scalar(miss_true_positive / max(miss_predicted, 1)),
            miss_recall=scalar(miss_true_positive / max(miss_actual, 1)),
            unclaimed_accuracy=scalar(unclaimed_correct / max(unclaimed_count, 1)),
            death_accuracy=scalar(death_correct / max(death_count, 1)),
            true_event_probability=true_probability,
            marginal_error=marginal_error,
            capacity_violation=capacity_violation,
            pair_events=pair_events,
            miss_events=miss_events,
            unclaimed_events=unclaimed_events,
            death_events=death_count,
            supervised_events=(
                pair_events + miss_events + unclaimed_events + death_count
            ),
        )
