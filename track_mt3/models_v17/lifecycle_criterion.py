from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .outputs import RecursiveFrameOutput


@dataclass(frozen=True)
class V17BirthLoss:
    total: torch.Tensor
    measurement_type: torch.Tensor
    newborn: torch.Tensor
    objectness_positive_mean: torch.Tensor
    objectness_negative_mean: torch.Tensor
    newborn_positive_mean: torch.Tensor
    newborn_negative_mean: torch.Tensor
    target_measurements: int
    clutter_measurements: int
    newborn_measurements: int
    poisson_count: torch.Tensor


def _balanced_probability_nll(
    log_probability: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Average positive and negative event NLL without clutter-count drift."""

    terms: list[torch.Tensor] = []
    if torch.any(labels):
        terms.append(-log_probability[labels].mean())
    negative = ~labels
    if torch.any(negative):
        # log(1-exp(x)) is stable for x<=0 in this form away from exactly zero.
        negative_log_probability = torch.log(
            (-torch.expm1(log_probability[negative])).clamp_min(1.0e-8)
        )
        terms.append(-negative_log_probability.mean())
    if not terms:
        return log_probability.sum() * 0.0
    return torch.stack(terms).mean()


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if torch.any(mask):
        return values[mask].mean()
    return values.sum() * 0.0


class V17BirthCriterion:
    """Supervise birth as target objectness times UNCLAIMED association mass.

    Measurement IDs and existing owner IDs only construct training labels.  They
    never enter either neural head.  Detaching UNCLAIMED makes G3a a clean birth
    gate that cannot damage a frozen/calibrated association model.
    """

    def __init__(
        self,
        *,
        measurement_type_weight: float = 1.0,
        newborn_weight: float = 1.0,
        detach_unclaimed: bool = True,
        poisson_count_weight: float = 0.0,
    ):
        self.measurement_type_weight = measurement_type_weight
        self.newborn_weight = newborn_weight
        self.detach_unclaimed = detach_unclaimed
        self.poisson_count_weight = poisson_count_weight
        if poisson_count_weight < 0.0:
            raise ValueError("poisson_count_weight must be non-negative")

    def __call__(self, output: RecursiveFrameOutput) -> V17BirthLoss:
        valid = ~output.measurement_padding_mask
        target = valid & (output.birth_measurement_ids >= 0)
        clutter = valid & ~target
        already_owned = (
            output.existing_mask[:, None, :]
            & (
                output.birth_measurement_ids[:, :, None]
                == output.existing_owner_ids[:, None, :]
            )
        ).any(-1)
        newborn = target & ~already_owned

        objectness_log_probability = F.logsigmoid(
            output.measurement_objectness_logits
        )
        measurement_type = _balanced_probability_nll(
            objectness_log_probability[valid], target[valid]
        )
        if output.predicted_unclaimed_log_probabilities is None:
            unclaimed_log_probability = torch.zeros_like(
                objectness_log_probability
            )
        else:
            unclaimed_log_probability = (
                output.predicted_unclaimed_log_probabilities.detach()
                if self.detach_unclaimed
                else output.predicted_unclaimed_log_probabilities
            )
        if output.birth_target_log_intensity is None:
            newborn_log_probability = (
                objectness_log_probability + unclaimed_log_probability
            ).clamp_max(-1.0e-8)
            poisson_count = objectness_log_probability.sum() * 0.0
        else:
            newborn_log_probability = F.logsigmoid(output.birth_logits)
            target_intensity = output.birth_target_log_intensity.exp().masked_fill(
                ~valid, 0.0
            )
            unclaimed_source = (
                output.birth_unclaimed_probabilities
                if output.birth_unclaimed_probabilities is not None
                else output.predicted_unclaimed_probabilities
            )
            if unclaimed_source is not None:
                unclaimed = unclaimed_source
                if self.detach_unclaimed:
                    unclaimed = unclaimed.detach()
                target_intensity = target_intensity * unclaimed
            predicted_count = target_intensity.sum(-1).clamp_min(1.0e-6)
            observed_count = newborn.sum(-1).to(predicted_count.dtype)
            poisson_count = (
                predicted_count - observed_count * predicted_count.log()
            ).mean()
        newborn_loss = _balanced_probability_nll(
            newborn_log_probability[valid], newborn[valid]
        )
        total = (
            self.measurement_type_weight * measurement_type
            + self.newborn_weight * newborn_loss
            + self.poisson_count_weight * poisson_count
        )
        objectness_probability = objectness_log_probability.exp()
        newborn_probability = newborn_log_probability.exp()
        return V17BirthLoss(
            total=total,
            measurement_type=measurement_type.detach(),
            newborn=newborn_loss.detach(),
            objectness_positive_mean=_masked_mean(
                objectness_probability, target
            ).detach(),
            objectness_negative_mean=_masked_mean(
                objectness_probability, clutter
            ).detach(),
            newborn_positive_mean=_masked_mean(
                newborn_probability, newborn
            ).detach(),
            newborn_negative_mean=_masked_mean(
                newborn_probability, valid & ~newborn
            ).detach(),
            target_measurements=int(target.sum()),
            clutter_measurements=int(clutter.sum()),
            newborn_measurements=int(newborn.sum()),
            poisson_count=poisson_count.detach(),
        )
