from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ContrastiveClassifier(nn.Module):
    """Pairwise association head over the measurement memory.

    Nothing in the collective average loss constrains how distinguishable the
    encoder memory is: the decoder can reach a low loss while every memory token
    collapses onto one direction, which is exactly what was measured (cosine
    similarity 0.9998 between unrelated measurements). This head projects each
    memory token onto a unit hypersphere and scores every pair, so an InfoNCE
    objective can require measurements of the same target to be close and
    everything else to be far apart.

    Follows the official MT3 contrastive classifier, with one deliberate
    difference: the cosine scores are divided by a temperature. Official MT3
    feeds raw cosines into the log_softmax, which bounds every score to [-1, 1].
    Over the ~200 measurements of a window that caps the objective at roughly
    ``1 - log(e + 199/e) = 3.3`` against a uniform baseline of ``log(200) = 5.3``:
    a 2-nat dynamic range with a vanishing gradient near the floor. Measured on
    v6, the term sat between 4.32 and 4.36 for 14000 consecutive steps while the
    encoder same-vs-different margin decayed from +0.83 to +0.11, so a
    near-optimal value is fully compatible with near-parallel memory and the term
    cannot prevent the collapse it exists to prevent.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        temperature: float = 0.1,
        learnable_temperature: bool = True,
    ):
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Parameterised as a log scale, as in CLIP, so the optimizer sees an
        # unconstrained variable and the scale can never reach zero or change sign.
        log_scale = torch.tensor(math.log(1.0 / temperature))
        if learnable_temperature:
            self.log_scale = nn.Parameter(log_scale)
        else:
            self.register_buffer("log_scale", log_scale)

    @property
    def scale(self) -> torch.Tensor:
        # Clamped so a runaway scale cannot saturate the softmax into all-or-nothing
        # rows; exp(log 1000) corresponds to a temperature of 0.001.
        return self.log_scale.clamp(max=math.log(1000.0)).exp()

    def forward(
        self, memory: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return row-wise log-probabilities over candidate partners.

        ``memory`` is ``(batch, measurements, hidden)`` and the result is
        ``(batch, measurements, measurements)`` with the diagonal and any padded
        entries masked out.
        """
        if memory.dim() != 3:
            raise ValueError("memory must be (batch, measurements, hidden)")
        batch, count, _ = memory.shape
        projected = F.normalize(self.projection(memory), dim=2)
        scores = (projected @ projected.transpose(1, 2)) * self.scale
        diagonal = torch.eye(count, dtype=torch.bool, device=memory.device)
        blocked = diagonal.unsqueeze(0).expand(batch, count, count)
        if padding_mask is not None:
            pairwise = padding_mask.unsqueeze(1).expand(batch, count, count)
            blocked = blocked | pairwise | pairwise.transpose(1, 2)
        # Rows that are entirely masked would make log_softmax produce NaN, so
        # they keep a finite score here and are dropped by the loss instead.
        empty_rows = blocked.all(dim=2, keepdim=True)
        scores = scores.masked_fill(blocked & ~empty_rows, float("-inf"))
        scores = scores.masked_fill(empty_rows & blocked, 0.0)
        return scores.log_softmax(dim=2)


def contrastive_association_loss(
    log_probabilities: torch.Tensor,
    measurement_ids: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """InfoNCE over measurements that share a target identity.

    Clutter carries id -1. Two clutter measurements are not associated with each
    other, so they must not be treated as a positive pair; only non-negative ids
    can form positives.
    """
    if log_probabilities.dim() != 3:
        raise ValueError("log_probabilities must be (batch, measurements, measurements)")
    batch, count, _ = log_probabilities.shape
    # Masked pairs hold -inf. Their weight is always zero, but 0 * -inf is NaN,
    # so the blocked scores are neutralised once, up front, and every downstream
    # expression uses the finite copy. Doing this only at the product site left
    # the degenerate branches below returning NaN.
    finite = log_probabilities.masked_fill(~torch.isfinite(log_probabilities), 0.0)
    identities = measurement_ids.to(log_probabilities.device)
    same = identities.unsqueeze(2) == identities.unsqueeze(1)
    real = (identities >= 0).unsqueeze(2) & (identities >= 0).unsqueeze(1)
    positives = (same & real).float()
    diagonal = torch.eye(count, dtype=torch.bool, device=log_probabilities.device)
    positives = positives.masked_fill(diagonal.unsqueeze(0), 0.0)
    if padding_mask is not None:
        pairwise = padding_mask.unsqueeze(1).expand(batch, count, count)
        positives = positives.masked_fill(pairwise | pairwise.transpose(1, 2), 0.0)
    counts = positives.sum(dim=2)
    eligible = counts > 0
    if not bool(eligible.any()):
        # A window can legitimately contain no positive pair: every measurement
        # is clutter, or each target was detected exactly once. Return zero while
        # keeping the graph connected so every DDP rank reduces the same buckets.
        return finite.sum() * 0.0
    # Average the log-likelihood over each measurement's positives so that a
    # target with many detections does not dominate the objective.
    weights = positives / counts.clamp(min=1.0).unsqueeze(2)
    per_measurement = -(finite * weights).sum(dim=2)
    return per_measurement[eligible].mean()
