from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linear_sum_assignment

from .gospa import _points


def pro_gospa(
    estimate_states: ArrayLike,
    existence_probabilities: ArrayLike,
    targets: ArrayLike,
    *,
    cutoff: float = 2.0,
    order: int = 1,
    alpha: float = 2.0,
) -> float:
    """Probabilistic GOSPA from equation (56) of the Track-MT3 paper.

    The published equation specializes the Bernoulli-set penalty to alpha=2.
    We retain alpha as an explicit guard so accidental metric drift is visible.
    """
    if cutoff <= 0 or order < 1 or alpha != 2:
        raise ValueError("paper Pro-GOSPA requires cutoff>0, order>=1 and alpha=2")
    estimates = _points(estimate_states)
    truth = _points(targets)
    probabilities = np.asarray(existence_probabilities, dtype=np.float64).reshape(-1)
    if len(probabilities) != len(estimates):
        raise ValueError("one existence probability is required per estimate")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("existence probabilities must be in [0, 1]")

    cutoff_power = cutoff**order
    if len(estimates) and len(truth):
        distances = np.linalg.norm(estimates[:, None, :] - truth[None, :, :], axis=-1)
        truncated = np.minimum(distances, cutoff) ** order
        pair_cost = truncated * probabilities[:, None] + (1.0 - probabilities[:, None]) * cutoff_power / 2.0
        estimate_indices, target_indices = linear_sum_assignment(pair_cost)
        matched_cost = float(pair_cost[estimate_indices, target_indices].sum())
        matched_estimates = np.zeros(len(estimates), dtype=bool)
        matched_estimates[estimate_indices] = True
        unmatched_estimate_cost = float(probabilities[~matched_estimates].sum() * cutoff_power / 2.0)
        missed_cost = float((len(truth) - len(target_indices)) * cutoff_power / 2.0)
    else:
        matched_cost = 0.0
        unmatched_estimate_cost = float(probabilities.sum() * cutoff_power / 2.0)
        missed_cost = float(len(truth) * cutoff_power / 2.0)
    return float((matched_cost + unmatched_estimate_cost + missed_cost) ** (1.0 / order))

