from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class GospaResult:
    value: float
    localization: float
    missed: float
    false: float
    matched_pairs: tuple[tuple[int, int], ...]


def _points(value: ArrayLike) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("point sets must have shape [N, D]")
    return array


def gospa(
    estimates: ArrayLike,
    targets: ArrayLike,
    *,
    cutoff: float = 2.0,
    order: int = 1,
    alpha: float = 2.0,
) -> GospaResult:
    """Compute GOSPA and its localization/missed/false decomposition."""
    if cutoff <= 0 or order < 1 or not 0 < alpha <= 2:
        raise ValueError("require cutoff>0, order>=1 and 0<alpha<=2")
    x = _points(estimates)
    y = _points(targets)
    n_estimates, n_targets = len(x), len(y)
    cutoff_power = cutoff**order
    if n_estimates and n_targets:
        distances = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)
        costs = np.minimum(distances, cutoff) ** order
        estimate_indices, target_indices = linear_sum_assignment(costs)
        assigned_costs = costs[estimate_indices, target_indices]
        valid = assigned_costs < cutoff_power
        localization_power = float(assigned_costs[valid].sum())
        pairs = tuple(
            (int(i), int(j)) for i, j, keep in zip(estimate_indices, target_indices, valid) if bool(keep)
        )
        matched = int(valid.sum())
    else:
        localization_power = 0.0
        pairs = ()
        matched = 0
    missed_power = cutoff_power / alpha * (n_targets - matched)
    false_power = cutoff_power / alpha * (n_estimates - matched)
    total_power = localization_power + missed_power + false_power
    root = 1.0 / order
    return GospaResult(
        value=float(total_power**root),
        localization=float(localization_power**root),
        missed=float(missed_power**root),
        false=float(false_power**root),
        matched_pairs=pairs,
    )

