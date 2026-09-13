"""Trajectory GOSPA metric used by the CNSF paper evaluation.

This is a small project-facing implementation of the LP T-GOSPA definition in
Garcia-Fernandez, Rahmathullah and Svensson (IEEE TSP 2020).  Its conventions
match the authors' reference Python implementation at commit ``8dcb0a6``:
unassigned cost ``c**p / 2`` and switching cost
``gamma**p / 2 * sum(abs(W[k+1] - W[k]))``.

Only tracker-native identities are accepted.  This module never links
detections across frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np
import scipy.sparse as sparse
from scipy.optimize import linprog


@dataclass(frozen=True)
class TGOSPAResult:
    total: float
    localization: float
    missed: float
    false: float
    switch: float


def _trajectory_tensor(
    positions: Sequence[np.ndarray], identities: Sequence[np.ndarray]
) -> tuple[np.ndarray, list[Hashable]]:
    """Convert frame outputs with native IDs to a [D,T,N] NaN-hole tensor."""
    if len(positions) != len(identities):
        raise ValueError("positions and identities must have equal frame counts")
    ids: list[Hashable] = []
    seen: set[Hashable] = set()
    dimension = 2
    for frame_positions, frame_ids in zip(positions, identities):
        pos = np.asarray(frame_positions)
        labels = np.asarray(frame_ids).reshape(-1)
        if pos.ndim != 2 or pos.shape[0] != labels.size:
            raise ValueError("each position frame must be [N,D] and match its IDs")
        if pos.shape[0]:
            dimension = int(pos.shape[1])
        if labels.size != len(set(labels.tolist())):
            raise ValueError("a native track ID occurs more than once in one frame")
        for label in labels.tolist():
            if label not in seen:
                ids.append(label)
                seen.add(label)
    tensor = np.full((dimension, len(positions), len(ids)), np.nan, dtype=np.float64)
    column = {label: index for index, label in enumerate(ids)}
    for time_index, (frame_positions, frame_ids) in enumerate(zip(positions, identities)):
        pos = np.asarray(frame_positions, dtype=np.float64).reshape(-1, dimension)
        for value, label in zip(pos, np.asarray(frame_ids).reshape(-1).tolist()):
            tensor[:, time_index, column[label]] = value
    return tensor, ids


def _connected_components(x: np.ndarray, y: np.ndarray, cutoff: float) -> list[tuple[list[int], list[int]]]:
    """Exact bipartite clustering used to split independent LPs."""
    nx, ny = x.shape[2], y.shape[2]
    adjacency: list[set[int]] = [set() for _ in range(nx + ny)]
    for i in range(nx):
        x_valid = ~np.isnan(x[0, :, i])
        for j in range(ny):
            valid = x_valid & ~np.isnan(y[0, :, j])
            if np.any(valid):
                distances = np.linalg.norm(x[:, valid, i] - y[:, valid, j], axis=0)
                if np.any(distances < cutoff):
                    adjacency[i].add(nx + j)
                    adjacency[nx + j].add(i)
    components: list[tuple[list[int], list[int]]] = []
    unseen = set(range(nx + ny))
    while unseen:
        root = unseen.pop()
        stack = [root]
        nodes = [root]
        while stack:
            node = stack.pop()
            for neighbour in adjacency[node]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
                    nodes.append(neighbour)
        components.append(
            ([node for node in nodes if node < nx], [node - nx for node in nodes if node >= nx])
        )
    return components


def _lp_component(x: np.ndarray, y: np.ndarray, c: float, p: float, gamma: float) -> TGOSPAResult:
    """Solve one connected component using the reference assignment LP."""
    nx, ny, horizon = x.shape[2], y.shape[2], x.shape[1]
    shape = (nx + 1, ny + 1)
    w_per_time = shape[0] * shape[1]
    w_count = horizon * w_per_time
    h_count = max(horizon - 1, 0) * nx * ny
    objective = np.zeros(w_count + h_count, dtype=np.float64)
    costs = np.zeros((nx + 1, ny + 1, horizon), dtype=np.float64)
    half = c**p / 2.0
    capped = c**p

    for time_index in range(horizon):
        x_real = ~np.isnan(x[0, time_index, :])
        y_real = ~np.isnan(y[0, time_index, :])
        for i in range(nx):
            for j in range(ny):
                if x_real[i] and y_real[j]:
                    costs[i, j, time_index] = min(
                        float(np.linalg.norm(x[:, time_index, i] - y[:, time_index, j], ord=p) ** p),
                        capped,
                    )
                elif x_real[i] != y_real[j]:
                    costs[i, j, time_index] = half
            if x_real[i]:
                costs[i, ny, time_index] = half
        for j in range(ny):
            if y_real[j]:
                costs[nx, j, time_index] = half

    objective[:w_count] = costs.reshape(-1, order="F")
    objective[w_count:] = gamma**p / 2.0

    # Every real X row and every real Y column sums to one at every time.
    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_data: list[float] = []
    rhs: list[float] = []
    row = 0
    for time_index in range(horizon):
        base = time_index * w_per_time
        for j in range(ny):
            for i in range(nx + 1):
                eq_rows.append(row); eq_cols.append(base + i + (nx + 1) * j); eq_data.append(1.0)
            rhs.append(1.0); row += 1
        for i in range(nx):
            for j in range(ny + 1):
                eq_rows.append(row); eq_cols.append(base + i + (nx + 1) * j); eq_data.append(1.0)
            rhs.append(1.0); row += 1
    a_eq = sparse.coo_matrix((eq_data, (eq_rows, eq_cols)), shape=(row, objective.size)).tocsr()

    # h_ijt >= |W_ij(t+1)-W_ij(t)| for every real-real pair.
    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_data: list[float] = []
    ub_row = 0
    for time_index in range(horizon - 1):
        for j in range(ny):
            for i in range(nx):
                now = time_index * w_per_time + i + (nx + 1) * j
                after = now + w_per_time
                h_index = w_count + time_index * nx * ny + j * nx + i
                for sign in (1.0, -1.0):
                    ub_rows.extend((ub_row, ub_row, ub_row))
                    ub_cols.extend((now, after, h_index))
                    ub_data.extend((sign, -sign, -1.0))
                    ub_row += 1
    a_ub = sparse.coo_matrix(
        (ub_data, (ub_rows, ub_cols)), shape=(ub_row, objective.size)
    ).tocsr()
    solved = linprog(
        objective,
        A_ub=a_ub if ub_row else None,
        b_ub=np.zeros(ub_row) if ub_row else None,
        A_eq=a_eq,
        b_eq=np.asarray(rhs),
        bounds=(0.0, None),
        method="highs",
    )
    if not solved.success:
        raise RuntimeError(f"T-GOSPA LP failed: {solved.message}")
    weights = solved.x[:w_count].reshape((nx + 1, ny + 1, horizon), order="F")

    localization = missed = false = 0.0
    for time_index in range(horizon):
        x_real = ~np.isnan(x[0, time_index, :])
        y_real = ~np.isnan(y[0, time_index, :])
        for i in range(nx):
            for j in range(ny):
                weight = weights[i, j, time_index]
                if x_real[i] and y_real[j]:
                    if costs[i, j, time_index] < capped:
                        localization += weight * costs[i, j, time_index]
                    else:
                        missed += weight * half
                        false += weight * half
                elif x_real[i]:
                    missed += weight * half
                elif y_real[j]:
                    false += weight * half
            if x_real[i]:
                missed += weights[i, ny, time_index] * half
        for j in range(ny):
            if y_real[j]:
                false += weights[nx, j, time_index] * half
    switch = (
        gamma**p / 2.0 * float(np.abs(np.diff(weights[:nx, :ny, :], axis=2)).sum())
        if horizon > 1 else 0.0
    )
    powered_total = localization + missed + false + switch
    return TGOSPAResult(
        total=float(powered_total ** (1.0 / p)),
        localization=float(localization),
        missed=float(missed),
        false=float(false),
        switch=float(switch),
    )


def trajectory_gospa(
    truth_positions: Sequence[np.ndarray],
    truth_ids: Sequence[np.ndarray],
    estimate_positions: Sequence[np.ndarray],
    estimate_ids: Sequence[np.ndarray],
    *,
    c: float = 2.0,
    p: float = 1.0,
    gamma: float = 1.0,
) -> TGOSPAResult:
    """Compute LP T-GOSPA and its four reference decomposition components."""
    if c <= 0 or p < 1 or gamma <= 0:
        raise ValueError("T-GOSPA requires c>0, p>=1 and gamma>0")
    if len(truth_positions) != len(estimate_positions):
        raise ValueError("truth and estimate horizons differ")
    x, _ = _trajectory_tensor(truth_positions, truth_ids)
    y, _ = _trajectory_tensor(estimate_positions, estimate_ids)
    if x.shape[0] != y.shape[0] and x.shape[2] and y.shape[2]:
        raise ValueError("truth and estimate state dimensions differ")
    horizon = x.shape[1]
    if x.shape[2] == 0 and y.shape[2] == 0:
        return TGOSPAResult(0.0, 0.0, 0.0, 0.0, 0.0)
    result = np.zeros(5, dtype=np.float64)
    for x_indices, y_indices in _connected_components(x, y, c):
        if not x_indices:
            value = c**p / 2.0 * float(np.count_nonzero(~np.isnan(y[0, :, y_indices])))
            result[0] += value; result[3] += value
        elif not y_indices:
            value = c**p / 2.0 * float(np.count_nonzero(~np.isnan(x[0, :, x_indices])))
            result[0] += value; result[2] += value
        else:
            part = _lp_component(x[:, :, x_indices], y[:, :, y_indices], c, p, gamma)
            result += np.asarray([part.total**p, part.localization, part.missed, part.false, part.switch])
    return TGOSPAResult(
        total=float(result[0] ** (1.0 / p)),
        localization=float(result[1]),
        missed=float(result[2]),
        false=float(result[3]),
        switch=float(result[4]),
    )
