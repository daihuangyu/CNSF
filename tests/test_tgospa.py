from __future__ import annotations

import numpy as np

from track_mt3.metrics.tgospa import trajectory_gospa


def _frames(values):
    return [np.asarray(frame, dtype=np.float64).reshape(-1, 2) for frame in values]


def _ids(values):
    return [np.asarray(frame, dtype=np.int64) for frame in values]


def test_perfect_trajectories_are_zero():
    positions = _frames([[[0, 0], [4, 0]], [[1, 0], [3, 0]], [[2, 0], [2, 1]]])
    identities = _ids([[10, 20], [10, 20], [10, 20]])
    result = trajectory_gospa(positions, identities, positions, identities)
    assert result.total < 1e-10
    assert result.switch < 1e-10


def test_identity_swap_has_switch_cost_but_zero_frame_error():
    truth = _frames([[[0, 0], [4, 0]], [[1, 0], [3, 0]], [[2, 0], [2, 1]], [[3, 0], [1, 1]]])
    truth_ids = _ids([[10, 20]] * 4)
    # Positions remain a perfect set, but prediction identities exchange after t=1.
    prediction = _frames([[[0, 0], [4, 0]], [[1, 0], [3, 0]], [[2, 1], [2, 0]], [[1, 1], [3, 0]]])
    prediction_ids = _ids([[100, 200]] * 4)
    result = trajectory_gospa(truth, truth_ids, prediction, prediction_ids)
    assert result.localization < 1e-10
    assert result.missed < 1e-10
    assert result.false < 1e-10
    assert result.total > 0
    assert result.switch > 0


def test_empty_sets_and_one_sided_empty():
    empty_positions = _frames([[], [], []])
    empty_ids = _ids([[], [], []])
    result = trajectory_gospa(empty_positions, empty_ids, empty_positions, empty_ids)
    assert result.total == 0
    truth = _frames([[[0, 0]], [[1, 0]], [[2, 0]]])
    truth_ids = _ids([[7], [7], [7]])
    missed = trajectory_gospa(truth, truth_ids, empty_positions, empty_ids)
    assert missed.total == 3.0
    assert missed.missed == 3.0
    assert missed.false == 0.0
    assert missed.switch == 0.0
