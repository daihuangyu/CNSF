import numpy as np

from track_mt3.config import SimulationConfig
from track_mt3.data.motion import process_covariance, transition_matrix
from track_mt3.data.simulator import MultiTargetSimulator
from track_mt3.data.window import build_sliding_windows, pad_measurement_windows
from track_mt3.metrics import gospa, pro_gospa


def test_constant_velocity_transition_and_covariance():
    state = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(transition_matrix(0.1) @ state, [1.3, 2.4, 3.0, 4.0])
    covariance = process_covariance(0.1, 0.5)
    np.testing.assert_allclose(covariance, covariance.T)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-12


def test_simulator_is_reproducible_and_ids_are_consistent():
    config = SimulationConfig(initial_targets=3, birth_rate=0.0, survival_probability=1.0, clutter_rate=2.0)
    first = MultiTargetSimulator(config, 12).simulate(5)
    second = MultiTargetSimulator(config, 12).simulate(5)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left.target_ids, right.target_ids)
        np.testing.assert_allclose(left.states, right.states)
        np.testing.assert_allclose(left.measurements, right.measurements)
        assert set(left.measurement_ids[left.measurement_ids >= 0]).issubset(set(left.target_ids))


def test_sliding_window_shapes_and_padding():
    config = SimulationConfig(initial_targets=2, birth_rate=0.0, clutter_rate=1.0)
    frames = MultiTargetSimulator(config, 2).simulate(7)
    windows = build_sliding_windows(frames, 4)
    assert len(windows) == 4
    assert windows[0].end_index == 3
    assert windows[-1].end_index == 6
    assert windows[0].time_indices.min() >= 0
    assert windows[0].time_indices.max() <= 3
    measurements, times, mask, measurement_ids = pad_measurement_windows(windows[:2])
    assert measurements.shape[:2] == times.shape == mask.shape == measurement_ids.shape
    assert (~mask).sum().item() == sum(len(window.measurements) for window in windows[:2])
    # Padding must stay distinguishable from clutter (-1) so the contrastive
    # objective cannot pair padded slots with each other.
    assert bool((measurement_ids[mask] == -2).all())
    assert bool((measurement_ids[~mask] >= -1).all())


def test_gospa_hand_calculated_cases():
    empty = np.empty((0, 2))
    assert gospa(empty, empty).value == 0
    missed = gospa(empty, [[0.0, 0.0]])
    assert missed.value == 1.0
    assert missed.missed == 1.0
    localized = gospa([[0.25, 0.0]], [[0.0, 0.0]])
    assert localized.value == 0.25
    assert localized.localization == 0.25
    cutoff_pair = gospa([[100.0, 0.0]], [[0.0, 0.0]])
    assert cutoff_pair.value == 2.0
    assert cutoff_pair.missed == 1.0
    assert cutoff_pair.false == 1.0


def test_pro_gospa_probability_boundaries():
    truth = np.array([[0.0, 0.0]])
    assert pro_gospa([[0.0, 0.0]], [1.0], truth) == 0.0
    assert pro_gospa([[0.0, 0.0]], [0.0], truth) == 1.0
    assert pro_gospa(np.empty((0, 2)), [], truth) == 1.0
    assert pro_gospa([[0.0, 0.0]], [1.0], np.empty((0, 2))) == 1.0



def test_partial_windows_are_right_aligned_and_start_at_frame_zero():
    """Frame k must be predictable from k+1 frames, with the current frame at the top offset."""
    import torch

    from track_mt3.config import load_config
    from track_mt3.data.simulator import MultiTargetSimulator
    from track_mt3.data.window import build_sliding_windows, pad_measurement_windows

    config = load_config("configs/paper.yaml")
    window_size = config.model.window_size
    frames = MultiTargetSimulator(config.simulation, 5).simulate(window_size + 3)
    full = build_sliding_windows(frames, window_size)
    partial = build_sliding_windows(frames, window_size, include_partial=True)
    assert len(partial) == len(frames)
    assert [window.end_index for window in partial[: len(frames)]] == [
        frame.index for frame in frames
    ]
    # Right alignment keeps full windows bit-identical to the left-aligned numbering.
    for reference, candidate in zip(full, partial[window_size - 1 :]):
        assert reference.end_index == candidate.end_index
        assert torch.equal(reference.time_indices, candidate.time_indices)
    for index, window in enumerate(partial[: window_size - 1]):
        covered = index + 1
        if len(window.time_indices):
            assert int(window.time_indices.min()) >= window_size - covered
            assert int(window.time_indices.max()) <= window_size - 1
    # An empty window must keep one unmasked slot, otherwise attention returns NaN.
    empty = partial[0].__class__(
        end_index=0,
        measurements=torch.zeros((0, 2)),
        time_indices=torch.zeros((0,), dtype=torch.long),
        measurement_ids=torch.zeros((0,), dtype=torch.long),
        target_positions=torch.zeros((0, 2)),
        target_ids=torch.zeros((0,), dtype=torch.long),
    )
    _, _, mask, _ = pad_measurement_windows([empty])
    assert not bool(mask.all())
