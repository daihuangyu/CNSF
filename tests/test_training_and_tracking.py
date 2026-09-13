from pathlib import Path

import numpy as np
import pytest
import torch

from track_mt3.config import load_config
from track_mt3.config_merge import load_merged_config
from track_mt3.data import MultiTargetSimulator, build_sliding_windows
from track_mt3.evaluation import paper_complexity
from track_mt3.models import TrackMT3
from track_mt3.tracking import OnlineTracker
from track_mt3.evaluation import load_trajectory
from track_mt3.training import Trainer, generate_training_clips


def save_test_trajectory(path: Path, config, frames: int = 100) -> None:
    trajectory = MultiTargetSimulator(config.simulation, 123).simulate(frames)
    state_lengths = [len(frame.states) for frame in trajectory]
    measurement_lengths = [len(frame.measurements) for frame in trajectory]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frame_indices=np.asarray([frame.index for frame in trajectory]),
        times=np.asarray([frame.time for frame in trajectory]),
        state_offsets=np.concatenate(([0], np.cumsum(state_lengths))),
        states=np.concatenate([frame.states for frame in trajectory]),
        target_ids=np.concatenate([frame.target_ids for frame in trajectory]),
        measurement_offsets=np.concatenate(([0], np.cumsum(measurement_lengths))),
        measurements=np.concatenate([frame.measurements for frame in trajectory]),
        measurement_ids=np.concatenate([frame.measurement_ids for frame in trajectory]),
    )


def test_partial_configs_merge_without_losing_base_values():
    config = load_merged_config("configs/paper.yaml", "configs/experiments/scenario2.yaml")
    assert config.model.hidden_dim == 256
    assert config.simulation.birth_rate == 0.08
    assert config.simulation.detection_probability == 0.90


def test_training_checkpoint_restore(tmp_path: Path):
    config = load_config("configs/smoke.yaml")
    config.training.output_dir = str(tmp_path / "first")
    config.training.batch_size = 1
    trainer = Trainer(config)
    metrics = trainer.train_step()
    assert metrics["step"] == 1
    checkpoint = trainer.checkpoint()
    restored_config = load_config("configs/smoke.yaml")
    restored_config.training.output_dir = str(tmp_path / "restored")
    restored_config.training.batch_size = 1
    restored = Trainer(restored_config)
    restored.restore(checkpoint)
    assert restored.state.step == 1
    for original, loaded in zip(trainer.model.parameters(), restored.model.parameters()):
        torch.testing.assert_close(original, loaded)
    original_next = generate_training_clips(config, trainer.seed_sequence)[0][0]
    restored_next = generate_training_clips(restored_config, restored.seed_sequence)[0][0]
    torch.testing.assert_close(original_next.measurements, restored_next.measurements)
    torch.testing.assert_close(original_next.target_positions, restored_next.target_positions)


def test_online_tracker_assigns_unique_monotonic_ids():
    config = load_config("configs/smoke.yaml")
    config.model.detection_threshold = 0.0
    config.model.tracking_threshold = 0.0
    model = TrackMT3(config)
    frames = MultiTargetSimulator(config.simulation, 77).simulate(6)
    windows = build_sliding_windows(frames, config.model.window_size)
    tracker = OnlineTracker(model)
    first = tracker.step(windows[0])
    assert first.track_ids.tolist() == list(range(len(first.track_ids)))
    second = tracker.step(windows[1])
    assert len(set(second.track_ids.tolist())) == len(second.track_ids)
    assert tracker.next_track_id >= len(first.track_ids)


def test_streaming_cache_encodes_only_the_new_frame_and_preserves_history():
    config = load_config("configs/smoke.yaml")
    config.model.window_size = 1
    config.model.memory_frames = 0
    config.model.streaming_cache_frames = 3
    config.model.streaming_cache_tokens_per_frame = 32
    config.model.contrastive_classifier = False
    model = TrackMT3(config).eval()
    frames = MultiTargetSimulator(config.simulation, 91).simulate(4)
    windows = build_sliding_windows(frames, 1)
    tracker = OnlineTracker(model)
    encoded_lengths = []
    handle = model.encoder.register_forward_pre_hook(
        lambda _module, args: encoded_lengths.append(args[0].shape[1])
    )
    tracker.step(windows[0])
    first = tracker.tracks.measurement_cache.detach().clone()
    tracker.step(windows[1])
    handle.remove()
    second = tracker.tracks.measurement_cache.detach()
    assert encoded_lengths == [max(len(windows[0].measurements), 1), max(len(windows[1].measurements), 1)]
    torch.testing.assert_close(second[:, -2], first[:, -1])
    assert second.shape == (1, 3, 32, config.model.hidden_dim)


def test_streaming_decoder_memory_drops_only_masked_cache_slots():
    config = load_config("configs/smoke.yaml")
    config.model.window_size = 1
    config.model.memory_frames = 0
    config.model.streaming_cache_frames = 4
    config.model.streaming_cache_tokens_per_frame = 32
    config.model.contrastive_classifier = False
    model = TrackMT3(config).eval()
    frames = MultiTargetSimulator(config.simulation, 92).simulate(3)
    windows = build_sliding_windows(frames, 1)
    tracks = [model.empty_tracks()]
    seen_lengths = []
    handle = model.decoder.register_forward_pre_hook(
        lambda _module, args: seen_lengths.append(args[1].shape[1])
    )
    with torch.no_grad():
        for window in windows:
            prediction = model.forward_windows([window], tracks)[0]
            tracks[0], _ = model.qtm(prediction, tracks[0])
    handle.remove()
    expected = []
    running = 0
    for window in windows:
        running += max(len(window.measurements), 1)
        expected.append(running)
    assert seen_lengths == expected
    assert all(length < 4 * 32 for length in seen_lengths)


def test_paper_complexity_increases_with_measurement_count():
    config = load_config("configs/smoke.yaml")
    small = paper_complexity(10, config.model, adjacent_max_targets=3, frame_max_targets=3)
    large = paper_complexity(20, config.model, adjacent_max_targets=3, frame_max_targets=3)
    assert small.total > 0
    assert large.total > small.total


def test_tiny_fixed_clip_can_be_overfit(tmp_path: Path):
    """The collective average loss alone must be drivable down on one clip.

    The contrastive term is disabled here so this stays a test of the set
    prediction objective; mixing in a second objective with weight 4 would make
    the threshold measure the sum of two unrelated quantities.
    """
    config = load_config("configs/smoke.yaml")
    config.training.output_dir = str(tmp_path / "overfit")
    config.training.batch_size = 1
    config.training.learning_rate = 5e-4
    config.model.contrastive_classifier = False
    trainer = Trainer(config)
    clip = generate_training_clips(config, np.random.SeedSequence(123))
    first = trainer.train_step(clip)["loss"]
    last = first
    for _ in range(29):
        last = trainer.train_step(clip)["loss"]
    assert last < first * 0.5


def test_contrastive_term_separates_memory_on_a_fixed_clip(tmp_path: Path):
    """The contrastive objective must be optimizable and must reduce collapse.

    Cosine similarity between measurements of different targets is the quantity
    that governs whether cross-attention can discriminate at all: once every key
    is parallel the shared component cancels in the softmax.
    """
    config = load_config("configs/smoke.yaml")
    config.training.output_dir = str(tmp_path / "contrastive")
    config.training.batch_size = 1
    config.training.learning_rate = 5e-4
    config.model.contrastive_classifier = True
    trainer = Trainer(config)
    clip = generate_training_clips(config, np.random.SeedSequence(123))

    def collapse() -> float:
        window = clip[0][0]
        with torch.no_grad():
            memory, _, _, _ = trainer.model.encode_window(window)
        features = memory[0, : len(window.measurements)]
        normalized = features / features.norm(dim=-1, keepdim=True)
        similarity = normalized @ normalized.T
        off_diagonal = ~torch.eye(len(features), dtype=torch.bool)
        return float(similarity[off_diagonal].mean())

    first = trainer.train_step(clip)["contrastive"]
    before = collapse()
    for _ in range(59):
        metrics = trainer.train_step(clip)
    assert metrics["contrastive"] < first
    assert collapse() < before


def test_loss_weight_schedule_interpolates_and_reaches_the_model(tmp_path: Path):
    """Scheduled weights must move with the step count and be visible to the loss."""
    config = load_config("configs/smoke.yaml")
    config.training.output_dir = str(tmp_path / "schedule")
    config.training.batch_size = 1
    config.loss.localization_weight = 1.0
    config.training.loss_weight_schedule = {"localization_weight": [1.0, 5.0, 4]}
    trainer = Trainer(config)
    clip = generate_training_clips(config, np.random.SeedSequence(7))
    weights = [trainer.train_step(clip)["weight_localization_weight"] for _ in range(6)]
    assert weights[0] == 1.0
    assert weights[1] > weights[0]
    assert weights[-1] == 5.0
    assert config.loss.localization_weight == 5.0


def test_cold_start_clips_keep_the_window_count_and_cover_the_sequence_start():
    """Cold-start clips must stay shape-compatible so DDP reductions stay identical."""
    config = load_config("configs/smoke.yaml")
    config.training.cold_start_clip_probability = 1.0
    clips = generate_training_clips(config, np.random.SeedSequence(11), 8)
    assert all(len(clip) == config.training.clip_windows for clip in clips)
    # Cold-start clips begin somewhere inside the incomplete-window region, so their
    # first window covers fewer than window_size frames.
    assert all(clip[0].end_index < config.model.window_size for clip in clips)
    assert any(clip[0].end_index > 0 for clip in clips)
    config.training.cold_start_clip_probability = 0.0
    warm = generate_training_clips(config, np.random.SeedSequence(11), 4)
    assert all(clip[0].end_index == config.model.window_size - 1 for clip in warm)


def test_demo_tracking_dump_is_written_and_covers_every_frame(tmp_path: Path):
    """The evaluation-time demo dump must cover all frames, including partial windows."""
    config = load_config("configs/smoke.yaml")
    config.training.output_dir = str(tmp_path / "demo")
    config.training.batch_size = 1
    config.training.demo_scene = "scenario3"
    config.training.demo_run = 0
    dataset_dir = tmp_path / "evaluation"
    trajectory_path = dataset_dir / "scenario3" / "run_000.npz"
    save_test_trajectory(trajectory_path, config)
    config.training.evaluation_dataset_dir = str(dataset_dir)
    trainer = Trainer(config)
    trainer._save_demo_tracking(trainer.model)
    dumps = sorted((Path(config.training.output_dir) / "demo_tracking").glob("step_*.npz"))
    assert len(dumps) == 1
    with np.load(dumps[0]) as archive:
        frames = len(load_trajectory(trajectory_path))
        assert len(archive["offsets"]) == frames + 1
        assert archive["positions"].shape[1] == 2
        assert len(archive["track_ids"]) == len(archive["positions"])


def test_demo_progress_figure_is_written_from_dumps(tmp_path: Path):
    """The in-training progress figure must render from the dumped estimates alone."""
    from track_mt3.visualization import plot_demo_progress

    demo_dir = tmp_path / "demo_tracking"
    demo_dir.mkdir()
    frames = 100
    for step in (1000, 2000):
        counts = np.full(frames, 2)
        offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        np.savez_compressed(
            demo_dir / f"step_{step:06d}.npz",
            step=np.asarray(step),
            scene=np.asarray("scenario3"),
            run=np.asarray(0),
            positions=np.zeros((int(offsets[-1]), 2), dtype=np.float32),
            track_ids=np.zeros((int(offsets[-1]),), dtype=np.int64),
            offsets=offsets,
        )
    config = load_config("configs/smoke.yaml")
    dataset_dir = tmp_path / "evaluation"
    save_test_trajectory(dataset_dir / "scenario3" / "run_000.npz", config)
    output = plot_demo_progress(
        demo_dir,
        dataset_dir,
        tmp_path / "progress.png",
        max_panels=4,
    )
    assert output.exists() and output.stat().st_size > 0


def test_curriculum_schedule_ramps_simulation_parameters(tmp_path: Path):
    config = load_config("configs/smoke.yaml")
    config.training.output_dir = str(tmp_path / "curriculum")
    config.training.batch_size = 1
    config.training.curriculum_warmup_steps = 4
    config.simulation.clutter_rate = 10.0
    config.simulation.process_noise = 0.08
    config.simulation.initial_targets = 8
    trainer = Trainer(config)
    clip = generate_training_clips(config, np.random.SeedSequence(42))
    metrics_0 = trainer.train_step(clip)
    assert metrics_0["sim_clutter_rate"] == 0.0
    assert metrics_0["sim_detection_probability"] == 1.0
    assert metrics_0["sim_initial_targets"] == 1
    assert metrics_0["sim_process_noise"] == pytest.approx(max(0.1, 0.2 * 0.08))
    for _ in range(2):
        trainer.train_step(clip)
    metrics_3 = trainer.train_step(clip)
    assert metrics_3["sim_clutter_rate"] > 0.0
    assert metrics_3["sim_clutter_rate"] < 10.0
    assert metrics_3["sim_detection_probability"] < 1.0
    for _ in range(4):
        trainer.train_step(clip)
    final_clutter = config.simulation.clutter_rate
    assert final_clutter == pytest.approx(10.0)
    assert config.simulation.detection_probability == pytest.approx(0.9)
    assert config.simulation.initial_targets == 8


def test_curriculum_disabled_when_warmup_is_zero(tmp_path: Path):
    config = load_config("configs/smoke.yaml")
    config.training.output_dir = str(tmp_path / "no_curriculum")
    config.training.batch_size = 1
    config.training.curriculum_warmup_steps = 0
    config.simulation.clutter_rate = 10.0
    trainer = Trainer(config)
    clip = generate_training_clips(config, np.random.SeedSequence(42))
    metrics = trainer.train_step(clip)
    assert "sim_clutter_rate" not in metrics
    assert config.simulation.clutter_rate == 10.0
