from __future__ import annotations

import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from track_mt3.config import ExperimentConfig, resolve_device
from track_mt3.data.simulator import MultiTargetSimulator
from track_mt3.data.window import MeasurementWindow, build_sliding_windows
from track_mt3.evaluation import evaluate_model, load_trajectory
from track_mt3.losses import CollectiveAverageCriterion
from track_mt3.models import TrackMT3
from track_mt3.tracking import OnlineTracker


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_training_clips(
    config: ExperimentConfig,
    seed_sequence: np.random.SeedSequence,
    batch_size: int | None = None,
) -> list[list[MeasurementWindow]]:
    """Training clips of exactly ``clip_windows`` windows each.

    A fraction of the clips start at the beginning of a sequence, where fewer than
    ``window_size`` frames exist. Those windows are right-aligned and therefore only
    populate the highest time offsets, which is exactly the input the tracker sees at
    deployment time on frames 1..window_size-1. Without them the sequence start is an
    out-of-distribution input and the tracker can only be run from frame window_size.
    The window count per clip is unchanged, so every DDP rank keeps producing the same
    reduction buckets.
    """
    child_seeds = seed_sequence.spawn(batch_size or config.training.batch_size)
    clip_windows = config.training.clip_windows
    window_size = config.model.window_size
    steps = window_size + clip_windows - 1
    probability = config.training.cold_start_clip_probability
    clips: list[list[MeasurementWindow]] = []
    for child_seed in child_seeds:
        generator = np.random.default_rng(child_seed)
        cold_start = probability > 0.0 and generator.random() < probability
        if cold_start:
            # The first window of the clip starts anywhere in the incomplete-window
            # region, so across training the model sees partial windows holding any
            # number of frames from 1 to window_size, not only the shortest ones.
            first_end = int(generator.integers(0, window_size))
            frames = MultiTargetSimulator(config.simulation, child_seed).simulate(
                first_end + clip_windows
            )
            windows = build_sliding_windows(frames, window_size, include_partial=True)[first_end:]
        else:
            frames = MultiTargetSimulator(config.simulation, child_seed).simulate(steps)
            windows = build_sliding_windows(frames, window_size)
        if len(windows) != clip_windows:
            raise RuntimeError("training clip construction produced an unexpected number of windows")
        clips.append(windows)
    return clips


@dataclass
class TrainingState:
    step: int = 0
    best_loss: float = float("inf")
    best_validation_loss: float = float("inf")
    validations_without_improvement: int = 0
    best_evaluation_metric: float = float("inf")
    evaluations_without_improvement: int = 0


class Trainer:
    def __init__(self, config: ExperimentConfig, model: TrackMT3 | None = None):
        self.config = config
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.distributed = self.world_size > 1
        if self.distributed:
            if config.training.batch_size % self.world_size:
                raise ValueError(
                    f"global batch_size={config.training.batch_size} must be divisible by "
                    f"world_size={self.world_size}"
                )
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            dist.init_process_group(backend="nccl", device_id=self.device)
        else:
            self.device = torch.device(resolve_device(config.training.device))
        set_random_seed(config.training.seed + self.rank)
        raw_model = (model or TrackMT3(config)).to(self.device)
        self.model = (
            DistributedDataParallel(
                raw_model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=config.training.ddp_find_unused_parameters,
            )
            if self.distributed
            else raw_model
        )
        self.criterion = CollectiveAverageCriterion(config.loss)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=config.training.lr_patience,
            factor=config.training.lr_factor,
        )
        self.state = TrainingState()
        self.seed_sequence = np.random.SeedSequence([config.training.seed, self.rank])
        if config.training.curriculum_warmup_steps > 0:
            self._curriculum_terminal: dict[str, float | int] = {
                name: getattr(config.simulation, name)
                for name in self._CURRICULUM_SCHEDULE
            }
        else:
            self._curriculum_terminal = {}
        self.output_dir = Path(config.training.output_dir)
        if self.rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "checkpoints").mkdir(exist_ok=True)
            snapshot = config.to_dict()
            snapshot["distributed"] = {"world_size": self.world_size}
            with (self.output_dir / "config.json").open("w", encoding="utf-8") as handle:
                json.dump(snapshot, handle, indent=2, ensure_ascii=False)
        if self.distributed:
            dist.barrier()
        self.wandb_run = None
        self.loss_history: deque[float] = deque(maxlen=500)
        self.gradient_history: deque[float] = deque(maxlen=200)
        if self.rank == 0 and config.training.wandb_enabled:
            import wandb

            self.wandb_run = wandb.init(
                project=config.training.wandb_project,
                entity=config.training.wandb_entity,
                name=config.training.wandb_run_name,
                config=config.to_dict(),
                dir=str(self.output_dir),
                resume="allow",
            )
            wandb.watch(
                raw_model,
                log="gradients",
                log_freq=max(config.training.wandb_log_interval, 1) * 10,
                log_graph=False,
            )

    def _apply_loss_weight_schedule(self) -> dict[str, float]:
        """Move scheduled loss weights to their value for the current step.

        The criterion and the model both read ``config.loss``, so mutating it in place
        is enough for the whole loss graph to see the new weights.
        """
        applied: dict[str, float] = {}
        for name, specification in self.config.training.loss_weight_schedule.items():
            if not hasattr(self.config.loss, name):
                raise ValueError(f"unknown loss weight in schedule: {name}")
            start, end, steps = (float(value) for value in specification)
            progress = min(self.state.step / max(steps, 1.0), 1.0) if steps > 0 else 1.0
            value = start + progress * (end - start)
            setattr(self.config.loss, name, value)
            applied[f"weight_{name}"] = value
        return applied

    _CURRICULUM_SCHEDULE = {
        "clutter_rate": lambda end: 0.0,
        "detection_probability": lambda end: 1.0,
        "process_noise": lambda end: max(0.1, 0.2 * end),
        "measurement_noise": lambda end: max(0.02, 0.2 * end),
        "initial_targets": lambda end: 1,
        "max_targets": lambda end: max(4, end),
        "birth_rate": lambda end: 0.0,
        "survival_probability": lambda end: 1.0,
    }
    _CURRICULUM_INT_PARAMS = {"initial_targets", "max_targets"}

    def _apply_curriculum_schedule(self) -> dict[str, float]:
        """Ramp simulation difficulty from easy to full over ``curriculum_warmup_steps``.

        Terminal values are read once at init time (``_curriculum_terminal``) so
        that the schedule is not corrupted by previous calls overwriting
        ``config.simulation`` in-place. When ``curriculum_warmup_steps`` is 0
        the method returns an empty dict and does nothing.
        """
        warmup = self.config.training.curriculum_warmup_steps
        if warmup <= 0:
            return {}
        progress = min(self.state.step / max(warmup, 1), 1.0)
        applied: dict[str, float] = {}
        for name, start_fn in self._CURRICULUM_SCHEDULE.items():
            end = self._curriculum_terminal[name]
            start = start_fn(end)
            value = start + progress * (end - start)
            if name in self._CURRICULUM_INT_PARAMS:
                value = int(round(value))
            setattr(self.config.simulation, name, value)
            applied[f"sim_{name}"] = value
        return applied

    def train_step(self, clips: Iterable[list[MeasurementWindow]] | None = None) -> dict[str, float]:
        step_started = time.perf_counter()
        scheduled_weights = self._apply_loss_weight_schedule()
        scheduled_sim = self._apply_curriculum_schedule()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.model.train()
        data_started = time.perf_counter()
        local_batch_size = self.config.training.batch_size // self.world_size
        clip_list = (
            list(clips)
            if clips is not None
            else generate_training_clips(self.config, self.seed_sequence, local_batch_size)
        )
        if not clip_list:
            raise ValueError("at least one training clip is required")
        data_seconds = time.perf_counter() - data_started
        self.optimizer.zero_grad(set_to_none=True)
        decay_steps = self.config.training.teacher_forcing_decay_steps
        progress = min(self.state.step / max(decay_steps, 1), 1.0) if decay_steps > 0 else 1.0
        teacher_forcing_probability = (
            self.config.training.teacher_forcing_start
            + progress
            * (
                self.config.training.teacher_forcing_end
                - self.config.training.teacher_forcing_start
            )
        )
        total, localization, confidence, auxiliary, contrastive, proposal = self.model(
            clip_list,
            criterion=self.criterion,
            teacher_forcing_probability=teacher_forcing_probability,
        )
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite training loss at step {self.state.step}: {total.item()}")
        total.backward()
        if self.config.training.gradient_clip_norm > 0:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.training.gradient_clip_norm
            )
        else:
            gradient_norm = torch.tensor(0.0, device=self.device)
        self.optimizer.step()
        step_seconds = time.perf_counter() - step_started
        reduced_metrics = torch.stack(
            (total.detach(), localization, confidence, auxiliary, contrastive, proposal)
        )
        if self.distributed:
            dist.all_reduce(reduced_metrics, op=dist.ReduceOp.SUM)
            reduced_metrics /= self.world_size
        value = float(reduced_metrics[0])
        self.state.step += 1
        self.state.best_loss = min(self.state.best_loss, value)
        metrics = {
            "step": self.state.step,
            "loss": value,
            "localization": float(reduced_metrics[1]),
            "confidence": float(reduced_metrics[2]),
            "auxiliary": float(reduced_metrics[3]),
            "contrastive": float(reduced_metrics[4]),
            "proposal": float(reduced_metrics[5]),
            "gradient_norm": float(gradient_norm),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "data_seconds": data_seconds,
            "step_seconds": step_seconds,
            "clips_per_second": self.config.training.batch_size / step_seconds,
            "windows_per_second": (
                self.config.training.batch_size
                * self.config.training.clip_windows
                / step_seconds
            ),
            "teacher_forcing_probability": teacher_forcing_probability,
            **scheduled_weights,
            **scheduled_sim,
        }
        # The contrastive scale is learnable, and its trajectory is the cheapest
        # check that the term is actually being optimised rather than sitting on a
        # floor, which is how the pre-v7 head failed silently for 14000 steps.
        unwrapped = (
            self.model.module
            if isinstance(self.model, DistributedDataParallel)
            else self.model
        )
        if unwrapped.contrastive_classifier is not None:
            metrics["contrastive_scale"] = float(
                unwrapped.contrastive_classifier.scale.detach()
            )
        if self.device.type == "cuda":
            metrics.update(
                {
                    "gpu_memory_allocated_gb": torch.cuda.memory_allocated(self.device)
                    / 2**30,
                    "gpu_memory_reserved_gb": torch.cuda.memory_reserved(self.device)
                    / 2**30,
                    "gpu_peak_memory_gb": torch.cuda.max_memory_allocated(self.device)
                    / 2**30,
                }
            )
        if self.rank == 0:
            with (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics) + "\n")
        return metrics

    def checkpoint(self, name: str | None = None) -> Path:
        filename = name or f"step_{self.state.step:06d}.pt"
        path = self.output_dir / "checkpoints" / filename
        if self.rank != 0:
            return path
        model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "training_state": self.state.__dict__,
                "config": self.config.to_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(),
                "data_seed_children_spawned": self.seed_sequence.n_children_spawned,
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            path,
        )
        return path

    def prune_checkpoints(self) -> None:
        if self.rank != 0 or self.config.training.checkpoint_keep <= 0:
            return
        paths = sorted((self.output_dir / "checkpoints").glob("step_*.pt"))
        for path in paths[: -self.config.training.checkpoint_keep]:
            path.unlink()

    @torch.no_grad()
    def evaluate_checkpoint(self) -> dict[str, float]:
        if self.distributed:
            dist.barrier()
        flat_summary: dict[str, float] = {}
        if self.rank == 0:
            model = (
                self.model.module
                if isinstance(self.model, DistributedDataParallel)
                else self.model
            )
            dataset_dir = Path(self.config.training.evaluation_dataset_dir)
            evaluation_dir = self.output_dir / "checkpoint_evaluations"
            evaluation_dir.mkdir(exist_ok=True)
            summaries = {}
            for scene in ("scenario1", "scenario2", "scenario3"):
                paths = sorted((dataset_dir / scene).glob("run_*.npz"))
                if not paths:
                    raise FileNotFoundError(
                        f"no fixed evaluation trajectories found in {dataset_dir / scene}"
                    )
                trajectories = [load_trajectory(path) for path in paths]
                result = evaluate_model(
                    model, self.config, trajectories=trajectories
                )
                summary = result.detailed_summary()
                summaries[scene] = summary
                flat_summary.update(
                    {f"{scene}/{key}": value for key, value in summary.items()}
                )
                np.savez_compressed(
                    evaluation_dir / f"step_{self.state.step:06d}_{scene}.npz",
                    gospa=result.gospa,
                    pro_gospa=result.pro_gospa,
                    localization=result.localization,
                    missed=result.missed,
                    false=result.false,
                    mean_probability=result.mean_probability,
                    max_probability=result.max_probability,
                    predicted_count=result.predicted_count,
                    target_count=result.target_count,
                    matched_count=result.matched_count,
                )
            flat_summary["mean_gospa"] = float(
                np.mean([summary["gospa"] for summary in summaries.values()])
            )
            self._save_demo_tracking(model)
            flat_summary["mean_pro_gospa"] = float(
                np.mean([summary["pro_gospa"] for summary in summaries.values()])
            )
            record = {
                "step": self.state.step,
                "scenes": summaries,
                "selection": {
                    "mean_gospa": flat_summary["mean_gospa"],
                    "mean_pro_gospa": flat_summary["mean_pro_gospa"],
                },
            }
            with (self.output_dir / "checkpoint_evaluation.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(record) + "\n")
            print("checkpoint evaluation " + json.dumps(record, ensure_ascii=False))
            if self.wandb_run is not None:
                self.wandb_run.log(
                    {f"evaluation/{key}": value for key, value in flat_summary.items()},
                    step=self.state.step,
                )
            model.train()
        if self.distributed:
            payload = [flat_summary if self.rank == 0 else None]
            dist.broadcast_object_list(payload, src=0, device=self.device)
            flat_summary = payload[0]
            dist.barrier()
        return flat_summary

    @torch.no_grad()
    @torch.no_grad()
    def _save_demo_tracking(self, model) -> None:
        """Dump per-frame estimates on one fixed trajectory at every evaluation.

        Aggregate metrics say whether the tracker improved but not how. Keeping one
        trajectory's estimates per evaluation lets the run be replayed afterwards as a
        stack of pictures, in the style of the GLMB and MHT demos, without retaining one
        checkpoint per step. Partial windows are included so the sequence start shows up
        as well.
        """
        scene = self.config.training.demo_scene
        run = self.config.training.demo_run
        path = Path(self.config.training.evaluation_dataset_dir) / scene / f"run_{run:03d}.npz"
        if not path.exists():
            return
        frames = load_trajectory(path)
        tracker = OnlineTracker(model)
        threshold = self.config.evaluation.existence_threshold
        positions: list[np.ndarray] = []
        identifiers: list[np.ndarray] = []
        for window in build_sliding_windows(
            frames, self.config.model.window_size, include_partial=True
        ):
            output = tracker.step(window.to(self.device))
            probabilities = output.probabilities.reshape(-1).cpu().numpy()
            keep = probabilities > threshold
            positions.append(output.positions.cpu().numpy()[keep].astype(np.float32))
            identifiers.append(output.track_ids.cpu().numpy()[: len(probabilities)][keep])
        offsets = np.cumsum([0] + [len(item) for item in positions]).astype(np.int64)
        demo_dir = self.output_dir / "demo_tracking"
        demo_dir.mkdir(exist_ok=True)
        np.savez_compressed(
            demo_dir / f"step_{self.state.step:06d}.npz",
            step=np.asarray(self.state.step),
            scene=np.asarray(scene),
            run=np.asarray(run),
            positions=np.concatenate(positions).astype(np.float32)
            if offsets[-1]
            else np.zeros((0, 2), np.float32),
            track_ids=np.concatenate(identifiers).astype(np.int64)
            if offsets[-1]
            else np.zeros((0,), np.int64),
            offsets=offsets,
        )
        self._render_demo_progress(demo_dir)

    def _render_demo_progress(self, demo_dir: Path) -> None:
        """Redraw the stacked progress picture after each new demo dump.

        Metrics alone do not show *how* the tracker fails, so every evaluation appends a
        panel to one figure. Plotting is best-effort: a missing matplotlib or a broken
        backend must never take down a multi-hour training run.
        """
        if not self.config.training.demo_plot:
            return
        try:
            from track_mt3.visualization.progress import plot_demo_progress

            plot_demo_progress(
                demo_dir,
                Path(self.config.training.evaluation_dataset_dir),
                self.output_dir / "demo_tracking.png",
                max_panels=self.config.training.demo_plot_panels,
            )
        except Exception as error:  # noqa: BLE001 - plotting must never stop training
            print(f"demo progress plot skipped: {error}", flush=True)

    def validate(self) -> dict[str, float]:
        self.model.eval()
        local_batch_size = self.config.training.batch_size // self.world_size
        validation_seed = np.random.SeedSequence(
            [self.config.training.seed, self.rank, 1_000_003]
        )
        clips = generate_training_clips(self.config, validation_seed, local_batch_size)
        # Two passes over identical clips. The teacher-forced pass is directly
        # comparable with the training objective, while the free-running pass
        # measures the regime the fixed evaluation actually uses. Comparing only
        # one of them against the training loss confuses exposure bias with
        # optimization failure.
        results: dict[str, float] = {"step": self.state.step}
        for tag, probability in (("teacher_forced", 1.0), ("free_running", 0.0)):
            values = torch.stack(
                self.model(
                    clips,
                    criterion=self.criterion,
                    teacher_forcing_probability=probability,
                )
            )
            if self.distributed:
                dist.all_reduce(values, op=dist.ReduceOp.SUM)
                values /= self.world_size
            results[f"{tag}_loss"] = float(values[0])
            results[f"{tag}_localization"] = float(values[1])
            results[f"{tag}_confidence"] = float(values[2])
            results[f"{tag}_auxiliary"] = float(values[3])
            results[f"{tag}_contrastive"] = float(values[4])
            results[f"{tag}_proposal"] = float(values[5])
        # Early stopping and the learning-rate schedule follow the free-running
        # regime because that is what the fixed evaluation measures.
        results["loss"] = results["free_running_loss"]
        results["localization"] = results["free_running_localization"]
        results["confidence"] = results["free_running_confidence"]
        results["auxiliary"] = results["free_running_auxiliary"]
        results["contrastive"] = results["free_running_contrastive"]
        results["exposure_gap"] = (
            results["free_running_loss"] - results["teacher_forced_loss"]
        )
        self.model.train()
        return results

    def restore(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.state = TrainingState(**checkpoint["training_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if checkpoint.get("cuda_rng_state") is not None and torch.cuda.is_available():
            cuda_states = [state.cpu() for state in checkpoint["cuda_rng_state"]]
            # A rank-0 checkpoint contains all visible-device states. Each DDP
            # worker restores only its local generator; using set_rng_state_all
            # on every rank assigns states to the wrong local device.
            state_index = min(self.local_rank, len(cuda_states) - 1)
            torch.cuda.set_rng_state(cuda_states[state_index], self.device)
        spawned = int(checkpoint.get("data_seed_children_spawned", 0))
        self.seed_sequence = np.random.SeedSequence([self.config.training.seed, self.rank])
        if spawned:
            self.seed_sequence.spawn(spawned)

    def fit(self) -> None:
        stopped_early = False
        while self.state.step < self.config.training.updates:
            metrics = self.train_step()
            self.loss_history.append(metrics["loss"])
            self.gradient_history.append(metrics["gradient_norm"])
            total_loss = metrics["loss"]
            if total_loss:
                metrics["localization_fraction"] = metrics["localization"] / total_loss
                metrics["confidence_fraction"] = metrics["confidence"] / total_loss
                metrics["auxiliary_fraction"] = metrics["auxiliary"] / total_loss
            for window in (10, 50, 200, 500):
                values = list(self.loss_history)[-window:]
                if len(values) == window:
                    metrics[f"loss_ma_{window}"] = sum(values) / window
            if len(self.gradient_history) >= 50:
                metrics["gradient_norm_ma_50"] = (
                    sum(list(self.gradient_history)[-50:]) / 50
                )
            if self.rank == 0 and self.state.step % self.config.training.log_interval == 0:
                print(
                    f"step={self.state.step} loss={metrics['loss']:.6f} "
                    f"lr={metrics['learning_rate']:.3e} grad={metrics['gradient_norm']:.3f}"
                )
            if (
                self.wandb_run is not None
                and self.state.step % self.config.training.wandb_log_interval == 0
            ):
                self.wandb_run.log(
                    {f"train/{key}": value for key, value in metrics.items() if key != "step"},
                    step=self.state.step,
                )
            if self.state.step % self.config.training.checkpoint_interval == 0:
                self.checkpoint()
                self.prune_checkpoints()
                if self.config.training.evaluate_on_checkpoint:
                    evaluation = self.evaluate_checkpoint()
                    metric_name = self.config.training.early_stopping_metric
                    if metric_name not in evaluation:
                        raise ValueError(
                            f"unknown evaluation early-stopping metric: {metric_name}"
                        )
                    metric = evaluation[metric_name]
                    if (
                        metric
                        < self.state.best_evaluation_metric
                        - self.config.training.early_stopping_min_delta
                    ):
                        self.state.best_evaluation_metric = metric
                        self.state.evaluations_without_improvement = 0
                        self.checkpoint("best.pt")
                    else:
                        self.state.evaluations_without_improvement += 1
                    if self.wandb_run is not None:
                        self.wandb_run.log(
                            {
                                "early_stopping/evaluation_metric": metric,
                                "early_stopping/best_evaluation_metric": self.state.best_evaluation_metric,
                                "early_stopping/evaluations_without_improvement": self.state.evaluations_without_improvement,
                            },
                            step=self.state.step,
                        )
                    if (
                        self.config.training.early_stopping_patience > 0
                        and self.state.evaluations_without_improvement
                        >= self.config.training.early_stopping_patience
                    ):
                        stopped_early = True
                        if self.rank == 0:
                            print(
                                f"evaluation early stopping at step={self.state.step} "
                                f"best_{metric_name}={self.state.best_evaluation_metric:.6f}"
                            )
                        break
            if self.state.step % self.config.training.validation_interval == 0:
                validation = self.validate()
                # Plateau detection uses the validation loss, not a single noisy
                # training batch, so one patience unit is one validation round.
                self.scheduler.step(validation["loss"])
                if self.rank == 0:
                    with (self.output_dir / "validation.jsonl").open(
                        "a", encoding="utf-8"
                    ) as handle:
                        handle.write(json.dumps(validation) + "\n")
                    if self.wandb_run is not None:
                        payload = {
                            f"validation/{key}": value
                            for key, value in validation.items()
                            if key != "step"
                        }
                        payload["validation/best_loss"] = min(
                            self.state.best_validation_loss, validation["loss"]
                        )
                        payload["early_stopping/checks_without_improvement"] = (
                            self.state.validations_without_improvement
                        )
                        self.wandb_run.log(payload, step=self.state.step)
                if (
                    validation["loss"]
                    < self.state.best_validation_loss
                    - self.config.training.early_stopping_min_delta
                ):
                    self.state.best_validation_loss = validation["loss"]
                    self.state.validations_without_improvement = 0
                else:
                    self.state.validations_without_improvement += 1
        self.checkpoint("final.pt")
        if self.wandb_run is not None:
            self.wandb_run.summary["best_validation_loss"] = self.state.best_validation_loss
            self.wandb_run.summary["best_evaluation_metric"] = self.state.best_evaluation_metric
            self.wandb_run.summary["early_stopping_metric"] = (
                self.config.training.early_stopping_metric
            )
            self.wandb_run.summary["final_step"] = self.state.step
            self.wandb_run.summary["stopped_early"] = stopped_early
            self.wandb_run.finish()
        if self.distributed:
            dist.barrier()
            dist.destroy_process_group()
