from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from track_mt3.data.simulator import Frame
from track_mt3.metrics import gospa, pro_gospa
from track_mt3.models_v17.batching import pad_current_frames
from track_mt3.models_v17.association_criterion import V17BAssociationCriterion
from track_mt3.models_v17.criterion import V17AOracleCriterion


def average_gradients(model: torch.nn.Module, world_size: int) -> None:
    """Average every parameter gradient in an identical collective order.

    Oracle birth/death events differ across ranks.  A head can therefore be
    unused on one rank (``grad is None``) while receiving a gradient on another.
    Every rank must still enter the corresponding collective; the unused rank
    contributes an explicit zero tensor.
    """

    if world_size == 1:
        return
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)


class V17ASequenceTrainer:
    """Small, inspectable trainer for the v17-A oracle filter gate."""

    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        criterion: V17AOracleCriterion,
        *,
        burn_in_frames: int = 0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.burn_in_frames = burn_in_frames

    @property
    def raw_model(self):
        return (
            self.model.module
            if isinstance(self.model, DistributedDataParallel)
            else self.model
        )

    def _sequence_loss(self, trajectories: Sequence[Sequence[Frame]]):
        if not trajectories or not trajectories[0]:
            raise ValueError("at least one non-empty trajectory is required")
        steps = len(trajectories[0])
        if any(len(trajectory) != steps for trajectory in trajectories):
            raise ValueError("all trajectories must have equal lengths")
        state = None
        losses = []
        for time_index in range(steps):
            batch = pad_current_frames(
                [trajectory[time_index] for trajectory in trajectories],
                self.raw_model.device,
            )
            measurements, mask, measurement_ids, frame_time, truth_states, truth_ids = (
                batch
            )
            if time_index < self.burn_in_frames:
                with torch.no_grad():
                    _, state = self.model(
                        measurements,
                        mask,
                        measurement_ids,
                        frame_time,
                        truth_ids,
                        state,
                    )
                state = state.detach()
                continue
            output, state = self.model(
                measurements, mask, measurement_ids, frame_time, truth_ids, state
            )
            losses.append(self.criterion(output, truth_states, truth_ids))
        if not losses:
            raise ValueError("burn_in_frames must be shorter than the trajectory")
        total = torch.stack([loss.total for loss in losses]).mean()
        diagnostics = {
            name: float(torch.stack([getattr(loss, name) for loss in losses]).mean())
            for name in (
                "localization",
                "confidence",
                "prior_nll",
                "posterior_nll",
                "prior_position_error",
                "posterior_position_error",
            )
        }
        diagnostics["matched_existing"] = sum(
            loss.matched_existing for loss in losses
        ) / len(losses)
        diagnostics["matched_births"] = sum(
            loss.matched_births for loss in losses
        ) / len(losses)
        return total, diagnostics

    def train_step(self, trajectories: Sequence[Sequence[Frame]]) -> dict[str, float]:
        self.model.train()
        started = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        total, metrics = self._sequence_loss(trajectories)
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite v17-A loss: {float(total)}")
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        return {
            "loss": float(total.detach()),
            "gradient_norm": float(gradient_norm),
            "step_seconds": time.perf_counter() - started,
            **metrics,
        }

    @torch.no_grad()
    def evaluate_oracle_filter(
        self,
        trajectories: Sequence[Sequence[Frame]],
        *,
        first_scored_frame: int = 0,
    ) -> dict[str, float]:
        """Evaluate the filter upper bound with oracle birth output selection."""

        self.raw_model.eval()
        state = None
        gospa_values: list[float] = []
        pro_values: list[float] = []
        prior_errors: list[float] = []
        posterior_errors: list[float] = []
        for time_index in range(len(trajectories[0])):
            batch = pad_current_frames(
                [trajectory[time_index] for trajectory in trajectories],
                self.raw_model.device,
            )
            measurements, mask, measurement_ids, frame_time, truth_states, truth_ids = (
                batch
            )
            output, state = self.raw_model.forward_step(
                measurements,
                mask,
                measurement_ids,
                frame_time,
                truth_ids,
                state,
            )
            if time_index < first_scored_frame:
                continue
            frame_loss = self.criterion(output, truth_states, truth_ids)
            if frame_loss.matched_existing:
                prior_errors.append(float(frame_loss.prior_position_error))
                posterior_errors.append(float(frame_loss.posterior_position_error))
            for row, targets in enumerate(truth_states):
                existing = output.existing_mask[row]
                true_birth = output.birth_mask[row] & (
                    output.birth_measurement_ids[row] >= 0
                )
                positions = (
                    torch.cat(
                        (
                            output.posterior_mean[row, existing, :2],
                            output.birth_positions[row, true_birth],
                        )
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                probabilities = np.ones(len(positions), dtype=np.float32)
                target_positions = targets[:, :2].detach().cpu().numpy()
                result = gospa(positions, target_positions)
                gospa_values.append(result.value)
                pro_values.append(pro_gospa(positions, probabilities, target_positions))
        return {
            "gospa": float(np.mean(gospa_values)),
            "pro_gospa": float(np.mean(pro_values)),
            "prior_position_error": float(np.mean(prior_errors))
            if prior_errors
            else 0.0,
            "posterior_position_error": float(np.mean(posterior_errors))
            if posterior_errors
            else 0.0,
        }


class V17BSequenceTrainer(V17ASequenceTrainer):
    """G2 trainer: direct association supervision plus scheduled state writes."""

    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        criterion: V17AOracleCriterion,
        association_criterion: V17BAssociationCriterion,
        *,
        burn_in_frames: int = 0,
        association_weight: float = 1.0,
        association_update_mode: str = "soft",
    ):
        super().__init__(model, optimizer, criterion, burn_in_frames=burn_in_frames)
        self.association_criterion = association_criterion
        self.association_weight = association_weight
        if association_update_mode not in {"soft", "hard", "moment"}:
            raise ValueError("invalid association_update_mode")
        self.association_update_mode = association_update_mode

    @torch.no_grad()
    def prepare_burn_in_state(
        self,
        trajectories: Sequence[Sequence[Frame]],
        association_oracle_probability: float,
    ):
        """Build one detached state that multiple scored branches can share."""
        if not trajectories or not trajectories[0]:
            raise ValueError("at least one non-empty trajectory is required")
        steps = len(trajectories[0])
        if any(len(trajectory) != steps for trajectory in trajectories):
            raise ValueError("all trajectories must have equal lengths")
        if self.burn_in_frames >= steps:
            raise ValueError("burn_in_frames must be shorter than the trajectory")
        state = None
        for time_index in range(self.burn_in_frames):
            batch = pad_current_frames(
                [trajectory[time_index] for trajectory in trajectories],
                self.raw_model.device,
            )
            measurements, mask, measurement_ids, frame_time, _, truth_ids = batch
            _, state = self.model(
                measurements,
                mask,
                measurement_ids,
                frame_time,
                truth_ids,
                state,
                association_oracle_probability,
                self.association_update_mode,
            )
        return None if state is None else state.detach()

    def _sequence_loss(
        self,
        trajectories: Sequence[Sequence[Frame]],
        association_oracle_probability: float,
        *,
        burn_in_association_oracle_probability: float | None = None,
        initial_state=None,
    ):
        if not trajectories or not trajectories[0]:
            raise ValueError("at least one non-empty trajectory is required")
        steps = len(trajectories[0])
        if any(len(trajectory) != steps for trajectory in trajectories):
            raise ValueError("all trajectories must have equal lengths")
        burn_in_probability = (
            association_oracle_probability
            if burn_in_association_oracle_probability is None
            else burn_in_association_oracle_probability
        )
        if not 0.0 <= burn_in_probability <= 1.0:
            raise ValueError("burn_in_association_oracle_probability must be in [0,1]")
        state = None if initial_state is None else initial_state.clone()
        frame_totals = []
        filter_losses = []
        association_losses = []
        for time_index in range(steps):
            batch = pad_current_frames(
                [trajectory[time_index] for trajectory in trajectories],
                self.raw_model.device,
            )
            measurements, mask, measurement_ids, frame_time, truth_states, truth_ids = (
                batch
            )
            if time_index < self.burn_in_frames:
                if initial_state is None:
                    with torch.no_grad():
                        _, state = self.model(
                            measurements,
                            mask,
                            measurement_ids,
                            frame_time,
                            truth_ids,
                            state,
                            burn_in_probability,
                            self.association_update_mode,
                        )
                    state = state.detach()
                continue
            output, state = self.model(
                measurements,
                mask,
                measurement_ids,
                frame_time,
                truth_ids,
                state,
                association_oracle_probability,
                self.association_update_mode,
            )
            filter_loss = self.criterion(output, truth_states, truth_ids)
            association_loss = self.association_criterion(output, truth_ids)
            frame_totals.append(
                filter_loss.total + self.association_weight * association_loss.total
            )
            filter_losses.append(filter_loss)
            association_losses.append(association_loss)
        if not frame_totals:
            raise ValueError("burn_in_frames must be shorter than the trajectory")
        total = torch.stack(frame_totals).mean()
        diagnostics = {
            name: float(
                torch.stack([getattr(loss, name) for loss in filter_losses]).mean()
            )
            for name in (
                "localization",
                "confidence",
                "prior_nll",
                "posterior_nll",
                "prior_position_error",
                "posterior_position_error",
            )
        }
        diagnostics.update(
            {
                "association_nll": float(
                    torch.stack(
                        [loss.total.detach() for loss in association_losses]
                    ).mean()
                ),
                **{
                    f"association_{name}": float(
                        torch.stack(
                            [getattr(loss, name) for loss in association_losses]
                        ).mean()
                    )
                    for name in (
                        "pair_nll",
                        "miss_nll",
                        "unclaimed_nll",
                        "top1_accuracy",
                        "miss_precision",
                        "miss_recall",
                        "unclaimed_accuracy",
                        "true_event_probability",
                        "marginal_error",
                        "capacity_violation",
                    )
                },
            }
        )
        diagnostics["matched_existing"] = sum(
            loss.matched_existing for loss in filter_losses
        ) / len(filter_losses)
        diagnostics["matched_births"] = sum(
            loss.matched_births for loss in filter_losses
        ) / len(filter_losses)
        diagnostics["association_supervised_events"] = sum(
            loss.supervised_events for loss in association_losses
        ) / len(association_losses)
        for event in ("pair", "miss", "unclaimed"):
            diagnostics[f"association_{event}_events"] = sum(
                getattr(loss, f"{event}_events") for loss in association_losses
            ) / len(association_losses)
        diagnostics["oracle_association_probability"] = float(
            association_oracle_probability
        )
        diagnostics["burn_in_oracle_association_probability"] = float(
            burn_in_probability
        )
        return total, diagnostics

    @torch.no_grad()
    def evaluate_association(
        self,
        trajectories: Sequence[Sequence[Frame]],
        *,
        association_oracle_probability: float,
        first_scored_frame: int = 0,
    ) -> dict[str, float]:
        self.raw_model.eval()
        state = None
        gospa_values: list[float] = []
        pro_values: list[float] = []
        prior_errors: list[float] = []
        posterior_errors: list[float] = []
        association_metrics: dict[str, list[float]] = {
            name: []
            for name in (
                "pair_nll",
                "miss_nll",
                "unclaimed_nll",
                "top1_accuracy",
                "miss_precision",
                "miss_recall",
                "unclaimed_accuracy",
                "true_event_probability",
                "marginal_error",
                "capacity_violation",
            )
        }
        for time_index in range(len(trajectories[0])):
            batch = pad_current_frames(
                [trajectory[time_index] for trajectory in trajectories],
                self.raw_model.device,
            )
            measurements, mask, measurement_ids, frame_time, truth_states, truth_ids = (
                batch
            )
            output, state = self.raw_model.forward_step(
                measurements,
                mask,
                measurement_ids,
                frame_time,
                truth_ids,
                state,
                association_oracle_probability,
                self.association_update_mode,
            )
            if time_index < first_scored_frame:
                continue
            frame_loss = self.criterion(output, truth_states, truth_ids)
            association_loss = self.association_criterion(output, truth_ids)
            for name in association_metrics:
                association_metrics[name].append(float(getattr(association_loss, name)))
            if frame_loss.matched_existing:
                prior_errors.append(float(frame_loss.prior_position_error))
                posterior_errors.append(float(frame_loss.posterior_position_error))
            for row, targets in enumerate(truth_states):
                existing = output.existing_mask[row]
                true_birth = output.birth_mask[row] & (
                    output.birth_measurement_ids[row] >= 0
                )
                positions = (
                    torch.cat(
                        (
                            output.posterior_mean[row, existing, :2],
                            output.birth_positions[row, true_birth],
                        )
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                probabilities = np.ones(len(positions), dtype=np.float32)
                target_positions = targets[:, :2].detach().cpu().numpy()
                gospa_values.append(gospa(positions, target_positions).value)
                pro_values.append(pro_gospa(positions, probabilities, target_positions))
        return {
            "gospa": float(np.mean(gospa_values)),
            "pro_gospa": float(np.mean(pro_values)),
            "prior_position_error": float(np.mean(prior_errors))
            if prior_errors
            else 0.0,
            "posterior_position_error": float(np.mean(posterior_errors))
            if posterior_errors
            else 0.0,
            **{
                f"association_{name}": float(np.mean(values)) if values else 0.0
                for name, values in association_metrics.items()
            },
        }


def save_v17a_checkpoint(
    path: str | Path,
    model,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: dict[str, float],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "metrics": metrics,
            "v17a_config": asdict(model.config),
        },
        path,
    )
    return path


def save_v17b_checkpoint(
    path: str | Path,
    model,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: dict[str, float],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "metrics": metrics,
            "v17b_config": asdict(model.config),
        },
        path,
    )
    return path


def append_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def archive_metrics_after_resume(path: str | Path, checkpoint_step: int) -> Path | None:
    """Remove abandoned post-checkpoint records while preserving an audit copy."""

    path = Path(path)
    if not path.is_file():
        return None
    kept: list[str] = []
    discarded: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        destination = (
            discarded
            if int(record.get("step", checkpoint_step)) > checkpoint_step
            else kept
        )
        destination.append(line)
    if not discarded:
        return None
    archive = path.with_name(
        f"{path.stem}.discarded_after_step_{checkpoint_step:06d}{path.suffix}"
    )
    if archive.exists():
        raise FileExistsError(f"refusing to overwrite metrics archive: {archive}")
    archive.write_text("\n".join(discarded) + "\n", encoding="utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    temporary.replace(path)
    return archive
