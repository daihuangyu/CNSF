#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import yaml

import _bootstrap  # noqa: F401

from track_mt3.config import SimulationConfig
from track_mt3.data.simulator import MultiTargetSimulator
from track_mt3.models_v17 import EndToEndRecursiveTracker, V17BConfig
from track_mt3.models_v17.association_criterion import V17BAssociationCriterion
from track_mt3.models_v17.batching import pad_current_frames
from track_mt3.models_v17.criterion import V17AOracleCriterion
from track_mt3.models_v17.death_criterion import (
    V18CBernoulliCriterion,
    V17DeathCriterion,
    negative_exposed_death_mean,
)
from track_mt3.models_v17.lifecycle_criterion import V17BirthCriterion
from track_mt3.training.curriculum_v17 import (
    v17_curriculum_at_step,
    v17_rollout_horizon_at_step,
    v17_trajectory_curriculum_at_step,
)
from track_mt3.training.trainer_v17 import (
    append_jsonl,
    archive_metrics_after_resume,
    average_gradients,
)


RUN_KIND = "cnsf_end_to_end_random_init"


def _config_digest(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _reduce_metrics(
    metrics: dict[str, float], device: torch.device, world_size: int
) -> dict[str, float]:
    names = sorted(metrics)
    values = torch.tensor([metrics[name] for name in names], device=device)
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= world_size
    return {name: float(value) for name, value in zip(names, values.cpu())}


def _save_checkpoint(
    path: Path,
    model: EndToEndRecursiveTracker,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    config_digest: str,
    run_id: str,
    effective_updates: int,
    metrics: dict[str, float],
    run_kind: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "run_kind": run_kind,
            "architecture": type(model).__name__,
            "initialization": "random",
            "config_digest": config_digest,
            "run_id": run_id,
            "effective_updates": effective_updates,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "metrics": metrics,
        },
        path,
    )


def _learning_rate_multiplier(
    step: int, *, warmup_steps: int, total_steps: int, minimum_ratio: float
) -> float:
    if step <= warmup_steps:
        return max(step, 1) / max(warmup_steps, 1)
    progress = min(
        max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0),
        1.0,
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CNSF end to end from a fresh initialization"
    )
    parser.add_argument(
        "--config", default="configs/training/cnsf_exact12k.yaml"
    )
    parser.add_argument("--updates", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument(
        "--batch-size",
        type=int,
        help="memory-recovery global batch override; valid only with --resume",
    )
    arguments = parser.parse_args()

    with Path(arguments.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    training = config["training"]
    run_kind = str(training.get("run_kind", RUN_KIND))
    if "initial_checkpoint" in training:
        raise ValueError(
            "formal end-to-end training forbids initial_checkpoint; use --resume "
            "only for a checkpoint created by this same run"
        )

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = int(training["seed"])
    # Every rank must start from identical parameters because gradients are
    # manually averaged rather than synchronized by a DDP constructor.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = EndToEndRecursiveTracker(V17BConfig(**config["model"])).to(device)
    trainable_parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    updates = arguments.updates or int(training["updates"])
    warmup_steps = int(training.get("learning_rate_warmup_steps", 0))
    minimum_ratio = float(training.get("minimum_learning_rate_ratio", 0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda current_step: _learning_rate_multiplier(
            current_step + 1,
            warmup_steps=warmup_steps,
            total_steps=updates,
            minimum_ratio=minimum_ratio,
        ),
    )
    digest = _config_digest(config)
    start_step = 0
    if arguments.resume:
        checkpoint = torch.load(
            arguments.resume, map_location=device, weights_only=False
        )
        if checkpoint.get("run_kind") != run_kind:
            raise ValueError("resume checkpoint was not created by this formal run")
        if checkpoint.get("architecture") != type(model).__name__:
            raise ValueError("resume checkpoint architecture does not match")
        if checkpoint.get("config_digest") != digest:
            raise ValueError("resume checkpoint config does not match")
        if checkpoint.get("initialization") != "random":
            raise ValueError("resume checkpoint did not originate from random init")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["step"])

    set_criterion = V17AOracleCriterion(**config["set_loss"])
    association_criterion = V17BAssociationCriterion(
        **config.get("association_loss", {})
    )
    birth_criterion = V17BirthCriterion(**config["birth_loss"])
    bernoulli_loss_config = config.get("bernoulli_loss")
    bernoulli_criterion = (
        V18CBernoulliCriterion(**bernoulli_loss_config)
        if bernoulli_loss_config is not None
        else None
    )
    death_criterion = (
        None
        if bernoulli_criterion is not None
        else V17DeathCriterion(**config["death_loss"])
    )
    natural_frequency_death = bool(
        config.get("death_loss", {}).get("natural_frequency", False)
    )
    weights = config["loss_weights"]
    set_risk_weight = float(weights.get("set_risk", 0.0))
    set_risk_curriculum = config.get("set_risk_curriculum", {})
    set_risk_start_step = int(set_risk_curriculum.get("start_step", 0))
    set_risk_full_step = int(
        set_risk_curriculum.get("full_step", set_risk_start_step)
    )
    if set_risk_weight < 0.0:
        raise ValueError("set_risk loss weight must be non-negative")
    if set_risk_start_step < 0 or set_risk_full_step < set_risk_start_step:
        raise ValueError("invalid set_risk_curriculum interval")
    oracle_shadow_weight = float(
        config.get("stability", {}).get("oracle_shadow_set_weight", 0.0)
    )
    if oracle_shadow_weight < 0.0:
        raise ValueError("oracle_shadow_set_weight must be non-negative")
    if oracle_shadow_weight and float(config["model"].get("dropout", 0.0)):
        raise ValueError(
            "dual-rollout encoder sharing requires deterministic dropout=0"
        )

    configured_global_batch = int(training["batch_size"])
    if arguments.batch_size is not None and not arguments.resume:
        raise ValueError("batch-size override is allowed only for same-run recovery")
    global_batch = (
        int(arguments.batch_size)
        if arguments.batch_size is not None
        else configured_global_batch
    )
    if global_batch < 1:
        raise ValueError("batch_size must be positive")
    if global_batch % world_size:
        raise ValueError("batch_size must be divisible by world size")
    local_batch = global_batch // world_size
    burn_in_frames = int(training["burn_in_frames"])
    supervised_frames = int(training["supervised_frames"])
    trajectory_steps = burn_in_frames + supervised_frames
    if burn_in_frames < 0:
        raise ValueError("burn_in_frames must be non-negative")
    if supervised_frames < 1:
        raise ValueError("supervised_frames must be positive")
    rollout_horizon_config = config.get("rollout_horizon")
    terminal_simulations = [
        SimulationConfig(**values) for values in config["simulations"]
    ]
    if not terminal_simulations:
        raise ValueError("at least one simulation configuration is required")
    trajectory_curriculum_config = config.get("trajectory_curriculum")
    curriculum_config = config["curriculum"]
    output_dir = Path(arguments.output_dir or training["output_dir"])
    run_id = hashlib.sha256(
        f"{digest}:{output_dir.resolve()}:{updates}".encode("utf-8")
    ).hexdigest()
    if arguments.resume:
        if checkpoint.get("run_id") != run_id:
            raise ValueError("resume checkpoint belongs to a different output run")
        if checkpoint.get("effective_updates") != updates:
            raise ValueError(
                "resume checkpoint used a different effective update count"
            )
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        discarded_metrics = (
            archive_metrics_after_resume(output_dir / "metrics.jsonl", start_step)
            if arguments.resume
            else None
        )
        (output_dir / "config.json").write_text(
            json.dumps(
                {
                    **config,
                    "distributed": {"world_size": world_size},
                    "runtime": {
                        "configured_global_batch": configured_global_batch,
                        "effective_global_batch": global_batch,
                        "memory_recovery_override": arguments.batch_size is not None,
                    },
                    "config_digest": digest,
                    "run_kind": run_kind,
                    "run_id": run_id,
                    "initialization": "random",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "event": "initialized",
                    "initialization": "random"
                    if start_step == 0
                    else "same_run_resume",
                    "resume": arguments.resume,
                    "start_step": start_step,
                    "architecture": type(model).__name__,
                    "parameters": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "configured_global_batch": configured_global_batch,
                    "effective_global_batch": global_batch,
                    "config_digest": digest,
                    "run_id": run_id,
                    "discarded_metrics_archive": (
                        str(discarded_metrics) if discarded_metrics else None
                    ),
                }
            ),
            flush=True,
        )
    if world_size > 1:
        dist.barrier()

    latest_metrics: dict[str, float] = {}
    for step in range(start_step + 1, updates + 1):
        if rollout_horizon_config is None:
            rollout_horizon_progress = 1.0
            current_trajectory_steps = trajectory_steps
            maximum_trajectory_steps = trajectory_steps
            current_burn_in_frames = burn_in_frames
            current_supervised_frames = supervised_frames
        else:
            rollout_horizon = v17_rollout_horizon_at_step(
                step, **rollout_horizon_config
            )
            rollout_horizon_progress = rollout_horizon.progress
            current_trajectory_steps = rollout_horizon.trajectory_steps
            maximum_trajectory_steps = rollout_horizon.maximum_trajectory_steps
            current_burn_in_frames = rollout_horizon.burn_in_frames
            current_supervised_frames = rollout_horizon.supervised_frames
        curriculum = v17_curriculum_at_step(step, **curriculum_config)
        if step < set_risk_start_step:
            set_risk_progress = 0.0
        elif set_risk_full_step == set_risk_start_step:
            set_risk_progress = 1.0
        else:
            set_risk_progress = min(
                1.0,
                (step - set_risk_start_step)
                / (set_risk_full_step - set_risk_start_step),
            )
        effective_set_risk_weight = set_risk_weight * set_risk_progress
        if trajectory_curriculum_config is None:
            simulations = terminal_simulations
            trajectory_curriculum_metrics: dict[str, float] = {}
        else:
            trajectory_curriculum = v17_trajectory_curriculum_at_step(
                terminal_simulations,
                step,
                **trajectory_curriculum_config,
            )
            simulations = trajectory_curriculum.simulations
            trajectory_curriculum_metrics = trajectory_curriculum.metrics
        birth_draw = np.random.default_rng(
            np.random.SeedSequence([seed, step, 9173])
        ).random()
        birth_state_mode = (
            "predicted"
            if birth_draw < curriculum.predicted_birth_probability
            else "oracle"
        )
        trajectories = [
            MultiTargetSimulator(
                simulations[
                    (step * global_batch + rank * local_batch + index)
                    % len(simulations)
                ],
                np.random.SeedSequence([seed, rank, step, index]),
            ).simulate(current_trajectory_steps)
            for index in range(local_batch)
        ]

        model.train()
        optimizer.zero_grad(set_to_none=True)
        state = None
        shadow_state = None
        set_losses = []
        shadow_set_losses = []
        association_losses = []
        birth_losses = []
        death_losses = []
        bernoulli_losses = []
        undetected_means = []
        secondary_hypothesis_weights = []
        previous_truth_ids = None
        started = time.perf_counter()
        for time_index in range(current_trajectory_steps):
            batch = pad_current_frames(
                [trajectory[time_index] for trajectory in trajectories], device
            )
            measurements, mask, measurement_ids, frame_time, truth_states, truth_ids = (
                batch
            )
            forward_arguments = dict(
                association_oracle_probability=curriculum.association_oracle_probability,
                association_update_mode="moment",
                birth_state_mode=birth_state_mode,
                birth_candidate_threshold=float(training["birth_candidate_threshold"]),
                confirmation_hits=int(training["confirmation_hits"]),
                death_state_mode=curriculum.death_state_mode,
                retention_threshold=float(training["retention_threshold"]),
                oracle_prune_delay=int(training["oracle_prune_delay"]),
                survival_warmup_frames=int(training["survival_warmup_frames"]),
                confirmed_association_bias=0.0,
            )
            if time_index < current_burn_in_frames:
                with torch.no_grad():
                    shared_embeddings = (
                        model.encode_current_frame(measurements, mask)
                        if oracle_shadow_weight
                        else None
                    )
                    _, state = model(
                        measurements,
                        mask,
                        measurement_ids,
                        frame_time,
                        truth_ids,
                        state,
                        **forward_arguments,
                        precomputed_measurement_embeddings=shared_embeddings,
                    )
                    if oracle_shadow_weight:
                        _, shadow_state = model(
                            measurements,
                            mask,
                            measurement_ids,
                            frame_time,
                            truth_ids,
                            shadow_state,
                            association_oracle_probability=1.0,
                            association_update_mode="moment",
                            birth_state_mode="oracle",
                            birth_candidate_threshold=float(
                                training["birth_candidate_threshold"]
                            ),
                            confirmation_hits=int(training["confirmation_hits"]),
                            death_state_mode="oracle_pre",
                            retention_threshold=float(training["retention_threshold"]),
                            oracle_prune_delay=int(training["oracle_prune_delay"]),
                            survival_warmup_frames=int(
                                training["survival_warmup_frames"]
                            ),
                            confirmed_association_bias=0.0,
                            precomputed_measurement_embeddings=shared_embeddings,
                        )
                state = state.detach()
                if shadow_state is not None:
                    shadow_state = shadow_state.detach()
                previous_truth_ids = truth_ids
                continue
            shared_embeddings = (
                model.encode_current_frame(measurements, mask)
                if oracle_shadow_weight
                else None
            )
            output, state = model(
                measurements,
                mask,
                measurement_ids,
                frame_time,
                truth_ids,
                state,
                **forward_arguments,
                precomputed_measurement_embeddings=shared_embeddings,
            )
            set_losses.append(set_criterion(output, truth_states, truth_ids))
            association_losses.append(association_criterion(output, truth_ids))
            birth_losses.append(birth_criterion(output))
            if output.predicted_undetected_mean is not None:
                undetected_means.append(output.predicted_undetected_mean.mean())
            if output.association_hypothesis_weights is not None:
                secondary_hypothesis_weights.append(
                    output.association_hypothesis_weights[:, 1].mean()
                )
            if bernoulli_criterion is None:
                assert death_criterion is not None
                death_losses.append(death_criterion(output, truth_ids))
            else:
                bernoulli_losses.append(
                    bernoulli_criterion(output, truth_ids, previous_truth_ids)
                )
            if oracle_shadow_weight:
                shadow_output, shadow_state = model(
                    measurements,
                    mask,
                    measurement_ids,
                    frame_time,
                    truth_ids,
                    shadow_state,
                    association_oracle_probability=1.0,
                    association_update_mode="moment",
                    birth_state_mode="oracle",
                    birth_candidate_threshold=float(
                        training["birth_candidate_threshold"]
                    ),
                    confirmation_hits=int(training["confirmation_hits"]),
                    death_state_mode="oracle_pre",
                    retention_threshold=float(training["retention_threshold"]),
                    oracle_prune_delay=int(training["oracle_prune_delay"]),
                    survival_warmup_frames=int(training["survival_warmup_frames"]),
                    confirmed_association_bias=0.0,
                    precomputed_measurement_embeddings=shared_embeddings,
                )
                shadow_set_losses.append(
                    set_criterion(shadow_output, truth_states, truth_ids)
                )
            previous_truth_ids = truth_ids

        set_total = torch.stack([loss.total for loss in set_losses]).mean()
        set_risk_total = torch.stack(
            [loss.set_risk for loss in set_losses]
        ).mean()
        association_total = torch.stack(
            [loss.total for loss in association_losses]
        ).mean()
        birth_total = torch.stack([loss.total for loss in birth_losses]).mean()
        if bernoulli_criterion is None:
            if natural_frequency_death:
                scored_death_losses = [
                    loss.total
                    for loss in death_losses
                    if loss.alive_tracks + loss.dead_tracks > 0
                ]
                death_total = (
                    torch.stack(scored_death_losses).mean()
                    if scored_death_losses
                    else set_total * 0.0
                )
                death_supervised_frames = len(scored_death_losses)
            else:
                death_total, death_supervised_frames = negative_exposed_death_mean(
                    death_losses, set_total
                )
            survival_transition_total = set_total.detach() * 0.0
            cardinality_total = torch.stack(
                [loss.cardinality for loss in death_losses]
            ).mean()
            predicted_cardinality = torch.stack(
                [loss.predicted_cardinality for loss in death_losses]
            ).mean()
            target_cardinality = torch.stack(
                [loss.target_cardinality for loss in death_losses]
            ).mean()
            alive_probability = torch.stack(
                [loss.alive_probability for loss in death_losses]
            ).mean()
            exposed_death_losses = [
                loss for loss in death_losses if loss.dead_tracks > 0
            ]
            dead_probability = (
                torch.stack(
                    [loss.dead_probability for loss in exposed_death_losses]
                ).mean()
                if exposed_death_losses
                else set_total.detach() * 0.0
            )
            missed_alive_probability = set_total.detach() * 0.0
            survival_alive_probability = set_total.detach() * 0.0
            survival_dead_probability = set_total.detach() * 0.0
            survival_transition_frames = 0
        else:
            posterior_losses = [
                loss.posterior for loss in bernoulli_losses if loss.posterior_tracks > 0
            ]
            transition_losses = [
                loss.survival_transition
                for loss in bernoulli_losses
                if loss.transition_tracks > 0
            ]
            death_total = (
                torch.stack(posterior_losses).mean()
                if posterior_losses
                else set_total * 0.0
            )
            survival_transition_total = (
                torch.stack(transition_losses).mean()
                if transition_losses
                else set_total * 0.0
            )
            cardinality_total = torch.stack(
                [loss.cardinality for loss in bernoulli_losses]
            ).mean()
            predicted_cardinality = torch.stack(
                [loss.predicted_cardinality for loss in bernoulli_losses]
            ).mean()
            target_cardinality = torch.stack(
                [loss.target_cardinality for loss in bernoulli_losses]
            ).mean()
            death_supervised_frames = len(posterior_losses)
            survival_transition_frames = len(transition_losses)
            alive_probability = torch.stack(
                [loss.alive_probability for loss in bernoulli_losses]
            ).mean()
            dead_probability = torch.stack(
                [loss.dead_probability for loss in bernoulli_losses]
            ).mean()
            missed_alive_probability = torch.stack(
                [loss.missed_alive_probability for loss in bernoulli_losses]
            ).mean()
            survival_alive_probability = torch.stack(
                [loss.survival_alive_probability for loss in bernoulli_losses]
            ).mean()
            survival_dead_probability = torch.stack(
                [loss.survival_dead_probability for loss in bernoulli_losses]
            ).mean()
        shadow_set_total = (
            torch.stack([loss.total for loss in shadow_set_losses]).mean()
            if shadow_set_losses
            else set_total.detach() * 0.0
        )
        total = (
            float(weights["set"]) * set_total
            + effective_set_risk_weight * set_risk_total
            + float(weights["association"]) * association_total
            + float(weights["birth"]) * birth_total
            + float(weights["survival"]) * death_total
            + float(weights.get("survival_transition", 0.0))
            * survival_transition_total
            + float(weights.get("cardinality", 0.0)) * cardinality_total
            + oracle_shadow_weight * shadow_set_total
        )
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite end-to-end loss at step {step}")
        total.backward()
        average_gradients(model, world_size)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters, float(training["gradient_clip_norm"])
        )
        optimizer.step()
        scheduler.step()

        metrics = _reduce_metrics(
            {
                "loss": float(total.detach()),
                "set_loss": float(set_total.detach()),
                "set_risk": float(set_risk_total.detach()),
                "set_risk_weight": effective_set_risk_weight,
                "prior_position_anchor": float(
                    torch.stack(
                        [loss.prior_position_anchor for loss in set_losses]
                    ).mean()
                ),
                "prior_velocity_anchor": float(
                    torch.stack(
                        [loss.prior_velocity_anchor for loss in set_losses]
                    ).mean()
                ),
                "posterior_velocity_anchor": float(
                    torch.stack(
                        [loss.posterior_velocity_anchor for loss in set_losses]
                    ).mean()
                ),
                "oracle_shadow_set_loss": float(shadow_set_total.detach()),
                "oracle_shadow_set_weight": oracle_shadow_weight,
                "association_nll": float(association_total.detach()),
                "association_top1": float(
                    torch.stack(
                        [loss.top1_accuracy for loss in association_losses]
                    ).mean()
                ),
                "association_true_probability": float(
                    torch.stack(
                        [loss.true_event_probability for loss in association_losses]
                    ).mean()
                ),
                "association_pair_nll": float(
                    torch.stack([loss.pair_nll for loss in association_losses]).mean()
                ),
                "association_miss_nll": float(
                    torch.stack([loss.miss_nll for loss in association_losses]).mean()
                ),
                "association_unclaimed_nll": float(
                    torch.stack(
                        [loss.unclaimed_nll for loss in association_losses]
                    ).mean()
                ),
                "association_death_nll": float(
                    torch.stack([loss.death_nll for loss in association_losses]).mean()
                ),
                "association_pair_events": float(
                    sum(loss.pair_events for loss in association_losses)
                ),
                "association_miss_events": float(
                    sum(loss.miss_events for loss in association_losses)
                ),
                "association_unclaimed_events": float(
                    sum(loss.unclaimed_events for loss in association_losses)
                ),
                "association_death_events": float(
                    sum(loss.death_events for loss in association_losses)
                ),
                "birth_loss": float(birth_total.detach()),
                "birth_positive_probability": float(
                    torch.stack(
                        [loss.newborn_positive_mean for loss in birth_losses]
                    ).mean()
                ),
                "birth_negative_probability": float(
                    torch.stack(
                        [loss.newborn_negative_mean for loss in birth_losses]
                    ).mean()
                ),
                "birth_poisson_count_loss": float(
                    torch.stack(
                        [loss.poisson_count for loss in birth_losses]
                    ).mean()
                ),
                "predicted_undetected_mean": float(
                    torch.stack(undetected_means).mean().detach()
                    if undetected_means
                    else 0.0
                ),
                "secondary_hypothesis_weight": float(
                    torch.stack(secondary_hypothesis_weights).mean().detach()
                    if secondary_hypothesis_weights
                    else 0.0
                ),
                "survival_loss": float(death_total.detach()),
                "survival_transition_loss": float(
                    survival_transition_total.detach()
                ),
                "cardinality_loss": float(cardinality_total.detach()),
                "predicted_cardinality": float(predicted_cardinality),
                "target_cardinality": float(target_cardinality),
                "alive_probability": float(alive_probability),
                "dead_probability": float(dead_probability),
                "missed_alive_probability": float(missed_alive_probability),
                "survival_alive_probability": float(survival_alive_probability),
                "survival_dead_probability": float(survival_dead_probability),
                "death_supervised_frames": float(death_supervised_frames),
                "survival_transition_frames": float(survival_transition_frames),
                "burn_in_frames": float(current_burn_in_frames),
                "supervised_frames": float(current_supervised_frames),
                "trajectory_steps": float(current_trajectory_steps),
                "maximum_trajectory_steps": float(maximum_trajectory_steps),
                "rollout_horizon_progress": float(rollout_horizon_progress),
                "association_oracle_probability": curriculum.association_oracle_probability,
                "predicted_birth_probability": curriculum.predicted_birth_probability,
                "predicted_birth_used": float(birth_state_mode == "predicted"),
                "predicted_death_used": float(
                    curriculum.death_state_mode == "predicted"
                ),
                "gradient_norm": float(gradient_norm),
                "learning_rate": scheduler.get_last_lr()[0],
                "step_seconds": time.perf_counter() - started,
                **trajectory_curriculum_metrics,
            },
            device,
            world_size,
        )
        latest_metrics = metrics
        if rank == 0:
            record = {"step": step, "curriculum": curriculum.name, **metrics}
            append_jsonl(output_dir / "metrics.jsonl", record)
            if step == 1 or step % int(training["log_interval"]) == 0:
                print(json.dumps(record), flush=True)
            if step % int(training["checkpoint_interval"]) == 0:
                _save_checkpoint(
                    output_dir / "checkpoints" / f"step_{step:06d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    step,
                    digest,
                    run_id,
                    updates,
                    latest_metrics,
                    run_kind,
                )

    if rank == 0:
        _save_checkpoint(
            output_dir / "checkpoints" / "final.pt",
            model,
            optimizer,
            scheduler,
            updates,
            digest,
            run_id,
            updates,
            latest_metrics,
            run_kind,
        )
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
