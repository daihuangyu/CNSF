from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from track_mt3.config import SimulationConfig


@dataclass(frozen=True)
class V17CurriculumState:
    name: str
    association_oracle_probability: float
    predicted_birth_probability: float
    death_state_mode: str


@dataclass(frozen=True)
class V17TrajectoryCurriculumState:
    """The physical training distribution at one optimizer step."""

    progress: float
    simulations: tuple[SimulationConfig, ...]
    metrics: dict[str, float]


@dataclass(frozen=True)
class V17RolloutHorizonState:
    """Recurrent training span and its supervised suffix at one update."""

    progress: float
    trajectory_steps: int
    burn_in_frames: int
    supervised_frames: int
    maximum_trajectory_steps: int


def v17_rollout_horizon_at_step(
    step: int,
    *,
    hold_steps: int,
    full_step: int,
    start_trajectory_steps: int,
    end_trajectory_steps: int,
    start_burn_in_frames: int,
    end_burn_in_frames: int,
    sampling_mode: str = "fixed",
    minimum_trajectory_steps: int | None = None,
    sampling_seed: int = 0,
    tail_probability: float = 0.25,
    short_maximum_trajectory_steps: int = 41,
) -> V17RolloutHorizonState:
    """Schedule a fixed or reproducibly sampled recurrent training span."""

    if step < 1:
        raise ValueError("step must be positive")
    if hold_steps < 0 or full_step < hold_steps:
        raise ValueError("invalid rollout-horizon curriculum boundaries")
    if start_trajectory_steps < 1 or end_trajectory_steps < start_trajectory_steps:
        raise ValueError("invalid trajectory-step range")
    burn_in_values = (start_burn_in_frames, end_burn_in_frames)
    if any(value < 0 for value in burn_in_values):
        raise ValueError("burn-in frames must be non-negative")
    if start_burn_in_frames >= start_trajectory_steps:
        raise ValueError("start burn-in must leave a supervised frame")
    if end_burn_in_frames >= end_trajectory_steps:
        raise ValueError("end burn-in must leave a supervised frame")
    if sampling_mode not in {"fixed", "uniform", "tail_balanced"}:
        raise ValueError(
            "sampling_mode must be 'fixed', 'uniform' or 'tail_balanced'"
        )
    minimum_steps = (
        start_trajectory_steps
        if minimum_trajectory_steps is None
        else minimum_trajectory_steps
    )
    if minimum_steps < 1 or minimum_steps > start_trajectory_steps:
        raise ValueError(
            "minimum_trajectory_steps must be between 1 and "
            "start_trajectory_steps"
        )
    if not 0.0 <= tail_probability <= 1.0:
        raise ValueError("tail_probability must lie in [0,1]")
    if short_maximum_trajectory_steps < minimum_steps:
        raise ValueError(
            "short_maximum_trajectory_steps must be at least the minimum"
        )

    progress = _linear_progress(step, hold_steps, full_step)
    maximum_trajectory_steps = int(
        round(
            start_trajectory_steps
            + progress * (end_trajectory_steps - start_trajectory_steps)
        )
    )
    maximum_burn_in_frames = int(
        round(
            start_burn_in_frames
            + progress * (end_burn_in_frames - start_burn_in_frames)
        )
    )
    if sampling_mode in {"uniform", "tail_balanced"}:
        # Depends only on seed and optimizer step: identical across DDP ranks
        # and exactly reproducible after a checkpoint resume.
        generator = random.Random(sampling_seed * 1_000_003 + step)
        if sampling_mode == "tail_balanced":
            # Ramp the long-tail probability with the maximum horizon.  At the
            # terminal curriculum a fixed fraction of updates reaches frame
            # 99, while the remaining updates stay short enough to preserve
            # training throughput.
            effective_tail_probability = progress * tail_probability
            if generator.random() < effective_tail_probability:
                trajectory_steps = maximum_trajectory_steps
            else:
                short_ceiling = min(
                    short_maximum_trajectory_steps,
                    maximum_trajectory_steps,
                )
                trajectory_steps = generator.randint(minimum_steps, short_ceiling)
        else:
            trajectory_steps = generator.randint(
                minimum_steps, maximum_trajectory_steps
            )
        burn_in_frames = int(
            round(
                trajectory_steps
                * maximum_burn_in_frames
                / maximum_trajectory_steps
            )
        )
    else:
        trajectory_steps = maximum_trajectory_steps
        burn_in_frames = maximum_burn_in_frames
    return V17RolloutHorizonState(
        progress=progress,
        trajectory_steps=trajectory_steps,
        burn_in_frames=burn_in_frames,
        supervised_frames=trajectory_steps - burn_in_frames,
        maximum_trajectory_steps=maximum_trajectory_steps,
    )


def v17_trajectory_curriculum_at_step(
    terminals: list[SimulationConfig] | tuple[SimulationConfig, ...],
    step: int,
    *,
    warmup_steps: int,
    start_initial_targets: int = 1,
    start_max_targets: int = 4,
    start_clutter_rate: float = 0.0,
    start_detection_probability: float = 1.0,
    start_birth_rate: float = 0.0,
    start_survival_probability: float = 1.0,
    process_noise_start_fraction: float = 0.2,
    measurement_noise_start_fraction: float = 0.2,
) -> V17TrajectoryCurriculumState:
    """Interpolate every terminal scenario from a shared easy regime.

    This is independent of :func:`v17_curriculum_at_step`: the latter controls
    oracle exposure, while this function controls only the simulated physical
    trajectory distribution.  Noise starts as a fraction of each terminal
    value rather than at an absolute floor, so a genuinely noiseless terminal
    scenario never becomes harder during an alleged easy-to-hard curriculum.
    """

    if step < 0:
        raise ValueError("step must be non-negative")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if not terminals:
        raise ValueError("at least one terminal simulation is required")
    if start_initial_targets < 0:
        raise ValueError("start_initial_targets must be non-negative")
    if start_max_targets < max(start_initial_targets, 1):
        raise ValueError("start_max_targets must cover the initial targets")
    probabilities = (
        start_detection_probability,
        start_survival_probability,
    )
    if any(not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("start probabilities must be in [0,1]")
    if start_clutter_rate < 0.0 or start_birth_rate < 0.0:
        raise ValueError("start clutter and birth rates must be non-negative")
    fractions = (process_noise_start_fraction, measurement_noise_start_fraction)
    if any(not 0.0 <= value <= 1.0 for value in fractions):
        raise ValueError("noise start fractions must be in [0,1]")

    progress = min(step / max(warmup_steps, 1), 1.0) if warmup_steps else 1.0
    simulations = []
    for terminal in terminals:
        if terminal.max_targets < terminal.initial_targets:
            raise ValueError("terminal max_targets must cover initial_targets")
        values = asdict(terminal)
        starts = {
            "initial_targets": float(start_initial_targets),
            "max_targets": float(start_max_targets),
            "clutter_rate": start_clutter_rate,
            "detection_probability": start_detection_probability,
            "birth_rate": start_birth_rate,
            "survival_probability": start_survival_probability,
            "process_noise": process_noise_start_fraction
            * terminal.process_noise,
            "measurement_noise": measurement_noise_start_fraction
            * terminal.measurement_noise,
        }
        for name, start in starts.items():
            end = float(getattr(terminal, name))
            value = start + progress * (end - start)
            values[name] = (
                int(round(value))
                if name in {"initial_targets", "max_targets"}
                else value
            )
        simulations.append(SimulationConfig(**values))

    metric_names = (
        "initial_targets",
        "max_targets",
        "clutter_rate",
        "detection_probability",
        "birth_rate",
        "survival_probability",
        "process_noise",
        "measurement_noise",
    )
    metrics = {"trajectory_curriculum_progress": progress}
    for name in metric_names:
        metrics[f"sim_{name}"] = sum(
            float(getattr(simulation, name)) for simulation in simulations
        ) / len(simulations)
    return V17TrajectoryCurriculumState(
        progress=progress,
        simulations=tuple(simulations),
        metrics=metrics,
    )


def _linear_progress(step: int, start: int, end: int) -> float:
    if end <= start:
        return float(step >= end)
    return min(max((step - start) / (end - start), 0.0), 1.0)


def v17_curriculum_at_step(
    step: int,
    *,
    association_oracle_hold_steps: int,
    association_oracle_end_step: int,
    predicted_birth_start_step: int,
    predicted_birth_full_step: int,
    predicted_death_start_step: int,
) -> V17CurriculumState:
    """Describe teacher forcing for one fixed-architecture training run."""

    if step < 1:
        raise ValueError("step must be positive")
    milestones = (
        association_oracle_hold_steps,
        association_oracle_end_step,
        predicted_birth_start_step,
        predicted_birth_full_step,
        predicted_death_start_step,
    )
    if any(value < 0 for value in milestones):
        raise ValueError("curriculum milestones must be non-negative")
    if association_oracle_end_step < association_oracle_hold_steps:
        raise ValueError("association oracle end must follow its hold")
    if predicted_birth_full_step < predicted_birth_start_step:
        raise ValueError("predicted birth full step must follow its start")
    if predicted_death_start_step < predicted_birth_full_step:
        raise ValueError("predicted death must start after predicted birth is full")

    association_oracle_probability = 1.0 - _linear_progress(
        step,
        association_oracle_hold_steps,
        association_oracle_end_step,
    )
    predicted_birth_probability = _linear_progress(
        step,
        predicted_birth_start_step,
        predicted_birth_full_step,
    )
    if step < predicted_birth_start_step:
        name = "oracle_bootstrap"
        death_state_mode = "oracle_pre"
    elif step < predicted_birth_full_step:
        name = "birth_transition"
        death_state_mode = "oracle_delayed"
    elif step < predicted_death_start_step:
        name = "lifecycle_exposure"
        death_state_mode = "oracle_delayed"
    else:
        name = "free_rollout"
        death_state_mode = "predicted"
    return V17CurriculumState(
        name=name,
        association_oracle_probability=association_oracle_probability,
        predicted_birth_probability=predicted_birth_probability,
        death_state_mode=death_state_mode,
    )
