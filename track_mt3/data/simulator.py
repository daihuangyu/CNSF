from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray

from track_mt3.config import SimulationConfig

from .motion import measurement_covariance, process_covariance, transition_matrix


@dataclass(frozen=True)
class TargetState:
    target_id: int
    state: NDArray[np.float64]


@dataclass(frozen=True)
class Frame:
    index: int
    time: float
    states: NDArray[np.float32]
    target_ids: NDArray[np.int64]
    target_ages: NDArray[np.int64]
    measurements: NDArray[np.float32]
    measurement_ids: NDArray[np.int64]


class MultiTargetSimulator:
    """Online simulator for equations (1)--(10) of the Track-MT3 paper."""

    def __init__(self, config: SimulationConfig, seed: int | np.random.SeedSequence = 0):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self._f = transition_matrix(config.dt)
        self._q = process_covariance(config.dt, config.process_noise, intensity_is_std=config.process_noise_is_std)
        self._r = measurement_covariance(config.measurement_noise, intensity_is_std=config.measurement_noise_is_std)
        self.targets: Dict[int, NDArray[np.float64]] = {}
        self.target_ages: Dict[int, int] = {}
        self.next_target_id = 0
        self.frame_index = 0

    def reset(self) -> Frame:
        self.targets = {}
        self.target_ages = {}
        self.next_target_id = 0
        self.frame_index = 0
        self._spawn(self.config.initial_targets)
        return self._observe()

    def _sample_initial_state(self) -> NDArray[np.float64]:
        position = self.rng.multivariate_normal(
            np.asarray(self.config.initial_position_mean), np.asarray(self.config.initial_position_covariance)
        )
        velocity = self.rng.multivariate_normal(
            np.asarray(self.config.initial_velocity_mean), np.asarray(self.config.initial_velocity_covariance)
        )
        return np.concatenate((position, velocity)).astype(np.float64)

    def _spawn(self, count: int) -> None:
        available = max(0, self.config.max_targets - len(self.targets))
        for _ in range(min(int(count), available)):
            target_id = self.next_target_id
            self.next_target_id += 1
            self.targets[target_id] = self._sample_initial_state()
            self.target_ages[target_id] = 0

    def _in_field_of_view(self, state: NDArray[np.float64]) -> bool:
        x_min, x_max, y_min, y_max = self.config.field_of_view
        return x_min <= state[0] <= x_max and y_min <= state[1] <= y_max

    def _propagate(self) -> None:
        propagated: Dict[int, NDArray[np.float64]] = {}
        propagated_ages: Dict[int, int] = {}
        for target_id, state in self.targets.items():
            if self.rng.random() > self.config.survival_probability:
                continue
            candidate = self._f @ state + self.rng.multivariate_normal(np.zeros(4), self._q)
            if self._in_field_of_view(candidate):
                propagated[target_id] = candidate
                propagated_ages[target_id] = self.target_ages[target_id] + 1
        self.targets = propagated
        self.target_ages = propagated_ages
        self._spawn(self.rng.poisson(self.config.birth_rate))

    def _observe(self) -> Frame:
        ids = np.fromiter(self.targets.keys(), dtype=np.int64, count=len(self.targets))
        ages = np.fromiter(self.target_ages.values(), dtype=np.int64, count=len(self.target_ages))
        states = (
            np.stack(list(self.targets.values())).astype(np.float32)
            if self.targets
            else np.empty((0, 4), dtype=np.float32)
        )
        if not self.targets:
            ages = np.empty((0,), dtype=np.int64)
        measurements: List[NDArray[np.float64]] = []
        measurement_ids: List[int] = []
        for target_id, state in self.targets.items():
            if self.rng.random() <= self.config.detection_probability:
                measurements.append(state[:2] + self.rng.multivariate_normal(np.zeros(2), self._r))
                measurement_ids.append(target_id)

        clutter_count = int(self.rng.poisson(self.config.clutter_rate))
        x_min, x_max, y_min, y_max = self.config.field_of_view
        if clutter_count:
            clutter = np.column_stack(
                (self.rng.uniform(x_min, x_max, clutter_count), self.rng.uniform(y_min, y_max, clutter_count))
            )
            measurements.extend(clutter)
            measurement_ids.extend([-1] * clutter_count)

        if measurements:
            order = self.rng.permutation(len(measurements))
            measurement_array = np.asarray(measurements, dtype=np.float32)[order]
            measurement_id_array = np.asarray(measurement_ids, dtype=np.int64)[order]
        else:
            measurement_array = np.empty((0, 2), dtype=np.float32)
            measurement_id_array = np.empty((0,), dtype=np.int64)

        return Frame(
            index=self.frame_index,
            time=self.frame_index * self.config.dt,
            states=states,
            target_ids=ids,
            target_ages=ages,
            measurements=measurement_array,
            measurement_ids=measurement_id_array,
        )

    def step(self) -> Frame:
        self.frame_index += 1
        self._propagate()
        return self._observe()

    def simulate(self, num_steps: int) -> List[Frame]:
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        frames = [self.reset()]
        frames.extend(self.step() for _ in range(num_steps - 1))
        return frames

