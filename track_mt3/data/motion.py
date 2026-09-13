from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def transition_matrix(dt: float) -> NDArray[np.float64]:
    return np.array(
        [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def process_covariance(dt: float, intensity: float, *, intensity_is_std: bool = False) -> NDArray[np.float64]:
    scale = intensity**2 if intensity_is_std else intensity
    base = np.array(
        [
            [dt**3 / 3.0, 0.0, dt**2 / 2.0, 0.0],
            [0.0, dt**3 / 3.0, 0.0, dt**2 / 2.0],
            [dt**2 / 2.0, 0.0, dt, 0.0],
            [0.0, dt**2 / 2.0, 0.0, dt],
        ],
        dtype=np.float64,
    )
    return scale * base


def measurement_covariance(intensity: float, *, intensity_is_std: bool = False) -> NDArray[np.float64]:
    scale = intensity**2 if intensity_is_std else intensity
    return scale * np.eye(2, dtype=np.float64)

