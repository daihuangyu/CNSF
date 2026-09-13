from __future__ import annotations

import math

import torch
from torch import nn


class SpatialEncoding(nn.Module):
    """Sinusoidal encoding of normalized coordinates.

    A single ``Linear(2, hidden_dim)`` gives every token a rank-2 spatial
    signature, so keys inside one frame become nearly indistinguishable and
    cross-attention collapses onto a uniform average over the measurement
    memory. Expanding each coordinate over a geometric frequency ladder
    restores the high-frequency detail that dot-product attention needs to
    perform spatial selection, matching the position encoder used by MT3.

    Inputs are expected on the unit interval so that measurement keys and
    decoder reference points share one embedding space.

    ``temperature`` is much smaller than the value used for image backbones
    because two normalized coordinates need their frequency ladder
    concentrated inside the field of view rather than spread over thousands of
    pixels.
    """

    def __init__(
        self,
        hidden_dim: int,
        coordinate_dim: int = 2,
        *,
        temperature: float = 20.0,
        scale: float = 2.0 * math.pi,
    ):
        super().__init__()
        if coordinate_dim < 1:
            raise ValueError("coordinate_dim must be positive")
        if hidden_dim % (2 * coordinate_dim):
            raise ValueError(
                "hidden_dim must be divisible by twice the coordinate dimension"
            )
        self.coordinate_dim = coordinate_dim
        self.features_per_coordinate = hidden_dim // coordinate_dim
        self.scale = scale
        divisor = temperature ** (
            2
            * (torch.arange(self.features_per_coordinate, dtype=torch.float32) // 2)
            / self.features_per_coordinate
        )
        self.register_buffer("divisor", divisor)

    def forward(self, unit_positions: torch.Tensor) -> torch.Tensor:
        if unit_positions.shape[-1] != self.coordinate_dim:
            raise ValueError(
                f"expected {self.coordinate_dim} coordinates, got {unit_positions.shape[-1]}"
            )
        scaled = unit_positions.unsqueeze(-1) * self.scale / self.divisor
        encoded = torch.stack(
            (scaled[..., 0::2].sin(), scaled[..., 1::2].cos()), dim=-1
        )
        return encoded.flatten(-3)
