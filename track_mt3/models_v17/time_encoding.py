from __future__ import annotations

import math

import torch
from torch import nn


class ContinuousTimeEncoding(nn.Module):
    """Physical relative-time features, independent of absolute frame index."""

    def __init__(self, output_dim: int, frequencies: int = 8):
        super().__init__()
        omega = torch.exp(torch.linspace(math.log(0.25), math.log(16.0), frequencies))
        self.register_buffer("omega", omega)
        input_dim = 2 + 2 * frequencies
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        value = delta_t.clamp_min(0.0).unsqueeze(-1)
        phase = value * self.omega
        features = torch.cat(
            (value, torch.log1p(value), torch.sin(phase), torch.cos(phase)), dim=-1
        )
        return self.projection(features)
