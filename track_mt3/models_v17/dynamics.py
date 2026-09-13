from __future__ import annotations

import torch
from torch import nn

from .config import V17AConfig
from .time_encoding import ContinuousTimeEncoding


def constant_velocity_matrix(delta_t: torch.Tensor) -> torch.Tensor:
    """Return batched constant-velocity transition matrices for ``delta_t``."""

    shape = (*delta_t.shape, 4, 4)
    matrix = torch.eye(4, device=delta_t.device, dtype=delta_t.dtype).expand(shape).clone()
    matrix[..., 0, 2] = delta_t
    matrix[..., 1, 3] = delta_t
    return matrix


class StructuredDynamics(nn.Module):
    """Kinematic prediction with learned acceleration and process uncertainty."""

    def __init__(self, config: V17AConfig):
        super().__init__()
        self.config = config
        self.time_encoding = ContinuousTimeEncoding(config.hidden_dim)
        self.transition = nn.Sequential(
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )
        self.acceleration_head = nn.Linear(config.hidden_dim, 2)
        self.process_variance_head = nn.Linear(config.hidden_dim, 4)
        nn.init.zeros_(self.acceleration_head.weight)
        nn.init.zeros_(self.acceleration_head.bias)
        nn.init.zeros_(self.process_variance_head.weight)
        nn.init.constant_(self.process_variance_head.bias, -5.0)

    def forward(
        self,
        mean: torch.Tensor,
        covariance: torch.Tensor,
        query: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_feature = self.time_encoding(delta_t)
        residual = self.transition(torch.cat((query, time_feature), dim=-1))
        predicted_query = query + residual
        acceleration = self.acceleration_head(predicted_query)

        transition = constant_velocity_matrix(delta_t)
        predicted_mean = (transition @ mean.unsqueeze(-1)).squeeze(-1)
        dt = delta_t.unsqueeze(-1)
        predicted_mean = predicted_mean + torch.cat(
            (0.5 * dt.square() * acceleration, dt * acceleration), dim=-1
        )

        process_variance = torch.nn.functional.softplus(
            self.process_variance_head(predicted_query)
        ) + self.config.min_variance
        process_variance = process_variance.clamp_max(self.config.max_variance)
        predicted_covariance = (
            transition @ covariance @ transition.transpose(-1, -2)
            + torch.diag_embed(process_variance * delta_t.unsqueeze(-1).clamp_min(1.0e-3))
        )
        predicted_covariance = 0.5 * (
            predicted_covariance + predicted_covariance.transpose(-1, -2)
        )
        return predicted_mean, predicted_covariance, predicted_query
