from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import V17AConfig
from .time_encoding import ContinuousTimeEncoding


@dataclass(frozen=True)
class PairUpdateCandidates:
    """All current-frame pair posteriors computed exactly once."""

    mean: torch.Tensor
    covariance: torch.Tensor
    query: torch.Tensor
    measurement_variance: torch.Tensor


class LearnedMeasurementUpdate(nn.Module):
    """Kalman-shaped update whose noise, gain residual and event write are learned."""

    def __init__(self, config: V17AConfig):
        super().__init__()
        self.config = config
        self.time_encoding = ContinuousTimeEncoding(config.hidden_dim)
        pair_dim = 3 * config.hidden_dim + 4
        self.pair_network = nn.Sequential(
            nn.Linear(pair_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )
        self.measurement_variance_head = nn.Linear(config.hidden_dim, 2)
        self.gain_residual_head = nn.Linear(config.hidden_dim, 8)
        self.event_update = nn.GRUCell(config.hidden_dim, config.hidden_dim)
        nn.init.zeros_(self.gain_residual_head.weight)
        nn.init.zeros_(self.gain_residual_head.bias)
        nn.init.zeros_(self.measurement_variance_head.weight)
        nn.init.constant_(self.measurement_variance_head.bias, -5.0)

    def _pair_embedding(
        self,
        prior_mean: torch.Tensor,
        prior_covariance: torch.Tensor,
        prior_query: torch.Tensor,
        measurements: torch.Tensor,
        measurement_embeddings: torch.Tensor,
        delta_t_association: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an aligned track/measurement pair and return its innovation."""
        innovation = measurements - prior_mean[..., :2]
        time_feature = self.time_encoding(delta_t_association)
        pair = self.pair_network(
            torch.cat(
                (
                    prior_query,
                    measurement_embeddings,
                    time_feature,
                    innovation / self.config.field_scale,
                    prior_covariance.diagonal(dim1=-2, dim2=-1)[..., :2].log1p(),
                ),
                dim=-1,
            )
        )
        return pair, innovation

    def _measurement_variance(self, pair: torch.Tensor) -> torch.Tensor:
        measurement_variance = torch.nn.functional.softplus(
            self.measurement_variance_head(pair)
        ) + self.config.min_variance
        return measurement_variance.clamp_max(self.config.max_variance)

    def pair_measurement_variance(
        self,
        prior_mean: torch.Tensor,
        prior_covariance: torch.Tensor,
        prior_query: torch.Tensor,
        measurements: torch.Tensor,
        measurement_embeddings: torch.Tensor,
        delta_t_association: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one observation variance for every track/measurement pair.

        v18-A calls this before association and reuses the identical tensor in
        the Kalman-shaped update.  The association likelihood and state update
        therefore describe one coherent learned observation model.
        """

        batch, slots, _ = prior_mean.shape
        count = measurements.shape[1]
        pair, _ = self._pair_embedding(
            prior_mean[:, :, None].expand(-1, -1, count, -1),
            prior_covariance[:, :, None].expand(-1, -1, count, -1, -1),
            prior_query[:, :, None].expand(-1, -1, count, -1),
            measurements[:, None].expand(-1, slots, -1, -1),
            measurement_embeddings[:, None].expand(-1, slots, -1, -1),
            delta_t_association[:, :, None].expand(-1, -1, count),
        )
        variance = self._measurement_variance(pair)
        if variance.shape != (batch, slots, count, 2):
            raise RuntimeError("invalid pair measurement variance shape")
        return variance

    def pair_update_candidates(
        self,
        prior_mean: torch.Tensor,
        prior_covariance: torch.Tensor,
        prior_query: torch.Tensor,
        measurements: torch.Tensor,
        measurement_embeddings: torch.Tensor,
        delta_t_association: torch.Tensor,
    ) -> PairUpdateCandidates:
        """Form every pair posterior once for association and state write."""

        batch, slots, _ = prior_mean.shape
        count = measurements.shape[1]
        mean, covariance, query, variance = self._candidate_update(
            prior_mean[:, :, None].expand(-1, -1, count, -1),
            prior_covariance[:, :, None].expand(-1, -1, count, -1, -1),
            prior_query[:, :, None].expand(-1, -1, count, -1),
            measurements[:, None].expand(-1, slots, -1, -1),
            measurement_embeddings[:, None].expand(-1, slots, -1, -1),
            delta_t_association[:, :, None].expand(-1, -1, count),
        )
        if mean.shape != (batch, slots, count, 4):
            raise RuntimeError("invalid pair posterior mean shape")
        return PairUpdateCandidates(mean, covariance, query, variance)

    def _candidate_update(
        self,
        prior_mean: torch.Tensor,
        prior_covariance: torch.Tensor,
        prior_query: torch.Tensor,
        measurements: torch.Tensor,
        measurement_embeddings: torch.Tensor,
        delta_t_association: torch.Tensor,
        measurement_variance_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Form one learned Kalman-shaped posterior for every aligned pair."""
        pair, innovation = self._pair_embedding(
            prior_mean,
            prior_covariance,
            prior_query,
            measurements,
            measurement_embeddings,
            delta_t_association,
        )
        measurement_variance = (
            self._measurement_variance(pair)
            if measurement_variance_override is None
            else measurement_variance_override
        )
        if measurement_variance.shape != (*pair.shape[:-1], 2):
            raise ValueError("measurement_variance_override has an invalid shape")
        measurement_covariance = torch.diag_embed(measurement_variance)

        pht = prior_covariance[..., :, :2]
        innovation_covariance = prior_covariance[..., :2, :2] + measurement_covariance
        cholesky = torch.linalg.cholesky(innovation_covariance)
        base_gain = torch.cholesky_solve(
            pht.transpose(-1, -2), cholesky
        ).transpose(-1, -2)
        residual_gain = self.config.gain_residual_scale * torch.tanh(
            self.gain_residual_head(pair).view(*pair.shape[:-1], 4, 2)
        )
        gain = base_gain + residual_gain

        candidate_mean = prior_mean + (gain @ innovation.unsqueeze(-1)).squeeze(-1)
        identity = torch.eye(
            4, device=prior_mean.device, dtype=prior_mean.dtype
        ).expand(*prior_mean.shape[:-1], 4, 4)
        kh = torch.zeros_like(identity)
        kh[..., :, :2] = gain
        left = identity - kh
        candidate_covariance = (
            left @ prior_covariance @ left.transpose(-1, -2)
            + gain @ measurement_covariance @ gain.transpose(-1, -2)
        )
        candidate_covariance = 0.5 * (
            candidate_covariance + candidate_covariance.transpose(-1, -2)
        )
        candidate_query = self.event_update(
            pair.reshape(-1, pair.shape[-1]),
            prior_query.reshape(-1, prior_query.shape[-1]),
        ).view_as(prior_query)
        return (
            candidate_mean,
            candidate_covariance,
            candidate_query,
            measurement_variance,
        )

    def forward(
        self,
        prior_mean: torch.Tensor,
        prior_covariance: torch.Tensor,
        prior_query: torch.Tensor,
        measurements: torch.Tensor,
        measurement_embeddings: torch.Tensor,
        delta_t_association: torch.Tensor,
        associated: torch.Tensor,
        measurement_variance_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate_mean, candidate_covariance, candidate_query, _ = (
            self._candidate_update(
                prior_mean,
                prior_covariance,
                prior_query,
                measurements,
                measurement_embeddings,
                delta_t_association,
                measurement_variance_override,
            )
        )

        # Boolean gates recover the exact G1 update. G2 may pass a Sinkhorn hit
        # probability, yielding a differentiable JPDA-like convex update.
        strength = associated.to(prior_mean.dtype).clamp(0.0, 1.0)
        gate = strength.unsqueeze(-1)
        mean = prior_mean + gate * (candidate_mean - prior_mean)
        query = prior_query + gate * (candidate_query - prior_query)
        covariance = prior_covariance + gate.unsqueeze(-1) * (
            candidate_covariance - prior_covariance
        )
        return mean, covariance, query

    def forward_mixture(
        self,
        prior_mean: torch.Tensor,
        prior_covariance: torch.Tensor,
        prior_query: torch.Tensor,
        measurements: torch.Tensor,
        measurement_embeddings: torch.Tensor,
        delta_t_association: torch.Tensor,
        pair_probabilities: torch.Tensor,
        pair_measurement_variance: torch.Tensor | None = None,
        pair_candidates: PairUpdateCandidates | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Moment-match pair posteriors and the MISS prior into one state.

        Unlike averaging raw measurements before the nonlinear update, this
        retains the between-hypothesis covariance required by a JPDA-like soft
        association. A one-hot pair row is exactly the ordinary update, while
        an all-zero row is exactly the prior/MISS state.
        """
        batch, slots, state_dim = prior_mean.shape
        count = measurements.shape[1]
        hidden_dim = prior_query.shape[-1]
        if pair_candidates is None:
            pair_mean, pair_covariance, pair_query, _ = self._candidate_update(
                prior_mean[:, :, None].expand(-1, -1, count, -1),
                prior_covariance[:, :, None].expand(-1, -1, count, -1, -1),
                prior_query[:, :, None].expand(-1, -1, count, -1),
                measurements[:, None].expand(-1, slots, -1, -1),
                measurement_embeddings[:, None].expand(-1, slots, -1, -1),
                delta_t_association[:, :, None].expand(-1, -1, count),
                pair_measurement_variance,
            )
        else:
            pair_mean = pair_candidates.mean
            pair_covariance = pair_candidates.covariance
            pair_query = pair_candidates.query
            if pair_mean.shape != (batch, slots, count, state_dim):
                raise ValueError("pair_candidates do not match the current frame")

        pair_weight = pair_probabilities.clamp_min(0.0)
        raw_pair_mass = pair_weight.sum(-1)
        pair_weight = pair_weight * torch.where(
            raw_pair_mass > 1.0,
            raw_pair_mass.clamp_min(1.0e-8).reciprocal(),
            torch.ones_like(raw_pair_mass),
        ).unsqueeze(-1)
        pair_mass = pair_weight.sum(-1)
        miss_weight = (1.0 - pair_mass).clamp_min(0.0)

        posterior_mean = (
            miss_weight.unsqueeze(-1) * prior_mean
            + (pair_weight.unsqueeze(-1) * pair_mean).sum(-2)
        )
        miss_delta = prior_mean - posterior_mean
        pair_delta = pair_mean - posterior_mean.unsqueeze(-2)
        posterior_covariance = miss_weight[..., None, None] * (
            prior_covariance + miss_delta.unsqueeze(-1) * miss_delta.unsqueeze(-2)
        )
        posterior_covariance = posterior_covariance + (
            pair_weight[..., None, None]
            * (
                pair_covariance
                + pair_delta.unsqueeze(-1) * pair_delta.unsqueeze(-2)
            )
        ).sum(-3)
        posterior_covariance = 0.5 * (
            posterior_covariance + posterior_covariance.transpose(-1, -2)
        )
        posterior_query = (
            miss_weight.unsqueeze(-1) * prior_query
            + (pair_weight.unsqueeze(-1) * pair_query).sum(-2)
        )
        if posterior_mean.shape != (batch, slots, state_dim):
            raise RuntimeError("invalid moment-matched mean shape")
        if posterior_query.shape != (batch, slots, hidden_dim):
            raise RuntimeError("invalid moment-matched query shape")
        return posterior_mean, posterior_covariance, posterior_query
