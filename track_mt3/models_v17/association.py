from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .config import V17BConfig
from .time_encoding import ContinuousTimeEncoding


class PosteriorAssociationDecoderLayer(nn.Module):
    """v9-style query reasoning restricted to current-frame association.

    Cross-attended values refine association queries only.  They are never
    written directly into the physical track posterior; the final Sinkhorn
    probabilities still control the pair-wise Kalman-shaped update.
    """

    def __init__(self, config: V17BConfig):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cross_attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.feedforward = nn.Sequential(
            nn.Linear(config.hidden_dim, config.feedforward_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.hidden_dim),
        )
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.norm3 = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    @staticmethod
    def _numerically_safe_mask(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose one dummy token for all-masked rows and report those rows."""

        safe = mask.clone()
        empty = safe.all(-1)
        # Boolean indexing is a no-op when ``empty`` has no true values.  Do
        # not branch on torch.any(empty): on CUDA that Python condition forces
        # a device synchronization in every decoder layer.
        safe[empty, 0] = False
        return safe, empty

    def forward(
        self,
        tracks: torch.Tensor,
        measurements: torch.Tensor,
        active_mask: torch.Tensor,
        measurement_padding_mask: torch.Tensor,
        track_positions: torch.Tensor,
    ) -> torch.Tensor:
        track_padding, no_tracks = self._numerically_safe_mask(~active_mask)
        measurement_padding, no_measurements = self._numerically_safe_mask(
            measurement_padding_mask
        )
        query = tracks + track_positions
        attended = self.self_attention(
            query,
            query,
            tracks,
            key_padding_mask=track_padding,
            need_weights=False,
        )[0]
        attended = attended.masked_fill(no_tracks[:, None, None], 0.0)
        tracks = self.norm1(tracks + self.dropout(attended))
        attended = self.cross_attention(
            tracks + track_positions,
            measurements,
            measurements,
            key_padding_mask=measurement_padding,
            need_weights=False,
        )[0]
        attended = attended.masked_fill(no_measurements[:, None, None], 0.0)
        tracks = self.norm2(tracks + self.dropout(attended))
        tracks = self.norm3(tracks + self.dropout(self.feedforward(tracks)))
        return torch.where(active_mask.unsqueeze(-1), tracks, torch.zeros_like(tracks))


@dataclass(frozen=True)
class AssociationTransport:
    """Capacity-constrained current-frame association probabilities."""

    pair_probabilities: torch.Tensor
    miss_probabilities: torch.Tensor
    unclaimed_probabilities: torch.Tensor
    death_probabilities: torch.Tensor | None
    pair_log_probabilities: torch.Tensor
    miss_log_probabilities: torch.Tensor
    unclaimed_log_probabilities: torch.Tensor
    death_log_probabilities: torch.Tensor | None
    marginal_error: torch.Tensor
    capacity_violation: torch.Tensor
    alternative_pair_probabilities: torch.Tensor | None = None
    alternative_miss_probabilities: torch.Tensor | None = None
    alternative_unclaimed_probabilities: torch.Tensor | None = None
    hypothesis_scores: torch.Tensor | None = None


def log_sinkhorn_transport(
    scores: torch.Tensor,
    row_mass: torch.Tensor,
    column_mass: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    """Log-space Sinkhorn with zero-mass padded rows and columns."""

    row_valid = row_mass > 0
    column_valid = column_mass > 0
    log_rows = torch.where(row_valid, row_mass.clamp_min(1.0e-12).log(), 0.0)
    log_columns = torch.where(column_valid, column_mass.clamp_min(1.0e-12).log(), 0.0)
    u = torch.zeros_like(row_mass)
    v = torch.zeros_like(column_mass)
    for _ in range(iterations):
        row_normalizer = torch.logsumexp(scores + v.unsqueeze(-2), dim=-1)
        u = torch.where(
            row_valid, log_rows - row_normalizer, torch.zeros_like(row_mass)
        )
        column_normalizer = torch.logsumexp(scores + u.unsqueeze(-1), dim=-2)
        v = torch.where(
            column_valid,
            log_columns - column_normalizer,
            torch.zeros_like(column_mass),
        )
    return scores + u.unsqueeze(-1) + v.unsqueeze(-2)


class JointAssociation(nn.Module):
    """Track-prior-conditioned association with MISS and UNCLAIMED bins."""

    def __init__(self, config: V17BConfig):
        super().__init__()
        self.config = config
        self.time_encoding = ContinuousTimeEncoding(config.hidden_dim)
        self.decoder_layers = nn.ModuleList(
            PosteriorAssociationDecoderLayer(config)
            for _ in range(config.association_decoder_layers)
        )
        if self.decoder_layers:
            self.track_state_projection = nn.Sequential(
                nn.Linear(8, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
            )
            self.physical_gate = nn.Linear(config.hidden_dim, 1)
            nn.init.zeros_(self.physical_gate.weight)
            nn.init.zeros_(self.physical_gate.bias)
        pair_dim = 3 * config.hidden_dim + 4
        self.pair_network = nn.Sequential(
            nn.Linear(pair_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )
        self.residual_score = nn.Linear(config.hidden_dim, 1)
        self.miss_score = nn.Sequential(
            nn.Linear(2 * config.hidden_dim + 2, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.unclaimed_score = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.dustbin_score = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.residual_score.weight)
        nn.init.zeros_(self.residual_score.bias)
        nn.init.zeros_(self.miss_score[-1].weight)
        nn.init.zeros_(self.miss_score[-1].bias)
        nn.init.zeros_(self.unclaimed_score[-1].weight)
        nn.init.zeros_(self.unclaimed_score[-1].bias)
        if self.decoder_layers:
            # Legacy v17 deliberately starts every learned score at exactly
            # zero.  With a deep decoder that would also block all first-step
            # gradients into its attention layers.  Tiny non-zero readouts keep
            # the initial transport physically anchored while opening an
            # immediate optimization path through the decoder.
            nn.init.normal_(self.residual_score.weight, mean=0.0, std=1.0e-3)
            nn.init.normal_(self.miss_score[-1].weight, mean=0.0, std=1.0e-3)

    def forward(
        self,
        prior_mean: torch.Tensor,
        prior_covariance: torch.Tensor,
        prior_query: torch.Tensor,
        measurements: torch.Tensor,
        measurement_embeddings: torch.Tensor,
        active_mask: torch.Tensor,
        measurement_padding_mask: torch.Tensor,
        delta_t_association: torch.Tensor,
        track_measurement_bias: torch.Tensor | None = None,
        survival_prior_logits: torch.Tensor | None = None,
        pair_measurement_variance: torch.Tensor | None = None,
    ) -> AssociationTransport:
        batch, slots, _ = prior_mean.shape
        count = measurements.shape[1]
        time_feature = self.time_encoding(delta_t_association)
        association_query = prior_query
        if self.decoder_layers:
            diagonal = prior_covariance.diagonal(dim1=-2, dim2=-1)
            track_positions = self.track_state_projection(
                torch.cat(
                    (prior_mean / self.config.field_scale, diagonal.log1p()), dim=-1
                )
            )
            association_query = prior_query + time_feature + track_positions
            for layer in self.decoder_layers:
                association_query = layer(
                    association_query,
                    measurement_embeddings,
                    active_mask,
                    measurement_padding_mask,
                    track_positions,
                )
        innovation = measurements[:, None] - prior_mean[:, :, None, :2]
        position_variance = prior_covariance.diagonal(dim1=-2, dim2=-1)[..., :2]
        pair_features = torch.cat(
            (
                association_query[:, :, None].expand(-1, -1, count, -1),
                measurement_embeddings[:, None].expand(-1, slots, -1, -1),
                time_feature[:, :, None].expand(-1, -1, count, -1),
                innovation / self.config.field_scale,
                position_variance.log1p()[:, :, None].expand(-1, -1, count, -1),
            ),
            dim=-1,
        )
        pair_hidden = self.pair_network(pair_features)
        residual = self.config.association_residual_scale * torch.tanh(
            self.residual_score(pair_hidden).squeeze(-1)
        )

        if pair_measurement_variance is None:
            measurement_variance = prior_mean.new_full(
                (batch, slots, count, 2),
                self.config.association_measurement_std**2,
            )
        else:
            if pair_measurement_variance.shape != (batch, slots, count, 2):
                raise ValueError("pair_measurement_variance must have shape [B,S,M,2]")
            measurement_variance = pair_measurement_variance
        innovation_covariance = (
            prior_covariance[..., :2, :2].unsqueeze(-3)
            + torch.diag_embed(measurement_variance)
        )
        cholesky = torch.linalg.cholesky(innovation_covariance)
        solved = torch.cholesky_solve(innovation.unsqueeze(-1), cholesky)
        mahalanobis = (innovation.unsqueeze(-2) @ solved).squeeze(-1).squeeze(-1)
        log_determinant = 2.0 * torch.log(cholesky.diagonal(dim1=-2, dim2=-1)).sum(-1)
        probabilistic_bias = -0.5 * (mahalanobis + log_determinant)
        probabilistic_bias = probabilistic_bias.clamp(min=-30.0, max=15.0)
        if self.config.association_learned_physical_gate:
            physical_weight = torch.sigmoid(
                self.physical_gate(pair_hidden).squeeze(-1)
            )
            pair_scores = physical_weight * probabilistic_bias + residual
        else:
            pair_scores = probabilistic_bias + residual
        if track_measurement_bias is not None:
            if track_measurement_bias.shape != (batch, slots):
                raise ValueError("track_measurement_bias must have shape [B,S]")
            pair_scores = pair_scores + track_measurement_bias.unsqueeze(-1)
        miss_scores = self.miss_score(
            torch.cat(
                (association_query, time_feature, position_variance.log1p()), dim=-1
            )
        ).squeeze(-1)
        unclaimed_scores = self.unclaimed_score(measurement_embeddings).squeeze(-1)

        valid_measurements = ~measurement_padding_mask
        valid_pairs = active_mask.unsqueeze(-1) & valid_measurements.unsqueeze(-2)
        # A finite sentinel avoids NaN gradients through fully padded
        # logsumexp rows while underflowing to exactly zero transport mass.
        masked_score = -1.0e4
        joint_lifecycle = self.config.joint_lifecycle_transport
        if joint_lifecycle and survival_prior_logits is None:
            raise ValueError("joint lifecycle transport requires survival_prior_logits")
        if survival_prior_logits is not None and survival_prior_logits.shape != (
            batch,
            slots,
        ):
            raise ValueError("survival_prior_logits must have shape [B,S]")
        extra_columns = 2 if joint_lifecycle else 1
        augmented = pair_scores.new_full(
            (batch, slots + 1, count + extra_columns), masked_score
        )
        if joint_lifecycle:
            log_survival = F.logsigmoid(survival_prior_logits)
            pair_scores = pair_scores + log_survival.unsqueeze(-1)
            miss_scores = miss_scores + log_survival
        augmented[:, :slots, :count] = pair_scores.masked_fill(
            ~valid_pairs, masked_score
        )
        augmented[:, :slots, count] = miss_scores.masked_fill(
            ~active_mask, masked_score
        )
        if joint_lifecycle:
            death_scores = F.logsigmoid(-survival_prior_logits)
            augmented[:, :slots, count + 1] = death_scores.masked_fill(
                ~active_mask, masked_score
            )
        augmented[:, slots, :count] = unclaimed_scores.masked_fill(
            ~valid_measurements, masked_score
        )
        active_count = active_mask.sum(-1).to(prior_mean.dtype)
        measurement_count = valid_measurements.sum(-1).to(prior_mean.dtype)
        dummy_mass = (
            measurement_count + active_count if joint_lifecycle else measurement_count
        )
        row_mass = torch.cat(
            (active_mask.to(prior_mean.dtype), dummy_mass[:, None]), dim=-1
        )
        lifecycle_columns = (
            torch.stack((active_count, active_count), dim=-1)
            if joint_lifecycle
            else active_count[:, None]
        )
        column_mass = torch.cat(
            (valid_measurements.to(prior_mean.dtype), lifecycle_columns), dim=-1
        )
        # Keep the vectorized iterations finite for a completely empty frame.
        empty = (active_count + measurement_count) == 0
        row_mass[empty, -1] = 1.0
        column_mass[empty, count] = 1.0
        dustbin_valid = (
            ((active_count > 0) | empty)
            if joint_lifecycle
            else ((active_count > 0) & (measurement_count > 0)) | empty
        )
        augmented[:, slots, count] = torch.where(
            dustbin_valid,
            self.dustbin_score.expand(batch),
            augmented.new_full((batch,), masked_score),
        )
        if joint_lifecycle:
            augmented[:, slots, count + 1] = torch.where(
                dustbin_valid,
                self.dustbin_score.expand(batch),
                augmented.new_full((batch,), masked_score),
            )
        augmented = augmented / self.config.association_temperature
        log_transport = log_sinkhorn_transport(
            augmented,
            row_mass,
            column_mass,
            self.config.association_sinkhorn_iterations,
        )
        transport = log_transport.exp()
        pair = transport[:, :slots, :count]
        miss = transport[:, :slots, count]
        death = transport[:, :slots, count + 1] if joint_lifecycle else None
        unclaimed = transport[:, slots, :count]

        alternative_pair = None
        alternative_miss = None
        alternative_unclaimed = None
        hypothesis_scores = None
        if self.config.association_mbm_hypotheses == 2:
            # A second globally feasible soft assignment is obtained by
            # suppressing the dominant PAIR edge of each active row and
            # re-solving the same capacity-constrained transport.  This is a
            # differentiable, bounded analogue of requesting the next Murty
            # assignment; it never forms independent per-track top-2 choices.
            repelled = augmented.clone()
            if count:
                dominant_column = pair.argmax(-1, keepdim=True)
                dominant_mass = pair.gather(-1, dominant_column).squeeze(-1)
                repel_row = active_mask & (dominant_mass > miss)
                penalty = torch.zeros_like(pair).scatter(
                    -1,
                    dominant_column,
                    repel_row.to(pair.dtype).unsqueeze(-1)
                    * self.config.association_mbm_repulsion,
                )
                repelled[:, :slots, :count] = (
                    repelled[:, :slots, :count] - penalty
                )
            alternative_log_transport = log_sinkhorn_transport(
                repelled,
                row_mass,
                column_mass,
                self.config.association_sinkhorn_iterations,
            )
            alternative_transport = alternative_log_transport.exp()
            alternative_pair = alternative_transport[:, :slots, :count]
            alternative_miss = alternative_transport[:, :slots, count]
            alternative_unclaimed = alternative_transport[:, slots, :count]
            normalizer = (active_count + measurement_count).clamp_min(1.0)
            primary_score = (transport * augmented).sum(dim=(-2, -1)) / normalizer
            alternative_score = (
                alternative_transport * augmented
            ).sum(dim=(-2, -1)) / normalizer
            hypothesis_scores = torch.stack(
                (primary_score, alternative_score), dim=-1
            )

        row_sum = pair.sum(-1) + miss
        if death is not None:
            row_sum = row_sum + death
        column_sum = pair.sum(-2) + unclaimed
        row_error = torch.where(
            active_mask, (row_sum - 1.0).abs(), torch.zeros_like(row_sum)
        )
        column_error = torch.where(
            valid_measurements,
            (column_sum - 1.0).abs(),
            torch.zeros_like(column_sum),
        )
        marginal_error = torch.maximum(row_error.amax(-1), column_error.amax(-1))
        capacity_violation = torch.relu(pair.sum(-2) - 1.0).amax(-1)
        return AssociationTransport(
            pair_probabilities=pair,
            miss_probabilities=miss,
            unclaimed_probabilities=unclaimed,
            death_probabilities=death,
            pair_log_probabilities=log_transport[:, :slots, :count],
            miss_log_probabilities=log_transport[:, :slots, count],
            unclaimed_log_probabilities=log_transport[:, slots, :count],
            death_log_probabilities=(
                log_transport[:, :slots, count + 1] if joint_lifecycle else None
            ),
            marginal_error=marginal_error,
            capacity_violation=capacity_violation,
            alternative_pair_probabilities=alternative_pair,
            alternative_miss_probabilities=alternative_miss,
            alternative_unclaimed_probabilities=alternative_unclaimed,
            hypothesis_scores=hypothesis_scores,
        )
