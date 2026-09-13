from __future__ import annotations

import copy

import torch
from torch import nn

from track_mt3.config import ModelConfig

from .spatial_encoding import SpatialEncoding
from .temporal_encoding import TemporalEncoding


class MeasurementEncoderLayer(nn.Module):
    """Post-norm encoder layer that re-injects positions into queries and keys.

    ``nn.TransformerEncoder`` only sees the positional signature once, in the
    input sum. Six post-norm layers then drive the tokens towards a common
    direction: measured layer by layer, the cosine similarity between distinct
    measurements rises 0.46 -> 0.9998 while the same-target margin decays from
    +0.0865 to +0.0001, an 865x loss. Once every key is nearly parallel, the
    shared component of ``q @ k`` cancels inside the softmax and cross-attention
    can only return a uniform average.

    Re-adding the positional signature at every layer keeps that signal from
    being normalized away, and matches the official MT3 encoder, which computes
    ``q = k = src + pos`` in each layer while leaving values untouched.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            config.hidden_dim, config.num_heads, config.dropout, batch_first=True
        )
        self.linear1 = nn.Linear(config.hidden_dim, config.feedforward_dim)
        self.linear2 = nn.Linear(config.feedforward_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        source: torch.Tensor,
        positions: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        query = source if positions is None else source + positions
        attended = self.self_attention(
            query,
            query,
            source,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        source = self.norm1(source + self.dropout(attended))
        feedforward = self.linear2(self.dropout(torch.relu(self.linear1(source))))
        return self.norm2(source + self.dropout(feedforward))


class MeasurementEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_projection = nn.Linear(config.measurement_dim, config.hidden_dim)
        self.temporal_encoding = TemporalEncoding(config.window_size, config.hidden_dim, config.temporal_encoding)
        self.spatial_encoding: SpatialEncoding | None = None
        if config.spatial_encoding == "sinusoidal":
            self.spatial_encoding = SpatialEncoding(
                config.hidden_dim,
                config.measurement_dim,
                temperature=config.spatial_encoding_temperature,
            )
            # A learnable projection lets the network scale the sinusoidal
            # signature against the temporal embedding instead of inheriting a
            # fixed amplitude ratio.
            self.spatial_projection = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
            )
        elif config.spatial_encoding != "none":
            raise ValueError(f"unknown spatial encoding: {config.spatial_encoding}")
        self.inject_positions_every_layer = config.encoder_position_every_layer
        prototype = MeasurementEncoderLayer(config)
        self.layers = nn.ModuleList(
            copy.deepcopy(prototype) for _ in range(config.encoder_layers)
        )

    def spatial_positions(self, unit_measurements: torch.Tensor) -> torch.Tensor | None:
        """Positional signature of measurements on the unit interval."""
        if self.spatial_encoding is None:
            return None
        return self.spatial_projection(self.spatial_encoding(unit_measurements))

    def forward(
        self,
        measurements: torch.Tensor,
        time_indices: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        spatial_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        positions = self.temporal_encoding(time_indices)
        if spatial_positions is not None:
            positions = positions + spatial_positions
        features = self.input_projection(measurements) + positions
        layer_positions = positions if self.inject_positions_every_layer else None
        for layer in self.layers:
            features = layer(features, layer_positions, padding_mask)
        return features
