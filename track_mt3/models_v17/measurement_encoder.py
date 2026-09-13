from __future__ import annotations

import copy

import torch
from torch import nn

from .config import V17AConfig


class CurrentFrameEncoderLayer(nn.Module):
    def __init__(self, config: V17AConfig):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.feedforward_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.hidden_dim),
        )
        self.norm2 = nn.LayerNorm(config.hidden_dim)

    def forward(
        self, value: torch.Tensor, position: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        query = value + position
        attended = self.attention(
            query, query, value, key_padding_mask=mask, need_weights=False
        )[0]
        value = self.norm1(value + attended)
        return self.norm2(value + self.ffn(value))


class CurrentFrameMeasurementEncoder(nn.Module):
    """Permutation-equivariant encoder that never receives historical tokens."""

    def __init__(self, config: V17AConfig):
        super().__init__()
        self.field_scale = config.field_scale
        self.input_projection = nn.Sequential(
            nn.Linear(2, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )
        self.position_projection = nn.Sequential(
            nn.Linear(6, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )
        prototype = CurrentFrameEncoderLayer(config)
        self.layers = nn.ModuleList(
            copy.deepcopy(prototype) for _ in range(config.encoder_layers)
        )

    def forward(
        self, measurements: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        original_mask = padding_mask
        # MultiheadAttention returns NaN for an all-masked row.  Keep one dummy
        # key numerically visible, then zero every originally padded output.
        padding_mask = padding_mask.clone()
        empty = padding_mask.all(dim=1)
        if torch.any(empty):
            padding_mask[empty, 0] = False
        normalized = measurements / self.field_scale
        x, y = normalized.unbind(-1)
        position_features = torch.stack(
            (x, y, torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)), dim=-1
        )
        position = self.position_projection(position_features)
        value = self.input_projection(normalized) + position
        for layer in self.layers:
            value = layer(value, position, padding_mask)
        return value.masked_fill(original_mask.unsqueeze(-1), 0.0)
