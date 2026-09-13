from __future__ import annotations

import copy
from typing import List, Tuple

import torch
from torch import nn

from track_mt3.config import ModelConfig

from .common import MLP, inverse_sigmoid
from .spatial_encoding import SpatialEncoding


class TrackingDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(config.hidden_dim, config.num_heads, config.dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(config.hidden_dim, config.num_heads, config.dropout, batch_first=True)
        self.linear1 = nn.Linear(config.hidden_dim, config.feedforward_dim)
        self.linear2 = nn.Linear(config.feedforward_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.norm3 = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor | None,
        query_padding_mask: torch.Tensor | None = None,
        query_positions: torch.Tensor | None = None,
        memory_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Positional signatures are added to queries and keys but never to
        # values, so attention gains a spatial metric while the transported
        # content stays untouched.
        def with_position(
            value: torch.Tensor, positions: torch.Tensor | None
        ) -> torch.Tensor:
            return value if positions is None else value + positions

        self_query = with_position(queries, query_positions)
        attended = self.self_attention(
            self_query,
            self_query,
            queries,
            key_padding_mask=query_padding_mask,
            need_weights=False,
        )[0]
        queries = self.norm1(queries + self.dropout(attended))
        attended, weights = self.cross_attention(
            with_position(queries, query_positions),
            with_position(memory, memory_positions),
            memory,
            key_padding_mask=memory_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        queries = self.norm2(queries + self.dropout(attended))
        feedforward = self.linear2(self.dropout(torch.relu(self.linear1(queries))))
        queries = self.norm3(queries + self.dropout(feedforward))
        return queries, weights


class TrackingDecoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        prototype = TrackingDecoderLayer(config)
        self.layers = nn.ModuleList(copy.deepcopy(prototype) for _ in range(config.decoder_layers))
        self.position_heads = nn.ModuleList(
            MLP(config.hidden_dim, config.prediction_hidden_dim, config.output_dim, config.prediction_layers)
            for _ in range(config.decoder_layers)
        )
        self.class_heads = nn.ModuleList(nn.Linear(config.hidden_dim, 1) for _ in range(config.decoder_layers))
        self.iterative_refinement = config.iterative_refinement
        self.detach_reference = config.detach_reference_between_layers
        self.query_position_mode = config.reference_query_embedding
        self.reference_encoding: SpatialEncoding | None = None
        if config.spatial_encoding not in ("sinusoidal", "none"):
            raise ValueError(f"unknown spatial encoding: {config.spatial_encoding}")
        if self.query_position_mode != "none":
            # Reference points are embedded with the same ladder used for the
            # measurement memory so that a query and a nearby measurement share
            # a comparable positional signature.
            self.reference_encoding = SpatialEncoding(
                config.hidden_dim,
                config.output_dim,
                temperature=config.spatial_encoding_temperature,
            )
            self.reference_projection = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
            )
        for head in self.position_heads:
            nn.init.zeros_(head.layers[-1].weight)
            nn.init.zeros_(head.layers[-1].bias)

    def query_positions(self, references: torch.Tensor) -> torch.Tensor | None:
        if self.reference_encoding is None:
            return None
        return self.reference_projection(self.reference_encoding(references))

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        references: torch.Tensor,
        memory_padding_mask: torch.Tensor | None = None,
        query_padding_mask: torch.Tensor | None = None,
        memory_positions: torch.Tensor | None = None,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
        states: List[torch.Tensor] = []
        logits: List[torch.Tensor] = []
        attention: List[torch.Tensor] = []
        current = queries
        current_reference = references
        # In "once" mode the positional signature is built from the initial
        # reference and reused unchanged by every layer, so the reference only
        # feeds the offset chain and never perturbs attention mid-stack. That is
        # the official MT3 arrangement. In "per_layer" mode it is rebuilt from the
        # refined reference, which only carries a learning signal when the
        # reference is not detached between layers.
        fixed_query_positions = (
            self.query_positions(current_reference)
            if self.query_position_mode == "once"
            else None
        )
        for layer, state_head, class_head in zip(self.layers, self.position_heads, self.class_heads):
            query_positions = (
                fixed_query_positions
                if self.query_position_mode != "per_layer"
                else self.query_positions(current_reference)
            )
            current, weights = layer(
                current,
                memory,
                memory_padding_mask,
                query_padding_mask,
                query_positions,
                memory_positions,
            )
            delta = state_head(current)
            state = torch.sigmoid(inverse_sigmoid(current_reference) + delta)
            states.append(state)
            logits.append(class_head(current))
            attention.append(weights)
            if self.iterative_refinement:
                current_reference = state.detach() if self.detach_reference else state
        return states, logits, current, torch.stack(attention)
