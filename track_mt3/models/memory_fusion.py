from __future__ import annotations

import torch
from torch import nn

from track_mt3.config import ModelConfig


class MemoryFusion(nn.Module):
    """DFSMN-style historical-measurement fusion for single-frame streaming.

    The encoder sees only the current frame's measurements (true incremental
    processing), but the current frame's encoding is fused with a compact
    memory of the *previous frames' measurement encodings* via cross-attention:

        fused = norm(current_memory + attn(current_memory, memory, memory))

    The memory is a fixed-size FIFO of representative measurement tokens from
    past frames. Each frame, the top-k most relevant current measurements
    (scored by a small learned gate) are pushed into the memory and the oldest
    tokens are dropped. This lets a detection query see each target's motion
    trail *through the fused current encoding* without re-feeding historical
    measurements to the encoder.

    This is the DFSMN idea (carry history forward in a memory block) applied to
    multi-target tracking, and it avoids the failure of v12 where the memory
    was handed to the decoder as a *parallel* second memory (which diluted the
    detection queries and blew up false alarms).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.memory_size = config.memory_frames * config.memory_tokens_per_frame
        self.tokens_per_frame = config.memory_tokens_per_frame

        self.attention = nn.MultiheadAttention(
            config.hidden_dim, config.num_heads, config.dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.memory_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

    def empty_memory(self, batch, device):
        return torch.zeros(batch, self.memory_size, self.hidden_dim, device=device)

    def forward(self, current_memory, memory, padding_mask=None):
        """Fuse the current frame's encoding with the historical memory.

        Args:
            current_memory: [B, S, D] encoder output of the current frame.
            memory: [B, K, D] historical measurement tokens (FIFO), or None.
            padding_mask: [B, S] True=padding.

        Returns:
            fused: [B, S, D] history-fused current encoding.
            new_memory: [B, K, D] updated FIFO memory.
        """
        B, S, D = current_memory.shape
        K = self.memory_size

        if memory is None or memory.shape[1] != K:
            memory = self.empty_memory(B, current_memory.device)

        has_history = bool((memory.abs().sum(dim=(1, 2)) > 0).any())
        if has_history:
            context, _ = self.attention(current_memory, memory, memory, need_weights=False)
            fused = self.norm(current_memory + context)
        else:
            fused = current_memory

        # Update memory: push top-k current tokens (FIFO, drop oldest).
        gate_scores = self.gate(current_memory).squeeze(-1)  # [B, S]
        if padding_mask is not None:
            gate_scores = gate_scores.masked_fill(padding_mask, -1e9)
        k = min(self.tokens_per_frame, S)
        if k > 0:
            top_values, top_indices = gate_scores.topk(k, dim=1)
            gathered = current_memory.gather(
                1, top_indices.unsqueeze(-1).expand(-1, -1, D)
            )
            weights = torch.sigmoid(top_values).unsqueeze(-1)
            new_tokens = self.memory_proj(gathered) * weights
            if k < K:
                retained = memory[:, k:]
            else:
                retained = memory[:, :0]
            new_memory = torch.cat([retained, new_tokens], dim=1)
        else:
            new_memory = memory

        return fused, new_memory
