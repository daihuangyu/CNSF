from __future__ import annotations

import math

import torch
from torch import nn


class TemporalEncoding(nn.Module):
    def __init__(self, window_size: int, hidden_dim: int, mode: str = "learned"):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.mode = mode
        if mode == "learned":
            self.embedding = nn.Embedding(window_size, hidden_dim)
        elif mode == "sinusoidal":
            self.embedding = None
            positions = torch.arange(window_size, dtype=torch.float32)[:, None]
            divisor = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float32) * (-math.log(10_000.0) / hidden_dim))
            encoding = torch.zeros(window_size, hidden_dim)
            encoding[:, 0::2] = torch.sin(positions * divisor)
            encoding[:, 1::2] = torch.cos(positions * divisor[: encoding[:, 1::2].shape[1]])
            self.register_buffer("encoding", encoding)
        else:
            raise ValueError(f"unknown temporal encoding: {mode}")

    def forward(self, time_indices: torch.Tensor) -> torch.Tensor:
        if torch.any((time_indices < 0) | (time_indices >= self.window_size)):
            raise ValueError("time index outside configured window")
        if self.embedding is not None:
            return self.embedding(time_indices)
        return self.encoding[time_indices]

