from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers: int, *, bias: bool = True):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        dimensions = [input_dim] + [hidden_dim] * (layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dimensions[index], dimensions[index + 1], bias=bias) for index in range(layers)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            value = layer(value)
            if index + 1 < len(self.layers):
                value = torch.relu(value)
        return value


def inverse_sigmoid(value: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return torch.log(value.clamp(min=epsilon) / (1.0 - value).clamp(min=epsilon))

