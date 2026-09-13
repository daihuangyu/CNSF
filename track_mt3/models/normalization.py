from __future__ import annotations

import torch
from torch import nn


class CoordinateNormalizer(nn.Module):
    """Paper normalization: z/s and s*(sigmoid(y)-0.5)."""

    def __init__(self, field_of_view: tuple[float, float, float, float]):
        super().__init__()
        x_min, x_max, y_min, y_max = field_of_view
        spans = torch.tensor([x_max - x_min, y_max - y_min], dtype=torch.float32)
        centers = torch.tensor([(x_max + x_min) / 2.0, (y_max + y_min) / 2.0], dtype=torch.float32)
        self.register_buffer("spans", spans)
        self.register_buffer("centers", centers)

    def normalize_measurements(self, positions: torch.Tensor) -> torch.Tensor:
        return (positions - self.centers) / self.spans

    def to_unit_interval(self, positions: torch.Tensor) -> torch.Tensor:
        return self.normalize_measurements(positions) + 0.5

    def from_unit_interval(self, positions: torch.Tensor) -> torch.Tensor:
        return self.centers + self.spans * (positions - 0.5)

