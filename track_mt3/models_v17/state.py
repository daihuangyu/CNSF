from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RecursiveTrackState:
    """Fixed-capacity differentiable state carried between physical frames."""

    mean: torch.Tensor
    covariance: torch.Tensor
    query: torch.Tensor
    evidence_hidden: torch.Tensor
    alternative_mean: torch.Tensor
    alternative_covariance: torch.Tensor
    alternative_query: torch.Tensor
    hypothesis_log_weights: torch.Tensor
    undetected_hidden: torch.Tensor
    undetected_log_mean: torch.Tensor
    undetected_initialized: torch.Tensor
    exist_logit: torch.Tensor
    active_mask: torch.Tensor
    confirmed_mask: torch.Tensor
    owner_ids: torch.Tensor
    supervision_ids: torch.Tensor
    ages: torch.Tensor
    hit_streak: torch.Tensor
    miss_streak: torch.Tensor
    last_update_time: torch.Tensor
    time_since_association: torch.Tensor
    oracle_dead_age: torch.Tensor
    next_runtime_id: torch.Tensor

    @classmethod
    def empty(
        cls,
        batch_size: int,
        slots: int,
        hidden_dim: int,
        device: torch.device | str,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> "RecursiveTrackState":
        return cls(
            mean=torch.zeros(batch_size, slots, 4, device=device, dtype=dtype),
            covariance=torch.eye(4, device=device, dtype=dtype)
            .view(1, 1, 4, 4)
            .repeat(batch_size, slots, 1, 1),
            query=torch.zeros(batch_size, slots, hidden_dim, device=device, dtype=dtype),
            evidence_hidden=torch.zeros(
                batch_size, slots, hidden_dim, device=device, dtype=dtype
            ),
            alternative_mean=torch.zeros(
                batch_size, slots, 4, device=device, dtype=dtype
            ),
            alternative_covariance=torch.eye(4, device=device, dtype=dtype)
            .view(1, 1, 4, 4)
            .repeat(batch_size, slots, 1, 1),
            alternative_query=torch.zeros(
                batch_size, slots, hidden_dim, device=device, dtype=dtype
            ),
            hypothesis_log_weights=torch.tensor(
                [0.0, -8.0], device=device, dtype=dtype
            ).view(1, 2).repeat(batch_size, 1),
            undetected_hidden=torch.zeros(
                batch_size, hidden_dim, device=device, dtype=dtype
            ),
            undetected_log_mean=torch.zeros(
                batch_size, device=device, dtype=dtype
            ),
            undetected_initialized=torch.zeros(
                batch_size, device=device, dtype=torch.bool
            ),
            exist_logit=torch.full(
                (batch_size, slots), -8.0, device=device, dtype=dtype
            ),
            active_mask=torch.zeros(batch_size, slots, device=device, dtype=torch.bool),
            confirmed_mask=torch.zeros(
                batch_size, slots, device=device, dtype=torch.bool
            ),
            owner_ids=torch.full(
                (batch_size, slots), -1, device=device, dtype=torch.long
            ),
            supervision_ids=torch.full(
                (batch_size, slots), -1, device=device, dtype=torch.long
            ),
            ages=torch.zeros(batch_size, slots, device=device, dtype=torch.long),
            hit_streak=torch.zeros(
                batch_size, slots, device=device, dtype=torch.long
            ),
            miss_streak=torch.zeros(batch_size, slots, device=device, dtype=torch.long),
            last_update_time=torch.zeros(batch_size, slots, device=device, dtype=dtype),
            time_since_association=torch.zeros(
                batch_size, slots, device=device, dtype=dtype
            ),
            oracle_dead_age=torch.zeros(
                batch_size, slots, device=device, dtype=torch.long
            ),
            next_runtime_id=torch.zeros(
                batch_size, device=device, dtype=torch.long
            ),
        )

    @property
    def batch_size(self) -> int:
        return self.mean.shape[0]

    @property
    def slots(self) -> int:
        return self.mean.shape[1]

    @property
    def device(self) -> torch.device:
        return self.mean.device

    def detach(self) -> "RecursiveTrackState":
        return RecursiveTrackState(
            **{
                name: value.detach()
                for name, value in self.__dict__.items()
            }
        )

    def clone(self) -> "RecursiveTrackState":
        return RecursiveTrackState(
            **{name: value.clone() for name, value in self.__dict__.items()}
        )

    def validate(self) -> None:
        batch, slots, state_dim = self.mean.shape
        if state_dim != 4:
            raise ValueError("mean must have shape [B,S,4]")
        expected = (batch, slots)
        if self.covariance.shape != (*expected, 4, 4):
            raise ValueError("covariance must have shape [B,S,4,4]")
        if self.query.shape[:2] != expected:
            raise ValueError("query must have shape [B,S,D]")
        if self.evidence_hidden.shape != self.query.shape:
            raise ValueError("evidence_hidden must have the same shape as query")
        if self.alternative_mean.shape != self.mean.shape:
            raise ValueError("alternative_mean must have the same shape as mean")
        if self.alternative_covariance.shape != self.covariance.shape:
            raise ValueError(
                "alternative_covariance must have the same shape as covariance"
            )
        if self.alternative_query.shape != self.query.shape:
            raise ValueError("alternative_query must have the same shape as query")
        if self.hypothesis_log_weights.shape != (batch, 2):
            raise ValueError("hypothesis_log_weights must have shape [B,2]")
        if self.undetected_hidden.shape != (batch, self.query.shape[-1]):
            raise ValueError("undetected_hidden must have shape [B,D]")
        if self.undetected_log_mean.shape != (batch,):
            raise ValueError("undetected_log_mean must have shape [B]")
        if self.undetected_initialized.shape != (batch,):
            raise ValueError("undetected_initialized must have shape [B]")
        for name in (
            "exist_logit",
            "active_mask",
            "confirmed_mask",
            "owner_ids",
            "supervision_ids",
            "ages",
            "hit_streak",
            "miss_streak",
            "last_update_time",
            "time_since_association",
            "oracle_dead_age",
        ):
            if getattr(self, name).shape != expected:
                raise ValueError(f"{name} must have shape [B,S]")
        if self.next_runtime_id.shape != (batch,):
            raise ValueError("next_runtime_id must have shape [B]")
        if torch.any(self.owner_ids[self.active_mask] < 0):
            raise ValueError("every active slot must have a non-negative owner id")
        if torch.any(self.confirmed_mask & ~self.active_mask):
            raise ValueError("only active slots may be confirmed")
        if (
            not torch.isfinite(self.mean).all()
            or not torch.isfinite(self.covariance).all()
            or not torch.isfinite(self.query).all()
            or not torch.isfinite(self.evidence_hidden).all()
            or not torch.isfinite(self.alternative_mean).all()
            or not torch.isfinite(self.alternative_covariance).all()
            or not torch.isfinite(self.alternative_query).all()
            or not torch.isfinite(self.hypothesis_log_weights).all()
            or not torch.isfinite(self.undetected_hidden).all()
            or not torch.isfinite(self.undetected_log_mean).all()
            or not torch.isfinite(self.exist_logit).all()
        ):
            raise ValueError("recursive state contains non-finite values")
