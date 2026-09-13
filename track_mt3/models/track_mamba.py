from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from track_mt3.config import ModelConfig


class TrackMamba(nn.Module):
    """Per-track selective SSM for QTM state synchronisation.

    Track-MT3's QTM carries a track's query feature across frames, but its
    TemporalFeatureNetwork only blends the current prediction with the
    *immediately previous* feature (single-step attention). A track is a
    dynamical object (position + velocity + existence), so its state deserves
    a proper recurrent memory. This module runs a Mamba-1 style selective SSM
    per track:

        h_t = exp(dt * A) * h_{t-1} + dt * B * x_t
        y_t = C @ h_t + D * x_t

    where ``x_t`` is the current frame's track feature (after QTM's
    prediction_to_query) and ``h_t`` is a per-track hidden state that persists
    across frames. Coasting tracks update their state too, so a missed
    detection is extrapolated by the recurrence rather than freezing.

    The module is a pure single-step SSM: the caller (QTM) owns the per-track
    state pool and slot bookkeeping.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_state = config.track_mamba_state_dim
        self.d_inner = config.hidden_dim
        self.dt_min = 0.001
        self.dt_max = 0.1

        # in_proj emits [z, x, B, C, dt] in one shot.
        self.in_proj = nn.Linear(
            config.hidden_dim,
            self.d_inner * 2 + 2 * self.d_state + self.d_inner,
            bias=False,
        )
        self.A_log = nn.Parameter(torch.randn(self.d_state) * 0.5)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        # Initialise dt bias so softplus(bias) lands log-uniformly in
        # [dt_min, dt_max], following the Mamba-1 scheme.
        dt_init = torch.exp(
            torch.rand(self.d_inner) * (math.log(self.dt_max) - math.log(self.dt_min))
            + math.log(self.dt_min)
        )
        self.dt_bias = nn.Parameter(torch.log(torch.exp(dt_init) - 1.0))
        self.out_proj = nn.Linear(self.d_inner, config.hidden_dim, bias=False)
        self.norm = nn.RMSNorm(config.hidden_dim)

    @property
    def ssm_shape(self) -> tuple[int, int]:
        return (self.d_inner, self.d_state)

    def forward(self, x, ssm_state=None):
        """Single-step SSM over N tracks.

        Args:
            x: [N, D] track features.
            ssm_state: [N, d_inner, d_state] previous hidden states, or None.

        Returns:
            out: [N, D] enhanced track features.
            new_state: [N, d_inner, d_state] updated hidden states.
        """
        N, D = x.shape
        d_inner = self.d_inner
        d_state = self.d_state

        projected = self.in_proj(x)
        z = projected[:, :d_inner]
        x_ssm = projected[:, d_inner:2 * d_inner]
        B = projected[:, 2 * d_inner:2 * d_inner + d_state]
        C = projected[:, 2 * d_inner + d_state:2 * d_inner + 2 * d_state]
        dt = projected[:, 2 * d_inner + 2 * d_state:]
        dt = F.softplus(dt + self.dt_bias).clamp(min=self.dt_min, max=self.dt_max)

        A = -F.softplus(self.A_log)

        decay = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        input_term = dt.unsqueeze(-1) * B.unsqueeze(1) * x_ssm.unsqueeze(-1)

        if ssm_state is None or ssm_state.shape[0] != N:
            h = input_term
        else:
            h = decay * ssm_state + input_term

        y = (C.unsqueeze(1) * h).sum(-1)
        y = y + self.D.unsqueeze(0) * x_ssm
        out = self.out_proj(y * F.silu(z))
        out = self.norm(out + x)
        return out, h
