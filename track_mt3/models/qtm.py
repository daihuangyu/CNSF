from __future__ import annotations

import torch
from torch import nn

from track_mt3.config import ModelConfig
from track_mt3.tracking.track_state import TrackState

from .common import MLP
from .outputs import FramePrediction
from .track_mamba import TrackMamba


class TemporalFeatureNetwork(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = nn.MultiheadAttention(config.hidden_dim, config.num_heads, config.dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.linear1 = nn.Linear(config.hidden_dim, config.feedforward_dim)
        self.linear2 = nn.Linear(config.feedforward_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.norm2 = nn.LayerNorm(config.hidden_dim)

    def forward(self, hidden: torch.Tensor, previous_queries: torch.Tensor) -> torch.Tensor:
        if not len(hidden):
            return hidden
        query_key = hidden + previous_queries
        attended = self.attention(
            query_key.unsqueeze(0), query_key.unsqueeze(0), hidden.unsqueeze(0), need_weights=False
        )[0].squeeze(0)
        value = self.norm1(hidden + self.dropout(attended))
        feedforward = self.linear2(self.dropout(torch.relu(self.linear1(value))))
        return self.norm2(value + self.dropout(feedforward))


class QueryTransformationModule(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        source_dim = 3 if config.qtm_source == "prediction" else config.hidden_dim
        self.prediction_to_query = MLP(source_dim, config.hidden_dim, config.hidden_dim, 2, bias=False)
        self.temporal_network = TemporalFeatureNetwork(config) if not config.track_mamba_enabled else None
        self.track_mamba = TrackMamba(config) if config.track_mamba_enabled else None
        self.output_projection = MLP(config.hidden_dim, config.hidden_dim, config.hidden_dim, 2)

    def _source(self, prediction: FramePrediction) -> torch.Tensor:
        if self.config.qtm_source == "prediction":
            return torch.cat((prediction.normalized_positions, prediction.logits), dim=-1)
        if self.config.qtm_source == "decoder_hidden":
            return prediction.hidden
        raise ValueError(f"unknown qtm_source: {self.config.qtm_source}")

    def _track_mamba_step(self, hidden, previous, keep_tracking):
        """Run the per-track SSM over surviving tracks and return (enhanced, pool, slot_map)."""
        N = len(hidden)
        if N == 0:
            return hidden, previous.ssm_states, previous.ssm_slot_map

        pool = previous.ssm_states
        slot_map = dict(previous.ssm_slot_map) if previous.ssm_slot_map is not None else {}
        old_track_ids = previous.track_ids[keep_tracking].tolist()

        used = set(slot_map.values())
        slots = []
        for tid in old_track_ids:
            if tid >= 0 and tid in slot_map:
                slots.append(slot_map[tid])
            elif tid >= 0:
                free = [s for s in range(pool.shape[0]) if s not in used]
                slot = free[0] if free else 0
                slot_map[tid] = slot
                used.add(slot)
                pool[slot] = 0.0
                slots.append(slot)
            else:
                slots.append(-1)

        d_inner, d_state = self.track_mamba.ssm_shape
        prev_states = torch.zeros(N, d_inner, d_state, device=hidden.device, dtype=hidden.dtype)
        for i, s in enumerate(slots):
            if s >= 0:
                prev_states[i] = pool[s]

        enhanced, new_states = self.track_mamba(hidden, prev_states)

        for i, s in enumerate(slots):
            if s >= 0:
                pool[s] = new_states[i].detach()

        alive = {tid for tid in old_track_ids if tid >= 0}
        for tid in list(slot_map.keys()):
            if tid not in alive:
                slot = slot_map.pop(tid)
                pool[slot] = 0.0

        return enhanced, pool, slot_map

    def forward(
        self,
        prediction: FramePrediction,
        previous: TrackState,
        query_target_ids: torch.Tensor | None = None,
        query_track_ids: torch.Tensor | None = None,
        teacher_force: bool | None = None,
    ) -> tuple[TrackState, torch.Tensor]:
        """Transform current predictions into next-frame tracking queries.

        Returns the new state and selected indices into the current prediction.
        """
        num_tracks = prediction.num_track_queries
        transformed = self.prediction_to_query(self._source(prediction))
        probabilities = prediction.probabilities.squeeze(-1)
        # Track-before-detect: tentative tracks (age < confirm_frames) must
        # clear the strict tracking_threshold every frame to survive; confirmed
        # tracks (age >= confirm_frames) may coast at the lax confirmed_threshold.
        track_probs = probabilities[:num_tracks]
        if self.config.track_confirmation_enabled:
            is_confirmed = previous.ages >= self.config.confirm_frames
            track_thresholds = torch.where(
                is_confirmed,
                self.config.confirmed_threshold,
                self.config.tracking_threshold,
            )
        else:
            # Legacy Track-MT3 (including the published v9 baseline) used the
            # strict tracking threshold for every propagated query.  Keeping
            # this explicit avoids silently changing old checkpoints when the
            # later track-before-detect lifecycle defaults are loaded.
            track_thresholds = torch.full_like(
                track_probs, self.config.tracking_threshold
            )
        keep_tracking = track_probs > track_thresholds
        keep_detection = probabilities[num_tracks:] > self.config.detection_threshold
        use_teacher_forcing = (
            self.config.teacher_force_matched_queries
            if teacher_force is None
            else teacher_force
        )
        # Teacher forcing is controlled by the caller rather than by the module
        # training flag, so an evaluation pass can report a teacher-forced loss
        # that is directly comparable with the training objective. Inference
        # paths never supply target identities, which keeps them free-running.
        if use_teacher_forcing and query_target_ids is not None:
            matched = query_target_ids >= 0
            keep_tracking = keep_tracking | matched[:num_tracks]
            keep_detection = keep_detection | matched[num_tracks:]
        track_indices = torch.arange(num_tracks, device=probabilities.device)[keep_tracking]
        detection_indices = torch.arange(num_tracks, len(probabilities), device=probabilities.device)[keep_detection]

        old_hidden = transformed[:num_tracks][keep_tracking]
        ssm_states = getattr(previous, "ssm_states", None)
        ssm_slot_map = getattr(previous, "ssm_slot_map", None)
        if self.config.qtm_enabled:
            if self.track_mamba is not None:
                old_hidden, ssm_states, ssm_slot_map = self._track_mamba_step(
                    old_hidden, previous, keep_tracking
                )
            else:
                old_hidden = self.temporal_network(old_hidden, previous.features[keep_tracking])
        new_hidden = transformed[num_tracks:][keep_detection]
        selected = torch.cat((track_indices, detection_indices))
        combined = torch.cat((old_hidden, new_hidden), dim=0)
        # MPS cannot backpropagate through a Linear applied to a zero-row
        # tensor. An empty selected set also has no query to transform, so the
        # mathematically correct operation is to keep the empty feature tensor.
        if self.config.qtm_enabled and len(combined):
            combined = self.output_projection(combined)

        if query_target_ids is None:
            old_target_ids = previous.target_ids[keep_tracking]
            new_target_ids = torch.full((int(keep_detection.sum()),), -1, dtype=torch.long, device=probabilities.device)
            target_ids = torch.cat((old_target_ids, new_target_ids))
        else:
            target_ids = query_target_ids[selected]

        if query_track_ids is None:
            old_track_ids = previous.track_ids[keep_tracking]
            new_track_ids = torch.full((int(keep_detection.sum()),), -1, dtype=torch.long, device=probabilities.device)
            track_ids = torch.cat((old_track_ids, new_track_ids))
        else:
            track_ids = query_track_ids[selected]

        ages = torch.cat(
            (
                previous.ages[keep_tracking] + 1,
                torch.ones((int(keep_detection.sum()),), dtype=torch.long, device=probabilities.device),
            )
        )
        if len(selected) > self.config.max_tracks:
            selected_scores = probabilities[selected]
            top = torch.topk(selected_scores, self.config.max_tracks).indices.sort().values
            selected = selected[top]
            combined = combined[top]
            target_ids = target_ids[top]
            track_ids = track_ids[top]
            ages = ages[top]
        state = TrackState(
            features=combined,
            references=prediction.normalized_positions[selected].detach(),
            target_ids=target_ids,
            track_ids=track_ids,
            ages=ages,
            ssm_states=ssm_states,
            ssm_slot_map=ssm_slot_map,
            memory=getattr(previous, "memory", None),
            measurement_cache=getattr(previous, "measurement_cache", None),
            measurement_cache_mask=getattr(previous, "measurement_cache_mask", None),
            measurement_cache_positions=getattr(previous, "measurement_cache_positions", None),
            measurement_cache_ids=getattr(previous, "measurement_cache_ids", None),
        )
        state.validate()
        return state, selected
