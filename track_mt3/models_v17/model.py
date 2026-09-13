from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from scipy.optimize import linear_sum_assignment

from .association import AssociationTransport, JointAssociation
from .config import V17AConfig, V17BConfig
from .dynamics import StructuredDynamics
from .measurement_encoder import CurrentFrameMeasurementEncoder
from .outputs import RecursiveFrameOutput
from .state import RecursiveTrackState
from .update import LearnedMeasurementUpdate


def _combine_birth_evidence(
    objectness_logits: torch.Tensor,
    unclaimed_probabilities: torch.Tensor,
    *,
    pmbm_odds_fusion: bool,
    unclaimed_log_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse target-vs-clutter evidence with measurement availability.

    The legacy v20 path multiplies two probabilities. The opt-in v21 path
    follows the PMBM intensity ratio: if objectness odds approximate e/kappa,
    scaling target intensity e by availability u gives log-odds
    log(e/kappa) + log(u). This remains fully neural because e/kappa comes from
    the learned current-frame objectness head and u from learned transport.
    """

    unclaimed = unclaimed_probabilities.clamp(1.0e-6, 1.0)
    if pmbm_odds_fusion:
        logits = objectness_logits + unclaimed_log_scale * torch.log(unclaimed)
        probabilities = logits.sigmoid()
    else:
        probabilities = (
            objectness_logits.sigmoid() * unclaimed
        ).clamp(1.0e-6, 1.0 - 1.0e-6)
        logits = torch.logit(probabilities)
    return probabilities, logits


class OracleRecursiveTracker(nn.Module):
    """v17-A upper bound with oracle association and slot lifecycle.

    Truth identities are used only by the non-learned state bookkeeping in
    :meth:`forward_step`.  They are never embedded or passed to a neural layer.
    This is an intentional diagnostic model: it isolates whether a per-track
    recursive posterior can replace the raw historical measurement window.
    """

    def __init__(self, config: V17AConfig = V17AConfig()):
        super().__init__()
        self.config = config
        self.encoder = CurrentFrameMeasurementEncoder(config)
        self.dynamics = StructuredDynamics(config)
        self.update = LearnedMeasurementUpdate(config)
        self.existence_head = nn.Sequential(
            nn.Linear(config.hidden_dim + 3, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.birth_position_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 2),
        )
        self.birth_velocity_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 2),
        )
        self.birth_variance_head = nn.Linear(config.hidden_dim, 4)
        self.birth_query_head = nn.Sequential(
            nn.Linear(config.hidden_dim + 2, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )
        self.birth_existence_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        if getattr(config, "birth_neural_ppp", False):
            self.ppp_initial_hidden = nn.Parameter(
                torch.zeros(config.hidden_dim)
            )
            self.ppp_initial_log_mean = nn.Parameter(
                torch.tensor(float(config.birth_ppp_initial_mean)).log()
            )
            self.ppp_survival_logit = nn.Parameter(
                torch.logit(torch.tensor(config.birth_ppp_survival_probability))
            )
            self.ppp_detection_logit = nn.Parameter(
                torch.logit(torch.tensor(config.birth_ppp_detection_probability))
            )
            self.ppp_new_log_mean = nn.Parameter(
                torch.tensor(max(config.birth_ppp_new_mean, 1.0e-6)).log()
            )
            self.ppp_target_intensity_head = nn.Sequential(
                nn.Linear(2 * config.hidden_dim, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, 1),
            )
            self.ppp_clutter_intensity_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, 1),
            )
            self.ppp_recurrent_input = nn.Sequential(
                nn.Linear(config.hidden_dim + 3, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            self.ppp_recurrent_cell = nn.GRUCell(
                config.hidden_dim, config.hidden_dim
            )
            nn.init.normal_(self.ppp_target_intensity_head[-1].weight, std=1.0e-3)
            nn.init.zeros_(self.ppp_target_intensity_head[-1].bias)
            nn.init.normal_(self.ppp_clutter_intensity_head[-1].weight, std=1.0e-3)
            nn.init.zeros_(self.ppp_clutter_intensity_head[-1].bias)
        nn.init.zeros_(self.birth_position_head[-1].weight)
        nn.init.zeros_(self.birth_position_head[-1].bias)
        nn.init.zeros_(self.birth_velocity_head[-1].weight)
        nn.init.zeros_(self.birth_velocity_head[-1].bias)
        nn.init.zeros_(self.birth_variance_head.weight)
        initial_variance = torch.tensor(
            [
                config.initial_position_std**2,
                config.initial_position_std**2,
                config.initial_velocity_std**2,
                config.initial_velocity_std**2,
            ]
        )
        with torch.no_grad():
            self.birth_variance_head.bias.copy_(
                torch.log(torch.expm1(initial_variance.clamp_min(1.0e-6)))
            )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def empty_state(self, batch_size: int) -> RecursiveTrackState:
        return RecursiveTrackState.empty(
            batch_size,
            self.config.max_tracks,
            self.config.hidden_dim,
            self.device,
        )

    def _predict_neural_ppp(
        self,
        state: RecursiveTrackState,
        embeddings: torch.Tensor,
        measurement_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate learned target/clutter intensities at current measurements.

        The recurrent scalar is the expected undetected-target mass.  The
        hidden vector carries its learned spatial/contextual sufficient
        statistic.  Both are strictly causal: the current-frame embeddings are
        used only to evaluate the predicted intensity and to form the posterior
        state written for the next frame.
        """

        initialized = state.undetected_initialized
        initial_hidden = self.ppp_initial_hidden.unsqueeze(0).expand(
            state.batch_size, -1
        )
        prior_hidden = torch.where(
            initialized.unsqueeze(-1), state.undetected_hidden, initial_hidden
        )
        previous_mean = torch.where(
            initialized,
            state.undetected_log_mean.exp(),
            self.ppp_initial_log_mean.exp().expand(state.batch_size),
        )
        predicted_mean = torch.where(
            initialized,
            self.ppp_survival_logit.sigmoid() * previous_mean
            + self.ppp_new_log_mean.exp(),
            previous_mean,
        ).clamp(1.0e-5, float(self.config.max_tracks))
        context = prior_hidden[:, None].expand(-1, embeddings.shape[1], -1)
        valid_count = (~measurement_padding_mask).sum(-1).clamp_min(1).to(
            embeddings.dtype
        )
        target_log_intensity = (
            self.ppp_target_intensity_head(
                torch.cat((embeddings, context), dim=-1)
            ).squeeze(-1)
            + predicted_mean.log().unsqueeze(-1)
            - valid_count.log().unsqueeze(-1)
        )
        clutter_log_intensity = self.ppp_clutter_intensity_head(
            embeddings
        ).squeeze(-1)
        target_log_intensity = target_log_intensity.masked_fill(
            measurement_padding_mask, -20.0
        )
        clutter_log_intensity = clutter_log_intensity.masked_fill(
            measurement_padding_mask, -20.0
        )
        return (
            target_log_intensity - clutter_log_intensity,
            target_log_intensity,
            clutter_log_intensity,
            predicted_mean,
        )

    def _update_neural_ppp_state(
        self,
        state: RecursiveTrackState,
        embeddings: torch.Tensor,
        measurement_padding_mask: torch.Tensor,
        birth_probabilities: torch.Tensor,
        predicted_mean: torch.Tensor,
    ) -> None:
        valid = ~measurement_padding_mask
        weights = birth_probabilities.masked_fill(~valid, 0.0)
        pooled = (
            weights.unsqueeze(-1) * embeddings
        ).sum(1) / weights.sum(-1, keepdim=True).clamp_min(1.0)
        scalars = torch.stack(
            (
                valid.sum(-1).to(embeddings.dtype) / 32.0,
                weights.sum(-1) / 16.0,
                state.active_mask.sum(-1).to(embeddings.dtype) / 32.0,
            ),
            dim=-1,
        )
        recurrent_input = self.ppp_recurrent_input(
            torch.cat((pooled, scalars), dim=-1)
        )
        previous_hidden = torch.where(
            state.undetected_initialized.unsqueeze(-1),
            state.undetected_hidden,
            self.ppp_initial_hidden.unsqueeze(0).expand(state.batch_size, -1),
        )
        state.undetected_hidden = self.ppp_recurrent_cell(
            recurrent_input, previous_hidden
        )
        posterior_mean = (
            predicted_mean * (1.0 - self.ppp_detection_logit.sigmoid())
        ).clamp_min(1.0e-5)
        state.undetected_log_mean = posterior_mean.log()
        state.undetected_initialized = torch.ones_like(
            state.undetected_initialized
        )

    def forward(
        self,
        measurements: torch.Tensor,
        measurement_padding_mask: torch.Tensor,
        measurement_ids: torch.Tensor,
        frame_time: torch.Tensor,
        truth_target_ids: Sequence[torch.Tensor],
        previous_state: RecursiveTrackState | None = None,
        association_oracle_probability: float = 1.0,
        association_update_mode: str = "soft",
        birth_state_mode: str = "oracle",
        birth_candidate_threshold: float = 0.5,
        confirmation_hits: int = 2,
        death_state_mode: str = "oracle_pre",
        retention_threshold: float = 0.5,
        oracle_prune_delay: int = 1,
        survival_warmup_frames: int = 0,
        confirmed_association_bias: float = 0.0,
        precomputed_measurement_embeddings: torch.Tensor | None = None,
    ) -> tuple[RecursiveFrameOutput, RecursiveTrackState]:
        """DDP-compatible alias for the causal single-frame step."""

        return self.forward_step(
            measurements,
            measurement_padding_mask,
            measurement_ids,
            frame_time,
            truth_target_ids,
            previous_state,
            association_oracle_probability,
            association_update_mode,
            birth_state_mode,
            birth_candidate_threshold,
            confirmation_hits,
            death_state_mode,
            retention_threshold,
            oracle_prune_delay,
            survival_warmup_frames,
            confirmed_association_bias,
            precomputed_measurement_embeddings,
        )

    def encode_current_frame(
        self,
        measurements: torch.Tensor,
        measurement_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode one physical frame for reuse by alternative state chains."""

        if measurements.ndim != 3 or measurements.shape[-1] != 2:
            raise ValueError("measurements must have shape [B,M,2]")
        batch, count, _ = measurements.shape
        if measurement_padding_mask.shape != (batch, count):
            raise ValueError("measurement_padding_mask must have shape [B,M]")
        return self.encoder(
            measurements.to(self.device),
            measurement_padding_mask.to(self.device),
        )

    def forward_inference_step(
        self,
        measurements: torch.Tensor,
        measurement_padding_mask: torch.Tensor,
        frame_time: torch.Tensor,
        previous_state: RecursiveTrackState | None = None,
        *,
        birth_candidate_threshold: float = 0.15,
        confirmation_hits: int = 3,
        retention_threshold: float = 0.175,
        survival_warmup_frames: int = 3,
        fast_inference: bool = False,
    ) -> tuple[RecursiveFrameOutput, RecursiveTrackState]:
        """Run one deployable frame without simulator IDs or truth inputs."""

        if measurements.ndim != 3:
            raise ValueError("measurements must have shape [B,M,2]")
        batch, count, _ = measurements.shape
        sentinel_ids = torch.full(
            (batch, count),
            -1,
            dtype=torch.long,
            device=measurements.device,
        )
        empty_truth_ids = [
            torch.empty(0, dtype=torch.long, device=measurements.device)
            for _ in range(batch)
        ]
        return self.forward_step(
            measurements,
            measurement_padding_mask,
            sentinel_ids,
            frame_time,
            empty_truth_ids,
            previous_state,
            association_oracle_probability=0.0,
            association_update_mode="moment",
            birth_state_mode="predicted",
            birth_candidate_threshold=birth_candidate_threshold,
            confirmation_hits=confirmation_hits,
            death_state_mode="predicted",
            retention_threshold=retention_threshold,
            survival_warmup_frames=survival_warmup_frames,
            confirmed_association_bias=0.0,
            fast_inference=fast_inference,
        )

    def forward_inference_step_fast(
        self,
        measurements: torch.Tensor,
        measurement_padding_mask: torch.Tensor,
        frame_time: torch.Tensor,
        previous_state: RecursiveTrackState | None = None,
        *,
        birth_candidate_threshold: float = 0.15,
        confirmation_hits: int = 3,
        retention_threshold: float = 0.175,
        survival_warmup_frames: int = 3,
    ) -> tuple[RecursiveFrameOutput, RecursiveTrackState]:
        """Deployable inference without Python/device synchronization points.

        Training and the historical inference entry point retain all runtime
        validation and scalar slot-allocation logic.  This opt-in path uses
        the same learned tensors and thresholds but vectorizes predicted birth
        admission and omits redundant validation of an internally owned state.
        """

        return self.forward_inference_step(
            measurements,
            measurement_padding_mask,
            frame_time,
            previous_state,
            birth_candidate_threshold=birth_candidate_threshold,
            confirmation_hits=confirmation_hits,
            retention_threshold=retention_threshold,
            survival_warmup_frames=survival_warmup_frames,
            fast_inference=True,
        )

    @staticmethod
    def _truth_sets(
        truth_target_ids: Sequence[torch.Tensor], device: torch.device
    ) -> list[set[int]]:
        return [
            {int(value) for value in ids.to(device="cpu").tolist() if int(value) >= 0}
            for ids in truth_target_ids
        ]

    def _prune_dead(
        self,
        state: RecursiveTrackState,
        truth_sets: list[set[int]],
    ) -> RecursiveTrackState:
        return self._advance_oracle_death(state, truth_sets, prune_after=1)

    def _advance_oracle_death(
        self,
        state: RecursiveTrackState,
        truth_sets: list[set[int]],
        *,
        prune_after: int,
    ) -> RecursiveTrackState:
        if prune_after < 1:
            raise ValueError("prune_after must be positive")
        state = state.clone()
        for batch_index, alive_ids in enumerate(truth_sets):
            seen_supervision_ids: set[int] = set()
            active_slots = torch.nonzero(
                state.active_mask[batch_index], as_tuple=False
            ).flatten()
            for slot in active_slots.tolist():
                supervision_id = int(state.supervision_ids[batch_index, slot])
                invalid = (
                    supervision_id not in alive_ids
                    or supervision_id in seen_supervision_ids
                )
                if invalid:
                    state.oracle_dead_age[batch_index, slot] += 1
                else:
                    state.oracle_dead_age[batch_index, slot] = 0
                if invalid and state.oracle_dead_age[batch_index, slot] >= prune_after:
                    state.active_mask[batch_index, slot] = False
                    state.confirmed_mask[batch_index, slot] = False
                    state.owner_ids[batch_index, slot] = -1
                    state.supervision_ids[batch_index, slot] = -1
                    state.exist_logit[batch_index, slot] = -8.0
                    state.evidence_hidden[batch_index, slot] = 0.0
                    state.ages[batch_index, slot] = 0
                    state.hit_streak[batch_index, slot] = 0
                    state.miss_streak[batch_index, slot] = 0
                    state.time_since_association[batch_index, slot] = 0.0
                    state.oracle_dead_age[batch_index, slot] = 0
                else:
                    if not invalid:
                        seen_supervision_ids.add(supervision_id)
        return state

    def _predict_survival(
        self,
        lifecycle_features: torch.Tensor,
        existing_logits: torch.Tensor,
    ) -> torch.Tensor:
        return existing_logits

    def _predict_track_measurement_bias(
        self,
        state: RecursiveTrackState,
        delta_t: torch.Tensor,
        confirmed_association_bias: float,
        survival_prior_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the per-track log-prior applied only to measurement edges."""

        del delta_t, survival_prior_logits
        return confirmed_association_bias * state.confirmed_mask.to(state.mean.dtype)

    def _next_confirmation_mask(
        self,
        state: RecursiveTrackState,
        existing_mask: torch.Tensor,
        existing_logits: torch.Tensor,
        confirmation_hits: int,
    ) -> torch.Tensor:
        """Update admission state without conflating it with existence."""

        if getattr(self.config, "track_evidence_recurrence", False):
            admitted = (
                existing_logits.sigmoid()
                >= self.config.track_evidence_confirmation_threshold
            )
            if self.config.track_evidence_persistent_confirmation:
                admitted = state.confirmed_mask | admitted
        else:
            admitted = state.confirmed_mask | (
                state.hit_streak >= confirmation_hits
            )
        return torch.where(existing_mask, admitted, state.confirmed_mask)

    def _association_write_pairs(
        self,
        state: RecursiveTrackState,
        prior_query: torch.Tensor,
        prior_covariance: torch.Tensor,
        pair_probabilities: torch.Tensor,
        measurement_padding_mask: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        """Return probabilities used to write the recursive state.

        Association supervision and lifecycle evidence always consume the
        original exclusive transport.  Historical configurations also write
        that transport unchanged; v19-B overrides only this state-write hook.
        """

        del state, prior_query, prior_covariance, measurement_padding_mask, delta_t
        return pair_probabilities

    def _predict_survival_prior(
        self,
        state: RecursiveTrackState,
        prior_query: torch.Tensor,
        prior_covariance: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return a causal Bernoulli survival prior for joint transport."""

        del state, prior_query, prior_covariance, delta_t
        return None

    def _update_track_evidence(
        self,
        state: RecursiveTrackState,
        posterior_query: torch.Tensor,
        instantaneous_logits: torch.Tensor,
        association_strength: torch.Tensor,
        selected_pair: torch.Tensor,
        elapsed: torch.Tensor,
        posterior_covariance: torch.Tensor,
        existing_mask: torch.Tensor,
        survival_prior_logits: torch.Tensor | None,
        measurement_objectness_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return deployable existence evidence and its recurrent state.

        Legacy models have no evidence recurrence.  The hook keeps their
        numerical path and checkpoints unchanged while v18-B overrides it.
        """

        del (
            posterior_query,
            association_strength,
            selected_pair,
            elapsed,
            posterior_covariance,
            existing_mask,
            survival_prior_logits,
            measurement_objectness_logits,
        )
        return instantaneous_logits, state.evidence_hidden

    def _predict_track_survival_prior(
        self,
        state: RecursiveTrackState,
        prior_query: torch.Tensor,
        prior_covariance: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return the pre-measurement survival logit used by v18-C."""

        del state, prior_query, prior_covariance, delta_t
        return None

    def _initialize_birth_evidence(
        self,
        birth_query: torch.Tensor,
        birth_logit: torch.Tensor,
    ) -> torch.Tensor:
        del birth_logit
        return torch.zeros_like(birth_query)

    def _predict_association(
        self,
        prior_mean: torch.Tensor,
        prior_covariance: torch.Tensor,
        prior_query: torch.Tensor,
        measurements: torch.Tensor,
        embeddings: torch.Tensor,
        existing_mask: torch.Tensor,
        measurement_padding_mask: torch.Tensor,
        delta_t_association: torch.Tensor,
        track_measurement_bias: torch.Tensor,
        survival_prior_logits: torch.Tensor | None,
        pair_measurement_variance: torch.Tensor | None,
    ) -> AssociationTransport | None:
        del survival_prior_logits, pair_measurement_variance
        return None

    @staticmethod
    def _oracle_pairs(
        supervision_ids: torch.Tensor,
        measurement_ids: torch.Tensor,
        active_mask: torch.Tensor,
        measurement_padding_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Match each active slot to its first same-ID valid measurement."""

        matches = (
            active_mask.unsqueeze(-1)
            & (~measurement_padding_mask).unsqueeze(1)
            & (supervision_ids.unsqueeze(-1) == measurement_ids.unsqueeze(1))
        )
        associated = matches.any(dim=-1)
        first_index = matches.to(torch.int64).argmax(dim=-1, keepdim=True)
        pair = torch.zeros(
            matches.shape,
            device=matches.device,
            dtype=dtype,
        ).scatter_(-1, first_index, associated.to(dtype).unsqueeze(-1))
        claimed_measurements = pair.to(torch.bool).any(dim=1)
        return pair, associated, claimed_measurements

    @staticmethod
    def _hard_association_pairs(
        pair_probabilities: torch.Tensor,
        miss_probabilities: torch.Tensor,
        active_mask: torch.Tensor,
        measurement_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode a one-to-one pair assignment with a private MISS per track."""

        hard = torch.zeros_like(pair_probabilities)
        for batch_index in range(pair_probabilities.shape[0]):
            slots = torch.nonzero(active_mask[batch_index], as_tuple=False).flatten()
            measurements = torch.nonzero(
                ~measurement_padding_mask[batch_index], as_tuple=False
            ).flatten()
            track_count = len(slots)
            measurement_count = len(measurements)
            if not track_count:
                continue
            costs = pair_probabilities.new_full(
                (track_count, measurement_count + track_count), 1.0e4
            )
            if measurement_count:
                pair = pair_probabilities[
                    batch_index, slots[:, None], measurements[None, :]
                ]
                costs[:, :measurement_count] = -pair.clamp_min(1.0e-12).log()
            for local_index, slot in enumerate(slots.tolist()):
                costs[local_index, measurement_count + local_index] = (
                    -miss_probabilities[batch_index, slot].clamp_min(1.0e-12).log()
                )
            rows, columns = linear_sum_assignment(costs.detach().cpu().numpy())
            for row, column in zip(rows.tolist(), columns.tolist()):
                if column < measurement_count:
                    hard[
                        batch_index,
                        int(slots[row]),
                        int(measurements[column]),
                    ] = 1.0
        return hard

    def forward_step(
        self,
        measurements: torch.Tensor,
        measurement_padding_mask: torch.Tensor,
        measurement_ids: torch.Tensor,
        frame_time: torch.Tensor,
        truth_target_ids: Sequence[torch.Tensor],
        previous_state: RecursiveTrackState | None = None,
        association_oracle_probability: float = 1.0,
        association_update_mode: str = "soft",
        birth_state_mode: str = "oracle",
        birth_candidate_threshold: float = 0.5,
        confirmation_hits: int = 2,
        death_state_mode: str = "oracle_pre",
        retention_threshold: float = 0.5,
        oracle_prune_delay: int = 1,
        survival_warmup_frames: int = 0,
        confirmed_association_bias: float = 0.0,
        precomputed_measurement_embeddings: torch.Tensor | None = None,
        fast_inference: bool = False,
    ) -> tuple[RecursiveFrameOutput, RecursiveTrackState]:
        """Process exactly one physical frame.

        ``measurement_ids`` and ``truth_target_ids`` are oracle bookkeeping for
        v17-A only.  No learned tensor is indexed by an ID value; IDs merely
        decide which current measurement event is written to which slot and
        which slots remain alive.
        """

        if measurements.ndim != 3 or measurements.shape[-1] != 2:
            raise ValueError("measurements must have shape [B,M,2]")
        batch, count, _ = measurements.shape
        if measurement_padding_mask.shape != (batch, count):
            raise ValueError("measurement_padding_mask must have shape [B,M]")
        if measurement_ids.shape != (batch, count):
            raise ValueError("measurement_ids must have shape [B,M]")
        if frame_time.shape != (batch,):
            raise ValueError("frame_time must have shape [B]")
        if len(truth_target_ids) != batch:
            raise ValueError("one truth-target-id tensor is required per batch row")

        measurements = measurements.to(self.device)
        measurement_padding_mask = measurement_padding_mask.to(self.device)
        measurement_ids = measurement_ids.to(self.device)
        frame_time = frame_time.to(self.device)
        state = previous_state or self.empty_state(batch)
        if not fast_inference:
            state.validate()
        if state.batch_size != batch:
            raise ValueError("state batch size does not match the input")

        if precomputed_measurement_embeddings is None:
            embeddings = self.encoder(measurements, measurement_padding_mask)
        else:
            expected = (batch, count, self.config.hidden_dim)
            if precomputed_measurement_embeddings.shape != expected:
                raise ValueError(
                    "precomputed_measurement_embeddings must have shape " f"{expected}"
                )
            embeddings = precomputed_measurement_embeddings.to(self.device)
        truth_sets = (
            [set() for _ in range(batch)]
            if fast_inference
            else self._truth_sets(truth_target_ids, self.device)
        )
        if death_state_mode not in {
            "oracle_pre",
            "oracle_post",
            "oracle_delayed",
            "predicted",
        }:
            raise ValueError(
                "death_state_mode must be oracle_pre/oracle_post/oracle_delayed/predicted"
            )
        if not 0.0 <= retention_threshold <= 1.0:
            raise ValueError("retention_threshold must be in [0,1]")
        if death_state_mode == "oracle_pre":
            state = self._prune_dead(state, truth_sets)
        if oracle_prune_delay < 1:
            raise ValueError("oracle_prune_delay must be positive")
        if survival_warmup_frames < 0:
            raise ValueError("survival_warmup_frames must be non-negative")
        if confirmed_association_bias < 0.0:
            raise ValueError("confirmed_association_bias must be non-negative")
        existing_mask = state.active_mask.clone()
        existing_owner_ids = state.owner_ids.clone()

        delta_t = (frame_time[:, None] - state.last_update_time).clamp_min(0.0)
        prior_mean, prior_covariance, prior_query = self.dynamics(
            state.mean, state.covariance, state.query, delta_t
        )
        # Inactive slots have no physical prior; retaining their zero state keeps
        # diagnostics finite and prevents their time embedding from leaking into
        # future births.
        prior_mean = torch.where(existing_mask.unsqueeze(-1), prior_mean, state.mean)
        prior_covariance = torch.where(
            existing_mask.unsqueeze(-1).unsqueeze(-1),
            prior_covariance,
            state.covariance,
        )
        prior_query = torch.where(existing_mask.unsqueeze(-1), prior_query, state.query)
        alternative_prior_mean = prior_mean
        alternative_prior_covariance = prior_covariance
        alternative_prior_query = prior_query
        if getattr(self.config, "association_mbm_hypotheses", 1) == 2:
            (
                alternative_prior_mean,
                alternative_prior_covariance,
                alternative_prior_query,
            ) = self.dynamics(
                state.alternative_mean,
                state.alternative_covariance,
                state.alternative_query,
                delta_t,
            )
            alternative_prior_mean = torch.where(
                existing_mask.unsqueeze(-1),
                alternative_prior_mean,
                state.alternative_mean,
            )
            alternative_prior_covariance = torch.where(
                existing_mask.unsqueeze(-1).unsqueeze(-1),
                alternative_prior_covariance,
                state.alternative_covariance,
            )
            alternative_prior_query = torch.where(
                existing_mask.unsqueeze(-1),
                alternative_prior_query,
                state.alternative_query,
            )

        if not 0.0 <= association_oracle_probability <= 1.0:
            raise ValueError("association_oracle_probability must be in [0,1]")
        if association_update_mode not in {"soft", "hard", "moment"}:
            raise ValueError(
                "association_update_mode must be 'soft', 'hard', or 'moment'"
            )
        if birth_state_mode not in {"oracle", "predicted"}:
            raise ValueError("birth_state_mode must be 'oracle' or 'predicted'")
        if not 0.0 <= birth_candidate_threshold <= 1.0:
            raise ValueError("birth_candidate_threshold must be in [0,1]")
        if confirmation_hits < 1:
            raise ValueError("confirmation_hits must be positive")
        if fast_inference:
            oracle_pair = measurements.new_zeros(
                batch, self.config.max_tracks, measurements.shape[1]
            )
            associated = torch.zeros_like(existing_mask)
            claimed_measurements = torch.zeros_like(measurement_padding_mask)
        else:
            oracle_pair, associated, claimed_measurements = self._oracle_pairs(
                state.supervision_ids,
                measurement_ids,
                existing_mask,
                measurement_padding_mask,
                measurements.dtype,
            )
        associated_measurements = torch.einsum(
            "bsm,bmd->bsd", oracle_pair, measurements
        )
        associated_embeddings = torch.einsum("bsm,bmd->bsd", oracle_pair, embeddings)

        survival_prior_logits = self._predict_survival_prior(
            state,
            prior_query,
            prior_covariance,
            delta_t,
        )
        track_survival_prior_logits = self._predict_track_survival_prior(
            state,
            prior_query,
            prior_covariance,
            delta_t,
        )
        track_measurement_bias = self._predict_track_measurement_bias(
            state,
            delta_t,
            confirmed_association_bias,
            track_survival_prior_logits,
        )
        pair_measurement_variance = None
        pair_update_candidates = None
        if getattr(self.config, "association_shared_observation_noise", False):
            pair_update_candidates = self.update.pair_update_candidates(
                prior_mean,
                prior_covariance,
                prior_query,
                measurements,
                embeddings,
                state.time_since_association + delta_t,
            )
            pair_measurement_variance = pair_update_candidates.measurement_variance
        transport = self._predict_association(
            prior_mean,
            prior_covariance,
            prior_query,
            measurements,
            embeddings,
            existing_mask,
            measurement_padding_mask,
            state.time_since_association + delta_t,
            track_measurement_bias,
            survival_prior_logits,
            pair_measurement_variance,
        )
        alternative_transport = None
        alternative_pair_update_candidates = None
        alternative_pair_measurement_variance = None
        if (
            transport is not None
            and getattr(self.config, "association_mbm_hypotheses", 1) == 2
        ):
            if getattr(self.config, "association_shared_observation_noise", False):
                alternative_pair_update_candidates = (
                    self.update.pair_update_candidates(
                        alternative_prior_mean,
                        alternative_prior_covariance,
                        alternative_prior_query,
                        measurements,
                        embeddings,
                        state.time_since_association + delta_t,
                    )
                )
                alternative_pair_measurement_variance = (
                    alternative_pair_update_candidates.measurement_variance
                )
            alternative_transport = self._predict_association(
                alternative_prior_mean,
                alternative_prior_covariance,
                alternative_prior_query,
                measurements,
                embeddings,
                existing_mask,
                measurement_padding_mask,
                state.time_since_association + delta_t,
                track_measurement_bias,
                survival_prior_logits,
                alternative_pair_measurement_variance,
            )
        association_strength = associated.to(prior_mean.dtype)
        selected_pair = oracle_pair
        selected_unclaimed_probability = None
        selected_death_probability = None
        oracle_association_used = None
        if transport is not None:
            selected_unclaimed_probability = transport.unclaimed_probabilities
            if association_oracle_probability == 1.0:
                oracle_association_used = torch.ones(
                    batch, dtype=torch.bool, device=self.device
                )
            elif association_oracle_probability == 0.0:
                oracle_association_used = torch.zeros(
                    batch, dtype=torch.bool, device=self.device
                )
            else:
                oracle_association_used = (
                    torch.rand(batch, device=self.device)
                    < association_oracle_probability
                )
            predicted_pair = transport.pair_probabilities
            if association_update_mode == "hard":
                predicted_pair = self._hard_association_pairs(
                    transport.pair_probabilities,
                    transport.miss_probabilities,
                    existing_mask,
                    measurement_padding_mask,
                )
            selected_pair = torch.where(
                oracle_association_used[:, None, None],
                oracle_pair,
                predicted_pair,
            )
            if transport.death_probabilities is not None and not fast_inference:
                padded_truth_ids = pad_sequence(
                    [identifiers.to(self.device) for identifiers in truth_target_ids],
                    batch_first=True,
                    padding_value=-2,
                )
                oracle_alive = (
                    state.supervision_ids.unsqueeze(-1) == padded_truth_ids.unsqueeze(1)
                ).any(-1)
                oracle_dead = (existing_mask & ~oracle_alive).to(
                    transport.death_probabilities.dtype
                )
                selected_death_probability = torch.where(
                    oracle_association_used[:, None],
                    oracle_dead,
                    transport.death_probabilities,
                )
            elif transport.death_probabilities is not None:
                selected_death_probability = transport.death_probabilities
            association_strength = selected_pair.sum(-1).clamp(0.0, 1.0)
            conditional_pair = selected_pair / association_strength.unsqueeze(
                -1
            ).clamp_min(1.0e-8)
            associated_measurements = torch.einsum(
                "bsm,bmd->bsd", conditional_pair, measurements
            )
            associated_embeddings = torch.einsum(
                "bsm,bmd->bsd", conditional_pair, embeddings
            )
        associated_hard = association_strength >= 0.5
        write_pair = self._association_write_pairs(
            state,
            prior_query,
            prior_covariance,
            selected_pair,
            measurement_padding_mask,
            state.time_since_association + delta_t,
        )

        if transport is not None and association_update_mode == "moment":
            posterior_mean, posterior_covariance, posterior_query = (
                self.update.forward_mixture(
                    prior_mean,
                    prior_covariance,
                    prior_query,
                    measurements,
                    embeddings,
                    state.time_since_association + delta_t,
                    write_pair,
                    pair_measurement_variance,
                    pair_update_candidates,
                )
            )
        else:
            posterior_mean, posterior_covariance, posterior_query = self.update(
                prior_mean,
                prior_covariance,
                prior_query,
                associated_measurements,
                associated_embeddings,
                state.time_since_association + delta_t,
                association_strength,
            )
        alternative_posterior_mean = posterior_mean
        alternative_posterior_covariance = posterior_covariance
        alternative_posterior_query = posterior_query
        updated_hypothesis_log_weights = state.hypothesis_log_weights
        if alternative_transport is not None and transport is not None:
            pair_mass = transport.pair_probabilities.sum(-1)
            conditional = transport.pair_probabilities / pair_mass.unsqueeze(
                -1
            ).clamp_min(1.0e-8)
            entropy = -(
                conditional * conditional.clamp_min(1.0e-8).log()
            ).sum(-1)
            entropy = torch.where(
                existing_mask & (pair_mass > 0.0),
                entropy,
                torch.zeros_like(entropy),
            )
            ambiguous = entropy.amax(-1) >= getattr(
                self.config, "association_mbm_entropy_threshold", 0.35
            )
            secondary_active = (
                state.hypothesis_log_weights.softmax(-1)[:, 1] > 1.0e-3
            )
            spawn_secondary = ambiguous & ~secondary_active
            seed_pair = transport.alternative_pair_probabilities
            assert seed_pair is not None
            alternative_pair = torch.where(
                spawn_secondary[:, None, None],
                seed_pair,
                alternative_transport.pair_probabilities,
            )
            alternative_pair = torch.where(
                oracle_association_used[:, None, None],
                oracle_pair,
                alternative_pair,
            )
            alternative_strength = alternative_pair.sum(-1).clamp(0.0, 1.0)
            alternative_base_mean = torch.where(
                spawn_secondary[:, None, None],
                prior_mean,
                alternative_prior_mean,
            )
            alternative_base_covariance = torch.where(
                spawn_secondary[:, None, None, None],
                prior_covariance,
                alternative_prior_covariance,
            )
            alternative_base_query = torch.where(
                spawn_secondary[:, None, None],
                prior_query,
                alternative_prior_query,
            )
            (
                alternative_posterior_mean,
                alternative_posterior_covariance,
                alternative_posterior_query,
            ) = self.update.forward_mixture(
                    alternative_base_mean,
                    alternative_base_covariance,
                    alternative_base_query,
                    measurements,
                    embeddings,
                    state.time_since_association + delta_t,
                    alternative_pair,
                    alternative_pair_measurement_variance,
                    alternative_pair_update_candidates,
                )
            assert transport.hypothesis_scores is not None
            assert alternative_transport.hypothesis_scores is not None
            primary_score = transport.hypothesis_scores[:, 0]
            alternative_score = torch.where(
                spawn_secondary,
                transport.hypothesis_scores[:, 1],
                alternative_transport.hypothesis_scores[:, 0],
            )
            candidate_weights = state.hypothesis_log_weights + torch.stack(
                (primary_score, alternative_score), dim=-1
            )
            initialized_weights = torch.stack(
                (primary_score, alternative_score), dim=-1
            )
            candidate_weights = torch.where(
                spawn_secondary.unsqueeze(-1),
                initialized_weights,
                candidate_weights,
            )
            candidate_weights = candidate_weights - torch.logsumexp(
                candidate_weights, dim=-1, keepdim=True
            )
            learned_row = ~oracle_association_used
            candidate_weights = torch.where(
                learned_row.unsqueeze(-1),
                candidate_weights,
                candidate_weights.new_tensor([0.0, -8.0]),
            )
            choose_alternative = learned_row & (
                candidate_weights[:, 1] > candidate_weights[:, 0]
            )
            primary_mean_before_choice = posterior_mean
            primary_covariance_before_choice = posterior_covariance
            primary_query_before_choice = posterior_query
            posterior_mean = torch.where(
                choose_alternative[:, None, None],
                alternative_posterior_mean,
                posterior_mean,
            )
            posterior_covariance = torch.where(
                choose_alternative[:, None, None, None],
                alternative_posterior_covariance,
                posterior_covariance,
            )
            posterior_query = torch.where(
                choose_alternative[:, None, None],
                alternative_posterior_query,
                posterior_query,
            )
            selected_pair = torch.where(
                choose_alternative[:, None, None],
                alternative_pair,
                selected_pair,
            )
            association_strength = selected_pair.sum(-1).clamp(0.0, 1.0)
            assert alternative_transport.unclaimed_probabilities is not None
            seed_unclaimed = transport.alternative_unclaimed_probabilities
            assert seed_unclaimed is not None
            alternative_unclaimed = torch.where(
                spawn_secondary[:, None],
                seed_unclaimed,
                alternative_transport.unclaimed_probabilities,
            )
            selected_unclaimed_probability = torch.where(
                choose_alternative[:, None],
                alternative_unclaimed,
                transport.unclaimed_probabilities,
            )
            alternative_posterior_mean = torch.where(
                choose_alternative[:, None, None],
                primary_mean_before_choice,
                alternative_posterior_mean,
            )
            alternative_posterior_covariance = torch.where(
                choose_alternative[:, None, None, None],
                primary_covariance_before_choice,
                alternative_posterior_covariance,
            )
            alternative_posterior_query = torch.where(
                choose_alternative[:, None, None],
                primary_query_before_choice,
                alternative_posterior_query,
            )
            updated_hypothesis_log_weights = torch.stack(
                (
                    torch.maximum(candidate_weights[:, 0], candidate_weights[:, 1]),
                    torch.minimum(candidate_weights[:, 0], candidate_weights[:, 1]),
                ),
                dim=-1,
            )
            associated_hard = association_strength >= 0.5
        covariance_diag = posterior_covariance.diagonal(dim1=-2, dim2=-1)
        existence_features = torch.cat(
            (
                posterior_query,
                association_strength.unsqueeze(-1),
                torch.log1p(state.time_since_association + delta_t).unsqueeze(-1),
                torch.log1p(covariance_diag[..., :2].mean(-1, keepdim=True)),
            ),
            dim=-1,
        )
        existing_logits = self.existence_head(existence_features).squeeze(-1)
        existing_logits = torch.where(
            existing_mask, existing_logits, torch.full_like(existing_logits, -8.0)
        )
        # Objectness is computed before lifecycle evidence so v18-D can tell a
        # target-supported association from a high-probability clutter edge.
        # Legacy models merely receive and ignore this tensor.
        birth_target_log_intensity = None
        birth_clutter_log_intensity = None
        predicted_undetected_mean = None
        measurement_objectness_logits = self.birth_existence_head(
            embeddings
        ).squeeze(-1)
        birth_evidence_logits = measurement_objectness_logits
        if getattr(self.config, "birth_neural_ppp", False):
            (
                birth_evidence_logits,
                birth_target_log_intensity,
                birth_clutter_log_intensity,
                predicted_undetected_mean,
            ) = self._predict_neural_ppp(
                state, embeddings, measurement_padding_mask
            )
        existing_logits, evidence_hidden = self._update_track_evidence(
            state,
            posterior_query,
            existing_logits,
            association_strength,
            selected_pair,
            state.time_since_association + delta_t,
            posterior_covariance,
            existing_mask,
            track_survival_prior_logits,
            measurement_objectness_logits,
        )
        lifecycle_features = torch.cat(
            (
                existence_features,
                torch.log1p(state.ages.to(posterior_mean.dtype)).unsqueeze(-1),
                torch.log1p(state.hit_streak.to(posterior_mean.dtype)).unsqueeze(-1),
                torch.log1p(state.miss_streak.to(posterior_mean.dtype)).unsqueeze(-1),
                state.exist_logit.sigmoid().unsqueeze(-1),
            ),
            dim=-1,
        )
        if transport is not None and transport.death_probabilities is not None:
            predicted_survival = (1.0 - transport.death_probabilities).clamp(
                1.0e-6, 1.0 - 1.0e-6
            )
            survival_logits = torch.logit(predicted_survival)
        else:
            survival_logits = self._predict_survival(
                lifecycle_features, existing_logits
            )
        if getattr(self.config, "track_evidence_recurrence", False):
            # v18-B has one Bernoulli posterior for output, association prior
            # and lifecycle supervision instead of two inconsistently
            # calibrated existence/survival heads.
            survival_logits = existing_logits
        reported_survival_prior_logits = (
            track_survival_prior_logits
            if track_survival_prior_logits is not None
            else survival_logits
        )
        survival_logits = torch.where(
            existing_mask,
            survival_logits,
            torch.full_like(survival_logits, -8.0),
        )
        reported_survival_prior_logits = torch.where(
            existing_mask,
            reported_survival_prior_logits,
            torch.full_like(reported_survival_prior_logits, -8.0),
        )

        oracle_birth_mask = (~measurement_padding_mask) & (~claimed_measurements)
        birth_positions = measurements + self.birth_position_head(embeddings)
        if birth_state_mode == "oracle":
            birth_mask = oracle_birth_mask
            birth_logits = birth_evidence_logits.masked_fill(~birth_mask, -8.0)
            birth_probabilities = birth_logits.sigmoid()
        else:
            birth_mask = ~measurement_padding_mask
            if transport is None:
                residual_probability = torch.ones_like(measurement_objectness_logits)
            else:
                assert selected_unclaimed_probability is not None
                residual_probability = selected_unclaimed_probability
            birth_probabilities, birth_logits = _combine_birth_evidence(
                birth_evidence_logits,
                residual_probability,
                pmbm_odds_fusion=(
                    getattr(self.config, "birth_pmbm_odds_fusion", False)
                    or getattr(self.config, "birth_neural_ppp", False)
                ),
                unclaimed_log_scale=getattr(
                    self.config, "birth_unclaimed_log_scale", 1.0
                ),
            )
            birth_logits = birth_logits.masked_fill(
                ~birth_mask, -8.0
            )

        output = RecursiveFrameOutput(
            prior_mean=prior_mean,
            prior_covariance=prior_covariance,
            posterior_mean=posterior_mean,
            posterior_covariance=posterior_covariance,
            existing_logits=existing_logits,
            survival_logits=survival_logits,
            survival_prior_logits=reported_survival_prior_logits,
            existing_mask=existing_mask,
            existing_confirmed_mask=state.confirmed_mask.clone(),
            existing_owner_ids=state.supervision_ids.clone(),
            existing_runtime_ids=existing_owner_ids,
            association_mask=associated_hard,
            birth_positions=birth_positions,
            birth_logits=birth_logits,
            birth_mask=birth_mask,
            birth_measurement_ids=measurement_ids,
            measurement_embeddings=embeddings,
            measurement_padding_mask=measurement_padding_mask,
            measurement_objectness_logits=measurement_objectness_logits,
            association_strength=association_strength,
            predicted_pair_probabilities=(
                transport.pair_probabilities if transport is not None else None
            ),
            predicted_miss_probabilities=(
                transport.miss_probabilities if transport is not None else None
            ),
            predicted_unclaimed_probabilities=(
                transport.unclaimed_probabilities if transport is not None else None
            ),
            predicted_death_probabilities=(
                transport.death_probabilities if transport is not None else None
            ),
            predicted_pair_log_probabilities=(
                transport.pair_log_probabilities if transport is not None else None
            ),
            predicted_miss_log_probabilities=(
                transport.miss_log_probabilities if transport is not None else None
            ),
            predicted_unclaimed_log_probabilities=(
                transport.unclaimed_log_probabilities if transport is not None else None
            ),
            predicted_death_log_probabilities=(
                transport.death_log_probabilities if transport is not None else None
            ),
            association_marginal_error=(
                transport.marginal_error if transport is not None else None
            ),
            association_capacity_violation=(
                transport.capacity_violation if transport is not None else None
            ),
            oracle_association_used=oracle_association_used,
            association_track_bias=track_measurement_bias,
            predicted_undetected_mean=predicted_undetected_mean,
            birth_target_log_intensity=birth_target_log_intensity,
            birth_clutter_log_intensity=birth_clutter_log_intensity,
            association_hypothesis_weights=(
                updated_hypothesis_log_weights.softmax(-1)
                if getattr(self.config, "association_mbm_hypotheses", 1) == 2
                else None
            ),
            alternative_posterior_mean=(
                alternative_posterior_mean
                if getattr(self.config, "association_mbm_hypotheses", 1) == 2
                else None
            ),
            birth_unclaimed_probabilities=selected_unclaimed_probability,
        )

        next_state = state.clone()
        next_state.mean = torch.where(
            existing_mask.unsqueeze(-1), posterior_mean, next_state.mean
        )
        next_state.covariance = torch.where(
            existing_mask.unsqueeze(-1).unsqueeze(-1),
            posterior_covariance,
            next_state.covariance,
        )
        next_state.query = torch.where(
            existing_mask.unsqueeze(-1), posterior_query, next_state.query
        )
        if getattr(self.config, "association_mbm_hypotheses", 1) == 2:
            next_state.alternative_mean = torch.where(
                existing_mask.unsqueeze(-1),
                alternative_posterior_mean,
                next_state.alternative_mean,
            )
            next_state.alternative_covariance = torch.where(
                existing_mask.unsqueeze(-1).unsqueeze(-1),
                alternative_posterior_covariance,
                next_state.alternative_covariance,
            )
            next_state.alternative_query = torch.where(
                existing_mask.unsqueeze(-1),
                alternative_posterior_query,
                next_state.alternative_query,
            )
            next_state.hypothesis_log_weights = updated_hypothesis_log_weights
        next_state.evidence_hidden = torch.where(
            existing_mask.unsqueeze(-1),
            evidence_hidden,
            next_state.evidence_hidden,
        )
        next_state.exist_logit = torch.where(
            existing_mask, existing_logits, next_state.exist_logit
        )
        next_state.ages = torch.where(
            existing_mask, next_state.ages + 1, next_state.ages
        )
        next_hit_streak = torch.where(
            associated_hard,
            next_state.hit_streak + 1,
            torch.zeros_like(next_state.hit_streak),
        )
        next_state.hit_streak = torch.where(
            existing_mask, next_hit_streak, next_state.hit_streak
        )
        next_state.confirmed_mask = self._next_confirmation_mask(
            next_state,
            existing_mask,
            existing_logits,
            confirmation_hits,
        )
        next_state.miss_streak = torch.where(
            existing_mask,
            torch.where(
                associated_hard,
                torch.zeros_like(next_state.miss_streak),
                next_state.miss_streak + 1,
            ),
            next_state.miss_streak,
        )
        next_state.last_update_time = torch.where(
            existing_mask, frame_time[:, None], next_state.last_update_time
        )
        next_state.time_since_association = torch.where(
            existing_mask,
            (1.0 - association_strength)
            * (next_state.time_since_association + delta_t),
            next_state.time_since_association,
        )
        if death_state_mode == "oracle_post":
            next_state = self._prune_dead(next_state, truth_sets)
        elif death_state_mode == "oracle_delayed":
            next_state = self._advance_oracle_death(
                next_state, truth_sets, prune_after=oracle_prune_delay
            )
        elif death_state_mode == "predicted":
            # A newborn query has only one measurement encoded in its state.
            # Keep it long enough to form a short recursive motion/association
            # history before asking the learned survival head to terminate it.
            # This uses state age only; no truth metadata enters the decision.
            eligible_for_termination = state.ages >= survival_warmup_frames
            if selected_death_probability is None:
                retain_probability = survival_logits.sigmoid()
            else:
                retain_probability = 1.0 - selected_death_probability
            terminate = (
                existing_mask
                & eligible_for_termination
                & (retain_probability < retention_threshold)
            )
            next_state.active_mask[terminate] = False
            next_state.confirmed_mask[terminate] = False
            next_state.owner_ids[terminate] = -1
            next_state.supervision_ids[terminate] = -1
            next_state.exist_logit[terminate] = -8.0
            next_state.evidence_hidden[terminate] = 0.0
            next_state.ages[terminate] = 0
            next_state.hit_streak[terminate] = 0
            next_state.miss_streak[terminate] = 0
            next_state.time_since_association[terminate] = 0.0
            next_state.oracle_dead_age[terminate] = 0

        birth_velocity = self.birth_velocity_head(embeddings)
        birth_variance = (
            torch.nn.functional.softplus(self.birth_variance_head(embeddings))
            + self.config.min_variance
        )
        birth_variance = birth_variance.clamp_max(self.config.max_variance)
        birth_query = self.birth_query_head(
            torch.cat((embeddings, measurements / self.config.field_scale), dim=-1)
        )
        # Slot decisions in predicted mode depend only on the factorized birth
        # posterior.  Measurement IDs are written solely as supervision metadata
        # after selection and are never consulted by a neural layer or threshold.
        if fast_inference:
            if birth_state_mode != "predicted" or death_state_mode != "predicted":
                raise ValueError(
                    "fast inference requires predicted birth and predicted death"
                )
            self._insert_predicted_births_vectorized(
                next_state,
                birth_mask,
                birth_probabilities,
                birth_logits,
                birth_positions,
                birth_velocity,
                birth_variance,
                birth_query,
                frame_time,
                birth_candidate_threshold,
                confirmation_hits,
            )
        else:
            self._insert_births_reference(
                next_state,
                truth_sets,
                measurement_ids,
                oracle_birth_mask,
                birth_mask,
                birth_probabilities,
                birth_logits,
                birth_positions,
                birth_velocity,
                birth_variance,
                birth_query,
                frame_time,
                birth_state_mode,
                birth_candidate_threshold,
                confirmation_hits,
            )
        if getattr(self.config, "birth_neural_ppp", False):
            assert predicted_undetected_mean is not None
            self._update_neural_ppp_state(
                next_state,
                embeddings,
                measurement_padding_mask,
                birth_probabilities,
                predicted_undetected_mean,
            )
        if getattr(self.config, "association_mbm_hypotheses", 1) == 2:
            newborn_slots = next_state.active_mask & ~existing_mask
            next_state.alternative_mean = torch.where(
                newborn_slots.unsqueeze(-1),
                next_state.mean,
                next_state.alternative_mean,
            )
            next_state.alternative_covariance = torch.where(
                newborn_slots.unsqueeze(-1).unsqueeze(-1),
                next_state.covariance,
                next_state.alternative_covariance,
            )
            next_state.alternative_query = torch.where(
                newborn_slots.unsqueeze(-1),
                next_state.query,
                next_state.alternative_query,
            )
        if not fast_inference:
            next_state.validate()
        return output, next_state

    def _insert_births_reference(
        self,
        next_state: RecursiveTrackState,
        truth_sets: list[set[int]],
        measurement_ids: torch.Tensor,
        oracle_birth_mask: torch.Tensor,
        birth_mask: torch.Tensor,
        birth_probabilities: torch.Tensor,
        birth_logits: torch.Tensor,
        birth_positions: torch.Tensor,
        birth_velocity: torch.Tensor,
        birth_variance: torch.Tensor,
        birth_query: torch.Tensor,
        frame_time: torch.Tensor,
        birth_state_mode: str,
        birth_candidate_threshold: float,
        confirmation_hits: int,
    ) -> None:
        """Historical scalar birth admission used by training and diagnostics."""

        for batch_index, alive_ids in enumerate(truth_sets):
            owned_supervision = {
                int(value)
                for value in next_state.supervision_ids[
                    batch_index, next_state.active_mask[batch_index]
                ]
                .detach()
                .cpu()
                .tolist()
                if int(value) >= 0
            }
            if birth_state_mode == "oracle":
                candidates = torch.nonzero(
                    oracle_birth_mask[batch_index], as_tuple=False
                ).flatten()
            else:
                candidates = torch.nonzero(
                    birth_mask[batch_index]
                    & (birth_probabilities[batch_index] >= birth_candidate_threshold),
                    as_tuple=False,
                ).flatten()
                candidates = candidates[
                    torch.argsort(
                        birth_probabilities[batch_index, candidates],
                        descending=True,
                    )
                ]
            for measurement_index in candidates.tolist():
                supervision_id = int(measurement_ids[batch_index, measurement_index])
                if birth_state_mode == "oracle" and (
                    supervision_id < 0
                    or supervision_id not in alive_ids
                    or supervision_id in owned_supervision
                ):
                    continue
                free = torch.nonzero(
                    ~next_state.active_mask[batch_index], as_tuple=False
                ).flatten()
                if not len(free):
                    if birth_state_mode == "oracle":
                        raise RuntimeError("v17-A ran out of oracle track slots")
                    break
                slot = int(free[0])
                next_state.active_mask[batch_index, slot] = True
                if birth_state_mode == "oracle":
                    runtime_id = supervision_id
                    next_state.next_runtime_id[batch_index] = torch.maximum(
                        next_state.next_runtime_id[batch_index],
                        next_state.next_runtime_id.new_tensor(runtime_id + 1),
                    )
                else:
                    runtime_id = int(next_state.next_runtime_id[batch_index])
                    next_state.next_runtime_id[batch_index] += 1
                next_state.owner_ids[batch_index, slot] = runtime_id
                next_state.supervision_ids[batch_index, slot] = supervision_id
                if getattr(self.config, "track_evidence_recurrence", False):
                    next_state.confirmed_mask[batch_index, slot] = (
                        birth_state_mode == "oracle"
                        or birth_probabilities[batch_index, measurement_index]
                        >= self.config.track_evidence_confirmation_threshold
                    )
                else:
                    next_state.confirmed_mask[batch_index, slot] = (
                        birth_state_mode == "oracle" or confirmation_hits <= 1
                    )
                next_state.mean[batch_index, slot] = torch.cat(
                    (
                        birth_positions[batch_index, measurement_index],
                        birth_velocity[batch_index, measurement_index],
                    )
                )
                next_state.covariance[batch_index, slot] = torch.diag(
                    birth_variance[batch_index, measurement_index]
                )
                next_state.query[batch_index, slot] = birth_query[
                    batch_index, measurement_index
                ]
                if getattr(self.config, "track_evidence_recurrence", False):
                    next_state.evidence_hidden[batch_index, slot] = (
                        self._initialize_birth_evidence(
                            birth_query[batch_index, measurement_index],
                            birth_logits[batch_index, measurement_index],
                        )
                    )
                next_state.exist_logit[batch_index, slot] = birth_logits[
                    batch_index, measurement_index
                ]
                next_state.ages[batch_index, slot] = 0
                next_state.hit_streak[batch_index, slot] = 1
                next_state.miss_streak[batch_index, slot] = 0
                next_state.last_update_time[batch_index, slot] = frame_time[batch_index]
                next_state.time_since_association[batch_index, slot] = 0.0
                next_state.oracle_dead_age[batch_index, slot] = 0
                if supervision_id >= 0:
                    owned_supervision.add(supervision_id)

    def _insert_predicted_births_vectorized(
        self,
        next_state: RecursiveTrackState,
        birth_mask: torch.Tensor,
        birth_probabilities: torch.Tensor,
        birth_logits: torch.Tensor,
        birth_positions: torch.Tensor,
        birth_velocity: torch.Tensor,
        birth_variance: torch.Tensor,
        birth_query: torch.Tensor,
        frame_time: torch.Tensor,
        birth_candidate_threshold: float,
        confirmation_hits: int,
    ) -> None:
        """Assign ranked predicted births to ranked free slots on-device."""

        candidate_mask = birth_mask & (
            birth_probabilities >= birth_candidate_threshold
        )
        ranked_measurements = torch.argsort(
            birth_probabilities.masked_fill(~candidate_mask, float("-inf")),
            dim=-1,
            descending=True,
        )
        free_mask = ~next_state.active_mask
        free_rank = free_mask.to(torch.long).cumsum(-1) - 1
        candidate_count = candidate_mask.sum(-1)
        assigned = free_mask & (free_rank < candidate_count.unsqueeze(-1))
        selected_measurement = ranked_measurements.gather(
            1,
            free_rank.clamp(min=0, max=ranked_measurements.shape[-1] - 1),
        )

        def gather(values: torch.Tensor) -> torch.Tensor:
            index = selected_measurement
            for _ in range(values.ndim - 2):
                index = index.unsqueeze(-1)
            return values.gather(
                1,
                index.expand(*selected_measurement.shape, *values.shape[2:]),
            )

        selected_probability = gather(birth_probabilities)
        selected_logit = gather(birth_logits)
        selected_query = gather(birth_query)
        selected_mean = torch.cat(
            (gather(birth_positions), gather(birth_velocity)), dim=-1
        )
        selected_covariance = torch.diag_embed(gather(birth_variance))
        selected_evidence = self._initialize_birth_evidence(
            selected_query, selected_logit
        )
        runtime_ids = next_state.next_runtime_id.unsqueeze(-1) + free_rank

        next_state.active_mask = next_state.active_mask | assigned
        next_state.owner_ids = torch.where(
            assigned, runtime_ids, next_state.owner_ids
        )
        next_state.supervision_ids = torch.where(
            assigned,
            torch.full_like(next_state.supervision_ids, -1),
            next_state.supervision_ids,
        )
        if getattr(self.config, "track_evidence_recurrence", False):
            admitted = (
                selected_probability
                >= self.config.track_evidence_confirmation_threshold
            )
        else:
            admitted = torch.full_like(assigned, confirmation_hits <= 1)
        next_state.confirmed_mask = torch.where(
            assigned, admitted, next_state.confirmed_mask
        )
        next_state.mean = torch.where(
            assigned.unsqueeze(-1), selected_mean, next_state.mean
        )
        next_state.covariance = torch.where(
            assigned.unsqueeze(-1).unsqueeze(-1),
            selected_covariance,
            next_state.covariance,
        )
        next_state.query = torch.where(
            assigned.unsqueeze(-1), selected_query, next_state.query
        )
        next_state.evidence_hidden = torch.where(
            assigned.unsqueeze(-1), selected_evidence, next_state.evidence_hidden
        )
        next_state.exist_logit = torch.where(
            assigned, selected_logit, next_state.exist_logit
        )
        next_state.ages = torch.where(
            assigned, torch.zeros_like(next_state.ages), next_state.ages
        )
        next_state.hit_streak = torch.where(
            assigned, torch.ones_like(next_state.hit_streak), next_state.hit_streak
        )
        next_state.miss_streak = torch.where(
            assigned, torch.zeros_like(next_state.miss_streak), next_state.miss_streak
        )
        next_state.last_update_time = torch.where(
            assigned, frame_time.unsqueeze(-1), next_state.last_update_time
        )
        next_state.time_since_association = torch.where(
            assigned,
            torch.zeros_like(next_state.time_since_association),
            next_state.time_since_association,
        )
        next_state.oracle_dead_age = torch.where(
            assigned,
            torch.zeros_like(next_state.oracle_dead_age),
            next_state.oracle_dead_age,
        )
        next_state.next_runtime_id = (
            next_state.next_runtime_id + assigned.sum(-1)
        )


class JointAssociationTracker(OracleRecursiveTracker):
    """v17-B: learned current-frame association with oracle lifecycle."""

    def __init__(self, config: V17BConfig = V17BConfig()):
        super().__init__(config)
        self.association = JointAssociation(config)

    def _predict_association(
        self,
        prior_mean: torch.Tensor,
        prior_covariance: torch.Tensor,
        prior_query: torch.Tensor,
        measurements: torch.Tensor,
        embeddings: torch.Tensor,
        existing_mask: torch.Tensor,
        measurement_padding_mask: torch.Tensor,
        delta_t_association: torch.Tensor,
        track_measurement_bias: torch.Tensor,
        survival_prior_logits: torch.Tensor | None,
        pair_measurement_variance: torch.Tensor | None,
    ) -> AssociationTransport:
        return self.association(
            prior_mean,
            prior_covariance,
            prior_query,
            measurements,
            embeddings,
            existing_mask,
            measurement_padding_mask,
            delta_t_association,
            track_measurement_bias,
            survival_prior_logits,
            pair_measurement_variance,
        )


class SeparatedLifecycleTracker(JointAssociationTracker):
    """G3d tracker with distinct output-existence and state-survival heads."""

    def __init__(self, config: V17BConfig = V17BConfig()):
        super().__init__(config)
        self.survival_head = nn.Sequential(
            nn.Linear(config.hidden_dim + 7, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def _predict_survival(
        self,
        lifecycle_features: torch.Tensor,
        existing_logits: torch.Tensor,
    ) -> torch.Tensor:
        return self.survival_head(lifecycle_features).squeeze(-1)


class EndToEndRecursiveTracker(SeparatedLifecycleTracker):
    """Final v17 architecture trained jointly from random initialization.

    The association prior is predicted from the recursive track state before
    seeing the current measurements.  It therefore plays the role of a neural
    Bernoulli-existence prior without re-reading historical measurements.
    """

    def __init__(self, config: V17BConfig = V17BConfig()):
        super().__init__(config)
        # Survival must start as an uninformative Bernoulli prior.  A tiny
        # non-zero weight keeps gradients flowing into the preceding layer on
        # the first negative-exposed update without imposing an alive/dead
        # preference before the rollout has produced both classes.
        nn.init.normal_(self.survival_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.survival_head[-1].bias)
        self.association_prior_head = nn.Sequential(
            nn.Linear(config.hidden_dim + 7, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        nn.init.zeros_(self.association_prior_head[-1].weight)
        nn.init.zeros_(self.association_prior_head[-1].bias)
        if config.ambiguity_aware_update:
            self.ambiguity_write_head = nn.Sequential(
                nn.Linear(config.hidden_dim + 5, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, 1),
            )
            nn.init.normal_(self.ambiguity_write_head[-1].weight, std=1.0e-3)
            # Begin close to the historical moment write while retaining an
            # immediate gradient path into the ambiguity controller.
            nn.init.constant_(self.ambiguity_write_head[-1].bias, -2.0)
        if config.track_evidence_recurrence:
            # Inputs are the posterior query plus nine causal scalars:
            # previous/current existence, PAIR/MISS evidence, ambiguity,
            # elapsed physical time, covariance and age.  The controlled
            # v18-B+ path appends target-weighted PAIR support while retaining
            # all original v18-B evidence.
            evidence_scalar_count = (
                10 if config.track_evidence_objectness_aware else 9
            )
            self.track_evidence_cell = nn.GRUCell(
                config.hidden_dim + evidence_scalar_count,
                config.hidden_dim,
            )
            evidence_head_scalar_count = (
                4 if config.track_evidence_objectness_aware else 3
            )
            self.track_evidence_head = nn.Sequential(
                nn.Linear(
                    config.hidden_dim + evidence_head_scalar_count,
                    config.hidden_dim,
                ),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, 2),
            )
            self.birth_evidence_head = nn.Sequential(
                nn.Linear(config.hidden_dim + 1, config.hidden_dim),
                nn.Tanh(),
            )
            # Start as a neutral fusion of prior and instantaneous evidence.
            # Tiny output weights allow gradients into the recurrent cell on
            # the first exposed lifecycle negative without an arbitrary bias.
            nn.init.normal_(self.track_evidence_head[-1].weight, std=1.0e-3)
            nn.init.zeros_(self.track_evidence_head[-1].bias)
            if config.track_evidence_log_odds_update:
                # The survival branch is strictly pre-measurement.  Its bias
                # starts at the simulator's middle survival regime while a
                # bounded residual learns state/time-dependent deviations.
                self.track_survival_prior_head = nn.Sequential(
                    nn.Linear(config.hidden_dim * 2 + 5, config.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(config.hidden_dim, 1),
                )
                self.track_likelihood_ratio_head = nn.Sequential(
                    nn.Linear(config.hidden_dim + 5, config.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(config.hidden_dim, 1),
                )
                nn.init.normal_(
                    self.track_survival_prior_head[-1].weight, std=1.0e-3
                )
                nn.init.zeros_(self.track_survival_prior_head[-1].bias)
                nn.init.normal_(
                    self.track_likelihood_ratio_head[-1].weight, std=1.0e-3
                )
                nn.init.zeros_(self.track_likelihood_ratio_head[-1].bias)

    def _initialize_birth_evidence(
        self,
        birth_query: torch.Tensor,
        birth_logit: torch.Tensor,
    ) -> torch.Tensor:
        if not self.config.track_evidence_recurrence:
            return torch.zeros_like(birth_query)
        return self.birth_evidence_head(
            torch.cat((birth_query, birth_logit.unsqueeze(-1)), dim=-1)
        )

    def _update_track_evidence(
        self,
        state: RecursiveTrackState,
        posterior_query: torch.Tensor,
        instantaneous_logits: torch.Tensor,
        association_strength: torch.Tensor,
        selected_pair: torch.Tensor,
        elapsed: torch.Tensor,
        posterior_covariance: torch.Tensor,
        existing_mask: torch.Tensor,
        survival_prior_logits: torch.Tensor | None,
        measurement_objectness_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.config.track_evidence_recurrence:
            return super()._update_track_evidence(
                state,
                posterior_query,
                instantaneous_logits,
                association_strength,
                selected_pair,
                elapsed,
                posterior_covariance,
                existing_mask,
                survival_prior_logits,
                measurement_objectness_logits,
            )

        pair_mass = selected_pair.sum(-1).clamp(0.0, 1.0)
        objectness_pair_mass = (
            selected_pair * measurement_objectness_logits.sigmoid().unsqueeze(1)
        ).sum(-1).clamp(0.0, 1.0)
        evidence_pair_mass = (
            objectness_pair_mass
            if self.config.track_evidence_objectness_aware
            else pair_mass
        )
        conditional_pair = selected_pair / pair_mass.unsqueeze(-1).clamp_min(1.0e-8)
        association_entropy = -(
            conditional_pair
            * conditional_pair.clamp_min(1.0e-8).log()
        ).sum(-1)
        if selected_pair.shape[-1] > 1:
            top_two = selected_pair.topk(2, dim=-1).values
            association_margin = top_two[..., 0] - top_two[..., 1]
        elif selected_pair.shape[-1] == 1:
            association_margin = selected_pair[..., 0]
        else:
            association_margin = torch.zeros_like(pair_mass)
        association_entropy = torch.where(
            pair_mass > 0.0,
            association_entropy,
            torch.zeros_like(association_entropy),
        )
        covariance_diag = posterior_covariance.diagonal(dim1=-2, dim2=-1)
        scalar_values = [
            state.exist_logit.sigmoid(),
            instantaneous_logits.sigmoid(),
            association_strength,
        ]
        if self.config.track_evidence_objectness_aware:
            scalar_values.append(objectness_pair_mass)
        scalar_values.extend(
            (
                (1.0 - pair_mass).clamp(0.0, 1.0),
                association_entropy,
                association_margin,
                torch.log1p(elapsed),
                torch.log1p(covariance_diag[..., :2].mean(-1)),
                torch.log1p(state.ages.to(posterior_query.dtype)),
            )
        )
        scalar_features = torch.stack(scalar_values, dim=-1)
        cell_input = torch.cat((posterior_query, scalar_features), dim=-1)
        flat_hidden = self.track_evidence_cell(
            cell_input.reshape(-1, cell_input.shape[-1]),
            state.evidence_hidden.reshape(-1, state.evidence_hidden.shape[-1]),
        ).view_as(state.evidence_hidden)
        if self.config.track_evidence_log_odds_update:
            if survival_prior_logits is None:
                raise RuntimeError("v18-C requires a causal survival prior")
            likelihood_features = torch.cat(
                (
                    flat_hidden,
                    evidence_pair_mass.unsqueeze(-1),
                    association_entropy.unsqueeze(-1),
                    association_margin.unsqueeze(-1),
                    torch.log1p(elapsed).unsqueeze(-1),
                    torch.log1p(covariance_diag[..., :2].mean(-1)).unsqueeze(-1),
                ),
                dim=-1,
            )
            residual_scale = self.config.track_evidence_likelihood_residual_scale
            learned_likelihood = residual_scale * torch.tanh(
                self.track_likelihood_ratio_head(likelihood_features).squeeze(-1)
                / residual_scale
            )
            pair_anchor = self.config.track_evidence_pair_anchor_scale * (
                evidence_pair_mass - self.config.track_evidence_pair_anchor_center
            )
            previous_probability = state.exist_logit.sigmoid()
            survival_probability = survival_prior_logits.sigmoid()
            if self.config.track_evidence_exact_bernoulli_prediction:
                # Bernoulli random-finite-set prediction: an object exists at
                # t only if it existed at t-1 and survived the transition.
                predicted_probability = previous_probability * survival_probability
            else:
                # Preserve the historical v18-C odds update exactly.
                predicted_probability = (
                    previous_probability * survival_probability
                ) / (
                    1.0
                    - previous_probability
                    + previous_probability * survival_probability
                ).clamp_min(1.0e-6)
            predicted_logit = torch.logit(
                predicted_probability.clamp(1.0e-6, 1.0 - 1.0e-6)
            )
            updated_logits = (
                predicted_logit + pair_anchor + learned_likelihood
            ).clamp(-8.0, 8.0)
            updated_logits = torch.where(
                existing_mask,
                updated_logits,
                torch.full_like(updated_logits, -8.0),
            )
            updated_hidden = torch.where(
                existing_mask.unsqueeze(-1),
                flat_hidden,
                state.evidence_hidden,
            )
            return updated_logits, updated_hidden
        head_values = [
            flat_hidden,
            state.exist_logit.sigmoid().unsqueeze(-1),
            instantaneous_logits.sigmoid().unsqueeze(-1),
            association_strength.unsqueeze(-1),
        ]
        if self.config.track_evidence_objectness_aware:
            head_values.append(objectness_pair_mass.unsqueeze(-1))
        head_input = torch.cat(head_values, dim=-1)
        gate_logit, residual_raw = self.track_evidence_head(head_input).unbind(-1)
        gate = gate_logit.sigmoid()
        residual_scale = self.config.track_evidence_residual_scale
        updated_logits = (
            gate * state.exist_logit
            + (1.0 - gate) * instantaneous_logits
            + residual_scale * torch.tanh(residual_raw / residual_scale)
        ).clamp(-8.0, 8.0)
        updated_logits = torch.where(
            existing_mask,
            updated_logits,
            torch.full_like(updated_logits, -8.0),
        )
        updated_hidden = torch.where(
            existing_mask.unsqueeze(-1),
            flat_hidden,
            state.evidence_hidden,
        )
        return updated_logits, updated_hidden

    def _predict_track_survival_prior(
        self,
        state: RecursiveTrackState,
        prior_query: torch.Tensor,
        prior_covariance: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.config.track_evidence_log_odds_update:
            return None
        elapsed = state.time_since_association + delta_t
        covariance_diag = prior_covariance.diagonal(dim1=-2, dim2=-1)
        features = torch.cat(
            (
                prior_query,
                state.evidence_hidden,
                state.exist_logit.sigmoid().unsqueeze(-1),
                torch.log1p(elapsed).unsqueeze(-1),
                torch.log1p(covariance_diag[..., :2].mean(-1, keepdim=True)),
                torch.log1p(state.ages.to(prior_query.dtype)).unsqueeze(-1),
                torch.log1p(state.miss_streak.to(prior_query.dtype)).unsqueeze(-1),
            ),
            dim=-1,
        )
        residual_scale = self.config.track_survival_logit_residual_scale
        residual = residual_scale * torch.tanh(
            self.track_survival_prior_head(features).squeeze(-1) / residual_scale
        )
        base = torch.logit(
            residual.new_tensor(self.config.track_survival_probability_init)
        )
        logits = (base + residual).clamp(-8.0, 8.0)
        return torch.where(
            state.active_mask, logits, torch.full_like(logits, -8.0)
        )

    def _predict_survival_prior(
        self,
        state: RecursiveTrackState,
        prior_query: torch.Tensor,
        prior_covariance: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.config.joint_lifecycle_transport:
            return None
        elapsed = state.time_since_association + delta_t
        covariance_diag = prior_covariance.diagonal(dim1=-2, dim2=-1)
        features = torch.cat(
            (
                prior_query,
                torch.exp(-elapsed).unsqueeze(-1),
                torch.log1p(elapsed).unsqueeze(-1),
                torch.log1p(covariance_diag[..., :2].mean(-1, keepdim=True)),
                torch.log1p(state.ages.to(state.mean.dtype)).unsqueeze(-1),
                torch.log1p(state.hit_streak.to(state.mean.dtype)).unsqueeze(-1),
                torch.log1p(state.miss_streak.to(state.mean.dtype)).unsqueeze(-1),
                state.exist_logit.sigmoid().unsqueeze(-1),
            ),
            dim=-1,
        )
        return self.survival_head(features).squeeze(-1)

    def _predict_track_measurement_bias(
        self,
        state: RecursiveTrackState,
        delta_t: torch.Tensor,
        confirmed_association_bias: float,
        survival_prior_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        covariance_diag = state.covariance.diagonal(dim1=-2, dim2=-1)
        features = torch.cat(
            (
                state.query,
                torch.log1p(state.ages.to(state.mean.dtype)).unsqueeze(-1),
                torch.log1p(state.hit_streak.to(state.mean.dtype)).unsqueeze(-1),
                torch.log1p(state.miss_streak.to(state.mean.dtype)).unsqueeze(-1),
                state.exist_logit.sigmoid().unsqueeze(-1),
                torch.log1p(covariance_diag[..., :2].mean(-1, keepdim=True)),
                torch.log1p(state.time_since_association + delta_t).unsqueeze(-1),
                state.confirmed_mask.to(state.mean.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        learned = 4.0 * torch.tanh(
            self.association_prior_head(features).squeeze(-1) / 4.0
        )
        manual = confirmed_association_bias * state.confirmed_mask.to(state.mean.dtype)
        bernoulli = torch.zeros_like(learned)
        if self.config.association_bernoulli_gate:
            existence_probability = state.exist_logit.sigmoid()
            if self.config.association_bernoulli_use_predicted_existence:
                if survival_prior_logits is None:
                    raise RuntimeError(
                        "predicted-existence association requires a survival prior"
                    )
                existence_probability = (
                    existence_probability * survival_prior_logits.sigmoid()
                )
            existence_probability = existence_probability.clamp(
                min=max(
                    self.config.association_bernoulli_probability_floor,
                    1.0e-6,
                ),
                max=1.0 - 1.0e-6,
            )
            bernoulli = (
                self.config.association_bernoulli_logit_scale
                * torch.logit(existence_probability)
            )
        return torch.where(
            state.active_mask,
            learned + manual + bernoulli,
            torch.zeros_like(learned),
        )

    def _association_write_pairs(
        self,
        state: RecursiveTrackState,
        prior_query: torch.Tensor,
        prior_covariance: torch.Tensor,
        pair_probabilities: torch.Tensor,
        measurement_padding_mask: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        if not self.config.ambiguity_aware_update:
            return pair_probabilities

        pair_mass = pair_probabilities.sum(-1).clamp(0.0, 1.0)
        conditional = pair_probabilities / pair_mass.unsqueeze(-1).clamp_min(1.0e-8)
        entropy = -(
            conditional * conditional.clamp_min(1.0e-8).log()
        ).sum(-1)
        valid_count = (~measurement_padding_mask).sum(-1).clamp_min(2).to(entropy.dtype)
        normalized_entropy = entropy / valid_count.log().unsqueeze(-1)
        normalized_entropy = torch.where(
            pair_mass > 0.0,
            normalized_entropy.clamp(0.0, 1.0),
            torch.zeros_like(normalized_entropy),
        )
        if pair_probabilities.shape[-1] > 1:
            top_two = conditional.topk(2, dim=-1).values
            margin = (top_two[..., 0] - top_two[..., 1]).clamp(0.0, 1.0)
        elif pair_probabilities.shape[-1] == 1:
            margin = conditional[..., 0].clamp(0.0, 1.0)
        else:
            margin = torch.zeros_like(pair_mass)
        covariance_diag = prior_covariance.diagonal(dim1=-2, dim2=-1)
        features = torch.cat(
            (
                prior_query,
                pair_mass.unsqueeze(-1),
                normalized_entropy.unsqueeze(-1),
                margin.unsqueeze(-1),
                torch.log1p(covariance_diag[..., :2].mean(-1, keepdim=True)),
                torch.log1p(delta_t).unsqueeze(-1),
            ),
            dim=-1,
        )
        reduction = self.ambiguity_write_head(features).squeeze(-1).sigmoid()
        ambiguity = normalized_entropy * (1.0 - margin)
        minimum = self.config.ambiguity_update_min_gate
        write_gate = 1.0 - ambiguity * (1.0 - minimum) * reduction
        write_gate = torch.where(
            state.active_mask, write_gate, torch.ones_like(write_gate)
        )
        return pair_probabilities * write_gate.unsqueeze(-1)
