from __future__ import annotations

import math

import torch
from torch import nn

from track_mt3.config import ExperimentConfig
from track_mt3.data.window import MeasurementWindow, pad_measurement_windows
from track_mt3.losses.criterion import CollectiveAverageCriterion, FrameLoss
from track_mt3.matching.cta import CTAResult, CrossFrameTargetAlignment
from track_mt3.tracking.track_state import TrackState

from .common import MLP, inverse_sigmoid
from .contrastive import ContrastiveClassifier, contrastive_association_loss
from .measurement_encoder import MeasurementEncoder
from .memory_fusion import MemoryFusion
from .normalization import CoordinateNormalizer
from .outputs import FramePrediction, LayerPrediction, ProposalOutputs
from .qtm import QueryTransformationModule
from .spatial_encoding import SpatialEncoding
from .tracking_decoder import TrackingDecoder
from .temporal_encoding import TemporalEncoding


class TrackMT3(nn.Module):
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        model = config.model
        self.normalizer = CoordinateNormalizer(config.simulation.field_of_view)
        self.encoder = MeasurementEncoder(model)
        self.memory_fusion = MemoryFusion(model) if model.memory_frames > 0 else None
        self.streaming_cache_frames = model.streaming_cache_frames
        self.streaming_cache_tokens = model.streaming_cache_tokens_per_frame
        self.cache_temporal_encoding = (
            TemporalEncoding(
                model.streaming_cache_frames,
                model.hidden_dim,
                model.temporal_encoding,
            )
            if model.streaming_cache_frames > 0 else None
        )
        self.decoder = TrackingDecoder(model)
        self.two_stage = model.two_stage
        # Two-stage replaces the learnable detection queries outright, so the
        # parameters are not created at all. Leaving them in place would hand DDP a
        # set of never-used parameters and force find_unused_parameters=True, which
        # 4.11 went to some trouble to switch off.
        if self.two_stage:
            self.detection_queries = None
            self.detection_reference = None
            self.detection_reference_offset = None
            # Selection mechanism, MT3 paper Section IV-B, equations (8)-(9).
            #
            #   m_i = Softmax(FFN1(e_i))        scores
            #   d_i = FFN2(e_i)                 offsets
            #   z~_i = z_{r_i} + d_{r_i}        refined coordinate, r = argsort(m)
            #   o_i = FFN(z~_i)                 object query
            #
            # The paper's ablation makes this the second most valuable component of
            # the architecture: removing it costs 25% GOSPA on Task 2 (3.662 ->
            # 4.587), behind only the intermediate decoder losses.
            #
            # FFN1. The paper specifies a feedforward net; the official code uses a
            # bare Linear (mt3.py:83, obj_classifier). One hidden layer is enough to
            # let the score depend nonlinearly on the embedding, which is what
            # "promising measurement" requires: a measurement is promising because of
            # how it relates to its neighbours, not through any single feature.
            self.proposal_class_head = MLP(
                model.hidden_dim, model.prediction_hidden_dim, 1, 2
            )
            # FFN2, equation (8). Predicts the offset added to the chosen
            # measurement.
            self.proposal_position_head = MLP(
                model.hidden_dim,
                model.prediction_hidden_dim,
                model.output_dim,
                model.prediction_layers,
            )
            # Equation (9), o_i = FFN(z~_i): the object query is generated from the
            # refined coordinate by a network, not taken as a raw encoder output.
            # This was previously a single Linear with no activation, which collapses
            # (9) into an affine map of the sinusoidal ladder and cannot represent
            # anything the ladder does not already encode linearly.
            #
            # Official MT3 widens the ladder to 2*d_model and splits the projection
            # into (query_pos, content). Only the content half is produced here,
            # because the decoder already builds query_pos from the reference with
            # the same ladder it uses for track queries; keeping one mechanism for
            # both query types is worth more than matching the split exactly.
            self.proposal_encoding = SpatialEncoding(
                2 * model.hidden_dim,
                model.output_dim,
                temperature=model.spatial_encoding_temperature,
            )
            self.proposal_to_query = nn.Sequential(
                nn.Linear(2 * model.hidden_dim, model.hidden_dim),
                nn.ReLU(),
                nn.Linear(model.hidden_dim, model.hidden_dim),
                nn.LayerNorm(model.hidden_dim),
            )
            # Start from the measurement itself: a zero offset means the reference
            # point is exactly the measurement coordinate.
            nn.init.zeros_(self.proposal_position_head.layers[-1].weight)
            nn.init.zeros_(self.proposal_position_head.layers[-1].bias)
        else:
            self.detection_queries = nn.Parameter(torch.empty(model.num_detection_queries, model.hidden_dim))
            self.detection_reference = nn.Linear(model.hidden_dim, model.output_dim)
            self.detection_reference_offset = nn.Parameter(
                self._initial_reference_offset(model, config.simulation)
            )
            nn.init.normal_(self.detection_queries, std=0.02)
            nn.init.xavier_uniform_(self.detection_reference.weight)
            nn.init.zeros_(self.detection_reference.bias)
        self.qtm = QueryTransformationModule(model)
        self.contrastive_classifier = (
            ContrastiveClassifier(
                model.hidden_dim,
                temperature=model.contrastive_temperature,
                learnable_temperature=model.contrastive_learnable_temperature,
            )
            if model.contrastive_classifier
            else None
        )

    @staticmethod
    def _initial_reference_offset(model, simulation) -> torch.Tensor:
        """Per-query anchor in inverse-sigmoid space.

        The shared linear projection has a single bias, so without a per-query
        term every anchor starts at sigmoid(0) = 0.5 and all queries compete for
        the centre of the scene.

        Spreading the anchors only helps if they are spread over the region
        where targets actually appear. A grid covering the whole field of view
        pushes most anchors into corners the simulator almost never populates,
        which measurably slows optimization, so "prior" scales the spread by the
        initial target covariance instead.
        """
        count = model.num_detection_queries
        mode = model.detection_reference_init
        if mode == "center":
            return torch.zeros(count, model.output_dim)
        if mode not in ("grid", "prior"):
            raise ValueError(f"unknown detection_reference_init: {mode}")
        divisions = math.ceil(count ** (1.0 / model.output_dim))
        centers = (torch.arange(divisions, dtype=torch.float32) + 0.5) / divisions
        grid = torch.cartesian_prod(*([centers] * model.output_dim))
        if model.output_dim == 1:
            grid = grid.unsqueeze(-1)
        grid = grid[:count]
        if mode == "prior":
            # Shrink the grid towards the centre so it spans roughly two
            # standard deviations of the initial target distribution rather than
            # the whole field of view.
            x_min, x_max, y_min, y_max = simulation.field_of_view
            spans = torch.tensor([x_max - x_min, y_max - y_min], dtype=torch.float32)
            deviations = torch.tensor(
                [
                    float(simulation.initial_position_covariance[index][index]) ** 0.5
                    for index in range(2)
                ],
                dtype=torch.float32,
            )
            scale = (4.0 * deviations / spans).clamp(max=1.0)[: model.output_dim]
            grid = 0.5 + (grid - 0.5) * scale
        return inverse_sigmoid(grid)

    def detection_references(self) -> torch.Tensor:
        """Anchor of every detection query on the unit interval."""
        return (
            self.detection_reference(self.detection_queries)
            + self.detection_reference_offset
        ).sigmoid()

    def _proposals(
        self,
        memory: torch.Tensor,
        unit_measurements: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        time_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encoder-scored measurement proposals for the two-stage decoder.

        Returns the top-k reference points, the matching query content, the full
        per-measurement scores and unit-interval coordinates that the auxiliary
        proposal loss supervises, and a boolean mask marking which of the k slots
        landed on a genuinely eligible measurement.

        The reference points are detached, exactly as in official MT3
        (mt3.py:181, ``topk_coords_unact.detach()``): the decoder is given a starting
        point, not a differentiable path back into the selection.

        With ``model.proposal_pool_frames > 0`` only measurements from the last that
        many frames may be selected, because a reference taken from the start of the
        window is up to ``(window_size - 1) * dt`` stale. The eligible pool is
        regularly smaller than ``num_detection_queries`` -- it holds 0-9 measurements
        during curriculum warmup -- so ``valid`` is returned instead of assuming all
        k slots are usable. A padded slot would supply a reference of
        ``inverse_sigmoid(0)``, so the caller must drop the invalid slots rather than
        mask them downstream.
        """
        # Equation (8), z~_i = z_{r_i} + d_{r_i}. The paper adds the offset in
        # measurement space; this adds it in inverse-sigmoid space, as the official
        # code does (mt3.py:174). The sigmoid then keeps every reference strictly
        # inside the field of view, which a raw coordinate sum does not guarantee.
        base = inverse_sigmoid(unit_measurements.clamp(1e-4, 1.0 - 1e-4))
        # Equation (8), m_i. The paper applies a Softmax over the n measurements;
        # raw logits are kept here, as in the official code, because Softmax is
        # monotonic and therefore cannot change the top-k, while logits are what the
        # BCE proposal loss consumes.
        scores = self.proposal_class_head(memory)
        coordinates = self.proposal_position_head(memory) + base
        # Padded slots must never win the top-k. They are still scored so the tensor
        # shape is uniform, but pushed below any real measurement. The returned
        # tensor keeps this masking because the proposal loss consumes it.
        scores = scores.masked_fill(memory_padding_mask.unsqueeze(-1), -1e8)
        eligible = ~memory_padding_mask
        pool_frames = self.config.model.proposal_pool_frames
        if pool_frames > 0 and time_indices is not None:
            recent = time_indices >= (self.config.model.window_size - pool_frames)
            restricted = eligible & recent
            # A row can hold no recent measurement at all: a low-clutter curriculum
            # scene can leave the last frames empty. Falling back to the
            # unrestricted pool for those rows keeps a stale reference, which still
            # beats an all-masked query row.
            eligible = torch.where(
                restricted.any(dim=1, keepdim=True), restricted, eligible
            )
        selection_scores = scores[..., 0].masked_fill(~eligible, -1e8)
        count = min(self.config.model.num_detection_queries, memory.shape[1])
        indices = selection_scores.topk(count, dim=1).indices
        gathered = coordinates.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, coordinates.shape[-1])
        )
        references = gathered.detach().sigmoid()
        content = self.proposal_to_query(self.proposal_encoding(references))
        valid = eligible.gather(1, indices)
        return references, content, scores, coordinates.sigmoid(), valid

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def empty_tracks(self) -> TrackState:
        m = self.config.model
        return TrackState.empty(
            m.hidden_dim,
            self.device,
            max_tracks=m.max_tracks if m.track_mamba_enabled else 0,
            track_d_inner=m.hidden_dim if m.track_mamba_enabled else 0,
            track_d_state=m.track_mamba_state_dim if m.track_mamba_enabled else 0,
            memory_size=m.memory_frames * m.memory_tokens_per_frame if self.memory_fusion is not None else 0,
            cache_frames=m.streaming_cache_frames,
            cache_tokens_per_frame=m.streaming_cache_tokens_per_frame,
        )

    def _append_streaming_cache(
        self,
        current: torch.Tensor,
        current_mask: torch.Tensor,
        current_positions: torch.Tensor | None,
        current_ids: torch.Tensor,
        tracks: list[TrackState],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Append one encoded frame and expose the cached 20-frame memory.

        Historical values are copied from the state without passing through the
        encoder again.  Only decoder queries attend across frames.  Spatial
        signatures are stored with the values, while relative frame age is
        regenerated after every shift so the newest slot is always F-1.
        """
        frames = self.streaming_cache_frames
        capacity = self.streaming_cache_tokens
        batch, _, hidden = current.shape
        spatial = current_positions
        if spatial is None:
            spatial = current.new_zeros(current.shape)
        # pad_measurement_windows keeps real tokens contiguous at the beginning,
        # so an entire DDP local batch can be appended without a per-sample loop.
        copied = min(current.shape[1], capacity)
        slot = current.new_zeros((batch, 1, capacity, hidden))
        slot_positions = current.new_zeros((batch, 1, capacity, hidden))
        slot_mask = torch.ones(
            (batch, 1, capacity), dtype=torch.bool, device=current.device
        )
        slot_ids = torch.full(
            (batch, 1, capacity), -2, dtype=torch.long, device=current.device
        )
        slot[:, 0, :copied] = current[:, :copied]
        slot_positions[:, 0, :copied] = spatial[:, :copied]
        slot_mask[:, 0, :copied] = current_mask[:, :copied]
        slot_ids[:, 0, :copied] = current_ids[:, :copied]

        old = torch.cat([track.measurement_cache for track in tracks], dim=0)
        old_mask = torch.cat([track.measurement_cache_mask for track in tracks], dim=0)
        old_positions = torch.cat(
            [track.measurement_cache_positions for track in tracks], dim=0
        )
        old_ids = torch.cat([track.measurement_cache_ids for track in tracks], dim=0)
        new = torch.cat((old[:, 1:], slot), dim=1)
        new_mask = torch.cat((old_mask[:, 1:], slot_mask), dim=1)
        new_positions = torch.cat((old_positions[:, 1:], slot_positions), dim=1)
        new_ids = torch.cat((old_ids[:, 1:], slot_ids), dim=1)
        for index, track in enumerate(tracks):
            track.measurement_cache = new[index : index + 1]
            track.measurement_cache_mask = new_mask[index : index + 1]
            track.measurement_cache_positions = new_positions[index : index + 1]
            track.measurement_cache_ids = new_ids[index : index + 1]

        memory = new.flatten(1, 2)
        mask = new_mask.flatten(1, 2)
        spatial_positions = new_positions.flatten(1, 2)
        measurement_ids = new_ids.flatten(1, 2)
        age_indices = torch.arange(frames, device=current.device).repeat_interleave(capacity)
        age_indices = age_indices.unsqueeze(0).expand(batch, -1)
        memory_positions = spatial_positions + self.cache_temporal_encoding(age_indices)

        # Key padding masks prevent padded slots from changing attention values,
        # but dense attention still performs arithmetic over all F*C=640 slots.
        # Pack only real tokens to the longest row in this local batch.  This is
        # numerically equivalent and especially valuable during curriculum warmup.
        lengths = (~mask).sum(dim=1)
        packed_memory = nn.utils.rnn.pad_sequence(
            [row[~row_mask] for row, row_mask in zip(memory, mask)],
            batch_first=True,
        )
        packed_positions = nn.utils.rnn.pad_sequence(
            [row[~row_mask] for row, row_mask in zip(memory_positions, mask)],
            batch_first=True,
        )
        packed_ids = nn.utils.rnn.pad_sequence(
            [row[~row_mask] for row, row_mask in zip(measurement_ids, mask)],
            batch_first=True,
            padding_value=-2,
        )
        packed_mask = (
            torch.arange(packed_memory.shape[1], device=current.device).unsqueeze(0)
            >= lengths.unsqueeze(1)
        )
        return packed_memory, packed_mask, packed_positions, packed_ids

    def _auxiliary_assignments(
        self,
        prediction: FramePrediction,
        tracks: TrackState,
        window: MeasurementWindow,
        alignment: CrossFrameTargetAlignment,
        final_assignment: CTAResult,
    ) -> list[CTAResult]:
        strategy = self.config.loss.auxiliary_matching
        if strategy == "shared":
            return [final_assignment] * len(prediction.auxiliary)
        if strategy != "per_layer":
            raise ValueError(f"unknown auxiliary_matching strategy: {strategy}")
        assignments = []
        # The inherited (track-query) half of the alignment does not depend on the
        # layer's own predictions, so it is computed once instead of once per layer.
        inherited = alignment.inherited_matches(
            tracks, window.target_ids, len(prediction.positions)
        )
        for layer in prediction.auxiliary:
            layer_prediction = FramePrediction(
                normalized_positions=layer.normalized_positions,
                positions=layer.positions,
                logits=layer.logits,
                hidden=prediction.hidden,
                num_track_queries=prediction.num_track_queries,
            )
            assignments.append(
                alignment(
                    layer_prediction,
                    tracks,
                    window.target_positions,
                    window.target_ids,
                    inherited=inherited,
                    include_query_target_ids=False,
                )
            )
        return assignments

    def forward(
        self,
        clips: list[list[MeasurementWindow]],
        *,
        criterion: CollectiveAverageCriterion,
        teacher_forcing_probability: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute one local training batch through the DDP forward boundary."""
        if not clips:
            raise ValueError("at least one clip is required")
        clip_length = len(clips[0])
        if clip_length == 0 or any(len(clip) != clip_length for clip in clips):
            raise ValueError("all clips must have the same non-zero length")
        tracks = [self.empty_tracks() for _ in clips]
        clip_losses: list[list[FrameLoss]] = [[] for _ in clips]
        contrastive_losses: list[torch.Tensor] = []
        proposal_losses: list[torch.Tensor] = []
        alignments = [CrossFrameTargetAlignment(self.config.loss) for _ in clips]
        # The proposal set carries no track inheritance, so it needs a plain
        # Hungarian pass rather than the CTA state used for the decoder queries.
        proposal_alignment = CrossFrameTargetAlignment(self.config.loss)
        if teacher_forcing_probability is None:
            teacher_forcing_probability = float(
                self.config.model.teacher_force_matched_queries
            )
        if not 0.0 <= teacher_forcing_probability <= 1.0:
            raise ValueError("teacher_forcing_probability must be in [0, 1]")
        for time_index in range(clip_length):
            if not self.config.model.propagate_tracks:
                tracks = [self.empty_tracks() for _ in clips]
            windows = [clip[time_index] for clip in clips]
            predictions, contrastive, proposals = self.forward_windows(
                windows, tracks, return_contrastive=True
            )
            if contrastive is not None:
                contrastive_losses.append(contrastive)
            if proposals is not None:
                for index, window in enumerate(windows):
                    proposal_losses.append(
                        self._proposal_loss(
                            proposals, index, window, criterion, proposal_alignment
                        )
                    )
            next_tracks = []
            for index, (window, prediction, track) in enumerate(
                zip(windows, predictions, tracks)
            ):
                assignment = alignments[index](
                    prediction, track, window.target_positions, window.target_ids
                )
                clip_losses[index].append(
                    criterion.frame_loss(
                        prediction,
                        window.target_positions,
                        assignment,
                        self._auxiliary_assignments(
                            prediction, track, window, alignments[index], assignment
                        ),
                        target_ages=window.target_ages if len(window.target_ages) else None,
                    )
                )
                if self.config.model.propagate_tracks:
                    # The QTM submodules only receive gradient through queries
                    # that survive selection: the output projection needs a
                    # non-empty selection, and the temporal network needs a
                    # track propagated from the previous window. Forcing the
                    # first two transitions of the first clip keeps the set of
                    # DDP reduction buckets identical on every rank, so
                    # scheduled sampling cannot desynchronise the collective
                    # order once teacher forcing anneals towards zero.
                    guaranteed = index == 0 and time_index < 2
                    teacher_force = guaranteed or bool(
                        torch.rand((), device=self.device)
                        < teacher_forcing_probability
                    )
                    next_track, _ = self.qtm(
                        prediction,
                        track,
                        query_target_ids=assignment.query_target_ids,
                        query_track_ids=assignment.query_target_ids,
                        teacher_force=teacher_force,
                    )
                    next_tracks.append(next_track)
            if self.config.model.propagate_tracks:
                tracks = next_tracks
        losses = [criterion.collective_average(values) for values in clip_losses]
        total = torch.stack([loss.total for loss in losses]).mean()
        if contrastive_losses:
            contrastive_total = torch.stack(contrastive_losses).mean()
            total = total + self.config.loss.contrastive_weight * contrastive_total
        else:
            contrastive_total = total.new_zeros(())
        if proposal_losses:
            proposal_total = torch.stack(proposal_losses).mean()
            total = total + self.config.loss.proposal_weight * proposal_total
        else:
            proposal_total = total.new_zeros(())
        return (
            total,
            torch.stack([loss.localization for loss in losses]).mean().detach(),
            torch.stack([loss.confidence for loss in losses]).mean().detach(),
            torch.stack([loss.auxiliary for loss in losses]).mean().detach(),
            contrastive_total.detach(),
            proposal_total.detach(),
        )

    def _proposal_loss(
        self,
        proposals: ProposalOutputs,
        index: int,
        window: MeasurementWindow,
        criterion: CollectiveAverageCriterion,
        alignment: CrossFrameTargetAlignment,
    ) -> torch.Tensor:
        """Auxiliary supervision of the encoder's per-measurement scores.

        Without this the top-k selection in ``_proposals`` is unsupervised: nothing
        tells the encoder which measurements belong to targets, so the ranking that
        chooses the decoder's starting points would be arbitrary. Official MT3 adds
        the same term at weight 1.0 (``enc_outputs`` in mt3.py:220).
        """
        normalized, positions, logits = proposals.row(index)
        prediction = FramePrediction(
            normalized_positions=normalized,
            positions=positions,
            logits=logits,
            # The proposal head consumes no hidden state downstream; the tensor only
            # has to exist for the dataclass.
            hidden=positions,
            num_track_queries=0,
        )
        assignment = alignment(
            prediction,
            TrackState.empty(self.config.model.hidden_dim, self.device),
            window.target_positions,
            window.target_ids,
            include_query_target_ids=False,
        )
        return criterion.frame_loss(
            prediction, window.target_positions, assignment
        ).total

    def forward_windows(
        self,
        windows: list[MeasurementWindow],
        tracks: list[TrackState],
        *,
        return_contrastive: bool = False,
    ) -> (
        list[FramePrediction]
        | tuple[list[FramePrediction], torch.Tensor | None, ProposalOutputs | None]
    ):
        """Vectorized window forward for a batch of independent clips.

        With ``return_contrastive`` the memory association loss and, when two-stage
        is enabled, the encoder proposal scores are returned alongside the
        predictions. Inference paths keep the plain list, so they never pay for the
        pairwise score matrix.
        """
        if len(windows) != len(tracks) or not windows:
            raise ValueError("windows and tracks must have the same non-zero length")
        measurements, times, memory_mask, measurement_ids = pad_measurement_windows(
            windows
        )
        measurements = self.normalizer.normalize_measurements(
            measurements.to(self.device)
        )
        times = times.to(self.device)
        memory_mask = memory_mask.to(self.device)
        memory_positions = self.encoder.spatial_positions(measurements + 0.5)
        memory = self.encoder(measurements, times, memory_mask, memory_positions)

        if self.cache_temporal_encoding is not None:
            memory, memory_mask, memory_positions, measurement_ids = (
                self._append_streaming_cache(
                    memory,
                    memory_mask,
                    memory_positions,
                    measurement_ids.to(self.device),
                    tracks,
                )
            )

        # DFSMN-style fusion with the previous frames' measurement encodings.
        if self.memory_fusion is not None:
            history = torch.cat([t.memory for t in tracks], dim=0)
            fused, new_memory = self.memory_fusion(memory, history, memory_mask)
            memory = fused
            for i, t in enumerate(tracks):
                t.memory = new_memory[i:i + 1]

        contrastive = None
        if return_contrastive and self.contrastive_classifier is not None:
            contrastive = contrastive_association_loss(
                self.contrastive_classifier(memory, memory_mask),
                measurement_ids.to(self.device),
                memory_mask,
            )
        proposals: ProposalOutputs | None = None
        if self.two_stage:
            references, content, scores, coordinates, valid = self._proposals(
                memory, measurements + 0.5, memory_mask, times
            )
            detection_batch = content
            reference_batch = references
            proposals = ProposalOutputs(
                logits=scores,
                normalized_positions=coordinates,
                positions=self.normalizer.from_unit_interval(coordinates),
                padding_mask=memory_mask,
            )
            # Rows keep only their eligible proposals, so a row whose pool held
            # fewer than k measurements contributes fewer detection queries rather
            # than queries anchored on padding. The lengths already vary with the
            # track count, so pad_sequence and query_mask below absorb this.
            query_parts = [
                torch.cat(
                    (track.to(self.device).features, detection_batch[index][valid[index]]),
                    dim=0,
                )
                for index, track in enumerate(tracks)
            ]
            reference_parts = [
                torch.cat(
                    (track.to(self.device).references, reference_batch[index][valid[index]]),
                    dim=0,
                )
                for index, track in enumerate(tracks)
            ]
        else:
            detection = self.detection_queries
            detection_references = self.detection_references()
            query_parts = [
                torch.cat((track.to(self.device).features, detection), dim=0)
                for track in tracks
            ]
            reference_parts = [
                torch.cat((track.to(self.device).references, detection_references), dim=0)
                for track in tracks
            ]
        lengths = torch.tensor(
            [len(part) for part in query_parts], device=self.device
        )
        queries = nn.utils.rnn.pad_sequence(query_parts, batch_first=True)
        references = nn.utils.rnn.pad_sequence(reference_parts, batch_first=True)
        query_mask = (
            torch.arange(queries.shape[1], device=self.device).unsqueeze(0)
            >= lengths.unsqueeze(1)
        )
        states, logits, hidden, attention = self.decoder(
            queries,
            memory,
            references,
            memory_mask,
            query_mask,
            memory_positions,
        )
        predictions = []
        for batch_index, (track, length) in enumerate(zip(tracks, lengths.tolist())):
            layers = tuple(
                LayerPrediction(
                    normalized_positions=state[batch_index, :length],
                    positions=self.normalizer.from_unit_interval(
                        state[batch_index, :length]
                    ),
                    logits=layer_logits[batch_index, :length],
                )
                for state, layer_logits in zip(states, logits)
            )
            final = layers[-1]
            predictions.append(
                FramePrediction(
                    normalized_positions=final.normalized_positions,
                    positions=final.positions,
                    logits=final.logits,
                    hidden=hidden[batch_index, :length],
                    num_track_queries=len(track),
                    auxiliary=(
                        layers[:-1] if self.config.model.auxiliary_loss else ()
                    ),
                    attention_maps=attention[
                        :, batch_index, :, :length
                    ],
                )
            )
        if return_contrastive:
            return predictions, contrastive, proposals
        return predictions

    def encode_window(
        self, window: MeasurementWindow
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        measurements = window.measurements.to(self.device)
        times = window.time_indices.to(self.device)
        if len(measurements) == 0:
            measurements = torch.zeros((1, 2), device=self.device)
            times = torch.zeros((1,), dtype=torch.long, device=self.device)
        normalized = self.normalizer.normalize_measurements(measurements).unsqueeze(0)
        spatial_positions = self.encoder.spatial_positions(normalized + 0.5)
        memory = self.encoder(
            normalized,
            times.unsqueeze(0),
            spatial_positions=spatial_positions,
        )
        mask = torch.zeros((1, measurements.shape[0]), dtype=torch.bool, device=self.device)
        return memory, mask, spatial_positions, normalized

    def forward_window(self, window: MeasurementWindow, tracks: TrackState | None = None) -> FramePrediction:
        if tracks is None:
            tracks = self.empty_tracks()
        tracks = tracks.to(self.device)
        tracks.validate()
        memory, mask, memory_positions, normalized = self.encode_window(window)
        if self.cache_temporal_encoding is not None:
            ids = window.measurement_ids.to(self.device)
            if len(window.measurements) == 0:
                ids = torch.full((1,), -2, dtype=torch.long, device=self.device)
            memory, mask, memory_positions, _ = self._append_streaming_cache(
                memory,
                mask,
                memory_positions,
                ids.unsqueeze(0),
                [tracks],
            )
        if self.memory_fusion is not None:
            history = tracks.memory if tracks.memory is not None else None
            fused, new_memory = self.memory_fusion(memory, history, mask)
            memory = fused
            tracks.memory = new_memory
        if self.two_stage:
            # encode_window substitutes a single zero measurement for an empty
            # window, so the time indices have to follow the same substitution to
            # stay aligned with the memory.
            times = window.time_indices.to(self.device)
            if len(window.measurements) == 0:
                times = torch.zeros((1,), dtype=torch.long, device=self.device)
            detection_references, detection, _, _, valid = self._proposals(
                memory, normalized + 0.5, mask, times.unsqueeze(0)
            )
            row = valid.squeeze(0)
            detection = detection.squeeze(0)[row]
            detection_references = detection_references.squeeze(0)[row]
        else:
            detection = self.detection_queries
            detection_references = self.detection_references()
        queries = torch.cat((tracks.features, detection), dim=0).unsqueeze(0)
        references = torch.cat((tracks.references, detection_references), dim=0).unsqueeze(0)
        states, logits, hidden, attention = self.decoder(
            queries,
            memory,
            references,
            mask,
            memory_positions=memory_positions,
        )
        layers = tuple(
            LayerPrediction(
                normalized_positions=state.squeeze(0),
                positions=self.normalizer.from_unit_interval(state.squeeze(0)),
                logits=layer_logits.squeeze(0),
            )
            for state, layer_logits in zip(states, logits)
        )
        final = layers[-1]
        auxiliary = layers[:-1] if self.config.model.auxiliary_loss else ()
        return FramePrediction(
            normalized_positions=final.normalized_positions,
            positions=final.positions,
            logits=final.logits,
            hidden=hidden.squeeze(0),
            num_track_queries=len(tracks),
            auxiliary=auxiliary,
            attention_maps=attention[:, 0],
        )

    def forward_clip(
        self,
        windows: list[MeasurementWindow],
        *,
        criterion: CollectiveAverageCriterion | None = None,
        alignment: CrossFrameTargetAlignment | None = None,
        teacher_forcing: bool = False,
    ) -> tuple[list[FramePrediction], TrackState, FrameLoss | None, list[CTAResult]]:
        tracks = self.empty_tracks()
        predictions: list[FramePrediction] = []
        assignments: list[CTAResult] = []
        frame_losses: list[FrameLoss] = []
        if criterion is not None and alignment is None:
            alignment = CrossFrameTargetAlignment(self.config.loss)
        for window in windows:
            if not self.config.model.propagate_tracks:
                tracks = self.empty_tracks()
            prediction = self.forward_window(window, tracks)
            predictions.append(prediction)
            if alignment is not None:
                assignment = alignment(prediction, tracks, window.target_positions, window.target_ids)
                assignments.append(assignment)
                query_ids = assignment.query_target_ids
                if criterion is not None:
                    frame_losses.append(
                        criterion.frame_loss(
                            prediction,
                            window.target_positions,
                            assignment,
                            self._auxiliary_assignments(
                                prediction, tracks, window, alignment, assignment
                            ),
                            target_ages=window.target_ages if len(window.target_ages) else None,
                        )
                    )
            else:
                query_ids = None
            if self.config.model.propagate_tracks:
                tracks, _ = self.qtm(
                    prediction,
                    tracks,
                    query_target_ids=query_ids,
                    query_track_ids=query_ids,
                    teacher_force=teacher_forcing,
                )
        cal = criterion.collective_average(frame_losses) if criterion is not None else None
        return predictions, tracks, cal, assignments
