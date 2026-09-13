from dataclasses import replace

import torch

from track_mt3.config import ExperimentConfig, LossConfig, ModelConfig, SimulationConfig
from track_mt3.data.simulator import MultiTargetSimulator
from track_mt3.data.window import build_sliding_windows
from track_mt3.losses import CollectiveAverageCriterion
from track_mt3.matching import CrossFrameTargetAlignment
from track_mt3.models import FramePrediction, TrackMT3
from track_mt3.models.common import MLP
from track_mt3.models.outputs import LayerPrediction
from track_mt3.tracking import TrackState


def small_config() -> ExperimentConfig:
    return ExperimentConfig(
        simulation=SimulationConfig(initial_targets=2, birth_rate=0.0, clutter_rate=2.0),
        model=ModelConfig(
            window_size=4,
            hidden_dim=32,
            num_heads=4,
            encoder_layers=2,
            decoder_layers=2,
            feedforward_dim=64,
            num_detection_queries=4,
            prediction_hidden_dim=16,
            max_tracks=4,
            dropout=0.0,
        ),
    )


def make_prediction(positions, logits, num_tracks):
    positions = torch.tensor(positions, dtype=torch.float32)
    logits = torch.tensor(logits, dtype=torch.float32).reshape(-1, 1)
    normalized = positions / 20.0 + 0.5
    return FramePrediction(normalized, positions, logits, torch.zeros(len(positions), 8), num_tracks)


def test_model_forward_shapes_and_auxiliary_layers():
    config = small_config()
    model = TrackMT3(config)
    frames = MultiTargetSimulator(config.simulation, 5).simulate(5)
    window = build_sliding_windows(frames, config.model.window_size)[0]
    output = model.forward_window(window)
    assert output.positions.shape == (4, 2)
    assert output.logits.shape == (4, 1)
    assert output.hidden.shape == (4, 32)
    assert len(output.auxiliary) == 1
    assert output.attention_maps.shape[0] == 2


def test_cta_inherits_tracks_and_matches_only_newborn_targets():
    prediction = make_prediction(
        positions=[[1.0, 0.0], [-1.0, 0.0], [5.1, 5.0], [-8.0, -8.0]],
        logits=[4.0, 4.0, 4.0, -4.0],
        num_tracks=2,
    )
    tracks = TrackState(
        features=torch.zeros(2, 8),
        references=torch.zeros(2, 2),
        target_ids=torch.tensor([10, 20]),
        track_ids=torch.tensor([10, 20]),
        ages=torch.ones(2, dtype=torch.long),
    )
    targets = torch.tensor([[-1.1, 0.0], [5.0, 5.0], [1.1, 0.0]])
    target_ids = torch.tensor([20, 30, 10])
    result = CrossFrameTargetAlignment(LossConfig())(prediction, tracks, targets, target_ids)
    assert result.inherited_count == 2
    assert result.newborn_count == 1
    assert result.query_target_ids.tolist() == [10, 20, 30, -1]
    assert set(zip(result.prediction_indices.tolist(), result.target_indices.tolist())) == {(0, 2), (1, 0), (2, 1)}


def test_frame_loss_and_cal_match_hand_calculation():
    """Both normalisation conventions are pinned to a hand calculation.

    Two queries at (0,0) and (3,0), one target at (1,0), both logits zero. The
    match is query 0, so |dx|+|dy| = 1 over 1 target and 2 coordinates, and the
    BCE is log(2) per query with one positive and one negative.

    "sum_per_target" divides both sums by the target count: localization
    1/1 = 1.0 and confidence 2*log(2)/1. "mean" is official MT3: localization
    1/(1 pair * 2 coords) = 0.5 and confidence 2*log(2)/2 queries = log(2). The
    ratio between the two terms therefore differs by 2.5x between conventions at
    identical weights, which is why the balance had to be documented rather than
    absorbed into localization_weight.
    """
    prediction = make_prediction([[0.0, 0.0], [3.0, 0.0]], [0.0, 0.0], num_tracks=0)
    tracks = TrackState.empty(8, "cpu")
    targets = torch.tensor([[1.0, 0.0]])
    target_ids = torch.tensor([4])
    log2 = torch.log(torch.tensor(2.0))
    expected = {
        "sum_per_target": (torch.tensor(1.0), 2.0 * log2),
        "mean": (torch.tensor(0.5), log2),
    }
    for reduction, (want_localization, want_confidence) in expected.items():
        config = LossConfig(reduction=reduction)
        alignment = CrossFrameTargetAlignment(config)(prediction, tracks, targets, target_ids)
        criterion = CollectiveAverageCriterion(config)
        loss = criterion.frame_loss(prediction, targets, alignment)
        assert torch.isclose(loss.localization, want_localization), reduction
        assert torch.isclose(loss.confidence, want_confidence), reduction
        average = criterion.collective_average([loss, loss])
        assert torch.isclose(average.total, loss.total), reduction


def test_qtm_thresholds_and_zero_bias_prediction_mapping():
    config = small_config()
    model = TrackMT3(config)
    previous = TrackState(
        features=torch.randn(1, 32),
        references=torch.tensor([[0.5, 0.5]]),
        target_ids=torch.tensor([7]),
        track_ids=torch.tensor([7]),
        ages=torch.tensor([2]),
    )
    positions = torch.zeros(5, 2)
    normalized = torch.full((5, 2), 0.5)
    logits = torch.tensor([[1.0], [2.0], [-2.0], [2.0], [-2.0]])
    prediction = FramePrediction(normalized, positions, logits, torch.randn(5, 32), num_track_queries=1)
    query_ids = torch.tensor([7, 8, -1, 9, -1])
    state, selected = model.qtm(prediction, previous, query_ids, query_ids)
    assert selected.tolist() == [0, 1, 3]
    assert state.target_ids.tolist() == [7, 8, 9]
    assert all(layer.bias is None for layer in model.qtm.prediction_to_query.layers)


def test_qtm_all_queries_rejected_returns_empty_state():
    config = small_config()
    model = TrackMT3(config)
    previous = TrackState.empty(32, "cpu")
    prediction = FramePrediction(
        normalized_positions=torch.full((4, 2), 0.5),
        positions=torch.zeros(4, 2),
        logits=torch.full((4, 1), -10.0),
        hidden=torch.randn(4, 32),
        num_track_queries=0,
    )
    state, selected = model.qtm(prediction, previous)
    assert len(state) == 0
    assert state.features.shape == (0, 32)
    assert selected.shape == (0,)


def test_qtm_teacher_forcing_is_controlled_by_the_caller_not_the_training_flag():
    """Teacher forcing must not depend on ``Module.training``.

    A validation pass runs under ``eval()`` but still needs a teacher-forced
    loss that is comparable with the training objective. Inference paths stay
    free-running because they never supply target identities.
    """
    config = small_config()
    config.model.teacher_force_matched_queries = True
    model = TrackMT3(config)
    previous = TrackState.empty(32, "cpu")
    prediction = FramePrediction(
        normalized_positions=torch.full((4, 2), 0.5),
        positions=torch.zeros(4, 2),
        logits=torch.full((4, 1), -10.0),
        hidden=torch.randn(4, 32),
        num_track_queries=0,
    )
    query_ids = torch.tensor([11, -1, 12, -1])
    for mode in (model.train, model.eval):
        mode()
        state, selected = model.qtm(prediction, previous, query_ids, query_ids)
        assert selected.tolist() == [0, 2]
        assert state.target_ids.tolist() == [11, 12]
    # Without target identities the module falls back to confidence gating, so
    # the sub-threshold logits above select nothing.
    model.eval()
    state, selected = model.qtm(prediction, previous)
    assert len(state) == 0
    assert len(selected) == 0


def test_qtm_teacher_force_override_supports_scheduled_sampling():
    config = small_config()
    config.model.teacher_force_matched_queries = True
    model = TrackMT3(config).train()
    previous = TrackState.empty(32, "cpu")
    prediction = FramePrediction(
        normalized_positions=torch.full((2, 2), 0.5),
        positions=torch.zeros(2, 2),
        logits=torch.full((2, 1), -10.0),
        hidden=torch.randn(2, 32),
        num_track_queries=0,
    )
    query_ids = torch.tensor([11, -1])
    state, selected = model.qtm(
        prediction,
        previous,
        query_ids,
        query_ids,
        teacher_force=False,
    )
    assert len(state) == 0
    assert len(selected) == 0


def test_full_clip_backward_reaches_detection_queries():
    config = small_config()
    model = TrackMT3(config)
    frames = MultiTargetSimulator(config.simulation, 11).simulate(7)
    windows = build_sliding_windows(frames, config.model.window_size)
    criterion = CollectiveAverageCriterion(config.loss)
    _, _, loss, assignments = model.forward_clip(windows, criterion=criterion)
    assert loss is not None and len(assignments) == len(windows)
    loss.total.backward()
    assert model.detection_queries.grad is not None
    assert torch.isfinite(model.detection_queries.grad).all()


def test_spatial_encoding_separates_measurements_within_one_frame():
    """Same-frame measurements must be distinguishable to attention.

    With only ``Linear(2, hidden_dim)`` the spatial signature is rank two, so
    measurements sharing a timestamp collapse onto nearly the same key and
    cross-attention degenerates into a uniform average over the memory. The
    sinusoidal ladder must dominate that residual variation.
    """
    from track_mt3.models.spatial_encoding import SpatialEncoding

    encoder = SpatialEncoding(32, 2, temperature=20.0)
    close = encoder(torch.tensor([[0.50, 0.50]]))
    nudged = encoder(torch.tensor([[0.51, 0.50]]))
    far = encoder(torch.tensor([[0.05, 0.90]]))
    near_distance = float((close - nudged).abs().mean())
    far_distance = float((close - far).abs().mean())
    assert near_distance < far_distance
    # A usable attention metric needs a wide dynamic range between nearby and
    # distant points, not merely a strict ordering.
    assert far_distance > 10.0 * near_distance


def test_spatial_encoding_is_shared_between_memory_keys_and_reference_points():
    """Queries and keys must live in one embedding space to be comparable."""
    config = small_config()
    config.model.spatial_encoding = "sinusoidal"
    model = TrackMT3(config)
    assert model.encoder.spatial_encoding is not None
    assert model.decoder.reference_encoding is not None
    torch.testing.assert_close(
        model.encoder.spatial_encoding.divisor,
        model.decoder.reference_encoding.divisor,
    )
    point = torch.tensor([[[0.25, 0.75]]])
    torch.testing.assert_close(
        model.encoder.spatial_encoding(point),
        model.decoder.reference_encoding(point),
    )


def test_spatial_encoding_switches_are_independently_ablatable():
    """The memory-key encoding and the query conditioning are separate knobs.

    They address different failure modes, so the whole 2x2 grid must be
    reachable in order to attribute any improvement to the right mechanism.
    """
    config = small_config()
    frames = MultiTargetSimulator(config.simulation, 0).simulate(config.model.window_size)
    window = build_sliding_windows(frames, config.model.window_size)[0]
    for encoding, reference in (
        ("none", "none"),
        ("sinusoidal", "none"),
        ("none", "once"),
        ("sinusoidal", "once"),
        ("none", "per_layer"),
        ("sinusoidal", "per_layer"),
    ):
        config.model.spatial_encoding = encoding
        config.model.reference_query_embedding = reference
        model = TrackMT3(config)
        assert (model.encoder.spatial_encoding is not None) == (encoding == "sinusoidal")
        assert (model.decoder.reference_encoding is not None) == (reference != "none")
        prediction = model.forward_window(window)
        assert prediction.logits.shape == (config.model.num_detection_queries, 1)


def test_detection_anchors_spread_without_leaving_the_target_prior():
    """Anchors must be spread, but only over the region targets occupy.

    A grid across the whole field of view puts most anchors where the simulator
    almost never places a target, which slows optimization instead of helping.
    """
    config = small_config()
    spreads = {}
    for mode in ("center", "prior", "grid"):
        config.model.detection_reference_init = mode
        model = TrackMT3(config)
        references = model.detection_references().detach()
        assert references.shape == (
            config.model.num_detection_queries,
            config.model.output_dim,
        )
        assert bool(((references > 0.0) & (references < 1.0)).all())
        world = model.normalizer.from_unit_interval(references)
        spreads[mode] = float(world.abs().max())
    deviation = float(config.simulation.initial_position_covariance[0][0]) ** 0.5
    half_span = (
        config.simulation.field_of_view[1] - config.simulation.field_of_view[0]
    ) / 2.0
    assert spreads["center"] < spreads["prior"] < spreads["grid"]
    # "prior" must stay inside a few standard deviations of the target prior,
    # while "grid" is what reaches the unpopulated corners.
    assert spreads["prior"] < 4.0 * deviation
    assert spreads["grid"] > 0.5 * half_span


def test_unknown_detection_reference_init_is_rejected():
    config = small_config()
    config.model.detection_reference_init = "spiral"
    try:
        TrackMT3(config)
    except ValueError as error:
        assert "spiral" in str(error)
    else:
        raise AssertionError("expected an unknown initialization to be rejected")


def test_contrastive_loss_is_finite_on_degenerate_windows():
    """Masked pairs hold -inf, and 0 * -inf is NaN.

    Every path has to neutralise the blocked scores, including the branch that
    returns early when a window has no positive pair at all: a window can be
    entirely clutter, or each target may be detected exactly once. A single NaN
    on one DDP rank kills the whole run, and these windows do occur in the
    simulator.
    """
    from track_mt3.models.contrastive import (
        ContrastiveClassifier,
        contrastive_association_loss,
    )

    torch.manual_seed(0)
    head = ContrastiveClassifier(32)
    batch, count, dim = 2, 6, 32
    memory = torch.randn(batch, count, dim, requires_grad=True)
    no_padding = torch.zeros(batch, count, dtype=torch.bool)
    trailing_padding = torch.zeros(batch, count, dtype=torch.bool)
    trailing_padding[:, 1:] = True
    ragged = torch.zeros(batch, count, dtype=torch.bool)
    ragged[0, 3:] = True

    cases = {
        "two targets plus clutter": (
            torch.tensor([[0, 0, 1, 1, -1, -1], [0, 0, 1, 1, -1, -1]]),
            no_padding,
        ),
        "all clutter": (torch.full((batch, count), -1), no_padding),
        "all padding": (torch.full((batch, count), -2), torch.ones(batch, count, dtype=torch.bool)),
        "one valid measurement per row": (
            torch.tensor([[0, -2, -2, -2, -2, -2]] * batch),
            trailing_padding,
        ),
        "every target detected once": (
            torch.tensor([[0, 1, 2, 3, 4, 5]] * batch),
            no_padding,
        ),
        "ragged padding across batch": (
            torch.tensor([[0, 0, 1, -2, -2, -2], [2, 2, 3, 3, -1, -1]]),
            ragged,
        ),
    }
    for name, (identities, padding) in cases.items():
        loss = contrastive_association_loss(
            head(memory, padding), identities, padding
        )
        assert torch.isfinite(loss), f"non-finite loss for {name}"
        # The graph must stay connected so every DDP rank reduces the same
        # buckets even when the objective degenerates to zero.
        assert loss.requires_grad, f"detached loss for {name}"
        head.zero_grad()
        loss.backward(retain_graph=True)
        gradient = head.projection.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all(), name

    # A window with a real positive pair must produce a non-zero objective,
    # otherwise the degenerate handling above would be masking a dead loss.
    identities, padding = cases["two targets plus clutter"]
    final = contrastive_association_loss(head(memory, padding), identities, padding)
    assert float(final.detach()) > 0.0


def test_single_measurement_window_does_not_produce_nan():
    """log_softmax over a fully masked row would be NaN without a guard."""
    from track_mt3.models.contrastive import (
        ContrastiveClassifier,
        contrastive_association_loss,
    )

    head = ContrastiveClassifier(16)
    memory = torch.randn(1, 1, 16)
    padding = torch.zeros(1, 1, dtype=torch.bool)
    scores = head(memory, padding)
    assert torch.isfinite(scores).all()
    assert torch.isfinite(contrastive_association_loss(scores, torch.tensor([[0]]), padding))


def test_progressive_auxiliary_weighting_favours_later_layers():
    """Later decoder layers must dominate while the total magnitude is preserved."""
    uniform = CollectiveAverageCriterion(LossConfig(auxiliary_layer_weighting="uniform"))
    progressive = CollectiveAverageCriterion(
        LossConfig(auxiliary_layer_weighting="progressive")
    )
    count = 5
    assert [uniform._layer_scale(index, count) for index in range(count)] == [1.0] * count
    scales = [progressive._layer_scale(index, count) for index in range(count)]
    assert scales == sorted(scales)
    assert scales[-1] > scales[0]
    assert abs(sum(scales) - count) < 1e-6
    assert progressive._layer_scale(0, 1) == 1.0


def test_reused_inherited_matches_equal_a_full_recomputation():
    """The auxiliary fast path must produce the same index pairs as the full call."""
    prediction = make_prediction(
        [[0.0, 0.0], [3.0, 0.0], [1.0, 1.0], [-2.0, 4.0]], [1.0, 0.5, 0.0, -1.0], num_tracks=2
    )
    tracks = TrackState(
        features=torch.zeros(2, 8),
        references=torch.zeros(2, 2),
        target_ids=torch.tensor([10, 20]),
        track_ids=torch.tensor([10, 20]),
        ages=torch.ones(2, dtype=torch.long),
    )
    targets = torch.tensor([[-1.1, 0.0], [5.0, 5.0], [1.1, 0.0]])
    target_ids = torch.tensor([20, 30, 10])
    alignment = CrossFrameTargetAlignment(LossConfig())
    reference = alignment(prediction, tracks, targets, target_ids)
    inherited = alignment.inherited_matches(tracks, target_ids, len(prediction.positions))
    fast = alignment(
        prediction, tracks, targets, target_ids, inherited=inherited,
        include_query_target_ids=False,
    )
    assert torch.equal(reference.prediction_indices, fast.prediction_indices)
    assert torch.equal(reference.target_indices, fast.target_indices)
    assert reference.inherited_count == fast.inherited_count
    assert reference.newborn_count == fast.newborn_count
    # The auxiliary path skips this tensor; only the final layer feeds QTM.
    assert len(fast.query_target_ids) == 0


def test_query_position_mode_once_holds_the_embedding_fixed_across_layers():
    """"once" must feed every decoder layer the same positional signature.

    This is the official MT3 arrangement: query_pos is computed from the initial
    reference and passed unchanged into all six layers, while the reference itself
    only feeds the offset chain. "per_layer" instead rebuilds it from the refined
    reference, so the two modes must be observably different at the layer boundary.
    """
    config = small_config()
    frames = MultiTargetSimulator(config.simulation, 1).simulate(config.model.window_size)
    window = build_sliding_windows(frames, config.model.window_size)[0]
    seen: dict[str, list[torch.Tensor]] = {}
    for mode in ("once", "per_layer"):
        config.model.reference_query_embedding = mode
        model = TrackMT3(config)
        model.eval()
        # Zero-initialised position heads would make every layer's reference
        # identical and hide the difference, so give the heads a real signal.
        for head in model.decoder.position_heads:
            torch.nn.init.normal_(head.layers[-1].weight, std=0.5)
            torch.nn.init.normal_(head.layers[-1].bias, std=0.5)
        captured: list[torch.Tensor] = []
        # query_positions is the fifth positional argument of the layer forward.
        for layer in model.decoder.layers:
            layer.forward = (
                lambda *args, _inner=layer.forward, **kwargs: (
                    captured.append(args[4]) or _inner(*args, **kwargs)
                )
            )
        with torch.no_grad():
            model.forward_window(window)
        assert len(captured) == config.model.decoder_layers
        seen[mode] = captured

    for tensor in seen["once"][1:]:
        torch.testing.assert_close(tensor, seen["once"][0])
    assert not torch.allclose(seen["per_layer"][-1], seen["per_layer"][0])


def test_reference_gradient_path_follows_the_detach_switch():
    """Detaching the reference cuts the only path that can teach refinement.

    With detach on (official MT3), layer l's prediction carries no gradient into
    the layers before it through the reference chain, so no layer can learn that
    its own offset helps the next one. Probing v6 showed exactly the resulting
    symptom: per-layer matched error 0.900, 0.655, 0.659, 0.662, 0.658, 0.664.
    """
    config = small_config()
    frames = MultiTargetSimulator(config.simulation, 2).simulate(config.model.window_size)
    window = build_sliding_windows(frames, config.model.window_size)[0]
    grads = {}
    for detach in (True, False):
        config.model.detach_reference_between_layers = detach
        torch.manual_seed(0)
        model = TrackMT3(config)
        for head in model.decoder.position_heads:
            torch.nn.init.normal_(head.layers[-1].weight, std=0.5)
            torch.nn.init.normal_(head.layers[-1].bias, std=0.5)
        prediction = model.forward_window(window)
        # Only the last layer's positions are used, so any gradient reaching the
        # first layer's position head must have travelled through the reference.
        model.zero_grad()
        prediction.positions.sum().backward()
        first = model.decoder.position_heads[0].layers[-1].weight.grad
        grads[detach] = 0.0 if first is None else float(first.abs().sum())
    assert grads[True] == 0.0
    assert grads[False] > 0.0


def test_contrastive_temperature_restores_the_objectives_dynamic_range():
    """Raw cosines cap the InfoNCE floor far above zero; a temperature removes it.

    With scores bounded to [-1, 1] and n candidates, even a perfect embedding pays
    a floor of log(1 + (n-2)*exp(-1/temperature)) nats, which is 1.8 nats for 16
    candidates at temperature 1 and grows with n. That is why the measured term sat
    at 4.32-4.36 for 14000 steps while the encoder margin kept decaying: the value
    was already near its floor and carried almost no gradient.
    """
    import math

    from track_mt3.models.contrastive import (
        ContrastiveClassifier,
        contrastive_association_loss,
    )

    count, dim = 16, 32
    identities = torch.arange(count) // 2  # perfectly separable pairs
    padding = torch.zeros(1, count, dtype=torch.bool)
    # An ideal memory: one unit direction per target, orthogonal across targets, so
    # the only thing left limiting the objective is the score range itself.
    directions = torch.eye(count // 2, dim)
    memory = directions.repeat_interleave(2, dim=0).unsqueeze(0)

    losses = {}
    for temperature in (1.0, 0.1):
        head = ContrastiveClassifier(
            dim, temperature=temperature, learnable_temperature=False
        )
        # Identity projection so the head cannot undo the temperature.
        with torch.no_grad():
            head.projection.weight.copy_(torch.eye(dim))
        losses[temperature] = float(
            contrastive_association_loss(
                head(memory, padding), identities.unsqueeze(0), padding
            ).detach()
        )
    for temperature, value in losses.items():
        floor = math.log(1.0 + (count - 2) * math.exp(-1.0 / temperature))
        assert abs(value - floor) < 0.05, (temperature, value, floor)
    assert losses[1.0] > 1.5
    assert losses[0.1] < 0.01


def test_learnable_contrastive_temperature_is_trained_and_bounded():
    from track_mt3.models.contrastive import (
        ContrastiveClassifier,
        contrastive_association_loss,
    )

    head = ContrastiveClassifier(16, temperature=0.1, learnable_temperature=True)
    assert isinstance(head.log_scale, torch.nn.Parameter)
    memory = torch.randn(1, 8, 16)
    padding = torch.zeros(1, 8, dtype=torch.bool)
    identities = torch.tensor([[0, 0, 1, 1, 2, 2, -1, -1]])
    loss = contrastive_association_loss(head(memory, padding), identities, padding)
    loss.backward()
    assert head.log_scale.grad is not None and torch.isfinite(head.log_scale.grad)

    fixed = ContrastiveClassifier(16, temperature=0.1, learnable_temperature=False)
    assert not isinstance(fixed.log_scale, torch.nn.Parameter)
    with torch.no_grad():
        head.log_scale.fill_(50.0)
        assert float(head.scale) <= 1000.0 + 1e-3


def test_boolean_reference_query_embedding_is_accepted_for_old_configs():
    """Configs and checkpoints predating the three-way switch used a boolean."""
    assert ModelConfig(reference_query_embedding=True).reference_query_embedding == "per_layer"
    assert ModelConfig(reference_query_embedding=False).reference_query_embedding == "none"
    assert ModelConfig().reference_query_embedding == "once"
    for bad in ("per-layer", "always", ""):
        try:
            ModelConfig(reference_query_embedding=bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should have been rejected")


def two_stage_config() -> ExperimentConfig:
    config = small_config()
    config.model.two_stage = True
    return config


def test_two_stage_reference_points_land_on_real_measurements():
    """The whole point of two-stage is that a query starts on a measurement.

    With the proposal offset head zero-initialised, every reference point must be
    exactly some measurement of the window. Localization is 1.32 on scenario1 even
    though the measurements there *are* the ground truth, so a starting point on a
    measurement is the most direct attack available on that number.
    """
    config = two_stage_config()
    frames = MultiTargetSimulator(config.simulation, 3).simulate(config.model.window_size)
    window = build_sliding_windows(frames, config.model.window_size)[0]
    model = TrackMT3(config)
    model.eval()
    with torch.no_grad():
        memory, mask, _, normalized = model.encode_window(window)
        references, content, scores, coordinates, valid = model._proposals(
            memory, normalized + 0.5, mask
        )
    unit_measurements = (normalized + 0.5).squeeze(0)
    count = min(config.model.num_detection_queries, unit_measurements.shape[0])
    assert references.shape == (1, count, config.model.output_dim)
    assert content.shape == (1, count, config.model.hidden_dim)
    assert scores.shape == (1, unit_measurements.shape[0], 1)
    assert valid.shape == (1, count) and bool(valid.all())
    # cdist switches to a matmul kernel above 25 rows, which loses several digits
    # on near-zero distances, so the exact kernel is required to assert an exact
    # hit rather than a 1e-2 near-miss.
    distances = torch.cdist(
        references.squeeze(0),
        unit_measurements,
        compute_mode="donot_use_mm_for_euclid_dist",
    )
    closest = distances.min(dim=1).values
    assert torch.allclose(closest, torch.zeros_like(closest), atol=1e-4), closest
    # And the proposal coordinates start as the measurements themselves.
    torch.testing.assert_close(
        coordinates.squeeze(0), unit_measurements, atol=1e-4, rtol=0
    )


def test_two_stage_introduces_no_unused_parameters_for_ddp():
    """Two-stage must not add parameters that never receive gradient.

    Unused parameters force find_unused_parameters=True, which 4.11 went to some
    trouble to switch off. The learnable detection queries are therefore not created
    at all rather than left in the graph collecting nothing.

    The assertion is relative to the single-stage model on the same clip, because
    some QTM submodules legitimately receive no gradient when no track survives
    selection -- that is the separate 4.11 condition and not what this test is about.
    """
    frames = MultiTargetSimulator(small_config().simulation, 4).simulate(
        small_config().model.window_size + 2
    )
    clip = build_sliding_windows(frames, small_config().model.window_size)[:2]

    ungraded = {}
    for two_stage in (False, True):
        config = small_config()
        config.model.two_stage = two_stage
        torch.manual_seed(0)
        model = TrackMT3(config)
        assert (model.detection_queries is None) == two_stage
        names = dict(model.named_parameters())
        assert any("detection_reference" in name for name in names) != two_stage
        assert any("proposal" in name for name in names) == two_stage

        criterion = CollectiveAverageCriterion(config.loss)
        outputs = model([clip], criterion=criterion, teacher_forcing_probability=1.0)
        total, proposal = outputs[0], outputs[5]
        assert torch.isfinite(total) and torch.isfinite(proposal)
        # The proposal term must be a real objective, not a zero placeholder.
        assert (float(proposal) > 0.0) == two_stage
        total.backward()
        ungraded[two_stage] = {
            name for name, parameter in names.items() if parameter.grad is None
        }
        if two_stage:
            proposal_parameters = [n for n in names if "proposal" in n]
            assert proposal_parameters
            for name in proposal_parameters:
                assert name not in ungraded[True], name

    # Two-stage may not widen the set of parameters that go ungraded.
    assert ungraded[True] <= ungraded[False], ungraded[True] - ungraded[False]


def test_two_stage_proposal_scores_never_select_padding():
    """A padded slot winning the top-k would put a query on a coordinate of zero."""
    config = two_stage_config()
    model = TrackMT3(config)
    model.eval()
    batch, count = 2, config.model.num_detection_queries + 6
    memory = torch.randn(batch, count, config.model.hidden_dim)
    unit = torch.rand(batch, count, config.model.output_dim)
    mask = torch.zeros(batch, count, dtype=torch.bool)
    # Leave exactly num_detection_queries real slots in row 0.
    mask[0, config.model.num_detection_queries :] = True
    with torch.no_grad():
        # Force the padded slots to look attractive before masking.
        model.proposal_class_head.layers[-1].bias.fill_(0.0)
        references, _, scores, _, valid = model._proposals(memory, unit, mask)
    assert float(scores[0, config.model.num_detection_queries :].max()) < -1e7
    assert bool(valid[0].all()), "row 0 has exactly k real slots, all must be valid"
    distances = torch.cdist(
        references[0],
        unit[0, : config.model.num_detection_queries],
        compute_mode="donot_use_mm_for_euclid_dist",
    )
    assert float(distances.min(dim=1).values.max()) < 1e-4


def test_proposal_pool_restricts_selection_to_recent_frames():
    """A restricted pool must never pick a measurement outside the recent frames.

    The window covers window_size frames but only the last frame's targets are
    predicted, so a reference point from the window's start is (window_size-1)*dt
    stale. Probing an untrained score head over a full window showed only 3 of 24
    selected measurements came from the last two frames.
    """
    config = two_stage_config()
    config.model.proposal_pool_frames = 2
    model = TrackMT3(config)
    model.eval()
    batch, count = 2, 40
    W = config.model.window_size
    memory = torch.randn(batch, count, config.model.hidden_dim)
    unit = torch.rand(batch, count, config.model.output_dim).clamp(0.05, 0.95)
    mask = torch.zeros(batch, count, dtype=torch.bool)
    # Spread measurements over every frame of the window.
    times = torch.arange(count).remainder(W).unsqueeze(0).repeat(batch, 1)
    with torch.no_grad():
        references, _, _, _, valid = model._proposals(memory, unit, mask, times)
    recent_unit = [unit[b][times[b] >= W - 2] for b in range(batch)]
    for b in range(batch):
        chosen = references[b][valid[b]]
        assert len(chosen) > 0
        distances = torch.cdist(
            chosen, recent_unit[b], compute_mode="donot_use_mm_for_euclid_dist"
        )
        assert float(distances.min(dim=1).values.max()) < 1e-4, b


def test_proposal_pool_smaller_than_k_never_yields_padded_queries():
    """The eligible pool is routinely smaller than k and must not be topped up.

    During curriculum warmup the last frames of a window hold 0-9 measurements
    while k is 24. A padded slot would carry a reference of inverse_sigmoid(0),
    i.e. a query anchored on a coordinate no measurement occupies, so the surplus
    slots have to be dropped instead.
    """
    config = two_stage_config()
    config.model.proposal_pool_frames = 1
    config.model.num_detection_queries = 12
    model = TrackMT3(config)
    model.eval()
    W = config.model.window_size
    count = 30
    memory = torch.randn(1, count, config.model.hidden_dim)
    unit = torch.rand(1, count, config.model.output_dim).clamp(0.05, 0.95)
    mask = torch.zeros(1, count, dtype=torch.bool)
    mask[0, 20:] = True  # only 20 real measurements
    times = torch.zeros(1, count, dtype=torch.long)
    times[0, :3] = W - 1  # exactly three measurements in the current frame
    with torch.no_grad():
        references, _, _, _, valid = model._proposals(memory, unit, mask, times)
    assert int(valid.sum()) == 3, valid
    chosen = references[0][valid[0]]
    distances = torch.cdist(
        chosen, unit[0, :3], compute_mode="donot_use_mm_for_euclid_dist"
    )
    assert float(distances.min(dim=1).values.max()) < 1e-4


def test_proposal_pool_falls_back_when_no_recent_measurement_exists():
    """An empty current frame must still produce at least one query.

    Otherwise the row reaches the decoder with no reference at all, and a fully
    masked query row makes the attention softmax return NaN.
    """
    config = two_stage_config()
    config.model.proposal_pool_frames = 2
    model = TrackMT3(config)
    model.eval()
    count = 10
    memory = torch.randn(1, count, config.model.hidden_dim)
    unit = torch.rand(1, count, config.model.output_dim).clamp(0.05, 0.95)
    mask = torch.zeros(1, count, dtype=torch.bool)
    times = torch.zeros(1, count, dtype=torch.long)  # every measurement is stale
    with torch.no_grad():
        _, _, _, _, valid = model._proposals(memory, unit, mask, times)
    assert int(valid.sum()) == min(config.model.num_detection_queries, count)


def test_restricted_pool_trains_and_tracks_end_to_end():
    """The variable-length selection must survive batching, loss and inference."""
    from track_mt3.tracking import OnlineTracker

    config = two_stage_config()
    config.model.proposal_pool_frames = 2
    frames = MultiTargetSimulator(config.simulation, 7).simulate(
        config.model.window_size + 3
    )
    windows = build_sliding_windows(frames, config.model.window_size)
    model = TrackMT3(config)
    criterion = CollectiveAverageCriterion(config.loss)
    total, _, _, _, _, proposal = model(
        [windows[:2]], criterion=criterion, teacher_forcing_probability=1.0
    )
    assert torch.isfinite(total) and torch.isfinite(proposal)
    assert float(proposal) > 0.0
    total.backward()
    for name, parameter in model.named_parameters():
        if "proposal" in name:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    model.eval()
    tracker = OnlineTracker(model)
    with torch.no_grad():
        for window in windows:
            estimate = tracker.step(window)
    assert estimate is not None


def test_selection_mechanism_query_generation_is_a_nonlinear_ffn():
    """MT3 paper equation (9) is o_i = FFN(z~_i), not an affine map.

    The object query is produced from the refined coordinate by a network. With a
    single Linear and no activation the whole of (9) collapses into an affine
    function of the sinusoidal ladder, so it can express nothing the ladder does
    not already encode linearly. The paper's ablation values this mechanism at 25%
    GOSPA on Task 2 (3.662 -> 4.587), so its capacity is worth pinning.
    """
    config = two_stage_config()
    model = TrackMT3(config)
    activations = [
        module
        for module in model.proposal_to_query
        if isinstance(module, (torch.nn.ReLU, torch.nn.GELU))
    ]
    linears = [m for m in model.proposal_to_query if isinstance(m, torch.nn.Linear)]
    assert activations, "equation (9) must be a nonlinear FFN"
    assert len(linears) >= 2, "an FFN needs at least one hidden layer"
    # FFN1 likewise has to be more than a bare projection.
    assert isinstance(model.proposal_class_head, MLP)
    assert len(model.proposal_class_head.layers) >= 2

    # And it must actually behave nonlinearly in the coordinate.
    model.eval()
    with torch.no_grad():
        a = torch.tensor([[[0.30, 0.30]]])
        b = torch.tensor([[[0.70, 0.70]]])
        mid = (a + b) / 2
        qa = model.proposal_to_query(model.proposal_encoding(a))
        qb = model.proposal_to_query(model.proposal_encoding(b))
        qmid = model.proposal_to_query(model.proposal_encoding(mid))
    gap = float((qmid - (qa + qb) / 2).abs().max())
    assert gap > 1e-3, f"query generation is behaving affinely (gap={gap:.2e})"


def test_two_stage_and_learnable_queries_both_track_a_clip():
    """Both query sources must survive the full QTM propagation path."""
    from track_mt3.tracking import OnlineTracker

    for two_stage in (False, True):
        config = small_config()
        config.model.two_stage = two_stage
        frames = MultiTargetSimulator(config.simulation, 6).simulate(
            config.model.window_size + 3
        )
        windows = build_sliding_windows(frames, config.model.window_size)
        model = TrackMT3(config)
        model.eval()
        tracker = OnlineTracker(model)
        with torch.no_grad():
            for window in windows:
                estimate = tracker.step(window)
        assert estimate is not None


def test_cta_result_records_newborn_prediction_indices():
    """newborn_prediction_indices must identify which prediction slots are newborn."""
    prediction = make_prediction(
        positions=[[1.0, 0.0], [-1.0, 0.0], [5.1, 5.0], [-8.0, -8.0]],
        logits=[4.0, 4.0, 4.0, -4.0],
        num_tracks=2,
    )
    tracks = TrackState(
        features=torch.zeros(2, 8),
        references=torch.zeros(2, 2),
        target_ids=torch.tensor([10, 20]),
        track_ids=torch.tensor([10, 20]),
        ages=torch.ones(2, dtype=torch.long),
    )
    targets = torch.tensor([[-1.1, 0.0], [5.0, 5.0], [1.1, 0.0]])
    target_ids = torch.tensor([20, 30, 10])
    result = CrossFrameTargetAlignment(LossConfig())(prediction, tracks, targets, target_ids)
    assert result.newborn_count == 1
    assert len(result.newborn_prediction_indices) == 1
    assert result.newborn_prediction_indices[0] == 2


def test_newborn_weight_reduces_localization_for_newborn_targets():
    """With newborn_weight<1, newborn matched pairs contribute less to localization."""
    prediction = make_prediction(
        positions=[[1.0, 0.0], [-1.0, 0.0], [5.1, 5.0]],
        logits=[4.0, 4.0, 4.0],
        num_tracks=2,
    )
    tracks = TrackState(
        features=torch.zeros(2, 8),
        references=torch.zeros(2, 2),
        target_ids=torch.tensor([10, 20]),
        track_ids=torch.tensor([10, 20]),
        ages=torch.ones(2, dtype=torch.long),
    )
    targets = torch.tensor([[-1.1, 0.0], [1.1, 0.0], [5.0, 5.0]])
    target_ids = torch.tensor([20, 10, 30])
    config_full = LossConfig(newborn_weight=1.0, age_ramp=0)
    config_discounted = LossConfig(newborn_weight=0.3, age_ramp=0)
    full_criterion = CollectiveAverageCriterion(config_full)
    disc_criterion = CollectiveAverageCriterion(config_discounted)
    alignment = CrossFrameTargetAlignment(LossConfig())(prediction, tracks, targets, target_ids)
    assert alignment.newborn_count == 1
    full_loss = full_criterion.frame_loss(prediction, targets, alignment)
    disc_loss = disc_criterion.frame_loss(prediction, targets, alignment)
    assert float(disc_loss.localization) < float(full_loss.localization)


def test_age_ramp_recovers_weight_for_older_targets():
    """age_ramp=3 with newborn_weight=0.3: age=0→0.3, age=3→1.0."""
    prediction = make_prediction(
        positions=[[1.0, 0.0], [-1.0, 0.0], [5.1, 5.0]],
        logits=[4.0, 4.0, 4.0],
        num_tracks=2,
    )
    tracks = TrackState(
        features=torch.zeros(2, 8),
        references=torch.zeros(2, 2),
        target_ids=torch.tensor([10, 20]),
        track_ids=torch.tensor([10, 20]),
        ages=torch.ones(2, dtype=torch.long),
    )
    targets = torch.tensor([[-1.1, 0.0], [1.1, 0.0], [5.0, 5.0]])
    target_ids = torch.tensor([20, 10, 30])
    target_ages_young = torch.tensor([5, 5, 0])
    target_ages_old = torch.tensor([5, 5, 3])
    config = LossConfig(newborn_weight=0.3, age_ramp=3)
    criterion = CollectiveAverageCriterion(config)
    alignment = CrossFrameTargetAlignment(LossConfig())(prediction, tracks, targets, target_ids)
    loss_young = criterion.frame_loss(prediction, targets, alignment, target_ages=target_ages_young)
    loss_old = criterion.frame_loss(prediction, targets, alignment, target_ages=target_ages_old)
    assert float(loss_young.localization) < float(loss_old.localization)


def test_simulator_tracks_target_ages():
    """Target ages must start at 0 and increment each frame."""
    from track_mt3.config import SimulationConfig
    config = SimulationConfig(initial_targets=2, birth_rate=0.0, clutter_rate=0.0)
    sim = MultiTargetSimulator(config, seed=0)
    frames = sim.simulate(5)
    assert all(age == 0 for age in frames[0].target_ages.tolist())
    assert all(age == 1 for age in frames[1].target_ages.tolist())
    assert all(age == 4 for age in frames[4].target_ages.tolist())


def test_newborn_weight_does_not_affect_inherited_only_matches():
    """When all matches are inherited, newborn_weight should not change the loss."""
    prediction = make_prediction(
        positions=[[1.0, 0.0], [-1.0, 0.0]],
        logits=[4.0, 4.0],
        num_tracks=2,
    )
    tracks = TrackState(
        features=torch.zeros(2, 8),
        references=torch.zeros(2, 2),
        target_ids=torch.tensor([10, 20]),
        track_ids=torch.tensor([10, 20]),
        ages=torch.ones(2, dtype=torch.long),
    )
    targets = torch.tensor([[-1.1, 0.0], [1.1, 0.0]])
    target_ids = torch.tensor([20, 10])
    config_full = LossConfig(newborn_weight=1.0, age_ramp=0)
    config_disc = LossConfig(newborn_weight=0.1, age_ramp=5)
    alignment = CrossFrameTargetAlignment(LossConfig())(prediction, tracks, targets, target_ids)
    assert alignment.newborn_count == 0
    full_loss = CollectiveAverageCriterion(config_full).frame_loss(prediction, targets, alignment)
    disc_loss = CollectiveAverageCriterion(config_disc).frame_loss(prediction, targets, alignment)
    torch.testing.assert_close(full_loss.localization, disc_loss.localization)
