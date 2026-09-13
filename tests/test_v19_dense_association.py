from __future__ import annotations

import pytest
import torch

from track_mt3.models_v17 import EndToEndRecursiveTracker, V17BConfig


def _config(**overrides) -> V17BConfig:
    values = {
        "hidden_dim": 32,
        "num_heads": 4,
        "encoder_layers": 1,
        "feedforward_dim": 64,
        "max_tracks": 4,
        "dropout": 0.0,
        "track_evidence_recurrence": True,
    }
    values.update(overrides)
    return V17BConfig(**values)


def test_bernoulli_gate_requires_recurrent_existence():
    with pytest.raises(ValueError, match="requires track_evidence_recurrence"):
        V17BConfig(association_bernoulli_gate=True)


def test_predicted_existence_gate_requires_log_odds_filter():
    with pytest.raises(ValueError, match="requires log-odds evidence update"):
        _config(association_bernoulli_use_predicted_existence=True)


def test_v19a_adds_explicit_existence_odds_only_to_active_pair_bias():
    model = EndToEndRecursiveTracker(
        _config(
            association_bernoulli_gate=True,
            association_bernoulli_logit_scale=0.5,
        )
    ).eval()
    state = model.empty_state(1)
    state.active_mask[0, :2] = True
    state.owner_ids[0, :2] = torch.tensor([0, 1])
    state.supervision_ids[0, :2] = torch.tensor([0, 1])
    state.exist_logit[0, :2] = torch.tensor([-2.0, 2.0])
    bias = model._predict_track_measurement_bias(state, torch.tensor([0.1]), 0.0)
    torch.testing.assert_close(bias[0, :2], torch.tensor([-1.0, 1.0]))
    torch.testing.assert_close(bias[0, 2:], torch.zeros(2))


def test_predicted_existence_gate_uses_survival_and_one_global_floor():
    model = EndToEndRecursiveTracker(
        _config(
            track_evidence_log_odds_update=True,
            association_bernoulli_gate=True,
            association_bernoulli_logit_scale=0.5,
            association_bernoulli_use_predicted_existence=True,
            association_bernoulli_probability_floor=0.2,
        )
    ).eval()
    torch.nn.init.zeros_(model.association_prior_head[-1].weight)
    torch.nn.init.zeros_(model.association_prior_head[-1].bias)
    state = model.empty_state(1)
    state.active_mask[0, :2] = True
    state.exist_logit[0, :2] = torch.logit(torch.tensor([0.8, 0.1]))
    survival = torch.full((1, 4), torch.logit(torch.tensor(0.5)))
    bias = model._predict_track_measurement_bias(
        state, torch.tensor([0.1]), 0.0, survival
    )
    expected = 0.5 * torch.logit(torch.tensor([0.4, 0.2]))
    torch.testing.assert_close(bias[0, :2], expected)


def test_persistent_confirmation_survives_ambiguous_low_posterior():
    model = EndToEndRecursiveTracker(
        _config(track_evidence_persistent_confirmation=True)
    ).eval()
    state = model.empty_state(1)
    state.active_mask[0, :2] = True
    state.confirmed_mask[0, 0] = True
    next_mask = model._next_confirmation_mask(
        state,
        state.active_mask,
        torch.tensor([[-4.0, 4.0, -8.0, -8.0]]),
        confirmation_hits=2,
    )
    assert next_mask[0, 0]
    assert next_mask[0, 1]


def test_fast_inference_matches_reference_state_chain():
    torch.manual_seed(7)
    model = EndToEndRecursiveTracker(
        _config(
            track_evidence_persistent_confirmation=True,
            track_evidence_log_odds_update=True,
            track_evidence_objectness_aware=True,
            track_evidence_exact_bernoulli_prediction=True,
            association_bernoulli_gate=True,
            association_bernoulli_use_predicted_existence=True,
        )
    ).eval()
    measurements = [
        torch.tensor(
            [
                [[0.1 + step, 0.2], [1.0, -0.5], [0.0, 0.0]],
                [[-0.3, 0.4 + step], [0.5, 0.7], [1.2, -0.8]],
            ]
        )
        for step in (0.0, 0.1, 0.2)
    ]
    masks = [
        torch.tensor([[False, False, True], [False, False, False]])
        for _ in measurements
    ]
    reference_state = None
    fast_state = None
    with torch.inference_mode():
        for step, (frame, mask) in enumerate(zip(measurements, masks)):
            time = torch.full((2,), 0.1 * step)
            reference_output, reference_state = model.forward_inference_step(
                frame,
                mask,
                time,
                reference_state,
                birth_candidate_threshold=0.0,
                confirmation_hits=2,
                retention_threshold=0.0,
                survival_warmup_frames=3,
            )
            fast_output, fast_state = model.forward_inference_step_fast(
                frame,
                mask,
                time,
                fast_state,
                birth_candidate_threshold=0.0,
                confirmation_hits=2,
                retention_threshold=0.0,
                survival_warmup_frames=3,
            )
            for name in (
                "existing_logits",
                "posterior_mean",
                "posterior_covariance",
                "birth_logits",
                "birth_positions",
            ):
                torch.testing.assert_close(
                    getattr(fast_output, name),
                    getattr(reference_output, name),
                    atol=2.0e-5,
                    rtol=2.0e-5,
                )
            for name, reference_value in reference_state.__dict__.items():
                fast_value = getattr(fast_state, name)
                if reference_value.is_floating_point():
                    torch.testing.assert_close(
                        fast_value, reference_value, atol=2.0e-5, rtol=2.0e-5
                    )
                else:
                    assert torch.equal(fast_value, reference_value)


def test_legacy_state_write_keeps_transport_exactly_unchanged():
    model = EndToEndRecursiveTracker(_config()).eval()
    state = model.empty_state(1)
    pair = torch.tensor([[[0.6, 0.3], [0.2, 0.1], [0.0, 0.0], [0.0, 0.0]]])
    written = model._association_write_pairs(
        state,
        state.query,
        state.covariance,
        pair,
        torch.zeros(1, 2, dtype=torch.bool),
        torch.full((1, 4), 0.1),
    )
    assert written is pair


def test_v19b_preserves_one_hot_write_and_reduces_ambiguous_write():
    model = EndToEndRecursiveTracker(
        _config(ambiguity_aware_update=True, ambiguity_update_min_gate=0.1)
    ).train()
    state = model.empty_state(1)
    state.active_mask[0, :2] = True
    state.owner_ids[0, :2] = torch.tensor([0, 1])
    state.supervision_ids[0, :2] = torch.tensor([0, 1])
    pair = torch.zeros(1, 4, 3)
    pair[0, 0, 0] = 1.0
    pair[0, 1, :2] = 0.5
    written = model._association_write_pairs(
        state,
        state.query,
        state.covariance,
        pair,
        torch.zeros(1, 3, dtype=torch.bool),
        torch.full((1, 4), 0.1),
    )
    torch.testing.assert_close(written[0, 0], pair[0, 0])
    assert written[0, 1].sum() < pair[0, 1].sum()
    assert written[0, 1].sum() > 0.1 * pair[0, 1].sum()
    written.sum().backward()
    gradient = model.ambiguity_write_head[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.norm() > 0
