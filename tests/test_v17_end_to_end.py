from __future__ import annotations

import copy
import json

import pytest
import torch

from track_mt3.config import SimulationConfig
from track_mt3.models_v17 import EndToEndRecursiveTracker, V17BConfig
from track_mt3.models_v17.model import _combine_birth_evidence
from track_mt3.models_v17.association_criterion import V17BAssociationCriterion
from track_mt3.models_v17.death_criterion import (
    V17DeathLoss,
    negative_exposed_death_mean,
)
from track_mt3.training.curriculum_v17 import (
    v17_curriculum_at_step,
    v17_rollout_horizon_at_step,
    v17_trajectory_curriculum_at_step,
)
from track_mt3.training.trainer_v17 import archive_metrics_after_resume


def _curriculum(step: int):
    return v17_curriculum_at_step(
        step,
        association_oracle_hold_steps=10,
        association_oracle_end_step=30,
        predicted_birth_start_step=20,
        predicted_birth_full_step=40,
        predicted_death_start_step=50,
    )


def test_v21_pmbm_birth_odds_preserve_local_odds_when_unclaimed():
    objectness_logits = torch.logit(torch.tensor([0.8, 0.8]))
    unclaimed = torch.tensor([1.0, 0.5])
    probabilities, logits = _combine_birth_evidence(
        objectness_logits,
        unclaimed,
        pmbm_odds_fusion=True,
        unclaimed_log_scale=1.0,
    )
    assert probabilities[0] == pytest.approx(0.8)
    assert probabilities[1] == pytest.approx(2.0 / 3.0)
    assert logits[1] == pytest.approx(float(objectness_logits[1] + torch.log(unclaimed[1])))


def test_v20_birth_product_remains_exact_when_v21_is_disabled():
    objectness_logits = torch.tensor([0.0, 1.0])
    unclaimed = torch.tensor([0.25, 0.75])
    probabilities, logits = _combine_birth_evidence(
        objectness_logits,
        unclaimed,
        pmbm_odds_fusion=False,
        unclaimed_log_scale=1.0,
    )
    expected = objectness_logits.sigmoid() * unclaimed
    torch.testing.assert_close(probabilities, expected)
    torch.testing.assert_close(logits, torch.logit(expected))


def test_v21b_neural_ppp_and_k2_hypothesis_state_are_causal_and_finite():
    torch.manual_seed(2121)
    model = EndToEndRecursiveTracker(
        V17BConfig(
            hidden_dim=32,
            num_heads=4,
            encoder_layers=1,
            feedforward_dim=64,
            max_tracks=6,
            dropout=0.0,
            birth_neural_ppp=True,
            association_mbm_hypotheses=2,
            association_mbm_entropy_threshold=0.0,
        )
    ).train()
    mask = torch.zeros(1, 3, dtype=torch.bool)
    ids = torch.tensor([[1, 2, -1]])
    first, state = model.forward_step(
        torch.tensor([[[-0.2, 0.0], [0.2, 0.0], [2.0, 2.0]]]),
        mask,
        ids,
        torch.tensor([0.0]),
        [torch.tensor([1, 2])],
        association_oracle_probability=0.0,
        association_update_mode="moment",
        birth_state_mode="predicted",
        birth_candidate_threshold=0.0,
        death_state_mode="predicted",
    )
    second, next_state = model.forward_step(
        torch.tensor([[[0.0, 0.0], [0.02, 0.0], [2.1, 2.0]]]),
        mask,
        ids,
        torch.tensor([1.0]),
        [torch.tensor([1, 2])],
        state,
        association_oracle_probability=0.0,
        association_update_mode="moment",
        birth_state_mode="predicted",
        birth_candidate_threshold=0.0,
        death_state_mode="predicted",
    )
    assert first.predicted_undetected_mean is not None
    assert second.birth_target_log_intensity is not None
    assert second.birth_clutter_log_intensity is not None
    assert second.association_hypothesis_weights is not None
    assert second.alternative_posterior_mean is not None
    assert next_state.undetected_initialized.all()
    torch.testing.assert_close(
        second.association_hypothesis_weights.sum(-1),
        torch.ones(1),
    )
    assert torch.isfinite(next_state.alternative_mean).all()
    assert torch.isfinite(next_state.undetected_log_mean).all()
    active = next_state.active_mask.unsqueeze(-1)
    assert torch.any(
        (next_state.alternative_mean - next_state.mean).abs().masked_fill(
            ~active, 0.0
        )
        > 1.0e-6
    )
    loss = second.birth_logits.sigmoid().sum() + second.posterior_mean.square().mean()
    loss.backward()
    assert model.ppp_target_intensity_head[-1].weight.grad is not None


def test_curriculum_retires_teacher_forcing_without_changing_model():
    assert _curriculum(1).name == "oracle_bootstrap"
    assert _curriculum(10).association_oracle_probability == 1.0
    assert _curriculum(20).association_oracle_probability == 0.5
    assert _curriculum(30).association_oracle_probability == 0.0
    assert _curriculum(20).predicted_birth_probability == 0.0
    assert _curriculum(30).predicted_birth_probability == 0.5
    assert _curriculum(40).predicted_birth_probability == 1.0
    assert _curriculum(49).death_state_mode == "oracle_delayed"
    assert _curriculum(50).death_state_mode == "predicted"
    assert _curriculum(50).name == "free_rollout"


def test_joint_lifecycle_curriculum_switches_association_and_death_together():
    before = v17_curriculum_at_step(
        799,
        association_oracle_hold_steps=800,
        association_oracle_end_step=800,
        predicted_birth_start_step=300,
        predicted_birth_full_step=600,
        predicted_death_start_step=800,
    )
    after = v17_curriculum_at_step(
        800,
        association_oracle_hold_steps=800,
        association_oracle_end_step=800,
        predicted_birth_start_step=300,
        predicted_birth_full_step=600,
        predicted_death_start_step=800,
    )
    assert before.association_oracle_probability == 1.0
    assert before.death_state_mode == "oracle_delayed"
    assert after.association_oracle_probability == 0.0
    assert after.death_state_mode == "predicted"


def test_trajectory_curriculum_is_independent_and_reaches_each_terminal():
    terminals = [
        SimulationConfig(
            initial_targets=6,
            max_targets=16,
            birth_rate=0.04,
            survival_probability=0.99,
            detection_probability=0.95,
            process_noise=0.0,
            measurement_noise=0.0,
            clutter_rate=5.0,
        ),
        SimulationConfig(
            initial_targets=10,
            max_targets=16,
            birth_rate=0.12,
            survival_probability=0.97,
            detection_probability=0.85,
            process_noise=0.08,
            measurement_noise=0.03,
            clutter_rate=15.0,
        ),
    ]
    easy = v17_trajectory_curriculum_at_step(
        terminals, 0, warmup_steps=5000
    )
    halfway = v17_trajectory_curriculum_at_step(
        terminals, 2500, warmup_steps=5000
    )
    full = v17_trajectory_curriculum_at_step(
        terminals, 5000, warmup_steps=5000
    )

    assert easy.progress == 0.0
    assert easy.simulations[0].initial_targets == 1
    assert easy.simulations[0].max_targets == 4
    assert easy.simulations[0].clutter_rate == 0.0
    assert easy.simulations[0].detection_probability == 1.0
    assert easy.simulations[0].birth_rate == 0.0
    assert easy.simulations[0].survival_probability == 1.0
    assert easy.simulations[0].process_noise == 0.0
    assert easy.simulations[1].process_noise == 0.016
    assert halfway.simulations[1].clutter_rate == 7.5
    assert halfway.simulations[1].process_noise == 0.048
    assert full.simulations == tuple(terminals)
    assert full.metrics["trajectory_curriculum_progress"] == 1.0
    assert full.metrics["sim_clutter_rate"] == 10.0
    # Curriculum construction must not mutate terminal configurations reused by
    # later steps or formal evaluation.
    assert terminals[0].initial_targets == 6
    assert terminals[1].clutter_rate == 15.0


def test_two_curricula_reach_the_registered_boundaries_independently():
    terminals = [SimulationConfig(initial_targets=10, clutter_rate=15.0)]
    state_3500 = v17_curriculum_at_step(
        3500,
        association_oracle_hold_steps=500,
        association_oracle_end_step=4000,
        predicted_birth_start_step=300,
        predicted_birth_full_step=600,
        predicted_death_start_step=800,
    )
    state_4000 = v17_curriculum_at_step(
        4000,
        association_oracle_hold_steps=500,
        association_oracle_end_step=4000,
        predicted_birth_start_step=300,
        predicted_birth_full_step=600,
        predicted_death_start_step=800,
    )
    physical_3500 = v17_trajectory_curriculum_at_step(
        terminals, 3500, warmup_steps=5000
    )
    physical_5000 = v17_trajectory_curriculum_at_step(
        terminals, 5000, warmup_steps=5000
    )

    assert state_3500.association_oracle_probability == pytest.approx(1.0 / 7.0)
    assert state_3500.death_state_mode == "predicted"
    assert state_4000.association_oracle_probability == 0.0
    assert physical_3500.progress == 0.7
    assert physical_3500.simulations[0].clutter_rate == 10.5
    assert physical_5000.simulations[0].clutter_rate == 15.0


def test_rollout_horizon_curriculum_reaches_exact_evaluation_span():
    arguments = dict(
        hold_steps=5000,
        full_step=14000,
        start_trajectory_steps=20,
        end_trajectory_steps=100,
        start_burn_in_frames=0,
        end_burn_in_frames=0,
    )
    early = v17_rollout_horizon_at_step(5000, **arguments)
    middle = v17_rollout_horizon_at_step(9500, **arguments)
    full = v17_rollout_horizon_at_step(14000, **arguments)

    assert early.progress == 0.0
    assert (early.trajectory_steps, early.burn_in_frames, early.supervised_frames) == (
        20,
        0,
        20,
    )
    assert middle.progress == 0.5
    assert (
        middle.trajectory_steps,
        middle.burn_in_frames,
        middle.supervised_frames,
    ) == (60, 0, 60)
    assert full.progress == 1.0
    assert (full.trajectory_steps, full.burn_in_frames, full.supervised_frames) == (
        100,
        0,
        100,
    )


def test_random_rollout_horizon_expands_reproducibly_to_full_range():
    arguments = dict(
        hold_steps=5000,
        full_step=14000,
        start_trajectory_steps=20,
        end_trajectory_steps=100,
        start_burn_in_frames=0,
        end_burn_in_frames=0,
        sampling_mode="uniform",
        minimum_trajectory_steps=20,
        sampling_seed=1811,
    )
    early = v17_rollout_horizon_at_step(5000, **arguments)
    middle = v17_rollout_horizon_at_step(9500, **arguments)
    repeated_middle = v17_rollout_horizon_at_step(9500, **arguments)
    final_samples = [
        v17_rollout_horizon_at_step(step, **arguments).trajectory_steps
        for step in range(14000, 14100)
    ]

    assert (early.trajectory_steps, early.maximum_trajectory_steps) == (20, 20)
    assert middle.maximum_trajectory_steps == 60
    assert 20 <= middle.trajectory_steps <= 60
    assert middle == repeated_middle
    assert min(final_samples) >= 20
    assert max(final_samples) <= 100
    assert len(set(final_samples)) > 50


def test_tail_balanced_horizon_reaches_tail_without_raising_average_cost():
    arguments = dict(
        hold_steps=5000,
        full_step=14000,
        start_trajectory_steps=20,
        end_trajectory_steps=100,
        start_burn_in_frames=0,
        end_burn_in_frames=0,
        sampling_mode="tail_balanced",
        minimum_trajectory_steps=20,
        sampling_seed=1919,
        tail_probability=0.25,
        short_maximum_trajectory_steps=41,
    )
    early = v17_rollout_horizon_at_step(5000, **arguments)
    repeated = v17_rollout_horizon_at_step(15000, **arguments)
    assert early.trajectory_steps == 20
    assert repeated == v17_rollout_horizon_at_step(15000, **arguments)

    samples = [
        v17_rollout_horizon_at_step(step, **arguments).trajectory_steps
        for step in range(14000, 15000)
    ]
    tail_fraction = sum(value == 100 for value in samples) / len(samples)
    assert 0.20 <= tail_fraction <= 0.30
    assert all(value == 100 or 20 <= value <= 41 for value in samples)
    assert 45.0 <= sum(samples) / len(samples) <= 51.0


def test_tail_balanced_horizon_rejects_invalid_probability():
    with pytest.raises(ValueError, match="tail_probability"):
        v17_rollout_horizon_at_step(
            1,
            hold_steps=0,
            full_step=1,
            start_trajectory_steps=20,
            end_trajectory_steps=100,
            start_burn_in_frames=0,
            end_burn_in_frames=0,
            sampling_mode="tail_balanced",
            tail_probability=1.1,
        )


def test_resume_archives_abandoned_metrics_without_duplicate_steps(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(json.dumps({"step": step}) for step in (1, 2, 3, 4)) + "\n",
        encoding="utf-8",
    )
    archive = archive_metrics_after_resume(path, 2)
    assert archive is not None
    assert [json.loads(line)["step"] for line in path.read_text().splitlines()] == [
        1,
        2,
    ]
    assert [json.loads(line)["step"] for line in archive.read_text().splitlines()] == [
        3,
        4,
    ]
    assert archive_metrics_after_resume(path, 2) is None


def test_dual_rollout_shared_encoder_matches_duplicate_forward_and_gradients():
    torch.manual_seed(71)
    config = V17BConfig(
        hidden_dim=32,
        num_heads=4,
        encoder_layers=1,
        feedforward_dim=64,
        max_tracks=4,
        dropout=0.0,
    )
    duplicate = EndToEndRecursiveTracker(config).train()
    shared = copy.deepcopy(duplicate).train()
    measurements = torch.tensor(
        [[[0.2, -0.1], [1.0, 0.5], [-2.0, 1.5]]], dtype=torch.float32
    )
    mask = torch.tensor([[False, False, False]])
    ids = torch.tensor([[2, -1, 7]])
    truth_ids = [torch.tensor([2, 7])]
    frame_time = torch.tensor([0.2])

    def objective(model, reuse):
        embeddings = model.encode_current_frame(measurements, mask) if reuse else None
        learned, _ = model.forward_step(
            measurements,
            mask,
            ids,
            frame_time,
            truth_ids,
            association_oracle_probability=0.0,
            association_update_mode="moment",
            birth_state_mode="predicted",
            death_state_mode="predicted",
            precomputed_measurement_embeddings=embeddings,
        )
        oracle, _ = model.forward_step(
            measurements,
            mask,
            ids,
            frame_time,
            truth_ids,
            association_oracle_probability=1.0,
            association_update_mode="moment",
            birth_state_mode="oracle",
            death_state_mode="oracle_pre",
            precomputed_measurement_embeddings=embeddings,
        )
        return (
            learned.birth_positions.square().mean()
            + learned.birth_logits.sigmoid().mean()
            + oracle.birth_positions.square().mean()
            + oracle.birth_logits.sigmoid().mean()
        )

    duplicate_loss = objective(duplicate, False)
    shared_loss = objective(shared, True)
    torch.testing.assert_close(shared_loss, duplicate_loss, rtol=0.0, atol=1.0e-7)
    duplicate_loss.backward()
    shared_loss.backward()
    for (duplicate_name, duplicate_parameter), (shared_name, shared_parameter) in zip(
        duplicate.named_parameters(), shared.named_parameters()
    ):
        assert duplicate_name == shared_name
        if duplicate_parameter.grad is None:
            assert shared_parameter.grad is None
        else:
            torch.testing.assert_close(
                shared_parameter.grad,
                duplicate_parameter.grad,
                rtol=2.0e-5,
                atol=2.0e-7,
                msg=lambda message: f"{duplicate_name}: {message}",
            )


def test_learned_track_prior_starts_neutral_and_receives_association_gradient():
    torch.manual_seed(23)
    model = EndToEndRecursiveTracker(
        V17BConfig(
            hidden_dim=32,
            num_heads=4,
            encoder_layers=1,
            feedforward_dim=64,
            max_tracks=4,
            dropout=0.0,
        )
    ).train()
    state = model.empty_state(1)
    state.active_mask[0, :2] = True
    state.confirmed_mask[0, 0] = True
    state.owner_ids[0, :2] = torch.tensor([3, 7])
    state.supervision_ids[0, :2] = torch.tensor([3, 7])
    state.next_runtime_id[0] = 8
    state.mean[0, 0, :2] = torch.tensor([1.0, 0.0])
    state.mean[0, 1, :2] = torch.tensor([-1.0, 0.0])
    output, _ = model.forward_step(
        torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [4.0, 4.0]]]),
        torch.zeros(1, 3, dtype=torch.bool),
        torch.tensor([[3, 7, -1]]),
        torch.tensor([0.1]),
        [torch.tensor([3, 7])],
        state,
        association_oracle_probability=0.0,
        association_update_mode="moment",
    )
    torch.testing.assert_close(
        output.association_track_bias,
        torch.zeros_like(output.association_track_bias),
    )
    V17BAssociationCriterion()(output).total.backward()
    gradient = model.association_prior_head[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.norm() > 0


def test_formal_survival_head_starts_near_neutral():
    torch.manual_seed(31)
    model = EndToEndRecursiveTracker(
        V17BConfig(
            hidden_dim=32,
            num_heads=4,
            encoder_layers=1,
            feedforward_dim=64,
            max_tracks=4,
            dropout=0.0,
        )
    )
    features = torch.randn(3, 4, 39)
    logits = model._predict_survival(features, torch.zeros(3, 4))
    assert logits.abs().max() < 0.1


def test_survival_loss_waits_for_negative_exposure():
    anchor = torch.tensor(2.0, requires_grad=True)
    positive_only = V17DeathLoss(
        total=torch.tensor(3.0, requires_grad=True),
        alive_probability=torch.tensor(0.7),
        dead_probability=torch.tensor(0.0),
        alive_tracks=4,
        dead_tracks=0,
    )
    negative_exposed = V17DeathLoss(
        total=torch.tensor(5.0, requires_grad=True),
        alive_probability=torch.tensor(0.6),
        dead_probability=torch.tensor(0.4),
        alive_tracks=3,
        dead_tracks=2,
    )
    waiting, waiting_frames = negative_exposed_death_mean([positive_only], anchor)
    torch.testing.assert_close(waiting, torch.tensor(0.0))
    assert waiting_frames == 0
    trained, trained_frames = negative_exposed_death_mean(
        [positive_only, negative_exposed], anchor
    )
    torch.testing.assert_close(trained, negative_exposed.total)
    assert trained_frames == 1


def test_inference_step_requires_no_truth_metadata():
    torch.manual_seed(29)
    model = EndToEndRecursiveTracker(
        V17BConfig(
            hidden_dim=32,
            num_heads=4,
            encoder_layers=1,
            feedforward_dim=64,
            max_tracks=6,
            dropout=0.0,
        )
    ).eval()
    measurements = torch.tensor([[[0.2, -0.1], [2.0, 3.0]]])
    mask = torch.zeros(1, 2, dtype=torch.bool)
    frame_time = torch.tensor([0.0])
    inference_output, inference_state = model.forward_inference_step(
        measurements,
        mask,
        frame_time,
        birth_candidate_threshold=0.0,
    )
    sentinel_output, sentinel_state = model.forward_step(
        measurements,
        mask,
        torch.full((1, 2), -1, dtype=torch.long),
        frame_time,
        [torch.empty(0, dtype=torch.long)],
        association_oracle_probability=0.0,
        association_update_mode="moment",
        birth_state_mode="predicted",
        birth_candidate_threshold=0.0,
        confirmation_hits=3,
        death_state_mode="predicted",
        retention_threshold=0.175,
        survival_warmup_frames=3,
    )
    torch.testing.assert_close(
        inference_output.birth_logits, sentinel_output.birth_logits
    )
    torch.testing.assert_close(inference_state.mean, sentinel_state.mean)
    torch.testing.assert_close(inference_state.active_mask, sentinel_state.active_mask)
    assert torch.all(inference_state.supervision_ids[inference_state.active_mask] == -1)
