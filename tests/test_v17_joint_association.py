from __future__ import annotations

import torch

from track_mt3.config import SimulationConfig
from track_mt3.data.simulator import MultiTargetSimulator
from track_mt3.models_v17 import (
    EndToEndRecursiveTracker,
    JointAssociationTracker,
    V17BConfig,
)
from track_mt3.models_v17.association_criterion import V17BAssociationCriterion
from track_mt3.models_v17.criterion import V17AOracleCriterion
from track_mt3.training.trainer_v17 import V17BSequenceTrainer


def _model() -> JointAssociationTracker:
    torch.manual_seed(17)
    return JointAssociationTracker(
        V17BConfig(
            hidden_dim=32,
            num_heads=4,
            encoder_layers=1,
            feedforward_dim=64,
            max_tracks=6,
            dropout=0.0,
            association_sinkhorn_iterations=30,
        )
    )


def _state(model: JointAssociationTracker):
    state = model.empty_state(1)
    state.active_mask[0, :2] = True
    state.confirmed_mask[0, :2] = True
    state.owner_ids[0, :2] = torch.tensor([3, 7])
    state.supervision_ids[0, :2] = torch.tensor([3, 7])
    state.next_runtime_id[0] = 8
    state.mean[0, 0, :2] = torch.tensor([1.0, 0.0])
    state.mean[0, 1, :2] = torch.tensor([-1.0, 0.0])
    state.covariance[0, :2] = 0.05 * torch.eye(4)
    return state


def _joint_lifecycle_model() -> EndToEndRecursiveTracker:
    torch.manual_seed(19)
    return EndToEndRecursiveTracker(
        V17BConfig(
            hidden_dim=32,
            num_heads=4,
            encoder_layers=1,
            feedforward_dim=64,
            max_tracks=6,
            dropout=0.0,
            association_sinkhorn_iterations=30,
            joint_lifecycle_transport=True,
        )
    )


def test_joint_lifecycle_transport_normalizes_pair_miss_and_death():
    model = _joint_lifecycle_model().eval()
    state = _state(model)
    state.active_mask[0, 2] = True
    state.owner_ids[0, 2] = 8
    state.supervision_ids[0, 2] = -1
    state.ages[0, :3] = 4
    output, _ = model.forward_step(
        torch.tensor([[[1.02, 0.01], [-1.02, -0.01], [4.0, 4.0]]]),
        torch.zeros(1, 3, dtype=torch.bool),
        torch.tensor([[3, 7, -1]]),
        torch.tensor([0.1]),
        [torch.tensor([3, 7])],
        state,
        association_oracle_probability=0.0,
        association_update_mode="moment",
        death_state_mode="predicted",
    )
    assert output.predicted_death_probabilities is not None
    row_mass = (
        output.predicted_pair_probabilities.sum(-1)
        + output.predicted_miss_probabilities
        + output.predicted_death_probabilities
    )
    column_mass = (
        output.predicted_pair_probabilities.sum(-2)
        + output.predicted_unclaimed_probabilities
    )
    torch.testing.assert_close(row_mass[0, :3], torch.ones(3), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(column_mass[0], torch.ones(3), atol=1e-4, rtol=1e-4)


def test_joint_lifecycle_loss_labels_false_track_as_death_and_trains_survival():
    model = _joint_lifecycle_model().train()
    state = _state(model)
    state.active_mask[0, 2] = True
    state.owner_ids[0, 2] = 8
    state.supervision_ids[0, 2] = -1
    output, _ = model.forward_step(
        torch.tensor([[[1.02, 0.01], [-1.02, -0.01], [4.0, 4.0]]]),
        torch.zeros(1, 3, dtype=torch.bool),
        torch.tensor([[3, 7, -1]]),
        torch.tensor([0.1]),
        [torch.tensor([3, 7])],
        state,
        association_oracle_probability=0.0,
        association_update_mode="moment",
        death_state_mode="predicted",
    )
    loss = V17BAssociationCriterion(
        balance_event_types=True,
        pair_weight=2.0,
        death_weight=2.0,
    )(output, [torch.tensor([3, 7])])
    assert loss.pair_events == 2
    assert loss.death_events == 1
    loss.total.backward()
    gradient = model.survival_head[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.norm() > 0


def test_oracle_association_jointly_terminates_dead_track():
    model = _joint_lifecycle_model().eval()
    state = _state(model)
    state.active_mask[0, 2] = True
    state.owner_ids[0, 2] = 8
    state.supervision_ids[0, 2] = -1
    state.ages[0, :3] = 4
    output, next_state = model.forward_step(
        torch.tensor([[[1.02, 0.01], [-1.02, -0.01]]]),
        torch.zeros(1, 2, dtype=torch.bool),
        torch.tensor([[3, 7]]),
        torch.tensor([0.1]),
        [torch.tensor([3, 7])],
        state,
        association_oracle_probability=1.0,
        association_update_mode="moment",
        birth_state_mode="predicted",
        death_state_mode="predicted",
        retention_threshold=0.3,
        survival_warmup_frames=3,
    )
    assert output.oracle_association_used.tolist() == [True]
    assert next_state.active_mask[0, :3].tolist() == [True, True, False]


def test_joint_lifecycle_empty_measurement_frame_is_finite():
    model = _joint_lifecycle_model().eval()
    state = _state(model)
    output, _ = model.forward_step(
        torch.zeros(1, 1, 2),
        torch.ones(1, 1, dtype=torch.bool),
        torch.full((1, 1), -2, dtype=torch.long),
        torch.tensor([0.1]),
        [torch.tensor([3, 7])],
        state,
        association_oracle_probability=0.0,
        association_update_mode="moment",
        death_state_mode="predicted",
    )
    row_mass = (
        output.predicted_pair_probabilities.sum(-1)
        + output.predicted_miss_probabilities
        + output.predicted_death_probabilities
    )
    torch.testing.assert_close(row_mass[0, :2], torch.ones(2), atol=1e-4, rtol=1e-4)
    assert torch.isfinite(output.posterior_mean).all()


def test_survival_prior_moves_live_and_dead_mass_in_opposite_directions():
    model = _joint_lifecycle_model().eval()
    state = _state(model)
    arguments = (
        torch.tensor([[[1.02, 0.01], [-1.02, -0.01]]]),
        torch.zeros(1, 2, dtype=torch.bool),
        torch.tensor([[3, 7]]),
        torch.tensor([0.1]),
        [torch.tensor([3, 7])],
        state,
    )
    with torch.no_grad():
        model.survival_head[-1].weight.zero_()
        model.survival_head[-1].bias.fill_(-4.0)
    low_survival, _ = model.forward_step(
        *arguments,
        association_oracle_probability=0.0,
        association_update_mode="moment",
    )
    with torch.no_grad():
        model.survival_head[-1].bias.fill_(4.0)
    high_survival, _ = model.forward_step(
        *arguments,
        association_oracle_probability=0.0,
        association_update_mode="moment",
    )
    assert torch.all(
        high_survival.predicted_death_probabilities[0, :2]
        < low_survival.predicted_death_probabilities[0, :2]
    )
    assert torch.all(
        high_survival.predicted_pair_probabilities[0, :2].sum(-1)
        + high_survival.predicted_miss_probabilities[0, :2]
        > low_survival.predicted_pair_probabilities[0, :2].sum(-1)
        + low_survival.predicted_miss_probabilities[0, :2]
    )


def test_sinkhorn_transport_enforces_track_and_measurement_capacity():
    model = _model().eval()
    output, _ = model.forward_step(
        torch.tensor([[[1.02, 0.01], [-1.02, -0.01], [4.0, 4.0]]]),
        torch.zeros(1, 3, dtype=torch.bool),
        torch.tensor([[3, 7, -1]]),
        torch.tensor([0.1]),
        [torch.tensor([3, 7])],
        _state(model),
        association_oracle_probability=0.0,
    )
    row_mass = (
        output.predicted_pair_probabilities.sum(-1)
        + output.predicted_miss_probabilities
    )
    column_mass = (
        output.predicted_pair_probabilities.sum(-2)
        + output.predicted_unclaimed_probabilities
    )
    torch.testing.assert_close(row_mass[0, :2], torch.ones(2), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(column_mass[0], torch.ones(3), atol=1e-4, rtol=1e-4)
    assert output.association_capacity_violation.max() < 1e-6


def test_association_is_equivariant_to_measurement_permutation():
    model = _model().eval()
    measurements = torch.tensor([[[1.02, 0.01], [-1.02, -0.01], [4.0, 4.0]]])
    identifiers = torch.tensor([[3, 7, -1]])
    permutation = torch.tensor([2, 0, 1])
    output, _ = model.forward_step(
        measurements,
        torch.zeros(1, 3, dtype=torch.bool),
        identifiers,
        torch.tensor([0.1]),
        [torch.tensor([3, 7])],
        _state(model),
        association_oracle_probability=0.0,
    )
    permuted, _ = model.forward_step(
        measurements[:, permutation],
        torch.zeros(1, 3, dtype=torch.bool),
        identifiers[:, permutation],
        torch.tensor([0.1]),
        [torch.tensor([3, 7])],
        _state(model),
        association_oracle_probability=0.0,
    )
    torch.testing.assert_close(
        permuted.predicted_pair_probabilities,
        output.predicted_pair_probabilities[:, :, permutation],
        atol=1e-5,
        rtol=1e-4,
    )


def test_confirmed_association_prior_changes_competition_not_capacity():
    model = _model().eval()
    state = _state(model)
    state.confirmed_mask[0, 1] = False
    arguments = (
        torch.tensor([[[0.0, 0.0]]]),
        torch.zeros(1, 1, dtype=torch.bool),
        torch.tensor([[-1]]),
        torch.tensor([0.1]),
        [torch.tensor([3, 7])],
        state,
    )
    unbiased, _ = model.forward_step(
        *arguments,
        association_oracle_probability=0.0,
    )
    biased, _ = model.forward_step(
        *arguments,
        association_oracle_probability=0.0,
        confirmed_association_bias=2.0,
    )
    assert (
        biased.predicted_pair_probabilities[0, 0, 0]
        > unbiased.predicted_pair_probabilities[0, 0, 0]
    )
    assert (
        biased.predicted_pair_probabilities[0, 1, 0]
        < unbiased.predicted_pair_probabilities[0, 1, 0]
    )
    torch.testing.assert_close(
        biased.predicted_pair_probabilities.sum(-2)
        + biased.predicted_unclaimed_probabilities,
        torch.ones(1, 1),
        atol=1e-4,
        rtol=1e-4,
    )


def test_association_supervision_reaches_new_parameters():
    model = _model().train()
    output, _ = model.forward_step(
        torch.tensor([[[1.02, 0.01], [-1.02, -0.01], [4.0, 4.0]]]),
        torch.zeros(1, 3, dtype=torch.bool),
        torch.tensor([[3, 7, -1]]),
        torch.tensor([0.1]),
        [torch.tensor([3, 7])],
        _state(model),
        association_oracle_probability=1.0,
    )
    loss = V17BAssociationCriterion()(output)
    loss.total.backward()
    gradient = model.association.residual_score.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.norm() > 0


def test_balanced_association_loss_does_not_dilute_pair_events():
    model = _model().eval()
    state = _state(model)
    state.active_mask[0, 2] = True
    state.owner_ids[0, 2] = 8
    state.supervision_ids[0, 2] = -1
    state.next_runtime_id[0] = 9
    output, _ = model.forward_step(
        torch.tensor([[[1.02, 0.01], [-1.02, -0.01], [4.0, 4.0]]]),
        torch.zeros(1, 3, dtype=torch.bool),
        torch.tensor([[3, 7, -1]]),
        torch.tensor([0.1]),
        [torch.tensor([3, 7])],
        state,
        association_oracle_probability=0.0,
        death_state_mode="predicted",
    )
    ordinary = V17BAssociationCriterion()(output)
    balanced = V17BAssociationCriterion(
        balance_event_types=True,
        pair_weight=2.0,
        miss_weight=1.0,
        unclaimed_weight=1.0,
    )(output)
    assert ordinary.pair_events == 2
    assert ordinary.miss_events == 1
    assert ordinary.unclaimed_events == 1
    ordinary_expected = (
        ordinary.pair_events * ordinary.pair_nll
        + ordinary.miss_events * ordinary.miss_nll
        + ordinary.unclaimed_events * ordinary.unclaimed_nll
    ) / ordinary.supervised_events
    torch.testing.assert_close(ordinary.total, ordinary_expected)
    balanced_expected = (
        2.0 * balanced.pair_nll + balanced.miss_nll + balanced.unclaimed_nll
    ) / 4.0
    torch.testing.assert_close(balanced.total, balanced_expected)


def test_empty_frame_transport_is_finite_and_all_tracks_miss():
    model = _model().eval()
    output, _ = model.forward_step(
        torch.zeros(1, 1, 2),
        torch.ones(1, 1, dtype=torch.bool),
        torch.full((1, 1), -2, dtype=torch.long),
        torch.tensor([0.1]),
        [torch.tensor([3, 7])],
        _state(model),
        association_oracle_probability=0.0,
    )
    torch.testing.assert_close(
        output.predicted_miss_probabilities[0, :2], torch.ones(2)
    )
    assert torch.isfinite(output.posterior_mean).all()


def test_oracle_schedule_selects_an_entire_frame_without_blending():
    model = _model().eval()
    output, _ = model.forward_step(
        torch.tensor([[[1.02, 0.01], [4.0, 4.0]]]),
        torch.zeros(1, 2, dtype=torch.bool),
        torch.tensor([[3, -1]]),
        torch.tensor([0.1]),
        [torch.tensor([3, 7])],
        _state(model),
        association_oracle_probability=1.0,
    )
    torch.testing.assert_close(
        output.association_strength[0, :2], torch.tensor([1.0, 0.0])
    )
    assert output.oracle_association_used.tolist() == [True]


def test_predicted_transport_backpropagates_through_recurrent_update():
    model = _model().train()
    state = _state(model)
    outputs = []
    for time_index in range(2):
        output, state = model.forward_step(
            torch.tensor(
                [[[1.02 + 0.05 * time_index, 0.01], [-1.02, -0.01], [4.0, 4.0]]]
            ),
            torch.zeros(1, 3, dtype=torch.bool),
            torch.tensor([[3, 7, -1]]),
            torch.tensor([0.1 * (time_index + 1)]),
            [torch.tensor([3, 7])],
            state,
            association_oracle_probability=0.0,
        )
        outputs.append(output)
    loss = sum(V17BAssociationCriterion()(output).total for output in outputs)
    loss = loss + outputs[-1].posterior_mean[0, :2, :2].square().mean()
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.association.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_hard_update_uses_each_measurement_at_most_once():
    model = _model().eval()
    pair = torch.tensor(
        [[[0.6, 0.3], [0.55, 0.2], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]
    )
    miss = torch.tensor([[0.1, 0.25, 0.0, 0.0, 0.0, 0.0]])
    hard = model._hard_association_pairs(
        pair,
        miss,
        torch.tensor([[True, True, False, False, False, False]]),
        torch.tensor([[False, False]]),
    )
    assert torch.all(hard.sum(-2) <= 1.0)
    assert torch.all(hard.sum(-1) <= 1.0)
    # Global assignment chooses track 0 -> measurement 1 and track 1 -> 0,
    # which has lower joint cost than greedily giving measurement 0 to track 0.
    assert hard[0, 0, 1] == 1.0
    assert hard[0, 1, 0] == 1.0


def test_shared_oracle_burn_in_matches_recomputed_state():
    model = _model().train()
    trainer = V17BSequenceTrainer(
        model,
        torch.optim.Adam(model.parameters(), lr=1e-4),
        V17AOracleCriterion(prior_nll_weight=0.0, posterior_nll_weight=0.0),
        V17BAssociationCriterion(),
        burn_in_frames=2,
    )
    simulation = SimulationConfig(
        initial_targets=2,
        max_targets=4,
        birth_rate=0.1,
        clutter_rate=2.0,
    )
    trajectories = [MultiTargetSimulator(simulation, seed=19).simulate(4)]

    recomputed, _ = trainer._sequence_loss(
        trajectories,
        0.0,
        burn_in_association_oracle_probability=1.0,
    )
    shared_state = trainer.prepare_burn_in_state(trajectories, 1.0)
    shared, diagnostics = trainer._sequence_loss(
        trajectories,
        0.0,
        burn_in_association_oracle_probability=1.0,
        initial_state=shared_state,
    )

    torch.testing.assert_close(shared, recomputed)
    assert diagnostics["burn_in_oracle_association_probability"] == 1.0


def test_moment_update_matches_ordinary_update_for_one_hot_association():
    model = _model().eval()
    measurements = torch.tensor([[[1.02, 0.01], [-1.02, -0.01], [4.0, 4.0]]])
    mask = torch.zeros(1, 3, dtype=torch.bool)
    identifiers = torch.tensor([[3, 7, -1]])
    truth_ids = [torch.tensor([3, 7])]
    soft_output, soft_state = model.forward_step(
        measurements,
        mask,
        identifiers,
        torch.tensor([0.1]),
        truth_ids,
        _state(model),
        association_oracle_probability=1.0,
        association_update_mode="soft",
    )
    moment_output, moment_state = model.forward_step(
        measurements,
        mask,
        identifiers,
        torch.tensor([0.1]),
        truth_ids,
        _state(model),
        association_oracle_probability=1.0,
        association_update_mode="moment",
    )
    torch.testing.assert_close(moment_output.posterior_mean, soft_output.posterior_mean)
    torch.testing.assert_close(
        moment_output.posterior_covariance, soft_output.posterior_covariance
    )
    torch.testing.assert_close(moment_state.query, soft_state.query)


def test_moment_update_adds_between_hypothesis_covariance():
    model = _model().eval()
    prior_mean = torch.zeros(1, 1, 4)
    prior_covariance = 0.1 * torch.eye(4).view(1, 1, 4, 4)
    prior_query = torch.zeros(1, 1, model.config.hidden_dim)
    measurements = torch.tensor([[[-1.0, 0.0], [1.0, 0.0]]])
    mask = torch.zeros(1, 2, dtype=torch.bool)
    embeddings = model.encoder(measurements, mask)
    probabilities = torch.full((1, 1, 2), 0.5)
    delta_t = torch.tensor([[0.1]])

    moment_mean, moment_covariance, _ = model.update.forward_mixture(
        prior_mean,
        prior_covariance,
        prior_query,
        measurements,
        embeddings,
        delta_t,
        probabilities,
    )
    averaged_mean, averaged_covariance, _ = model.update(
        prior_mean,
        prior_covariance,
        prior_query,
        torch.zeros(1, 1, 2),
        embeddings.mean(1, keepdim=True),
        delta_t,
        torch.ones(1, 1),
    )

    torch.testing.assert_close(moment_mean, averaged_mean, atol=1e-6, rtol=1e-5)
    assert moment_covariance[0, 0, 0, 0] > averaged_covariance[0, 0, 0, 0]
    assert torch.linalg.eigvalsh(moment_covariance[0, 0]).min() > 0


def test_moment_update_backpropagates_through_recurrent_state():
    model = _model().train()
    state = _state(model)
    outputs = []
    for time_index in range(2):
        output, state = model.forward_step(
            torch.tensor(
                [[[1.02 + 0.05 * time_index, 0.01], [-1.02, -0.01], [4.0, 4.0]]]
            ),
            torch.zeros(1, 3, dtype=torch.bool),
            torch.tensor([[3, 7, -1]]),
            torch.tensor([0.1 * (time_index + 1)]),
            [torch.tensor([3, 7])],
            state,
            association_oracle_probability=0.0,
            association_update_mode="moment",
        )
        outputs.append(output)
    loss = V17BAssociationCriterion()(outputs[-1]).total
    loss = loss + outputs[-1].posterior_mean[0, :2, :2].square().mean()
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in (*model.association.parameters(), *model.update.parameters())
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_neural_posterior_decoder_is_capacity_constrained_and_finite():
    torch.manual_seed(117)
    model = JointAssociationTracker(
        V17BConfig(
            hidden_dim=32,
            num_heads=4,
            encoder_layers=1,
            feedforward_dim=64,
            max_tracks=6,
            dropout=0.0,
            association_sinkhorn_iterations=30,
            association_decoder_layers=2,
            association_residual_scale=8.0,
            association_shared_observation_noise=True,
            association_learned_physical_gate=True,
        )
    ).eval()
    output, state = model.forward_step(
        torch.tensor([[[1.02, 0.01], [-1.02, -0.01], [4.0, 4.0]]]),
        torch.zeros(1, 3, dtype=torch.bool),
        torch.tensor([[3, 7, -1]]),
        torch.tensor([0.1]),
        [torch.tensor([3, 7])],
        _state(model),
        association_oracle_probability=0.0,
        association_update_mode="moment",
    )
    row_mass = (
        output.predicted_pair_probabilities.sum(-1)
        + output.predicted_miss_probabilities
    )
    column_mass = (
        output.predicted_pair_probabilities.sum(-2)
        + output.predicted_unclaimed_probabilities
    )
    torch.testing.assert_close(row_mass[0, :2], torch.ones(2), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(column_mass[0], torch.ones(3), atol=1e-4, rtol=1e-4)
    assert torch.isfinite(output.posterior_mean).all()
    assert torch.isfinite(output.posterior_covariance).all()
    state.validate()


def test_shared_observation_model_trains_from_association_and_decoder():
    torch.manual_seed(119)
    model = JointAssociationTracker(
        V17BConfig(
            hidden_dim=32,
            num_heads=4,
            encoder_layers=1,
            feedforward_dim=64,
            max_tracks=6,
            dropout=0.0,
            association_sinkhorn_iterations=30,
            association_decoder_layers=2,
            association_residual_scale=8.0,
            association_shared_observation_noise=True,
            association_learned_physical_gate=True,
        )
    ).train()
    output, _ = model.forward_step(
        torch.tensor([[[1.02, 0.01], [-1.02, -0.01], [4.0, 4.0]]]),
        torch.zeros(1, 3, dtype=torch.bool),
        torch.tensor([[3, 7, -1]]),
        torch.tensor([0.1]),
        [torch.tensor([3, 7])],
        _state(model),
        association_oracle_probability=0.0,
        association_update_mode="moment",
    )
    V17BAssociationCriterion()(output).total.backward()
    variance_gradient = model.update.measurement_variance_head.weight.grad
    decoder_gradient = (
        model.association.decoder_layers[0].cross_attention.in_proj_weight.grad
    )
    assert variance_gradient is not None
    assert decoder_gradient is not None
    assert torch.isfinite(variance_gradient).all()
    assert torch.isfinite(decoder_gradient).all()
    assert variance_gradient.norm() > 0
    assert decoder_gradient.norm() > 0


def test_neural_posterior_decoder_handles_empty_tracks_and_measurements():
    torch.manual_seed(121)
    model = JointAssociationTracker(
        V17BConfig(
            hidden_dim=32,
            num_heads=4,
            encoder_layers=1,
            feedforward_dim=64,
            max_tracks=6,
            dropout=0.0,
            association_decoder_layers=1,
            association_shared_observation_noise=True,
            association_learned_physical_gate=True,
        )
    ).eval()
    output, state = model.forward_step(
        torch.zeros(1, 1, 2),
        torch.ones(1, 1, dtype=torch.bool),
        torch.full((1, 1), -2, dtype=torch.long),
        torch.tensor([0.1]),
        [torch.empty(0, dtype=torch.long)],
        model.empty_state(1),
        association_oracle_probability=0.0,
        association_update_mode="moment",
        birth_state_mode="predicted",
        death_state_mode="predicted",
    )
    assert torch.isfinite(output.predicted_pair_probabilities).all()
    assert torch.isfinite(output.predicted_unclaimed_probabilities).all()
    state.validate()


def test_precomputed_pair_candidates_preserve_moment_update_exactly():
    torch.manual_seed(123)
    model = _model().eval()
    prior_mean = torch.randn(1, 2, 4)
    prior_covariance = 0.2 * torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
    prior_query = torch.randn(1, 2, model.config.hidden_dim)
    measurements = torch.randn(1, 3, 2)
    mask = torch.zeros(1, 3, dtype=torch.bool)
    embeddings = model.encoder(measurements, mask)
    delta_t = torch.full((1, 2), 0.1)
    probabilities = torch.tensor([[[0.6, 0.1, 0.0], [0.0, 0.2, 0.5]]])
    candidates = model.update.pair_update_candidates(
        prior_mean,
        prior_covariance,
        prior_query,
        measurements,
        embeddings,
        delta_t,
    )
    direct = model.update.forward_mixture(
        prior_mean,
        prior_covariance,
        prior_query,
        measurements,
        embeddings,
        delta_t,
        probabilities,
        candidates.measurement_variance,
    )
    cached = model.update.forward_mixture(
        prior_mean,
        prior_covariance,
        prior_query,
        measurements,
        embeddings,
        delta_t,
        probabilities,
        candidates.measurement_variance,
        candidates,
    )
    for cached_value, direct_value in zip(cached, direct):
        torch.testing.assert_close(cached_value, direct_value, rtol=0.0, atol=0.0)
