from __future__ import annotations

import torch

from track_mt3.models_v17 import (
    EndToEndRecursiveTracker,
    JointAssociationTracker,
    SeparatedLifecycleTracker,
    V17BConfig,
)
from track_mt3.models_v17.death_criterion import (
    V18CBernoulliCriterion,
    V17DeathCriterion,
)
from track_mt3.models_v17.lifecycle_criterion import V17BirthCriterion


def _model() -> JointAssociationTracker:
    torch.manual_seed(19)
    return JointAssociationTracker(
        V17BConfig(
            hidden_dim=32,
            num_heads=4,
            encoder_layers=1,
            feedforward_dim=64,
            max_tracks=6,
            dropout=0.0,
        )
    )


def _evidence_model() -> EndToEndRecursiveTracker:
    torch.manual_seed(1811)
    return EndToEndRecursiveTracker(
        V17BConfig(
            hidden_dim=32,
            num_heads=4,
            encoder_layers=1,
            feedforward_dim=64,
            max_tracks=6,
            dropout=0.0,
            track_evidence_recurrence=True,
        )
    )


def _log_odds_model() -> EndToEndRecursiveTracker:
    torch.manual_seed(1813)
    return EndToEndRecursiveTracker(
        V17BConfig(
            hidden_dim=32,
            num_heads=4,
            encoder_layers=1,
            feedforward_dim=64,
            max_tracks=6,
            dropout=0.0,
            track_evidence_recurrence=True,
            track_evidence_log_odds_update=True,
            track_survival_probability_init=0.98,
            track_evidence_pair_anchor_scale=2.0,
            track_evidence_pair_anchor_center=0.15,
        )
    )


def test_birth_factorization_uses_all_measurements_without_changing_oracle_slots():
    model = _model().train()
    state = model.empty_state(1)
    state.active_mask[0, 0] = True
    state.confirmed_mask[0, 0] = True
    state.owner_ids[0, 0] = 3
    state.supervision_ids[0, 0] = 3
    state.next_runtime_id[0] = 4
    output, next_state = model.forward_step(
        torch.tensor([[[0.1, 0.0], [2.0, 1.0], [-3.0, 4.0]]]),
        torch.zeros(1, 3, dtype=torch.bool),
        torch.tensor([[3, 5, -1]]),
        torch.tensor([0.1]),
        [torch.tensor([3, 5])],
        state,
        association_oracle_probability=0.0,
        association_update_mode="moment",
    )
    loss = V17BirthCriterion()(output)
    assert loss.target_measurements == 2
    assert loss.clutter_measurements == 1
    assert loss.newborn_measurements == 1
    assert output.measurement_objectness_logits.shape == (1, 3)
    torch.testing.assert_close(
        output.newborn_probabilities(),
        output.measurement_objectness_logits.sigmoid()
        * output.predicted_unclaimed_probabilities,
    )
    # Oracle lifecycle remains isolated in G3a: only the true newborn is written.
    assert set(next_state.owner_ids[0, next_state.active_mask[0]].tolist()) == {3, 5}
    loss.total.backward()
    gradient = model.birth_existence_head[0].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.norm() > 0
    assert model.association.residual_score.weight.grad is None


def test_birth_loss_is_invariant_to_repeating_clutter_count():
    model = _model().eval()
    outputs = []
    for clutter_count in (1, 4):
        measurements = torch.zeros(1, 1 + clutter_count, 2)
        identifiers = torch.tensor([[7] + [-1] * clutter_count])
        output, _ = model.forward_step(
            measurements,
            torch.zeros_like(identifiers, dtype=torch.bool),
            identifiers,
            torch.tensor([0.0]),
            [torch.tensor([7])],
            association_oracle_probability=0.0,
            association_update_mode="moment",
        )
        outputs.append(V17BirthCriterion()(output).measurement_type)
    torch.testing.assert_close(outputs[0], outputs[1], atol=1e-5, rtol=1e-5)


def _force_birth_candidates(model: JointAssociationTracker) -> None:
    with torch.no_grad():
        model.birth_existence_head[-1].weight.zero_()
        model.birth_existence_head[-1].bias.fill_(20.0)


def test_predicted_birth_slot_decisions_do_not_depend_on_measurement_labels():
    model = _model().eval()
    _force_birth_candidates(model)
    measurements = torch.tensor([[[1.0, 2.0], [-3.0, 0.5]]])
    states = []
    for identifiers, truth_ids in (
        (torch.tensor([[5, -1]]), [torch.tensor([5])]),
        (torch.tensor([[-1, 8]]), [torch.tensor([8])]),
    ):
        _, state = model.forward_step(
            measurements,
            torch.zeros(1, 2, dtype=torch.bool),
            identifiers,
            torch.tensor([0.0]),
            truth_ids,
            birth_state_mode="predicted",
            birth_candidate_threshold=0.5,
            confirmation_hits=2,
        )
        states.append(state)
    torch.testing.assert_close(states[0].active_mask, states[1].active_mask)
    torch.testing.assert_close(states[0].owner_ids, states[1].owner_ids)
    torch.testing.assert_close(states[0].mean, states[1].mean)
    assert not states[0].confirmed_mask.any()
    assert not torch.equal(states[0].supervision_ids, states[1].supervision_ids)


def test_tentative_true_birth_confirms_on_next_association():
    model = _model().eval()
    _force_birth_candidates(model)
    measurement = torch.tensor([[[1.0, -0.5]]])
    _, state = model.forward_step(
        measurement,
        torch.zeros(1, 1, dtype=torch.bool),
        torch.tensor([[5]]),
        torch.tensor([0.0]),
        [torch.tensor([5])],
        birth_state_mode="predicted",
        birth_candidate_threshold=0.5,
        confirmation_hits=2,
    )
    assert state.active_mask[0, 0]
    assert not state.confirmed_mask[0, 0]
    output, state = model.forward_step(
        measurement,
        torch.zeros(1, 1, dtype=torch.bool),
        torch.tensor([[5]]),
        torch.tensor([0.1]),
        [torch.tensor([5])],
        state,
        association_oracle_probability=0.0,
        association_update_mode="moment",
        birth_state_mode="predicted",
        birth_candidate_threshold=0.5,
        confirmation_hits=2,
    )
    assert output.association_strength[0, 0] >= 0.5
    assert state.confirmed_mask[0, 0]
    assert state.hit_streak[0, 0] >= 2


def test_oracle_death_gate_removes_false_predicted_birth_before_next_update():
    model = _model().eval()
    _force_birth_candidates(model)
    _, state = model.forward_step(
        torch.tensor([[[2.0, 3.0]]]),
        torch.zeros(1, 1, dtype=torch.bool),
        torch.tensor([[-1]]),
        torch.tensor([0.0]),
        [torch.empty(0, dtype=torch.long)],
        birth_state_mode="predicted",
        birth_candidate_threshold=0.5,
    )
    assert state.active_mask[0, 0]
    _, state = model.forward_step(
        torch.zeros(1, 1, 2),
        torch.ones(1, 1, dtype=torch.bool),
        torch.full((1, 1), -2, dtype=torch.long),
        torch.tensor([0.1]),
        [torch.empty(0, dtype=torch.long)],
        state,
        association_oracle_probability=0.0,
        association_update_mode="moment",
        birth_state_mode="predicted",
        birth_candidate_threshold=0.5,
    )
    assert not state.active_mask.any()
    assert state.next_runtime_id.item() == 1


def test_oracle_post_prune_exposes_dead_tracks_to_loss_then_cleans_state():
    model = _model().eval()
    state = model.empty_state(1)
    state.active_mask[0, :2] = True
    state.confirmed_mask[0, :2] = True
    state.owner_ids[0, :2] = torch.tensor([10, 11])
    state.supervision_ids[0, :2] = torch.tensor([5, -1])
    state.next_runtime_id[0] = 12
    output, next_state = model.forward_step(
        torch.tensor([[[0.1, 0.0]]]),
        torch.zeros(1, 1, dtype=torch.bool),
        torch.tensor([[5]]),
        torch.tensor([0.1]),
        [torch.tensor([5])],
        state,
        association_oracle_probability=0.0,
        association_update_mode="moment",
        birth_state_mode="predicted",
        birth_candidate_threshold=1.0,
        death_state_mode="oracle_post",
    )
    death_loss = V17DeathCriterion()(output, [torch.tensor([5])])
    assert death_loss.alive_tracks == 1
    assert death_loss.dead_tracks == 1
    assert output.existing_mask.sum() == 2
    assert next_state.active_mask.sum() == 1
    assert next_state.supervision_ids[0, next_state.active_mask[0]].item() == 5


def test_predicted_death_decision_does_not_depend_on_truth_metadata():
    model = _model().eval()
    base = model.empty_state(1)
    base.active_mask[0, 0] = True
    base.confirmed_mask[0, 0] = True
    base.owner_ids[0, 0] = 10
    base.next_runtime_id[0] = 11
    resulting_states = []
    for supervision_id, truth_id in ((5, 5), (-1, 99)):
        state = base.clone()
        state.supervision_ids[0, 0] = supervision_id
        _, state = model.forward_step(
            torch.zeros(1, 1, 2),
            torch.ones(1, 1, dtype=torch.bool),
            torch.full((1, 1), -2, dtype=torch.long),
            torch.tensor([0.1]),
            [torch.tensor([truth_id])],
            state,
            association_oracle_probability=0.0,
            association_update_mode="moment",
            birth_state_mode="predicted",
            birth_candidate_threshold=1.0,
            death_state_mode="predicted",
            retention_threshold=0.5,
        )
        resulting_states.append(state)
    torch.testing.assert_close(
        resulting_states[0].active_mask, resulting_states[1].active_mask
    )
    torch.testing.assert_close(
        resulting_states[0].confirmed_mask, resulting_states[1].confirmed_mask
    )


def test_delayed_oracle_death_exposes_three_negative_frames():
    model = _model().eval()
    state = model.empty_state(1)
    state.active_mask[0, 0] = True
    state.confirmed_mask[0, 0] = True
    state.owner_ids[0, 0] = 10
    state.supervision_ids[0, 0] = -1
    state.next_runtime_id[0] = 11
    exposed = []
    for time_index in range(3):
        output, state = model.forward_step(
            torch.zeros(1, 1, 2),
            torch.ones(1, 1, dtype=torch.bool),
            torch.full((1, 1), -2, dtype=torch.long),
            torch.tensor([0.1 * (time_index + 1)]),
            [torch.empty(0, dtype=torch.long)],
            state,
            association_oracle_probability=0.0,
            association_update_mode="moment",
            birth_state_mode="predicted",
            birth_candidate_threshold=1.0,
            death_state_mode="oracle_delayed",
            oracle_prune_delay=3,
        )
        exposed.append(bool(output.existing_mask[0, 0]))
    assert exposed == [True, True, True]
    assert not state.active_mask.any()


def test_separate_survival_head_receives_hard_negative_gradient():
    base = _model().eval()
    model = SeparatedLifecycleTracker(base.config).train()
    incompatible = model.load_state_dict(base.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert set(incompatible.missing_keys) == {
        name for name in model.state_dict() if name.startswith("survival_head.")
    }
    state = model.empty_state(1)
    state.active_mask[0, :2] = True
    state.owner_ids[0, :2] = torch.tensor([10, 11])
    state.supervision_ids[0, :2] = torch.tensor([5, -1])
    state.next_runtime_id[0] = 12
    output, _ = model.forward_step(
        torch.tensor([[[0.0, 0.0]]]),
        torch.zeros(1, 1, dtype=torch.bool),
        torch.tensor([[5]]),
        torch.tensor([0.1]),
        [torch.tensor([5])],
        state,
        association_oracle_probability=0.0,
        association_update_mode="moment",
        birth_state_mode="predicted",
        birth_candidate_threshold=1.0,
        death_state_mode="oracle_delayed",
        oracle_prune_delay=3,
    )
    loss = V17DeathCriterion(
        logit_field="survival_logits",
        hard_negative_fraction=0.25,
        hard_negative_weight=1.0,
    )(output, [torch.tensor([5])])
    loss.total.backward()
    gradient = model.survival_head[0].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.norm() > 0
    assert model.existence_head[0].weight.grad is None


def test_predicted_survival_warmup_protects_only_young_tracks():
    base = _model().eval()
    model = SeparatedLifecycleTracker(base.config).eval()
    with torch.no_grad():
        model.survival_head[-1].weight.zero_()
        model.survival_head[-1].bias.fill_(-20.0)
    state = model.empty_state(1)
    state.active_mask[0, :2] = True
    state.owner_ids[0, :2] = torch.tensor([10, 11])
    state.supervision_ids[0, :2] = torch.tensor([5, 6])
    state.ages[0, :2] = torch.tensor([0, 3])
    state.next_runtime_id[0] = 12
    _, next_state = model.forward_step(
        torch.zeros(1, 1, 2),
        torch.ones(1, 1, dtype=torch.bool),
        torch.full((1, 1), -2, dtype=torch.long),
        torch.tensor([0.1]),
        [torch.tensor([5, 6])],
        state,
        association_oracle_probability=0.0,
        association_update_mode="moment",
        birth_state_mode="predicted",
        birth_candidate_threshold=1.0,
        death_state_mode="predicted",
        retention_threshold=0.5,
        survival_warmup_frames=3,
    )
    assert next_state.active_mask[0, 0]
    assert not next_state.active_mask[0, 1]


def test_neural_evidence_is_one_posterior_for_output_and_survival():
    model = _evidence_model().train()
    state = model.empty_state(1)
    state.active_mask[0, :2] = True
    state.owner_ids[0, :2] = torch.tensor([10, 11])
    state.supervision_ids[0, :2] = torch.tensor([5, -1])
    state.next_runtime_id[0] = 12
    output, next_state = model.forward_step(
        torch.tensor([[[0.0, 0.0]]]),
        torch.zeros(1, 1, dtype=torch.bool),
        torch.tensor([[5]]),
        torch.tensor([0.1]),
        [torch.tensor([5])],
        state,
        association_oracle_probability=0.0,
        association_update_mode="moment",
        birth_state_mode="predicted",
        birth_candidate_threshold=1.0,
        death_state_mode="oracle_delayed",
        oracle_prune_delay=3,
    )
    torch.testing.assert_close(output.existing_logits, output.survival_logits)
    assert not torch.equal(next_state.evidence_hidden, state.evidence_hidden)
    loss = V17DeathCriterion(logit_field="survival_logits")(
        output, [torch.tensor([5])]
    )
    loss.total.backward()
    for gradient in (
        model.track_evidence_head[-1].weight.grad,
        model.track_evidence_cell.weight_ih.grad,
    ):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0


def test_neural_evidence_confirmation_does_not_use_hit_count():
    model = _evidence_model().eval()
    with torch.no_grad():
        model.existence_head[-1].weight.zero_()
        model.existence_head[-1].bias.fill_(8.0)
        model.track_evidence_head[-1].weight.zero_()
        model.track_evidence_head[-1].bias.zero_()
    state = model.empty_state(1)
    state.active_mask[0, 0] = True
    state.owner_ids[0, 0] = 10
    state.supervision_ids[0, 0] = 5
    state.exist_logit[0, 0] = 8.0
    state.next_runtime_id[0] = 11
    _, next_state = model.forward_step(
        torch.zeros(1, 1, 2),
        torch.ones(1, 1, dtype=torch.bool),
        torch.full((1, 1), -2, dtype=torch.long),
        torch.tensor([0.1]),
        [torch.tensor([5])],
        state,
        association_oracle_probability=0.0,
        association_update_mode="moment",
        birth_state_mode="predicted",
        birth_candidate_threshold=1.0,
        confirmation_hits=99,
        death_state_mode="predicted",
        retention_threshold=0.3,
        survival_warmup_frames=100,
    )
    assert next_state.hit_streak[0, 0] == 0
    assert next_state.confirmed_mask[0, 0]


def test_neural_birth_confirmation_uses_birth_posterior_not_hit_count():
    model = _evidence_model().eval()
    _force_birth_candidates(model)
    _, state = model.forward_step(
        torch.tensor([[[1.0, -0.5]]]),
        torch.zeros(1, 1, dtype=torch.bool),
        torch.tensor([[5]]),
        torch.tensor([0.0]),
        [torch.tensor([5])],
        birth_state_mode="predicted",
        birth_candidate_threshold=0.35,
        confirmation_hits=99,
    )
    assert state.active_mask[0, 0]
    assert state.confirmed_mask[0, 0]
    assert state.hit_streak[0, 0] == 1
    assert state.evidence_hidden[0, 0].norm() > 0


def test_neural_evidence_can_deconfirm_without_releasing_slot():
    model = _evidence_model().eval()
    with torch.no_grad():
        model.existence_head[-1].weight.zero_()
        model.existence_head[-1].bias.fill_(-20.0)
        model.track_evidence_head[-1].weight.zero_()
        model.track_evidence_head[-1].bias.zero_()
    state = model.empty_state(1)
    state.active_mask[0, 0] = True
    state.confirmed_mask[0, 0] = True
    state.owner_ids[0, 0] = 10
    state.supervision_ids[0, 0] = 5
    state.exist_logit[0, 0] = -8.0
    state.next_runtime_id[0] = 11
    _, next_state = model.forward_step(
        torch.zeros(1, 1, 2),
        torch.ones(1, 1, dtype=torch.bool),
        torch.full((1, 1), -2, dtype=torch.long),
        torch.tensor([0.1]),
        [torch.tensor([5])],
        state,
        association_oracle_probability=0.0,
        association_update_mode="moment",
        birth_state_mode="predicted",
        birth_candidate_threshold=1.0,
        death_state_mode="predicted",
        retention_threshold=0.3,
        survival_warmup_frames=100,
    )
    assert next_state.active_mask[0, 0]
    assert not next_state.confirmed_mask[0, 0]


def test_log_odds_evidence_predicts_survival_then_adds_miss_likelihood():
    model = _log_odds_model().eval()
    with torch.no_grad():
        model.track_survival_prior_head[-1].weight.zero_()
        model.track_survival_prior_head[-1].bias.zero_()
        model.track_likelihood_ratio_head[-1].weight.zero_()
        model.track_likelihood_ratio_head[-1].bias.zero_()
    state = model.empty_state(1)
    state.active_mask[0, 0] = True
    state.owner_ids[0, 0] = 10
    state.supervision_ids[0, 0] = 5
    state.exist_logit[0, 0] = torch.logit(torch.tensor(0.8))
    state.next_runtime_id[0] = 11
    output, _ = model.forward_step(
        torch.zeros(1, 1, 2),
        torch.ones(1, 1, dtype=torch.bool),
        torch.full((1, 1), -2, dtype=torch.long),
        torch.tensor([0.1]),
        [torch.tensor([5])],
        state,
        association_oracle_probability=1.0,
        association_update_mode="moment",
        birth_state_mode="predicted",
        birth_candidate_threshold=1.0,
        death_state_mode="oracle_delayed",
        oracle_prune_delay=3,
    )
    predicted_probability = (0.8 * 0.98) / (1.0 - 0.8 + 0.8 * 0.98)
    expected = torch.logit(torch.tensor(predicted_probability)) - 0.3
    torch.testing.assert_close(
        output.survival_prior_logits[0, 0], torch.logit(torch.tensor(0.98))
    )
    torch.testing.assert_close(output.existing_logits[0, 0], expected)
    torch.testing.assert_close(output.existing_logits, output.survival_logits)


def test_log_odds_criterion_scores_natural_posterior_and_true_transition():
    model = _log_odds_model().train()
    state = model.empty_state(1)
    state.active_mask[0, :2] = True
    state.owner_ids[0, :2] = torch.tensor([10, 11])
    state.supervision_ids[0, :2] = torch.tensor([5, 6])
    state.exist_logit[0, :2] = torch.logit(torch.tensor([0.8, 0.8]))
    state.next_runtime_id[0] = 12
    output, _ = model.forward_step(
        torch.zeros(1, 1, 2),
        torch.ones(1, 1, dtype=torch.bool),
        torch.full((1, 1), -2, dtype=torch.long),
        torch.tensor([0.1]),
        [torch.tensor([5])],
        state,
        association_oracle_probability=0.0,
        association_update_mode="moment",
        birth_state_mode="predicted",
        birth_candidate_threshold=1.0,
        death_state_mode="oracle_delayed",
        oracle_prune_delay=3,
    )
    loss = V18CBernoulliCriterion(missed_alive_weight=1.0)(
        output,
        [torch.tensor([5])],
        [torch.tensor([5, 6])],
    )
    assert loss.posterior_tracks == 2
    assert loss.transition_tracks == 2
    assert loss.missed_alive_probability > 0
    (loss.posterior + loss.survival_transition).backward()
    for gradient in (
        model.track_survival_prior_head[-1].weight.grad,
        model.track_likelihood_ratio_head[-1].weight.grad,
        model.track_evidence_cell.weight_ih.grad,
    ):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0
