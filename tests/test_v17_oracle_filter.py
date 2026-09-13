from __future__ import annotations

from dataclasses import replace

import torch

from track_mt3.models_v17 import OracleRecursiveTracker, V17AConfig
from track_mt3.models_v17.criterion import V17AOracleCriterion
from track_mt3.training.trainer_v17 import average_gradients


def _model() -> OracleRecursiveTracker:
    torch.manual_seed(7)
    return OracleRecursiveTracker(
        V17AConfig(
            hidden_dim=32,
            num_heads=4,
            encoder_layers=1,
            feedforward_dim=64,
            max_tracks=6,
            dropout=0.0,
        )
    )


def _active_state(model: OracleRecursiveTracker, owner: int = 3):
    state = model.empty_state(1)
    state.active_mask[0, 0] = True
    state.confirmed_mask[0, 0] = True
    state.owner_ids[0, 0] = owner
    state.supervision_ids[0, 0] = owner
    state.next_runtime_id[0] = owner + 1
    state.covariance[0, 0] = torch.diag(torch.tensor([1.0, 1.0, 2.0, 2.0]))
    return state


def test_current_frame_encoder_is_measurement_permutation_equivariant():
    model = _model().eval()
    measurements = torch.tensor([[[1.0, 2.0], [-3.0, 0.5], [0.2, -1.0]]])
    mask = torch.zeros(1, 3, dtype=torch.bool)
    permutation = torch.tensor([2, 0, 1])
    encoded = model.encoder(measurements, mask)
    permuted = model.encoder(measurements[:, permutation], mask[:, permutation])
    torch.testing.assert_close(permuted, encoded[:, permutation], atol=1e-6, rtol=1e-5)


def test_oracle_miss_coasts_without_shrinking_covariance():
    model = _model().eval()
    state = _active_state(model)
    before = state.covariance[0, 0].clone()
    output, next_state = model.forward_step(
        torch.zeros(1, 1, 2),
        torch.ones(1, 1, dtype=torch.bool),
        torch.full((1, 1), -2, dtype=torch.long),
        torch.tensor([0.1]),
        [torch.tensor([3])],
        state,
    )
    assert not output.association_mask[0, 0]
    torch.testing.assert_close(
        output.posterior_covariance[0, 0], output.prior_covariance[0, 0]
    )
    assert torch.all(
        output.posterior_covariance[0, 0].diagonal() >= before.diagonal()
    )
    assert next_state.miss_streak[0, 0] == 1


def test_vectorized_oracle_pairs_preserve_first_match_and_masks():
    supervision_ids = torch.tensor([[7, -1, 4], [3, 8, 9]])
    measurement_ids = torch.tensor([[7, 5, 7, -1], [8, 3, 3, 9]])
    active_mask = torch.tensor([[True, True, False], [True, True, False]])
    padding_mask = torch.tensor(
        [[False, False, False, False], [False, False, False, True]]
    )

    pair, associated, claimed = OracleRecursiveTracker._oracle_pairs(
        supervision_ids,
        measurement_ids,
        active_mask,
        padding_mask,
        torch.float32,
    )

    expected_pair = torch.zeros(2, 3, 4)
    expected_pair[0, 0, 0] = 1.0
    expected_pair[0, 1, 3] = 1.0
    expected_pair[1, 0, 1] = 1.0
    expected_pair[1, 1, 0] = 1.0
    assert torch.equal(pair, expected_pair)
    assert torch.equal(
        associated,
        torch.tensor([[True, True, False], [True, True, False]]),
    )
    assert torch.equal(
        claimed,
        torch.tensor([[True, False, False, True], [True, True, False, False]]),
    )


def test_oracle_measurement_update_moves_posterior_towards_measurement():
    model = _model().eval()
    state = _active_state(model)
    measurement = torch.tensor([[[1.0, -0.5]]])
    output, _ = model.forward_step(
        measurement,
        torch.zeros(1, 1, dtype=torch.bool),
        torch.tensor([[3]]),
        torch.tensor([0.0]),
        [torch.tensor([3])],
        state,
    )
    prior_error = torch.linalg.vector_norm(output.prior_mean[0, 0, :2] - measurement[0, 0])
    posterior_error = torch.linalg.vector_norm(
        output.posterior_mean[0, 0, :2] - measurement[0, 0]
    )
    assert posterior_error < prior_error
    assert torch.linalg.eigvalsh(output.posterior_covariance[0, 0]).min() > 0.0


def test_oracle_lifecycle_never_allocates_clutter_or_duplicate_owner():
    model = _model().eval()
    _, state = model.forward_step(
        torch.tensor([[[1.0, 1.0], [1.1, 1.0], [-4.0, 2.0]]]),
        torch.zeros(1, 3, dtype=torch.bool),
        torch.tensor([[5, 5, -1]]),
        torch.tensor([0.0]),
        [torch.tensor([5])],
    )
    owners = state.owner_ids[0, state.active_mask[0]].tolist()
    assert owners == [5]
    assert -1 not in owners


def test_recursive_state_contains_no_raw_measurement_cache():
    state = _model().empty_state(2)
    names = set(state.__dict__)
    assert not any("measurement" in name or "cache" in name for name in names)


def test_next_frame_loss_backpropagates_through_birth_state():
    model = _model().train()
    truth_state_0 = torch.tensor([[0.5, -0.25, 0.8, 0.1]])
    _, state = model.forward_step(
        torch.tensor([[[0.5, -0.25]]]),
        torch.zeros(1, 1, dtype=torch.bool),
        torch.tensor([[9]]),
        torch.tensor([0.0]),
        [torch.tensor([9])],
    )
    output, _ = model.forward_step(
        torch.tensor([[[0.58, -0.24]]]),
        torch.zeros(1, 1, dtype=torch.bool),
        torch.tensor([[9]]),
        torch.tensor([0.1]),
        [torch.tensor([9])],
        state,
    )
    truth_state_1 = truth_state_0.clone()
    truth_state_1[:, 0] += 0.08
    truth_state_1[:, 1] += 0.01
    loss = V17AOracleCriterion()(output, [truth_state_1], [torch.tensor([9])])
    loss.total.backward()
    gradient = model.birth_query_head[0].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.norm() > 0.0


def test_deployment_chain_kinematic_anchor_supervises_inherited_state():
    model = _model().train()
    _, state = model.forward_step(
        torch.tensor([[[0.5, -0.25]]]),
        torch.zeros(1, 1, dtype=torch.bool),
        torch.tensor([[9]]),
        torch.tensor([0.0]),
        [torch.tensor([9])],
    )
    output, _ = model.forward_step(
        torch.tensor([[[0.58, -0.24]]]),
        torch.zeros(1, 1, dtype=torch.bool),
        torch.tensor([[9]]),
        torch.tensor([0.1]),
        [torch.tensor([9])],
        state,
    )
    truth_state = torch.tensor([[0.58, -0.24, 0.8, 0.1]])
    truth_ids = [torch.tensor([9])]
    base = V17AOracleCriterion()(output, [truth_state], truth_ids)
    anchored = V17AOracleCriterion(
        prior_position_weight=0.5,
        prior_velocity_weight=0.25,
        posterior_velocity_weight=0.25,
    )(output, [truth_state], truth_ids)
    expected = (
        base.total
        + 0.5 * anchored.prior_position_anchor
        + 0.25 * anchored.prior_velocity_anchor
        + 0.25 * anchored.posterior_velocity_anchor
    )
    torch.testing.assert_close(anchored.total, expected)
    assert anchored.prior_position_anchor > 0.0
    assert anchored.prior_velocity_anchor > 0.0
    anchored.total.backward()
    gradient = model.dynamics.transition[0].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.norm() > 0.0


def test_cta_probabilistic_set_risk_matches_pro_gospa_bernoulli_cost():
    model = _model().eval()
    state = _active_state(model, owner=9)
    output, _ = model.forward_step(
        torch.zeros(1, 1, 2),
        torch.ones(1, 1, dtype=torch.bool),
        torch.full((1, 1), -2, dtype=torch.long),
        torch.tensor([0.1]),
        [torch.tensor([9])],
        state,
    )
    posterior = output.posterior_mean.clone()
    posterior[0, 0, :2] = torch.tensor([0.3, 0.4])
    logits = output.existing_logits.clone()
    logits[0, 0] = 0.0
    output = replace(output, posterior_mean=posterior, existing_logits=logits)
    truth = [torch.tensor([[0.0, 0.0, 0.0, 0.0]])]
    loss = V17AOracleCriterion(set_risk_cutoff=2.0)(
        output, truth, [torch.tensor([9])]
    )
    # p=0.5, distance=0.5, cutoff/2=1: 0.5*0.5 + 0.5*1.
    torch.testing.assert_close(loss.set_risk, torch.tensor(0.75))


def test_constant_velocity_skip_uses_physical_delta_t():
    model = _model().eval()
    state = _active_state(model)
    state.mean[0, 0] = torch.tensor([1.0, 2.0, 3.0, -2.0])
    output, _ = model.forward_step(
        torch.zeros(1, 1, 2),
        torch.ones(1, 1, dtype=torch.bool),
        torch.full((1, 1), -2, dtype=torch.long),
        torch.tensor([0.2]),
        [torch.tensor([3])],
        state,
    )
    # Acceleration head is zero-initialized, so the analytical CV skip is exact.
    torch.testing.assert_close(
        output.prior_mean[0, 0], torch.tensor([1.6, 1.6, 3.0, -2.0])
    )


def test_distributed_gradient_average_participates_for_unused_parameters(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    # Only the second layer has a local gradient, reproducing a rank with no
    # birth event for the first head while another rank may have used it.
    model[1].weight.grad = torch.ones_like(model[1].weight)
    calls = []

    def fake_all_reduce(tensor, op):
        calls.append(tensor)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    average_gradients(model, world_size=2)
    parameters = list(model.parameters())
    assert len(calls) == len(parameters)
    assert all(parameter.grad is not None for parameter in parameters)
    assert torch.count_nonzero(parameters[0].grad) == 0
    torch.testing.assert_close(
        model[1].weight.grad, torch.full_like(model[1].weight, 0.5)
    )


def test_distributed_gradient_average_skips_frozen_parameters(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    model[0].requires_grad_(False)
    model[1].weight.grad = torch.ones_like(model[1].weight)
    calls = []

    def fake_all_reduce(tensor, op):
        calls.append(tensor)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    average_gradients(model, world_size=2)
    assert len(calls) == sum(parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model[0].parameters())
