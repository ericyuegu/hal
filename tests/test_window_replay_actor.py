"""Joint PPO, direct KL and fixed-storage CUDA actor capture."""

import dataclasses

import pytest
import torch
from test_self_play_model import model
from test_window_replay_model import inputs
from torch import Tensor

from hal.inference.graphed_actor import GraphedActorStep
from hal.models.replay_actor import ActorTargets
from hal.models.replay_actor import actor_loss
from hal.models.replay_actor import validate_actor_targets
from hal.models.self_play import DecoderPrecision
from hal.models.self_play import SelfPlayTransformer
from hal.models.window_replay import ReplayOutput
from hal.models.window_replay import replay_windows


def targets_for(output: ReplayOutput) -> ActorTargets:
    shape = output.log_probs_WKF.shape[:2]
    device = output.log_probs_WKF.device
    return ActorTargets(
        output.log_probs_WKF.detach() - 0.01,
        torch.linspace(-2, 2, shape.numel(), device=device).reshape(shape),
        output.factor_mask_WKF.any(-1),
        torch.ones(shape, device=device) * 0.2,
        tuple(p.detach().clone() for p in output.distributions),
        torch.tensor(0.1, device=device),
        torch.tensor(1.3, device=device),
        output.factor_mask_WKF.any(-1).sum().float(),
    )


def test_actor_joint_ratio_masks_and_stop_gradients() -> None:
    policy = model()
    hidden, observed, slots = inputs(policy)
    output = replay_windows(policy, hidden, observed, slots, grad=True)
    targets = targets_for(output)
    validate_actor_targets(output, targets)
    for field in (
        "behavior_log_probs_WKF",
        "advantages_WK",
        "kl_coefficient_WK",
        "advantage_mean",
        "advantage_sigma",
        "denominator",
    ):
        getattr(targets, field).requires_grad_(True)
    for p in targets.reference_distributions:
        p.requires_grad_(True)
    actual = actor_loss(output, targets)
    count = output.factor_mask_WKF.sum(-1)
    torch.testing.assert_close(actual.joint_ratio_WK, (count * 0.01).exp())
    assert float(actual.kl_loss.detach()) == 0
    actual.loss.backward()
    assert targets.advantages_WK.grad is None
    assert targets.behavior_log_probs_WKF.grad is None
    assert targets.kl_coefficient_WK.grad is None
    assert targets.advantage_mean.grad is None
    assert targets.advantage_sigma.grad is None
    assert targets.denominator.grad is None
    assert all(p.grad is None for p in targets.reference_distributions)
    for p in policy.critic_parameters():
        assert p.grad is None
    assert any(p.grad is not None and bool(p.grad.any()) for p in policy.actor_parameters())


def test_actor_ignores_nan_padding_and_matches_enumerated_objective() -> None:
    # A single factor makes the expected PPO+KL gradient explicit.
    logits = torch.tensor([0.2, -0.4, 0.1], requires_grad=True)
    p = logits.softmax(-1)
    q = torch.tensor([0.2, 0.3, 0.5], requires_grad=True)
    mask = torch.zeros(1, 2, 16, dtype=torch.bool)
    mask[0, 0, :4] = True
    policy = model()
    hidden, observed, slots = inputs(policy)
    features = replay_windows(policy, hidden, observed, slots, grad=False).features
    probabilities = p.expand(1, 2, 4, 3)
    logs = p[0].log().expand(1, 2, 16)
    output = ReplayOutput(logs, (probabilities,) * 4, mask, features)
    selected = torch.tensor([[True, False]])
    ref = q.expand_as(probabilities).clone()
    ref[:, 1] = torch.nan
    old = torch.full((1, 2, 16), torch.nan)
    old[:, 0] = float(p[0].detach().log())
    targets = ActorTargets(
        old,
        torch.tensor([[0.8, torch.nan]]),
        selected,
        torch.tensor([[0.3, torch.nan]]),
        (ref,) * 4,
        torch.tensor(0.1),
        torch.tensor(1.4),
        torch.tensor(1.0),
    )
    validate_actor_targets(output, targets)
    loss = actor_loss(output, targets)
    actual = torch.autograd.grad(loss.loss, logits, retain_graph=True)[0]
    expected_loss = (
        -(4 * (p[0].log() - old[0, 0, 0])).exp() * (0.7 / 1.4)
        + 0.3 / 1.4 * 4 * (p * (p.log() - q.detach().log())).sum()
    )
    expected = torch.autograd.grad(expected_loss, logits)[0]
    torch.testing.assert_close(loss.loss, expected_loss)
    torch.testing.assert_close(actual, expected)
    assert q.grad is None


@pytest.mark.parametrize("clip_epsilon", [0.05, 0.2, 0.4])
def test_joint_ppo_clips_both_advantage_signs(clip_epsilon: float) -> None:
    policy = model()
    hidden, observed, slots = inputs(policy)
    output = replay_windows(policy, hidden, observed, slots, grad=False)
    mask = torch.ones(2, 3, 16, dtype=torch.bool)
    logs = torch.zeros(2, 3, 16, requires_grad=True)
    output = dataclasses.replace(output, log_probs_WKF=logs, factor_mask_WKF=mask)
    ratios = torch.tensor([[1.5, 0.5, 1.5], [0.5, 1.0, 1.0]])
    advantages = torch.tensor([[1.0, 1.0, -1.0], [-1.0, 1.0, -1.0]])
    target = ActorTargets(
        -ratios.log()[..., None].expand_as(logs) / 16,
        advantages,
        torch.ones(2, 3, dtype=torch.bool),
        torch.zeros(2, 3),
        output.distributions,
        torch.tensor(0.0),
        torch.tensor(1.0),
        torch.tensor(6.0),
    )
    loss = actor_loss(output, target, clip_epsilon=clip_epsilon)
    torch.testing.assert_close(
        loss.policy_loss, torch.tensor(-((1 + clip_epsilon) + 0.5 - 1.5 - (1 - clip_epsilon)) / 6)
    )
    loss.loss.backward()
    assert logs.grad is not None
    assert not bool(logs.grad[0, 0].any()) and not bool(logs.grad[1, 0].any())
    assert bool((logs.grad[0, 1] < 0).all()) and bool((logs.grad[0, 2] > 0).all())


@pytest.mark.parametrize("case", ["sigma", "reference", "empty", "invalid_slot", "behavior", "coefficient", "dtype"])
def test_actor_rejects_invalid_targets(case: str) -> None:
    policy = model()
    hidden, observed, slots = inputs(policy)
    output = replay_windows(policy, hidden, observed, slots, grad=False)
    targets = targets_for(output)
    if case == "sigma":
        targets = dataclasses.replace(targets, advantage_sigma=torch.tensor(0.0))
    elif case == "reference":
        targets.reference_distributions[0][0, 0, 0] = 0
    elif case == "empty":
        targets = dataclasses.replace(
            targets, selected_WK=torch.zeros_like(targets.selected_WK), denominator=torch.tensor(0.0)
        )
    elif case == "invalid_slot":
        targets.selected_WK[0, 2] = True
    elif case == "behavior":
        targets.behavior_log_probs_WKF[0, 0, 0] = torch.nan
    elif case == "coefficient":
        targets.kl_coefficient_WK[0, 0] = -0.1
    else:
        targets = dataclasses.replace(targets, advantages_WK=targets.advantages_WK.double())
    with pytest.raises(ValueError):
        validate_actor_targets(output, targets)


def test_empty_local_rank_has_zero_loss_and_gradients_with_positive_global_count() -> None:
    policy = model()
    hidden, observed, slots = inputs(policy)
    output = replay_windows(policy, hidden, observed, slots, grad=True)
    targets = targets_for(output)
    targets = dataclasses.replace(
        targets, selected_WK=torch.zeros_like(targets.selected_WK), denominator=torch.tensor(1.0)
    )
    targets.advantages_WK.fill_(torch.nan)
    targets.behavior_log_probs_WKF.fill_(torch.nan)
    targets.kl_coefficient_WK.fill_(torch.nan)
    for q in targets.reference_distributions:
        q.fill_(torch.nan)
    validate_actor_targets(output, targets)
    loss = actor_loss(output, targets)
    assert float(loss.loss.detach()) == 0
    loss.loss.backward()
    assert all(p.grad is None or not bool(p.grad.any()) for p in policy.parameters())


def test_fullgraph_capture_has_no_breaks() -> None:
    policy = model()
    hidden, observed, slots = inputs(policy)
    compiled = torch.compile(replay_windows, backend="eager", fullgraph=True)
    expected = replay_windows(policy, hidden, observed, slots, grad=True)
    actual = compiled(policy, hidden, observed, slots, grad=True)
    torch.testing.assert_close(actual.log_probs_WKF, expected.log_probs_WKF)
    targets = targets_for(expected)
    compiled_loss = torch.compile(actor_loss, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled_loss(actual, targets).loss, actor_loss(expected, targets).loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_graphed_actor_matches_eager_and_reads_updated_inputs_and_weights(precision: DecoderPrecision) -> None:
    torch.manual_seed(8)
    base = model()
    policy = (
        SelfPlayTransformer(
            dataclasses.replace(base.cfg, temporal_heads=2),
            base.ranges,
            max_opponent_queue=14,
            critic_hidden_dim=16,
            decoder_precision=precision,
        )
        .cuda()
        .eval()
    )
    hidden, observed, slots = inputs(policy, "cuda")
    compiled_replay = torch.compile(
        replay_windows,
        fullgraph=True,
        dynamic=False,
        options={"triton.cudagraphs": False, "emulate_precision_casts": True},
    )

    # Include a learned operation before replay, so repeated backward must build
    # the graph inside capture instead of reusing an external autograd history.
    def forward(values: tuple[Tensor, ...]) -> ReplayOutput:
        return compiled_replay(
            policy,
            values[0] + policy.player_projection(policy.player_embedding(values[1]))[:, None],
            observed,
            slots,
            grad=True,
        )

    values = (hidden, torch.zeros(2, dtype=torch.long, device="cuda"))
    output = forward(values)
    targets = targets_for(output)
    del output
    parameters = policy.actor_parameters()
    with torch.inference_mode(), pytest.raises(ValueError, match="inference_mode"):
        GraphedActorStep(forward, values, targets, parameters)
    graph = GraphedActorStep(forward, values, targets, parameters, clip_epsilon=0.07)
    saved: tuple[Tensor, Tensor] | None = None
    saved_evaluation: tuple[Tensor, Tensor] | None = None
    for step in range(3):
        current = (values[0] + step * 0.05, values[1])
        policy.zero_grad(set_to_none=False)
        expected = actor_loss(forward(current), targets, clip_epsilon=0.07)
        expected.loss.backward()
        gradients = tuple(p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in parameters)
        actual = graph.backward(current, targets)
        evaluated = graph.evaluation()
        torch.testing.assert_close(
            evaluated.log_probs_WKF, forward(current).log_probs_WKF.detach(), atol=2e-6, rtol=2e-5
        )
        assert not evaluated.log_probs_WKF.requires_grad
        torch.testing.assert_close(actual.loss, expected.loss, atol=2e-5, rtol=2e-4)
        for parameter, gradient in zip(parameters, gradients, strict=True):
            torch.testing.assert_close(parameter.grad, gradient, atol=3e-5, rtol=5e-4)
        if saved is not None:
            torch.testing.assert_close(saved[0], saved[1], rtol=0, atol=0)
        if saved_evaluation is not None:
            torch.testing.assert_close(saved_evaluation[0], saved_evaluation[1], rtol=0, atol=0)
        saved = (actual.loss, actual.loss.clone())
        saved_evaluation = (evaluated.log_probs_WKF, evaluated.log_probs_WKF.clone())
        with torch.no_grad():
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-0.001)
    policy.zero_grad(set_to_none=True)
    with pytest.raises(ValueError, match="storage changed"):
        graph.backward(values, targets)


@pytest.mark.parametrize("clip_epsilon", [0.0, 1.0, float("nan"), float("inf")])
def test_actor_rejects_invalid_clip_before_capture(clip_epsilon: float) -> None:
    policy = model()
    hidden, observed, slots = inputs(policy)
    output = replay_windows(policy, hidden, observed, slots, grad=False)
    targets = targets_for(output)
    with pytest.raises(ValueError, match="clip epsilon"):
        actor_loss(output, targets, clip_epsilon=clip_epsilon)
    with pytest.raises(ValueError, match="clip epsilon"):
        GraphedActorStep(lambda _: output, (), targets, (), clip_epsilon=clip_epsilon)


def test_actor_does_not_drop_new_mass_outside_reference_support() -> None:
    policy = model()
    hidden, observed, slots = inputs(policy)
    output = replay_windows(policy, hidden, observed, slots, grad=False)
    target = targets_for(output)
    probabilities = output.distributions[0]
    positive = torch.nonzero((probabilities > 0) & output.factor_mask_WKF[..., 0::4, None])[0]
    target.reference_distributions[0][tuple(positive)] = 0
    assert bool(torch.isinf(actor_loss(output, target).kl_loss))
