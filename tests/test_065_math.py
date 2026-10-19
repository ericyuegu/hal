import dataclasses
import importlib.util
import itertools
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

_SPEC = importlib.util.spec_from_file_location(
    "hal_experiment_065_math", Path(__file__).parents[1] / "experiments" / "065_selfplay_ppo.py"
)
assert _SPEC is not None and _SPEC.loader is not None
exp = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = exp
_SPEC.loader.exec_module(exp)
PROXY_OFFSETS = (*range(1, 13), 16, 20, 24, 28)
PRODUCTION_OFFSETS = (*range(1, 25), 27, 30)
DEVICES = (
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
)


def _decisions(
    count: int,
    *,
    device: str = "cpu",
    delay: int = 0,
    replan: int = 2,
    remaining: Tensor | None = None,
) -> Any:
    d = torch.full((count,), delay, dtype=torch.int64, device=device)
    r = torch.full_like(d, replan)
    if remaining is None:
        remaining = d + r
    zeros = torch.zeros_like(d)
    return exp.DecisionBatch(
        observation_frame_J=zeros,
        decision_frame_J=zeros,
        sampled_indices_JF=torch.zeros(count, 16, dtype=torch.int64, device=device),
        committed_prefix_JD4=torch.zeros(count, 24, 4, dtype=torch.int64, device=device),
        behavior_log_probs_JF=torch.zeros(count, 16, dtype=torch.float64, device=device),
        executed_factors_JF=exp.executed_factor_mask(d, r, remaining),
        duration_J=r,
        bootstrap_mask_J=torch.ones(count, dtype=torch.bool, device=device),
        trace_mask_J=torch.zeros(count, dtype=torch.bool, device=device),
        kl_coefficient_J=torch.ones(count, dtype=torch.float64, device=device),
        delay_J=d,
        replan_J=r,
        policy_version_J=zeros,
        episode_id_J=zeros,
        segment_id_J=zeros,
    )


def _statistics(device: str = "cpu", *, mean: float = 0, sigma: float = 1, count: int = 1) -> Any:
    return exp.AdvantageStatistics(
        torch.tensor(mean, dtype=torch.float64, device=device),
        torch.tensor(sigma, dtype=torch.float64, device=device),
        torch.tensor(float(count), dtype=torch.float64, device=device),
    )


def test_development_and_production_configuration_contracts() -> None:
    dev = exp.development_config(PROXY_OFFSETS)
    assert dev.configurations == (exp.DelayReplan(8, 4), exp.DelayReplan(10, 2))
    assert dev.supervised_offsets == PROXY_OFFSETS
    assert dev.rollout_frames == 256
    production = exp.production_config(PRODUCTION_OFFSETS)
    assert len(production.configurations) == 11
    assert {(c.delay, c.replan) for c in production.configurations} == {
        (16, 2),
        (16, 4),
        (17, 2),
        (17, 4),
        (18, 2),
        (18, 4),
        (19, 2),
        (19, 4),
        (20, 2),
        (20, 4),
        (21, 2),
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        dev.rollout_frames = 128
    assert not hasattr(dev, "__dict__")


@pytest.mark.parametrize("delay,replan", [(-1, 2), (8, 3), (23, 2), (21, 4), (True, 2)])
def test_delay_replan_rejects_invalid_pairs(delay: int, replan: int) -> None:
    with pytest.raises(ValueError):
        exp.DelayReplan(delay, replan)


@pytest.mark.parametrize("frames", [0, -4, 255, 258, True])
def test_rollout_length_must_divide_every_cadence(frames: int) -> None:
    with pytest.raises(ValueError, match="rollout_frames"):
        exp.development_config(PROXY_OFFSETS, rollout_frames=frames)


def test_config_checks_all_executed_offsets_including_first_and_last() -> None:
    for missing in (9, 10, 11, 12):
        with pytest.raises(ValueError, match="never supervised"):
            exp.development_config(tuple(o for o in PROXY_OFFSETS if o != missing))
    with pytest.raises(ValueError, match="never supervised"):
        exp.production_config(PROXY_OFFSETS)
    # A supervised endpoint does not imply supervision of the interior offsets.
    with pytest.raises(ValueError, match="17, 18, 19"):
        exp.ScheduleConfig((exp.DelayReplan(16, 4),), PROXY_OFFSETS)
    for offsets in ((), (2, 1), (1, 1), (0, 1)):
        with pytest.raises(ValueError, match="supervised offsets"):
            exp.development_config(offsets)
    for pairs in ((), (exp.DelayReplan(8, 4),) * 2):
        with pytest.raises(ValueError, match="configurations"):
            exp.ScheduleConfig(pairs, PROXY_OFFSETS)


@pytest.mark.parametrize("device", DEVICES)
def test_executed_offsets_terminal_frame_and_loss_mask(device: str) -> None:
    remaining = torch.tensor([8, 9, 11, 12, 100], device=device)
    decisions = _decisions(5, device=device, delay=8, replan=4, remaining=remaining)
    assert decisions.executed_factors_JF.sum(-1).tolist() == [0, 4, 12, 16, 16]
    assert decisions.loss_mask_J.tolist() == [False, True, True, True, True]
    short = _decisions(1, device=device, delay=10, replan=2)
    assert short.executed_factors_JF.tolist() == [[True] * 8 + [False] * 8]


def test_record_rejects_partial_factors_holes_and_invalid_transitions() -> None:
    decisions = _decisions(1)
    partial = decisions.executed_factors_JF.clone()
    partial[0, 1] = False
    hole = torch.zeros_like(partial)
    hole[0, 4:8] = True
    for mask in (partial, hole, torch.ones_like(partial)):
        with pytest.raises(ValueError, match="complete sampled frames"):
            dataclasses.replace(decisions, executed_factors_JF=mask)
    with pytest.raises(ValueError, match="duration"):
        dataclasses.replace(decisions, duration_J=torch.tensor([3]))
    with pytest.raises(ValueError, match="terminal"):
        dataclasses.replace(decisions, bootstrap_mask_J=torch.tensor([False]), trace_mask_J=torch.tensor([True]))
    with pytest.raises(ValueError, match="positive"):
        dataclasses.replace(decisions, kl_coefficient_J=torch.tensor([0.0]))
    with pytest.raises(ValueError, match="observation frame"):
        dataclasses.replace(decisions, observation_frame_J=torch.tensor([1]))
    with pytest.raises(ValueError, match="dtype"):
        dataclasses.replace(decisions, replan_J=torch.tensor([2.0]))


@pytest.mark.parametrize("device", DEVICES)
def test_factor_order_exact_kl_and_shared_button_mask(device: str) -> None:
    torch.manual_seed(9)
    policy_groups = tuple(
        torch.randn(2, 4, k, dtype=torch.float64, device=device, requires_grad=True) for k in (2, 3, 4, 5)
    )
    ref_groups = tuple(torch.randn_like(p, requires_grad=True) for p in policy_groups)
    button_mask = torch.zeros_like(policy_groups[3], dtype=torch.bool)
    button_mask[:, :, 1] = True
    indices = torch.zeros(2, 4, 4, dtype=torch.int64, device=device)
    indices[:, :, 1] = 2
    indices[:, :, 2] = 3
    indices[:, :, 3] = 4
    evaluation = exp.evaluate_factors(
        exp.ControllerLogits(*policy_groups), exp.ControllerLogits(*ref_groups), indices.flatten(1), button_mask
    )
    expected_log_probs: list[Tensor] = []
    expected_kl: list[Tensor] = []
    for group, (policy, ref) in enumerate(zip(policy_groups, ref_groups, strict=True)):
        legal = [0, 2, 3, 4] if group == 3 else list(range(policy.shape[-1]))
        p = policy[..., legal].softmax(-1)
        q = ref[..., legal].softmax(-1)
        expected_kl.append((p * (p.log() - q.log())).sum(-1))
        index = legal.index(int(indices[0, 0, group]))
        expected_log_probs.append(p[..., index].log())
    torch.testing.assert_close(evaluation.exact_kl_JF, torch.stack(expected_kl, -1).flatten(1))
    torch.testing.assert_close(evaluation.log_probs_JF, torch.stack(expected_log_probs, -1).flatten(1))
    evaluation.exact_kl_JF.sum().backward()
    for logits in policy_groups:
        assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert policy_groups[3].grad is not None
    assert torch.count_nonzero(policy_groups[3].grad[:, :, 1]) == 0
    assert all(logits.grad is None for logits in ref_groups)


def test_categorical_validation_and_masked_gradient() -> None:
    logits = torch.tensor([[0.3, math.nan, -0.2]], requires_grad=True)
    mask = torch.tensor([[False, True, False]])
    log_p = exp.categorical_log_probs(logits, mask)
    ref = exp.categorical_log_probs(torch.zeros_like(logits), mask)
    exp.categorical_kl(log_p, ref).sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1] == 0
    with pytest.raises(ValueError, match="identical support"):
        exp.categorical_kl(log_p, torch.zeros_like(logits).log_softmax(-1))
    with pytest.raises(ValueError, match="normalized"):
        exp.categorical_kl(log_p + 0.2, ref)
    with pytest.raises(ValueError, match="legal category"):
        exp.categorical_log_probs(logits, torch.ones_like(mask))
    with pytest.raises(ValueError, match="finite"):
        exp.categorical_log_probs(logits)
    with pytest.raises(ValueError, match="normalized"):
        exp.categorical_kl(torch.full_like(log_p, -torch.inf), torch.full_like(ref, -torch.inf))


def test_reward_discounts_actual_interval_and_scales_sum_kl_by_ego_lambda() -> None:
    decisions = dataclasses.replace(
        _decisions(3, replan=4, delay=8, remaining=torch.tensor([12, 10, 8])),
        duration_J=torch.tensor([4, 2, 1]),
        kl_coefficient_J=torch.tensor([0.1, 2.0, 7.0]),
    )
    rewards = torch.tensor(
        [[1, 2, 3, 4], [5, 6, math.nan, math.nan], [7, math.nan, math.nan, math.nan]], requires_grad=True
    )
    kl = torch.full((3, 16), 0.25, requires_grad=True)
    result = exp.regularized_reward(decisions, rewards, kl, gamma=0.9)
    expected = torch.tensor([1 + 0.9 * 2 + 0.9**2 * 3 + 0.9**3 * 4 - 0.1 * 4, 5 + 0.9 * 6 - 2 * 2, 7])
    torch.testing.assert_close(result.float(), expected)
    assert not result.requires_grad
    reference = torch.full((3, 16), -0.2, requires_grad=True)
    k1 = exp.sampled_kl_penalty(decisions, reference)
    torch.testing.assert_close(k1, decisions.kl_coefficient_J.double() * torch.tensor([3.2, 1.6, 0.0]))
    assert not k1.requires_grad


@pytest.mark.parametrize("estimator", ["exact", "k1"])
@pytest.mark.parametrize("device", DEVICES)
def test_regularized_gradient_matches_enumerated_categorical_chain(estimator: str, device: str) -> None:
    theta = torch.tensor(
        [[0.5, -0.4], [0.1, 0.6], [-0.7, 0.2]], dtype=torch.float64, device=device, requires_grad=True
    )
    ref_logits = torch.tensor([[-0.2, 0.4], [0.8, -0.1], [0.3, -0.2]], dtype=torch.float64, device=device)
    log_p = theta.log_softmax(-1)
    log_q = ref_logits.log_softmax(-1)
    coefficient, sigma = 0.7, 2.3
    objective = theta.sum() * 0
    estimator_loss = theta.sum() * 0
    score_only_loss = theta.sum() * 0
    for a, b in itertools.product(range(2), repeat=2):
        sampled = torch.stack((log_p[0, a], log_p[1 + a, b], *([theta.new_zeros(())] * 14)))[None]
        reference = torch.stack((log_q[0, a], log_q[1 + a, b], *([theta.new_zeros(())] * 14)))[None]
        path_kl = torch.stack(
            (
                exp.categorical_kl(log_p[0], log_q[0]),
                exp.categorical_kl(log_p[1 + a], log_q[1 + a]),
                *([theta.new_zeros(())] * 14),
            )
        )[None]
        probability = (log_p[0, a] + log_p[1 + a, b]).exp()
        reward = float(2 * a - b + 3 * a * b)
        objective = objective + probability * (reward - coefficient * (sampled - reference).sum())
        decisions = dataclasses.replace(
            _decisions(1, device=device),
            behavior_log_probs_JF=sampled.detach(),
            kl_coefficient_J=theta.new_tensor([coefficient]),
        )
        frames = theta.new_tensor([[reward, 0, 0, 0]])
        if estimator == "exact":
            advantage = exp.regularized_reward(decisions, frames, path_kl, gamma=1)
        else:
            advantage = frames[:, 0] - exp.sampled_kl_penalty(decisions, reference)
        stats = _statistics(device, mean=0.4, sigma=sigma)
        selected = decisions.loss_mask_J
        policy = exp.ppo_loss(decisions, sampled, advantage, selected, stats, clip_epsilon=0.2).loss
        score_only_loss = score_only_loss + probability.detach() * policy
        loss = policy
        if estimator == "exact":
            loss = loss + exp.differentiable_kl_loss(decisions, path_kl, selected, stats)
        estimator_loss = estimator_loss + probability.detach() * loss
    expected = torch.autograd.grad(-objective / sigma, theta, retain_graph=True)[0]
    actual = torch.autograd.grad(estimator_loss, theta, retain_graph=True)[0]
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    if estimator == "exact":
        missing_direct = torch.autograd.grad(score_only_loss, theta)[0]
        assert not torch.allclose(missing_direct, expected, atol=1e-3, rtol=1e-3)


def test_gae_terminal_timeout_and_rollout_end_are_distinct() -> None:
    # Columns are parallel streams: terminal/reset, timeout/reset, ongoing.
    rewards = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float64, requires_grad=True)
    values = torch.tensor([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]], dtype=torch.float64, requires_grad=True)
    next_values = torch.tensor([[math.nan, 10.0, 4.0], [5.0, 5.0, 5.0]], dtype=torch.float64, requires_grad=True)
    durations = torch.tensor([[1, 1, 4], [2, 2, 2]])
    bootstrap = torch.tensor([[False, True, True], [True, True, True]])
    trace = torch.tensor([[False, False, True], [False, False, False]])
    result = exp.generalized_advantage(
        rewards, values, next_values, durations, bootstrap, trace, gamma=0.9, gae_lambda=0.8
    )
    tail = 2 + 0.9**2 * 5 - 4
    expected = values.new_tensor([[-2, 1 + 0.9 * 10 - 3, 1 + 0.9**4 * 4 - 3 + (0.9 * 0.8) ** 4 * tail], [tail] * 3])
    torch.testing.assert_close(result.advantages, expected)
    torch.testing.assert_close(result.value_targets, expected + values)
    assert not result.advantages.requires_grad and not result.value_targets.requires_grad


@pytest.mark.parametrize("device", DEVICES)
def test_delayed_actions_receive_unshifted_credit_through_untrained_tail(device: str) -> None:
    # A decision at frame 0 enters a four-frame queue and executes at frame 5.
    # Decision 2 receives the reward; its sampled actions fall after the match end.
    decisions = _decisions(4, device=device, delay=4, remaining=torch.tensor([8, 6, 4, 2], device=device))
    assert decisions.loss_mask_J.tolist() == [True, True, False, False]
    rewards = torch.tensor(
        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.float64, device=device
    )
    interval = exp.regularized_reward(decisions, rewards, torch.zeros(4, 16, device=device), gamma=0.9)
    values = torch.tensor([0.0, 0.3, 0.7, 1.0], dtype=torch.float64, device=device)
    bootstrap = torch.tensor([True, True, True, False], device=device)
    trace = torch.tensor([True, True, True, False], device=device)
    result = exp.generalized_advantage(
        interval,
        values,
        torch.cat((values[1:], values.new_tensor([math.nan]))),
        decisions.duration_J,
        bootstrap,
        trace,
        gamma=0.9,
        gae_lambda=1,
    )
    torch.testing.assert_close(result.value_targets, values.new_tensor([0.9**4, 0.9**2, 1, 0]))
    theta = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device, requires_grad=True)
    log_p = theta.log_softmax(-1)
    expected_loss = theta.sum() * 0
    for action in (0, 1):
        sampled = torch.stack((log_p[action], *([theta.new_zeros(())] * 15)))[None].expand(4, -1)
        current = dataclasses.replace(decisions, behavior_log_probs_JF=sampled.detach())
        selected = torch.tensor([True, False, False, False], device=device)
        expected_loss = (
            expected_loss
            + log_p[action].exp().detach()
            * exp.ppo_loss(
                current,
                sampled,
                action * result.advantages,
                selected,
                _statistics(device),
                clip_epsilon=0.2,
            ).loss
        )
    gradient = torch.autograd.grad(expected_loss, theta)[0]
    torch.testing.assert_close(gradient, theta.new_tensor([0.9**4 / 4, -(0.9**4) / 4]))


def test_joint_ppo_clipping_is_per_decision_and_all_targets_are_detached() -> None:
    decisions = _decisions(4)
    joint = torch.tensor([1.5, 0.5, 1.1, 0.9], dtype=torch.float64)
    old_joint = torch.tensor([2.0, 0.25, 1.0, 1.0], dtype=torch.float64)
    old = (old_joint.log()[:, None] / 8).expand(4, 16).clone().requires_grad_()
    new = (old.detach() + joint.log()[:, None] / 8).requires_grad_()
    advantages = torch.tensor([2.0, -2.0, 2.0, -2.0], dtype=torch.float64, requires_grad=True)
    behavior = old.detach().clone().requires_grad_()
    decisions = dataclasses.replace(decisions, behavior_log_probs_JF=behavior)
    stats = exp.advantage_statistics(exp.advantage_moments(advantages, decisions.loss_mask_J))
    result = exp.ppo_loss(decisions, new, advantages, decisions.loss_mask_J, stats, clip_epsilon=0.2)
    torch.testing.assert_close(result.joint_ratio_J, joint)
    expected = -(1.2 - 0.8 + 1.1 - 0.9) / 4
    assert result.loss.item() == pytest.approx(expected)
    result.loss.backward()
    assert old.grad is None and advantages.grad is None and behavior.grad is None
    assert new.grad is not None
    torch.testing.assert_close(new.grad[:2], torch.zeros_like(new.grad[:2]))
    assert torch.count_nonzero(new.grad[:, 8:]) == 0
    assert torch.count_nonzero(new.grad[2:, :8]) == 16


def test_joint_log_ratio_subtracts_before_reducing_factors() -> None:
    decisions = _decisions(1)
    old = torch.tensor([[-1e8, -1.0] * 8])
    new = old.clone()
    new[:, 1::2] += 0.01
    expected = (new[0, :8] - old[0, :8]).double().sum().exp()
    assert (new[0, :8].sum() - old[0, :8].sum()).exp() == 1
    decisions = dataclasses.replace(decisions, behavior_log_probs_JF=old)
    result = exp.ppo_loss(
        decisions,
        new,
        torch.ones(1),
        decisions.loss_mask_J,
        _statistics(),
        clip_epsilon=0.2,
    )
    torch.testing.assert_close(result.joint_ratio_J.double(), expected[None])
    changed_behavior = dataclasses.replace(decisions, behavior_log_probs_JF=old - (new - old))
    torch.testing.assert_close(
        exp.sampled_kl_penalty(changed_behavior, old), -(new[0, :8] - old[0, :8]).double().sum()[None]
    )


def test_ratio_overflow_is_rejected_even_when_clipping_would_hide_it() -> None:
    decisions = _decisions(1)
    old = torch.full((1, 16), -1000.0)
    new = torch.zeros_like(old, requires_grad=True)
    selected = decisions.loss_mask_J
    decisions = dataclasses.replace(decisions, behavior_log_probs_JF=old)
    with pytest.raises(ValueError, match="joint ratio must be finite"):
        exp.ppo_loss(decisions, new, torch.ones(1), selected, _statistics(), clip_epsilon=0.2)


def test_policy_hyperparameters_are_validated_and_statistics_are_detached() -> None:
    decisions = _decisions(1)
    new = torch.zeros(1, 16, requires_grad=True)
    mean = torch.tensor(0.0, requires_grad=True)
    sigma = torch.tensor(2.0, requires_grad=True)
    stats = exp.AdvantageStatistics(mean, sigma, torch.tensor(1.0))
    for epsilon in (0.0, 1.0, math.nan):
        with pytest.raises(ValueError):
            exp.ppo_loss(
                decisions,
                new,
                torch.ones(1),
                decisions.loss_mask_J,
                stats,
                clip_epsilon=epsilon,
            )
    loss = exp.ppo_loss(decisions, new, torch.ones(1), decisions.loss_mask_J, stats, clip_epsilon=0.2).loss
    loss = loss + exp.differentiable_kl_loss(decisions, new.square(), decisions.loss_mask_J, stats)
    loss.backward()
    assert mean.grad is None and sigma.grad is None


def test_partial_execution_masks_padding_and_rejects_training_unexecuted_actions() -> None:
    decisions = _decisions(3, delay=8, replan=4, remaining=torch.tensor([9, 8, 12]))
    values = torch.full((3, 16), math.nan, dtype=torch.float64)
    values[0, :4] = math.log(1.1) / 4
    values[2] = 1000  # Unselected rows are not evaluated, even if they overflow exp.
    values.requires_grad_()
    selected = torch.tensor([True, False, False])
    result = exp.ppo_loss(
        decisions,
        values,
        torch.ones(3),
        selected,
        _statistics(),
        clip_epsilon=0.2,
    )
    assert result.joint_ratio_J[0].item() == pytest.approx(1.1)
    result.loss.backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert torch.count_nonzero(values.grad) == 4
    with pytest.raises(ValueError, match="no executed factors"):
        exp.differentiable_kl_loss(decisions, values, torch.tensor([False, True, False]), _statistics())


def test_global_advantage_and_loss_hooks_match_a_single_rank_with_unequal_counts() -> None:
    advantage = torch.tensor([-3.0, 0.0, 2.0, 9.0], dtype=torch.float64)
    selected = torch.ones(4, dtype=torch.bool)
    moments = exp.advantage_moments(advantage, selected)
    pieces = [slice(0, 1), slice(1, 4), slice(4, 4)]
    reduced = sum(
        (exp.advantage_moments(advantage[s], selected[s]) for s in pieces), torch.zeros(3, dtype=torch.float64)
    )
    torch.testing.assert_close(reduced, moments)
    stats = exp.advantage_statistics(reduced)
    torch.testing.assert_close(stats.mean, advantage.mean())
    torch.testing.assert_close(stats.sigma, advantage.std(correction=0))
    theta = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    x = torch.tensor([1.0, 2.0, -1.0, 0.5], dtype=torch.float64)

    def loss(part: slice, reduction: Any = None) -> Tensor:
        count = len(advantage[part])
        decisions = _decisions(count)
        logs = (theta * x[part, None]).expand(count, 16)
        actor = exp.ppo_loss(
            decisions,
            logs,
            advantage[part],
            selected[part],
            stats,
            clip_epsilon=0.2,
            reduction=reduction,
        ).loss
        kl = exp.differentiable_kl_loss(decisions, logs.square(), selected[part], stats, reduction=reduction)
        critic = exp.critic_loss(theta * x[part], advantage[part], selected[part], reduction=reduction)
        return actor + kl + critic

    expected = torch.autograd.grad(loss(slice(None)), theta)[0]
    reduction = exp.DecisionReduction(stats.count, gradient_average_ranks=3)
    rank_gradients = [torch.autograd.grad(loss(part, reduction), theta)[0] for part in pieces]
    torch.testing.assert_close(torch.stack(rank_gradients).mean(), expected)
    assert rank_gradients[-1].item() == 0


def test_differentiable_kl_uses_same_sigma_ego_lambda_and_decision_count() -> None:
    decisions = dataclasses.replace(
        _decisions(2, replan=4), kl_coefficient_J=torch.tensor([0.5, 2.0], requires_grad=True)
    )
    kl = torch.full((2, 16), 0.2, requires_grad=True)
    stats = _statistics(sigma=4, count=2)
    loss = exp.differentiable_kl_loss(decisions, kl, decisions.loss_mask_J, stats)
    assert loss.item() == pytest.approx((0.5 + 2) / 4 * 16 * 0.2 / 2)
    loss.backward()
    assert kl.grad is not None
    torch.testing.assert_close(kl.grad, torch.tensor([[0.5 / 4 / 2] * 16, [2 / 4 / 2] * 16]))
    assert decisions.kl_coefficient_J.grad is None


def test_critic_matches_o64_scale_and_includes_decisions_without_executed_factors() -> None:
    decisions = _decisions(3, delay=8, remaining=torch.tensor([10, 8, 8]))
    assert decisions.loss_mask_J.tolist() == [True, False, False]
    values = torch.tensor([150.0, -300.0, math.nan], requires_grad=True)
    targets = torch.tensor([0.0, 0.0, math.nan], requires_grad=True)
    selected = torch.tensor([True, True, False])
    loss = exp.critic_loss(values, targets, selected)
    assert loss.item() == pytest.approx(2.5)
    loss.backward()
    torch.testing.assert_close(values.grad, torch.tensor([1 / 150, -2 / 150, 0]))
    assert targets.grad is None


def test_loss_reduction_preserves_float64_precision_for_non_power_of_two_counts() -> None:
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    loss = exp.critic_loss(values, torch.zeros_like(values), torch.ones(3, dtype=torch.bool), value_scale=1)
    torch.testing.assert_close(loss, values.square().mean(), atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("mutate", ["last_trace", "terminal_trace", "duration", "bootstrap_nan"])
def test_gae_rejects_invalid_transitions(mutate: str) -> None:
    reward = torch.ones(2)
    value = torch.zeros(2)
    next_value = torch.zeros(2)
    duration = torch.full((2,), 2, dtype=torch.int64)
    bootstrap = torch.ones(2, dtype=torch.bool)
    trace = torch.tensor([True, False])
    if mutate == "last_trace":
        trace[-1] = True
    elif mutate == "terminal_trace":
        bootstrap[0] = False
    elif mutate == "duration":
        duration[0] = 0
    else:
        next_value[-1] = math.nan
    with pytest.raises(ValueError):
        exp.generalized_advantage(reward, value, next_value, duration, bootstrap, trace)


def test_constant_advantages_empty_local_rank_and_invalid_normalization() -> None:
    stats = exp.advantage_statistics(exp.advantage_moments(torch.ones(3), torch.ones(3, dtype=torch.bool)))
    assert stats.sigma.item() == pytest.approx(1e-8)
    invalid: tuple[Callable[[], object], ...] = (
        lambda: exp.advantage_statistics(torch.zeros(3)),
        lambda: exp.advantage_statistics(torch.tensor([1.5, 2, 3])),
        lambda: exp.advantage_statistics(torch.tensor([1.0, 2, math.nan])),
        lambda: exp.advantage_statistics(torch.tensor([1.0, 10, 1])),
        lambda: _statistics(sigma=-1),
        lambda: _statistics(sigma=math.nan),
        lambda: exp.DecisionReduction(torch.tensor(0)),
        lambda: exp.DecisionReduction(torch.tensor(True)),
        lambda: exp.DecisionReduction(torch.tensor(1), gradient_average_ranks=0),
        lambda: exp.advantage_moments(torch.ones(3, dtype=torch.int64), torch.ones(3, dtype=torch.bool)),
        lambda: exp.critic_loss(torch.ones(1, dtype=torch.int64), torch.zeros(1), torch.tensor([True])),
        lambda: exp.critic_loss(torch.zeros(1), torch.zeros(1), torch.tensor([False])),
        lambda: exp.critic_loss(torch.zeros(1), torch.zeros(1), torch.tensor([True]), value_scale=0),
        lambda: exp.regularized_reward(_decisions(1), torch.zeros(1, 4), torch.zeros(1, 16), gamma=math.nan),
    )
    for call in invalid:
        with pytest.raises(ValueError):
            call()


def test_actor_rejects_advantage_statistics_for_a_different_selection() -> None:
    decisions = _decisions(2)
    for reduction in (None, exp.DecisionReduction(torch.tensor(3))):
        with pytest.raises(ValueError, match="same global decision count"):
            exp.ppo_loss(
                decisions,
                torch.zeros(2, 16),
                torch.ones(2),
                decisions.loss_mask_J,
                _statistics(),
                clip_epsilon=0.2,
                reduction=reduction,
            )
        with pytest.raises(ValueError, match="same global decision count"):
            exp.differentiable_kl_loss(
                decisions, torch.zeros(2, 16), decisions.loss_mask_J, _statistics(), reduction=reduction
            )


def test_mixed_cadence_records_and_invalid_controller_rows() -> None:
    decisions = _decisions(2, delay=8, replan=4)
    delay = torch.tensor([8, 10])
    replan = torch.tensor([4, 2])
    mixed = dataclasses.replace(
        decisions,
        delay_J=delay,
        replan_J=replan,
        duration_J=replan,
        executed_factors_JF=exp.executed_factor_mask(delay, replan, torch.tensor([12, 12])),
    )
    joint = exp.ppo_loss(
        mixed,
        torch.full((2, 16), 0.01),
        torch.ones(2),
        mixed.loss_mask_J,
        _statistics(count=2),
        clip_epsilon=0.2,
    ).joint_ratio_J
    torch.testing.assert_close(joint, torch.tensor([0.16, 0.08], dtype=joint.dtype).exp())
    with pytest.raises(ValueError, match="invalid delay"):
        exp.executed_factor_mask(torch.tensor([torch.iinfo(torch.int64).max]), torch.tensor([4]), torch.tensor([20]))
    with pytest.raises(ValueError, match="non-negative"):
        exp.executed_factor_mask(torch.tensor([8]), torch.tensor([4]), torch.tensor([-1]))
    with pytest.raises(ValueError, match="shortened interval"):
        dataclasses.replace(_decisions(1), duration_J=torch.tensor([1]), trace_mask_J=torch.tensor([True]))
    with pytest.raises(ValueError, match="sampled indices"):
        dataclasses.replace(_decisions(1), sampled_indices_JF=torch.full((1, 16), -1, dtype=torch.int64))
    with pytest.raises(ValueError, match="committed indices"):
        dataclasses.replace(_decisions(1, delay=8), committed_prefix_JD4=torch.full((1, 24, 4), -1, dtype=torch.int64))

    logits = torch.zeros(1, 4, 2)
    controller = exp.ControllerLogits(logits, logits, logits, logits)
    picks = torch.zeros(1, 16, dtype=torch.int64)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    with pytest.raises(ValueError, match="shape"):
        exp.evaluate_factors(dataclasses.replace(controller, c_stick=torch.tensor(1.0)), controller, picks, mask)
    picks[0, 0] = 2
    with pytest.raises(ValueError, match="outside categorical support"):
        exp.evaluate_factors(controller, controller, picks, mask)
    picks[0, 0] = 0
    mask[:, :, 0] = True
    with pytest.raises(ValueError, match="sampled log-probs must be finite"):
        exp.evaluate_factors(controller, controller, picks, mask)
