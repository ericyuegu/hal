import dataclasses
import importlib.util
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


@pytest.mark.parametrize("delay,replan", [(-1, 2), (8, 3), (23, 2), (21, 4)])
def test_delay_replan_rejects_invalid_pairs(delay: int, replan: int) -> None:
    with pytest.raises(ValueError):
        exp.DelayReplan(delay, replan)


@pytest.mark.parametrize("frames", [0, -4, 255, 258])
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


def test_gae_terminal_timeout_and_rollout_end_are_distinct() -> None:
    # Columns are parallel streams: terminal/reset, timeout/reset, ongoing.
    rewards = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float64, requires_grad=True)
    values = torch.tensor([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]], dtype=torch.float64, requires_grad=True)
    next_values = torch.tensor([[math.nan, 10.0, 4.0], [5.0, 5.0, 5.0]], dtype=torch.float64, requires_grad=True)
    durations = torch.tensor([[1, 1, 4], [2, 2, 2]])
    bootstrap = torch.tensor([[False, True, True], [True, True, True]])
    trace = torch.tensor([[False, False, True], [False, False, False]])
    result = exp.dense_advantages(rewards, values, next_values, durations, bootstrap, trace, torch.ones_like(trace))
    gamma, gae = exp.GAMMA, exp.GAE_LAMBDA
    tail = 2 + gamma**2 * 5 - 4
    expected = values.new_tensor(
        [[-2, 1 + gamma * 10 - 3, 1 + gamma**4 * 4 - 3 + (gamma * gae) ** 4 * tail], [tail] * 3]
    )
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
    interval = exp.regularized_reward(decisions, rewards, torch.zeros(4, 16, device=device), gamma=exp.GAMMA)
    values = torch.tensor([0.0, 0.3, 0.7, 1.0], dtype=torch.float64, device=device)
    bootstrap = torch.tensor([True, True, True, False], device=device)
    trace = torch.tensor([True, True, True, False], device=device)
    result = exp.dense_advantages(
        interval,
        values,
        torch.cat((values[1:], values.new_tensor([math.nan]))),
        decisions.duration_J,
        bootstrap,
        trace,
        torch.ones_like(trace),
    )
    discount = exp.GAMMA**2
    trace_discount = (exp.GAMMA * exp.GAE_LAMBDA) ** 2
    last = 0.3 + discount - trace_discount
    middle = 0.7 * discount - 0.3 + trace_discount * last
    expected = values.new_tensor([0.3 * discount + trace_discount * middle, middle, last, -1])
    torch.testing.assert_close(result.advantages, expected)
    assert result.advantages[0] > 0


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


def test_constant_advantages_empty_local_rank_and_invalid_normalization() -> None:
    stats = exp.advantage_statistics(exp.advantage_moments(torch.ones(3), torch.ones(3, dtype=torch.bool)))
    assert stats.sigma.item() == pytest.approx(1e-8)
    invalid: tuple[Callable[[], object], ...] = (
        lambda: exp.advantage_statistics(torch.zeros(3)),
        lambda: exp.advantage_statistics(torch.tensor([1.5, 2, 3])),
        lambda: exp.advantage_statistics(torch.tensor([1.0, 2, math.nan])),
        lambda: exp.advantage_statistics(torch.tensor([1.0, 10, 1])),
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
