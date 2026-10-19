"""Window packing parity, execution masks and captured actor gradients."""

import dataclasses

import pytest
import torch
from test_self_play_model import model
from torch import Tensor

from hal.inference.decision_decoder import DecisionDecoder
from hal.models.attention import KVMemory
from hal.models.controller_codec import CONTROLLER_DECODE_ORDER
from hal.models.controller_codec import CONTROLLER_GROUP_INDEX
from hal.models.self_play import SelfPlayTransformer
from hal.models.window_replay import ReplaySlots
from hal.models.window_replay import replay_windows
from hal.models.window_replay import validate_replay_inputs
from hal.wire import ACTION_DIM


def inputs(policy: SelfPlayTransformer, device: str = "cpu") -> tuple[Tensor, Tensor, ReplaySlots]:
    torch.manual_seed(79)
    hidden = torch.randn(2, 32, 32, device=device)
    observed = policy.codec.quantize(torch.rand(2, 32, ACTION_DIM, device=device))
    positions = torch.tensor([[8, 15, 31], [7, 23, 30]], device=device)
    delay = torch.tensor([[8, 10, 8], [10, 8, 10]], device=device)
    replan = 12 - delay
    condition = policy.ranges.normalize(torch.zeros(2, 3, device=device), delay, replan)
    starts = torch.zeros(2, 32, dtype=torch.long, device=device)
    starts[0, 16:], starts[1, 24:] = 16, 24
    valid = torch.tensor([[True, True, False], [True, True, True]], device=device)
    executed = (
        (torch.arange(4, device=device) < replan[..., None])[..., None].expand(-1, -1, -1, 4).flatten(-2).clone()
    )
    executed[0, 0, 8:] = False
    slots = ReplaySlots(
        positions,
        valid,
        starts,
        policy.codec.quantize(torch.rand(2, 3, 10, ACTION_DIM, device=device)),
        delay,
        replan,
        condition,
        -condition,
        policy.codec.quantize(torch.rand(2, 3, 14, ACTION_DIM, device=device)),
        delay + replan,
        policy.codec.quantize(torch.rand(2, 3, 4, ACTION_DIM, device=device)),
        executed,
    )
    return hidden, observed, slots


def individual(
    policy: SelfPlayTransformer, hidden: Tensor, observed: Tensor, slots: ReplaySlots
) -> tuple[Tensor, Tensor]:
    decoder = DecisionDecoder(policy, ((8, 4), (10, 2)), compiled=False)
    logs, states = [], []
    order = [CONTROLLER_GROUP_INDEX[name] for name in CONTROLLER_DECODE_ORDER]
    for window in range(2):
        for slot in range(3):
            pos = int(slots.positions_WK[window, slot])
            key, value = policy.temporal.history_attention.project_memory(hidden[window : window + 1])
            memory_positions = torch.arange(32, device=hidden.device)[None]
            start = slots.history_starts_WL[window, pos]
            memory_positions = torch.where(memory_positions >= start, memory_positions, -1)
            result = decoder(
                hidden[window : window + 1, pos : pos + 1],
                observed[window : window + 1, pos],
                slots.forced_queue_WKD4[window : window + 1, slot],
                torch.zeros(1, device=hidden.device),
                slots.delay_WK[window : window + 1, slot],
                slots.replan_WK[window : window + 1, slot],
                torch.full((4, 4, 1), 0.5, device=hidden.device),
                sampled_offsets=4,
                sampled_actions=slots.sampled_actions_WK44[window : window + 1, slot],
                history=KVMemory(
                    torch.stack((key, value)), memory_positions, torch.tensor([pos], device=hidden.device), 256
                ),
            )
            logs.append(result.log_probs[..., order].flatten(-2))
            states.append(result.decoder_state)
    mask = slots.executed_WKF & slots.valid_WK[..., None]
    return torch.where(mask, torch.cat(logs).reshape(2, 3, 16), 0), torch.cat(states).reshape(2, 3, 32)


def test_window_replay_matches_individual_outputs_features_and_gradients() -> None:
    policy = model()
    for projection in policy.temporal.return_conditioner.projections:
        assert isinstance(projection, torch.nn.Linear)
        torch.nn.init.normal_(projection.weight, std=0.02)
    hidden, observed, slots = inputs(policy)
    validate_replay_inputs(policy, hidden, observed, slots)
    hidden.requires_grad_(True)
    expected_logs, expected_state = individual(policy, hidden, observed, slots)
    parameters = (hidden, *policy.actor_parameters())
    expected_grad = torch.autograd.grad(expected_logs.sum(), parameters, allow_unused=True)
    result = replay_windows(policy, hidden, observed, slots, grad=True)
    actual_grad = torch.autograd.grad(result.log_probs_WKF.sum(), parameters, allow_unused=True)
    torch.testing.assert_close(result.log_probs_WKF, expected_logs, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(
        result.features.decoder_WKt[slots.valid_WK], expected_state[slots.valid_WK], atol=2e-6, rtol=2e-5
    )
    for actual, expected in zip(actual_grad, expected_grad, strict=True):
        assert (actual is None) == (expected is None)
        if actual is not None and expected is not None:
            torch.testing.assert_close(actual, expected, atol=4e-6, rtol=4e-4)
    assert all(not getattr(result.features, field.name).requires_grad for field in dataclasses.fields(result.features))
    assert not replay_windows(policy, hidden, observed, slots, grad=False).log_probs_WKF.requires_grad
    with torch.inference_mode(), pytest.raises(ValueError, match="inference_mode"):
        replay_windows(policy, hidden, observed, slots, grad=True)


def test_padding_is_ignored_and_history_does_not_cross_resets() -> None:
    policy = model()
    hidden, observed, slots = inputs(policy)
    expected = replay_windows(policy, hidden, observed, slots, grad=False)
    poisoned = {}
    for field in dataclasses.fields(slots):
        value = getattr(slots, field.name).clone()
        if field.name not in ("valid_WK", "history_starts_WL"):
            value[~slots.valid_WK] = torch.nan if value.is_floating_point() else 999
        poisoned[field.name] = value
    padded = ReplaySlots(**poisoned)
    validate_replay_inputs(policy, hidden, observed, padded)
    actual = replay_windows(policy, hidden, observed, padded, grad=False)
    torch.testing.assert_close(actual.log_probs_WKF, expected.log_probs_WKF, rtol=0, atol=0)
    hidden[1, :24] = torch.randn_like(hidden[1, :24]) * 100
    changed = replay_windows(policy, hidden, observed, slots, grad=False)
    torch.testing.assert_close(changed.log_probs_WKF[1, 2], expected.log_probs_WKF[1, 2], rtol=0, atol=0)


def test_critic_state_precedes_ego_draw_and_replay_rejects_unsupervised_offsets() -> None:
    policy = model()
    hidden, observed, slots = inputs(policy)
    expected = replay_windows(policy, hidden, observed, slots, grad=False)
    changed_actions = policy.codec.quantize(torch.rand(2, 3, 4, ACTION_DIM))
    changed = replay_windows(
        policy, hidden, observed, dataclasses.replace(slots, sampled_actions_WK44=changed_actions), grad=False
    )
    torch.testing.assert_close(changed.features.decoder_WKt, expected.features.decoder_WKt, rtol=0, atol=0)
    unsupported = SelfPlayTransformer(
        dataclasses.replace(policy.cfg, head_offsets=tuple(range(1, 11))),
        policy.ranges,
        max_opponent_queue=14,
        critic_hidden_dim=16,
    )
    with pytest.raises(ValueError, match="supervised"):
        validate_replay_inputs(unsupported, hidden, observed, slots)


def test_unexecuted_valid_decision_retains_critic_features() -> None:
    policy = model()
    hidden, observed, slots = inputs(policy)
    expected = replay_windows(policy, hidden, observed, slots, grad=False)
    executed = slots.executed_WKF.clone()
    executed[0, 0] = False
    output = replay_windows(policy, hidden, observed, dataclasses.replace(slots, executed_WKF=executed), grad=False)
    assert not bool(output.factor_mask_WKF[0, 0].any())
    assert not bool(output.log_probs_WKF[0, 0].any())
    torch.testing.assert_close(output.features.decoder_WKt[0, 0], expected.features.decoder_WKt[0, 0], rtol=0, atol=0)
    features = output.features
    value = policy.self_play_value(
        features.trunk_WKd[0, :1],
        features.decoder_WKt[0, :1],
        features.condition_WK3[0, :1],
        features.opponent_condition_WK3[0, :1],
        features.opponent_queue_WKQ4[0, :1],
        features.opponent_lengths_WK[0, :1],
    )
    value.sum().backward()
    assert any(p.grad is not None and bool(p.grad.any()) for p in policy.critic_parameters())
    assert all(p.grad is None for p in policy.actor_parameters())


@pytest.mark.parametrize(
    "field",
    [
        "positions_WK",
        "delay_WK",
        "replan_WK",
        "executed_WKF",
        "sampled_actions_WK44",
        "condition_WK3",
        "history_starts_WL",
    ],
)
def test_invalid_slots_are_rejected(field: str) -> None:
    policy = model()
    hidden, observed, slots = inputs(policy)
    value = getattr(slots, field).clone()
    if field == "executed_WKF":
        value[0, 0, 1] = False
    elif field == "condition_WK3":
        value[0, 0, 1] = -0.5
    elif field == "history_starts_WL":
        value[0, 2] = 4
    else:
        value[0, 0] = 999
    with pytest.raises(ValueError):
        validate_replay_inputs(policy, hidden, observed, dataclasses.replace(slots, **{field: value}))


def test_invalid_window_actions_are_rejected_even_outside_live_slot_positions() -> None:
    policy = model()
    hidden, observed, slots = inputs(policy)
    observed[0, 0] = 999
    with pytest.raises(ValueError, match="controller factor"):
        validate_replay_inputs(policy, hidden, observed, slots)
