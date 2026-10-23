"""Keep gathered actor windows immutable across captured forward and backward."""

import copy
import dataclasses
import math

import pytest
import torch
from test_065_loop import exp
from test_065_loop import transformer_trainer


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_actor_step_reuses_windows_without_writes_or_gradient_changes(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    run = transformer_trainer(device, decoder_precision="bf16" if device == "cuda" else "fp32")
    batch = run.prepare(run.engine.collect(0))
    original = run.policy.actor_step
    checked = False

    def compare(
        windows: exp.ReplayBatch, targets: exp.ActorTargets, *, clip_epsilon: float
    ) -> tuple[exp.ActorLoss, exp.ReplayOutput]:
        nonlocal checked
        if checked:
            return original(windows, targets, clip_epsilon=clip_epsilon)
        checked = True
        ids = batch.steps[0]
        fresh = batch.windows.take(ids, batch.placement.positions, ids >= 0)
        before = tuple(value.clone() for value in windows.inputs())
        actual, actual_output = original(windows, targets, clip_epsilon=clip_epsilon)
        gradients = tuple(parameter.grad.clone() for parameter in run.policy.actor_parameters())
        expected, expected_output = original(fresh, targets, clip_epsilon=clip_epsilon)
        for field in dataclasses.fields(actual):
            torch.testing.assert_close(getattr(actual, field.name), getattr(expected, field.name), rtol=0, atol=0)
        torch.testing.assert_close(actual_output.log_probs_WKF, expected_output.log_probs_WKF, rtol=0, atol=0)
        for gradient, parameter in zip(gradients, run.policy.actor_parameters(), strict=True):
            torch.testing.assert_close(parameter.grad, gradient, rtol=1e-5, atol=1e-7)
        for value, saved in zip(windows.inputs(), before, strict=True):
            torch.testing.assert_close(value, saved, rtol=0, atol=0)
        return expected, expected_output

    monkeypatch.setattr(run.policy, "actor_step", compare)
    run.fit(batch)
    assert checked


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_grouped_reference_matches_padded_replay_for_every_valid_decision(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    run = transformer_trainer(device, development=True, decoder_precision="bf16" if device == "cuda" else "fp32")
    batch = run.prepare(run.engine.collect(0))
    windows = batch.windows
    reference = run.reference
    reference.reference_forward_model = copy.deepcopy(reference.model).requires_grad_(True)
    positions = torch.arange(64, device=device)[None].expand(len(windows.rows), -1)
    with torch.no_grad():
        expected = reference.reference_forward(windows.inputs(), len(windows.rows), 16)
        # A partial order must also score all other windows and all unsampled decisions.
        actual = reference.evaluate_reference(windows, positions[:, :16], torch.tensor([[1, -1, 0]], device=device))
    mask = expected.factor_mask_WKF
    assert int(mask.sum()) > 0
    torch.testing.assert_close(actual.factor_mask_WKF, mask)
    tolerance = {"rtol": 0.02, "atol": 0.02} if device == "cuda" else {"rtol": 1e-4, "atol": 1e-5}
    torch.testing.assert_close(actual.log_probs_WKF[mask], expected.log_probs_WKF[mask], **tolerance)
    frame_mask = mask.reshape(*mask.shape[:2], 4, 4)
    for group, (p, q) in enumerate(zip(actual.distributions, expected.distributions, strict=True)):
        torch.testing.assert_close(p[frame_mask[..., group]], q[frame_mask[..., group]], **tolerance)
    # Reward KL consumes every live slot, not only the 16 actor samples.
    assert torch.equal(actual.factor_mask_WKF.any(-1), windows.slots.valid_WK & windows.slots.executed_WKF.any(-1))


def test_grouped_reference_covers_all_production_shapes_and_partial_groups() -> None:
    run = transformer_trainer("cpu", decoder_precision="fp32")
    windows = run.prepare(run.engine.collect(0)).windows
    reference = run.reference
    schedule = exp.production_config(tuple(range(1, 29)))
    reference.schedule = schedule
    reference.model.ranges = exp.ConditioningRanges((math.log(0.01), math.log(0.02)), (16, 21), (2, 4))
    ids = torch.zeros(len(schedule.configurations), dtype=torch.long)
    positions = torch.arange(64)[None].expand(len(windows.rows), -1)
    windows = windows.take(ids, positions, ids >= 0)
    delay = torch.tensor([c.delay for c in schedule.configurations])[:, None].expand(-1, 64)
    replan = torch.tensor([c.replan for c in schedule.configurations])[:, None].expand(-1, 64)
    valid = torch.arange(64)[None] < 128 // replan
    executed = (torch.arange(4)[None, None, :] < replan[..., None]).repeat_interleave(4, -1)
    slots = dataclasses.replace(
        windows.slots,
        valid_WK=valid,
        delay_WK=delay,
        replan_WK=replan,
        condition_WK4=reference.model.ranges.normalize(
            torch.full_like(delay, math.log(0.01), dtype=torch.float32), delay, replan
        ),
        forced_queue_WKD4=torch.zeros(len(ids), 64, 21, 4, dtype=torch.long),
        executed_WKF=executed,
    )
    windows = dataclasses.replace(windows, slots=slots)
    reference.reference_forward_model = copy.deepcopy(reference.model).requires_grad_(True)
    with torch.no_grad():
        expected = reference.reference_forward(windows.inputs(), len(ids), 16)
        # Each configuration has one window and pads to a three-window graph.
        actual = reference.evaluate_reference(windows, positions[ids, :16], torch.tensor([[4, -1, 2]]))
    mask = expected.factor_mask_WKF
    torch.testing.assert_close(actual.factor_mask_WKF, mask)
    torch.testing.assert_close(actual.log_probs_WKF[mask], expected.log_probs_WKF[mask], rtol=1e-4, atol=1e-5)
    for group, (p, q) in enumerate(zip(actual.distributions, expected.distributions, strict=True)):
        active = mask.reshape(len(ids), 64, 4, 4)[..., group]
        torch.testing.assert_close(p[active], q[active], rtol=1e-4, atol=1e-5)


def test_grouping_rejects_a_window_with_mixed_configuration() -> None:
    run = transformer_trainer("cpu", decoder_precision="fp32")
    windows = run.prepare(run.engine.collect(0)).windows
    delay = windows.slots.delay_WK.clone()
    live = torch.nonzero(windows.slots.valid_WK[0]).flatten()
    delay[0, live[-1]] += 1
    windows = dataclasses.replace(windows, slots=dataclasses.replace(windows.slots, delay_WK=delay))
    with pytest.raises(ValueError, match="one delay and replan"):
        exp.window_configurations(windows, run.policy.schedule.configurations)


def test_grouped_reference_preserves_an_empty_update() -> None:
    run = transformer_trainer("cpu", decoder_precision="fp32")
    windows = run.prepare(run.engine.collect(0)).windows
    windows = dataclasses.replace(
        windows, slots=dataclasses.replace(windows.slots, valid_WK=torch.zeros_like(windows.slots.valid_WK))
    )
    positions = torch.arange(16)[None].expand(len(windows.rows), -1)
    actual = run.reference.evaluate_reference(windows, positions, torch.tensor([[-1, -1, -1]]))
    assert actual.log_probs_WKF.shape == (len(windows.rows), 64, 16)
    assert not bool(actual.factor_mask_WKF.any())
    assert not bool(actual.log_probs_WKF.any())
