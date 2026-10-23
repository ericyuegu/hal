"""Keep gathered actor windows immutable across captured forward and backward."""

import dataclasses

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
