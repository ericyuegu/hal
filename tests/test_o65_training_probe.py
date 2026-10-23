"""Capture the actual first PPO inputs without changing the learner."""

from pathlib import Path

import pytest
import torch
from test_065_loop import exp
from test_065_loop import transformer_trainer


@pytest.mark.parametrize("warm,nsys,rollout", [(2, False, False), (1, True, False), (1, False, True)])
def test_training_capture_rejects_other_profile_modes(warm: int, nsys: bool, rollout: bool) -> None:
    with pytest.raises(ValueError, match="training input capture"):
        exp.ProfileConfig(warm_updates=warm, nsys=nsys, rollout_only=rollout, capture_training_inputs=True)


def test_training_probe_preserves_the_actual_first_step_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_065_loop import PROVENANCE

    run = transformer_trainer("cpu", decoder_precision="fp32")
    targets_seen: list[exp.ActorTargets] = []
    original = run.policy.actor_step

    def record(
        windows: exp.ReplayBatch, targets: exp.ActorTargets, *, clip_epsilon: float
    ) -> tuple[exp.ActorLoss, exp.ReplayOutput]:
        targets_seen.append(targets)
        return original(windows, targets, clip_epsilon=clip_epsilon)

    monkeypatch.setattr(run.policy, "actor_step", record)
    batch, _ = run.run_update()
    before = exp._state_hash(run.policy.model.state_dict())
    path = tmp_path / "inputs.pt"
    exp.save_training_probe(run, batch, path, PROVENANCE)
    saved = torch.load(path, weights_only=True)
    assert saved["format"] == "hal.o65.training-probe.v1"
    assert saved["provenance"] == PROVENANCE
    step = batch.steps[0]
    windows = batch.windows.take(step, batch.placement.positions, step >= 0)
    actual = saved["cohorts"][0]
    for got, expected in zip(actual["inputs"], windows.inputs(), strict=True):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual["behavior"], batch.rollout.table.behavior[windows.rows], rtol=0, atol=0)
    selected = actual["selected"]
    targets = targets_seen[0]
    torch.testing.assert_close(selected, targets.selected_WK, rtol=0, atol=0)
    torch.testing.assert_close(actual["advantages"][selected], targets.advantages_WK[selected], rtol=0, atol=0)
    torch.testing.assert_close(actual["coefficients"], targets.kl_coefficient_WK, rtol=0, atol=0)
    assert selected.sum() == targets.denominator
    assert exp._state_hash(run.policy.model.state_dict()) == before
