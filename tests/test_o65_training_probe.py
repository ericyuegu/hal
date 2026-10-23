"""Capture the actual first PPO inputs without changing the learner."""

import dataclasses
from pathlib import Path

import pytest
import torch
from test_065_loop import exp
from test_065_loop import transformer_trainer


@pytest.mark.parametrize("warm,nsys,rollout", [(2, False, False), (1, True, False), (1, False, True)])
def test_training_capture_rejects_other_profile_modes(warm: int, nsys: bool, rollout: bool) -> None:
    with pytest.raises(ValueError, match="training input capture"):
        exp.ProfileConfig(warm_updates=warm, nsys=nsys, rollout_only=rollout, capture_training_inputs=True)


@pytest.mark.parametrize("decisions", [16, "all"])
def test_training_probe_preserves_the_actual_first_step_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decisions: int | str
) -> None:
    from test_065_loop import PROVENANCE

    from scripts.o65_replay_probe import load_cohort

    run = transformer_trainer(
        "cpu", development=True, decoder_precision="fp32", matches=(0, 1, 2, 3), episode_length=10000
    )
    run.config = dataclasses.replace(run.config, decisions_per_window=decisions, actor_microbatches=2)
    targets_seen: list[exp.ActorTargets] = []
    inputs_seen: list[tuple[torch.Tensor, ...]] = []
    original = run.policy.actor_step

    def record(
        windows: exp.ReplayBatch, targets: exp.ActorTargets, *, clip_epsilon: float
    ) -> tuple[exp.ActorLoss, exp.ReplayOutput]:
        targets_seen.append(targets)
        inputs_seen.append(tuple(value.clone() for value in windows.inputs()))
        return original(windows, targets, clip_epsilon=clip_epsilon)

    monkeypatch.setattr(run.policy, "actor_step", record)
    batch, _ = run.run_update()
    before = exp._state_hash(run.policy.model.state_dict())
    path = tmp_path / "inputs.pt"
    exp.save_training_probe(run, batch, path, PROVENANCE)
    saved = torch.load(path, weights_only=True)
    assert saved["format"] == "hal.o65.training-probe.v2"
    assert saved["provenance"] == PROVENANCE
    for index, actual in enumerate(saved["cohorts"]):
        targets = targets_seen[index]
        for got, expected in zip(actual["inputs"], inputs_seen[index], strict=True):
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
        for key, value in (
            ("selected", targets.selected_WK),
            ("behavior", targets.behavior_log_probs_WKF),
            ("advantages", targets.advantages_WK),
            ("coefficients", targets.kl_coefficient_WK),
            ("advantage_mean", targets.advantage_mean),
            ("advantage_sigma", targets.advantage_sigma),
            ("denominator", targets.denominator),
        ):
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
        assert actual["selected"].sum() < actual["denominator"]
    loaded = load_cohort(path, PROVENANCE["checkpoint_sha256"], 0)
    torch.testing.assert_close(loaded["denominator"], targets_seen[0].denominator, rtol=0, atol=0)
    assert exp._state_hash(run.policy.model.state_dict()) == before
