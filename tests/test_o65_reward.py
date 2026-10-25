"""Reward treatment identity and critic warmup after learner migration."""

import dataclasses
from pathlib import Path
from typing import Any

import pytest
import torch
from test_065_loop import PROVENANCE
from test_065_loop import exp
from test_o65_fixed_layout import fixed_trainer
from test_o65_fixed_layout import legacy_checkpoint

from hal.sim.native import RewardWeights

TREATMENT = RewardWeights(0.1, 120.0, 120.0)


def migration_trainer(*, initialize: bool) -> Any:
    run = fixed_trainer(length=1000000, initialize=initialize)
    run.engine.sampler.config = dataclasses.replace(run.engine.sampler.config, lambda_max=100)
    run.engine.sampler.lambda_max = 100
    return run


@pytest.mark.parametrize("changed", [False, True])
def test_learner_reward_change_restarts_32_critic_updates_and_persists_progress(tmp_path: Path, changed: bool) -> None:
    source = migration_trainer(initialize=True)
    source.run_update()
    source.config = dataclasses.replace(source.config, critic_warmup_updates=32)
    path = legacy_checkpoint(source, tmp_path / "source")
    target = migration_trainer(initialize=False)
    target.config = dataclasses.replace(source.config, reward_weights=TREATMENT if changed else RewardWeights())
    exp.restore_pilot_learner(target, path, PROVENANCE)
    assert target.warming == changed
    assert target.warmup_start_update == (256 if changed else 0)
    if not changed:
        assert target.warmup_moments == source.warmup_moments
        _, metrics = target.run_update()
        assert target.actor_steps > source.actor_steps
        assert "train/warmup_ended" not in metrics
        return
    assert target.warmup_moments == {}
    actor_before = [p.detach().clone() for p in target.policy.actor_parameters()]
    optimizer_before = exp._state_hash(target.actor_optimizer.state_dict())
    for update in range(257, 289):
        batch, metrics = target.run_update()
        assert batch.reference is not None
        assert batch.kl.sum() > 0
        assert target.warming == (update < 288)
        assert ("train/warmup_ended" in metrics) == (update == 288)
        assert target.actor_steps == source.actor_steps
        if update == 257:
            checkpoint = exp.save_checkpoint(target, tmp_path / "target", PROVENANCE)
            assert exp.read_training_manifest(checkpoint)["reward_weights"] == dataclasses.asdict(TREATMENT)
            restored = migration_trainer(initialize=False)
            restored.config = target.config
            exp.restore_checkpoint(restored, checkpoint, PROVENANCE)
            assert restored.warming and restored.warmup_start_update == 256
            target = restored
    for before, parameter in zip(actor_before, target.policy.actor_parameters(), strict=True):
        torch.testing.assert_close(before, parameter, rtol=0, atol=0)
    assert exp._state_hash(target.actor_optimizer.state_dict()) == optimizer_before
    target.run_update()
    assert target.actor_steps > source.actor_steps and target.update == 289


@pytest.mark.parametrize("field", ["damage", "stock", "game_win"])
def test_full_resume_rejects_reward_change(tmp_path: Path, field: str) -> None:
    source = fixed_trainer()
    source.run_update()
    path = exp.save_checkpoint(source, tmp_path, PROVENANCE)
    target = fixed_trainer(initialize=False)
    weights = dataclasses.replace(source.config.reward_weights, **{field: 0.0})
    target.config = dataclasses.replace(source.config, reward_weights=weights)
    with pytest.raises(ValueError, match="resume configuration mismatch"):
        exp.restore_checkpoint(target, path, PROVENANCE)
