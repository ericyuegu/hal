"""Exercise the exploitability probe: a pi_ref learner against one frozen opponent."""

import dataclasses
from pathlib import Path
from typing import Any

import pytest
import torch
from test_065_loop import PROVENANCE
from test_065_loop import FakeShard
from test_065_loop import ToyPolicy
from test_065_loop import exp


def frozen_config(**changes: Any) -> Any:
    config = exp.TrainConfig(
        actor_optimizer="adam",
        actor_lr=0.03,
        critic_lr=0.01,
        critic_epochs=1,
        critic_warmup_updates=0,
        advantage_scale="batch",
        warmup_steps=0,
        windows_per_step=8,
        snapshot_interval=1,
        lane_layout="frozen_opponent",
        league_selection="frozen",
        opponent_weight_power=0,
    )
    return dataclasses.replace(config, **changes)


def opponent_state() -> dict[str, torch.Tensor]:
    # Action 1 takes stocks in the fake simulator, so this opponent wins exchanges.
    torch.manual_seed(7)
    opponent = ToyPolicy()
    with torch.no_grad():
        opponent.model.logits.copy_(torch.tensor([[-1.0, 1.0]] * 4))
        opponent.model.value_head.bias.fill_(0.5)
    return opponent.model.state_dict()


def frozen_trainer(*, opponent: dict[str, torch.Tensor] | None = None, length: int = 1000000) -> Any:
    torch.manual_seed(23)
    policy = ToyPolicy()
    sampling = exp.EpisodeSampler(
        exp.SamplerConfig((exp.IdentityCount(1, 1, 100),), (exp.Matchup((1, 1), 1),), (25,), lambda_max=0.02),
        exp.development_config(tuple(range(1, 13))),
        exp.CounterRng(65),
    )
    engine = exp.RolloutEngine(
        ((FakeShard(tuple(range(8)), length=length),),),
        policy,
        sampling,
        {0: policy.frozen_copy()},
        lane_layout="frozen_opponent",
    )
    run = exp.Trainer(engine, frozen_config())
    run.install_frozen_opponent(opponent_state() if opponent is None else opponent)
    engine.initialize()
    return run


def assert_same_state(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    assert left.keys() == right.keys()
    for name, value in left.items():
        torch.testing.assert_close(value.cpu(), right[name].cpu(), atol=0, rtol=0)


def test_learner_starts_as_pi_ref_and_opponent_holds_the_installed_actor() -> None:
    run = frozen_trainer()
    assert_same_state(run.policy.model.state_dict(), run.reference.model.state_dict())
    assert_same_state(run.engine.anchors[1].model.state_dict(), opponent_state())
    assert run.slot_entries == {0: 0, 1: 1, 2: 0, 3: 0}
    assert [entry.entry_id for entry in run.league] == [0, 1]
    assert run.reference is run.engine.anchors[0]


def test_every_lane_plays_the_frozen_opponent() -> None:
    run = frozen_trainer()
    assert set(run.engine.lane_roles.values()) == {1}
    rollout = run.engine.collect(0)
    for state in run.engine.matches.values():
        assert sorted((player.anchor is None, player.anchor) for player in state.episode.players) == [
            (False, 1),
            (True, None),
        ]
    owners = run.engine.owners[0].reshape(-1, 2).sort(-1).values
    assert owners.tolist() == [[-1, 1]] * 8
    table, ids = rollout.table, rollout.ids
    assert len(ids) and bool((table.owner[ids] == -1).all())
    assert bool((table.opponent_anchor[ids] == 1).all())
    batch = run.prepare(rollout)
    torch.testing.assert_close(batch.opponent_weights, torch.ones_like(batch.opponent_weights))


def test_updates_change_only_the_learner() -> None:
    run = frozen_trainer()
    opponent = exp._state_hash(run.engine.anchors[1].model.state_dict())
    reference = exp._state_hash(run.reference.model.state_dict())
    learner = exp._state_hash(run.policy.model.state_dict())
    for _ in range(3):
        _, metrics = run.run_update()
        assert "league/rotated_slots" not in metrics
        for name in ("collapse/student_smash_fraction", "collapse/opponent_smash_fraction", "policy/entropy"):
            assert name in metrics
        assert metrics["gameplay/anchor1_stocks_taken"] + metrics["gameplay/anchor1_stocks_lost"] > 0
        assert 0 <= metrics["gameplay/anchor1_stock_share_32_updates"] <= 1
    assert exp._state_hash(run.engine.anchors[1].model.state_dict()) == opponent
    assert exp._state_hash(run.reference.model.state_dict()) == reference
    assert exp._state_hash(run.policy.model.state_dict()) != learner
    assert all(not parameter.requires_grad for parameter in run.engine.anchors[1].model.parameters())
    assert [entry.entry_id for entry in run.league] == [0, 1] and run.slot_entries[1] == 1


def test_resume_restores_the_opponent_and_rejects_another(tmp_path: Path) -> None:
    run = frozen_trainer(length=23)
    run.run_update()
    manifest = exp.save_checkpoint(run, tmp_path, PROVENANCE)
    resumed = frozen_trainer(length=23)
    exp.restore_checkpoint(resumed, manifest, PROVENANCE)
    resumed.check_frozen_opponent(opponent_state())
    assert resumed.anchor_stock_history == run.anchor_stock_history
    other = opponent_state()
    other["logits"] = torch.zeros(4, 2)
    with pytest.raises(ValueError, match="differs from the requested actor"):
        resumed.check_frozen_opponent(other)


def test_frozen_league_rejects_mixed_settings() -> None:
    with pytest.raises(ValueError, match="frozen_opponent lanes"):
        frozen_config(lane_layout="anchor_heavy")
    with pytest.raises(ValueError, match="frozen_opponent lanes"):
        frozen_config(league_selection="pfsp")
    with pytest.raises(ValueError, match="opponent_weight_power 0"):
        frozen_config(opponent_weight_power=1.0)
    run = frozen_trainer()
    with pytest.raises(ValueError, match="before the pool starts"):
        run.install_frozen_opponent(opponent_state())
    with pytest.raises(ValueError, match="actor tensors differ"):
        exp.frozen_opponent_weights(run.policy.model.state_dict(), {"logits": torch.zeros(4, 2)})
