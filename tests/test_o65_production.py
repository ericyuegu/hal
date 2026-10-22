"""Production cadence, long-rollout boundaries, and behavior instrumentation."""

import dataclasses
from pathlib import Path
from typing import Any

import melee
import numpy as np
import pytest
import torch
from test_065_loop import PROVENANCE
from test_065_loop import assert_batch_equal
from test_065_loop import exp
from test_065_loop import trainer
from test_o65_evaluation import small_model

from hal.data.policy_world_schema import POLICY_WORLD_SCHEMA_VERSION
from hal.data.schema import SCHEMA_VERSION
from hal.eval.action_behavior import COUNT_NAMES
from hal.eval.action_behavior import EVENT_STATES
from hal.eval.action_behavior import BehaviorTracker
from hal.eval.action_behavior import sequence_counts
from hal.eval.self_play_checkpoint import load_evaluation_checkpoint
from hal.models.action_sequence import ActionSequenceConfig
from hal.models.self_play import validate_executed_offsets
from hal.training.self_play_flops import SelfPlayFlops

O64_OFFSETS = (*range(1, 25), 27, 30)


def test_o64_supervised_offsets_admit_all_eleven_production_cadences() -> None:
    schedule = exp.production_config(O64_OFFSETS, rollout_frames=512)
    pairs = tuple((pair.delay, pair.replan) for pair in schedule.configurations)
    assert len(pairs) == 11
    assert (21, 2) in pairs and (21, 4) not in pairs
    validate_executed_offsets(O64_OFFSETS, pairs)
    with pytest.raises(ValueError, match="supervised"):
        validate_executed_offsets(O64_OFFSETS, ((23, 2),))
    with pytest.raises(ValueError, match="never supervised"):
        exp.production_config(tuple(range(1, 24)), rollout_frames=512)


def long_trainer() -> Any:
    run = trainer(groups=1, development=True)
    run.engine.sampler.schedule = dataclasses.replace(run.engine.sampler.schedule, rollout_frames=512)
    return run


def test_512_frame_windows_bootstrap_and_next_update_resume(tmp_path: Path) -> None:
    run = long_trainer()
    batch, _ = run.run_update()
    assert batch.rollout.frames == 512 * 2
    assert batch.rollout.table.ticks == 256
    assert batch.windows.rows.shape[1] == 64
    assert len(batch.windows.rows) == 4 * 4  # four blocks, four player streams
    table = batch.rollout.table
    last = batch.rollout.ids[table.integers[batch.rollout.ids, exp.CLOCK] == 510]
    assert len(last) > 0
    assert table.boundary[last].all() and not table.terminal[last].any()
    assert table.bootstrap[table.following[last]].all()
    assert (table.duration[last] == 2).all()
    checkpoint = exp.save_checkpoint(run, tmp_path, PROVENANCE)
    expected, expected_metrics = run.run_update()
    restored = long_trainer()
    exp.restore_checkpoint(restored, checkpoint, PROVENANCE)
    actual, actual_metrics = restored.run_update()
    assert_batch_equal(expected, actual)
    for name, value in run.policy.model.state_dict().items():
        torch.testing.assert_close(value, restored.policy.model.state_dict()[name], atol=0, rtol=0)
    assert {k: v for k, v in expected_metrics.items() if k.startswith("behavior/")} == {
        k: v for k, v in actual_metrics.items() if k.startswith("behavior/")
    }
    np.testing.assert_array_equal(run.engine.behavior_tracker.state, restored.engine.behavior_tracker.state)


def test_512_frame_gae_keeps_the_final_bootstrap() -> None:
    ticks = 256
    reward = torch.ones(ticks, 1)
    value = torch.full_like(reward, 2)
    duration = torch.full((ticks, 1), 2, dtype=torch.long)
    bootstrap = torch.ones_like(reward, dtype=torch.bool)
    trace = bootstrap.clone()
    trace[-1] = False
    targets = exp.dense_advantages(reward, value, value, duration, bootstrap, trace, bootstrap)
    delta = 1 + exp.GAMMA**2 * 2 - 2
    discount = (exp.GAMMA * exp.GAE_LAMBDA) ** 2
    expected = torch.tensor([delta * sum(discount**k for k in range(ticks - at)) for at in range(ticks)])
    torch.testing.assert_close(targets.advantages[:, 0], expected)
    assert targets.advantages[-1, 0] == pytest.approx(delta)


def test_behavior_events_active_exposure_and_zero_damage_stocks_survive_boundaries() -> None:
    states = [melee.Action.STANDING.value]
    for group in EVENT_STATES.values():
        states.extend((min(group), min(group), melee.Action.STANDING.value))
    states.extend((melee.Action.SHIELD_START.value, melee.Action.SHIELD.value, melee.Action.STANDING.value))
    states.extend((melee.Action.DEAD_DOWN.value, melee.Action.ON_HALO_WAIT.value, melee.Action.STANDING.value))
    states.extend((melee.Action.DEAD_DOWN.value, melee.Action.ON_HALO_WAIT.value, melee.Action.STANDING.value))
    action = np.stack((states, np.full(len(states), melee.Action.STANDING.value)), -1).astype(np.int64)
    stock = np.full_like(action, 4)
    stock[-6:, 0] = 3
    stock[-3:, 0] = 2
    percent = np.zeros(action.shape, dtype=np.float64)
    percent[-4, 0] = 5  # the second stock took damage before its loss
    x = np.zeros_like(percent)
    x[1:3, 0] = 200
    offline = sequence_counts(action, stock, percent, x, stage=25)
    tracker = BehaviorTracker(2)
    split = len(states) - 3
    for at in range(len(states)):
        if at == split:
            restored = BehaviorTracker(2)
            restored.restore(tracker.state.copy())
            restored.counts[:] = tracker.counts
            tracker = restored
        tracker.observe(
            np.array([0, 1]),
            action[at : at + 1],
            stock[at : at + 1],
            percent[at : at + 1],
            x[at : at + 1],
            np.array([25]),
        )
    for side in (0, 1):
        np.testing.assert_array_equal(tracker.counts[side], offline[side].counts)
    metrics = offline[0].metrics("student")
    assert metrics["behavior/student_zero_damage_self_destruct_count"] == 1
    assert metrics["behavior/student_shield_count"] == 2
    assert metrics["behavior/student_offstage_frames_count"] == 2
    assert metrics["behavior/student_smash_count"] >= 3
    assert metrics["behavior/student_forward_smash_per_min"] > 0
    assert all(key.count("/") == 1 for key in metrics)
    assert len(COUNT_NAMES) == tracker.counts.shape[1]
    with pytest.raises(ValueError, match="walkable edge"):
        tracker.observe(np.array([0, 1]), action[:1], stock[:1], percent[:1], x[:1], np.array([255]))


def test_flops_scale_with_actual_architecture_and_padded_work() -> None:
    model = small_model()
    cost = SelfPlayFlops.from_model(model)
    assert cost.replay(2, 16, 14, 4) == 2 * cost.replay(1, 16, 14, 4)
    assert cost.decode(25, 4) > cost.decode(14, 4)
    assert cost.rollout(8, 1, 25, 4, restart=True) > cost.rollout(8, 2, 23, 2, restart=False)
    assert cost.critic > 0


def test_native_cli_rejects_invalid_rollout_and_active_match_counts(tmp_path: Path) -> None:
    import os

    cpu = min(os.sched_getaffinity(0))
    args = exp.NativeCli(
        checkpoint=tmp_path / "bc.pt",
        sampler=tmp_path / "sampler.json",
        output=tmp_path / "run",
        data_dir=tmp_path,
        cpus=(cpu,),
        updates=256,
        cadence="production",
        rollout_frames=512,
        active_matches_per_gpu=64,
    )
    assert args.active_matches_per_gpu == 64
    with pytest.raises(ValueError, match="128-frame blocks"):
        dataclasses.replace(args, rollout_frames=510)
    with pytest.raises(ValueError, match="active matches"):
        dataclasses.replace(args, active_matches_per_gpu=3)


@pytest.mark.parametrize("field", ["head_offsets", "unknown_field"])
def test_o64_rejects_missing_or_unknown_architecture_fields(tmp_path: Path, field: str) -> None:
    architecture = dataclasses.asdict(ActionSequenceConfig(sample_chunk_length=30, head_offsets=O64_OFFSETS))
    for name in (
        "main_stick_layout",
        "player_vocab_sha256",
        "hidden_std_multiplier",
        "player_vocab_size",
        "return_conditioning",
        "depth_alpha",
    ):
        architecture.pop(name)
    if field == "head_offsets":
        architecture.pop(field)
    else:
        architecture[field] = 1
    path = tmp_path / "invalid.pt"
    torch.save(
        {
            "cfg": {
                "experiment_id": "064_production_percentile_v1",
                "checkpoint_format_version": 6,
                "arm": "bootstrap30",
                "mds_schema_version": SCHEMA_VERSION,
                "policy_world_schema_version": POLICY_WORLD_SCHEMA_VERSION,
                "return_conditioning": True,
                "architecture": architecture,
            }
        },
        path,
    )
    with pytest.raises(ValueError, match="architecture fields"):
        load_evaluation_checkpoint(path, device=torch.device("cpu"))


def test_behavior_exposure_excludes_countdown() -> None:
    tracker = BehaviorTracker(2)
    action = np.full((1, 2), melee.Action.FSMASH_MID.value, dtype=np.int64)
    stock = np.full((1, 2), 4, dtype=np.int64)
    zero = np.zeros((1, 2), dtype=np.float64)
    tracker.observe(np.array([0, 1]), action, stock, zero, zero, np.array([25]), playing=np.array([False]))
    assert not tracker.counts.any()
    assert (tracker.state[:, 0] == melee.Action.FSMASH_MID.value).all()
