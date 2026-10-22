"""Fixed policy batches survive resets and in-place snapshot promotion."""

import copy
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from test_065_loop import PROVENANCE
from test_065_loop import FakeShard
from test_065_loop import ToyPolicy
from test_065_loop import exp


def fixed_trainer(*, length: int = 67, groups: int = 1, initialize: bool = True, shards: int = 1) -> Any:
    torch.manual_seed(19)
    policy = ToyPolicy()
    config = exp.SamplerConfig((exp.IdentityCount(1, 1, 100),), (exp.Matchup((1, 1), 1),), (25,), lambda_max=0.02)
    sampling = exp.EpisodeSampler(config, exp.development_config(tuple(range(1, 13))), exp.CounterRng(65))
    pools = tuple(
        tuple(
            FakeShard(
                tuple(range(g * 16 + shard * (16 // shards), g * 16 + (shard + 1) * (16 // shards))),
                length=length,
                timeout=True,
            )
            for shard in range(shards)
        )
        for g in range(groups)
    )
    engine = exp.RolloutEngine(pools, policy, sampling, {0: policy.frozen_copy()})
    run = exp.Trainer(
        engine,
        exp.TrainConfig(
            actor_optimizer="adam",
            critic_warmup_updates=0,
            windows_per_step=32,
            critic_epochs=1,
            snapshot_interval=1,
            warmup_steps=0,
        ),
    )
    if initialize:
        engine.initialize()
    return run


def test_layout_has_exact_counts_and_spreads_anchors() -> None:
    roles = exp.opponent_slots(64)
    assert [roles.count(slot) for slot in (None, 0, 1, 2, 3)] == [54, 4, 2, 2, 2]
    assert [exp.opponent_slots(16).count(slot) for slot in (None, 0, 1, 2, 3)] == [12, 1, 1, 1, 1]
    assert all(any(role is not None for role in roles[start : start + 8]) for start in range(0, 64, 8))
    with pytest.raises(ValueError):
        exp.opponent_slots(0)
    with pytest.raises(TypeError):
        exp.SamplerConfig((), (), (), anchor_rate=0.15)


def test_resets_and_promotions_keep_layouts_and_policy_addresses() -> None:
    run = fixed_trainer(groups=2, shards=2)
    engine = run.engine
    identities = {slot: id(policy) for slot, policy in engine.anchors.items()}
    pointers = {slot: [p.data_ptr() for p in policy.model.parameters()] for slot, policy in engine.anchors.items()}
    layouts = {}
    for _ in range(5):
        batch, metrics = run.run_update()
        assert batch.rollout.resets > 0
        current = {key: layout.streams for key, layout in engine.group_layouts[batch.rollout.group].items()}
        if batch.rollout.group in layouts:
            assert current == layouts[batch.rollout.group]
        layouts[batch.rollout.group] = current
        assert metrics["systems/post_startup_compiles"] == metrics["systems/post_startup_captures"] == 0
        assert identities == {slot: id(policy) for slot, policy in engine.anchors.items()}
        assert pointers == {
            slot: [p.data_ptr() for p in policy.model.parameters()] for slot, policy in engine.anchors.items()
        }
        for match, state in engine.matches.items():
            assert (
                [p.anchor for p in state.episode.players if p.anchor is not None] == [engine.lane_roles[match]]
                if engine.lane_roles[match] is not None
                else all(p.anchor is None for p in state.episode.players)
            )
    assert sorted(run.slot_updates.values()) == [3, 4, 5]


def test_promotion_truncates_only_its_slot_and_preserves_boundary_bootstrap() -> None:
    run = fixed_trainer(length=1000000, groups=2)
    batch, metrics = run.run_update()
    engine = run.engine
    for match, state in engine.matches.items():
        group, lane = engine.locations[match]
        assert (state.end_clock is not None) == (engine.lane_roles[match] == 1)
        assert bool(engine.active[group][lane]) == (state.end_clock is None)
    table = batch.rollout.table
    rows = batch.rollout.ids
    selected = rows[(table.opponent_anchor[rows] == 1) & table.boundary[rows]]
    assert len(selected) == 1
    assert not table.terminal[selected].any()
    assert table.bootstrap[table.following[selected]].all()
    assert metrics["gameplay/anchor1_count"] == 0
    assert not any(
        2 * match + side == stream
        for match, role in engine.lane_roles.items()
        if role == 1
        for side in (0, 1)
        for stream, _ in run.episode_metrics
    )
    for name, value in engine.anchors[1].model.state_dict().items():
        torch.testing.assert_close(value, run.policy.model.state_dict()[name], rtol=0, atol=0)
    run.run_update()
    assert all(
        engine.matches[match].episode.number == int(engine.locations[match][0] == 1)
        for match, role in engine.lane_roles.items()
        if role == 1
    )


def test_fixed_slot_checkpoint_restores_slot_versions_and_next_update(tmp_path: Path) -> None:
    run = fixed_trainer(length=1000000)
    run.run_update()
    checkpoint = exp.save_checkpoint(run, tmp_path, PROVENANCE)
    expected, metrics = run.run_update()
    restored = fixed_trainer(length=1000000)
    exp.restore_checkpoint(restored, checkpoint, PROVENANCE)
    assert restored.slot_updates == {1: 1, 2: 0, 3: 0}
    actual, restored_metrics = restored.run_update()
    assert exp._batch_hash(expected) == exp._batch_hash(actual)
    assert metrics["gameplay/anchor2_weights_update"] == restored_metrics["gameplay/anchor2_weights_update"]
    for value, other in zip(run.policy.model.parameters(), restored.policy.model.parameters(), strict=True):
        torch.testing.assert_close(value, other, rtol=0, atol=0)


def test_capture_guard_names_missing_graph_key() -> None:
    policy = object.__new__(exp.TransformerPolicy)
    policy.startup_finished = True
    policy.post_startup_captures = 0
    with pytest.raises(RuntimeError, match="after startup.*tick.*missing"):
        policy.admit_capture(("tick", "missing"))
    assert policy.post_startup_captures == 1


def test_masked_evaluations_preserve_the_other_policy_rows() -> None:
    run = fixed_trainer(length=1000000)
    batch, _ = run.run_update()
    table = batch.rollout.table
    rows = batch.rollout.ids
    # Frozen anchors must never overwrite the learner's behavior or value rows.
    behavior = table.behavior[rows].clone()
    features = exp._take_features(table.evaluation.features, rows)
    output = exp.PolicyEvaluation(
        table.actions[rows], behavior + 9, tuple(p[rows] for p in table.evaluation.distributions), features
    )
    table.record_evaluation(rows, output, table.values[rows] + 9, torch.zeros_like(rows, dtype=torch.bool))
    torch.testing.assert_close(table.behavior[rows], behavior, rtol=0, atol=0)


def legacy_checkpoint(run: Any, directory: Path) -> Path:
    run.config = dataclasses.replace(run.config, snapshot_interval=256)
    run.update = 256
    run.engine.completed_update = 255
    snapshot = exp.capture_checkpoint(run, PROVENANCE)
    state = copy.deepcopy(snapshot.state)
    state.update(
        version=8, anchors={key: state["anchors"][key] for key in (0, 1)}, anchor_candidates=(0, 1), next_anchor=2
    )
    groups = len(run.engine.groups)
    frames = run.engine.sampler.schedule.rollout_frames
    initial = ((run.config.head_start_frames + frames - 1) // frames) * frames
    state["clocks"] = [initial + ((256 + groups - 1 - group) // groups) * frames for group in range(groups)]
    state["cursor"] = 256 % groups
    state.pop("slot_updates")
    state.pop("pool_start_update")
    state.pop("warmup_start_update")
    provenance = {**PROVENANCE, "git_sha": exp.PILOT_SOURCE, "adapter_sha256": exp.PILOT_ADAPTER_SHA256}
    manifest = {
        **snapshot.manifest,
        "format": "hal.o65.training.v8",
        "version": 8,
        "provenance": provenance,
        "configuration_sha256": exp.legacy_pilot_digest(run),
    }
    manifest.pop("reward_weights")
    actor = {
        "format": exp.EVALUATION_FORMAT,
        "model": state["model"],
        "decoder_precision": "bf16",
        "provenance": {
            "update": 256,
            "git_sha": exp.PILOT_SOURCE,
            "reference_sha256": PROVENANCE["checkpoint_sha256"],
            "data_manifests": {"training": PROVENANCE["data_manifest_sha256"]},
        },
    }
    # Toy policies do not publish actors; this envelope exercises the same migration validator.
    return exp.write_checkpoint(dataclasses.replace(snapshot, state=state, manifest=manifest, actor=actor), directory)


def test_learner_migration_preserves_optimizers_history_and_fresh_pool(tmp_path: Path) -> None:
    source = fixed_trainer(length=1000000)
    source.run_update()
    source.anchor_win_history = [{0: (1, 2), 1: (0, 1)}]
    path = legacy_checkpoint(source, tmp_path)
    target = fixed_trainer(length=1000000, initialize=False)
    target.config = dataclasses.replace(target.config, snapshot_interval=256)
    exp.restore_pilot_learner(target, path, PROVENANCE)
    assert target.update == target.pool_start_update == 256
    assert target.anchor_win_history == source.anchor_win_history
    for name in ("actor_optimizer", "critic_optimizer", "scheduler"):
        assert exp._state_hash(getattr(target, name).state_dict()) == exp._state_hash(
            getattr(source, name).state_dict()
        )
    assert target.actor_steps == source.actor_steps
    assert target.engine.clocks == [0]
    assert all(not shard.ticks.any() for group in target.engine.groups for shard in group)
    assert all(state.episode.number == 0 for state in target.engine.matches.values())
    assert not target.episode_metrics
    assert target.slot_updates == {1: 256, 2: 0, 3: 0}
    target.run_update()
    assert target.update == 257
    resumed = exp.save_checkpoint(target, tmp_path / "new", PROVENANCE)
    restored = fixed_trainer(length=1000000)
    restored.config = dataclasses.replace(restored.config, snapshot_interval=256)
    exp.restore_checkpoint(restored, resumed, PROVENANCE)
    assert restored.pool_start_update == 256 and restored.engine.clocks == [256]


def test_learner_migration_samples_fresh_head_start_with_saved_lambda(tmp_path: Path) -> None:
    source = fixed_trainer(length=1000000)
    source.run_update()
    source.config = dataclasses.replace(source.config, head_start_frames=16)
    source.engine.sampler.lambda_max = 0.015
    path = legacy_checkpoint(source, tmp_path)
    target = fixed_trainer(length=1000000, initialize=False)
    target.config = dataclasses.replace(target.config, snapshot_interval=256, head_start_frames=16)
    head = exp.restore_pilot_learner(target, path, PROVENANCE)
    assert head is not None and max(head.episode_ages) <= 16
    assert target.update == 256 and target.engine.clocks == [256]
    assert all(
        p.log_lambda <= math.log(0.015) for state in target.engine.matches.values() for p in state.episode.players
    )
    assert all(
        target.engine.matches[match].end_clock == 256 for match, role in target.engine.lane_roles.items() if role == 1
    )
    assert not target.episode_metrics
    target.run_update()
    assert target.update == 257


@pytest.mark.parametrize("field,value", [("update", 255), ("version", 9), ("configuration_sha256", "f" * 64)])
def test_learner_migration_rejects_other_checkpoints(tmp_path: Path, field: str, value: object) -> None:
    source = fixed_trainer()
    path = legacy_checkpoint(source, tmp_path)
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    target = fixed_trainer()
    with pytest.raises(ValueError):
        exp.restore_pilot_learner(target, path, PROVENANCE)


def test_short_gpu_run_has_no_compiles_or_captures_after_startup() -> None:
    import os

    if os.environ.get("HAL_O65_FIXED_GPU") != "1":
        pytest.skip("run once explicitly with HAL_O65_FIXED_GPU=1 on the RTX 3060")
    assert torch.cuda.is_available()
    assert "3060" in torch.cuda.get_device_name()
    output = Path("runs/fixed/gpu")
    config = exp.NativeCli(
        checkpoint=Path("runs/proxy/final.pt"),
        sampler=Path("experiments/o65/pilot-sampler.json"),
        output=output,
        data_dir=Path("/home/ericgu/src/melee-sim-light/data"),
        cpus=(min(os.sched_getaffinity(0)),),
        groups=1,
        matches_per_shard=16,
        updates=3,
        max_frame=0,
        training=exp.TrainConfig(
            critic_warmup_updates=0,
            head_start_frames=16,
            windows_per_step=32,
            snapshot_interval=1,
            checkpoint_interval=3,
        ),
    )
    exp.train_native(config)
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == 3
    assert all(row["systems/post_startup_compiles"] == row["systems/post_startup_captures"] == 0 for row in metrics)
    assert all(row["data/resets"] > 0 for row in metrics)
    assert [row["global_step"] for row in metrics] == [1, 2, 3]
    (output / "measurement.json").write_text(
        json.dumps(
            {
                "startup_capture_seconds": metrics[0]["systems/startup_capture_seconds"],
                "seconds_per_update": [row["systems/update_seconds"] for row in metrics],
                "compiles": [row["systems/post_startup_compiles"] for row in metrics],
                "captures": [row["systems/post_startup_captures"] for row in metrics],
                "peak_allocated_bytes": max(row["systems/peak_allocated_bytes"] for row in metrics),
                "peak_reserved_bytes": max(row["systems/peak_reserved_bytes"] for row in metrics),
            },
            indent=2,
        )
    )
