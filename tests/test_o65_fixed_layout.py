"""Fixed policy batches survive resets and in-place snapshot promotion."""

import copy
import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from test_065_loop import PROVENANCE
from test_065_loop import FakeShard
from test_065_loop import ToyPolicy
from test_065_loop import exp


def fixed_trainer(
    *, length: int = 67, groups: int = 1, initialize: bool = True, shards: int = 1, warmup: int = 0
) -> Any:
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
            critic_warmup_updates=warmup,
            advantage_scale="warmup" if warmup else "batch",
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
    assert [roles.count(slot) for slot in (None, 0, 1, 2, 3)] == [16, 12, 12, 12, 12]
    assert [exp.opponent_slots(16).count(slot) for slot in (None, 0, 1, 2, 3)] == [4, 3, 3, 3, 3]
    # Both self-play ports train, so 32 of 80 student streams are self-play.
    assert 2 * roles.count(None) / (2 * roles.count(None) + 48) == 0.4
    assert all(
        [roles[start : start + 16].count(slot) for slot in (None, 0, 1, 2, 3)] == [4, 3, 3, 3, 3]
        for start in range(0, 64, 16)
    )
    assert all(roles[start : start + 8].count(None) == 2 for start in range(0, 64, 8))
    assert exp.opponent_slots(4) == (0, 1, 2, 3) and exp.opponent_slots(3) == (None,) * 3
    assert [exp.opponent_slots(8).count(slot) for slot in (None, 0, 1, 2, 3)] == [4, 1, 1, 1, 1]
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
    assert [entry.entry_id for entry in run.league] == list(range(6))
    assert 5 in run.slot_entries.values() and run.slot_entries[0] == 0
    weights = {entry.entry_id: entry.weights for entry in run.league}
    for slot, entry_id in run.slot_entries.items():
        for name, value in engine.anchors[slot].model.state_dict().items():
            torch.testing.assert_close(value, weights[entry_id][name], rtol=0, atol=0)


def test_rotation_swaps_slot_weights_without_ending_episodes() -> None:
    run = fixed_trainer(length=1000000, groups=2)
    run.run_update()
    engine = run.engine
    assert run.slot_entries[1] == 1
    assert all(state.end_clock is None for state in engine.matches.values())
    assert all(engine.active[group][lane] for group, lane in engine.locations.values())
    for name, value in engine.anchors[1].model.state_dict().items():
        torch.testing.assert_close(value, run.policy.model.state_dict()[name], rtol=0, atol=0)
    batch, _ = run.run_update()
    # Slot 1 lanes of the next group continue their first episode against the new weights.
    assert batch.rollout.group == 1 and batch.rollout.resets == 0
    assert all(state.episode.number == 0 for state in engine.matches.values())
    table, rows = batch.rollout.table, batch.rollout.ids
    assert bool((table.opponent_anchor[rows] == 1).any())


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


def record_actor_targets(run: Any, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    seen: list[Any] = []
    original = run.policy.actor_step

    def record(batch: Any, targets: Any, *, clip_epsilon: float) -> Any:
        seen.append(targets)
        return original(batch, targets, clip_epsilon=clip_epsilon)

    monkeypatch.setattr(run.policy, "actor_step", record)
    return seen


def test_warmup_advantage_scale_is_the_pooled_kl_free_spread(monkeypatch: pytest.MonkeyPatch) -> None:
    run = fixed_trainer(length=1000000, warmup=2)
    run.engine.sampler.config = dataclasses.replace(run.engine.sampler.config, lambda_max=100)
    run.engine.sampler.lambda_max = 100
    first, _ = run.run_update()
    second, metrics = run.run_update()
    assert first.reference is None and second.reference is None and not run.warming
    # The scale pools every warmup update, not only the last one.
    expected = float(torch.cat((first.advantages, second.advantages)).double().std(unbiased=False))
    assert expected != pytest.approx(float(second.advantages.double().std(unbiased=False)))
    assert run.warmup_advantage_sigma() == pytest.approx(expected)
    assert metrics["train/warmup_advantage_sigma"] == pytest.approx(expected)
    seen = record_actor_targets(run, monkeypatch)
    # Larger KL coefficients widen the regularized spread but must not shrink the KL term.
    run.engine.sampler.lambda_max = 50
    _, metrics = run.run_update()
    assert seen and all(float(targets.advantage_sigma) == pytest.approx(expected) for targets in seen)
    assert metrics["policy/advantage_sigma"] == pytest.approx(expected)
    assert metrics["policy/batch_advantage_sigma"] != pytest.approx(expected)
    with pytest.raises(ValueError, match="warmup advantage scale"):
        exp.TrainConfig(critic_warmup_updates=0)


def test_opponent_weights_scale_only_centered_anchor_advantages(monkeypatch: pytest.MonkeyPatch) -> None:
    runs, records = [], []
    # Distinct per-slot rates give weights 0.5, 1.5, 1 and 1.75 for slots 0..3.
    rates = {0: 0.75, 1: 0.25, 2: 0.5, 3: 0.0}
    for treated in (False, True):
        run = fixed_trainer(length=1000000)
        if treated:
            for slot in (1, 2, 3):
                run.league.append(exp.league_entry(slot, 0, run.reference.model.state_dict()))
                run.slot_entries[slot] = slot
            run.league_stock_shares.update(rates)
        records.append(record_actor_targets(run, monkeypatch))
        runs.append(run.run_update()[0])
    control, treatment = runs
    opponents = control.rollout.table.opponent_anchor[control.rollout.ids]
    assert set(opponents.tolist()) == {-1, 0, 1, 2, 3}
    slot_weights = torch.tensor([0.5, 1.5, 1.0, 1.75])
    expected = torch.where(opponents >= 0, slot_weights[opponents.clamp_min(0)], 1.0)
    torch.testing.assert_close(treatment.opponent_weights, expected.to(treatment.opponent_weights.dtype))
    torch.testing.assert_close(control.opponent_weights, torch.ones_like(control.opponent_weights))
    ratios = []
    for base, weighted in zip(*records, strict=True):
        assert float(base.advantage_mean) == float(weighted.advantage_mean)
        assert float(base.advantage_sigma) == float(weighted.advantage_sigma)
        selected = base.selected_WK & ((base.advantages_WK - base.advantage_mean).abs() > 1e-2)
        ratios.append(
            ((weighted.advantages_WK - weighted.advantage_mean) / (base.advantages_WK - base.advantage_mean))[selected]
        )
    ratio = torch.cat(ratios)
    matches = [(ratio - weight).abs() < 1e-3 for weight in (0.5, 1.0, 1.5, 1.75)]
    assert all(bool(match.any()) for match in matches)
    assert bool(torch.stack(matches).any(0).all())


def test_opponent_weight_formula_clips_and_power_zero_disables() -> None:
    run = fixed_trainer(initialize=False)
    run.league_stock_shares.update({1: 0.3, 2: 1.0, 3: 0.0})
    run.league_stock_shares[0] = 0.75
    run.slot_entries.update({1: 1, 2: 2, 3: 3})
    torch.testing.assert_close(run.opponent_weights(), torch.tensor([0.5, 1.4, 0.25, 1.75], dtype=torch.float64))
    run.config = dataclasses.replace(run.config, opponent_weight_power=0.0)
    torch.testing.assert_close(run.opponent_weights(), torch.ones(4, dtype=torch.float64))


def test_pfsp_rotation_prefers_hard_entries_and_evicts_the_easiest() -> None:
    run = fixed_trainer(length=1000000)
    run.config = dataclasses.replace(run.config, pfsp_uniform=0.0)
    run.update = 1
    assert run.rotate_league() == (1,)
    assert run.slot_entries == {0: 0, 1: 1, 2: 0, 3: 0}
    hard = 1
    for update in range(2, 22):
        for entry in run.league:
            run.league_stock_shares[entry.entry_id] = 0.0 if entry.entry_id == hard else 0.95
        run.update = update
        run.rotate_league()
        slots = [run.slot_entries[slot] for slot in (1, 2, 3)]
        assert run.league[-1].entry_id == update and update in slots and hard in slots
    run = fixed_trainer(length=1000000)
    run.config = dataclasses.replace(run.config, max_league_size=6)
    for update in range(1, 6):
        run.update = update
        run.rotate_league()
    run.league_stock_shares.update({1: 0.99, 2: 0.99, 3: 0.99, 4: 0.6, 5: 0.9})
    run.update = 6
    run.rotate_league()
    # Entries 0..3 are protected; entry 5 is the easiest unprotected one.
    assert [entry.entry_id for entry in run.league] == [0, 1, 2, 3, 4, 6]
    assert set(run.league_stock_shares) == {0, 1, 2, 3, 4, 6}
    assert 5 not in run.slot_entries.values() and 6 in run.slot_entries.values()
    # Equal win rates evict the oldest unprotected entry.
    run.league_stock_shares.update({4: 0.8, 6: 0.8})
    run.update = 7
    run.rotate_league()
    assert [entry.entry_id for entry in run.league] == [0, 1, 2, 3, 6, 7]


def test_league_checkpoint_reuses_shards_and_restores_stock_shares(tmp_path: Path) -> None:
    run = fixed_trainer(length=1000000)
    run.run_update()
    run.league_stock_shares[0] = 0.7
    checkpoint = exp.save_checkpoint(run, tmp_path, PROVENANCE)
    assert len(list(tmp_path.glob("*.league.pt"))) == 2
    slot_weights = {slot: copy.deepcopy(policy.model.state_dict()) for slot, policy in run.engine.anchors.items()}
    expected, _ = run.run_update()
    restored = fixed_trainer(length=1000000)
    exp.restore_checkpoint(restored, checkpoint, PROVENANCE)
    assert restored.league_stock_shares == {0: 0.7, 1: 0.5}
    assert restored.slot_entries == {0: 0, 1: 1, 2: 0, 3: 0}
    for slot, policy in restored.engine.anchors.items():
        for name, value in policy.model.state_dict().items():
            torch.testing.assert_close(value, slot_weights[slot][name], rtol=0, atol=0)
    assert [(e.entry_id, e.update, e.sha256) for e in restored.league] == [
        (e.entry_id, e.update, e.sha256) for e in run.league[:2]
    ]
    actual, _ = restored.run_update()
    assert exp._batch_hash(expected) == exp._batch_hash(actual)
    # A later checkpoint writes only the new entry and refers to the banked ones.
    exp.save_checkpoint(restored, tmp_path, PROVENANCE)
    assert len(list(tmp_path.glob("*.league.pt"))) == 3
    manifest = json.loads(checkpoint.read_text())
    manifest["shards"] = [shard for shard in manifest["shards"] if shard["role"] != "league_1"]
    checkpoint.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="league"):
        exp.restore_checkpoint(fixed_trainer(length=1000000), checkpoint, PROVENANCE)


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
    anchors = {key: copy.deepcopy(run.engine.anchors[key].model.state_dict()) for key in (0, 1)}
    state.update(version=8, anchors=anchors, anchor_candidates=(0, 1), next_anchor=2)
    groups = len(run.engine.groups)
    frames = run.engine.sampler.schedule.rollout_frames
    initial = ((run.config.head_start_frames + frames - 1) // frames) * frames
    state["clocks"] = [initial + ((256 + groups - 1 - group) // groups) * frames for group in range(groups)]
    state["cursor"] = 256 % groups
    for name in ("league", "league_stock_shares", "slot_entries", "pool_start_update"):
        state.pop(name)
    provenance = {**PROVENANCE, "git_sha": exp.PILOT_SOURCE}
    manifest = {
        **snapshot.manifest,
        "format": "hal.o65.training.v8",
        "version": 8,
        "provenance": provenance,
        "configuration_sha256": exp.legacy_pilot_digest(run),
    }
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
    return exp.write_checkpoint(
        dataclasses.replace(snapshot, state=state, manifest=manifest, actor=actor, league=()), directory
    )


def test_learner_migration_preserves_optimizers_history_and_fresh_pool(tmp_path: Path) -> None:
    source = fixed_trainer(length=1000000)
    source.run_update()
    source.anchor_win_history = [{0: (1, 2), 1: (0, 1)}]
    timing = source.engine.sampler.schedule.configurations[0]
    source.warmup_moments = {timing: (4, 2.0, 5.0)}
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
    # The fixed advantage scale after migration comes from the pilot's warmup.
    assert target.warmup_moments == source.warmup_moments
    assert target.warmup_advantage_sigma() == pytest.approx(1.0)
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
            advantage_scale="batch",
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


def test_legacy_pilot_digest_ignores_only_the_league_treatment() -> None:
    run = fixed_trainer(initialize=False, warmup=2)
    baseline = exp.legacy_pilot_digest(run)
    treated = fixed_trainer(initialize=False, warmup=2)
    treated.config = dataclasses.replace(
        treated.config,
        advantage_scale="batch",
        snapshot_interval=7,
        max_league_size=9,
        pfsp_power=3.0,
        stock_share_ema_alpha=0.3,
        pfsp_uniform=0.5,
        opponent_weight_power=2.0,
    )
    treated.engine.sampler.config = dataclasses.replace(
        treated.engine.sampler.config, percentile_alpha=2.0, percentile_beta=3.0
    )
    assert exp.legacy_pilot_digest(treated) == baseline
    assert set(exp.LEAGUE_TRAIN_FIELDS) <= set(exp._configuration_identity(run)["train"])
    changed = fixed_trainer(initialize=False, warmup=2)
    changed.config = dataclasses.replace(changed.config, actor_lr=2 * changed.config.actor_lr)
    assert exp.legacy_pilot_digest(changed) != baseline


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="NativeCli validates Linux CPU affinity")
def test_profile_and_zero_warmup_configs_use_batch_advantage_scale() -> None:
    native = exp.GpuProfileCli(
        Path("checkpoint.pt"), Path("data"), Path("output"), cpus=tuple(sorted(os.sched_getaffinity(0)))[:16]
    ).native()
    assert native.training.critic_warmup_updates == 0 and native.training.advantage_scale == "batch"
    with pytest.raises(ValueError, match="warmup advantage scale"):
        exp.TrainConfig(critic_warmup_updates=0)


def test_pfsp_uniform_fraction_keeps_easy_entries_drawable() -> None:
    def easy_draws(uniform: float) -> int:
        run = fixed_trainer(initialize=False)
        run.config = dataclasses.replace(run.config, pfsp_uniform=uniform, max_league_size=40)
        count = 0
        for update in range(1, 33):
            run.update = update
            # Entry 1 looks fully beaten, so plain PFSP gives it 0.0025 of a hard entry's weight.
            run.league_stock_shares.update({entry.entry_id: 0.0 for entry in run.league})
            run.league_stock_shares[1] = 1.0
            run.rotate_league()
            # Two draws from fewer than three other entries take every entry.
            count += 1 in [run.slot_entries[slot] for slot in (1, 2, 3)] and update > 2
        return count

    assert easy_draws(0.0) == 0
    assert easy_draws(1.0) > 2
