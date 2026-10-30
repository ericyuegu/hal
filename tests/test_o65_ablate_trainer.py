"""CPU checks for O65 reward, opponent, and collapse ablations."""

import dataclasses
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import melee
import numpy as np
import pytest
import torch
import tyro
from test_065_loop import PROVENANCE
from test_065_loop import ROOT
from test_065_loop import exp
from test_065_loop import trainer
from test_o65_fixed_layout import fixed_trainer


def reward_table() -> tuple[Any, torch.Tensor, np.ndarray, np.ndarray]:
    stocks = np.array([(4, 4), (4, 4), (4, 4), (4, 3), (3, 3), (3, 1), (3, 0), (3, 4)], dtype=np.int32)
    percents = np.array(
        [(0, 0), (0, 0), (20, 10), (25, 0), (0, 7), (2, 8), (math.inf, 0), (math.nan, 5)],
        dtype=np.float32,
    )
    records = np.zeros((8, 1), dtype=exp.native.FRAME_DTYPE)
    for side in range(2):
        for name, values in (("stock", stocks), ("percent", percents)):
            records["columns"][:, 0, exp.NATIVE_COLUMN_LAYOUT[f"p{side + 1}_{name}"][0]] = values[:, side].view(
                np.uint32
            )
    store = exp.FrameStore(8, 1, torch.device("cpu"))
    store.records.copy_(torch.from_numpy(records.view(np.uint8).reshape(8, 1, -1)))
    table = exp.DenseRollout(64, store)
    ids = torch.arange(4)
    table.integers[ids, exp.START] = torch.tensor([1, 1, 3, 3])
    table.integers[ids, exp.LENGTH] = 2
    table.integers[ids, exp.SIDE] = torch.tensor([0, 1, 0, 1])
    table.duration[ids] = torch.tensor([4, 4, 3, 3])
    for row in ids.tolist():
        side = int(table.integers[row, exp.SIDE])
        start = int(table.integers[row, exp.START] + table.integers[row, exp.LENGTH] - 1)
        for offset in range(int(table.duration[row])):
            frame = start + offset
            losses = [int(stocks[frame + 1, p] < stocks[frame, p]) for p in (0, 1)]
            final = [int(losses[p] and stocks[frame + 1, p] == 0) for p in (0, 1)]
            base = 120 * (losses[1] - losses[0]) + 50 * (final[1] - final[0])
            table.rewards[row, offset] = base if side == 0 else -base
    return table, ids, stocks, percents


@pytest.mark.parametrize("weight", [0.0, 1.0, 0.25])
def test_shaping_matches_removed_c_formula_and_keeps_stored_rewards(weight: float) -> None:
    table, ids, stocks, percents = reward_table()
    before = table.rewards.clone()
    actual = exp.training_frame_rewards(table, ids, weight)
    expected = torch.zeros_like(actual)
    for row in ids.tolist():
        side = int(table.integers[row, exp.SIDE])
        start = int(table.integers[row, exp.START] + table.integers[row, exp.LENGTH] - 1)
        for offset in range(int(table.duration[row])):
            frame = start + offset
            loss = [int(stocks[frame + 1, p] < stocks[frame, p]) for p in (0, 1)]
            final = [int(loss[p] and stocks[frame + 1, p] == 0) for p in (0, 1)]
            damage = []
            for p in (0, 1):
                increase = float(percents[frame + 1, p]) - float(percents[frame, p])
                damage.append(increase if increase > 0 and math.isfinite(increase) else 0.0)
            base = 120 * (loss[1] - loss[0]) + 50 * (final[1] - final[0])
            reward = base + weight * (damage[1] - damage[0])
            expected[row, offset] = reward if side == 0 else -reward
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(table.rewards, before, atol=0, rtol=0)


def test_stock_check_rejects_swapped_student_side_and_corrupted_reward() -> None:
    table, ids, _, _ = reward_table()
    table.integers[ids, exp.SIDE] = 1 - table.integers[ids, exp.SIDE]
    with pytest.raises(ValueError, match="stock reward.*student side"):
        exp.training_frame_rewards(table, ids, 1)
    run = trainer(groups=1)
    rollout = run.engine.collect(0)
    rollout.table.rewards[rollout.ids[0], 0] += 1
    with pytest.raises(ValueError, match="stock reward"):
        run.prepare(rollout)


def test_missing_stock_sentinel_does_not_count_as_a_loss() -> None:
    table, ids, _, _ = reward_table()
    columns = table.store.records.view(torch.int32)
    columns[2, 0, exp.NATIVE_COLUMN_LAYOUT["p2_stock"][0]] = torch.iinfo(torch.int32).max
    losses, _, _ = exp.frame_components(table, ids[:1])
    assert losses[0, 0, 1] == 0


@pytest.mark.parametrize("lanes", [1, 2, 3, 4, 5, 8, 16, 64])
def test_both_lane_layouts_match_their_historical_counts(lanes: int) -> None:
    heavy = exp.opponent_slots(lanes)
    light = exp.self_play_heavy_opponent_slots(lanes)
    if lanes < 4:
        assert heavy == light == (None,) * lanes
        return
    heavy_count = max(1, min((lanes - 1) // 4, round(lanes * 3 / 16)))
    light_counts = [max(1, round(lanes / 16)), *([max(1, round(lanes / 32))] * 3)]
    assert [heavy.count(slot) for slot in range(4)] == [heavy_count] * 4
    assert [light.count(slot) for slot in range(4)] == light_counts
    if lanes == 64:
        assert heavy.count(None) == 16 and light.count(None) == 54


def test_engine_uses_configured_layout_and_rejects_trainer_mismatch() -> None:
    run = fixed_trainer(initialize=False)
    base = run.engine
    engine = exp.RolloutEngine(
        base.groups, base.policy, base.sampler, {0: base.anchors[0]}, lane_layout="self_play_heavy"
    )
    assert tuple(engine.lane_roles.values()) == exp.self_play_heavy_opponent_slots(16)
    with pytest.raises(ValueError, match="lane layouts"):
        exp.Trainer(engine, run.config)
    exp.Trainer(engine, dataclasses.replace(run.config, lane_layout="self_play_heavy"))


class NoLeagueDrawRng(exp.CounterRng):
    def choice(self, weights: Sequence[float], domain: str, stream: int, episode: int, frame: int) -> int:
        assert domain != "league", "latest selection must not draw PFSP entries"
        return super().choice(weights, domain, stream, episode, frame)


@pytest.mark.parametrize("share", [0.0, 1.0])
def test_latest_promotes_oldest_slot_independent_of_stock_share(share: float) -> None:
    run = fixed_trainer()
    run.config = dataclasses.replace(run.config, league_selection="latest", max_league_size=5)
    run.engine.sampler.rng = NoLeagueDrawRng(65)
    expected = [(1, 0, 0), (1, 2, 0), (1, 2, 3), (4, 2, 3), (4, 5, 3), (4, 5, 6), (7, 5, 6)]
    for update, slots in enumerate(expected, start=1):
        run.update = update * 256
        run.league_stock_shares.update({entry.entry_id: share for entry in run.league})
        assert run.rotate_league() == ((update - 1) % 3 + 1,)
        assert tuple(run.slot_entries[slot] for slot in (1, 2, 3)) == slots
        assert run.slot_updates == {slot: entry_id * 256 for slot, entry_id in enumerate(slots, start=1)}
        for slot, entry_id in run.slot_entries.items():
            weights = next(entry.weights for entry in run.league if entry.entry_id == entry_id)
            assert exp._state_hash(run.engine.anchors[slot].model.state_dict()) == exp._state_hash(weights)
    assert all(
        state.end_clock is not None
        for match, state in run.engine.matches.items()
        if run.engine.lane_roles[match] in (1, 2, 3)
    )
    assert len(run.league) == 5 and run.league_metrics()["league/size"] == 5


def test_smash_states_match_installed_enum_and_include_all_angles() -> None:
    names = ("FSMASH_HIGH", "FSMASH_MID_HIGH", "FSMASH_MID", "FSMASH_MID_LOW", "FSMASH_LOW", "UPSMASH", "DOWNSMASH")
    assert tuple(melee.Action[name].value for name in names) == exp.SMASH_ACTION_STATES
    assert not any("SMASH" in action.name and "CHARGE" in action.name for action in melee.Action)


def test_histogram_uses_post_step_states_student_side_slot_zero_and_duration() -> None:
    table, ids, _, _ = reward_table()
    columns = table.store.records.view(torch.int32)
    columns[:, 0, exp.NATIVE_COLUMN_LAYOUT["p1_action"][0]] = torch.arange(8) + 58
    columns[:, 0, exp.NATIVE_COLUMN_LAYOUT["p2_action"][0]] = torch.arange(8) + 10
    table.opponent_anchor[ids] = torch.tensor([0, 0, 1, -1])
    actual = exp.action_state_histogram(table, ids)
    expected = torch.zeros(exp.ACTION_STATE_COUNT, dtype=torch.long)
    expected[61:65] = 1
    expected[13:17] = 1
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_collapse_statistics_use_nats_and_explicit_zero_baselines() -> None:
    baseline = torch.zeros(exp.ACTION_STATE_COUNT, dtype=torch.long)
    baseline[0], baseline[58] = 3, 1
    current = baseline.clone()
    current[0], current[58] = 1, 3
    metrics = exp.collapse_statistics(current, baseline)
    assert metrics["collapse/smash_fraction"] == 0.75
    assert metrics["collapse/smash_ratio"] == 3
    expected = 0.25 * math.log(0.5) + 0.75 * math.log(1.5)
    assert metrics["collapse/action_js"] == pytest.approx(expected)
    assert exp.collapse_statistics(baseline, baseline)["collapse/action_js"] == 0
    baseline[0], baseline[58] = 1, 0
    assert exp.collapse_statistics(current, baseline)["collapse/smash_ratio"] == math.inf
    empty = exp.collapse_statistics(current, torch.zeros_like(baseline))
    assert empty["collapse/baseline_frames"] == 0 and math.isnan(empty["collapse/action_js"])


def test_warmup_histogram_resume_matches_next_batch_and_update(tmp_path: Path) -> None:
    run = fixed_trainer(length=1000000, warmup=2)
    run.run_update()
    histogram = run.warmup_action_histogram.clone()
    assert histogram.sum() > 0
    manifest = exp.save_checkpoint(run, tmp_path, PROVENANCE)
    expected, _ = run.run_update()
    expected_hash = exp._batch_hash(expected)
    expected_histogram = run.warmup_action_histogram.clone()
    _, metrics = run.run_update()
    expected_model = exp._state_hash(run.policy.model.state_dict())
    resumed = fixed_trainer(length=1000000, warmup=2)
    exp.restore_checkpoint(resumed, manifest, PROVENANCE)
    torch.testing.assert_close(resumed.warmup_action_histogram, histogram)
    actual, _ = resumed.run_update()
    assert exp._batch_hash(actual) == expected_hash
    torch.testing.assert_close(resumed.warmup_action_histogram, expected_histogram)
    _, resumed_metrics = resumed.run_update()
    assert resumed_metrics["collapse/action_js"] == metrics["collapse/action_js"] == 0
    assert resumed_metrics["collapse/baseline_frames"] == int(expected_histogram.sum())
    assert exp._state_hash(resumed.policy.model.state_dict()) == expected_model
    torch.testing.assert_close(resumed.warmup_action_histogram, expected_histogram)


def test_prepare_uses_discount_and_damage_and_reports_lane_explained_variance() -> None:
    run = fixed_trainer(length=1000000)
    run.config = dataclasses.replace(run.config, gamma=0.8, damage_reward=0.5)
    rollout = run.engine.collect(0)
    table, ids = rollout.table, rollout.ids
    columns = table.store.records.view(torch.int32)
    percent = torch.arange(len(columns), dtype=torch.float32)[:, None].expand(-1, columns.shape[1]).contiguous()
    columns[..., exp.NATIVE_COLUMN_LAYOUT["p1_percent"][0]] = percent.view(torch.int32)
    batch = run.prepare(rollout)
    frame_rewards = exp.training_frame_rewards(table, ids, 0.5)
    interval = exp.regularized_reward(batch.decisions, frame_rewards, batch.kl, gamma=0.8)
    assert batch.metrics["return/interval_reward"] == pytest.approx(float(frame_rewards.sum(-1).mean()))
    opponents = table.opponent_anchor[ids]
    for label, mask in (("self_play", opponents < 0), ("anchor", opponents >= 0)):
        expected = exp.explained_variance(batch.values[mask], batch.targets[mask])
        assert batch.metrics[f"critic/explained_variance_{label}"] == pytest.approx(expected)
    # With lambda 1 each non-boundary decision equals its reward plus the next target.
    following = table.following[ids]
    lookup = {int(row): index for index, row in enumerate(ids)}
    for index, row in enumerate(ids.tolist()):
        if table.boundary[row] or table.terminal[row]:
            continue
        next_index = lookup[int(following[index])]
        expected = interval[index] + 0.8 ** int(table.duration[row]) * batch.targets[next_index]
        torch.testing.assert_close(batch.targets[index], expected, rtol=1e-5, atol=1e-4)

    kl = torch.full_like(batch.kl, 0.1)
    run.completed_metrics(rollout, kl)
    frames = torch.arange(4)
    for stream in table.integers[ids, exp.STREAM].unique().tolist():
        mask = table.integers[ids, exp.STREAM] == stream
        age = table.age[ids][mask]
        expected_return = (frame_rewards[mask] * 0.8 ** (age[:, None] + frames)).sum()
        costs = (kl[mask] * batch.decisions.executed_factors_JF[mask]).sum(-1) * table.coefficients[ids][mask]
        expected_cost = (costs * 0.8**age).sum()
        assert run.episode_metrics[(stream, 0)][0] == pytest.approx(float(expected_return), abs=1e-5)
        assert run.episode_metrics[(stream, 0)][3] == pytest.approx(float(expected_cost), abs=1e-5)


def test_cli_prior_overrides_reach_sampler_and_resolved_file(tmp_path: Path) -> None:
    config = tyro.cli(
        exp.NativeCli,
        args=[
            "--checkpoint",
            "actor.pt",
            "--sampler",
            "sampler.json",
            "--output",
            str(tmp_path),
            "--data-dir",
            "data",
            "--cpus",
            str(min(os.sched_getaffinity(0))),
            "--updates",
            "1",
            "--percentile-alpha",
            "5",
            "--percentile-beta",
            "1.3",
            "--training.gamma",
            "0.99855",
            "--training.damage-reward",
            "1",
            "--training.lane-layout",
            "self_play_heavy",
            "--training.league-selection",
            "latest",
        ],
    )
    assert config.training.gamma == 0.99855 and config.training.damage_reward == 1
    assert config.training.lane_layout == "self_play_heavy" and config.training.league_selection == "latest"
    sampling = exp.read_native_sampler(
        ROOT / "experiments/o65/pilot-sampler.json",
        percentile_alpha=config.percentile_alpha,
        percentile_beta=config.percentile_beta,
    )
    assert (sampling.percentile_alpha, sampling.percentile_beta) == (5, 1.3)
    run = fixed_trainer(warmup=1)
    run.engine.sampler.config = dataclasses.replace(run.engine.sampler.config, percentile_alpha=5, percentile_beta=1.3)
    run.config = dataclasses.replace(run.config, gamma=0.99855, damage_reward=1)
    exp.run_training(run, 2, tmp_path, PROVENANCE)
    resolved = json.loads((tmp_path / "resolved.json").read_text())
    assert resolved["returns"]["gamma_per_frame"] == 0.99855
    assert "damage weight 1" in resolved["returns"]["reward"]
    assert (resolved["sampler"]["percentile_alpha"], resolved["sampler"]["percentile_beta"]) == (5, 1.3)
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert "collapse/action_js" not in rows[0]
    assert "collapse/action_js" in rows[1]
    assert sum(resolved["warmup_action_histogram"]) == rows[1]["collapse/baseline_frames"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("gamma", math.nan),
        ("gamma", -0.1),
        ("gamma", 1.01),
        ("damage_reward", -1),
        ("damage_reward", math.inf),
        ("lane_layout", "invalid"),
        ("league_selection", "invalid"),
    ],
)
def test_ablation_config_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        exp.TrainConfig(**{field: value})


@pytest.mark.parametrize(
    "invalid", [None, torch.zeros(4, dtype=torch.int64), torch.full((exp.ACTION_STATE_COUNT,), -1, dtype=torch.int64)]
)
def test_resume_rejects_invalid_histogram_before_model_mutation(tmp_path: Path, invalid: object) -> None:
    run = fixed_trainer(warmup=2)
    run.run_update()
    snapshot = exp.capture_checkpoint(run, PROVENANCE)
    state = {**snapshot.state, "warmup_action_histogram": invalid}
    manifest = exp.write_checkpoint(dataclasses.replace(snapshot, state=state), tmp_path)
    restored = fixed_trainer(warmup=2)
    before = exp._state_hash(restored.policy.model.state_dict())
    with pytest.raises(ValueError, match="warmup action histogram"):
        exp.restore_checkpoint(restored, manifest, PROVENANCE)
    assert exp._state_hash(restored.policy.model.state_dict()) == before


def test_default_configuration_and_legacy_checkpoint_remain_compatible(tmp_path: Path) -> None:
    run = fixed_trainer()
    run.run_update()
    identity = exp._configuration_identity(run)
    assert all(name not in identity["train"] for name in ("gamma", "damage_reward", "lane_layout", "league_selection"))
    snapshot = exp.capture_checkpoint(run, PROVENANCE)
    state = dict(snapshot.state)
    del state["warmup_action_histogram"]
    manifest = exp.write_checkpoint(dataclasses.replace(snapshot, state=state), tmp_path)
    restored = fixed_trainer()
    exp.restore_checkpoint(restored, manifest, PROVENANCE)
    assert restored.warmup_action_histogram.sum() == 0
    _, metrics = restored.run_update()
    assert metrics["collapse/baseline_frames"] == 0
    assert math.isnan(metrics["collapse/action_js"])
