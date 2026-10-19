"""Dense decision selection and stored behavior contracts."""

import dataclasses
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from test_065_loop import exp
from test_065_loop import trainer


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("windows_per_step", [1, 3, 8, 100])
def test_device_selection_places_each_window_once_and_separates_matches(device: str, windows_per_step: int) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    store = exp.FrameStore(1, 1, torch.device(device), exp.native.OBSERVATION_FIELDS)
    metadata = []
    for match, windows in enumerate((4, 1, 3, 2, 4, 1)):
        for window in range(windows):
            for clock in range(0, 32, 2):
                row = [0] * 18
                row[exp.STREAM] = 2 * match + window % 2
                row[exp.BLOCK] = window // 2 * 128
                row[exp.EPISODE] = window // 2
                row[exp.CLOCK] = window // 2 * 128 + clock
                metadata.append(row)
    table = exp.DecisionTable(len(metadata), store)
    table.count = len(metadata)
    table.meta[:] = metadata
    table.integers.copy_(torch.tensor(metadata, device=device))
    rows = exp.DecisionRecords(table.rows(np.arange(table.count)))
    eligible = torch.arange(table.count, device=device) % 5 != 0
    eligible[16:32] = False
    selected, steps = exp.select_decisions(rows, eligible, exp.CounterRng(65), 7, 5, windows_per_step)
    assert not bool((selected & ~eligible).any())
    placed = torch.cat(steps)
    assert len(placed) == int(selected.sum())
    assert len(placed.unique()) == len(placed)
    counts = Counter()
    ownership = {}
    for step_index, indices in enumerate(steps):
        windows = set()
        matches = set()
        for index in indices.cpu().tolist():
            row = metadata[index]
            key = row[exp.STREAM], row[exp.BLOCK]
            windows.add(key)
            matches.add(row[exp.STREAM] // 2)
            counts[key] += 1
            assert ownership.setdefault(key, step_index) == step_index
        assert len(windows) <= windows_per_step
        assert len(matches) == len(windows)
    assert all(value == 5 for value in counts.values())
    again, repeated = exp.select_decisions(rows, eligible, exp.CounterRng(65), 7, 5, windows_per_step)
    torch.testing.assert_close(selected, again, rtol=0, atol=0)
    for a, b in zip(steps, repeated, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_prepare_uses_behavior_outputs_without_student_scoring() -> None:
    run = trainer(groups=1, anchor_rate=0)
    rollout = run.engine.collect(0)
    original = run.policy.evaluate
    calls = []

    def bootstrap_only(states: Sequence[object], actions: torch.Tensor) -> object:
        calls.append(len(states))
        return original(states, actions)

    run.policy.evaluate = bootstrap_only
    batch = run.prepare(rollout)
    assert sum(calls) < len(rollout.records)
    torch.testing.assert_close(batch.old.log_probs, batch.decisions.behavior_log_probs_JF, rtol=0, atol=0)
    assert batch.metrics["numerics/mu_old_gap_max"] == 0


def test_tensor_counter_domains_keys_and_cpu_cuda_parity() -> None:
    streams = torch.arange(2000)
    episodes = streams // 7
    frames = streams * 128
    rng = exp.CounterRng(65)
    draws = rng.tensor_uniform(1, streams, episodes, frames)
    assert bool(((draws >= 0) & (draws < 1)).all())
    assert len(draws.unique()) > 1990
    assert abs(float(draws.mean()) - 0.5) < 0.02
    assert not torch.equal(draws, rng.tensor_uniform(2, streams, episodes, frames))
    assert not torch.equal(draws, exp.CounterRng(66).tensor_uniform(1, streams, episodes, frames))
    assert not torch.equal(draws, exp.CounterRng(65 + 2**32).tensor_uniform(1, streams, episodes, frames))
    assert not torch.equal(draws, rng.tensor_uniform(1, streams + 2**32, episodes, frames))
    assert not torch.equal(draws, rng.tensor_uniform(1, streams, episodes, frames + 2**32))
    at = torch.tensor([17, 0, 83])
    torch.testing.assert_close(draws[at], rng.tensor_uniform(1, streams[at], episodes[at], frames[at]), rtol=0, atol=0)
    if torch.cuda.is_available():
        gpu = rng.tensor_uniform(1, streams.cuda(), episodes.cuda(), frames.cuda())
        torch.testing.assert_close(draws, gpu.cpu(), rtol=0, atol=0)


def test_ppo_uses_detached_behavior_logs_as_old_policy() -> None:
    run = trainer(groups=1, anchor_rate=0)
    batch = run.prepare(run.engine.collect(0))
    behavior = batch.decisions.behavior_log_probs_JF.clone().requires_grad_()
    decisions = dataclasses.replace(batch.decisions, behavior_log_probs_JF=behavior)
    new = (behavior.detach() + 0.001).requires_grad_()
    selected = decisions.loss_mask_J
    stats = exp.advantage_statistics(exp.advantage_moments(batch.advantages, selected))
    result = exp.ppo_loss(decisions, new, batch.advantages, selected, stats, clip_epsilon=0.2)
    expected = (decisions.executed_factors_JF.sum(-1) * 0.001).exp()
    torch.testing.assert_close(result.joint_ratio_J, expected)
    result.loss.backward()
    assert behavior.grad is None
    assert new.grad is not None


@pytest.mark.parametrize("timeout", [False, True])
def test_dense_slots_keep_cadence_and_episode_execution_masks(timeout: bool) -> None:
    run = trainer(groups=1, length=13, timeout=timeout, delay=8, anchor_rate=0)
    rollout = run.engine.collect(0)
    table = rollout.records.states.table
    players = 4
    valid = table.valid[: 128 * players].reshape(128, players)
    meta = table.integers[: 128 * players].reshape(128, players, 18)
    expected = meta[..., exp.CLOCK] % meta[..., exp.REPLAN] == 0
    expected &= (meta[..., exp.CLOCK] % 128) < 13
    torch.testing.assert_close(valid, expected)
    for record in rollout.records:
        assert record.frames_remaining == 13 - record.state.clock % 128
        assert len(record.rewards) == min(record.state.player.timing.replan, record.frames_remaining)
    batch = run.prepare(rollout)
    expected_frames = (
        batch.decisions.duration_J * 0 + 13 - batch.decisions.decision_frame_J % 128 - batch.decisions.delay_J
    ).clamp_min(0)
    expected_frames = torch.minimum(expected_frames, batch.decisions.replan_J)
    torch.testing.assert_close(batch.decisions.executed_factors_JF.sum(-1), 4 * expected_frames)


def test_final_timeout_bootstrap_precedes_reset() -> None:
    run = trainer(groups=1, length=256, timeout=True, delay=8, anchor_rate=0)
    rollout = run.engine.collect(0)
    for stream in range(4):
        record = next(row for row in reversed(rollout.records) if row.state.stream == stream)
        assert record.boundary and not record.terminal
        assert record.next_state.episode.number == 0
        assert record.next_state.frames[-1].columns["state"] == 256
        assert run.engine.matches[stream // 2].episode.number == 1
        assert run.engine.matches[stream // 2].history[-1].columns["state"] == 0


def test_dense_gae_matches_decision_grid_with_mixed_cadences_and_boundaries() -> None:
    torch.manual_seed(65)
    ticks, players = 8, 3
    decided = torch.ones(ticks, players, dtype=torch.bool)
    decided[1::2, 0] = False
    decided[2:4, 1] = False
    duration = torch.where(decided, 2, 0)
    duration[::2, 0] = 4
    duration[1, 1] = 1
    rewards = torch.randn(ticks, players, dtype=torch.float64)
    values = torch.randn_like(rewards)
    following = torch.randn_like(rewards)
    bootstrap = torch.ones_like(decided)
    bootstrap[1, 1] = False
    trace = decided.clone()
    trace[1, 1] = False
    trace[-1] = False
    trace[-2, 0] = False
    result = exp.dense_advantages(rewards, values, following, duration, bootstrap, trace, decided)
    for player in range(players):
        at = torch.nonzero(decided[:, player]).flatten()
        reference = exp.generalized_advantage(
            rewards[at, player],
            values[at, player],
            following[at, player],
            duration[at, player],
            bootstrap[at, player],
            trace[at, player],
        )
        torch.testing.assert_close(result.advantages[at, player], reference.advantages, rtol=0, atol=0)
        torch.testing.assert_close(result.value_targets[at, player], reference.value_targets, rtol=0, atol=0)


def test_two_frame_queue_does_not_overwrite_actions_with_padding() -> None:
    from test_065_loop import transformer_trainer

    base = transformer_trainer("cpu")
    base.engine.groups[0][0].length = 1000000
    schedule = dataclasses.replace(base.engine.sampler.schedule, configurations=(exp.DelayReplan(0, 2),))
    sampling = exp.EpisodeSampler(base.engine.sampler.config, schedule, exp.CounterRng(65))
    policy = exp.TransformerPolicy(base.policy.model, schedule, base.policy.statistics)
    engine = exp.RolloutEngine(base.engine.groups, policy, sampling, {0: policy.frozen_copy()})
    engine.initialize()
    rollout = engine.collect(0)
    assert engine.rings[0].capacity == 2
    shard = engine.groups[0][0]
    for record in rollout.records:
        expected = policy.device_wire_actions(record.actions[:2]).numpy()
        actual = np.stack(shard.trace[record.state.clock : record.state.clock + 2])[:, 0, record.state.side]
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("field", ["identity", "conditioning", "configuration"])
def test_reset_rejects_episode_settings_before_mutating_simulator(field: str) -> None:
    from test_065_loop import transformer_trainer

    run = transformer_trainer("cpu")
    shard = run.engine.groups[0][0]
    before = shard.save()
    if field == "identity":
        identity = run.policy.model.cfg.player_vocab_size
        run.engine.sampler.config = dataclasses.replace(
            run.engine.sampler.config,
            identities=(exp.IdentityCount(identity, 1, 400),),
            deployment_identity=identity,
            masked_rate=0,
            deployment_rate=1,
        )
    elif field == "conditioning":
        run.engine.sampler.config = dataclasses.replace(run.engine.sampler.config, lambda_min=1, lambda_max=1)
        run.engine.sampler.lambda_max = 1
    else:
        run.engine.sampler.schedule = dataclasses.replace(
            run.engine.sampler.schedule, configurations=(exp.DelayReplan(8, 4),)
        )
    with pytest.raises(ValueError, match=field):
        run.engine._reset(shard, 0, 0, 0, 1)
    assert shard.save() == before


def test_shards_step_while_the_other_buffer_samples() -> None:
    import threading

    from test_065_loop import FakeShard
    from test_065_loop import ToyPolicy
    from test_065_loop import sampler

    simulation_started = threading.Event()
    second_buffer_sampled = threading.Event()

    class WaitingShard(FakeShard):
        def step(self, inputs: NDArray[np.float32], active: NDArray[np.bool_]) -> tuple[tuple[Any, ...], ...]:
            if self.ticks[0] == 0:
                simulation_started.set()
                if not second_buffer_sampled.wait(5):
                    raise RuntimeError("second buffer did not sample while the first shard waited")
            return super().step(inputs, active)

    class CheckingPolicy(ToyPolicy):
        def sample(self, states: Any, rng: Any) -> Any:
            if int(states.meta[0, exp.LANE]) == 1 and int(states.meta[0, exp.CLOCK]) == 0:
                if not simulation_started.wait(5):
                    raise RuntimeError("first shard did not start before second-buffer inference")
                second_buffer_sampled.set()
            return super().sample(states, rng)

        def frozen_copy(self) -> Any:
            frozen = ToyPolicy()
            frozen.model.load_state_dict(self.model.state_dict())
            return frozen.frozen_copy()

    policy = CheckingPolicy()
    shards = (WaitingShard((0,)), FakeShard((1,)))
    engine = exp.RolloutEngine((shards,), policy, sampler(anchor_rate=0), {0: policy.frozen_copy()})
    engine.initialize()
    engine.collect(0)
    assert simulation_started.is_set() and second_buffer_sampled.is_set()
    assert len(engine.buffers[0]) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_two_stream_resets_preserve_both_matches_and_resume(tmp_path: Path) -> None:
    from test_065_loop import PROVENANCE
    from test_065_loop import transformer_trainer

    base = transformer_trainer("cuda", development=True, compiled=True)
    shard_type = type(base.engine.groups[0][0])
    shards = (shard_type((0,), length=67, timeout=True), shard_type((1,), length=67, timeout=True))
    base.engine.sampler.config = dataclasses.replace(base.engine.sampler.config, anchor_rate=1)
    engine = exp.RolloutEngine((shards,), base.policy, base.engine.sampler, {0: base.policy.frozen_copy()})
    engine.initialize()
    run = exp.Trainer(engine, base.config)
    buffers = engine.buffers[0]
    assert len(buffers) == 2 and buffers[0].stream != buffers[1].stream
    batch, _ = run.run_update()
    assert batch.rollout.timeouts == 4
    for record in batch.rollout.records:
        assert record.frames_remaining == 67 - record.state.clock % 128
    exp.verify_checkpoint_resume(run, tmp_path, PROVENANCE)
    assert run.update == 1
