"""Deterministic full-loop tests. Simulator state contains no torch objects."""

import copy
import dataclasses
import importlib.util
import io
import json
import math
import os
import random
import struct
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from hal.data.schema import MDS_PER_FRAME_DTYPES
from hal.wire import ACTION_DIM

_SPEC = importlib.util.spec_from_file_location(
    "hal_experiment_065_loop", Path(__file__).parents[1] / "experiments" / "065_selfplay_ppo.py"
)
assert _SPEC is not None and _SPEC.loader is not None
exp = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = exp
_SPEC.loader.exec_module(exp)


class FakeShard:
    def __init__(self, ids: tuple[int, ...], *, length: int = 1000000, timeout: bool = False) -> None:
        self.match_ids = ids
        self.length = length
        self.timeout = timeout
        self.ticks = [0] * len(ids)
        self.episodes = [-1] * len(ids)
        self.seeds = [0] * len(ids)
        self.trace: list[NDArray[np.float32]] = []

    def reset(self, lane: int, episode: Any) -> Any:
        self.ticks[lane] = 0
        self.episodes[lane] = episode.number
        self.seeds[lane] = episode.seed
        return exp.SimFrame({"state": 0.0}, np.zeros((2, ACTION_DIM), dtype=np.float32))

    def step(self, inputs: NDArray[np.float32], active: NDArray[np.bool_]) -> tuple[tuple[Any, ...], ...]:
        rows = []
        for frame in inputs:
            observations = []
            self.trace.append(frame.copy())
            for lane in range(len(self.match_ids)):
                if active[lane]:
                    self.ticks[lane] += 1
                end = self.ticks[lane] >= self.length
                # Independently reward each side's first button.
                rewards = (float(frame[lane, 0, 6]), float(frame[lane, 1, 6]))
                observations.append(
                    exp.SimFrame(
                        {"state": float(self.ticks[lane])},
                        frame[lane].copy(),
                        rewards,
                        end and not self.timeout,
                        end and self.timeout,
                    )
                )
            rows.append(tuple(observations))
            if any(enabled and (f.terminal or f.timeout) for enabled, f in zip(active, observations, strict=True)):
                break
        return tuple(rows)

    def save(self) -> bytes:
        return json.dumps(
            {
                "version": 1,
                "length": self.length,
                "timeout": self.timeout,
                "ticks": self.ticks,
                "episodes": self.episodes,
                "seeds": self.seeds,
            }
        ).encode()

    def restore(self, state: bytes) -> None:
        saved = json.loads(state)
        if (saved["version"], saved["length"], saved["timeout"]) != (1, self.length, self.timeout):
            raise ValueError("fake simulator identity changed")
        self.ticks = saved["ticks"]
        self.episodes = saved["episodes"]
        self.seeds = saved["seeds"]


class ToyModel(torch.nn.Module):
    def __init__(self, device: str) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(4, 2, device=device))
        self.value_head = torch.nn.Linear(1, 1, device=device)
        torch.nn.init.zeros_(self.value_head.weight)
        torch.nn.init.zeros_(self.value_head.bias)
        self.percentile_head = torch.nn.Linear(1, 1, device=device).requires_grad_(False)


class ToyPolicy:
    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = ToyModel(device)

    def actor_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return (self.model.logits,)

    def critic_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return tuple(self.model.value_head.parameters())

    def clear_caches(self) -> None:
        pass

    def frozen_copy(self) -> Any:
        policy = copy.deepcopy(self)
        policy.model.requires_grad_(False)
        return policy

    def _evaluation(self, states: Any, actions: torch.Tensor) -> Any:
        count = len(states)
        # The native storage order is buttons, main stick, C-stick, triggers.
        probabilities = self.model.logits.softmax(-1)[None, None].expand(count, 4, 4, 2)
        log_probs = probabilities.log().gather(-1, actions[..., None]).squeeze(-1)
        for j, state in enumerate(states):
            log_probs = log_probs.clone()
            log_probs[j, state.player.timing.replan :] = 0
        trunk = torch.ones(count, 1, device=self.device)
        zeros = torch.zeros(count, 3, device=self.device)
        features = exp.CriticFeatures(
            trunk,
            trunk,
            zeros,
            zeros,
            torch.zeros(count, 24, 4, dtype=torch.long, device=self.device),
            torch.zeros(count, dtype=torch.long, device=self.device),
        )
        return exp.PolicyEvaluation(
            actions,
            log_probs[..., exp.DECODE_GROUP_INDICES].flatten(1),
            tuple(probabilities[:, :, g] for g in exp.DECODE_GROUP_INDICES),
            features,
        )

    @torch.no_grad()
    def sample(self, states: Any, rng: Any) -> Any:
        probabilities = self.model.logits.softmax(-1)
        draws = torch.tensor(
            [
                [
                    [
                        rng.uniform(
                            "action",
                            state.stream,
                            state.episode.number,
                            state.clock,
                            state.player.timing.delay + offset + 1,
                            factor,
                        )
                        for factor in range(4)
                    ]
                    for offset in range(4)
                ]
                for state in states
            ],
            device=self.device,
        )
        actions = (draws >= probabilities[:, 0]).long()
        return self._evaluation(states, actions)

    def evaluate(self, states: Any, actions: torch.Tensor) -> Any:
        return self._evaluation(states, actions)

    def values(self, features: Any) -> torch.Tensor:
        return self.model.value_head(features.trunk.detach()).squeeze(-1)

    def wire_actions(self, indices: torch.Tensor) -> NDArray[np.float32]:
        actions = np.zeros((*indices.shape[:-1], ACTION_DIM), dtype=np.float32)
        actions[..., 6] = indices[..., 0].cpu().numpy()
        return actions


def sampler(*, delay: int = 0, mixed: bool = True, anchor_rate: float = 0.15) -> Any:
    schedule = exp.ScheduleConfig(
        (exp.DelayReplan(delay, 2), exp.DelayReplan(delay, 4)) if mixed else (exp.DelayReplan(delay, 2),),
        tuple(range(1, 25)),
    )
    cfg = exp.SamplerConfig(
        (exp.IdentityCount(1, 1, 400), exp.IdentityCount(2, 1, 100), exp.IdentityCount(3, 1, 900)),
        (exp.Matchup((1, 1), 1.0),),
        (2,),
        data_floor=100,
        anchor_rate=anchor_rate,
        deployment_identity=1,
        lambda_max=0.02,
    )
    return exp.EpisodeSampler(cfg, schedule, exp.CounterRng(65))


def trainer(
    device: str = "cpu",
    *,
    groups: int = 2,
    length: int = 1000000,
    timeout: bool = False,
    delay: int = 0,
    warmup: int = 0,
    anchor_rate: float = 0.15,
    development: bool = False,
) -> Any:
    torch.manual_seed(19)
    policy = ToyPolicy(device)
    shards = tuple(
        (FakeShard((2 * g,), length=length, timeout=timeout), FakeShard((2 * g + 1,), length=length, timeout=timeout))
        for g in range(groups)
    )
    sampling = sampler(delay=delay, anchor_rate=anchor_rate)
    if development:
        sampling.schedule = exp.development_config((*range(1, 13), 16, 20, 24, 28))
    engine = exp.RolloutEngine(shards, policy, sampling, {0: policy.frozen_copy()})
    engine.initialize()
    cfg = exp.TrainConfig(
        actor_optimizer="adam",
        actor_lr=0.03,
        critic_lr=0.01,
        critic_epochs=2,
        critic_warmup_updates=warmup,
        warmup_steps=0,
        windows_per_step=2,
        snapshot_interval=2,
        snapshot_capacity=2,
    )
    return exp.Trainer(engine, cfg)


PROVENANCE = {
    key: "a" * (40 if key == "git_sha" else 64)
    for key in ("git_sha", "checkpoint_sha256", "data_manifest_sha256", "simulator_sha256", "adapter_sha256")
}


def assert_batch_equal(left: Any, right: Any) -> None:
    for field in dataclasses.fields(left.decisions):
        torch.testing.assert_close(
            getattr(left.decisions, field.name), getattr(right.decisions, field.name), atol=0, rtol=0
        )
    for name in ("advantages", "targets", "selected", "values"):
        torch.testing.assert_close(getattr(left, name), getattr(right, name), atol=0, rtol=0)
    assert left.steps == right.steps
    for a, b in zip(left.rollout.records, right.rollout.records, strict=True):
        assert (a.state.stream, a.state.episode, a.state.clock, a.rewards) == (
            b.state.stream,
            b.state.episode,
            b.state.clock,
            b.rewards,
        )
        torch.testing.assert_close(a.state.prefix, b.state.prefix, atol=0, rtol=0)
        torch.testing.assert_close(a.state.opponent_queue, b.state.opponent_queue, atol=0, rtol=0)
        assert a.state.frames[-1].columns.keys() == b.state.frames[-1].columns.keys()
        for name in a.state.frames[-1].columns:
            np.testing.assert_array_equal(a.state.frames[-1].columns[name], b.state.frames[-1].columns[name])


def test_counter_rng_domains_and_rebatching() -> None:
    rng = exp.CounterRng(65)
    keys = [
        (stream, episode, frame, offset, factor)
        for stream in range(2)
        for episode in range(2)
        for frame in range(2)
        for offset in range(2)
        for factor in range(4)
    ]
    values = {key: rng.uniform("action", *key) for key in keys}
    assert values == {key: rng.uniform("action", *key) for key in reversed(keys)}
    assert len(set(values.values())) == len(values)
    assert all(0 <= value < 1 for value in values.values())
    assert all(value != rng.uniform("reset", *key) for key, value in values.items())
    with pytest.raises(ValueError, match="sampling weights"):
        rng.choice((0.0, 0.0), "bad", 0, 0, 0)


def test_sampler_prior_floor_identity_rates_and_anchors() -> None:
    sampling = sampler()
    identities: dict[int, int] = {}
    anchors = 0
    lambdas = []
    for episode in range(3000):
        match = sampling.sample(7, episode, (0, 1, 2))
        assert sum(p.anchor is not None for p in match.players) <= 1
        anchors += any(p.anchor is not None for p in match.players)
        for player in match.players:
            identities[player.identity] = identities.get(player.identity, 0) + 1
            assert player.timing in sampling.schedule.configurations
            lambdas.append(player.log_lambda)
    assert 0.08 < identities[0] / 6000 < 0.12
    assert 0.20 < identities[1] / 6000 < 0.25
    assert 2.6 < identities[3] / identities[2] < 3.4  # sqrt(900/100)
    assert 0.12 < anchors / 3000 < 0.18
    assert abs(sum(lambdas) / len(lambdas) - (math.log(0.01) + math.log(0.02)) / 2) < 0.01
    with pytest.raises(ValueError, match="deployment identity"):
        dataclasses.replace(sampling.config, deployment_identity=9)
    with pytest.raises(ValueError, match="ordinary identities"):
        dataclasses.replace(sampling.config, data_floor=1000)


def test_stock_count_is_sampled_once_per_match() -> None:
    sampling = sampler()
    stocks = set()
    for match in range(10):
        for episode in range(20):
            players = sampling.sample(match, episode, (0,)).players
            assert players[0].stocks == players[1].stocks
            stocks.add(players[0].stocks)
    assert stocks == {1, 2, 3, 4}


@pytest.mark.parametrize("length", [13, 128, 129, 259])
@pytest.mark.parametrize("timeout", [False, True])
def test_finished_matches_wait_without_records_or_steps(length: int, timeout: bool) -> None:
    run = trainer(groups=1, length=length, timeout=timeout, anchor_rate=0)
    records = []
    frames = idle = 0
    for update in range(3):
        rollout = run.engine.collect(update)
        records.extend(rollout.records)
        frames += rollout.frames
        idle += rollout.idle_frames
    assert frames + idle == 2 * 3 * 256
    period = ((length + 127) // 128) * 128
    for record in records:
        start = record.state.episode.number * period
        assert start <= record.state.clock < start + length
        assert record.state.frames[-1].columns["state"] == record.state.clock - start
        assert record.state.window[2] - start == ((record.state.clock - start) // 128) * 128
    for shard in run.engine.groups[0]:
        # Idle lanes never enter a native step, including the odd terminal frame.
        assert len(shard.trace) == frames // 2
    if length == 13:
        assert frames == 2 * 6 * 13
        assert idle == 2 * 6 * 115


def test_idle_lane_does_not_stop_its_live_shard_peer() -> None:
    policy = ToyPolicy()
    shard = FakeShard((0, 1), length=13)
    engine = exp.RolloutEngine(((shard,),), policy, sampler(anchor_rate=0), {0: policy.frozen_copy()})
    engine.initialize()
    shard.ticks[1] = -120
    rollout = engine.collect(0)
    assert sorted(result.duration for result in rollout.results) == [13, 13, 133]
    assert rollout.frames == 2 * 13 + 133
    assert rollout.idle_frames == 512 - rollout.frames
    assert not any(r.state.match == 0 and 13 <= r.state.clock < 128 for r in rollout.records)
    assert any(r.state.match == 1 and r.state.clock == 128 for r in rollout.records)


@torch.inference_mode()
def test_training_and_deployment_blocks_share_episode_frames_after_reset() -> None:
    from hal.inference.kv_cache import DualBlockRestartCache

    run = transformer_trainer("cpu")
    run.engine.groups[0][0].length = 301
    records = []
    for update in range(4):
        records.extend(run.engine.collect(update).records)
    by_episode: dict[int, dict[int, list[Any]]] = {}
    for record in records:
        episode_frames = by_episode.setdefault(record.state.episode.number, {})
        start = record.state.episode.number * 384
        episode_frames.setdefault(record.state.clock - start, []).append(record)
    assert len(by_episode) == 3
    cache = DualBlockRestartCache(run.policy.model, 1, torch.device("cpu"))
    for episode_frames in by_episode.values():
        cache.reset()
        for frame in range(max(episode_frames) + 1):
            cache.append(torch.zeros(1, 1, run.policy.model.cfg.d_model))
            for record in episode_frames.get(frame, []):
                assert len(record.state.frames) == int(cache.active.cache_seqlens[0])
                assert record.state.window[2] % 128 == 0


@pytest.mark.parametrize("timeout", [False, True])
def test_odd_frame_boundaries_and_mixed_cadence(timeout: bool) -> None:
    run = trainer(groups=1, length=13, timeout=timeout, anchor_rate=0)
    batch, metrics = run.run_update()
    assert metrics["data/resets"] > 0
    assert metrics["data/timeouts"] == (metrics["data/resets"] if timeout else 0)
    for record in batch.rollout.records:
        assert record.state.clock % record.state.player.timing.replan == 0
        assert 1 <= len(record.rewards) <= record.state.player.timing.replan
        if record.frames_remaining is not None and record.frames_remaining <= record.state.player.timing.replan:
            assert record.boundary
            assert record.terminal == (not timeout)
            if timeout:
                assert record.next_state is not None
                assert record.next_state.frames[-1].timeout
                assert record.next_state.frames[-1].columns["state"] == 13
    assert {r.state.player.timing.replan for r in batch.rollout.records} == {2, 4}
    assert any(len(r.rewards) < r.state.player.timing.replan for r in batch.rollout.records)
    for step in batch.steps:
        windows: dict[tuple[int, ...], int] = {}
        matches: dict[int, set[tuple[int, ...]]] = {}
        for index in step:
            record = batch.rollout.records[index]
            windows[record.state.window[:3]] = windows.get(record.state.window[:3], 0) + int(batch.selected[index])
            matches.setdefault(record.state.match, set()).add(record.state.window[:3])
        assert max(windows.values()) <= 16
        assert all(len(keys) == 1 for keys in matches.values())


def test_delayed_queue_masks_and_rollout_tail() -> None:
    run = trainer(groups=1, length=13, delay=8, anchor_rate=0)
    rollout = run.engine.collect(0)
    # Offsets execute at t+D+1; an ending at t+13 includes offset 13.
    first = next(r for r in rollout.records if r.state.clock == 0)
    assert first.frames_remaining == 13
    assert not first.state.prefix.any()
    assert any(r.frames_remaining is not None and r.frames_remaining <= 8 for r in rollout.records)
    batch = run.prepare(rollout)
    for index, record in enumerate(rollout.records):
        if record.frames_remaining is None:
            assert int(batch.decisions.executed_factors_JF[index].sum()) == 4 * record.state.player.timing.replan


def test_rollout_tail_stays_executed_when_next_rollout_ends_early() -> None:
    run = trainer(groups=1, length=259, delay=8, anchor_rate=0)
    first, _ = run.run_update()
    tails = [i for i, r in enumerate(first.rollout.records) if r.state.clock >= 248]
    saved_mask = first.decisions.executed_factors_JF.clone()
    for index in tails:
        record = first.rollout.records[index]
        assert record.frames_remaining is None
        assert int(saved_mask[index].sum()) == 4 * record.state.player.timing.replan
    second, _ = run.run_update()
    early = [i for i, r in enumerate(second.rollout.records) if r.state.episode.number == 0]
    assert early
    assert not second.decisions.executed_factors_JF[early].any()
    assert all(second.rollout.records[i].frames_remaining <= 3 for i in early)
    torch.testing.assert_close(first.decisions.executed_factors_JF, saved_mask, atol=0, rtol=0)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_exact_next_batch_and_update_after_resume(tmp_path: Path, device: str) -> None:
    run = trainer(device, length=23, timeout=True)
    for _ in range(3):
        run.run_update()
    random.seed(21)
    np.random.seed(21)
    torch.manual_seed(21)
    manifest = exp.save_checkpoint(run, tmp_path, PROVENANCE)
    expected_rng = (random.random(), float(np.random.random()), torch.rand(3))
    expected, _ = run.run_update()
    expected_weights = copy.deepcopy(run.policy.model.state_dict())
    expected_optimizers = (
        copy.deepcopy(run.actor_optimizer.state_dict()),
        copy.deepcopy(run.critic_optimizer.state_dict()),
    )
    resumed = trainer(device, length=23, timeout=True)
    exp.restore_checkpoint(resumed, manifest, PROVENANCE)
    actual_rng = (random.random(), float(np.random.random()), torch.rand(3))
    assert expected_rng[:2] == actual_rng[:2]
    torch.testing.assert_close(expected_rng[2], actual_rng[2], atol=0, rtol=0)
    actual, _ = resumed.run_update()
    assert_batch_equal(expected, actual)
    for name, value in expected_weights.items():
        torch.testing.assert_close(value, resumed.policy.model.state_dict()[name], atol=0, rtol=0)
    assert resumed.actor_steps == run.actor_steps
    assert resumed.scheduler.state_dict() == run.scheduler.state_dict()
    for expected_state, actual_state in zip(
        expected_optimizers, (resumed.actor_optimizer.state_dict(), resumed.critic_optimizer.state_dict()), strict=True
    ):
        assert expected_state["param_groups"] == actual_state["param_groups"]
        for index, parameters in expected_state["state"].items():
            for name, value in parameters.items():
                torch.testing.assert_close(value, actual_state["state"][index][name], atol=0, rtol=0)


def test_checkpoint_rejects_corruption_and_changed_config(tmp_path: Path) -> None:
    run = trainer()
    run.run_update()
    manifest = exp.save_checkpoint(run, tmp_path, PROVENANCE)
    altered = trainer(delay=1)
    with pytest.raises(ValueError, match="configuration mismatch"):
        exp.restore_checkpoint(altered, manifest, PROVENANCE)
    saved = json.loads(manifest.read_text())
    path = tmp_path / saved["shards"][-1]["name"]
    path.write_bytes(path.read_bytes() + b"bad")
    with pytest.raises(ValueError, match="hash mismatch"):
        exp.restore_checkpoint(trainer(), manifest, PROVENANCE)


def test_toy_reward_increases_action_probability() -> None:
    run = trainer(groups=1, anchor_rate=0)
    before = float(run.policy.model.logits[0].softmax(-1)[1].detach())
    for _ in range(8):
        run.run_update()
    after = float(run.policy.model.logits[0].softmax(-1)[1].detach())
    assert after > before + 0.15


def test_critic_only_warmup_does_not_step_actor() -> None:
    run = trainer(groups=1, warmup=100, anchor_rate=0)
    before = copy.deepcopy(run.policy.model.state_dict())
    run.run_update()
    assert not run.actor_optimizer.state
    assert run.actor_steps == 0
    torch.testing.assert_close(before["logits"], run.policy.model.logits, atol=0, rtol=0)
    torch.testing.assert_close(
        before["percentile_head.weight"], run.policy.model.percentile_head.weight, atol=0, rtol=0
    )
    assert not torch.equal(before["value_head.bias"], run.policy.model.value_head.bias)


def test_warmup_skips_reference_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    run = trainer(groups=1, warmup=100, anchor_rate=0)

    def fail_evaluation(states: Any, actions: torch.Tensor) -> Any:
        raise AssertionError("warmup must not evaluate the reference")

    monkeypatch.setattr(run.reference, "evaluate", fail_evaluation)
    batch, metrics = run.run_update()
    assert batch.reference is None
    assert metrics["policy/kl_per_frame"] == 0
    assert all(value == 0 for key, value in metrics.items() if key.startswith("policy/kl_per_frame_lambda"))
    assert run.actor_steps == 0


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_student_and_reference_evaluations_are_identical_before_first_actor_step(device: str) -> None:
    run = transformer_trainer(device, development=True)
    run = exp.Trainer(run.engine, dataclasses.replace(run.config, critic_warmup_updates=100))
    match = run.engine.matches[0]
    players = tuple(
        dataclasses.replace(player, timing=timing)
        for player, timing in zip(match.episode.players, run.engine.sampler.schedule.configurations, strict=True)
    )
    episode = dataclasses.replace(match.episode, players=players)
    states = tuple(
        exp.DecisionState(
            side,
            0,
            side,
            episode,
            128,
            (0, side, 128, 0),
            tuple(match.history) * 129,
            torch.zeros(players[side].timing.delay, 4, dtype=torch.long),
            torch.zeros(players[1 - side].timing.delay, 4, dtype=torch.long),
        )
        for side in range(2)
    )
    actions = torch.zeros(2, 4, 4, dtype=torch.long, device=device)

    def compare() -> None:
        with torch.no_grad():
            student = run.policy.evaluate(states, actions)
            reference = run.reference.evaluate(states, actions)
        torch.testing.assert_close(student.log_probs, reference.log_probs, atol=0, rtol=0)
        for actual, expected in zip(student.distributions, reference.distributions, strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(
            exp.factor_kl(student, reference), torch.zeros_like(student.log_probs), atol=0, rtol=0
        )

    compare()
    run.run_update()
    assert run.actor_steps == 0
    assert not run.actor_optimizer.state
    compare()


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_full_batch_critic_loss_and_gradient_match_shuffled_order(device: str) -> None:
    run = transformer_trainer(device, development=True)
    policy = run.policy
    count = 257
    with torch.no_grad():
        policy.model.critic_residual.residual.down.weight.fill_(0.03)
    features = exp.CriticFeatures(
        torch.randn(count, 16, device=device),
        torch.randn(count, 16, device=device),
        torch.randn(count, 3, device=device),
        torch.randn(count, 3, device=device),
        torch.zeros(count, 14, 4, dtype=torch.long, device=device),
        torch.arange(count, device=device) % 15,
    )
    targets = torch.randn(count, device=device) * 100
    selected = torch.ones(count, dtype=torch.bool, device=device)
    order = sorted(range(count), key=lambda j: run.engine.sampler.rng.uniform("critic_shuffle", j % 2, 0, 0, 0, j))
    at = torch.tensor(order, device=device)
    loss = exp.critic_loss(policy.values(features), targets, selected, value_scale=run.config.value_scale)
    gradients = torch.autograd.grad(loss, policy.critic_parameters())
    shuffled_loss = exp.critic_loss(
        policy.values(exp._take_features(features, at)), targets[at], selected[at], value_scale=run.config.value_scale
    )
    shuffled_gradients = torch.autograd.grad(shuffled_loss, policy.critic_parameters())
    torch.testing.assert_close(loss, shuffled_loss, rtol=1e-6, atol=1e-7)
    for actual, expected in zip(gradients, shuffled_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)


def transformer_trainer(
    device: str, *, development: bool = False, deterministic: bool = False, compiled: bool = False
) -> Any:
    from hal.data.feature_stats import FeatureStats
    from hal.data.feature_stats import consolidate_key
    from hal.inference.column_context import ColumnContextBuilder
    from hal.models.action_sequence import ActionSequenceConfig
    from hal.models.conditioning import ConditioningRanges
    from hal.models.self_play import SelfPlayTransformer
    from hal.representation.features import BASE_ITEMS_PROJECTION
    from hal.representation.features import ITEM_COLUMNS
    from hal.representation.features import feature_kind
    from hal.representation.player_identity import PlayerVocabulary

    torch.manual_seed(129)
    stats = {
        consolidate_key(name): FeatureStats(0, 1, -1, 1)
        for name in BASE_ITEMS_PROJECTION.columns
        if feature_kind(name, ITEM_COLUMNS) == "float" and not name.startswith("ego_button_")
    }
    builder = ColumnContextBuilder(stats, 1)
    template = {source: 0 if kind == "cat" else 0.0 for source, _, kind, _, _, _ in builder.routes}
    sampling = sampler(mixed=True, anchor_rate=0)
    vocabulary = PlayerVocabulary(())
    cfg = ActionSequenceConfig(
        d_model=16,
        n_layers=1,
        n_heads=2,
        temporal_d_model=16,
        temporal_layers=1,
        temporal_heads=2,
        temporal_ff_dim=32,
        group_head_dim=8,
        value_hidden_dim=8,
        return_embed_dim=8,
        player_vocab_size=vocabulary.size,
        player_vocab_sha256=vocabulary.sha256,
    )
    sampling.schedule = (
        exp.development_config(cfg.head_offsets)
        if development
        else exp.ScheduleConfig(sampling.schedule.configurations, cfg.head_offsets)
    )
    ranges = ConditioningRanges((math.log(0.01), math.log(0.02)), (8, 10) if development else (0, 0), (2, 4))
    model = (
        SelfPlayTransformer(cfg, ranges, vocabulary, max_opponent_queue=14 if development else 4, critic_hidden_dim=8)
        .to(device)
        .eval()
    )
    policy = exp.TransformerPolicy(model, sampling.schedule, stats, compiled=compiled)

    class ColumnsShard(FakeShard):
        def reset(self, lane: int, episode: Any) -> Any:
            frame = super().reset(lane, episode)
            return dataclasses.replace(frame, columns=dict(template))

        def step(self, inputs: NDArray[np.float32], active: NDArray[np.bool_]) -> tuple[tuple[Any, ...], ...]:
            rows = super().step(inputs, active)
            return tuple(tuple(dataclasses.replace(frame, columns=dict(template)) for frame in row) for row in rows)

    shard = ColumnsShard((0,), length=67, timeout=True)
    engine = exp.RolloutEngine(((shard,),), policy, sampling, {0: policy.frozen_copy()})
    engine.initialize()
    cfg = exp.TrainConfig(
        deterministic=deterministic,
        actor_optimizer="adam",
        actor_lr=1e-4,
        critic_epochs=1,
        critic_warmup_updates=0,
        warmup_steps=0,
        snapshot_interval=100,
    )
    return exp.Trainer(engine, cfg)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_transformer_fake_simulator_checkpoint_and_kernel_parity(
    tmp_path: Path, device: str, deterministic_execution: None
) -> None:
    run = transformer_trainer(device, deterministic=True)
    first, metrics = run.run_update()
    assert metrics["numerics/mu_old_gap_p99"] < (0.15 if device == "cuda" else 1e-4)
    assert first.old.features.trunk.requires_grad is False
    assert first.old.features.decoder.requires_grad is False
    assert run.policy.model.percentile_head.down.weight.requires_grad is False
    manifest = exp.save_checkpoint(run, tmp_path, PROVENANCE)
    expected, _ = run.run_update()
    weights = copy.deepcopy(run.policy.model.state_dict())
    resumed = transformer_trainer(device, deterministic=True)
    exp.restore_checkpoint(resumed, manifest, PROVENANCE)
    actual, _ = resumed.run_update()
    assert_batch_equal(expected, actual)
    for name, weight in weights.items():
        torch.testing.assert_close(weight, resumed.policy.model.state_dict()[name], rtol=0, atol=0)


def test_anchor_side_never_trains_and_history_survives_rotation() -> None:
    run = trainer(groups=1, anchor_rate=1)
    reference = copy.deepcopy(run.reference.model.state_dict())
    first, _ = run.run_update()
    assert all(record.state.player.anchor is None for record in first.rollout.records)
    second, _ = run.run_update()
    assert all(len(record.state.frames) == 129 for record in second.rollout.records if record.state.clock == 256)
    for name, value in reference.items():
        torch.testing.assert_close(value, run.reference.model.state_dict()[name], rtol=0, atol=0)
    assert run.engine.anchor_candidates == (0, 1)


def test_masked_deployment_rate_is_total_masked_rate() -> None:
    sampling = sampler()
    config = dataclasses.replace(sampling.config, deployment_identity=0)
    sampling = exp.EpisodeSampler(config, sampling.schedule, sampling.rng)
    masked = sum(
        player.identity == 0 for episode in range(3000) for player in sampling.sample(0, episode, (0,)).players
    )
    assert 0.20 < masked / 6000 < 0.25


def test_lambda_calibration_uses_every_timing_and_preserves_actor() -> None:
    run = trainer(groups=1, warmup=1, anchor_rate=0)
    run.engine.sampler.config = dataclasses.replace(run.engine.sampler.config, lambda_max=100)
    run.engine.sampler.lambda_max = 100
    run.config = dataclasses.replace(run.config, plateau_patience=1, warmup_match_frames=0)
    before = run.policy.model.logits.detach().clone()
    batch, _ = run.run_update()
    expected = 3 * max(
        float(batch.advantages[batch.decisions.replan_J == replan].double().std(unbiased=False)) for replan in (2, 4)
    )
    assert run.engine.sampler.lambda_max == pytest.approx(expected)
    assert not run.warming
    assert run.actor_steps == 0
    assert batch.reference is None
    torch.testing.assert_close(before, run.policy.model.logits, rtol=0, atol=0)
    first_actor, metrics = run.run_update()
    assert first_actor.reference is not None
    assert metrics["policy/kl_per_frame"] == 0
    assert run.actor_steps > 0
    rollout = run.engine.collect(run.update)
    following = run.prepare(rollout)
    assert following.reference is not None
    kl = exp.factor_kl(following.old, following.reference)
    assert float(kl.max()) > 0
    executed = following.decisions.executed_factors_JF
    expected_kl = torch.where(executed, kl, 0).sum() / following.decisions.replan_J.sum()
    assert following.metrics["policy/kl_per_frame"] == pytest.approx(float(expected_kl))
    run.warming = True
    unregularized = run.prepare(rollout)
    run.warming = False
    streams: dict[int, list[int]] = {}
    for index, record in enumerate(rollout.records):
        streams.setdefault(record.state.stream, []).append(index)
    zeros = torch.zeros_like(following.values)
    penalty = -following.decisions.kl_coefficient_J * torch.where(executed, kl, 0).sum(-1)
    effect = exp.stream_advantages(
        penalty,
        zeros,
        zeros,
        following.decisions.duration_J,
        following.decisions.bootstrap_mask_J,
        following.decisions.trace_mask_J,
        list(streams.values()),
    )
    torch.testing.assert_close(following.advantages, unregularized.advantages + effect.advantages)
    torch.testing.assert_close(following.targets, unregularized.targets + effect.advantages)


def test_actor_fit_rejects_missing_reference() -> None:
    run = trainer(groups=1, anchor_rate=0)
    batch = run.prepare(run.engine.collect(0))
    before = copy.deepcopy(run.policy.model.state_dict())
    with pytest.raises(ValueError, match="actor update requires a reference evaluation"):
        run.fit(dataclasses.replace(batch, reference=None))
    assert not run.actor_optimizer.state
    assert not run.critic_optimizer.state
    for name, value in before.items():
        torch.testing.assert_close(run.policy.model.state_dict()[name], value, rtol=0, atol=0)


def test_unfinished_update_cannot_be_checkpointed(tmp_path: Path) -> None:
    run = trainer(groups=1, delay=8)
    # Missing a queued input fails after entering an update.
    run.engine.matches[0].queues[0].clear()
    with pytest.raises(ValueError, match="exactly D"):
        run.run_update()
    with pytest.raises(ValueError, match="complete update boundary"):
        exp.save_checkpoint(run, tmp_path, PROVENANCE)
    with pytest.raises(ValueError, match="previous update did not finish"):
        run.run_update()


def test_training_dispatch_writes_flat_metrics_and_manifests(tmp_path: Path) -> None:
    run = trainer(groups=1, anchor_rate=0)
    exp.run_training(run, 2, tmp_path, PROVENANCE)
    lines = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [row["global_step"] for row in lines] == [1, 2]
    assert all(key.count("/") <= 1 for row in lines for key in row)
    assert (tmp_path / "checkpoints" / "update-00000002.json").is_file()
    assert json.loads((tmp_path / "resolved.json").read_text())["train"]["seed"] == 65


@pytest.fixture
def native_rollout_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from hal.sim import native_build

    root = Path(os.environ.get("HAL_MSL_SOURCE", "~/src/melee-sim-light")).expanduser().resolve()
    assert (root / "data/raw/manifest.json").is_file(), "required MSL data fixture missing"
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.setenv("MSL_CORE_LIBRARY", str(root / "build/melee_core/python/libmelee_core.so"))
    monkeypatch.setenv("HAL_NATIVE_LIBRARY", str(native_build.build(root, tmp_path / "adapter.so")))
    return root / "data"


@pytest.mark.integration
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_native_processed_features_and_exact_resume(
    tmp_path: Path, native_rollout_data: Path, device: str, deterministic_execution: None
) -> None:
    from hal.eval.conditioned_matchups import native_observation_rows
    from hal.eval.self_play_checkpoint import load_evaluation_checkpoint
    from hal.eval.self_play_policy import PlayerCondition
    from hal.eval.self_play_policy import SelfPlayPlayer
    from hal.sim.rollout import Slot

    run = transformer_trainer(device, development=True, deterministic=True, compiled=device == "cuda")
    sampling = run.engine.sampler
    sampling.config = dataclasses.replace(sampling.config, stages=(25,), stocks=(4,), percents=(0.0,))
    shard = exp.NativeSimulatorShard(
        native_rollout_data, (0,), cpu=min(os.sched_getaffinity(0)), device=torch.device(device), max_frame=140
    )
    try:
        engine = exp.RolloutEngine(((shard,),), run.policy, sampling, {0: run.policy.frozen_copy()})
        engine.initialize()
        run = exp.Trainer(engine, run.config)
        # Independently build Dolphin's processed readback at the observation boundary.
        inputs = np.zeros((1, 1, 2, ACTION_DIM), dtype=np.float32)
        inputs[..., :2] = [0.9875, 0.1625]
        observed = shard.step(inputs)[0][0]
        np.testing.assert_array_equal(
            observed.applied_actions[:, :2], np.array([[0.975, 0], [0.975, 0]], dtype=np.float32)
        )
        frame = exp.native.NativeFrameBatch(
            {
                name: shard._records["columns"][:, index].view(MDS_PER_FRAME_DTYPES[name])
                for index, name in enumerate(exp.native.OBSERVATION_FIELDS)
            },
            shard._records["frame_id"],
            shard._records["applied_action"],
            shard._records["wire_inputs"],
            shard._records["reward"],
            shard._records["terminated"],
            shard._records["truncated"],
            shard._records["reset"],
        )
        rows = native_observation_rows(frame)
        state = dataclasses.replace(engine._state(0, 0, 0, 0), frames=(observed,))
        features, previous = run.policy._contexts((state,))
        cpu_frame = dataclasses.replace(observed, device_record=None, device_actions=None)
        baseline, baseline_previous = run.policy._contexts((dataclasses.replace(state, frames=(cpu_frame,)),))
        for name in features:
            torch.testing.assert_close(features[name], baseline[name], rtol=0, atol=0)
        torch.testing.assert_close(previous, baseline_previous, rtol=0, atol=0)
        player = SelfPlayPlayer(
            run.policy.model,
            run.policy.statistics,
            PlayerCondition(0.01, 8, 4, state.player.identity),
            port=1,
            seed=17,
            logical_player=0,
        )
        row = rows[Slot(0, 1)][0]
        np.testing.assert_array_equal(row.action, observed.applied_actions[0])
        expected = torch.tensor([0.975, 0], device=device)
        torch.testing.assert_close(
            torch.stack((features["ego_main_stick_x"], features["ego_main_stick_y"])).flatten(),
            expected,
            rtol=0,
            atol=0,
        )
        actions = torch.zeros(1, 1, 2, ACTION_DIM, device=device)
        actions[0, 0, 0] = torch.as_tensor(row.action, device=device)
        columns = {
            name: torch.tensor([[value]], device=device, dtype=torch.long if isinstance(value, int) else torch.float32)
            for name, value in row.flat.items()
        }
        dolphin = player.builder(
            columns,
            actions,
            torch.zeros(1, 1, dtype=torch.long, device=device),
            torch.zeros(1, 1, dtype=torch.bool, device=device),
            torch.tensor([state.player.identity], device=device),
        )
        for name in features:
            torch.testing.assert_close(features[name], dolphin.features[name], rtol=0, atol=0)
        # Return to an actual pool boundary before testing the next rollout.
        shard.reset(0, engine.matches[0].episode)
        first, _ = run.run_update()
        del first
        manifest = exp.save_checkpoint(run, tmp_path / "checkpoint", PROVENANCE)
        actor = load_evaluation_checkpoint(manifest, device=torch.device("cpu"))
        assert actor.provenance["update"] == 1
        expected_batch, _ = run.run_update()
        expected_weights = {
            name: value.detach().cpu().clone() for name, value in run.policy.model.state_dict().items()
        }
        exp.restore_checkpoint(run, manifest, PROVENANCE)
        actual, _ = run.run_update()
        assert_batch_equal(expected_batch, actual)
        for name, value in expected_weights.items():
            torch.testing.assert_close(value, run.policy.model.state_dict()[name].cpu(), rtol=0, atol=0)
    finally:
        shard.close()


@pytest.mark.integration
def test_native_pool_snapshot_restores_all_lanes_and_rejects_corruption(native_rollout_data: Path) -> None:
    sampling = sampler(anchor_rate=0)
    sampling.config = dataclasses.replace(sampling.config, stages=(25,), stocks=(4,), percents=(0.0,))
    shard = exp.NativeSimulatorShard(
        native_rollout_data, (0, 1), cpu=min(os.sched_getaffinity(0)), device=torch.device("cpu")
    )
    try:
        assert shard.shard.library_sha256 == exp.sha256(
            native_rollout_data.parent / "build/melee_core/python/libmelee_core.so"
        )
        for lane in (0, 1):
            shard.reset(lane, sampling.sample(lane, 0, (0,)))
        inputs = np.zeros((150, 2, 2, ACTION_DIM), dtype=np.float32)
        inputs[:, 0, 0, 0] = 0.75
        inputs[:, 1, 1, 0] = -0.75
        shard.step(inputs)
        saved = shard.save()
        expected = shard.step(inputs[:2])
        shard.restore(saved)
        actual = shard.step(inputs[:2])
        for left_row, right_row in zip(expected, actual, strict=True):
            for left, right in zip(left_row, right_row, strict=True):
                for name in left.columns:
                    np.testing.assert_array_equal(left.columns[name], right.columns[name])
                np.testing.assert_array_equal(left.applied_actions, right.applied_actions)
                assert left.rewards == right.rewards
        for malformed in (saved[:3], saved[:-1], saved + b"trailing"):
            with pytest.raises(ValueError, match="snapshot"):
                shard.restore(malformed)
        header_length = struct.unpack_from("<I", saved)[0]
        header = json.loads(saved[4 : 4 + header_length])
        header["format"] = "hal.o65.native_pool.v1"
        old_header = json.dumps(header).encode()
        old_snapshot = struct.pack("<I", len(old_header)) + old_header + saved[4 + header_length :]
        with pytest.raises(ValueError, match="snapshot identity"):
            shard.restore(old_snapshot)
        with pytest.raises(ValueError, match="distinct lanes"):
            shard.shard.save_many((0, 0)).result()
    finally:
        shard.close()


@pytest.mark.integration
def test_native_shard_close_releases_batch_after_copy_failure(native_rollout_data: Path) -> None:
    class FailedCopyShard(exp.NativeSimulatorShard):
        def _before_write(self) -> None:
            raise RuntimeError("copy failed")

    shard = FailedCopyShard(native_rollout_data, (0,), cpu=min(os.sched_getaffinity(0)), device=torch.device("cpu"))
    with pytest.raises(RuntimeError, match="copy failed"):
        shard.close()
    with pytest.raises(RuntimeError, match="closed"):
        shard.shard.save(0)


def test_proxy_loader_rejects_wrong_artifact_before_loading(tmp_path: Path) -> None:
    checkpoint = tmp_path / "wrong.pt"
    checkpoint.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="recorded O63 bootstrap30"):
        exp.load_development_policy(checkpoint, {}, device="cpu")


def test_training_manifest_binds_exported_actor_and_update(tmp_path: Path) -> None:
    from hal.eval.self_play_checkpoint import evaluation_actor_path
    from hal.eval.self_play_checkpoint import load_evaluation_checkpoint

    run = transformer_trainer("cpu")
    run.run_update()
    manifest = exp.save_checkpoint(run, tmp_path, PROVENANCE)
    loaded = load_evaluation_checkpoint(manifest, device=torch.device("cpu"))
    for name, value in run.policy.model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[name], value, rtol=0, atol=0)
    saved = json.loads(manifest.read_text())
    saved["update"] += 1
    manifest.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="provenance mismatch"):
        evaluation_actor_path(manifest)
    saved["update"] -= 1
    saved["version"] = 1
    manifest.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="unsupported checkpoint identity"):
        evaluation_actor_path(manifest)
    saved["version"] = exp.CHECKPOINT_VERSION
    manifest.write_text(json.dumps(saved))
    actor = evaluation_actor_path(manifest)
    actor.write_bytes(actor.read_bytes() + b"bad")
    with pytest.raises(ValueError, match="actor hash mismatch"):
        load_evaluation_checkpoint(manifest, device=torch.device("cpu"))


def test_checkpoint_rejects_changed_decode_compilation(tmp_path: Path) -> None:
    run = transformer_trainer("cpu")
    manifest = exp.save_checkpoint(run, tmp_path, PROVENANCE)
    policy = exp.TransformerPolicy(
        copy.deepcopy(run.policy.model), run.engine.sampler.schedule, run.policy.statistics, compiled=True
    )
    engine = exp.RolloutEngine(run.engine.groups, policy, run.engine.sampler, {0: policy.frozen_copy()})
    resumed = exp.Trainer(engine, run.config)
    with pytest.raises(ValueError, match="configuration mismatch"):
        exp.restore_checkpoint(resumed, manifest, PROVENANCE)


@pytest.mark.parametrize("saved_mode", [False, True])
def test_resume_allows_execution_mode_change_and_records_the_original_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, saved_mode: bool
) -> None:
    run = trainer(groups=1, anchor_rate=0)
    run.run_update()
    initial_mode = torch.are_deterministic_algorithms_enabled()
    try:
        run.config = dataclasses.replace(run.config, deterministic=saved_mode)
        torch.use_deterministic_algorithms(saved_mode)
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        digest = exp._configuration_digest(run)
        manifest = exp.save_checkpoint(run, tmp_path, PROVENANCE)
        raw = json.loads(manifest.read_text())
        shard = next(row for row in raw["shards"] if row["role"] == "training")
        saved = torch.load(tmp_path / shard["name"], weights_only=False)
        assert saved["execution"] == {"deterministic": saved_mode, "cublas_workspace_config": ":4096:8"}
        run.config = dataclasses.replace(run.config, deterministic=not saved_mode)
        torch.use_deterministic_algorithms(not saved_mode)
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
        assert exp._configuration_digest(run) == digest
        exp.restore_checkpoint(run, manifest, PROVENANCE)
        assert run.config.deterministic is not saved_mode
        run.config = dataclasses.replace(run.config, actor_lr=run.config.actor_lr * 2)
        with pytest.raises(ValueError, match="configuration mismatch"):
            exp.restore_checkpoint(run, manifest, PROVENANCE)
    finally:
        torch.use_deterministic_algorithms(initial_mode)


@pytest.mark.parametrize(
    "execution",
    [
        None,
        {"deterministic": 1, "cublas_workspace_config": None},
        {"deterministic": False, "cublas_workspace_config": False},
    ],
)
def test_resume_rejects_invalid_execution_metadata_before_model_mutation(tmp_path: Path, execution: object) -> None:
    run = trainer(groups=1, anchor_rate=0)
    run.run_update()
    manifest = exp.save_checkpoint(run, tmp_path, PROVENANCE)
    raw = json.loads(manifest.read_text())
    shard = next(row for row in raw["shards"] if row["role"] == "training")
    state = torch.load(tmp_path / shard["name"], weights_only=False)
    state["execution"] = execution
    buffer = io.BytesIO()
    torch.save(state, buffer)
    shard["name"], shard["sha256"] = exp._durable_file(tmp_path, buffer.getvalue(), "rank0.pt")
    manifest.write_text(json.dumps(raw))
    before = exp._state_hash(run.policy.model.state_dict())
    with pytest.raises(ValueError, match="execution mode"):
        exp.restore_checkpoint(run, manifest, PROVENANCE)
    assert exp._state_hash(run.policy.model.state_dict()) == before


def test_native_defaults_reserve_deterministic_algorithms_for_resume_verification(tmp_path: Path) -> None:
    config = exp.NativeCli(
        checkpoint=tmp_path / "proxy.pt",
        sampler=tmp_path / "sampler.json",
        output=tmp_path / "run",
        data_dir=tmp_path / "data",
        cpus=(min(os.sched_getaffinity(0)),),
        updates=1,
    )
    assert not config.training.deterministic
    checked = dataclasses.replace(config, verify_resume=True)
    assert checked.verify_resume and not checked.training.deterministic


@pytest.mark.parametrize(
    "field,message",
    [
        ("provenance", "provenance hash"),
        ("environment", "environment"),
        ("configuration_sha256", "configuration hash"),
        ("role", "role or filename"),
        ("name", "shard identity"),
        ("sha256", "shard identity"),
        ("shards", "nonempty list"),
    ],
)
def test_manifest_metadata_is_validated_by_resume_and_evaluation(tmp_path: Path, field: str, message: str) -> None:
    from hal.eval.self_play_checkpoint import load_evaluation_checkpoint

    run = trainer(groups=1)
    run.run_update()
    manifest = exp.save_checkpoint(run, tmp_path, PROVENANCE)
    raw = json.loads(manifest.read_text())
    if field == "provenance":
        raw[field]["adapter_sha256"] = "invalid"
    elif field == "environment":
        raw[field]["world_size"] = 1
    elif field in ("role", "name", "sha256"):
        raw["shards"][0][field] = "invalid"
    else:
        raw[field] = "invalid"
    manifest.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=message):
        exp.restore_checkpoint(run, manifest, PROVENANCE)
    with pytest.raises(ValueError, match=message):
        load_evaluation_checkpoint(manifest, device=torch.device("cpu"))


@pytest.mark.parametrize("warmup", [0, 100, 1])
def test_resume_diagnostic_covers_next_batch_and_both_optimizers(tmp_path: Path, warmup: int) -> None:
    run = trainer(groups=1, development=True, anchor_rate=0, warmup=warmup)
    if warmup == 1:
        run.engine.sampler.config = dataclasses.replace(run.engine.sampler.config, lambda_max=100)
        run.engine.sampler.lambda_max = 100
        run.config = dataclasses.replace(run.config, plateau_patience=2, warmup_match_frames=0)
    run.run_update()
    exp.verify_checkpoint_resume(run, tmp_path, PROVENANCE)
    result = json.loads((tmp_path / "resume-check.json").read_text())
    assert result["next_batch_expected_sha256"] == result["next_batch_actual_sha256"]
    assert result["optimizer_expected_sha256"] == result["optimizer_actual_sha256"]
    assert result["maximum_parameter_error"] == 0
    assert result["deterministic"] is True
    assert not run.config.deterministic
    assert not torch.are_deterministic_algorithms_enabled()
    assert run.update == 1
    if warmup == 1:
        assert run.warming
        assert result["uninterrupted_metrics"]["train/warming"] == 0
        assert result["uninterrupted_metrics"]["train/actor_steps"] == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_muon_resume_repeats_the_next_fit(tmp_path: Path, device: str) -> None:
    run = transformer_trainer(device, development=True)
    run = exp.Trainer(run.engine, dataclasses.replace(run.config, actor_optimizer="muon", actor_lr=1e-5))
    run.run_update()
    manifest = exp.save_checkpoint(run, tmp_path, PROVENANCE)
    optimizer_before = exp._state_hash(
        (run.actor_optimizer.state_dict(), run.critic_optimizer.state_dict(), run.scheduler.state_dict())
    )
    batch = run.prepare(run.engine.collect(run.update))
    run.fit(batch)
    expected = {name: value.detach().cpu().clone() for name, value in run.policy.model.state_dict().items()}
    exp.restore_checkpoint(run, manifest, PROVENANCE)
    assert (
        exp._state_hash(
            (run.actor_optimizer.state_dict(), run.critic_optimizer.state_dict(), run.scheduler.state_dict())
        )
        == optimizer_before
    )
    run.fit(batch)
    for name, value in expected.items():
        torch.testing.assert_close(run.policy.model.state_dict()[name].cpu(), value, rtol=0, atol=0, msg=name)


@pytest.fixture
def deterministic_execution(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    previous = torch.are_deterministic_algorithms_enabled()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="proxy fit requires CUDA")
def test_proxy_muon_repeated_fit_on_long_history(deterministic_execution: None) -> None:
    from hal.eval.self_play_checkpoint import load_evaluation_checkpoint

    source = os.environ.get("HAL_O65_PROXY_CHECKPOINT")
    if source is None:
        pytest.skip("set HAL_O65_PROXY_CHECKPOINT to the O63 bootstrap30 final.pt")
    checkpoint = load_evaluation_checkpoint(Path(source), device=torch.device("cuda"))
    checkpoint.model.requires_grad_(True)
    checkpoint.model.percentile_head.requires_grad_(False)
    sampling = sampler(anchor_rate=0)
    sampling.schedule = exp.development_config(checkpoint.model.head_offsets)
    policy = exp.TransformerPolicy(checkpoint.model, sampling.schedule, checkpoint.statistics)
    engine = exp.RolloutEngine(((FakeShard((0,)),),), policy, sampling, {0: policy.frozen_copy()})
    run = exp.Trainer(engine, exp.TrainConfig(critic_warmup_updates=0, warmup_steps=0, deterministic=True))
    episode = sampling.sample(0, 0, (0,))
    players = tuple(dataclasses.replace(player, timing=exp.DelayReplan(8, 4)) for player in episode.players)
    episode = dataclasses.replace(episode, players=players)
    columns = {
        name: 0 if np.dtype(MDS_PER_FRAME_DTYPES[name]).kind == "i" else 0.0 for name in exp.native.OBSERVATION_FIELDS
    }
    columns.update(stage=25, p1_character=1, p2_character=1, p1_stock=4, p2_stock=4)
    frame = exp.SimFrame(columns, np.zeros((2, ACTION_DIM), dtype=np.float32))
    states = tuple(
        exp.DecisionState(
            0,
            0,
            0,
            episode,
            clock,
            (0, 0, 128, 0),
            (frame,) * (clock + 1),
            torch.zeros(8, 4, dtype=torch.long),
            torch.zeros(8, 4, dtype=torch.long),
        )
        for clock in range(128, 256, 4)
    )
    actions = torch.zeros(32, 4, 4, device="cuda", dtype=torch.long)
    with torch.no_grad():
        old = policy.evaluate(states, actions)
        reference = run.reference.evaluate(states, actions)
        values = policy.values(old.features)
    records = tuple(
        exp.DecisionRecord(
            state, actions[index].cpu(), old.log_probs[index].cpu(), [0.0] * 4, boundary=True, next_state=state
        )
        for index, state in enumerate(states)
    )
    rollout = exp.Rollout(records, 0, 0, 128, 0, 0, 0.0, ())
    advantages = torch.linspace(-1, 1, 32, device="cuda")
    batch = exp.PreparedUpdate(
        rollout,
        exp._decision_batch(records, policy.device),
        old,
        reference,
        values,
        advantages,
        values + advantages,
        torch.ones(32, dtype=torch.bool, device="cuda"),
        (tuple(range(32)),),
        {},
    )
    model = copy.deepcopy(policy.model.state_dict())
    actor = copy.deepcopy(run.actor_optimizer.state_dict())
    critic = copy.deepcopy(run.critic_optimizer.state_dict())
    scheduler = copy.deepcopy(run.scheduler.state_dict())
    run.fit(batch)
    expected = {name: value.detach().cpu().clone() for name, value in policy.model.state_dict().items()}
    policy.model.load_state_dict(model)
    run.actor_optimizer.load_state_dict(copy.deepcopy(actor))
    run.critic_optimizer.load_state_dict(copy.deepcopy(critic))
    run.scheduler.load_state_dict(scheduler)
    run.fit(batch)
    for name, value in expected.items():
        torch.testing.assert_close(policy.model.state_dict()[name].cpu(), value, rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("version", [1, 2])
def test_periodic_metrics_preserve_unscored_budget_matches(tmp_path: Path, version: int) -> None:
    result = tmp_path / "results.json"
    result.write_text(
        json.dumps(
            {
                "format": f"hal.o65.evaluation-results.v{version}",
                "split": "selection",
                "checkpoint_provenance": {"update": 2},
                "groups": [
                    {
                        "metrics": {
                            "msl": {
                                "matches": 2,
                                "completed_pairs": 0,
                                "win_rate": {"mean": None},
                                "net_stocks_per_min": {"mean": 0.0},
                                "net_stock_lcb": None,
                            }
                        }
                    }
                ],
            }
        )
    )
    metrics = exp.periodic_evaluation_metrics(result)
    assert metrics["evaluation/update"] == 2
    assert metrics["evaluation/msl_cell0_matches"] == 2
    assert metrics["evaluation/msl_cell0_completed_pairs"] == 0
    assert "evaluation/msl_cell0_win_rate" not in metrics


def test_periodic_and_native_evaluations_have_independent_cadence(tmp_path: Path) -> None:
    from hal.eval.periodic_self_play import PeriodicEvaluator

    class RecordingEvaluator(PeriodicEvaluator):
        def __init__(self) -> None:
            super().__init__(tmp_path / "evaluations", every=1, arguments=())
            self.updates: list[int] = []

        def submit(self, checkpoint: Path, *, update: int) -> bool:
            assert checkpoint.is_file()
            self.updates.append(update)
            return True

    run = trainer(groups=1, anchor_rate=0)
    evaluator = RecordingEvaluator()
    exp.run_training(
        run,
        3,
        tmp_path,
        PROVENANCE,
        periodic_evaluator=evaluator,
        evaluation_shards=(StockShard((90, 91), length=13),),
        evaluation_config=exp.EvaluationConfig((0.01,), 1065, 2, max_rollouts=1),
        evaluation_interval=1,
    )
    assert evaluator.updates == [1, 2, 3]
    assert run.evaluation_cursor == 6


class StockShard(FakeShard):
    def reset(self, lane: int, episode: Any) -> Any:
        return dataclasses.replace(super().reset(lane, episode), columns={"p1_stock": 2, "p2_stock": 2})

    def step(self, inputs: NDArray[np.float32], active: NDArray[np.bool_]) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            tuple(dataclasses.replace(frame, columns={"p1_stock": 2, "p2_stock": 0}) for frame in row)
            for row in super().step(inputs, active)
        )


def test_fixed_anchor_evaluation_preserves_training_pool() -> None:
    run = trainer(groups=1, anchor_rate=0)
    run.run_update()
    run.run_update()
    pool = [shard.save() for group in run.engine.groups for shard in group]
    actor_steps = run.actor_steps
    sampling_cursor = run.engine.sampler.cursor
    shards = (StockShard((90, 91), length=13),)
    config = exp.EvaluationConfig((0.01, 0.02), 1065, 2, max_rollouts=1)
    first = exp.evaluate_anchors(run, shards, config)
    second = exp.evaluate_anchors(run, shards, config)
    assert first == second
    assert any("win_rate" in key for key in first)
    assert any("a1_" in key for key in first)
    assert run.evaluation_cursor == 2
    assert run.actor_steps == actor_steps
    assert run.engine.sampler.cursor == sampling_cursor
    assert pool == [shard.save() for group in run.engine.groups for shard in group]
    with pytest.raises(ValueError, match="training shard"):
        exp.evaluate_anchors(run, run.engine.groups[0], config)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
@torch.no_grad()
def test_subsampled_window_preserves_old_policy_trunk_shape(device: str) -> None:
    run = transformer_trainer(device)
    match = run.engine.matches[0]
    state = exp.DecisionState(
        0,
        0,
        0,
        match.episode,
        0,
        (0, 0, 0, 0),
        tuple(match.history),
        torch.empty(0, 4, dtype=torch.long),
        torch.empty(0, 4, dtype=torch.long),
    )
    later = dataclasses.replace(state, clock=16, frames=state.frames * 17)
    actions = torch.zeros(2, 4, 4, dtype=torch.long, device=device)
    full = run.policy.evaluate((state, later), actions)
    subset = run.policy.evaluate((state,), actions[:1])
    torch.testing.assert_close(full.features.trunk[:1], subset.features.trunk, rtol=0, atol=0)
    torch.testing.assert_close(full.log_probs[:1], subset.log_probs, rtol=2e-5, atol=2e-6)
    for first, second in zip(full.distributions, subset.distributions, strict=True):
        torch.testing.assert_close(first[:1], second, rtol=2e-5, atol=2e-6)


def test_batched_replay_preserves_row_order_and_actor_gradients() -> None:
    run = transformer_trainer("cpu", development=True)
    match = run.engine.matches[0]
    states = []
    for stream, side, clock in ((1, 1, 16), (0, 0, 0), (1, 1, 4), (2, 0, 8)):
        player = match.episode.players[side]
        opponent = match.episode.players[1 - side]
        states.append(
            exp.DecisionState(
                stream,
                0,
                side,
                match.episode,
                clock,
                (0, stream, 0, 0),
                tuple(match.history) * (clock + 1),
                torch.zeros(player.timing.delay, 4, dtype=torch.long),
                torch.zeros(opponent.timing.delay, 4, dtype=torch.long),
            )
        )
    actions = run.policy.model.codec.quantize(torch.rand(4, 4, ACTION_DIM))
    batched = run.policy.evaluate(states, actions)
    (-batched.log_probs.sum()).backward()
    gradients = tuple(None if p.grad is None else p.grad.clone() for p in run.policy.actor_parameters())
    run.policy.model.zero_grad(set_to_none=True)
    individual = [run.policy.evaluate((state,), actions[index : index + 1]) for index, state in enumerate(states)]
    expected = exp._concatenate_evaluations(individual)
    torch.testing.assert_close(batched.log_probs, expected.log_probs, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(batched.features.trunk, expected.features.trunk, rtol=2e-5, atol=2e-6)
    (-expected.log_probs.sum()).backward()
    for gradient, parameter in zip(gradients, run.policy.actor_parameters(), strict=True):
        if gradient is None:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None
            torch.testing.assert_close(parameter.grad, gradient, rtol=3e-4, atol=5e-6)


class BorrowingShard(FakeShard):
    def __init__(self) -> None:
        super().__init__((0,))
        self.buffer = np.zeros((2, 1, 2, ACTION_DIM), dtype=np.float32)

    def step(self, inputs: NDArray[np.float32], active: NDArray[np.bool_]) -> tuple[tuple[Any, ...], ...]:
        rows = super().step(inputs, active)
        for at, row in enumerate(rows):
            self.buffer[at, 0] = row[0].applied_actions
        return tuple((dataclasses.replace(row[0], applied_actions=self.buffer[at, 0]),) for at, row in enumerate(rows))


def test_rollout_owns_history_after_shard_reuses_output_buffers() -> None:
    policy = ToyPolicy()
    shard = BorrowingShard()
    engine = exp.RolloutEngine(((shard,),), policy, sampler(anchor_rate=0), {0: policy.frozen_copy()})
    engine.initialize()
    rollout = engine.collect(0)
    snapshots = [record.state.frames[-1].applied_actions.copy() for record in rollout.records]
    shard.buffer.fill(123)
    for record, snapshot in zip(rollout.records, snapshots, strict=True):
        np.testing.assert_array_equal(record.state.frames[-1].applied_actions, snapshot)


def test_episode_monte_carlo_diagnostic_uses_full_discounted_episode() -> None:
    run = trainer(groups=1, length=13, anchor_rate=1)
    rollout = run.engine.collect(0)
    records = tuple(record for record in rollout.records if record.state.episode.number == 0)
    results = {result.match: result for result in rollout.results if result.episode.number == 0}
    diagnostics = exp._episode_diagnostics(run.policy, run.reference, records, results)
    targets = []
    for match in results:
        rows = [record for record in records if record.state.match == match]
        rewards = [reward for record in rows for reward in record.rewards]
        for record in rows:
            targets.append(
                sum(exp.GAMMA**offset * reward for offset, reward in enumerate(rewards[record.state.clock :]))
            )
    expected_rmse = math.sqrt(sum(target**2 for target in targets) / len(targets))
    assert diagnostics["monte_carlo_rmse"] == pytest.approx(expected_rmse, rel=1e-6)
    assert diagnostics["monte_carlo_count"] == len(targets)
    assert diagnostics["kl_cost"] == 0
    assert diagnostics["kl_per_frame"] == 0


def test_transformer_rejects_invalid_feature_statistics() -> None:
    run = transformer_trainer("cpu")
    key = next(iter(run.policy.statistics))
    stats = dict(run.policy.statistics)
    stats[key] = dataclasses.replace(stats[key], std=-1)
    with pytest.raises(ValueError, match="observation statistics"):
        exp.TransformerPolicy(run.policy.model, run.engine.sampler.schedule, stats)


def test_rollout_with_no_executed_actions_only_fits_critic() -> None:
    run = trainer(groups=1, length=1, delay=8, anchor_rate=0)
    before = run.policy.model.logits.detach().clone()
    batch, metrics = run.run_update()
    assert not batch.selected.any()
    assert not batch.steps
    assert metrics["policy/empty_update"] == 1
    assert run.actor_steps == 0
    assert run.critic_optimizer.state
    torch.testing.assert_close(run.policy.model.logits, before, rtol=0, atol=0)


def test_mixed_development_delays_resume_when_execution_masks_are_known(tmp_path: Path) -> None:
    run = trainer(groups=2, length=128, anchor_rate=0.15, development=True)
    for _ in range(2):
        batch, _ = run.run_update()
        assert {
            (record.state.player.timing.delay, record.state.player.timing.replan) for record in batch.rollout.records
        } == {(8, 4), (10, 2)}
    manifest = exp.save_checkpoint(run, tmp_path, PROVENANCE)
    expected, _ = run.run_update()
    weights = copy.deepcopy(run.policy.model.state_dict())
    resumed = trainer(groups=2, length=128, anchor_rate=0.15, development=True)
    exp.restore_checkpoint(resumed, manifest, PROVENANCE)
    actual, _ = resumed.run_update()
    assert_batch_equal(expected, actual)
    for name, weight in weights.items():
        torch.testing.assert_close(weight, resumed.policy.model.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
@torch.no_grad()
def test_transformer_development_queues_factor_order_and_pre_action_critic(device: str) -> None:
    run = transformer_trainer(device, development=True)
    match = run.engine.matches[0]
    players = (
        dataclasses.replace(match.episode.players[0], timing=exp.DelayReplan(8, 4)),
        dataclasses.replace(match.episode.players[1], timing=exp.DelayReplan(10, 2)),
    )
    episode = dataclasses.replace(match.episode, players=players)
    states = tuple(
        exp.DecisionState(
            side,
            0,
            side,
            episode,
            0,
            (0, side, 0, 0),
            tuple(match.history),
            torch.zeros(players[side].timing.delay, 4, dtype=torch.long),
            torch.zeros(players[1 - side].timing.delay, 4, dtype=torch.long),
        )
        for side in range(2)
    )
    sampled = run.policy.sample(states, run.engine.sampler.rng)
    replay = run.policy.evaluate(states, sampled.actions)
    tolerance = 0.15 if device == "cuda" else 1e-5
    torch.testing.assert_close(sampled.log_probs, replay.log_probs, atol=tolerance, rtol=0)
    indices = sampled.actions[..., exp.DECODE_GROUP_INDICES]
    expected = torch.stack(
        [
            distribution.gather(-1, indices[..., group, None]).squeeze(-1).log()
            for group, distribution in enumerate(replay.distributions)
        ],
        dim=-1,
    )
    expected[1, 2:] = 0
    torch.testing.assert_close(replay.log_probs, expected.flatten(1), atol=1e-6, rtol=0)
    altered = run.policy.evaluate(states, torch.zeros_like(sampled.actions))
    torch.testing.assert_close(replay.features.decoder, altered.features.decoder, atol=0, rtol=0)
    recovered = run.policy.model.codec.quantize(torch.tensor(run.policy.wire_actions(sampled.actions), device=device))
    torch.testing.assert_close(recovered, sampled.actions, atol=0, rtol=0)
    at_two = dataclasses.replace(states[1], clock=2, frames=states[1].frames * 3)
    short = run.policy.sample((at_two,), run.engine.sampler.rng)
    assert not short.actions[:, 2:].any()
    for distribution in short.distributions:
        assert not distribution[:, 2:, 1:].any()
        assert bool((distribution[:, 2:, 0] == 1).all())


def test_rollout_rejects_mutable_or_shared_anchors() -> None:
    policy = ToyPolicy()
    shards = ((FakeShard((0,)),),)
    for anchor in (policy, copy.deepcopy(policy)):
        with pytest.raises(ValueError, match="frozen and have independent parameters"):
            exp.RolloutEngine(shards, policy, sampler(), {0: anchor})
    engine = exp.RolloutEngine(shards, policy, sampler(), {0: policy.frozen_copy(), 4: policy.frozen_copy()})
    run = exp.Trainer(engine, exp.TrainConfig(actor_optimizer="adam"))
    assert run.next_anchor == 5


@pytest.mark.parametrize(
    "construct",
    [
        lambda: exp.CounterRng(True),
        lambda: exp.IdentityCount(1.5, 1, 100),
        lambda: exp.IdentityCount(1, 1, 2.5),
        lambda: exp.Matchup((1, 1, 1), 1),
        lambda: exp.Matchup((True, 1), 1),
        lambda: dataclasses.replace(sampler().config, stocks=(2.5,)),
        lambda: dataclasses.replace(sampler().config, stages=(True,)),
        lambda: exp.TrainConfig(warmup_steps=0.5),
    ],
)
def test_loop_configuration_rejects_invalid_integer_fields(construct: Any) -> None:
    with pytest.raises(ValueError):
        construct()


def test_disabled_deployment_identity_does_not_require_its_data_row() -> None:
    sampling = sampler()
    rows = tuple(row for row in sampling.config.identities if row.identity != 1)
    cfg = dataclasses.replace(sampling.config, identities=rows, deployment_rate=0)
    sampling = exp.EpisodeSampler(cfg, sampling.schedule, sampling.rng)
    assert all(player.identity != 1 for episode in range(50) for player in sampling.sample(0, episode, (0,)).players)


@dataclasses.dataclass(slots=True)
class CompileCounter:
    calls: int = 0

    def __call__(self, graph: torch.fx.GraphModule, inputs: list[torch.Tensor]) -> Any:
        self.calls += 1
        return graph.forward


def test_compiled_decision_replay_keeps_history_shape_and_actor_gradients() -> None:
    run = transformer_trainer("cpu")
    counter = CompileCounter()
    run.policy.decoder.replay_variants[4] = torch.compile(
        run.policy.decoder.replay_variants[4], backend=counter, dynamic=False, fullgraph=True
    )
    match = run.engine.matches[0]
    state = exp.DecisionState(
        0,
        0,
        0,
        match.episode,
        0,
        (0, 0, 0, 0),
        tuple(match.history),
        torch.empty(0, 4, dtype=torch.long),
        torch.empty(0, 4, dtype=torch.long),
    )
    actions = torch.zeros(1, 4, 4, dtype=torch.long)
    for length in range(1, 50, 4):
        current = dataclasses.replace(state, clock=length - 1, frames=state.frames * length)
        with torch.no_grad():
            output = run.policy.evaluate((current,), actions)
        assert torch.isfinite(output.log_probs).all()
    assert counter.calls == 1
    for length in (9, 25):
        run.policy.model.zero_grad(set_to_none=True)
        current = dataclasses.replace(state, clock=length - 1, frames=state.frames * length)
        output = run.policy.evaluate((current,), actions)
        output.log_probs.sum().backward()
        grad = run.policy.model.temporal.history_attention.key_value.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and bool(grad.any())
    assert counter.calls == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seed", [65, -(2**127), 10**120, 10**250])
def test_gpu_action_counters_match_scalar_bits_and_rebatching(seed: int) -> None:
    from hal.inference.action_rng import action_uniforms

    rng = exp.CounterRng(seed)
    counters = [(index * 100003, index // 5, index * 256 - 123, (0, 8, 10, 21, 95)[index % 5]) for index in range(257)]
    actual = action_uniforms(seed, counters, torch.device("cuda")).cpu()
    expected = torch.tensor(
        [
            [
                [
                    rng.uniform("action", stream, episode, frame, delay + offset + 1, factor)
                    for stream, episode, frame, delay in counters
                ]
                for factor in exp.CODEC_FACTOR_INDICES
            ]
            for offset in range(4)
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    reordered = action_uniforms(seed, counters[::-2], torch.device("cuda")).cpu()
    torch.testing.assert_close(reordered, expected[:, :, torch.arange(256, -1, -2)], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("count", [1, 2, 4])
@pytest.mark.parametrize("inference", [False, True])
def test_packed_append_graph_reads_current_weights_and_preserves_cache(count: int, inference: bool) -> None:
    from hal.inference.kv_cache import BlockRestartCache

    run = transformer_trainer("cuda", compiled=True)
    policy = run.policy
    episode = run.engine.matches[0].episode
    frame = run.engine.matches[0].history[0]
    records = np.zeros(1, dtype=exp.native.FRAME_DTYPE)
    for index, name in enumerate(exp.native.OBSERVATION_FIELDS):
        value = np.array([frame.columns.get(name, 0)], dtype=MDS_PER_FRAME_DTYPES[name])
        records["columns"][0, index] = value.view(np.uint32)[0]
    packed = torch.as_tensor(records.view(np.uint8), device="cuda")
    frame = dataclasses.replace(
        frame, device_record=packed, device_actions=torch.as_tensor(frame.applied_actions, device="cuda")
    )
    states = [
        exp.DecisionState(
            side,
            0,
            side,
            episode,
            count,
            (0, side, 0, 0),
            (frame,) * (count + 1),
            torch.zeros(0, 4, dtype=torch.long),
            torch.zeros(0, 4, dtype=torch.long),
        )
        for side in range(2)
    ]
    with torch.inference_mode(inference), torch.no_grad():
        cached = BlockRestartCache(policy.model, 2, policy.device)
        eager = BlockRestartCache(policy.model, 2, policy.device)
        for iteration in range(3):
            cached.reset()
            eager.reset()
            if iteration:
                next(policy.model.observation_encoder.parameters()).add_(0.03125)
                next(policy.model.trunk.parameters()).add_(0.015625)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                features, observed = policy._contexts(states, tail=count)
                expected = eager.append(policy.model.context_tokens(features, observed))
                actual, actual_observed = policy._append_packed(states, count, cached)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual_observed, observed[:, -1], rtol=0, atol=0)
            for left, right in zip(cached.cache.buffers(), eager.cache.buffers(), strict=True):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert len(cached.cache.updates) == 1
        policy.rollout_caches[(0, 1)] = cached
        policy.rollout_caches[(2, 3)] = eager
        policy.used_cache_keys.update(((0, 1), (2, 3)))
        policy.clear_caches()
        assert set(policy.rollout_caches) == {(0, 1)}


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_stream_gae_matches_individual_recurrences(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    generator = torch.Generator().manual_seed(65)
    count = 23
    values = torch.randn(count, generator=generator).to(device)
    next_values = torch.randn(count, generator=generator).to(device)
    rewards = torch.randn(count, generator=generator).to(device)
    durations = torch.tensor([1, 2, 3, 4] * 5 + [1, 2, 4], device=device)
    streams = ([0, 4, 6, 9, 12, 17, 22], [1, 5, 10, 13, 18, 20], [2, 3, 7, 8, 11, 14, 15, 16, 19, 21])
    bootstrap = torch.ones(count, dtype=torch.bool, device=device)
    bootstrap[[6, 13, 16]] = False
    next_values[~bootstrap] = float("nan")
    trace = bootstrap.clone()
    trace[[9, 12, 22, 20, 21]] = False
    actual = exp.stream_advantages(rewards, values, next_values, durations, bootstrap, trace, streams)
    for indices in streams:
        expected = exp.generalized_advantage(
            rewards[indices],
            values[indices],
            next_values[indices],
            durations[indices],
            bootstrap[indices],
            trace[indices],
        )
        torch.testing.assert_close(actual.advantages[indices], expected.advantages, rtol=0, atol=0)
        torch.testing.assert_close(actual.value_targets[indices], expected.value_targets, rtol=0, atol=0)
    with pytest.raises(ValueError, match="flat decision"):
        exp.stream_advantages(rewards[:, None], values, next_values, durations, bootstrap, trace, streams)
    with pytest.raises(ValueError, match="partition"):
        exp.stream_advantages(rewards, values, next_values, durations, bootstrap, trace, streams[:2])
    with pytest.raises(ValueError, match="partition"):
        exp.stream_advantages(rewards, values, next_values, durations, bootstrap, trace, (streams[0], *streams))
    trace[22] = True
    with pytest.raises(ValueError, match="every stream"):
        exp.stream_advantages(rewards, values, next_values, durations, bootstrap, trace, streams)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.no_grad()
def test_wire_lookup_matches_codec_bits_and_rejects_invalid_factors(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    run = transformer_trainer(device)
    policy = run.policy
    count = max(policy.model.codec.group_vocabs)
    indices = torch.stack([torch.arange(count) % size for size in policy.model.codec.group_vocabs], dim=-1)
    expected = policy.model.codec.dequantize(indices.to(device)).cpu().numpy()
    np.testing.assert_array_equal(policy.wire_actions(indices), expected)
    np.testing.assert_array_equal(policy.device_wire_actions(indices.to(device)).cpu().numpy(), expected)
    policy.model.codec.main_centers = policy.model.codec.main_centers * 0.5
    policy.clear_caches()
    expected = policy.model.codec.dequantize(indices.to(device)).cpu().numpy()
    np.testing.assert_array_equal(policy.wire_actions(indices), expected)
    np.testing.assert_array_equal(policy.device_wire_actions(indices.to(device)).cpu().numpy(), expected)
    indices[0, 0] = -1
    with pytest.raises(ValueError, match="invalid controller"):
        policy.wire_actions(indices)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("storage", ["cpu", "cuda"])
@torch.no_grad()
def test_bulk_packed_histories_match_owned_frame_views(storage: str) -> None:
    run = transformer_trainer("cuda")
    blocks = [
        exp.DeviceFrameBlock(
            torch.randint(0, 256, (128, 3, exp.native.FRAME_DTYPE.itemsize), device=storage, dtype=torch.uint8),
            torch.rand(128, 3, 2, ACTION_DIM, device=storage),
        )
        for _ in range(2)
    ]
    initial = run.engine.matches[0].history[0]
    states = []
    frames = []
    for row, slots in enumerate((((0, 126, 0), (0, 127, 0), (1, 0, 0)), ((1, 1, 2),), ((0, 126, 0), (1, 1, 1)))):
        history = tuple(
            dataclasses.replace(
                initial,
                device_record=blocks[block].records[frame, lane],
                device_actions=blocks[block].actions[frame, lane],
                device_block=blocks[block],
                device_slot=(frame, lane),
            )
            for block, frame, lane in slots
        )
        frames.append(history)
        states.append(dataclasses.replace(run.engine._state(0, row % 2, 0, 0), frames=history))
    actual = run.policy._packed_inputs(states, frames, 5)
    legacy = [
        tuple(dataclasses.replace(frame, device_block=None, device_slot=None) for frame in row) for row in frames
    ]
    expected = run.policy._packed_inputs(states, legacy, 5)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_cache_preparation_survives_both_update_boundary_clears() -> None:
    from hal.inference.kv_cache import BlockRestartCache

    policy = transformer_trainer("cpu").policy
    cache = BlockRestartCache(policy.model, 2, policy.device)
    policy.rollout_caches[(0, 1)] = cache
    policy.used_cache_keys.add((0, 1))
    policy.clear_caches()
    policy.clear_caches()
    assert policy.rollout_caches[(0, 1)] is cache
    policy.rollout_caches[(2, 3)] = cache
    policy.used_cache_keys.add((2, 3))
    policy.clear_caches()
    assert set(policy.rollout_caches) == {(2, 3)}


def test_native_columns_keep_adapter_words_until_read() -> None:
    words = np.zeros(len(exp.native.OBSERVATION_FIELDS), dtype=np.uint32)
    expected = {}
    for index, name in enumerate(exp.native.OBSERVATION_FIELDS):
        value = index - 30 if np.dtype(MDS_PER_FRAME_DTYPES[name]).kind == "i" else np.float32(index / 13)
        expected[name] = int(value) if isinstance(value, int) else float(value)
        words[index] = np.array([value], dtype=MDS_PER_FRAME_DTYPES[name]).view(np.uint32)[0]
    with pytest.raises(ValueError, match="immutable"):
        exp.NativeColumns(words)
    words.flags.writeable = False
    actual = exp.NativeColumns(words)
    assert dict(actual) == expected
    with pytest.raises(KeyError):
        actual["timer"]


def test_policy_rejects_invalid_side_before_packed_column_selection() -> None:
    run = transformer_trainer("cpu")
    state = run.engine._state(0, 0, 0, 0)
    for side in (-1, 2, True):
        with pytest.raises(ValueError, match="decision side"):
            run.policy.sample((dataclasses.replace(state, side=side),), run.engine.sampler.rng)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_action_counter_blake_block_boundaries() -> None:
    from hal.inference.action_rng import action_uniforms

    base_length = len(json.dumps((0, "action", 0, 0, 0, 9, 3), separators=(",", ":")))
    for length in (127, 128, 129, 255, 256, 257):
        seed = 10 ** (length - base_length)
        assert len(json.dumps((seed, "action", 0, 0, 0, 9, 3), separators=(",", ":"))) == length
        actual = action_uniforms(seed, ((0, 0, 0, 8),), torch.device("cuda")).cpu()
        rng = exp.CounterRng(seed)
        expected = torch.tensor(
            [
                [[rng.uniform("action", 0, 0, 0, 8 + offset + 1, factor)] for factor in exp.CODEC_FACTOR_INDICES]
                for offset in range(4)
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
