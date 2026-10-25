"""Dense tensor loop tests with a deterministic, packed simulator seam."""

import copy
import dataclasses
import importlib.util
import io
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any
from typing import Literal

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from hal.models.window_replay import ReplayCriticFeatures
from hal.models.window_replay import ReplayOutput
from hal.wire import ACTION_DIM

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("hal_experiment_065_loop", ROOT / "experiments/065_selfplay_ppo.py")
assert _SPEC is not None and _SPEC.loader is not None
exp = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = exp
_SPEC.loader.exec_module(exp)


class FakeShard:
    def __init__(
        self, ids: tuple[int, ...], *, length: int = 1000000, timeout: bool = False, device: str = "cpu"
    ) -> None:
        self.match_ids = ids
        self.length = length
        self.timeout = timeout
        self.device = torch.device(device)
        self.ticks = np.zeros(len(ids), dtype=np.int64)
        self.episodes = np.full(len(ids), -1, dtype=np.int64)
        self.seeds = np.zeros(len(ids), dtype=np.uint64)
        self.trace = []
        self.stage = np.full(len(ids), 25, dtype=np.int32)
        self.characters = np.ones((len(ids), 2), dtype=np.int32)
        self.stocks = np.full((len(ids), 2), 4, dtype=np.int32)
        self.percent = np.zeros((len(ids), 2), dtype=np.float32)
        self.returns = np.zeros((len(ids), 2), dtype=np.float32)

    def _frames(
        self,
        actions: NDArray[np.float32],
        rewards: NDArray[np.float32],
        terminal: NDArray[np.bool_],
        timeout: NDArray[np.bool_],
    ) -> Any:
        records = np.zeros(len(self.match_ids), dtype=exp.native.FRAME_DTYPE)
        records["frame_id"] = self.ticks
        records["applied_action"] = actions
        records["reward"] = rewards
        records["terminated"] = terminal
        records["truncated"] = timeout
        fields = records["columns"]
        fields[:, exp.NATIVE_COLUMN_LAYOUT["p1_position_x"][0]] = self.ticks.astype(np.float32).view(np.uint32)
        fields[:, exp.NATIVE_COLUMN_LAYOUT["stage"][0]] = self.stage.view(np.uint32)
        for side in range(2):
            for name, values in (
                (f"p{side + 1}_character", self.characters[:, side]),
                (f"p{side + 1}_stock", self.stocks[:, side]),
                (f"p{side + 1}_percent", self.percent[:, side]),
            ):
                fields[:, exp.NATIVE_COLUMN_LAYOUT[name][0]] = values.view(np.uint32)
        columns = fields.copy()
        columns.flags.writeable = False
        return exp.ShardFrames(
            torch.from_numpy(records.view(np.uint8).reshape(len(records), -1)).to(self.device),
            torch.from_numpy(actions.copy()).to(self.device),
            columns,
            rewards.copy(),
            terminal.copy(),
            timeout.copy(),
            actions.copy(),
            records["frame_id"].copy(),
        )

    def reset(self, lane: int, episode: Any) -> Any:
        self.ticks[lane] = 0
        self.episodes[lane] = episode.number
        self.seeds[lane] = episode.seed
        self.returns[lane] = 0
        self.stage[lane] = episode.stage
        self.characters[lane] = [player.character for player in episode.players]
        self.stocks[lane] = [player.stocks for player in episode.players]
        self.percent[lane] = [player.percent for player in episode.players]
        n = len(self.match_ids)
        return self._frames(
            np.zeros((n, 2, ACTION_DIM), dtype=np.float32),
            np.zeros((n, 2), dtype=np.float32),
            np.zeros(n, dtype=np.bool_),
            np.zeros(n, dtype=np.bool_),
        )

    def step(self, inputs: torch.Tensor, active: NDArray[np.bool_]) -> tuple[Any, ...]:
        rows = []
        for frame in inputs.detach().cpu().numpy():
            self.trace.append(frame.copy())
            self.ticks += active
            rewards = frame[..., 6].copy()
            self.returns += np.where(active[:, None], rewards, 0)
            end = self.ticks >= self.length
            rows.append(self._frames(frame, rewards, end & ~np.bool_(self.timeout), end & np.bool_(self.timeout)))
            if bool((end & active).any()):
                break
        return tuple(rows)

    def save(self) -> bytes:
        return json.dumps(
            {
                "format": 1,
                "length": self.length,
                "timeout": self.timeout,
                "ticks": self.ticks.tolist(),
                "episodes": self.episodes.tolist(),
                "seeds": self.seeds.tolist(),
                "returns": self.returns.tolist(),
                "stage": self.stage.tolist(),
                "characters": self.characters.tolist(),
                "stocks": self.stocks.tolist(),
                "percent": self.percent.tolist(),
            }
        ).encode()

    def restore(self, state: bytes) -> None:
        saved = json.loads(state)
        if (saved["format"], saved["length"], saved["timeout"]) != (1, self.length, self.timeout):
            raise ValueError("fake simulator identity changed")
        self.ticks[:] = saved["ticks"]
        self.episodes[:] = saved["episodes"]
        self.seeds[:] = saved["seeds"]
        self.returns[:] = saved["returns"]
        self.stage[:] = saved["stage"]
        self.characters[:] = saved["characters"]
        self.stocks[:] = saved["stocks"]
        self.percent[:] = saved["percent"]


class ToyModel(torch.nn.Module):
    def __init__(self, device: str) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(4, 2, device=device))
        self.value_head = torch.nn.Linear(1, 1, device=device)
        torch.nn.init.zeros_(self.value_head.weight)
        torch.nn.init.zeros_(self.value_head.bias)
        self.percentile_head = torch.nn.Linear(1, 1, device=device).requires_grad_(False)


class ToyPolicy:
    max_delay = 20
    max_opponent_queue = 24
    player_vocab_size = 4
    condition_ranges = exp.ConditioningRanges((math.log(0.01), math.log(100)), (0, 20), (2, 4))

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = ToyModel(device)

    def actor_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return (self.model.logits,)

    def critic_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return tuple(self.model.value_head.parameters())

    def clear_caches(self) -> None:
        pass

    def retain_tick_layouts(self, ring: Any, layouts: Any) -> None:
        pass

    def tick_ready(self, layout: Any, ring: Any, *, clock: int) -> bool:
        return True

    def frozen_copy(self) -> Any:
        result = copy.deepcopy(self)
        result.model.requires_grad_(False)
        return result

    def device_wire_actions(self, indices: torch.Tensor, *, validate: bool = True) -> torch.Tensor:
        if validate and not bool(((indices >= 0) & (indices < 2)).all()):
            raise ValueError("invalid controller factor")
        result = torch.zeros(*indices.shape[:-1], ACTION_DIM, device=indices.device)
        result[..., 6] = indices[..., 0]
        return result

    def _output(self, meta: torch.Tensor, actions: torch.Tensor) -> Any:
        n = len(meta)
        p = self.model.logits.softmax(-1)[None, None].expand(n, 4, 4, 2)
        logs = p.log().gather(-1, actions[..., None]).squeeze(-1)[..., exp.DECODE_GROUP_INDICES].flatten(-2)
        frame = torch.arange(4, device=self.device)[None] < meta[:, exp.REPLAN, None]
        logs = torch.where(frame.repeat_interleave(4, -1), logs, 0)
        trunk = torch.ones(n, 1, device=self.device)
        zeros = torch.zeros(n, 4, device=self.device)
        features = exp.CriticFeatures(
            trunk,
            trunk,
            zeros,
            zeros,
            torch.zeros(n, 24, 4, device=self.device, dtype=torch.long),
            torch.zeros(n, device=self.device, dtype=torch.long),
        )
        return exp.PolicyEvaluation(
            actions, logs, tuple(p[..., g, :].log() for g in exp.DECODE_GROUP_INDICES), features
        )

    @torch.no_grad()
    def sample_tick(self, states: Any, rng: Any, layout: Any, ring: Any, *, clock: int) -> None:
        meta = states.tensor(states.table.integers)
        offsets = meta[:, exp.DELAY, None, None] + torch.arange(1, 5, device=self.device)[None, :, None]
        factors = torch.tensor(exp.CODEC_FACTOR_INDICES, device=self.device)[None, None]
        draws = rng.tensor_uniform(
            3,
            meta[:, exp.STREAM, None, None].expand(-1, 4, 4),
            meta[:, exp.EPISODE, None, None],
            meta[:, exp.CLOCK, None, None] * 128 + offsets * 4 + factors,
        )
        actions = (draws >= self.model.logits.softmax(-1)[:, 0]).long()
        output = self._output(meta, actions)
        due = states.tensor(states.table.active) & (clock % meta[:, exp.REPLAN] == 0)
        ring.commit(meta, due, actions, self.device_wire_actions(actions), count=4 if clock % 4 == 0 else 2)
        states.table.record_evaluation(states.ids, output, self.values(output.features), states.enabled)

    def window_output(self, batch: Any) -> ReplayOutput:
        slots = batch.slots
        w, k = slots.valid_WK.shape
        p = self.model.logits.softmax(-1)[None, None, None].expand(w, k, 4, 4, 2)
        logs = (
            p.log()
            .gather(-1, slots.sampled_actions_WK44[..., None])
            .squeeze(-1)[..., exp.DECODE_GROUP_INDICES]
            .flatten(-2)
        )
        mask = slots.executed_WKF & slots.valid_WK[..., None]
        trunk = torch.ones(w, k, 1, device=self.device)
        features = ReplayCriticFeatures(
            trunk,
            trunk,
            slots.condition_WK4,
            slots.opponent_condition_WK4,
            slots.opponent_queue_WKQ4,
            slots.opponent_lengths_WK,
        )
        return ReplayOutput(
            torch.where(mask, logs, 0), tuple(p[..., g, :].log() for g in exp.DECODE_GROUP_INDICES), mask, features
        )

    def evaluate_reference(self, batch: Any, positions: Any, window_order: Any) -> Any:
        return self.window_output(batch)

    def evaluate_values(self, batch: Any) -> torch.Tensor:
        return self.model.value_head(torch.ones(*batch.slots.valid_WK.shape, 1, device=self.device)).squeeze(-1)

    def actor_step(self, batch: Any, targets: Any, *, clip_epsilon: float) -> Any:
        self.model.zero_grad(set_to_none=True)
        output = self.window_output(batch)
        loss = exp.actor_loss(output, targets, clip_epsilon=clip_epsilon)
        loss.loss.backward()
        return loss, dataclasses.replace(
            output,
            log_probs_WKF=output.log_probs_WKF.detach(),
            log_distributions=tuple(p.detach() for p in output.log_distributions),
        )

    def values(self, features: Any) -> torch.Tensor:
        return self.model.value_head(features.trunk.detach()).squeeze(-1)


def sampler(*, delay: int = 0, mixed: bool = True) -> Any:
    cfg = exp.SamplerConfig(
        (exp.IdentityCount(1, 1, 400), exp.IdentityCount(2, 1, 100), exp.IdentityCount(3, 1, 900)),
        (exp.Matchup((1, 1), 1.0),),
        (25,),
        data_floor=100,
        deployment_identity=1,
        deployment_rate=0.225,
        tempering=0.5,
        lambda_max=0.02,
    )
    timings = (exp.DelayReplan(delay, 2), exp.DelayReplan(delay, 4)) if mixed else (exp.DelayReplan(delay, 2),)
    return exp.EpisodeSampler(cfg, exp.ScheduleConfig(timings, tuple(range(1, 25))), exp.CounterRng(65))


def trainer(
    device: str = "cpu",
    *,
    groups: int = 2,
    length: int = 1000000,
    timeout: bool = False,
    delay: int = 0,
    warmup: int = 0,
    development: bool = False,
) -> Any:
    torch.manual_seed(19)
    policy = ToyPolicy(device)
    sampling = sampler(delay=delay)
    if development:
        sampling.schedule = exp.development_config((*range(1, 13), 16, 20, 24, 28))
    shards = tuple(
        (
            FakeShard((2 * g,), length=length, timeout=timeout, device=device),
            FakeShard((2 * g + 1,), length=length, timeout=timeout, device=device),
        )
        for g in range(groups)
    )
    engine = exp.RolloutEngine(shards, policy, sampling, {0: policy.frozen_copy()})
    engine.initialize()
    config = exp.TrainConfig(
        actor_optimizer="adam",
        actor_lr=0.03,
        critic_lr=0.01,
        critic_epochs=2,
        critic_warmup_updates=warmup,
        warmup_steps=0,
        windows_per_step=2,
        decisions_per_window=16,
        snapshot_interval=2,
    )
    return exp.Trainer(engine, config)


PROVENANCE = {
    key: "a" * (40 if key == "git_sha" else 64)
    for key in ("git_sha", "checkpoint_sha256", "data_manifest_sha256", "simulator_sha256", "adapter_sha256")
}


def transformer_trainer(
    device: str = "cpu",
    *,
    development: bool = False,
    deterministic: bool = False,
    decoder_precision: Literal["bf16", "fp32"] = "bf16",
    matches: tuple[int, ...] = (0,),
    episode_length: int = 67,
) -> Any:
    from hal.data.feature_stats import FeatureStats
    from hal.data.feature_stats import consolidate_key
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
    vocabulary = PlayerVocabulary(())
    cfg = ActionSequenceConfig(
        d_model=16,
        n_layers=1,
        n_heads=2,
        temporal_d_model=32,
        temporal_layers=1,
        temporal_heads=2,
        temporal_ff_dim=64,
        group_head_dim=8,
        value_hidden_dim=8,
        return_embed_dim=8,
        player_vocab_size=vocabulary.size,
        player_vocab_sha256=vocabulary.sha256,
    )
    sampling = sampler()
    sampling.schedule = (
        exp.development_config(cfg.head_offsets)
        if development
        else exp.ScheduleConfig(sampling.schedule.configurations, cfg.head_offsets)
    )
    ranges = ConditioningRanges((math.log(0.01), math.log(0.02)), (8, 10) if development else (0, 0), (2, 4))
    model = (
        SelfPlayTransformer(
            cfg,
            ranges,
            vocabulary,
            max_opponent_queue=14 if development else 4,
            critic_hidden_dim=8,
            decoder_precision=decoder_precision,
        )
        .to(device)
        .eval()
    )
    policy = exp.TransformerPolicy(model, sampling.schedule, stats)
    shards = (FakeShard(matches, length=episode_length, timeout=True, device=device),)
    engine = exp.RolloutEngine((shards,), policy, sampling, {0: policy.frozen_copy()})
    engine.initialize()
    config = exp.TrainConfig(
        deterministic=deterministic,
        actor_optimizer="adam",
        actor_lr=1e-4,
        critic_epochs=1,
        windows_per_step=4,
        decisions_per_window=16,
        critic_warmup_updates=0,
        warmup_steps=0,
        snapshot_interval=100,
    )
    return exp.Trainer(engine, config)


def assert_batch_equal(left: Any, right: Any) -> None:
    assert exp._batch_hash(left) == exp._batch_hash(right)


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
        match = sampling.sample(7, episode, episode % 3)
        assert sum(p.anchor is not None for p in match.players) <= 1
        anchors += any(p.anchor is not None for p in match.players)
        for player in match.players:
            identities[player.identity] = identities.get(player.identity, 0) + 1
            assert player.timing in sampling.schedule.configurations
            lambdas.append(player.log_lambda)
    assert 0.08 < identities[0] / 6000 < 0.12
    assert 0.20 < identities[1] / 6000 < 0.25
    assert 2.6 < identities[3] / identities[2] < 3.4  # sqrt(900/100)
    assert anchors == 3000
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
            players = sampling.sample(match, episode, None).players
            assert players[0].stocks == players[1].stocks
            stocks.add(players[0].stocks)
    assert stocks == {1, 2, 3, 4}


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
    run = trainer(groups=1)
    before = float(run.policy.model.logits[0].softmax(-1)[1].detach())
    for _ in range(8):
        run.run_update()
    after = float(run.policy.model.logits[0].softmax(-1)[1].detach())
    assert after > before + 0.15


def test_critic_only_warmup_does_not_step_actor() -> None:
    run = trainer(groups=1, warmup=100)
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
    run = trainer(groups=1, warmup=100)

    def fail_evaluation(batch: Any, positions: Any, window_order: Any) -> Any:
        raise AssertionError("warmup must not evaluate the reference")

    monkeypatch.setattr(run.reference, "evaluate_reference", fail_evaluation)
    batch, metrics = run.run_update()
    assert batch.reference is None
    assert metrics["policy/kl_per_frame"] == 0
    assert all(value == 0 for key, value in metrics.items() if key.startswith("policy/kl_per_frame_lambda"))
    assert run.actor_steps == 0


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
        torch.randn(count, policy.model.cfg.d_model, device=device),
        torch.randn(count, policy.model.cfg.temporal_d_model, device=device),
        torch.randn(count, 4, device=device),
        torch.randn(count, 4, device=device),
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


def test_masked_deployment_rate_is_total_masked_rate() -> None:
    sampling = sampler()
    config = dataclasses.replace(sampling.config, deployment_identity=0)
    sampling = exp.EpisodeSampler(config, sampling.schedule, sampling.rng)
    masked = sum(
        player.identity == 0 for episode in range(3000) for player in sampling.sample(0, episode, None).players
    )
    assert 0.20 < masked / 6000 < 0.25


def test_lambda_calibration_uses_every_timing_and_preserves_actor() -> None:
    run = trainer(groups=1, warmup=1)
    run.engine.sampler.config = dataclasses.replace(run.engine.sampler.config, lambda_max=100)
    run.engine.sampler.lambda_max = 100
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
    kl = exp.factor_kl(following.old, following.reference, following.decisions.executed_factors_JF)
    assert float(kl.max()) > 0
    executed = following.decisions.executed_factors_JF
    expected_kl = torch.where(executed, kl, 0).sum() / following.decisions.replan_J.sum()
    assert following.metrics["policy/kl_per_frame"] == pytest.approx(float(expected_kl))
    run.warming = True
    unregularized = run.prepare(rollout)
    run.warming = False
    table = rollout.table
    size = table.ticks * table.players
    ids = rollout.ids
    penalty = torch.zeros_like(table.values)
    penalty[ids] = -following.decisions.kl_coefficient_J * torch.where(executed, kl, 0).sum(-1)
    zeros = torch.zeros_like(penalty[:size]).reshape(table.ticks, table.players)
    effect = exp.dense_advantages(
        penalty[:size].reshape_as(zeros),
        zeros,
        zeros,
        table.duration[:size].reshape_as(zeros),
        ~table.terminal[:size].reshape_as(zeros),
        ~table.boundary[:size].reshape_as(zeros),
        table.valid[:size].reshape_as(zeros),
    ).advantages.flatten()[ids]
    torch.testing.assert_close(following.advantages, unregularized.advantages + effect)
    torch.testing.assert_close(following.targets, unregularized.targets + effect)


def test_warmup_ends_after_32_updates_with_calibrated_lambda() -> None:
    run = trainer(groups=1, warmup=exp.TrainConfig().critic_warmup_updates)
    run.engine.sampler.config = dataclasses.replace(run.engine.sampler.config, lambda_max=100)
    run.engine.sampler.lambda_max = 100
    before = run.policy.model.logits.detach().clone()
    advantages = {replan: [] for replan in (2, 4)}
    for update in range(1, 33):
        batch, metrics = run.run_update()
        for replan, parts in advantages.items():
            parts.append(batch.advantages[batch.decisions.replan_J == replan].double())
        assert batch.reference is None and run.actor_steps == 0
        assert run.warming == (update < 32)
        assert ("train/warmup_ended" in metrics) == (update == 32)
    expected = 3 * max(float(torch.cat(parts).std(unbiased=False)) for parts in advantages.values())
    assert metrics["train/lambda_max"] == pytest.approx(expected)
    torch.testing.assert_close(before, run.policy.model.logits, rtol=0, atol=0)
    batch, metrics = run.run_update()
    assert run.update == 33 and run.actor_steps > 0 and batch.reference is not None


def test_warmup_rejects_a_negative_length() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        exp.TrainConfig(critic_warmup_updates=-1)


def test_actor_fit_rejects_missing_reference() -> None:
    run = trainer(groups=1)
    batch = run.prepare(run.engine.collect(0))
    before = copy.deepcopy(run.policy.model.state_dict())
    with pytest.raises(ValueError, match="actor update requires a reference evaluation"):
        run.fit(dataclasses.replace(batch, reference=None))
    assert not run.actor_optimizer.state
    assert not run.critic_optimizer.state
    for name, value in before.items():
        torch.testing.assert_close(run.policy.model.state_dict()[name], value, rtol=0, atol=0)


def test_unfinished_update_cannot_be_checkpointed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = trainer(groups=1, delay=8)

    def fail_collect(update: int) -> Any:
        raise ValueError("failed collection")

    monkeypatch.setattr(run.engine, "collect", fail_collect)
    with pytest.raises(ValueError, match="failed collection"):
        run.run_update()
    with pytest.raises(ValueError, match="complete update boundary"):
        exp.save_checkpoint(run, tmp_path, PROVENANCE)
    with pytest.raises(ValueError, match="previous update did not finish"):
        run.run_update()


def test_training_dispatch_writes_flat_metrics_and_manifests(tmp_path: Path) -> None:
    run = trainer(groups=1)
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
def test_native_shard_close_releases_batch_after_copy_failure(native_rollout_data: Path) -> None:
    class FailedCopyShard(exp.NativeSimulatorShard):
        def _before_write(self) -> None:
            raise RuntimeError("copy failed")

    shard = FailedCopyShard(native_rollout_data, (0,), cpu=min(os.sched_getaffinity(0)), device=torch.device("cpu"))
    with pytest.raises(RuntimeError, match="copy failed"):
        shard.close()
    with pytest.raises(RuntimeError, match="closed"):
        shard.shard.save(0)


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
    with pytest.raises(ValueError, match="checkpoint manifest schema changed"):
        evaluation_actor_path(manifest)
    saved["version"] = exp.CHECKPOINT_VERSION
    manifest.write_text(json.dumps(saved))
    actor = evaluation_actor_path(manifest)
    actor.write_bytes(actor.read_bytes() + b"bad")
    with pytest.raises(ValueError, match="actor hash mismatch"):
        load_evaluation_checkpoint(manifest, device=torch.device("cpu"))


@pytest.mark.parametrize("saved_mode", [False, True])
def test_resume_allows_execution_mode_change_and_records_the_original_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, saved_mode: bool
) -> None:
    run = trainer(groups=1)
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
    run = trainer(groups=1)
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


@pytest.mark.parametrize("partial", [None, [(True, 0)], [(0, -1)], [(99, 0)], [[0, 0]], [(0, 0)], [(0, 0), (0, 0)]])
def test_resume_rejects_invalid_partial_episodes_before_model_mutation(tmp_path: Path, partial: object) -> None:
    run = trainer(groups=1)
    run.run_update()
    manifest = exp.save_checkpoint(run, tmp_path, PROVENANCE)
    raw = json.loads(manifest.read_text())
    shard = next(row for row in raw["shards"] if row["role"] == "training")
    state = torch.load(tmp_path / shard["name"], weights_only=False)
    state["partial_episodes"] = partial
    buffer = io.BytesIO()
    torch.save(state, buffer)
    shard["name"], shard["sha256"] = exp._durable_file(tmp_path, buffer.getvalue(), "rank0.pt")
    manifest.write_text(json.dumps(raw))
    before = exp._state_hash(run.policy.model.state_dict())
    with pytest.raises(ValueError, match="partial episodes"):
        exp.restore_checkpoint(run, manifest, PROVENANCE)
    assert exp._state_hash(run.policy.model.state_dict()) == before


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
    run = trainer(groups=1, development=True, warmup=warmup)
    if warmup == 1:
        run.engine.sampler.config = dataclasses.replace(run.engine.sampler.config, lambda_max=100)
        run.engine.sampler.lambda_max = 100
        run.config = dataclasses.replace(run.config, critic_warmup_updates=2)
    run.run_update()
    exp.verify_checkpoint_resume(run, tmp_path, PROVENANCE)
    result = json.loads((tmp_path / "resume-check.json").read_text())
    assert result["format"] == "hal.o65.resume-check.v5"
    assert len(set(result["next_batch_sha256"])) == 1
    assert result["restored"] == [result["boundary"]] * 3
    assert result["maximum_parameter_error"] == 0
    assert result["passed"]
    assert result["deterministic"] is False
    assert not run.config.deterministic
    assert not torch.are_deterministic_algorithms_enabled()
    assert run.update == 1
    if warmup == 1:
        assert run.warming
        assert result["control_metrics"][0]["train/warming"] == 0
        assert result["control_metrics"][0]["train/actor_steps"] == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_muon_resume_restores_state_and_matches_measured_next_update(tmp_path: Path, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    run = transformer_trainer(device, development=True)
    engine = exp.RolloutEngine(run.engine.groups, run.policy, run.engine.sampler, {0: run.reference})
    engine.initialize()
    run = exp.Trainer(engine, dataclasses.replace(run.config, actor_optimizer="muon", actor_lr=1e-5))
    run.run_update()
    boundary = exp._resume_state_hashes(run)
    exp.verify_checkpoint_resume(run, tmp_path, PROVENANCE)
    assert exp._resume_state_hashes(run) == boundary
    result = json.loads((tmp_path / "resume-check.json").read_text())
    assert result["passed"]
    assert all(row["passed"] for row in result["groups"].values())
    assert not result["deterministic"]


@pytest.mark.parametrize("error,passed", [(0.0015, True), (0.003, False)])
def test_resume_update_tolerance_uses_measured_control_spread(error: float, passed: bool) -> None:
    first = {"weight": torch.tensor([1.0, 2.0]), "step": torch.tensor(1)}
    second = {"weight": torch.tensor([1.001, 2.0]), "step": torch.tensor(1)}
    resumed = {"weight": torch.tensor([1.0, 2.0 + error]), "step": torch.tensor(1)}
    report = exp._compare_resume_updates(first, second, resumed)
    assert report.groups["weight"]["passed"] is passed
    assert report.groups["step"]["passed"]
    assert report.tensors["weight"]["control_spread"] == pytest.approx(0.001, rel=1e-4)
    assert report.groups["weight"]["tolerance_max"] == pytest.approx(0.002, abs=3e-6)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_resume_comparison_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        exp._compare_resume_updates({"w": torch.tensor(0.0)}, {"w": torch.tensor(0.0)}, {"w": torch.tensor(value)})


def test_resume_comparison_rejects_changed_integer_state_and_schema() -> None:
    assert not exp._compare_resume_updates(
        {"step": torch.tensor(1)}, {"step": torch.tensor(2)}, {"step": torch.tensor(2)}
    ).passed
    with pytest.raises(ValueError, match="schema"):
        exp._compare_resume_updates({"w": torch.zeros(1)}, {"w": torch.zeros(1)}, {"w": torch.zeros(2)})
    with pytest.raises(ValueError, match="identities"):
        exp._compare_resume_updates({"w": torch.zeros(1)}, {}, {"w": torch.zeros(1)})


def test_transformer_rejects_invalid_feature_statistics() -> None:
    run = transformer_trainer("cpu")
    key = next(iter(run.policy.statistics))
    stats = dict(run.policy.statistics)
    stats[key] = dataclasses.replace(stats[key], std=-1)
    with pytest.raises(ValueError, match="observation statistics"):
        exp.TransformerPolicy(run.policy.model, run.engine.sampler.schedule, stats)


def test_rollout_with_no_executed_actions_only_fits_critic() -> None:
    run = trainer(groups=1, length=1, delay=8)
    before = run.policy.model.logits.detach().clone()
    batch, metrics = run.run_update()
    assert not batch.selected.any()
    assert not batch.steps
    assert metrics["policy/empty_update"] == 1
    assert run.actor_steps == 0
    assert run.critic_optimizer.state
    torch.testing.assert_close(run.policy.model.logits, before, rtol=0, atol=0)


def test_mixed_development_delays_resume_when_execution_masks_are_known(tmp_path: Path) -> None:
    run = trainer(groups=2, length=128, development=True)
    for _ in range(2):
        batch, _ = run.run_update()
        meta = batch.rollout.table.integers[batch.rollout.ids]
        assert set(map(tuple, meta[:, [exp.DELAY, exp.REPLAN]].tolist())) == {(8, 4), (10, 2)}
    manifest = exp.save_checkpoint(run, tmp_path, PROVENANCE)
    expected, _ = run.run_update()
    weights = copy.deepcopy(run.policy.model.state_dict())
    resumed = trainer(groups=2, length=128, development=True)
    exp.restore_checkpoint(resumed, manifest, PROVENANCE)
    actual, _ = resumed.run_update()
    assert_batch_equal(expected, actual)
    for name, weight in weights.items():
        torch.testing.assert_close(weight, resumed.policy.model.state_dict()[name], rtol=0, atol=0)


def test_rollout_rejects_mutable_or_shared_anchors() -> None:
    policy = ToyPolicy()
    shards = ((FakeShard((0,)),),)
    for anchor in (policy, copy.deepcopy(policy)):
        with pytest.raises(ValueError, match="frozen and have independent parameters"):
            exp.RolloutEngine(shards, policy, sampler(), {0: anchor})
    engine = exp.RolloutEngine(shards, policy, sampler(), {0: policy.frozen_copy(), 4: policy.frozen_copy()})
    with pytest.raises(ValueError, match="only pi_ref"):
        exp.Trainer(engine, exp.TrainConfig(actor_optimizer="adam"))


def test_disabled_deployment_identity_does_not_require_its_data_row() -> None:
    sampling = sampler()
    rows = tuple(row for row in sampling.config.identities if row.identity != 1)
    cfg = dataclasses.replace(sampling.config, identities=rows, deployment_rate=0)
    sampling = exp.EpisodeSampler(cfg, sampling.schedule, sampling.rng)
    assert all(player.identity != 1 for episode in range(50) for player in sampling.sample(0, episode, None).players)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["loss", "policy_loss", "kl_loss", "joint_ratio_WK", "factor_kl_WKF", "gradient"])
def test_trainer_rejects_nonfinite_actor_result_before_optimizer_step(value: float, field: str) -> None:
    class FaultPolicy(ToyPolicy):
        def actor_step(self, batch: Any, targets: Any, *, clip_epsilon: float) -> Any:
            result, output = super().actor_step(batch, targets, clip_epsilon=clip_epsilon)
            if field == "gradient":
                assert self.model.logits.grad is not None
                self.model.logits.grad.fill_(value)
            else:
                bad = torch.full_like(getattr(result, field), value)
                result = dataclasses.replace(result, **{field: bad})
            return result, output

    base = trainer(groups=1)
    policy = FaultPolicy("cpu")
    engine = exp.RolloutEngine(base.engine.groups, policy, base.engine.sampler, {0: policy.frozen_copy()})
    engine.initialize()
    run = exp.Trainer(engine, base.config)
    initial = policy.model.logits.detach().clone()
    error = RuntimeError if field == "gradient" else ValueError
    with pytest.raises(error, match="non-finite|finite"):
        run.run_update()
    torch.testing.assert_close(policy.model.logits, initial, rtol=0, atol=0)
    assert not run.actor_optimizer.state
    assert run.updating


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_trainer_rejects_nonfinite_critic_gradient_before_optimizer_step(value: float) -> None:
    run = trainer(groups=1)
    parameter = run.policy.critic_parameters()[0]
    before = parameter.detach().clone()
    handle = parameter.register_hook(lambda gradient: torch.full_like(gradient, value))
    try:
        with pytest.raises(RuntimeError, match="non-finite"):
            run.run_update()
    finally:
        handle.remove()
    torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    assert not run.critic_optimizer.state and not run.actor_optimizer.state
    assert run.updating


@pytest.mark.parametrize("fault", ["normalization", "support", "selected_action", "mask"])
@pytest.mark.parametrize("capture", [False, True])
def test_live_actor_head_failure_preserves_actor_optimizer_and_schedule(
    fault: str, capture: bool, tmp_path: Path
) -> None:
    class FaultPolicy(ToyPolicy):
        corrupt = False

        def actor_step(self, batch: Any, targets: Any, *, clip_epsilon: float) -> Any:
            result, output = super().actor_step(batch, targets, clip_epsilon=clip_epsilon)
            if self.corrupt:
                if fault == "selected_action":
                    output = dataclasses.replace(output, log_probs_WKF=output.log_probs_WKF - 0.01)
                elif fault == "mask":
                    mask = output.factor_mask_WKF.clone()
                    mask[..., 0] = False
                    output = dataclasses.replace(output, factor_mask_WKF=mask)
                else:
                    heads = tuple(logs.clone() for logs in output.log_distributions)
                    if fault == "normalization":
                        heads[1].sub_(0.05)
                    else:
                        heads[1][..., 0] = -torch.inf
                    output = dataclasses.replace(output, log_distributions=heads)
            return result, output

    base = trainer(groups=1)
    policy = FaultPolicy("cpu")
    engine = exp.RolloutEngine(base.engine.groups, policy, base.engine.sampler, {0: policy.frozen_copy()})
    engine.initialize()
    run = exp.Trainer(engine, base.config, actor_failure_directory=tmp_path if capture else None)
    run.run_update()
    assert run.actor_optimizer.state and run.actor_steps > 0
    parameters = tuple(parameter.detach().clone() for parameter in policy.actor_parameters())
    optimizer = copy.deepcopy(run.actor_optimizer.state_dict())
    scheduler = copy.deepcopy(run.scheduler.state_dict())
    steps, update = run.actor_steps, run.update
    policy.corrupt = True
    with pytest.raises(ValueError, match="normalized|identical support|selected-action|execution mask"):
        run.run_update()
    for actual, expected in zip(policy.actor_parameters(), parameters, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_optimizer = run.actor_optimizer.state_dict()
    assert actual_optimizer["param_groups"] == optimizer["param_groups"]
    for index, state in optimizer["state"].items():
        for name, expected in state.items():
            torch.testing.assert_close(actual_optimizer["state"][index][name], expected, rtol=0, atol=0)
    assert run.scheduler.state_dict() == scheduler
    assert (run.actor_steps, run.update) == (steps, update)
    if capture:
        (path,) = tmp_path.glob("*.pt")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["format"] == "hal.o65.actor-failure.v1"
        assert (saved["actor_steps"], saved["update"], saved["rank"]) == (steps, update, 0)
        assert saved["scheduler"] == scheduler
        assert len(saved["inputs"]) == 5 + len(dataclasses.fields(exp.ReplaySlots))
        assert saved["targets"]["selected_WK"].any()
        assert saved["failures"][0]
        torch.testing.assert_close(saved["model"]["logits"], parameters[0], rtol=0, atol=0)
        assert not tuple(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("last_noise,passed", [(0.0005, True), (0.1, False)])
def test_resume_verifier_accepts_only_calibrated_update_noise(tmp_path: Path, last_noise: float, passed: bool) -> None:
    class NoisyTrainer(exp.Trainer):
        def __init__(self, engine: Any, config: Any) -> None:
            super().__init__(engine, config)
            self.calls = 0

        def run_update(self) -> Any:
            result = super().run_update()
            noise = (0.0, 0.001, last_noise)[self.calls]
            self.calls += 1
            with torch.no_grad():
                self.policy.model.logits.add_(noise)
            return result

    base = trainer(groups=1)
    engine = exp.RolloutEngine(base.engine.groups, base.policy, base.engine.sampler, {0: base.reference})
    engine.initialize()
    run = NoisyTrainer(engine, base.config)
    before = exp._resume_state_hashes(run)
    if passed:
        exp.verify_checkpoint_resume(run, tmp_path, PROVENANCE)
    else:
        with pytest.raises(ValueError, match="exceeded measured update noise"):
            exp.verify_checkpoint_resume(run, tmp_path, PROVENANCE)
    result = json.loads((tmp_path / "resume-check.json").read_text())
    assert result["passed"] is passed
    assert result["comparisons"]["model/logits"]["control_spread"] == pytest.approx(0.001, rel=1e-4)
    assert exp._resume_state_hashes(run) == before


def test_resume_verifier_rejects_inexact_model_restoration(tmp_path: Path) -> None:
    class CorruptRestoreModel(ToyModel):
        def load_state_dict(self, state_dict: Any, strict: bool = True, assign: bool = False) -> Any:
            result = super().load_state_dict(state_dict, strict=strict, assign=assign)
            if self.logits.requires_grad:
                with torch.no_grad():
                    self.logits.add_(0.25)
            return result

    class CorruptRestorePolicy(ToyPolicy):
        def __init__(self, device: str) -> None:
            super().__init__(device)
            self.model = CorruptRestoreModel(device)

    base = trainer(groups=1)
    policy = CorruptRestorePolicy("cpu")
    engine = exp.RolloutEngine(base.engine.groups, policy, base.engine.sampler, {0: policy.frozen_copy()})
    engine.initialize()
    run = exp.Trainer(engine, base.config)
    with pytest.raises(ValueError, match="did not restore exact state.*model"):
        exp.verify_checkpoint_resume(run, tmp_path, PROVENANCE)


def test_resume_noise_can_move_between_model_tensors() -> None:
    first = {"model/a": torch.zeros(4), "model/b": torch.zeros(4)}
    second = {"model/a": torch.full((4,), 0.001), "model/b": torch.zeros(4)}
    resumed = {"model/a": torch.zeros(4), "model/b": torch.full((4,), 0.001)}
    report = exp._compare_resume_updates(first, second, resumed)
    assert report.passed
    assert report.tensors["model/b"]["control_spread"] == 0
    assert report.tensors["model/b"]["error"] > 0


def test_resume_moment_noise_cannot_hide_a_second_moment_error() -> None:
    first = {
        "actor_optimizer/state/0/exp_avg": torch.tensor(1.0),
        "actor_optimizer/state/0/exp_avg_sq": torch.tensor(0.0),
    }
    second = {
        "actor_optimizer/state/0/exp_avg": torch.tensor(1.1),
        "actor_optimizer/state/0/exp_avg_sq": torch.tensor(0.0),
    }
    resumed = {
        "actor_optimizer/state/0/exp_avg": torch.tensor(1.05),
        "actor_optimizer/state/0/exp_avg_sq": torch.tensor(0.01),
    }
    report = exp._compare_resume_updates(first, second, resumed)
    assert report.groups["actor_optimizer/exp_avg"]["passed"]
    assert not report.groups["actor_optimizer/exp_avg_sq"]["passed"]
    assert not report.passed


def test_resume_integer_comparison_does_not_lose_low_bits_to_float_conversion() -> None:
    first = {"step": torch.tensor(2**62)}
    resumed = {"step": torch.tensor(2**62 + 1)}
    assert not exp._compare_resume_updates(first, first, resumed).passed


def test_resume_comparison_rejects_overflow_in_error_statistics() -> None:
    first = {"weight": torch.tensor(1e200, dtype=torch.float64)}
    second = {"weight": torch.tensor(-1e200, dtype=torch.float64)}
    with pytest.raises(ValueError, match="errors must be finite"):
        exp._compare_resume_updates(first, second, second)


def test_trainer_rejects_critic_loss_sum_overflow_before_optimizer_step() -> None:
    run = trainer(groups=1)
    with torch.no_grad():
        run.policy.model.value_head.weight.fill_(2e21)
    batch = run.prepare(run.engine.collect(0))
    loss = exp.critic_loss(run.policy.values(batch.old.features), batch.targets, torch.ones_like(batch.selected))
    assert bool(torch.isinf(loss))
    before = tuple(p.detach().clone() for p in run.policy.critic_parameters())
    with pytest.raises(ValueError, match="critic loss must be finite"):
        run.fit(batch)
    assert not run.critic_optimizer.state and not run.actor_optimizer.state
    for actual, expected in zip(run.policy.critic_parameters(), before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_resume_maximum_is_diagnostic_when_vector_noise_is_unchanged() -> None:
    first = {"model/weight": torch.zeros(10000)}
    second = {"model/weight": torch.full((10000,), 0.001)}
    resumed = {"model/weight": torch.zeros(10000)}
    resumed["model/weight"][0] = 0.003
    report = exp._compare_resume_updates(first, second, resumed)
    assert report.groups["model"]["error_max"] > report.groups["model"]["tolerance_max"]
    assert report.passed


def test_actor_microbatches_match_the_whole_step() -> None:
    run = transformer_trainer("cpu", decoder_precision="fp32")
    batch = run.prepare(run.engine.collect(0))
    actor = copy.deepcopy(run.policy.model.state_dict())
    optimizer = copy.deepcopy(run.actor_optimizer.state_dict())
    critic = copy.deepcopy(run.critic_optimizer.state_dict())
    scheduler = copy.deepcopy(run.scheduler.state_dict())
    run.fit(batch)
    whole = copy.deepcopy(run.policy.model.state_dict())
    assert any(not torch.equal(whole[name], value) for name, value in actor.items())
    run.policy.model.load_state_dict(actor)
    run.actor_optimizer.load_state_dict(optimizer)
    run.critic_optimizer.load_state_dict(critic)
    run.scheduler.load_state_dict(scheduler)
    run.config = dataclasses.replace(run.config, actor_microbatches=2)
    run.fit(batch)
    for name, value in run.policy.model.state_dict().items():
        torch.testing.assert_close(value, whole[name], rtol=1e-5, atol=1e-7, msg=name)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_background_checkpoint_owns_state_and_resumes_next_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    import threading

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    run = trainer(device, length=23, timeout=True)
    for _ in range(3):
        run.run_update()
    synchronous = exp.save_checkpoint(run, tmp_path / "sync", PROVENANCE)
    entered, release = threading.Event(), threading.Event()
    original = exp.write_checkpoint

    def delayed_write(snapshot: Any, directory: Path) -> Path:
        entered.set()
        assert release.wait(timeout=60)
        return original(snapshot, directory)

    monkeypatch.setattr(exp, "write_checkpoint", delayed_write)
    worker = exp.CheckpointWorker(tmp_path / "async", PROVENANCE)
    try:
        worker.submit(run)
        assert entered.wait(timeout=60)
        expected, _ = run.run_update()
        control = exp._update_tensors(run)
        assert not list((tmp_path / "async").glob("update-*.json"))
    finally:
        release.set()
        worker.close()
    asynchronous = tmp_path / "async/update-00000003.json"
    results = []
    for manifest in (synchronous, synchronous, asynchronous):
        exp.restore_checkpoint(run, manifest, PROVENANCE)
        actual, _ = run.run_update()
        assert_batch_equal(expected, actual)
        results.append(exp._update_tensors(run))
        assert run.scheduler.last_epoch == run.actor_steps
    assert exp._compare_resume_updates(results[0], results[1], results[2]).passed
    assert exp._compare_resume_updates(results[0], results[1], control).passed


def test_checkpoint_worker_failure_stops_before_next_update(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    failed = threading.Event()
    run = trainer(groups=1)
    run.config = dataclasses.replace(run.config, checkpoint_interval=1)
    original = run.run_update
    calls = []

    def fail_write(snapshot: Any, directory: Path) -> Path:
        failed.set()
        raise OSError("checkpoint disk failed")

    def update() -> Any:
        calls.append(run.update)
        assert len(calls) == 1, "failed checkpoint allowed the next update"
        return original()

    submit = exp.CheckpointWorker.submit

    def submit_and_observe(worker: Any, current: Any) -> None:
        submit(worker, current)
        assert failed.wait(timeout=60)
        # Observe completion without consuming its exception.
        assert worker.pending is not None
        from concurrent.futures import wait

        wait((worker.pending,), timeout=60)
        assert worker.pending.done()

    monkeypatch.setattr(exp, "write_checkpoint", fail_write)
    monkeypatch.setattr(exp.CheckpointWorker, "submit", submit_and_observe)
    monkeypatch.setattr(run, "run_update", update)
    with pytest.raises(OSError, match="checkpoint disk failed"):
        exp.run_training(run, 3, tmp_path, PROVENANCE)
    assert calls == [0]
    assert not list((tmp_path / "checkpoints").glob("update-*.json"))


def test_training_stdout_reports_updates_checkpoints_and_warmup(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = trainer(groups=1, warmup=1)
    run.engine.sampler.config = dataclasses.replace(run.engine.sampler.config, lambda_max=100)
    exp.run_training(run, 2, tmp_path, PROVENANCE)
    lines = capsys.readouterr().out.splitlines()
    assert sum(line.startswith("update=") for line in lines) == 2
    assert any(line.startswith("critic_warmup_ended update=1") for line in lines)
    assert any(line.startswith("snapshot update=2") for line in lines)
    checkpoint = next(line for line in lines if line.startswith("checkpoint update=2"))
    assert all(field in checkpoint for field in ("training_seconds=", "worker_seconds=", "wait_seconds="))
    progress = next(line for line in lines if line.startswith("update=1"))
    assert "warming=1" in progress and "pi_ref_win32=n/a games=0" in progress
    assert all(field in progress for field in ("rollout=", "preparation=", "actor=", "critic=", "kl/frame=", "ev="))


def test_failed_checkpoint_write_keeps_latest_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = trainer(groups=1)
    run.run_update()
    first = exp.save_checkpoint(run, tmp_path, PROVENANCE)
    original = first.read_bytes()
    run.run_update()
    snapshot = exp.capture_checkpoint(run, PROVENANCE)
    durable_file = exp._durable_file

    def fail_simulator(directory: Path, payload: bytes, suffix: str) -> tuple[str, str]:
        if suffix == "sim.zlib":
            raise OSError("simulator write failed")
        return durable_file(directory, payload, suffix)

    monkeypatch.setattr(exp, "_durable_file", fail_simulator)
    with pytest.raises(OSError, match="simulator write failed"):
        exp.write_checkpoint(snapshot, tmp_path)
    assert first.read_bytes() == original
    assert not (tmp_path / "update-00000002.json").exists()
    exp.restore_checkpoint(run, first, PROVENANCE)
    assert run.update == 1


@pytest.mark.integration
def test_native_background_checkpoint_matches_synchronous_next_update(
    tmp_path: Path, native_rollout_data: Path
) -> None:
    policy = ToyPolicy()
    shard = exp.NativeSimulatorShard(
        native_rollout_data, (0, 1), cpu=min(os.sched_getaffinity(0)), device=policy.device, max_frame=0
    )
    try:
        engine = exp.RolloutEngine(((shard,),), policy, sampler(), {0: policy.frozen_copy()})
        engine.initialize()
        run = exp.Trainer(
            engine,
            exp.TrainConfig(
                actor_optimizer="adam",
                critic_warmup_updates=0,
                windows_per_step=4,
                decisions_per_window=16,
                critic_epochs=1,
                snapshot_interval=100,
            ),
        )
        run.run_update()
        synchronous = exp.save_checkpoint(run, tmp_path / "sync", PROVENANCE)
        worker = exp.CheckpointWorker(tmp_path / "async", PROVENANCE)
        try:
            worker.submit(run)
            run.run_update()
        finally:
            worker.close()
        batches, updates = [], []
        for manifest in (synchronous, synchronous, tmp_path / "async/update-00000001.json"):
            exp.restore_checkpoint(run, manifest, PROVENANCE)
            batch, _ = run.run_update()
            batches.append(exp._batch_hash(batch))
            updates.append(exp._update_tensors(run))
        assert len(set(batches)) == 1
        assert exp._compare_resume_updates(*updates).passed
    finally:
        shard.close()
