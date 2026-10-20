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
        zeros = torch.zeros(n, 3, device=self.device)
        features = exp.CriticFeatures(
            trunk,
            trunk,
            zeros,
            zeros,
            torch.zeros(n, 24, 4, device=self.device, dtype=torch.long),
            torch.zeros(n, device=self.device, dtype=torch.long),
        )
        return exp.PolicyEvaluation(actions, logs, tuple(p[..., g, :] for g in exp.DECODE_GROUP_INDICES), features)

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
        states.table.record_evaluation(states.ids, output, self.values(output.features))

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
            slots.condition_WK3,
            slots.opponent_condition_WK3,
            slots.opponent_queue_WKQ4,
            slots.opponent_lengths_WK,
        )
        return ReplayOutput(
            torch.where(mask, logs, 0), tuple(p[..., g, :] for g in exp.DECODE_GROUP_INDICES), mask, features
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
            distributions=tuple(p.detach() for p in output.distributions),
        )

    def values(self, features: Any) -> torch.Tensor:
        return self.model.value_head(features.trunk.detach()).squeeze(-1)


def sampler(*, delay: int = 0, mixed: bool = True, anchor_rate: float = 0.15) -> Any:
    cfg = exp.SamplerConfig(
        (exp.IdentityCount(1, 1, 400), exp.IdentityCount(2, 1, 100), exp.IdentityCount(3, 1, 900)),
        (exp.Matchup((1, 1), 1.0),),
        (25,),
        data_floor=100,
        anchor_rate=anchor_rate,
        deployment_identity=1,
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
    anchor_rate: float = 0.15,
    development: bool = False,
) -> Any:
    torch.manual_seed(19)
    policy = ToyPolicy(device)
    sampling = sampler(delay=delay, anchor_rate=anchor_rate)
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
        snapshot_interval=2,
        snapshot_capacity=2,
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
    sampling = sampler(anchor_rate=0)
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
    shards = (FakeShard((0,), length=67, timeout=True, device=device),)
    engine = exp.RolloutEngine((shards,), policy, sampling, {0: policy.frozen_copy()})
    engine.initialize()
    config = exp.TrainConfig(
        deterministic=deterministic,
        actor_optimizer="adam",
        actor_lr=1e-4,
        critic_epochs=1,
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
        meta = batch.rollout.table.integers[batch.rollout.ids]
        assert set(map(tuple, meta[:, [exp.DELAY, exp.REPLAN]].tolist())) == {(8, 4), (10, 2)}
    manifest = exp.save_checkpoint(run, tmp_path, PROVENANCE)
    expected, _ = run.run_update()
    weights = copy.deepcopy(run.policy.model.state_dict())
    resumed = trainer(groups=2, length=128, anchor_rate=0.15, development=True)
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
