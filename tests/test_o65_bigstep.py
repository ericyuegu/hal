"""Pilot sampling, partial actor steps, conditioning, and cheap diagnostics."""

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from test_065_loop import PROVENANCE
from test_065_loop import exp
from test_065_loop import sampler
from test_065_loop import trainer
from test_065_loop import transformer_trainer

from hal.models.action_sequence import RETURN_SCALE
from hal.models.replay_actor import ActorTargets


@dataclasses.dataclass(frozen=True, slots=True)
class HeadStartRng(exp.CounterRng):
    seed: int
    offsets: tuple[int, ...] = (0, 1, 127, 1200)
    maximum: int = 1200

    def uniform(self, domain: str, stream: int, episode: int, frame: int, offset: int = 0, factor: int = 0) -> float:
        if domain == "head_start":
            return (self.offsets[stream] + 0.5) / (self.maximum + 1)
        return exp.CounterRng.uniform(self, domain, stream, episode, frame, offset, factor)


def test_head_start_preserves_offsets_queues_and_next_update_on_resume(tmp_path: Path) -> None:
    run = trainer(groups=2, development=True)
    run.config = dataclasses.replace(run.config, head_start_frames=1200)
    run.engine.sampler.rng = HeadStartRng(65)
    before = exp._state_hash(run.policy.model.state_dict())
    result = run.head_start()
    assert result.offsets == (0, 1, 127, 1200)
    assert result.episode_ages == (0, 1, 127, 1200)
    assert run.engine.clocks == [1280, 1280] and run.engine.cursor == 0
    assert run.update == run.actor_steps == 0 and run.engine.completed_update == -1
    assert not run.actor_optimizer.state and not run.critic_optimizer.state
    assert exp._state_hash(run.policy.model.state_dict()) == before
    assert run.partial_episodes == {(stream, 0) for stream in range(2, 8)}
    assert not run.episode_metrics and not run.anchor_win_history
    assert result.metrics()["coverage/head_start_age_max"] == 1200
    for group, matches in zip(run.engine.groups, ((0, 1), (2, 3)), strict=True):
        assert [int(shard.ticks[0]) for shard in group] == [result.offsets[match] for match in matches]
    exp.verify_checkpoint_resume(run, tmp_path, PROVENANCE)
    assert json.loads((tmp_path / "resume-check.json").read_text())["passed"]
    assert run.partial_episodes == {(stream, 0) for stream in range(2, 8)}


@pytest.mark.parametrize("timeout", [False, True])
def test_head_start_handles_ends_and_excludes_partial_monte_carlo_episodes(timeout: bool) -> None:
    run = trainer(groups=2, length=129, timeout=timeout, development=True)
    run.engine.sampler.rng = HeadStartRng(65)
    run.config = dataclasses.replace(run.config, head_start_frames=1200)
    result = run.head_start()
    assert all(0 <= age <= 129 for age in result.episode_ages)
    for update in range(4):
        _, metrics = run.run_update()
        assert metrics["critic/monte_carlo_episode_count"] == (0 if timeout else (2 if update < 2 else 4))
    assert not run.partial_episodes


def test_partial_episodes_do_not_report_zero_complete_match_kl_costs() -> None:
    run = trainer(groups=1, length=129, development=True)
    run.engine.sampler.rng = HeadStartRng(65, (127, 127))
    run.config = dataclasses.replace(run.config, head_start_frames=1200)
    run.head_start()
    _, metrics = run.run_update()
    assert metrics["critic/monte_carlo_episode_count"] == 0
    assert "return/completed_match_kl_cost" not in metrics


@pytest.mark.parametrize("maximum", [-1, 1.5, True])
def test_head_start_rejects_invalid_lengths(maximum: int) -> None:
    with pytest.raises(ValueError, match="nonnegative integers"):
        exp.TrainConfig(head_start_frames=maximum)


@pytest.mark.parametrize("offset", [3, 4, 5, 6])
def test_staggered_first_decision_matches_uncached_reference(offset: int) -> None:
    run = transformer_trainer("cpu", development=True, decoder_precision="fp32")
    run.engine.groups[0][0].length = 1000000
    run.engine.sampler.rng = HeadStartRng(65, (offset,), 8)
    run.engine.stagger_starts(8)
    rollout = run.engine.collect(-1)
    batch = run.prepare(rollout)
    assert batch.reference is not None and len(rollout.ids)
    valid = batch.decisions.executed_factors_JF
    torch.testing.assert_close(batch.old.log_probs[valid], batch.reference.log_probs[valid], rtol=1e-5, atol=1e-5)


def test_episode_sampling_matches_character_prior_identity_counts_and_beta() -> None:
    base = sampler()
    config = dataclasses.replace(
        base.config,
        identities=(exp.IdentityCount(1, 1, 100), exp.IdentityCount(3, 1, 300), exp.IdentityCount(2, 22, 100)),
        matchups=(exp.Matchup((1, 1), 1), exp.Matchup((1, 22), 3)),
        stages=(8, 18, 6, 26, 24, 25),
        tempering=1,
        deployment_rate=0,
    )
    sampling = exp.EpisodeSampler(config, base.schedule, base.rng)
    counts = {"mixed": 0, "orientation": 0, "master": 0, "ordinary": 0, "masked": 0}
    stages = dict.fromkeys(config.stages, 0)
    draws = []
    for number in range(4000):
        episode = sampling.sample(3, number, None)
        assert episode == sampling.sample(3, number, None)
        characters = tuple(player.character for player in episode.players)
        counts["mixed"] += 22 in characters
        counts["orientation"] += characters == (22, 1)
        stages[episode.stage] += 1
        for player in episode.players:
            assert player.percent == 0
            assert 0.1 <= player.percentile <= 0.999
            draws.append(player.percentile)
            counts["masked"] += player.identity == 0
            if player.character == 1 and player.identity:
                counts["ordinary"] += 1
                counts["master"] += player.identity == 3
    assert 0.72 < counts["mixed"] / 4000 < 0.78
    assert 0.47 < counts["orientation"] / counts["mixed"] < 0.53
    assert 0.72 < counts["master"] / counts["ordinary"] < 0.78
    assert 0.08 < counts["masked"] / 8000 < 0.12
    assert all(0.14 < value / 4000 < 0.19 for value in stages.values())
    u = torch.tensor(draws).reshape(-1, 2)
    assert 0.81 < float(u.median()) < 0.85
    assert 0.035 < float((u < 0.5).float().mean()) < 0.075
    assert abs(float(torch.corrcoef(u.T)[0, 1])) < 0.06


def test_partial_512_window_step_uses_every_window_once_and_masks_padding() -> None:
    run = trainer(groups=1)
    prepared = run.prepare(run.engine.collect(0))
    source = prepared.windows
    windows = torch.arange(513) % len(source.rows)
    positions = torch.arange(64)[None].expand(len(source.rows), -1)
    batch = source.take(windows, positions, torch.ones(513, dtype=torch.bool))
    meta = batch.metadata.clone()
    meta[..., exp.MATCH] = torch.arange(513)[:, None]
    batch = dataclasses.replace(batch, metadata=meta)
    placement = exp.place_windows(batch, run.engine.sampler.rng, 0, 16, 512)
    used = placement.windows[placement.windows >= 0]
    assert torch.equal(used.sort().values, torch.arange(513))
    assert placement.windows.shape == (2, 512)
    ids = placement.windows[-1]
    live = ids >= 0
    padded = batch.take(ids, placement.positions, live)
    assert not bool(padded.slots.valid_WK[~live].any())
    compact = batch.take(ids[live], placement.positions, torch.ones(int(live.sum()), dtype=torch.bool))
    gradients = []
    for current in (compact, padded):
        selected = current.slots.valid_WK & current.slots.executed_WKF.any(-1)
        old = run.policy.window_output(current)
        advantages = torch.arange(selected.numel()).reshape(selected.shape).float() % 3 - 1
        targets = ActorTargets(
            old.log_probs_WKF.detach(),
            advantages,
            selected,
            torch.zeros_like(advantages),
            tuple(p.detach() for p in old.log_distributions),
            torch.tensor(0.0),
            torch.tensor(1.0),
            selected.sum().float(),
        )
        run.policy.actor_step(current, targets, clip_epsilon=0.2)
        gradients.append(run.policy.model.logits.grad.clone())
    torch.testing.assert_close(*gradients, rtol=1e-5, atol=1e-6)


def test_episode_u_reaches_rollout_reference_replay_and_both_critic_inputs() -> None:
    run = transformer_trainer(development=True, decoder_precision="fp32")
    observed = []

    def capture(_module, args):
        observed.append((args[0].detach().clone(), args[2].detach().clone()))

    handle = run.policy.model.temporal.return_conditioner.register_forward_pre_hook(capture)
    try:
        rollout = run.engine.collect(0)
    finally:
        handle.remove()
    table, ids = rollout.table, rollout.ids
    assert table.evaluation is not None
    meta = table.integers[ids]
    expected = torch.tensor(
        [
            run.engine.sampler.sample(int(m), int(ep), None).players[int(side)].percentile
            for m, ep, side in meta[:, [exp.MATCH, exp.EPISODE, exp.SIDE]]
        ]
    )
    torch.testing.assert_close(table.conditions[ids, 2], expected)
    handle = run.reference.model.temporal.return_conditioner.register_forward_pre_hook(
        lambda _module, args: observed.append((args[0].detach().clone(), args[2].detach().clone()))
    )
    try:
        batch = run.prepare(rollout)
    finally:
        handle.remove()
    assert observed
    for value, condition in observed:
        torch.testing.assert_close(value, condition[..., 3] * RETURN_SCALE)
    features = batch.old.features
    torch.testing.assert_close(features.condition[:, 3], 2 * expected - 1)
    torch.testing.assert_close(features.opponent_condition[:, 3], 2 * table.conditions[ids, 3] - 1)
    assert len(expected.unique()) > 1
    head = run.policy.model.critic_residual
    torch.nn.init.normal_(head.residual.down.weight)
    ego = features.condition.clone().requires_grad_()
    opponent = features.opponent_condition.clone().requires_grad_()
    value = run.policy.model.self_play_value(
        features.trunk, features.decoder, ego, opponent, features.opponent_queue, features.opponent_lengths
    )
    value.sum().backward()
    assert bool(ego.grad[:, 3].abs().sum() > 0)
    assert bool(opponent.grad[:, 3].abs().sum() > 0)


def test_resume_preserves_u_and_episode_metric_totals(tmp_path: Path) -> None:
    run = trainer(groups=1, length=300)
    run.run_update()
    before = dict(run.episode_metrics)
    run.anchor_win_history = [{0: (1, 2), 7: (0, 1)}]
    anchor_history = list(run.anchor_win_history)
    checkpoint = exp.save_checkpoint(run, tmp_path, PROVENANCE)
    expected, metrics = run.run_update()
    restored = trainer(groups=1, length=300)
    exp.restore_checkpoint(restored, checkpoint, PROVENANCE)
    assert restored.episode_metrics == before
    assert restored.anchor_win_history == anchor_history
    actual, resumed_metrics = restored.run_update()
    torch.testing.assert_close(
        actual.rollout.table.conditions[actual.rollout.ids],
        expected.rollout.table.conditions[expected.rollout.ids],
        rtol=0,
        atol=0,
    )
    assert resumed_metrics["return/completed_match_kl_cost"] == metrics["return/completed_match_kl_cost"]
    assert resumed_metrics["critic/monte_carlo_rmse"] == metrics["critic/monte_carlo_rmse"]
    assert restored.episode_metrics == run.episode_metrics
    assert restored.anchor_win_history == run.anchor_win_history


def test_anchor_monitor_counts_active_frames_ports_resets_and_completed_wins() -> None:
    run = trainer(groups=1)
    run.engine.anchors[7] = run.policy.frozen_copy()
    store = exp.FrameStore(16, 2, torch.device("cpu"))
    table = exp.DenseRollout(64, store)
    records = np.zeros((16, 2), dtype=exp.native.FRAME_DTYPE)
    for lane, stocks, percents in (
        (0, [(4, 4), (4, 3), (3, 3), (4, 4), (3, 4)], [(20, 10), (25, 0), (0, 7), (0, 0), (0, 9)]),
        (1, [(4, 4), (4, 4), (3, 4), (3, 4)], [(0, 0), (10, 3), (0, 5), (2, 8)]),
    ):
        for side in range(2):
            for name, values, dtype in (("stock", stocks, np.int32), ("percent", percents, np.float32)):
                records["columns"][: len(values), lane, exp.NATIVE_COLUMN_LAYOUT[f"p{side + 1}_{name}"][0]] = np.array(
                    values, dtype=dtype
                )[:, side].view(np.uint32)
    store.records.copy_(torch.from_numpy(records.view(np.uint8).reshape(16, 2, -1)))
    table.integers[:4, exp.LANE] = torch.tensor([0, 0, 1, 1])
    table.integers[:4, exp.SIDE] = torch.tensor([0, 1, 1, 0])
    table.integers[:4, exp.START] = torch.tensor([0, 3, 0, 0])
    table.integers[:4, exp.LENGTH] = 1
    table.duration[:4] = torch.tensor([2, 1, 3, 3])
    table.opponent_anchor[:4] = torch.tensor([0, 7, 0, -1])
    episode = run.engine.sampler.sample(0, 0, None)

    def completed(anchor: int, anchor_side: int, *, timeout: bool = False):
        players = tuple(
            dataclasses.replace(player, anchor=anchor if side == anchor_side else None)
            for side, player in enumerate(episode.players)
        )
        return exp.EpisodeResult(0, dataclasses.replace(episode, players=players), 300, (0.0, 0.0), timeout, (1, 0))

    rollout = exp.Rollout(
        table, torch.arange(4), 0, 0, 6, 1, 1, 0.0, (completed(0, 1), completed(7, 0), completed(0, 0, timeout=True))
    )
    assert exp.reward_component_metrics(rollout) == {
        "return/damage_positive": 6.75,
        "return/damage_negative": -8.5,
        "return/damage_reward": -1.75,
        "return/stock_positive": 90.0,
        "return/stock_negative": -60.0,
        "return/stock_reward": 30.0,
        "return/win_positive": 0.0,
        "return/win_negative": 0.0,
        "return/win_reward": 0.0,
    }
    metrics = run.anchor_metrics(rollout)
    assert metrics["gameplay/pi_ref_net_stocks_per_min"] == pytest.approx(720)
    assert metrics["gameplay/pi_ref_net_damage_per_min"] == pytest.approx(4320)
    assert metrics["gameplay/anchor7_net_stocks_per_min"] == pytest.approx(3600)
    assert metrics["gameplay/anchor7_net_damage_per_min"] == pytest.approx(-32400)
    assert metrics["gameplay/pi_ref_count"] == metrics["gameplay/pi_ref_win_rate"] == 1
    assert metrics["gameplay/anchor7_count"] == 1 and metrics["gameplay/anchor7_win_rate"] == 0
    second = dataclasses.replace(rollout, results=(completed(0, 0), completed(0, 0)))
    metrics = run.anchor_metrics(second)
    assert metrics["gameplay/pi_ref_win_rate_32_updates"] == pytest.approx(1 / 3)
    empty = dataclasses.replace(rollout, results=())
    for _ in range(30):
        metrics = run.anchor_metrics(empty)
    assert metrics["gameplay/pi_ref_count"] == 0
    assert "gameplay/pi_ref_win_rate" not in metrics
    assert metrics["gameplay/pi_ref_win_rate_32_updates"] == pytest.approx(1 / 3)
    metrics = run.anchor_metrics(empty)
    assert metrics["gameplay/pi_ref_count_32_updates"] == 2
    assert metrics["gameplay/pi_ref_win_rate_32_updates"] == 0
    metrics = run.anchor_metrics(empty)
    assert "gameplay/pi_ref_win_rate_32_updates" not in metrics


def test_gradient_noise_estimate_uses_preclip_gradients_and_window_counts() -> None:
    parameter = torch.nn.Parameter(torch.zeros(1))
    noise = exp.GradientNoise((parameter,))
    for gradient in (1.0, 3.0):
        parameter.grad = torch.tensor([gradient])
        noise.add(4, 64)
        torch.nn.utils.clip_grad_norm_((parameter,), 0.1)
    metrics = noise.metrics()
    assert metrics["actor/gradient_noise_variance"] == 128
    assert metrics["actor/gradient_signal_squared"] == 3
    assert metrics["actor/gradient_noise_scale_windows"] == pytest.approx(8 / 3)
    assert all(math.isfinite(value) for value in metrics.values())


def test_phase_memory_preserves_oom_phase_and_partial_peaks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 123)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 456)
    memory = exp.PhaseMemory(torch.device("cuda"), 512)
    memory.output, memory.active, memory.label = tmp_path, True, "first_ppo"
    with pytest.raises(RuntimeError, match="OOM in actor.*512 windows"), memory.phase("actor"):
        raise torch.cuda.OutOfMemoryError("synthetic allocation failure")
    import json

    peaks = json.loads((tmp_path / "phase-memory.json").read_text())
    assert peaks["memory/first_ppo_actor_peak_allocated_bytes"] == 123
    assert peaks["memory/first_ppo_actor_peak_reserved_bytes"] == 456
    assert not memory.actor_checked
    with memory.phase("actor"):
        pass
    assert memory.actor_checked


def test_gradient_noise_weights_the_actual_selected_decisions() -> None:
    parameter = torch.nn.Parameter(torch.zeros(1))
    noise = exp.GradientNoise((parameter,))
    for gradient, count in ((1.0, 4), (3.0, 12)):
        parameter.grad = torch.tensor([gradient])
        noise.add(4, count)
    metrics = noise.metrics()
    assert metrics["actor/gradient_signal_squared"] == 5.5
    assert metrics["actor/gradient_noise_scale_windows"] == pytest.approx(12 / 5.5 / 2)
