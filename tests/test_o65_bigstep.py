"""Pilot sampling, partial actor steps, conditioning, and cheap diagnostics."""

import dataclasses
import math
from pathlib import Path

import pytest
import torch
from test_065_loop import PROVENANCE
from test_065_loop import exp
from test_065_loop import sampler
from test_065_loop import trainer
from test_065_loop import transformer_trainer

from hal.models.action_sequence import RETURN_SCALE
from hal.models.replay_actor import ActorTargets


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
        episode = sampling.sample(3, number, (0,))
        assert episode == sampling.sample(3, number, (0,))
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
    run = trainer(groups=1, anchor_rate=0)
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
            tuple(p.detach() for p in old.distributions),
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
            run.engine.sampler.sample(int(m), int(ep), (0,)).players[int(side)].percentile
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
    run = trainer(groups=1, length=300, anchor_rate=0)
    run.run_update()
    before = dict(run.episode_metrics)
    checkpoint = exp.save_checkpoint(run, tmp_path, PROVENANCE)
    expected, metrics = run.run_update()
    restored = trainer(groups=1, length=300, anchor_rate=0)
    exp.restore_checkpoint(restored, checkpoint, PROVENANCE)
    assert restored.episode_metrics == before
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
