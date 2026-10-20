"""Dense decision selection and stored behavior contracts."""

import dataclasses
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
    store = exp.FrameStore(1, 6, torch.device(device))
    metadata = []
    for match, windows in enumerate((4, 1, 3, 2, 4, 1)):
        for window in range(windows):
            window_rows = []
            for clock in range(0, 32, 2):
                row = [0] * 18
                row[exp.STREAM] = 2 * match + window % 2
                row[exp.MATCH] = match
                row[exp.SIDE] = window % 2
                row[exp.BLOCK] = window // 2 * 128
                row[exp.EPISODE] = window // 2
                row[exp.CLOCK] = window // 2 * 128 + clock
                row[exp.REPLAN] = 4
                row[exp.LENGTH] = 1
                window_rows.append(row)
            metadata.append(window_rows)
    meta = torch.tensor(metadata, device=device)
    rows = torch.arange(meta.shape[0] * 16, device=device).reshape(-1, 16)
    table = exp.DenseRollout(128, store)
    table.integers[: rows.numel()].copy_(meta.flatten(0, 1))
    eligible = rows % 5 != 0
    eligible[1] = False
    table.valid[: rows.numel()].copy_(eligible.flatten())
    batch = exp.replay_batch(trainer().policy, table, rows, eligible)
    placement = exp.place_windows(batch, exp.CounterRng(65), 7, 5, windows_per_step)
    assert not bool((placement.selected & ~eligible).any())
    placed = placement.windows[placement.windows >= 0]
    assert len(placed) == int(placement.selected.any(-1).sum())
    assert len(placed.unique()) == len(placed)
    assert bool((placement.selected.sum(-1)[placement.selected.any(-1)] == 5).all())
    for step in placement.windows:
        live = step >= 0
        chosen = step[live]
        matches = meta[chosen, 0, exp.MATCH]
        assert len(matches.unique()) == len(chosen) <= windows_per_step
        padded = batch.take(step, placement.positions, live)
        assert padded.rows.shape == (windows_per_step, 5)
        assert not bool(padded.slots.valid_WK[~live].any())
    repeated = exp.place_windows(batch, exp.CounterRng(65), 7, 5, windows_per_step)
    for name in ("selected", "positions", "windows"):
        torch.testing.assert_close(getattr(placement, name), getattr(repeated, name), rtol=0, atol=0)


def test_prepare_uses_behavior_outputs_without_student_scoring() -> None:
    run = trainer(groups=1, anchor_rate=0)
    rollout = run.engine.collect(0)
    original = run.policy.evaluate_values
    calls = []

    def bootstrap_only(batch: Any) -> Any:
        calls.append(batch.slots.valid_WK.shape[1])
        return original(batch)

    def reject_student_scoring(batch: Any) -> Any:
        raise AssertionError("preparation must use the stored behavior evaluation")

    run.policy.evaluate_values = bootstrap_only
    run.policy.evaluate_reference = reject_student_scoring
    batch = run.prepare(rollout)
    assert calls and all(slots == 1 for slots in calls)
    torch.testing.assert_close(batch.old.log_probs, batch.decisions.behavior_log_probs_JF, rtol=0, atol=0)
    assert "numerics/mu_old_gap_max" not in batch.metrics


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
    table = rollout.table
    players = 4
    valid = table.valid[: 128 * players].reshape(128, players)
    meta = table.integers[: 128 * players].reshape(128, players, 18)
    expected = meta[..., exp.CLOCK] % meta[..., exp.REPLAN] == 0
    expected &= (meta[..., exp.CLOCK] % 128) < 13
    torch.testing.assert_close(valid, expected)
    ids = rollout.ids
    remaining = 13 - table.integers[ids, exp.CLOCK] % 128
    torch.testing.assert_close(table.remaining[ids], remaining)
    torch.testing.assert_close(table.duration[ids], torch.minimum(table.integers[ids, exp.REPLAN], remaining))
    batch = run.prepare(rollout)
    expected_frames = (
        batch.decisions.duration_J * 0 + 13 - batch.decisions.decision_frame_J % 128 - batch.decisions.delay_J
    ).clamp_min(0)
    expected_frames = torch.minimum(expected_frames, batch.decisions.replan_J)
    torch.testing.assert_close(batch.decisions.executed_factors_JF.sum(-1), 4 * expected_frames)


def test_final_timeout_bootstrap_precedes_reset() -> None:
    run = trainer(groups=1, length=256, timeout=True, delay=8, anchor_rate=0)
    rollout = run.engine.collect(0)
    table = rollout.table
    for stream in range(4):
        ids = rollout.ids[table.integers[rollout.ids, exp.STREAM] == stream]
        row = ids[-1]
        next_row = table.following[row]
        assert table.boundary[row] and not table.terminal[row]
        assert table.integers[next_row, exp.EPISODE] == 0
        packed = exp.TickInputs(table, next_row.reshape(1)).packed(256)
        raw = packed[0].cpu().numpy().reshape(-1).view(exp.native.FRAME_DTYPE)
        assert raw[255]["frame_id"] == 256
        assert run.engine.matches[stream // 2].episode.number == 1
        assert run.engine.observed_clocks[0][stream // 2] == 256


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
    table = rollout.table
    trace = np.stack(shard.trace)
    for row in rollout.ids.tolist():
        clock, side = table.integers[row, [exp.CLOCK, exp.SIDE]].tolist()
        expected = policy.device_wire_actions(table.actions[row, :2]).numpy()
        np.testing.assert_array_equal(trace[clock : clock + 2, 0, side], expected)


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
        def step(self, inputs: torch.Tensor, active: NDArray[np.bool_]) -> tuple[tuple[Any, ...], ...]:
            if self.ticks[0] == 0:
                simulation_started.set()
                if not second_buffer_sampled.wait(5):
                    raise RuntimeError("second buffer did not sample while the first shard waited")
            return super().step(inputs, active)

    class CheckingPolicy(ToyPolicy):
        def sample_tick(self, states: Any, rng: Any, layout: Any, ring: Any, *, clock: int = 0) -> None:
            if (
                int(states.table.integers[states.ids[0], exp.LANE]) == 1
                and int(states.table.integers[states.ids[0], exp.CLOCK]) == 0
            ):
                if not simulation_started.wait(5):
                    raise RuntimeError("first shard did not start before second-buffer inference")
                second_buffer_sampled.set()
            super().sample_tick(states, rng, layout, ring, clock=clock)

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

    base = transformer_trainer("cuda", development=True)
    shard_type = type(base.engine.groups[0][0])
    shards = (
        shard_type((0,), length=67, timeout=True, device="cuda"),
        shard_type((1,), length=67, timeout=True, device="cuda"),
    )
    base.engine.sampler.config = dataclasses.replace(base.engine.sampler.config, anchor_rate=1)
    engine = exp.RolloutEngine((shards,), base.policy, base.engine.sampler, {0: base.policy.frozen_copy()})
    engine.initialize()
    run = exp.Trainer(engine, base.config)
    buffers = engine.buffers[0]
    assert len(buffers) == 2 and buffers[0].stream != buffers[1].stream
    batch, _ = run.run_update()
    assert batch.rollout.timeouts == 4
    table, ids = batch.rollout.table, batch.rollout.ids
    torch.testing.assert_close(table.remaining[ids], 67 - table.integers[ids, exp.CLOCK] % 128)
    with pytest.raises(ValueError, match="clip epsilon"):
        run.policy.actor_step(batch.windows, None, clip_epsilon=0.3)
    exp.verify_checkpoint_resume(run, tmp_path, PROVENANCE)
    assert run.update == 1


@pytest.mark.parametrize("timeout", [False, True])
def test_bootstrap_only_runs_occupied_boundary_roles(timeout: bool) -> None:
    run = trainer(groups=1, anchor_rate=0, length=67, timeout=timeout, delay=8)
    rollout = run.engine.collect(0)
    original = run.policy.evaluate_values
    calls: list[tuple[int, int]] = []

    def values(batch: Any) -> Any:
        calls.append(tuple(batch.slots.valid_WK.shape))
        return original(batch)

    run.policy.evaluate_values = values
    run.prepare(rollout)
    assert calls == ([(4, 1), (4, 1)] if timeout else [])


@pytest.mark.parametrize("field", ["stream", "episode", "clock", "written"])
def test_prepare_rejects_invalid_next_state(field: str) -> None:
    run = trainer(groups=1, anchor_rate=0, length=1000000)
    rollout = run.engine.collect(0)
    table = rollout.table
    row = rollout.ids[0]
    following = table.following[row]
    if field == "written":
        table.valid[following] = False
    else:
        column = {"stream": exp.STREAM, "episode": exp.EPISODE, "clock": exp.CLOCK}[field]
        table.integers[following, column] += 1
    with pytest.raises(ValueError, match="decision transition"):
        run.prepare(rollout)


@pytest.mark.parametrize("field", ["counter_rng", "counter_rng_type", "clocks", "cursor", "sampler_cursor"])
def test_restore_rejects_invalid_progress_before_model_mutation(field: str, tmp_path: Path) -> None:
    import io
    import json

    from test_065_loop import PROVENANCE

    run = trainer(groups=1, anchor_rate=0)
    run.run_update()
    manifest = exp.save_checkpoint(run, tmp_path, PROVENANCE)
    raw = json.loads(manifest.read_text())
    shard = next(shard for shard in raw["shards"] if shard["role"] == "training")
    saved = torch.load(tmp_path / shard["name"], weights_only=False)
    if field == "counter_rng_type":
        saved["counter_rng"]["seed"] = 65.0
    else:
        saved[field] = {"seed": 65, "key": 0} if field == "counter_rng" else ([0] if field == "clocks" else True)
    payload = io.BytesIO()
    torch.save(saved, payload)
    shard["name"], shard["sha256"] = exp._durable_file(tmp_path, payload.getvalue(), "rank0.pt")
    manifest.write_text(json.dumps(raw))
    before = exp._state_hash(run.policy.model.state_dict())
    with pytest.raises(ValueError, match="RNG key|progress"):
        exp.restore_checkpoint(run, manifest, PROVENANCE)
    assert exp._state_hash(run.policy.model.state_dict()) == before


def test_reused_rollout_clears_reward_padding_before_metrics() -> None:
    run = trainer(groups=1, anchor_rate=0, length=13)
    first = run.engine.collect(0)
    first.table.rewards.fill_(123)
    following = run.engine.collect(1)
    table, ids = following.table, following.ids
    live = torch.arange(4)[None] < table.duration[ids, None]
    assert not bool(table.rewards[ids][~live].any())
    prepared = run.prepare(following)
    expected = torch.where(live, table.rewards[ids], 0).sum(-1).mean()
    assert prepared.metrics["return/interval_reward"] == pytest.approx(float(expected))


@pytest.mark.parametrize("count", [3, 16])
def test_reference_cohorts_preserve_slots_and_frozen_parameters(count: int) -> None:
    from test_065_loop import transformer_trainer

    run = transformer_trainer("cpu", development=True, decoder_precision="fp32")
    rollout = run.engine.collect(0)
    table = rollout.table
    rows = torch.arange(table.ticks * table.players).reshape(-1, 64, table.players).transpose(1, 2).reshape(-1, 64)
    windows = exp.replay_batch(run.policy, table, rows, table.valid[rows])
    positions = torch.stack([torch.randperm(64)[:count] for _ in range(len(rows))])
    expected = run.reference.window_forward(windows.inputs(), grad=False)
    order = torch.tensor([[2, 0, -1]]) if count == 16 else torch.empty(0, 3, dtype=torch.long)
    actual = run.reference.evaluate_reference(windows, positions, order)
    torch.testing.assert_close(actual.log_probs_WKF, expected.log_probs_WKF, rtol=3e-5, atol=5e-6)
    torch.testing.assert_close(actual.factor_mask_WKF, expected.factor_mask_WKF, rtol=0, atol=0)
    for first, second in zip(actual.distributions, expected.distributions, strict=True):
        torch.testing.assert_close(first, second, rtol=3e-5, atol=5e-6)
        assert not first.requires_grad
    assert all(not p.requires_grad and p.grad is None for p in run.reference.model.parameters())
    scoring = run.reference.reference_forward_model
    assert scoring is not None
    assert all(p.grad is None for p in scoring.parameters())
    for original, copied in zip(run.reference.model.parameters(), scoring.parameters(), strict=True):
        assert original.data_ptr() != copied.data_ptr()
        torch.testing.assert_close(original, copied, rtol=0, atol=0)
    with pytest.raises(ValueError, match="distinct slots"):
        run.reference.evaluate_reference(windows, torch.zeros(len(rows), 2, dtype=torch.long), order)
    with pytest.raises(ValueError, match="actor shape"):
        run.reference.evaluate_reference(windows, positions, torch.zeros(2, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="distinct windows"):
        run.reference.evaluate_reference(windows, positions, torch.zeros(2, 3, dtype=torch.long))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["reward", "value", "bootstrap", "overflow"])
def test_dense_gae_rejects_nonfinite_training_values(value: float, field: str) -> None:
    rewards = torch.ones(2, 2)
    values = torch.zeros_like(rewards)
    following = torch.zeros_like(rewards)
    duration = torch.full((2, 2), 2, dtype=torch.long)
    valid = torch.ones(2, 2, dtype=torch.bool)
    trace = valid.clone()
    trace[-1] = False
    if field == "overflow":
        rewards.fill_(torch.finfo(torch.float32).max)
    else:
        {"reward": rewards, "value": values, "bootstrap": following}[field][0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        exp.dense_advantages(rewards, values, following, duration, valid, trace, valid)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -0.1])
@pytest.mark.parametrize("reference", [False, True])
def test_kl_rejects_invalid_executed_probabilities(value: float, reference: bool) -> None:
    run = trainer(groups=1, anchor_rate=0)
    batch = run.prepare(run.engine.collect(0))
    policy = batch.old
    target = batch.reference
    probabilities = target.distributions if reference else policy.distributions
    row, frame = torch.nonzero(batch.decisions.executed_factors_JF[:, ::4])[0]
    probabilities[0][row, frame, 0] = value
    with pytest.raises(ValueError, match="executed KL probabilities"):
        exp.factor_kl(policy, target, batch.decisions.executed_factors_JF)
