"""Dense decision selection and stored behavior contracts."""

import dataclasses
from collections import Counter

import numpy as np
import pytest
import torch
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

    def bootstrap_only(states: object, actions: torch.Tensor) -> object:
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
