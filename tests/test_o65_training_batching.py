"""Keep gathered actor windows immutable across captured forward and backward."""

import copy
import dataclasses
import math
from pathlib import Path

import pytest
import torch
from test_065_loop import PROVENANCE
from test_065_loop import assert_batch_equal
from test_065_loop import exp
from test_065_loop import transformer_trainer

from hal.inference.cuda_graph import prepared_shape_compilation


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_actor_step_reuses_windows_without_writes_or_gradient_changes(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    run = transformer_trainer(device, decoder_precision="bf16" if device == "cuda" else "fp32")
    batch = run.prepare(run.engine.collect(0))
    original = run.policy.actor_step
    checked = False

    def compare(
        windows: exp.ReplayBatch, targets: exp.ActorTargets, *, clip_epsilon: float
    ) -> tuple[exp.ActorLoss, exp.ReplayOutput]:
        nonlocal checked
        if checked:
            return original(windows, targets, clip_epsilon=clip_epsilon)
        checked = True
        ids = batch.steps[0]
        fresh = batch.windows.take(ids, batch.placement.positions, ids >= 0)
        before = tuple(value.clone() for value in windows.inputs())
        actual, actual_output = original(windows, targets, clip_epsilon=clip_epsilon)
        gradients = tuple(parameter.grad.clone() for parameter in run.policy.actor_parameters())
        expected, expected_output = original(fresh, targets, clip_epsilon=clip_epsilon)
        for field in dataclasses.fields(actual):
            torch.testing.assert_close(getattr(actual, field.name), getattr(expected, field.name), rtol=0, atol=0)
        torch.testing.assert_close(actual_output.log_probs_WKF, expected_output.log_probs_WKF, rtol=0, atol=0)
        for gradient, parameter in zip(gradients, run.policy.actor_parameters(), strict=True):
            torch.testing.assert_close(parameter.grad, gradient, rtol=1e-5, atol=1e-7)
        for value, saved in zip(windows.inputs(), before, strict=True):
            torch.testing.assert_close(value, saved, rtol=0, atol=0)
        return expected, expected_output

    monkeypatch.setattr(run.policy, "actor_step", compare)
    run.fit(batch)
    assert checked


@pytest.mark.parametrize("device,count", [("cpu", 3), ("cpu", 16), ("cuda", 16)])
def test_grouped_reference_matches_padded_replay_for_every_valid_decision(device: str, count: int) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    run = transformer_trainer(device, development=True, decoder_precision="bf16" if device == "cuda" else "fp32")
    batch = run.prepare(run.engine.collect(0))
    windows = batch.windows
    reference = run.reference
    reference.reference_forward_model = copy.deepcopy(reference.model).requires_grad_(True)
    positions = torch.arange(64, device=device)[None].expand(len(windows.rows), -1)
    with torch.no_grad(), prepared_shape_compilation(64):
        expected = reference.reference_forward(windows.inputs(), len(windows.rows), 16)
        # A partial order must also score all other windows and all unsampled decisions.
        actual = reference.evaluate_reference(windows, positions[:, :count], torch.tensor([[1, -1, 0]], device=device))
    mask = expected.factor_mask_WKF
    assert int(mask.sum()) > 0
    torch.testing.assert_close(actual.factor_mask_WKF, mask)
    tolerance = {"rtol": 0.02, "atol": 0.02} if device == "cuda" else {"rtol": 1e-4, "atol": 1e-5}
    torch.testing.assert_close(actual.log_probs_WKF[mask], expected.log_probs_WKF[mask], **tolerance)
    frame_mask = mask.reshape(*mask.shape[:2], 4, 4)
    for group, (p, q) in enumerate(zip(actual.distributions, expected.distributions, strict=True)):
        torch.testing.assert_close(p[frame_mask[..., group]], q[frame_mask[..., group]], **tolerance)
    # Reward KL consumes every live slot, not only the 16 actor samples.
    assert torch.equal(actual.factor_mask_WKF.any(-1), windows.slots.valid_WK & windows.slots.executed_WKF.any(-1))


def test_grouped_reference_covers_all_production_shapes_and_partial_groups() -> None:
    run = transformer_trainer("cpu", decoder_precision="fp32")
    windows = run.prepare(run.engine.collect(0)).windows
    reference = run.reference
    schedule = exp.production_config(tuple(range(1, 29)))
    reference.schedule = schedule
    reference.model.ranges = exp.ConditioningRanges((math.log(0.01), math.log(0.02)), (16, 21), (2, 4))
    ids = torch.zeros(len(schedule.configurations), dtype=torch.long)
    positions = torch.arange(64)[None].expand(len(windows.rows), -1)
    windows = windows.take(ids, positions, ids >= 0)
    delay = torch.tensor([c.delay for c in schedule.configurations])[:, None].expand(-1, 64)
    replan = torch.tensor([c.replan for c in schedule.configurations])[:, None].expand(-1, 64)
    valid = torch.arange(64)[None] < 128 // replan
    executed = (torch.arange(4)[None, None, :] < replan[..., None]).repeat_interleave(4, -1)
    slots = dataclasses.replace(
        windows.slots,
        valid_WK=valid,
        delay_WK=delay,
        replan_WK=replan,
        condition_WK4=reference.model.ranges.normalize(
            torch.full_like(delay, math.log(0.01), dtype=torch.float32), delay, replan
        ),
        forced_queue_WKD4=torch.zeros(len(ids), 64, 21, 4, dtype=torch.long),
        executed_WKF=executed,
    )
    windows = dataclasses.replace(windows, slots=slots)
    reference.reference_forward_model = copy.deepcopy(reference.model).requires_grad_(True)
    with torch.no_grad():
        expected = reference.reference_forward(windows.inputs(), len(ids), 16)
        # Each configuration has one window and pads to a three-window graph.
        actual = reference.evaluate_reference(windows, positions[ids, :16], torch.tensor([[4, -1, 2]]))
    mask = expected.factor_mask_WKF
    torch.testing.assert_close(actual.factor_mask_WKF, mask)
    torch.testing.assert_close(actual.log_probs_WKF[mask], expected.log_probs_WKF[mask], rtol=1e-4, atol=1e-5)
    for group, (p, q) in enumerate(zip(actual.distributions, expected.distributions, strict=True)):
        active = mask.reshape(len(ids), 64, 4, 4)[..., group]
        torch.testing.assert_close(p[active], q[active], rtol=1e-4, atol=1e-5)


def test_grouping_rejects_a_window_with_mixed_configuration() -> None:
    run = transformer_trainer("cpu", decoder_precision="fp32")
    windows = run.prepare(run.engine.collect(0)).windows
    delay = windows.slots.delay_WK.clone()
    live = torch.nonzero(windows.slots.valid_WK[0]).flatten()
    delay[0, live[-1]] += 1
    windows = dataclasses.replace(windows, slots=dataclasses.replace(windows.slots, delay_WK=delay))
    with pytest.raises(ValueError, match="one delay and replan"):
        exp.window_configurations(windows, run.policy.schedule.configurations)


def test_grouped_reference_preserves_an_empty_update() -> None:
    run = transformer_trainer("cpu", decoder_precision="fp32")
    windows = run.prepare(run.engine.collect(0)).windows
    windows = dataclasses.replace(
        windows, slots=dataclasses.replace(windows.slots, valid_WK=torch.zeros_like(windows.slots.valid_WK))
    )
    positions = torch.arange(16)[None].expand(len(windows.rows), -1)
    actual = run.reference.evaluate_reference(windows, positions, torch.tensor([[-1, -1, -1]]))
    assert actual.log_probs_WKF.shape == (len(windows.rows), 64, 16)
    assert not bool(actual.factor_mask_WKF.any())
    assert not bool(actual.log_probs_WKF.any())


def test_all_decisions_preserves_windows_optimizer_steps_and_microbatch_updates() -> None:
    run = transformer_trainer("cpu", development=True, decoder_precision="fp32")
    run.engine.groups[0][0].length = 10000
    rollout = run.engine.collect(0)
    sampled = run.prepare(rollout)
    run.config = dataclasses.replace(run.config, decisions_per_window="all")
    batch = run.prepare(rollout)
    eligible = batch.windows.slots.valid_WK & batch.windows.slots.executed_WKF.any(-1)
    assert torch.equal(batch.placement.selected, eligible)
    assert int(batch.selected.sum()) > int(sampled.selected.sum())
    torch.testing.assert_close(batch.placement.windows, sampled.placement.windows, rtol=0, atol=0)
    for configuration, members in exp.window_configurations(batch.windows, run.policy.schedule.configurations):
        counts = eligible[members].sum(-1)
        assert bool((counts == 128 // configuration.replan).all())
    state = copy.deepcopy(run.policy.model.state_dict())
    actor = copy.deepcopy(run.actor_optimizer.state_dict())
    critic = copy.deepcopy(run.critic_optimizer.state_dict())
    scheduler = copy.deepcopy(run.scheduler.state_dict())
    steps = run.actor_steps
    run.fit(batch)
    whole = copy.deepcopy(run.policy.model.state_dict())
    assert run.actor_steps - steps == len(batch.steps)
    run.policy.model.load_state_dict(state)
    run.actor_optimizer.load_state_dict(actor)
    run.critic_optimizer.load_state_dict(critic)
    run.scheduler.load_state_dict(scheduler)
    run.config = dataclasses.replace(run.config, actor_microbatches=2)
    steps = run.actor_steps
    run.fit(batch)
    assert run.actor_steps - steps == len(batch.steps)
    for name, value in run.policy.model.state_dict().items():
        torch.testing.assert_close(value, whole[name], rtol=1e-5, atol=1e-7, msg=name)


@pytest.mark.parametrize("decisions", [16, "all"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_grouped_actor_matches_padded_gradients_for_mixed_shapes(decisions: int | str, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    run = transformer_trainer(device, development=True, decoder_precision="bf16" if device == "cuda" else "fp32")
    windows = run.prepare(run.engine.collect(0)).windows
    source = int(torch.nonzero(windows.slots.valid_WK.any(-1))[0, 0])
    ids = torch.full((4,), source, dtype=torch.long, device=device)
    positions = torch.arange(64, device=device)[None].expand(len(windows.rows), -1)
    windows = windows.take(ids, positions, ids >= 0)
    delay = torch.tensor([8, 10, 8, 10], device=device)[:, None].expand(-1, 64)
    replan = torch.tensor([4, 2, 4, 2], device=device)[:, None].expand(-1, 64)
    valid = (replan == 2) | (torch.arange(64, device=device)[None] % 2 == 0)
    # One missing decision and an entirely invalid window exercise both masks.
    valid = valid.clone()
    valid[0, -2] = False
    valid[-1] = False
    slots = dataclasses.replace(
        windows.slots,
        delay_WK=delay,
        replan_WK=replan,
        valid_WK=valid,
        condition_WK4=run.policy.model.ranges.normalize(
            torch.full_like(delay, math.log(0.01), dtype=torch.float32), delay, replan
        ),
        executed_WKF=(torch.arange(4, device=device)[None, None] < replan[..., None]).repeat_interleave(4, -1),
    )
    windows = dataclasses.replace(windows, slots=slots)
    placement = exp.place_windows(windows, run.engine.sampler.rng, 0, decisions, 4)
    windows = windows.take(
        torch.arange(4, device=device), placement.positions, torch.ones(4, dtype=torch.bool, device=device)
    )
    selected = placement.selected.gather(1, placement.positions)
    with torch.no_grad(), prepared_shape_compilation(64):
        initial = run.policy.window_forward(windows.inputs(), grad=False)
    reference = []
    for p in initial.distributions:
        q = p * torch.linspace(0.9, 1.1, p.shape[-1], device=device)
        reference.append(q / q.sum(-1, keepdim=True))
    targets = exp.ActorTargets(
        initial.log_probs_WKF - 0.005,
        torch.randn(selected.shape, device=device) + 0.4,
        selected,
        torch.full(selected.shape, 0.1, device=device),
        tuple(reference),
        torch.tensor(0.0, device=device),
        torch.tensor(1.0, device=device),
        selected.sum().float(),
    )
    expected, expected_output = run.policy.actor_step(windows, targets, clip_epsilon=0.2)
    gradients = tuple(torch.zeros_like(p) if p.grad is None else p.grad.clone() for p in run.policy.actor_parameters())
    before = tuple(value.clone() for value in windows.inputs())
    actual, actual_output = run.policy.grouped_actor_step(windows, targets, windows_per_chunk=3, clip_epsilon=0.2)
    tolerance = {"rtol": 0.05, "atol": 0.005} if device == "cuda" else {"rtol": 1e-4, "atol": 1e-6}
    for field in dataclasses.fields(actual):
        torch.testing.assert_close(getattr(actual, field.name), getattr(expected, field.name), **tolerance)
    mask = expected_output.factor_mask_WKF
    torch.testing.assert_close(actual_output.factor_mask_WKF, mask)
    torch.testing.assert_close(actual_output.log_probs_WKF[mask], expected_output.log_probs_WKF[mask], **tolerance)
    control = torch.cat([g.flatten() for g in gradients])
    candidate = torch.cat(
        [(torch.zeros_like(p) if p.grad is None else p.grad).flatten() for p in run.policy.actor_parameters()]
    )
    if device == "cuda":
        assert all(graph.gradients is run.policy.actor_gradients for graph in run.policy.actor_graphs.values())
        assert float(torch.nn.functional.cosine_similarity(candidate, control, dim=0)) > 0.995
        assert float((candidate - control).norm() / control.norm()) < 0.05
    else:
        torch.testing.assert_close(candidate, control, rtol=1e-4, atol=1e-6)
    for actual_input, saved in zip(windows.inputs(), before, strict=True):
        torch.testing.assert_close(actual_input, saved, rtol=0, atol=0)


@pytest.mark.parametrize("windows,microbatches,chunk", [(512, 1, 64), (512, 8, 64), (96, 1, 64), (4, 2, 2)])
def test_actor_chunk_cap_preserves_the_configured_optimizer_batch(windows: int, microbatches: int, chunk: int) -> None:
    config = exp.TrainConfig(windows_per_step=windows, actor_microbatches=microbatches)
    assert config.windows_per_chunk == chunk
    assert config.windows_per_step == windows


@pytest.mark.parametrize("decisions", [0, 17, True, "unsupported"])
def test_decision_mode_rejects_invalid_values(decisions: object) -> None:
    with pytest.raises(ValueError, match="decisions per window"):
        exp.TrainConfig(decisions_per_window=decisions)


def test_all_decision_resume_restores_next_batch_and_optimizer_update(tmp_path: Path) -> None:
    run = transformer_trainer("cpu", development=True, decoder_precision="fp32")
    run.config = dataclasses.replace(run.config, decisions_per_window="all", actor_microbatches=2)
    run.run_update()
    manifest = exp.save_checkpoint(run, tmp_path, PROVENANCE)
    expected, _ = run.run_update()
    state = exp._state_hash(
        (
            run.policy.model.state_dict(),
            run.actor_optimizer.state_dict(),
            run.critic_optimizer.state_dict(),
            run.scheduler.state_dict(),
        )
    )
    resumed = transformer_trainer("cpu", development=True, decoder_precision="fp32")
    resumed.config = dataclasses.replace(resumed.config, decisions_per_window="all", actor_microbatches=2)
    exp.restore_checkpoint(resumed, manifest, PROVENANCE)
    actual, _ = resumed.run_update()
    assert_batch_equal(expected, actual)
    assert state == exp._state_hash(
        (
            resumed.policy.model.state_dict(),
            resumed.actor_optimizer.state_dict(),
            resumed.critic_optimizer.state_dict(),
            resumed.scheduler.state_dict(),
        )
    )


def test_empty_local_actor_chunk_has_zero_gradient_with_a_global_denominator() -> None:
    run = transformer_trainer("cpu", development=True, decoder_precision="fp32")
    windows = run.prepare(run.engine.collect(0)).windows
    windows = dataclasses.replace(
        windows, slots=dataclasses.replace(windows.slots, valid_WK=torch.zeros_like(windows.slots.valid_WK))
    )
    with torch.no_grad():
        output = run.policy.window_forward(windows.inputs(), grad=False)
    zero = torch.zeros_like(windows.slots.valid_WK, dtype=torch.float32)
    targets = exp.ActorTargets(
        output.log_probs_WKF,
        zero,
        windows.slots.valid_WK,
        zero,
        output.distributions,
        zero.new_zeros(()),
        zero.new_ones(()),
        zero.new_tensor(128.0),
    )
    loss, current = run.policy.grouped_actor_step(windows, targets, windows_per_chunk=3, clip_epsilon=0.2)
    assert float(loss.loss) == 0
    assert not bool(current.factor_mask_WKF.any())
    assert bool((loss.joint_ratio_WK == 1).all())
    assert all(p.grad is None or not bool(p.grad.any()) for p in run.policy.actor_parameters())
