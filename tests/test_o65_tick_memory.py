"""CUDA graph and retained-score storage preserve training inputs."""

import copy
import dataclasses
import weakref
from typing import Any

import pytest
import torch
from test_065_loop import FakeShard
from test_065_loop import exp
from test_065_loop import transformer_trainer


def test_reference_releases_each_decoder_graph_before_the_next_chunk() -> None:
    policy = transformer_trainer("cpu", development=True).reference
    store = exp.FrameStore(256, 1, policy.device)
    table = exp.DenseRollout(128, store)
    rows = torch.zeros(1, 64, dtype=torch.long)
    batch = exp.replay_batch(policy, table, rows, torch.zeros_like(rows, dtype=torch.bool))
    original = policy.window_replay
    outputs: list[weakref.ReferenceType[torch.Tensor]] = []

    def replay(model: Any, hidden: torch.Tensor, observed: torch.Tensor, slots: Any, *, grad: bool) -> Any:
        assert all(output() is None for output in outputs)
        output = original(model, hidden, observed, slots, grad=grad)
        assert output.log_probs_WKF.grad_fn is not None
        outputs.append(weakref.ref(output.log_probs_WKF))
        return output

    policy.window_replay = replay
    positions = torch.arange(16)[None]
    output = policy.evaluate_reference(batch, positions, torch.zeros(1, 1, dtype=torch.long))
    assert len(outputs) == 4
    assert all(item() is None for item in outputs)
    assert output.log_probs_WKF.grad_fn is None


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("count", [3, 16])
def test_reference_padding_preserves_real_windows_and_slot_order(device: str, count: int) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    run = transformer_trainer(device, development=True, decoder_precision="fp32")
    rollout = run.engine.collect(0)
    table = rollout.table
    rows = (
        torch.arange(table.ticks * table.players, device=run.policy.device)
        .reshape(-1, 64, table.players)
        .transpose(1, 2)
        .reshape(-1, 64)
    )
    batch = exp.replay_batch(run.policy, table, rows, table.valid[rows])
    positions = torch.stack([torch.randperm(64, device=run.policy.device)[:count] for _ in rows])
    order = torch.tensor([[2, -1, 0], [-1, -1, -1]], device=run.policy.device)
    actual = run.reference.evaluate_reference(batch, positions, order)
    source = batch
    if device == "cuda":
        # Keep B=3 and K=count for every BF16 trunk/decoder call. A final
        # B=1 call selects different GEMM arithmetic even without compaction.
        columns = torch.arange(64, device=run.policy.device)[None].expand(len(rows), -1)
        columns = torch.nn.functional.pad(columns, (0, (-64) % count))
        source = batch.take(
            torch.tensor([0, 1, 2, 3, -1, -1], device=run.policy.device),
            columns,
            torch.tensor([True, True, True, True, False, False], device=run.policy.device),
        )
        source = dataclasses.replace(
            source,
            slots=dataclasses.replace(
                source.slots,
                valid_WK=source.slots.valid_WK & (torch.arange(columns.shape[1], device=run.policy.device) < 64),
            ),
        )
    with torch.no_grad():
        expected = run.reference.reference_forward(source.inputs(), 3, count)
    expected = exp.ReplayOutput(
        expected.log_probs_WKF[:4, :64],
        tuple(value[:4, :64] for value in expected.distributions),
        expected.factor_mask_WKF[:4, :64],
        exp.ReplayCriticFeatures(
            *(getattr(expected.features, field.name)[:4, :64] for field in dataclasses.fields(expected.features))
        ),
    )
    torch.testing.assert_close(actual.log_probs_WKF, expected.log_probs_WKF, atol=5e-6, rtol=3e-5)
    torch.testing.assert_close(actual.factor_mask_WKF, expected.factor_mask_WKF, atol=0, rtol=0)
    for first, second in zip(actual.distributions, expected.distributions, strict=True):
        torch.testing.assert_close(first, second, atol=5e-6, rtol=3e-5)
    for field in dataclasses.fields(actual.features):
        torch.testing.assert_close(
            getattr(actual.features, field.name), getattr(expected.features, field.name), atol=5e-6, rtol=3e-5
        )


def test_retained_reference_rows_own_only_the_selected_storage() -> None:
    values = torch.arange(5 * 3 * 16, dtype=torch.float32).reshape(5, 3, 16)
    output = exp.ReplayOutput(
        values,
        tuple(values.clone() for _ in range(4)),
        values.bool(),
        exp.ReplayCriticFeatures(*(values.clone() for _ in range(6))),
    )
    for selected in (torch.tensor([True, False, False, True, False]), torch.zeros(5, dtype=torch.bool)):
        retained = exp.take_replay_windows(output, selected)
        for name in ("log_probs_WKF", "factor_mask_WKF"):
            value = getattr(retained, name)
            torch.testing.assert_close(value, getattr(output, name)[selected], atol=0, rtol=0)
            assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
        for value in retained.distributions:
            assert value.shape[0] == int(selected.sum())
            assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
    selected = torch.tensor([True, False, False, True, False])
    retained = exp.take_replay_windows(output, selected)
    expected = retained.log_probs_WKF.clone()
    output.log_probs_WKF.fill_(99)
    torch.testing.assert_close(retained.log_probs_WKF, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rollout_groups_share_streams_without_sharing_match_state() -> None:
    runs = []
    for separate in (False, True):
        base = transformer_trainer("cuda", development=True)
        groups = tuple(
            (FakeShard((2 * group, 2 * group + 1), length=67, timeout=True, device="cuda"),) for group in range(2)
        )
        engine = exp.RolloutEngine(groups, base.policy, base.engine.sampler, {0: base.policy.frozen_copy()})
        exp.Trainer(engine, base.config)
        engine.initialize()
        assert engine.buffers[0][0].stream is engine.buffers[1][0].stream
        if separate:
            buffers = []
            for group in engine.buffers:
                replaced = []
                for buffer in group:
                    updated = dataclasses.replace(buffer, stream=torch.cuda.Stream(device=engine.policy.device))
                    engine.buffer_device_lanes[id(updated)] = engine.buffer_device_lanes.pop(id(buffer))
                    replaced.append(updated)
                buffers.append(tuple(replaced))
            engine.buffers = tuple(buffers)
        runs.append(engine)
    for update in range(4):
        actual, expected = [engine.collect(update) for engine in runs]
        assert exp._state_hash(runs[0].snapshot()) == exp._state_hash(runs[1].snapshot())
        for name in ("actions", "behavior", "values", "valid"):
            torch.testing.assert_close(getattr(actual.table, name), getattr(expected.table, name), atol=0, rtol=0)


class PrivateTickPolicy(exp.TransformerPolicy):
    def frozen_copy(self) -> Any:
        return PrivateTickPolicy(copy.deepcopy(self.model).requires_grad_(False), self.schedule, self.statistics)

    def retain_tick_layouts(self, ring: Any, layouts: Any) -> None:
        self.tick_pools.clear()
        super().retain_tick_layouts(ring, layouts)


def pooled_tick_trainer(*, shared: bool) -> Any:
    base = transformer_trainer("cuda", development=True)
    policy_type = exp.TransformerPolicy if shared else PrivateTickPolicy
    policy = policy_type(base.policy.model, base.policy.schedule, base.policy.statistics)
    groups = tuple(
        tuple(
            FakeShard(
                tuple(range(group * 8 + shard * 4, group * 8 + (shard + 1) * 4)),
                length=67,
                timeout=True,
                device="cuda",
            )
            for shard in range(2)
        )
        for group in range(2)
    )
    engine = exp.RolloutEngine(groups, policy, base.engine.sampler, {0: policy.frozen_copy()})
    config = dataclasses.replace(base.config, critic_warmup_updates=2, snapshot_interval=1)
    run = exp.Trainer(engine, config)
    engine.initialize()
    run.capture_startup()
    return run


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rollout_graph_pools_preserve_updates_across_resets_and_promotions() -> None:
    private = pooled_tick_trainer(shared=False)
    shared = pooled_tick_trainer(shared=True)
    by_buffer: dict[int, set[tuple[int, int]]] = {}
    for policy in (shared.policy, *shared.engine.anchors.values()):
        for (key, _appended, _count), graph in policy.tick_graphs.items():
            buffer, _rows = key[1]
            by_buffer.setdefault(buffer, set()).add(graph.graph.pool())
    assert set(by_buffer) == {0, 1}
    assert all(len(pools) == 1 for pools in by_buffer.values())
    assert len(set.union(*by_buffer.values())) == 2
    assert shared.policy.training_pool.handle not in set.union(*by_buffer.values())
    private_pools = {
        graph.graph.pool()
        for policy in (private.policy, *private.engine.anchors.values())
        for graph in policy.tick_graphs.values()
    }
    assert len(private_pools) == 60
    for update in range(4):
        expected, metrics = private.run_update()
        actual, actual_metrics = shared.run_update()
        assert actual.rollout.resets > 0
        if update < 2:
            assert exp._batch_hash(actual) == exp._batch_hash(expected)
        for name in ("selected", "steps"):
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name), atol=0, rtol=0)
        for name in ("actions", "valid", "integers"):
            torch.testing.assert_close(
                getattr(actual.rollout.table, name), getattr(expected.rollout.table, name), atol=0, rtol=0
            )
        for name in ("values", "advantages", "targets"):
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name), atol=1e-6, rtol=1e-5)
        for name in ("critic/loss", "critic/explained_variance", "policy/kl_per_frame"):
            assert actual_metrics[name] == pytest.approx(metrics[name], abs=1e-7, rel=1e-5)
        for value, other in zip(shared.policy.model.parameters(), private.policy.model.parameters(), strict=True):
            # Parallel backward reductions can differ with graph allocation addresses.
            torch.testing.assert_close(value, other, atol=1e-8, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_shared_reference_bf16_scores_survive_scratch_and_allocator_reuse() -> None:
    run = transformer_trainer("cuda", development=True, decoder_precision="bf16")
    table = run.engine.collect(0).table
    rows = (
        torch.arange(table.ticks * table.players, device=run.policy.device)
        .reshape(-1, 64, table.players)
        .transpose(1, 2)
        .reshape(-1, 64)
    )
    batch = exp.replay_batch(run.policy, table, rows, table.valid[rows])
    positions = torch.arange(16, device=run.policy.device)[None].expand(len(rows), -1)
    order = torch.tensor([[2, -1, 0], [-1, -1, -1]], device=run.policy.device)
    pool = exp.CapturePool(run.policy.device)
    run.reference.training_pool = pool
    expected = run.reference.evaluate_reference(batch, positions, order)
    scratch = torch.zeros(16384, device=run.policy.device)
    other = exp.CapturedCall(lambda: scratch.sin(), (scratch,), device=run.policy.device, pool=pool)
    pool.capture_all()
    for _ in range(3):
        junk = [torch.full((65536,), torch.nan, device=run.policy.device) for _ in range(16)]
        torch.cuda.synchronize()
        del junk
        actual = run.reference.evaluate_reference(batch, positions, order)
        other((scratch,))
        torch.testing.assert_close(actual.factor_mask_WKF, expected.factor_mask_WKF, atol=0, rtol=0)
        for factor, (first, second) in enumerate(zip(actual.distributions, expected.distributions, strict=True)):
            live = expected.factor_mask_WKF[..., factor::4]
            p = first[live].double().clamp_min(1e-30)
            q = second[live].double().clamp_min(1e-30)
            divergence = (p * (p.log() - q.log())).sum(-1).abs().mean()
            assert float(divergence) < 1e-5
        torch.testing.assert_close(actual.features.trunk_WKd, expected.features.trunk_WKd, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(actual.features.decoder_WKt, expected.features.decoder_WKt, atol=0.02, rtol=0.01)
