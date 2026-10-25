"""Stacked anchors preserve distinct weights, queues, caches and refreshes."""

import copy
import weakref
from pathlib import Path

import pytest
import torch
from test_065_loop import PROVENANCE
from test_065_loop import FakeShard
from test_065_loop import exp
from test_065_loop import transformer_trainer


class FirstCategoryRng(exp.CounterRng):
    def tensor_uniform(
        self, domain: int, stream: torch.Tensor, episode: torch.Tensor, frame: torch.Tensor
    ) -> torch.Tensor:
        # Fixed quantiles keep the action chain equal across BF16 GEMM kernels.
        return torch.zeros_like(stream, dtype=torch.float64)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.no_grad()
def test_stacked_anchor_ticks_match_separate_models_and_weight_refresh(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    source = transformer_trainer(device).reference
    sources = tuple(source.frozen_copy() for _ in range(4))
    torch.manual_seed(173)
    for owner, policy in enumerate(sources):
        for parameter in policy.model.parameters():
            parameter.add_(torch.randn_like(parameter) * (0.01 * owner))
        policy.model.critic_residual.residual.down.weight.normal_(std=0.02)
        policy.prepare_tick_weights()
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    stack = exp.StackedAnchorPolicy((sources[0], *sources))
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    stack.prepare_tick_weights()
    assert stack.anchor_groups == 5
    assert stack.flops == sources[0].flops
    batch = 20
    store = exp.FrameStore(140, batch // 2, torch.device(device))
    store.actions.uniform_()
    table = exp.DenseRollout(64, store)
    table.allocate_evaluation(source.model)
    ids = torch.arange(batch, device=device)
    meta = table.integers[:batch]
    meta[:, exp.STREAM] = ids
    meta[:, exp.LANE] = ids // 2
    meta[:, exp.SIDE] = ids % 2
    meta[:, exp.REPLAN] = torch.tensor([2, 2, 4, 2, 4, 4, 4, 4, 2, 4, 4, 4, 2, 2, 4, 4, 2, 2, 2, 4], device=device)
    meta[:, exp.OPP_REPLAN] = 2
    table.conditions[:batch, :2] = -4
    table.conditions[:batch, 2:] = 0.5
    table.active[:batch] = True
    table.active[1] = False
    serial = copy.deepcopy(table)
    ring = exp.ActionRing(batch // 2, 4, torch.device(device))
    serial_ring = exp.ActionRing(batch // 2, 4, torch.device(device))
    layout = exp.TickLayout(tuple(range(batch)), tuple(row // 2 for row in range(batch)), exp.ALL_BUFFERS)
    counts = (8, 4, 4, 4)
    offsets = (0, 8, 12, 16)
    layouts = tuple(
        exp.TickLayout(
            tuple(range(start, start + count)), tuple(row // 2 for row in range(start, start + count)), exp.ALL_BUFFERS
        )
        for start, count in zip(offsets, counts, strict=True)
    )
    rng = FirstCategoryRng(173)
    captures = 0
    for clock in (0, 2, 4, 128, 130):
        if clock == 128:
            sources[2].model.temporal.outputs["buttons"].down.bias[0].add_(0.1)
            sources[2].model.player_embedding.weight.add_(0.01)
            for policy in sources:
                policy.prepare_tick_weights()
            stack.prepare_tick_weights()
        for current in (table, serial):
            current.integers[:batch, exp.CLOCK] = clock
            current.integers[:batch, exp.LENGTH] = min(clock + 1, 129)
        ring.length_device.zero_()
        serial_ring.length_device.zero_()
        due = table.active[:batch] & (clock % meta[:, exp.REPLAN] == 0)
        selected = tuple(torch.nonzero(due).flatten().tolist())
        stack.sample_tick(exp.TickInputs(table, ids, decoder_rows=selected), rng, layout, ring, clock=clock)
        for policy, start, count, own_layout in zip(sources, offsets, counts, layouts, strict=True):
            own_rows = tuple(row - start for row in selected if start <= row < start + count)
            policy.sample_tick(
                exp.TickInputs(serial, ids[start : start + count], decoder_rows=own_rows),
                rng,
                own_layout,
                serial_ring,
                clock=clock,
            )
        torch.testing.assert_close(ring.values, serial_ring.values, atol=0, rtol=0)
        torch.testing.assert_close(ring.controllers, serial_ring.controllers, atol=0, rtol=0)
        torch.testing.assert_close(ring.length_device, serial_ring.length_device, atol=0, rtol=0)
        assert table.evaluation is not None and serial.evaluation is not None
        torch.testing.assert_close(table.evaluation.actions[:batch], serial.evaluation.actions[:batch], atol=0, rtol=0)
        torch.testing.assert_close(
            table.evaluation.log_probs[:batch], serial.evaluation.log_probs[:batch], atol=3e-2, rtol=3e-2
        )
        torch.testing.assert_close(table.values[:batch], serial.values[:batch], atol=3e-2, rtol=3e-2)
        for actual, expected in zip(
            table.evaluation.head_probabilities, serial.evaluation.head_probabilities, strict=True
        ):
            torch.testing.assert_close(actual[:batch], expected[:batch], atol=1.6e-2, rtol=1.6e-2)
        combined = stack.tick_caches[ring, (exp.ALL_BUFFERS, batch)].cache
        separate = [
            policy.tick_caches[serial_ring, (exp.ALL_BUFFERS, count)].cache
            for policy, count in zip(sources, counts, strict=True)
        ]
        torch.testing.assert_close(
            combined.positions, torch.cat([cache.positions for cache in separate]), atol=0, rtol=0
        )
        torch.testing.assert_close(
            combined.hidden, torch.cat([cache.hidden for cache in separate]), atol=3e-2, rtol=3e-2
        )
        for layer, actual in enumerate(combined.layers):
            torch.testing.assert_close(
                actual, torch.cat([cache.layers[layer] for cache in separate], dim=1), atol=3e-2, rtol=3e-2
            )
        torch.testing.assert_close(
            combined.history, torch.cat([cache.history for cache in separate], dim=1), atol=3e-2, rtol=3e-2
        )
        if clock == 4:
            captures = stack.capture_count
        if clock == 130:
            assert stack.capture_count == captures
    if device == "cuda":
        for changed in (copy.copy(table), table):
            if changed is table:
                changed.store = copy.deepcopy(table.store)
            with pytest.raises(ValueError, match="tick graph storage changed"):
                stack.sample_tick(exp.TickInputs(changed, ids, decoder_rows=selected), rng, layout, ring, clock=130)
        stack.clear_caches()
        assert not stack.tick_sources


def test_stacked_decoder_padding_uses_due_rows_and_preserves_weight_groups() -> None:
    source = transformer_trainer().reference
    stack = exp.StackedAnchorPolicy((source, source, source))
    selected = (0, 2, 3, 9)
    bucket = stack.decoder_bucket(selected, 12, 2)
    rows, valid = stack.decoder_layout(selected, 12, bucket)
    assert tuple(row for row, enabled in zip(rows, valid, strict=True) if enabled) == selected
    width = bucket // 3
    assert all(row in selected for row in rows)
    assert all(
        group * 4 <= row < (group + 1) * 4
        for group in range(3)
        for row, enabled in zip(
            rows[group * width : (group + 1) * width], valid[group * width : (group + 1) * width], strict=True
        )
        if enabled
    )
    assert stack.decoder_bucket((), 12, 2) == 0
    assert stack.decoder_layout((), 12, 0) == ((), ())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("count", [2, 4])
@torch.no_grad()
def test_stacked_decoder_replay_matches_separate_models_for_fixed_actions(device: str, count: int) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    source = transformer_trainer(device).reference
    sources = tuple(source.frozen_copy() for _ in range(3))
    torch.manual_seed(179)
    for owner, policy in enumerate(sources):
        for parameter in policy.model.temporal.parameters():
            parameter.add_(torch.randn_like(parameter) * (0.01 * owner))
    stack = exp.StackedAnchorPolicy(sources)
    batch, width = 12, 4
    hidden = torch.randn(batch, 7, source.model.cfg.d_model, device=device)
    observed = torch.zeros(batch, 4, dtype=torch.long, device=device)
    queue = torch.zeros(batch, source.max_delay, 4, dtype=torch.long, device=device)
    log_lambda = torch.full((batch,), -4.0, device=device)
    delays = torch.zeros(batch, dtype=torch.long, device=device)
    replans = torch.full_like(delays, count)
    percentile = torch.full((batch,), 0.5, device=device)
    uniforms = torch.rand(count, 4, batch, device=device)
    controls = []
    for index, policy in enumerate(sources):
        section = slice(index * width, (index + 1) * width)
        controls.append(
            policy.tick_decoder(
                hidden[section],
                observed[section],
                queue[section],
                log_lambda[section],
                delays[section],
                replans[section],
                uniforms[:, :, section],
                sampled_offsets=count,
                percentile=percentile[section],
            )
        )
    actions = torch.cat([control.actions for control in controls])
    actual = stack.tick_decoder(
        hidden,
        observed,
        queue,
        log_lambda,
        delays,
        replans,
        uniforms,
        sampled_offsets=count,
        sampled_actions=actions,
        percentile=percentile,
    )
    torch.testing.assert_close(actual.actions, actions, atol=0, rtol=0)
    torch.testing.assert_close(
        actual.log_probs, torch.cat([control.log_probs for control in controls]), atol=3e-2, rtol=3e-2
    )
    torch.testing.assert_close(
        actual.decoder_state, torch.cat([control.decoder_state for control in controls]), atol=3e-2, rtol=3e-2
    )
    for group, probabilities in enumerate(actual.head_probabilities):
        expected = torch.cat([control.head_probabilities[group] for control in controls])
        torch.testing.assert_close(probabilities, expected, atol=1.6e-2, rtol=1.6e-2)


@pytest.mark.parametrize("index", [-1, 3])
def test_stacked_embedding_rejects_indices_that_would_cross_anchor_tables(index: int) -> None:
    embedding = exp.AnchorEmbedding((torch.nn.Embedding(3, 2), torch.nn.Embedding(3, 2)))
    indices = torch.zeros(2, 1, dtype=torch.long)
    indices[0 if index >= 0 else 1] = index
    with pytest.raises(RuntimeError, match="out of range"):
        embedding(indices)


def test_stacked_anchor_layout_rejects_changed_weight_order_or_cohort_sizes() -> None:
    source = transformer_trainer().reference
    other = source.frozen_copy()
    stack = exp.StackedAnchorPolicy((source, source, other))
    stack.validate_layout((source, other), (8, 4))
    with pytest.raises(ValueError, match="weights disagree"):
        stack.validate_layout((other, source), (8, 4))
    with pytest.raises(ValueError, match="row groups changed"):
        stack.validate_layout((source, other), (7, 5))


def test_stacked_anchors_reject_different_representation_contracts() -> None:
    source = transformer_trainer().reference
    other = transformer_trainer(development=True).reference
    with pytest.raises(ValueError, match="matching model and representation"):
        exp.StackedAnchorPolicy((source, other))


def test_stacked_anchors_reject_parameters_without_a_stacked_operator() -> None:
    source = transformer_trainer().reference
    source.model.register_parameter("extra", torch.nn.Parameter(torch.ones(1)))
    with pytest.raises(ValueError, match="linear or embedding parameters"):
        exp.StackedAnchorPolicy((source, source))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.no_grad()
def test_checkpoint_restore_keeps_stack_identity_and_releases_its_caches(device: str, tmp_path: Path) -> None:
    from hal.inference.kv_cache import BlockRestartCache

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    run = transformer_trainer(device)
    stack = exp.StackedAnchorPolicy((run.reference, *run.snapshot_slots))
    run.engine.anchor_stacks[0] = stack
    expected = stack.model.observation_encoder.weight.clone()
    cache = BlockRestartCache(stack.model, 4, torch.device(device))
    stack.tick_caches[run.engine.rings[0], (exp.ALL_BUFFERS, 4)] = cache
    cached = weakref.ref(cache)
    del cache
    manifest = exp.save_checkpoint(run, tmp_path, PROVENANCE)
    run.engine.anchors[1].model.observation_encoder.weight.add_(0.125)
    stack.prepare_tick_weights()
    assert not torch.equal(stack.model.observation_encoder.weight, expected)
    exp.restore_checkpoint(run, manifest, PROVENANCE)
    assert run.engine.anchor_stacks[0] is stack
    assert cached() is None
    assert not stack.tick_caches and not stack.tick_graphs and not stack.restart_graphs
    torch.testing.assert_close(stack.model.observation_encoder.weight, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@torch.no_grad()
def test_anchor_preparation_and_snapshot_load_refresh_captured_weight_storage() -> None:
    from hal.inference.cuda_graph import CapturedCall

    original = transformer_trainer("cuda")
    sources = {owner: original.reference.frozen_copy() for owner in range(4)}
    engine = exp.RolloutEngine(
        ((FakeShard(tuple(range(64)), device="cuda"),),),
        original.policy,
        copy.deepcopy(original.engine.sampler),
        sources,
    )
    engine.prepare_anchor_stacks()
    stack = engine.anchor_stacks[0]
    assert stack is not None
    assert stack.capture_priority < original.policy.capture_priority
    for policy in sources.values():
        rotary = policy.model.temporal.blocks[0].rotary
        for length in (policy.max_delay, policy.max_delay + 2, policy.max_delay + 4):
            assert (length, policy.device, torch.bfloat16) in rotary.prepared
    assert stack.anchor_sources == (sources[0], *sources.values())
    stack.validate_layout(tuple(sources.values()), (8, 4, 4, 4))
    assert len(engine.execution_policies()) == 6
    linear = stack.model.observation_encoder
    inputs = torch.randn(20, linear.in_features, device="cuda")
    pointer = linear.tick_weight.data_ptr()
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False), stack._tick_weight_casts():
        operation = torch.compile(lambda: linear(inputs), fullgraph=True, options={"triton.cudagraphs": False})
        captured = CapturedCall(operation, (inputs,), device=torch.device("cuda"))
    for promotion in (False, True):
        if promotion:
            state = copy.deepcopy(sources[2].model.state_dict())
            state["observation_encoder.weight"].add_(0.125)
            sources[2].model.load_state_dict(state, strict=True)
            for policy in engine.execution_policies():
                policy.prepare_tick_weights()
        assert linear.tick_weight.data_ptr() == pointer
        actual = captured((inputs,))
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            expected = torch.cat(
                [
                    policy.model.observation_encoder(inputs[index * 4 : (index + 1) * 4])
                    for index, policy in enumerate(stack.anchor_sources)
                ]
            )
        torch.testing.assert_close(actual, expected, atol=1.6e-2, rtol=1.6e-2)
    for policy in engine.execution_policies():
        policy.clear_caches()
    assert not stack.tick_graphs and not stack.restart_graphs


@torch.no_grad()
def test_stacked_engine_layout_and_owned_writes_match_separate_anchors() -> None:
    original = transformer_trainer()
    sources = {owner: original.reference.frozen_copy() for owner in range(4)}
    engines = tuple(
        exp.RolloutEngine(
            ((FakeShard(tuple(range(32))),),),
            original.policy.frozen_copy(),
            copy.deepcopy(original.engine.sampler),
            sources,
        )
        for _ in range(2)
    )
    stack = exp.StackedAnchorPolicy((sources[0], *sources.values()))
    engines[1].anchor_stacks[0] = stack
    tables = []
    for engine in engines:
        engine.initialize()
        table = exp.DenseRollout(256, engine.stores[0])
        table.allocate_evaluation(original.policy.model)
        tables.append(table)
        pending = torch.full((table.players,), -1, dtype=torch.long)
        ids = engine._stage_tick(0, 0, table, pending)
        for buffer in engine.buffers[0]:
            engine._sample_buffer(buffer, 0, table, ids)
    key = exp.ALL_BUFFERS, exp.STACKED_ANCHORS
    assert key in engines[1].tick_layouts
    assert all(owner is None or owner == exp.STACKED_ANCHORS for _, owner in engines[1].tick_layouts)
    assert len(engines[1].tick_layouts[key].streams) == 10
    expected_rows = sum(len(rows) for (_, owner), rows in engines[0].tick_decoder_rows.items() if owner is not None)
    assert len(engines[1].tick_decoder_rows[key]) == expected_rows
    assert stack in engines[1].execution_policies()
    for name in ("values", "controllers", "length_device"):
        torch.testing.assert_close(
            getattr(engines[0].rings[0], name), getattr(engines[1].rings[0], name), atol=0, rtol=0
        )
    for table in tables:
        assert table.evaluation is not None
    torch.testing.assert_close(tables[0].evaluation.log_probs, tables[1].evaluation.log_probs, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_anchor_stream_has_priority_over_student_and_is_reused() -> None:
    import numpy as np
    from test_065_loop import ToyPolicy

    class StreamProbe(exp.RolloutEngine):
        def __init__(self) -> None:
            self.policy = ToyPolicy("cuda")
            self.cursor = 0
            buffer = exp.RolloutBuffer(0, (), np.empty(0, dtype=np.int64), torch.cuda.Stream())
            self.buffers = ((buffer,),)
            self.anchor_streams = {}
            self.tick_layouts = dict.fromkeys(((exp.ALL_BUFFERS, exp.STACKED_ANCHORS), (0, None)))
            self.priorities: dict[tuple[int, int | None], int] = {}

        def _sample_layout(self, key: tuple[int, int | None], clock: int, table: object, ids: torch.Tensor) -> None:
            self.priorities[key] = torch.cuda.current_stream(self.policy.device).priority

    probe = StreamProbe()
    buffer = probe.buffers[0][0]
    key = exp.ALL_BUFFERS, exp.STACKED_ANCHORS
    ids = torch.empty(0, dtype=torch.long, device="cuda")
    probe._sample_buffer(buffer, 2, None, ids, concurrent=True)
    assert probe.priorities[key] < probe.priorities[0, None]
    first = probe.anchor_streams[key]
    probe._sample_buffer(buffer, 4, None, ids, concurrent=True)
    assert probe.anchor_streams[key] is first
