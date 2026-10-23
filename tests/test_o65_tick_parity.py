"""Compare fixed tick inputs with a0ae12df, which must be in Git history."""

import importlib.util
import io
import math
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest
import torch

from hal.data.feature_stats import FeatureStats
from hal.data.feature_stats import consolidate_key
from hal.models.action_sequence import ActionSequenceConfig
from hal.models.conditioning import ConditioningRanges
from hal.models.self_play import SelfPlayTransformer
from hal.representation.features import BASE_ITEMS_PROJECTION
from hal.representation.features import ITEM_COLUMNS
from hal.representation.features import feature_kind
from hal.representation.player_identity import PlayerVocabulary

CONTROL = "a0ae12df"
ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def tick_outputs(root: Path = ROOT, device: str = "cpu", frozen: bool = False) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("o65_tick_parity_runtime", root / "experiments/065_selfplay_ppo.py")
    assert spec is not None and spec.loader is not None
    exp = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = exp
    spec.loader.exec_module(exp)
    torch.set_num_threads(2)
    torch.manual_seed(171)
    vocabulary = PlayerVocabulary(())
    cfg = ActionSequenceConfig(
        d_model=32,
        n_layers=2,
        n_heads=2,
        temporal_d_model=32,
        temporal_layers=2,
        temporal_heads=2,
        temporal_ff_dim=64,
        group_head_dim=16,
        value_hidden_dim=16,
        return_embed_dim=16,
        head_offsets=tuple(range(1, 29)),
        player_vocab_size=vocabulary.size,
        player_vocab_sha256=vocabulary.sha256,
    )
    model = (
        SelfPlayTransformer(
            cfg,
            ConditioningRanges((math.log(0.0001), math.log(0.1)), (16, 21), (2, 4)),
            vocabulary,
            max_opponent_queue=24,
            critic_hidden_dim=16,
        )
        .to(device)
        .eval()
    )
    # Exercise the decoder contribution to values, which starts at zero in O64.
    model.critic_residual.residual.down.weight.normal_(std=0.02)
    model.requires_grad_(not frozen)
    stats = {
        consolidate_key(name): FeatureStats(0, 1, -1, 1)
        for name in BASE_ITEMS_PROJECTION.columns
        if feature_kind(name, ITEM_COLUMNS) == "float" and not name.startswith("ego_button_")
    }

    class RecordingPolicy(exp.TransformerPolicy):
        def __init__(self, model: SelfPlayTransformer, schedule: Any, statistics: dict[str, FeatureStats]) -> None:
            super().__init__(model, schedule, statistics)
            self.sampled_streams: list[int] = []

        def _tensor_action_uniforms(self, meta: torch.Tensor, rng: Any) -> torch.Tensor:
            if meta.device.type == "cpu":
                self.sampled_streams.extend(meta[:, exp.STREAM].tolist())
            return super()._tensor_action_uniforms(meta, rng)

    policy = RecordingPolicy(model, exp.production_config(cfg.head_offsets), stats)
    store = exp.FrameStore(140, 10, torch.device(device))
    store.actions.uniform_()
    table = exp.DenseRollout(64, store)
    table.allocate_evaluation(model)
    ids = torch.tensor([3, 1, 5, 0, 4, 2, 9, 6, 8, 7], device=device)
    streams = (7, 2, 9, 0, 5, 10, 13, 14, 17, 18)
    lanes = (3, 1, 4, 0, 2, 5, 6, 7, 8, 9)
    layout = exp.TickLayout(streams, lanes)
    meta = torch.zeros(10, 18, dtype=torch.long, device=device)
    meta[:, exp.STREAM] = torch.tensor(streams, device=device)
    meta[:, exp.LANE] = torch.tensor(lanes, device=device)
    meta[:, exp.SIDE] = torch.tensor(streams, device=device) % 2
    meta[:, exp.LENGTH] = torch.tensor([129, 1, 2, 65, 80, 129, 7, 100, 63, 127], device=device)
    meta[:, exp.DELAY] = torch.tensor([16, 17, 18, 19, 20, 21, 16, 17, 18, 19], device=device)
    meta[:, exp.REPLAN] = torch.tensor([4, 2, 4, 2, 4, 2, 4, 2, 4, 2], device=device)
    meta[:, exp.OPP_DELAY] = meta[:, exp.DELAY].flip(0)
    meta[:, exp.OPP_REPLAN] = meta[:, exp.REPLAN].flip(0)
    meta[:, exp.OPP_LENGTH] = meta[:, exp.OPP_DELAY]
    table.conditions[ids] = torch.tensor([-5.0, -6.0, 0.3, 0.7], device=device)
    table.prefix[ids] = model.codec.quantize(torch.rand(10, 24, exp.ACTION_DIM, device=device))
    table.opponent[ids] = model.codec.quantize(torch.rand(10, 24, exp.ACTION_DIM, device=device))
    table.active[ids] = torch.tensor([True, False, True, True, False, True, False, True, False, False], device=device)
    ring = exp.ActionRing(10, 32, torch.device(device))
    ring.length_device[meta[:, exp.LANE], meta[:, exp.SIDE]] = meta[:, exp.DELAY]
    states = exp.TickInputs(table, ids)
    ownership = torch.ones(10, dtype=torch.bool, device=device)

    class EndpointRng(exp.CounterRng):
        def tensor_uniform(
            self, domain: int, stream: torch.Tensor, episode: torch.Tensor, frame: torch.Tensor
        ) -> torch.Tensor:
            # Endpoint uniforms keep discrete draws stable under BF16 rounding.
            return torch.where(
                stream % 2 == 0,
                torch.zeros_like(stream, dtype=torch.float64),
                torch.full_like(stream, 1 - 2**-24, dtype=torch.float64),
            )

    rng = EndpointRng(65) if device == "cuda" else exp.CounterRng(65)
    inputs = [value.clone() for value in model.parameters()]
    inputs.extend([store.actions.clone(), table.prefix[ids], table.opponent[ids], meta.clone()])
    outputs = []
    for clock in (128, 130, 132, 134, 136):
        if clock == 132:
            # Replays read current table rows and observations, including a
            # different row assignment for the same stream layout.
            active = table.active[ids].clone()
            ids = ids.roll(1)
            table.active[ids] = active
            table.conditions[ids, 0] += 0.125
            table.prefix[ids] = table.prefix[ids].flip(1)
            table.opponent[ids] = table.opponent[ids].flip(1)
            store.actions[129:] *= 0.5
            ownership[2] = False
        if clock == 134:
            # R2 grows from bucket 4 to bucket 6. Bucket 6 was captured with
            # inactive synthetic rows; production must replay it on live rows.
            table.active[ids[[1, 9]]] = True
        if clock == 136:
            table.active[ids] = False
        meta[:, exp.CLOCK] = clock
        table.integers[ids] = meta
        states = exp.TickInputs(table, ids, enabled=ownership)
        if device == "cuda" and "decoder_rows" in exp.TickInputs.__dataclass_fields__:
            due = table.active[ids] & ownership & (clock % meta[:, exp.REPLAN] == 0)
            decoder_rows = tuple(torch.nonzero(due).flatten().cpu().tolist())
            states = exp.TickInputs(table, ids, enabled=ownership, decoder_rows=decoder_rows)
        uniforms = policy._tensor_action_uniforms(meta, rng)
        policy.sampled_streams.clear()
        if device == "cuda" and clock == 134 and hasattr(policy, "tick_graphs"):
            graphs = len(policy.tick_graphs)
            cache_key = ring, (layout.buffer, len(ids))
            if "decoder_rows" in exp.TickInputs.__dataclass_fields__:
                assert (cache_key, 2, 2, 6) in policy.tick_graphs
        policy.sample_tick(states, rng, layout, ring, clock=clock)
        if device == "cuda" and clock == 134 and hasattr(policy, "tick_graphs"):
            assert len(policy.tick_graphs) == graphs
        due = table.active[ids] & ownership & (clock % meta[:, exp.REPLAN] == 0)
        if device == "cpu" and "decoder_rows" in exp.TickInputs.__dataclass_fields__:
            assert policy.sampled_streams == meta[due, exp.STREAM].tolist()
        rows = ids[due]
        cache = policy.tick_caches[ring, (layout.buffer, len(ids))]
        outputs.append(
            {
                "actions": table.actions[rows].clone(),
                "log_probs": table.behavior[rows].clone(),
                "values": table.values[rows].clone(),
                "distributions": tuple(value[rows].clone() for value in table.evaluation.distributions),
                "decoder": table.evaluation.features.decoder[rows].clone(),
                "ring": ring.values.clone(),
                "controllers": ring.controllers.clone(),
                "lengths": ring.length_device.clone(),
                "cache": tuple(value.clone() for value in cache.cache.buffers()),
                "uniforms": uniforms,
            }
        )
        meta[:, exp.LENGTH] += 2
    return {"inputs": inputs, "ticks": outputs}


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("frozen", [False, True])
def test_tick_matches_control_actions_log_probs_values_and_ring(tmp_path: Path, device: str, frozen: bool) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    control = tmp_path / "control"
    control.mkdir()
    archive = subprocess.check_output(["git", "archive", CONTROL, "hal", "experiments/065_selfplay_ppo.py"], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as files:
        files.extractall(control, filter="data")
    output = tmp_path / "control.pt"
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(output), device, str(int(frozen))],
        cwd=control,
        env={**os.environ, "PYTHONPATH": str(control)},
        check=True,
        timeout=300,
    )
    expected = torch.load(output, weights_only=False)
    actual = tick_outputs(device=device, frozen=frozen)
    for value, reference in zip(actual["inputs"], expected["inputs"], strict=True):
        torch.testing.assert_close(value, reference, atol=0, rtol=0)
    for tick, control_tick in zip(actual["ticks"], expected["ticks"], strict=True):
        for name in ("actions", "ring", "controllers", "lengths", "uniforms"):
            torch.testing.assert_close(tick[name], control_tick[name], atol=0, rtol=0)
        for name in ("log_probs", "values", "decoder"):
            torch.testing.assert_close(tick[name], control_tick[name], atol=1.6e-2, rtol=1.6e-2)
        for value, reference in zip(tick["distributions"], control_tick["distributions"], strict=True):
            # Compare every categorical log-probability under the same BF16
            # contract as the sampled factors, including exact zero support.
            positive = reference > 0
            assert torch.equal(value > 0, positive)
            torch.testing.assert_close(value[positive].log(), reference[positive].log(), atol=1.6e-2, rtol=1.6e-2)
        for value, reference in zip(tick["cache"], control_tick["cache"], strict=True):
            torch.testing.assert_close(
                value, reference, atol=1.6e-2 if device == "cuda" else 2e-6, rtol=1.6e-2 if device == "cuda" else 2e-5
            )


@pytest.mark.parametrize(
    "clock,decoder_rows,message",
    [(0, (0,), "omit a due row"), (2, (0,), "not due"), (0, (0, 0), "distinct"), (0, (2,), "row indices")],
)
def test_tick_rejects_incorrect_decoder_rows(clock: int, decoder_rows: tuple[int, ...], message: str) -> None:
    from test_065_loop import exp
    from test_065_loop import transformer_trainer

    run = transformer_trainer(development=True)
    engine = run.engine
    table = exp.DenseRollout(64, engine.stores[0])
    table.allocate_evaluation(run.policy.model)
    ids = engine._stage_tick(clock, 0, table, torch.full((2,), -1, dtype=torch.long))
    table.integers[ids, exp.REPLAN] = 4
    meta = table.integers[ids]
    layout = exp.TickLayout(tuple(meta[:, exp.STREAM].tolist()), tuple(meta[:, exp.LANE].tolist()))
    states = exp.TickInputs(table, ids, decoder_rows=decoder_rows)
    with pytest.raises((ValueError, RuntimeError), match=message):
        run.policy.sample_tick(states, engine.sampler.rng, layout, engine.rings[0], clock=clock)


def test_staged_decoder_rows_match_device_metadata_and_ownership() -> None:
    from test_065_loop import exp
    from test_o65_fixed_layout import fixed_trainer

    engine = fixed_trainer().engine
    table = exp.DenseRollout(64, engine.stores[0])
    pending = torch.full((table.players,), -1, dtype=torch.long)
    for clock in (0, 2):
        if clock == 2:
            engine.active[0][0] = False
            engine.live_host[0][0] = False
        ids = engine._stage_tick(clock, 0, table, pending)
        for key in engine.tick_layouts:
            slots = engine.group_slots[0][key]
            selected = ids[slots]
            owner = -1 if key[1] is None else key[1]
            metadata = table.integers[selected]
            due = table.active[selected] & (engine.owners[0][slots] == owner)
            due &= metadata[:, exp.CLOCK] % metadata[:, exp.REPLAN] == 0
            assert engine.tick_decoder_rows[key] == tuple(torch.nonzero(due).flatten().tolist())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_head_start_preserves_captured_store_addresses(device: str) -> None:
    from test_065_loop import exp
    from test_065_loop import trainer

    from hal.inference.cuda_graph import CapturedCall

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    engine = trainer(device, groups=1).engine
    count = engine.sampler.schedule.rollout_frames
    previous = engine.stores[0]
    store = exp.FrameStore(129 + count + count // 128 + 1, len(engine.group_matches[0]), engine.policy.device)
    for name in ("records", "actions", "rewards", "flags"):
        getattr(store, name)[: len(previous.records)].copy_(getattr(previous, name))
    engine.stores[0] = store
    table = exp.DenseRollout(count // 2, store)
    engine.tables[0] = table
    captured = (
        CapturedCall(lambda: store.records.clone(), (store.records,), device=engine.policy.device)
        if device == "cuda"
        else None
    )
    engine.stagger_starts(64)
    assert engine.stores[0] is store
    engine.collect(-1)
    assert engine.stores[0] is table.store is store
    assert torch.count_nonzero(store.records[1:])
    if captured:
        torch.testing.assert_close(captured((store.records,)), table.store.records, atol=0, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tick_linear_casts_refresh_without_changing_master_weights_or_gradients(device: str) -> None:
    from test_065_loop import exp

    from hal.inference.cuda_graph import CapturedCall

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(91)
    source = torch.nn.Linear(64, 96, device=device)
    layer = exp.TickLinear(source)
    assert layer.weight is source.weight and layer.bias is source.bias
    assert layer.state_dict().keys() == source.state_dict().keys()
    inputs = torch.randn(8, 64, device=device, requires_grad=True)
    parameters = inputs, source.weight, source.bias
    expected = source(inputs)
    actual = layer(inputs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for gradient, reference in zip(
        torch.autograd.grad(actual.sum(), parameters), torch.autograd.grad(expected.sum(), parameters), strict=True
    ):
        torch.testing.assert_close(gradient, reference, atol=0, rtol=0)
    layer.tick = True
    # The critic explicitly disables autocast and must keep its FP32 path.
    torch.testing.assert_close(layer(inputs), source(inputs), atol=0, rtol=0)
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, cache_enabled=False):
        static = inputs.detach().clone()
        captured = (
            CapturedCall(
                torch.compile(lambda: layer(static), fullgraph=True, options={"triton.cudagraphs": False}),
                (static,),
                device=torch.device(device),
            )
            if device == "cuda"
            else None
        )
        for _ in range(2):
            source.weight.add_(0.125)
            source.bias.mul_(0.75)
            layer.refresh()
            actual = layer(static) if captured is None else captured((inputs.detach(),))
            torch.testing.assert_close(actual, source(inputs), atol=0, rtol=0)


def test_split_staging_matches_whole_tick() -> None:
    from test_065_loop import exp
    from test_065_loop import trainer

    engines = [trainer(groups=1, delay=8).engine for _ in range(2)]
    tables = [exp.DenseRollout(256, engine.stores[0]) for engine in engines]
    pending = [torch.full((table.players,), -1, dtype=torch.long) for table in tables]
    for engine, table, previous in zip(engines, tables, pending, strict=True):
        engine._stage_tick(0, 0, table, previous)
    expected = engines[0]._stage_tick(2, 0, tables[0], pending[0])
    for buffer in engines[1].buffers[0]:
        actual = engines[1]._stage_tick(2, 0, tables[1], pending[1], buffer=buffer)
        assert torch.equal(actual, expected)
    for name, reference in vars(tables[0]).items():
        if isinstance(reference, torch.Tensor):
            assert torch.equal(getattr(tables[1], name), reference), name
    assert torch.equal(pending[1], pending[0])
    assert engines[1].tick_decoder_rows == engines[0].tick_decoder_rows
    with pytest.raises(ValueError, match="block boundary"):
        engines[1]._stage_tick(128, 0, tables[1], pending[1], buffer=engines[1].buffers[0][0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    "length,timeout,stagger", [(67, True, False), (1, False, False), (67, True, True), (1000, False, True)]
)
def test_prefetched_ticks_match_serial_rollouts(length: int, timeout: bool, stagger: bool) -> None:
    import dataclasses

    from test_065_loop import FakeShard
    from test_065_loop import ToyPolicy
    from test_065_loop import exp
    from test_065_loop import sampler

    class AsyncPolicy(ToyPolicy):
        def device_wire_actions(self, indices: torch.Tensor, *, validate: bool = True) -> torch.Tensor:
            return super().device_wire_actions(indices, validate=False)

        def sample_tick(self, *args: Any, **kwargs: Any) -> None:
            super().sample_tick(*args, **kwargs)
            if self.model.logits.requires_grad or kwargs["clock"] != 2:
                return
            states, ring = args[0], args[3]
            metadata = states.tensor(states.table.integers)[states.enabled]
            lanes, sides = metadata[:, exp.LANE], metadata[:, exp.SIDE]
            saved = ring.controller_storage[lanes, sides].clone()
            # The toy policy's tensor constructors synchronize. Delay a final
            # write after those constructors to expose a missing controller join.
            ring.controller_storage[lanes, sides] = 0
            torch.cuda._sleep(200_000_000)
            ring.controller_storage[lanes, sides] = saved

    class AsyncEngine(exp.RolloutEngine):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.copied_inputs: dict[int, tuple[torch.Tensor, torch.cuda.Event]] = {}

        def _submit_buffer(self, workers: Any, buffer: Any) -> list[Any]:
            inputs = self._controller_inputs(self.buffer_device_lanes[id(buffer)], 2).clone()
            ready = torch.cuda.Event()
            ready.record()
            self.copied_inputs[id(buffer)] = inputs, ready
            return []

        def _complete_buffer(self, buffer: Any, pending: Any) -> list[Any]:
            inputs, ready = self.copied_inputs[id(buffer)]
            ready.synchronize()
            completed = []
            start = 0
            for shard in buffer.shards:
                stop = start + len(shard.match_ids)
                if self._active(shard).any():
                    completed.append((shard, shard.step(inputs[:, start:stop], self._active(shard))))
                start = stop
            return completed

    class SerialEngine(AsyncEngine):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.accept_frame = exp.accept_frame_tensors
            self.stage_tick_tensors = self._stage_tick_tensors

        def _accept(self, *args: Any, **kwargs: Any) -> tuple[int, int]:
            kwargs["behavior"] = None
            return super()._accept(*args, **kwargs)

        def _prefetch_tick(self, *args: Any, **kwargs: Any) -> None:
            return None

    class PrefetchEngine(AsyncEngine):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.prefetched_clocks: list[int] = []

        def _prefetch_tick(self, clock: int, *args: Any, **kwargs: Any) -> torch.cuda.Event | None:
            prefetched = super()._prefetch_tick(clock, *args, **kwargs)
            if prefetched:
                self.prefetched_clocks.append(clock)
            return prefetched

    engines = []
    for cls in (SerialEngine, PrefetchEngine):
        torch.manual_seed(19)
        policy = AsyncPolicy("cuda")
        shards = tuple(
            FakeShard(tuple(range(start, start + 8)), length=length, timeout=timeout, device="cuda")
            for start in (0, 8)
        )
        sampling = sampler()
        sampling.schedule = dataclasses.replace(sampling.schedule, rollout_frames=128)
        engine = cls((shards,), policy, sampling, {0: policy.frozen_copy()})
        exp.Trainer(engine, exp.TrainConfig(actor_optimizer="adam", windows_per_step=32))
        engine.initialize()
        if stagger:
            engine.stagger_starts(64)
        engines.append(engine)
    for update in range(2):
        expected, actual = [engine.collect(update) for engine in engines]
        assert (actual.frames, actual.resets, actual.timeouts) == (expected.frames, expected.resets, expected.timeouts)
        assert torch.equal(actual.ids, expected.ids)
        for name, reference in vars(expected.table).items():
            if isinstance(reference, torch.Tensor):
                if name == "coefficients":
                    torch.testing.assert_close(getattr(actual.table, name), reference, atol=0, rtol=1e-6)
                else:
                    assert torch.equal(getattr(actual.table, name), reference), name
        for name in ("records", "actions", "rewards", "flags", "lengths"):
            assert torch.equal(getattr(actual.table.store, name), getattr(expected.table.store, name)), name
        assert exp._state_hash(engines[1].snapshot()) == exp._state_hash(engines[0].snapshot())
        for (role, counts), (expected_role, reference) in zip(actual.behavior, expected.behavior, strict=True):
            assert role == expected_role
            assert (counts.counts == reference.counts).all()
    if stagger:
        assert not engines[1].prefetched_clocks
        return
    assert engines[1].prefetched_clocks
    assert all(clock % 128 and clock % 512 for clock in engines[1].prefetched_clocks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_prefetched_ids_survive_asynchronous_consumer_and_allocator_reuse() -> None:
    import numpy as np
    from test_065_loop import ToyPolicy
    from test_065_loop import exp

    size = 1 << 20

    class PrefetchProbe(exp.RolloutEngine):
        def __init__(self) -> None:
            self.policy = ToyPolicy("cuda")
            self.cursor = 0
            self.staggering = False
            self.profile = None
            self.buffers = (
                tuple(
                    exp.RolloutBuffer(index, (), np.empty(0, dtype=np.int64), torch.cuda.Stream())
                    for index in range(2)
                ),
            )
            self.indices = torch.arange(size, device="cuda")
            torch.cuda.synchronize()
            # Avoid a first allocation's cudaMalloc synchronization draining
            # the delayed consumer before the producer can reuse its storage.
            for buffer in self.buffers[0]:
                with torch.cuda.stream(buffer.stream):
                    warm = self.indices.index_select(0, self.indices)
                    warm.fill_(-1)
                    torch.cuda._sleep(1)
                buffer.stream.synchronize()
                del warm

        def _stage_tick(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            return torch.arange(size, device="cuda")

        def _sample_layout(self, key: Any, clock: int, table: Any, ids: torch.Tensor) -> None:
            torch.cuda._sleep(20_000_000)
            self.observed = ids.index_select(0, self.indices)

    probe = PrefetchProbe()
    assert probe._prefetch_tick(2, 128, 0, None, None, ready=True)
    with torch.cuda.stream(probe.buffers[0][1].stream):
        for _ in range(16):
            replacement = torch.empty(size, dtype=torch.long, device="cuda")
            replacement.fill_(-1)
    for buffer in probe.buffers[0]:
        buffer.stream.synchronize()
    assert torch.equal(probe.observed, torch.arange(size, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_tick_tensor_preparation_covers_groups_layouts_and_future_clocks() -> None:
    from test_065_loop import exp
    from test_065_loop import trainer

    from hal.inference.cuda_graph import count_compilation_starts
    from hal.inference.cuda_graph import prepared_shape_compilation

    engine = trainer("cuda", groups=2).engine
    frames = engine.sampler.schedule.rollout_frames
    with count_compilation_starts() as counter:
        with prepared_shape_compilation(64, instance_count=2):
            for group, matches in enumerate(engine.group_matches):
                engine.cursor = group
                engine.tick_layouts = engine.group_layouts[group]
                previous = engine.stores[group]
                store = exp.FrameStore(129 + frames + frames // 128 + 1, len(matches), engine.policy.device)
                for name in ("records", "actions", "rewards", "flags"):
                    getattr(store, name)[: len(previous.records)].copy_(getattr(previous, name))
                engine.stores[group] = store
                table = exp.DenseRollout(frames // 2, store)
                engine.tables[group] = table
                pending = torch.full((table.players,), -1, device="cuda", dtype=torch.long)
                for clock in (0, 2, 4):
                    engine._stage_tick(clock, 0, table, pending)
                    if clock == 2:
                        for buffer in engine.buffers[group]:
                            engine._stage_tick(clock, 0, table, pending, buffer=buffer)
                engine.prepare_frame_acceptance(table)
                table.reset()
        engine.cursor = 0
        before = counter.snapshot()
        for update in range(3):
            engine.collect(update)
        assert counter.snapshot() == before


if __name__ == "__main__":
    torch.save(tick_outputs(Path.cwd(), sys.argv[2], bool(int(sys.argv[3]))), sys.argv[1])
