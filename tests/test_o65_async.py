"""Host staging must preserve every frame and keep CUDA off simulator threads."""

import os
from concurrent.futures import Future
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_transfer_parity_resets_and_staggered_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    import contextlib
    import dataclasses
    import threading

    import numpy as np
    import torch
    from test_065_loop import ToyPolicy
    from test_065_loop import exp
    from test_065_loop import sampler
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves

    from hal.sim import native_build
    from hal.sim import native_shard

    if device == "cuda":
        assert torch.cuda.is_available(), "required CUDA fixture missing"
    source = Path(os.environ.get("HAL_MSL_SOURCE", "~/src/melee-sim-light")).expanduser()
    monkeypatch.syspath_prepend(str(source))
    monkeypatch.setenv("MSL_CORE_LIBRARY", str(source / "build/melee_core/python/libmelee_core.so"))
    monkeypatch.setenv("HAL_NATIVE_LIBRARY", str(native_build.build(source, tmp_path / "adapter.so")))
    main = threading.get_ident()
    cuda_calls = []

    def guard(original):
        def call(*args, **kwargs):
            if threading.get_ident() != main:
                cuda_calls.append(original.__name__)
                raise AssertionError("worker made a CUDA call")
            return original(*args, **kwargs)

        return call

    for name in ("stream", "current_stream", "synchronize", "Event", "Stream"):
        monkeypatch.setattr(torch.cuda, name, guard(getattr(torch.cuda, name)))

    class NoWorkerCuda(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = {} if kwargs is None else kwargs
            if any(isinstance(t, torch.Tensor) and t.is_cuda for t in tree_leaves((args, kwargs))):
                raise AssertionError("worker accessed a CUDA tensor")
            requested = kwargs.get("device")
            if requested is not None and torch.device(requested).type == "cuda":
                raise AssertionError("worker allocated a CUDA tensor")
            result = func(*args, **kwargs)
            assert not any(isinstance(t, torch.Tensor) and t.is_cuda for t in tree_leaves(result))
            return result

    step_packed = native_shard.NativeShard._step_packed

    def guarded_step(*args, **kwargs):
        with NoWorkerCuda():
            return step_packed(*args, **kwargs)

    monkeypatch.setattr(native_shard.NativeShard, "_step_packed", guarded_step)

    class LegacyShard:
        def __init__(self, shard):
            self.shard = shard
            self.match_ids = shard.match_ids

        def step(self, inputs, active):
            return self.shard.step(inputs, active)

        def reset(self, lane, episode):
            return self.shard.reset(lane, episode)

        def save(self):
            return self.shard.save()

        def restore(self, state):
            self.shard.restore(state)

    torch.set_num_threads(2)
    with contextlib.ExitStack() as resources:
        engines = []
        for legacy in (True, False):
            torch.manual_seed(19)
            policy = ToyPolicy(device)
            shards = []
            for index, cpu in enumerate(sorted(os.sched_getaffinity(0))[:4]):
                shard = exp.NativeSimulatorShard(
                    source / "data", (2 * index, 2 * index + 1), cpu=cpu, device=torch.device(device), max_frame=1
                )
                resources.callback(shard.close)
                shards.append(shard)
            sampling = sampler(delay=8)
            engine = exp.RolloutEngine(
                (tuple(LegacyShard(s) for s in shards) if legacy else tuple(shards),),
                policy,
                sampling,
                {0: policy.frozen_copy()},
            )
            engine.initialize()
            for shard in shards:
                shard._before_write()
                shard._configs[0] = dataclasses.replace(shard._configs[0], max_frame=0)
                shard.shard.reset(shard._configs, np.ones(2, dtype=np.bool_)).result()
            engines.append(engine)
        # Exercise restore before transfers. MSL relocates arena pointers between batches.
        for a, b in zip(engines[0].groups[0], engines[1].groups[0], strict=True):
            b.restore(a.save())
        for update in range(2):
            expected, actual = [engine.collect(update) for engine in engines]
            assert expected.frames == actual.frames
            assert expected.resets == actual.resets > 0
            assert expected.timeouts == actual.timeouts > 0
            assert torch.equal(expected.ids, actual.ids)
            assert len(engines[1].transfer_timings) <= 256
            assert all(len(shard.timings) <= 128 for shard in engines[1].groups[0])
            for name, reference in vars(expected.table).items():
                if isinstance(reference, torch.Tensor):
                    assert torch.equal(reference, getattr(actual.table, name)), name
            for name in ("records", "actions", "rewards", "flags", "lengths"):
                assert torch.equal(getattr(expected.table.store, name), getattr(actual.table.store, name)), name
            assert exp._state_hash(engines[0].snapshot()) == exp._state_hash(engines[1].snapshot())
            for shard in engines[1].groups[0]:
                before = shard.save()
                shard.restore(before)
                assert shard.save() == before
        assert not cuda_calls


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_failed_tick_releases_host_waiters(failure: type[BaseException]) -> None:
    from test_065_loop import exp
    from test_065_loop import trainer

    engine = trainer(groups=1).engine
    staging = exp.BufferStaging(1, engine.policy.device)
    staging.ready = Future()
    buffer = engine.buffers[0][0]
    engine.staging[id(buffer)] = staging
    with pytest.raises(failure, match="failed inference"), engine._host_transfers((buffer,)):
        raise failure("failed inference")
    assert staging.ready.done()
    with pytest.raises(failure, match="failed inference"):
        staging.ready.result()


@pytest.mark.integration
def test_staging_keeps_fp32_controller_bits() -> None:
    import numpy as np
    import torch
    from test_065_loop import exp

    assert torch.cuda.is_available(), "required CUDA fixture missing"
    source = np.random.default_rng(65).uniform(-1, 1, (2, 8, 2, 14)).astype(np.float32)
    source[0, 0, 0, 0] = -0.0
    staging = exp.BufferStaging(8, torch.device("cuda"))
    staging.ready = Future()
    staging.inputs.copy_(torch.from_numpy(source).cuda(), non_blocking=True)
    staging.input_copied.record()
    staging.release_inputs()
    np.testing.assert_array_equal(staging.host_inputs.view(np.uint32), source.view(np.uint32))


def test_failed_copy_cancels_both_buffers() -> None:
    from test_065_loop import exp
    from test_065_loop import trainer

    class FailedCopy:
        def synchronize(self) -> None:
            raise RuntimeError("copy failed")

    engine = trainer(groups=1).engine
    buffers = engine.buffers[0]
    for buffer in buffers:
        staging = exp.BufferStaging(1, engine.policy.device)
        staging.ready = Future()
        staging.input_copied = FailedCopy()
        engine.staging[id(buffer)] = staging
    with pytest.raises(RuntimeError, match="copy failed"), engine._host_transfers(buffers):
        pass
    for buffer in buffers:
        ready = engine.staging[id(buffer)].ready
        assert ready.done()
        with pytest.raises(RuntimeError, match="copy failed"):
            ready.result()


@pytest.mark.integration
def test_cancelled_inputs_do_not_step_native(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from hal.sim import native
    from hal.sim import native_build
    from hal.sim import native_shard

    source = Path(os.environ.get("HAL_MSL_SOURCE", "~/src/melee-sim-light")).expanduser()
    monkeypatch.syspath_prepend(str(source))
    monkeypatch.setenv("MSL_CORE_LIBRARY", str(source / "build/melee_core/python/libmelee_core.so"))
    monkeypatch.setenv("HAL_NATIVE_LIBRARY", str(native_build.build(source, tmp_path / "adapter.so")))
    with native_shard.NativeShard(source / "data", 1, cpu=min(os.sched_getaffinity(0))) as shard:
        active = np.ones(1, dtype=np.bool_)
        match = native.NativeMatch(25, (native.NativePlayer(1, 1), native.NativePlayer(2, 22)), 65)
        shard.reset((match,), active).result()
        before = shard.save(0).result()
        ready: Future[None] = Future()
        ready.set_exception(RuntimeError("copy failed"))
        future = shard.step_packed(
            np.zeros((2, 1, 2, 14), dtype=np.float32),
            active,
            np.empty((2, 1), dtype=native_shard.STAGING_DTYPE),
            ready,
        )
        with pytest.raises(RuntimeError, match="copy failed"):
            future.result(timeout=5)
        assert shard.save(0).result(timeout=5) == before
