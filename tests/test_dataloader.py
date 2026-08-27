import hashlib
import json
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from multiprocessing import resource_tracker
from pathlib import Path
from threading import Event
from unittest.mock import patch

import numpy as np
import pytest
import torch
from streaming import MDSWriter
from streaming import StreamingDataset
from streaming.base.shared.memory import SharedMemory
from torch.utils.data import DataLoader

import hal.data.streaming_compat as streaming_compat
import hal.training.dataloader as dataloader
from hal.data.schema import SCHEMA_VERSION
from hal.streams import StreamSource
from hal.training.dataloader import ResumableStreamingDataLoader
from hal.training.dataloader import WindowDataset
from hal.training.dataloader import _choose_chunk_starts
from hal.training.dataloader import _loader_generator
from hal.training.dataloader import _make_streaming_dataset
from hal.training.features import Context
from hal.training.features import TrainBatch
from hal.training.replay_reservoir import OneBatchPrefetch
from hal.training.replay_reservoir import ReplayPack
from hal.training.replay_reservoir import ReplayReservoir
from hal.training.replay_reservoir import ReservoirLoader
from hal.training.replay_reservoir import _stable_replay_rng

L_CTX, L_CHUNK = 6, 4
_L = L_CTX + L_CHUNK


def _write_scalar_mds(root: Path, split: str, values: range) -> None:
    with MDSWriter(out=str(root / split), columns={"value": "int"}, compression="zstd") as writer:
        for value in values:
            writer.write({"value": value})


def test_multistream_dataset_uses_every_source_and_exposes_counts(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_scalar_mds(first, "val", range(0, 3))
    _write_scalar_mds(second, "val", range(10, 15))
    sources = (
        StreamSource("first", None, first),  # type: ignore[arg-type]
        StreamSource("second", None, second),  # type: ignore[arg-type]
    )

    dataset, names = _make_streaming_dataset(
        None,
        "val",
        sources=sources,
        remote=None,
        shuffle=True,
        shuffle_seed=123,
        cache_limit=None,
        shuffle_block_size=16,
        predownload=None,
    )

    assert names == ("first", "second")
    assert [stream.repeat for stream in dataset.streams] == [1, 1]
    assert [stream.proportion for stream in dataset.streams] == pytest.approx([3 / 8, 5 / 8])
    assert dataset.samples_per_stream.tolist() == [3, 5]
    assert dataset.shuffle and dataset.shuffle_seed == 123
    assert sorted(int(sample["value"]) for sample in dataset) == [0, 1, 2, 10, 11, 12, 13, 14]


def test_multistream_dataset_applies_explicit_source_weights(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_scalar_mds(first, "train", range(0, 3))
    _write_scalar_mds(second, "train", range(10, 15))
    sources = (
        StreamSource("first", None, first),  # type: ignore[arg-type]
        StreamSource("second", None, second),  # type: ignore[arg-type]
    )

    dataset, _ = _make_streaming_dataset(
        None,
        "train",
        sources=sources,
        source_weights=(1.0, 2.0),
        remote=None,
        shuffle=False,
        shuffle_seed=123,
        cache_limit=None,
        shuffle_block_size=16,
        predownload=None,
    )

    assert [stream.proportion for stream in dataset.streams] == pytest.approx([1 / 3, 2 / 3])
    assert dataset.samples_per_stream.tolist() == [3, 5]


def test_source_weights_must_match_multistream_sources(tmp_path: Path) -> None:
    source = StreamSource("source", "s3://example/source", Path("cache/source"))
    kwargs = {
        "split": "train",
        "remote": None,
        "shuffle": None,
        "shuffle_seed": None,
        "cache_limit": None,
        "shuffle_block_size": None,
        "predownload": None,
    }

    with pytest.raises(ValueError, match="length"):
        _make_streaming_dataset(None, sources=[source], source_weights=(1.0, 2.0), **kwargs)
    with pytest.raises(ValueError, match="finite and positive"):
        _make_streaming_dataset(None, sources=[source], source_weights=(0.0,), **kwargs)
    with pytest.raises(ValueError, match="requires sources"):
        _make_streaming_dataset(str(tmp_path), sources=None, source_weights=(1.0,), **kwargs)


def test_streaming_dataset_mode_selection_is_strict(tmp_path: Path) -> None:
    source = StreamSource("source", "s3://example/source", Path("cache/source"))
    kwargs = {
        "split": "train",
        "remote": None,
        "shuffle": None,
        "shuffle_seed": None,
        "cache_limit": None,
        "shuffle_block_size": None,
        "predownload": None,
    }
    with pytest.raises(ValueError, match="cannot be combined"):
        _make_streaming_dataset(str(tmp_path), sources=[source], **kwargs)
    with pytest.raises(ValueError, match="must not be empty"):
        _make_streaming_dataset(None, sources=[], **kwargs)
    with pytest.raises(ValueError, match="data_root is required"):
        _make_streaming_dataset(None, sources=None, **kwargs)


def test_single_stream_reservoir_count_metadata_is_empty() -> None:
    loader = object.__new__(ReservoirLoader)
    loader._source_names = ()
    loader._dataset = type("Dataset", (), {"samples_per_stream": np.array([3])})()

    assert loader.source_sample_counts == {}


def test_streaming_resource_tracker_forwards_without_extra_self() -> None:
    memory = object.__new__(SharedMemory)
    with patch.object(resource_tracker._resource_tracker, "register") as register:
        memory.fix_register("/semaphore", "semaphore")
    register.assert_called_once_with("/semaphore", "semaphore")
    with patch.object(resource_tracker._resource_tracker, "unregister") as unregister:
        memory.fix_unregister("/semaphore", "semaphore")
    unregister.assert_called_once_with("/semaphore", "semaphore")


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="requires Linux procfs")
def test_streaming_prefix_probes_do_not_leak_file_descriptors(tmp_path: Path) -> None:
    _write_scalar_mds(tmp_path, "train", range(1))
    program = textwrap.dedent(
        """\
        import gc
        import json
        import os
        import sys

        import hal.training.dataloader
        from streaming import StreamingDataset


        def count_shared_memory_descriptors():
            count = 0
            for name in os.listdir("/proc/self/fd"):
                try:
                    target = os.readlink(f"/proc/self/fd/{name}")
                except FileNotFoundError:
                    continue
                count += target.startswith("/dev/shm/")
            return count


        counts = [count_shared_memory_descriptors()]
        for _ in range(5):
            dataset = StreamingDataset(local=sys.argv[1], batch_size=1, shuffle=False)
            del dataset
            gc.collect()
            counts.append(count_shared_memory_descriptors())
        print(json.dumps(counts))
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path / "train")],
        check=True,
        capture_output=True,
        text=True,
    )
    counts = json.loads(completed.stdout.splitlines()[-1])
    increments = [after - before for before, after in zip(counts[:-1], counts[1:], strict=True)]

    assert increments == [increments[0]] * len(increments), counts


def test_loader_generator_does_not_change_process_rng() -> None:
    torch.manual_seed(123)
    before = torch.random.get_rng_state()

    generator = _loader_generator(7)

    assert torch.equal(torch.random.get_rng_state(), before)
    assert generator.initial_seed() == 7


def test_data_loader_iteration_uses_only_its_private_rng() -> None:
    train = DataLoader(range(4), batch_size=2, generator=_loader_generator(7))
    val = DataLoader(range(4), batch_size=2, generator=_loader_generator(8))
    torch.manual_seed(123)
    before = torch.random.get_rng_state()

    list(train)
    list(val)

    assert torch.equal(torch.random.get_rng_state(), before)


def _train_batch(index: int) -> TrainBatch:
    return TrainBatch(
        context=Context(
            features={
                "float": torch.tensor([[index + 0.25]], dtype=torch.float32),
                "int": torch.tensor([[index]], dtype=torch.int64),
            },
            ctx_pad=torch.tensor([index % 3]),
        ),
        target=torch.full((1, 2, 14), float(index)),
        replay_ids=(f"replay-{index}",),
    )


def _batch_digest(batch: TrainBatch) -> bytes:
    parts = ["\0".join(batch.replay_ids or ()).encode()]
    tensors = [batch.context.ctx_pad, batch.target]
    tensors.extend(batch.context.features[name] for name in sorted(batch.context.features))
    for tensor in tensors:
        parts.extend((str(tensor.dtype).encode(), str(tuple(tensor.shape)).encode(), tensor.numpy().tobytes()))
    return hashlib.sha256(b"\0".join(parts)).digest()


def test_one_batch_prefetch_preserves_32_complete_batches_and_rng() -> None:
    expected = tuple(_train_batch(index) for index in range(32))
    torch.manual_seed(123)
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    actual = tuple(OneBatchPrefetch(iter(expected)))

    assert [_batch_digest(batch) for batch in actual] == [_batch_digest(batch) for batch in expected]
    assert torch.equal(torch.random.get_rng_state(), cpu_before)
    if cuda_before is not None:
        for actual_state, expected_state in zip(torch.cuda.get_rng_state_all(), cuda_before, strict=True):
            assert torch.equal(actual_state, expected_state)


def test_one_batch_prefetch_overlaps_the_next_item() -> None:
    started = Event()
    release = Event()

    def source() -> Iterator[int]:
        yield 1
        started.set()
        if not release.wait(timeout=2):
            raise TimeoutError("the test did not release the producer")
        yield 2

    prefetch = OneBatchPrefetch(source())
    assert next(prefetch) == 1
    assert started.wait(timeout=1)
    release.set()
    assert next(prefetch) == 2
    with np.testing.assert_raises(StopIteration):
        next(prefetch)


def test_one_batch_prefetch_prepares_the_full_requested_depth() -> None:
    prepared = Event()

    def source() -> Iterator[int]:
        for index in range(4):
            if index == 2:
                prepared.set()
            yield index

    prefetch = OneBatchPrefetch(source(), depth=3)
    assert prepared.wait(timeout=1)
    assert [next(prefetch) for _ in range(4)] == [0, 1, 2, 3]
    with np.testing.assert_raises(StopIteration):
        next(prefetch)


def test_one_batch_prefetch_rejects_invalid_depth() -> None:
    with np.testing.assert_raises_regex(ValueError, "positive integer"):
        OneBatchPrefetch(iter(()), depth=0)


def test_one_batch_prefetch_propagates_errors_and_closes_early() -> None:
    started = Event()
    release = Event()
    closed = Event()

    def source() -> Iterator[int]:
        try:
            yield 1
            started.set()
            if not release.wait(timeout=2):
                raise TimeoutError("the test did not release the producer")
            yield 2
            raise RuntimeError("source failed")
        finally:
            closed.set()

    prefetch = OneBatchPrefetch(source())
    assert next(prefetch) == 1
    assert started.wait(timeout=1)
    release.set()
    prefetch.close()
    assert closed.wait(timeout=1)
    with np.testing.assert_raises(StopIteration):
        next(prefetch)

    failed = OneBatchPrefetch(iter([1]))
    assert next(failed) == 1
    with np.testing.assert_raises(StopIteration):
        next(failed)

    def error_source() -> Iterator[int]:
        yield 1
        raise RuntimeError("source failed")

    failed = OneBatchPrefetch(error_source())
    assert next(failed) == 1
    with np.testing.assert_raises_regex(RuntimeError, "source failed"):
        next(failed)


def _fake_mds(n_samples: int = 6, length: int = 60) -> list[dict[str, np.ndarray]]:
    """In-memory stand-in for a StreamingDataset: each sample is one replay."""
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "frame": np.arange(length, dtype=np.int32),
            "p1_position_x": np.arange(length, dtype=np.float32),
            "p2_position_x": np.arange(length, dtype=np.float32) + 1000.0,
        }
        for _ in range(n_samples)
    ]


def _fingerprint(sampler: WindowDataset) -> list[tuple[int, str]]:
    """(window start, ego side) per yielded window — observable proxy for the
    sampler's two random draws (start offset + ego_prefix)."""
    out = []
    for w in sampler:
        start = int(w["frame"][0])
        ego_side = "p1" if w["ego_position_x"][0] < 500 else "p2"
        out.append((start, ego_side))
    return out


def test_same_seed_same_windows() -> None:
    """Two fresh samplers with the same seed yield identical windows — this is
    what makes cached val loss comparable across runs."""
    a = _fingerprint(WindowDataset(_fake_mds(), L_CTX, L_CHUNK, seed=0))
    b = _fingerprint(WindowDataset(_fake_mds(), L_CTX, L_CHUNK, seed=0))
    assert a == b


def test_different_seed_different_windows() -> None:
    a = _fingerprint(WindowDataset(_fake_mds(), L_CTX, L_CHUNK, seed=0))
    b = _fingerprint(WindowDataset(_fake_mds(), L_CTX, L_CHUNK, seed=1))
    assert a != b


def test_windows_vary_across_epochs() -> None:
    """A single sampler iterated twice (two epochs) draws different windows, so
    a fixed seed doesn't freeze train augmentation to one window per replay."""
    s = WindowDataset(_fake_mds(), L_CTX, L_CHUNK, seed=0)
    epoch0 = _fingerprint(s)
    epoch1 = _fingerprint(s)
    assert epoch0 != epoch1


def test_replay_identity_makes_mid_epoch_window_resume_exact() -> None:
    rows = _fake_mds(n_samples=8)
    for index, row in enumerate(rows):
        row[dataloader._STREAMING_REPLAY_ID] = f"replay-{index}"  # type: ignore[assignment]

    expected = _fingerprint(WindowDataset(rows, L_CTX, L_CHUNK, seed=17))[3:]
    resumed = WindowDataset(rows[3:], L_CTX, L_CHUNK, seed=17)
    resumed.resume_epoch(0)

    assert _fingerprint(resumed) == expected


class _StatefulRows(torch.utils.data.IterableDataset):
    def __init__(self) -> None:
        self.resumed_epoch: int | None = None

    def __iter__(self):
        yield from range(8)

    def resume_epoch(self, epoch: int) -> None:
        self.resumed_epoch = epoch


class _ValueRows(torch.utils.data.IterableDataset):
    def __init__(self, dataset: StreamingDataset) -> None:
        self.dataset = dataset
        self.resumed_epoch: int | None = None

    def __iter__(self):
        for sample in self.dataset:
            yield int(sample["value"])

    def resume_epoch(self, epoch: int) -> None:
        self.resumed_epoch = epoch


class _MosaicStateStub:
    replication = None

    def __init__(self) -> None:
        self.saved_num_samples: int | None = None
        self.loaded: dict | None = None

    def state_dict(self, num_samples: int, from_beginning: bool) -> dict:
        assert not from_beginning
        self.saved_num_samples = num_samples
        return {"epoch": 3, "sample_in_epoch": 11}

    def load_state_dict(self, state: dict) -> None:
        self.loaded = state


def test_resumable_streaming_loader_routes_mosaic_cursor_through_wrapper() -> None:
    rows = _StatefulRows()
    mds = _MosaicStateStub()
    loader = ResumableStreamingDataLoader(
        rows,
        streaming_dataset=mds,  # type: ignore[arg-type]
        window_dataset=rows,  # type: ignore[arg-type]
        batch_size=2,
        num_workers=0,
    )
    next(iter(loader))

    state = loader.state_dict()

    assert mds.saved_num_samples == 2
    restored_rows = _StatefulRows()
    restored_mds = _MosaicStateStub()
    restored = ResumableStreamingDataLoader(
        restored_rows,
        streaming_dataset=restored_mds,  # type: ignore[arg-type]
        window_dataset=restored_rows,  # type: ignore[arg-type]
        batch_size=2,
        num_workers=0,
    )
    restored.load_state_dict(state)
    assert restored_mds.loaded == state["mds"]
    assert restored_rows.resumed_epoch == 3


def test_resumable_streaming_loader_accepts_legacy_unwrapped_state() -> None:
    rows = _StatefulRows()
    mds = _MosaicStateStub()
    loader = ResumableStreamingDataLoader(
        rows,
        streaming_dataset=mds,  # type: ignore[arg-type]
        window_dataset=rows,  # type: ignore[arg-type]
        batch_size=2,
        num_workers=0,
    )
    legacy_state = {"epoch": 3, "sample_in_epoch": 11}

    loader.load_state_dict(legacy_state)

    assert mds.loaded == legacy_state
    assert rows.resumed_epoch == 3


@pytest.mark.parametrize(
    ("state", "error", "message"),
    [
        ({"schema": 2, "mds": {}}, ValueError, "unsupported"),
        ({"schema": 1, "mds": None}, TypeError, "mds"),
        ({"schema": 1, "mds": {}}, TypeError, "epoch"),
    ],
)
def test_resumable_streaming_loader_rejects_invalid_state(
    state: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    rows = _StatefulRows()
    loader = ResumableStreamingDataLoader(
        rows,
        streaming_dataset=_MosaicStateStub(),  # type: ignore[arg-type]
        window_dataset=rows,  # type: ignore[arg-type]
        batch_size=2,
        num_workers=0,
    )

    with pytest.raises(error, match=message):
        loader.load_state_dict(state)


def test_resumable_streaming_loader_restores_exact_mosaic_sequence(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _write_scalar_mds(first_root, "train", range(40))
    _write_scalar_mds(second_root, "train", range(40))

    def make(root: Path) -> ResumableStreamingDataLoader:
        mds = StreamingDataset(
            local=str(root / "train"),
            batch_size=1,
            shuffle=True,
            shuffle_seed=91,
            shuffle_block_size=16,
        )
        rows = _ValueRows(mds)
        return ResumableStreamingDataLoader(
            rows,
            streaming_dataset=mds,
            window_dataset=rows,  # type: ignore[arg-type]
            batch_size=2,
            num_workers=0,
        )

    uninterrupted = make(first_root)
    iterator = iter(uninterrupted)
    for _ in range(3):
        next(iterator)
    state = uninterrupted.state_dict()
    expected = [next(iterator) for _ in range(4)]

    resumed = make(second_root)
    resumed.load_state_dict(state)
    actual_iterator = iter(resumed)
    actual = [next(actual_iterator) for _ in range(4)]

    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want)


def test_failed_streaming_download_does_not_poison_shared_shard_state(monkeypatch) -> None:
    remote = streaming_compat.streaming_dataset._ShardState.REMOTE
    preparing = streaming_compat.streaming_dataset._ShardState.PREPARING

    class Dataset:
        _shard_states = np.array([remote], dtype=np.uint8)

    def fail(dataset, shard_id, blocking):
        del blocking
        dataset._shard_states[shard_id] = preparing
        raise RuntimeError("transient object-store failure")

    monkeypatch.setattr(streaming_compat, "_ORIGINAL_PREPARE_SHARD", fail)
    with pytest.raises(RuntimeError, match="object-store"):
        streaming_compat._prepare_shard_without_poisoned_state(Dataset(), 0)  # type: ignore[arg-type]
    assert Dataset._shard_states[0] == remote


def test_replay_transform_runs_after_validation_and_before_windowing() -> None:
    calls: list[int] = []

    def add_return(sample: dict[str, np.ndarray | int]) -> dict[str, np.ndarray | int]:
        calls.append(int(sample["schema_version"]))
        out = dict(sample)
        frames = len(np.asarray(sample["frame"]))
        out["p1_return"] = np.arange(frames, dtype=np.float32)
        out["p2_return"] = np.arange(frames, dtype=np.float32) + 100
        return out

    windows = list(WindowDataset(_fake_mds(n_samples=1), L_CTX, L_CHUNK, seed=2, replay_transform=add_return))

    assert calls == [SCHEMA_VERSION]
    assert len(windows) == 1
    assert {"ego_return", "opp_return"} <= windows[0].keys()

    invalid = _fake_mds(n_samples=1)
    invalid[0]["schema_version"] = SCHEMA_VERSION - 1
    with np.testing.assert_raises_regex(ValueError, "schema_version"):
        list(WindowDataset(invalid, L_CTX, L_CHUNK, seed=2, replay_transform=add_return))
    assert calls == [SCHEMA_VERSION]


def test_batch_transform_receives_windows_and_normal_train_batch() -> None:
    normal = _train_batch(5)
    windows = [{"ego_return": np.array([3.5], dtype=np.float32)}]

    def wrap(source_windows: list[dict[str, np.ndarray]], batch: TrainBatch) -> tuple[str, TrainBatch]:
        assert source_windows is windows
        assert source_windows[0]["ego_return"].item() == 3.5
        assert batch is normal
        return "wrapped", batch

    with patch.object(dataloader, "collate_train_batch", return_value=normal):
        result = dataloader._collate_with_batch_transform(
            windows, stats={}, L_ctx=L_CTX, extra=None, projection=None, batch_transform=wrap
        )

    assert isinstance(result, tuple)
    assert result[0] == "wrapped"
    assert result[1] is normal


def test_window_length_and_ctx_pad() -> None:
    """Every emitted window is exactly L_ctx + L_chunk frames and carries an int
    ctx_pad — the neutral [ctx | chunk] contract, no bridge frames."""
    for w in WindowDataset(_fake_mds(), L_CTX, L_CHUNK, seed=0):
        assert len(w["frame"]) == _L
        assert "ctx_pad" in w


def test_cold_start_floor_skips_too_short() -> None:
    """cs_min=1 needs >=1 real context frame and the L_chunk chunk in-episode, so
    a replay of exactly L_chunk frames (cs_max=0 < 1) yields nothing."""
    assert list(WindowDataset(_fake_mds(n_samples=2, length=L_CHUNK), L_CTX, L_CHUNK, seed=0)) == []
    # one extra frame is enough for a single anchor (cs=1, fully left-padded ctx).
    assert list(WindowDataset(_fake_mds(n_samples=2, length=L_CHUNK + 1), L_CTX, L_CHUNK, seed=0))


def test_choose_chunk_starts_nonoverlapping_and_bounded() -> None:
    """Up to K chunk-starts, stratified with pairwise gap >= _L so the
    [cs - L_ctx, cs + L_chunk) windows never overlap; all in [1, T - L_chunk];
    count clamps to what the episode can fit."""
    rng = np.random.default_rng(0)
    for T in [11, 30, 44, 60, 100, 300]:
        for K in [1, 2, 4, 16]:
            cs = _choose_chunk_starts(T, L_CTX, L_CHUNK, K, rng)
            fit = max(1, (T - L_CHUNK) // _L)
            assert len(cs) == min(K, fit)
            assert (cs >= 1).all() and (cs <= T - L_CHUNK).all()
            assert (np.diff(np.sort(cs)) >= _L).all()


def test_yields_k_nonoverlapping_windows_per_replay() -> None:
    """K=4 over a long replay emits 4 windows whose real (non-pad) frames are
    pairwise disjoint — distinct training examples, not near-duplicate slices."""
    K = 4
    wins = list(WindowDataset(_fake_mds(n_samples=1, length=60), L_CTX, L_CHUNK, seed=0, windows_per_replay=K))
    assert len(wins) == K
    seen: set[int] = set()
    for w in wins:
        real = set(w["frame"][int(w["ctx_pad"]) :].tolist())
        assert real and real.isdisjoint(seen)
        seen |= real


def test_windows_per_replay_clamps_to_short_replay() -> None:
    """A replay too short for K disjoint windows yields only what fits (here 2),
    never overlapping ones to hit the count."""
    wins = list(WindowDataset(_fake_mds(n_samples=1, length=30), L_CTX, L_CHUNK, seed=0, windows_per_replay=4))
    assert len(wins) == 2


def test_windows_per_replay_default_is_one() -> None:
    """Default K=1 keeps the historical one-window-per-replay behavior."""
    wins = list(WindowDataset(_fake_mds(n_samples=3, length=60), L_CTX, L_CHUNK, seed=0))
    assert len(wins) == 3


def _packs(n_replays: int, windows: int) -> Iterator[ReplayPack]:
    for replay in range(n_replays):
        yield ReplayPack(str(replay), tuple((replay, window) for window in range(windows)))


def test_reservoir_batches_have_distinct_replays_and_no_repeated_windows() -> None:
    reservoir = ReplayReservoir(_packs(12, 4), batch_size=4, capacity=8, seed=7)
    batches = list(reservoir)
    seen = set()
    for batch in batches:
        assert len(batch.replay_ids) == len(set(batch.replay_ids)) == 4
        assert not (set(batch.windows) & seen)
        seen.update(batch.windows)
    assert len(seen) == len(batches) * 4
    assert reservoir.emitted_windows == len(seen)
    assert reservoir.emitted_windows + reservoir.dropped_windows == 12 * 4


def test_reservoir_keeps_weighted_repeats_without_batch_duplicates() -> None:
    packs = iter(
        (
            ReplayPack("repeat", (("repeat", 0),)),
            ReplayPack("repeat", (("repeat", 1),)),
            ReplayPack("other", (("other", 0),)),
        )
    )

    batches = list(ReplayReservoir(packs, batch_size=1, capacity=2, seed=7, cooldown_batches=0))

    assert sorted(batch.windows[0] for batch in batches) == [("other", 0), ("repeat", 0), ("repeat", 1)]
    assert all(len(batch.replay_ids) == len(set(batch.replay_ids)) for batch in batches)


def test_reservoir_is_deterministic_and_bounded() -> None:
    def run() -> tuple[list[tuple[str, ...]], int]:
        reservoir = ReplayReservoir(_packs(20, 3), batch_size=4, capacity=8, seed=2)
        batches = []
        high_water = 0
        for batch in reservoir:
            batches.append(batch.replay_ids)
            high_water = max(high_water, reservoir.active_replays)
        return batches, high_water

    first, high_water = run()
    second, _ = run()
    assert first == second
    assert high_water <= 8


def test_reservoir_cooldown_avoids_adjacent_replay_reuse() -> None:
    batches = list(ReplayReservoir(_packs(8, 3), batch_size=4, capacity=8, seed=0, cooldown_batches=1))
    for previous, current in zip(batches, batches[1:], strict=False):
        assert set(previous.replay_ids).isdisjoint(current.replay_ids)


def test_reservoir_rejects_capacity_that_cannot_enforce_cooldown() -> None:
    with np.testing.assert_raises_regex(ValueError, "need at least 12"):
        ReplayReservoir(_packs(12, 2), batch_size=4, capacity=11, seed=0, cooldown_batches=2)


def test_reservoir_never_breaks_cooldown_at_epoch_tail() -> None:
    reservoir = ReplayReservoir(_packs(10, 3), batch_size=4, capacity=8, seed=4, cooldown_batches=1)
    batches = list(reservoir)
    for previous, current in zip(batches, batches[1:], strict=False):
        assert set(previous.replay_ids).isdisjoint(current.replay_ids)
    assert reservoir.emitted_windows % 4 == 0
    assert reservoir.emitted_windows + reservoir.dropped_windows == 30
    assert reservoir.dropped_windows > 0
    assert reservoir.active_replays == 0


def test_reservoir_epoch_tail_emits_complete_batches_and_counts_drop() -> None:
    reservoir = ReplayReservoir(_packs(10, 1), batch_size=4, capacity=4, seed=1, cooldown_batches=0)
    batches = list(reservoir)
    assert len(batches) == 2
    assert reservoir.emitted_windows == 8
    assert reservoir.dropped_windows == 2
    assert reservoir.dropped_replays == 2
    assert reservoir.active_replays == 0
    with np.testing.assert_raises(StopIteration):
        next(reservoir)


def _per_replay_draws(replay_ids: list[str], worker_count: int, epoch: int) -> dict[str, tuple]:
    draws = {}
    for worker in range(worker_count):
        for replay_id in replay_ids[worker::worker_count]:
            rng = _stable_replay_rng(17, epoch, replay_id)
            starts = tuple(_choose_chunk_starts(100, L_CTX, L_CHUNK, 4, rng).tolist())
            ego = tuple(bool(rng.integers(0, 2)) for _ in starts)
            draws[replay_id] = (starts, ego)
    return draws


def test_replay_draws_do_not_depend_on_worker_partition() -> None:
    replay_ids = [f"replay-{i}" for i in range(31)]
    expected = _per_replay_draws(replay_ids, worker_count=1, epoch=3)
    assert _per_replay_draws(replay_ids, worker_count=2, epoch=3) == expected
    assert _per_replay_draws(replay_ids, worker_count=7, epoch=3) == expected


def test_replay_draws_are_deterministic_and_change_each_epoch() -> None:
    replay_ids = [f"replay-{i}" for i in range(31)]
    epoch0 = _per_replay_draws(replay_ids, worker_count=4, epoch=0)
    assert _per_replay_draws(replay_ids, worker_count=4, epoch=0) == epoch0
    assert _per_replay_draws(replay_ids, worker_count=4, epoch=1) != epoch0


def test_train_batch_record_stream_covers_optional_context_fields(monkeypatch) -> None:
    recorded: list[torch.Tensor] = []
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: recorded.append(self))
    features = {"ego_position_x": torch.zeros(2, L_CTX), "opp_position_x": torch.zeros(2, L_CTX)}
    target = torch.zeros(2, L_CHUNK, 14)
    full = TrainBatch(
        context=Context(
            features=features,
            ctx_pad=torch.zeros(2, dtype=torch.int64),
            slot_ids=torch.arange(2),
            reset=torch.ones(2, dtype=torch.bool),
        ),
        target=target,
    )
    full.record_stream(object())
    assert len(recorded) == len(features) + 4  # ctx_pad, target, slot_ids, reset
    recorded.clear()
    bare = TrainBatch(context=Context(features=features, ctx_pad=torch.zeros(2, dtype=torch.int64)), target=target)
    bare.record_stream(object())
    assert len(recorded) == len(features) + 2
