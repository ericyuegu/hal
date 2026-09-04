"""Dense, shard-ordered replay loading for experiment O51.

Mosaic Streaming owns manifests, shard download, decompression, and MDS row
decoding.  This module owns the order in which physical shards are read and the
identity-uniform sampling of four-window replay generations.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import shutil
import threading
import time
from collections import deque
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Final
from typing import cast

import numpy as np
import torch
from streaming import Stream
from streaming import StreamingDataset
from streaming.base.format.mds import MDSReader
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Sampler

from hal import streams as streams_lib
from hal.data.policy_world_schema import decode_policy_world_replay_slices
from hal.data.schema import check_schema_version
from hal.training.dataloader import ReplayLabels
from hal.training.dataloader import _make_window
from hal.training.dataloader import collate_train_batch
from hal.training.features import ITEM_COLUMNS
from hal.training.features import AWRBatch as O51AWRBatch
from hal.training.features import ExtraColumns
from hal.training.features import FeatureProjection
from hal.training.features import TrainBatch
from hal.training.o51_data import DATA_PROTOCOL
from hal.training.o51_data import O51_RETURN_SUFFIX
from hal.training.o51_data import SOURCE_MANIFEST_SHA256
from hal.training.o51_data import TierSelection

WINDOWS_PER_GENERATION: Final[int] = 4
DEFAULT_REPLAY_SLOTS: Final[int] = 131_072
PREFETCH_FACTOR: Final[int] = 1
CHECKPOINT_SCHEMA: Final[int] = 1
RESERVED_DISK_BYTES: Final[int] = 256 * 2**30
O51_L_CTX: Final[int] = 256
O51_L_CHUNK: Final[int] = 20
O51_SCHEMA_VERSION: Final[int] = 7
O51_EXTRA_COLUMNS: Final[ExtraColumns] = ExtraColumns(
    floats=ITEM_COLUMNS.floats,
    cats={**ITEM_COLUMNS.cats, "player_id": None},
)

type BatchTransform = Callable[[list[dict[str, np.ndarray]], TrainBatch], object]


def _collate_o51_batch(windows: list[dict[str, np.ndarray]], batch: TrainBatch) -> O51AWRBatch:
    next_frames = slice(1, O51_L_CTX + 1)
    return_name = f"ego_{O51_RETURN_SUFFIX}"
    valid_name = f"{return_name}_valid"
    returns = np.stack([window[return_name] for window in windows])[:, next_frames]
    eligible = np.stack([window[valid_name] for window in windows])[:, next_frames]
    return O51AWRBatch(
        batch=batch,
        returns=torch.from_numpy(np.ascontiguousarray(returns)),
        eligible=torch.from_numpy(np.ascontiguousarray(eligible)).bool(),
    )


@dataclass(frozen=True, slots=True)
class SourceManifest:
    """The shard geometry needed by the pure O51 planner."""

    source: str
    samples_per_shard: tuple[int, ...]
    raw_bytes_per_shard: tuple[int, ...] = ()
    zip_bytes_per_shard: tuple[int, ...] = ()
    raw_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if not self.source or not self.samples_per_shard:
            raise ValueError("a source manifest must contain at least one shard")
        if any(samples < 1 for samples in self.samples_per_shard):
            raise ValueError(f"{self.source} contains an empty shard")
        for name, values in (
            ("raw byte", self.raw_bytes_per_shard),
            ("compressed byte", self.zip_bytes_per_shard),
            ("raw path", self.raw_paths),
        ):
            if values and len(values) != len(self.samples_per_shard):
                raise ValueError(f"{self.source} {name} metadata does not cover every shard")


@dataclass(frozen=True, slots=True)
class HostMemoryEstimate:
    """Conservative O51 host-memory components."""

    central_buffer: int
    queued_shards: int
    worker_outputs_and_workspaces: int
    ipc_copies: int
    parent_result: int
    pinned_batches: int
    validation_cache: int
    compiler_and_process: int

    @property
    def peak_bytes(self) -> int:
        return sum(
            (
                self.central_buffer,
                self.queued_shards,
                self.worker_outputs_and_workspaces,
                self.ipc_copies,
                self.parent_result,
                self.pinned_batches,
                self.validation_cache,
                self.compiler_and_process,
            )
        )


def estimate_host_memory(
    *,
    central_buffer_bytes: int,
    decoded_shard_bytes: int,
    replay_workspace_bytes: int,
    pinned_batch_bytes: int,
    validation_cache_bytes: int,
    compiler_and_process_bytes: int,
    workers: int,
) -> HostMemoryEstimate:
    """Model every resident buffer involved in one-shard prefetch."""
    values = (
        central_buffer_bytes,
        decoded_shard_bytes,
        replay_workspace_bytes,
        pinned_batch_bytes,
        validation_cache_bytes,
        compiler_and_process_bytes,
        workers,
    )
    if any(value < 0 for value in values):
        raise ValueError("host-memory inputs must be non-negative")
    queued = workers * PREFETCH_FACTOR * decoded_shard_bytes
    worker_outputs = workers * (decoded_shard_bytes + replay_workspace_bytes)
    return HostMemoryEstimate(
        central_buffer=central_buffer_bytes,
        queued_shards=queued,
        worker_outputs_and_workspaces=worker_outputs,
        ipc_copies=queued,
        parent_result=decoded_shard_bytes,
        pinned_batches=2 * pinned_batch_bytes,
        validation_cache=validation_cache_bytes,
        compiler_and_process=compiler_and_process_bytes,
    )


@dataclass(frozen=True, slots=True)
class ShardTask:
    """The selected local rows from one physical MDS shard."""

    source: str
    shard: int
    row_start: int
    row_stop: int
    excluded_rows: tuple[int, ...] = ()
    source_index: int = 0
    global_shard: int = -1

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("shard task source must not be empty")
        if self.shard < 0 or self.source_index < 0:
            raise ValueError("shard indices must be non-negative")
        if not 0 <= self.row_start < self.row_stop:
            raise ValueError("a shard task must select a non-empty row range")
        if self.excluded_rows != tuple(sorted(set(self.excluded_rows))):
            raise ValueError("shard exclusions must be sorted and unique")
        if any(not self.row_start <= row < self.row_stop for row in self.excluded_rows):
            raise ValueError("a shard exclusion is outside the selected row range")

    @property
    def row_count(self) -> int:
        return self.row_stop - self.row_start - len(self.excluded_rows)

    @property
    def selected_rows(self) -> tuple[int, ...]:
        excluded = set(self.excluded_rows)
        return tuple(row for row in range(self.row_start, self.row_stop) if row not in excluded)


@dataclass(frozen=True, slots=True)
class PhysicalRow:
    """A stable row locator independent of Mosaic's global sample map."""

    source: str
    shard: int
    row: int


@dataclass(frozen=True, slots=True)
class ShardRequest:
    """One ordered worker request."""

    epoch: int
    task_offset: int
    task_index: int


@dataclass(frozen=True, slots=True)
class ShardTimings:
    """Worker-side wall times for one physical shard."""

    prepare_seconds: float = 0.0
    download_seconds: float = 0.0
    decompress_seconds: float = 0.0
    read_seconds: float = 0.0
    decode_seconds: float = 0.0
    worker_seconds: float = 0.0

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.values()):
            raise ValueError("shard timings must be non-negative")

    def values(self) -> tuple[float, ...]:
        return (
            self.prepare_seconds,
            self.download_seconds,
            self.decompress_seconds,
            self.read_seconds,
            self.decode_seconds,
            self.worker_seconds,
        )


@dataclass(frozen=True, slots=True)
class DecodedShard:
    """Fixed four-window rows returned by one worker."""

    task: ShardTask
    epoch: int
    task_offset: int
    replay_ids: tuple[str, ...]
    locators: tuple[PhysicalRow, ...]
    columns: Mapping[str, np.ndarray]
    raw_bytes_read: int = 0
    timings: ShardTimings = ShardTimings()
    worker_finished_ns: int = 0

    def __post_init__(self) -> None:
        rows = len(self.replay_ids)
        if len(self.locators) != rows:
            raise ValueError("decoded shard row metadata does not match its task")
        if rows < 1:
            raise ValueError("a decoded shard must contain at least one row")
        if self.task_offset >= 0 and rows != self.task.row_count:
            raise ValueError("worker output does not cover every selected task row")
        if not self.columns:
            raise ValueError("a decoded shard has no columns")
        if self.raw_bytes_read < 0:
            raise ValueError("raw bytes read must be non-negative")
        if self.worker_finished_ns < 0:
            raise ValueError("worker completion time must be non-negative")
        bad = {
            name: value.shape
            for name, value in self.columns.items()
            if not isinstance(value, np.ndarray) or value.ndim < 2 or value.shape[:2] != (rows, WINDOWS_PER_GENERATION)
        }
        if bad:
            raise ValueError(f"decoded shard columns must begin [R, 4], got {bad}")

    def windows(self, row: int) -> tuple[dict[str, np.ndarray], ...]:
        if not 0 <= row < len(self.replay_ids):
            raise IndexError(row)
        return tuple(
            {name: values[row, window] for name, values in self.columns.items()}
            for window in range(WINDOWS_PER_GENERATION)
        )


def _coerce_manifest(source: str, value: SourceManifest | Sequence[int]) -> SourceManifest:
    if isinstance(value, SourceManifest):
        if value.source != source:
            raise ValueError(f"manifest key {source!r} does not match manifest source {value.source!r}")
        return value
    return SourceManifest(source, tuple(int(samples) for samples in value))


def build_shard_plan(
    tier: TierSelection,
    manifests: Mapping[str, SourceManifest | Sequence[int]],
) -> tuple[ShardTask, ...]:
    """Expose each selected prefix row exactly once, grouped by physical shard."""
    if set(manifests) != {view.source for view in tier.sources}:
        missing = {view.source for view in tier.sources} - set(manifests)
        extra = set(manifests) - {view.source for view in tier.sources}
        raise ValueError(
            f"manifest sources do not match U{tier.scale}: missing={sorted(missing)}, extra={sorted(extra)}"
        )

    tasks: list[ShardTask] = []
    global_shard = 0
    for source_index, view in enumerate(tier.sources):
        manifest = _coerce_manifest(view.source, manifests[view.source])
        source_row = 0
        selected = 0
        for shard, samples in enumerate(manifest.samples_per_shard):
            selected_stop = min(samples, max(0, view.stop - source_row))
            if selected_stop:
                exclusions = tuple(
                    row - source_row for row in view.excluded_rows if source_row <= row < source_row + selected_stop
                )
                task = ShardTask(
                    source=view.source,
                    shard=shard,
                    row_start=0,
                    row_stop=selected_stop,
                    excluded_rows=exclusions,
                    source_index=source_index,
                    global_shard=global_shard,
                )
                tasks.append(task)
                selected += task.row_count
            source_row += samples
            global_shard += 1
        if view.stop > source_row:
            raise ValueError(
                f"U{tier.scale} selects {view.stop} rows from {view.source}, but its manifest has {source_row}"
            )
        if selected != view.unique_replays:
            raise RuntimeError(
                f"shard plan selected {selected} rows from {view.source}, expected {view.unique_replays}"
            )
    if sum(task.row_count for task in tasks) != tier.unique_replays:
        raise RuntimeError("shard plan row total does not match the tier")
    return tuple(tasks)


def _task_key(task: ShardTask, seed: int, epoch: int) -> bytes:
    value = f"{seed}\0{epoch}\0{task.source}\0{task.shard}".encode()
    return hashlib.blake2b(value, digest_size=16, person=b"hal-o51-shards").digest()


def permute_shard_tasks(tasks: Sequence[ShardTask], *, seed: int, epoch: int) -> tuple[int, ...]:
    """Return the complete keyed shard permutation for one source epoch."""
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    return tuple(
        sorted(
            range(len(tasks)),
            key=lambda index: (_task_key(tasks[index], seed, epoch), tasks[index].source, tasks[index].shard),
        )
    )


def _uniform_weak_composition(total: int, parts: int, rng: np.random.Generator) -> np.ndarray:
    if total < 0 or parts < 1:
        raise ValueError("weak-composition inputs are invalid")
    if parts == 1:
        return np.asarray([total], dtype=np.int64)
    bars = np.sort(rng.choice(total + parts - 1, size=parts - 1, replace=False))
    boundaries = np.concatenate((np.asarray([-1]), bars, np.asarray([total + parts - 1])))
    return np.diff(boundaries).astype(np.int64) - 1


def choose_o51_window_starts(
    frames: int,
    L_ctx: int,
    L_chunk: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose four uniformly phased, circularly spaced full windows."""
    length = L_ctx + L_chunk
    starts = frames - length + 1
    required = WINDOWS_PER_GENERATION * length
    if starts < required:
        raise ValueError(f"{frames} frames provide {starts} starts; four windows require at least {required}")
    gaps = _uniform_weak_composition(starts - required, WINDOWS_PER_GENERATION, rng)
    phase = int(rng.integers(starts))
    offsets = np.concatenate((np.asarray([0]), np.cumsum(length + gaps[:-1])))
    spaced = (phase + offsets) % starts
    return spaced[rng.permutation(WINDOWS_PER_GENERATION)].astype(np.int64, copy=False)


def _replay_checksum(replay_id: str) -> int:
    return int.from_bytes(hashlib.blake2b(replay_id.encode(), digest_size=8).digest(), "little")


def _generation_rng(seed: int, epoch: int, replay_id: str) -> np.random.Generator:
    checksum = _replay_checksum(replay_id)
    return np.random.default_rng((seed, epoch, checksum & 0xFFFFFFFF, checksum >> 32))


def _stack_window_rows(rows: Sequence[tuple[dict[str, np.ndarray], ...]]) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("cannot stack an empty shard")
    names = tuple(sorted(rows[0][0]))
    expected_names = set(names)
    arrays: dict[str, np.ndarray] = {}
    for row in rows:
        if len(row) != WINDOWS_PER_GENERATION or any(set(window) != expected_names for window in row):
            raise ValueError("decoded windows do not have fixed keys")
    for name in names:
        reference = rows[0][0][name]
        shape = reference.shape
        dtype = reference.dtype
        out = np.empty((len(rows), WINDOWS_PER_GENERATION, *shape), dtype=dtype)
        for row_index, row in enumerate(rows):
            for window_index, window in enumerate(row):
                value = window[name]
                if value.shape != shape or value.dtype != dtype:
                    raise ValueError(f"decoded column {name!r} changed shape or dtype within a shard")
                out[row_index, window_index] = value
        arrays[name] = out
    return arrays


def _decode_generation(
    compact: Mapping[str, object],
    *,
    task: ShardTask,
    row: int,
    epoch: int,
    seed: int,
    L_ctx: int,
    L_chunk: int,
    schema_version: int,
    labels: ReplayLabels,
    projection: FeatureProjection | None,
) -> tuple[str, tuple[dict[str, np.ndarray], ...]]:
    replay_id = str(compact["replay_id"])
    frames = int(cast(Any, compact["num_frames"]))
    required_frames = 5 * (L_ctx + L_chunk) - 1
    if frames < required_frames:
        raise ValueError(
            f"short replay {replay_id!r} at source={task.source} shard={task.shard} row={row}: "
            f"frame_count={frames}, required_count={required_frames}"
        )
    source_schema_version = int(cast(Any, compact["source_schema_version"]))
    check_schema_version({"schema_version": source_schema_version}, expected=schema_version)
    rng = _generation_rng(seed, epoch, replay_id)
    starts = choose_o51_window_starts(frames, L_ctx, L_chunk, rng)
    ranges = tuple((int(start), int(start) + L_ctx + L_chunk) for start in starts)
    decoded = decode_policy_world_replay_slices(compact, ranges)
    replay_labels = {name: np.asarray(value) for name, value in labels(compact).items()}
    wrong_labels = {name: value.shape for name, value in replay_labels.items() if value.shape not in ((), (frames,))}
    if wrong_labels:
        raise ValueError(f"replay labels have invalid shapes {wrong_labels}; expected scalar or {(frames,)}")

    windows: list[dict[str, np.ndarray]] = []
    for start, decoded_slice in zip(starts, decoded, strict=True):
        start_int = int(start)
        sample = dict(decoded_slice)
        sample.update(
            {
                name: (
                    np.full(L_ctx + L_chunk, value.item(), dtype=value.dtype)
                    if value.shape == ()
                    else value[start_int : start_int + L_ctx + L_chunk]
                )
                for name, value in replay_labels.items()
            }
        )
        ego = "p1" if rng.random() < 0.5 else "p2"
        window = _make_window(
            sample,
            ego_prefix=ego,
            start=0,
            pad=0,
            length=L_ctx + L_chunk,
            projection=projection,
        )
        window["ctx_pad"] = np.asarray(0, dtype=np.int64)
        windows.append({name: np.ascontiguousarray(value) for name, value in window.items()})
    return replay_id, tuple(windows)


class MDSStorageAdapter:
    """Narrow Mosaic Streaming 0.13 adapter used by O51 workers."""

    def __init__(
        self,
        tier: TierSelection,
        *,
        split: str = "train",
        download_retry: int = 8,
    ) -> None:
        installed = importlib.metadata.version("mosaicml-streaming")
        if installed != "0.13.0":
            raise RuntimeError(f"O51 requires mosaicml-streaming==0.13.0, found {installed}")
        sources = tuple(streams_lib.BY_NAME[view.source] for view in tier.sources)
        mosaic_streams = [
            Stream(
                remote=source.remote,
                local=str(source.local_root),
                split=split,
                choose=view.unique_replays,
                download_retry=download_retry,
                keep_zip=False,
            )
            for source, view in zip(sources, tier.sources, strict=True)
        ]
        self.dataset = StreamingDataset(
            streams=mosaic_streams,
            shuffle=False,
            predownload=None,
            cache_limit=None,
            keep_zip=False,
            batch_size=1,
        )
        self.tier = tier
        self.split = split
        self.manifests = self._manifests()
        self.source_manifest_sha256 = {view.source: SOURCE_MANIFEST_SHA256[view.source] for view in tier.sources}
        self.last_read_bytes = 0

    def _manifests(self) -> dict[str, SourceManifest]:
        manifests: dict[str, SourceManifest] = {}
        for source_index, view in enumerate(self.tier.sources):
            begin = int(self.dataset.shard_offset_per_stream[source_index])
            stop = begin + int(self.dataset.shards_per_stream[source_index])
            shards = self.dataset.shards[begin:stop]
            stream = self.dataset.streams[source_index]
            manifests[view.source] = SourceManifest(
                source=view.source,
                samples_per_shard=tuple(int(shard.samples) for shard in shards),
                raw_bytes_per_shard=tuple(int(shard.raw_data.bytes) for shard in shards),
                zip_bytes_per_shard=tuple(
                    0 if shard.zip_data is None else int(shard.zip_data.bytes) for shard in shards
                ),
                raw_paths=tuple(Path(stream.local) / stream.split / shard.raw_data.basename for shard in shards),
            )
        return manifests

    def _reader(self, task: ShardTask) -> MDSReader:
        reader = self.dataset.shards[task.global_shard]
        if not isinstance(reader, MDSReader):
            raise TypeError(f"O51 requires MDS shards, found {type(reader).__name__}")
        return reader

    def _prepare_shard_profiled(self, task: ShardTask) -> tuple[float, float, float]:
        """Prepare one shard while timing Mosaic's download and decompression calls."""
        stream_id = int(self.dataset.stream_per_shard[task.global_shard])
        stream = self.dataset.streams[stream_id]
        original_download = stream._download_file
        original_decompress = stream._decompress_shard_part
        stream_attributes = vars(stream)
        prior_download = stream_attributes.get("_download_file")
        prior_decompress = stream_attributes.get("_decompress_shard_part")
        had_download_override = "_download_file" in stream_attributes
        had_decompress_override = "_decompress_shard_part" in stream_attributes
        download_seconds = 0.0
        decompress_seconds = 0.0

        def timed_download(*args: Any, **kwargs: Any) -> Any:
            nonlocal download_seconds
            started = time.perf_counter()
            try:
                return original_download(*args, **kwargs)
            finally:
                download_seconds += time.perf_counter() - started

        def timed_decompress(*args: Any, **kwargs: Any) -> Any:
            nonlocal decompress_seconds
            started = time.perf_counter()
            try:
                return original_decompress(*args, **kwargs)
            finally:
                decompress_seconds += time.perf_counter() - started

        stream._download_file = timed_download  # type: ignore[invalid-assignment]
        stream._decompress_shard_part = timed_decompress  # type: ignore[invalid-assignment]
        started = time.perf_counter()
        try:
            self.dataset.prepare_shard(task.global_shard)
        finally:
            prepare_seconds = time.perf_counter() - started
            if had_download_override:
                stream._download_file = prior_download  # type: ignore[invalid-assignment]
            else:
                del stream._download_file
            if had_decompress_override:
                stream._decompress_shard_part = prior_decompress  # type: ignore[invalid-assignment]
            else:
                del stream._decompress_shard_part
        return prepare_seconds, download_seconds, decompress_seconds

    def read_rows(self, task: ShardTask, rows: Sequence[int]) -> dict[int, Mapping[str, object]]:
        """Prepare a shard, open it once, and decode requested rows in byte order."""
        requested = tuple(sorted(set(int(row) for row in rows)))
        if not requested:
            return {}
        if any(row < task.row_start or row >= task.row_stop or row in task.excluded_rows for row in requested):
            raise ValueError("requested row is outside its shard task")
        self.dataset.prepare_shard(task.global_shard)
        reader = self._reader(task)
        filename = Path(reader.dirname) / (reader.split or "") / reader.raw_data.basename
        with filename.open("rb", buffering=0) as handle:
            table = handle.read(4 * (reader.samples + 2))
            offsets = np.frombuffer(table, dtype=np.uint32)
            if len(offsets) != reader.samples + 2 or int(offsets[0]) != reader.samples:
                raise ValueError(f"invalid MDS offset table in {filename}")
            decoded: dict[int, Mapping[str, object]] = {}
            payload_bytes = 0
            for row in requested:
                begin, end = int(offsets[row + 1]), int(offsets[row + 2])
                handle.seek(begin)
                payload = handle.read(end - begin)
                if len(payload) != end - begin:
                    raise EOFError(f"short MDS row read from {filename} at row {row}")
                payload_bytes += len(payload)
                decoded[row] = reader.decode_sample(payload)
        self.last_read_bytes = len(table) + payload_bytes
        return decoded

    def decode_generations(
        self,
        task: ShardTask,
        requests: Sequence[tuple[int, int]],
        *,
        seed: int,
        L_ctx: int,
        L_chunk: int,
        schema_version: int,
        labels: ReplayLabels,
        projection: FeatureProjection | None,
    ) -> dict[tuple[int, int], tuple[str, tuple[dict[str, np.ndarray], ...]]]:
        """Rebuild requested row/epoch generations with one physical shard read."""
        compact_rows = self.read_rows(task, [row for row, _ in requests])
        out = {}
        for row, epoch in requests:
            out[(row, epoch)] = _decode_generation(
                compact_rows[row],
                task=task,
                row=row,
                epoch=epoch,
                seed=seed,
                L_ctx=L_ctx,
                L_chunk=L_chunk,
                schema_version=schema_version,
                labels=labels,
                projection=projection,
            )
        return out

    def decode_task(
        self,
        task: ShardTask,
        epoch: int,
        task_offset: int,
        *,
        seed: int,
        L_ctx: int,
        L_chunk: int,
        schema_version: int,
        labels: ReplayLabels,
        projection: FeatureProjection | None,
    ) -> DecodedShard:
        worker_started = time.perf_counter()
        rows = task.selected_rows
        prepare_seconds, download_seconds, decompress_seconds = self._prepare_shard_profiled(task)
        reader = self._reader(task)
        filename = Path(reader.dirname) / (reader.split or "") / reader.raw_data.basename
        replay_ids: list[str] = []
        columns: dict[str, np.ndarray] = {}
        raw_bytes_read = 0
        read_seconds = 0.0
        decode_seconds = 0.0
        with filename.open("rb", buffering=0) as handle:
            read_started = time.perf_counter()
            table = handle.read(4 * (reader.samples + 2))
            offsets = np.frombuffer(table, dtype=np.uint32)
            read_seconds += time.perf_counter() - read_started
            if len(offsets) != reader.samples + 2 or int(offsets[0]) != reader.samples:
                raise ValueError(f"invalid MDS offset table in {filename}")
            raw_bytes_read += len(table)
            for output_row, row in enumerate(rows):
                read_started = time.perf_counter()
                begin, end = int(offsets[row + 1]), int(offsets[row + 2])
                handle.seek(begin)
                payload = handle.read(end - begin)
                read_seconds += time.perf_counter() - read_started
                if len(payload) != end - begin:
                    raise EOFError(f"short MDS row read from {filename} at row {row}")
                raw_bytes_read += len(payload)
                decode_started = time.perf_counter()
                replay_id, windows = _decode_generation(
                    reader.decode_sample(payload),
                    task=task,
                    row=row,
                    epoch=epoch,
                    seed=seed,
                    L_ctx=L_ctx,
                    L_chunk=L_chunk,
                    schema_version=schema_version,
                    labels=labels,
                    projection=projection,
                )
                replay_ids.append(replay_id)
                if not columns:
                    columns = {
                        name: np.empty((len(rows), WINDOWS_PER_GENERATION, *value.shape), dtype=value.dtype)
                        for name, value in sorted(windows[0].items())
                    }
                expected_names = set(columns)
                if any(set(window) != expected_names for window in windows):
                    raise ValueError("decoded windows do not have fixed keys")
                for window_index, window in enumerate(windows):
                    for name, destination in columns.items():
                        value = window[name]
                        if value.shape != destination.shape[2:] or value.dtype != destination.dtype:
                            raise ValueError(f"decoded column {name!r} changed shape or dtype within a shard")
                        destination[output_row, window_index] = value
                decode_seconds += time.perf_counter() - decode_started
        worker_seconds = time.perf_counter() - worker_started
        return DecodedShard(
            task=task,
            epoch=epoch,
            task_offset=task_offset,
            replay_ids=tuple(replay_ids),
            locators=tuple(PhysicalRow(task.source, task.shard, row) for row in rows),
            columns=columns,
            raw_bytes_read=raw_bytes_read,
            timings=ShardTimings(
                prepare_seconds=prepare_seconds,
                download_seconds=download_seconds,
                decompress_seconds=decompress_seconds,
                read_seconds=read_seconds,
                decode_seconds=decode_seconds,
                worker_seconds=worker_seconds,
            ),
            worker_finished_ns=time.monotonic_ns(),
        )


def disk_requirement_bytes(
    tasks: Sequence[ShardTask],
    manifests: Mapping[str, SourceManifest],
    *,
    workers: int,
    reserved_bytes: int = RESERVED_DISK_BYTES,
) -> int:
    """Return new disk bytes needed for selected shards and concurrent downloads."""
    if workers < 0 or reserved_bytes < 0:
        raise ValueError("disk requirement inputs must be non-negative")
    missing_raw = 0
    compressed: list[int] = []
    for task in tasks:
        manifest = manifests[task.source]
        raw_bytes = manifest.raw_bytes_per_shard[task.shard]
        raw_path = manifest.raw_paths[task.shard]
        valid_raw = raw_path.is_file() and raw_path.stat().st_size == raw_bytes
        if not valid_raw:
            missing_raw += raw_bytes
            compressed.append(manifest.zip_bytes_per_shard[task.shard])
    compressed.sort(reverse=True)
    return missing_raw + sum(compressed[:workers]) + reserved_bytes


class _ShardDataset(Dataset[DecodedShard]):
    def __init__(
        self,
        adapter: MDSStorageAdapter,
        tasks: tuple[ShardTask, ...],
        *,
        seed: int,
        L_ctx: int,
        L_chunk: int,
        schema_version: int,
        labels: ReplayLabels,
        projection: FeatureProjection | None,
    ) -> None:
        self.adapter = adapter
        self.tasks = tasks
        self.seed = seed
        self.L_ctx = L_ctx
        self.L_chunk = L_chunk
        self.schema_version = schema_version
        self.labels = labels
        self.projection = projection

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, index: ShardRequest) -> DecodedShard:
        return self.adapter.decode_task(
            self.tasks[index.task_index],
            index.epoch,
            index.task_offset,
            seed=self.seed,
            L_ctx=self.L_ctx,
            L_chunk=self.L_chunk,
            schema_version=self.schema_version,
            labels=self.labels,
            projection=self.projection,
        )


class _ShardSampler(Sampler[ShardRequest]):
    def __init__(self, tasks: tuple[ShardTask, ...], seed: int, cursor: tuple[int, int, int]) -> None:
        self.tasks = tasks
        self.seed = seed
        self.cursor = cursor

    def __iter__(self) -> Iterator[ShardRequest]:
        epoch, task_offset, _ = self.cursor
        while True:
            order = permute_shard_tasks(self.tasks, seed=self.seed, epoch=epoch)
            for offset in range(task_offset, len(order)):
                yield ShardRequest(epoch, offset, order[offset])
            epoch += 1
            task_offset = 0

    def __len__(self) -> int:
        return 2**63 - 1


def _limit_worker_threads(_worker_id: int) -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)


def _identity(value: DecodedShard) -> DecodedShard:
    return value


class _Fenwick:
    """Order statistics over canonical identity representatives."""

    def __init__(self, size: int) -> None:
        self.values = np.zeros(size, dtype=np.uint8)
        self.tree = np.zeros(size + 1, dtype=np.int64)

    @property
    def total(self) -> int:
        return int(self.tree[-1]) if len(self.tree) == 2 else int(self.prefix(len(self.values)))

    def prefix(self, stop: int) -> int:
        total = 0
        index = stop
        while index:
            total += int(self.tree[index])
            index -= index & -index
        return total

    def set(self, index: int, present: bool) -> None:
        target = int(present)
        delta = target - int(self.values[index])
        if not delta:
            return
        self.values[index] = target
        cursor = index + 1
        while cursor < len(self.tree):
            self.tree[cursor] += delta
            cursor += cursor & -cursor

    def select(self, rank: int) -> int:
        if not 0 <= rank < self.prefix(len(self.values)):
            raise IndexError(rank)
        index = 0
        bit = 1 << (len(self.values).bit_length() - 1)
        while bit:
            candidate = index + bit
            if candidate < len(self.tree) and self.tree[candidate] <= rank:
                index = candidate
                rank -= int(self.tree[candidate])
            bit >>= 1
        return index


@dataclass(frozen=True, slots=True)
class GenerationDescriptor:
    slot: int
    locator: PhysicalRow
    epoch: int
    replay_checksum: int
    next_window: int


class ReplayBuffer:
    """Columnar replay generations sampled uniformly by active identity."""

    def __init__(self, capacity: int, batch_size: int, seed: int) -> None:
        if capacity < batch_size:
            raise ValueError("replay capacity must cover one identity-distinct batch")
        self.capacity = capacity
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.columns: dict[str, np.ndarray] = {}
        self.replay_ids: list[str | None] = [None] * capacity
        self.locators: list[PhysicalRow | None] = [None] * capacity
        self.epochs = np.full(capacity, -1, dtype=np.int64)
        self.next_windows = np.zeros(capacity, dtype=np.uint8)
        self.identity_slots: dict[str, set[int]] = {}
        self.representatives = _Fenwick(capacity)
        self.free_slots = list(range(capacity - 1, -1, -1))
        self.batch_index = 0
        self.last_seen: dict[str, int] = {}
        self.last_metrics: dict[str, float] = {}
        self.last_sampled_identity_ranks: tuple[int, ...] = ()
        self.last_sampled_identity_count = 0

    @property
    def size(self) -> int:
        return self.capacity - len(self.free_slots)

    @property
    def active_identities(self) -> int:
        return len(self.identity_slots)

    def _set_representative(self, replay_id: str, old: int | None) -> None:
        if old is not None:
            self.representatives.set(old, False)
        slots = self.identity_slots.get(replay_id)
        if slots:
            self.representatives.set(min(slots), True)

    def _initialize_columns(self, shard: DecodedShard) -> None:
        if self.columns:
            return
        for name, values in shard.columns.items():
            self.columns[name] = np.empty((self.capacity, *values.shape[1:]), dtype=values.dtype)

    def admit(self, shard: DecodedShard, row: int, *, slot: int | None = None) -> int:
        """Copy one decoded generation into a free slot."""
        self._initialize_columns(shard)
        if set(self.columns) != set(shard.columns):
            raise ValueError("decoded shard columns changed after buffer allocation")
        if slot is None:
            if not self.free_slots:
                raise RuntimeError("replay buffer is full")
            slot = self.free_slots.pop()
        elif slot in self.free_slots:
            if self.free_slots[-1] == slot:
                self.free_slots.pop()
            else:
                self.free_slots.remove(slot)
        elif self.replay_ids[slot] is not None:
            raise RuntimeError(f"replay slot {slot} is already occupied")
        for name, destination in self.columns.items():
            source = shard.columns[name][row]
            if source.shape != destination.shape[1:] or source.dtype != destination.dtype:
                raise ValueError(f"decoded column {name!r} does not match the central buffer")
            destination[slot] = source
        replay_id = shard.replay_ids[row]
        existing = self.identity_slots.get(replay_id)
        old_representative = None if not existing else min(existing)
        self.replay_ids[slot] = replay_id
        self.locators[slot] = shard.locators[row]
        self.epochs[slot] = shard.epoch
        self.next_windows[slot] = 0
        self.identity_slots.setdefault(replay_id, set()).add(slot)
        self._set_representative(replay_id, old_representative)
        return slot

    def remove(self, slot: int) -> None:
        replay_id = self.replay_ids[slot]
        if replay_id is None:
            raise RuntimeError(f"replay slot {slot} is empty")
        slots = self.identity_slots[replay_id]
        old_representative = min(slots)
        slots.remove(slot)
        if not slots:
            del self.identity_slots[replay_id]
        self._set_representative(replay_id, old_representative)
        self.replay_ids[slot] = None
        self.locators[slot] = None
        self.epochs[slot] = -1
        self.next_windows[slot] = 0
        self.free_slots.append(slot)

    def sample(self) -> tuple[tuple[str, ...], list[dict[str, np.ndarray]], tuple[int, ...]]:
        """Consume one window from 512 distinct, identity-uniform replays."""
        if self.active_identities < self.batch_size:
            raise RuntimeError(
                f"replay buffer has {self.active_identities} identities, but a batch needs {self.batch_size}"
            )
        self.last_sampled_identity_count = self.active_identities
        ranks = self.rng.choice(self.last_sampled_identity_count, size=self.batch_size, replace=False)
        self.last_sampled_identity_ranks = tuple(int(rank) for rank in ranks)
        slots: list[int] = []
        replay_ids: list[str] = []
        windows: list[dict[str, np.ndarray]] = []
        exhausted: list[int] = []
        ages: list[int] = []
        for rank in ranks:
            representative = self.representatives.select(int(rank))
            replay_id = self.replay_ids[representative]
            assert replay_id is not None
            generations = sorted(self.identity_slots[replay_id])
            slot = generations[int(self.rng.integers(len(generations)))]
            ordinal = int(self.next_windows[slot])
            replay_ids.append(replay_id)
            slots.append(slot)
            windows.append({name: values[slot, ordinal] for name, values in self.columns.items()})
            self.next_windows[slot] += 1
            if self.next_windows[slot] == WINDOWS_PER_GENERATION:
                exhausted.append(slot)
            previous = self.last_seen.get(replay_id)
            if previous is not None:
                ages.append(self.batch_index - previous)
            self.last_seen[replay_id] = self.batch_index
        self.batch_index += 1
        age_values = np.asarray(ages, dtype=np.float64)
        self.last_metrics = {
            "data/replay_age_p01": float(np.percentile(age_values, 1)) if ages else float("nan"),
            "data/replay_age_p05": float(np.percentile(age_values, 5)) if ages else float("nan"),
            "data/replay_age_p50": float(np.percentile(age_values, 50)) if ages else float("nan"),
            "data/replay_age_p95": float(np.percentile(age_values, 95)) if ages else float("nan"),
            "data/replay_age_le_1_fraction": float(np.mean(age_values <= 1)) if ages else 0.0,
            "data/replay_age_le_16_fraction": float(np.mean(age_values <= 16)) if ages else 0.0,
            "data/active_replay_identities": float(self.active_identities),
            "data/duplicate_generations": float(self.size - self.active_identities),
        }
        return tuple(replay_ids), windows, tuple(exhausted)

    def descriptors(self) -> tuple[GenerationDescriptor, ...]:
        out = []
        for slot, replay_id in enumerate(self.replay_ids):
            if replay_id is None:
                continue
            locator = self.locators[slot]
            assert locator is not None
            out.append(
                GenerationDescriptor(
                    slot=slot,
                    locator=locator,
                    epoch=int(self.epochs[slot]),
                    replay_checksum=_replay_checksum(replay_id),
                    next_window=int(self.next_windows[slot]),
                )
            )
        return tuple(out)


class _O51Iterator(Iterator[object]):
    def __init__(self, loader: O51ReplayLoader, shards: Iterator[DecodedShard]) -> None:
        self.loader = loader
        self.shards = shards
        self.current: DecodedShard | None = None
        self._active = False

    def __iter__(self) -> _O51Iterator:
        return self

    def __next__(self) -> object:
        if self._active:
            raise RuntimeError("concurrent next() calls are not supported")
        self._active = True
        self.loader._parent_next_active = True
        try:
            return self.loader._next_batch(self)
        finally:
            self.loader._parent_next_active = False
            self._active = False

    def take_generation(self) -> tuple[DecodedShard, int]:
        while self.current is None or self.loader._cursor[2] >= len(self.current.replay_ids):
            if self.current is not None:
                self.loader._advance_task_cursor()
            wait_started = time.perf_counter()
            decoded = next(self.shards)
            received_ns = time.monotonic_ns()
            parent_wait_seconds = time.perf_counter() - wait_started
            self.loader._raw_bytes_read += decoded.raw_bytes_read
            decoded_bytes = sum(values.nbytes for values in decoded.columns.values())
            self.loader._max_decoded_shard_bytes = max(self.loader._max_decoded_shard_bytes, decoded_bytes)
            epoch, task_offset, row_offset = self.loader._cursor
            if (decoded.epoch, decoded.task_offset) != (epoch, task_offset):
                raise RuntimeError(
                    "DataLoader released a shard out of committed order: "
                    f"got {(decoded.epoch, decoded.task_offset)}, expected {(epoch, task_offset)}"
                )
            self.loader._record_shard_profile(decoded, received_ns, parent_wait_seconds)
            self.current = decoded
            if row_offset > len(decoded.replay_ids):
                raise RuntimeError("committed row cursor exceeds its shard")
        row = self.loader._cursor[2]
        self.loader._cursor = (self.loader._cursor[0], self.loader._cursor[1], row + 1)
        self.loader._generations_read += 1
        return self.current, row


class O51ReplayLoader:
    """Infinite O51 batch loader with exact descriptor-only checkpoints."""

    separate_identity_checkpoint: Final[bool] = True

    def __init__(
        self,
        *,
        tier: TierSelection,
        adapter: MDSStorageAdapter,
        tasks: tuple[ShardTask, ...],
        stats: dict[str, Any],
        labels: ReplayLabels,
        projection: FeatureProjection | None,
        batch_size: int,
        replay_slots: int,
        seed: int,
        num_workers: int,
        L_ctx: int,
        L_chunk: int,
        schema_version: int,
        extra: ExtraColumns | None,
        batch_transform: BatchTransform | None,
        pin_memory: bool,
    ) -> None:
        if replay_slots > tier.unique_replays:
            replay_slots = tier.unique_replays
        if replay_slots < batch_size:
            raise ValueError("selected tier is too small for one identity-distinct batch")
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        self.tier = tier
        self.adapter = adapter
        self.tasks = tasks
        self.stats = stats
        self.labels = labels
        self.projection = projection
        self.batch_size = batch_size
        self.replay_slots = replay_slots
        self.seed = seed
        self.num_workers = num_workers
        self.L_ctx = L_ctx
        self.L_chunk = L_chunk
        self.schema_version = schema_version
        self.extra = extra
        self.batch_transform = batch_transform
        self.pin_memory = pin_memory
        self.source_sample_counts = tier.source_replay_counts()
        self._selection_hash = tier.sha256
        self._source_manifest_sha256 = dict(getattr(adapter, "source_manifest_sha256", {}))
        self._cursor: tuple[int, int, int] = (0, 0, 0)
        self._buffer = ReplayBuffer(replay_slots, batch_size, seed ^ 0x51B0FF)
        self._resume_descriptors: tuple[GenerationDescriptor, ...] | None = None
        self._resume_rng_state: dict[str, Any] | None = None
        self._iterator: _O51Iterator | None = None
        self._data_iterator: Iterator[DecodedShard] | None = None
        self._parent_next_active = False
        self._raw_bytes_read = 0
        self._generations_read = 0
        self._max_decoded_shard_bytes = 0
        self._shard_profile_values: dict[str, deque[float]] = {}
        self._shard_profile_events: deque[dict[str, object]] = deque(maxlen=8_192)
        self.reset_shard_profile()

    @property
    def metrics(self) -> dict[str, float]:
        return dict(self._buffer.last_metrics)

    @property
    def buffer_bytes(self) -> int:
        return sum(values.nbytes for values in self._buffer.columns.values())

    @property
    def raw_bytes_read(self) -> int:
        return self._raw_bytes_read

    @property
    def active_replay_ids(self) -> tuple[str, ...]:
        return tuple(self._buffer.identity_slots)

    @property
    def active_identity_count(self) -> int:
        return self._buffer.active_identities

    def count_active_replay_ids(self, replay_ids: Iterable[str]) -> int:
        """Count candidate identities that are live before the next sample."""
        return sum(replay_id in self._buffer.identity_slots for replay_id in replay_ids)

    @property
    def sampled_identity_ranks(self) -> tuple[int, ...]:
        return self._buffer.last_sampled_identity_ranks

    @property
    def sampled_identity_count(self) -> int:
        return self._buffer.last_sampled_identity_count

    @property
    def generations_read(self) -> int:
        return self._generations_read

    @property
    def shard_profile(self) -> dict[str, float | int]:
        """Return bounded per-shard phase distributions since the last reset."""
        metrics: dict[str, float | int] = {"shard_profile_count": len(self._shard_profile_events)}
        for name, samples in self._shard_profile_values.items():
            values = np.asarray(samples, dtype=np.float64)
            prefix = f"shard_{name}_seconds"
            metrics[f"{prefix}_total"] = float(values.sum()) if len(values) else 0.0
            metrics[f"{prefix}_mean"] = float(values.mean()) if len(values) else 0.0
            metrics[f"{prefix}_p50"] = float(np.percentile(values, 50)) if len(values) else 0.0
            metrics[f"{prefix}_p95"] = float(np.percentile(values, 95)) if len(values) else 0.0
            metrics[f"{prefix}_p99"] = float(np.percentile(values, 99)) if len(values) else 0.0
            metrics[f"{prefix}_max"] = float(values.max()) if len(values) else 0.0
        return metrics

    @property
    def slowest_shards(self) -> dict[str, tuple[dict[str, object], ...]]:
        """Return the five slowest recent shard events for each blocking phase."""
        events = tuple(self._shard_profile_events)
        fields = (
            "parent_wait_seconds",
            "worker_seconds",
            "download_seconds",
            "result_delivery_and_order_seconds",
        )
        return {
            field: tuple(sorted(events, key=lambda event: float(event[field]), reverse=True)[:5]) for field in fields
        }

    def reset_shard_profile(self) -> None:
        """Start a new bounded shard-profile interval between parent ``next`` calls."""
        if self._parent_next_active:
            raise RuntimeError("cannot reset shard profiling while parent-side next() is active")
        names = (
            "prepare",
            "prepare_other",
            "download",
            "decompress",
            "read",
            "decode",
            "worker",
            "result_delivery_and_order",
            "parent_wait",
        )
        self._shard_profile_values = {name: deque(maxlen=8_192) for name in names}
        self._shard_profile_events = deque(maxlen=8_192)

    def _record_shard_profile(
        self,
        shard: DecodedShard,
        received_ns: int,
        parent_wait_seconds: float,
    ) -> None:
        timings = shard.timings
        result_delivery_seconds = (
            max(0.0, (received_ns - shard.worker_finished_ns) / 1e9) if shard.worker_finished_ns else 0.0
        )
        values = {
            "prepare": timings.prepare_seconds,
            "prepare_other": max(
                0.0,
                timings.prepare_seconds - timings.download_seconds - timings.decompress_seconds,
            ),
            "download": timings.download_seconds,
            "decompress": timings.decompress_seconds,
            "read": timings.read_seconds,
            "decode": timings.decode_seconds,
            "worker": timings.worker_seconds,
            "result_delivery_and_order": result_delivery_seconds,
            "parent_wait": parent_wait_seconds,
        }
        for name, value in values.items():
            self._shard_profile_values[name].append(value)
        self._shard_profile_events.append(
            {
                "source": shard.task.source,
                "shard": shard.task.shard,
                "epoch": shard.epoch,
                "task_offset": shard.task_offset,
                "rows": len(shard.replay_ids),
                **{f"{name}_seconds": value for name, value in values.items()},
            }
        )

    @property
    def max_decoded_shard_bytes(self) -> int:
        return self._max_decoded_shard_bytes

    @property
    def required_disk_bytes(self) -> int:
        return disk_requirement_bytes(self.tasks, self.adapter.manifests, workers=self.num_workers)

    @property
    def disk_free_bytes(self) -> int:
        first_manifest = next(iter(self.adapter.manifests.values()))
        return shutil.disk_usage(first_manifest.raw_paths[0].parent).free

    def __iter__(self) -> _O51Iterator:
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("create the O51 iterator on the main thread")
        if self._iterator is not None:
            raise RuntimeError("O51ReplayLoader supports one process-lifetime iterator")
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            raise RuntimeError("start O51 loader workers before CUDA initialization")
        dataset = _ShardDataset(
            self.adapter,
            self.tasks,
            seed=self.seed,
            L_ctx=self.L_ctx,
            L_chunk=self.L_chunk,
            schema_version=self.schema_version,
            labels=self.labels,
            projection=self.projection,
        )
        sampler = _ShardSampler(self.tasks, self.seed, self._cursor)
        if self.num_workers:
            torch.multiprocessing.set_sharing_strategy("file_system")
            data_loader = DataLoader(
                dataset,
                batch_size=None,
                sampler=sampler,
                num_workers=self.num_workers,
                collate_fn=cast(Any, _identity),
                pin_memory=False,
                worker_init_fn=_limit_worker_threads,
                generator=torch.Generator().manual_seed(self.seed),
                in_order=True,
                persistent_workers=True,
                prefetch_factor=PREFETCH_FACTOR,
                multiprocessing_context="spawn",
            )
        else:
            data_loader = DataLoader(
                dataset,
                batch_size=None,
                sampler=sampler,
                num_workers=0,
                collate_fn=cast(Any, _identity),
                pin_memory=False,
                worker_init_fn=_limit_worker_threads,
                generator=torch.Generator().manual_seed(self.seed),
                in_order=True,
            )
        self._data_iterator = iter(data_loader)
        self._iterator = _O51Iterator(self, self._data_iterator)
        return self._iterator

    def _advance_task_cursor(self) -> None:
        epoch, task_offset, _ = self._cursor
        task_offset += 1
        if task_offset == len(self.tasks):
            epoch += 1
            task_offset = 0
        self._cursor = (epoch, task_offset, 0)

    def _next_batch(self, iterator: _O51Iterator) -> object:
        if self._resume_descriptors is not None:
            self._restore_buffer(self._resume_descriptors)
            assert self._resume_rng_state is not None
            self._buffer.rng.bit_generator.state = self._resume_rng_state
            self._resume_descriptors = None
            self._resume_rng_state = None
        while self._buffer.size < self.replay_slots:
            shard, row = iterator.take_generation()
            self._buffer.admit(shard, row)
        replay_ids, windows, exhausted = self._buffer.sample()
        batch = collate_train_batch(
            windows,
            stats=self.stats,
            L_ctx=self.L_ctx,
            extra=self.extra,
            projection=self.projection,
        )
        batch = TrainBatch(context=batch.context, target=batch.target, replay_ids=replay_ids)
        transformed = self.batch_transform(windows, batch) if self.batch_transform is not None else batch
        for slot in exhausted:
            self._buffer.remove(slot)
            shard, row = iterator.take_generation()
            self._buffer.admit(shard, row, slot=slot)
        if not self.pin_memory:
            return transformed
        pin_memory = getattr(transformed, "pin_memory", None)
        if not callable(pin_memory):
            raise TypeError("O51 batch transform returned an object without pin_memory()")
        return pin_memory()

    def state_dict(self) -> dict[str, object]:
        if self._iterator is None:
            raise RuntimeError("the O51 loader has not started")
        if self._parent_next_active:
            raise RuntimeError("cannot checkpoint while parent-side next() is active")
        if self._buffer.size != self.replay_slots:
            raise RuntimeError("cannot checkpoint before the replay buffer is full")
        return {
            "schema": CHECKPOINT_SCHEMA,
            "data_protocol": DATA_PROTOCOL,
            "source_selection_sha256": self._selection_hash,
            "source_manifest_sha256": self._source_manifest_sha256,
            "cursor": self._cursor,
            "batch_sampler_rng_state": self._buffer.rng.bit_generator.state,
            "slots": self._buffer.descriptors(),
            "buffer_geometry": {
                "replay_slots": self.replay_slots,
                "windows_per_generation": WINDOWS_PER_GENERATION,
                "batch_size": self.batch_size,
                "window_length": self.L_ctx + self.L_chunk,
            },
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Schedule a strict O51 restore before workers are started."""
        if self._iterator is not None:
            raise RuntimeError("load O51 state before creating its iterator")
        if state.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError(f"unsupported O51 loader schema {state.get('schema')!r}")
        if state.get("data_protocol") != DATA_PROTOCOL:
            raise ValueError("O51 data protocol changed across resume")
        if state.get("source_selection_sha256") != self._selection_hash:
            raise ValueError("O51 source selection changed across resume")
        if state.get("source_manifest_sha256") != self._source_manifest_sha256:
            raise ValueError("O51 source manifests changed across resume")
        expected_geometry = {
            "replay_slots": self.replay_slots,
            "windows_per_generation": WINDOWS_PER_GENERATION,
            "batch_size": self.batch_size,
            "window_length": self.L_ctx + self.L_chunk,
        }
        if state.get("buffer_geometry") != expected_geometry:
            raise ValueError("O51 replay-buffer geometry changed across resume")
        cursor = state.get("cursor")
        if not isinstance(cursor, tuple) or len(cursor) != 3:
            raise ValueError("O51 committed source cursor is invalid")
        typed_cursor = cast(tuple[int, int, int], cursor)
        if any(not isinstance(value, int) or value < 0 for value in typed_cursor) or typed_cursor[1] >= len(
            self.tasks
        ):
            raise ValueError("O51 committed source cursor is invalid")
        descriptors = state.get("slots")
        rng_state = state.get("batch_sampler_rng_state")
        if not isinstance(descriptors, tuple) or len(descriptors) != self.replay_slots:
            raise ValueError("O51 checkpoint does not describe every replay slot")
        if not all(isinstance(descriptor, GenerationDescriptor) for descriptor in descriptors):
            raise ValueError("O51 checkpoint contains an invalid replay descriptor")
        if not isinstance(rng_state, dict):
            raise ValueError("O51 checkpoint has no batch-sampler RNG state")
        epoch, task_offset, row_offset = typed_cursor
        task_index = permute_shard_tasks(self.tasks, seed=self.seed, epoch=epoch)[task_offset]
        if row_offset > self.tasks[task_index].row_count:
            raise ValueError("O51 committed row cursor exceeds its shard")
        typed_descriptors = cast(tuple[GenerationDescriptor, ...], descriptors)
        if sorted(descriptor.slot for descriptor in typed_descriptors) != list(range(self.replay_slots)):
            raise ValueError("O51 checkpoint replay slots are not a complete permutation")
        tasks_by_shard = {(task.source, task.shard): task for task in self.tasks}
        for descriptor in typed_descriptors:
            task = tasks_by_shard.get((descriptor.locator.source, descriptor.locator.shard))
            if (
                descriptor.epoch < 0
                or descriptor.replay_checksum < 0
                or not 0 <= descriptor.next_window < WINDOWS_PER_GENERATION
                or task is None
                or not task.row_start <= descriptor.locator.row < task.row_stop
                or descriptor.locator.row in task.excluded_rows
            ):
                raise ValueError("O51 checkpoint contains an invalid replay descriptor")
        self._cursor = typed_cursor
        self._resume_descriptors = typed_descriptors
        self._resume_rng_state = cast(dict[str, Any], rng_state)

    def _restore_buffer(self, descriptors: tuple[GenerationDescriptor, ...]) -> None:
        tasks = {(task.source, task.shard): task for task in self.tasks}
        cohorts: dict[tuple[str, int], list[GenerationDescriptor]] = {}
        for descriptor in descriptors:
            cohorts.setdefault((descriptor.locator.source, descriptor.locator.shard), []).append(descriptor)
        for key in sorted(cohorts):
            task = tasks.get(key)
            if task is None:
                raise ValueError(f"checkpoint row shard {key} is not in the source plan")
            cohort = cohorts[key]
            requests = [(item.locator.row, item.epoch) for item in cohort]
            generations = self.adapter.decode_generations(
                task,
                requests,
                seed=self.seed,
                L_ctx=self.L_ctx,
                L_chunk=self.L_chunk,
                schema_version=self.schema_version,
                labels=self.labels,
                projection=self.projection,
            )
            ordered = sorted(cohort, key=lambda item: item.slot)
            replay_ids = tuple(generations[(descriptor.locator.row, descriptor.epoch)][0] for descriptor in ordered)
            windows = tuple(generations[(descriptor.locator.row, descriptor.epoch)][1] for descriptor in ordered)
            shard = DecodedShard(
                task=task,
                epoch=-1,
                task_offset=-1,
                replay_ids=replay_ids,
                locators=tuple(descriptor.locator for descriptor in ordered),
                columns=_stack_window_rows(windows),
            )
            for row, descriptor in enumerate(ordered):
                replay_id = replay_ids[row]
                if _replay_checksum(replay_id) != descriptor.replay_checksum:
                    raise ValueError(f"replay identity changed at {descriptor.locator}")
                self._buffer.admit(shard, row, slot=descriptor.slot)
                self._buffer.epochs[descriptor.slot] = descriptor.epoch
                if not 0 <= descriptor.next_window < WINDOWS_PER_GENERATION:
                    raise ValueError("checkpoint next-window ordinal is invalid")
                self._buffer.next_windows[descriptor.slot] = descriptor.next_window

    def close(self) -> None:
        iterator = self._data_iterator
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()


def make_o51_replay_loader(
    *,
    tier: TierSelection,
    stats: dict[str, Any],
    labels: ReplayLabels,
    projection: FeatureProjection | None,
    batch_size: int,
    replay_slots: int = DEFAULT_REPLAY_SLOTS,
    seed: int,
    num_workers: int,
) -> O51ReplayLoader:
    """Build O51's fixed four-window, one-shard-per-worker loader."""
    adapter = MDSStorageAdapter(tier, download_retry=8)
    tasks = build_shard_plan(tier, adapter.manifests)
    return O51ReplayLoader(
        tier=tier,
        adapter=adapter,
        tasks=tasks,
        stats=stats,
        labels=labels,
        projection=projection,
        batch_size=batch_size,
        replay_slots=replay_slots,
        seed=seed,
        num_workers=num_workers,
        L_ctx=O51_L_CTX,
        L_chunk=O51_L_CHUNK,
        schema_version=O51_SCHEMA_VERSION,
        extra=O51_EXTRA_COLUMNS,
        batch_transform=_collate_o51_batch,
        pin_memory=torch.cuda.is_available(),
    )
