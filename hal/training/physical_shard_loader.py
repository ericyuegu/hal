"""Replay loading in deterministic physical-shard order.

Mosaic Streaming owns manifests, shard download, decompression, and MDS row
decoding. This module owns physical-shard order, bounded prefetch, and
balanced replay sampling.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import shutil
import threading
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Final
from typing import Protocol
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
from hal.training.dataloader import make_window
from hal.training.features import FeatureProjection

PREFETCH_FACTOR: Final[int] = 2
CHECKPOINT_SCHEMA: Final[int] = 2
MIN_REPLAY_GAP_BATCHES: Final[int] = 200
SHUFFLE_BLOCK_BATCHES: Final[int] = 32

type Window = dict[str, np.ndarray]
type Generation = tuple[str, tuple[Window, ...]]

type BatchTransform[T] = Callable[[tuple[str, ...], Mapping[str, np.ndarray]], T]


@dataclass(frozen=True, slots=True)
class SourceRowSelection:
    """A selected prefix of one source, less explicit excluded rows."""

    source: str
    stop: int
    excluded_rows: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("source name must not be empty")
        if self.stop < 1:
            raise ValueError(f"source row stop must be positive for {self.source}")
        if self.excluded_rows != tuple(sorted(set(self.excluded_rows))):
            raise ValueError(f"excluded rows are not sorted and unique for {self.source}")
        if any(not 0 <= row < self.stop for row in self.excluded_rows):
            raise ValueError(f"excluded row is outside {self.source}[0:{self.stop}]")

    @property
    def row_count(self) -> int:
        return self.stop - len(self.excluded_rows)


@dataclass(frozen=True, slots=True)
class PhysicalShardSelection:
    """Rows and stable identity for one physical-shard data selection."""

    sources: tuple[SourceRowSelection, ...]
    sha256: str

    def __post_init__(self) -> None:
        source_names = [source.source for source in self.sources]
        if not source_names or len(set(source_names)) != len(source_names):
            raise ValueError("source selections must be non-empty and unique")
        if len(self.sha256) != 64 or any(character not in "0123456789abcdef" for character in self.sha256):
            raise ValueError("selection SHA-256 is invalid")

    @property
    def row_count(self) -> int:
        return sum(source.row_count for source in self.sources)

    def row_counts_by_source(self) -> dict[str, int]:
        return {source.source: source.row_count for source in self.sources}


@dataclass(frozen=True, slots=True)
class SourceManifest:
    """The physical geometry of one source manifest."""

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
    """Conservative host-memory components for shard prefetch."""

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
class DecodedShard:
    """Fixed-window rows returned by one worker."""

    task: ShardTask
    epoch: int
    task_offset: int
    replay_ids: tuple[str, ...]
    locators: tuple[PhysicalRow, ...]
    columns: Mapping[str, np.ndarray]
    windows_per_generation: int
    raw_bytes_read: int = 0

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
        if self.windows_per_generation < 1:
            raise ValueError("windows per generation must be positive")
        expected_shape = (rows, self.windows_per_generation)
        bad = {
            name: value.shape
            for name, value in self.columns.items()
            if not isinstance(value, np.ndarray) or value.ndim < 2 or value.shape[:2] != expected_shape
        }
        if bad:
            raise ValueError(f"decoded shard columns must begin [R, {self.windows_per_generation}], got {bad}")

    def windows(self, row: int) -> tuple[dict[str, np.ndarray], ...]:
        if not 0 <= row < len(self.replay_ids):
            raise IndexError(row)
        return tuple(
            {name: values[row, window] for name, values in self.columns.items()}
            for window in range(self.windows_per_generation)
        )


def _coerce_manifest(source: str, value: SourceManifest | Sequence[int]) -> SourceManifest:
    if isinstance(value, SourceManifest):
        if value.source != source:
            raise ValueError(f"manifest key {source!r} does not match manifest source {value.source!r}")
        return value
    return SourceManifest(source, tuple(int(samples) for samples in value))


def build_shard_plan(
    selection: PhysicalShardSelection,
    manifests: Mapping[str, SourceManifest | Sequence[int]],
) -> tuple[ShardTask, ...]:
    """Expose each selected prefix row exactly once, grouped by physical shard."""
    selected_sources = {view.source for view in selection.sources}
    if set(manifests) != selected_sources:
        missing = selected_sources - set(manifests)
        extra = set(manifests) - selected_sources
        raise ValueError(f"manifest sources do not match selection: missing={sorted(missing)}, extra={sorted(extra)}")

    tasks: list[ShardTask] = []
    global_shard = 0
    for source_index, view in enumerate(selection.sources):
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
            raise ValueError(f"selection needs {view.stop} rows from {view.source}, but its manifest has {source_row}")
        if selected != view.row_count:
            raise RuntimeError(f"shard plan selected {selected} rows from {view.source}, expected {view.row_count}")
    if sum(task.row_count for task in tasks) != selection.row_count:
        raise RuntimeError("shard plan row total does not match the selection")
    return tuple(tasks)


def _task_key(task: ShardTask, seed: int) -> bytes:
    value = f"{seed}\0{task.source}\0{task.shard}".encode()
    return hashlib.blake2b(value, digest_size=16, person=b"hal-o51-shards").digest()


def permute_shard_tasks(tasks: Sequence[ShardTask], *, seed: int, epoch: int) -> tuple[int, ...]:
    """Return one keyed shard order reused across source epochs.

    A stable order prevents a physical row from wrapping early across an epoch
    boundary. The replay schedule supplies the changing batch composition.
    """
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    return tuple(
        sorted(
            range(len(tasks)),
            key=lambda index: (_task_key(tasks[index], seed), tasks[index].source, tasks[index].shard),
        )
    )


def choose_generation_window_starts(
    frames: int,
    context_length: int,
    chunk_length: int,
    windows_per_generation: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose distinct window starts uniformly without replacement."""
    if windows_per_generation < 1:
        raise ValueError("windows per generation must be positive")
    length = context_length + chunk_length
    if length < 1:
        raise ValueError("window length must be positive")
    starts = frames - length + 1
    if starts < windows_per_generation:
        raise ValueError(
            f"{frames} frames provide {starts} starts; "
            f"{windows_per_generation} windows require at least {windows_per_generation}"
        )
    return rng.choice(starts, size=windows_per_generation, replace=False).astype(np.int64, copy=False)


def _replay_checksum(replay_id: str) -> int:
    return int.from_bytes(hashlib.blake2b(replay_id.encode(), digest_size=8).digest(), "little")


def _generation_rng(seed: int, epoch: int, replay_id: str) -> np.random.Generator:
    checksum = _replay_checksum(replay_id)
    return np.random.default_rng((seed, epoch, checksum & 0xFFFFFFFF, checksum >> 32))


def _stack_window_rows(rows: Sequence[tuple[Window, ...]], *, windows_per_generation: int) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("cannot stack an empty shard")
    names = tuple(sorted(rows[0][0]))
    expected_names = set(names)
    arrays: dict[str, np.ndarray] = {}
    for row in rows:
        if len(row) != windows_per_generation or any(set(window) != expected_names for window in row):
            raise ValueError("decoded windows do not have fixed keys")
    for name in names:
        reference = rows[0][0][name]
        shape = reference.shape
        dtype = reference.dtype
        out = np.empty((len(rows), windows_per_generation, *shape), dtype=dtype)
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
    context_length: int,
    chunk_length: int,
    windows_per_generation: int,
    schema_version: int,
    labels: ReplayLabels,
    projection: FeatureProjection | None,
) -> Generation:
    replay_id = str(compact["replay_id"])
    frames = int(cast(Any, compact["num_frames"]))
    window_length = context_length + chunk_length
    required_frames = window_length + windows_per_generation - 1
    if frames < required_frames:
        raise ValueError(
            f"short replay {replay_id!r} at source={task.source} shard={task.shard} row={row}: "
            f"frame_count={frames}, required_count={required_frames}"
        )
    source_schema_version = int(cast(Any, compact["source_schema_version"]))
    check_schema_version({"schema_version": source_schema_version}, expected=schema_version)
    rng = _generation_rng(seed, epoch, replay_id)
    starts = choose_generation_window_starts(
        frames,
        context_length,
        chunk_length,
        windows_per_generation,
        rng,
    )
    ranges = tuple((int(start), int(start) + window_length) for start in starts)
    decoded = decode_policy_world_replay_slices(compact, ranges)
    replay_labels = {name: np.asarray(value) for name, value in labels(compact).items()}
    wrong_labels = {name: value.shape for name, value in replay_labels.items() if value.shape not in ((), (frames,))}
    if wrong_labels:
        raise ValueError(f"replay labels have invalid shapes {wrong_labels}; expected scalar or {(frames,)}")

    windows: list[Window] = []
    for start, decoded_slice in zip(starts, decoded, strict=True):
        start_int = int(start)
        sample = dict(decoded_slice)
        sample.update(
            {
                name: (
                    np.full(window_length, value.item(), dtype=value.dtype)
                    if value.shape == ()
                    else value[start_int : start_int + window_length]
                )
                for name, value in replay_labels.items()
            }
        )
        ego = "p1" if rng.random() < 0.5 else "p2"
        window = make_window(
            sample,
            ego_prefix=ego,
            start=0,
            pad=0,
            length=window_length,
            projection=projection,
        )
        window["ctx_pad"] = np.asarray(0, dtype=np.int64)
        windows.append(
            {
                name: np.asarray(value) if np.ndim(value) == 0 else np.ascontiguousarray(value)
                for name, value in window.items()
            }
        )
    return replay_id, tuple(windows)


class MDSStorageAdapter:
    """Narrow Mosaic Streaming 0.13 adapter for physical-shard reads."""

    def __init__(
        self,
        selection: PhysicalShardSelection,
        *,
        split: str = "train",
        download_retry: int = 8,
    ) -> None:
        installed = importlib.metadata.version("mosaicml-streaming")
        if installed != "0.13.0":
            raise RuntimeError(f"physical shard loading requires mosaicml-streaming==0.13.0, found {installed}")
        sources = tuple(streams_lib.BY_NAME[view.source] for view in selection.sources)
        mosaic_streams = [
            Stream(
                remote=source.remote,
                local=str(source.local_root),
                split=split,
                choose=view.row_count,
                download_retry=download_retry,
                keep_zip=False,
            )
            for source, view in zip(sources, selection.sources, strict=True)
        ]
        self.dataset = StreamingDataset(
            streams=mosaic_streams,
            shuffle=False,
            predownload=None,
            cache_limit=None,
            keep_zip=False,
            batch_size=1,
        )
        self.selection = selection
        self.split = split
        self.manifests = self._manifests()
        self.last_read_bytes = 0

    def _manifests(self) -> dict[str, SourceManifest]:
        manifests: dict[str, SourceManifest] = {}
        for source_index, view in enumerate(self.selection.sources):
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
            raise TypeError(f"physical shard loading requires MDS shards, found {type(reader).__name__}")
        return reader

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
        context_length: int,
        chunk_length: int,
        windows_per_generation: int,
        schema_version: int,
        labels: ReplayLabels,
        projection: FeatureProjection | None,
    ) -> dict[tuple[int, int], Generation]:
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
                context_length=context_length,
                chunk_length=chunk_length,
                windows_per_generation=windows_per_generation,
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
        context_length: int,
        chunk_length: int,
        windows_per_generation: int,
        schema_version: int,
        labels: ReplayLabels,
        projection: FeatureProjection | None,
    ) -> DecodedShard:
        rows = task.selected_rows
        self.dataset.prepare_shard(task.global_shard)
        reader = self._reader(task)
        filename = Path(reader.dirname) / (reader.split or "") / reader.raw_data.basename
        replay_ids: list[str] = []
        columns: dict[str, np.ndarray] = {}
        raw_bytes_read = 0
        with filename.open("rb", buffering=0) as handle:
            table = handle.read(4 * (reader.samples + 2))
            offsets = np.frombuffer(table, dtype=np.uint32)
            if len(offsets) != reader.samples + 2 or int(offsets[0]) != reader.samples:
                raise ValueError(f"invalid MDS offset table in {filename}")
            raw_bytes_read += len(table)
            for output_row, row in enumerate(rows):
                begin, end = int(offsets[row + 1]), int(offsets[row + 2])
                handle.seek(begin)
                payload = handle.read(end - begin)
                if len(payload) != end - begin:
                    raise EOFError(f"short MDS row read from {filename} at row {row}")
                raw_bytes_read += len(payload)
                replay_id, windows = _decode_generation(
                    reader.decode_sample(payload),
                    task=task,
                    row=row,
                    epoch=epoch,
                    seed=seed,
                    context_length=context_length,
                    chunk_length=chunk_length,
                    windows_per_generation=windows_per_generation,
                    schema_version=schema_version,
                    labels=labels,
                    projection=projection,
                )
                replay_ids.append(replay_id)
                if not columns:
                    columns = {
                        name: np.empty((len(rows), windows_per_generation, *value.shape), dtype=value.dtype)
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
        return DecodedShard(
            task=task,
            epoch=epoch,
            task_offset=task_offset,
            replay_ids=tuple(replay_ids),
            locators=tuple(PhysicalRow(task.source, task.shard, row) for row in rows),
            columns=columns,
            windows_per_generation=windows_per_generation,
            raw_bytes_read=raw_bytes_read,
        )


def disk_requirement_bytes(
    tasks: Sequence[ShardTask],
    manifests: Mapping[str, SourceManifest],
    *,
    workers: int,
    reserved_bytes: int,
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


class ShardStorageAdapter(Protocol):
    """Storage boundary used by worker processes and checkpoint restore."""

    @property
    def manifests(self) -> Mapping[str, SourceManifest]: ...

    def decode_task(
        self,
        task: ShardTask,
        epoch: int,
        task_offset: int,
        *,
        seed: int,
        context_length: int,
        chunk_length: int,
        windows_per_generation: int,
        schema_version: int,
        labels: ReplayLabels,
        projection: FeatureProjection | None,
    ) -> DecodedShard: ...

    def decode_generations(
        self,
        task: ShardTask,
        requests: Sequence[tuple[int, int]],
        *,
        seed: int,
        context_length: int,
        chunk_length: int,
        windows_per_generation: int,
        schema_version: int,
        labels: ReplayLabels,
        projection: FeatureProjection | None,
    ) -> Mapping[tuple[int, int], Generation]: ...


class _ShardDataset(Dataset[DecodedShard]):
    def __init__(
        self,
        adapter: ShardStorageAdapter,
        tasks: tuple[ShardTask, ...],
        *,
        seed: int,
        context_length: int,
        chunk_length: int,
        windows_per_generation: int,
        schema_version: int,
        labels: ReplayLabels,
        projection: FeatureProjection | None,
    ) -> None:
        self.adapter = adapter
        self.tasks = tasks
        self.seed = seed
        self.context_length = context_length
        self.chunk_length = chunk_length
        self.windows_per_generation = windows_per_generation
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
            context_length=self.context_length,
            chunk_length=self.chunk_length,
            windows_per_generation=self.windows_per_generation,
            schema_version=self.schema_version,
            labels=self.labels,
            projection=self.projection,
        )


class _ShardSampler(Sampler[ShardRequest]):
    def __init__(self, tasks: tuple[ShardTask, ...], seed: int, cursor: tuple[int, int, int]) -> None:
        self.order = permute_shard_tasks(tasks, seed=seed, epoch=0)
        self.cursor = cursor

    def __iter__(self) -> Iterator[ShardRequest]:
        epoch, task_offset, _ = self.cursor
        while True:
            for offset in range(task_offset, len(self.order)):
                yield ShardRequest(epoch, offset, self.order[offset])
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


@dataclass(frozen=True, slots=True)
class _DecodedRows:
    shard: DecodedShard
    start: int
    stop: int

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.stop <= len(self.shard.replay_ids):
            raise ValueError("decoded row range is invalid")

    @property
    def count(self) -> int:
        return self.stop - self.start


class _BalancedReplaySchedule:
    """Visit every slot once per pass while keeping consecutive visits far apart."""

    def __init__(self, capacity: int, batch_size: int, phases: int, seed: int) -> None:
        if capacity % batch_size or capacity % phases or batch_size % phases:
            raise ValueError("replay geometry must divide evenly into batches and turnover phases")
        self.capacity = capacity
        self.batch_size = batch_size
        self.phases = phases
        self.pass_batches = capacity // batch_size
        self.phase_batch_size = batch_size // phases
        # A slot moves by at most one block minus one between passes.
        self.shuffle_block_batches = min(
            SHUFFLE_BLOCK_BATCHES,
            max(1, self.pass_batches - MIN_REPLAY_GAP_BATCHES + 1),
        )
        self.minimum_gap_batches = self.pass_batches - self.shuffle_block_batches + 1
        self.rng = np.random.default_rng(seed)
        self.decks = np.empty(
            (phases, self.pass_batches, self.phase_batch_size),
            dtype=np.int32,
        )
        for phase in range(phases):
            slots = np.arange(phase, capacity, phases, dtype=np.int32)
            self.rng.shuffle(slots)
            self.decks[phase] = slots.reshape(self.pass_batches, self.phase_batch_size)
        self.row = 0
        self.pass_index = 0

    def _shuffle_pass(self) -> None:
        block = self.shuffle_block_batches
        offset = int(self.rng.integers(block)) if block > 1 else 0
        for deck in self.decks:
            if offset:
                self.rng.shuffle(deck[:offset].reshape(-1))
            for start in range(offset, self.pass_batches, block):
                self.rng.shuffle(deck[start : start + block].reshape(-1))
        self.row = 0
        self.pass_index += 1

    def next(self) -> np.ndarray:
        if self.row == self.pass_batches:
            self._shuffle_pass()
        slots = np.ascontiguousarray(self.decks[:, self.row].reshape(-1))
        self.row += 1
        return slots

    def state_dict(self) -> dict[str, object]:
        return {
            "decks": self.decks.copy(),
            "pass_index": self.pass_index,
            "row": self.row,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if set(state) != {"decks", "pass_index", "row", "rng_state"}:
            raise ValueError("batch schedule state has invalid keys")
        decks = state["decks"]
        pass_index = state["pass_index"]
        row = state["row"]
        rng_state = state["rng_state"]
        if not isinstance(decks, np.ndarray) or decks.shape != self.decks.shape or decks.dtype != np.int32:
            raise ValueError("batch schedule decks have invalid geometry")
        if not isinstance(pass_index, int) or pass_index < 0:
            raise ValueError("batch schedule pass is invalid")
        if not isinstance(row, int) or not 0 <= row <= self.pass_batches:
            raise ValueError("batch schedule row is invalid")
        if not isinstance(rng_state, dict):
            raise ValueError("batch schedule RNG state is invalid")
        if not np.array_equal(np.sort(cast(Any, decks).reshape(-1)), np.arange(self.capacity)):
            raise ValueError("batch schedule decks are not a slot permutation")
        for phase, deck in enumerate(decks):
            if np.any(deck % self.phases != phase):
                raise ValueError("batch schedule changed a slot's turnover phase")
        self.decks[:] = decks
        self.pass_index = pass_index
        self.row = row
        self.rng.bit_generator.state = cast(dict[str, Any], rng_state)


@dataclass(frozen=True, slots=True)
class GenerationDescriptor:
    slot: int
    locator: PhysicalRow
    epoch: int
    replay_checksum: int
    next_window: int


class ReplayBuffer:
    """Columnar replay generations consumed by a balanced slot schedule."""

    def __init__(self, capacity: int, batch_size: int, windows_per_generation: int, seed: int) -> None:
        if capacity < batch_size:
            raise ValueError("replay capacity must cover one identity-distinct batch")
        if windows_per_generation < 1:
            raise ValueError("windows per generation must be positive")
        self.capacity = capacity
        self.batch_size = batch_size
        self.windows_per_generation = windows_per_generation
        self.schedule = _BalancedReplaySchedule(capacity, batch_size, windows_per_generation, seed)
        self.columns: dict[str, np.ndarray] = {}
        self.replay_ids: list[str | None] = [None] * capacity
        self.locators: list[PhysicalRow | None] = [None] * capacity
        self.epochs = np.full(capacity, -1, dtype=np.int64)
        self.next_windows = np.zeros(capacity, dtype=np.uint16)
        self.identity_slots: dict[str, int] = {}
        self.last_selected_batches = np.full(capacity, -1, dtype=np.int64)
        self._size = 0
        self.batch_index = 0
        self.last_metrics: dict[str, float] = {}
        self.last_sampled_identity_ranks: tuple[int, ...] = ()
        self.last_sampled_identity_count = 0

    @property
    def size(self) -> int:
        return self._size

    @property
    def active_identities(self) -> int:
        return len(self.identity_slots)

    def _initialize_columns(self, shard: DecodedShard) -> None:
        if self.columns:
            return
        for name, values in shard.columns.items():
            self.columns[name] = np.empty((self.capacity, *values.shape[1:]), dtype=values.dtype)

    def _admit(
        self,
        shard: DecodedShard,
        selector: slice | np.ndarray,
        replay_ids: tuple[str, ...],
        locators: tuple[PhysicalRow, ...],
        slots: np.ndarray,
        *,
        stagger: bool,
    ) -> None:
        if shard.windows_per_generation != self.windows_per_generation:
            raise ValueError("decoded shard does not match replay-buffer window geometry")
        self._initialize_columns(shard)
        if set(self.columns) != set(shard.columns):
            raise ValueError("decoded shard columns changed after buffer allocation")
        slots = np.asarray(slots, dtype=np.int64)
        if slots.shape != (len(replay_ids),) or len(np.unique(slots)) != len(replay_ids):
            raise ValueError("destination slots do not match decoded rows")
        if np.any(slots < 0) or np.any(slots >= self.capacity):
            raise ValueError("destination slot is outside the replay buffer")
        if any(self.replay_ids[int(slot)] is not None for slot in slots):
            raise RuntimeError("destination replay slot is already occupied")
        if len(set(replay_ids)) != len(replay_ids):
            raise ValueError("decoded rows repeat a replay identity")
        duplicate = next((replay_id for replay_id in replay_ids if replay_id in self.identity_slots), None)
        if duplicate is not None:
            raise ValueError(f"replay identity {duplicate!r} is already active")
        for name, destination in self.columns.items():
            source = shard.columns[name][selector]
            if source.shape[1:] != destination.shape[1:] or source.dtype != destination.dtype:
                raise ValueError(f"decoded column {name!r} does not match the central buffer")
            destination[slots] = source
        for slot_value, replay_id, locator in zip(slots, replay_ids, locators, strict=True):
            slot = int(slot_value)
            self.replay_ids[slot] = replay_id
            self.locators[slot] = locator
            self.identity_slots[replay_id] = slot
        self.epochs[slots] = shard.epoch
        # Stagger only the initial fill. One phase then expires in every batch;
        # all replacement generations consume every window from ordinal zero.
        self.next_windows[slots] = slots % self.windows_per_generation if stagger else 0
        self.last_selected_batches[slots] = -1
        self._size += len(replay_ids)

    def admit_rows(self, rows: _DecodedRows, slots: np.ndarray, *, stagger: bool) -> None:
        """Copy a contiguous decoded row range into the selected empty slots."""
        selector = slice(rows.start, rows.stop)
        self._admit(
            rows.shard,
            selector,
            rows.shard.replay_ids[selector],
            rows.shard.locators[selector],
            slots,
            stagger=stagger,
        )

    def replacement_indices(self, rows: _DecodedRows) -> tuple[np.ndarray, int]:
        """Select rows whose earlier-epoch generation is no longer active."""
        accepted: list[int] = []
        seen: set[str] = set()
        skipped = 0
        for row in range(rows.start, rows.stop):
            replay_id = rows.shard.replay_ids[row]
            if replay_id in seen:
                raise ValueError("decoded rows repeat a replay identity")
            seen.add(replay_id)
            active_slot = self.identity_slots.get(replay_id)
            if active_slot is None:
                accepted.append(row)
                continue
            active_epoch = int(self.epochs[active_slot])
            if rows.shard.epoch <= active_epoch:
                raise ValueError(f"replay identity {replay_id!r} repeats within a source epoch")
            skipped += 1
        return np.asarray(accepted, dtype=np.int64), skipped

    def admit_replacement_indices(
        self,
        shard: DecodedShard,
        row_indices: np.ndarray,
        slots: np.ndarray,
    ) -> None:
        row_indices = np.asarray(row_indices, dtype=np.int64)
        if row_indices.ndim != 1 or not len(row_indices):
            raise ValueError("replacement row selection must not be empty")
        if len(np.unique(row_indices)) != len(row_indices):
            raise ValueError("replacement row indices must be unique")
        if np.any(row_indices < 0) or np.any(row_indices >= len(shard.replay_ids)):
            raise ValueError("replacement row index is outside its shard")
        if np.array_equal(row_indices, np.arange(row_indices[0], row_indices[-1] + 1)):
            start = int(row_indices[0])
            stop = int(row_indices[-1]) + 1
            selector: slice | np.ndarray = slice(start, stop)
            replay_ids = shard.replay_ids[start:stop]
            locators = shard.locators[start:stop]
        else:
            selector = row_indices
            replay_ids = tuple(shard.replay_ids[int(row)] for row in row_indices)
            locators = tuple(shard.locators[int(row)] for row in row_indices)
        self._admit(shard, selector, replay_ids, locators, slots, stagger=False)

    def record_duplicate_generations(self, count: int) -> None:
        if count < 0:
            raise ValueError("duplicate generation count must be non-negative")
        self.last_metrics["data/duplicate_generations"] += float(count)

    def append_rows(self, rows: _DecodedRows) -> None:
        stop = self.size + rows.count
        if stop > self.capacity:
            raise RuntimeError("decoded rows exceed replay-buffer capacity")
        self.admit_rows(rows, np.arange(self.size, stop), stagger=True)

    def retire(self, slots: np.ndarray) -> None:
        slots = np.asarray(slots, dtype=np.int64)
        for slot_value in slots:
            slot = int(slot_value)
            replay_id = self.replay_ids[slot]
            if replay_id is None:
                raise RuntimeError(f"replay slot {slot} is empty")
            del self.identity_slots[replay_id]
            self.replay_ids[slot] = None
            self.locators[slot] = None
        self.epochs[slots] = -1
        self.next_windows[slots] = 0
        self.last_selected_batches[slots] = -1
        self._size -= len(slots)

    def sample(self) -> tuple[tuple[str, ...], dict[str, np.ndarray], np.ndarray]:
        """Consume one window from every slot in the next balanced batch."""
        if self.size != self.capacity or self.active_identities != self.capacity:
            raise RuntimeError("replay buffer must be full of unique identities before sampling")
        self.last_sampled_identity_count = self.active_identities
        slots = self.schedule.next()
        self.last_sampled_identity_ranks = tuple(int(slot) for slot in slots)
        replay_ids = tuple(cast(str, self.replay_ids[int(slot)]) for slot in slots)
        if len(set(replay_ids)) != self.batch_size:
            raise RuntimeError("balanced batch contains a repeated replay identity")
        ordinals = self.next_windows[slots].astype(np.int64)
        columns = {name: values[slots, ordinals] for name, values in self.columns.items()}
        self.next_windows[slots] += 1
        exhausted = slots[self.next_windows[slots] == self.windows_per_generation]
        previous = self.last_selected_batches[slots]
        ages = self.batch_index - previous[previous >= 0]
        if len(ages) and int(ages.min()) < self.schedule.minimum_gap_batches:
            raise RuntimeError("balanced replay schedule violated its minimum reuse gap")
        self.last_selected_batches[slots] = self.batch_index
        self.batch_index += 1
        age_values = ages.astype(np.float64, copy=False)
        self.last_metrics = {
            "data/replay_age_le_1_fraction": float(np.mean(age_values <= 1)) if len(ages) else 0.0,
            "data/replay_age_le_16_fraction": float(np.mean(age_values <= 16)) if len(ages) else 0.0,
            "data/active_replay_identities": float(self.active_identities),
            "data/duplicate_generations": 0.0,
        }
        if len(ages):
            self.last_metrics.update(
                {
                    "data/replay_age_p01": float(np.percentile(age_values, 1)),
                    "data/replay_age_p05": float(np.percentile(age_values, 5)),
                    "data/replay_age_p50": float(np.percentile(age_values, 50)),
                    "data/replay_age_p95": float(np.percentile(age_values, 95)),
                }
            )
        return replay_ids, columns, exhausted

    def sampler_state_dict(self) -> dict[str, object]:
        return {"batch_index": self.batch_index, "schedule": self.schedule.state_dict()}

    def load_sampler_state_dict(self, state: Mapping[str, object]) -> None:
        if set(state) != {"batch_index", "schedule"}:
            raise ValueError("batch sampler state has invalid keys")
        batch_index = state["batch_index"]
        schedule = state["schedule"]
        if not isinstance(batch_index, int) or batch_index < 0 or not isinstance(schedule, Mapping):
            raise ValueError("batch sampler state is invalid")
        self.schedule.load_state_dict(cast(Mapping[str, object], schedule))
        scheduled_batches = self.schedule.pass_index * self.schedule.pass_batches + self.schedule.row
        if batch_index != scheduled_batches:
            raise ValueError("batch sampler state has an inconsistent cursor")
        self.batch_index = batch_index

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


class _PhysicalShardIterator[BatchT](Iterator[BatchT]):
    def __init__(self, loader: PhysicalShardReplayLoader[BatchT], shards: Iterator[DecodedShard]) -> None:
        self.loader = loader
        self.shards = shards
        self.current: DecodedShard | None = None
        self._active = False

    def __iter__(self) -> _PhysicalShardIterator[BatchT]:
        return self

    def __next__(self) -> BatchT:
        if self._active:
            raise RuntimeError("concurrent next() calls are not supported")
        self._active = True
        self.loader._parent_next_active = True
        try:
            return self.loader._next_batch(self)
        finally:
            self.loader._parent_next_active = False
            self._active = False

    def take_rows(self, limit: int) -> _DecodedRows:
        if limit < 1:
            raise ValueError("decoded row limit must be positive")
        while self.current is None or self.loader._cursor[2] >= len(self.current.replay_ids):
            if self.current is not None:
                self.loader._advance_task_cursor()
            decoded = next(self.shards)
            self.loader._raw_bytes_read += decoded.raw_bytes_read
            decoded_bytes = sum(values.nbytes for values in decoded.columns.values())
            self.loader._max_decoded_shard_bytes = max(self.loader._max_decoded_shard_bytes, decoded_bytes)
            epoch, task_offset, row_offset = self.loader._cursor
            if (decoded.epoch, decoded.task_offset) != (epoch, task_offset):
                raise RuntimeError(
                    "DataLoader released a shard out of committed order: "
                    f"got {(decoded.epoch, decoded.task_offset)}, expected {(epoch, task_offset)}"
                )
            self.current = decoded
            if row_offset > len(decoded.replay_ids):
                raise RuntimeError("committed row cursor exceeds its shard")
        start = self.loader._cursor[2]
        stop = min(start + limit, len(self.current.replay_ids))
        self.loader._cursor = (self.loader._cursor[0], self.loader._cursor[1], stop)
        self.loader._generations_read += stop - start
        return _DecodedRows(self.current, start, stop)


def _shutdown_data_loader_workers(iterator: object | None) -> None:
    """Use PyTorch's private worker shutdown until it exposes a public close."""
    if iterator is None:
        return
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


class PhysicalShardReplayLoader[BatchT]:
    """Infinite physical-shard loader with decoded-array-free checkpoints."""

    separate_identity_checkpoint: Final[bool] = True

    def __init__(
        self,
        *,
        selection: PhysicalShardSelection,
        adapter: ShardStorageAdapter,
        tasks: tuple[ShardTask, ...],
        data_protocol: str,
        source_manifest_sha256: Mapping[str, str],
        labels: ReplayLabels,
        projection: FeatureProjection | None,
        batch_transform: BatchTransform[BatchT],
        batch_size: int,
        replay_slots: int,
        seed: int,
        num_workers: int,
        context_length: int,
        chunk_length: int,
        windows_per_generation: int,
        schema_version: int,
        reserved_disk_bytes: int,
        pin_memory: bool,
    ) -> None:
        if replay_slots > selection.row_count:
            replay_slots = selection.row_count
        if replay_slots < batch_size:
            raise ValueError("selection is too small for one identity-distinct batch")
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if context_length < 1 or chunk_length < 0 or windows_per_generation < 1:
            raise ValueError("window geometry is invalid")
        if reserved_disk_bytes < 0:
            raise ValueError("reserved disk bytes must be non-negative")
        if not data_protocol:
            raise ValueError("data protocol must not be empty")
        selected_sources = {source.source for source in selection.sources}
        if set(source_manifest_sha256) != selected_sources:
            raise ValueError("source manifest hashes do not cover the selection")
        if any(
            len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
            for digest in source_manifest_sha256.values()
        ):
            raise ValueError("a source manifest SHA-256 is invalid")
        self.selection = selection
        self.adapter = adapter
        self.tasks = tasks
        self.data_protocol = data_protocol
        self.source_manifest_sha256 = dict(source_manifest_sha256)
        self.labels = labels
        self.projection = projection
        self.batch_transform = batch_transform
        self.batch_size = batch_size
        self.replay_slots = replay_slots
        self.seed = seed
        self.num_workers = num_workers
        self.context_length = context_length
        self.chunk_length = chunk_length
        self.windows_per_generation = windows_per_generation
        self.schema_version = schema_version
        self.reserved_disk_bytes = reserved_disk_bytes
        self.pin_memory = pin_memory
        self.source_sample_counts = selection.row_counts_by_source()
        self._selection_hash = selection.sha256
        self._cursor: tuple[int, int, int] = (0, 0, 0)
        self._buffer = ReplayBuffer(replay_slots, batch_size, windows_per_generation, seed ^ 0x51B0FF)
        # Keep one shuffle block outside the buffer before an identity can return.
        minimum_source_rows = replay_slots + batch_size * (self._buffer.schedule.shuffle_block_batches - 1)
        if selection.row_count <= minimum_source_rows:
            raise ValueError(
                "source selection is too small to keep retired replay identities out for one shuffle block"
            )
        self._resume_descriptors: tuple[GenerationDescriptor, ...] | None = None
        self._resume_sampler_state: Mapping[str, object] | None = None
        self._iterator: _PhysicalShardIterator[BatchT] | None = None
        self._data_iterator: Iterator[DecodedShard] | None = None
        self._parent_next_active = False
        self._raw_bytes_read = 0
        self._generations_read = 0
        self._max_decoded_shard_bytes = 0
        self._closed = False

    @property
    def metrics(self) -> dict[str, float]:
        metrics = dict(self._buffer.last_metrics)
        metrics["data/generations_read"] = float(self._generations_read)
        return metrics

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

    @property
    def minimum_replay_gap_batches(self) -> int:
        return self._buffer.schedule.minimum_gap_batches

    @property
    def schedule_pass_batches(self) -> int:
        return self._buffer.schedule.pass_batches

    @property
    def missing_raw_shards(self) -> int:
        missing = 0
        for task in self.tasks:
            manifest = self.adapter.manifests[task.source]
            path = manifest.raw_paths[task.shard]
            expected_bytes = manifest.raw_bytes_per_shard[task.shard]
            missing += not path.is_file() or path.stat().st_size != expected_bytes
        return missing

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
    def max_decoded_shard_bytes(self) -> int:
        return self._max_decoded_shard_bytes

    @property
    def required_disk_bytes(self) -> int:
        return disk_requirement_bytes(
            self.tasks,
            self.adapter.manifests,
            workers=self.num_workers,
            reserved_bytes=self.reserved_disk_bytes,
        )

    @property
    def disk_free_bytes(self) -> int:
        first_manifest = next(iter(self.adapter.manifests.values()))
        return shutil.disk_usage(first_manifest.raw_paths[0].parent).free

    def __iter__(self) -> _PhysicalShardIterator[BatchT]:
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("create the physical-shard iterator on the main thread")
        if self._closed:
            raise RuntimeError("physical-shard loader is closed")
        if self._iterator is not None:
            raise RuntimeError("PhysicalShardReplayLoader supports one process-lifetime iterator")
        dataset = _ShardDataset(
            self.adapter,
            self.tasks,
            seed=self.seed,
            context_length=self.context_length,
            chunk_length=self.chunk_length,
            windows_per_generation=self.windows_per_generation,
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
        self._iterator = _PhysicalShardIterator(self, self._data_iterator)
        return self._iterator

    def _advance_task_cursor(self) -> None:
        epoch, task_offset, _ = self._cursor
        task_offset += 1
        if task_offset == len(self.tasks):
            epoch += 1
            task_offset = 0
        self._cursor = (epoch, task_offset, 0)

    def _next_batch(self, iterator: _PhysicalShardIterator[BatchT]) -> BatchT:
        if self._closed:
            raise RuntimeError("physical-shard loader is closed")
        if self._resume_descriptors is not None:
            self._restore_buffer(self._resume_descriptors)
            assert self._resume_sampler_state is not None
            self._buffer.load_sampler_state_dict(self._resume_sampler_state)
            self._resume_descriptors = None
            self._resume_sampler_state = None
        while self._buffer.size < self.replay_slots:
            rows = iterator.take_rows(self.replay_slots - self._buffer.size)
            self._buffer.append_rows(rows)
        replay_ids, columns, exhausted = self._buffer.sample()
        transformed = self.batch_transform(replay_ids, columns)
        self._buffer.retire(exhausted)
        replaced = 0
        scanned_without_admission = 0
        while replaced < len(exhausted):
            rows = iterator.take_rows(len(exhausted) - replaced)
            row_indices, skipped = self._buffer.replacement_indices(rows)
            self._buffer.record_duplicate_generations(skipped)
            if not len(row_indices):
                scanned_without_admission += rows.count
                if scanned_without_admission >= self.selection.row_count:
                    raise RuntimeError("source selection has no inactive replay identity for replacement")
                continue
            slots = exhausted[replaced : replaced + len(row_indices)]
            self._buffer.admit_replacement_indices(rows.shard, row_indices, slots)
            replaced += len(row_indices)
            scanned_without_admission = 0
        if not self.pin_memory:
            return transformed
        pin_memory = getattr(transformed, "pin_memory", None)
        if not callable(pin_memory):
            raise TypeError("batch transform returned an object without pin_memory()")
        return cast(BatchT, pin_memory())

    def state_dict(self) -> dict[str, object]:
        if self._iterator is None:
            raise RuntimeError("the physical-shard loader has not started")
        if self._parent_next_active:
            raise RuntimeError("cannot checkpoint while parent-side next() is active")
        if self._buffer.size != self.replay_slots:
            raise RuntimeError("cannot checkpoint before the replay buffer is full")
        return {
            "schema": CHECKPOINT_SCHEMA,
            "data_protocol": self.data_protocol,
            "source_selection_sha256": self._selection_hash,
            "source_manifest_sha256": self.source_manifest_sha256,
            "cursor": self._cursor,
            "batch_sampler_state": self._buffer.sampler_state_dict(),
            "slots": self._buffer.descriptors(),
            "buffer_geometry": {
                "replay_slots": self.replay_slots,
                "windows_per_generation": self.windows_per_generation,
                "batch_size": self.batch_size,
                "window_length": self.context_length + self.chunk_length,
            },
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Schedule a strict restore before workers are started."""
        if self._iterator is not None:
            raise RuntimeError("load state before creating the physical-shard iterator")
        if state.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError(f"unsupported physical-shard loader schema {state.get('schema')!r}")
        if state.get("data_protocol") != self.data_protocol:
            raise ValueError("data protocol changed across resume")
        if state.get("source_selection_sha256") != self._selection_hash:
            raise ValueError("source selection changed across resume")
        if state.get("source_manifest_sha256") != self.source_manifest_sha256:
            raise ValueError("source manifests changed across resume")
        expected_geometry = {
            "replay_slots": self.replay_slots,
            "windows_per_generation": self.windows_per_generation,
            "batch_size": self.batch_size,
            "window_length": self.context_length + self.chunk_length,
        }
        if state.get("buffer_geometry") != expected_geometry:
            raise ValueError("replay-buffer geometry changed across resume")
        cursor = state.get("cursor")
        if not isinstance(cursor, tuple) or len(cursor) != 3:
            raise ValueError("committed source cursor is invalid")
        typed_cursor = cast(tuple[int, int, int], cursor)
        if any(not isinstance(value, int) or value < 0 for value in typed_cursor) or typed_cursor[1] >= len(
            self.tasks
        ):
            raise ValueError("committed source cursor is invalid")
        descriptors = state.get("slots")
        sampler_state = state.get("batch_sampler_state")
        if not isinstance(descriptors, tuple) or len(descriptors) != self.replay_slots:
            raise ValueError("checkpoint does not describe every replay slot")
        if not all(isinstance(descriptor, GenerationDescriptor) for descriptor in descriptors):
            raise ValueError("checkpoint contains an invalid replay descriptor")
        if not isinstance(sampler_state, Mapping):
            raise ValueError("checkpoint has no batch-sampler state")
        epoch, task_offset, row_offset = typed_cursor
        task_index = permute_shard_tasks(self.tasks, seed=self.seed, epoch=epoch)[task_offset]
        if row_offset > self.tasks[task_index].row_count:
            raise ValueError("committed row cursor exceeds its shard")
        typed_descriptors = cast(tuple[GenerationDescriptor, ...], descriptors)
        if sorted(descriptor.slot for descriptor in typed_descriptors) != list(range(self.replay_slots)):
            raise ValueError("checkpoint replay slots are not a complete permutation")
        tasks_by_shard = {(task.source, task.shard): task for task in self.tasks}
        for descriptor in typed_descriptors:
            task = tasks_by_shard.get((descriptor.locator.source, descriptor.locator.shard))
            if (
                descriptor.epoch < 0
                or descriptor.replay_checksum < 0
                or not 0 <= descriptor.next_window < self.windows_per_generation
                or task is None
                or not task.row_start <= descriptor.locator.row < task.row_stop
                or descriptor.locator.row in task.excluded_rows
            ):
                raise ValueError("checkpoint contains an invalid replay descriptor")
        self._cursor = typed_cursor
        self._resume_descriptors = typed_descriptors
        self._resume_sampler_state = cast(Mapping[str, object], sampler_state)

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
                context_length=self.context_length,
                chunk_length=self.chunk_length,
                windows_per_generation=self.windows_per_generation,
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
                columns=_stack_window_rows(windows, windows_per_generation=self.windows_per_generation),
                windows_per_generation=self.windows_per_generation,
            )
            for row, descriptor in enumerate(ordered):
                replay_id = replay_ids[row]
                if _replay_checksum(replay_id) != descriptor.replay_checksum:
                    raise ValueError(f"replay identity changed at {descriptor.locator}")
            slots = np.asarray([descriptor.slot for descriptor in ordered])
            self._buffer.admit_rows(_DecodedRows(shard, 0, len(ordered)), slots, stagger=False)
            self._buffer.epochs[slots] = [descriptor.epoch for descriptor in ordered]
            self._buffer.next_windows[slots] = [descriptor.next_window for descriptor in ordered]

    def __enter__(self) -> PhysicalShardReplayLoader[BatchT]:
        if self._closed:
            raise RuntimeError("physical-shard loader is closed")
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        iterator = self._data_iterator
        self._data_iterator = None
        _shutdown_data_loader_workers(iterator)
