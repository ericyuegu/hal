"""Replay loading in deterministic physical-shard order.

Mosaic Streaming owns manifests, shard download, decompression, and MDS row
decoding. This module owns physical-shard order, bounded decoding, and replay
ring sampling.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import threading
import time
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
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
from streaming.base.dataset import _ShardState
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
CHECKPOINT_SCHEMA: Final[int] = 3
MAX_DECODE_CHUNK_ROWS: Final[int] = 64
MATERIALIZATION_LOG_INTERVAL_S: Final[float] = 60.0

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

    @classmethod
    def from_sources(cls, sources: tuple[SourceRowSelection, ...]) -> PhysicalShardSelection:
        """Construct a selection with its canonical persisted identity."""
        payload = [
            {
                "source": source.source,
                "stop": source.stop,
                "excluded_rows": list(source.excluded_rows),
            }
            for source in sources
        ]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return cls(sources, hashlib.sha256(encoded).hexdigest())

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


def _mds_schema_sha256(shard: Mapping[str, object]) -> str:
    fields = {
        name: shard.get(name)
        for name in (
            "column_names",
            "column_encodings",
            "column_sizes",
        )
    }
    encoded = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class HostMemoryEstimate:
    """Conservative host-memory components for bounded chunk prefetch."""

    central_buffer: int
    queued_chunks: int
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
                self.queued_chunks,
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
    decoded_chunk_bytes: int,
    replay_workspace_bytes: int,
    pinned_batch_bytes: int,
    pinned_batch_count: int,
    validation_cache_bytes: int,
    compiler_and_process_bytes: int,
    workers: int,
) -> HostMemoryEstimate:
    """Model every resident buffer involved in bounded chunk prefetch."""
    values = (
        central_buffer_bytes,
        decoded_chunk_bytes,
        replay_workspace_bytes,
        pinned_batch_bytes,
        pinned_batch_count,
        validation_cache_bytes,
        compiler_and_process_bytes,
        workers,
    )
    if any(value < 0 for value in values):
        raise ValueError("host-memory inputs must be non-negative")
    queued = workers * PREFETCH_FACTOR * decoded_chunk_bytes
    worker_outputs = workers * (decoded_chunk_bytes + replay_workspace_bytes)
    return HostMemoryEstimate(
        central_buffer=central_buffer_bytes,
        queued_chunks=queued,
        worker_outputs_and_workspaces=worker_outputs,
        ipc_copies=queued,
        parent_result=decoded_chunk_bytes,
        pinned_batches=pinned_batch_count * pinned_batch_bytes,
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
class _DecodeChunkRequest:
    """One bounded worker request in the fixed logical source order."""

    sequence: int
    epoch: int
    task_offset: int
    task_index: int
    row_offset: int
    rows: tuple[int, ...]

    def __post_init__(self) -> None:
        if min(self.sequence, self.epoch, self.task_offset, self.task_index, self.row_offset) < 0:
            raise ValueError("decode-chunk indices must be non-negative")
        if not self.rows or len(self.rows) > MAX_DECODE_CHUNK_ROWS:
            raise ValueError(f"a decode chunk must contain 1–{MAX_DECODE_CHUNK_ROWS} rows")
        if self.rows != tuple(sorted(set(self.rows))):
            raise ValueError("decode-chunk rows must be sorted and unique")


@dataclass(frozen=True, slots=True)
class DecodedChunk:
    """Fixed-window rows returned by one bounded worker request."""

    request: _DecodeChunkRequest
    task: ShardTask
    replay_ids: tuple[str, ...]
    locators: tuple[PhysicalRow, ...]
    columns: Mapping[str, np.ndarray]
    windows_per_generation: int
    raw_bytes_read: int = 0

    def __post_init__(self) -> None:
        rows = len(self.replay_ids)
        if rows != len(self.request.rows) or len(self.locators) != rows:
            raise ValueError("decoded chunk row metadata does not match its request")
        if any(
            locator.source != self.task.source or locator.shard != self.task.shard or locator.row != row
            for locator, row in zip(self.locators, self.request.rows, strict=True)
        ):
            raise ValueError("decoded chunk locators do not match its request")
        if not self.columns:
            raise ValueError("a decoded chunk has no columns")
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
            raise ValueError(f"decoded chunk columns must begin [R, {self.windows_per_generation}], got {bad}")

    @property
    def epoch(self) -> int:
        return self.request.epoch


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

    def validate_manifests(
        self,
        *,
        expected_sha256: Mapping[str, str],
        expected_index_version: int,
        expected_schema_sha256: str,
        expected_rows: Mapping[str, int],
    ) -> None:
        """Validate the exact MDS indexes exposed by this adapter."""
        selected_sources = {source.source for source in self.selection.sources}
        for name, values in (
            ("manifest hashes", expected_sha256),
            ("manifest row counts", expected_rows),
        ):
            if set(values) != selected_sources:
                raise ValueError(f"{name} do not cover the selected sources")
        if any(
            len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
            for digest in expected_sha256.values()
        ):
            raise ValueError("an expected manifest SHA-256 is invalid")
        if len(expected_schema_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_schema_sha256
        ):
            raise ValueError("expected MDS schema SHA-256 is invalid")

        for source in self.selection.sources:
            name = source.source
            path = streams_lib.BY_NAME[name].local_root / self.split / "index.json"
            payload = path.read_bytes()
            actual_hash = hashlib.sha256(payload).hexdigest()
            if actual_hash != expected_sha256[name]:
                raise ValueError(f"{name} {self.split}/index.json SHA-256 {actual_hash} != {expected_sha256[name]}")
            manifest = json.loads(payload)
            if not isinstance(manifest, dict) or manifest.get("version") != expected_index_version:
                raise ValueError(f"{name} {self.split}/index.json has an unsupported MDS index version")
            shards = manifest.get("shards")
            if not isinstance(shards, list) or not shards or not all(isinstance(shard, dict) for shard in shards):
                raise ValueError(f"{name} {self.split}/index.json has invalid shards")
            schema_hashes = {_mds_schema_sha256(shard) for shard in shards}
            if schema_hashes != {expected_schema_sha256}:
                raise ValueError(f"{name} {self.split}/index.json schema differs from the expected MDS schema")
            try:
                rows = sum(int(shard["samples"]) for shard in shards)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{name} {self.split}/index.json has invalid shard row counts") from error
            if rows != expected_rows[name]:
                raise ValueError(f"{name} {self.split}/index.json has {rows} rows, expected {expected_rows[name]}")
            adapter_rows = sum(self.manifests[name].samples_per_shard)
            if adapter_rows != rows:
                raise ValueError(f"{name} adapter exposes {adapter_rows} rows, expected {rows}")

    def _reader(self, task: ShardTask) -> MDSReader:
        reader = self.dataset.shards[task.global_shard]
        if not isinstance(reader, MDSReader):
            raise TypeError(f"physical shard loading requires MDS shards, found {type(reader).__name__}")
        return reader

    def prepare_shard(self, task: ShardTask) -> None:
        """Materialize one raw shard and reject an incomplete download."""
        try:
            self.dataset.prepare_shard(task.global_shard)
            manifest = self.manifests[task.source]
            if not _raw_shard_is_materialized(manifest, task.shard):
                raise FileNotFoundError(
                    f"Mosaic prepared {task.source} shard {task.shard} without its complete raw file"
                )
        except Exception:
            self._reset_failed_prepare(task)
            raise

    def _reset_failed_prepare(self, task: ShardTask) -> None:
        """Release Mosaic 0.13 waiters after its downloader raises.

        Mosaic Streaming 0.13.0 leaves the shared state at PREPARING when
        ``Stream.prepare_shard`` raises. Remove this workaround when Mosaic
        resets failed downloads itself.
        """
        lock = self.dataset._cache_filelock
        with lock:
            if self.dataset._shard_states[task.global_shard] != _ShardState.PREPARING:
                return
            manifest = self.manifests[task.source]
            state = _ShardState.LOCAL if _raw_shard_is_materialized(manifest, task.shard) else _ShardState.REMOTE
            self.dataset._shard_states[task.global_shard] = state

    def read_rows(self, task: ShardTask, rows: Sequence[int]) -> dict[int, Mapping[str, object]]:
        """Prepare a shard, open it once, and decode requested rows in byte order."""
        requested = tuple(sorted(set(int(row) for row in rows)))
        if not requested:
            return {}
        if any(row < task.row_start or row >= task.row_stop or row in task.excluded_rows for row in requested):
            raise ValueError("requested row is outside its shard task")
        self.prepare_shard(task)
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
        if not requests or len(requests) > MAX_DECODE_CHUNK_ROWS:
            raise ValueError(f"generation decode must contain 1–{MAX_DECODE_CHUNK_ROWS} rows")
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

    def decode_chunk(
        self,
        request: _DecodeChunkRequest,
        task: ShardTask,
        *,
        seed: int,
        context_length: int,
        chunk_length: int,
        windows_per_generation: int,
        schema_version: int,
        labels: ReplayLabels,
        projection: FeatureProjection | None,
    ) -> DecodedChunk:
        compact_rows = self.read_rows(task, request.rows)
        generations = tuple(
            _decode_generation(
                compact_rows[row],
                task=task,
                row=row,
                epoch=request.epoch,
                seed=seed,
                context_length=context_length,
                chunk_length=chunk_length,
                windows_per_generation=windows_per_generation,
                schema_version=schema_version,
                labels=labels,
                projection=projection,
            )
            for row in request.rows
        )
        return DecodedChunk(
            request=request,
            task=task,
            replay_ids=tuple(generation[0] for generation in generations),
            locators=tuple(PhysicalRow(task.source, task.shard, row) for row in request.rows),
            columns=_stack_window_rows(
                tuple(generation[1] for generation in generations),
                windows_per_generation=windows_per_generation,
            ),
            windows_per_generation=windows_per_generation,
            raw_bytes_read=self.last_read_bytes,
        )


def _raw_shard_is_materialized(manifest: SourceManifest, shard: int) -> bool:
    raw_path = manifest.raw_paths[shard]
    raw_bytes = manifest.raw_bytes_per_shard[shard]
    return raw_path.is_file() and raw_path.stat().st_size == raw_bytes


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
        if not _raw_shard_is_materialized(manifest, task.shard):
            missing_raw += raw_bytes
            compressed.append(manifest.zip_bytes_per_shard[task.shard])
    compressed.sort(reverse=True)
    return missing_raw + sum(compressed[:workers]) + reserved_bytes


class ShardStorageAdapter(Protocol):
    """Storage boundary used by worker processes and checkpoint restore."""

    @property
    def manifests(self) -> Mapping[str, SourceManifest]: ...

    def prepare_shard(self, task: ShardTask) -> None: ...

    def decode_chunk(
        self,
        request: _DecodeChunkRequest,
        task: ShardTask,
        *,
        seed: int,
        context_length: int,
        chunk_length: int,
        windows_per_generation: int,
        schema_version: int,
        labels: ReplayLabels,
        projection: FeatureProjection | None,
    ) -> DecodedChunk: ...

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


class _ChunkDataset(Dataset[DecodedChunk]):
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
        return 2**63 - 1

    def __getitem__(self, index: _DecodeChunkRequest) -> DecodedChunk:
        return self.adapter.decode_chunk(
            index,
            self.tasks[index.task_index],
            seed=self.seed,
            context_length=self.context_length,
            chunk_length=self.chunk_length,
            windows_per_generation=self.windows_per_generation,
            schema_version=self.schema_version,
            labels=self.labels,
            projection=self.projection,
        )


class _ChunkSampler(Sampler[_DecodeChunkRequest]):
    """Split the infinite logical row stream at cohort and shard boundaries."""

    def __init__(
        self,
        tasks: tuple[ShardTask, ...],
        seed: int,
        cursor: tuple[int, int, int],
        chunk_rows: int,
    ) -> None:
        if not tasks:
            raise ValueError("a chunk sampler needs at least one shard task")
        if chunk_rows < 1:
            raise ValueError("replay-cohort rows must be positive")
        self.order = permute_shard_tasks(tasks, seed=seed, epoch=0)
        self.selected_rows = tuple(task.selected_rows for task in tasks)
        self.cursor = cursor
        self.chunk_rows = chunk_rows
        epoch, task_offset, row_offset = cursor
        if epoch < 0 or not 0 <= task_offset < len(tasks):
            raise ValueError("chunk-sampler cursor is invalid")
        task_rows = len(self.selected_rows[self.order[task_offset]])
        if not 0 <= row_offset < task_rows:
            raise ValueError("chunk-sampler row cursor is invalid")
        rows_per_epoch = sum(task.row_count for task in tasks)
        prior_rows = sum(tasks[self.order[offset]].row_count for offset in range(task_offset))
        if (epoch * rows_per_epoch + prior_rows + row_offset) % chunk_rows:
            raise ValueError("chunk-sampler cursor is not on a replay-cohort boundary")

    def __iter__(self) -> Iterator[_DecodeChunkRequest]:
        epoch, task_offset, row_offset = self.cursor
        sequence = 0
        remaining = self.chunk_rows
        while True:
            task_index = self.order[task_offset]
            task_rows = self.selected_rows[task_index]
            count = min(remaining, len(task_rows) - row_offset, MAX_DECODE_CHUNK_ROWS)
            stop = row_offset + count
            yield _DecodeChunkRequest(
                sequence=sequence,
                epoch=epoch,
                task_offset=task_offset,
                task_index=task_index,
                row_offset=row_offset,
                rows=task_rows[row_offset:stop],
            )
            sequence += 1
            remaining -= count
            row_offset = stop
            if row_offset == len(task_rows):
                task_offset += 1
                row_offset = 0
                if task_offset == len(self.order):
                    epoch += 1
                    task_offset = 0
            if remaining == 0:
                remaining = self.chunk_rows

    def __len__(self) -> int:
        return 2**63 - 1


def _limit_worker_threads(_worker_id: int) -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)


def _identity(value: DecodedChunk) -> DecodedChunk:
    return value


class _OrderedChunks(Iterator[DecodedChunk]):
    """Restore logical request order after out-of-order worker completion."""

    def __init__(self, chunks: Iterator[DecodedChunk]) -> None:
        self._chunks = chunks
        self._expected = 0
        self._pending: dict[int, DecodedChunk] = {}
        self._max_pending = 0

    def __iter__(self) -> _OrderedChunks:
        return self

    def __next__(self) -> DecodedChunk:
        while self._expected not in self._pending:
            chunk = next(self._chunks)
            sequence = chunk.request.sequence
            if sequence < self._expected or sequence in self._pending:
                raise RuntimeError(f"worker returned duplicate decode chunk {sequence}")
            self._pending[sequence] = chunk
            self._max_pending = max(self._max_pending, len(self._pending))
        chunk = self._pending.pop(self._expected)
        self._expected += 1
        return chunk

    @property
    def pending(self) -> int:
        return len(self._pending)

    @property
    def max_pending(self) -> int:
        return self._max_pending


class _ReplayRingSchedule:
    """Derive replay slots and window ordinals from a FIFO ring position."""

    def __init__(
        self,
        capacity: int,
        batch_size: int,
        windows_per_generation: int,
        phase_block_batches: int,
        seed: int,
    ) -> None:
        if capacity < 1 or batch_size < 1:
            raise ValueError("replay-ring capacity and batch size must be positive")
        if not 2 <= windows_per_generation <= phase_block_batches:
            raise ValueError("phase block must cover at least two windows per generation")
        if capacity % batch_size or batch_size % windows_per_generation:
            raise ValueError("replay-ring geometry must divide evenly")
        self.capacity = capacity
        self.batch_size = batch_size
        self.windows_per_generation = windows_per_generation
        self.replay_lanes = batch_size // windows_per_generation
        self.period_batches = capacity // batch_size
        self.cohort_count = capacity // self.replay_lanes
        if self.period_batches < phase_block_batches:
            raise ValueError(f"replay period must cover the {phase_block_batches}-batch phase block")
        self.minimum_gap_batches = self.period_batches - phase_block_batches + 1
        self.maximum_gap_batches = self.period_batches + phase_block_batches - 1
        rng = np.random.default_rng(seed)
        self.phase_offsets = np.stack(
            [
                rng.choice(
                    phase_block_batches,
                    size=self.windows_per_generation,
                    replace=False,
                )
                for _ in range(self.replay_lanes)
            ]
        ).astype(np.int16, copy=False)
        self.phase_offsets.flags.writeable = False
        self._window_ordinals = np.repeat(
            np.arange(self.windows_per_generation, dtype=np.int64),
            self.replay_lanes,
        )
        self._window_ordinals.flags.writeable = False
        self._lane_indices = np.tile(np.arange(self.replay_lanes, dtype=np.int64), self.windows_per_generation)
        self._lane_indices.flags.writeable = False
        phases = self.phase_offsets[self._lane_indices, self._window_ordinals]
        ages = self._window_ordinals * self.period_batches + phases
        self._cohort_offsets = (-1 - ages) % self.cohort_count
        self._cohort_offsets.flags.writeable = False
        gaps = self.period_batches + np.diff(self.phase_offsets, axis=1)
        self.reuse_gaps = gaps.reshape(-1).astype(np.int64, copy=False)
        self.reuse_gaps.flags.writeable = False
        age_values = self.reuse_gaps.astype(np.float64, copy=False)
        self.replay_age_metrics = {
            "data/replay_age_le_1_fraction": float(np.mean(age_values <= 1)),
            "data/replay_age_le_16_fraction": float(np.mean(age_values <= 16)),
            "data/replay_age_p01": float(np.percentile(age_values, 1)),
            "data/replay_age_p05": float(np.percentile(age_values, 5)),
            "data/replay_age_p50": float(np.percentile(age_values, 50)),
            "data/replay_age_p95": float(np.percentile(age_values, 95)),
        }
        if len(np.unique(self.selected_slots(0))) != self.batch_size:
            raise RuntimeError("replay-ring schedule selected one slot twice")

    @property
    def window_ordinals(self) -> np.ndarray:
        return self._window_ordinals

    def selected_slots(self, fifo_head: int) -> np.ndarray:
        if not 0 <= fifo_head < self.cohort_count:
            raise ValueError("FIFO head is outside the replay ring")
        cohorts = (fifo_head + self._cohort_offsets) % self.cohort_count
        return cohorts * self.replay_lanes + self._lane_indices


@dataclass(frozen=True, slots=True)
class GenerationDescriptor:
    """Legacy experiment-051 v5/v6 pickle payload."""

    slot: int
    locator: PhysicalRow
    epoch: int
    replay_checksum: int
    next_window: int


@dataclass(frozen=True, slots=True)
class RingSlotDescriptor:
    slot: int
    locator: PhysicalRow
    epoch: int
    replay_checksum: int


def _materialization_task_order(
    tasks: tuple[ShardTask, ...],
    task_order: tuple[int, ...],
    cursor_task_offset: int,
    resume_descriptors: Sequence[RingSlotDescriptor],
) -> tuple[int, ...]:
    """Prioritize the cursor and resume ring within deterministic shard order."""
    if sorted(task_order) != list(range(len(tasks))):
        raise ValueError("materialization task order is not a permutation")
    if not 0 <= cursor_task_offset < len(task_order):
        raise ValueError("materialization cursor is outside the shard order")
    tasks_by_shard = {(task.source, task.shard): index for index, task in enumerate(tasks)}
    if len(tasks_by_shard) != len(tasks):
        raise ValueError("materialization tasks repeat a physical shard")
    resume_shards = {(item.locator.source, item.locator.shard) for item in resume_descriptors}
    missing_resume_shards = resume_shards - tasks_by_shard.keys()
    if missing_resume_shards:
        raise ValueError(f"resume ring contains shards outside the source plan: {sorted(missing_resume_shards)}")

    prioritized: list[int] = []
    seen: set[int] = set()

    def add(task_index: int) -> None:
        if task_index not in seen:
            seen.add(task_index)
            prioritized.append(task_index)

    add(task_order[cursor_task_offset])
    for task_index in task_order:
        task = tasks[task_index]
        if (task.source, task.shard) in resume_shards:
            add(task_index)
    for task_index in (*task_order[cursor_task_offset:], *task_order[:cursor_task_offset]):
        add(task_index)
    return tuple(prioritized)


class _ShardMaterializer:
    """Materialize raw shards once with bounded background concurrency."""

    def __init__(
        self,
        adapter: ShardStorageAdapter,
        tasks: tuple[ShardTask, ...],
        order: tuple[int, ...],
        *,
        workers: int,
    ) -> None:
        if workers < 1:
            raise ValueError("a shard materializer needs at least one worker")
        if sorted(order) != list(range(len(tasks))):
            raise ValueError("materialization order is not a task permutation")
        self._adapter = adapter
        self._tasks = tasks
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._materialized = {
            task_index
            for task_index, task in enumerate(tasks)
            if _raw_shard_is_materialized(adapter.manifests[task.source], task.shard)
        }
        self._prepared_shards = 0
        self._prepared_bytes = 0
        self._download_errors = 0
        self._first_failure: tuple[ShardTask, Exception] | None = None
        self._started = time.monotonic()
        self._last_progress_log = self._started
        self._finished = self._started if len(self._materialized) == len(tasks) else None
        self._closed = False
        missing = len(tasks) - len(self._materialized)
        missing_bytes = sum(
            adapter.manifests[task.source].raw_bytes_per_shard[task.shard]
            for task_index, task in enumerate(tasks)
            if task_index not in self._materialized
        )
        if missing:
            print(
                "[loader] starting background shard materialization: "
                f"{missing:,} shards ({missing_bytes / 2**30:.1f} GiB) to download with {workers} workers; "
                f"{len(self._materialized):,}/{len(tasks):,} already local",
                flush=True,
            )
        else:
            print(f"[loader] background shard materialization: all {len(tasks):,} shards already local", flush=True)
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="raw-shard-materializer")
        self._futures: list[Future[None]] = []
        try:
            self._futures = [
                self._executor.submit(self._prepare, task_index)
                for task_index in order
                if task_index not in self._materialized
            ]
        except RuntimeError:
            self._stop.set()
            self._executor.shutdown(wait=True, cancel_futures=True)
            raise

    def _prepare(self, task_index: int) -> None:
        if self._stop.is_set():
            return
        task = self._tasks[task_index]
        try:
            self._adapter.prepare_shard(task)
            manifest = self._adapter.manifests[task.source]
            if not _raw_shard_is_materialized(manifest, task.shard):
                raise FileNotFoundError(f"materialized {task.source} shard {task.shard} has no complete raw file")
        except Exception as error:
            with self._lock:
                self._download_errors += 1
                if self._first_failure is None:
                    self._first_failure = (task, error)
                    self._stop.set()
            raise
        progress: tuple[int, int, int, float, bool] | None = None
        with self._lock:
            self._materialized.add(task_index)
            self._prepared_shards += 1
            self._prepared_bytes += manifest.raw_bytes_per_shard[task.shard]
            now = time.monotonic()
            complete = len(self._materialized) == len(self._tasks)
            if complete:
                self._finished = now
            if complete or now - self._last_progress_log >= MATERIALIZATION_LOG_INTERVAL_S:
                self._last_progress_log = now
                progress = (
                    len(self._materialized),
                    self._prepared_shards,
                    self._prepared_bytes,
                    now - self._started,
                    complete,
                )
        if progress is not None:
            ready, downloaded, downloaded_bytes, elapsed, complete = progress
            state = "complete" if complete else "progress"
            rate_mib_s = downloaded_bytes / 2**20 / max(elapsed, 1e-12)
            print(
                f"[loader] background shard materialization {state}: "
                f"{ready:,}/{len(self._tasks):,} shards ready; "
                f"downloaded {downloaded:,} shards ({downloaded_bytes / 2**30:.1f} GiB) "
                f"in {elapsed:.1f}s ({rate_mib_s:.1f} MiB/s)",
                flush=True,
            )

    def raise_if_failed(self) -> None:
        with self._lock:
            failure = self._first_failure
        if failure is None:
            return
        task, error = failure
        raise RuntimeError(f"background materialization failed for {task.source} shard {task.shard}") from error

    def metrics(self, task_order: tuple[int, ...], cursor_task_offset: int) -> dict[str, float]:
        with self._lock:
            materialized = set(self._materialized)
            prepared_shards = self._prepared_shards
            prepared_bytes = self._prepared_bytes
            download_errors = self._download_errors
            finished = self._finished
        ordered_lead = (*task_order[cursor_task_offset:], *task_order[:cursor_task_offset])
        contiguous_lead = 0
        for task_index in ordered_lead:
            if task_index not in materialized:
                break
            contiguous_lead += 1
        elapsed = max((finished or time.monotonic()) - self._started, 1e-12)
        return {
            "loader/materialized_shards": float(len(materialized)),
            "loader/remaining_shards": float(len(self._tasks) - len(materialized)),
            "loader/contiguous_materialized_shard_lead": float(contiguous_lead),
            "loader/materialization_shards_per_s": prepared_shards / elapsed,
            "loader/materialization_bytes_per_s": prepared_bytes / elapsed,
            "loader/materialization_download_errors": float(download_errors),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._executor.shutdown(wait=True, cancel_futures=True)


class _ReplayRing:
    """Columnar replay generations ordered as FIFO cohorts."""

    def __init__(
        self,
        capacity: int,
        batch_size: int,
        windows_per_generation: int,
        phase_block_batches: int,
        seed: int,
    ) -> None:
        if capacity < batch_size:
            raise ValueError("replay ring must cover one batch")
        self.schedule = _ReplayRingSchedule(
            capacity,
            batch_size,
            windows_per_generation,
            phase_block_batches,
            seed,
        )
        self.columns: dict[str, np.ndarray] = {}
        self.replay_ids: list[str | None] = [None] * capacity
        self.locators: list[PhysicalRow | None] = [None] * capacity
        self.epochs = np.full(capacity, -1, dtype=np.int64)
        self._size = 0
        self.fifo_head = 0
        self.batch_index = 0

    @property
    def size(self) -> int:
        return self._size

    def _initialize_columns(self, chunk: DecodedChunk) -> None:
        if self.columns:
            return
        for name, values in chunk.columns.items():
            self.columns[name] = np.empty((self.schedule.capacity, *values.shape[1:]), dtype=values.dtype)

    def _write_chunk(
        self,
        chunk: DecodedChunk,
        slots: np.ndarray,
        *,
        require_empty: bool,
    ) -> None:
        if chunk.windows_per_generation != self.schedule.windows_per_generation:
            raise ValueError("decoded chunk does not match replay-ring window geometry")
        self._initialize_columns(chunk)
        if set(self.columns) != set(chunk.columns):
            raise ValueError("decoded chunk columns changed after ring allocation")
        slots = np.asarray(slots, dtype=np.int64)
        if slots.shape != (len(chunk.replay_ids),) or len(np.unique(slots)) != len(chunk.replay_ids):
            raise ValueError("destination slots do not match decoded rows")
        if np.any(slots < 0) or np.any(slots >= self.schedule.capacity):
            raise ValueError("destination slot is outside the replay ring")
        occupied = [self.replay_ids[int(slot)] is not None for slot in slots]
        if require_empty and any(occupied):
            raise RuntimeError("destination replay slot is already occupied")
        if not require_empty and not all(occupied):
            raise RuntimeError("FIFO replacement encountered an empty replay slot")
        if len(set(chunk.replay_ids)) != len(chunk.replay_ids):
            raise ValueError("decoded rows repeat a replay identity")
        for name, destination in self.columns.items():
            source = chunk.columns[name]
            if source.shape[1:] != destination.shape[1:] or source.dtype != destination.dtype:
                raise ValueError(f"decoded column {name!r} does not match the replay ring")
            destination[slots] = source
        for slot_value, replay_id, locator in zip(slots, chunk.replay_ids, chunk.locators, strict=True):
            slot = int(slot_value)
            self.replay_ids[slot] = replay_id
            self.locators[slot] = locator
        self.epochs[slots] = chunk.epoch

    def append_chunk(self, chunk: DecodedChunk) -> None:
        stop = self.size + len(chunk.replay_ids)
        if stop > self.schedule.capacity:
            raise RuntimeError("decoded rows exceed replay-ring capacity")
        self._write_chunk(chunk, np.arange(self.size, stop), require_empty=True)
        self._size = stop

    def restore_chunk(self, chunk: DecodedChunk, slots: np.ndarray) -> None:
        if self.size + len(chunk.replay_ids) > self.schedule.capacity:
            raise RuntimeError("restored rows exceed replay-ring capacity")
        self._write_chunk(chunk, slots, require_empty=True)
        self._size += len(chunk.replay_ids)

    def sample(self) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
        """Select the derived window from each replay in the current batch."""
        if self.size != self.schedule.capacity:
            raise RuntimeError("replay ring must be full before sampling")
        if self.fifo_head != self.batch_index % self.schedule.cohort_count:
            raise RuntimeError("replay-ring FIFO and batch cursors diverged")
        slots = self.schedule.selected_slots(self.fifo_head)
        selected_ids = tuple(self.replay_ids[int(slot)] for slot in slots)
        if any(replay_id is None for replay_id in selected_ids):
            raise RuntimeError("replay-ring schedule selected an empty slot")
        replay_ids = cast(tuple[str, ...], selected_ids)
        if len(set(replay_ids)) != self.schedule.batch_size:
            raise RuntimeError("replay-ring batch contains a repeated replay identity")
        ordinals = self.schedule.window_ordinals
        columns = {name: values[slots, ordinals] for name, values in self.columns.items()}
        return replay_ids, columns

    def replace_oldest(self, chunks: Sequence[DecodedChunk]) -> None:
        rows = sum(len(chunk.replay_ids) for chunk in chunks)
        if rows != self.schedule.replay_lanes:
            raise ValueError(f"FIFO replacement needs {self.schedule.replay_lanes} rows, got {rows}")
        replay_ids = [replay_id for chunk in chunks for replay_id in chunk.replay_ids]
        if len(set(replay_ids)) != rows:
            raise ValueError("replacement cohort repeats a replay identity")
        start = self.fifo_head * self.schedule.replay_lanes
        offset = 0
        for chunk in chunks:
            stop = offset + len(chunk.replay_ids)
            self._write_chunk(chunk, np.arange(start + offset, start + stop), require_empty=False)
            offset = stop
        self.fifo_head = (self.fifo_head + 1) % self.schedule.cohort_count
        self.batch_index += 1

    def set_position(self, fifo_head: int, batch_index: int) -> None:
        if self.size != self.schedule.capacity:
            raise RuntimeError("cannot position an incomplete replay ring")
        if not 0 <= fifo_head < self.schedule.cohort_count or batch_index < 0:
            raise ValueError("replay-ring position is invalid")
        if fifo_head != batch_index % self.schedule.cohort_count:
            raise ValueError("replay-ring FIFO and batch cursors are inconsistent")
        self.fifo_head = fifo_head
        self.batch_index = batch_index

    def descriptors(self) -> tuple[RingSlotDescriptor, ...]:
        out = []
        for slot, replay_id in enumerate(self.replay_ids):
            if replay_id is None:
                continue
            locator = self.locators[slot]
            assert locator is not None
            out.append(
                RingSlotDescriptor(
                    slot=slot,
                    locator=locator,
                    epoch=int(self.epochs[slot]),
                    replay_checksum=_replay_checksum(replay_id),
                )
            )
        return tuple(out)


class _PhysicalShardIterator[BatchT](Iterator[BatchT]):
    def __init__(self, loader: PhysicalShardReplayLoader[BatchT], chunks: Iterator[DecodedChunk]) -> None:
        self.loader = loader
        self.chunks = chunks
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

    def take_cohort(self) -> tuple[DecodedChunk, ...]:
        chunks: list[DecodedChunk] = []
        rows = 0
        cohort_rows = self.loader._ring.schedule.replay_lanes
        while rows < cohort_rows:
            self.loader._raise_materialization_failure()
            decoded = next(self.chunks)
            self.loader._raise_materialization_failure()
            count = len(decoded.replay_ids)
            if rows + count > cohort_rows:
                raise RuntimeError("decode chunk crosses a replay-cohort boundary")
            epoch, task_offset, row_offset = self.loader._cursor
            request = decoded.request
            expected_task_index = self.loader._task_order[task_offset]
            if (
                request.epoch,
                request.task_offset,
                request.task_index,
                request.row_offset,
            ) != (epoch, task_offset, expected_task_index, row_offset):
                raise RuntimeError(
                    "DataLoader released a chunk out of committed order: "
                    f"got {(request.epoch, request.task_offset, request.row_offset)}, "
                    f"expected {(epoch, task_offset, row_offset)}"
                )
            task_rows = self.loader._selected_rows[expected_task_index]
            stop = row_offset + count
            if request.rows != task_rows[row_offset:stop]:
                raise RuntimeError("decode chunk rows differ from the committed source order")
            if stop == len(task_rows):
                self.loader._advance_task_cursor()
            else:
                self.loader._cursor = (epoch, task_offset, stop)
            self.loader._raw_bytes_read += decoded.raw_bytes_read
            decoded_bytes = sum(values.nbytes for values in decoded.columns.values())
            self.loader._max_decoded_chunk_bytes = max(self.loader._max_decoded_chunk_bytes, decoded_bytes)
            self.loader._max_decoded_chunk_size = max(self.loader._max_decoded_chunk_size, count)
            self.loader._decoded_generations += count
            chunks.append(decoded)
            rows += count
        return tuple(chunks)


def _shutdown_data_loader_workers(iterator: object | None) -> None:
    """Use PyTorch's private worker shutdown until it exposes a public close."""
    if iterator is None:
        return
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


class PhysicalShardReplayLoader[BatchT]:
    """Infinite physical-shard loader with decoded-array-free checkpoints."""

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
        replay_phase_block_batches: int,
        schema_version: int,
        reserved_disk_bytes: int,
        pin_memory: bool,
        materialization_threads: int = 0,
    ) -> None:
        if replay_slots > selection.row_count:
            raise ValueError("replay slots exceed the selected source rows")
        if replay_slots < batch_size:
            raise ValueError("selection is too small for one identity-distinct batch")
        if not tasks or sum(task.row_count for task in tasks) != selection.row_count:
            raise ValueError("shard tasks do not cover the source selection")
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if context_length < 1 or chunk_length < 0 or windows_per_generation < 1:
            raise ValueError("window geometry is invalid")
        if reserved_disk_bytes < 0:
            raise ValueError("reserved disk bytes must be non-negative")
        if materialization_threads < 0:
            raise ValueError("materialization threads must be non-negative")
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
        if materialization_threads:
            for task in tasks:
                manifest = adapter.manifests[task.source]
                if task.global_shard < 0:
                    raise ValueError("materialized shard tasks need global shard indices")
                if not manifest.raw_bytes_per_shard or not manifest.zip_bytes_per_shard or not manifest.raw_paths:
                    raise ValueError(f"materialization needs complete file metadata for {task.source}")
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
        self.materialization_threads = materialization_threads
        self.source_sample_counts = selection.row_counts_by_source()
        self._selection_hash = selection.sha256
        self._task_order = permute_shard_tasks(tasks, seed=seed, epoch=0)
        self._selected_rows = tuple(task.selected_rows for task in tasks)
        self._cursor: tuple[int, int, int] = (0, 0, 0)
        self._ring = _ReplayRing(
            replay_slots,
            batch_size,
            windows_per_generation,
            replay_phase_block_batches,
            seed ^ 0x51B0FF,
        )
        self._resume_descriptors: tuple[RingSlotDescriptor, ...] | None = None
        self._resume_position: tuple[int, int] | None = None
        self._iterator: _PhysicalShardIterator[BatchT] | None = None
        self._data_iterator: Iterator[DecodedChunk] | None = None
        self._ordered_chunks: _OrderedChunks | None = None
        self._materializer: _ShardMaterializer | None = None
        self._parent_next_active = False
        self._raw_bytes_read = 0
        self._decoded_generations = 0
        self._max_decoded_chunk_size = 0
        self._max_decoded_chunk_bytes = 0
        self._closed = False

    @property
    def metrics(self) -> dict[str, float]:
        metrics = dict(self._ring.schedule.replay_age_metrics)
        generation_count = self.generation_count
        generation_epoch = generation_count / self.selection.row_count
        metrics.update(
            {
                "data/epoch": generation_epoch,
                "data/decoded_generations": float(self._decoded_generations),
                "data/replay_generations": float(generation_count),
                "data/raw_bytes_read": float(self._raw_bytes_read),
                "data/replay_generation_epoch": generation_epoch,
                "data/max_decoded_chunk_size": float(self._max_decoded_chunk_size),
                "loader/ring_size": float(self._ring.size),
                "loader/ring_batch_index": float(self._ring.batch_index),
                "loader/ring_fifo_head": float(self._ring.fifo_head),
            }
        )
        if self._ordered_chunks is not None:
            metrics["loader/reorder_backlog"] = float(self._ordered_chunks.pending)
            metrics["loader/reorder_backlog_max"] = float(self._ordered_chunks.max_pending)
        if self._materializer is not None:
            metrics.update(self._materializer.metrics(self._task_order, self._cursor[1]))
        return metrics

    @property
    def buffer_bytes(self) -> int:
        return sum(values.nbytes for values in self._ring.columns.values())

    @property
    def raw_bytes_read(self) -> int:
        return self._raw_bytes_read

    @property
    def minimum_replay_gap_batches(self) -> int:
        return self._ring.schedule.minimum_gap_batches

    @property
    def reuse_period_batches(self) -> int:
        return self._ring.schedule.period_batches

    @property
    def missing_raw_shards(self) -> int:
        missing = 0
        for task in self.tasks:
            manifest = self.adapter.manifests[task.source]
            missing += not _raw_shard_is_materialized(manifest, task.shard)
        return missing

    @property
    def decoded_generations(self) -> int:
        return self._decoded_generations

    @property
    def generation_count(self) -> int:
        """Return logical source generations committed to the replay ring."""
        return self._ring.size + self._ring.batch_index * self._ring.schedule.replay_lanes

    @property
    def max_decoded_chunk_size(self) -> int:
        return self._max_decoded_chunk_size

    @property
    def max_decoded_chunk_bytes(self) -> int:
        return self._max_decoded_chunk_bytes

    @property
    def required_disk_bytes(self) -> int:
        return disk_requirement_bytes(
            self.tasks,
            self.adapter.manifests,
            workers=self.num_workers + self.materialization_threads,
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
        dataset = _ChunkDataset(
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
        sampler = _ChunkSampler(
            self.tasks,
            self.seed,
            self._cursor,
            self._ring.schedule.replay_lanes,
        )
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
                in_order=False,
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
                in_order=False,
            )
        data_iterator = iter(data_loader)
        ordered_chunks = _OrderedChunks(data_iterator)
        materializer = None
        if self.materialization_threads:
            materialization_order = _materialization_task_order(
                self.tasks,
                self._task_order,
                self._cursor[1],
                self._resume_descriptors or (),
            )
            try:
                materializer = _ShardMaterializer(
                    self.adapter,
                    self.tasks,
                    materialization_order,
                    workers=self.materialization_threads,
                )
            except Exception:
                _shutdown_data_loader_workers(data_iterator)
                raise
        self._data_iterator = data_iterator
        self._ordered_chunks = ordered_chunks
        self._materializer = materializer
        self._iterator = _PhysicalShardIterator(self, ordered_chunks)
        return self._iterator

    def _raise_materialization_failure(self) -> None:
        if self._materializer is not None:
            self._materializer.raise_if_failed()

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
        self._raise_materialization_failure()
        if self._resume_descriptors is not None:
            assert self._resume_position is not None
            self._restore_ring(self._resume_descriptors, *self._resume_position)
            self._resume_descriptors = None
            self._resume_position = None
        while self._ring.size < self.replay_slots:
            chunks = iterator.take_cohort()
            for chunk in chunks:
                self._ring.append_chunk(chunk)
        replay_ids, columns = self._ring.sample()
        transformed = self.batch_transform(replay_ids, columns)
        replacements = iterator.take_cohort()
        self._ring.replace_oldest(replacements)
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
        if self._ring.size != self.replay_slots:
            raise RuntimeError("cannot checkpoint before the replay ring is full")
        return {
            "schema": CHECKPOINT_SCHEMA,
            "data_protocol": self.data_protocol,
            "source_selection_sha256": self._selection_hash,
            "source_manifest_sha256": self.source_manifest_sha256,
            "cursor": self._cursor,
            "fifo_head": self._ring.fifo_head,
            "batch_index": self._ring.batch_index,
            "slots": self._ring.descriptors(),
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
            raise ValueError("replay-ring geometry changed across resume")
        cursor = state.get("cursor")
        if not isinstance(cursor, tuple) or len(cursor) != 3:
            raise ValueError("committed source cursor is invalid")
        typed_cursor = cast(tuple[int, int, int], cursor)
        if any(not isinstance(value, int) or value < 0 for value in typed_cursor) or typed_cursor[1] >= len(
            self.tasks
        ):
            raise ValueError("committed source cursor is invalid")
        descriptors = state.get("slots")
        fifo_head = state.get("fifo_head")
        batch_index = state.get("batch_index")
        if not isinstance(descriptors, tuple) or len(descriptors) != self.replay_slots:
            raise ValueError("checkpoint does not describe every replay slot")
        if not all(isinstance(descriptor, RingSlotDescriptor) for descriptor in descriptors):
            raise ValueError("checkpoint contains an invalid replay descriptor")
        if not isinstance(fifo_head, int) or not isinstance(batch_index, int):
            raise ValueError("checkpoint has no replay-ring position")
        if (
            not 0 <= fifo_head < self._ring.schedule.cohort_count
            or batch_index < 0
            or fifo_head != batch_index % self._ring.schedule.cohort_count
        ):
            raise ValueError("checkpoint replay-ring position is invalid")
        epoch, task_offset, row_offset = typed_cursor
        source_task_index = self._task_order[task_offset]
        if row_offset >= self.tasks[source_task_index].row_count:
            raise ValueError("committed row cursor exceeds its shard")
        prior_rows = sum(self.tasks[self._task_order[offset]].row_count for offset in range(task_offset))
        source_position = epoch * self.selection.row_count + prior_rows + row_offset
        expected_source_position = self.replay_slots + batch_index * self._ring.schedule.replay_lanes
        if source_position != expected_source_position:
            raise ValueError("committed source cursor does not match the replay-ring position")
        typed_descriptors = cast(tuple[RingSlotDescriptor, ...], descriptors)
        if sorted(descriptor.slot for descriptor in typed_descriptors) != list(range(self.replay_slots)):
            raise ValueError("checkpoint replay slots are not a complete permutation")
        tasks_by_shard = {(task.source, task.shard): task for task in self.tasks}
        for descriptor in typed_descriptors:
            task = tasks_by_shard.get((descriptor.locator.source, descriptor.locator.shard))
            if (
                descriptor.epoch < 0
                or descriptor.replay_checksum < 0
                or task is None
                or not task.row_start <= descriptor.locator.row < task.row_stop
                or descriptor.locator.row in task.excluded_rows
            ):
                raise ValueError("checkpoint contains an invalid replay descriptor")
        self._cursor = typed_cursor
        self._resume_descriptors = typed_descriptors
        self._resume_position = (fifo_head, batch_index)

    def _restore_ring(
        self,
        descriptors: tuple[RingSlotDescriptor, ...],
        fifo_head: int,
        batch_index: int,
    ) -> None:
        tasks = {(task.source, task.shard): (task_index, task) for task_index, task in enumerate(self.tasks)}
        cohorts: dict[tuple[str, int, int], list[RingSlotDescriptor]] = {}
        for descriptor in descriptors:
            key = (descriptor.locator.source, descriptor.locator.shard, descriptor.epoch)
            cohorts.setdefault(key, []).append(descriptor)
        for key in sorted(cohorts):
            self._raise_materialization_failure()
            task_entry = tasks.get(key[:2])
            if task_entry is None:
                raise ValueError(f"checkpoint row shard {key[:2]} is not in the source plan")
            task_index, task = task_entry
            cohort = sorted(cohorts[key], key=lambda item: item.locator.row)
            for start in range(0, len(cohort), MAX_DECODE_CHUNK_ROWS):
                ordered = cohort[start : start + MAX_DECODE_CHUNK_ROWS]
                requests = [(item.locator.row, item.epoch) for item in ordered]
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
                replay_ids = tuple(
                    generations[(descriptor.locator.row, descriptor.epoch)][0] for descriptor in ordered
                )
                windows = tuple(generations[(descriptor.locator.row, descriptor.epoch)][1] for descriptor in ordered)
                for replay_id, descriptor in zip(replay_ids, ordered, strict=True):
                    if _replay_checksum(replay_id) != descriptor.replay_checksum:
                        raise ValueError(f"replay identity changed at {descriptor.locator}")
                rows = tuple(descriptor.locator.row for descriptor in ordered)
                request = _DecodeChunkRequest(
                    sequence=0,
                    epoch=key[2],
                    task_offset=0,
                    task_index=task_index,
                    row_offset=0,
                    rows=rows,
                )
                chunk = DecodedChunk(
                    request=request,
                    task=task,
                    replay_ids=replay_ids,
                    locators=tuple(descriptor.locator for descriptor in ordered),
                    columns=_stack_window_rows(
                        windows,
                        windows_per_generation=self.windows_per_generation,
                    ),
                    windows_per_generation=self.windows_per_generation,
                )
                self._ring.restore_chunk(
                    chunk,
                    np.asarray([descriptor.slot for descriptor in ordered]),
                )
                count = len(ordered)
                self._decoded_generations += count
                self._max_decoded_chunk_size = max(self._max_decoded_chunk_size, count)
                decoded_bytes = sum(values.nbytes for values in chunk.columns.values())
                self._max_decoded_chunk_bytes = max(self._max_decoded_chunk_bytes, decoded_bytes)
        self._ring.set_position(fifo_head, batch_index)

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
        materializer = self._materializer
        self._materializer = None
        if materializer is not None:
            materializer.close()
        iterator = self._data_iterator
        self._data_iterator = None
        self._ordered_chunks = None
        _shutdown_data_loader_workers(iterator)
