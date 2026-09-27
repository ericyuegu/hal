"""Measure the 059 buffered MDS replay loader without model execution.

This measures physical-shard decoding and replay-ring scheduling. The batch
transform keeps projected arrays on the CPU, so these numbers do not include
059's return labels, feature normalization, or training prefetch.
"""

from __future__ import annotations

import configparser
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from typing import cast

import numpy as np
import tyro

from hal import streams
from hal.representation.features import BASE_ACTION_PROJECTION
from hal.training.buffered_mds_replay_loader import MAX_DECODE_CHUNK_ROWS
from hal.training.buffered_mds_replay_loader import BufferedMDSReplayLoader
from hal.training.buffered_mds_replay_loader import MDSStorageAdapter
from hal.training.buffered_mds_replay_loader import PhysicalShardSelection
from hal.training.buffered_mds_replay_loader import ShardTask
from hal.training.buffered_mds_replay_loader import SourceManifest
from hal.training.buffered_mds_replay_loader import SourceRowSelection
from hal.training.buffered_mds_replay_loader import build_shard_plan
from hal.training.buffered_mds_replay_loader import estimate_host_memory
from hal.training.system_metrics import process_tree_pids
from hal.training.system_metrics import read_process_tree_memory

O59_SCHEMA_SHA256: Final[str] = "405199de9494fe01350506734f0b2ec392fe79b0122d69cbcb5cae2afabc0d49"
O59_SELECTION_SHA256: Final[str] = "2593361352b92e705be3fbeae1b4e9bb1a3c9f1787cd713014a7a95b7df62477"


@dataclass(frozen=True, slots=True)
class Args:
    sources: tuple[str, ...] = tuple(source.name for source in streams.POLICY_WORLD_V8_SOURCES)
    warm_batches: int = 200
    measured_batches: int = 500
    batch_size: int = 512
    replay_slots: int = 131_072
    windows_per_generation: int = 8
    phase_block_batches: int = 25
    num_workers: int = 24
    materialization_threads: int = 4
    seed: int = 0
    hash_batches: bool = False
    label: str = "loader-core"
    output: Path | None = None
    local_repo: Path | None = None
    rclone_config: Path | None = None
    estimate_only: bool = False
    cache_status_only: bool = False
    materialize_only: bool = False
    max_memory_fraction: float = 0.5


@dataclass(frozen=True, slots=True)
class _Batch:
    replay_ids: tuple[str, ...]
    columns: Mapping[str, np.ndarray]


def _no_labels(_row: Mapping[str, object]) -> dict[str, np.ndarray]:
    return {}


def _keep_columns(replay_ids: tuple[str, ...], columns: Mapping[str, np.ndarray]) -> _Batch:
    return _Batch(replay_ids, columns)


def _batch_digest(batch: _Batch) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for name, values in sorted(batch.columns.items()):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _host_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo does not report MemAvailable")


def _ring_slot_bytes(windows_per_generation: int) -> int:
    # Schema-7 policy-world decoding yields 4-byte values for these 61
    # per-frame columns. Each window also stores one int64 ctx_pad scalar.
    frame_bytes = len(BASE_ACTION_PROJECTION.columns) * 4 * (256 + 28)
    return windows_per_generation * (frame_bytes + 8)


def _memory_preflight(args: Args) -> dict[str, object]:
    if not 0 < args.max_memory_fraction <= 1:
        raise ValueError("max_memory_fraction must be in (0, 1]")
    if args.replay_slots < args.batch_size * args.phase_block_batches:
        raise ValueError("replay slots must cover the entire phase block")
    if args.replay_slots % args.batch_size or args.batch_size % args.windows_per_generation:
        raise ValueError("batch, replay slots, and windows per generation are misaligned")
    per_slot = _ring_slot_bytes(args.windows_per_generation)
    estimate = estimate_host_memory(
        central_buffer_bytes=args.replay_slots * per_slot,
        decoded_chunk_bytes=MAX_DECODE_CHUNK_ROWS * per_slot,
        replay_workspace_bytes=0,
        pinned_batch_bytes=0,
        pinned_batch_count=0,
        validation_cache_bytes=0,
        compiler_and_process_bytes=0,
        workers=args.num_workers,
    )
    available = _host_available_bytes()
    return {
        "decoded_slot_bytes": per_slot,
        "host_available_bytes_before_start": available,
        "estimated_peak_lower_bound_bytes": estimate.peak_bytes,
        "estimated_components_bytes": asdict(estimate),
        "estimate_excludes": ["worker replay workspaces", "Python processes", "filesystem cache"],
        "max_memory_fraction": args.max_memory_fraction,
        "safe_to_start": estimate.peak_bytes <= available * args.max_memory_fraction,
    }


def _configure_r2(path: Path | None) -> None:
    if path is None:
        return
    settings = configparser.ConfigParser()
    if not settings.read(path) or "r2" not in settings:
        raise ValueError(f"R2 credentials are absent from {path}")
    r2 = settings["r2"]
    os.environ["AWS_ENDPOINT_URL"] = r2["endpoint"]
    os.environ["AWS_ACCESS_KEY_ID"] = r2["access_key_id"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = r2["secret_access_key"]
    os.environ["AWS_DEFAULT_REGION"] = "auto"


def _process_tree_counters() -> dict[str, float | int]:
    cpu_ticks = 0
    read_bytes = 0
    write_bytes = 0
    clock_ticks = os.sysconf("SC_CLK_TCK")
    for pid in process_tree_pids(os.getpid()):
        process = Path("/proc") / str(pid)
        try:
            fields = (process / "stat").read_text().rsplit(")", 1)[1].split()
            cpu_ticks += int(fields[11]) + int(fields[12])
            for line in (process / "io").read_text().splitlines():
                name, _, value = line.partition(":")
                if name == "read_bytes":
                    read_bytes += int(value)
                elif name == "write_bytes":
                    write_bytes += int(value)
        except FileNotFoundError, ProcessLookupError:
            continue
    return {
        "cpu_seconds": cpu_ticks / clock_ticks,
        "disk_read_bytes": read_bytes,
        "disk_write_bytes": write_bytes,
    }


def _process_tree_high_water_rss_bytes() -> int:
    total = 0
    for pid in process_tree_pids(os.getpid()):
        try:
            for line in (Path("/proc") / str(pid) / "status").read_text().splitlines():
                if line.startswith("VmHWM:"):
                    total += int(line.split()[1]) * 1024
                    break
        except FileNotFoundError, ProcessLookupError:
            continue
    return total


def _source_metadata() -> dict[str, object]:
    source_file = Path(__file__).resolve()
    loader_file = Path(inspect.getfile(BufferedMDSReplayLoader)).resolve()
    checkout = loader_file.parents[2]
    source_sha = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "source_git_sha": source_sha,
        "benchmark_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
        "loader_sha256": hashlib.sha256(loader_file.read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "torch_version": importlib.metadata.version("torch"),
        "mosaic_version": importlib.metadata.version("mosaicml-streaming"),
        "cpu_affinity": len(os.sched_getaffinity(0)),
        "cpu_model": next(
            (
                line.partition(":")[2].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        ),
    }


def _emit(result: dict[str, object], output: Path | None) -> None:
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as handle:
            handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        summary = {name: value for name, value in result.items() if name not in ("batch_wait_seconds", "batch_hashes")}
        summary["record"] = str(output)
        print(json.dumps(summary, sort_keys=True), flush=True)
    else:
        print(json.dumps(result, sort_keys=True), flush=True)


def _cache_status(tasks: Sequence[ShardTask], manifests: Mapping[str, SourceManifest]) -> dict[str, int]:
    missing = 0
    missing_bytes = 0
    total_bytes = 0
    for task in tasks:
        manifest = manifests[task.source]
        raw_bytes = manifest.raw_bytes_per_shard[task.shard]
        raw_path = manifest.raw_paths[task.shard]
        total_bytes += raw_bytes
        if not raw_path.is_file() or raw_path.stat().st_size != raw_bytes:
            missing += 1
            missing_bytes += raw_bytes
    return {
        "planned_shards": len(tasks),
        "raw_bytes_total": total_bytes,
        "missing_raw_shards": missing,
        "missing_raw_bytes": missing_bytes,
    }


def measure(
    loader: BufferedMDSReplayLoader[_Batch],
    *,
    warm_batches: int,
    measured_batches: int,
    hash_batches: bool,
) -> dict[str, object]:
    """Measure complete next-batch waits after a declared warmup."""
    if warm_batches < 1 or measured_batches < 1:
        raise ValueError("warm_batches and measured_batches must be positive")
    iterator = iter(loader)
    startup = time.perf_counter()
    next(iterator)
    first_batch_seconds = time.perf_counter() - startup
    for _ in range(warm_batches - 1):
        next(iterator)
    counters_before = _process_tree_counters()
    measurement_start = time.perf_counter()
    durations: list[float] = []
    diversities: list[int] = []
    reuse_gaps: list[int] = []
    last_seen: dict[str, int] = {}
    hashes: list[str] = []
    batch_bytes = 0
    for batch_index in range(measured_batches):
        start = time.perf_counter()
        batch = next(iterator)
        durations.append(time.perf_counter() - start)
        batch_bytes = sum(values.nbytes for values in batch.columns.values())
        diversities.append(len(set(batch.replay_ids)))
        for replay_id in batch.replay_ids:
            if replay_id in last_seen:
                reuse_gaps.append(batch_index - last_seen[replay_id])
            last_seen[replay_id] = batch_index
        if hash_batches:
            hashes.append(_batch_digest(batch))
    measurement_wall_seconds = time.perf_counter() - measurement_start
    counters_after = _process_tree_counters()
    samples = measured_batches * loader.batch_size
    return {
        "protocol": "o59-replay-ring-v2-loader-core",
        "first_batch_seconds": first_batch_seconds,
        "warm_batches": warm_batches,
        "measured_batches": measured_batches,
        "samples": samples,
        "samples_per_second": samples / measurement_wall_seconds,
        "wait_only_samples_per_second": samples / sum(durations),
        "measurement_wall_seconds": measurement_wall_seconds,
        "measurement_cpu_seconds": counters_after["cpu_seconds"] - counters_before["cpu_seconds"],
        "measurement_disk_read_bytes": counters_after["disk_read_bytes"] - counters_before["disk_read_bytes"],
        "measurement_disk_write_bytes": counters_after["disk_write_bytes"] - counters_before["disk_write_bytes"],
        "process_tree_rss_gib_at_end": read_process_tree_memory(os.getpid())["system/process_tree/rss_gib"],
        "process_tree_peak_rss_upper_bound_bytes": _process_tree_high_water_rss_bytes(),
        "batch_wait_seconds_median": statistics.median(durations),
        "batch_wait_seconds_p95": float(np.percentile(durations, 95)),
        "batch_wait_seconds": durations,
        "batch_array_bytes": batch_bytes,
        "distinct_replays_min_per_batch": min(diversities),
        "distinct_replays_total": len(last_seen),
        "reuse_gap_batches_min": min(reuse_gaps) if reuse_gaps else None,
        "reuse_gap_batches_median": statistics.median(reuse_gaps) if reuse_gaps else None,
        "batch_hashes": hashes,
        "loader_metrics": loader.metrics,
    }


def main(args: Args) -> None:
    if sum((args.estimate_only, args.cache_status_only, args.materialize_only)) > 1:
        raise ValueError("estimate-only, cache-status-only, and materialize-only are mutually exclusive")
    if args.local_repo is not None:
        streams.REPO_DIR = str(args.local_repo.resolve())
    _configure_r2(args.rclone_config)
    if len(set(args.sources)) != len(args.sources) or not args.sources:
        raise ValueError("sources must be non-empty and unique")
    unknown = sorted(set(args.sources) - set(streams.POLICY_WORLD_V8_TRAIN_REPLAYS))
    if unknown:
        raise ValueError(f"unknown policy-world-v8 sources: {unknown}")
    selection = PhysicalShardSelection.from_sources(
        tuple(SourceRowSelection(name, streams.POLICY_WORLD_V8_TRAIN_REPLAYS[name]) for name in args.sources)
    )
    if (
        args.sources == tuple(source.name for source in streams.POLICY_WORLD_V8_SOURCES)
        and selection.sha256 != O59_SELECTION_SHA256
    ):
        raise ValueError("059 source selection identity changed")
    manifest_hashes = {name: streams.POLICY_WORLD_V8_TRAIN_MANIFEST_SHA256[name] for name in args.sources}
    preflight = _memory_preflight(args)
    identity = {
        "label": args.label,
        "sources": args.sources,
        "selection_sha256": selection.sha256,
        "source_manifest_sha256": manifest_hashes,
        "batch_size": args.batch_size,
        "replay_slots": args.replay_slots,
        "windows_per_generation": args.windows_per_generation,
        "phase_block_batches": args.phase_block_batches,
        "num_workers": args.num_workers,
        "materialization_threads": args.materialization_threads,
        "seed": args.seed,
        "memory_preflight": preflight,
        **_source_metadata(),
    }
    if args.estimate_only:
        _emit(identity, args.output)
        return
    if not preflight["safe_to_start"]:
        estimated_bytes = cast(int, preflight["estimated_peak_lower_bound_bytes"])
        available_bytes = cast(int, preflight["host_available_bytes_before_start"])
        raise RuntimeError(
            "loader memory lower bound exceeds the configured safety fraction of available RAM; "
            f"estimated {estimated_bytes / 2**30:.2f} GiB, "
            f"available {available_bytes / 2**30:.2f} GiB"
        )
    row_counts = {name: streams.POLICY_WORLD_V8_TRAIN_REPLAYS[name] for name in args.sources}
    startup_start = time.perf_counter()
    adapter = MDSStorageAdapter(selection)
    try:
        adapter.validate_manifests(
            expected_sha256=manifest_hashes,
            expected_index_version=2,
            expected_schema_sha256=O59_SCHEMA_SHA256,
            expected_rows=row_counts,
        )
        tasks = build_shard_plan(selection, adapter.manifests)
        if args.cache_status_only:
            _emit({**identity, "cache_status": _cache_status(tasks, adapter.manifests)}, args.output)
            return
        if args.materialize_only:
            before = _cache_status(tasks, adapter.manifests)
            materialization_start = time.perf_counter()
            for task in tasks:
                manifest = adapter.manifests[task.source]
                raw_path = manifest.raw_paths[task.shard]
                raw_bytes = manifest.raw_bytes_per_shard[task.shard]
                if not raw_path.is_file() or raw_path.stat().st_size != raw_bytes:
                    adapter.prepare_shard(task)
            after = _cache_status(tasks, adapter.manifests)
            if after["missing_raw_shards"]:
                raise RuntimeError("loader cache materialization left incomplete raw shards")
            _emit(
                {
                    **identity,
                    "cache_status_before": before,
                    "cache_status_after": after,
                    "materialization_seconds": time.perf_counter() - materialization_start,
                },
                args.output,
            )
            return
        loader = BufferedMDSReplayLoader[_Batch](
            selection=selection,
            adapter=adapter,
            tasks=tasks,
            data_protocol="o59-replay-ring-v2",
            source_manifest_sha256=manifest_hashes,
            labels=_no_labels,
            projection=BASE_ACTION_PROJECTION,
            batch_transform=_keep_columns,
            batch_size=args.batch_size,
            replay_slots=args.replay_slots,
            seed=args.seed,
            num_workers=args.num_workers,
            context_length=256,
            chunk_length=28,
            windows_per_generation=args.windows_per_generation,
            replay_phase_block_batches=args.phase_block_batches,
            schema_version=7,
            reserved_disk_bytes=0,
            pin_memory=False,
            materialization_threads=args.materialization_threads,
        )
        try:
            setup_seconds = time.perf_counter() - startup_start
            result = measure(
                loader,
                warm_batches=args.warm_batches,
                measured_batches=args.measured_batches,
                hash_batches=args.hash_batches,
            )
            result["setup_seconds"] = setup_seconds
            result["startup_seconds_total"] = setup_seconds + cast(float, result["first_batch_seconds"])
            result["host_available_bytes_after_measurement"] = _host_available_bytes()
            _emit({**identity, **result}, args.output)
        finally:
            loader.close()
    finally:
        adapter.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
