"""Measure complete native stepping, sharded threads, and compressed restore."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import statistics
import struct
import time
import zlib
from pathlib import Path

import numpy as np

from hal.sim import native
from hal.sim import native_shard


def measure_threads(data_dir: Path, cpus: tuple[int, ...], lanes: int, frames: int) -> dict[str, object]:
    if not cpus or len(set(cpus)) != len(cpus) or lanes < 1 or frames < 1:
        raise ValueError("distinct CPUs and positive lane/frame counts are required")
    match = native.NativeMatch(25, (native.NativePlayer(1, 1), native.NativePlayer(2, 22)), 17)
    mask = np.ones(lanes, dtype=np.bool_)
    actions = np.zeros((lanes, 2, 14), dtype=np.float32)
    actions[:, 0, 0] = 0.5
    actions[:, 1, 0] = -0.5
    timings = []
    digests = []
    with contextlib.ExitStack() as stack:
        shards = [stack.enter_context(native_shard.NativeShard(data_dir, lanes, cpu=cpu)) for cpu in cpus]
        for parallel in (False, True):
            for shard in shards:
                shard.reset((match,) * lanes, mask).result()
            started = time.perf_counter()
            for _ in range(frames):
                if parallel:
                    futures = [shard.step(actions, mask) for shard in shards]
                    for future in futures:
                        future.result()
                else:
                    for shard in shards:
                        shard.step(actions, mask).result()
            timings.append(time.perf_counter() - started)
            # The same workload must finish at identical states in both schedules.
            digests.append([hashlib.sha256(shard.save(0).result()).hexdigest() for shard in shards])
    if digests[0] != digests[1]:
        raise AssertionError("parallel stepping changed simulator state")
    return {
        "cpus": cpus,
        "lanes_per_shard": lanes,
        "frames": frames,
        "serial_seconds": timings[0],
        "parallel_seconds": timings[1],
        "serial_match_frames_per_second": frames * lanes * len(cpus) / timings[0],
        "parallel_match_frames_per_second": frames * lanes * len(cpus) / timings[1],
        "state_sha256": digests[0],
    }


def measure_restore(data_dir: Path, lanes: int) -> dict[str, object]:
    if lanes < 1:
        raise ValueError("lanes must be positive")
    mask = np.ones(lanes, dtype=np.bool_)
    actions = np.zeros((lanes, 2, 14), dtype=np.float32)
    match = native.NativeMatch(25, (native.NativePlayer(1, 1), native.NativePlayer(2, 22)), 17)
    with native.NativeRolloutBatch(data_dir, lanes) as batch:
        batch.reset((match,) * lanes, mask)
        for _ in range(150):
            batch.step(actions, mask)
        started = time.perf_counter()
        saved = tuple(batch.save(lane) for lane in range(lanes))
        save_seconds = time.perf_counter() - started
        before = []
        after = []
        for _ in range(3):
            started = time.perf_counter()
            for lane, state in enumerate(saved):
                batch.restore(lane, state)
            before.append(time.perf_counter() - started)
            started = time.perf_counter()
            batch.restore_many(tuple(range(lanes)), saved)
            after.append(time.perf_counter() - started)
        metadata_size = struct.unpack_from("<I", saved[0], 8)[0]
        raw_size = len(zlib.decompress(saved[0][12 + metadata_size :]))
        return {
            "lanes": lanes,
            "native_bytes_per_match": raw_size,
            "compressed_hal_bytes_per_match": len(saved[0]),
            "save_seconds": save_seconds,
            "lane_restore_seconds": statistics.median(before),
            "batch_restore_seconds": statistics.median(after),
            "library_sha256": batch.library_sha256,
            "data_manifest_sha256": batch.manifest_sha256,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cpus", type=int, nargs="+", required=True)
    parser.add_argument("--lanes", type=int, default=32)
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--restore-lanes", type=int, default=128)
    args = parser.parse_args()
    result = {
        "host": platform.platform(),
        "allowed_cpus": sorted(os.sched_getaffinity(0)),
        "threads": [
            measure_threads(args.data_dir, tuple(args.cpus[:count]), args.lanes, args.frames)
            for count in (1, 2, 4, 6)
            if count <= len(args.cpus)
        ],
        "restore": measure_restore(args.data_dir, args.restore_lanes),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
