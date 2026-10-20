"""Measure O65's native simulator floor on one pinned CPU, without Torch."""

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from hal.sim import native
from hal.sim import native_bridge
from hal.sim import native_shard
from hal.wire import ACTION_DIM


def benchmark(data_dir: Path, cpu: int, *, matches: int = 64, frames: int = 256) -> dict[str, object]:
    if matches < 1 or frames < 1:
        raise ValueError("matches and frames must be positive")
    # O65 pilot: Final Destination, Fox/Falco, four stocks and sampled starting percents.
    configurations = tuple(
        native.NativeMatch(
            25,
            (
                native.NativePlayer(1, 1, start_percent=(lane % 4) * 30),
                native.NativePlayer(2, 22, start_percent=((lane + 1) % 4) * 30),
            ),
            65 + lane,
            max_frame=28800,
            stocks=4,
        )
        for lane in range(matches)
    )
    active = np.ones(matches, dtype=np.bool_)
    actions = np.zeros((matches, 2, ACTION_DIM), dtype=np.float32)
    actions[:, 0, 0] = 0.5
    actions[:, 1, 0] = -0.5
    simulator = 0.0
    with native_shard.NativeShard(data_dir, matches, cpu=cpu) as shard:
        shard.reset(configurations, active).result()
        for _ in range(256):
            warm = shard.step(actions, active).result()
            if (warm.terminated | warm.truncated).any():
                raise ValueError("benchmark ended during warmup")
        start_frames = warm.frame_id.copy().tolist()
        started = time.perf_counter()
        for _ in range(frames):
            result = shard.step(actions, active).result()
            simulator += shard.last_step_seconds
            if (result.terminated | result.truncated).any():
                raise ValueError("benchmark ended before its configured frame count")
        wall = time.perf_counter() - started
        end_frames = result.frame_id.copy().tolist()
        states = shard.save_many(tuple(range(matches))).result()
        simulator_hash = shard.library_sha256
    if "torch" in sys.modules:
        raise RuntimeError("simulator benchmark imported Torch")
    return {
        "format": "hal.o65.simulator-floor.v1",
        "host": platform.platform(),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip(),
        "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "adapter_sha256": native_bridge.library_sha256(),
        "cpu": cpu,
        "matches": matches,
        "frames": frames,
        "warmup_frames": 256,
        "start_frame_ids": start_frames,
        "end_frame_ids": end_frames,
        "settings": [dataclasses.asdict(config) for config in configurations],
        "actions": "fixed inward half-stick; no policy inference",
        "wall_seconds": wall,
        "simulator_seconds": simulator,
        "dispatch_seconds": wall - simulator,
        "match_frames_per_second": matches * frames / wall,
        "library_sha256": simulator_hash,
        "manifest_sha256": hashlib.sha256((data_dir / "raw/manifest.json").read_bytes()).hexdigest(),
        "state_sha256": hashlib.sha256(b"".join(states)).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=min(os.sched_getaffinity(0)))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark(args.data_dir, args.cpu)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("wall_seconds", "simulator_seconds", "match_frames_per_second")}))


if __name__ == "__main__":
    main()
