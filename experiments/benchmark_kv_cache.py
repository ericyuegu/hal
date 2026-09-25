"""Replay fixed observations to measure O59 batch-size-one inference on CUDA."""

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import POLICY_BUTTON_MASK
from hal.controller import ControllerAction
from hal.data.extract import extract_replay
from hal.inference.api import PolicyInput
from hal.inference.api import RuntimeConfig
from hal.inference.o59 import load_o59_policy
from hal.wire import ACTION_CHANNELS
from hal.wire import BUTTON_BITS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentiles(values: list[float]) -> dict[str, float]:
    return dict(zip(("p50", "p95", "p99"), map(float, np.percentile(values, (50, 95, 99))), strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--history-mode", choices=("window", "kv_cache"), default="kv_cache")
    parser.add_argument("--update-frames", type=int, choices=(1, 2), default=2)
    parser.add_argument("--compiled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16-window-linears", action="store_true")
    parser.add_argument("--frames", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=1001)
    args = parser.parse_args()
    if args.frames < 600:
        raise ValueError("benchmark requires at least 600 frames, including 300 warmup frames")
    torch.set_num_threads(1)
    rows = extract_replay(str(args.replay))
    if rows is None or len(rows["stage"]) < args.frames:
        raise ValueError("replay does not contain enough usable frames")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    policy = load_o59_policy(
        args.bundle,
        device="cuda",
        seed=args.seed,
        compiled=args.compiled,
        history_mode=args.history_mode,
        kv_update_frames=args.update_frames,
        kv_cuda_graphs=args.cuda_graphs,
    )
    if args.bf16_window_linears:
        for module in policy.model.modules():
            if isinstance(module, torch.nn.Linear):
                module.to(dtype=torch.bfloat16)
    policy.prepare(RuntimeConfig(1, (2,)))
    prepare_seconds = time.perf_counter() - started
    inputs = []
    for index in range(args.frames):
        action = ControllerAction(
            *(float(rows["p1_" + name][index]) for name in ACTION_CHANNELS[:6]),
            sum(
                BUTTON_BITS[name.removeprefix("button_")]
                for name in ACTION_CHANNELS[6:]
                if rows["p1_" + name][index] > 0.5
            )
            & POLICY_BUTTON_MASK,
        )
        observation = {name: rows[name][index].item() for name in policy.spec.required_observation_fields}
        inputs.append(
            PolicyInput(
                0,
                index,
                1,
                observation,
                action,
                (NEUTRAL_CONTROLLER_ACTION,) * 2,
                player_identity="IBDW#0",
                reset=index == 0,
            )
        )
    times = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.compiler.set_stance("fail_on_recompile"):
        for item in inputs:
            started = time.perf_counter()
            policy.step((item,))
            torch.cuda.synchronize()
            times.append(1000 * (time.perf_counter() - started))
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as profile:
            for index in range(20):
                policy.step((replace(inputs[-1], frame_id=args.frames + index),))
            torch.cuda.synchronize()
    profile.export_chrome_trace(str(args.output / "trace.json"))
    (args.output / "profile.txt").write_text(
        profile.key_averages().table(sort_by="self_cuda_time_total", row_limit=50)
    )
    result = {
        "config": {
            **vars(args),
            "batch_size": 1,
            "transport_frames": 2,
            "replan_frames": 2,
            "temperature": 1.0,
            "desired_return": 20.0,
            "identity": "IBDW#0",
        },
        "prepare_seconds": prepare_seconds,
        "cold_ms": times[0],
        "frame_ms": percentiles(times[300:]),
        "replan_ms": percentiles(times[300::2]),
        "queued_frame_ms": percentiles(times[301::2]),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "bundle_sha256": sha256(args.bundle),
        "replay_sha256": sha256(args.replay),
        "source_sha256": {
            str(path): sha256(path)
            for path in (
                Path(__file__),
                *sorted(Path("hal/inference").glob("*.py")),
            )
        },
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
