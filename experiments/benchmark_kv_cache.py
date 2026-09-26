"""Replay fixed observations to measure O59 batch-size-one inference on CUDA."""

import argparse
import gc
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
from hal.inference.api import PredictionRequest
from hal.inference.api import RuntimeConfig
from hal.inference.backends.history_decoder.policy import load_o59_policy
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
    parser.add_argument("--update-frames", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--compiled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16-window-linears", action="store_true")
    parser.add_argument("--frames", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--prediction-horizon", type=int, default=4)
    parser.add_argument("--fixed-prefix", type=int, default=2)
    parser.add_argument("--replan-interval", type=int, default=2)
    args = parser.parse_args()
    if args.replan_interval < 1 or args.fixed_prefix < 2 or args.prediction_horizon <= args.fixed_prefix:
        raise ValueError("invalid prediction workload")
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
    policy.prepare_prediction(RuntimeConfig(1, (2,)), args.prediction_horizon, args.fixed_prefix)
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
    gc.collect()
    gc.freeze()
    times = []
    source_frames = []
    sequence = 0
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.compiler.set_stance("fail_on_recompile"):
        for start in range(0, len(inputs), args.replan_interval):
            observations = tuple(inputs[start : start + args.replan_interval])
            request = PredictionRequest(
                0,
                1,
                sequence,
                observations[-1].frame_id,
                observations,
                (NEUTRAL_CONTROLLER_ACTION,) * args.fixed_prefix,
            )
            started = time.perf_counter()
            policy.predict((request,))
            torch.cuda.synchronize()
            times.append(1000 * (time.perf_counter() - started))
            source_frames.append(request.source_frame)
            sequence += 1
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as profile:
            for index in range(20):
                observations = tuple(
                    replace(inputs[-1], frame_id=args.frames + index * args.replan_interval + offset, reset=False)
                    for offset in range(args.replan_interval)
                )
                request = PredictionRequest(
                    0,
                    1,
                    sequence,
                    observations[-1].frame_id,
                    observations,
                    (NEUTRAL_CONTROLLER_ACTION,) * args.fixed_prefix,
                )
                policy.predict((request,))
                sequence += 1
            torch.cuda.synchronize()
    profile.export_chrome_trace(str(args.output / "trace.json"))
    (args.output / "profile.txt").write_text(
        profile.key_averages().table(sort_by="self_cuda_time_total", row_limit=50)
    )
    result = {
        "schema_version": 2,
        "gc_frozen": True,
        "config": {
            **vars(args),
            "batch_size": 1,
            "input_delay_frames": 2,
            "request_observations": "incremental",
            "temperature": 1.0,
            "desired_return": 20.0,
            "identity": "IBDW#0",
        },
        "prepare_seconds": prepare_seconds,
        "cold_ms": times[0],
        "source_frames": source_frames,
        "prediction_samples_ms": times,
        "prediction_ms": percentiles(
            [value for frame, value in zip(source_frames, times, strict=True) if frame >= 300]
        ),
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
                *sorted(Path("hal/inference").rglob("*.py")),
            )
        },
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
