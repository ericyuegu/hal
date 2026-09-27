"""Replay fixed observations to measure O59 batch-size-one inference on CUDA."""

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import action_vec_to_controller
from hal.data.extract import extract_replay
from hal.inference.action_sequence_artifact import REQUIRED_OBSERVATION_FIELDS
from hal.inference.action_sequence_artifact import load_action_sequence_policy
from hal.inference.api import PolicyInput
from hal.inference.api import PredictionRequest
from hal.inference.api import RuntimeConfig
from hal.inference.benchmark import measure_process_ready_pair
from hal.inference.engine import configure_inference_process
from hal.inference.engine import freeze_inference_runtime
from hal.training.runs import source_git_sha
from hal.wire import ACTION_CHANNELS


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


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
    parser.add_argument("--frames", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--prediction-horizon", type=int, default=4)
    parser.add_argument("--fixed-prefix", type=int, default=2)
    parser.add_argument("--replan-interval", type=int, default=2)
    parser.add_argument("--physical-delay", type=int, choices=(0, 2, 3), default=2)
    parser.add_argument("--process-batching", action="store_true")
    parser.add_argument("--capacity", type=int, default=2)
    parser.add_argument("--serial-requests", action="store_true")
    parser.add_argument("--warmup-calls", type=int, default=20)
    parser.add_argument("--measured-calls", type=int, default=200)
    args = parser.parse_args()
    if (
        args.replan_interval < 1
        or args.fixed_prefix < args.physical_delay
        or args.prediction_horizon <= args.fixed_prefix
    ):
        raise ValueError("invalid prediction workload")
    if not args.process_batching and args.frames < 600:
        raise ValueError("benchmark requires at least 600 frames, including 300 warmup frames")
    if args.process_batching and (
        args.history_mode != "kv_cache"
        or args.update_frames != 4
        or args.prediction_horizon != 8
        or args.fixed_prefix != 3
        or args.replan_interval != 4
        or args.physical_delay != 2
        or not args.compiled
        or not args.cuda_graphs
        or args.seed != 1001
    ):
        raise ValueError("process batching measures the declared H8/prefix3/delay2/Q4 CUDA-graph profile")
    configure_inference_process()
    rows = extract_replay(str(args.replay))
    needed_frames = 4 * (args.warmup_calls + args.measured_calls) if args.process_batching else args.frames
    if rows is None or len(rows["stage"]) < needed_frames:
        raise ValueError("replay does not contain enough usable frames")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.process_batching:
        inputs = []
        for index in range(needed_frames):
            action = action_vec_to_controller(
                np.asarray([rows["p1_" + name][index] for name in ACTION_CHANNELS], dtype=np.float32)
            )
            observation = {name: rows[name][index].item() for name in REQUIRED_OBSERVATION_FIELDS}
            inputs.append(PolicyInput(0, index, 1, observation, action, player_identity="IBDW#0"))
        measurement = measure_process_ready_pair(
            args.bundle,
            inputs,
            capacity=args.capacity,
            concurrent=not args.serial_requests,
            warmup_calls=args.warmup_calls,
            measured_calls=args.measured_calls,
        )
        result = {
            **measurement,
            "config": vars(args),
            "bundle_sha256": sha256(args.bundle),
            "replay_sha256": sha256(args.replay),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "source_sha256": {
                str(path): sha256(path)
                for path in (
                    Path(__file__),
                    *sorted(Path("hal/inference").rglob("*.py")),
                )
            },
        }
        (args.output / "results.json").write_text(json.dumps(result, indent=2, default=str))
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "capacity",
                        "concurrent",
                        "preparation_seconds",
                        "startup_seconds",
                        "peak_allocated_mib",
                        "complete_pair_ms",
                        "engine_batch_calls",
                        "engine_batch_items",
                        "engine_max_batch_items",
                    )
                },
                indent=2,
            ),
            flush=True,
        )
        return
    started = time.perf_counter()
    policy = load_action_sequence_policy(
        args.bundle,
        device="cuda",
        seed=args.seed,
        compiled=args.compiled,
        history_mode=args.history_mode,
        kv_update_frames=args.update_frames,
        kv_cuda_graphs=args.cuda_graphs,
    )
    policy.prepare_prediction(
        RuntimeConfig(1, (args.physical_delay,), replan_interval_frames=args.replan_interval),
        args.prediction_horizon,
        args.fixed_prefix,
    )
    prepare_seconds = time.perf_counter() - started
    inputs = []
    for index in range(args.frames):
        action = action_vec_to_controller(
            np.asarray([rows["p1_" + name][index] for name in ACTION_CHANNELS], dtype=np.float32)
        )
        observation = {name: rows[name][index].item() for name in policy.spec.required_observation_fields}
        inputs.append(
            PolicyInput(
                0,
                index,
                1,
                observation,
                action,
                player_identity="IBDW#0",
                reset=index == 0,
            )
        )
    freeze_inference_runtime()
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
            "physical_delay_frames": args.physical_delay,
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
        "git_sha": source_git_sha(),
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
    print(
        json.dumps(
            {key: result[key] for key in ("prepare_seconds", "cold_ms", "prediction_ms", "peak_allocated_mib", "gpu")},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
