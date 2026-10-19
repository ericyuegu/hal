"""Train a self-play actor with decision-grid rewards and KL-regularized PPO.

Flatten each frame's factors in decode order: C-stick, main stick, triggers,
buttons. Pad mixed cadences to four frames. Mask execution at match ends,
retain live rollout tails, and compute GAE from every decision's value.
"""

import contextlib
import copy
import dataclasses
import hashlib
import json
import math
import os
import random
import struct
import subprocess
import threading
import time
import zlib
from collections import defaultdict
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal
from typing import Protocol
from typing import Self
from typing import TypedDict
from typing import cast
from typing import overload

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from hal.data.feature_stats import FeatureStats
from hal.data.schema import MDS_PER_FRAME_DTYPES
from hal.data.schema import Rank
from hal.eval.periodic_self_play import PeriodicEvaluator
from hal.eval.self_play_checkpoint import EVALUATION_FORMAT
from hal.eval.self_play_checkpoint import PROXY_SHA256
from hal.eval.self_play_checkpoint import TRAINING_FORMAT
from hal.eval.self_play_checkpoint import TRAINING_VERSION
from hal.eval.self_play_checkpoint import evaluation_actor_path
from hal.eval.self_play_checkpoint import load_evaluation_checkpoint
from hal.eval.self_play_checkpoint import read_training_manifest
from hal.eval.self_play_checkpoint import save_evaluation_checkpoint
from hal.eval.self_play_checkpoint import sha256
from hal.inference.column_context import ColumnContextBuilder
from hal.inference.column_context import PackedColumnContextBuilder
from hal.inference.cuda_graph import CapturedCall
from hal.inference.cuda_graph import prepared_shape_compilation
from hal.inference.decision_decoder import DecisionDecoder
from hal.inference.graphed_actor import GraphedActorStep
from hal.inference.kv_cache import BlockRestartCache
from hal.inference.kv_cache import forward_tokens_with_kv_cache
from hal.models.action_sequence import ActionSequenceConfig
from hal.models.attention import KVMemory
from hal.models.conditioning import ConditioningRanges
from hal.models.controller_codec import CONTROLLER_GROUP_NAMES
from hal.models.replay_actor import ActorLoss
from hal.models.replay_actor import ActorTargets
from hal.models.replay_actor import actor_loss
from hal.models.self_play import DecoderPrecision
from hal.models.self_play import SelfPlayTransformer
from hal.models.self_play import validate_decoder_precision
from hal.models.window_replay import ReplayOutput
from hal.models.window_replay import ReplaySlots
from hal.models.window_replay import replay_windows
from hal.representation.player_identity import vocabulary_from_checkpoint_buffer
from hal.sim import native
from hal.sim import native_bridge
from hal.sim.inputs import native_controller_readback
from hal.sim.native_shard import NativeShard
from hal.training.muon import SingleDeviceMuonWithAuxAdam
from hal.wire import ACTION_DIM

FACTOR_ORDER = ("c_stick", "main_stick", "triggers", "buttons")
MAX_REPLAN = 4
GAMMA = 0.99855
GAE_LAMBDA = 0.99


class TraceEvent(TypedDict, total=False):
    cat: str
    name: str
    ts: float
    dur: float
    args: dict[str, str | int | float]


@dataclasses.dataclass(frozen=True, slots=True)
class ProfileConfig:
    warm_updates: int = 4

    def __post_init__(self) -> None:
        if type(self.warm_updates) is not int or self.warm_updates < 1:
            raise ValueError("profiling requires at least one warm update")


@dataclasses.dataclass(frozen=True, slots=True)
class ReplayChunks:
    """Chunk sizes for no-grad full-window scoring; actor shapes stay fixed."""

    windows: int = 16
    decisions: int = 128

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 1 for value in (self.windows, self.decisions)):
            raise ValueError("replay chunk sizes must be positive integers")


DEFAULT_REPLAY_CHUNKS = ReplayChunks()


def interval_union(intervals: list[tuple[float, float]]) -> float:
    end = total = 0.0
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def summarize_trace(path: Path) -> dict[str, object]:
    """Count device activities, not CPU operators that launch those activities."""
    events = cast(list[TraceEvent], json.loads(path.read_text())["traceEvents"])
    roots = [
        (event["ts"], event["ts"] + event["dur"])
        for event in events
        if event.get("cat") == "user_annotation" and event.get("name", "").startswith("o65_profile/")
    ]
    markers = [
        (event["ts"], event["ts"] + event["dur"])
        for event in events
        if event.get("cat") == "user_annotation" and event.get("name") == "o65_profile_sync"
    ]
    if len(roots) != 1:
        raise ValueError("trace requires exactly one O65 profile annotation")
    root_start, root_stop = roots[0]
    kernels: dict[str, float] = defaultdict(float)
    copies: dict[str, int] = defaultdict(int)
    syncs: dict[str, int] = defaultdict(int)
    measurement_syncs: dict[str, int] = defaultdict(int)
    intervals: list[tuple[float, float]] = []
    kernel_count = launches = graph_launches = 0
    launch_seconds = 0.0
    for event in events:
        category = event.get("cat", "")
        name = event.get("name", "")
        duration = event.get("dur", 0.0)
        timestamp = event.get("ts", -1.0)
        in_root = root_start <= timestamp <= root_stop
        if category == "cuda_runtime" and "Synchronize" in name:
            measurement = not in_root or any(start <= timestamp <= stop for start, stop in markers)
            if measurement:
                measurement_syncs[name] += 1
            else:
                syncs[name] += 1
        if not in_root:
            continue
        if category in ("kernel", "gpu_memcpy", "gpu_memset"):
            start = event["ts"]
            intervals.append((start, start + duration))
        if category == "kernel":
            kernels[name] += duration
            kernel_count += 1
        if category == "cuda_runtime" and ("LaunchKernel" in name or "GraphLaunch" in name):
            launches += 1
            graph_launches += int("GraphLaunch" in name)
            launch_seconds += duration / 1e6
        if category == "gpu_memcpy":
            if "HtoD" in name:
                direction = "h2d"
            elif "DtoH" in name:
                direction = "d2h"
            else:
                direction = "other"
            args = event.get("args", {})
            if "bytes" not in args:
                raise ValueError(f"copy event has no byte count: {name}")
            copies[f"{direction}_count"] += 1
            copies[f"{direction}_bytes"] += int(args["bytes"])
    if not kernel_count:
        raise ValueError("profile captured no CUDA kernels; check CUPTI availability")
    total = sum(kernels.values())
    top = sorted(kernels.items(), key=lambda item: item[1], reverse=True)[:15]
    return {
        "device_busy_seconds": interval_union(intervals) / 1e6,
        "kernel_seconds": total / 1e6,
        "kernel_count": kernel_count,
        "runtime_launch_count": launches,
        "runtime_eager_launch_count": launches - graph_launches,
        "runtime_graph_launch_count": graph_launches,
        "runtime_launch_seconds": launch_seconds,
        "kernel_seconds_by_name": {name: value / 1e6 for name, value in kernels.items()},
        "top15_kernel_share": sum(value for _, value in top) / total if total else 0.0,
        "top15_kernels": [{"name": name, "seconds": value / 1e6} for name, value in top],
        "copies": dict(copies),
        "synchronizations": dict(syncs),
        "measurement_synchronizations_excluded": dict(measurement_syncs),
    }


class UpdateProfile:
    """Measure one update, then trace bounded samples in a separate update.

    CUDA event spans contain idle gaps. Only trace interval unions measure busy
    time. Extrapolated samples remain estimates and are stored separately.
    """

    def __init__(self, output: Path, device: torch.device, *, traces: bool) -> None:
        self.output = output
        self.device = device
        self.traces = traces
        self.phase_name = "outside"
        self.active_trace = False
        self.phases: dict[str, dict[str, float]] = {}
        self.calls: dict[str, int] = defaultdict(int)
        self.work: dict[str, float] = defaultdict(float)
        self.samples: list[dict[str, object]] = []
        self.host_seconds: dict[str, float] = defaultdict(float)

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        previous = self.phase_name
        self.phase_name = name
        torch.cuda.synchronize(self.device)
        start_event = torch.cuda.Event(enable_timing=True)
        stop_event = torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        cpu_started = time.thread_time()
        started_unix = time.time()
        start_event.record()
        try:
            yield
        finally:
            stop_event.record()
            stop_event.synchronize()
            self.phases[name] = {
                "started_unix": started_unix,
                "finished_unix": time.time(),
                "wall_seconds": time.perf_counter() - started,
                "main_thread_cpu_seconds": time.thread_time() - cpu_started,
                "cuda_span_seconds": start_event.elapsed_time(stop_event) / 1000,
            }
            self.phase_name = previous

    @contextlib.contextmanager
    def sample(self, name: str, *, work: float = 0.0, limit: int = 1, skip: int = 0) -> Iterator[None]:
        started = time.perf_counter()
        cpu_started = time.thread_time()
        key = f"{self.phase_name}/{name}"
        try:
            with self._sample(name, work=work, limit=limit, skip=skip):
                yield
        finally:
            self.host_seconds[key] += time.perf_counter() - started
            self.host_seconds[f"{key}/main_thread_cpu"] += time.thread_time() - cpu_started

    @contextlib.contextmanager
    def _sample(self, name: str, *, work: float, limit: int, skip: int) -> Iterator[None]:
        key = f"{self.phase_name}/{name}"
        index = self.calls[key]
        self.calls[key] += 1
        self.work[key] += work
        if not self.traces or self.active_trace or not skip <= index < skip + limit:
            yield
            return
        self.active_trace = True
        try:
            torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            with (
                torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                ) as profiler,
                torch.profiler.record_function(f"o65_profile/{key}"),
            ):
                with torch.profiler.record_function("o65_profile_sync"):
                    torch.cuda.synchronize(self.device)
                yield
                with torch.profiler.record_function("o65_profile_sync"):
                    torch.cuda.synchronize(self.device)
            path = self.output / f"{key.replace('/', '-')}-{index}.json"
            profiler.export_chrome_trace(str(path))
            self.samples.append(
                {
                    "site": key,
                    "index": index,
                    "work": work,
                    "instrumented_seconds_including_export": time.perf_counter() - started,
                    "trace": path.name,
                    **summarize_trace(path),
                }
            )
        finally:
            self.active_trace = False

    def write(self, metrics: dict[str, float]) -> None:
        name = "traced" if self.traces else "timed"
        payload = {
            "format": "hal.o65.gpu-profile.v1",
            "mode": name,
            "phases": self.phases,
            "calls": dict(self.calls),
            "work": dict(self.work),
            "site_host_seconds": dict(self.host_seconds),
            "samples": self.samples,
            "metrics": metrics,
        }
        (self.output / f"{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


def estimate_profile(timed: UpdateProfile, traced: UpdateProfile) -> dict[str, object]:
    """Scale bounded samples by timed work; preserve the coverage limitation."""
    estimates: dict[str, dict[str, object]] = {}
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in traced.samples:
        grouped[cast(str, row["site"])].append(row)
    for site, rows in grouped.items():
        sample_work = sum(cast(float, row["work"]) for row in rows)
        if sample_work <= 0 or site not in timed.work:
            raise ValueError(f"missing timed work for trace site {site}")
        scale = timed.work[site] / sample_work
        call_scale = timed.calls[site] / len(rows)
        kernels: dict[str, float] = defaultdict(float)
        copies: dict[str, float] = defaultdict(float)
        syncs: dict[str, float] = defaultdict(float)
        for row in rows:
            for name, duration in cast(dict[str, float], row["kernel_seconds_by_name"]).items():
                kernels[name] += duration * scale
            for name, count in cast(dict[str, int], row["copies"]).items():
                copies[name] += count * (call_scale if name.endswith("_count") else scale)
            for name, count in cast(dict[str, int], row["synchronizations"]).items():
                syncs[name] += count * call_scale
        total = sum(kernels.values())
        top = sorted(kernels.items(), key=lambda item: item[1], reverse=True)[:15]
        estimates[site] = {
            "scale": scale,
            "call_scale": call_scale,
            "timed_calls": timed.calls[site],
            "sampled_calls": len(rows),
            "device_busy_seconds": sum(cast(float, row["device_busy_seconds"]) for row in rows) * scale,
            "kernel_seconds": total,
            "kernel_count": sum(cast(int, row["kernel_count"]) for row in rows) * call_scale,
            "runtime_launch_count": sum(cast(int, row["runtime_launch_count"]) for row in rows) * call_scale,
            "runtime_eager_launch_count": sum(
                cast(int, row.get("runtime_eager_launch_count", row["runtime_launch_count"])) for row in rows
            )
            * call_scale,
            "runtime_graph_launch_count": sum(cast(int, row.get("runtime_graph_launch_count", 0)) for row in rows)
            * call_scale,
            "runtime_launch_seconds": sum(cast(float, row["runtime_launch_seconds"]) for row in rows) * call_scale,
            "copies": dict(copies),
            "synchronizations_excluding_injected": dict(syncs),
            "top15_kernel_share": sum(value for _, value in top) / total if total else 0,
            "top15_kernels": [{"name": name, "seconds": value} for name, value in top],
        }
    flops: dict[str, float] = {}
    for name in timed.phases:
        forward = 2 * (
            9_400_000 * timed.work.get(f"{name}/trunk_tokens", 0)
            + 6_700_000 * timed.work.get(f"{name}/decoder_tokens", 0)
        )
        flops[name] = forward * (3 if name == "actor" else 1)
    return {
        "format": "hal.o65.gpu-profile-estimates.v1",
        "sites": estimates,
        "coverage": "sampled sites only; gaps between chunks and rollout setup are untraced",
        "flops": flops,
        "tflops_per_wall_second": {
            name: value / timed.phases[name]["wall_seconds"] / 1e12 for name, value in flops.items()
        },
    }


@contextlib.contextmanager
def phase(profile: UpdateProfile | None, name: str) -> Iterator[None]:
    with profile.phase(name) if profile is not None else contextlib.nullcontext():
        yield


@contextlib.contextmanager
def sample(
    profile: UpdateProfile | None,
    name: str,
    *,
    work: float = 0.0,
    limit: int = 1,
    skip: int = 0,
) -> Iterator[None]:
    with profile.sample(name, work=work, limit=limit, skip=skip) if profile is not None else contextlib.nullcontext():
        yield


class HostSampler:
    """Sample per-core and per-thread CPU ticks and GPU use during timing only."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="o65-host-sampler")
        self.failure: OSError | subprocess.SubprocessError | None = None

    def _run(self) -> None:
        try:
            with (self.output / "host.jsonl").open("x") as log:
                while not self.stop.is_set():
                    threads: dict[str, list[int]] = {}
                    thread_affinity: dict[str, list[int]] = {}
                    for path in Path(f"/proc/{os.getpid()}/task").glob("*/stat"):
                        try:
                            fields = path.read_text().rsplit(")", 1)[1].split()
                            affinity = sorted(os.sched_getaffinity(int(path.parent.name)))
                        except FileNotFoundError, ProcessLookupError:
                            continue
                        threads[path.parent.name] = [int(fields[11]), int(fields[12])]
                        thread_affinity[path.parent.name] = affinity
                    cores = [line for line in Path("/proc/stat").read_text().splitlines() if line.startswith("cpu")]
                    gpu = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,utilization.memory,power.draw,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        timeout=10,
                    ).strip()
                    log.write(
                        json.dumps(
                            {
                                "time": time.time(),
                                "pid": os.getpid(),
                                "threads": threads,
                                "thread_affinity": thread_affinity,
                                "cores": cores,
                                "gpu": gpu,
                            }
                        )
                        + "\n"
                    )
                    log.flush()
                    self.stop.wait(0.5)
        except (OSError, subprocess.SubprocessError) as error:
            self.failure = error

    def __enter__(self) -> Self:
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join()
        if self.failure is not None:
            raise RuntimeError("host sampling failed") from self.failure


@dataclasses.dataclass(frozen=True, slots=True)
class DelayReplan:
    delay: int
    replan: int

    def __post_init__(self) -> None:
        if type(self.delay) is not int or self.delay < 0:
            raise ValueError("delay must be a non-negative integer")
        if type(self.replan) is not int or self.replan not in (2, 4):
            raise ValueError("replan must be 2 or 4")
        if self.delay + self.replan > 24:
            raise ValueError("delay + replan must not exceed 24")


@dataclasses.dataclass(frozen=True, slots=True)
class ScheduleConfig:
    configurations: tuple[DelayReplan, ...]
    supervised_offsets: tuple[int, ...]
    rollout_frames: int = 256

    def __post_init__(self) -> None:
        if not self.configurations or len(set(self.configurations)) != len(self.configurations):
            raise ValueError("configurations must be nonempty and unique")
        if (
            not self.supervised_offsets
            or any(type(offset) is not int or offset < 1 for offset in self.supervised_offsets)
            or tuple(sorted(set(self.supervised_offsets))) != self.supervised_offsets
        ):
            raise ValueError("supervised offsets must be positive, ordered, unique integers")
        if type(self.rollout_frames) is not int or self.rollout_frames < 1:
            raise ValueError("rollout_frames must be a positive integer")
        for config in self.configurations:
            if self.rollout_frames % config.replan:
                raise ValueError("rollout_frames must be divisible by every replan")
            missing = set(range(config.delay + 1, config.delay + config.replan + 1)) - set(self.supervised_offsets)
            if missing:
                raise ValueError(f"checkpoint never supervised executed offsets {sorted(missing)}")


def development_config(supervised_offsets: tuple[int, ...], *, rollout_frames: int = 256) -> ScheduleConfig:
    return ScheduleConfig((DelayReplan(8, 4), DelayReplan(10, 2)), supervised_offsets, rollout_frames)


def production_config(supervised_offsets: tuple[int, ...], *, rollout_frames: int = 256) -> ScheduleConfig:
    configurations = tuple(
        DelayReplan(delay, replan) for delay in range(16, 22) for replan in (2, 4) if delay + replan <= 24
    )
    return ScheduleConfig(configurations, supervised_offsets, rollout_frames)


def _check_tensor(
    name: str, value: Tensor, shape: tuple[int, ...], kind: Literal["bool", "int", "float"], device: torch.device
) -> None:
    if value.shape != shape or value.device != device:
        raise ValueError(f"{name} must have shape {shape} on {device}")
    valid_dtype = value.dtype == torch.bool if kind == "bool" else value.dtype == torch.int64
    if kind == "float":
        valid_dtype = value.is_floating_point()
    if not valid_dtype:
        raise ValueError(f"{name} must have {kind} dtype")


def _check_finite(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _loss_precision(value: Tensor) -> Tensor:
    return value if value.dtype == torch.float64 else value.float()


def _check_discount(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")


def _check_positive_count(count: Tensor) -> None:
    if count.ndim != 0 or (not count.is_floating_point() and count.dtype not in (torch.int32, torch.int64)):
        raise ValueError("count must be a real numeric scalar")
    _check_finite("count", count)
    if bool(count <= 0) or bool(count != count.round()):
        raise ValueError("count must be a positive integer")


def executed_factor_mask(delay_J: Tensor, replan_J: Tensor, frames_remaining_J: Tensor) -> Tensor:
    """Include offset D+1 through min(D+R, frames remaining in the match).

    The terminal frame counts as an executed frame. For an ongoing match, the
    caller supplies at least D+R; the rollout's remaining length is irrelevant.
    A time-out also discards queued actions and uses its last pre-reset frame.
    """
    shape = tuple(delay_J.shape)
    if len(shape) != 1:
        raise ValueError("delay must have shape [J]")
    for name, value in (("delay", delay_J), ("replan", replan_J), ("frames_remaining", frames_remaining_J)):
        _check_tensor(name, value, shape, "int", delay_J.device)
    if bool(((delay_J < 0) | ((replan_J != 2) & (replan_J != 4)) | (delay_J > 24 - replan_J)).any()):
        raise ValueError("invalid delay or replan")
    if bool((frames_remaining_J < 0).any()):
        raise ValueError("frames_remaining must be non-negative")
    frame = torch.arange(1, MAX_REPLAN + 1, device=delay_J.device)
    executed = (frame <= replan_J[:, None]) & (delay_J[:, None] + frame <= frames_remaining_J[:, None])
    return executed.repeat_interleave(4, dim=-1)


@dataclasses.dataclass(frozen=True, slots=True)
class DecisionBatch:
    """Decision rows, with all action indices stored in FACTOR_ORDER.

    The prefix is padded to [J, 24, 4]; only its first D frames are read.
    Sampled indices and behavior log-probs are padded to [J, 16]. The execution
    mask consists of complete frames and can be shorter than the sampled chunk.
    Observation/decision frames use the same clock. Lambda belongs to ego.
    """

    observation_frame_J: Tensor
    decision_frame_J: Tensor
    sampled_indices_JF: Tensor
    committed_prefix_JD4: Tensor
    behavior_log_probs_JF: Tensor
    executed_factors_JF: Tensor
    duration_J: Tensor
    bootstrap_mask_J: Tensor
    trace_mask_J: Tensor
    kl_coefficient_J: Tensor
    delay_J: Tensor
    replan_J: Tensor
    policy_version_J: Tensor
    episode_id_J: Tensor
    segment_id_J: Tensor

    def __post_init__(self) -> None:
        shape = tuple(self.delay_J.shape)
        if len(shape) != 1:
            raise ValueError("decision columns must have shape [J]")
        device = self.delay_J.device
        for name, value in (
            ("observation frame", self.observation_frame_J),
            ("decision frame", self.decision_frame_J),
            ("duration", self.duration_J),
            ("delay", self.delay_J),
            ("replan", self.replan_J),
            ("policy version", self.policy_version_J),
            ("episode id", self.episode_id_J),
            ("segment id", self.segment_id_J),
        ):
            _check_tensor(name, value, shape, "int", device)
        for name, value in (("bootstrap mask", self.bootstrap_mask_J), ("trace mask", self.trace_mask_J)):
            _check_tensor(name, value, shape, "bool", device)
        _check_tensor("lambda", self.kl_coefficient_J, shape, "float", device)
        _check_finite("lambda", self.kl_coefficient_J)
        if bool((self.kl_coefficient_J <= 0).any()):
            raise ValueError("lambda must be positive")
        j = shape[0]
        _check_tensor("indices", self.sampled_indices_JF, (j, 16), "int", device)
        _check_tensor("prefix", self.committed_prefix_JD4, (j, 24, 4), "int", device)
        _check_tensor("behavior log-probs", self.behavior_log_probs_JF, (j, 16), "float", device)
        _check_tensor("execution mask", self.executed_factors_JF, (j, 16), "bool", device)
        full_mask = executed_factor_mask(self.delay_J, self.replan_J, self.delay_J + self.replan_J)
        if bool((self.sampled_indices_JF[full_mask] < 0).any()):
            raise ValueError("sampled indices must be non-negative")
        prefix_mask = torch.arange(24, device=device) < self.delay_J[:, None]
        if bool((self.committed_prefix_JD4[prefix_mask] < 0).any()):
            raise ValueError("committed indices must be non-negative")
        executed = self.executed_factors_JF.reshape(j, 4, 4)
        if (
            bool((self.executed_factors_JF & ~full_mask).any())
            or not torch.equal(executed, executed[:, :, :1].expand_as(executed))
            or bool((~executed[:, :-1, 0] & executed[:, 1:, 0]).any())
        ):
            raise ValueError("execution mask must be a prefix of complete sampled frames")
        _check_finite("executed behavior log-probs", self.behavior_log_probs_JF[self.executed_factors_JF])
        if bool(((self.duration_J < 1) | (self.duration_J > self.replan_J)).any()):
            raise ValueError("duration must be in [1, R]")
        if bool((self.trace_mask_J & ~self.bootstrap_mask_J).any()):
            raise ValueError("a terminal decision cannot continue a trace")
        if bool(((self.duration_J < self.replan_J) & self.trace_mask_J).any()):
            raise ValueError("a shortened interval must end the trace")
        if bool((self.observation_frame_J > self.decision_frame_J).any()):
            raise ValueError("observation frame cannot follow the decision frame")

    @property
    def loss_mask_J(self) -> Tensor:
        return self.executed_factors_JF.any(-1)


@dataclasses.dataclass(frozen=True, slots=True)
class ControllerLogits:
    """Categorical logits [J, 4, K_group], in decoder order."""

    c_stick: Tensor
    main_stick: Tensor
    triggers: Tensor
    buttons: Tensor


@dataclasses.dataclass(frozen=True, slots=True)
class FactorEvaluation:
    log_probs_JF: Tensor
    reference_log_probs_JF: Tensor
    exact_kl_JF: Tensor


def categorical_log_probs(logits: Tensor, invalid_mask: Tensor | None = None) -> Tensor:
    """Apply the same support mask before normalizing either policy."""
    if not logits.is_floating_point() or logits.ndim < 1 or logits.shape[-1] < 1:
        raise ValueError("logits must be floating point with a nonempty category dimension")
    if invalid_mask is None:
        invalid_mask = torch.zeros_like(logits, dtype=torch.bool)
    _check_tensor("category mask", invalid_mask, tuple(logits.shape), "bool", logits.device)
    if bool(invalid_mask.all(-1).any()):
        raise ValueError("a categorical distribution needs a legal category")
    _check_finite("legal logits", logits[~invalid_mask])
    return _loss_precision(logits).masked_fill(invalid_mask, -torch.inf).log_softmax(-1)


def categorical_kl(log_probs: Tensor, reference_log_probs: Tensor) -> Tensor:
    """Exact forward KL; the reference is always a stop-gradient target.

    Inputs must be normalized log-probabilities with identical support. Illegal
    categories have log-probability -inf; they contribute zero, including in
    the backward pass.
    """
    _check_tensor("reference", reference_log_probs, tuple(log_probs.shape), "float", log_probs.device)
    if not log_probs.is_floating_point() or log_probs.ndim < 1 or log_probs.shape[-1] < 1:
        raise ValueError("log-probs must have a category dimension")
    log_probs = _loss_precision(log_probs)
    reference_log_probs = _loss_precision(reference_log_probs.detach())
    legal = torch.isfinite(log_probs)
    if not torch.equal(legal, torch.isfinite(reference_log_probs)):
        raise ValueError("policy and reference must have identical support")
    for value in (log_probs, reference_log_probs):
        if bool((torch.isnan(value) | torch.isposinf(value)).any()) or not bool(
            torch.isclose(value.logsumexp(-1), torch.zeros_like(value[..., 0]), atol=1e-5).all()
        ):
            raise ValueError("log-probs must be normalized")
    difference = torch.where(legal, log_probs, 0) - torch.where(legal, reference_log_probs, 0)
    return (log_probs.exp() * difference).sum(-1)


def evaluate_factors(
    policy: ControllerLogits,
    reference: ControllerLogits,
    sampled_indices_JF: Tensor,
    button_mask_JRK: Tensor,
) -> FactorEvaluation:
    """Evaluate sampled contexts, applying one trigger-derived button mask.

    The caller supplies logits conditioned on the identical forced prefix and
    earlier sampled factors. The mask uses HAL's convention: True is illegal.
    """
    if policy.c_stick.ndim != 3:
        raise ValueError("controller logits must have shape [J, 4, K_group]")
    j = policy.c_stick.shape[0]
    _check_tensor("sampled indices", sampled_indices_JF, (j, 16), "int", policy.c_stick.device)
    indices = sampled_indices_JF.reshape(j, 4, 4)
    log_probs: list[Tensor] = []
    reference_log_probs: list[Tensor] = []
    kls: list[Tensor] = []
    groups = (
        (policy.c_stick, reference.c_stick),
        (policy.main_stick, reference.main_stick),
        (policy.triggers, reference.triggers),
        (policy.buttons, reference.buttons),
    )
    for group, (logits, ref_logits) in enumerate(groups):
        if logits.ndim != 3 or logits.shape[:2] != (j, 4):
            raise ValueError("controller logits must have shape [J, 4, K_group]")
        _check_tensor("reference logits", ref_logits, tuple(logits.shape), "float", logits.device)
        mask = button_mask_JRK if group == 3 else None
        log_p = categorical_log_probs(logits, mask)
        log_ref = categorical_log_probs(ref_logits.detach(), mask)
        picks = indices[:, :, group, None]
        if bool(((picks < 0) | (picks >= logits.shape[-1])).any()):
            raise ValueError("sampled index outside categorical support")
        sampled = log_p.gather(-1, picks).squeeze(-1)
        _check_finite("sampled log-probs", sampled)
        log_probs.append(sampled)
        reference_log_probs.append(log_ref.gather(-1, picks).squeeze(-1))
        kls.append(categorical_kl(log_p, log_ref))
    return FactorEvaluation(
        torch.stack(log_probs, -1).flatten(1),
        torch.stack(reference_log_probs, -1).flatten(1),
        torch.stack(kls, -1).flatten(1),
    )


def _masked_factors(values_JF: Tensor, executed_JF: Tensor) -> Tensor:
    _check_tensor("factor values", values_JF, tuple(executed_JF.shape), "float", executed_JF.device)
    _check_finite("executed factor values", values_JF[executed_JF])
    return torch.where(executed_JF, _loss_precision(values_JF), 0)


def _factor_sum(values_JF: Tensor, executed_JF: Tensor) -> Tensor:
    return _masked_factors(values_JF, executed_JF).sum(-1)


@torch.no_grad()
def regularized_reward(
    decisions: DecisionBatch, frame_rewards_JR: Tensor, factor_kl_JF: Tensor, *, gamma: float = GAMMA
) -> Tensor:
    """Discount unshifted interval rewards and subtract lambda times old-policy KL."""
    _check_discount("gamma", gamma)
    _check_tensor("frame rewards", frame_rewards_JR, (len(decisions.delay_J), 4), "float", decisions.delay_J.device)
    frame = torch.arange(4, device=frame_rewards_JR.device)
    valid = frame < decisions.duration_J[:, None]
    _check_finite("interval rewards", frame_rewards_JR[valid])
    rewards = torch.where(valid, _loss_precision(frame_rewards_JR), 0)
    rewards = rewards * gamma ** frame.to(rewards.dtype)
    kl = _factor_sum(factor_kl_JF, decisions.executed_factors_JF)
    return rewards.sum(-1) - decisions.kl_coefficient_J * kl


@torch.no_grad()
def sampled_kl_penalty(decisions: DecisionBatch, reference_log_probs_JF: Tensor) -> Tensor:
    """The k1 cross-check uses log mu, with no differentiable KL loss."""
    behavior = _masked_factors(decisions.behavior_log_probs_JF, decisions.executed_factors_JF)
    reference = _masked_factors(reference_log_probs_JF, decisions.executed_factors_JF)
    return decisions.kl_coefficient_J * (behavior - reference).sum(-1)


@dataclasses.dataclass(frozen=True, slots=True)
class AdvantageTargets:
    advantages: Tensor
    value_targets: Tensor


@torch.no_grad()
def generalized_advantage(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    durations: Tensor,
    bootstrap_mask: Tensor,
    trace_mask: Tensor,
    *,
    gamma: float = GAMMA,
    gae_lambda: float = GAE_LAMBDA,
) -> AdvantageTargets:
    """Time-major GAE [T, ...], with explicit pre-reset and rollout-end values.

    next_values is V' for every interval, not a shifted tensor spanning resets.
    The final trace mask must be false. Terminal next-values may be NaN because
    the bootstrap mask removes them before arithmetic.
    """
    _check_discount("gamma", gamma)
    _check_discount("gae_lambda", gae_lambda)
    if rewards.ndim < 1 or rewards.shape[0] == 0:
        raise ValueError("GAE needs a nonempty time dimension")
    shape = tuple(rewards.shape)
    for name, value in (("rewards", rewards), ("values", values), ("next_values", next_values)):
        _check_tensor(name, value, shape, "float", rewards.device)
    _check_tensor("durations", durations, shape, "int", rewards.device)
    for name, value in (("bootstrap mask", bootstrap_mask), ("trace mask", trace_mask)):
        _check_tensor(name, value, shape, "bool", rewards.device)
    if bool(((durations < 1) | (durations > MAX_REPLAN)).any()):
        raise ValueError("durations must be in [1, 4]")
    if bool(trace_mask[-1].any()) or bool((trace_mask & ~bootstrap_mask).any()):
        raise ValueError("traces must stop at terminals and rollout end")
    _check_finite("rewards", rewards)
    _check_finite("values", values)
    _check_finite("bootstrap values", next_values[bootstrap_mask])
    values = _loss_precision(values)
    next_values = torch.where(bootstrap_mask, _loss_precision(next_values), 0)
    delta = _loss_precision(rewards) + gamma ** durations.to(values.dtype) * next_values - values
    trace_discount = (gamma * gae_lambda) ** durations.to(values.dtype)
    advantages = torch.empty_like(delta)
    tail = torch.zeros_like(delta[-1])
    for t in range(len(delta) - 1, -1, -1):
        tail = delta[t] + trace_discount[t] * torch.where(trace_mask[t], tail, 0)
        advantages[t] = tail
    return AdvantageTargets(advantages, advantages + values)


@torch.no_grad()
def stream_advantages(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    durations: Tensor,
    bootstrap_mask: Tensor,
    trace_mask: Tensor,
    streams: Sequence[Sequence[int]],
) -> AdvantageTargets:
    """Run the same reverse recurrence over padded independent stream columns."""
    if rewards.ndim != 1:
        raise ValueError("stream GAE needs flat decision rows")
    count = len(rewards)
    if (
        not streams
        or any(not stream for stream in streams)
        or sorted(index for stream in streams for index in stream) != list(range(count))
    ):
        raise ValueError("GAE streams must partition every decision exactly once")
    if bool(trace_mask[[stream[-1] for stream in streams]].any()):
        raise ValueError("traces must stop at every stream's rollout end")
    positions = np.full((max(map(len, streams)), len(streams)), count, dtype=np.int64)
    inverse = np.empty(count, dtype=np.int64)
    for column, stream in enumerate(streams):
        positions[: len(stream), column] = stream
        inverse[list(stream)] = np.arange(len(stream)) * len(streams) + column
    at = torch.as_tensor(positions, device=rewards.device)
    result = generalized_advantage(
        *(
            torch.cat((value, value.new_full((1,), padding)))[at]
            for value, padding in (
                (rewards, 0),
                (values, 0),
                (next_values, 0),
                (durations, 1),
                (bootstrap_mask, False),
                (trace_mask, False),
            )
        )
    )
    order = torch.as_tensor(inverse, device=rewards.device)
    return AdvantageTargets(result.advantages.flatten()[order], result.value_targets.flatten()[order])


@torch.no_grad()
def dense_advantages(
    rewards_TP: Tensor,
    values_TP: Tensor,
    next_values_TP: Tensor,
    duration_TP: Tensor,
    bootstrap_TP: Tensor,
    trace_TP: Tensor,
    decided_TP: Tensor,
) -> AdvantageTargets:
    """Empty ticks preserve the next decision's trace without adding a discount."""
    values = torch.where(decided_TP, values_TP, 0)
    next_values = torch.where(decided_TP & bootstrap_TP, next_values_TP, 0)
    rewards = torch.where(decided_TP, rewards_TP, 0)
    duration = duration_TP.to(values.dtype)
    delta = rewards + GAMMA**duration * next_values - values
    discount = (GAMMA * GAE_LAMBDA) ** duration
    advantages = torch.zeros_like(delta)
    tail = torch.zeros_like(delta[-1])
    for tick in range(len(delta) - 1, -1, -1):
        candidate = delta[tick] + discount[tick] * torch.where(trace_TP[tick], tail, 0)
        tail = torch.where(decided_TP[tick], candidate, tail)
        advantages[tick] = torch.where(decided_TP[tick], tail, 0)
    return AdvantageTargets(advantages, advantages + values)


@torch.no_grad()
def advantage_moments(advantages_J: Tensor, selected_J: Tensor) -> Tensor:
    """Return float64 [count, sum, sum of squares] for an external SUM all-reduce."""
    if not advantages_J.is_floating_point():
        raise ValueError("advantages must have floating dtype")
    _check_tensor("selection mask", selected_J, tuple(advantages_J.shape), "bool", advantages_J.device)
    _check_finite("selected advantages", advantages_J[selected_J])
    values = torch.where(selected_J, advantages_J.double(), 0)
    return torch.stack((selected_J.sum().double(), values.sum(), values.square().sum()))


@dataclasses.dataclass(frozen=True, slots=True)
class AdvantageStatistics:
    mean: Tensor
    sigma: Tensor
    count: Tensor

    def __post_init__(self) -> None:
        _check_positive_count(self.count)
        for name, value in (("mean", self.mean), ("sigma", self.sigma)):
            _check_tensor(name, value, (), "float", self.count.device)
            _check_finite(name, value)
        if bool(self.sigma <= 0):
            raise ValueError("sigma must be positive")


@torch.no_grad()
def advantage_statistics(global_moments: Tensor, *, minimum_sigma: float = 1e-8) -> AdvantageStatistics:
    """Population standard deviation with an explicit numerical floor.

    The actor and differentiable KL must use this same sigma. Moments cover
    exactly the selected training decisions across all ranks.
    """
    if global_moments.shape != (3,) or not global_moments.is_floating_point():
        raise ValueError("global moments must have shape [3] and floating dtype")
    _check_finite("global moments", global_moments)
    if not math.isfinite(minimum_sigma) or minimum_sigma <= 0:
        raise ValueError("minimum_sigma must be finite and positive")
    count, total, squares = global_moments.unbind()
    _check_positive_count(count)
    if bool(squares < 0):
        raise ValueError("sum of squares must be non-negative")
    mean = total / count
    variance = squares / count - mean.square()
    roundoff = 8 * torch.finfo(global_moments.dtype).eps * (squares / count + mean.square())
    if bool(variance < -roundoff):
        raise ValueError("inconsistent advantage moments give negative variance")
    return AdvantageStatistics(mean, variance.clamp_min(0).sqrt().clamp_min(minimum_sigma), count)


@dataclasses.dataclass(frozen=True, slots=True)
class DecisionReduction:
    """External global count and the number of ranks averaging gradients.

    DDP averages rank gradients, so each local sum is scaled by world_size / J.
    A critic whose gradients are SUM-reduced instead uses gradient_average_ranks=1.
    """

    global_count: Tensor
    gradient_average_ranks: int = 1

    def __post_init__(self) -> None:
        _check_positive_count(self.global_count)
        if type(self.gradient_average_ranks) is not int or self.gradient_average_ranks < 1:
            raise ValueError("gradient_average_ranks must be a positive integer")


def _decision_mean(values_J: Tensor, selected_J: Tensor, reduction: DecisionReduction | None) -> Tensor:
    _check_tensor("selected decisions", selected_J, tuple(values_J.shape), "bool", values_J.device)
    _check_finite("selected loss values", values_J[selected_J])
    local_count = selected_J.sum()
    if reduction is None:
        reduction = DecisionReduction(local_count)
    if bool(reduction.global_count < local_count):
        raise ValueError("global count cannot be smaller than the local count")
    return torch.where(selected_J, values_J, 0).sum() * (
        reduction.gradient_average_ranks
        / reduction.global_count.detach().to(device=values_J.device, dtype=values_J.dtype)
    )


def _training_mask(
    decisions: DecisionBatch,
    selected_J: Tensor,
    statistics: AdvantageStatistics,
    reduction: DecisionReduction | None,
) -> None:
    _check_tensor("selected decisions", selected_J, tuple(decisions.delay_J.shape), "bool", decisions.delay_J.device)
    if bool((selected_J & ~decisions.loss_mask_J).any()):
        raise ValueError("cannot train a decision with no executed factors")
    count = selected_J.sum() if reduction is None else reduction.global_count
    if bool(statistics.count.to(count.device) != count):
        raise ValueError("advantage statistics and loss must use the same global decision count")


@dataclasses.dataclass(frozen=True, slots=True)
class PolicyLoss:
    loss: Tensor
    joint_ratio_J: Tensor


def ppo_loss(
    decisions: DecisionBatch,
    new_log_probs_JF: Tensor,
    advantages_J: Tensor,
    selected_J: Tensor,
    statistics: AdvantageStatistics,
    *,
    clip_epsilon: float,
    reduction: DecisionReduction | None = None,
) -> PolicyLoss:
    """Joint PPO on selected decisions using detached global advantage statistics."""
    _training_mask(decisions, selected_J, statistics, reduction)
    if not math.isfinite(clip_epsilon) or not 0 < clip_epsilon < 1:
        raise ValueError("clip_epsilon must be in (0, 1)")
    _check_tensor("advantages", advantages_J, tuple(decisions.delay_J.shape), "float", decisions.delay_J.device)
    executed = decisions.executed_factors_JF & selected_J[:, None]
    new = _masked_factors(new_log_probs_JF, executed)
    old = _masked_factors(decisions.behavior_log_probs_JF.detach(), executed)
    # Subtract per factor before summing to preserve small kernel differences.
    ratio = (new - old).sum(-1).exp()
    _check_finite("joint ratio", ratio)
    advantages = torch.where(selected_J, advantages_J.detach(), 0)
    normalized = (advantages - statistics.mean.detach()) / statistics.sigma.detach()
    surrogate = torch.minimum(ratio * normalized, ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * normalized)
    return PolicyLoss(_decision_mean(-surrogate, selected_J, reduction), ratio)


def differentiable_kl_loss(
    decisions: DecisionBatch,
    factor_kl_JF: Tensor,
    selected_J: Tensor,
    statistics: AdvantageStatistics,
    *,
    reduction: DecisionReduction | None = None,
) -> Tensor:
    """Direct KL derivative on the same decisions and sigma used by PPO."""
    _training_mask(decisions, selected_J, statistics, reduction)
    kl = _factor_sum(factor_kl_JF, decisions.executed_factors_JF & selected_J[:, None])
    weighted = decisions.kl_coefficient_J.detach() / statistics.sigma.detach() * kl
    return _decision_mean(weighted, selected_J, reduction)


def critic_loss(
    values_J: Tensor,
    targets_J: Tensor,
    selected_J: Tensor,
    *,
    value_scale: float = 150.0,
    reduction: DecisionReduction | None = None,
) -> Tensor:
    """O64's mean squared error in beta=150 reward units; targets are detached.

    Selection includes all valid critic decisions, even ones with no executed
    factors. Cached critic features must be detached by the model boundary.
    """
    if not math.isfinite(value_scale) or value_scale <= 0:
        raise ValueError("value_scale must be finite and positive")
    if not values_J.is_floating_point():
        raise ValueError("critic values must have floating dtype")
    _check_tensor("targets", targets_J, tuple(values_J.shape), "float", values_J.device)
    _check_tensor("selection mask", selected_J, tuple(values_J.shape), "bool", values_J.device)
    values = torch.where(selected_J, _loss_precision(values_J), 0)
    targets = torch.where(selected_J, _loss_precision(targets_J.detach()), 0)
    return _decision_mean(((values - targets) / value_scale).square(), selected_J, reduction)


# PPO rows use FACTOR_ORDER. Index codec tensors with DECODE_GROUP_INDICES
# to read that order; CODEC_FACTOR_INDICES orders RNG factors for the codec.
DECODE_GROUP_INDICES = (2, 1, 3, 0)
CODEC_FACTOR_INDICES = (3, 1, 0, 2)
CHECKPOINT_VERSION = TRAINING_VERSION


@dataclasses.dataclass(frozen=True, slots=True)
class CounterRng:
    seed: int
    key: int = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise ValueError("counter RNG seed must be an integer")
        digest = hashlib.blake2b(str(self.seed).encode(), digest_size=8, person=b"hal-o65-key-v3").digest()
        object.__setattr__(self, "key", int.from_bytes(digest, "little") & 0x7FFFFFFFFFFFFFFF)

    def uniform(self, domain: str, stream: int, episode: int, frame: int, offset: int = 0, factor: int = 0) -> float:
        key = json.dumps((self.seed, domain, stream, episode, frame, offset, factor), separators=(",", ":"))
        digest = hashlib.blake2b(key.encode(), digest_size=8, person=b"hal-o65-rng-v1").digest()
        return (int.from_bytes(digest, "little") >> 11) * 2.0**-53

    def tensor_uniform(self, domain: int, stream: Tensor, episode: Tensor, frame: Tensor) -> Tensor:
        """Stateless 63-bit integer counters with identical CPU and CUDA arithmetic."""
        mask = 0x7FFFFFFFFFFFFFFF
        value = torch.full_like(stream, self.key)
        for counter in (domain, stream, episode, frame):
            value = (value ^ counter) & mask
            value = ((value ^ (value >> 30)) * 6364136223846793005) & mask
            value = ((value ^ (value >> 27)) * 1442695040888963407) & mask
            value = value ^ (value >> 31)
        return (value >> 10).double() * 2.0**-53

    def choice(self, weights: Sequence[float], domain: str, stream: int, episode: int, frame: int) -> int:
        if not weights or any(not math.isfinite(x) or x < 0 for x in weights) or sum(weights) <= 0:
            raise ValueError("sampling weights must be finite, nonnegative, and have positive mass")
        draw = self.uniform(domain, stream, episode, frame) * sum(weights)
        cumulative = 0.0
        for index, weight in enumerate(weights):
            cumulative += weight
            if draw < cumulative:
                return index
        raise ArithmeticError("categorical CDF did not cover the draw")


@dataclasses.dataclass(frozen=True, slots=True)
class IdentityCount:
    identity: int
    character: int
    count: int

    def __post_init__(self) -> None:
        if (
            any(type(value) is not int for value in (self.identity, self.character, self.count))
            or min(self.identity, self.character) < 0
            or self.count < 1
        ):
            raise ValueError("identity counts require nonnegative ids and positive counts")


@dataclasses.dataclass(frozen=True, slots=True)
class Matchup:
    characters: tuple[int, int]
    weight: float

    def __post_init__(self) -> None:
        if (
            len(self.characters) != 2
            or any(type(value) is not int or value < 0 for value in self.characters)
            or not math.isfinite(self.weight)
            or self.weight <= 0
        ):
            raise ValueError("invalid matchup prior")


@dataclasses.dataclass(frozen=True, slots=True)
class SamplerConfig:
    identities: tuple[IdentityCount, ...]
    matchups: tuple[Matchup, ...]
    stages: tuple[int, ...]
    data_floor: int = 100
    tempering: float = 0.5
    masked_rate: float = 0.1
    deployment_identity: int = int(Rank.MASTER)
    deployment_rate: float = 0.225
    lambda_min: float = 0.01
    lambda_max: float = 100.0
    stocks: tuple[int, ...] = (1, 2, 3, 4)
    percents: tuple[float, ...] = (0.0, 30.0, 60.0, 90.0)
    anchor_rate: float = 0.15

    def __post_init__(self) -> None:
        if (
            not self.identities
            or not self.matchups
            or not self.stages
            or any(type(stage) is not int or stage < 0 for stage in self.stages)
        ):
            raise ValueError("sampler tables and stages must be nonempty")
        if len({(row.identity, row.character) for row in self.identities}) != len(self.identities):
            raise ValueError("duplicate identity and character pair")
        if (
            type(self.data_floor) is not int
            or self.data_floor < 1
            or not math.isfinite(self.tempering)
            or not 0 <= self.tempering <= 1
        ):
            raise ValueError("invalid identity floor or tempering")
        for rate in (self.masked_rate, self.deployment_rate, self.anchor_rate):
            if not math.isfinite(rate) or not 0 <= rate <= 1:
                raise ValueError("sampler rates must be in [0, 1]")
        ordinary_rate = 1 - (
            self.deployment_rate if self.deployment_identity == 0 else self.masked_rate + self.deployment_rate
        )
        if type(self.deployment_identity) is not int or self.deployment_identity < 0 or ordinary_rate < 0:
            raise ValueError("invalid deployment identity or identity rates")
        if not 0 < self.lambda_min <= self.lambda_max or not math.isfinite(self.lambda_max):
            raise ValueError("invalid lambda range")
        if not self.stocks or any(type(stock) is not int or not 1 <= stock <= 4 for stock in self.stocks):
            raise ValueError("stocks must be in [1, 4]")
        if not self.percents or any(not math.isfinite(x) or x < 0 for x in self.percents):
            raise ValueError("starting percents must be finite and nonnegative")
        for matchup in self.matchups:
            for character in matchup.characters:
                rows = [row for row in self.identities if row.character == character and row.count >= self.data_floor]
                ordinary = [row for row in rows if row.identity not in (0, self.deployment_identity)]
                if not rows or (ordinary_rate > 0 and not ordinary):
                    raise ValueError(f"character {character} has no admitted ordinary identities")
                if (
                    self.deployment_identity
                    and self.deployment_rate > 0
                    and not any(row.identity == self.deployment_identity for row in rows)
                ):
                    raise ValueError(f"character {character} has no admitted deployment identity")


@dataclasses.dataclass(frozen=True, slots=True)
class PlayerEpisode:
    identity: int
    character: int
    log_lambda: float
    timing: DelayReplan
    stocks: int
    percent: float
    anchor: int | None


@dataclasses.dataclass(frozen=True, slots=True)
class Episode:
    number: int
    stage: int
    players: tuple[PlayerEpisode, PlayerEpisode]
    seed: int


class EpisodeSampler:
    def __init__(self, config: SamplerConfig, schedule: ScheduleConfig, rng: CounterRng) -> None:
        self.config = config
        self.schedule = schedule
        self.rng = rng
        self.lambda_max = config.lambda_max
        self.cursor = 0

    def sample(self, match_id: int, episode: int, anchors: tuple[int, ...]) -> Episode:
        cfg = self.config
        rng = self.rng
        matchup = cfg.matchups[rng.choice([m.weight for m in cfg.matchups], "matchup", match_id, episode, 0)]
        stage = cfg.stages[rng.choice([1.0] * len(cfg.stages), "stage", match_id, episode, 0)]
        stocks = cfg.stocks[rng.choice([1.0] * len(cfg.stocks), "stocks", match_id, episode, 0)]
        anchor_side = None
        if rng.uniform("anchor_share", match_id, episode, 0) < cfg.anchor_rate:
            anchor_side = rng.choice((1.0, 1.0), "anchor_side", match_id, episode, 0)
        players: list[PlayerEpisode] = []
        for side, character in enumerate(matchup.characters):
            stream = 2 * match_id + side
            identity_draw = rng.uniform("identity_kind", stream, episode, 0)
            masked_mass = cfg.deployment_rate if cfg.deployment_identity == 0 else cfg.masked_rate
            if identity_draw < masked_mass:
                identity = 0
            elif cfg.deployment_identity != 0 and identity_draw < masked_mass + cfg.deployment_rate:
                identity = cfg.deployment_identity
            else:
                rows = [
                    row
                    for row in cfg.identities
                    if row.character == character
                    and row.count >= cfg.data_floor
                    and row.identity not in (0, cfg.deployment_identity)
                ]
                identity = rows[
                    rng.choice([row.count**cfg.tempering for row in rows], "identity", stream, episode, 0)
                ].identity
            log_lambda = math.log(cfg.lambda_min) + rng.uniform("lambda", stream, episode, 0) * (
                math.log(self.lambda_max) - math.log(cfg.lambda_min)
            )
            timing = self.schedule.configurations[
                rng.choice([1.0] * len(self.schedule.configurations), "timing", stream, episode, 0)
            ]
            percent = cfg.percents[rng.choice([1.0] * len(cfg.percents), "percent", stream, episode, 0)]
            anchor = None
            if side == anchor_side:
                if not anchors:
                    raise ValueError("anchor sampling requires a reference policy")
                anchor = anchors[rng.choice([1.0] * len(anchors), "anchor", stream, episode, 0)]
            players.append(PlayerEpisode(identity, character, log_lambda, timing, stocks, percent, anchor))
        self.cursor += 1
        seed = int(rng.uniform("simulator_seed", match_id, episode, 0) * 2**32)
        return Episode(episode, stage, (players[0], players[1]), seed)


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class NativeColumns(Mapping[str, int | float]):
    """Decode an immutable adapter row when diagnostics or CPU features need columns."""

    words: NDArray[np.uint32]

    def __post_init__(self) -> None:
        if (
            self.words.shape != (len(native.OBSERVATION_FIELDS),)
            or self.words.dtype != np.uint32
            or self.words.flags.writeable
        ):
            raise ValueError("native columns require an immutable uint32 adapter row")

    def __getitem__(self, name: str) -> int | float:
        index, integer = NATIVE_COLUMN_LAYOUT[name]
        word = self.words[index]
        return int(word.view(np.int32)) if integer else float(word.view(np.float32))

    def __iter__(self) -> Iterator[str]:
        return iter(native.OBSERVATION_FIELDS)

    def __len__(self) -> int:
        return len(native.OBSERVATION_FIELDS)


NATIVE_COLUMN_LAYOUT = {
    name: (index, np.dtype(MDS_PER_FRAME_DTYPES[name]).kind == "i")
    for index, name in enumerate(native.OBSERVATION_FIELDS)
}


@dataclasses.dataclass(frozen=True, slots=True)
class DeviceFrameBlock:
    """Stage write-once native transfers before copying them into group storage."""

    records: Tensor  # [128, lanes, bytes]
    actions: Tensor  # [128, lanes, 2, ACTION_DIM]


@dataclasses.dataclass(frozen=True, slots=True)
class SimFrame:
    """One post-frame observation. No timer column is admitted by the builder."""

    columns: Mapping[str, int | float]
    applied_actions: NDArray[np.float32]  # [2, ACTION_DIM], HAL wire units
    rewards: tuple[float, float] = (0.0, 0.0)
    terminal: bool = False
    timeout: bool = False
    device_record: Tensor | None = None  # packed HalFrame bytes, copied from pinned output
    device_actions: Tensor | None = None  # processed readback [2, ACTION_DIM]
    device_block: DeviceFrameBlock | None = None
    device_slot: tuple[int, int] | None = None  # frame, lane in device_block

    def __post_init__(self) -> None:
        if self.applied_actions.shape != (2, ACTION_DIM) or self.applied_actions.dtype != np.float32:
            raise ValueError("simulator actions must be float32 [2, ACTION_DIM]")
        if not np.isfinite(self.applied_actions).all() or not all(math.isfinite(x) for x in self.rewards):
            raise ValueError("simulator actions and rewards must be finite")
        if self.terminal and self.timeout:
            raise ValueError("a match end cannot also be a time-out")


class SimulatorShard(Protocol):
    """Step queued actions until the first terminal or timeout in the shard.

    Inputs contain [frames, lanes, 2, ACTION_DIM]; step returns 1..frames rows.
    Reset returns the initial observation while the group clock stays fixed.
    Callers copy borrowed observations before the next write and serialize calls
    on each shard. Save and restore include simulator RNG and build identity.
    """

    @property
    def match_ids(self) -> tuple[int, ...]: ...
    def step(self, inputs: NDArray[np.float32], active: NDArray[np.bool_]) -> Sequence[Sequence[SimFrame]]: ...
    def reset(self, lane: int, episode: Episode) -> SimFrame: ...
    def save(self) -> bytes: ...
    def restore(self, state: bytes) -> None: ...


class NativeSimulatorShard:
    """Step native matches with pinned buffers and batch their device transfers.

    Wait for the previous GPU copy before writing output again. The rollout owns
    each batch's observations after it copies them into its group storage.
    """

    def __init__(
        self, data_dir: Path, match_ids: tuple[int, ...], *, cpu: int, device: torch.device, max_frame: int = 28800
    ) -> None:
        if not match_ids or len(set(match_ids)) != len(match_ids) or min(match_ids) < 0:
            raise ValueError("native shard needs distinct nonnegative match ids")
        self.match_ids = match_ids
        self.device = device
        self.max_frame = max_frame
        count = len(match_ids)
        pinned = device.type == "cuda"
        self._output = torch.zeros(count * native.FRAME_DTYPE.itemsize, dtype=torch.uint8, pin_memory=pinned)
        self._records = self._output.numpy().view(native.FRAME_DTYPE)
        self._step_input = torch.empty(2, count, 2, ACTION_DIM, dtype=torch.float32, pin_memory=pinned)
        self._observed = torch.empty(count, 2, ACTION_DIM, dtype=torch.float32, pin_memory=pinned)
        self._event: torch.cuda.Event | None = None
        self._device_block: DeviceFrameBlock | None = None
        self._device_frame = 128
        self.step_seconds = 0.0
        self.reset_seconds = 0.0
        self.stepped_frames = 0
        self.copied_bytes = 0
        self._mask = np.ones(count, dtype=np.bool_)
        self._configs = [native.NativeMatch(25, (native.NativePlayer(1, 1), native.NativePlayer(2, 22)), 0)] * count
        self.shard = NativeShard(data_dir, count, cpu=cpu, records=self._records)

    def _before_write(self) -> None:
        if self._event is not None:
            self._event.synchronize()

    def _frames(self, frame: native.NativeFrameBatch) -> FrameBatch:
        if frame.records is None:
            raise ValueError("native rollout requires packed adapter records")
        columns = frame.records["columns"].copy()
        columns.flags.writeable = False
        observed = native_controller_readback(frame.applied_action)
        self._observed.copy_(torch.from_numpy(observed))
        if self._device_frame == 128:
            self._device_block = DeviceFrameBlock(
                torch.zeros(
                    128, len(self.match_ids), native.FRAME_DTYPE.itemsize, device=self.device, dtype=torch.uint8
                ),
                torch.zeros(128, len(self.match_ids), 2, ACTION_DIM, device=self.device),
            )
            self._device_frame = 0
        block = self._device_block
        if block is None:
            raise RuntimeError("native device frame storage was not allocated")
        packed = block.records[self._device_frame]
        actions = block.actions[self._device_frame]
        packed.copy_(self._output.reshape(len(self.match_ids), -1), non_blocking=True)
        actions.copy_(self._observed, non_blocking=True)
        if self.device.type == "cuda":
            self._event = torch.cuda.Event()
            self._event.record(torch.cuda.current_stream(self.device))
            self.copied_bytes += self._output.numel() + self._observed.numel() * self._observed.element_size()
        self._device_frame += 1
        return FrameBatch(
            packed, actions, columns, frame.reward.copy(), frame.terminated.copy(), frame.truncated.copy(), observed
        )

    def step(
        self, inputs: Tensor | NDArray[np.float32], active: NDArray[np.bool_] | None = None
    ) -> tuple[FrameBatch, ...]:
        started = time.perf_counter()
        source = torch.from_numpy(inputs) if isinstance(inputs, np.ndarray) else inputs
        if (
            source.dtype != torch.float32
            or source.ndim != 4
            or source.shape[1:] != (len(self.match_ids), 2, ACTION_DIM)
            or not len(source)
        ):
            raise ValueError("native step needs float32 [frames, lanes, 2, ACTION_DIM]")
        if active is None:
            active = self._mask
        if active.shape != (len(self.match_ids),) or active.dtype != np.bool_:
            raise ValueError("native active mask must be bool [lanes]")
        if self._step_input.shape[0] < len(source):
            self._step_input = torch.empty(source.shape, dtype=torch.float32, pin_memory=self.device.type == "cuda")
        # A single device-to-pinned-host transfer supplies the whole tick.
        self._step_input[: len(source)].copy_(source)
        rows = []
        for actions in self._step_input[: len(source)]:
            frame = self.shard.step(actions.numpy(), active, before_write=self._before_write).result()
            rows.append(self._frames(frame))
            self.stepped_frames += int(active.sum())
            if ((frame.terminated | frame.truncated) & active).any():
                break
        self.step_seconds += time.perf_counter() - started
        return tuple(rows)

    def reset(self, lane: int, episode: Episode) -> SimFrame:
        started = time.perf_counter()
        if type(lane) is not int or not 0 <= lane < len(self.match_ids):
            raise ValueError("native reset lane is out of range")
        left, right = episode.players
        if left.stocks != right.stocks:
            raise ValueError("native reset does not support unequal starting stocks")
        if any(p.percent != int(p.percent) for p in episode.players):
            raise ValueError("native reset requires integer starting percents")
        config = native.NativeMatch(
            episode.stage,
            (
                native.NativePlayer(1, left.character, start_percent=int(left.percent)),
                native.NativePlayer(2, right.character, start_percent=int(right.percent)),
            ),
            episode.seed,
            max_frame=self.max_frame,
            stocks=left.stocks,
        )
        self._configs[lane] = config
        mask = np.arange(len(self.match_ids)) == lane
        frame = self.shard.reset(self._configs, mask, before_write=self._before_write).result()
        result = self._frames(frame)[lane]
        self.reset_seconds += time.perf_counter() - started
        return result

    def save(self) -> bytes:
        states = self.shard.save_many(tuple(range(len(self.match_ids)))).result()
        header = json.dumps(
            {"format": "hal.o65.native_pool.v2", "matches": self.match_ids, "max_frame": self.max_frame}
        ).encode()
        return (
            struct.pack("<I", len(header))
            + header
            + b"".join(struct.pack("<I", len(state)) + state for state in states)
        )

    def restore(self, state: bytes) -> None:
        if len(state) < 4:
            raise ValueError("truncated native pool snapshot")
        length = struct.unpack_from("<I", state)[0]
        at = 4 + length
        expected = {"format": "hal.o65.native_pool.v2", "matches": list(self.match_ids), "max_frame": self.max_frame}
        if json.loads(state[4:at]) != expected:
            raise ValueError("native pool snapshot identity mismatch")
        states = []
        for _ in self.match_ids:
            if at + 4 > len(state):
                raise ValueError("truncated native pool snapshot")
            size = struct.unpack_from("<I", state, at)[0]
            at += 4
            if at + size > len(state):
                raise ValueError("truncated native pool snapshot")
            states.append(state[at : at + size])
            at += size
        if at != len(state):
            raise ValueError("unexpected native pool snapshot bytes")
        self.shard.restore_many(tuple(range(len(self.match_ids))), states, before_write=self._before_write).result()

    def close(self) -> None:
        try:
            self._before_write()
        finally:
            self.shard.close()


def _step_shard(
    shard: SimulatorShard,
    inputs: Tensor,
    active: NDArray[np.bool_],
    stream: torch.cuda.Stream | None = None,
) -> Sequence[Sequence[SimFrame]]:
    with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
        if isinstance(shard, NativeSimulatorShard):
            return shard.step(inputs, active)
        return shard.step(inputs.cpu().numpy(), active)


@dataclasses.dataclass(frozen=True, slots=True)
class DecisionState:
    stream: int
    match: int
    side: int
    episode: Episode
    clock: int
    window: tuple[int, int, int, int]  # update, stream, block, episode
    frames: Sequence[SimFrame]
    prefix: Tensor  # [D, 4], codec order
    opponent_queue: Tensor  # [up to D_opp+R_opp, 4], codec order

    @property
    def player(self) -> PlayerEpisode:
        return self.episode.players[self.side]


@dataclasses.dataclass(frozen=True, slots=True)
class CriticFeatures:
    trunk: Tensor
    decoder: Tensor
    condition: Tensor
    opponent_condition: Tensor
    opponent_queue: Tensor
    opponent_lengths: Tensor

    def detached(self) -> CriticFeatures:
        return CriticFeatures(*(getattr(self, field.name).detach().clone() for field in dataclasses.fields(self)))


CRITIC_FEATURE_NAMES = ("trunk", "decoder", "condition", "opponent_condition", "opponent_queue", "opponent_lengths")


@dataclasses.dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    actions: Tensor  # [J, 4, 4], codec order
    log_probs: Tensor  # [J, 16], decode order
    distributions: tuple[Tensor, ...]  # [J, 4, K], decode order; button mask already applied
    features: CriticFeatures


@dataclasses.dataclass(frozen=True, slots=True)
class CachedStream:
    window: tuple[int, int, int, int]
    history_length: int
    cache: BlockRestartCache
    row: int


class LoopPolicy(Protocol):
    """Run rollout sampling and differentiable replay through separate kernels."""

    @property
    def model(self) -> torch.nn.Module: ...
    @property
    def device(self) -> torch.device: ...
    @property
    def max_delay(self) -> int: ...
    @property
    def max_opponent_queue(self) -> int: ...
    @property
    def condition_ranges(self) -> ConditioningRanges: ...
    def evaluate_windows(self, batch: ReplayBatch) -> ReplayOutput: ...
    def evaluate_values(self, batch: ReplayBatch) -> Tensor: ...
    def actor_step(
        self, batch: ReplayBatch, targets: ActorTargets, *, clip_epsilon: float
    ) -> tuple[ActorLoss, ReplayOutput]: ...
    def actor_parameters(self) -> tuple[torch.nn.Parameter, ...]: ...
    def critic_parameters(self) -> tuple[torch.nn.Parameter, ...]: ...
    def sample(self, states: Sequence[DecisionState], rng: CounterRng) -> PolicyEvaluation: ...
    def evaluate(self, states: Sequence[DecisionState], actions: Tensor) -> PolicyEvaluation: ...
    def values(self, features: CriticFeatures) -> Tensor: ...
    def wire_actions(self, indices: Tensor) -> NDArray[np.float32]: ...
    def clear_caches(self) -> None: ...
    def frozen_copy(self) -> LoopPolicy: ...


def _concatenate_evaluations(rows: Sequence[PolicyEvaluation]) -> PolicyEvaluation:
    return PolicyEvaluation(
        torch.cat([row.actions for row in rows]),
        torch.cat([row.log_probs for row in rows]),
        tuple(torch.cat([row.distributions[group] for row in rows]) for group in range(4)),
        CriticFeatures(
            *(
                torch.cat([getattr(row.features, field.name) for row in rows])
                for field in dataclasses.fields(CriticFeatures)
            )
        ),
    )


def _take_evaluation(evaluation: PolicyEvaluation, indices: Tensor) -> PolicyEvaluation:
    return PolicyEvaluation(
        evaluation.actions[indices],
        evaluation.log_probs[indices],
        tuple(value[indices] for value in evaluation.distributions),
        _take_features(evaluation.features, indices),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class TickLayout:
    streams: tuple[int, ...]
    lanes: tuple[int, ...]
    buffer: int = 0

    def __post_init__(self) -> None:
        if type(self.buffer) is not int or self.buffer not in (0, 1):
            raise ValueError("tick buffer must be zero or one")
        if not self.streams or len(self.streams) != len(self.lanes):
            raise ValueError("tick layout lanes must match nonempty streams")
        if len(set(self.streams)) != len(self.streams):
            raise ValueError("tick layout streams must be distinct")
        if any(type(value) is not int or value < 0 for value in (*self.streams, *self.lanes)):
            raise ValueError("tick layout coordinates must be nonnegative integers")
        if len({(lane, stream % 2) for lane, stream in zip(self.lanes, self.streams, strict=True)}) != len(
            self.streams
        ):
            raise ValueError("tick layout streams must own distinct queues")


@dataclasses.dataclass(frozen=True, slots=True)
class RolloutBuffer:
    index: int
    shards: tuple[SimulatorShard, ...]
    lanes: NDArray[np.int64]
    stream: torch.cuda.Stream | None


type PendingStep = tuple[RolloutBuffer, SimulatorShard, Future[Sequence[Sequence[SimFrame]]]]


class TransformerPolicy:
    """Single-device correctness path; each training window encodes once.

    Rollout uses the block cache. Training and the no-grad pass use the same
    full-window trunk and replay the recorded conditional action path.
    """

    def __init__(
        self,
        model: SelfPlayTransformer,
        schedule: ScheduleConfig,
        statistics: dict[str, FeatureStats],
        *,
        compiled: bool = False,
        replay_chunks: ReplayChunks = DEFAULT_REPLAY_CHUNKS,
    ) -> None:
        self.profile: UpdateProfile | None = None
        self.model = model.eval()
        self.compiled = compiled
        self.replay_chunks = replay_chunks
        self.schedule = schedule
        self.statistics = statistics
        for stat in statistics.values():
            if (
                not all(math.isfinite(value) for value in (stat.mean, stat.std, stat.min, stat.max))
                or stat.std < 0
                or stat.min > stat.max
            ):
                raise ValueError("invalid observation statistics")
        self.device = next(model.parameters()).device
        if schedule.supervised_offsets != model.head_offsets:
            raise ValueError("schedule supervised offsets must match the model checkpoint")
        configurations = tuple((c.delay, c.replan) for c in schedule.configurations)
        self.decoder = DecisionDecoder(model, configurations, compiled=compiled)
        self.group_sizes_cpu = self.decoder.group_sizes.cpu()
        self.button_valid_cpu = self.model.codec.button_valid_for_trigger.cpu()
        self.builders = (ColumnContextBuilder(statistics, 1), ColumnContextBuilder(statistics, 2))
        self.packed_builder = PackedColumnContextBuilder(statistics, native.OBSERVATION_FIELDS, self.device)
        self.decode_order = torch.tensor(DECODE_GROUP_INDICES, device=self.device)
        self.tick_decoder = DecisionDecoder(model, configurations, compiled=False)
        self.tick_caches: dict[tuple[ActionRing, tuple[int, ...]], BlockRestartCache] = {}
        self.tick_graphs: dict[
            tuple[tuple[ActionRing, tuple[int, ...]], int, int], CapturedCall[tuple[PolicyEvaluation, Tensor]]
        ] = {}
        self.last_tick_values: Tensor | None = None
        self.current_layouts: dict[tuple[ActionRing, int], TickLayout] = {}
        self.tick_windows: dict[tuple[ActionRing, int], tuple[int, int]] = {}
        self.window_graphs: dict[tuple[tuple[int, ...], ...], CapturedCall[ReplayOutput]] = {}
        self.value_graphs: dict[tuple[tuple[int, ...], ...], CapturedCall[Tensor]] = {}
        self.queue_replay = (
            torch.compile(
                self.model.temporal.replay_window_queue_state,
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False, "emulate_precision_casts": True},
            )
            if self.device.type == "cuda"
            else self.model.temporal.replay_window_queue_state
        )
        self.actor_graph: GraphedActorStep | None = None
        self.actor_clip_epsilon: float | None = None
        self.window_replay = (
            torch.compile(
                replay_windows,
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False, "emulate_precision_casts": True},
            )
            if self.device.type == "cuda"
            else replay_windows
        )
        self.caches: dict[int, CachedStream] = {}
        self.rollout_caches: dict[tuple[int, ...], BlockRestartCache] = {}
        self.used_cache_keys: set[tuple[int, ...]] = set()
        self.sample_group_counts: dict[int, int] = defaultdict(int)
        self.sample_batch_counts: dict[int, int] = defaultdict(int)
        self.device_wire_tables: tuple[Tensor, ...] = ()
        self.wire_tables: tuple[NDArray[np.float32], ...] = ()
        self._prepare_wire_tables()

    @torch.no_grad()
    def _prepare_wire_tables(self) -> None:
        tables = tuple(
            self.model.codec.semantic_values(name, torch.arange(size, device=self.device)).float()
            for name, size in zip(CONTROLLER_GROUP_NAMES, self.model.codec.group_vocabs, strict=True)
        )
        if self.device_wire_tables:
            for target, source in zip(self.device_wire_tables, tables, strict=True):
                target.copy_(source)
        else:
            self.device_wire_tables = tables
        self.wire_tables = tuple(value.cpu().numpy() for value in self.device_wire_tables)

    @property
    def max_delay(self) -> int:
        return self.decoder.max_delay

    @property
    def max_opponent_queue(self) -> int:
        return self.model.critic_residual.max_queue

    @property
    def condition_ranges(self) -> ConditioningRanges:
        return self.model.ranges

    def actor_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return self.model.actor_parameters()

    def critic_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return self.model.critic_parameters()

    def clear_caches(self) -> None:
        self.caches.clear()
        self.tick_windows.clear()
        self._prepare_wire_tables()
        if self.used_cache_keys:
            self.rollout_caches = {
                key: cache
                for key, cache in self.rollout_caches.items()
                if key in self.used_cache_keys
                and (not self.compiled or self.device.type != "cuda" or cache.cache.updates)
            }
            self.used_cache_keys.clear()

    def frozen_copy(self) -> LoopPolicy:
        model = copy.deepcopy(self.model).requires_grad_(False)
        model.zero_grad(set_to_none=True)
        return TransformerPolicy(
            model, self.schedule, self.statistics, compiled=self.compiled, replay_chunks=self.replay_chunks
        )

    def _validate_states(self, states: Sequence[DecisionState], actions: Tensor | None = None) -> None:
        if not states:
            raise ValueError("policy batch must be nonempty")
        if isinstance(states, DecisionRows):
            meta = states.meta
            if np.any((meta[:, SIDE] < 0) | (meta[:, SIDE] > 1)):
                raise ValueError("decision side must be HAL port 1 or 2's zero-based index")
            if np.any((meta[:, LENGTH] < 1) | (meta[:, LENGTH] > 256)):
                raise ValueError("invalid rollout history span")
            if np.any((meta[:, IDENTITY] < 0) | (meta[:, IDENTITY] >= self.model.cfg.player_vocab_size)):
                raise ValueError("episode identity is outside the checkpoint vocabulary")
            allowed = {(timing.delay, timing.replan) for timing in self.schedule.configurations}
            for delay, replan in ((DELAY, REPLAN), (OPP_DELAY, OPP_REPLAN)):
                if any((int(row[delay]), int(row[replan])) not in allowed for row in meta):
                    raise ValueError("decision batch contains an unsupported configuration")
            logs = states.table.log_lambda[states.ids]
            low, high = self.model.ranges.log_lambda
            if not np.isfinite(logs).all() or np.any((logs < low) | (logs > high)):
                raise ValueError("episode conditioning is outside its declared range")
            if np.any((meta[:, OPP_LENGTH] < 0) | (meta[:, OPP_LENGTH] > self.model.critic_residual.max_queue)):
                raise ValueError("opponent queue exceeds the critic capacity or has the wrong shape")
            queues = torch.cat((states.tensor(states.table.prefix), states.tensor(states.table.opponent)), dim=1)
            self._validate_actions(queues.flatten(0, 1))
            if actions is not None:
                self._validate_actions(actions)
            return
        for state in states:
            if type(state.side) is not int or state.side not in (0, 1):
                raise ValueError("decision side must be HAL port 1 or 2's zero-based index")
            if not 1 <= len(state.frames) <= 256:
                raise ValueError("invalid rollout history span")
            if not 0 <= state.player.identity < self.model.cfg.player_vocab_size:
                raise ValueError("episode identity is outside the checkpoint vocabulary")
            for player in state.episode.players:
                if player.timing not in self.schedule.configurations:
                    raise ValueError("decision batch contains an unsupported configuration")
                if (
                    not math.isfinite(player.log_lambda)
                    or not self.model.ranges.log_lambda[0] <= player.log_lambda <= self.model.ranges.log_lambda[1]
                ):
                    raise ValueError("episode conditioning is outside its declared range")
            if state.prefix.shape != (state.player.timing.delay, 4):
                raise ValueError("decision prefix must contain D committed frames")
            if (
                state.opponent_queue.ndim != 2
                or state.opponent_queue.shape[1] != 4
                or len(state.opponent_queue) > self.model.critic_residual.max_queue
            ):
                raise ValueError("opponent queue exceeds the critic capacity or has the wrong shape")
        queues = torch.cat([queue for state in states for queue in (state.prefix, state.opponent_queue)])
        self._validate_actions(queues)
        if actions is not None:
            self._validate_actions(actions)

    def _validate_actions(self, actions: Tensor) -> None:
        if actions.dtype not in (torch.int32, torch.int64) or actions.shape[-1] != 4:
            raise ValueError("actions must contain integer controller factors")
        sizes = self.group_sizes_cpu if actions.device.type == "cpu" else self.decoder.group_sizes
        if not bool(((actions >= 0) & (actions < sizes)).all()):
            raise ValueError("actions contain an invalid controller factor")
        valid_buttons = (
            self.button_valid_cpu if actions.device.type == "cpu" else self.model.codec.button_valid_for_trigger
        )
        if bool((~valid_buttons[actions[..., 3], actions[..., 0]]).any()):
            raise ValueError("actions contain buttons incompatible with their triggers")

    def _packed_inputs(
        self, states: Sequence[DecisionState], frames: Sequence[tuple[SimFrame, ...]], length: int
    ) -> tuple[Tensor, ...]:
        if all(frame.device_block is not None and frame.device_slot is not None for row in frames for frame in row):
            blocks: dict[int, tuple[DeviceFrameBlock, list[tuple[int, int, int]]]] = {}
            for row, history in enumerate(frames):
                for position, frame in enumerate(history):
                    block = cast(DeviceFrameBlock, frame.device_block)
                    slot = cast(tuple[int, int], frame.device_slot)
                    if id(block) not in blocks:
                        blocks[id(block)] = (block, [])
                    blocks[id(block)][1].append((row * length + position, *slot))
            coordinates = torch.tensor([item for _, items in blocks.values() for item in items])
            device_coordinates = coordinates.to(self.device, non_blocking=True)
            records = torch.zeros(
                len(states) * length, native.FRAME_DTYPE.itemsize, dtype=torch.uint8, device=self.device
            )
            actions = torch.zeros(len(states) * length, 2, ACTION_DIM, device=self.device)
            start = 0
            for block, items in blocks.values():
                end = start + len(items)
                at = device_coordinates[start:end]
                source_at = at if block.records.device == self.device else coordinates[start:end]
                records.index_copy_(0, at[:, 0], block.records[source_at[:, 1], source_at[:, 2]].to(self.device))
                actions.index_copy_(0, at[:, 0], block.actions[source_at[:, 1], source_at[:, 2]].to(self.device))
                start = end
            records = records.reshape(len(states), length, -1)
            actions = actions.reshape(len(states), length, 2, ACTION_DIM)
        else:
            packed = [self._packed_history(row) for row in frames]
            records = torch.stack([torch.nn.functional.pad(row[0], (0, 0, 0, length - len(row[0]))) for row in packed])
            actions = torch.stack(
                [torch.nn.functional.pad(row[1], (0, 0, 0, 0, 0, length - len(row[1]))) for row in packed]
            )
        sides = torch.tensor([state.side for state in states]).to(self.device, non_blocking=True)
        identities = torch.tensor([state.player.identity for state in states]).to(self.device, non_blocking=True)
        lengths = torch.tensor([len(row) for row in frames]).to(self.device, non_blocking=True)
        return records, actions, sides, identities, lengths

    def _packed_history(self, frames: tuple[SimFrame, ...]) -> tuple[Tensor, Tensor]:
        return (
            torch.stack([cast(Tensor, frame.device_record).to(self.device, non_blocking=True) for frame in frames]),
            torch.stack([cast(Tensor, frame.device_actions).to(self.device, non_blocking=True) for frame in frames]),
        )

    def _packed_context(self, inputs: tuple[Tensor, ...]) -> tuple[dict[str, Tensor], Tensor]:
        records, actions, sides, identities, lengths = inputs
        features = self.packed_builder(records, actions, sides, identities, lengths)
        length = records.shape[1]
        ego = actions.gather(2, sides[:, None, None, None].expand(-1, length, 1, ACTION_DIM)).squeeze(2)
        valid = torch.arange(length, device=self.device)[None] < lengths[:, None]
        return features, torch.where(valid[..., None], self.model.codec.quantize(ego), 0)

    def _append_packed(
        self, states: Sequence[DecisionState], count: int, cache: BlockRestartCache
    ) -> tuple[Tensor, Tensor]:
        if not 1 <= count <= 4:
            raise ValueError("cached context append must contain one to four frames")
        clocks = (
            states.meta[:, CLOCK] - states.meta[:, BLOCK]
            if isinstance(states, DecisionRows)
            else np.array([state.clock - state.window[2] for state in states])
        )
        if np.any((clocks < 0) | (clocks >= 128)):
            raise ValueError("block is exhausted; restart before appending")
        if isinstance(states, DecisionRows):
            inputs = states.packed(count, tail=count)
        else:
            frames = [tuple(state.frames[-count:]) for state in states]
            inputs = self._packed_inputs(states, frames, count)
        key = 2 * count + int(torch.is_inference_mode_enabled())
        captured = cache.cache.updates.get(key)
        if captured is None:
            static = tuple(value.clone() for value in inputs)

            def append() -> tuple[Tensor, Tensor]:
                # Autocast's eager weight cache dies when sample() returns.
                # Capture the casts so replay also reads updated actor weights.
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    features, observed = self._packed_context(static)
                    tokens = self.model.context_tokens(features, observed)
                    hidden = forward_tokens_with_kv_cache(self.model, tokens, cache.cache)
                    cache.frames_in_block.add_(count)
                    return hidden, observed[:, -1]

            captured = CapturedCall(
                append, static, (*cache.cache.buffers(), cache.frames_in_block), device=self.device
            )
            cache.cache.updates[key] = captured
        hidden, observed = cast(CapturedCall[tuple[Tensor, Tensor]], captured)(inputs)
        return hidden, observed

    def _contexts(
        self, states: Sequence[DecisionState], *, length: int | None = None, tail: int | None = None
    ) -> tuple[dict[str, Tensor], Tensor]:
        if isinstance(states, DecisionRows):
            required = {route[0] for route in self.builders[0].routes}
            if not required.issubset(states.table.store.fields):
                raise ValueError("simulator observations are missing required policy columns")
            lengths = states.meta[:, LENGTH]
            length = (
                int(lengths.max() if tail is None else min(tail, int(lengths.max()))) if length is None else length
            )
            return self._packed_context(states.packed(length, tail))
        frames = [tuple(state.frames if tail is None else state.frames[-tail:]) for state in states]
        length = max(len(row) for row in frames) if length is None else length
        if any(len(row) > length for row in frames):
            raise ValueError("context padding is shorter than its history")
        columns: dict[str, Tensor] = {}
        packed = all(
            frame.device_record is not None and frame.device_actions is not None for row in frames for frame in row
        )
        if packed:
            return self._packed_context(self._packed_inputs(states, frames, length))
        for name, value in frames[0][0].columns.items():
            dtype = torch.int64 if isinstance(value, int) else torch.float32
            columns[name] = torch.tensor(
                [[frame.columns[name] for frame in row] + [0] * (length - len(row)) for row in frames],
                dtype=dtype,
                device=self.device,
            )
        host_actions = np.zeros((len(states), length, 2, ACTION_DIM), dtype=np.float32)
        for index, row in enumerate(frames):
            host_actions[index, : len(row)] = np.stack([frame.applied_actions for frame in row])
        actions = torch.as_tensor(host_actions, device=self.device)
        parts: list[dict[str, Tensor]] = []
        observed_parts: list[Tensor] = []
        order: list[int] = []
        for side in range(2):
            indices = [index for index, state in enumerate(states) if state.side == side]
            if not indices:
                continue
            at = torch.tensor(indices, device=self.device)
            segments = torch.zeros(len(indices), length, device=self.device, dtype=torch.long)
            context = self.builders[side](
                {name: value[at] for name, value in columns.items()},
                actions[at],
                segments,
                torch.zeros_like(segments, dtype=torch.bool),
                torch.tensor([states[index].player.identity for index in indices], device=self.device),
            )
            valid = (
                torch.arange(length, device=self.device)[None]
                < torch.tensor([len(frames[index]) for index in indices], device=self.device)[:, None]
            )
            parts.append({name: torch.where(valid, value, 0) for name, value in context.features.items()})
            observed_parts.append(torch.where(valid[..., None], self.model.codec.quantize(actions[at, :, side]), 0))
            order.extend(indices)
        inverse = torch.tensor(np.argsort(order)).to(self.device, non_blocking=True)
        return (
            {name: torch.cat([part[name] for part in parts])[inverse] for name in parts[0]},
            torch.cat(observed_parts)[inverse],
        )

    def _columnar_states(self, states: Sequence[DecisionState]) -> DecisionRows:
        if isinstance(states, DecisionRows):
            return states
        # Explicit test and CPU-proxy inputs enter the same packed tensor path.
        self._validate_states(states)
        length = max(len(state.frames) for state in states)
        fields = tuple(states[0].frames[0].columns)
        store = FrameStore(length, len(states), self.device, fields)
        records = np.zeros((length, len(states)), dtype=native.FRAME_DTYPE)
        applied = np.zeros((length, len(states), 2, ACTION_DIM), dtype=np.float32)
        encoded: dict[int, NDArray] = {}
        packed_inputs = all(
            frame.device_record is not None and frame.device_actions is not None
            for state in states
            for frame in state.frames
        )
        for lane, state in enumerate(states):
            for position, frame in enumerate(state.frames):
                if packed_inputs:
                    records[position : position + 1, lane] = (
                        cast(Tensor, frame.device_record).cpu().numpy().view(native.FRAME_DTYPE)
                    )
                    applied[position, lane] = cast(Tensor, frame.device_actions).detach().cpu().numpy()
                else:
                    packed = encoded.get(id(frame))
                    if packed is None:
                        packed = np.zeros(1, dtype=native.FRAME_DTYPE)
                        for name, value in frame.columns.items():
                            if name in NATIVE_COLUMN_LAYOUT:
                                index, _ = NATIVE_COLUMN_LAYOUT[name]
                                packed["columns"][0, index] = np.array([value], dtype=MDS_PER_FRAME_DTYPES[name]).view(
                                    np.uint32
                                )[0]
                        encoded[id(frame)] = packed
                    records[position, lane] = packed[0]
                    applied[position, lane] = frame.applied_actions
        store.records.copy_(torch.from_numpy(records.view(np.uint8).reshape(length, len(states), -1)).to(self.device))
        store.actions.copy_(torch.from_numpy(applied).to(self.device))
        table = DecisionTable(len(states), store)
        meta = []
        for lane, state in enumerate(states):
            player, opponent = state.player, state.episode.players[1 - state.side]
            meta.append(
                [
                    state.stream,
                    state.match,
                    state.side,
                    state.clock,
                    state.window[0],
                    state.window[2],
                    state.episode.number,
                    lane,
                    0,
                    len(state.frames),
                    player.identity,
                    player.character,
                    state.episode.stage,
                    player.timing.delay,
                    player.timing.replan,
                    opponent.timing.delay,
                    opponent.timing.replan,
                    len(state.opponent_queue),
                ]
            )
        table.count = len(states)
        table.meta[: table.count] = meta
        table.integers.copy_(torch.tensor(meta, device=self.device))
        table.log_lambda[:] = [
            (state.player.log_lambda, state.episode.players[1 - state.side].log_lambda) for state in states
        ]
        table.conditions.copy_(torch.from_numpy(table.log_lambda).to(device=self.device, dtype=torch.float32))
        for index, state in enumerate(states):
            table.episodes[state.match, state.episode.number] = state.episode
            table.prefix[index, : len(state.prefix)] = state.prefix.to(self.device)
            table.opponent[index, : len(state.opponent_queue)] = state.opponent_queue.to(self.device)
        return table.rows(np.arange(table.count))

    def _decode(
        self,
        states: DecisionRows,
        hidden: Tensor,
        observed: Tensor,
        uniforms: Tensor,
        actions: Tensor | None = None,
        *,
        history: KVMemory,
        count: int,
    ) -> PolicyEvaluation:
        return self._decode_tensors(
            states.tensor(states.table.integers),
            states.tensor(states.table.conditions),
            states.tensor(states.table.prefix),
            states.tensor(states.table.opponent),
            hidden,
            observed,
            uniforms,
            actions,
            history=history,
            count=count,
            decoder=self.decoder,
        )

    def _decode_tensors(
        self,
        integers: Tensor,
        conditions: Tensor,
        prefix: Tensor,
        opponent: Tensor,
        hidden: Tensor,
        observed: Tensor,
        uniforms: Tensor,
        actions: Tensor | None,
        *,
        history: KVMemory,
        count: int,
        decoder: DecisionDecoder,
    ) -> PolicyEvaluation:
        delay = integers[:, DELAY].contiguous()
        replan = integers[:, REPLAN].contiguous()
        log_lambda = conditions[:, 0].contiguous()
        queue = prefix[:, : decoder.max_delay].contiguous()
        output = decoder(
            hidden,
            observed,
            queue,
            log_lambda,
            delay,
            replan,
            uniforms[:count],
            sampled_offsets=count,
            history=history,
            sampled_actions=None if actions is None else actions[:, :count],
            validate=False,
        )
        condition = self.model.ranges.normalize(log_lambda, delay, replan, validate=False)
        opponent_condition = self.model.ranges.normalize(
            conditions[:, 1].contiguous(),
            integers[:, OPP_DELAY].contiguous(),
            integers[:, OPP_REPLAN].contiguous(),
            validate=False,
        )
        capacity = self.model.critic_residual.max_queue
        features = CriticFeatures(
            hidden[:, -1].detach(),
            output.decoder_state.detach(),
            condition.detach(),
            opponent_condition.detach(),
            opponent[:, :capacity].contiguous(),
            integers[:, OPP_LENGTH].contiguous(),
        )
        distributions = []
        for group in DECODE_GROUP_INDICES:
            probabilities = output.distributions[group]
            padding = probabilities.new_zeros(hidden.shape[0], 4 - count, probabilities.shape[-1])
            padding[..., 0] = 1
            distributions.append(torch.cat((probabilities, padding), dim=1))
        return PolicyEvaluation(
            torch.nn.functional.pad(output.actions, (0, 0, 0, 4 - count)),
            torch.nn.functional.pad(
                output.log_probs.index_select(-1, self.decode_order), (0, 0, 0, 4 - count)
            ).flatten(1),
            tuple(distributions),
            features,
        )

    @torch.no_grad()
    def sample(self, states: Sequence[DecisionState], rng: CounterRng) -> PolicyEvaluation:
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            return self._sample(states, rng)

    def _sample(self, states: Sequence[DecisionState], rng: CounterRng) -> PolicyEvaluation:
        states = self._columnar_states(states)
        self._validate_states(states)
        meta = states.meta
        if np.any(meta[:, CLOCK] % meta[:, REPLAN]) or np.any(meta[:, CLOCK] % 4 != meta[0, CLOCK] % 4):
            raise ValueError("sampling batch must share a decision tick and respect each cadence")
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, row in enumerate(meta):
            entry = self.caches.get(int(row[STREAM]))
            window = _window_key(row)
            seen = entry.history_length if entry is not None and entry.window == window else 0
            groups[int(row[LENGTH]), seen].append(index)
        self.sample_group_counts[len(groups)] += 1
        rows: list[PolicyEvaluation] = []
        order: list[int] = []
        for (_, seen), indices in groups.items():
            batch = states.take(indices)
            self.sample_batch_counts[len(batch)] += 1
            if self.profile is not None:
                self.profile.work["rollout/trunk_tokens"] += len(batch) * (int(batch.meta[0, LENGTH]) - seen)
                self.profile.work["rollout/decoder_tokens"] += len(batch) * (
                    self.decoder.max_delay + (4 if meta[indices[0], CLOCK] % 4 == 0 else 2)
                )
            hidden, last_observed, cache = self._advance_rollout_cache(batch, seen)
            uniforms = self._action_uniforms(batch, rng)
            count = 4 if meta[indices[0], CLOCK] % 4 == 0 else 2
            rows.append(
                self._decode(batch, hidden[:, -1:], last_observed, uniforms, history=cache.memory(), count=count)
            )
            order.extend(indices)
        return _take_evaluation(
            _concatenate_evaluations(rows), torch.tensor(np.argsort(order)).to(self.device, non_blocking=True)
        )

    @torch.no_grad()
    def sample_tick(
        self, states: DecisionRows, rng: CounterRng, layout: TickLayout, ring: ActionRing, *, clock: int = 0
    ) -> PolicyEvaluation:
        if len(layout.streams) != len(states):
            raise ValueError("tick layout contains an unknown stream or row count")
        key = ring, (layout.buffer, len(states))
        layout_key = ring, layout.buffer
        previous_layout = self.current_layouts.get(layout_key)
        if previous_layout is not layout:
            if previous_layout is not None and clock % 128 and previous_layout.streams != layout.streams:
                raise ValueError("tick layout may change only at a block boundary")
            expected_streams = torch.tensor(layout.streams, device=self.device)
            expected_lanes = torch.tensor(layout.lanes, device=self.device)
            declared = states.tensor(states.table.integers)
            if bool((declared[:, STREAM] != expected_streams).any()):
                raise ValueError("rollout tick contains an unknown stream")
            if bool((declared[:, LANE] != expected_lanes).any()):
                raise ValueError("rollout tick metadata disagrees with its lane layout")
            self.current_layouts[layout_key] = layout
        cache = self.tick_caches.get(key)
        if cache is None:
            cache = BlockRestartCache(
                self.model,
                len(states),
                self.device,
                dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            )
            self.tick_caches[key] = cache
        meta = states.tensor(states.table.integers)
        restart = clock % 128 == 0
        enabled = self.device.type == "cuda"
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=enabled, cache_enabled=False):
            if restart:
                features, observed = self._packed_context(states.packed(129))
                tokens = self.model.context_tokens(features, observed)
                lengths = meta[:, LENGTH] - 1
                positions = torch.arange(128, device=self.device)[None] - (128 - lengths[:, None])
                previous = tokens.gather(1, positions.clamp_min(0)[..., None].expand(-1, -1, tokens.shape[-1]))
                previous = torch.where((positions >= 0)[..., None], previous, 0)
                cache.restart(previous, lengths)
            appended = 1 if restart else 2
            inputs = (
                *states.packed(appended, tail=appended),
                meta,
                states.tensor(states.table.conditions),
                states.tensor(states.table.prefix),
                states.tensor(states.table.opponent),
                states.tensor(states.table.tick_active),
                states.device_ids,
            )
            count = 4 if clock % 4 == 0 else 2
            graph_key = key, appended, count
            captured = self.tick_graphs.get(graph_key)
            if captured is None:
                static = tuple(value.clone() for value in inputs)

                def tick() -> tuple[PolicyEvaluation, Tensor]:
                    with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=enabled, cache_enabled=False):
                        context, observed = self._packed_context(static[:5])
                        tokens = self.model.context_tokens(context, observed)
                        hidden = forward_tokens_with_kv_cache(self.model, tokens, cache.cache)
                        cache.frames_in_block.add_(appended)
                        result = self._decode_tensors(
                            static[5],
                            static[6],
                            static[7],
                            static[8],
                            hidden[:, -1:],
                            observed[:, -1],
                            self._tensor_action_uniforms(static[5], rng),
                            None,
                            history=cache.memory(),
                            count=count,
                            decoder=self.tick_decoder,
                        )
                        rows = static[5]
                        lanes, sides = rows[:, LANE], rows[:, SIDE]
                        due = static[9] & (rows[:, CLOCK] % rows[:, REPLAN] == 0)
                        lengths = ring.length_device[lanes, sides]
                        written = min(count, ring.capacity)
                        offsets = torch.arange(written, device=self.device)[None]
                        positions = (ring.head[lanes, sides, None] + lengths[:, None] + offsets) % ring.capacity
                        valid = due[:, None] & (offsets < rows[:, REPLAN, None])
                        current = ring.values[lanes[:, None], sides[:, None], positions]
                        ring.values[lanes[:, None], sides[:, None], positions] = torch.where(
                            valid[..., None],
                            result.actions[:, :written],
                            current,
                        )
                        if ring.controllers is None:
                            raise ValueError("rollout tick requires controller storage")
                        controller = self.device_wire_actions(result.actions[:, :written])
                        previous = ring.controllers[lanes[:, None], sides[:, None], positions]
                        ring.controllers[lanes[:, None], sides[:, None], positions] = torch.where(
                            valid[..., None],
                            controller,
                            previous,
                        )
                        ring.length_device[lanes, sides] += torch.where(due, rows[:, REPLAN], 0)
                        values = self.values(result.features, validate=False)
                        states.table.record_evaluation(static[10], result, values)
                        return result, values

                if enabled:
                    operation = torch.compile(
                        tick, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}
                    )
                    with prepared_shape_compilation(64):
                        captured = CapturedCall(
                            operation,
                            static,
                            (
                                *cache.cache.buffers(),
                                cache.frames_in_block,
                                ring.values,
                                cast(Tensor, ring.controllers),
                                ring.length_device,
                            ),
                            device=self.device,
                        )
                    self.tick_graphs[graph_key] = captured
                else:
                    # CPU tests execute the same fixed-slot operation without capture.
                    captured = None
            if self.profile is not None:
                self.profile.work["rollout/trunk_tokens"] += len(states) * (appended + (128 if restart else 0))
                self.profile.work["rollout/decoder_tokens"] += len(states) * (self.tick_decoder.max_delay + count)
            output, self.last_tick_values = tick() if captured is None else captured(inputs)
        self.sample_group_counts[1] += 1
        self.sample_batch_counts[len(states)] += 1
        return output

    def _tensor_action_uniforms(self, meta: Tensor, rng: CounterRng) -> Tensor:
        offsets = meta[:, DELAY][None, None] + torch.arange(1, 5, device=self.device)[:, None, None]
        factors = torch.tensor(CODEC_FACTOR_INDICES, device=self.device)[None, :, None]
        return rng.tensor_uniform(
            3,
            meta[:, STREAM][None, None].expand(4, 4, -1),
            meta[:, EPISODE][None, None],
            meta[:, CLOCK][None, None] * 128 + offsets * 4 + factors,
        )

    def _advance_rollout_cache(self, batch: DecisionRows, seen: int) -> tuple[Tensor, Tensor, BlockRestartCache]:
        meta = batch.meta
        length = int(meta[0, LENGTH])
        if not seen:
            cache = self._rollout_cache_for(tuple(int(stream) for stream in meta[:, STREAM]))
            features, observed = self._contexts(batch)
            tokens = self.model.context_tokens(features, observed)
            previous = tokens[:, :-1]
            if previous.shape[1] > 128:
                raise ValueError("first decision in a block must prefill at most 128 previous frames")
            cache.restart(previous, torch.full((len(batch),), previous.shape[1], device=self.device))
            tokens = tokens[:, -1:]
        else:
            sources = [self.caches[int(stream)] for stream in meta[:, STREAM]]
            cache = sources[0].cache
            if cache.cache.batch_size != len(batch) or any(
                source.cache is not cache or source.row != row for row, source in enumerate(sources)
            ):
                cache = self._rollout_cache_for(tuple(int(stream) for stream in meta[:, STREAM]))
                for target_index, target in enumerate(cache.cache.buffers()):
                    axis = 1 if target.ndim == 5 else 0
                    target.copy_(
                        torch.cat(
                            [
                                source.cache.cache.buffers()[target_index].narrow(axis, source.row, 1)
                                for source in sources
                            ],
                            dim=axis,
                        )
                    )
                cache.frames_in_block.copy_(
                    torch.cat([source.cache.frames_in_block[source.row : source.row + 1] for source in sources])
                )
        if seen and self.compiled and self.device.type == "cuda":
            hidden, last_observed = self._append_packed(batch, length - seen, cache)
        else:
            if seen:
                features, observed = self._contexts(batch, tail=length - seen)
                tokens = self.model.context_tokens(features, observed)
            hidden = cache.append(tokens)
            last_observed = observed[:, -1]
        for row, data in enumerate(meta):
            window = _window_key(data)
            self.caches[int(data[STREAM])] = CachedStream(window, length, cache, row)
        return hidden, last_observed, cache

    def _rollout_cache_for(self, streams: tuple[int, ...]) -> BlockRestartCache:
        self.used_cache_keys.add(streams)
        cache = self.rollout_caches.get(streams)
        if cache is None:
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
            cache = BlockRestartCache(self.model, len(streams), self.device, dtype=dtype)
            self.rollout_caches[streams] = cache
        return cache

    def _action_uniforms(self, batch: DecisionRows, rng: CounterRng) -> Tensor:
        counters = [(int(row[STREAM]), int(row[EPISODE]), int(row[CLOCK]), int(row[DELAY])) for row in batch.meta]
        if self.device.type == "cuda":
            from hal.inference.action_rng import action_uniforms

            return action_uniforms(rng.seed, counters, self.device)
        draws = [
            [
                [
                    rng.uniform("action", stream, episode, clock, delay + offset + 1, factor)
                    for stream, episode, clock, delay in counters
                ]
                for factor in CODEC_FACTOR_INDICES
            ]
            for offset in range(4)
        ]
        return torch.tensor(draws, dtype=torch.float64, device=self.device)

    def evaluate(self, states: Sequence[DecisionState], actions: Tensor) -> PolicyEvaluation:
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            return self._evaluate(states, actions)

    def _evaluate(self, states: Sequence[DecisionState], actions: Tensor) -> PolicyEvaluation:
        chunks = self.replay_chunks if not torch.is_grad_enabled() else DEFAULT_REPLAY_CHUNKS
        states = self._columnar_states(states)
        self._validate_states(states, actions)
        meta = states.meta
        windows: dict[tuple[int, int, int, int], int] = {}
        row_windows = []
        for index, row in enumerate(meta):
            key = _window_key(row)
            row_windows.append(key)
            if key not in windows or row[LENGTH] > meta[windows[key], LENGTH]:
                windows[key] = index
        window_states = states.take(list(windows.values()))
        encoded: dict[tuple[int, int, int, int], tuple[Tensor, Tensor, Tensor]] = {}
        for start in range(0, len(window_states), chunks.windows):
            batch = window_states[start : start + chunks.windows]
            if self.profile is not None:
                self.profile.work[f"{self.profile.phase_name}/trunk_tokens"] += len(batch) * 256
            with sample(self.profile, "trunk_chunk", work=float(len(batch) * 256)):
                features, observed = self._contexts(batch, length=256)
                hidden = self.model(features, torch.zeros(len(batch), dtype=torch.long, device=self.device), observed)
                keys, values = self.model.temporal.history_attention.project_memory(hidden)
                memory = torch.stack((keys, values))
                for row, data in enumerate(batch.meta):
                    encoded[_window_key(data)] = (hidden[row], observed[row], memory[:, row])
        outputs: list[PolicyEvaluation] = []
        for start in range(0, len(states), chunks.decisions):
            batch = states[start : start + chunks.decisions]
            if self.profile is not None:
                self.profile.work[f"{self.profile.phase_name}/decoder_tokens"] += len(batch) * (
                    self.decoder.max_delay + 4
                )
            with sample(self.profile, "decoder_chunk", work=float(len(batch))):
                keys = row_windows[start : start + len(batch)]
                positions_host = batch.meta[:, LENGTH] - 1
                hidden = torch.stack(
                    [encoded[key][0][int(position)] for key, position in zip(keys, positions_host, strict=True)]
                )[:, None]
                observed = torch.stack(
                    [encoded[key][1][int(position)] for key, position in zip(keys, positions_host, strict=True)]
                )
                memory = KVMemory(
                    torch.stack([encoded[key][2] for key in keys], dim=1),
                    torch.arange(256, device=self.device)[None].expand(len(batch), -1),
                    torch.from_numpy(positions_host.copy()).to(self.device),
                    256,
                )
                outputs.append(
                    self._decode(
                        batch,
                        hidden,
                        observed,
                        torch.full((4, 4, len(batch)), 0.5, device=self.device),
                        actions[start : start + len(batch)],
                        history=memory,
                        count=4,
                    )
                )
        return _concatenate_evaluations(outputs)

    def window_forward(self, inputs: tuple[Tensor, ...], *, grad: bool) -> ReplayOutput:
        with (
            torch.set_grad_enabled(grad),
            torch.autocast(
                self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda", cache_enabled=False
            ),
        ):
            context, observed = self._packed_context(inputs[:5])
            hidden = self.model(context, torch.zeros(len(observed), dtype=torch.long, device=self.device), observed)
        # Torch 2.11 whole-trunk compilation failed the proxy precision gate.
        # Capture the eager trunk together with the compiled shared-history decoder.
        return self.window_replay(self.model, hidden, observed, ReplaySlots(*inputs[5:]), grad=grad)

    @torch.no_grad()
    def evaluate_windows(self, batch: ReplayBatch) -> ReplayOutput:
        if self.profile is not None:
            phase_name = self.profile.phase_name
            self.profile.work[f"{phase_name}/trunk_tokens"] += batch.records.shape[0] * batch.records.shape[1]
            self.profile.work[f"{phase_name}/decoder_tokens"] += batch.rows.numel() * (self.max_delay + 4)
        inputs = batch.inputs()
        if self.device.type == "cpu":
            return self.window_forward(inputs, grad=False)
        key = tuple(tuple(value.shape) for value in inputs)
        captured = self.window_graphs.get(key)
        if captured is None:
            static = tuple(value.clone() for value in inputs)
            with prepared_shape_compilation(64):
                captured = CapturedCall(lambda: self.window_forward(static, grad=False), static, device=self.device)
            self.window_graphs[key] = captured
        return captured(inputs)

    def value_forward(self, inputs: tuple[Tensor, ...]) -> Tensor:
        slots = ReplaySlots(*inputs[5:])
        valid = slots.valid_WK
        positions = torch.where(valid, slots.positions_WK, 0)
        delay = torch.where(valid, slots.delay_WK, 0)
        condition = torch.where(valid[..., None], slots.condition_WK3, 0)
        with torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda", cache_enabled=False
        ):
            context, observed = self._packed_context(inputs[:5])
            hidden = self.model(context, torch.zeros(len(observed), dtype=torch.long, device=self.device), observed)
        with torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.model.decoder_precision == "bf16", cache_enabled=False
        ):
            state = self.queue_replay(
                hidden.float(), observed, positions, slots.history_starts_WL, slots.forced_queue_WKD4, delay, condition
            )
        rows = torch.arange(len(observed), device=self.device)[:, None]
        lengths = torch.where(valid, slots.opponent_lengths_WK, 0)
        opponent_queue = torch.where(
            (torch.arange(self.max_opponent_queue, device=self.device) < lengths[..., None])[..., None],
            slots.opponent_queue_WKQ4,
            0,
        )
        features = CriticFeatures(
            hidden[rows, positions].flatten(0, 1).detach(),
            state.flatten(0, 1).detach(),
            condition.flatten(0, 1),
            torch.where(valid[..., None], slots.opponent_condition_WK3, 0).flatten(0, 1),
            opponent_queue.flatten(0, 1),
            lengths.flatten(),
        )
        return self.values(features, validate=False).reshape(valid.shape)

    @torch.no_grad()
    def evaluate_values(self, batch: ReplayBatch) -> Tensor:
        if self.profile is not None:
            phase_name = self.profile.phase_name
            self.profile.work[f"{phase_name}/trunk_tokens"] += batch.records.shape[0] * batch.records.shape[1]
            self.profile.work[f"{phase_name}/decoder_tokens"] += batch.rows.numel() * (self.max_delay + 1)
        inputs = batch.inputs()
        if self.device.type == "cpu":
            return self.value_forward(inputs)
        key = tuple(tuple(value.shape) for value in inputs)
        captured = self.value_graphs.get(key)
        if captured is None:
            static = tuple(value.clone() for value in inputs)
            with prepared_shape_compilation(64):
                captured = CapturedCall(lambda: self.value_forward(static), static, device=self.device)
            self.value_graphs[key] = captured
        return captured(inputs)

    def actor_step(
        self, batch: ReplayBatch, targets: ActorTargets, *, clip_epsilon: float
    ) -> tuple[ActorLoss, ReplayOutput]:
        if self.profile is not None:
            phase_name = self.profile.phase_name
            self.profile.work[f"{phase_name}/trunk_tokens"] += batch.records.shape[0] * batch.records.shape[1]
            self.profile.work[f"{phase_name}/decoder_tokens"] += batch.rows.numel() * (self.max_delay + 4)
        inputs = batch.inputs()
        if self.device.type == "cpu":
            self.model.zero_grad(set_to_none=True)
            output = self.window_forward(inputs, grad=True)
            result = actor_loss(output, targets, clip_epsilon=clip_epsilon)
            result.loss.backward()
            detached = ReplayOutput(
                output.log_probs_WKF.detach(),
                tuple(value.detach() for value in output.distributions),
                output.factor_mask_WKF,
                output.features,
            )
            return result, detached
        if self.actor_graph is not None and self.actor_clip_epsilon != clip_epsilon:
            raise ValueError("actor clip epsilon differs from the captured objective")
        if self.actor_graph is None:
            with prepared_shape_compilation(64):
                self.actor_graph = GraphedActorStep(
                    lambda values: self.window_forward(values, grad=True),
                    inputs,
                    targets,
                    self.actor_parameters(),
                    clip_epsilon=clip_epsilon,
                )
            self.actor_clip_epsilon = clip_epsilon
        return self.actor_graph.backward(inputs, targets), self.actor_graph.evaluation()

    def values(self, features: CriticFeatures, *, validate: bool = True) -> Tensor:
        return self.model.self_play_value(
            features.trunk,
            features.decoder,
            features.condition,
            features.opponent_condition,
            features.opponent_queue,
            features.opponent_lengths,
            validate=validate,
        )

    def device_wire_actions(self, indices: Tensor) -> Tensor:
        tables = self.device_wire_tables
        result = torch.empty(*indices.shape[:-1], ACTION_DIM, device=indices.device, dtype=torch.float32)
        result[..., :2] = tables[1][indices[..., 1]]
        result[..., 2:4] = tables[2][indices[..., 2]]
        result[..., 4:6] = tables[3][indices[..., 3]]
        result[..., 6:] = tables[0][indices[..., 0]]
        return result

    def wire_actions(self, indices: Tensor) -> NDArray[np.float32]:
        host = indices.detach().cpu().numpy()
        if host.dtype not in (np.int32, np.int64) or host.shape[-1:] != (4,):
            raise ValueError("wire actions require integer controller factors")
        if any(
            ((host[..., group] < 0) | (host[..., group] >= len(table))).any()
            for group, table in enumerate(self.wire_tables)
        ):
            raise ValueError("wire actions contain an invalid controller factor")
        result = np.empty((*host.shape[:-1], ACTION_DIM), dtype=np.float32)
        result[..., :2] = self.wire_tables[1][host[..., 1]]
        result[..., 2:4] = self.wire_tables[2][host[..., 2]]
        result[..., 4:6] = self.wire_tables[3][host[..., 3]]
        result[..., 6:] = self.wire_tables[0][host[..., 0]]
        return result


@dataclasses.dataclass(slots=True)
class MatchState:
    episode: Episode
    episode_start: int
    history_start: int
    history: FrameHistory
    queues: tuple[QueueView, QueueView]
    returns: tuple[float, float] = (0.0, 0.0)
    end_clock: int | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class EpisodeResult:
    match: int
    episode: Episode
    duration: int
    returns: tuple[float, float]
    timeout: bool
    stocks: tuple[int, int] | None


@dataclasses.dataclass(slots=True)
class DecisionRecord:
    state: DecisionState
    actions: Tensor  # [4, 4], codec order
    behavior: Tensor  # [16], decode order
    rewards: list[float]
    terminal: bool = False
    boundary: bool = False
    frames_remaining: int | None = None
    next_state: DecisionState | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class Rollout:
    """Dense group storage stays valid until that group is collected again."""

    records: DecisionRecords
    group: int
    update: int
    frames: int
    resets: int
    timeouts: int
    elapsed: float
    results: tuple[EpisodeResult, ...]
    idle_frames: int = 0


def _decision_batch(records: Sequence[DecisionRecord | DecisionRecordView], device: torch.device) -> DecisionBatch:
    if not records:
        raise ValueError("rollout has no student decisions")
    if isinstance(records, DecisionRecords):
        states = records.states
        meta = states.tensor(states.table.integers)
        delay, replan = meta[:, DELAY].contiguous(), meta[:, REPLAN].contiguous()
        remaining = states.tensor(states.table.remaining_device)
        remaining = torch.where(remaining < 0, delay + replan, remaining)
        return DecisionBatch(
            observation_frame_J=meta[:, CLOCK].contiguous(),
            decision_frame_J=meta[:, CLOCK].contiguous(),
            sampled_indices_JF=states.tensor(states.table.actions)[..., DECODE_GROUP_INDICES].flatten(1),
            committed_prefix_JD4=states.tensor(states.table.prefix)[..., DECODE_GROUP_INDICES],
            behavior_log_probs_JF=states.tensor(states.table.behavior),
            executed_factors_JF=executed_factor_mask(delay, replan, remaining),
            duration_J=states.tensor(states.table.duration_device),
            bootstrap_mask_J=~states.tensor(states.table.terminal_device),
            trace_mask_J=~states.tensor(states.table.boundary_device),
            kl_coefficient_J=states.tensor(states.table.coefficients),
            delay_J=delay,
            replan_J=replan,
            policy_version_J=meta[:, UPDATE].contiguous(),
            episode_id_J=meta[:, EPISODE].contiguous(),
            segment_id_J=meta[:, EPISODE].contiguous(),
        )

    def integers(values: Sequence[int]) -> Tensor:
        return torch.tensor(values, dtype=torch.long, device=device)

    delay = integers([r.state.player.timing.delay for r in records])
    replan = integers([r.state.player.timing.replan for r in records])
    prefixes = torch.zeros(len(records), 24, 4, dtype=torch.long)
    for index, record in enumerate(records):
        prefixes[index, : len(record.state.prefix)] = record.state.prefix[..., DECODE_GROUP_INDICES].cpu()
    # Only an end observed in this rollout masks execution. A live match's
    # queued tail trains in full, even if the next rollout ends before it runs.
    remaining = integers(
        [
            r.frames_remaining
            if r.frames_remaining is not None
            else r.state.player.timing.delay + r.state.player.timing.replan
            for r in records
        ]
    )
    return DecisionBatch(
        observation_frame_J=integers([r.state.clock for r in records]),
        decision_frame_J=integers([r.state.clock for r in records]),
        sampled_indices_JF=torch.stack([r.actions[..., DECODE_GROUP_INDICES].flatten() for r in records]).to(device),
        committed_prefix_JD4=prefixes.to(device, non_blocking=True),
        behavior_log_probs_JF=torch.stack([r.behavior for r in records]).to(device),
        executed_factors_JF=executed_factor_mask(delay, replan, remaining),
        duration_J=integers([len(r.rewards) for r in records]),
        bootstrap_mask_J=torch.tensor([not r.terminal for r in records], device=device),
        trace_mask_J=torch.tensor([not r.boundary for r in records], device=device),
        kl_coefficient_J=torch.tensor([math.exp(r.state.player.log_lambda) for r in records], device=device),
        delay_J=delay,
        replan_J=replan,
        policy_version_J=integers([r.state.window[0] for r in records]),
        episode_id_J=integers([r.state.episode.number for r in records]),
        segment_id_J=integers([r.state.episode.number for r in records]),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class FrameBatch(Sequence[SimFrame]):
    records: Tensor  # [lanes, bytes]
    actions: Tensor  # processed readback [lanes, 2, ACTION_DIM]
    columns: NDArray[np.uint32]
    rewards: NDArray[np.float32]
    terminal: NDArray[np.bool_]
    timeout: NDArray[np.bool_]
    host_actions: NDArray[np.float32]

    def __post_init__(self) -> None:
        count = len(self.rewards)
        if (
            self.records.shape != (count, native.FRAME_DTYPE.itemsize)
            or self.records.dtype != torch.uint8
            or self.actions.shape != (count, 2, ACTION_DIM)
            or self.actions.dtype != torch.float32
            or self.host_actions.shape != (count, 2, ACTION_DIM)
            or self.host_actions.dtype != np.float32
            or self.columns.shape != (count, len(native.OBSERVATION_FIELDS))
            or self.columns.dtype != np.uint32
            or self.columns.flags.writeable
            or self.rewards.shape != (count, 2)
            or self.rewards.dtype != np.float32
            or self.terminal.shape != (count,)
            or self.timeout.shape != (count,)
            or self.terminal.dtype != np.bool_
            or self.timeout.dtype != np.bool_
        ):
            raise ValueError("native frame batch schema mismatch")
        if not np.isfinite(self.host_actions).all() or not np.isfinite(self.rewards).all():
            raise ValueError("simulator actions and rewards must be finite")
        if np.any(self.terminal & self.timeout):
            raise ValueError("a match end cannot also be a time-out")

    def __len__(self) -> int:
        return len(self.rewards)

    @overload
    def __getitem__(self, index: int) -> SimFrame: ...
    @overload
    def __getitem__(self, index: slice) -> tuple[SimFrame, ...]: ...
    def __getitem__(self, index: int | slice) -> SimFrame | tuple[SimFrame, ...]:
        if isinstance(index, slice):
            return tuple(self[lane] for lane in range(*index.indices(len(self))))
        return SimFrame(
            NativeColumns(self.columns[index]),
            self.host_actions[index],
            (float(self.rewards[index, 0]), float(self.rewards[index, 1])),
            bool(self.terminal[index]),
            bool(self.timeout[index]),
            self.records[index],
            self.actions[index],
        )


class FrameStore:
    """Own contiguous observations. Only diagnostic views decode Python frames."""

    def __init__(self, capacity: int, matches: int, device: torch.device, fields: tuple[str, ...]) -> None:
        self.records = torch.zeros(capacity, matches, native.FRAME_DTYPE.itemsize, dtype=torch.uint8, device=device)
        self.actions = torch.zeros(capacity, matches, 2, ACTION_DIM, device=device)
        self.rewards = torch.zeros(capacity, matches, 2, device=device)
        self.flags = torch.zeros(capacity, matches, 2, device=device, dtype=torch.bool)
        self.fields = fields
        self.extra = {
            name: torch.zeros(capacity, matches, dtype=torch.float64, device=device)
            for name in fields
            if name not in NATIVE_COLUMN_LAYOUT
        }
        self._host: tuple[NDArray, NDArray, NDArray, NDArray, dict[str, NDArray]] | None = None
        self._views: dict[tuple[int, int], SimFrame] = {}

    def write(self, positions: Tensor, lanes: Tensor, batch: FrameBatch, source: Tensor) -> Tensor:
        self._host = None
        self._views.clear()
        selected = batch.records.index_select(0, source)
        self.records[positions, lanes] = selected
        self.actions[positions, lanes] = batch.actions.index_select(0, source)
        fields = native.FRAME_DTYPE.fields
        if fields is None:
            raise RuntimeError("native records must have a structured dtype")
        reward_offset = fields["reward"][1]
        flag_offset = fields["terminated"][1]
        rewards = selected[:, reward_offset : reward_offset + 8].contiguous().view(torch.float32)
        self.rewards[positions, lanes] = rewards
        self.flags[positions, lanes] = selected[:, flag_offset : flag_offset + 2].bool()
        return rewards

    def write_frame(self, position: int, lane: int, frame: SimFrame) -> None:
        self._host = None
        self._views.clear()
        if tuple(frame.columns) != self.fields:
            raise ValueError("simulator observation schema changed within the pool")
        packed = np.zeros(1, dtype=native.FRAME_DTYPE)
        for name, value in frame.columns.items():
            if name in NATIVE_COLUMN_LAYOUT:
                index, _ = NATIVE_COLUMN_LAYOUT[name]
                packed["columns"][0, index] = np.array([value], dtype=MDS_PER_FRAME_DTYPES[name]).view(np.uint32)[0]
            else:
                self.extra[name][position, lane] = value
        packed["applied_action"][0] = frame.applied_actions
        packed["reward"][0] = frame.rewards
        packed["terminated"][0] = frame.terminal
        packed["truncated"][0] = frame.timeout
        device = self.records.device
        source = (
            torch.from_numpy(packed.view(np.uint8)).to(device) if frame.device_record is None else frame.device_record
        )
        self.records[position, lane].copy_(source)
        self.actions[position, lane].copy_(torch.from_numpy(frame.applied_actions).to(device))
        self.rewards[position, lane] = torch.tensor(frame.rewards, device=device)
        self.flags[position, lane] = torch.tensor((frame.terminal, frame.timeout), device=device)

    def frame(self, position: int, lane: int) -> SimFrame:
        cached = self._views.get((position, lane))
        if cached is not None:
            return cached
        if self._host is None:
            self._host = (
                self.records.cpu().numpy().copy(),
                self.actions.cpu().numpy().copy(),
                self.rewards.cpu().numpy().copy(),
                self.flags.cpu().numpy().copy(),
                {name: value.cpu().numpy().copy() for name, value in self.extra.items()},
            )
        packed, actions, rewards, flags, extra = self._host
        words = packed[position, lane, : 4 * len(native.OBSERVATION_FIELDS)].copy().view(np.uint32)
        words.flags.writeable = False
        native_columns = NativeColumns(words)
        columns = {
            name: float(extra[name][position, lane]) if name in extra else native_columns[name] for name in self.fields
        }
        result = SimFrame(
            columns,
            actions[position, lane],
            (float(rewards[position, lane, 0]), float(rewards[position, lane, 1])),
            bool(flags[position, lane, 0]),
            bool(flags[position, lane, 1]),
            self.records[position, lane],
            self.actions[position, lane],
        )
        self._views[position, lane] = result
        return result


@dataclasses.dataclass(slots=True)
class FrameHistory(Sequence[SimFrame]):
    """Track a lane's retained span; decode frames only for diagnostics."""

    store: FrameStore
    lane: int
    start: int
    stop: int

    def __len__(self) -> int:
        return self.stop - self.start

    @overload
    def __getitem__(self, index: int) -> SimFrame: ...
    @overload
    def __getitem__(self, index: slice) -> FrameHistory: ...
    def __getitem__(self, index: int | slice) -> SimFrame | FrameHistory:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                raise ValueError("frame history only supports contiguous slices")
            return FrameHistory(self.store, self.lane, self.start + start, self.start + stop)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.store.frame(self.start + index, self.lane)


class ActionRing:
    """Store committed factors and mirror lengths on the host for scheduling."""

    def __init__(self, matches: int, capacity: int, device: torch.device) -> None:
        self.values = torch.zeros(matches, 2, capacity, 4, dtype=torch.long, device=device)
        self.controllers: Tensor | None = None
        self.head = torch.zeros(matches, 2, dtype=torch.long, device=device)
        self.length = np.zeros((matches, 2), dtype=np.int64)
        self.length_device = torch.zeros(matches, 2, dtype=torch.long, device=device)
        self.capacity = capacity
        self.offsets = torch.arange(capacity, device=device)

    def reset(self, lane: int, delays: tuple[int, int]) -> None:
        self.values[lane].zero_()
        self.head[lane].zero_()
        self.length[lane] = delays
        self.length_device[lane] = torch.tensor(delays, device=self.values.device)

    def read(self, lanes: Tensor, sides: Tensor, count: int) -> Tensor:
        positions = (self.head[lanes, sides, None] + self.offsets[:count]) % self.capacity
        return self.values[lanes[:, None], sides[:, None], positions]

    def append(
        self, host_lanes: NDArray[np.int64], host_sides: NDArray[np.int64], actions: Tensor, replans: NDArray[np.int64]
    ) -> None:
        lengths = self.length[host_lanes, host_sides]
        if np.any(lengths + replans > self.capacity):
            raise ValueError("action queue exceeded its capacity")
        lanes = torch.from_numpy(host_lanes.copy()).to(self.values.device)
        sides = torch.from_numpy(host_sides.copy()).to(self.values.device)
        lengths_device = torch.from_numpy(lengths).to(self.values.device)
        # Write each cadence together so the two-frame rows do not touch padding.
        for count in (2, 4):
            indices = np.flatnonzero(replans == count)
            if not len(indices):
                continue
            at = torch.from_numpy(indices).to(self.values.device)
            left, side = lanes[at], sides[at]
            positions = (self.head[left, side, None] + lengths_device[at, None] + self.offsets[:count]) % self.capacity
            self.values[left[:, None], side[:, None], positions] = actions[at, :count]
        self.length[host_lanes, host_sides] += replans
        self.length_device[lanes, sides] += torch.from_numpy(replans.copy()).to(self.values.device)

    def consume(self, host: NDArray[np.int64], *, device_lanes: Tensor | None = None) -> None:
        if np.any(self.length[host] < 1):
            raise ValueError("simulator step would consume an uncommitted action")
        lanes = torch.from_numpy(host).to(self.values.device) if device_lanes is None else device_lanes
        self.head[lanes] = (self.head[lanes] + 1) % self.capacity
        self.length[host] -= 1
        self.length_device[lanes] -= 1


@dataclasses.dataclass(frozen=True, slots=True)
class QueueView(Sequence[Tensor]):
    ring: ActionRing
    lane: int
    side: int

    def __len__(self) -> int:
        return int(self.ring.length[self.lane, self.side])

    @overload
    def __getitem__(self, index: int) -> Tensor: ...
    @overload
    def __getitem__(self, index: slice) -> list[Tensor]: ...
    def __getitem__(self, index: int | slice) -> Tensor | list[Tensor]:
        device = self.ring.values.device
        values = self.ring.read(
            torch.tensor([self.lane], device=device), torch.tensor([self.side], device=device), len(self)
        )[0]
        if isinstance(index, slice):
            return list(values[index].unbind())
        return values[index]

    def clear(self) -> None:
        self.ring.length[self.lane, self.side] = 0
        self.ring.length_device[self.lane, self.side] = 0


# Integer table columns. The host mirror drives schedules and counter keys;
# tensor columns drive context and queue gathers without observation readback.
(
    STREAM,
    MATCH,
    SIDE,
    CLOCK,
    UPDATE,
    BLOCK,
    EPISODE,
    LANE,
    START,
    LENGTH,
    IDENTITY,
    CHARACTER,
    STAGE,
    DELAY,
    REPLAN,
    OPP_DELAY,
    OPP_REPLAN,
    OPP_LENGTH,
) = range(18)


def _window_key(row: NDArray[np.int64]) -> tuple[int, int, int, int]:
    return int(row[UPDATE]), int(row[STREAM]), int(row[BLOCK]), int(row[EPISODE])


class DecisionTable:
    """Store sampled decisions and bootstrap states; records select student rows."""

    def __init__(self, capacity: int, store: FrameStore) -> None:
        self.store = store
        self.device = store.records.device
        self.meta = np.zeros((capacity, 18), dtype=np.int64)
        self.integers = torch.zeros(capacity, 18, dtype=torch.long, device=self.device)
        self.log_lambda = np.zeros((capacity, 2), dtype=np.float64)
        self.conditions = torch.zeros(capacity, 2, device=self.device)
        self.coefficients = torch.zeros(capacity, device=self.device)
        self.prefix = torch.zeros(capacity, 24, 4, dtype=torch.long, device=self.device)
        self.opponent = torch.zeros_like(self.prefix)
        self.actions = torch.zeros(capacity, 4, 4, dtype=torch.long, device=self.device)
        self.behavior = torch.zeros(capacity, 16, device=self.device)
        self.rewards = torch.zeros(capacity, 4, device=self.device)
        self.duration = np.zeros(capacity, dtype=np.int64)
        self.remaining = np.full(capacity, -1, dtype=np.int64)
        self.terminal = np.zeros(capacity, dtype=np.bool_)
        self.boundary = np.zeros(capacity, dtype=np.bool_)
        self.following = np.full(capacity, -1, dtype=np.int64)
        self.duration_device = torch.zeros(capacity, dtype=torch.long, device=self.device)
        self.remaining_device = torch.full((capacity,), -1, dtype=torch.long, device=self.device)
        self.terminal_device = torch.zeros(capacity, dtype=torch.bool, device=self.device)
        self.boundary_device = torch.zeros(capacity, dtype=torch.bool, device=self.device)
        self.following_device = torch.full((capacity,), -1, dtype=torch.long, device=self.device)
        self.episodes: dict[tuple[int, int], Episode] = {}
        self.episode_slots = np.zeros(capacity, dtype=np.int64)
        self.episode_ends = np.full(capacity, -1, dtype=np.int64)
        self.episode_indices: dict[tuple[int, int], int] = {}
        self.count = 0
        self.evaluation: PolicyEvaluation | None = None
        self.values = torch.zeros(capacity, device=self.device)
        self.tick_active = torch.zeros(capacity, device=self.device, dtype=torch.bool)
        self.valid = torch.zeros(capacity, device=self.device, dtype=torch.bool)
        self.queue_error = torch.zeros((), device=self.device, dtype=torch.bool)

    def allocate_evaluation(self, model: SelfPlayTransformer) -> None:
        capacity = len(self.meta)
        self.evaluation = PolicyEvaluation(
            self.actions,
            self.behavior,
            tuple(
                torch.zeros(capacity, 4, model.codec.group_vocabs[group], device=self.device)
                for group in DECODE_GROUP_INDICES
            ),
            CriticFeatures(
                torch.zeros(capacity, model.cfg.d_model, device=self.device),
                torch.zeros(capacity, model.cfg.temporal_d_model, device=self.device),
                torch.zeros(capacity, 3, device=self.device),
                torch.zeros(capacity, 3, device=self.device),
                torch.zeros(capacity, model.critic_residual.max_queue, 4, dtype=torch.long, device=self.device),
                torch.zeros(capacity, dtype=torch.long, device=self.device),
            ),
        )

    def reset(self) -> None:
        self.valid.zero_()
        self.tick_active.zero_()
        self.duration_device.zero_()
        self.remaining_device.fill_(-1)
        self.terminal_device.zero_()
        self.boundary_device.zero_()
        self.following_device.fill_(-1)
        self.queue_error.zero_()

    def record_evaluation(self, ids: Tensor, output: PolicyEvaluation, values: Tensor) -> None:
        if self.evaluation is None:

            def storage(value: Tensor) -> Tensor:
                return value.new_zeros((len(self.meta), *value.shape[1:]))

            self.evaluation = PolicyEvaluation(
                self.actions,
                self.behavior,
                tuple(storage(p) for p in output.distributions),
                CriticFeatures(*(storage(getattr(output.features, name)) for name in CRITIC_FEATURE_NAMES)),
            )
        self.actions[ids] = output.actions
        self.behavior[ids] = output.log_probs
        self.values[ids] = values
        for target, source in zip(self.evaluation.distributions, output.distributions, strict=True):
            target[ids] = source
        for name in CRITIC_FEATURE_NAMES:
            target = getattr(self.evaluation.features, name)
            target[ids] = getattr(output.features, name).to(target.dtype)

    def rows(self, ids: Sequence[int] | NDArray[np.int64]) -> DecisionRows:
        return DecisionRows(self, np.asarray(ids, dtype=np.int64))

    def publish(self) -> None:
        self.meta[:] = self.integers.cpu().numpy()
        self.log_lambda[:] = self.conditions.cpu().numpy()
        for target, source in (
            (self.duration, self.duration_device),
            (self.remaining, self.remaining_device),
            (self.terminal, self.terminal_device),
            (self.boundary, self.boundary_device),
            (self.following, self.following_device),
        ):
            target[:] = source.cpu().numpy()

    def state(self, index: int, metadata: NDArray[np.int64] | None = None) -> DecisionState:
        row = self.meta[index] if metadata is None else metadata
        episode = self.episodes[int(row[MATCH]), int(row[EPISODE])]
        history = FrameHistory(self.store, int(row[LANE]), int(row[START]), int(row[START] + row[LENGTH]))
        return DecisionState(
            int(row[STREAM]),
            int(row[MATCH]),
            int(row[SIDE]),
            episode,
            int(row[CLOCK]),
            _window_key(row),
            history,
            self.prefix[index, : int(row[DELAY])],
            self.opponent[index, : int(row[OPP_LENGTH])],
        )

    def add(self, meta: list[list[int]], episodes: list[Episode], ring: ActionRing) -> DecisionRows:
        end = self.count + len(meta)
        if end > len(self.meta):
            raise ValueError("decision table capacity exhausted")
        ids = np.arange(self.count, end)
        self.meta[ids] = meta
        data = self.meta[ids]
        at = torch.from_numpy(ids).to(self.device)
        integers = torch.from_numpy(data).to(self.device)
        self.integers[at] = integers
        logs = []
        coefficients = []
        for index, (row, episode) in enumerate(zip(data, episodes, strict=True)):
            side = int(row[SIDE])
            key = (int(row[MATCH]), episode.number)
            self.episodes[key] = episode
            self.episode_slots[ids[index]] = self.episode_indices.setdefault(key, len(self.episode_indices))
            logs.append((episode.players[side].log_lambda, episode.players[1 - side].log_lambda))
            coefficients.append(math.exp(episode.players[side].log_lambda))
        self.log_lambda[ids] = logs
        self.conditions[at] = torch.tensor(logs, device=self.device)
        self.coefficients[at] = torch.tensor(coefficients, device=self.device)
        lanes, sides = integers[:, LANE], integers[:, SIDE]
        offsets = torch.arange(24, device=self.device)[None]
        count = min(24, ring.capacity)
        prefix = torch.nn.functional.pad(ring.read(lanes, sides, count), (0, 0, 0, 24 - count))
        opponent = torch.nn.functional.pad(ring.read(lanes, 1 - sides, count), (0, 0, 0, 24 - count))
        self.prefix[at] = torch.where((offsets < integers[:, DELAY, None])[..., None], prefix, 0)
        self.opponent[at] = torch.where((offsets < integers[:, OPP_LENGTH, None])[..., None], opponent, 0)
        self.count = end
        return self.rows(ids)


@dataclasses.dataclass(frozen=True, slots=True)
class DecisionRows(Sequence[DecisionState]):
    """Select rows without copying their stored queues or observation histories."""

    table: DecisionTable
    ids: NDArray[np.int64]
    metadata: NDArray[np.int64] | None = None
    device_ids: Tensor = dataclasses.field(init=False, repr=False)
    device_metadata: Tensor | None = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "device_ids", torch.from_numpy(self.ids).to(self.table.device, non_blocking=True))
        object.__setattr__(
            self,
            "device_metadata",
            None
            if self.metadata is None
            else torch.from_numpy(self.metadata).to(self.table.device, non_blocking=True),
        )

    @property
    def meta(self) -> NDArray[np.int64]:
        return self.table.meta[self.ids] if self.metadata is None else self.metadata

    def tensor(self, value: Tensor) -> Tensor:
        if self.device_metadata is not None and value is self.table.integers:
            return self.device_metadata
        return value.index_select(0, self.device_ids)

    def take(self, indices: Sequence[int] | NDArray[np.int64]) -> DecisionRows:
        at = np.asarray(indices, dtype=np.int64)
        return DecisionRows(self.table, self.ids[at], None if self.metadata is None else self.metadata[at])

    def tail(self, count: int) -> DecisionRows:
        meta = self.meta.copy()
        lengths = np.minimum(meta[:, LENGTH], count)
        meta[:, START] += meta[:, LENGTH] - lengths
        meta[:, LENGTH] = lengths
        meta[:, BLOCK] = meta[:, CLOCK]
        return DecisionRows(self.table, self.ids, meta)

    def __len__(self) -> int:
        return len(self.ids)

    @overload
    def __getitem__(self, index: int) -> DecisionState: ...
    @overload
    def __getitem__(self, index: slice) -> DecisionRows: ...
    def __getitem__(self, index: int | slice) -> DecisionState | DecisionRows:
        if isinstance(index, slice):
            return self.take(np.arange(len(self))[index])
        at = int(self.ids[index])
        return self.table.state(at, self.meta[index])

    def packed(self, length: int, tail: int | None = None) -> tuple[Tensor, ...]:
        meta = self.tensor(self.table.integers)
        lengths = meta[:, LENGTH] if tail is None else meta[:, LENGTH].clamp_max(tail)
        starts = meta[:, START] + meta[:, LENGTH] - lengths
        valid = torch.arange(length, device=meta.device)[None] < lengths[:, None]
        positions = starts[:, None] + torch.arange(length, device=meta.device)[None]
        # Padding may extend past a lane's allocation at the final boundary.
        positions = torch.where(valid, positions, 0)
        flat = (positions * self.table.store.records.shape[1] + meta[:, LANE, None]).flatten()
        records = self.table.store.records.flatten(0, 1).index_select(0, flat).reshape(len(self), length, -1)
        actions = (
            self.table.store.actions.flatten(0, 1).index_select(0, flat).reshape(len(self), length, 2, ACTION_DIM)
        )
        records = torch.where(valid[..., None], records, 0)
        actions = torch.where(valid[..., None, None], actions, 0)
        return records, actions, meta[:, SIDE], meta[:, IDENTITY], lengths


class DecisionRecordView:
    """Expose diagnostic fields without retaining per-decision Python state."""

    def __init__(self, table: DecisionTable, row: int) -> None:
        self.table = table
        self.row = row

    @property
    def state(self) -> DecisionState:
        return self.table.state(self.row)

    @property
    def actions(self) -> Tensor:
        return self.table.actions[self.row]

    @property
    def behavior(self) -> Tensor:
        return self.table.behavior[self.row]

    @property
    def rewards(self) -> list[float]:
        return self.table.rewards[self.row, : int(self.table.duration[self.row])].cpu().tolist()

    @property
    def terminal(self) -> bool:
        return bool(self.table.terminal[self.row])

    @property
    def boundary(self) -> bool:
        return bool(self.table.boundary[self.row])

    @property
    def frames_remaining(self) -> int | None:
        value = int(self.table.remaining[self.row])
        return None if value == -1 else value

    @frames_remaining.setter
    def frames_remaining(self, value: int) -> None:
        self.table.remaining[self.row] = value
        self.table.remaining_device[self.row] = value

    @property
    def next_state(self) -> DecisionState | None:
        following = int(self.table.following[self.row])
        return None if following == -1 else self.table.state(following)


@dataclasses.dataclass(frozen=True, slots=True)
class DecisionRecords(Sequence[DecisionRecordView]):
    states: DecisionRows

    def __len__(self) -> int:
        return len(self.states)

    @overload
    def __getitem__(self, index: int) -> DecisionRecordView: ...
    @overload
    def __getitem__(self, index: slice) -> DecisionRecords: ...
    def __getitem__(self, index: int | slice) -> DecisionRecordView | DecisionRecords:
        if isinstance(index, slice):
            return DecisionRecords(self.states[index])
        return DecisionRecordView(self.states.table, int(self.states.ids[index]))


@dataclasses.dataclass(frozen=True, slots=True)
class PoolMatchSnapshot:
    match: int
    episode: Episode
    episode_start: int
    history_start: int
    history_length: int
    returns: tuple[float, float]
    end_clock: int | None


@dataclasses.dataclass(frozen=True, slots=True)
class PoolGroupSnapshot:
    fields: tuple[str, ...]
    records: Tensor
    actions: Tensor
    rewards: Tensor
    flags: Tensor
    extra: dict[str, Tensor]
    queue: Tensor
    head: Tensor
    length: Tensor
    matches: tuple[PoolMatchSnapshot, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class PoolSnapshot:
    format: str
    groups: tuple[PoolGroupSnapshot, ...]


class RolloutEngine:
    def __init__(
        self,
        groups: tuple[tuple[SimulatorShard, ...], ...],
        policy: LoopPolicy,
        sampler: EpisodeSampler,
        anchors: dict[int, LoopPolicy],
        *,
        fixed_anchor_lanes: bool = False,
    ) -> None:
        if not groups or any(not group for group in groups):
            raise ValueError("rollout groups and shards must be nonempty")
        if type(fixed_anchor_lanes) is not bool:
            raise ValueError("fixed_anchor_lanes must be a boolean")
        ids = [match for group in groups for shard in group for match in shard.match_ids]
        if not ids or len(set(ids)) != len(ids) or min(ids) < 0:
            raise ValueError("logical match ids must be nonempty, nonnegative, and globally unique")
        if 0 not in anchors:
            raise ValueError("anchor zero must be the frozen reference")
        student_parameters = {id(parameter) for parameter in policy.model.parameters()}
        for anchor in anchors.values():
            if any(parameter.requires_grad for parameter in anchor.model.parameters()) or any(
                id(parameter) in student_parameters for parameter in anchor.model.parameters()
            ):
                raise ValueError("anchors must be frozen and have independent parameters")
        self.profile: UpdateProfile | None = None
        self.groups = groups
        self.policy = policy
        self.sampler = sampler
        self.anchors = anchors
        self.fixed_anchor_lanes = fixed_anchor_lanes
        self.tick_layouts: dict[tuple[int, int | None], TickLayout] = {}
        self.group_layouts: list[dict[tuple[int, int | None], TickLayout]] = [{} for _ in groups]
        self.tick_layout_clock: tuple[int, int] | None = None
        self.anchor_candidates = tuple(sorted(anchors))
        self.clocks = [0] * len(groups)
        self.cursor = 0
        self.matches: dict[int, MatchState] = {}
        self._initialized = False
        self.completed_update = -1
        self.results: list[EpisodeResult] = []
        self.group_matches = tuple(tuple(match for shard in group for match in shard.match_ids) for group in groups)
        capacity = max(t.delay + t.replan for t in sampler.schedule.configurations)
        self.rings = tuple(ActionRing(len(matches), capacity, policy.device) for matches in self.group_matches)
        if isinstance(policy, TransformerPolicy):
            for ring in self.rings:
                ring.controllers = policy.device_wire_actions(ring.values)
        self.anchor_counts = tuple(
            math.floor(len(matches) * sampler.config.anchor_rate + 0.5) for matches in self.group_matches
        )
        self.locations = {
            match: (group, lane)
            for group, matches in enumerate(self.group_matches)
            for lane, match in enumerate(matches)
        }
        self.shard_lanes = {
            id(shard): np.array([self.locations[match][1] for match in shard.match_ids], dtype=np.int64)
            for group in groups
            for shard in group
        }
        buffers: list[tuple[RolloutBuffer, ...]] = []
        for shards in groups:
            split = (len(shards) + 1) // 2
            prepared: list[RolloutBuffer] = []
            for index, half in enumerate((shards[:split], shards[split:])):
                if not half:
                    continue
                lanes = np.concatenate([self.shard_lanes[id(shard)] for shard in half])
                lanes.flags.writeable = False
                stream = torch.cuda.Stream(device=policy.device) if policy.device.type == "cuda" else None
                prepared.append(RolloutBuffer(index, half, lanes, stream))
            buffers.append(tuple(prepared))
        self.buffers = tuple(buffers)
        self.group_slots: list[dict[tuple[int, int | None], NDArray[np.int64]]] = [{} for _ in groups]
        self.stores: list[FrameStore | None] = [None] * len(groups)
        self.dense_meta = [
            torch.zeros(2 * len(ids), 18, dtype=torch.long, device=policy.device) for ids in self.group_matches
        ]
        self.dense_logs = [torch.zeros(2 * len(ids), 2, device=policy.device) for ids in self.group_matches]
        self.stops = [torch.zeros(len(ids), dtype=torch.long, device=policy.device) for ids in self.group_matches]
        self.episode_clocks = [torch.zeros_like(value) for value in self.stops]
        self.observed_clocks = [torch.zeros_like(value) for value in self.stops]
        self.active = [torch.ones(len(ids), dtype=torch.bool, device=policy.device) for ids in self.group_matches]
        self.live_host = [np.ones(len(ids), dtype=np.bool_) for ids in self.group_matches]
        self.assignments = [np.full(2 * len(ids), -1, dtype=np.int64) for ids in self.group_matches]
        self.owners = [
            torch.full((2 * len(ids),), -1, dtype=torch.long, device=policy.device) for ids in self.group_matches
        ]
        self.layout_assignments = [np.full_like(values, -2) for values in self.assignments]
        self._table: DecisionTable | None = None
        self.tables: list[DecisionTable | None] = [None] * len(groups)
        self.episode_returns = [torch.zeros(len(ids), 2, device=policy.device) for ids in self.group_matches]

    def snapshot(self) -> PoolSnapshot:
        groups = []
        for group_id, matches in enumerate(self.group_matches):
            store = self.stores[group_id]
            if store is None:
                raise ValueError("pool has no frame storage")
            # Checkpoints contain only the retained context, without the rollout's
            # decision table or diagnostic copies of observations.
            packed = FrameStore(129, len(matches), self.policy.device, store.fields)
            metadata = []
            for lane, match in enumerate(matches):
                state = self.matches[match]
                history = state.history
                length = len(history)
                span = slice(history.start, history.stop)
                for target, source in (
                    (packed.records, store.records),
                    (packed.actions, store.actions),
                    (packed.rewards, store.rewards),
                    (packed.flags, store.flags),
                ):
                    target[:length, lane].copy_(source[span, lane])
                for name in packed.extra:
                    packed.extra[name][:length, lane].copy_(store.extra[name][span, lane])
                metadata.append(
                    PoolMatchSnapshot(
                        match,
                        state.episode,
                        state.episode_start,
                        state.history_start,
                        length,
                        state.returns,
                        state.end_clock,
                    )
                )
            ring = self.rings[group_id]
            if not torch.equal(ring.length_device.cpu(), torch.from_numpy(ring.length)):
                raise ValueError("action queue length mirrors disagree")
            groups.append(
                PoolGroupSnapshot(
                    store.fields,
                    packed.records.cpu(),
                    packed.actions.cpu(),
                    packed.rewards.cpu(),
                    packed.flags.cpu(),
                    {name: value.cpu() for name, value in packed.extra.items()},
                    ring.values.to(device="cpu", copy=True),
                    ring.head.to(device="cpu", copy=True),
                    ring.length_device.to(device="cpu", copy=True),
                    tuple(metadata),
                )
            )
        return PoolSnapshot("hal.o65.rollout_pool.v2", tuple(groups))

    def restore_pool(self, snapshot: PoolSnapshot) -> None:
        if not isinstance(snapshot, PoolSnapshot) or snapshot.format != "hal.o65.rollout_pool.v2":
            raise ValueError("unsupported rollout pool snapshot identity")
        if len(snapshot.groups) != len(self.groups):
            raise ValueError("rollout pool topology mismatch")
        restored: dict[int, MatchState] = {}
        for group_id, (saved, ids) in enumerate(zip(snapshot.groups, self.group_matches, strict=True)):
            if tuple(match.match for match in saved.matches) != ids or len(set(saved.fields)) != len(saved.fields):
                raise ValueError("rollout pool match or schema mismatch")
            store = FrameStore(129, len(ids), self.policy.device, saved.fields)
            ring = self.rings[group_id]
            pairs = (
                (store.records, saved.records),
                (store.actions, saved.actions),
                (store.rewards, saved.rewards),
                (store.flags, saved.flags),
                (ring.values, saved.queue),
                (ring.head, saved.head),
                (ring.length_device, saved.length),
            )
            for target, source in pairs:
                if target.shape != source.shape or target.dtype != source.dtype:
                    raise ValueError("rollout pool tensor schema mismatch")
                target.copy_(source)
            if (
                bool(((saved.head < 0) | (saved.head >= ring.capacity)).any())
                or bool(((saved.length < 0) | (saved.length > ring.capacity)).any())
                or set(saved.extra) != set(store.extra)
            ):
                raise ValueError("invalid rollout pool queues or columns")
            ring.length[:] = saved.length.cpu().numpy()
            if ring.controllers is not None:
                if not isinstance(self.policy, TransformerPolicy):
                    raise ValueError("controller storage requires a transformer policy")
                ring.controllers.copy_(self.policy.device_wire_actions(ring.values))
            for name, target in store.extra.items():
                source = saved.extra[name]
                if target.shape != source.shape or target.dtype != source.dtype:
                    raise ValueError("rollout pool extra column schema mismatch")
                target.copy_(source)
            for lane, match in enumerate(saved.matches):
                if not 1 <= match.history_length <= 129:
                    raise ValueError("invalid rollout pool history length")
                restored[match.match] = MatchState(
                    match.episode,
                    match.episode_start,
                    match.history_start,
                    FrameHistory(store, lane, 0, match.history_length),
                    (QueueView(ring, lane, 0), QueueView(ring, lane, 1)),
                    match.returns,
                    match.end_clock,
                )
            self.stores[group_id] = store
        self.matches = restored
        self.tables = [None] * len(self.groups)
        # Captured tick writes own the old table's addresses.
        for policy in (self.policy, *self.anchors.values()):
            if isinstance(policy, TransformerPolicy):
                policy.tick_graphs.clear()
                policy.tick_caches.clear()
                policy.current_layouts.clear()
                policy.window_graphs.clear()
                policy.value_graphs.clear()
                policy.actor_graph = None
        for group_id, matches in enumerate(self.group_matches):
            for lane, match in enumerate(matches):
                state = self.matches[match]
                self._publish_episode(group_id, lane, match, state.history.start)
                self.stops[group_id][lane] = state.history.stop
                self.observed_clocks[group_id][lane] = state.history_start + len(state.history) - 1
                self.active[group_id][lane] = state.end_clock is None
                self.live_host[group_id][lane] = state.end_clock is None
                self.episode_returns[group_id][lane] = torch.tensor(state.returns, device=self.policy.device)

    def initialize(self) -> None:
        if self._initialized:
            raise ValueError("pool is already initialized")
        for group in self.groups:
            for shard in group:
                for lane, match in enumerate(shard.match_ids):
                    self._reset(shard, lane, match, 0, 0)
        self._initialized = True

    def _reset(self, shard: SimulatorShard, lane: int, match: int, clock: int, number: int) -> None:
        if clock % 128:
            raise ValueError("episodes must start on a block boundary")
        episode = self.sampler.sample(match, number, self.anchor_candidates)
        if self.fixed_anchor_lanes:
            group, lane_index = self.locations[match]
            side = lane_index % 2 if lane_index < self.anchor_counts[group] else None
            players = []
            for player_side, player in enumerate(episode.players):
                anchor = None
                if player_side == side:
                    stream = 2 * match + player_side
                    choice = self.sampler.rng.choice([1.0] * len(self.anchor_candidates), "anchor", stream, number, 0)
                    anchor = self.anchor_candidates[choice]
                players.append(dataclasses.replace(player, anchor=anchor))
            episode = dataclasses.replace(episode, players=(players[0], players[1]))
        if isinstance(self.policy, TransformerPolicy):
            for player in episode.players:
                if not 0 <= player.identity < self.policy.model.cfg.player_vocab_size:
                    raise ValueError("episode identity is outside the checkpoint vocabulary")
                if player.timing not in self.policy.schedule.configurations:
                    raise ValueError("episode uses an unsupported decision configuration")
                low, high = self.policy.model.ranges.log_lambda
                if not low <= player.log_lambda <= high:
                    raise ValueError("episode conditioning is outside its declared range")
        frame = shard.reset(lane, episode)
        if frame.terminal or frame.timeout or bool(frame.applied_actions.any()):
            raise ValueError("reset must return a live initial state with neutral applied inputs")
        group, at = self.locations[match]
        store = self.stores[group]
        if store is None:
            store = FrameStore(1, len(self.group_matches[group]), self.policy.device, tuple(frame.columns))
            self.stores[group] = store
        previous = self.matches.get(match)
        position = int(self.stops[group][at]) if previous is not None else 0
        if position >= len(store.records):
            raise ValueError("frame store capacity exhausted during reset")
        store.write_frame(position, at, frame)
        ring = self.rings[group]
        ring.reset(at, (episode.players[0].timing.delay, episode.players[1].timing.delay))
        if ring.controllers is not None:
            if not isinstance(self.policy, TransformerPolicy):
                raise ValueError("controller storage requires a transformer policy")
            ring.controllers[at].copy_(self.policy.device_wire_actions(ring.values[at]))
        self.matches[match] = MatchState(
            episode,
            clock,
            clock,
            FrameHistory(store, at, position, position + 1),
            (QueueView(ring, at, 0), QueueView(ring, at, 1)),
        )
        self._publish_episode(group, at, match, position)

    def _publish_episode(self, group: int, lane: int, match: int, position: int) -> None:
        state = self.matches[match]
        episode = state.episode
        rows = []
        for side, player in enumerate(episode.players):
            opponent = episode.players[1 - side]
            rows.append(
                [
                    2 * match + side,
                    match,
                    side,
                    state.episode_start,
                    0,
                    state.episode_start,
                    episode.number,
                    lane,
                    position,
                    1,
                    player.identity,
                    player.character,
                    episode.stage,
                    player.timing.delay,
                    player.timing.replan,
                    opponent.timing.delay,
                    opponent.timing.replan,
                    opponent.timing.delay,
                ]
            )
            self.assignments[group][2 * lane + side] = -1 if player.anchor is None else player.anchor
        at = slice(2 * lane, 2 * lane + 2)
        self.dense_meta[group][at] = torch.tensor(rows, device=self.policy.device)
        self.owners[group][at] = torch.from_numpy(self.assignments[group][at].copy()).to(self.policy.device)
        logs = tuple(player.log_lambda for player in episode.players)
        self.dense_logs[group][at] = torch.tensor((logs, logs[::-1]), device=self.policy.device)
        self.stops[group][lane] = position + 1
        self.episode_clocks[group][lane] = state.episode_start
        self.observed_clocks[group][lane] = state.episode_start
        self.active[group][lane] = True
        self.live_host[group][lane] = True
        self.episode_returns[group][lane].zero_()
        if self._table is not None:
            self._table.episodes[match, episode.number] = episode

    def _dense_metadata(self, clock: int, update: int, *, bootstrap: bool = False) -> Tensor:
        group = self.cursor
        meta = self.dense_meta[group].clone()
        observed = self.observed_clocks[group].repeat_interleave(2)
        clocks = observed if bootstrap else torch.full_like(observed, clock)
        block = ((clocks - int(bootstrap)) // 128) * 128
        start = torch.maximum(self.episode_clocks[group].repeat_interleave(2), block - 128)
        start = torch.maximum(start, clocks - 255)
        # Inactive rows still occupy slots. Their observations stop at the end.
        length = (observed - start + 1).clamp(1, 256)
        meta[:, CLOCK] = clocks
        meta[:, UPDATE] = update
        meta[:, BLOCK] = block
        meta[:, START] = self.stops[group].repeat_interleave(2) - length
        meta[:, LENGTH] = length
        meta[:, OPP_LENGTH] = self.rings[group].length_device.flip(1).flatten()
        return meta

    def _metadata(self, match: int, side: int, clock: int, update: int, *, bootstrap: bool = False) -> list[int]:
        if type(side) is not int or side not in (0, 1):
            raise ValueError("decision side must be HAL port 1 or 2's zero-based index")
        state = self.matches[match]
        block = (clock // 128) * 128
        if bootstrap and clock % 128 == 0:
            block -= 128
        start = max(state.episode_start, block - 128, clock - 255)
        history_start = state.history.start + start - state.history_start
        length = clock - start + 1
        if (
            not 1 <= length <= 256
            or history_start < state.history.start
            or history_start + length > state.history.stop
        ):
            raise ValueError("invalid rollout history span")
        group, lane = self.locations[match]
        player, opponent = state.episode.players[side], state.episode.players[1 - side]
        if self.rings[group].length[lane, side] < player.timing.delay:
            raise ValueError("committed queue is shorter than D")
        return [
            2 * match + side,
            match,
            side,
            clock,
            update,
            block,
            state.episode.number,
            lane,
            history_start,
            length,
            player.identity,
            player.character,
            state.episode.stage,
            player.timing.delay,
            player.timing.replan,
            opponent.timing.delay,
            opponent.timing.replan,
            int(self.rings[group].length[lane, 1 - side]),
        ]

    def _state(self, match: int, side: int, clock: int, update: int, *, bootstrap: bool = False) -> DecisionState:
        meta = self._metadata(match, side, clock, update, bootstrap=bootstrap)
        group, _ = self.locations[match]
        store = self.stores[group]
        if store is None:
            raise ValueError("pool has no frame storage")
        table = DecisionTable(1, store)
        return table.add([meta], [self.matches[match].episode], self.rings[group])[0]

    def _stage_tick(
        self,
        clock: int,
        update: int,
        table: DecisionTable,
        pending: Tensor,
    ) -> tuple[NDArray[np.int64], Tensor, Tensor]:
        group = self.cursor
        players = len(self.dense_meta[group])
        tick = (clock - self.clocks[group]) // 2
        ids = np.arange(tick * players, (tick + 1) * players)
        at = torch.from_numpy(ids).to(self.policy.device)
        meta = self._dense_metadata(clock, update)
        table.integers[at] = meta
        table.remaining_device[at] = -1
        table.conditions[at] = self.dense_logs[group]
        table.coefficients[at] = self.dense_logs[group][:, 0].exp()
        ring = self.rings[group]
        lanes, sides = meta[:, LANE], meta[:, SIDE]
        offsets = torch.arange(24, device=self.policy.device)[None]
        queue = ring.read(lanes, sides, min(24, ring.capacity))
        queue = torch.nn.functional.pad(queue, (0, 0, 0, 24 - queue.shape[1]))
        table.prefix[at] = torch.where((offsets < meta[:, DELAY, None])[..., None], queue, 0)
        opponent = queue.reshape(-1, 2, 24, 4).flip(1).flatten(0, 1)
        table.opponent[at] = torch.where((offsets < meta[:, OPP_LENGTH, None])[..., None], opponent, 0)
        active = self.active[group].repeat_interleave(2)
        table.tick_active[at] = active
        due = active & (clock % meta[:, REPLAN] == 0)
        table.queue_error.logical_or_((due & (ring.length_device.flatten() != meta[:, DELAY])).any())
        assignments = self.assignments[group]
        student = self.owners[group] < 0
        table.valid[at] = due & student
        previous = torch.where(pending >= 0, pending, torch.arange(players, device=self.policy.device))
        continuing = due & student & (pending >= 0)
        table.following_device[previous] = torch.where(continuing, at, table.following_device[previous])
        pending.copy_(torch.where(due & student, at, pending))
        if clock % 128 == 0 and not np.array_equal(assignments, self.layout_assignments[group]):
            self.tick_layouts.clear()
            self.group_slots[group].clear()
            self.layout_assignments[group][:] = assignments
            logical = np.asarray(self.group_matches[group], dtype=np.int64).repeat(2) * 2 + np.tile(
                (0, 1), len(self.group_matches[group])
            )
            for buffer in self.buffers[group]:
                in_buffer = np.isin(np.arange(players) // 2, buffer.lanes)
                for owner in np.unique(assignments[in_buffer]):
                    selected = np.flatnonzero(in_buffer & (assignments == owner))
                    anchor = None if owner < 0 else int(owner)
                    key = buffer.index, anchor
                    self.group_slots[group][key] = selected
                    self.tick_layouts[key] = TickLayout(
                        tuple(logical[selected].tolist()),
                        tuple((selected // 2).tolist()),
                        buffer.index,
                    )
        return ids, meta, due

    def _sample_buffer(
        self,
        buffer: RolloutBuffer,
        clock: int,
        table: DecisionTable,
        ids: NDArray[np.int64],
        meta: Tensor,
        due: Tensor,
    ) -> None:
        group = self.cursor
        ring = self.rings[group]
        for key, layout in self.tick_layouts.items():
            buffer_index, anchor = key
            if buffer_index != buffer.index:
                continue
            slots = self.group_slots[group][key]
            rows = table.rows(ids[slots])
            policy = self.policy if anchor is None else self.anchors[anchor]
            with torch.no_grad():
                if isinstance(policy, TransformerPolicy):
                    output = policy.sample_tick(rows, self.sampler.rng, layout, ring, clock=clock)
                else:
                    table.meta[ids] = meta.cpu().numpy()
                    output = policy.sample(rows, self.sampler.rng)
                    chosen = rows.tensor(table.integers)
                    # The small test policy executes the same dense queues.
                    for count in (2, 4):
                        if count > ring.capacity:
                            continue
                        take = torch.nonzero(
                            due[torch.from_numpy(slots).to(self.policy.device)] & (chosen[:, REPLAN] == count)
                        ).flatten()
                        target_lanes, target_sides = chosen[take, LANE], chosen[take, SIDE]
                        positions = (
                            ring.head[target_lanes, target_sides, None]
                            + ring.length_device[target_lanes, target_sides, None]
                            + ring.offsets[:count]
                        ) % ring.capacity
                        ring.values[target_lanes[:, None], target_sides[:, None], positions] = output.actions[
                            take, :count
                        ]
                        ring.length_device[target_lanes, target_sides] += count
                values = (
                    policy.last_tick_values
                    if isinstance(policy, TransformerPolicy)
                    else policy.values(output.features)
                )
                if values is None:
                    raise ValueError("tick graph did not produce critic values")
                if not isinstance(policy, TransformerPolicy):
                    table.record_evaluation(rows.device_ids, output, values)

    def _tick_ready(self, buffer: RolloutBuffer, clock: int) -> bool:
        if self.policy.device.type != "cuda":
            return True
        ring = self.rings[self.cursor]
        for (buffer_index, anchor), layout in self.tick_layouts.items():
            if buffer_index != buffer.index:
                continue
            policy = self.policy if anchor is None else self.anchors[anchor]
            if not isinstance(policy, TransformerPolicy):
                continue
            key = ring, (buffer.index, len(layout.streams))
            graph_key = key, (1 if clock % 128 == 0 else 2), (4 if clock % 4 == 0 else 2)
            if graph_key not in policy.tick_graphs:
                return False
        return True

    def _submit_buffer(self, workers: ThreadPoolExecutor, buffer: RolloutBuffer) -> list[PendingStep]:
        pending: list[PendingStep] = []
        for shard in buffer.shards:
            active = self._active(shard)
            if active.any():
                inputs = self._inputs(shard, 2)
                pending.append((buffer, shard, workers.submit(_step_shard, shard, inputs, active, buffer.stream)))
        return pending

    def _inputs(self, shard: SimulatorShard, count: int) -> Tensor:
        lanes = torch.from_numpy(self.shard_lanes[id(shard)]).to(self.policy.device)
        ring = self.rings[self.cursor]
        sides = torch.arange(2, device=self.policy.device)[None, :, None]
        positions = (ring.head[lanes, :, None] + ring.offsets[:count]) % ring.capacity
        indices = ring.values[lanes[:, None, None], sides, positions].permute(2, 0, 1, 3)
        if isinstance(self.policy, TransformerPolicy):
            if ring.controllers is None:
                raise ValueError("dense rollout requires device controller storage")
            result = ring.controllers[lanes[:, None, None], sides, positions].permute(2, 0, 1, 3)
        else:
            result = torch.from_numpy(self.policy.wire_actions(indices)).to(self.policy.device)
        return torch.where(self.active[self.cursor][lanes][None, :, None, None], result, 0)

    def _accept(
        self,
        shard: SimulatorShard,
        rows: Sequence[Sequence[SimFrame]],
        clock: int,
        update: int,
        table: DecisionTable,
        pending: Tensor,
    ) -> tuple[int, int]:
        group = self.cursor
        players = len(self.dense_meta[group])
        ring = self.rings[group]
        host_lanes = self.shard_lanes[id(shard)]
        lanes = torch.from_numpy(host_lanes).to(self.policy.device)
        slots = (2 * lanes[:, None] + torch.arange(2, device=self.policy.device)[None]).flatten()
        frames_stepped = timeouts = 0
        for offset, frames in enumerate(rows, start=1):
            if len(frames) != len(shard.match_ids):
                raise ValueError("simulator returned the wrong lane count")
            active_host = self._active(shard)
            active = self.active[group][lanes]
            positions = self.stops[group][lanes]
            if isinstance(frames, FrameBatch):
                rewards = table.store.write(positions, lanes, frames, torch.arange(len(lanes), device=lanes.device))
                terminals, truncations = frames.terminal, frames.timeout
            else:
                # Test simulators enter through the same packed storage.
                for source, lane in enumerate(host_lanes):
                    table.store.write_frame(int(positions[source]), int(lane), frames[source])
                rewards = table.store.rewards[positions, lanes]
                terminals = np.array([frame.terminal for frame in frames], dtype=np.bool_)
                truncations = np.array([frame.timeout for frame in frames], dtype=np.bool_)
            ids = torch.where(pending[slots] >= 0, pending[slots], slots)
            valid = (pending[slots] >= 0) & active.repeat_interleave(2)
            duration = table.duration_device[ids]
            table.rewards[ids, duration.clamp_max(3)] = torch.where(
                valid, rewards.flatten(), table.rewards[ids, duration.clamp_max(3)]
            )
            table.duration_device[ids] += valid.long()
            self.episode_returns[group][lanes] += torch.where(active[:, None], rewards, 0)
            self.stops[group][lanes] += active.long()
            self.observed_clocks[group][lanes] += active.long()
            ring.head[lanes] = (ring.head[lanes] + active[:, None].long()) % ring.capacity
            ring.length_device[lanes] -= active[:, None].long()
            frames_stepped += int(active_host.sum())
            ends = active_host & (terminals | truncations)
            for source in np.flatnonzero(ends):
                match = shard.match_ids[source]
                lane = int(host_lanes[source])
                state = self.matches[match]
                end_clock = clock + offset
                if offset != len(rows):
                    raise ValueError("simulator stepped beyond an episode boundary")
                history = state.history
                history.stop = int(self.stops[group][lane])
                state.returns = (
                    float(self.episode_returns[group][lane, 0]),
                    float(self.episode_returns[group][lane, 1]),
                )
                stocks = None
                if isinstance(frames, FrameBatch):
                    stocks = tuple(
                        int(frames.columns[source, NATIVE_COLUMN_LAYOUT[f"p{side}_stock"][0]].view(np.int32))
                        for side in (1, 2)
                    )
                elif "p1_stock" in frames[source].columns:
                    stocks = (int(frames[source].columns["p1_stock"]), int(frames[source].columns["p2_stock"]))
                self.results.append(
                    EpisodeResult(
                        match,
                        state.episode,
                        end_clock - state.episode_start,
                        state.returns,
                        bool(truncations[source]),
                        cast(tuple[int, int] | None, stocks),
                    )
                )
                match_slots = torch.arange(2 * lane, 2 * lane + 2, device=self.policy.device)
                pending_ids = pending[match_slots]
                valid_pending = pending_ids >= 0
                target = pending_ids[valid_pending]
                table.boundary_device[target] = True
                table.terminal_device[target] = bool(terminals[source])
                episode_rows = (table.integers[:, MATCH] == match) & (
                    table.integers[:, EPISODE] == state.episode.number
                )
                episode_rows &= table.valid
                ended_rows = torch.nonzero(episode_rows).flatten()
                table.remaining_device[ended_rows] = end_clock - table.integers[ended_rows, CLOCK]
                if truncations[source]:
                    boot = (
                        self.sampler.schedule.rollout_frames // 2 + (end_clock - self.clocks[group] - 1) // 128
                    ) * players + match_slots
                    self._write_bootstrap(table, boot, match_slots, end_clock, update)
                    table.following_device[target] = boot[valid_pending]
                pending[match_slots] = -1
                state.end_clock = end_clock
                self.active[group][lane] = False
                self.live_host[group][lane] = False
                timeouts += int(truncations[source])
        return frames_stepped, timeouts

    def _write_bootstrap(self, table: DecisionTable, ids: Tensor, slots: Tensor, clock: int, update: int) -> None:
        meta = self._dense_metadata(clock, update, bootstrap=True)[slots]
        table.integers[ids] = meta
        table.conditions[ids] = self.dense_logs[self.cursor][slots]
        ring = self.rings[self.cursor]
        lanes, sides = meta[:, LANE], meta[:, SIDE]
        count = min(24, ring.capacity)
        queue = torch.nn.functional.pad(ring.read(lanes, sides, count), (0, 0, 0, 24 - count))
        opponent = torch.nn.functional.pad(ring.read(lanes, 1 - sides, count), (0, 0, 0, 24 - count))
        offsets = torch.arange(24, device=self.policy.device)[None]
        table.prefix[ids] = torch.where((offsets < meta[:, DELAY, None])[..., None], queue, 0)
        table.opponent[ids] = torch.where((offsets < meta[:, OPP_LENGTH, None])[..., None], opponent, 0)

    def _reset_waiting(self, shards: Sequence[SimulatorShard], clock: int) -> int:
        if clock % 128:
            return 0
        resets = 0
        for shard in shards:
            waiting = np.flatnonzero(~self._active(shard))
            for lane in waiting:
                match = shard.match_ids[lane]
                state = self.matches[match]
                self._reset(shard, int(lane), match, clock, state.episode.number + 1)
                resets += 1
        return resets

    def _active(self, shard: SimulatorShard) -> NDArray[np.bool_]:
        return self.live_host[self.cursor][self.shard_lanes[id(shard)]]

    def collect(self, update: int) -> Rollout:
        if not self._initialized:
            raise ValueError("initialize or restore the pool before rollout")
        started = time.perf_counter()
        self.results = []
        group_id = self.cursor
        shards = self.groups[group_id]
        start = self.clocks[group_id]
        count = self.sampler.schedule.rollout_frames
        if count % 128:
            raise ValueError("rollouts must end on a 128-frame block boundary")
        self.tick_layouts = self.group_layouts[group_id]
        previous_store = self.stores[group_id]
        if previous_store is None:
            raise ValueError("pool has no frame storage")
        matches = self.group_matches[group_id]
        capacity = 129 + count + count // 128 + 1
        store = (
            previous_store
            if len(previous_store.records) == capacity
            else FrameStore(capacity, len(matches), self.policy.device, previous_store.fields)
        )
        lengths = (self.observed_clocks[group_id] - self.episode_clocks[group_id] + 1).clamp_max(129)
        positions = self.stops[group_id][None] - lengths[None] + torch.arange(129, device=self.policy.device)[:, None]
        valid = torch.arange(129, device=self.policy.device)[:, None] < lengths[None]
        positions = torch.where(valid, positions, 0)
        lanes = torch.arange(len(matches), device=self.policy.device)[None]
        for target, source in (
            (store.records, previous_store.records),
            (store.actions, previous_store.actions),
            (store.rewards, previous_store.rewards),
            (store.flags, previous_store.flags),
        ):
            mask = valid.reshape(*valid.shape, *([1] * (target.ndim - 2)))
            target[:129].copy_(torch.where(mask, source[positions, lanes], 0))
        for name in store.extra:
            store.extra[name][:129].copy_(torch.where(valid, previous_store.extra[name][positions, lanes], 0))
        self.stops[group_id].copy_(lengths)
        self.stores[group_id] = store
        table = self.tables[group_id]
        if table is None:
            table = DecisionTable(len(matches) * (count + 2 * (count // 128 + 2)), store)
            self.tables[group_id] = table
            if isinstance(self.policy, TransformerPolicy):
                table.allocate_evaluation(self.policy.model)
        table.reset()
        self._table = table
        table.count = len(table.meta)
        table.episodes = {
            (match, self.matches[match].episode.number): self.matches[match].episode for match in matches
        }
        self.policy.clear_caches()
        for anchor in self.anchors.values():
            anchor.clear_caches()
        pending = torch.full((2 * len(matches),), -1, dtype=torch.long, device=self.policy.device)
        resets = timeouts = frames_stepped = 0
        buffers = self.buffers[group_id]
        default_stream = torch.cuda.current_stream(self.policy.device) if self.policy.device.type == "cuda" else None
        with ThreadPoolExecutor(max_workers=len(shards)) as workers:
            for clock in range(start, start + count, 2):
                boundary = clock % 128 == 0
                with sample(
                    self.profile,
                    "boundary_tick" if boundary else "tick",
                    work=float(len(matches)),
                    limit=2 if boundary else 4,
                    skip=0 if boundary else 16,
                ):
                    if default_stream is not None:
                        for buffer in buffers:
                            default_stream.wait_stream(cast(torch.cuda.Stream, buffer.stream))
                    resets += self._reset_waiting(shards, clock)
                    with sample(self.profile, "tick_state", work=float(len(matches)), limit=0):
                        ids, meta, due = self._stage_tick(clock, update, table, pending)
                    ready = all(self._tick_ready(buffer, clock) for buffer in buffers)
                    futures: list[PendingStep] = []
                    for buffer in buffers:
                        inference_stream = buffer.stream if ready else default_stream
                        if ready and buffer.stream is not None:
                            buffer.stream.wait_stream(cast(torch.cuda.Stream, default_stream))
                        with (
                            torch.cuda.stream(inference_stream)
                            if inference_stream is not None
                            else contextlib.nullcontext()
                        ):
                            with sample(self.profile, "gpu_tick", work=float(len(buffer.lanes)), limit=0):
                                self._sample_buffer(buffer, clock, table, ids, meta, due)
                            if ready:
                                futures.extend(self._submit_buffer(workers, buffer))
                    if not ready:
                        # Cold captures restore the shared ring; serialize them before workers issue CUDA calls.
                        for buffer in buffers:
                            if buffer.stream is not None:
                                buffer.stream.wait_stream(cast(torch.cuda.Stream, default_stream))
                            with (
                                torch.cuda.stream(buffer.stream)
                                if buffer.stream is not None
                                else contextlib.nullcontext()
                            ):
                                futures.extend(self._submit_buffer(workers, buffer))
                    for buffer, shard, future in futures:
                        rows = future.result()
                        if not 1 <= len(rows) <= 2:
                            raise ValueError("simulator must return between one and two frames")
                        with (
                            torch.cuda.stream(buffer.stream) if buffer.stream is not None else contextlib.nullcontext()
                        ):
                            stepped, ended_timeouts = self._accept(shard, rows, clock, update, table, pending)
                            frames_stepped += stepped
                            timeouts += ended_timeouts
                            if len(rows) == 1 and self._active(shard).any():
                                rest = _step_shard(shard, self._inputs(shard, 1), self._active(shard), buffer.stream)
                                if len(rest) != 1:
                                    raise ValueError("one-frame step returned the wrong duration")
                                stepped, ended_timeouts = self._accept(shard, rest, clock + 1, update, table, pending)
                                frames_stepped += stepped
                                timeouts += ended_timeouts
                    if default_stream is not None:
                        for buffer in buffers:
                            default_stream.wait_stream(cast(torch.cuda.Stream, buffer.stream))
        end = start + count
        live_slots = torch.nonzero(pending >= 0).flatten()
        pending_ids = pending[live_slots]
        table.boundary_device[pending_ids] = True
        boot_ids = (count // 2 + count // 128) * (2 * len(matches)) + live_slots
        self._write_bootstrap(table, boot_ids, live_slots, end, update)
        table.following_device[pending_ids] = boot_ids
        # Preserve the pre-reset observation and queue before resetting a final boundary.
        resets += self._reset_waiting(shards, end)
        for lane, match in enumerate(matches):
            state = self.matches[match]
            state.returns = (
                float(self.episode_returns[group_id][lane, 0]),
                float(self.episode_returns[group_id][lane, 1]),
            )
            state.history.stop = int(self.stops[group_id][lane])
            length = int(
                (self.observed_clocks[group_id][lane] - self.episode_clocks[group_id][lane] + 1).clamp_max(129)
            )
            state.history = FrameHistory(store, lane, state.history.stop - length, state.history.stop)
            state.history_start = int(self.observed_clocks[group_id][lane]) - length + 1
        if bool(table.queue_error):
            raise ValueError("decision must see exactly D committed frames")
        table.publish()
        records = torch.nonzero(table.valid).flatten().cpu().numpy()
        self._table = None
        ring = self.rings[group_id]
        ring.length[:] = ring.length_device.cpu().numpy()
        self.clocks[group_id] = end
        self.cursor = (group_id + 1) % len(self.groups)
        self.completed_update = update
        return Rollout(
            DecisionRecords(table.rows(records)),
            group_id,
            update,
            frames_stepped,
            resets,
            timeouts,
            time.perf_counter() - started,
            tuple(self.results),
            count * len(matches) - frames_stepped,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class TrainConfig:
    seed: int = 65
    actor_optimizer: Literal["muon", "adam"] = "muon"
    actor_lr: float = 1e-5
    critic_lr: float = 3e-4
    critic_epochs: int = 4
    windows_per_step: int = 32
    decisions_per_window: int = 16
    clip_epsilon: float = 0.2
    gradient_clip: float = 1.0
    value_scale: float = 150.0
    warmup_steps: int = 32
    critic_warmup_updates: int = 172
    warmup_match_frames: int = 11008
    plateau_patience: int = 8
    plateau_tolerance: float = 0.01
    lambda_spread_multiplier: float = 3.0
    snapshot_interval: int = 256
    snapshot_capacity: int = 3
    checkpoint_interval: int = 32
    deterministic: bool = False

    def __post_init__(self) -> None:
        if type(self.deterministic) is not bool:
            raise ValueError("deterministic must be a boolean")
        for name in (
            "critic_epochs",
            "windows_per_step",
            "decisions_per_window",
            "plateau_patience",
            "snapshot_interval",
            "snapshot_capacity",
            "checkpoint_interval",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 1 <= self.decisions_per_window <= 16:
            raise ValueError("at most 16 decisions can train per window")
        for value in (self.warmup_steps, self.critic_warmup_updates, self.warmup_match_frames):
            if type(value) is not int or value < 0:
                raise ValueError("warmup lengths must be nonnegative integers")
        for name in (
            "actor_lr",
            "critic_lr",
            "gradient_clip",
            "value_scale",
            "lambda_spread_multiplier",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.clip_epsilon < 1 or not math.isfinite(self.plateau_tolerance) or self.plateau_tolerance < 0:
            raise ValueError("invalid clip or plateau tolerance")
        if self.actor_optimizer not in ("muon", "adam"):
            raise ValueError("unsupported actor optimizer")


def make_actor_optimizer(policy: LoopPolicy, config: TrainConfig) -> torch.optim.Optimizer:
    parameters = policy.actor_parameters()
    if config.actor_optimizer == "adam":
        return torch.optim.AdamW(parameters, lr=config.actor_lr, weight_decay=0)
    allowed = {id(p) for p in parameters}
    buckets: dict[tuple[bool, int], list[torch.nn.Parameter]] = defaultdict(list)
    for name, parameter in policy.model.named_parameters():
        if id(parameter) not in allowed:
            continue
        muon = parameter.ndim == 2 and name.startswith(
            ("trunk.blocks.", "temporal.blocks.", "temporal.history_attention.", "temporal.token_projection.weight")
        )
        if name.startswith(("temporal.outputs.", "temporal.trunk_outputs.")) and name.endswith("up.weight"):
            muon = True
        splits = 3 if name.endswith(("attn.c_attn.weight", "qkv.weight")) else 1
        if name.endswith("key_value.weight"):
            splits = 2
        buckets[muon, splits if muon else 1].append(parameter)
    groups: list[dict[str, object]] = []
    for (muon, splits), values in buckets.items():
        group: dict[str, object] = {"params": values, "lr": config.actor_lr, "weight_decay": 0.0, "use_muon": muon}
        if muon:
            group.update(momentum=0.95, logical_splits=splits, muon_scale_clamp_min_one=False)
        else:
            group.update(betas=(0.9, 0.95), eps=1e-8)
        groups.append(group)
    return SingleDeviceMuonWithAuxAdam(groups)


def _take_features(features: CriticFeatures, indices: Tensor) -> CriticFeatures:
    return CriticFeatures(
        *(getattr(features, field.name).index_select(0, indices) for field in dataclasses.fields(features))
    )


def _take_decisions(decisions: DecisionBatch, indices: Tensor) -> DecisionBatch:
    return DecisionBatch(
        *(getattr(decisions, field.name).index_select(0, indices) for field in dataclasses.fields(decisions))
    )


def _log_distribution(probabilities: Tensor) -> Tensor:
    positive = probabilities > 0
    return torch.where(positive, probabilities, 1).log().masked_fill(~positive, -torch.inf)


def factor_kl(policy: PolicyEvaluation, reference: PolicyEvaluation, executed_JF: Tensor) -> Tensor:
    rows = []
    for group, (p, q) in enumerate(zip(policy.distributions, reference.distributions, strict=True)):
        valid = executed_JF[:, group::4, None]
        neutral = torch.nn.functional.one_hot(torch.zeros_like(p[..., 0], dtype=torch.long), p.shape[-1]).to(p.dtype)
        p = torch.where(valid, p, neutral)
        q = torch.where(valid, q, neutral)
        rows.append(categorical_kl(_log_distribution(p), _log_distribution(q)))
    return torch.stack(rows, -1).flatten(1)


def explained_variance(values: Tensor, targets: Tensor) -> float:
    variance = targets.double().var(unbiased=False)
    if float(variance) == 0:
        return math.nan
    return float(1 - (targets.double() - values.double()).var(unbiased=False) / variance)


@dataclasses.dataclass(frozen=True, slots=True)
class PreparedUpdate:
    rollout: Rollout
    decisions: DecisionBatch
    old: PolicyEvaluation
    reference: PolicyEvaluation | None
    values: Tensor
    advantages: Tensor
    targets: Tensor
    selected: Tensor
    steps: tuple[Tensor, ...]
    metrics: dict[str, float]
    windows: ReplayBatch
    placement: WindowPlacement


@dataclasses.dataclass(frozen=True, slots=True)
class ReplayBatch:
    """Packed windows and their fixed decision slots; rows index dense storage."""

    records: Tensor
    applied_actions: Tensor
    sides: Tensor
    identities: Tensor
    lengths: Tensor
    slots: ReplaySlots
    rows: Tensor
    metadata: Tensor

    def inputs(self) -> tuple[Tensor, ...]:
        return (
            self.records,
            self.applied_actions,
            self.sides,
            self.identities,
            self.lengths,
            *(getattr(self.slots, name) for name in REPLAY_SLOT_NAMES),
        )

    def take(self, windows: Tensor, positions: Tensor, live: Tensor) -> ReplayBatch:
        """Gather a padded actor step without materializing per-decision history."""
        at = windows.clamp(0, len(self.records) - 1)
        selected_positions = positions[at]
        fields = []
        for name in REPLAY_SLOT_NAMES:
            value = getattr(self.slots, name)
            selected = value[at]
            if name != "history_starts_WL":
                selected = selected[torch.arange(len(at), device=at.device)[:, None], selected_positions]
            if name == "valid_WK":
                selected = selected & live[:, None]
            fields.append(selected)
        return ReplayBatch(
            self.records[at],
            self.applied_actions[at],
            self.sides[at],
            self.identities[at],
            self.lengths[at],
            ReplaySlots(*fields),
            self.rows[at[:, None], selected_positions],
            self.metadata[at[:, None], selected_positions],
        )


REPLAY_SLOT_NAMES = tuple(field.name for field in dataclasses.fields(ReplaySlots))


def replay_batch(policy: LoopPolicy, table: DecisionTable, rows: Tensor, valid: Tensor) -> ReplayBatch:
    """Assemble device slots from one episode per block, guaranteed by reset cadence."""
    meta = table.integers[rows]
    active_lengths = torch.where(valid, meta[..., LENGTH], 0)
    longest = active_lengths.argmax(1)
    index = torch.arange(len(rows), device=rows.device)
    window = meta[index, longest]
    lengths = active_lengths.amax(1).clamp_min(1)
    time = torch.arange(256, device=rows.device)[None]
    in_history = time < lengths[:, None]
    positions = torch.where(in_history, window[:, START, None] + time, 0)
    flat = (positions * table.store.records.shape[1] + window[:, LANE, None]).flatten()
    records = table.store.records.flatten(0, 1)[flat].reshape(len(rows), 256, -1)
    applied = table.store.actions.flatten(0, 1)[flat].reshape(len(rows), 256, 2, ACTION_DIM)
    records = torch.where(in_history[..., None], records, 0)
    applied = torch.where(in_history[..., None, None], applied, 0)
    conditions = table.conditions[rows]
    remaining = table.remaining_device[rows]
    remaining = torch.where(remaining < 0, meta[..., DELAY] + meta[..., REPLAN], remaining)
    frame = torch.arange(1, 5, device=rows.device)
    mask = (frame <= meta[..., REPLAN, None]) & (meta[..., DELAY, None] + frame <= remaining[..., None])
    mask = mask.repeat_interleave(4, -1)
    slots = ReplaySlots(
        (meta[..., LENGTH] - 1).clamp_min(0),
        valid,
        torch.zeros(len(rows), 256, dtype=torch.long, device=rows.device),
        table.prefix[rows, : policy.max_delay],
        meta[..., DELAY],
        meta[..., REPLAN],
        policy.condition_ranges.normalize(conditions[..., 0], meta[..., DELAY], meta[..., REPLAN], validate=False),
        policy.condition_ranges.normalize(
            conditions[..., 1], meta[..., OPP_DELAY], meta[..., OPP_REPLAN], validate=False
        ),
        table.opponent[rows, : policy.max_opponent_queue],
        meta[..., OPP_LENGTH],
        table.actions[rows],
        mask.reshape(*rows.shape, 16),
    )
    return ReplayBatch(records, applied, window[:, SIDE], window[:, IDENTITY], lengths, slots, rows, meta)


@dataclasses.dataclass(frozen=True, slots=True)
class WindowPlacement:
    selected: Tensor  # [W,K]
    positions: Tensor  # [W,decisions_per_window]
    windows: Tensor  # [steps,windows_per_step], -1 padding


def place_windows(
    batch: ReplayBatch, rng: CounterRng, update: int, count: int, windows_per_step: int
) -> WindowPlacement:
    if not 1 <= count <= batch.rows.shape[1] or windows_per_step < 1:
        raise ValueError("invalid decision/window selection shape")
    eligible = batch.slots.valid_WK & batch.slots.executed_WKF.any(-1)
    meta = batch.metadata
    draws = rng.tensor_uniform(1, meta[..., STREAM], meta[..., EPISODE], meta[..., CLOCK])
    positions = torch.argsort(torch.where(eligible, draws, torch.inf), stable=True)[:, :count]
    selected = torch.zeros_like(eligible)
    selected.scatter_(1, positions, eligible.gather(1, positions))
    live = selected.any(-1)
    total = int(live.sum())
    if not total:
        return WindowPlacement(
            selected, positions, torch.empty(0, windows_per_step, dtype=torch.long, device=meta.device)
        )
    matches, inverse = torch.unique(meta[:, 0, MATCH], return_inverse=True)
    sizes = torch.zeros(len(matches), dtype=torch.long, device=meta.device)
    sizes.scatter_add_(0, inverse, live.long())
    steps = max(int(sizes.max()), math.ceil(total / windows_per_step))
    match_order = torch.argsort(
        rng.tensor_uniform(2, matches, torch.zeros_like(matches), torch.full_like(matches, update)), stable=True
    )
    rank = torch.empty_like(match_order)
    rank[match_order] = torch.arange(len(matches), device=meta.device)
    ordered = torch.argsort(rank[inverse], stable=True)
    ordered = ordered[live[ordered]]
    index = torch.arange(total, device=meta.device)
    placed = torch.full((steps, windows_per_step), -1, dtype=torch.long, device=meta.device)
    placed[index % steps, index // steps] = ordered
    return WindowPlacement(selected, positions, placed)


def window_indices(rows: Tensor, players: int) -> Tensor:
    tick = rows // players
    return ((tick // 64) * players + rows % players) * 64 + tick % 64


def window_evaluation(output: ReplayOutput, actions: Tensor, indices: Tensor) -> PolicyEvaluation:
    def flatten(value: Tensor) -> Tensor:
        return value.flatten(0, 1)[indices]

    features = output.features
    return PolicyEvaluation(
        actions,
        flatten(output.log_probs_WKF),
        tuple(flatten(value) for value in output.distributions),
        CriticFeatures(
            *(
                flatten(getattr(features, name))
                for name in (
                    "trunk_WKd",
                    "decoder_WKt",
                    "condition_WK3",
                    "opponent_condition_WK3",
                    "opponent_queue_WKQ4",
                    "opponent_lengths_WK",
                )
            )
        ),
    )


class Trainer:
    def __init__(self, engine: RolloutEngine, config: TrainConfig) -> None:
        if torch.are_deterministic_algorithms_enabled() != config.deterministic:
            raise ValueError("trainer determinism differs from the configured execution mode")
        if engine.sampler.rng.seed != config.seed:
            raise ValueError("trainer and counter RNG seeds must agree")
        self.profile: UpdateProfile | None = None
        self.engine = engine
        self.policy = engine.policy
        self.reference = engine.anchors[0]
        self.config = config
        actor_ids = {id(p) for p in self.policy.actor_parameters()}
        critic_ids = {id(p) for p in self.policy.critic_parameters()}
        if not actor_ids or not critic_ids or actor_ids & critic_ids:
            raise ValueError("actor and critic parameter sets must be nonempty and disjoint")
        self.actor_optimizer = make_actor_optimizer(self.policy, config)
        self.critic_optimizer = torch.optim.AdamW(self.policy.critic_parameters(), lr=config.critic_lr, weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.actor_optimizer, lambda step: min(1.0, (step + 1) / max(1, config.warmup_steps))
        )
        self.update = 0
        self.actor_steps = 0
        self.warming = config.critic_warmup_updates > 0
        self.warmup_moments: dict[DelayReplan, tuple[int, float, float]] = {}
        self.fresh_variances: list[float] = []
        self.next_anchor = max(engine.anchors) + 1
        self.evaluation_cursor = 0
        self.updating = False

    @torch.no_grad()
    def prepare(self, rollout: Rollout) -> PreparedUpdate:
        started = time.perf_counter()
        device = self.policy.device
        records = rollout.records
        decisions = _decision_batch(records, device)
        states = records.states if isinstance(records, DecisionRecords) else [r.state for r in records]
        if not isinstance(states, DecisionRows) or states.table.evaluation is None:
            raise ValueError("rollout is missing its behavior evaluation")
        table = states.table
        players = 2 * len(self.engine.group_matches[rollout.group])
        ticks = self.engine.sampler.schedule.rollout_frames // 2
        size = ticks * players
        rows = torch.arange(size, device=device).reshape(-1, 64, players).transpose(1, 2).reshape(-1, 64)
        windows = replay_batch(self.policy, table, rows, table.valid[rows])
        compact = window_indices(states.device_ids, players)
        assert table.evaluation is not None
        old = _take_evaluation(table.evaluation, states.device_ids)
        with phase(self.profile, "reference"), sample(self.profile, "windows", work=float(rows.numel())):
            reference = (
                None
                if self.warming
                else window_evaluation(self.reference.evaluate_windows(windows), old.actions, compact)
            )
        kl = (
            torch.zeros_like(old.log_probs)
            if reference is None
            else factor_kl(old, reference, decisions.executed_factors_JF)
        )
        frame_rewards = (
            states.tensor(states.table.rewards)
            if isinstance(states, DecisionRows)
            else torch.tensor(
                [record.rewards + [0.0] * (4 - len(record.rewards)) for record in records], device=device
            )
        )
        rewards = regularized_reward(decisions, frame_rewards, kl)
        values = states.tensor(states.table.values)
        next_values = torch.zeros_like(values)
        table = states.table
        following = states.tensor(table.following_device)
        terminal = states.tensor(table.terminal_device)
        boundary = states.tensor(table.boundary_device)
        if bool((~terminal & (following < 0)).any()):
            raise ValueError("nonterminal boundary is missing its bootstrap state")
        interior = ~terminal & ~boundary
        next_episode = table.integers[following.clamp_min(0), EPISODE]
        if bool((interior & (next_episode != decisions.episode_id_J)).any()):
            raise ValueError("GAE trace crossed an episode reset")
        boundary_mask = boundary & ~terminal
        boundary_at = torch.nonzero(boundary_mask).flatten()
        all_values = table.values.clone()
        bootstrap_rows = torch.arange(size, len(table.values), device=device)[:, None]
        bootstrap_valid = torch.zeros_like(table.valid)
        bootstrap_valid[following[boundary_at]] = True
        if len(boundary_at):
            with sample(self.profile, "bootstrap", work=float(len(bootstrap_rows))):
                bootstrap_batch = replay_batch(self.policy, table, bootstrap_rows, bootstrap_valid[bootstrap_rows])
                all_values[size:] = self.policy.evaluate_values(bootstrap_batch).flatten()
        next_values = torch.where(terminal, 0, all_values[following.clamp_min(0)])
        dense_rewards = torch.zeros_like(table.values)
        dense_next = torch.zeros_like(table.values)
        dense_rewards[states.device_ids] = rewards
        dense_next[states.device_ids] = next_values
        result = dense_advantages(
            dense_rewards[:size].reshape(ticks, players),
            table.values[:size].reshape(ticks, players),
            dense_next[:size].reshape(ticks, players),
            table.duration_device[:size].reshape(ticks, players),
            ~table.terminal_device[:size].reshape(ticks, players),
            ~table.boundary_device[:size].reshape(ticks, players),
            table.valid[:size].reshape(ticks, players),
        )
        advantages = result.advantages.flatten()[states.device_ids]
        targets = result.value_targets.flatten()[states.device_ids]
        placement = place_windows(
            windows,
            self.engine.sampler.rng,
            rollout.update,
            self.config.decisions_per_window,
            self.config.windows_per_step,
        )
        selected = placement.selected.flatten()[compact]
        steps = tuple(placement.windows.unbind(0))
        executed = decisions.executed_factors_JF
        gap = torch.where(executed, old.log_probs - decisions.behavior_log_probs_JF, 0).sum(-1)
        metrics = {
            "critic/explained_variance": explained_variance(values, targets),
            "numerics/mu_old_gap_mean": float(-gap.mean()),
            "numerics/mu_old_gap_p50": float(gap.abs().quantile(0.5)),
            "numerics/mu_old_gap_p99": float(gap.abs().quantile(0.99)),
            "numerics/mu_old_gap_max": float(gap.abs().max()),
            "policy/kl_per_frame": float(torch.where(executed, kl, 0).sum() / decisions.replan_J.sum()),
            "return/interval_reward": float(frame_rewards.sum(-1).mean()),
            "data/anchor_share": sum(
                player.anchor is not None for state in self.engine.matches.values() for player in state.episode.players
            )
            / len(self.engine.matches)
            / 2,
            "data/resets": float(rollout.resets),
            "data/timeouts": float(rollout.timeouts),
            "systems/rollout_seconds": rollout.elapsed,
            "systems/preparation_seconds": time.perf_counter() - started,
            "systems/simulator_fps": rollout.frames / rollout.elapsed,
            "systems/rollout_frames": float(rollout.frames),
            "systems/reset_idle_frames": float(rollout.idle_frames),
            "systems/reset_idle_fraction": rollout.idle_frames / (rollout.frames + rollout.idle_frames),
            "data/decisions": float(len(records)),
            "data/trained_decisions": float(selected.sum()),
        }
        log_min = math.log(self.engine.sampler.config.lambda_min)
        log_max = math.log(self.engine.sampler.config.lambda_max)
        log_values = decisions.kl_coefficient_J.log()
        bins = ((log_values - log_min) / max(log_max - log_min, 1e-12) * 4).long().clamp(0, 3)
        for index in range(4):
            mask = bins == index
            if bool(mask.any()):
                metrics[f"return/interval_reward_lambda{index}"] = float(frame_rewards[mask].sum(-1).mean())
                metrics[f"policy/kl_per_frame_lambda{index}"] = float(
                    torch.where(executed[mask], kl[mask], 0).sum() / decisions.replan_J[mask].sum()
                )
        for position in range(4):
            mask = (decisions.decision_frame_J % 128) // 32 == position
            if bool(mask.any()):
                metrics[f"critic/explained_variance_position{position}"] = explained_variance(
                    values[mask], targets[mask]
                )
        for name, column in (("identity", IDENTITY), ("character", CHARACTER), ("stage", STAGE)):
            if isinstance(states, DecisionRows):
                keys, counts_array = np.unique(states.meta[:, column], return_counts=True)
                for key, value in zip(keys, counts_array, strict=True):
                    metrics[f"data/{name}_{key}"] = float(value)
            else:
                counts: dict[int, int] = defaultdict(int)
                for record in records:
                    value = {
                        "identity": record.state.player.identity,
                        "character": record.state.player.character,
                        "stage": record.state.episode.stage,
                    }[name]
                    counts[value] += 1
                for key, value in counts.items():
                    metrics[f"data/{name}_{key}"] = float(value)
        final_rows = following[boundary_at]
        final = table.integers[final_rows, CLOCK] % self.engine.sampler.schedule.rollout_frames == 0
        subset = final_rows[final][:32]
        if len(subset):
            shorter = replay_batch(
                self.policy, table, subset[:, None], torch.ones(len(subset), 1, device=device, dtype=torch.bool)
            )
            length = shorter.lengths.clamp_max(129)
            shift = shorter.lengths - length
            frame = torch.arange(256, device=device)[None]
            source = (shift[:, None] + frame).clamp_max(255)
            records_short = shorter.records.gather(1, source[..., None].expand(-1, -1, shorter.records.shape[-1]))
            applied_short = shorter.applied_actions.gather(1, source[..., None, None].expand(-1, -1, 2, ACTION_DIM))
            shorter = dataclasses.replace(
                shorter,
                records=records_short,
                applied_actions=applied_short,
                lengths=length,
                slots=dataclasses.replace(shorter.slots, positions_WK=length[:, None] - 1),
            )
            metrics["critic/bootstrap_context_gap"] = float(
                (self.policy.evaluate_values(shorter).flatten() - all_values[subset]).abs().mean()
            )
        for timing in self.engine.sampler.schedule.configurations:
            mask = (decisions.delay_J == timing.delay) & (decisions.replan_J == timing.replan)
            if not bool(mask.any()):
                continue
            suffix = f"d{timing.delay}_r{timing.replan}"
            metrics[f"critic/advantage_mean_{suffix}"] = float(advantages[mask].mean())
            metrics[f"critic/advantage_std_{suffix}"] = float(advantages[mask].std(unbiased=False))
            metrics[f"critic/explained_variance_{suffix}"] = explained_variance(values[mask], targets[mask])
            metrics[f"numerics/mu_old_gap_p99_{suffix}"] = float(gap[mask].abs().quantile(0.99))
        for factor, distribution in zip(FACTOR_ORDER, old.distributions, strict=True):
            entropy = -(distribution * torch.where(distribution > 0, distribution, 1).log()).sum(-1)
            valid = torch.arange(4, device=device)[None] < decisions.replan_J[:, None]
            metrics[f"policy/entropy_{factor}"] = float(entropy[valid].mean())
        return PreparedUpdate(
            rollout,
            decisions,
            old,
            reference,
            values,
            advantages,
            targets,
            selected,
            steps,
            metrics,
            windows,
            placement,
        )

    def fit(self, batch: PreparedUpdate) -> dict[str, float]:
        if not self.warming and batch.reference is None:
            raise ValueError("actor update requires a reference evaluation")
        cfg = self.config
        metrics = dict(batch.metrics)
        started = time.perf_counter()
        with (
            phase(self.profile, "critic"),
            sample(self.profile, "fit", work=float(len(batch.values))),
        ):
            features = batch.old.features.detached()
            count = len(batch.values)
            selected = torch.ones(count, dtype=torch.bool, device=self.policy.device)
            for _ in range(cfg.critic_epochs):
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss = critic_loss(
                    self.policy.values(features),
                    batch.targets,
                    selected,
                    value_scale=cfg.value_scale,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.critic_parameters(), cfg.gradient_clip, error_if_nonfinite=True
                )
                self.critic_optimizer.step()
            metrics["critic/loss"] = float(loss.detach())
            metrics["systems/critic_seconds"] = time.perf_counter() - started
        if self.warming:
            self._warmup(batch)
            return metrics
        metrics["policy/empty_update"] = float(not batch.steps)
        if not batch.steps:
            return metrics
        assert batch.reference is not None
        started = time.perf_counter()
        policy_losses: list[Tensor] = []
        kl_losses: list[Tensor] = []
        ratios: list[Tensor] = []
        old_log_ratios: list[Tensor] = []
        table = batch.rollout.records.states.table
        dense_advantage = torch.zeros_like(table.values)
        dense_advantage[batch.rollout.records.states.device_ids] = batch.advantages
        assert batch.reference is not None
        reference_dense = tuple(
            value.new_zeros(len(table.values), *value.shape[1:]) for value in batch.reference.distributions
        )
        for target, value in zip(reference_dense, batch.reference.distributions, strict=True):
            target[batch.rollout.records.states.device_ids] = value
        with phase(self.profile, "actor"):
            for step, window_ids in enumerate(batch.steps):
                with sample(
                    self.profile, "step", work=float(cfg.windows_per_step * cfg.decisions_per_window), limit=2
                ):
                    live = window_ids >= 0
                    windows = batch.windows.take(window_ids, batch.placement.positions, live)
                    selected = (
                        batch.placement.selected[window_ids.clamp_min(0)].gather(
                            1, batch.placement.positions[window_ids.clamp_min(0)]
                        )
                        & live[:, None]
                    )
                    advantages = dense_advantage[windows.rows]
                    statistics = advantage_statistics(advantage_moments(advantages.flatten(), selected.flatten()))
                    targets = ActorTargets(
                        table.behavior[windows.rows],
                        advantages,
                        selected,
                        table.coefficients[windows.rows],
                        tuple(value[windows.rows] for value in reference_dense),
                        statistics.mean.float(),
                        statistics.sigma.float(),
                        selected.sum().float(),
                    )
                    result, current = self.policy.actor_step(windows, targets, clip_epsilon=cfg.clip_epsilon)
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.actor_parameters(), cfg.gradient_clip, error_if_nonfinite=True
                    )
                    self.actor_optimizer.step()
                    self.scheduler.step()
                    self.actor_steps += 1
                    policy_losses.append(result.policy_loss.detach())
                    kl_losses.append(result.kl_loss.detach())
                    ratio = result.joint_ratio_WK[selected].detach()
                    ratios.append(ratio)
                    old_log_ratios.append(ratio.log())
                    metrics[f"policy/old_log_ratio_step{step}"] = float(ratio.log().mean())
                    assert table.evaluation is not None
                    new = window_evaluation(
                        current,
                        table.actions[windows.rows].flatten(0, 1),
                        torch.arange(windows.rows.numel(), device=windows.rows.device),
                    )
                    previous = _take_evaluation(table.evaluation, windows.rows.flatten())
                    mask = (current.factor_mask_WKF & selected[..., None]).flatten(0, 1)
                    old_kl = factor_kl(new, previous, mask)
                    metrics[f"policy/old_kl_step{step}"] = float(torch.where(mask, old_kl, 0).sum() / selected.sum())
        ratio = torch.cat(ratios)
        metrics.update(
            {
                "policy/loss": float(torch.stack(policy_losses).mean()),
                "policy/kl_loss": float(torch.stack(kl_losses).mean()),
                "policy/clip_fraction": float(((ratio - 1).abs() > cfg.clip_epsilon).float().mean()),
                "policy/ratio_p01": float(ratio.quantile(0.01)),
                "policy/ratio_p99": float(ratio.quantile(0.99)),
                "policy/old_log_ratio_p99": float(torch.cat(old_log_ratios).abs().quantile(0.99)),
                "systems/actor_seconds": time.perf_counter() - started,
            }
        )
        return metrics

    def _warmup(self, batch: PreparedUpdate) -> None:
        for timing in self.engine.sampler.schedule.configurations:
            mask = (batch.decisions.delay_J == timing.delay) & (batch.decisions.replan_J == timing.replan)
            values = batch.advantages[mask].double()
            previous = self.warmup_moments.get(timing, (0, 0.0, 0.0))
            self.warmup_moments[timing] = (
                previous[0] + len(values),
                previous[1] + float(values.sum()),
                previous[2] + float(values.square().sum()),
            )
        self.fresh_variances.append(batch.metrics["critic/explained_variance"])
        patience = self.config.plateau_patience
        if self.update + 1 < self.config.critic_warmup_updates or len(self.fresh_variances) < patience:
            return
        if min(self.engine.clocks) < self.config.warmup_match_frames:
            return
        recent = self.fresh_variances[-patience:]
        if not all(math.isfinite(x) for x in recent) or max(recent) - min(recent) > self.config.plateau_tolerance:
            return
        spreads = []
        for count, total, squares in self.warmup_moments.values():
            if count < 2:
                raise ValueError("lambda calibration needs advantages for every timing configuration")
            spreads.append(math.sqrt(max(0.0, squares / count - (total / count) ** 2)))
        maximum = self.config.lambda_spread_multiplier * max(spreads)
        sampler = self.engine.sampler
        if not sampler.config.lambda_min < maximum <= sampler.config.lambda_max:
            raise ValueError("calibrated lambda_max is outside the declared conditioner range")
        sampler.lambda_max = maximum
        self.warming = False

    def run_update(self) -> tuple[PreparedUpdate, dict[str, float]]:
        if self.updating:
            raise ValueError("previous update did not finish; restore an update-boundary checkpoint")
        self.updating = True
        started = time.perf_counter()
        native_shards = [
            shard for shard in self.engine.groups[self.engine.cursor] if isinstance(shard, NativeSimulatorShard)
        ]
        before = [
            (shard.step_seconds, shard.reset_seconds, shard.stepped_frames, shard.copied_bytes)
            for shard in native_shards
        ]
        if self.policy.device.type == "cuda":
            torch.cuda.synchronize(self.policy.device)
            torch.cuda.reset_peak_memory_stats(self.policy.device)
        with phase(self.profile, "rollout"):
            rollout = self.engine.collect(self.update)
        self.policy.clear_caches()
        for anchor in self.engine.anchors.values():
            anchor.clear_caches()
        with phase(self.profile, "preparation"):
            batch = self.prepare(rollout)
        metrics = self.fit(batch)
        for index, (shard, previous) in enumerate(zip(native_shards, before, strict=True)):
            elapsed = shard.step_seconds - previous[0]
            metrics[f"systems/native_shard{index}_seconds"] = elapsed
            metrics[f"systems/native_shard{index}_fps"] = (shard.stepped_frames - previous[2]) / elapsed
            metrics[f"systems/native_shard{index}_reset_seconds"] = shard.reset_seconds - previous[1]
            metrics[f"systems/native_shard{index}_h2d_bytes"] = float(shard.copied_bytes - previous[3])
        self.update += 1
        metrics.update(
            global_step=float(self.update),
            **{
                "train/actor_steps": float(self.actor_steps),
                "train/warming": float(self.warming),
                "train/lambda_max": self.engine.sampler.lambda_max,
            },
        )
        if not self.warming and self.update % self.config.snapshot_interval == 0:
            assigned = {p.anchor for m in self.engine.matches.values() for p in m.episode.players}
            snapshots = sorted(key for key in self.engine.anchors if key)
            retired = snapshots[: max(0, len(snapshots) - self.config.snapshot_capacity + 1)]
            available = [key for key in retired if key not in assigned]
            if available:
                anchor = self.engine.anchors.pop(available[0])
                anchor.model.load_state_dict(self.policy.model.state_dict(), strict=True)
                anchor.clear_caches()
            else:
                anchor = self.policy.frozen_copy()
            self.engine.anchors[self.next_anchor] = anchor
            self.next_anchor += 1
            self.engine.anchor_candidates = (
                0,
                *sorted(key for key in self.engine.anchors if key)[-self.config.snapshot_capacity :],
            )
            # Keep an assigned snapshot until its last episode finishes.
            assigned = {p.anchor for m in self.engine.matches.values() for p in m.episode.players}
            snapshots = sorted(key for key in self.engine.anchors if key)
            for key in snapshots[: -self.config.snapshot_capacity]:
                if key not in assigned:
                    del self.engine.anchors[key]
        self.policy.clear_caches()
        if self.policy.device.type == "cuda":
            torch.cuda.synchronize(self.policy.device)
            metrics["systems/peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated(self.policy.device))
            metrics["systems/peak_reserved_bytes"] = float(torch.cuda.max_memory_reserved(self.policy.device))
        metrics["systems/update_seconds"] = time.perf_counter() - started
        metrics["systems/match_frames_per_second"] = rollout.frames / metrics["systems/update_seconds"]
        self.updating = False
        return batch, metrics


def _durable_file(directory: Path, payload: bytes, suffix: str) -> tuple[str, str]:
    digest = hashlib.sha256(payload).hexdigest()
    name = f"{digest}.{suffix}"
    target = directory / name
    temporary = directory / f".{name}.{os.getpid()}.partial"
    with temporary.open("wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, target)
    return name, digest


def _environment() -> dict[str, str]:
    import platform

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": str(torch.version.cuda),
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "world_size": "1",
        "tf32_matmul": str(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": str(torch.backends.cudnn.allow_tf32),
    }


def _configuration_digest(trainer: Trainer) -> str:
    engine = trainer.engine
    model = trainer.policy.model
    schema = [(name, tuple(value.shape), str(value.dtype)) for name, value in model.state_dict().items()]
    training = dataclasses.asdict(trainer.config)
    del training["deterministic"]
    identity = {
        "train": training,
        "sampler": dataclasses.asdict(engine.sampler.config),
        "schedule": dataclasses.asdict(engine.sampler.schedule),
        "seed": engine.sampler.rng.seed,
        "schema": schema,
        "execution_device": str(trainer.policy.device),
        "resume_contract": "execution_mode_independent_v1",
        "groups": [[s.match_ids for s in group] for group in engine.groups],
        "native_limits": [
            [s.max_frame if isinstance(s, NativeSimulatorShard) else None for s in group] for group in engine.groups
        ],
        "episode_reset": "shared_stocks_block_wait_v1",
        "rollout_storage": "stored_behavior_device_selection_v3",
        "selection_rng": "integer_counter_v3",
        "opponent_layout": "fixed_anchor_ranges_v1" if engine.fixed_anchor_lanes else "sampled_anchor_lanes_v1",
    }
    if isinstance(trainer.policy, TransformerPolicy):
        identity["compiled"] = trainer.policy.compiled
        identity["decision_kernels"] = "decoder_precision_fixed_tick_graph_v4"
        identity["decoder_precision"] = trainer.policy.model.decoder_precision
        identity["no_grad_chunks"] = dataclasses.asdict(trainer.policy.replay_chunks)
        identity["model"] = dataclasses.asdict(trainer.policy.model.cfg)
        identity["ranges"] = dataclasses.asdict(trainer.policy.model.ranges)
        identity["statistics"] = {key: dataclasses.asdict(value) for key, value in trainer.policy.statistics.items()}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def save_checkpoint(trainer: Trainer, directory: Path, provenance: dict[str, str]) -> Path:
    """Publish one manifest after the model and every simulator shard are durable.

    No cache is saved: every resumed rollout prefills under restored weights.
    Filenames are content hashes, so an interrupted replacement cannot corrupt
    a checkpoint which already has a manifest.
    """
    import io

    if torch.are_deterministic_algorithms_enabled() != trainer.config.deterministic:
        raise ValueError("checkpoint execution mode differs from the trainer configuration")
    if not trainer.engine._initialized or trainer.updating or trainer.engine.completed_update != trainer.update - 1:
        raise ValueError("checkpoint requires a complete update boundary")
    required = {"git_sha", "checkpoint_sha256", "data_manifest_sha256", "simulator_sha256", "adapter_sha256"}
    if set(provenance) != required or any(not value for value in provenance.values()):
        raise ValueError("checkpoint provenance must identify source, BC checkpoint, data, simulator, and adapter")
    directory.mkdir(parents=True, exist_ok=True)
    engine = trainer.engine
    state = {
        "version": CHECKPOINT_VERSION,
        "execution": {
            "deterministic": trainer.config.deterministic,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "model": trainer.policy.model.state_dict(),
        "actor_optimizer": trainer.actor_optimizer.state_dict(),
        "critic_optimizer": trainer.critic_optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict(),
        "reference": trainer.reference.model.state_dict(),
        "anchors": {key: value.model.state_dict() for key, value in engine.anchors.items()},
        "anchor_candidates": engine.anchor_candidates,
        "pool": engine.snapshot(),
        "clocks": engine.clocks,
        "cursor": engine.cursor,
        "sampler_cursor": engine.sampler.cursor,
        "lambda_max": engine.sampler.lambda_max,
        "update": trainer.update,
        "actor_steps": trainer.actor_steps,
        "warming": trainer.warming,
        "warmup_moments": trainer.warmup_moments,
        "fresh_variances": trainer.fresh_variances,
        "next_anchor": trainer.next_anchor,
        "evaluation_cursor": trainer.evaluation_cursor,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    buffer = io.BytesIO()
    torch.save(state, buffer)
    name, digest = _durable_file(directory, buffer.getvalue(), "rank0.pt")
    shards = [{"name": name, "sha256": digest, "role": "training"}]
    for group_id, group in enumerate(engine.groups):
        for shard_id, shard in enumerate(group):
            name, digest = _durable_file(directory, zlib.compress(shard.save()), "sim.zlib")
            shards.append({"name": name, "sha256": digest, "role": f"simulator_{group_id}_{shard_id}"})
    if isinstance(trainer.policy, TransformerPolicy):
        actor = directory / f".actor-{os.getpid()}-{time.time_ns()}.pt"
        try:
            save_evaluation_checkpoint(
                actor,
                trainer.policy.model,
                trainer.policy.statistics,
                provenance={
                    "reference_sha256": provenance["checkpoint_sha256"],
                    "git_sha": provenance["git_sha"],
                    "update": trainer.update,
                    "data_manifests": {"training": provenance["data_manifest_sha256"]},
                },
            )
            name, digest = _durable_file(directory, actor.read_bytes(), "actor.pt")
            shards.append({"name": name, "sha256": digest, "role": "actor"})
        finally:
            actor.unlink(missing_ok=True)
    manifest = {
        "format": TRAINING_FORMAT,
        "version": CHECKPOINT_VERSION,
        "experiment": "065_selfplay_ppo",
        "update": trainer.update,
        "configuration_sha256": _configuration_digest(trainer),
        "environment": _environment(),
        "provenance": provenance,
        "shards": shards,
    }
    temporary = directory / f".manifest-{trainer.update}-{os.getpid()}.partial"
    target = directory / f"update-{trainer.update:08d}.json"
    with temporary.open("w") as output:
        json.dump(manifest, output, sort_keys=True)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, target)
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return target


def restore_checkpoint(trainer: Trainer, manifest_path: Path, provenance: dict[str, str]) -> None:
    manifest = read_training_manifest(manifest_path)
    if manifest["configuration_sha256"] != _configuration_digest(trainer):
        raise ValueError("resume configuration mismatch")
    if manifest["environment"] != _environment() or manifest["provenance"] != provenance:
        raise ValueError("resume environment or checkpoint mismatch")
    paths: dict[str, Path] = {}
    for shard in manifest["shards"]:
        if set(shard) != {"name", "sha256", "role"} or Path(shard["name"]).name != shard["name"]:
            raise ValueError("invalid checkpoint shard entry")
        path = manifest_path.parent / shard["name"]
        if shard["role"] in paths or sha256(path) != shard["sha256"]:
            raise ValueError("checkpoint shard hash mismatch or duplicate role")
        paths[shard["role"]] = path
    expected_roles = {"training"} | {
        f"simulator_{g}_{s}" for g, group in enumerate(trainer.engine.groups) for s in range(len(group))
    }
    if isinstance(trainer.policy, TransformerPolicy):
        expected_roles.add("actor")
        evaluation_actor_path(manifest_path)
    if set(paths) != expected_roles:
        raise ValueError("checkpoint simulator topology mismatch")
    saved = torch.load(paths["training"], map_location="cpu", weights_only=False)
    if saved["version"] != CHECKPOINT_VERSION or saved["update"] != manifest["update"]:
        raise ValueError("checkpoint shard identity mismatch")
    execution = saved.get("execution")
    if (
        not isinstance(execution, dict)
        or set(execution) != {"deterministic", "cublas_workspace_config"}
        or type(execution["deterministic"]) is not bool
        or (
            execution["cublas_workspace_config"] is not None
            and not isinstance(execution["cublas_workspace_config"], str)
        )
    ):
        raise ValueError("invalid checkpoint execution mode")
    if "actor" in paths:
        published = torch.load(paths["actor"], map_location="cpu", weights_only=False)
        if (
            published.get("format") != EVALUATION_FORMAT
            or published.get("decoder_precision") != trainer.policy.model.decoder_precision
        ):
            raise ValueError("checkpoint evaluation actor precision or format mismatch")
        actor = published["model"]
        if actor.keys() != saved["model"].keys() or any(
            not torch.equal(value, saved["model"][name]) for name, value in actor.items()
        ):
            raise ValueError("checkpoint training state and evaluation actor disagree")
    if trainer.policy.device.type == "cuda":
        torch.cuda.synchronize(trainer.policy.device)
    trainer.policy.model.load_state_dict(saved["model"], strict=True)
    trainer.actor_optimizer.load_state_dict(saved["actor_optimizer"])
    trainer.critic_optimizer.load_state_dict(saved["critic_optimizer"])
    trainer.scheduler.load_state_dict(saved["scheduler"])
    engine = trainer.engine
    engine.anchors.clear()
    for key, tensors in saved["anchors"].items():
        policy = trainer.policy.frozen_copy()
        policy.model.load_state_dict(tensors, strict=True)
        engine.anchors[key] = policy
    trainer.reference = engine.anchors[0]
    for name, value in saved["reference"].items():
        if not torch.equal(trainer.reference.model.state_dict()[name].cpu(), value):
            raise ValueError("checkpoint reference and anchor zero disagree")
    for group_id, group in enumerate(engine.groups):
        for shard_id, shard in enumerate(group):
            shard.restore(zlib.decompress(paths[f"simulator_{group_id}_{shard_id}"].read_bytes()))
    engine.restore_pool(saved["pool"])
    engine.clocks = saved["clocks"]
    engine.cursor = saved["cursor"]
    engine.anchor_candidates = saved["anchor_candidates"]
    engine.sampler.cursor = saved["sampler_cursor"]
    engine.sampler.lambda_max = saved["lambda_max"]
    engine._initialized = True
    trainer.update = saved["update"]
    trainer.updating = False
    engine.completed_update = trainer.update - 1
    trainer.actor_steps = saved["actor_steps"]
    trainer.warming = saved["warming"]
    trainer.warmup_moments = saved["warmup_moments"]
    trainer.fresh_variances = saved["fresh_variances"]
    trainer.next_anchor = saved["next_anchor"]
    trainer.evaluation_cursor = saved["evaluation_cursor"]
    trainer.policy.clear_caches()
    random.setstate(saved["python_rng"])
    np.random.set_state(saved["numpy_rng"])
    torch.set_rng_state(saved["torch_rng"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(saved["cuda_rng"])


def load_development_policy(
    checkpoint: Path,
    statistics: dict[str, FeatureStats],
    *,
    lambda_min: float = 0.01,
    lambda_max: float = 100.0,
    device: str = "cuda",
    compiled: bool = False,
    decoder_precision: DecoderPrecision = "bf16",
) -> TransformerPolicy:
    """Load the recorded O63 bootstrap30 checkpoint with fresh RL optimizer state."""
    if sha256(checkpoint) != PROXY_SHA256:
        raise ValueError("development checkpoint is not the recorded O63 bootstrap30 final.pt")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = saved["cfg"]
    if (config["experiment_id"], config["checkpoint_format_version"], config["arm"], saved["step"]) != (
        "063_percentile_conditioning_v1",
        5,
        "bootstrap30",
        16383,
    ):
        raise ValueError("development checkpoint identity mismatch")
    cfg = ActionSequenceConfig(
        d_model=256,
        n_layers=12,
        n_heads=4,
        temporal_d_model=256,
        temporal_layers=6,
        temporal_heads=4,
        temporal_ff_dim=1024,
        group_head_dim=256,
        value_hidden_dim=128,
    )
    if config["architecture"] != {name: getattr(cfg, name) for name in config["architecture"]}:
        raise ValueError("development checkpoint architecture mismatch")
    offsets = tuple(config["architecture"]["head_offsets"])
    schedule = development_config(offsets)
    if not 0 < lambda_min < lambda_max or not math.isfinite(lambda_max):
        raise ValueError("invalid development lambda range")
    state = saved["model"]
    vocabulary = vocabulary_from_checkpoint_buffer(state["player_code_bytes"].numpy())
    ranges = ConditioningRanges((math.log(lambda_min), math.log(lambda_max)), (8, 10), (2, 4))
    model = SelfPlayTransformer(
        cfg, ranges, vocabulary, max_opponent_queue=14, critic_hidden_dim=128, decoder_precision=decoder_precision
    )
    model.load_bc_state_dict(state)
    return TransformerPolicy(model.to(device).eval(), schedule, statistics, compiled=compiled)


def run_training(
    trainer: Trainer,
    updates: int,
    output: Path,
    provenance: dict[str, str],
    *,
    wandb_project: str | None = None,
    evaluation_shards: tuple[SimulatorShard, ...] | None = None,
    evaluation_config: EvaluationConfig | None = None,
    evaluation_interval: int = 128,
    periodic_evaluator: PeriodicEvaluator | None = None,
    run_configuration: dict[str, object] | None = None,
) -> None:
    """Train from initialized shards, publish checkpoints, and collect periodic results."""
    import wandb

    if updates <= trainer.update:
        raise ValueError("updates must exceed the restored update cursor")
    if (evaluation_shards is None) != (evaluation_config is None) or evaluation_interval < 1:
        raise ValueError("evaluation needs dedicated shards, a configuration, and a positive interval")
    output.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, object] = {
        "configuration_sha256": _configuration_digest(trainer),
        "environment": _environment(),
        "provenance": provenance,
        "train": dataclasses.asdict(trainer.config),
        "sampler": dataclasses.asdict(trainer.engine.sampler.config),
        "schedule": dataclasses.asdict(trainer.engine.sampler.schedule),
        "treatment": "KL-regularized joint PPO from pi_ref at u95",
        "control": "frozen pi_ref with matched sampling, starts, u95, and schedules",
        "invariants": "BC checkpoint, representation, data, simulator, adapter, reward, evaluation seeds",
        "run": run_configuration,
    }
    if isinstance(trainer.policy, TransformerPolicy):
        resolved["decoder_precision"] = trainer.policy.model.decoder_precision
    (output / "resolved.json").write_text(json.dumps(resolved, indent=2, sort_keys=True))
    with wandb.init(project=wandb_project, mode="disabled" if wandb_project is None else "online", config=resolved):
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")
        with (output / "metrics.jsonl").open("a") as log:
            while trainer.update < updates:
                batch, metrics = trainer.run_update()
                del batch
                if periodic_evaluator is not None:
                    completed = periodic_evaluator.poll()
                    if completed is not None:
                        metrics.update(periodic_evaluation_metrics(completed))
                    if trainer.update % periodic_evaluator.every == 0:
                        manifest = save_checkpoint(trainer, output / "checkpoints", provenance)
                        if periodic_evaluator.submit(manifest, update=trainer.update):
                            trainer.evaluation_cursor += 1
                if (
                    evaluation_shards is not None
                    and evaluation_config is not None
                    and trainer.update % evaluation_interval == 0
                ):
                    metrics.update(evaluate_anchors(trainer, evaluation_shards, evaluation_config))
                resolved["calibrated_lambda_max"] = trainer.engine.sampler.lambda_max
                (output / "resolved.json").write_text(json.dumps(resolved, indent=2, sort_keys=True))
                wandb.log(metrics)
                log.write(json.dumps(metrics, sort_keys=True) + "\n")
                log.flush()
                if trainer.update % trainer.config.checkpoint_interval == 0:
                    save_checkpoint(trainer, output / "checkpoints", provenance)
            save_checkpoint(trainer, output / "checkpoints", provenance)
            if periodic_evaluator is not None:
                completed = periodic_evaluator.finish()
                if completed is not None:
                    metrics = {"global_step": float(trainer.update), **periodic_evaluation_metrics(completed)}
                    wandb.log(metrics)
                    log.write(json.dumps(metrics, sort_keys=True) + "\n")


def periodic_evaluation_metrics(path: Path) -> dict[str, float]:
    result = json.loads(path.read_text())
    if (
        result.get("format") not in ("hal.o65.evaluation-results.v1", "hal.o65.evaluation-results.v2")
        or result.get("split") != "selection"
    ):
        raise ValueError("unsupported periodic evaluation result identity")
    metrics = {"evaluation/update": float(result["checkpoint_provenance"]["update"])}
    for index, group in enumerate(result["groups"]):
        for simulator, values in group["metrics"].items():
            prefix = f"evaluation/{simulator}_cell{index}"
            metrics[f"{prefix}_matches"] = float(values["matches"])
            metrics[f"{prefix}_completed_pairs"] = float(values["completed_pairs"])
            for key in ("win_rate", "net_stocks_per_min"):
                value = values[key]["mean"]
                if value is not None:
                    metrics[f"{prefix}_{key}"] = float(value)
            if values["net_stock_lcb"] is not None:
                metrics[f"{prefix}_net_stock_lcb"] = float(values["net_stock_lcb"])
    return metrics


@dataclasses.dataclass(frozen=True, slots=True)
class ProxyCli:
    checkpoint: Path
    statistics: Path
    output: Path
    device: str = "cuda"
    compiled: bool = False
    decoder_precision: DecoderPrecision = "bf16"


@dataclasses.dataclass(frozen=True, slots=True)
class NativeCli:
    checkpoint: Path
    sampler: Path
    output: Path
    data_dir: Path
    cpus: tuple[int, ...]
    updates: int
    groups: int = 4
    matches_per_shard: int = 2
    device: str = "cuda"
    compiled: bool = False
    max_frame: int = 28800
    resume: Path | None = None
    training: TrainConfig = TrainConfig()
    evaluation_every: int = 128
    evaluation_protocol: Path | None = None
    evaluation_simulators: tuple[Literal["msl", "dolphin"], ...] = ("msl", "dolphin")
    evaluation_device: str = "cpu"
    verify_resume: bool = False
    fixed_anchor_lanes: bool = True
    wandb_project: str | None = None
    profiling: ProfileConfig | None = None
    decoder_precision: DecoderPrecision = "bf16"
    replay_chunks: ReplayChunks = ReplayChunks()

    def __post_init__(self) -> None:
        validate_decoder_precision(self.decoder_precision)
        if self.profiling is not None and (
            self.device != "cuda"
            or self.training.critic_warmup_updates
            or self.resume is not None
            or self.evaluation_every
            or self.wandb_project is not None
            or self.verify_resume
        ):
            raise ValueError(
                "profiling requires CUDA, warmup off, fresh state, no evaluation, no uploads, and no resume check"
            )
        if type(self.fixed_anchor_lanes) is not bool:
            raise ValueError("fixed_anchor_lanes must be a boolean")
        if (
            not self.cpus
            or len(set(self.cpus)) != len(self.cpus)
            or any(cpu not in os.sched_getaffinity(0) for cpu in self.cpus)
        ):
            raise ValueError("native runs need distinct available CPUs")
        if any(type(value) is not int or value < 1 for value in (self.groups, self.matches_per_shard, self.updates)):
            raise ValueError("groups, matches per shard, and updates must be positive integers")
        if type(self.evaluation_every) is not int or self.evaluation_every < 0:
            raise ValueError("evaluation interval must be nonnegative")
        if self.max_frame != -1 and (type(self.max_frame) is not int or self.max_frame < 0):
            raise ValueError("max frame must be -1 or nonnegative")
        if self.profiling is None and self.replay_chunks != DEFAULT_REPLAY_CHUNKS:
            raise ValueError("nondefault replay chunks are qualified for profiling only")
        if self.profiling is not None and (
            self.profiling.warm_updates < self.groups or self.updates != self.profiling.warm_updates + 2
        ):
            raise ValueError("profiling must warm every group and run exactly two measured updates")


def read_native_sampler(path: Path) -> SamplerConfig:
    raw = json.loads(path.read_text())
    if (
        not isinstance(raw, dict)
        or set(raw) != {"format", "identities", "matchups", "stages", "stocks", "percents"}
        or raw["format"] != "hal.o65.sampler.v1"
    ):
        raise ValueError("unsupported native sampler schema")
    identities = []
    for row in raw["identities"]:
        if set(row) != {"identity", "character", "count"}:
            raise ValueError("invalid sampler identity fields")
        identities.append(IdentityCount(row["identity"], row["character"], row["count"]))
    matchups = []
    for row in raw["matchups"]:
        if set(row) != {"characters", "weight"} or len(row["characters"]) != 2:
            raise ValueError("invalid sampler matchup fields")
        matchups.append(Matchup((row["characters"][0], row["characters"][1]), row["weight"]))
    stocks = tuple(raw["stocks"])
    return SamplerConfig(
        tuple(identities), tuple(matchups), tuple(raw["stages"]), stocks=stocks, percents=tuple(raw["percents"])
    )


def train_native(config: NativeCli) -> None:
    torch.set_num_threads(2)
    if config.training.deterministic or config.verify_resume:
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace not in (None, ":4096:8"):
            raise ValueError("deterministic native runs require CUBLAS_WORKSPACE_CONFIG=:4096:8")
        if torch.cuda.is_initialized() and workspace is None:
            raise ValueError("configure deterministic execution before CUDA initialization")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(config.training.deterministic)
    if sha256(config.checkpoint) != PROXY_SHA256:
        raise ValueError("native development run requires the recorded O63 proxy")
    torch.manual_seed(config.training.seed)
    checkpoint = load_evaluation_checkpoint(
        config.checkpoint, device=torch.device(config.device), decoder_precision=config.decoder_precision
    )
    checkpoint.model.requires_grad_(True)
    checkpoint.model.percentile_head.requires_grad_(False)
    schedule = development_config(checkpoint.model.head_offsets)
    policy = TransformerPolicy(
        checkpoint.model, schedule, checkpoint.statistics, compiled=config.compiled, replay_chunks=config.replay_chunks
    )
    sampling = read_native_sampler(config.sampler)
    for row in sampling.identities:
        if row.identity >= policy.model.cfg.player_vocab_size:
            raise ValueError("sampler identity is outside the checkpoint vocabulary")
    manifests = checkpoint.provenance["data_manifests"]
    if not isinstance(manifests, dict) or set(manifests) != {"ranked-anonymized-1-policy-world-v8"}:
        raise ValueError("development checkpoint data manifests changed")
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if config.output.exists() and config.resume is None:
        raise FileExistsError(config.output)
    with contextlib.ExitStack() as resources:
        groups = []
        next_match = 0
        for _ in range(config.groups):
            shards: list[NativeSimulatorShard] = []
            for cpu in config.cpus:
                ids = tuple(range(next_match, next_match + config.matches_per_shard))
                shard = NativeSimulatorShard(
                    config.data_dir, ids, cpu=cpu, device=policy.device, max_frame=config.max_frame
                )
                resources.callback(shard.close)
                shards.append(shard)
                next_match += config.matches_per_shard
            groups.append(tuple(shards))
        provenance = {
            "git_sha": git_sha,
            "checkpoint_sha256": PROXY_SHA256,
            "data_manifest_sha256": str(cast(dict[str, object], manifests)["ranked-anonymized-1-policy-world-v8"]),
            "simulator_sha256": groups[0][0].shard.library_sha256,
            "adapter_sha256": native_bridge.library_sha256(),
        }
        engine = RolloutEngine(
            tuple(groups),
            policy,
            EpisodeSampler(sampling, schedule, CounterRng(config.training.seed)),
            {0: policy.frozen_copy()},
            fixed_anchor_lanes=config.fixed_anchor_lanes,
        )
        trainer = Trainer(engine, config.training)
        if config.resume is None:
            engine.initialize()
        else:
            restore_checkpoint(trainer, config.resume, provenance)
        evaluator = None
        if config.evaluation_every:
            arguments = [
                "--anchors",
                str(config.checkpoint.resolve()),
                "--data-dir",
                str(config.data_dir.resolve()),
                "--simulators",
                *config.evaluation_simulators,
                "--device",
                config.evaluation_device,
                "--decoder-precision",
                config.decoder_precision,
                "--cadence-grid",
                "8/4",
                "10/2",
            ]
            if config.evaluation_protocol is not None:
                arguments.extend(("--protocol", str(config.evaluation_protocol.resolve())))
            evaluator = PeriodicEvaluator(
                config.output / "evaluation", every=config.evaluation_every, arguments=tuple(arguments)
            )
            evaluator.last_update = trainer.update
            resources.callback(evaluator.close)
        native_config = json.loads(json.dumps(dataclasses.asdict(config), default=str))
        native_config["simulator_manifest_sha256"] = sha256(config.data_dir / "raw" / "manifest.json")
        native_config["sampler_sha256"] = sha256(config.sampler)
        if config.profiling is not None:
            profile_native(trainer, config, provenance, native_config)
            return
        run_training(
            trainer,
            config.updates,
            config.output,
            provenance,
            wandb_project=config.wandb_project,
            periodic_evaluator=evaluator,
            run_configuration=native_config,
        )
        if config.verify_resume:
            verify_checkpoint_resume(trainer, config.output, provenance)


def validate_profile_sites(sites: set[str], *, bootstrap: bool) -> None:
    required = {"rollout/tick", "rollout/boundary_tick", "reference/windows", "critic/fit", "actor/step"}
    if bootstrap:
        required.add("preparation/bootstrap")
    missing = required - sites
    if missing:
        raise ValueError(f"profile is missing required sites: {sorted(missing)}")


def profile_native(
    trainer: Trainer,
    config: NativeCli,
    provenance: dict[str, str],
    native_config: dict[str, object],
) -> None:
    if config.profiling is None:
        raise ValueError("missing profile configuration")
    config.output.mkdir(parents=True, exist_ok=False)
    (config.output / "resolved.json").write_text(
        json.dumps(
            {
                "format": "hal.o65.gpu-profile-inputs.v1",
                "provenance": provenance,
                "trainer_source_sha256": sha256(Path(__file__)),
                "configuration": native_config,
                "environment": _environment(),
                "treatment": "bounded profiler instrumentation",
                "control": "same configuration without tracing",
                "invariants": "checkpoint, sampler, seed, representation, reward, optimizer, simulator and adapter",
                "gpu": torch.cuda.get_device_name(),
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "clock_ticks_per_second": os.sysconf("SC_CLK_TCK"),
                "flop_model": {
                    "multiply_add_flops": 2,
                    "trunk_parameter_equivalent": 9_400_000,
                    "decoder_parameter_equivalent": 6_700_000,
                    "actor_forward_backward_factor": 3,
                    "critic_parameters": sum(p.numel() for p in trainer.policy.critic_parameters()),
                    "formula": "2*9.4e6*trunk_tokens + 2*6.7e6*decoder_tokens; actor times 3; critic 6*parameters*rows*epochs",
                    "excludes": "attention score/value products, nonlinearities, feature building, optimizer, copies",
                    "decoder_precision": config.decoder_precision,
                    "trunk_precision": "BF16",
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    with (config.output / "warm.jsonl").open("x") as log:
        for _ in range(config.profiling.warm_updates):
            batch, metrics = trainer.run_update()
            del batch
            log.write(json.dumps(metrics) + "\n")
            log.flush()
    recorders: list[UpdateProfile] = []
    try:
        for traces in (False, True):
            recorder = UpdateProfile(config.output, trainer.policy.device, traces=traces)
            recorders.append(recorder)
            trainer.profile = trainer.engine.profile = recorder
            for policy in (trainer.policy, *trainer.engine.anchors.values()):
                if not isinstance(policy, TransformerPolicy):
                    raise TypeError("profiling requires TransformerPolicy")
                policy.profile = recorder
            sampler = contextlib.nullcontext() if traces else HostSampler(config.output)
            with sampler:
                batch, metrics = trainer.run_update()
            del batch
            recorder.write(metrics)
        estimates = estimate_profile(*recorders)
        validate_profile_sites(
            set(cast(dict[str, object], estimates["sites"])),
            bootstrap=bool(recorders[0].calls.get("preparation/bootstrap", 0)),
        )
        critic_flops = (
            6
            * sum(p.numel() for p in trainer.policy.critic_parameters())
            * recorders[0].work["critic/fit"]
            * trainer.config.critic_epochs
        )
        cast(dict[str, float], estimates["flops"])["critic"] = critic_flops
        cast(dict[str, float], estimates["tflops_per_wall_second"])["critic"] = (
            critic_flops / recorders[0].phases["critic"]["wall_seconds"] / 1e12
        )
        (config.output / "estimates.json").write_text(json.dumps(estimates, indent=2, sort_keys=True))
    finally:
        trainer.profile = trainer.engine.profile = None
        for policy in (trainer.policy, *trainer.engine.anchors.values()):
            if isinstance(policy, TransformerPolicy):
                policy.profile = None


@dataclasses.dataclass(frozen=True, slots=True)
class GpuProfileCli:
    checkpoint: Path
    data_dir: Path
    output: Path
    sampler: Path = Path("experiments/o65/pilot-sampler.json")
    cpus: tuple[int, ...] = tuple(range(16))
    matches_per_shard: int = 64
    groups: int = 4
    warm_updates: int = 4
    decoder_precision: DecoderPrecision = "bf16"
    replay_chunks: ReplayChunks = ReplayChunks()

    def native(self) -> NativeCli:
        return NativeCli(
            self.checkpoint,
            self.sampler,
            self.output,
            self.data_dir,
            self.cpus,
            updates=self.warm_updates + 2,
            groups=self.groups,
            matches_per_shard=self.matches_per_shard,
            compiled=True,
            training=TrainConfig(critic_warmup_updates=0, windows_per_step=32),
            evaluation_every=0,
            profiling=ProfileConfig(self.warm_updates),
            decoder_precision=self.decoder_precision,
            replay_chunks=self.replay_chunks,
        )


def _state_hash(value: object) -> str:
    digest = hashlib.sha256()

    def visit(item: object) -> None:
        if isinstance(item, Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype, item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(key)
                visit(child)
        elif isinstance(item, tuple | list):
            digest.update(str(len(item)).encode())
            for child in item:
                visit(child)
        elif item is None or isinstance(item, str | int | float | bool):
            digest.update(json.dumps(item).encode() + b"\x00")
        else:
            raise TypeError(f"unsupported resume diagnostic value: {type(item)}")

    visit(value)
    return digest.hexdigest()


def _batch_hash(batch: PreparedUpdate) -> str:
    return _state_hash(
        {
            "decisions": {
                field.name: getattr(batch.decisions, field.name) for field in dataclasses.fields(batch.decisions)
            },
            "selected": batch.selected,
            "steps": batch.steps,
            "values": batch.values,
            "advantages": batch.advantages,
            "targets": batch.targets,
            "old": batch.old.log_probs,
            "reference": None if batch.reference is None else batch.reference.log_probs,
            "observations": tuple(
                (
                    record.state.stream,
                    record.state.clock,
                    record.state.window,
                    dataclasses.asdict(record.state.episode),
                    record.state.opponent_queue,
                    tuple(
                        (dict(frame.columns), frame.applied_actions, frame.rewards, frame.terminal, frame.timeout)
                        for frame in record.state.frames
                    ),
                )
                for record in batch.rollout.records
            ),
        }
    )


def verify_checkpoint_resume(trainer: Trainer, output: Path, provenance: dict[str, str]) -> None:
    """Compare two deterministic next updates, then restore the training boundary."""
    if trainer.policy.device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("configure CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA resume verification")
    manifest = save_checkpoint(trainer, output / "checkpoints", provenance)
    config = trainer.config
    deterministic = torch.are_deterministic_algorithms_enabled()
    trainer.config = dataclasses.replace(config, deterministic=True)
    try:
        torch.use_deterministic_algorithms(True)
        _clear_execution_caches(trainer)
        restore_checkpoint(trainer, manifest, provenance)
        _verify_checkpoint_resume(trainer, output, provenance, manifest)
    finally:
        try:
            restore_checkpoint(trainer, manifest, provenance)
        finally:
            trainer.config = config
            torch.use_deterministic_algorithms(deterministic)
            _clear_execution_caches(trainer)


def _clear_execution_caches(trainer: Trainer) -> None:
    # A graph retains the algorithms selected during capture.
    for policy in (trainer.policy, *trainer.engine.anchors.values()):
        policy.clear_caches()
        if isinstance(policy, TransformerPolicy):
            policy.rollout_caches.clear()
            policy.decoder.graphs.clear()
            policy.tick_caches.clear()
            policy.tick_graphs.clear()
            policy.current_layouts.clear()
            policy.window_graphs.clear()
            policy.value_graphs.clear()
            policy.actor_graph = None


def _verify_checkpoint_resume(trainer: Trainer, output: Path, provenance: dict[str, str], manifest: Path) -> None:
    boundary_optimizer = _state_hash(
        (trainer.actor_optimizer.state_dict(), trainer.critic_optimizer.state_dict(), trainer.scheduler.state_dict())
    )
    started = time.perf_counter()
    expected, expected_metrics = trainer.run_update()
    expected_batch = _batch_hash(expected)
    expected_model = {name: value.detach().cpu().clone() for name, value in trainer.policy.model.state_dict().items()}
    expected_optimizer = _state_hash(
        (trainer.actor_optimizer.state_dict(), trainer.critic_optimizer.state_dict(), trainer.scheduler.state_dict())
    )
    del expected
    restore_checkpoint(trainer, manifest, provenance)
    restored_optimizer = _state_hash(
        (trainer.actor_optimizer.state_dict(), trainer.critic_optimizer.state_dict(), trainer.scheduler.state_dict())
    )
    if restored_optimizer != boundary_optimizer:
        raise ValueError("checkpoint did not restore the exact optimizer and scheduler state")
    actual, actual_metrics = trainer.run_update()
    actual_batch = _batch_hash(actual)
    actual_optimizer = _state_hash(
        (trainer.actor_optimizer.state_dict(), trainer.critic_optimizer.state_dict(), trainer.scheduler.state_dict())
    )
    maximum = max(
        float((expected_model[name].float() - value.detach().cpu().float()).abs().max())
        for name, value in trainer.policy.model.state_dict().items()
        if value.numel()
    )
    report = {
        "format": "hal.o65.resume-check.v1",
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "manifest": str(manifest),
        "next_batch_expected_sha256": expected_batch,
        "next_batch_actual_sha256": actual_batch,
        "optimizer_expected_sha256": expected_optimizer,
        "optimizer_actual_sha256": actual_optimizer,
        "boundary_optimizer_sha256": boundary_optimizer,
        "restored_optimizer_sha256": restored_optimizer,
        "maximum_parameter_error": maximum,
        "uninterrupted_metrics": expected_metrics,
        "resumed_metrics": actual_metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "resume-check.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    if expected_batch != actual_batch or expected_optimizer != actual_optimizer or maximum != 0:
        raise ValueError(f"exact native resume failed; see {output / 'resume-check.json'}")


def qualify_development(config: ProxyCli) -> None:
    raw = json.loads(config.statistics.read_text())
    if not isinstance(raw, dict) or any(set(value) != {"mean", "std", "min", "max"} for value in raw.values()):
        raise ValueError("statistics must contain finalized FeatureStats objects")
    stats = {
        name: FeatureStats(value["mean"], value["std"], value["min"], value["max"]) for name, value in raw.items()
    }
    policy = load_development_policy(
        config.checkpoint,
        stats,
        device=config.device,
        compiled=config.compiled,
        decoder_precision=config.decoder_precision,
    )
    report = {
        "checkpoint_sha256": PROXY_SHA256,
        "schedule": dataclasses.asdict(policy.schedule),
        "parameters": sum(p.numel() for p in policy.model.parameters()),
        "actor_parameters": sum(p.numel() for p in policy.actor_parameters()),
        "critic_parameters": sum(p.numel() for p in policy.critic_parameters()),
        "statistics_sha256": sha256(config.statistics),
        "decoder_precision": policy.model.decoder_precision,
        "environment": _environment(),
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(report, sort_keys=True, indent=2))


@dataclasses.dataclass(frozen=True, slots=True)
class EvaluationConfig:
    lambdas: tuple[float, ...]
    seed: int
    matches_per_cell: int
    max_rollouts: int = 64

    def __post_init__(self) -> None:
        if not self.lambdas or any(not math.isfinite(x) or x <= 0 for x in self.lambdas):
            raise ValueError("evaluation lambdas must be finite and positive")
        if min(self.matches_per_cell, self.max_rollouts) < 1:
            raise ValueError("evaluation needs positive match and rollout counts")


@torch.no_grad()
def _episode_diagnostics(
    policy: LoopPolicy,
    reference: LoopPolicy,
    records: Sequence[DecisionRecord | DecisionRecordView],
    results: dict[int, EpisodeResult],
) -> dict[str, float]:
    for record in records:
        record.frames_remaining = results[record.state.match].duration - record.state.clock
    batch = _decision_batch(records, policy.device)
    actions = torch.stack([record.actions for record in records]).to(policy.device)
    states = [record.state for record in records]
    current = policy.evaluate(states, actions)
    control = reference.evaluate(states, actions)
    kl = _factor_sum(factor_kl(current, control, batch.executed_factors_JF), batch.executed_factors_JF)
    metrics = {
        "kl_cost": float((kl * batch.kl_coefficient_J).sum()) / len(results),
        "kl_per_frame": float(kl.sum()) / sum(result.duration for result in results.values()),
    }
    rewards = torch.zeros(len(records), 4, device=policy.device)
    by_match: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        rewards[index, : len(record.rewards)] = torch.tensor(record.rewards, device=policy.device)
        if not results[record.state.match].timeout:
            by_match[record.state.match].append(index)
    regularized = regularized_reward(batch, rewards, factor_kl(current, control, batch.executed_factors_JF))
    targets = torch.zeros(len(records), device=policy.device)
    selected = torch.zeros(len(records), device=policy.device, dtype=torch.bool)
    for indices in by_match.values():
        tail = torch.zeros((), device=policy.device)
        for index in reversed(indices):
            tail = regularized[index] + GAMMA ** len(records[index].rewards) * tail
            targets[index] = tail
            selected[index] = True
    metrics["monte_carlo_count"] = float(selected.sum())
    if bool(selected.any()):
        values = policy.values(current.features)[selected]
        metrics["monte_carlo_rmse"] = float((values - targets[selected]).square().mean().sqrt())
        metrics["monte_carlo_explained_variance"] = explained_variance(values, targets[selected])
    return metrics


@torch.no_grad()
def evaluate_anchors(
    trainer: Trainer, shards: tuple[SimulatorShard, ...], config: EvaluationConfig
) -> dict[str, float]:
    """Fixed first-episode seeds and starts, separate from every training lane.

    Every cell uses the same logical matches at matched D and R. An incomplete
    cell is an error, rather than a selection biased toward short matches.
    The caller supplies dedicated shards and controls the evaluation interval.
    """
    if any(shard is training for shard in shards for group in trainer.engine.groups for training in group):
        raise ValueError("evaluation cannot step a training shard")
    ids = sorted(match for shard in shards for match in shard.match_ids)
    if len(ids) < config.matches_per_cell:
        raise ValueError("evaluation requires one lane per fixed initial match")
    wanted = set(ids[: config.matches_per_cell])
    metrics: dict[str, float] = {}
    for anchor_id in trainer.engine.anchor_candidates:
        for timing in trainer.engine.sampler.schedule.configurations:
            for bin_id, coefficient in enumerate(config.lambdas):
                sampler_config = dataclasses.replace(
                    trainer.engine.sampler.config, lambda_min=coefficient, lambda_max=coefficient, anchor_rate=1.0
                )
                schedule = ScheduleConfig((timing,), trainer.engine.sampler.schedule.supervised_offsets)
                evaluation_seed = int(CounterRng(config.seed).uniform("evaluation", 0, 0, 0) * 2**63)
                sampling = EpisodeSampler(sampler_config, schedule, CounterRng(evaluation_seed))
                policy = trainer.policy.frozen_copy()
                anchors = {0: trainer.engine.anchors[anchor_id].frozen_copy()}
                engine = RolloutEngine((shards,), policy, sampling, anchors)
                engine.initialize()
                results: dict[int, EpisodeResult] = {}
                records: list[DecisionRecord | DecisionRecordView] = []
                for update in range(config.max_rollouts):
                    rollout = engine.collect(update)
                    retained = [
                        record
                        for record in rollout.records
                        if record.state.match in wanted and record.state.episode.number == 0
                    ]
                    # Evaluation retains episodes across reuse of the dense group buffers.
                    records.extend(copy.deepcopy(retained))
                    for result in rollout.results:
                        if result.match in wanted and result.episode.number == 0:
                            results[result.match] = result
                    if set(results) == wanted:
                        break
                if set(results) != wanted:
                    raise ValueError("evaluation cell did not finish all fixed matches")
                scores: list[float] = []
                lengths: list[float] = []
                stocks: list[float] = []
                wins: list[float] = []
                for result in results.values():
                    student = 0 if result.episode.players[0].anchor is None else 1
                    scores.append(result.returns[student])
                    lengths.append(float(result.duration))
                    if result.stocks is None:
                        raise ValueError("gameplay evaluation requires stock columns")
                    stock = result.stocks[student] - result.stocks[1 - student]
                    initial = result.episode.players[student].stocks - result.episode.players[1 - student].stocks
                    stocks.append(float(stock - initial))
                    if not result.timeout:
                        wins.append(float(stock > 0) + 0.5 * float(stock == 0))
                suffix = f"a{anchor_id}_d{timing.delay}_r{timing.replan}_lambda{bin_id}"
                metrics[f"eval/return_{suffix}"] = float(np.mean(scores))
                metrics[f"eval/match_frames_{suffix}"] = float(np.mean(lengths))
                metrics[f"eval/net_stocks_{suffix}"] = float(np.mean(stocks))
                metrics[f"eval/timeout_fraction_{suffix}"] = 1 - len(wins) / len(results)
                if wins:
                    metrics[f"eval/win_rate_{suffix}"] = float(np.mean(wins))
                diagnostics = _episode_diagnostics(policy, trainer.reference, records, results)
                for name, value in diagnostics.items():
                    metrics[f"eval/{name}_{suffix}"] = value
    trainer.evaluation_cursor += 1
    return metrics


def main() -> None:
    import argparse

    import tyro

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train", add_help=False)
    commands.add_parser("qualify", add_help=False)
    commands.add_parser("profile", add_help=False)
    args, remaining = parser.parse_known_args()
    if args.command == "train":
        train_native(tyro.cli(NativeCli, args=remaining))
    elif args.command == "profile":
        train_native(tyro.cli(GpuProfileCli, args=remaining).native())
    else:
        qualify_development(tyro.cli(ProxyCli, args=remaining))


if __name__ == "__main__":
    main()
