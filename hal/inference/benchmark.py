"""Measure complete prediction delivery for a declared request shape."""

import math
import multiprocessing as mp
import threading
import time
from collections import deque
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path

import numpy as np
import torch

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.inference.action_sequence_artifact import load_action_sequence_policy
from hal.inference.api import PolicyInput
from hal.inference.api import PolicySpec
from hal.inference.api import PredictionPolicy
from hal.inference.api import PredictionRequest
from hal.inference.api import PreparedInferenceProfile
from hal.inference.api import RuntimeConfig
from hal.inference.client import InferenceClient
from hal.inference.client import StopSignal
from hal.inference.engine import InferenceEngine
from hal.inference.engine import configure_inference_process
from hal.inference.engine import freeze_inference_runtime
from hal.inference.warmup import make_warmup_observations

WARMUP_CALLS = 20
MEASURED_CALLS = 200
MAX_DECODE_TIMING_SAMPLES = 8192


def _decode_timing_samples() -> deque[float]:
    return deque(maxlen=MAX_DECODE_TIMING_SAMPLES)


def _serve_prediction_worker(
    worker: InferenceEngine,
    stop: threading.Event,
    lost: threading.Event,
    errors: list[BaseException],
) -> None:
    try:
        worker.serve(stop)
    except BaseException as error:
        errors.append(error)
        lost.set()


@dataclass(slots=True)
class DecodeTelemetry:
    """Timing and batch-size statistics for policy decode calls."""

    calls: int = 0
    rows: int = 0
    executed_frames: int = 0
    seconds: float = 0.0
    max_seconds: float = 0.0
    calls_over_100ms: int = 0
    durations: deque[float] = field(default_factory=_decode_timing_samples)

    def record(self, *, rows: int, horizon: int, seconds: float) -> None:
        self.calls += 1
        self.rows += rows
        self.executed_frames += rows * horizon
        self.seconds += seconds
        self.max_seconds = max(self.max_seconds, seconds)
        self.calls_over_100ms += int(seconds > 0.1)
        self.durations.append(seconds)

    def metrics(self) -> dict[str, float]:
        durations = np.asarray(self.durations, dtype=np.float64)
        return {
            "decode_calls": float(self.calls),
            "decode_rows": float(self.rows),
            "decode_seconds": self.seconds,
            "decode_max_seconds": self.max_seconds,
            "decode_p50_ms": float(np.percentile(durations, 50) * 1000) if durations.size else 0.0,
            "decode_p95_ms": float(np.percentile(durations, 95) * 1000) if durations.size else 0.0,
            "decode_p99_ms": float(np.percentile(durations, 99) * 1000) if durations.size else 0.0,
            "decode_calls_over_100ms": float(self.calls_over_100ms),
            "decode_mean_rows": self.rows / max(self.calls, 1),
            "decode_replans_per_s": self.calls / max(self.seconds, 1e-12),
            "decode_executed_frames_per_s": self.executed_frames / max(self.seconds, 1e-12),
        }


@dataclass(frozen=True, slots=True)
class LatencyMeasurement:
    prediction_horizon_frames: int
    fixed_prefix_frames: int
    seconds: tuple[float, ...]

    @property
    def p99_seconds(self) -> float:
        if not self.seconds:
            raise ValueError("latency measurement has no samples")
        return sorted(self.seconds)[math.ceil(0.99 * len(self.seconds)) - 1]


@dataclass(frozen=True, slots=True)
class _BenchmarkReady:
    spec: PolicySpec
    context_frames: int
    checkpoint_sha256: str
    preparation_seconds: float
    peak_allocated_mib: float


@dataclass(frozen=True, slots=True)
class _BenchmarkDone:
    batch_calls: int
    batch_items: int
    max_batch_items: int
    timing_p95_ms: tuple[float | None, float | None]


@dataclass(frozen=True, slots=True)
class _BenchmarkError:
    name: str
    reason: str


def measure_prediction_shape(
    policy: PredictionPolicy,
    runtime: RuntimeConfig,
    horizon: int,
    prefix_frames: int,
    batch_wait_seconds: float,
) -> LatencyMeasurement:
    """Time IPC, batching, validation, and the model at full configured load."""
    delay = runtime.require_single_delay()
    if prefix_frames < delay or prefix_frames >= horizon:
        raise ValueError("prediction shape must cover the physical delay and leave a predicted tail")
    if not math.isfinite(batch_wait_seconds) or batch_wait_seconds < 0:
        raise ValueError("batch wait must be finite and non-negative")
    request_interval = runtime.replan_interval_frames or max(1, prefix_frames - delay)
    if request_interval > policy.context_frames:
        raise ValueError("replan interval exceeds the observation history capacity")
    torch.compiler.reset()
    policy.prepare_prediction(runtime, horizon, prefix_frames)
    checkpoint_sha256 = policy.checkpoint_sha256
    if checkpoint_sha256 is None:
        raise ValueError("prediction benchmarking requires a checkpoint-backed policy")
    profile = PreparedInferenceProfile(
        "qualification",
        checkpoint_sha256,
        policy.history_mode,
        horizon,
        prefix_frames,
        policy.prepared_update_shapes,
        runtime.max_batch_size,
    )
    stop = threading.Event()
    lost = threading.Event()
    errors: list[BaseException] = []
    samples = []
    with ExitStack() as cleanup:
        pairs = []
        for _ in range(runtime.max_batch_size):
            parent, child = Pipe()
            cleanup.callback(parent.close)
            cleanup.callback(child.close)
            pairs.append((parent, child))
        parents = {slot: pair[0] for slot, pair in enumerate(pairs)}
        clients = [
            InferenceClient(
                policy.spec,
                policy.context_frames,
                pair[1],
                lost,
                {prefix_frames: profile},
                timeout_seconds=120.0,
            )
            for pair in pairs
        ]
        for client in clients:
            cleanup.callback(client.close)
        worker = InferenceEngine({profile: policy}, parents, batch_wait_seconds=batch_wait_seconds)
        worker.forbid_compilation = False

        thread = threading.Thread(target=_serve_prediction_worker, args=(worker, stop, lost, errors), daemon=True)
        try:
            thread.start()
            policy.reset_prediction()
            for slot, client in enumerate(clients):
                client.start_match(slot, prefix_frames)
            worker.forbid_compilation = False
            for sequence in range(WARMUP_CALLS + MEASURED_CALLS):
                if sequence == WARMUP_CALLS:
                    worker.forbid_compilation = True
                    freeze_inference_runtime()
                source = policy.context_frames + sequence * request_interval
                observation_count = policy.context_frames if sequence == 0 else request_interval
                for slot, client in enumerate(clients):
                    client.submit(
                        PredictionRequest(
                            slot,
                            1,
                            sequence,
                            source,
                            make_warmup_observations(
                                policy.spec,
                                observation_count,
                                slot,
                                source,
                                delay,
                                reset_first=sequence == 0,
                            ),
                            (NEUTRAL_CONTROLLER_ACTION,) * prefix_frames,
                        )
                    )
                deadline = time.monotonic() + 120
                while any(client.busy for client in clients):
                    for client in clients:
                        client.poll()
                    if time.monotonic() > deadline:
                        raise TimeoutError("prediction benchmark request did not finish within 120 seconds")
                    time.sleep(0.0001)
                if sequence >= WARMUP_CALLS:
                    samples.append(max(client.last_latency for client in clients))
        finally:
            for client in clients:
                if not lost.is_set():
                    client.close_match()
            stop.set()
            if thread.is_alive():
                thread.join(timeout=2)
            if thread.is_alive():
                raise RuntimeError("prediction benchmark worker did not stop; policy cannot be reused")
            policy.reset_prediction()
    if errors:
        raise RuntimeError("prediction benchmark inference failed") from errors[0]
    return LatencyMeasurement(horizon, prefix_frames, tuple(samples))


def _serve_process_batch_benchmark(
    bundle: Path,
    capacity: int,
    stop: StopSignal,
    connections: tuple[Connection, ...],
    status: Connection,
) -> None:
    """Prepare one model in a spawned GPU process before admitting any streams."""
    try:
        configure_inference_process()
        started = time.perf_counter()
        policy = load_action_sequence_policy(
            bundle,
            device="cuda",
            seed=1001,
            compiled=True,
            history_mode="kv_cache",
            kv_update_frames=4,
            kv_cuda_graphs=True,
        )
        policy.prepare_prediction(RuntimeConfig(capacity, (2,), replan_interval_frames=4), 8, 3)
        torch.cuda.synchronize()
        preparation_seconds = time.perf_counter() - started
        peak_allocated_mib = torch.cuda.max_memory_allocated() / 2**20
        freeze_inference_runtime()
        profile = PreparedInferenceProfile(
            "batching-3060", policy.checkpoint_sha256, "kv_cache", 8, 3, (1, 2, 4), capacity
        )
        engine = InferenceEngine(
            {profile: policy},
            {index: connection for index, connection in enumerate(connections)},
            batch_wait_seconds=0.0005,
        )
        status.send(
            _BenchmarkReady(
                policy.spec, policy.context_frames, policy.checkpoint_sha256, preparation_seconds, peak_allocated_mib
            )
        )
        engine.serve(stop)
        status.send(
            _BenchmarkDone(engine.batch_calls, engine.batch_items, engine.max_batch_items, engine.timing_p95_ms())
        )
    except Exception as error:
        status.send(_BenchmarkError(type(error).__name__, str(error)))
        raise
    finally:
        status.close()
        for connection in connections:
            connection.close()


def _receive_process_status(
    process: BaseProcess, status: Connection, timeout_seconds: float
) -> _BenchmarkReady | _BenchmarkDone | _BenchmarkError:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if status.poll(0.1):
            message = status.recv()
            if isinstance(message, (_BenchmarkReady, _BenchmarkDone, _BenchmarkError)):
                return message
            raise RuntimeError("inference benchmark process sent invalid status")
        if not process.is_alive():
            raise RuntimeError(f"inference benchmark process exited with status {process.exitcode}")
    raise TimeoutError("inference benchmark process did not report readiness in time")


def _await_plan(client: InferenceClient, process: BaseProcess, lost: threading.Event) -> None:
    while client.busy:
        if not process.is_alive():
            lost.set()
        client.poll()
        time.sleep(0.00005)


def measure_process_ready_pair(
    bundle: Path,
    frames: Sequence[PolicyInput],
    *,
    capacity: int,
    concurrent: bool,
    warmup_calls: int = 20,
    measured_calls: int = 200,
    preparation_timeout_seconds: float = 300.0,
) -> dict[str, object]:
    """Measure two serial or concurrent ready responses through a spawned engine."""
    if capacity < 2 or warmup_calls < 1 or measured_calls < 1:
        raise ValueError("batching benchmark needs two rows and positive warmup and measurement counts")
    if not math.isfinite(preparation_timeout_seconds) or preparation_timeout_seconds <= 0:
        raise ValueError("benchmark preparation timeout must be finite and positive")
    if len(frames) < 4 * (warmup_calls + measured_calls):
        raise ValueError("replay does not contain the requested benchmark frames")
    context = mp.get_context("spawn")
    stop = context.Event()
    lost = threading.Event()
    pairs = [context.Pipe() for _ in range(capacity)]
    status_parent, status_child = context.Pipe(duplex=False)
    process = context.Process(
        target=_serve_process_batch_benchmark,
        args=(bundle, capacity, stop, tuple(parent for parent, _ in pairs), status_child),
    )
    clients: list[InferenceClient] = []
    measurements: list[float] = []
    first_ids = (1901, 9001)
    stream_ids = (*first_ids, *(20_000 + index for index in range(capacity - 2)))
    startup_started = time.perf_counter()
    try:
        process.start()
        for parent, _ in pairs:
            parent.close()
        status_child.close()
        ready = _receive_process_status(process, status_parent, preparation_timeout_seconds)
        if not isinstance(ready, _BenchmarkReady):
            raise RuntimeError(f"inference benchmark preparation failed: {ready}")
        profile = PreparedInferenceProfile(
            "batching-3060", ready.checkpoint_sha256, "kv_cache", 8, 3, (1, 2, 4), capacity
        )
        clients = [InferenceClient(ready.spec, ready.context_frames, child, lost, {3: profile}) for _, child in pairs]
        for stream_id, client in zip(stream_ids, clients, strict=True):
            client.start_match(stream_id, 3)
        startup_seconds = time.perf_counter() - startup_started
        for sequence in range(warmup_calls + measured_calls):
            start = sequence * 4
            requests = tuple(
                PredictionRequest(
                    stream_id,
                    client.generation,
                    sequence,
                    start + 3,
                    tuple(
                        replace(frames[frame], stream_id=stream_id, reset=sequence == 0 and frame == start)
                        for frame in range(start, start + 4)
                    ),
                    (NEUTRAL_CONTROLLER_ACTION,) * 3,
                )
                for stream_id, client in zip(first_ids, clients[:2], strict=True)
            )
            started = time.perf_counter()
            for client, request in zip(clients[:2], requests, strict=True):
                client.submit(request)
                if not concurrent:
                    _await_plan(client, process, lost)
            if concurrent:
                for client in clients[:2]:
                    _await_plan(client, process, lost)
            elapsed = time.perf_counter() - started
            if sequence >= warmup_calls:
                measurements.append(elapsed)
        for client in clients:
            client.close_match()
        stop.set()
        worker_stats = _receive_process_status(process, status_parent, 5.0)
        if not isinstance(worker_stats, _BenchmarkDone):
            raise RuntimeError(f"inference benchmark worker failed: {worker_stats}")
        process.join(timeout=2.0)
        if process.exitcode != 0:
            raise RuntimeError(f"inference benchmark worker exited with status {process.exitcode}")
        values = np.asarray(measurements, dtype=np.float64) * 1000
        return {
            "capacity": capacity,
            "ready_streams": 2,
            "concurrent": concurrent,
            "warmup_calls": warmup_calls,
            "measured_calls": measured_calls,
            "preparation_timeout_seconds": preparation_timeout_seconds,
            "preparation_seconds": ready.preparation_seconds,
            "startup_seconds": startup_seconds,
            "peak_allocated_mib": ready.peak_allocated_mib,
            "complete_pair_ms": dict(
                zip(("p50", "p95", "p99"), np.percentile(values, (50, 95, 99)).tolist(), strict=True)
            ),
            "pair_samples_ms": values.tolist(),
            "engine_batch_calls": worker_stats.batch_calls,
            "engine_batch_items": worker_stats.batch_items,
            "engine_max_batch_items": worker_stats.max_batch_items,
            "engine_timing_p95_ms": worker_stats.timing_p95_ms,
        }
    finally:
        stop.set()
        for client in clients:
            client.close()
        for parent, child in pairs:
            parent.close()
            child.close()
        status_parent.close()
        status_child.close()
        if process.pid is not None:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
