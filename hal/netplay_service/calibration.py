"""Qualify the complete chunk request path before accepting a match."""

import math
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import Pipe

import torch

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.inference.api import RuntimeConfig
from hal.inference.chunks import ChunkPolicy
from hal.inference.chunks import ChunkRequest
from hal.inference.chunks import TimingSchedule
from hal.inference.chunks import latency_frames
from hal.netplay_service.chunks import ChunkBatcher
from hal.netplay_service.chunks import RemoteChunkPolicy

WARMUP_CALLS = 20
MEASURED_CALLS = 200


@dataclass(frozen=True, slots=True)
class Measurement:
    horizon: int
    prefix_frames: int
    seconds: tuple[float, ...]

    @property
    def p99_seconds(self) -> float:
        return sorted(self.seconds)[math.ceil(0.99 * len(self.seconds)) - 1]


@dataclass(frozen=True, slots=True)
class Calibration:
    schedules: tuple[TimingSchedule, ...]
    measurements: tuple[Measurement, ...]


def select_schedule(measurements: tuple[Measurement, ...], transport: int) -> TimingSchedule:
    candidates = []
    for measurement in measurements:
        compute = latency_frames(measurement.p99_seconds)
        # The measured forced-prefix shape must itself cover the measured latency.
        budget = measurement.prefix_frames - transport
        if budget == compute + 1 and measurement.horizon >= budget * 2 + transport:
            candidates.append(TimingSchedule(budget - 1, transport, measurement.horizon))
    if not candidates:
        raise RuntimeError("server unavailable: no trained horizon can cover measured inference and transport")
    return min(candidates, key=lambda schedule: (schedule.budget_frames, -schedule.horizon))


def measure_shape(
    policy: ChunkPolicy, runtime: RuntimeConfig, horizon: int, prefix: int, batch_wait: float
) -> Measurement:
    torch.compiler.reset()
    policy.prepare_chunks(runtime, horizon, prefix)
    stop = threading.Event()
    lost = threading.Event()
    pairs = [Pipe() for _ in range(runtime.max_batch_size)]
    parents = {slot: pair[0] for slot, pair in enumerate(pairs)}
    clients = [RemoteChunkPolicy(policy.spec, policy.context_frames, (), pair[1], lost) for pair in pairs]
    batcher = ChunkBatcher(policy, horizon, parents, batch_wait_seconds=batch_wait)
    batcher.forbid_compilation = False
    errors: list[BaseException] = []

    def serve() -> None:
        try:
            batcher.serve(stop)
        except BaseException as error:
            errors.append(error)
            lost.set()

    engine = threading.Thread(target=serve, daemon=True)
    engine.start()
    samples = []
    try:
        for delay in runtime.transport_delays:
            policy.reset_chunks()
            batcher.forbid_compilation = False
            for sequence in range(WARMUP_CALLS + MEASURED_CALLS):
                if sequence == WARMUP_CALLS:
                    batcher.forbid_compilation = True
                source = policy.context_frames + sequence * max(1, prefix - max(runtime.transport_delays))
                for slot, client in enumerate(clients):
                    client.submit(
                        ChunkRequest(
                            slot,
                            0,
                            sequence,
                            source,
                            policy.warmup_context(slot, source, delay),
                            (NEUTRAL_CONTROLLER_ACTION,) * (prefix - max(runtime.transport_delays) + delay),
                        )
                    )
                deadline = time.monotonic() + 120
                while any(client.busy for client in clients):
                    for client in clients:
                        client.poll()
                    if time.monotonic() > deadline:
                        raise TimeoutError("calibration request did not finish within 120 seconds")
                    time.sleep(0.0001)
                if sequence >= WARMUP_CALLS:
                    samples.append(max(client.last_latency for client in clients))
    finally:
        stop.set()
        engine.join(timeout=2)
        for pair in pairs:
            for connection in pair:
                connection.close()
        policy.reset_chunks()
    if errors:
        raise RuntimeError("calibration inference failed") from errors[0]
    return Measurement(horizon, prefix, tuple(samples))


def calibrate(policy: ChunkPolicy, runtime: RuntimeConfig, batch_wait: float) -> Calibration:
    transport = max(runtime.transport_delays)
    measurements = []
    for horizon in policy.supported_horizons:
        if horizon < transport + 2:
            continue
        # Forced work and sampled work have different costs. Measure every
        # feasible prefix shape rather than assuming latency is monotone in B.
        for budget in range(1, (horizon - transport) // 2 + 1):
            measurements.append(measure_shape(policy, runtime, horizon, transport + budget, batch_wait))
    selected = select_schedule(tuple(measurements), transport)
    # Prepare the selected shape again, then qualify it after all candidate compilations.
    result = measure_shape(policy, runtime, selected.horizon, selected.prefix_frames, batch_wait)
    measurements.append(result)
    if latency_frames(result.p99_seconds) > selected.compute_frames:
        raise RuntimeError("server unavailable: selected schedule failed final qualification")
    policy.reset_chunks()
    return Calibration(
        tuple(TimingSchedule(selected.compute_frames, delay, selected.horizon) for delay in runtime.transport_delays),
        tuple(measurements),
    )


@contextmanager
def local_chunk_service(policy: ChunkPolicy, runtime: RuntimeConfig) -> Iterator[RemoteChunkPolicy]:
    """Use the same calibrated delivery path for a single interactive match."""
    if runtime.max_batch_size != 1:
        raise ValueError("local chunk service requires one stream")
    result = calibrate(policy, runtime, 0.0005)
    parent, child = Pipe()
    stop = threading.Event()
    lost = threading.Event()
    batcher = ChunkBatcher(policy, result.schedules[0].horizon, {0: parent}, batch_wait_seconds=0.0005)

    def serve() -> None:
        try:
            batcher.serve(stop)
        except BaseException:
            lost.set()

    engine = threading.Thread(target=serve, daemon=True)
    engine.start()
    try:
        yield RemoteChunkPolicy(policy.spec, policy.context_frames, result.schedules, child, lost)
    finally:
        stop.set()
        engine.join(timeout=2)
        parent.close()
        child.close()
