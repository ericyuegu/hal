"""Measure complete prediction delivery for a declared request shape."""

import math
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass
from multiprocessing import Pipe

import torch

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.inference.api import PredictionPolicy
from hal.inference.api import PredictionRequest
from hal.inference.api import RuntimeConfig
from hal.inference.warmup import make_warmup_observations
from hal.inference.worker import InferenceClient
from hal.inference.worker import InferenceWorker

WARMUP_CALLS = 20
MEASURED_CALLS = 200


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


def measure_prediction_shape(
    policy: PredictionPolicy,
    runtime: RuntimeConfig,
    horizon: int,
    prefix_frames: int,
    batch_wait_seconds: float,
) -> LatencyMeasurement:
    """Time IPC, batching, validation, and the model at full configured load."""
    if prefix_frames < max(runtime.transport_delays) or prefix_frames >= horizon:
        raise ValueError("prediction shape must cover every input delay and leave a predicted tail")
    if not math.isfinite(batch_wait_seconds) or batch_wait_seconds < 0:
        raise ValueError("batch wait must be finite and non-negative")
    request_interval = runtime.replan_interval_frames or max(1, prefix_frames - max(runtime.transport_delays))
    if request_interval > policy.context_frames:
        raise ValueError("replan interval exceeds the observation history capacity")
    torch.compiler.reset()
    policy.prepare_prediction(runtime, horizon, prefix_frames)
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
        clients = [InferenceClient(policy.spec, policy.context_frames, pair[1], lost) for pair in pairs]
        worker = InferenceWorker(policy, horizon, parents, batch_wait_seconds=batch_wait_seconds)
        worker.forbid_compilation = False

        def serve() -> None:
            try:
                worker.serve(stop)
            except BaseException as error:
                errors.append(error)
                lost.set()

        thread = threading.Thread(target=serve, daemon=True)
        try:
            thread.start()
            max_delay = max(runtime.transport_delays)
            for delay in runtime.transport_delays:
                policy.reset_prediction()
                worker.forbid_compilation = False
                fixed_length = prefix_frames - max_delay + delay
                for sequence in range(WARMUP_CALLS + MEASURED_CALLS):
                    if sequence == WARMUP_CALLS:
                        worker.forbid_compilation = True
                    source = policy.context_frames + sequence * request_interval
                    observation_count = policy.context_frames if sequence == 0 else request_interval
                    for slot, client in enumerate(clients):
                        client.submit(
                            PredictionRequest(
                                slot,
                                0,
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
                                (NEUTRAL_CONTROLLER_ACTION,) * fixed_length,
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
            stop.set()
            if thread.is_alive():
                thread.join(timeout=2)
            if thread.is_alive():
                raise RuntimeError("prediction benchmark worker did not stop; policy cannot be reused")
            policy.reset_prediction()
    if errors:
        raise RuntimeError("prediction benchmark inference failed") from errors[0]
    return LatencyMeasurement(horizon, prefix_frames, tuple(samples))
