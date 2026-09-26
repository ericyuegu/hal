"""Deliver prediction requests without blocking a live Dolphin session."""

import math
import threading
import time
from collections import deque
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from contextlib import contextmanager
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from multiprocessing.connection import wait
from typing import Protocol
from typing import cast

import torch

from hal.inference.api import ActionPlan
from hal.inference.api import PolicySpec
from hal.inference.api import PredictionPolicy
from hal.inference.api import PredictionRequest
from hal.inference.api import validate_action_plan


class StopSignal(Protocol):
    def is_set(self) -> bool: ...


class InferenceUnavailable(RuntimeError):
    """Confirmed inference failure; the caller must drain its current plan."""


@dataclass(frozen=True, slots=True)
class WorkerFailure:
    reason: str


class InferenceClient:
    """Send one outstanding prediction through a pipe on a delivery thread."""

    def __init__(
        self,
        spec: PolicySpec,
        context_frames: int,
        connection: Connection,
        worker_lost: StopSignal,
    ) -> None:
        self.spec = spec
        self.context_frames = context_frames
        self.connection = connection
        self.worker_lost = worker_lost
        self.generation = 0
        self._pending: PredictionRequest | None = None
        self._response: ActionPlan | WorkerFailure | None = None
        self._lock = threading.Lock()
        self._started = 0.0
        self.last_latency = 0.0

    @property
    def busy(self) -> bool:
        return self._pending is not None

    def start_match(self) -> int:
        self.generation += 1
        return self.generation

    def submit(self, request: PredictionRequest) -> None:
        if self.busy:
            raise RuntimeError("a stream already has an outstanding prediction request")
        if self.worker_lost.is_set():
            raise InferenceUnavailable("inference worker stopped")
        self._pending = request
        self._started = time.perf_counter()
        try:
            threading.Thread(
                target=self._exchange, args=(request,), daemon=True, name="hal-prediction-delivery"
            ).start()
        except RuntimeError as error:
            self._pending = None
            raise InferenceUnavailable("inference delivery thread could not start") from error

    def _exchange(self, request: PredictionRequest) -> None:
        try:
            self.connection.send(request)
            response = self.connection.recv()
            if not isinstance(response, (ActionPlan, WorkerFailure)):
                response = WorkerFailure("invalid inference response")
        except (OSError, EOFError) as error:
            response = WorkerFailure(f"inference connection lost: {type(error).__name__}")
        with self._lock:
            self.last_latency = time.perf_counter() - self._started
            self._response = response

    def poll(self) -> ActionPlan | None:
        if self.worker_lost.is_set():
            raise InferenceUnavailable("inference worker stopped")
        with self._lock:
            response = self._response
            self._response = None
        if response is None:
            return None
        self._pending = None
        if isinstance(response, WorkerFailure):
            raise InferenceUnavailable(response.reason)
        return response


class InferenceWorker:
    """Collect ready streams, run one policy batch, and return their plans."""

    def __init__(
        self,
        policy: PredictionPolicy,
        horizon: int,
        connections: Mapping[int, Connection],
        *,
        batch_wait_seconds: float,
    ) -> None:
        if not math.isfinite(batch_wait_seconds) or batch_wait_seconds < 0 or not connections:
            raise ValueError("invalid inference worker configuration")
        self.policy = policy
        self.horizon = horizon
        self.connections = dict(connections)
        self.batch_wait_seconds = batch_wait_seconds
        self.batch_calls = 0
        self.batch_items = 0
        self.max_batch_items = 0
        self.forbid_compilation = True
        self._seconds: deque[float] = deque(maxlen=1200)
        self._waits: deque[float] = deque(maxlen=1200)
        self._lock = threading.Lock()

    def timing_p95_ms(self) -> tuple[float | None, float | None]:
        with self._lock:
            samples = (tuple(self._seconds), tuple(self._waits))
        values = tuple(
            None if not sample else 1000 * sorted(sample)[math.ceil(0.95 * len(sample)) - 1] for sample in samples
        )
        return values[0], values[1]

    def serve(self, stop: StopSignal) -> None:
        while not stop.is_set():
            ready = cast(list[Connection], wait(self.connections.values(), timeout=0.05))
            if not ready:
                continue
            started = time.perf_counter()
            deadline = started + self.batch_wait_seconds
            while len(ready) < len(self.connections):
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                more = cast(
                    list[Connection],
                    wait(
                        [connection for connection in self.connections.values() if connection not in ready],
                        timeout=remaining,
                    ),
                )
                if not more:
                    break
                ready.extend(more)
            self.serve_batch(ready, time.perf_counter() - started)

    def serve_batch(self, connections: Sequence[Connection], batch_wait: float) -> None:
        try:
            requests = []
            for connection in connections:
                request = connection.recv()
                if not isinstance(request, PredictionRequest):
                    raise ValueError("invalid prediction request")
                requests.append(request)
            self.batch_calls += 1
            self.batch_items += len(requests)
            self.max_batch_items = max(self.max_batch_items, len(requests))
            started = time.perf_counter()
            with torch.compiler.set_stance("fail_on_recompile" if self.forbid_compilation else "default"):
                plans = tuple(self.policy.predict(requests))
            if len(plans) != len(requests):
                raise ValueError("inference returned the wrong plan batch size")
            for request, plan in zip(requests, plans, strict=True):
                validate_action_plan(request, plan, self.horizon)
            for plan, connection in zip(plans, connections, strict=True):
                connection.send(plan)
            with self._lock:
                self._seconds.append(time.perf_counter() - started)
                self._waits.append(batch_wait)
        except BaseException:
            # A failed worker must notify every client, including idle streams.
            for connection in self.connections.values():
                with suppress(OSError, EOFError):
                    connection.send(WorkerFailure("inference worker failed"))
            raise


@contextmanager
def start_inference_worker(
    policy: PredictionPolicy,
    horizon: int,
    batch_wait_seconds: float,
) -> Iterator[InferenceClient]:
    """Serve one prepared policy stream until the caller leaves the scope."""
    parent, child = Pipe()
    try:
        stop = threading.Event()
        lost = threading.Event()
        worker = InferenceWorker(policy, horizon, {0: parent}, batch_wait_seconds=batch_wait_seconds)

        def serve() -> None:
            try:
                worker.serve(stop)
            except BaseException:
                lost.set()

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        try:
            yield InferenceClient(policy.spec, policy.context_frames, child, lost)
        finally:
            stop.set()
            thread.join(timeout=2)
    finally:
        parent.close()
        child.close()
