"""Nonblocking chunk clients and a continuously batched local inference service."""

import math
import threading
import time
from collections import deque
from collections.abc import Mapping
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.connection import wait
from typing import Protocol
from typing import cast

import torch

from hal.inference.api import PolicySpec
from hal.inference.chunks import ChunkPolicy
from hal.inference.chunks import ChunkRequest
from hal.inference.chunks import ChunkResponse
from hal.inference.chunks import TimingSchedule
from hal.inference.chunks import validate_chunk_response


class StopSignal(Protocol):
    def is_set(self) -> bool: ...


class EngineLost(RuntimeError):
    """Confirmed inference failure; the worker must drain its current plan."""


@dataclass(frozen=True, slots=True)
class EngineFailure:
    reason: str


class RemoteChunkPolicy:
    """One outstanding request; pipe delivery never blocks the Dolphin worker."""

    def __init__(
        self,
        spec: PolicySpec,
        context_frames: int,
        schedules: tuple[TimingSchedule, ...],
        connection: Connection,
        engine_lost: StopSignal,
    ) -> None:
        self.spec = spec
        self.context_frames = context_frames
        self.schedules = schedules
        self.connection = connection
        self.engine_lost = engine_lost
        self.generation = 0
        self._pending: ChunkRequest | None = None
        self._response: ChunkResponse | EngineFailure | None = None
        self._lock = threading.Lock()
        self._started = 0.0
        self.last_latency = 0.0

    @property
    def busy(self) -> bool:
        return self._pending is not None

    def start_match(self) -> int:
        self.generation += 1
        return self.generation

    def submit(self, request: ChunkRequest) -> None:
        if self.busy:
            raise RuntimeError("a stream already has an outstanding chunk request")
        if self.engine_lost.is_set():
            raise EngineLost("inference engine stopped")
        self._pending = request
        self._started = time.perf_counter()
        threading.Thread(target=self._exchange, args=(request,), daemon=True, name="hal-chunk-delivery").start()

    def _exchange(self, request: ChunkRequest) -> None:
        try:
            self.connection.send(request)
            response = self.connection.recv()
            if not isinstance(response, (ChunkResponse, EngineFailure)):
                response = EngineFailure("invalid inference response")
        except (OSError, EOFError) as error:
            response = EngineFailure(f"inference connection lost: {type(error).__name__}")
        with self._lock:
            self.last_latency = time.perf_counter() - self._started
            self._response = response

    def poll(self) -> ChunkResponse | None:
        if self.engine_lost.is_set():
            raise EngineLost("inference engine stopped")
        with self._lock:
            response = self._response
            self._response = None
        if response is None:
            return None
        self._pending = None
        if isinstance(response, EngineFailure):
            raise EngineLost(response.reason)
        return response


class ChunkBatcher:
    def __init__(
        self,
        policy: ChunkPolicy,
        horizon: int,
        connections: Mapping[int, Connection],
        *,
        batch_wait_seconds: float,
    ) -> None:
        if not math.isfinite(batch_wait_seconds) or batch_wait_seconds < 0 or not connections:
            raise ValueError("invalid chunk batcher configuration")
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
                    list[Connection], wait([c for c in self.connections.values() if c not in ready], timeout=remaining)
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
                if not isinstance(request, ChunkRequest):
                    raise ValueError("invalid inference chunk request")
                requests.append(request)
            self.batch_calls += 1
            self.batch_items += len(requests)
            self.max_batch_items = max(self.max_batch_items, len(requests))
            started = time.perf_counter()
            with torch.compiler.set_stance("fail_on_recompile" if self.forbid_compilation else "default"):
                responses = tuple(self.policy.plan_chunks(requests))
            if len(responses) != len(requests):
                raise ValueError("inference returned the wrong chunk batch size")
            for request, response, connection in zip(requests, responses, connections, strict=True):
                validate_chunk_response(request, response, self.horizon)
                connection.send(response)
            with self._lock:
                self._seconds.append(time.perf_counter() - started)
                self._waits.append(batch_wait)
        except BaseException:
            # A failed engine must notify every worker, including idle streams.
            for connection in self.connections.values():
                with suppress(OSError, EOFError):
                    connection.send(EngineFailure("inference engine failed"))
            raise
