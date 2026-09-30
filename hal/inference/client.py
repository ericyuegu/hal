"""Persistent local-process delivery for one admitted inference stream."""

import math
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Literal
from typing import NoReturn
from typing import Protocol

from hal.inference.api import ActionPlan
from hal.inference.api import PolicySpec
from hal.inference.api import PredictionRequest
from hal.inference.api import PreparedInferenceProfile

REQUEST_TIMEOUT_SECONDS = 1.0


class StopSignal(Protocol):
    def is_set(self) -> bool: ...


class InferenceUnavailable(RuntimeError):
    """Confirmed inference failure; the caller must drain its current plan."""


@dataclass(frozen=True, slots=True)
class WorkerFailure:
    reason: str


@dataclass(frozen=True, slots=True)
class StreamAdmission:
    stream_id: int
    generation: int
    profile: PreparedInferenceProfile


@dataclass(frozen=True, slots=True)
class StreamRelease:
    stream_id: int
    generation: int


@dataclass(frozen=True, slots=True)
class StreamInvalidate:
    """Clear a dead client's stream before a replacement slot is admitted."""

    stream_id: int
    token: str


@dataclass(frozen=True, slots=True)
class StreamInvalidated:
    stream_id: int
    token: str


@dataclass(frozen=True, slots=True)
class StreamAck:
    stream_id: int
    generation: int
    operation: Literal["admitted", "released"]


_Control = StreamAdmission | StreamRelease
_Pending = PredictionRequest | _Control
_Response = ActionPlan | StreamAck | WorkerFailure


class InferenceClient:
    """Keep one delivery thread and at most one outstanding wire message."""

    def __init__(
        self,
        spec: PolicySpec,
        context_frames: int,
        connection: Connection,
        worker_lost: StopSignal,
        profiles: Mapping[int, PreparedInferenceProfile],
        *,
        timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        if (
            type(context_frames) is not int
            or context_frames < 1
            or type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("inference client needs positive context and timeout")
        if not profiles or any(prefix != profile.fixed_prefix_frames for prefix, profile in profiles.items()):
            raise ValueError("inference client needs distinct prepared fixed-prefix profiles")
        self.spec = spec
        self.context_frames = context_frames
        self.connection = connection
        self.worker_lost = worker_lost
        self.profiles = dict(profiles)
        self.timeout_seconds = timeout_seconds
        self.generation = 0
        self.last_latency = 0.0
        self._pending: _Pending | None = None
        self._response: _Response | None = None
        self._started = 0.0
        self._received_at = 0.0
        self._failed: str | None = None
        self._stream_id: int | None = None
        self._active_profile: PreparedInferenceProfile | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._pending is not None

    def start_match(self, stream_id: int, prefix_frames: int) -> int:
        if type(stream_id) is not int or type(prefix_frames) is not int:
            raise ValueError("stream ID and fixed prefix must be integers")
        try:
            profile = self.profiles[prefix_frames]
        except KeyError as error:
            raise ValueError("match requests an unprepared fixed prefix") from error
        with self._lock:
            self._require_available()
            if self._pending is not None:
                raise RuntimeError("cannot start a match with an outstanding prediction request")
            if self._stream_id is not None:
                raise RuntimeError("close the current match before starting another")
            generation = self.generation + 1
        self._round_trip_control(StreamAdmission(stream_id, generation, profile), "admitted")
        with self._lock:
            self.generation = generation
            self._stream_id = stream_id
            self._active_profile = profile
        return generation

    def close_match(self) -> None:
        with self._lock:
            stream_id = self._stream_id
            generation = self.generation
        if stream_id is None:
            return
        deadline = time.monotonic() + self.timeout_seconds
        while self.busy:
            self.poll()
            if time.monotonic() >= deadline:
                with self._lock:
                    self._fail("inference request timed out before stream release")
            time.sleep(0.001)
        self._round_trip_control(StreamRelease(stream_id, generation), "released")
        with self._lock:
            self._stream_id = None
            self._active_profile = None

    def submit(self, request: PredictionRequest) -> None:
        with self._lock:
            self._require_available()
            if self._pending is not None:
                raise RuntimeError("a stream already has an outstanding prediction request")
            if request.generation != self.generation or request.stream_id != self._stream_id:
                raise ValueError("prediction stream or generation differs from this client's match")
            if self._active_profile is None or len(request.fixed_actions) != self._active_profile.fixed_prefix_frames:
                raise ValueError("prediction fixed prefix differs from admitted profile")
            self._queue(request)

    def poll(self) -> ActionPlan | None:
        with self._lock:
            self._require_available()
            if self._pending is not None and not isinstance(self._pending, PredictionRequest):
                raise RuntimeError("cannot poll an admission or release as a prediction")
            response = self._response
            if response is None:
                if self._pending is not None and time.monotonic() - self._started >= self.timeout_seconds:
                    self._fail("inference request timed out")
                return None
            request = self._pending
            self._pending = None
            self._response = None
            self.last_latency = self._received_at - self._started
            if self.last_latency > self.timeout_seconds:
                self._fail("inference request timed out")
            if isinstance(response, WorkerFailure):
                self._fail(response.reason)
            if (
                not isinstance(request, PredictionRequest)
                or not isinstance(response, ActionPlan)
                or (
                    response.stream_id,
                    response.generation,
                    response.sequence,
                    response.source_frame,
                )
                != (request.stream_id, request.generation, request.sequence, request.source_frame)
                or type(response.state_value) is not float
                or not math.isfinite(response.state_value)
                or any(
                    type(value) is not int
                    for value in (response.stream_id, response.generation, response.sequence, response.source_frame)
                )
            ):
                self._fail("inference response identity differs from its request")
            return response

    def close(self) -> None:
        self._closed.set()
        self._wake.set()
        self.connection.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _round_trip_control(self, request: _Control, operation: Literal["admitted", "released"]) -> None:
        with self._lock:
            self._require_available()
            if self._pending is not None:
                raise RuntimeError("cannot change stream admission with an outstanding request")
            self._queue(request)
        while True:
            with self._lock:
                self._require_available()
                response = self._response
                if response is not None:
                    self._pending = None
                    self._response = None
                    if self._received_at - self._started > self.timeout_seconds:
                        self._fail("inference stream acknowledgement timed out")
                    if isinstance(response, WorkerFailure):
                        self._fail(response.reason)
                    if (
                        not isinstance(response, StreamAck)
                        or any(type(value) is not int for value in (response.stream_id, response.generation))
                        or (
                            response.stream_id,
                            response.generation,
                            response.operation,
                        )
                        != (request.stream_id, request.generation, operation)
                    ):
                        self._fail("inference stream acknowledgement differs from its request")
                    return
                if time.monotonic() - self._started >= self.timeout_seconds:
                    self._fail("inference stream acknowledgement timed out")
            time.sleep(0.001)

    def _queue(self, request: _Pending) -> None:
        self._pending = request
        self._response = None
        self._started = time.monotonic()
        self._received_at = 0.0
        if self._thread is None:
            thread = threading.Thread(target=self._deliver, daemon=True, name="hal-prediction-delivery")
            try:
                thread.start()
            except RuntimeError as error:
                self._pending = None
                self._failed = "inference delivery thread could not start"
                raise InferenceUnavailable(self._failed) from error
            self._thread = thread
        self._wake.set()

    def _require_available(self) -> None:
        if self.worker_lost.is_set():
            self._fail("inference worker stopped")
        if self._closed.is_set():
            self._fail("inference client closed")
        if self._failed is not None:
            raise InferenceUnavailable(self._failed)

    def _fail(self, reason: str) -> NoReturn:
        self._failed = reason
        self._pending = None
        self._response = None
        raise InferenceUnavailable(reason)

    def _deliver(self) -> None:
        while not self._closed.is_set():
            self._wake.wait(0.1)
            if self._closed.is_set():
                return
            if not self._wake.is_set():
                continue
            self._wake.clear()
            with self._lock:
                request = self._pending
            if request is None:
                continue
            try:
                self.connection.send(request)
                while not self._closed.is_set():
                    if self.connection.poll(0.05):
                        response = self.connection.recv()
                        break
                    with self._lock:
                        if self._failed is not None or self._pending is not request:
                            return
                        if time.monotonic() - self._started >= self.timeout_seconds:
                            response = WorkerFailure("inference request timed out")
                            break
                else:
                    return
                if not isinstance(response, (ActionPlan, StreamAck, WorkerFailure)):
                    response = WorkerFailure("invalid inference response")
            except (OSError, EOFError, ValueError) as error:
                response = WorkerFailure(f"inference connection lost: {type(error).__name__}")
            received_at = time.monotonic()
            with self._lock:
                if self._pending is request and self._failed is None:
                    self._received_at = received_at
                    self._response = (
                        WorkerFailure("inference request timed out")
                        if received_at - self._started > self.timeout_seconds
                        else response
                    )
