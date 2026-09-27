"""Coalesce ready local-process requests for a prepared prediction policy."""

from __future__ import annotations

import gc
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from contextlib import contextmanager
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from multiprocessing.connection import wait
from typing import cast

import torch

from hal.inference.action_sequence_artifact import ActionSequenceArtifact
from hal.inference.action_sequence_artifact import build_action_sequence_model
from hal.inference.api import PredictionPolicy
from hal.inference.api import PredictionRequest
from hal.inference.api import PreparedInferenceProfile
from hal.inference.api import validate_action_plan
from hal.inference.client import InferenceClient
from hal.inference.client import StopSignal
from hal.inference.client import StreamAck
from hal.inference.client import StreamAdmission
from hal.inference.client import StreamRelease
from hal.inference.client import WorkerFailure
from hal.models.action_sequence import ActionSequenceTransformer


def configure_inference_process() -> None:
    torch.set_num_threads(1)


def freeze_inference_runtime() -> None:
    """Exclude prepared long-lived objects from collections during frame delivery."""
    gc.collect()
    gc.freeze()


def _deadline_order(item: tuple[Connection, PredictionRequest]) -> float:
    return float("inf") if item[1].deadline_monotonic is None else item[1].deadline_monotonic


def _serve_worker_thread(
    worker: InferenceEngine,
    stop: StopSignal,
    lost: threading.Event,
    errors: list[Exception],
) -> None:
    try:
        worker.serve(stop)
    except Exception as error:
        errors.append(error)
        lost.set()


class ModelRegistry:
    """Construct checkpoint weights only after a device/dtype registry miss."""

    def __init__(self) -> None:
        self._models: dict[tuple[str, str, torch.dtype], ActionSequenceTransformer] = {}
        self._lock = threading.Lock()

    @property
    def model_count(self) -> int:
        with self._lock:
            return len(self._models)

    def register_artifact(
        self,
        artifact: ActionSequenceArtifact,
        *,
        device: str | torch.device,
        inference_dtype: torch.dtype,
    ) -> ActionSequenceTransformer:
        resolved = torch.device(device)
        if resolved.type == "cuda" and resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
        elif resolved.type == "cpu":
            resolved = torch.device("cpu")
        expected = torch.bfloat16 if resolved.type == "cuda" else torch.float32
        if resolved.type not in ("cpu", "cuda") or inference_dtype != expected:
            raise ValueError(f"unsupported inference device/dtype {resolved}/{inference_dtype}")
        key = (artifact.checkpoint_sha256, str(resolved), inference_dtype)
        with self._lock:
            model = self._models.get(key)
            if model is None:
                model = build_action_sequence_model(artifact, device=resolved, inference_dtype=inference_dtype)
                self._models[key] = model
            return model


@dataclass(frozen=True, slots=True)
class _StreamRoute:
    connection: Connection
    generation: int
    profile: PreparedInferenceProfile


class InferenceEngine:
    """Collect ready streams, run one policy batch, and return their plans."""

    def __init__(
        self,
        profiles: Mapping[PreparedInferenceProfile, PredictionPolicy],
        connections: Mapping[int, Connection],
        *,
        batch_wait_seconds: float,
    ) -> None:
        if not math.isfinite(batch_wait_seconds) or not 0 <= batch_wait_seconds <= 0.0005 or not connections:
            raise ValueError("invalid inference worker configuration")
        if not profiles or len(set(connections.values())) != len(connections):
            raise ValueError("engine needs prepared profiles and distinct connections")
        for profile, policy in profiles.items():
            policy.validate_prepared_profile(profile)
        self.profiles = dict(profiles)
        self.connections = dict(connections)
        self._routes: dict[int, _StreamRoute] = {}
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

    def serve(self, stop: StopSignal, pulse: Callable[[], None] | None = None) -> None:
        while not stop.is_set():
            ready = cast(list[Connection], wait(self.connections.values(), timeout=0.05))
            if not ready:
                if pulse is not None:
                    pulse()
                continue
            started = time.monotonic()
            try:
                arrived = self._receive(ready)
                until = started + self.batch_wait_seconds
                while arrived:
                    nearest = min(
                        (
                            request.deadline_monotonic
                            for _, request in arrived
                            if request.deadline_monotonic is not None
                        ),
                        default=until,
                    )
                    seen = {connection for connection, _ in arrived}
                    available = [connection for connection in self.connections.values() if connection not in seen]
                    if not available:
                        break
                    more = cast(list[Connection], wait(available, timeout=0))
                    if not more and len(arrived) == 1:
                        remaining = min(until, nearest) - time.monotonic()
                        if remaining > 0:
                            more = cast(list[Connection], wait(available, timeout=remaining))
                    if not more:
                        break
                    arrived.extend(self._receive(more))
                if arrived:
                    self._execute_batch(arrived, time.monotonic() - started)
            except Exception:
                self._notify_failure()
                raise
            if pulse is not None:
                pulse()

    def serve_batch(self, connections: Sequence[Connection], batch_wait: float) -> None:
        try:
            arrived = self._receive(connections)
            if arrived:
                self._execute_batch(arrived, batch_wait)
        except Exception:
            self._notify_failure()
            raise

    def _receive(self, connections: Sequence[Connection]) -> list[tuple[Connection, PredictionRequest]]:
        arrived: list[tuple[Connection, PredictionRequest]] = []
        for connection in connections:
            if not any(known is connection for known in self.connections.values()):
                raise ValueError("inference message came from an unregistered connection")
            message = connection.recv()
            if isinstance(message, StreamAdmission):
                self._admit(connection, message)
            elif isinstance(message, StreamRelease):
                self._release(connection, message)
            elif isinstance(message, PredictionRequest):
                route = self._routes.get(message.stream_id)
                if route is None or route.connection is not connection or route.generation != message.generation:
                    raise ValueError("prediction stream was not admitted on this connection and generation")
                if len(message.fixed_actions) != route.profile.fixed_prefix_frames:
                    raise ValueError("prediction fixed prefix differs from the admitted profile")
                if any(request.stream_id == message.stream_id for _, request in arrived):
                    raise ValueError("a stream has more than one outstanding prediction")
                arrived.append((connection, message))
            else:
                raise ValueError("invalid inference wire message")
        return arrived

    def _admit(self, connection: Connection, request: StreamAdmission) -> None:
        if (
            request.profile not in self.profiles
            or type(request.generation) is not int
            or request.generation < 1
            or type(request.stream_id) is not int
        ):
            raise ValueError("stream requested an unknown profile or invalid identity")
        for stream_id, route in self._routes.items():
            if route.connection is connection and stream_id != request.stream_id:
                raise ValueError("connection already owns a different stream")
        prior = self._routes.get(request.stream_id)
        if prior is not None:
            if prior.connection is not connection or request.generation <= prior.generation:
                raise ValueError("stream admission reuses an active connection or stale generation")
            self.profiles[prior.profile].release_stream(request.stream_id)
            del self._routes[request.stream_id]
        if sum(route.profile == request.profile for route in self._routes.values()) >= request.profile.capacity:
            raise ValueError("prepared inference profile has no available stream rows")
        self._routes[request.stream_id] = _StreamRoute(connection, request.generation, request.profile)
        connection.send(StreamAck(request.stream_id, request.generation, "admitted"))

    def _release(self, connection: Connection, request: StreamRelease) -> None:
        if type(request.stream_id) is not int or type(request.generation) is not int:
            raise ValueError("stream release identity must contain exact integers")
        route = self._routes.get(request.stream_id)
        if route is None or route.connection is not connection or route.generation != request.generation:
            raise ValueError("stream release does not match an active admission")
        self.profiles[route.profile].release_stream(request.stream_id)
        del self._routes[request.stream_id]
        connection.send(StreamAck(request.stream_id, request.generation, "released"))

    def _execute_batch(self, arrived: Sequence[tuple[Connection, PredictionRequest]], batch_wait: float) -> None:
        ordered = sorted(arrived, key=_deadline_order)
        groups: dict[tuple[PreparedInferenceProfile, int], list[tuple[Connection, PredictionRequest]]] = {}
        for connection, request in ordered:
            shape = (self._routes[request.stream_id].profile, len(request.observations))
            groups.setdefault(shape, []).append((connection, request))
        started = time.perf_counter()
        for (profile, _), group in groups.items():
            requests = tuple(request for _, request in group)
            policy = self.profiles[profile]
            self.batch_calls += 1
            self.batch_items += len(requests)
            self.max_batch_items = max(self.max_batch_items, len(requests))
            with torch.compiler.set_stance("fail_on_recompile" if self.forbid_compilation else "default"):
                plans = tuple(policy.predict(requests))
            if len(plans) != len(requests):
                raise ValueError("inference returned the wrong plan batch size")
            deliveries = []
            for (connection, request), plan in zip(group, plans, strict=True):
                validate_action_plan(request, plan, profile.prediction_horizon_frames)
                deliveries.append((connection, plan))
            for connection, plan in deliveries:
                connection.send(plan)
        with self._lock:
            self._seconds.append(time.perf_counter() - started)
            self._waits.append(batch_wait)

    def _notify_failure(self) -> None:
        # A failed engine must notify every client, including idle streams.
        for connection in self.connections.values():
            with suppress(OSError, EOFError):
                connection.send(WorkerFailure("inference worker failed"))


@contextmanager
def start_inference_worker(
    policy: PredictionPolicy,
    profile: PreparedInferenceProfile,
    batch_wait_seconds: float,
) -> Iterator[InferenceClient]:
    """Serve one prepared policy stream until the caller leaves the scope."""
    parent, child = Pipe()
    try:
        stop = threading.Event()
        lost = threading.Event()
        errors: list[Exception] = []
        worker = InferenceEngine({profile: policy}, {0: parent}, batch_wait_seconds=batch_wait_seconds)

        thread = threading.Thread(target=_serve_worker_thread, args=(worker, stop, lost, errors), daemon=True)
        thread.start()
        client: InferenceClient | None = None
        try:
            client = InferenceClient(
                policy.spec, policy.context_frames, child, lost, {profile.fixed_prefix_frames: profile}
            )
            yield client
        finally:
            if client is not None:
                client.close()
            stop.set()
            thread.join(timeout=2)
        if errors:
            raise RuntimeError("local inference worker failed") from errors[0]
    finally:
        parent.close()
        child.close()
