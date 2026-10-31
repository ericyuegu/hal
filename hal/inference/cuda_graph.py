"""Explicit CUDA graph ownership for inference with persistent mutable state."""

import time
from collections.abc import Callable
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from torch._dynamo import callback_handler
from torch._dynamo import config as dynamo_config


def _require_tested_compiler_version(version: str) -> None:
    if version.split("+", 1)[0] != "2.11.0":
        raise RuntimeError(f"compiler instrumentation requires tested Torch 2.11.0, got {version}")


@contextmanager
def prepared_shape_compilation(shape_count: int) -> Iterator[None]:
    """Allow Torch 2.11 to specialize declared batch/update shapes before admission."""
    _require_tested_compiler_version(str(torch.__version__))
    if shape_count < 1:
        raise ValueError("at least one compiler shape must be prepared")
    # Torch 2.11.0+cu130 defaults to eight variants per code object, which
    # raises FailOnRecompileLimitHit for prepared 1/2/4 updates and buckets.
    # Remove this private-API patch once another Torch version qualifies its
    # per-call specialization budget or dynamic-shape capture for these profiles.
    # The budget stays scoped to preparation and below Torch's accumulated cap.
    limit = min(dynamo_config.accumulated_recompile_limit, max(dynamo_config.recompile_limit, shape_count * 4))
    with dynamo_config.patch(recompile_limit=limit):
        yield


@dataclass(frozen=True, slots=True)
class CaptureCounts:
    attempts: int
    completed: int


class CaptureCounter:
    """Count explicit CUDA graph captures owned by one policy."""

    def __init__(self) -> None:
        self.attempts = 0
        self.completed = 0

    def snapshot(self) -> CaptureCounts:
        return CaptureCounts(self.attempts, self.completed)


class CompilationStartCounter:
    """Count Torch compiler starts after policy preparation."""

    def __init__(self) -> None:
        self.starts = 0

    def __call__(self, _event: object) -> None:
        self.starts += 1

    def snapshot(self) -> int:
        if sum(callback is self for callback in callback_handler.start_callbacks) != 1:
            raise RuntimeError("Torch removed the post-preparation compilation listener")
        return self.starts


@contextmanager
def count_compilation_starts() -> Iterator[CompilationStartCounter]:
    """Observe first compiles as well as recompiles in the serving scope."""
    _require_tested_compiler_version(str(torch.__version__))
    # Torch 2.11's private callback sees first compiles that fail_on_recompile allows.
    # Replace it when Torch exposes a public compile-start listener.
    counter = CompilationStartCounter()
    callback_handler.register_start_callback(counter)
    try:
        yield counter
    finally:
        try:
            callback_handler.remove_start_callback(counter)
        except ValueError as error:
            raise RuntimeError("Torch removed the post-preparation compilation listener") from error


class CapturePool:
    """Prepare operations first, then capture shared scratch for one replay stream.

    Call capture_all after preparing every operation. Before that, calls execute
    eagerly on the preparation stream. Compilation and persistent constants stay
    outside the graph pool, even when they are first created by a later operation.
    Consume or copy a graph's outputs before replaying another graph in this
    pool. Persistent inputs and mutable state must live outside the pool.
    """

    def __init__(self, device: torch.device) -> None:
        if device.type != "cuda":
            raise ValueError("CUDA graph pools need a CUDA device")
        self.device = torch.device("cuda", torch.cuda.current_device()) if device.index is None else device
        self.phase: Literal["preparing", "capturing", "ready", "failed"] = "preparing"
        self.pending: list[Callable[[], None]] = []
        with torch.cuda.device(self.device):
            self.handle = torch.cuda.graph_pool_handle()
            self.capture_stream = torch.cuda.Stream(device=self.device)
            self.replay_stream = torch.cuda.current_stream(self.device)

    def check_stream(self) -> None:
        if self.phase == "failed":
            raise RuntimeError("shared CUDA graph pool capture failed")
        if torch.cuda.current_stream(self.device) != self.replay_stream:
            raise ValueError("shared CUDA graphs must replay on their pool's stream")

    def capture_all(self) -> None:
        self.check_stream()
        if self.phase != "preparing":
            raise ValueError("shared CUDA graph pool is already captured")
        with torch.cuda.device(self.device):
            torch.cuda.synchronize(self.device)
            # Warmup uses one stream and the general allocator. Its unused cache
            # must not remain reserved alongside the graph pool.
            torch.cuda.empty_cache()
            self.phase = "capturing"
            completed = False
            try:
                for capture in self.pending:
                    capture()
                completed = True
            finally:
                self.phase = "ready" if completed else "failed"
                # Discard pending operation closures after a partial capture too.
                for capture in self.pending:
                    capture()
                self.pending.clear()


class CapturedCall[T]:
    """Replay an operation on fixed input storage without copying its persistent state."""

    def __init__(
        self,
        operation: Callable[[], T],
        inputs: tuple[Tensor, ...],
        state: tuple[Tensor, ...] = (),
        *,
        device: torch.device,
        counter: CaptureCounter | None = None,
        pool: CapturePool | None = None,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("CUDA graph capture needs a CUDA device")
        self.device = torch.device("cuda", torch.cuda.current_device()) if device.index is None else device
        if pool is not None:
            if pool.device != self.device:
                raise ValueError("CUDA graph pool must use the declared device")
            pool.check_stream()
            if pool.phase != "preparing":
                raise ValueError("shared CUDA graph pool is closed to new operations")
        self.pool = pool
        self.grad_enabled = torch.is_grad_enabled()
        self.autocast_enabled = torch.is_autocast_enabled(self.device.type)
        self.autocast_dtype = torch.get_autocast_dtype(self.device.type)
        self.autocast_cache_enabled = torch.is_autocast_cache_enabled()
        if any(value.device != self.device for value in (*inputs, *state)):
            raise ValueError("CUDA graph inputs and persistent state must use the declared device")
        self.inputs = inputs
        self.state = state
        self.counter = counter
        self.operation = operation if pool is not None else None
        self.capture_seconds = 0.0
        preparation_started = time.perf_counter()
        with torch.cuda.device(self.device):
            self.graph = torch.cuda.CUDAGraph()
            current = torch.cuda.current_stream(self.device)
            self.capture_stream = torch.cuda.Stream(device=self.device) if pool is None else pool.capture_stream
            capture = self.capture_stream
            saved = tuple(value.clone() for value in state)
            capture.wait_stream(current)
            try:
                with torch.cuda.stream(capture):
                    for _ in range(2):
                        if pool is None:
                            operation()
                        else:
                            self.output = operation()
                    warmup_done = torch.cuda.Event()
                    warmup_done.record(capture)
                    warmup_done.synchronize()
                    self.preparation_seconds = time.perf_counter() - preparation_started
                    if pool is None:
                        self._capture(operation)
                    else:
                        pool.pending.append(self._capture_prepared)
            finally:
                current.wait_stream(capture)
                for target, original in zip(state, saved, strict=True):
                    target.copy_(original)

    def _capture(self, operation: Callable[[], T]) -> None:
        started = time.perf_counter()
        if self.counter is not None:
            self.counter.attempts += 1
        with torch.cuda.graph(
            self.graph, pool=None if self.pool is None else self.pool.handle, stream=self.capture_stream
        ):
            self.output = operation()
        self.capture_seconds = time.perf_counter() - started
        if self.counter is not None:
            self.counter.completed += 1

    def _capture_prepared(self) -> None:
        operation, self.operation = self.operation, None
        if operation is None or self.pool is None or self.pool.phase != "capturing":
            return
        # A compiled operation guards the ambient autocast state, even when
        # its body owns another autocast context.
        with (
            torch.cuda.device(self.device),
            torch.set_grad_enabled(self.grad_enabled),
            torch.autocast(
                self.device.type,
                dtype=self.autocast_dtype,
                enabled=self.autocast_enabled,
                cache_enabled=self.autocast_cache_enabled,
            ),
        ):
            current = torch.cuda.current_stream(self.device)
            saved = tuple(value.clone() for value in self.state)
            self.capture_stream.wait_stream(current)
            try:
                with torch.cuda.stream(self.capture_stream):
                    self._capture(operation)
            finally:
                current.wait_stream(self.capture_stream)
                for target, original in zip(self.state, saved, strict=True):
                    target.copy_(original)

    def __call__(self, inputs: tuple[Tensor, ...]) -> T:
        with torch.cuda.device(self.device):
            if self.pool is not None:
                self.pool.check_stream()
            for target, source in zip(self.inputs, inputs, strict=True):
                target.copy_(source)
            if self.operation is None:
                self.graph.replay()
            else:
                current = torch.cuda.current_stream(self.device)
                self.capture_stream.wait_stream(current)
                with (
                    torch.cuda.stream(self.capture_stream),
                    torch.set_grad_enabled(self.grad_enabled),
                    torch.autocast(
                        self.device.type,
                        dtype=self.autocast_dtype,
                        enabled=self.autocast_enabled,
                        cache_enabled=self.autocast_cache_enabled,
                    ),
                ):
                    self.output = self.operation()
                current.wait_stream(self.capture_stream)
        return self.output
