"""Explicit CUDA graph ownership for inference with persistent mutable state."""

from collections.abc import Callable
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

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


class CapturedCall:
    """Replay an operation on fixed input storage without copying its persistent state."""

    def __init__(
        self,
        operation: Callable[[], Tensor],
        inputs: tuple[Tensor, ...],
        state: tuple[Tensor, ...] = (),
        *,
        device: torch.device,
        counter: CaptureCounter | None = None,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("CUDA graph capture needs a CUDA device")
        self.device = torch.device("cuda", torch.cuda.current_device()) if device.index is None else device
        if any(value.device != self.device for value in (*inputs, *state)):
            raise ValueError("CUDA graph inputs and persistent state must use the declared device")
        self.inputs = inputs
        with torch.cuda.device(self.device):
            self.graph = torch.cuda.CUDAGraph()
            current = torch.cuda.current_stream(self.device)
            capture = torch.cuda.Stream(device=self.device)
            saved = tuple(value.clone() for value in state)
            capture.wait_stream(current)
            try:
                with torch.cuda.stream(capture):
                    for _ in range(2):
                        operation()
                    if counter is not None:
                        counter.attempts += 1
                    with torch.cuda.graph(self.graph, stream=capture):
                        self.output = operation()
            finally:
                current.wait_stream(capture)
                for target, original in zip(state, saved, strict=True):
                    target.copy_(original)
        if counter is not None:
            counter.completed += 1

    def __call__(self, inputs: tuple[Tensor, ...]) -> Tensor:
        with torch.cuda.device(self.device):
            for target, source in zip(self.inputs, inputs, strict=True):
                target.copy_(source)
            self.graph.replay()
        return self.output
