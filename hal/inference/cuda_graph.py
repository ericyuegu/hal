"""Explicit CUDA graph ownership for inference with persistent mutable state."""

from collections.abc import Callable
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import Tensor
from torch._dynamo import config as dynamo_config


def _require_tested_compiler_version(version: str) -> None:
    if version.split("+", 1)[0] != "2.11.0":
        raise RuntimeError(f"prepared-shape compilation requires tested Torch 2.11.0, got {version}")


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


class CapturedCall:
    """Replay an operation on fixed input storage without copying its persistent state."""

    def __init__(
        self,
        operation: Callable[[], Tensor],
        inputs: tuple[Tensor, ...],
        state: tuple[Tensor, ...] = (),
        *,
        device: torch.device,
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
                    with torch.cuda.graph(self.graph, stream=capture):
                        self.output = operation()
            finally:
                current.wait_stream(capture)
                for target, original in zip(state, saved, strict=True):
                    target.copy_(original)

    def __call__(self, inputs: tuple[Tensor, ...]) -> Tensor:
        with torch.cuda.device(self.device):
            for target, source in zip(self.inputs, inputs, strict=True):
                target.copy_(source)
            self.graph.replay()
        return self.output
