"""Explicit CUDA graph ownership for inference with persistent mutable state."""

from collections.abc import Callable

import torch
from torch import Tensor


class CapturedCall:
    """Replay an operation on fixed input storage without copying its persistent state."""

    def __init__(
        self, operation: Callable[[], Tensor], inputs: tuple[Tensor, ...], state: tuple[Tensor, ...] = ()
    ) -> None:
        self.inputs = inputs
        self.graph = torch.cuda.CUDAGraph()
        current = torch.cuda.current_stream()
        capture = torch.cuda.Stream()
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
        for target, source in zip(self.inputs, inputs, strict=True):
            target.copy_(source)
        self.graph.replay()
        return self.output
