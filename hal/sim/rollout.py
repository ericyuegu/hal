"""Runtime contract for asynchronous action-chunk policies.

The simulator must know when it needs another plan, but it must not know which
model produced that plan.  ``PolicyRuntimeSpec`` is the complete scheduling
contract between a Session worker and the inference broker.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class PolicyRuntimeSpec:
    """Policy-independent rollout sizes for one model-driven slot.

    ``prediction_frames`` is the number of actions in one returned plan.
    ``execution_stride`` is the number of new actions executed before the next
    request.  ``committed_frames`` is the prefix that can execute while the next
    plan is in flight.  A synchronous policy uses zero committed frames.
    """

    context_frames: int
    prediction_frames: int
    execution_stride: int
    committed_frames: int
    action_dim: int
    action_token_groups: int = 0

    def __post_init__(self) -> None:
        if self.context_frames < 1:
            raise ValueError(f"context_frames must be >= 1, got {self.context_frames}")
        if self.prediction_frames < 1:
            raise ValueError(f"prediction_frames must be >= 1, got {self.prediction_frames}")
        if not 1 <= self.execution_stride <= self.prediction_frames:
            raise ValueError(
                "execution_stride must be between 1 and prediction_frames, got "
                f"{self.execution_stride} for {self.prediction_frames}"
            )
        available = self.prediction_frames - self.execution_stride
        if not 0 <= self.committed_frames <= available:
            raise ValueError(f"committed_frames must be between 0 and {available}, got {self.committed_frames}")
        if self.action_dim < 1:
            raise ValueError(f"action_dim must be >= 1, got {self.action_dim}")
        if self.action_token_groups < 0:
            raise ValueError(f"action_token_groups must be >= 0, got {self.action_token_groups}")

    @property
    def raw_ring_capacity(self) -> int:
        """Rows needed to retain one context and all possible unpublished work."""
        return max(self.context_frames, self.execution_stride + self.committed_frames + 1)


@dataclass(frozen=True, slots=True)
class ObservationRow:
    """One numeric observation and the action that produced it.

    ``flat`` can be a normal dictionary or a zero-copy view over shared memory.
    The broker consumes rows before it releases that shared ring generation.
    """

    frame_id: int
    flat: Mapping[str, float | int]
    action: np.ndarray
    reset: bool = False


class ChunkPolicy(Protocol):
    """Structural documentation for the worker/broker action-chunk contract."""

    @property
    def runtime_spec(self) -> PolicyRuntimeSpec: ...

    def plan_rows(self, rows: Mapping[object, Sequence[ObservationRow]]) -> Mapping[object, np.ndarray]: ...


def nearest_power_of_two(value: int) -> int:
    """Return the nearest power of two, selecting the larger power on a tie."""
    if value < 1:
        raise ValueError(f"value must be >= 1, got {value}")
    lower = 1 << (value.bit_length() - 1)
    if lower == value:
        return value
    upper = lower << 1
    return upper if value - lower >= upper - value else lower


def covering_power_of_two(value: int) -> int:
    """Return the smallest power of two that can contain ``value`` rows."""
    if value < 1:
        raise ValueError(f"value must be >= 1, got {value}")
    return 1 << (value - 1).bit_length()
