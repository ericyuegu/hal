"""Absolute-frame requests for inference that overlaps game execution."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from typing import runtime_checkable

from hal.controller import ControllerAction
from hal.inference.api import PolicyInput
from hal.inference.api import PolicySpec
from hal.inference.api import RuntimeConfig
from hal.inference.api import validate_controller_action
from hal.inference.api import validate_policy_inputs


@dataclass(frozen=True, slots=True)
class TimingSchedule:
    compute_frames: int
    transport_frames: int
    horizon: int

    def __post_init__(self) -> None:
        for value in (self.compute_frames, self.transport_frames, self.horizon):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("schedule values must be non-negative integers")
        if self.horizon < self.prefix_frames + self.replan_frames:
            raise ValueError("prediction horizon must cover the forced prefix and replan interval")

    @property
    def budget_frames(self) -> int:
        return self.compute_frames + 1

    @property
    def replan_frames(self) -> int:
        return self.budget_frames

    @property
    def prefix_frames(self) -> int:
        return self.budget_frames + self.transport_frames

    @property
    def reserve_frames(self) -> int:
        return self.horizon - self.prefix_frames - self.replan_frames


def latency_frames(seconds: float) -> int:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("latency must be finite and non-negative")
    return math.ceil(seconds * 60)


def contiguous_horizons(offsets: tuple[int, ...]) -> tuple[int, ...]:
    count = 0
    for expected, actual in enumerate(offsets, start=1):
        if actual != expected:
            break
        count += 1
    return tuple(range(1, count + 1))


@dataclass(frozen=True, slots=True)
class ChunkRequest:
    stream_id: int
    generation: int
    sequence: int
    source_frame: int
    context: tuple[PolicyInput, ...]
    forced_prefix: tuple[ControllerAction, ...]

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (self.stream_id, self.generation, self.sequence, self.source_frame)
        ):
            raise ValueError("chunk identity fields must be integers")
        if self.generation < 0 or self.sequence < 0:
            raise ValueError("chunk generation and sequence must be non-negative")
        if not self.context or self.context[-1].frame_id != self.source_frame:
            raise ValueError("chunk context must end at the source frame")
        for index, item in enumerate(self.context):
            if item.stream_id != self.stream_id:
                raise ValueError("chunk context contains another stream")
            if index and item.frame_id != self.context[index - 1].frame_id + 1:
                raise ValueError("chunk context must be contiguous")
        for action in self.forced_prefix:
            validate_controller_action(action)


@dataclass(frozen=True, slots=True)
class FrameAction:
    target_frame: int
    action: ControllerAction


@dataclass(frozen=True, slots=True)
class ChunkResponse:
    stream_id: int
    generation: int
    sequence: int
    source_frame: int
    actions: tuple[FrameAction, ...]


def chunk_response(request: ChunkRequest, tail: Sequence[ControllerAction]) -> ChunkResponse:
    actions = (*request.forced_prefix, *tail)
    return ChunkResponse(
        request.stream_id,
        request.generation,
        request.sequence,
        request.source_frame,
        tuple(FrameAction(request.source_frame + offset, action) for offset, action in enumerate(actions, 1)),
    )


def validate_chunk_request(
    spec: PolicySpec, runtime: RuntimeConfig, request: ChunkRequest, *, context_frames: int, prefix_frames: int
) -> None:
    if len(request.context) > context_frames or len(request.forced_prefix) != prefix_frames:
        raise ValueError("chunk context or forced prefix differs from the prepared shape")
    for item in request.context:
        validate_policy_inputs(spec, runtime, (item,))


def validate_chunk_response(request: ChunkRequest, response: ChunkResponse, horizon: int) -> None:
    if (response.stream_id, response.generation, response.sequence, response.source_frame) != (
        request.stream_id,
        request.generation,
        request.sequence,
        request.source_frame,
    ):
        raise ValueError("chunk response does not match its request")
    if len(response.actions) != horizon:
        raise ValueError("chunk response has the wrong horizon")
    for offset, item in enumerate(response.actions, 1):
        if item.target_frame != request.source_frame + offset:
            raise ValueError("chunk response target frames are not contiguous")
        validate_controller_action(item.action)
    if tuple(item.action for item in response.actions[: len(request.forced_prefix)]) != request.forced_prefix:
        raise ValueError("chunk response changed the forced prefix")


@runtime_checkable
class ChunkPolicy(Protocol):
    @property
    def spec(self) -> PolicySpec: ...

    @property
    def sampling_seed(self) -> int: ...

    @property
    def context_frames(self) -> int: ...

    @property
    def supported_horizons(self) -> tuple[int, ...]: ...

    def prepare_chunks(self, runtime: RuntimeConfig, horizon: int, prefix_frames: int) -> None: ...

    def reset_chunks(self) -> None: ...

    def warmup_context(self, stream_id: int, source_frame: int, transport: int) -> tuple[PolicyInput, ...]: ...

    def plan_chunks(self, requests: Sequence[ChunkRequest]) -> Sequence[ChunkResponse]: ...
