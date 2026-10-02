"""Run a chunk policy in the native simulator and retain every frame seam."""

import time
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from hal.data.schema import MDS_PER_FRAME_DTYPES
from hal.sim.native import OBSERVATION_FIELDS
from hal.sim.native import NativeFrameBatch
from hal.sim.native import NativeMatch
from hal.sim.rollout import ChunkPolicy
from hal.sim.rollout import ObservationRow
from hal.sim.rollout import Slot
from hal.wire import ACTION_DIM


class NativeSimulator(Protocol):
    """The lane operations used by the synchronous evaluator."""

    def reset(self, configs: Sequence[NativeMatch], mask: NDArray[np.bool_]) -> NativeFrameBatch: ...

    def step(self, actions: NDArray[np.float32], mask: NDArray[np.bool_]) -> NativeFrameBatch: ...


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """One episode, including the reset row and each resulting post-frame."""

    arrays: Mapping[str, np.ndarray]
    wall_seconds: float
    simulator_seconds: float
    policy_seconds: float
    stop_reason: str

    @property
    def steps(self) -> int:
        return len(self.arrays["frame_id"]) - 1

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.arrays)


def _row(frame: NativeFrameBatch, slot: Slot) -> ObservationRow:
    flat = {name: value[0].item() for name, value in frame.columns.items()}
    return ObservationRow(
        int(frame.frame_id[0]),
        flat,
        np.array(frame.applied_action[0, slot.port - 1], copy=True),
        reset=bool(frame.reset[0]),
    )


class _FrameTrace:
    """Own frame history while the simulator reuses its borrowed frame buffers."""

    def __init__(self, frame: NativeFrameBatch, names: tuple[str, ...], capacity: int) -> None:
        self.names = names
        self.length = 0
        self.records: NDArray | None = None
        source = frame.records
        if source is not None:
            expected = {
                "columns",
                "frame_id",
                "applied_action",
                "wire_inputs",
                "reward",
                "terminated",
                "truncated",
                "reset",
            }
            if (
                source.shape != (1,)
                or not expected.issubset(source.dtype.names or ())
                or source["columns"].shape != (1, len(OBSERVATION_FIELDS))
                or source["columns"].dtype != np.uint32
                or source["frame_id"].shape != (1,)
                or source["frame_id"].dtype != np.int32
                or source["applied_action"].shape != (1, 2, ACTION_DIM)
                or source["applied_action"].dtype != np.float32
                or source["wire_inputs"].shape != (1, 2, 7)
                or source["wire_inputs"].dtype != np.int16
                or source["reward"].shape != (1, 2)
                or source["reward"].dtype != np.float32
                or any(
                    source[name].shape != (1,) or source[name].dtype != np.bool_
                    for name in ("terminated", "truncated", "reset")
                )
            ):
                raise ValueError("native frame record does not match the trace schema")
            self.records = np.empty(capacity, dtype=source.dtype)
            words = self.records["columns"]
            offsets = {name: index for index, name in enumerate(OBSERVATION_FIELDS)}
            arrays = {name: words[:, offsets[name]].view(np.dtype(MDS_PER_FRAME_DTYPES[name])) for name in names}
            arrays.update(
                {
                    "frame_id": self.records["frame_id"],
                    "submitted": self.records["applied_action"],
                    "wire_inputs": self.records["wire_inputs"],
                    "reward": self.records["reward"],
                    "terminated": self.records["terminated"],
                    "truncated": self.records["truncated"],
                    "reset": self.records["reset"],
                }
            )
            self.arrays = arrays
            return
        self.arrays = {name: np.empty(capacity, dtype=MDS_PER_FRAME_DTYPES[name]) for name in names}
        self.arrays.update(
            {
                "frame_id": np.empty(capacity, dtype=np.int32),
                "submitted": np.empty((capacity, 2, ACTION_DIM), dtype=np.float32),
                "wire_inputs": np.empty((capacity, 2, 7), dtype=np.int16),
                "reward": np.empty((capacity, 2), dtype=np.float32),
                "terminated": np.empty(capacity, dtype=np.bool_),
                "truncated": np.empty(capacity, dtype=np.bool_),
                "reset": np.empty(capacity, dtype=np.bool_),
            }
        )

    def append(self, frame: NativeFrameBatch) -> None:
        index = self.length
        if self.records is not None:
            source = frame.records
            if source is None or source.shape != (1,) or source.dtype != self.records.dtype:
                raise ValueError("native frame record changed during the episode")
            self.records[index] = source[0]
        else:
            arrays = self.arrays
            for name in self.names:
                arrays[name][index] = frame.columns[name][0]
            arrays["frame_id"][index] = frame.frame_id[0]
            arrays["submitted"][index] = frame.applied_action[0]
            arrays["wire_inputs"][index] = frame.wire_inputs[0]
            arrays["reward"][index] = frame.reward[0]
            arrays["terminated"][index] = frame.terminated[0]
            arrays["truncated"][index] = frame.truncated[0]
            arrays["reset"][index] = frame.reset[0]
        self.length += 1

    def result(self) -> dict[str, np.ndarray]:
        return {name: values[: self.length] for name, values in self.arrays.items()}


def run_episode(
    simulator: NativeSimulator,
    policy: ChunkPolicy,
    match: NativeMatch,
    *,
    max_frames: int,
) -> EpisodeResult:
    """Execute two due actions per plan, then ingest their two post-frames."""
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    spec = policy.runtime_spec
    if (
        spec.prediction_frames != 4
        or spec.execution_stride != 2
        or spec.committed_frames != 2
        or spec.action_dim != ACTION_DIM
        or spec.observed_actions
    ):
        raise ValueError("native 059 evaluation requires submitted-action 4/2/2 timing")

    started = time.perf_counter()
    reset_started = time.perf_counter()
    frame = simulator.reset((match,), np.array([True]))
    sim_seconds = time.perf_counter() - reset_started
    required_fields = tuple(sorted(frame.columns))
    if set(required_fields) != set(OBSERVATION_FIELDS):
        raise ValueError("native observation columns differ from HAL's 71-column schema")
    trace = _FrameTrace(frame, required_fields, max_frames + 1)
    trace.append(frame)
    slots = (Slot(0, 1), Slot(0, 2))
    pending = {slot: [_row(frame, slot)] for slot in slots}
    policy_seconds = 0.0
    steps = 0
    stop_reason = "frame_budget"
    while steps < max_frames:
        plan_started = time.perf_counter()
        plans = policy.plan_rows(pending)
        policy_seconds += time.perf_counter() - plan_started
        if plans.keys() != pending.keys() or any(np.shape(plans[slot]) != (4, ACTION_DIM) for slot in slots):
            raise ValueError("policy plan has the wrong slots or action shape")
        pending = {slot: [] for slot in slots}
        for offset in range(min(2, max_frames - steps)):
            action = np.stack([plans[slot][offset] for slot in slots]).astype(np.float32, copy=False)
            step_started = time.perf_counter()
            frame = simulator.step(action[None], np.array([True]))
            sim_seconds += time.perf_counter() - step_started
            steps += 1
            trace.append(frame)
            for slot in slots:
                pending[slot].append(_row(frame, slot))
            if bool(frame.terminated[0]) or bool(frame.truncated[0]):
                stop_reason = "native_terminated" if bool(frame.terminated[0]) else "native_truncated"
                break
        if stop_reason != "frame_budget":
            break
    return EpisodeResult(trace.result(), time.perf_counter() - started, sim_seconds, policy_seconds, stop_reason)


def first_difference(reference: EpisodeResult, repeat: EpisodeResult) -> tuple[str, int] | None:
    """Report the first byte-level difference, including NaN payloads."""
    if reference.arrays.keys() != repeat.arrays.keys():
        raise ValueError("episode traces have different columns")
    for name, before in reference.arrays.items():
        after = repeat.arrays[name]
        if before.dtype != after.dtype:
            return name, 0
        if before.shape != after.shape:
            return name, min(len(before), len(after))
        before_bytes = np.ascontiguousarray(before).view(np.uint8).reshape(len(before), -1)
        after_bytes = np.ascontiguousarray(after).view(np.uint8).reshape(len(after), -1)
        differing = np.flatnonzero(np.any(before_bytes != after_bytes, axis=1))
        if len(differing):
            return name, int(differing[0])
    return None


def replay_inputs(
    simulator: NativeSimulator,
    match: NativeMatch,
    reference: EpisodeResult,
) -> tuple[tuple[str, int] | None, float]:
    """Repeat the recorded inputs without policy inference and compare all rows."""
    started = time.perf_counter()
    frame = simulator.reset((match,), np.array([True]))
    names = tuple(name for name in frame.columns)
    trace = _FrameTrace(frame, names, len(reference.arrays["submitted"]))
    trace.append(frame)
    for action in reference.arrays["submitted"][1:]:
        frame = simulator.step(action[None], np.array([True]))
        trace.append(frame)
    repeat = EpisodeResult(trace.result(), 0.0, 0.0, 0.0, reference.stop_reason)
    return first_difference(reference, repeat), time.perf_counter() - started
