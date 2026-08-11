"""Fixed binary control messages and shared numeric rollout storage.

Large observations and plans stay in POSIX shared memory.  A 64-byte control
message only transfers ownership.  Runtime calls use ``send_bytes`` and never
pickle a frame, action chunk, or trajectory.
"""

import hashlib
import math
import struct
from collections.abc import Iterator
from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from typing import Final

import numpy as np

from hal.data.schema import MDS_PER_FRAME_DTYPES
from hal.wire import BUTTON_BITS
from hal.wire import MASK_INT32
from hal.wire import POST_FIELD_SUFFIXES

CONTROL_MAGIC: Final[int] = 0x314C4148  # little-endian bytes: HAL1
CONTROL_VERSION: Final[int] = 1
CONTROL_SIZE: Final[int] = 64
_CONTROL = struct.Struct("<IHHIIQQQQIHHII")
if _CONTROL.size != CONTROL_SIZE:
    raise AssertionError(f"control message is {_CONTROL.size} bytes, expected {CONTROL_SIZE}")


class MessageType(IntEnum):
    HELLO = 1
    START_TASK = 2
    TASK_STARTED = 3
    PLAN_REQUEST = 4
    PLAN_READY = 5
    TASK_DONE = 6
    RESULT_READY = 7
    RESULT_RELEASED = 8
    ERROR = 9
    SHUTDOWN = 10
    SHUTDOWN_ACK = 11


@dataclass(frozen=True, slots=True)
class ControlMessage:
    """One fixed-width worker notification.

    Field meanings depend on ``message_type``.  Shared-memory schemas own bulk
    data; this object only identifies the published generation and row range.
    """

    message_type: MessageType
    worker_id: int
    flags: int = 0
    task_generation: int = 0
    task_id: int = 0
    sequence: int = 0
    auxiliary_sequence: int = 0
    count: int = 0
    plan_slot: int = 0
    port_or_slot: int = 0
    status_code: int = 0

    def pack_into(self, out: bytearray) -> None:
        if len(out) != CONTROL_SIZE:
            raise ValueError(f"control output must be {CONTROL_SIZE} bytes, got {len(out)}")
        _CONTROL.pack_into(
            out,
            0,
            CONTROL_MAGIC,
            CONTROL_VERSION,
            int(self.message_type),
            self.worker_id,
            self.flags,
            self.task_generation,
            self.task_id,
            self.sequence,
            self.auxiliary_sequence,
            self.count,
            self.plan_slot,
            self.port_or_slot,
            self.status_code,
            0,
        )

    def pack(self) -> bytes:
        out = bytearray(CONTROL_SIZE)
        self.pack_into(out)
        return bytes(out)

    @classmethod
    def unpack(cls, payload: bytes | bytearray | memoryview) -> ControlMessage:
        if len(payload) != CONTROL_SIZE:
            raise ValueError(f"control message must be {CONTROL_SIZE} bytes, got {len(payload)}")
        (
            magic,
            version,
            message_type,
            worker_id,
            flags,
            task_generation,
            task_id,
            sequence,
            auxiliary_sequence,
            count,
            plan_slot,
            port_or_slot,
            status_code,
            reserved,
        ) = _CONTROL.unpack(payload)
        if magic != CONTROL_MAGIC:
            raise ValueError(f"invalid control magic 0x{magic:08x}")
        if version != CONTROL_VERSION:
            raise ValueError(f"unsupported control version {version}")
        if reserved != 0:
            raise ValueError(f"control reserved field must be zero, got {reserved}")
        try:
            kind = MessageType(message_type)
        except ValueError as exc:
            raise ValueError(f"unknown control message type {message_type}") from exc
        return cls(
            message_type=kind,
            worker_id=worker_id,
            flags=flags,
            task_generation=task_generation,
            task_id=task_id,
            sequence=sequence,
            auxiliary_sequence=auxiliary_sequence,
            count=count,
            plan_slot=plan_slot,
            port_or_slot=port_or_slot,
            status_code=status_code,
        )


def send_control(connection: Connection, message: ControlMessage, buffer: bytearray | None = None) -> bytearray:
    """Send one fixed message without pickle; return the reusable send buffer."""
    out = bytearray(CONTROL_SIZE) if buffer is None else buffer
    message.pack_into(out)
    connection.send_bytes(out)
    return out


def receive_control(connection: Connection, buffer: bytearray | None = None) -> tuple[ControlMessage, bytearray]:
    """Receive one fixed message into reusable storage without pickle."""
    out = bytearray(CONTROL_SIZE) if buffer is None else buffer
    received = connection.recv_bytes_into(out)
    if received != CONTROL_SIZE:
        raise ValueError(f"control message must be {CONTROL_SIZE} bytes, got {received}")
    return ControlMessage.unpack(out), out


_CONTROLLER_SUFFIXES: Final[tuple[str, ...]] = (
    *(f"button_{name}" for name in BUTTON_BITS),
    "main_stick_x",
    "main_stick_y",
    "c_stick_x",
    "c_stick_y",
    "trigger_l",
    "trigger_r",
)
_CONTROLLER_COLUMNS: Final[frozenset[str]] = frozenset(
    f"p{port}_{suffix}" for port in (1, 2) for suffix in _CONTROLLER_SUFFIXES
)
_DROP_COLUMNS: Final[frozenset[str]] = frozenset({"frame", "p1_rank", "p2_rank"}) | _CONTROLLER_COLUMNS
LIVE_COLUMN_DTYPES: Final[dict[str, np.dtype]] = {
    name: np.dtype(dtype).newbyteorder("<")
    for name, dtype in MDS_PER_FRAME_DTYPES.items()
    if name not in _DROP_COLUMNS
}
LIVE_FLOAT_COLUMNS: Final[tuple[str, ...]] = tuple(
    name for name, dtype in LIVE_COLUMN_DTYPES.items() if dtype.kind == "f"
)
LIVE_INT_COLUMNS: Final[tuple[str, ...]] = tuple(
    name for name, dtype in LIVE_COLUMN_DTYPES.items() if dtype.kind in ("i", "u")
)
_FLOAT_AT: Final[dict[str, int]] = {name: i for i, name in enumerate(LIVE_FLOAT_COLUMNS)}
_INT_AT: Final[dict[str, int]] = {name: i for i, name in enumerate(LIVE_INT_COLUMNS)}


def live_layout_hash() -> str:
    """Stable hash of the ordered shared observation columns and dtypes."""
    lines = [f"{name}:{LIVE_COLUMN_DTYPES[name].str}" for name in LIVE_COLUMN_DTYPES]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


class FlatFrameView(Mapping[str, float | int]):
    """Read-only mapping over one shared numeric observation row."""

    __slots__ = ("_floats", "_ints")

    def __init__(self, floats: np.ndarray, ints: np.ndarray) -> None:
        self._floats = floats
        self._ints = ints

    def __getitem__(self, key: str) -> float | int:
        float_at = _FLOAT_AT.get(key)
        if float_at is not None:
            return float(self._floats[float_at])
        int_at = _INT_AT.get(key)
        if int_at is not None:
            return int(self._ints[int_at])
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(LIVE_COLUMN_DTYPES)

    def __len__(self) -> int:
        return len(LIVE_COLUMN_DTYPES)


@dataclass(frozen=True, slots=True)
class ArenaSpec:
    workers: int
    ring_capacity: int
    prediction_frames: int
    action_dim: int
    action_token_groups: int = 0

    def __post_init__(self) -> None:
        for name in ("workers", "ring_capacity", "prediction_frames", "action_dim"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if self.action_token_groups < 0:
            raise ValueError(f"action_token_groups must be >= 0, got {self.action_token_groups}")


@dataclass(frozen=True, slots=True)
class ArenaDescriptor:
    name: str
    size: int
    spec: ArenaSpec
    layout_hash: str


@dataclass(frozen=True, slots=True)
class ResultSpec:
    frames: int
    segments: int
    ports: int

    def __post_init__(self) -> None:
        if self.frames < 1 or self.segments < 1 or self.ports < 1:
            raise ValueError(f"invalid result shape: {self}")


def result_shm_name(arena_name: str, worker: int) -> str:
    """A result name both endpoints can derive without serializing a string."""
    return f"halr_{arena_name.removeprefix('/')}_{worker}"


def discard_result_shm(name: str) -> None:
    """Remove one abandoned deterministic result slab, if it exists."""
    try:
        shm = SharedMemory(name=name, create=False, track=False)
    except FileNotFoundError:
        return
    try:
        shm.unlink()
    finally:
        shm.close()


class ResultArena:
    """Exact-size, cold-path trajectory slab owned and unlinked by the parent."""

    def __init__(self, shm: SharedMemory, spec: ResultSpec, *, unlink_on_close: bool) -> None:
        self._shm = shm
        self.spec = spec
        self.unlink_on_close = unlink_on_close
        at = 0
        self.frame_id, at = self._view(at, (spec.frames,), np.dtype("<i4"))
        self.random_seed, at = self._view(at, (spec.frames,), np.dtype("<u4"))
        self.segment_start, at = self._view(at, (spec.segments,), np.dtype("<i4"))
        self.segment_length, at = self._view(at, (spec.segments,), np.dtype("<i4"))
        self.post, at = self._view(
            at,
            (spec.ports, len(POST_FIELD_SUFFIXES), spec.frames),
            np.dtype("<f8"),
        )
        if _aligned(at) != shm.size:
            raise AssertionError(f"result layout used {_aligned(at)} bytes, allocation has {shm.size}")

    @staticmethod
    def required_bytes(spec: ResultSpec) -> int:
        total = 0
        for shape, dtype in (
            ((spec.frames,), np.dtype("<i4")),
            ((spec.frames,), np.dtype("<u4")),
            ((spec.segments,), np.dtype("<i4")),
            ((spec.segments,), np.dtype("<i4")),
            ((spec.ports, len(POST_FIELD_SUFFIXES), spec.frames), np.dtype("<f8")),
        ):
            total = _aligned(total)
            total += int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        return _aligned(total)

    @classmethod
    def create(cls, name: str, spec: ResultSpec) -> ResultArena:
        shm = SharedMemory(name=name, create=True, size=cls.required_bytes(spec), track=False)
        return cls(shm, spec, unlink_on_close=False)

    @classmethod
    def attach(cls, name: str, spec: ResultSpec) -> ResultArena:
        shm = SharedMemory(name=name, create=False, track=False)
        return cls(shm, spec, unlink_on_close=True)

    def _view(self, at: int, shape: tuple[int, ...], dtype: np.dtype) -> tuple[np.ndarray, int]:
        at = _aligned(at)
        width = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        return np.ndarray(shape, dtype=dtype, buffer=self._shm.buf, offset=at), at + width

    def close(self) -> None:
        for name in ("frame_id", "random_seed", "segment_start", "segment_length", "post"):
            if hasattr(self, name):
                delattr(self, name)
        self._shm.close()
        if self.unlink_on_close:
            self._shm.unlink()


def _aligned(at: int, alignment: int = 64) -> int:
    return (at + alignment - 1) // alignment * alignment


class RolloutArena:
    """Typed views over one parent-owned shared-memory allocation."""

    def __init__(self, shm: SharedMemory, descriptor: ArenaDescriptor, *, owner: bool) -> None:
        self._shm = shm
        self.descriptor = descriptor
        self.owner = owner
        if descriptor.layout_hash != live_layout_hash():
            raise ValueError(
                f"live layout hash mismatch: arena {descriptor.layout_hash}, process {live_layout_hash()}"
            )
        if shm.size < descriptor.size:
            raise ValueError(f"shared memory has {shm.size} bytes, expected at least {descriptor.size}")
        self._build_views()

    @classmethod
    def create(cls, spec: ArenaSpec) -> RolloutArena:
        size = cls.required_bytes(spec)
        shm = SharedMemory(create=True, size=size)
        descriptor = ArenaDescriptor(name=shm.name, size=size, spec=spec, layout_hash=live_layout_hash())
        arena = cls(shm, descriptor, owner=True)
        arena.obs_sequence.fill(np.iinfo(np.uint64).max)
        return arena

    @classmethod
    def attach(cls, descriptor: ArenaDescriptor) -> RolloutArena:
        shm = SharedMemory(name=descriptor.name, create=False, track=False)
        return cls(shm, descriptor, owner=False)

    @staticmethod
    def required_bytes(spec: ArenaSpec) -> int:
        total = 0
        shapes = (
            ((spec.workers, spec.ring_capacity), np.dtype("<u8")),
            ((spec.workers, spec.ring_capacity), np.dtype("<i4")),
            ((spec.workers, spec.ring_capacity), np.dtype("u1")),
            ((spec.workers, spec.ring_capacity, len(LIVE_FLOAT_COLUMNS)), np.dtype("<f4")),
            ((spec.workers, spec.ring_capacity, len(LIVE_INT_COLUMNS)), np.dtype("<i4")),
            ((spec.workers, spec.ring_capacity, spec.action_dim), np.dtype("<f4")),
            ((spec.workers, spec.ring_capacity, spec.action_token_groups), np.dtype("<i4")),
            ((spec.workers, 2, spec.prediction_frames, spec.action_dim), np.dtype("<f4")),
            ((spec.workers, 2, spec.prediction_frames, spec.action_token_groups), np.dtype("<i4")),
        )
        for shape, dtype in shapes:
            total = _aligned(total)
            total += int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        return _aligned(total)

    def _view(self, at: int, shape: tuple[int, ...], dtype: np.dtype) -> tuple[np.ndarray, int]:
        at = _aligned(at)
        count = int(np.prod(shape, dtype=np.int64))
        width = count * dtype.itemsize
        view = np.ndarray(shape, dtype=dtype, buffer=self._shm.buf, offset=at)
        return view, at + width

    def _build_views(self) -> None:
        spec = self.descriptor.spec
        at = 0
        self.obs_sequence, at = self._view(at, (spec.workers, spec.ring_capacity), np.dtype("<u8"))
        self.obs_frame_id, at = self._view(at, (spec.workers, spec.ring_capacity), np.dtype("<i4"))
        self.obs_reset, at = self._view(at, (spec.workers, spec.ring_capacity), np.dtype("u1"))
        self.obs_floats, at = self._view(
            at, (spec.workers, spec.ring_capacity, len(LIVE_FLOAT_COLUMNS)), np.dtype("<f4")
        )
        self.obs_ints, at = self._view(at, (spec.workers, spec.ring_capacity, len(LIVE_INT_COLUMNS)), np.dtype("<i4"))
        self.obs_actions, at = self._view(at, (spec.workers, spec.ring_capacity, spec.action_dim), np.dtype("<f4"))
        self.obs_tokens, at = self._view(
            at, (spec.workers, spec.ring_capacity, spec.action_token_groups), np.dtype("<i4")
        )
        self.plan_actions, at = self._view(
            at, (spec.workers, 2, spec.prediction_frames, spec.action_dim), np.dtype("<f4")
        )
        self.plan_tokens, at = self._view(
            at, (spec.workers, 2, spec.prediction_frames, spec.action_token_groups), np.dtype("<i4")
        )
        if _aligned(at) != self.descriptor.size:
            raise AssertionError(f"arena layout used {_aligned(at)} bytes, descriptor has {self.descriptor.size}")

    def write_observation(
        self,
        worker: int,
        sequence: int,
        frame_id: int,
        flat: Mapping[str, float | int],
        action: np.ndarray,
        *,
        reset: bool,
        tokens: np.ndarray | None = None,
    ) -> None:
        """Write one complete row, then publish its absolute sequence last."""
        spec = self.descriptor.spec
        at = sequence % spec.ring_capacity
        float_row = self.obs_floats[worker, at]
        int_row = self.obs_ints[worker, at]
        for i, name in enumerate(LIVE_FLOAT_COLUMNS):
            float_row[i] = flat[name]
        for i, name in enumerate(LIVE_INT_COLUMNS):
            value = flat[name]
            int_row[i] = MASK_INT32 if isinstance(value, float) and not math.isfinite(value) else value
        action_row = np.asarray(action, dtype=np.float32)
        if action_row.shape != (spec.action_dim,):
            raise ValueError(f"action has shape {action_row.shape}, expected {(spec.action_dim,)}")
        self.obs_actions[worker, at] = action_row
        if spec.action_token_groups:
            if tokens is None:
                self.obs_tokens[worker, at].fill(-1)
            else:
                token_row = np.asarray(tokens, dtype=np.int32)
                if token_row.shape != (spec.action_token_groups,):
                    raise ValueError(f"tokens have shape {token_row.shape}, expected {(spec.action_token_groups,)}")
                self.obs_tokens[worker, at] = token_row
        self.obs_frame_id[worker, at] = frame_id
        self.obs_reset[worker, at] = reset
        self.obs_sequence[worker, at] = sequence

    def observation(self, worker: int, sequence: int) -> tuple[FlatFrameView, np.ndarray, bool]:
        spec = self.descriptor.spec
        at = sequence % spec.ring_capacity
        stored = int(self.obs_sequence[worker, at])
        if stored != sequence:
            raise ValueError(f"worker {worker} row {sequence} was overwritten or torn; stored sequence is {stored}")
        return (
            FlatFrameView(self.obs_floats[worker, at], self.obs_ints[worker, at]),
            self.obs_actions[worker, at],
            bool(self.obs_reset[worker, at]),
        )

    def close(self) -> None:
        # Drop our exported ndarray views before unmapping the allocation. A
        # dangling NumPy view can segfault during later exception formatting.
        for name in (
            "obs_sequence",
            "obs_frame_id",
            "obs_reset",
            "obs_floats",
            "obs_ints",
            "obs_actions",
            "obs_tokens",
            "plan_actions",
            "plan_tokens",
        ):
            if hasattr(self, name):
                delattr(self, name)
        self._shm.close()

    def unlink(self) -> None:
        if not self.owner:
            raise RuntimeError("only the arena owner can unlink shared memory")
        self._shm.unlink()

    def __enter__(self) -> RolloutArena:
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            self.close()
        finally:
            if self.owner:
                self.unlink()
