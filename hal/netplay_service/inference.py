"""Continuous policy batching for isolated netplay Dolphin processes."""

import math
import threading
import time
from collections import deque
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.connection import wait
from multiprocessing.shared_memory import SharedMemory
from numbers import Integral
from numbers import Real
from typing import cast

import numpy as np
from loguru import logger

from hal.controller import ControllerAction
from hal.inference.api import ObservationScalar
from hal.inference.api import Policy
from hal.inference.api import PolicyInput
from hal.inference.api import PolicyOutput
from hal.inference.api import PolicySpec
from hal.inference.api import RuntimeConfig
from hal.inference.api import validate_policy_inputs
from hal.inference.api import validate_policy_outputs
from hal.sim.ipc import ControlMessage
from hal.sim.ipc import MessageType
from hal.sim.ipc import receive_control
from hal.sim.ipc import send_control

_ACTION_WIDTH = 7
_EMPTY_SEQUENCE = np.iinfo(np.uint64).max


def _p95_ms(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return 1_000.0 * ordered[math.ceil(0.95 * len(ordered)) - 1]


@dataclass(frozen=True, slots=True)
class ServingArenaDescriptor:
    """Immutable identity and shape of a serving shared-memory block."""

    name: str
    size: int
    slots: int
    max_delay: int
    observation_fields: tuple[str, ...]
    max_identity_bytes: int = 64

    def __post_init__(self) -> None:
        if not self.name or self.size < 1:
            raise ValueError("serving arena needs a name and positive size")
        if self.slots < 1:
            raise ValueError("serving arena slots must be positive")
        if self.max_delay < 0:
            raise ValueError("serving arena max_delay must be non-negative")
        if len(set(self.observation_fields)) != len(self.observation_fields):
            raise ValueError("serving observation fields must be unique")
        if self.max_identity_bytes < 1:
            raise ValueError("max_identity_bytes must be positive")


class _ObservationView(Mapping[str, ObservationScalar]):
    """Typed read-only view over one shared observation row."""

    __slots__ = ("_fields", "_index", "_integer", "_values")

    def __init__(
        self,
        fields: tuple[str, ...],
        index: Mapping[str, int],
        values: np.ndarray,
        integer: np.ndarray,
    ) -> None:
        self._fields = fields
        self._index = index
        self._values = values
        self._integer = integer

    def __getitem__(self, key: str) -> ObservationScalar:
        try:
            index = self._index[key]
        except KeyError as error:
            raise KeyError(key) from error
        value = self._values[index]
        return int(value) if self._integer[index] else float(value)

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)


def _aligned(offset: int, alignment: int = 64) -> int:
    return (offset + alignment - 1) // alignment * alignment


def _action_values(action: ControllerAction) -> tuple[float, ...]:
    return (
        action.main_x,
        action.main_y,
        action.c_x,
        action.c_y,
        action.trigger_l,
        action.trigger_r,
        float(action.buttons),
    )


def _controller_action(values: np.ndarray) -> ControllerAction:
    return ControllerAction(
        main_x=float(values[0]),
        main_y=float(values[1]),
        c_x=float(values[2]),
        c_y=float(values[3]),
        trigger_l=float(values[4]),
        trigger_r=float(values[5]),
        buttons=int(values[6]),
    )


class ServingArena:
    """One request and response row per fixed inference slot."""

    def __init__(self, shared_memory: SharedMemory, descriptor: ServingArenaDescriptor, *, owner: bool) -> None:
        if shared_memory.size < descriptor.size:
            raise ValueError(f"serving shared memory has {shared_memory.size} bytes, expected {descriptor.size}")
        self._shared_memory = shared_memory
        self.descriptor = descriptor
        self.owner = owner
        self._build_views()

    @classmethod
    def create(
        cls,
        slots: int,
        max_delay: int,
        observation_fields: Sequence[str],
        *,
        max_identity_bytes: int = 64,
    ) -> ServingArena:
        fields = tuple(observation_fields)
        size = cls._required_bytes(slots, max_delay, len(fields), max_identity_bytes)
        shared_memory = SharedMemory(create=True, size=size)
        descriptor = ServingArenaDescriptor(
            shared_memory.name,
            size,
            slots,
            max_delay,
            fields,
            max_identity_bytes,
        )
        arena = cls(shared_memory, descriptor, owner=True)
        arena.request_sequence.fill(_EMPTY_SEQUENCE)
        arena.response_sequence.fill(_EMPTY_SEQUENCE)
        return arena

    @classmethod
    def attach(cls, descriptor: ServingArenaDescriptor) -> ServingArena:
        shared_memory = SharedMemory(name=descriptor.name, create=False, track=False)
        return cls(shared_memory, descriptor, owner=False)

    @staticmethod
    def _required_bytes(slots: int, max_delay: int, fields: int, identity_bytes: int) -> int:
        total = 0
        shapes = (
            ((slots,), np.dtype("<u8")),
            ((slots,), np.dtype("<u8")),
            ((slots,), np.dtype("<i8")),
            ((slots,), np.dtype("<i8")),
            ((slots,), np.dtype("u1")),
            ((slots,), np.dtype("u1")),
            ((slots,), np.dtype("u1")),
            ((slots,), np.dtype("<u2")),
            ((slots, identity_bytes), np.dtype("u1")),
            ((slots, fields), np.dtype("<f8")),
            ((slots, fields), np.dtype("u1")),
            ((slots, max_delay + 1, _ACTION_WIDTH), np.dtype("<f8")),
            ((slots, _ACTION_WIDTH), np.dtype("<f8")),
        )
        for shape, dtype in shapes:
            total = _aligned(total)
            total += int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        return _aligned(total)

    def _view(self, offset: int, shape: tuple[int, ...], dtype: np.dtype) -> tuple[np.ndarray, int]:
        offset = _aligned(offset)
        count = int(np.prod(shape, dtype=np.int64))
        width = count * dtype.itemsize
        view = np.ndarray(shape, dtype=dtype, buffer=self._shared_memory.buf, offset=offset)
        return view, offset + width

    def _build_views(self) -> None:
        descriptor = self.descriptor
        self._observation_index = {name: index for index, name in enumerate(descriptor.observation_fields)}
        offset = 0
        self.request_sequence, offset = self._view(offset, (descriptor.slots,), np.dtype("<u8"))
        self.response_sequence, offset = self._view(offset, (descriptor.slots,), np.dtype("<u8"))
        self.stream_id, offset = self._view(offset, (descriptor.slots,), np.dtype("<i8"))
        self.frame_id, offset = self._view(offset, (descriptor.slots,), np.dtype("<i8"))
        self.controlled_port, offset = self._view(offset, (descriptor.slots,), np.dtype("u1"))
        self.delay, offset = self._view(offset, (descriptor.slots,), np.dtype("u1"))
        self.reset, offset = self._view(offset, (descriptor.slots,), np.dtype("u1"))
        self.identity_length, offset = self._view(offset, (descriptor.slots,), np.dtype("<u2"))
        self.identity, offset = self._view(
            offset,
            (descriptor.slots, descriptor.max_identity_bytes),
            np.dtype("u1"),
        )
        fields = len(descriptor.observation_fields)
        self.observation, offset = self._view(offset, (descriptor.slots, fields), np.dtype("<f8"))
        self.observation_integer, offset = self._view(offset, (descriptor.slots, fields), np.dtype("u1"))
        self.actions, offset = self._view(
            offset,
            (descriptor.slots, descriptor.max_delay + 1, _ACTION_WIDTH),
            np.dtype("<f8"),
        )
        self.output, offset = self._view(offset, (descriptor.slots, _ACTION_WIDTH), np.dtype("<f8"))
        if _aligned(offset) != descriptor.size:
            raise AssertionError(f"serving arena layout used {_aligned(offset)} bytes, expected {descriptor.size}")

    def write_request(self, slot: int, sequence: int, item: PolicyInput) -> None:
        """Write one complete input and publish its sequence last."""
        self._validate_slot(slot)
        missing = set(self.descriptor.observation_fields) - item.observation.keys()
        if missing:
            raise ValueError(f"serving observation is missing fields {sorted(missing)}")
        identity = b"" if item.player_identity is None else item.player_identity.encode("utf-8")
        if len(identity) > self.descriptor.max_identity_bytes:
            raise ValueError(
                f"player identity uses {len(identity)} bytes; maximum is {self.descriptor.max_identity_bytes}"
            )
        delay = len(item.pending_actions)
        if delay > self.descriptor.max_delay:
            raise ValueError(f"request delay {delay} exceeds serving maximum {self.descriptor.max_delay}")
        for index, name in enumerate(self.descriptor.observation_fields):
            value = item.observation[name]
            if not isinstance(value, (Real, Integral)) or isinstance(value, bool):
                raise ValueError(f"observation {name!r} must be numeric")
            self.observation[slot, index] = value
            self.observation_integer[slot, index] = isinstance(value, Integral)
        self.identity[slot].fill(0)
        self.identity[slot, : len(identity)] = np.frombuffer(identity, dtype=np.uint8)
        self.identity_length[slot] = len(identity)
        self.stream_id[slot] = item.stream_id
        self.frame_id[slot] = item.frame_id
        self.controlled_port[slot] = item.controlled_port
        self.delay[slot] = delay
        self.reset[slot] = item.reset
        self.actions[slot, 0] = _action_values(item.applied_action)
        for index, action in enumerate(item.pending_actions, start=1):
            self.actions[slot, index] = _action_values(action)
        self.request_sequence[slot] = sequence

    def read_request(self, slot: int, sequence: int) -> PolicyInput:
        """Read one published request, preserving integer observation types."""
        self._validate_slot(slot)
        stored = int(self.request_sequence[slot])
        if stored != sequence:
            raise RuntimeError(f"slot {slot} request {sequence} is torn or stale; stored {stored}")
        delay = int(self.delay[slot])
        identity_length = int(self.identity_length[slot])
        try:
            identity = bytes(self.identity[slot, :identity_length]).decode("utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeError(f"slot {slot} player identity is not UTF-8") from error
        return PolicyInput(
            stream_id=int(self.stream_id[slot]),
            frame_id=int(self.frame_id[slot]),
            controlled_port=int(self.controlled_port[slot]),
            observation=_ObservationView(
                self.descriptor.observation_fields,
                self._observation_index,
                self.observation[slot],
                self.observation_integer[slot],
            ),
            applied_action=_controller_action(self.actions[slot, 0]),
            pending_actions=tuple(_controller_action(self.actions[slot, index]) for index in range(1, delay + 1)),
            player_identity=identity or None,
            reset=bool(self.reset[slot]),
        )

    def write_response(self, slot: int, sequence: int, action: ControllerAction) -> None:
        self._validate_slot(slot)
        self.output[slot] = _action_values(action)
        self.response_sequence[slot] = sequence

    def read_response(self, slot: int, sequence: int) -> ControllerAction:
        self._validate_slot(slot)
        stored = int(self.response_sequence[slot])
        if stored != sequence:
            raise RuntimeError(f"slot {slot} response {sequence} is torn or stale; stored {stored}")
        return _controller_action(self.output[slot])

    def _validate_slot(self, slot: int) -> None:
        if not 0 <= slot < self.descriptor.slots:
            raise ValueError(f"serving slot {slot} is outside [0, {self.descriptor.slots})")

    def close(self) -> None:
        for name in (
            "request_sequence",
            "response_sequence",
            "stream_id",
            "frame_id",
            "controlled_port",
            "delay",
            "reset",
            "identity_length",
            "identity",
            "observation",
            "observation_integer",
            "actions",
            "output",
        ):
            if hasattr(self, name):
                delattr(self, name)
        self._shared_memory.close()

    def unlink(self) -> None:
        if not self.owner:
            raise RuntimeError("only the serving arena owner can unlink it")
        self._shared_memory.unlink()

    def __enter__(self) -> ServingArena:
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            self.close()
        finally:
            if self.owner:
                self.unlink()


class RemotePolicy:
    """A one-slot policy client used inside a Dolphin process."""

    def __init__(
        self,
        spec: PolicySpec,
        runtime: RuntimeConfig,
        arena: ServingArena,
        connection: Connection,
        slot: int,
        *,
        initial_sequence: int = 0,
    ) -> None:
        if runtime.max_batch_size < 1:
            raise ValueError("remote policy runtime needs a positive batch size")
        self._spec = spec
        self._runtime = runtime
        self._arena = arena
        self._connection = connection
        self._slot = slot
        if initial_sequence < 0:
            raise ValueError("initial_sequence must be non-negative")
        self._sequence = initial_sequence

    @property
    def spec(self) -> PolicySpec:
        return self._spec

    def prepare(self, config: RuntimeConfig) -> None:
        if config != self._runtime:
            raise ValueError("remote policy runtime differs from the inference engine")

    def step(self, inputs: Sequence[PolicyInput]) -> tuple[PolicyOutput, ...]:
        if len(inputs) != 1:
            raise ValueError(f"a Dolphin slot must submit one policy input, got {len(inputs)}")
        item = inputs[0]
        validate_policy_inputs(self._spec, self._runtime, inputs)
        sequence = self._sequence
        self._sequence += 1
        self._arena.write_request(self._slot, sequence, item)
        send_control(
            self._connection,
            ControlMessage(MessageType.PLAN_REQUEST, worker_id=self._slot, sequence=sequence),
        )
        response, _ = receive_control(self._connection)
        if response.worker_id != self._slot or response.sequence != sequence:
            raise RuntimeError("inference response does not match the outstanding request")
        if response.message_type is MessageType.ERROR:
            raise RuntimeError("inference engine rejected the request")
        if response.message_type is not MessageType.PLAN_READY:
            raise RuntimeError(f"unexpected inference response {response.message_type.name}")
        action = self._arena.read_response(self._slot, sequence)
        return (PolicyOutput(item.stream_id, action),)


class ContinuousBatcher:
    """Serve fixed Dolphin slots through one prepared policy instance."""

    def __init__(
        self,
        policy: Policy,
        runtime: RuntimeConfig,
        arena: ServingArena,
        connections: Mapping[int, Connection],
        *,
        batch_wait_seconds: float = 0.0005,
    ) -> None:
        if not math.isfinite(batch_wait_seconds) or batch_wait_seconds < 0:
            raise ValueError("batch_wait_seconds must be finite and non-negative")
        if set(connections) != set(range(arena.descriptor.slots)):
            raise ValueError("inference connections must cover every serving slot")
        if arena.descriptor.observation_fields != policy.spec.required_observation_fields:
            raise ValueError("serving arena observation fields differ from the policy specification")
        if arena.descriptor.max_delay < max(runtime.transport_delays):
            raise ValueError("serving arena cannot hold the configured transport delays")
        if runtime.max_batch_size < arena.descriptor.slots:
            raise ValueError("policy batch size is smaller than the serving slot count")
        self.policy = policy
        self.runtime = runtime
        self.arena = arena
        self.connections = dict(connections)
        self.batch_wait_seconds = batch_wait_seconds
        self._slot_of = {connection: slot for slot, connection in connections.items()}
        self.batch_calls = 0
        self.batch_items = 0
        self.max_batch_items = 0
        self._policy_seconds: deque[float] = deque(maxlen=1_200)
        self._batch_wait_seconds: deque[float] = deque(maxlen=1_200)
        self._timing_lock = threading.Lock()

    def timing_p95_ms(self) -> tuple[float | None, float | None]:
        """Return model and request-coalescing p95 latency."""
        with self._timing_lock:
            policy_seconds = tuple(self._policy_seconds)
            batch_wait_seconds = tuple(self._batch_wait_seconds)
        return _p95_ms(policy_seconds), _p95_ms(batch_wait_seconds)

    def serve(self, stop: threading.Event) -> None:
        """Run until stopped; policy or protocol failures terminate the engine."""
        while not stop.is_set():
            ready = cast(list[Connection], wait(self.connections.values(), timeout=0.05))
            if not ready:
                continue
            wait_started = time.perf_counter()
            deadline = time.perf_counter() + self.batch_wait_seconds
            pending = list(ready)
            seen = set(pending)
            while len(pending) < len(self.connections):
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                more = cast(
                    list[Connection],
                    wait(
                        [connection for connection in self.connections.values() if connection not in seen],
                        timeout=remaining,
                    ),
                )
                if not more:
                    break
                pending.extend(more)
                seen.update(more)
            self._serve_batch(pending, time.perf_counter() - wait_started)

    def _serve_batch(self, connections: Sequence[Connection], batch_wait_seconds: float) -> None:
        requests: list[tuple[int, int, PolicyInput, Connection]] = []
        try:
            for connection in connections:
                message, _ = receive_control(connection)
                slot = self._slot_of[connection]
                if message.message_type is not MessageType.PLAN_REQUEST:
                    raise RuntimeError(f"slot {slot} sent {message.message_type.name}, expected PLAN_REQUEST")
                if message.worker_id != slot:
                    raise RuntimeError(f"connection for slot {slot} claimed slot {message.worker_id}")
                item = self.arena.read_request(slot, message.sequence)
                requests.append((slot, message.sequence, item, connection))
            inputs = tuple(request[2] for request in requests)
            validate_policy_inputs(self.policy.spec, self.runtime, inputs)
            policy_started = time.perf_counter()
            outputs = validate_policy_outputs(inputs, tuple(self.policy.step(inputs)))
            with self._timing_lock:
                self._policy_seconds.append(time.perf_counter() - policy_started)
                self._batch_wait_seconds.append(batch_wait_seconds)
            self.batch_calls += 1
            self.batch_items += len(inputs)
            self.max_batch_items = max(self.max_batch_items, len(inputs))
            if self.batch_calls == 1:
                logger.info(
                    "inference engine active batch_size={} streams={} delays={} resets={}",
                    len(inputs),
                    [item.stream_id for item in inputs],
                    [len(item.pending_actions) for item in inputs],
                    [item.reset for item in inputs],
                )
            elif self.batch_calls % 600 == 0:
                ordered_seconds = sorted(self._policy_seconds)
                p95_seconds = ordered_seconds[math.ceil(0.95 * len(ordered_seconds)) - 1]
                logger.info(
                    "inference batches={} items={} mean_batch={:.2f} max_batch={} "
                    "recent_policy_p95={:.1f}ms recent_policy_max={:.1f}ms",
                    self.batch_calls,
                    self.batch_items,
                    self.batch_items / self.batch_calls,
                    self.max_batch_items,
                    1_000.0 * p95_seconds,
                    1_000.0 * max(self._policy_seconds),
                )
            for slot, sequence, item, connection in requests:
                self.arena.write_response(slot, sequence, outputs[item.stream_id])
                send_control(
                    connection,
                    ControlMessage(MessageType.PLAN_READY, worker_id=slot, sequence=sequence),
                )
        except BaseException:
            for slot, sequence, _item, connection in requests:
                with suppress(BrokenPipeError, EOFError, OSError):
                    send_control(
                        connection,
                        ControlMessage(MessageType.ERROR, worker_id=slot, sequence=sequence),
                    )
            raise
