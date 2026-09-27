"""Spawned Session worker for shared-memory closed-loop rollout.

The worker owns Dolphin, libmelee, canonical-frame parsing, live flattening,
controller conversion, and trajectory transposition.  Only numeric rows and
numeric plans cross the hot process boundary, through ``RolloutArena``.
"""

import faulthandler
import math
import time
from collections.abc import Mapping
from contextlib import suppress
from multiprocessing.connection import Connection
from typing import Any

import numpy as np
from loguru import logger

from hal.controller import action_vec_to_controller
from hal.controller import controller_to_action_vec
from hal.representation.observations import flatten_canonical_frame
from hal.sim.inputs import ControllerInputs
from hal.sim.inputs import canonical_pre_to_action
from hal.sim.ipc import ControlMessage
from hal.sim.ipc import MessageType
from hal.sim.ipc import ResultArena
from hal.sim.ipc import ResultSpec
from hal.sim.ipc import RolloutArena
from hal.sim.ipc import receive_control
from hal.sim.ipc import result_shm_name
from hal.sim.ipc import send_control
from hal.sim.rollout import PolicyRuntimeSpec
from hal.sim.session import Matchup
from hal.sim.session import Session
from hal.sim.session import SessionOptions
from hal.sim.trajectory import Trajectory
from hal.wire import POST_FIELD_SUFFIXES


def _frame_id(frame: Mapping, previous: int) -> int:
    value = frame.get("id", previous + 1)
    if isinstance(value, int):
        return value
    if not math.isfinite(value):
        raise ValueError(f"canonical frame id is {value!r}; Dolphin produced a torn frame")
    return int(value)


def _match_metadata(matchup: Matchup) -> dict[str, object]:
    return {
        "stage": int(matchup.stage.value),
        "character": {player.port: int(player.character.value) for player in matchup.players},
    }


class _SessionWorkerState:
    """Own one worker's observation, plan, and controller-flush state."""

    def __init__(
        self,
        worker_id: int,
        connection: Connection,
        arena: RolloutArena,
        matchup: Matchup,
        model_ports: tuple[int, ...],
        arena_slots: tuple[int, ...],
        runtime: PolicyRuntimeSpec,
    ) -> None:
        if len(model_ports) != len(arena_slots):
            raise ValueError(f"got {len(model_ports)} model ports but {len(arena_slots)} shared arena slots")
        self.worker_id = worker_id
        self.connection = connection
        self.arena = arena
        self.model_ports = model_ports
        self.arena_slot_of = dict(zip(model_ports, arena_slots, strict=True))
        self.runtime = runtime
        self.ports = tuple(player.port for player in matchup.players)
        self.metadata = _match_metadata(matchup)
        self.neutral_actions = {port: np.zeros(runtime.action_dim, dtype=np.float32) for port in model_ports}
        self.trajectories: list[Trajectory] = []
        self.segment: list[dict] = []
        self.sequence = 1
        self.plan_generation = {port: 0 for port in model_ports}
        self.observation_started_ns: dict[int, int] = {}
        self.plan_started_ns = {port: 0 for port in model_ports}
        self.plan_ack_generation = {port: 0 for port in model_ports}
        self.send_buffer = bytearray(64)
        self.receive_buffer = bytearray(64)

    def close_segment(self) -> None:
        if self.segment:
            self.trajectories.append(Trajectory.from_capture(self.segment, self.ports))
        self.segment = []

    def publish(self, current: dict, actions: Mapping[int, np.ndarray], *, reset: bool) -> None:
        self.metadata["stage"] = current.get("stage", self.metadata["stage"])
        view = {**current, "_matchup": self.metadata}
        flat = flatten_canonical_frame(view)
        for port in self.model_ports:
            applied = actions[port]
            if self.runtime.observed_actions:
                applied = controller_to_action_vec(canonical_pre_to_action(current["ports"][port]["leader"]["pre"]))
            self.arena.write_observation(
                self.arena_slot_of[port], self.sequence, int(current["id"]), flat, applied, reset=reset
            )

    def request_plans(self, first_sequence: int, count: int) -> dict[int, np.ndarray]:
        for port in self.model_ports:
            generation = self.plan_generation[port]
            self.send_buffer = send_control(
                self.connection,
                ControlMessage(
                    message_type=MessageType.PLAN_REQUEST,
                    worker_id=self.worker_id,
                    task_generation=self.observation_started_ns[first_sequence],
                    task_id=self.arena_slot_of[port],
                    sequence=first_sequence,
                    auxiliary_sequence=generation,
                    count=count,
                    plan_slot=generation & 1,
                    port_or_slot=port,
                ),
                self.send_buffer,
            )
        for published_sequence in range(first_sequence, first_sequence + count):
            self.observation_started_ns.pop(published_sequence, None)
        plans: dict[int, np.ndarray] = {}
        for _ in self.model_ports:
            reply, self.receive_buffer = receive_control(self.connection, self.receive_buffer)
            if reply.message_type is not MessageType.PLAN_READY:
                raise RuntimeError(f"worker {self.worker_id} expected PLAN_READY, got {reply.message_type.name}")
            port = reply.port_or_slot
            if port not in self.arena_slot_of or reply.task_id != self.arena_slot_of[port]:
                raise RuntimeError(
                    f"worker {self.worker_id} got a plan for unknown port {port} in row {reply.task_id}"
                )
            generation = self.plan_generation[port]
            plan_slot = generation & 1
            if reply.auxiliary_sequence != generation or reply.plan_slot != plan_slot:
                raise RuntimeError(
                    f"worker {self.worker_id} port {port} got stale plan generation "
                    f"{reply.auxiliary_sequence} in slot {reply.plan_slot}; expected {generation} in {plan_slot}"
                )
            plans[port] = self.arena.plan_actions[self.arena_slot_of[port], plan_slot]
            self.plan_started_ns[port] = reply.task_generation
            self.plan_ack_generation[port] = reply.auxiliary_sequence
            self.plan_generation[port] += 1
        return plans

    def prepare_first_inputs(self, plans: Mapping[int, np.ndarray]) -> dict[int, ControllerInputs]:
        return {port: action_vec_to_controller(plans[port][0]) for port in self.model_ports}

    def acknowledge_plans(self) -> None:
        for port in self.model_ports:
            self.send_buffer = send_control(
                self.connection,
                ControlMessage(
                    message_type=MessageType.PLAN_APPLIED,
                    worker_id=self.worker_id,
                    task_generation=self.plan_started_ns[port],
                    auxiliary_sequence=self.plan_ack_generation[port],
                    port_or_slot=port,
                ),
                self.send_buffer,
            )


def session_worker(
    worker_id: int,
    connection: Connection,
    arena_descriptor: Any,
    session_kwargs: SessionOptions,
    matchup: Matchup,
    model_ports: tuple[int, ...],
    arena_slots: tuple[int, ...],
    runtime: PolicyRuntimeSpec,
    max_frames: int,
    instant_restart: bool,
) -> None:
    """Process target. All arguments are cold, spawn-time values."""
    faulthandler.enable()
    state: _SessionWorkerState | None = None
    try:
        arena = RolloutArena.attach(arena_descriptor)
        try:
            state = _SessionWorkerState(worker_id, connection, arena, matchup, model_ports, arena_slots, runtime)
            with Session(**session_kwargs) as session:
                frame = session.start_match(matchup)
                frame_id = _frame_id(frame, -1)
                state.segment.append(frame)
                state.observation_started_ns[state.sequence] = time.perf_counter_ns()
                state.publish(frame, state.neutral_actions, reset=True)
                plans = state.request_plans(state.sequence, 1)
                first_inputs = state.prepare_first_inputs(plans)
                captured = 1
                done = False
                while captured < max_frames and not done:
                    first_unpublished = state.sequence + 1
                    executed = 0
                    while executed < runtime.execution_stride and captured < max_frames:
                        actions = {port: plans[port][executed] for port in model_ports}
                        controller_inputs: dict[int, ControllerInputs] = (
                            first_inputs
                            if executed == 0
                            else {port: action_vec_to_controller(action) for port, action in actions.items()}
                        )
                        frame, in_game = session.step(
                            controller_inputs,
                            on_inputs_flushed=state.acknowledge_plans if executed == 0 else None,
                        )
                        next_id = _frame_id(frame, frame_id)
                        state.sequence += 1
                        captured += 1
                        reset = instant_restart and next_id < frame_id
                        if reset:
                            state.close_segment()
                            state.segment = [frame]
                        else:
                            state.segment.append(frame)
                        frame_id = next_id
                        executed += 1
                        if not in_game:
                            # Match-end frames can omit live player fields. Capture but do not flatten them.
                            state.close_segment()
                            done = True
                            break
                        state.observation_started_ns[state.sequence] = time.perf_counter_ns()
                        state.publish(frame, state.neutral_actions if reset else actions, reset=reset)
                        if reset:
                            plans = state.request_plans(state.sequence, 1)
                            first_inputs = state.prepare_first_inputs(plans)
                            break
                    else:
                        if captured < max_frames:
                            plans = state.request_plans(first_unpublished, executed)
                            first_inputs = state.prepare_first_inputs(plans)
                        continue
                    if done:
                        break
                    if reset:
                        continue
                state.close_segment()
        finally:
            # Plan arrays are zero-copy slices of the arena mapping.
            plans = None
            actions = None
            arena.close()
        assert state is not None
        trajectories = state.trajectories
        ports = state.ports
        frame_count = sum(len(trajectory) for trajectory in trajectories)
        result_spec = ResultSpec(frames=frame_count, segments=len(trajectories), ports=len(ports))
        result = ResultArena.create(result_shm_name(arena_descriptor.name, worker_id), result_spec)
        try:
            at = 0
            for segment_index, trajectory in enumerate(trajectories):
                stop = at + len(trajectory)
                result.segment_start[segment_index] = at
                result.segment_length[segment_index] = len(trajectory)
                result.frame_id[at:stop] = trajectory.frame_id
                result.random_seed[at:stop] = trajectory.random_seed
                for port_index, port in enumerate(ports):
                    for field_index, field in enumerate(POST_FIELD_SUFFIXES):
                        result.post[port_index, field_index, at:stop] = trajectory.post[port][field]
                at = stop
            state.send_buffer = send_control(
                connection,
                ControlMessage(
                    message_type=MessageType.RESULT_READY,
                    worker_id=worker_id,
                    auxiliary_sequence=len(trajectories),
                    count=frame_count,
                ),
                state.send_buffer,
            )
            reply, state.receive_buffer = receive_control(connection, state.receive_buffer)
            if reply.message_type is not MessageType.RESULT_RELEASED:
                raise RuntimeError(f"worker {worker_id} expected RESULT_RELEASED, got {reply.message_type.name}")
        finally:
            result.close()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.opt(exception=exc).warning(f"Session worker {worker_id} failed: {error}")
        with suppress(BrokenPipeError, EOFError, OSError):
            send_control(
                connection,
                ControlMessage(message_type=MessageType.ERROR, worker_id=worker_id, status_code=1),
                bytearray(64) if state is None else state.send_buffer,
            )
    finally:
        connection.close()
