"""Spawned Session worker for shared-memory closed-loop rollout.

The worker owns Dolphin, libmelee, canonical-frame parsing, live flattening,
controller conversion, and trajectory transposition.  Only numeric rows and
numeric plans cross the hot process boundary, through ``RolloutArena``.
"""

import faulthandler
import math
from collections.abc import Mapping
from contextlib import suppress
from multiprocessing.connection import Connection
from typing import Any

import numpy as np
from loguru import logger

from hal.sim.inputs import action_vec_to_controller
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
from hal.sim.trajectory import Trajectory
from hal.training.canonical import flatten_canonical_frame
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


def _session_worker(
    worker_id: int,
    connection: Connection,
    arena_descriptor: Any,
    session_kwargs: dict[str, object],
    matchup: Matchup,
    model_ports: tuple[int, ...],
    arena_slots: tuple[int, ...],
    runtime: PolicyRuntimeSpec,
    max_frames: int,
    instant_restart: bool,
) -> None:
    """Process target. All arguments are cold, spawn-time values."""
    faulthandler.enable()
    send_buffer = bytearray(64)
    receive_buffer = bytearray(64)
    trajectories: list[Trajectory] = []
    segment: list[dict] = []
    ports = tuple(player.port for player in matchup.players)
    if len(model_ports) != len(arena_slots):
        raise ValueError(f"got {len(model_ports)} model ports but {len(arena_slots)} shared arena slots")
    arena_slot_of = dict(zip(model_ports, arena_slots, strict=True))

    def close_segment() -> None:
        nonlocal segment
        if segment:
            trajectories.append(Trajectory.from_capture(segment, ports))
        segment = []

    try:
        arena = RolloutArena.attach(arena_descriptor)
        try:
            with Session(**session_kwargs) as session:
                frame = session.start_match(matchup)
                frame_id = _frame_id(frame, -1)
                segment.append(frame)
                metadata = _match_metadata(matchup)
                neutral = np.zeros(runtime.action_dim, dtype=np.float32)
                sequence = 1
                plan_generation = {port: 0 for port in model_ports}

                def publish(current: dict, actions: Mapping[int, np.ndarray], *, reset: bool) -> None:
                    metadata["stage"] = current.get("stage", metadata["stage"])
                    view = {**current, "_matchup": metadata}
                    flat = flatten_canonical_frame(view)
                    for port in model_ports:
                        arena.write_observation(
                            arena_slot_of[port],
                            sequence,
                            int(current["id"]),
                            flat,
                            actions[port],
                            reset=reset,
                        )

                def request_plans(first_sequence: int, count: int) -> dict[int, np.ndarray]:
                    nonlocal send_buffer, receive_buffer
                    for port in model_ports:
                        generation = plan_generation[port]
                        send_buffer = send_control(
                            connection,
                            ControlMessage(
                                message_type=MessageType.PLAN_REQUEST,
                                worker_id=worker_id,
                                task_id=arena_slot_of[port],
                                sequence=first_sequence,
                                auxiliary_sequence=generation,
                                count=count,
                                plan_slot=generation & 1,
                                port_or_slot=port,
                            ),
                            send_buffer,
                        )
                    plans: dict[int, np.ndarray] = {}
                    for _ in model_ports:
                        reply, receive_buffer = receive_control(connection, receive_buffer)
                        if reply.message_type is not MessageType.PLAN_READY:
                            raise RuntimeError(
                                f"worker {worker_id} expected PLAN_READY, got {reply.message_type.name}"
                            )
                        port = reply.port_or_slot
                        if port not in arena_slot_of or reply.task_id != arena_slot_of[port]:
                            raise RuntimeError(
                                f"worker {worker_id} got a plan for unknown port {port} in row {reply.task_id}"
                            )
                        generation = plan_generation[port]
                        plan_slot = generation & 1
                        if reply.auxiliary_sequence != generation or reply.plan_slot != plan_slot:
                            raise RuntimeError(
                                f"worker {worker_id} port {port} got stale plan generation "
                                f"{reply.auxiliary_sequence} in slot {reply.plan_slot}; "
                                f"expected {generation} in {plan_slot}"
                            )
                        plans[port] = arena.plan_actions[arena_slot_of[port], plan_slot]
                        plan_generation[port] += 1
                    return plans

                neutral_actions = {port: neutral for port in model_ports}
                publish(frame, neutral_actions, reset=True)
                plans = request_plans(sequence, 1)
                captured = 1
                done = False
                while captured < max_frames and not done:
                    first_unpublished = sequence + 1
                    executed = 0
                    while executed < runtime.execution_stride and captured < max_frames:
                        actions = {port: plans[port][executed] for port in model_ports}
                        frame, in_game = session.step(
                            {port: action_vec_to_controller(action) for port, action in actions.items()}
                        )
                        next_id = _frame_id(frame, frame_id)
                        sequence += 1
                        captured += 1
                        reset = instant_restart and next_id < frame_id
                        if reset:
                            close_segment()
                            segment = [frame]
                        else:
                            segment.append(frame)
                        frame_id = next_id
                        executed += 1
                        if not in_game:
                            # A normal match-end frame can omit live player fields.
                            # Keep it in the trajectory, but never feed it to the
                            # policy or try to flatten it as an observation.
                            close_segment()
                            done = True
                            break
                        publish(frame, neutral_actions if reset else actions, reset=reset)
                        if reset:
                            plans = request_plans(sequence, 1)
                            break
                    else:
                        if captured < max_frames:
                            plans = request_plans(first_unpublished, executed)
                        continue
                    if done:
                        break
                    # A reset replanned inside the inner loop. Continue from the
                    # new match with a fresh execution stride.
                    if reset:
                        continue
                close_segment()
        finally:
            # These are zero-copy slices of ``arena``. Release the last local
            # aliases before the shared mapping is closed.
            plans = None
            actions = None
            arena.close()
        frame_count = sum(len(trajectory) for trajectory in trajectories)
        result_spec = ResultSpec(frames=frame_count, segments=len(trajectories), ports=len(ports))
        result = ResultArena.create(result_shm_name(arena_descriptor.name, worker_id), result_spec)
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
        send_buffer = send_control(
            connection,
            ControlMessage(
                message_type=MessageType.RESULT_READY,
                worker_id=worker_id,
                auxiliary_sequence=len(trajectories),
                count=frame_count,
            ),
            send_buffer,
        )
        reply, receive_buffer = receive_control(connection, receive_buffer)
        if reply.message_type is not MessageType.RESULT_RELEASED:
            raise RuntimeError(f"worker {worker_id} expected RESULT_RELEASED, got {reply.message_type.name}")
        result.close()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.opt(exception=exc).warning(f"Session worker {worker_id} failed: {error}")
        with suppress(BrokenPipeError, EOFError, OSError):
            send_control(
                connection,
                ControlMessage(message_type=MessageType.ERROR, worker_id=worker_id, status_code=1),
                send_buffer,
            )
    finally:
        connection.close()
