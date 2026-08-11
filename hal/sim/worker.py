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
    model_port: int,
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
                plan_generation = 0

                def publish(current: dict, action: np.ndarray, *, reset: bool) -> None:
                    metadata["stage"] = current.get("stage", metadata["stage"])
                    view = {**current, "_matchup": metadata}
                    arena.write_observation(
                        worker_id,
                        sequence,
                        int(current["id"]),
                        flatten_canonical_frame(view),
                        action,
                        reset=reset,
                    )

                def request_plan(first_sequence: int, count: int) -> np.ndarray:
                    nonlocal plan_generation, send_buffer, receive_buffer
                    plan_slot = plan_generation & 1
                    send_buffer = send_control(
                        connection,
                        ControlMessage(
                            message_type=MessageType.PLAN_REQUEST,
                            worker_id=worker_id,
                            sequence=first_sequence,
                            auxiliary_sequence=plan_generation,
                            count=count,
                            plan_slot=plan_slot,
                            port_or_slot=model_port,
                        ),
                        send_buffer,
                    )
                    reply, receive_buffer = receive_control(connection, receive_buffer)
                    if reply.message_type is not MessageType.PLAN_READY:
                        raise RuntimeError(f"worker {worker_id} expected PLAN_READY, got {reply.message_type.name}")
                    if reply.auxiliary_sequence != plan_generation or reply.plan_slot != plan_slot:
                        raise RuntimeError(
                            f"worker {worker_id} got stale plan generation {reply.auxiliary_sequence} "
                            f"in slot {reply.plan_slot}; expected {plan_generation} in {plan_slot}"
                        )
                    plan_generation += 1
                    return arena.plan_actions[worker_id, plan_slot]

                publish(frame, neutral, reset=True)
                plan = request_plan(sequence, 1)
                captured = 1
                done = False
                while captured < max_frames and not done:
                    first_unpublished = sequence + 1
                    executed = 0
                    while executed < runtime.execution_stride and captured < max_frames:
                        action = plan[executed]
                        frame, in_game = session.step({model_port: action_vec_to_controller(action)})
                        next_id = _frame_id(frame, frame_id)
                        sequence += 1
                        captured += 1
                        reset = instant_restart and next_id < frame_id
                        if reset:
                            close_segment()
                            segment = [frame]
                        else:
                            segment.append(frame)
                        publish(frame, neutral if reset else action, reset=reset)
                        frame_id = next_id
                        executed += 1
                        if not in_game:
                            close_segment()
                            done = True
                            break
                        if reset:
                            plan = request_plan(sequence, 1)
                            break
                    else:
                        if captured < max_frames:
                            plan = request_plan(first_unpublished, executed)
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
            plan = None
            action = None
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
        logger.warning(f"Session worker {worker_id} failed: {error}")
        with suppress(BrokenPipeError, EOFError, OSError):
            send_control(
                connection,
                ControlMessage(message_type=MessageType.ERROR, worker_id=worker_id, status_code=1),
                send_buffer,
            )
    finally:
        connection.close()
