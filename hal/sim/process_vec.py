"""Main-process inference broker for spawned Session workers."""

import multiprocessing as mp
import time
from collections.abc import Mapping
from collections.abc import Sequence
from multiprocessing.connection import Connection
from multiprocessing.connection import wait
from typing import Protocol

import numpy as np
from loguru import logger

from hal.sim.ipc import ArenaSpec
from hal.sim.ipc import ControlMessage
from hal.sim.ipc import MessageType
from hal.sim.ipc import ResultArena
from hal.sim.ipc import ResultSpec
from hal.sim.ipc import RolloutArena
from hal.sim.ipc import discard_result_shm
from hal.sim.ipc import receive_control
from hal.sim.ipc import result_shm_name
from hal.sim.ipc import send_control
from hal.sim.rollout import ObservationRow
from hal.sim.rollout import PolicyRuntimeSpec
from hal.sim.trajectory import Trajectory
from hal.sim.vec import Slot
from hal.sim.vec import VecMatch
from hal.sim.worker import _session_worker
from hal.wire import POST_FIELD_SUFFIXES


class SharedChunkPolicy(Protocol):
    @property
    def runtime_spec(self) -> PolicyRuntimeSpec: ...

    def plan_rows(self, rows: Mapping[Slot, Sequence[ObservationRow]]) -> Mapping[Slot, np.ndarray]: ...


def drive_process_vec(
    session_kwargs: Sequence[dict[str, object]],
    matches: Sequence[VecMatch],
    policy: SharedChunkPolicy,
    *,
    max_frames: int,
    instant_restart: bool = False,
    progress_every: int = 600,
) -> list[list[Trajectory]]:
    """Drive one wave with one spawned process per Dolphin Session.

    The parent owns inference and shared memory. Each child owns all Session CPU
    work. The hot IPC path contains fixed 64-byte notifications only.
    """
    if len(session_kwargs) != len(matches):
        raise ValueError(f"got {len(session_kwargs)} Session configs for {len(matches)} matches")
    if not matches:
        return []
    for index, match in enumerate(matches):
        if len(match.model_ports) != 1:
            raise ValueError(
                f"spawned driver currently needs one model port per match; match {index} has {match.model_ports}"
            )
    runtime = policy.runtime_spec
    if runtime.action_token_groups:
        raise ValueError("spawned driver does not yet support tokenized controller plans")

    context = mp.get_context("spawn")
    spec = ArenaSpec(
        workers=len(matches),
        ring_capacity=runtime.raw_ring_capacity,
        prediction_frames=runtime.prediction_frames,
        action_dim=runtime.action_dim,
        action_token_groups=runtime.action_token_groups,
    )
    out: list[list[Trajectory]] = [[] for _ in matches]
    errors: dict[int, str] = {}
    with RolloutArena.create(spec) as arena:
        parents: dict[int, Connection] = {}
        processes: dict[int, mp.Process] = {}
        receive_buffers = {worker: bytearray(64) for worker in range(len(matches))}
        send_buffers = {worker: bytearray(64) for worker in range(len(matches))}
        for worker, (kwargs, match) in enumerate(zip(session_kwargs, matches, strict=True)):
            parent, child = context.Pipe(duplex=True)
            process = context.Process(
                target=_session_worker,
                args=(
                    worker,
                    child,
                    arena.descriptor,
                    kwargs,
                    match.matchup,
                    match.model_ports[0],
                    runtime,
                    max_frames,
                    instant_restart,
                ),
                name=f"hal-session-{worker}",
            )
            process.start()
            child.close()
            parents[worker] = parent
            processes[worker] = process

        active = set(parents)
        pending: dict[int, ControlMessage] = {}
        connection_to_worker = {connection: worker for worker, connection in parents.items()}
        plan_calls = 0
        started = time.monotonic()
        try:
            while active:
                ready = wait([parents[worker] for worker in active])
                for connection in ready:
                    worker = connection_to_worker[connection]
                    try:
                        message, receive_buffers[worker] = receive_control(connection, receive_buffers[worker])
                    except EOFError as exc:
                        processes[worker].join(timeout=0.2)
                        raise RuntimeError(
                            f"worker {worker} closed its control pipe without a result; "
                            f"exit code is {processes[worker].exitcode}"
                        ) from exc
                    if message.worker_id != worker:
                        raise RuntimeError(f"connection {worker} received a message for worker {message.worker_id}")
                    if message.message_type is MessageType.PLAN_REQUEST:
                        if worker in pending:
                            raise RuntimeError(f"worker {worker} sent two plan requests without a reply")
                        pending[worker] = message
                    elif message.message_type is MessageType.RESULT_READY:
                        ports = tuple(player.port for player in matches[worker].matchup.players)
                        result_spec = ResultSpec(
                            frames=message.count,
                            segments=message.auxiliary_sequence,
                            ports=len(ports),
                        )
                        result = ResultArena.attach(result_shm_name(arena.descriptor.name, worker), result_spec)
                        try:
                            trajectories: list[Trajectory] = []
                            for segment in range(result_spec.segments):
                                start = int(result.segment_start[segment])
                                stop = start + int(result.segment_length[segment])
                                trajectories.append(
                                    Trajectory(
                                        frame_id=np.array(result.frame_id[start:stop]),
                                        random_seed=np.array(result.random_seed[start:stop]),
                                        post={
                                            port: {
                                                field: np.array(result.post[port_index, field_index, start:stop])
                                                for field_index, field in enumerate(POST_FIELD_SUFFIXES)
                                            }
                                            for port_index, port in enumerate(ports)
                                        },
                                    )
                                )
                        finally:
                            result.close()
                        out[worker] = trajectories
                        active.remove(worker)
                        pending.pop(worker, None)
                        send_buffers[worker] = send_control(
                            parents[worker],
                            ControlMessage(message_type=MessageType.RESULT_RELEASED, worker_id=worker),
                            send_buffers[worker],
                        )
                    elif message.message_type is MessageType.ERROR:
                        active.remove(worker)
                        pending.pop(worker, None)
                        errors[worker] = f"worker error status {message.status_code}"
                    else:
                        raise RuntimeError(
                            f"broker received unexpected {message.message_type.name} from worker {worker}"
                        )

                # Preserve one GPU call per lockstep boundary. A fast worker waits
                # in recv while the slowest active worker completes its stride.
                if pending and pending.keys() >= active:
                    plan_workers = sorted(active)
                    requests: dict[Slot, list[ObservationRow]] = {}
                    for worker in plan_workers:
                        message = pending[worker]
                        slot = Slot(worker, message.port_or_slot)
                        rows: list[ObservationRow] = []
                        for sequence in range(message.sequence, message.sequence + message.count):
                            flat, action, reset = arena.observation(worker, sequence)
                            rows.append(
                                ObservationRow(
                                    frame_id=int(arena.obs_frame_id[worker, sequence % spec.ring_capacity]),
                                    flat=flat,
                                    action=action,
                                    reset=reset,
                                )
                            )
                        requests[slot] = rows
                    plans = policy.plan_rows(requests)
                    for worker in plan_workers:
                        message = pending.pop(worker)
                        slot = Slot(worker, message.port_or_slot)
                        plan = np.asarray(plans[slot], dtype=np.float32)
                        expected = (runtime.prediction_frames, runtime.action_dim)
                        if plan.shape != expected:
                            raise ValueError(f"policy plan for {slot} has shape {plan.shape}, expected {expected}")
                        arena.plan_actions[worker, message.plan_slot] = plan
                        send_buffers[worker] = send_control(
                            parents[worker],
                            ControlMessage(
                                message_type=MessageType.PLAN_READY,
                                worker_id=worker,
                                auxiliary_sequence=message.auxiliary_sequence,
                                count=runtime.prediction_frames,
                                plan_slot=message.plan_slot,
                                port_or_slot=message.port_or_slot,
                            ),
                            send_buffers[worker],
                        )
                    plan_calls += 1
                    if progress_every and plan_calls % max(1, progress_every // runtime.execution_stride) == 0:
                        elapsed = time.monotonic() - started
                        frames = plan_calls * runtime.execution_stride
                        logger.info(
                            f"drive_process_vec: about {frames}/{max_frames} frames | "
                            f"live {len(active)}/{len(matches)} | {frames / elapsed:.1f} lockstep fps"
                        )
        finally:
            for connection in parents.values():
                connection.close()
            for worker, process in processes.items():
                process.join(timeout=2.0)
                if process.is_alive():
                    logger.warning(f"drive_process_vec: force-stopping worker {worker}")
                    process.kill()
                    process.join(timeout=2.0)
                discard_result_shm(result_shm_name(arena.descriptor.name, worker))
    if errors:
        logger.warning(f"drive_process_vec: {len(errors)} worker(s) failed: {errors}")
    return out
