"""Main-process inference broker for spawned Session workers."""

import json
import math
import multiprocessing as mp
import os
import platform
import sys
import time
import traceback
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from multiprocessing.connection import Connection
from multiprocessing.connection import wait
from pathlib import Path
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


class PolicyExecutionError(RuntimeError):
    """A systemic parent-side policy failure; retrying workers cannot recover it."""

    def __init__(self, message: str, *, capsule_path: Path | None = None) -> None:
        super().__init__(message)
        self.capsule_path = capsule_path


def _write_policy_fault_capsule(
    failure_dir: Path | None,
    policy: SharedChunkPolicy,
    requests: Mapping[Slot, Sequence[ObservationRow]],
    *,
    plan_call: int,
) -> Path | None:
    if failure_dir is None:
        return None
    failure_dir.mkdir(parents=True, exist_ok=True)
    stem = failure_dir / f"policy_fault_{time.time_ns()}"
    metadata: dict[str, object] = {}
    arrays: dict[str, np.ndarray] = {}
    snapshot = getattr(policy, "fault_snapshot", None)
    if callable(snapshot):
        try:
            metadata, arrays = snapshot()
        except Exception:
            metadata = {"snapshot_error": traceback.format_exc()}
            arrays = {}
    payload = {
        "schema_version": 1,
        "plan_call": plan_call,
        "pid": os.getpid(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_sha": os.environ.get("HAL_GIT_SHA", "unknown"),
        "exception": traceback.format_exc(),
        "requests": [
            {
                "match": slot.match,
                "port": slot.port,
                "frame_ids": [row.frame_id for row in rows],
                "reset": [row.reset for row in rows],
            }
            for slot, rows in requests.items()
        ],
        "policy": metadata,
        "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()},
    }
    json_tmp = stem.with_suffix(".json.tmp")
    json_path = stem.with_suffix(".json")
    json_tmp.write_text(json.dumps(payload, sort_keys=True, indent=2))
    json_tmp.replace(json_path)
    if arrays:
        npz_tmp = stem.with_suffix(".npz.tmp")
        npz_path = stem.with_suffix(".npz")
        with npz_tmp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        npz_tmp.replace(npz_path)
    return json_path


@dataclass(slots=True)
class ProcessVecTelemetry:
    """Cumulative wall-clock accounting for spawned-driver waves.

    The fields are accumulated by :func:`drive_process_vec`, including across
    retries. Supplying this object is optional, so the normal evaluation API does
    not allocate timing records.
    """

    startup_seconds: float = 0.0
    control_wait_seconds: float = 0.0
    request_read_seconds: float = 0.0
    policy_seconds: float = 0.0
    plan_write_seconds: float = 0.0
    result_read_seconds: float = 0.0
    total_seconds: float = 0.0
    plan_calls: int = 0
    plan_rows: int = 0
    min_plan_rows: int = 0
    max_plan_rows: int = 0
    failed_workers: int = 0
    timed_out_workers: int = 0
    cohort_count: int = 1
    inference_latency_ms: list[float] = field(default_factory=list)
    inference_rows_by_call: list[int] = field(default_factory=list)
    inference_batch_rows_by_call: list[int] = field(default_factory=list)
    inference_rows: int = 0
    inference_batch_rows: int = 0

    def metrics(self) -> dict[str, float]:
        calls = max(self.plan_calls, 1)
        latency = np.asarray(self.inference_latency_ms, dtype=np.float64)
        return {
            "broker_startup_seconds": self.startup_seconds,
            "broker_control_wait_seconds": self.control_wait_seconds,
            "broker_request_read_seconds": self.request_read_seconds,
            "broker_policy_seconds": self.policy_seconds,
            "broker_plan_write_seconds": self.plan_write_seconds,
            "broker_result_read_seconds": self.result_read_seconds,
            "broker_unaccounted_seconds": max(
                0.0,
                self.total_seconds
                - self.control_wait_seconds
                - self.request_read_seconds
                - self.policy_seconds
                - self.plan_write_seconds
                - self.result_read_seconds,
            ),
            "broker_total_seconds": self.total_seconds,
            "broker_plan_calls": float(self.plan_calls),
            "broker_plan_rows": float(self.plan_rows),
            "broker_mean_plan_rows": self.plan_rows / calls,
            "broker_min_plan_rows": float(self.min_plan_rows),
            "broker_max_plan_rows": float(self.max_plan_rows),
            "broker_failed_workers": float(self.failed_workers),
            "broker_timed_out_workers": float(self.timed_out_workers),
            "broker_cohort_count": float(self.cohort_count),
            "broker_inference_latency_p50_ms": float(np.percentile(latency, 50)) if len(latency) else 0.0,
            "broker_inference_latency_p95_ms": float(np.percentile(latency, 95)) if len(latency) else 0.0,
            "broker_inference_rows": float(self.inference_rows),
            "broker_inference_batch_rows": float(self.inference_batch_rows),
            "broker_sustained_inference_rows_per_s": self.inference_batch_rows
            / max(float(latency.sum()) / 1_000, 1e-12),
        }


def _worker_cohort_ids(n_workers: int, cohort_count: int) -> tuple[int, ...]:
    """Balanced, stable worker-to-cohort assignment.

    Every port owned by a Session stays in the same cohort.  Keeping the mapping
    purely index-based makes retries and fixed-seed slot identities independent of
    process timing.
    """
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    if isinstance(cohort_count, bool) or not 1 <= cohort_count <= n_workers:
        raise ValueError(f"cohort_count must be between 1 and {n_workers}, got {cohort_count}")
    return tuple(min(cohort_count - 1, worker * cohort_count // n_workers) for worker in range(n_workers))


def _next_ready_cohort(
    active_slots: set[Slot], pending_slots: set[Slot], cohort_of_worker: Sequence[int]
) -> list[Slot]:
    """Return the first complete cohort, or an empty list if none is ready."""
    active_cohorts = sorted({cohort_of_worker[slot.match] for slot in active_slots})
    for cohort in active_cohorts:
        slots = {slot for slot in active_slots if cohort_of_worker[slot.match] == cohort}
        if slots and pending_slots >= slots:
            return sorted(slots, key=lambda slot: (slot.match, slot.port))
    return []


def _cohort_latency_start_ns(acknowledged: Sequence[tuple[int, int, int, int]]) -> int:
    """Return the earliest preprocessing start in an inferred worker cohort.

    End-to-end batch latency begins when the first worker starts preprocessing,
    not when the last worker reaches the synchronization gate.  Using the
    earliest timestamp therefore includes cohort synchronization in the measured
    deployment-path latency.
    """
    if not acknowledged:
        raise ValueError("latency accounting requires at least one acknowledged plan")
    return min(started for _, _, _, started in acknowledged)


def drive_process_vec(
    session_kwargs: Sequence[dict[str, object]],
    matches: Sequence[VecMatch],
    policy: SharedChunkPolicy,
    *,
    max_frames: int,
    instant_restart: bool = False,
    progress_every: int = 600,
    worker_timeout_seconds: float = 60.0,
    telemetry: ProcessVecTelemetry | None = None,
    failure_dir: Path | None = None,
    cohort_count: int = 1,
) -> list[list[Trajectory]]:
    """Drive one wave with one spawned process per Dolphin Session.

    The parent owns inference and shared memory. Each child owns all Session CPU
    work. The hot IPC path contains fixed 64-byte notifications only. A worker that
    stops producing control messages is named, killed, and reaped after
    ``worker_timeout_seconds`` without aborting healthy workers in the wave.
    """
    drive_started = time.monotonic()
    if len(session_kwargs) != len(matches):
        raise ValueError(f"got {len(session_kwargs)} Session configs for {len(matches)} matches")
    if not math.isfinite(worker_timeout_seconds) or worker_timeout_seconds <= 0:
        raise ValueError(f"worker_timeout_seconds must be finite and positive, got {worker_timeout_seconds}")
    if not matches:
        return []
    cohort_of_worker = _worker_cohort_ids(len(matches), cohort_count)
    if telemetry is not None:
        telemetry.cohort_count = max(telemetry.cohort_count, cohort_count)
    for index, match in enumerate(matches):
        if not match.model_ports:
            raise ValueError(f"spawned driver needs at least one model port; match {index} has none")
    runtime = policy.runtime_spec
    if runtime.action_token_groups:
        raise ValueError("spawned driver does not yet support tokenized controller plans")

    context = mp.get_context("spawn")
    arena_slots_of: list[tuple[int, ...]] = []
    arena_slot = 0
    for match in matches:
        slots = tuple(range(arena_slot, arena_slot + len(match.model_ports)))
        arena_slots_of.append(slots)
        arena_slot += len(slots)
    spec = ArenaSpec(
        workers=arena_slot,
        ring_capacity=runtime.raw_ring_capacity,
        prediction_frames=runtime.prediction_frames,
        action_dim=runtime.action_dim,
        action_token_groups=runtime.action_token_groups,
    )
    out: list[list[Trajectory]] = [[] for _ in matches]
    errors: dict[int, str] = {}
    timed_out_workers = 0
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
                    match.model_ports,
                    arena_slots_of[worker],
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

        active_workers = set(parents)
        active_slots = {Slot(worker, port) for worker, match in enumerate(matches) for port in match.model_ports}
        arena_slot_of = {
            Slot(worker, port): slot_index
            for worker, match in enumerate(matches)
            for port, slot_index in zip(match.model_ports, arena_slots_of[worker], strict=True)
        }
        pending: dict[Slot, ControlMessage] = {}
        pending_acks: dict[tuple[int, int, int], tuple[int | None, int]] = {}
        latency_groups: dict[int, list[int]] = {}
        connection_to_worker = {connection: worker for worker, connection in parents.items()}
        plan_calls = 0
        startup_recorded = False
        started = time.monotonic()
        progress_frames = 0
        progress_time = started

        def retire_worker(worker: int, reason: str | None = None, *, kill: bool = False) -> None:
            """Remove one worker without discarding the rest of the wave."""
            if worker not in active_workers:
                return
            active_workers.remove(worker)
            retired_slots = {slot for slot in active_slots if slot.match == worker}
            active_slots.difference_update(retired_slots)
            for slot in retired_slots:
                pending.pop(slot, None)
            for key in tuple(pending_acks):
                if key[0] == worker:
                    pending_acks.pop(key)
            process = processes[worker]
            if reason is not None:
                errors[worker] = reason
            if kill and process.is_alive():
                process.kill()
                process.join(timeout=2.0)
            else:
                process.join(timeout=0.2)
            parents[worker].close()

        try:
            while active_workers:
                # If a complete cohort remained pending after the previous dispatch,
                # serve it immediately instead of sleeping in wait(). With one cohort
                # (the public default), this is exactly the historical all-slot gate.
                plan_slots = _next_ready_cohort(active_slots, set(pending), cohort_of_worker)
                waited_for_control = not plan_slots
                if waited_for_control:
                    wait_started = time.monotonic()
                    ready = wait(
                        [parents[worker] for worker in active_workers],
                        timeout=worker_timeout_seconds,
                    )
                    if telemetry is not None:
                        telemetry.control_wait_seconds += time.monotonic() - wait_started
                else:
                    ready = []
                for connection in ready:
                    worker = connection_to_worker[connection]
                    if worker not in active_workers:
                        continue
                    try:
                        message, receive_buffers[worker] = receive_control(connection, receive_buffers[worker])
                    except EOFError:
                        process = processes[worker]
                        retire_worker(
                            worker,
                            f"closed its control pipe without a result; pid={process.pid}, "
                            f"exitcode={process.exitcode}",
                            kill=process.is_alive(),
                        )
                        continue
                    if message.worker_id != worker:
                        raise RuntimeError(f"connection {worker} received a message for worker {message.worker_id}")
                    if message.message_type is MessageType.PLAN_REQUEST:
                        slot = Slot(worker, message.port_or_slot)
                        if slot not in active_slots:
                            raise RuntimeError(f"worker {worker} requested an inactive or unknown slot {slot}")
                        if message.task_id != arena_slot_of[slot]:
                            raise RuntimeError(
                                f"worker {worker} published slot {slot} in arena row {message.task_id}; "
                                f"expected {arena_slot_of[slot]}"
                            )
                        if slot in pending:
                            raise RuntimeError(f"slot {slot} sent two plan requests without a reply")
                        pending[slot] = message
                    elif message.message_type is MessageType.PLAN_APPLIED:
                        key = (worker, int(message.auxiliary_sequence), int(message.port_or_slot))
                        if key not in pending_acks:
                            raise RuntimeError(
                                f"worker {worker} acknowledged unknown plan generation {key[1]} port {key[2]}"
                            )
                        latency_group, expected_start = pending_acks.pop(key)
                        if message.task_generation != expected_start:
                            raise RuntimeError(
                                f"worker {worker} changed plan start timestamp "
                                f"{expected_start} -> {message.task_generation}"
                            )
                        if latency_group is not None and telemetry is not None:
                            info = latency_groups[latency_group]
                            info[3] -= 1
                            if info[3] == 0:
                                started_ns, inference_rows, batch_rows, _ = latency_groups.pop(latency_group)
                                telemetry.inference_latency_ms.append(
                                    (time.perf_counter_ns() - started_ns) / 1_000_000
                                )
                                telemetry.inference_rows_by_call.append(inference_rows)
                                telemetry.inference_batch_rows_by_call.append(batch_rows)
                                telemetry.inference_rows += inference_rows
                                telemetry.inference_batch_rows += batch_rows
                    elif message.message_type is MessageType.RESULT_READY:
                        result_started = time.monotonic()
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
                        try:
                            send_buffers[worker] = send_control(
                                parents[worker],
                                ControlMessage(message_type=MessageType.RESULT_RELEASED, worker_id=worker),
                                send_buffers[worker],
                            )
                        except (BrokenPipeError, EOFError, OSError) as exc:
                            logger.warning(
                                f"drive_process_vec: worker {worker} published a valid result but closed "
                                f"before release acknowledgement ({type(exc).__name__}: {exc})"
                            )
                        retire_worker(worker)
                        if telemetry is not None:
                            telemetry.result_read_seconds += time.monotonic() - result_started
                    elif message.message_type is MessageType.ERROR:
                        retire_worker(worker, f"reported error status {message.status_code}")
                    else:
                        raise RuntimeError(
                            f"broker received unexpected {message.message_type.name} from worker {worker}"
                        )

                # A dead worker's pipe may remain open in a descendant. Reap it from
                # process state rather than waiting for EOF on an inherited handle.
                for worker in tuple(active_workers):
                    process = processes[worker]
                    if process.exitcode is not None and not parents[worker].poll():
                        retire_worker(
                            worker,
                            f"exited without a result; pid={process.pid}, exitcode={process.exitcode}",
                        )

                if waited_for_control and not ready and active_workers:
                    blockers = []
                    for worker in sorted(active_workers):
                        worker_slots = {slot for slot in active_slots if slot.match == worker}
                        if not worker_slots or not pending.keys() >= worker_slots:
                            blockers.append(worker)
                    for worker in blockers:
                        process = processes[worker]
                        reason = (
                            f"timed out after {worker_timeout_seconds:.1f}s without a control message; "
                            f"pid={process.pid}, exitcode={process.exitcode}"
                        )
                        logger.warning(f"drive_process_vec: worker {worker} {reason}")
                        retire_worker(worker, reason, kill=True)
                        timed_out_workers += 1

                # One cohort per GPU call. With cohort_count=1 a fast worker waits
                # for every live worker, preserving the original lockstep behavior.
                plan_slots = _next_ready_cohort(active_slots, set(pending), cohort_of_worker)
                if plan_slots:
                    if telemetry is not None and not startup_recorded:
                        telemetry.startup_seconds += time.monotonic() - drive_started
                        startup_recorded = True
                    request_started = time.monotonic()
                    requests: dict[Slot, list[ObservationRow]] = {}
                    for slot in plan_slots:
                        message = pending[slot]
                        shared_slot = arena_slot_of[slot]
                        rows: list[ObservationRow] = []
                        for sequence in range(message.sequence, message.sequence + message.count):
                            flat, action, reset = arena.observation(shared_slot, sequence)
                            rows.append(
                                ObservationRow(
                                    frame_id=int(arena.obs_frame_id[shared_slot, sequence % spec.ring_capacity]),
                                    flat=flat,
                                    action=action,
                                    reset=reset,
                                )
                            )
                        requests[slot] = rows
                    if telemetry is not None:
                        telemetry.request_read_seconds += time.monotonic() - request_started
                    policy_started = time.monotonic()
                    try:
                        plans = policy.plan_rows(requests)
                    except Exception as exc:
                        try:
                            capsule = _write_policy_fault_capsule(
                                failure_dir, policy, requests, plan_call=plan_calls + 1
                            )
                        except Exception as capsule_exc:
                            logger.opt(exception=capsule_exc).error("failed to write policy crash capsule")
                            capsule = None
                        suffix = "" if capsule is None else f"; crash capsule: {capsule}"
                        raise PolicyExecutionError(
                            f"parent policy failed on plan call {plan_calls + 1}{suffix}", capsule_path=capsule
                        ) from exc
                    if telemetry is not None:
                        telemetry.policy_seconds += time.monotonic() - policy_started
                    write_started = time.monotonic()
                    sent_rows = 0
                    acknowledged: list[tuple[int, int, int, int]] = []
                    for slot in plan_slots:
                        worker = slot.match
                        if worker not in active_workers:
                            continue
                        message = pending.pop(slot, None)
                        if message is None:
                            continue
                        shared_slot = arena_slot_of[slot]
                        plan = np.asarray(plans[slot], dtype=np.float32)
                        expected = (runtime.prediction_frames, runtime.action_dim)
                        if plan.shape != expected:
                            raise ValueError(f"policy plan for {slot} has shape {plan.shape}, expected {expected}")
                        arena.plan_actions[shared_slot, message.plan_slot] = plan
                        try:
                            send_buffers[worker] = send_control(
                                parents[worker],
                                ControlMessage(
                                    message_type=MessageType.PLAN_READY,
                                    worker_id=worker,
                                    task_generation=message.task_generation,
                                    task_id=shared_slot,
                                    auxiliary_sequence=message.auxiliary_sequence,
                                    count=runtime.prediction_frames,
                                    plan_slot=message.plan_slot,
                                    port_or_slot=message.port_or_slot,
                                ),
                                send_buffers[worker],
                            )
                            sent_rows += 1
                            acknowledged.append(
                                (
                                    worker,
                                    int(message.auxiliary_sequence),
                                    int(message.port_or_slot),
                                    int(message.task_generation),
                                )
                            )
                        except (BrokenPipeError, EOFError, OSError) as exc:
                            retire_worker(
                                worker,
                                f"closed while receiving a plan ({type(exc).__name__}: {exc})",
                                kill=processes[worker].is_alive(),
                            )
                    if telemetry is not None:
                        telemetry.plan_calls += 1
                        telemetry.plan_rows += sent_rows
                        if sent_rows:
                            if telemetry.min_plan_rows == 0:
                                telemetry.min_plan_rows = sent_rows
                            else:
                                telemetry.min_plan_rows = min(telemetry.min_plan_rows, sent_rows)
                            telemetry.max_plan_rows = max(telemetry.max_plan_rows, sent_rows)
                        telemetry.plan_write_seconds += time.monotonic() - write_started
                    inferred = bool(getattr(policy, "last_plan_inferred", True))
                    inference_rows = int(getattr(policy, "last_inference_rows", sent_rows))
                    batch_rows = int(getattr(policy, "last_inference_batch_rows", inference_rows))
                    latency_group = plan_calls if telemetry is not None and inferred and acknowledged else None
                    if latency_group is not None:
                        latency_groups[latency_group] = [
                            _cohort_latency_start_ns(acknowledged),
                            inference_rows,
                            batch_rows,
                            len(acknowledged),
                        ]
                    for worker, generation, port, started in acknowledged:
                        pending_acks[(worker, generation, port)] = (latency_group, started)
                    plan_calls += 1
                    if progress_every and plan_calls % max(1, progress_every // runtime.execution_stride) == 0:
                        now = time.monotonic()
                        elapsed = now - started
                        frames = plan_calls * runtime.execution_stride
                        interval_fps = (frames - progress_frames) / max(now - progress_time, 1e-12)
                        progress_frames = frames
                        progress_time = now
                        logger.info(
                            f"drive_process_vec: about {frames}/{max_frames} frames | "
                            f"live {len(active_workers)}/{len(matches)} | {interval_fps:.1f} interval fps | "
                            f"{frames / elapsed:.1f} cumulative fps"
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
                if not process.is_alive():
                    process.close()
                discard_result_shm(result_shm_name(arena.descriptor.name, worker))
    if errors:
        logger.warning(f"drive_process_vec: {len(errors)} worker(s) failed: {errors}")
    if telemetry is not None:
        telemetry.failed_workers += len(errors)
        telemetry.timed_out_workers += timed_out_workers
        telemetry.total_seconds += time.monotonic() - drive_started
    return out
