"""Long-running netplay reservation workers and inference engine."""

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import platform
import secrets
import signal
import socket
import sys
import threading
import time
from collections.abc import Sequence
from contextlib import ExitStack
from contextlib import closing
from contextlib import suppress
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from datetime import UTC
from datetime import datetime
from functools import partial
from multiprocessing.connection import Connection
from multiprocessing.context import SpawnContext
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Protocol

import melee
import torch
from loguru import logger
from peppi_py.game import EndMethod
from prometheus_client import Gauge
from prometheus_client import start_http_server
from websockets.exceptions import ConnectionClosed

from hal import r2
from hal.eval.match_summary import summarize_trajectory
from hal.eval.netplay import DolphinConnectionLost
from hal.eval.netplay import NoUsableActionPlan
from hal.eval.netplay import run_netplay_match
from hal.eval.qualification import RealtimeBudgetCheck
from hal.eval.qualification import check_realtime_budget
from hal.eval.replays import read_new_replay_end
from hal.eval.results import NetplayProgress
from hal.eval.results import PlayResult
from hal.eval.scheduling import ActionScheduler
from hal.eval.scheduling import FrameTiming
from hal.fixtures import ISO
from hal.fixtures import NETPLAY_EMULATOR
from hal.fixtures import ensure
from hal.inference.action_sequence_artifact import read_action_sequence_artifact
from hal.inference.action_sequence_policy import ActionSequencePolicy
from hal.inference.api import PolicySpec
from hal.inference.api import PreparedInferenceProfile
from hal.inference.api import RuntimeConfig
from hal.inference.client import InferenceClient
from hal.inference.client import InferenceUnavailable
from hal.inference.client import StreamInvalidate
from hal.inference.client import StreamInvalidated
from hal.inference.cuda_graph import CaptureCounter
from hal.inference.cuda_graph import CompilationStartCounter
from hal.inference.cuda_graph import count_compilation_starts
from hal.inference.engine import InferenceEngine
from hal.inference.engine import ModelRegistry
from hal.inference.engine import configure_inference_process
from hal.inference.engine import freeze_inference_runtime
from hal.netplay_service.assets import AssetCache
from hal.netplay_service.assets import LocalSource
from hal.netplay_service.assets import PinnedAsset
from hal.netplay_service.assets import R2Source
from hal.netplay_service.domain import IDLE_TIMEOUT_SECONDS
from hal.netplay_service.domain import TERMINAL_STATUSES
from hal.netplay_service.domain import Job
from hal.netplay_service.domain import JobStatus
from hal.netplay_service.domain import validate_player_code
from hal.netplay_service.health import FRAME_STALL_SECONDS
from hal.netplay_service.health import SLOT_HEARTBEAT_MAX_AGE_SECONDS
from hal.netplay_service.health import SLOT_STARTUP_GRACE_SECONDS
from hal.netplay_service.health import ChunkHealth
from hal.netplay_service.health import RunnerState
from hal.netplay_service.health import RunnerStatus
from hal.netplay_service.health import RuntimeHealth
from hal.netplay_service.health import RuntimeSnapshot
from hal.netplay_service.health import SlotState
from hal.netplay_service.health import SlotStatus
from hal.netplay_service.health import aggregate_runner_status
from hal.netplay_service.health import read_runner_status
from hal.netplay_service.health import read_slot_status
from hal.netplay_service.health import write_runner_status
from hal.netplay_service.health import write_slot_status
from hal.netplay_service.queue_client import QueueEndpoint
from hal.netplay_service.queue_client import QueueProtocolError
from hal.netplay_service.queue_client import QueueUnavailableError
from hal.netplay_service.queue_client import RemoteQueue
from hal.netplay_service.queue_client import RunnerClient
from hal.netplay_service.queue_client import SessionReporter
from hal.netplay_service.queue_client import StartedSession
from hal.netplay_service.queue_client import new_session_id
from hal.netplay_service.queue_client import runner_endpoint
from hal.netplay_service.queue_client import slot_worker_id
from hal.netplay_service.queue_contract import InvalidTransitionError
from hal.netplay_service.queue_contract import RunnerQueue
from hal.netplay_service.queue_contract import SessionEndedError
from hal.netplay_service.replays import ReplayMetadata
from hal.netplay_service.replays import upload_replay
from hal.netplay_service.stream import DisplayGroup
from hal.netplay_service.stream import GameStreamState
from hal.netplay_service.stream import IdleStreamState
from hal.netplay_service.stream import PulseAudio
from hal.netplay_service.stream import StreamSupervisor
from hal.netplay_service.stream import write_stream_state
from hal.sim.netplay import ConnectAbandoned
from hal.sim.netplay import NetplaySession
from hal.sim.netplay import NetplaySetup
from hal.sim.session import DolphinGraphicsBackend
from hal.sim.session import FrameTimeout

_PENDING_UPLOAD_RETRY_SECONDS = 60.0
_SLOT_STATUS_INTERVAL_SECONDS = 1.0
_SLOT_RESTART_LEASE_GRACE_SECONDS = 60.0
_SESSION_HEARTBEAT = Gauge(
    "hal_netplay_session_heartbeat_unixtime",
    "Unix time when the runner prepared its latest session status report.",
)
_HEALTHY_SLOTS = Gauge("hal_netplay_healthy_slots", "Number of healthy netplay slots.")


class _StopEvent(Protocol):
    def is_set(self) -> bool: ...

    def wait(self, timeout: float | None = None) -> bool: ...

    def set(self) -> None: ...


class _PipeContext(Protocol):
    def Pipe(self, duplex: bool = True) -> tuple[Connection, Connection]: ...


@dataclass(frozen=True, slots=True)
class SlotConfig:
    """Cold configuration for one long-running Dolphin worker."""

    slot: int
    worker_id: str
    stream_id: int
    queue_endpoint: QueueEndpoint
    session_id: str
    user_json: Path
    bot_connect_code: str
    slippi_port: int
    iso_path: Path
    dolphin_path: Path
    replay_dir: Path
    status_path: Path
    policy_sha256: str
    checkpoint_sha256: str
    git_sha: str
    graphics_backend: DolphinGraphicsBackend = "Vulkan"
    max_frames: int = 54_000
    recovery_cooldown_seconds: float = 2.0
    measurement_dir: Path | None = None
    publish_replays: bool = True
    display: str | None = None
    stream_output: bool = False
    stream_state_path: Path | None = None
    pulse_environment: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.graphics_backend not in ("Vulkan", "OGL"):
            raise ValueError(f"unsupported Dolphin graphics backend {self.graphics_backend!r}")
        if (
            type(self.recovery_cooldown_seconds) not in (int, float)
            or not math.isfinite(self.recovery_cooldown_seconds)
            or self.recovery_cooldown_seconds < 0
        ):
            raise ValueError("slot recovery cooldown must be non-negative")
        if type(self.publish_replays) is not bool:
            raise ValueError("slot replay publication must be a boolean")
        if type(self.stream_output) is not bool:
            raise ValueError("slot stream output must be a boolean")
        if self.stream_output and (self.slot != 0 or self.stream_state_path is None):
            raise ValueError("only slot 0 can have stream output and it needs a state path")


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    """Validated deployment configuration for one GPU runner."""

    queue_endpoint: QueueEndpoint
    session_id: str
    policy: Path
    user_jsons: tuple[Path, ...]
    slippi_ports: tuple[int, ...]
    iso_path: Path
    dolphin_path: Path
    replay_dir: Path
    status_path: Path
    git_sha: str
    graphics_backend: DolphinGraphicsBackend = "Vulkan"
    device: str = "cuda"
    seed: int | None = None
    compiled: bool = False
    batch_wait_seconds: float = 0.0005
    preparation_timeout_seconds: float = 1800.0
    max_frames: int = 54_000
    measurement_dir: Path | None = None
    publish_replays: bool = True
    display_base: int | None = None
    wants_stream: bool = False
    stream_display: str | None = None
    twitch_bandwidth_test: bool = False

    def __post_init__(self) -> None:
        if self.graphics_backend not in ("Vulkan", "OGL"):
            raise ValueError(f"unsupported Dolphin graphics backend {self.graphics_backend!r}")
        if not self.user_jsons:
            raise ValueError("runner needs at least one Slippi account")
        if len(self.user_jsons) != len(self.slippi_ports):
            raise ValueError("runner needs one Slippi port per account slot")
        if len(set(self.slippi_ports)) != len(self.slippi_ports):
            raise ValueError("runner Slippi ports must be unique")
        if any(type(port) is not int or not 1 <= port <= 65_535 for port in self.slippi_ports):
            raise ValueError("runner Slippi ports must be in [1, 65535]")
        if not self.git_sha or "/" in self.git_sha:
            raise ValueError("runner git_sha must be a non-empty identifier")
        if (
            isinstance(self.batch_wait_seconds, bool)
            or not isinstance(self.batch_wait_seconds, (float, int))
            or not math.isfinite(self.batch_wait_seconds)
            or not 0 <= self.batch_wait_seconds <= 0.0005
        ):
            raise ValueError("netplay batch coalescing must be in [0, 0.5] ms")
        if type(self.max_frames) is not int or self.max_frames < 6:
            raise ValueError("runner max_frames must be at least 6")
        if (
            type(self.preparation_timeout_seconds) not in (float, int)
            or not math.isfinite(self.preparation_timeout_seconds)
            or self.preparation_timeout_seconds <= 0
        ):
            raise ValueError("runner preparation timeout must be finite and positive")
        if type(self.publish_replays) is not bool:
            raise ValueError("runner replay publication must be a boolean")
        if self.display_base is not None and (
            type(self.display_base) is not int
            or self.display_base < 0
            or self.display_base + len(self.user_jsons) > 65_535
        ):
            raise ValueError("runner X display range is invalid")
        if type(self.wants_stream) is not bool or type(self.twitch_bandwidth_test) is not bool:
            raise ValueError("runner stream flags must be boolean")
        if self.wants_stream and self.display_base is None:
            raise ValueError("streaming requires a managed X display")
        if self.wants_stream and not self.stream_display:
            raise ValueError("streaming requires --stream-display with a dedicated NVIDIA Xorg display")


_NETPLAY_TIMINGS = (
    FrameTiming(2, 1, 3, 4, 8),
    FrameTiming(3, 1, 4, 4, 8),
)
_ENGINE_RECOVERY_TIMEOUT_SECONDS = 120.0
_ENGINE_PROGRESS_TIMEOUT_SECONDS = 1.0
_SLOT_RESET_TIMEOUT_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class _InferenceProcessConfig:
    policy: Path
    device: str
    seed: int | None
    compiled: bool
    capacity: int
    batch_wait_seconds: float
    generation_id: str
    source_git_sha: str
    bundle_sha256: str
    measurement_dir: Path | None
    standby_for: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class _EngineReady:
    spec: PolicySpec
    context_frames: int
    checkpoint_sha256: str
    capability_version: int
    budgets: tuple[RealtimeBudgetCheck, ...]
    sampling_seeds: tuple[int, ...]
    hardware: str
    profiles: tuple[PreparedInferenceProfile, ...]


@dataclass(frozen=True, slots=True)
class _PreparedNetplayEngine:
    engine: InferenceEngine
    ready: _EngineReady
    capture_counters: tuple[tuple[PreparedInferenceProfile, CaptureCounter], ...]


@dataclass(frozen=True, slots=True)
class _EnginePulse:
    model_inference_p95_ms: float | None
    batch_wait_p95_ms: float | None


@dataclass(frozen=True, slots=True)
class _EngineFailure:
    reason: str


class _EngineLost(RuntimeError):
    pass


class _ShutdownRequested(RuntimeError):
    pass


class _ShutdownFlag:
    def __init__(self) -> None:
        self._signals = 0

    @property
    def requested(self) -> bool:
        return self._signals > 0

    @property
    def forced(self) -> bool:
        return self._signals > 1

    def __call__(self, _signum: int, _frame: object) -> None:
        self._signals += 1


def _prepare_netplay_engine(
    config: _InferenceProcessConfig,
    connections: dict[int, Connection],
) -> _PreparedNetplayEngine:
    artifact = read_action_sequence_artifact(config.policy)
    if artifact.capability_version < 2 or not {2, 3}.issubset(artifact.spec.supported_transport_delays):
        raise ValueError("netplay delay-2 and delay-3 profiles require a qualified 059 capability-v2 bundle")
    inference_dtype = torch.bfloat16 if torch.device(config.device).type == "cuda" else torch.float32
    registry = ModelRegistry()
    model = registry.register_artifact(artifact, device=config.device, inference_dtype=inference_dtype)
    device = next(model.parameters()).device
    policies: dict[PreparedInferenceProfile, ActionSequencePolicy] = {}
    budgets: list[RealtimeBudgetCheck] = []
    seeds: list[int] = []
    for timing in _NETPLAY_TIMINGS:
        runtime = RuntimeConfig(config.capacity, (timing.physical_delay_frames,), timing.replan_interval_frames)
        policy = ActionSequencePolicy(
            model,
            artifact.statistics,
            artifact.vocabulary.codes,
            spec=artifact.spec,
            checkpoint_sha256=artifact.checkpoint_sha256,
            return_p90=artifact.return_p90,
            capability_version=artifact.capability_version,
            device=device,
            seed=config.seed,
            compiled=config.compiled,
            history_mode="kv_cache",
            kv_update_frames=4,
        )
        budget = check_realtime_budget(
            policy,
            runtime,
            config.batch_wait_seconds,
            shape=(timing.prediction_horizon_frames, timing.fixed_prefix_frames),
        )
        if budget.timings != (timing,):
            raise RuntimeError("netplay timing qualification changed the declared profile")
        profile = PreparedInferenceProfile(
            name=f"netplay-delay-{timing.physical_delay_frames}",
            checkpoint_sha256=artifact.checkpoint_sha256,
            execution_mode="kv_cache",
            prediction_horizon_frames=timing.prediction_horizon_frames,
            fixed_prefix_frames=timing.fixed_prefix_frames,
            update_shapes=(1, 2, 4),
            capacity=config.capacity,
        )
        policies[profile] = policy
        budgets.append(budget)
        seeds.append(policy.sampling_seed)
    if registry.model_count != 1:
        raise RuntimeError("netplay profiles allocated more than one checkpoint model")
    freeze_inference_runtime()
    hardware = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
    engine = InferenceEngine(
        policies,
        connections,
        batch_wait_seconds=config.batch_wait_seconds,
        standby_for=dict(config.standby_for),
    )
    ready = _EngineReady(
        artifact.spec,
        model.cfg.L_ctx,
        artifact.checkpoint_sha256,
        artifact.capability_version,
        tuple(budgets),
        tuple(seeds),
        hardware,
        tuple(policies),
    )
    return _PreparedNetplayEngine(
        engine,
        ready,
        tuple((profile, policy.capture_counter) for profile, policy in policies.items()),
    )


def _write_engine_audit_record(directory: Path, generation_id: str, phase: str, payload: dict[str, object]) -> None:
    audit_directory = directory / "engine-audits"
    audit_directory.mkdir(parents=True, exist_ok=True)
    path = audit_directory / f"{generation_id}-{phase}.json"
    pending = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with pending.open("x") as output:
            output.write(json.dumps(payload, allow_nan=False, sort_keys=True))
        os.link(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def _engine_audit_snapshot(
    config: _InferenceProcessConfig,
    prepared: _PreparedNetplayEngine,
    compiler: CompilationStartCounter,
    phase: str,
    outcome: str,
) -> dict[str, object]:
    captures = tuple((profile, counter.snapshot()) for profile, counter in prepared.capture_counters)
    return {
        "schema_version": 1,
        "phase": phase,
        "engine_generation_id": config.generation_id,
        "source_git_sha": config.source_git_sha,
        "policy_bundle_sha256": config.bundle_sha256,
        "checkpoint_sha256": prepared.ready.checkpoint_sha256,
        "capability_version": prepared.ready.capability_version,
        "compiled": config.compiled,
        "outcome": outcome,
        "compilation_starts": compiler.snapshot(),
        "profiles": [
            {
                "profile": asdict(profile),
                "capture_attempts": counts.attempts,
                "capture_completed": counts.completed,
            }
            for profile, counts in captures
        ],
    }


class _EnginePulseSender:
    def __init__(self, connection: Connection, engine: InferenceEngine) -> None:
        self.connection = connection
        self.engine = engine
        self.last_sent = 0.0

    def __call__(self) -> None:
        now = time.monotonic()
        if now - self.last_sent < 0.2:
            return
        self.connection.send(_EnginePulse(*self.engine.timing_p95_ms()))
        self.last_sent = now


def _gpu_inference_process(
    config: _InferenceProcessConfig,
    connections: dict[int, Connection],
    status: Connection,
    stop: _StopEvent,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        configure_inference_process()
        prepared = _prepare_netplay_engine(config, connections)
        if config.measurement_dir is None:
            status.send(prepared.ready)
            prepared.engine.serve(stop, _EnginePulseSender(status, prepared.engine))
        else:
            with count_compilation_starts() as compiler:
                ready_record = _engine_audit_snapshot(config, prepared, compiler, "ready", "serving")
                _write_engine_audit_record(config.measurement_dir, config.generation_id, "ready", ready_record)
                outcome = "stopped"
                try:
                    status.send(prepared.ready)
                    prepared.engine.serve(stop, _EnginePulseSender(status, prepared.engine))
                except BaseException:
                    outcome = "error"
                    raise
                finally:
                    final_record = _engine_audit_snapshot(config, prepared, compiler, "final", outcome)
                    _write_engine_audit_record(config.measurement_dir, config.generation_id, "final", final_record)
    except BaseException as error:
        with suppress(OSError, EOFError):
            status.send(_EngineFailure(f"{type(error).__name__}: {error}"))
        raise
    finally:
        status.close()
        for connection in connections.values():
            connection.close()


class _RecoverableRuntimeError(RuntimeError):
    pass


class _SlotHealthReporter:
    """Publish one slot heartbeat without file I/O in the frame loop."""

    def __init__(self, slot: int, path: Path) -> None:
        self._slot = slot
        self._path = path
        self._monitor = RuntimeHealth()
        self._state = SlotState.STARTING
        self._reason: str | None = None
        self._delay: int | None = None
        self._recoveries = 0
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._stop = threading.Event()
        self._publisher_error: BaseException | None = None
        self._publisher = threading.Thread(
            target=self._publish_loop,
            name=f"hal-netplay-slot-{slot}-health",
            daemon=True,
        )

    def __enter__(self) -> _SlotHealthReporter:
        self._publish()
        self._publisher.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._publisher.join(timeout=2.0)

    def idle(self) -> None:
        with self._lock:
            self._raise_if_publisher_failed()
            if self._state is SlotState.IDLE:
                return
            self._monitor.finish()
            self._state = SlotState.IDLE
            self._reason = None
            self._delay = None
        self._publish()

    def connecting(self, delay: int) -> None:
        with self._lock:
            self._raise_if_publisher_failed()
            self._monitor.finish()
            self._state = SlotState.CONNECTING
            self._reason = None
            self._delay = delay
        self._publish()

    def playing(self) -> None:
        with self._lock:
            self._raise_if_publisher_failed()
            if self._delay is None:
                raise RuntimeError("slot health has no online delay")
            self._monitor.begin(self._delay, time.monotonic())
            self._state = SlotState.PLAYING
            self._reason = None
        self._publish()

    def observe_frame(self, frame_id: int, dolphin_step_seconds: float) -> None:
        with self._lock:
            self._raise_if_publisher_failed()
            snapshot = self._monitor.observe_frame(frame_id, dolphin_step_seconds, time.monotonic())
        self._raise_if_recovery_required(snapshot)

    def configure_schedule(self, schedule: FrameTiming) -> None:
        with self._lock:
            self._monitor.chunk_health = ChunkHealth(schedule)

    def observe_schedule(self, schedule: ActionScheduler) -> None:
        with self._lock:
            self._monitor.observe_chunks(
                ChunkHealth(
                    schedule.timing,
                    schedule.deadline_misses,
                    schedule.prefix_mismatches,
                    schedule.exhausted_chunks,
                    schedule.neutral_fallback_frames,
                    schedule.submission_gaps,
                    schedule.transport_corrections,
                ),
                time.monotonic(),
            )

    def observe_policy(self, seconds: float) -> None:
        with self._lock:
            self._raise_if_publisher_failed()
            snapshot = self._monitor.observe_policy(seconds, time.monotonic())
        self._raise_if_recovery_required(snapshot)

    def recovering(self, reason: str) -> None:
        with self._lock:
            self._raise_if_publisher_failed()
            self._monitor.finish()
            self._state = SlotState.RECOVERING
            self._reason = reason
            self._recoveries += 1
        self._publish()

    def status(self) -> SlotStatus:
        with self._lock:
            self._raise_if_publisher_failed()
            snapshot = self._monitor.snapshot(time.monotonic())
            state = self._state
            reason = self._reason
            if state is SlotState.PLAYING and snapshot.reason is not None:
                state = SlotState.RECOVERING if snapshot.recovery_required else SlotState.DEGRADED
                reason = snapshot.reason
            return SlotStatus(
                slot=self._slot,
                state=state,
                game_fps=snapshot.game_fps,
                frame_interval_p95_ms=snapshot.frame_interval_p95_ms,
                dolphin_step_p95_ms=snapshot.dolphin_step_p95_ms,
                policy_round_trip_p95_ms=snapshot.policy_round_trip_p95_ms,
                reason=reason,
                recoveries=self._recoveries,
                updated_at=time.time(),
                chunk_health=self._monitor.chunk_health,
            )

    @staticmethod
    def _raise_if_recovery_required(snapshot: RuntimeSnapshot) -> None:
        if snapshot.recovery_required:
            raise _RecoverableRuntimeError(snapshot.reason or "runtime_degraded")

    def _raise_if_publisher_failed(self) -> None:
        if self._publisher_error is not None:
            raise RuntimeError("slot health publisher failed") from self._publisher_error

    def _publish(self) -> None:
        with self._write_lock:
            write_slot_status(self._path, self.status())

    def _publish_loop(self) -> None:
        try:
            while not self._stop.wait(_SLOT_STATUS_INTERVAL_SECONDS):
                self._publish()
        except BaseException as error:
            with self._lock:
                self._publisher_error = error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _slot_status_path(path: Path, slot: int) -> Path:
    return path.with_name(f"{path.name}.slot-{slot}.json")


def _unavailable_slot_status(slot: int, now: float, *, starting: bool) -> SlotStatus:
    return SlotStatus(
        slot=slot,
        state=SlotState.STARTING if starting else SlotState.RECOVERING,
        game_fps=None,
        frame_interval_p95_ms=None,
        dolphin_step_p95_ms=None,
        policy_round_trip_p95_ms=None,
        reason="slot_starting" if starting else "slot_heartbeat_missing",
        recoveries=0,
        updated_at=now,
    )


def _load_slot_statuses(paths: tuple[Path, ...], *, now: float, started_at: float) -> tuple[SlotStatus, ...]:
    statuses: list[SlotStatus] = []
    starting = now - started_at <= SLOT_STARTUP_GRACE_SECONDS
    for slot, path in enumerate(paths):
        try:
            status = read_slot_status(path)
        except ValueError:
            statuses.append(_unavailable_slot_status(slot, now, starting=starting))
            continue
        age = now - status.updated_at
        if status.slot != slot or not -SLOT_HEARTBEAT_MAX_AGE_SECONDS <= age <= SLOT_HEARTBEAT_MAX_AGE_SECONDS:
            statuses.append(_unavailable_slot_status(slot, now, starting=False))
            continue
        statuses.append(status)
    return tuple(statuses)


def _write_status(
    path: Path,
    *,
    policy_sha256: str,
    slot_paths: tuple[Path, ...],
    started_at: float,
    model_inference_p95_ms: float | None,
    batch_wait_p95_ms: float | None,
) -> RunnerStatus:
    now = time.time()
    slots = _load_slot_statuses(slot_paths, now=now, started_at=started_at)
    status = aggregate_runner_status(
        policy_sha256,
        slots,
        now,
        model_inference_p95_ms=model_inference_p95_ms,
        batch_wait_p95_ms=batch_wait_p95_ms,
    )
    write_runner_status(path, status)
    return status


def _bot_connect_code(path: Path) -> str:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read Slippi account JSON {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Slippi account JSON {path} must contain an object")
    value = payload.get("connectCode")
    if not isinstance(value, str):
        raise ValueError(f"Slippi account JSON {path} has no connectCode")
    return validate_player_code(value)


def _bot_connect_codes(paths: Sequence[Path]) -> tuple[str, ...]:
    codes = tuple(_bot_connect_code(path) for path in paths)
    if len(set(codes)) != len(codes):
        raise ValueError("runner Slippi accounts must have distinct connect codes")
    return codes


def _human_result(result: PlayResult) -> str:
    summary = summarize_trajectory(result.trajectory)
    bot_stocks = summary.p1_stocks_left if result.ego_port == 1 else summary.p2_stocks_left
    human_stocks = summary.p2_stocks_left if result.ego_port == 1 else summary.p1_stocks_left
    if human_stocks > bot_stocks:
        return "win"
    if human_stocks < bot_stocks:
        return "loss"
    return "tie"


def _stage_name(stage_id: int) -> str:
    try:
        stage = melee.Stage(stage_id)
    except ValueError as error:
        raise RuntimeError(f"netplay returned unknown stage {stage_id}") from error
    if stage in (melee.Stage.NO_STAGE, melee.Stage.RANDOM_STAGE):
        raise RuntimeError(f"netplay returned non-playable stage {stage.name}")
    return stage.name


def _setup(job: Job, *, rematch: bool) -> NetplaySetup:
    try:
        character = melee.Character[job.choices.character]
        if rematch:
            if job.choices.requested_stage is None:
                raise ValueError(f"rematch job {job.id} has no requested stage")
            stage = melee.Stage[job.choices.requested_stage]
        else:
            stage = melee.Stage.RANDOM_STAGE
    except KeyError as error:
        raise ValueError(f"job {job.id} contains an unknown Melee selection") from error
    return NetplaySetup(character, job.player_code, stage=stage)


def _upload_sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".upload.json")


def _write_pending_upload(path: Path, metadata: ReplayMetadata, worker_id: str) -> Path:
    payload = asdict(metadata)
    payload["started_at"] = metadata.started_at.astimezone(UTC).isoformat()
    payload["ended_at"] = metadata.ended_at.astimezone(UTC).isoformat()
    # The queue accepts a replay only from the worker that played the game.
    payload["worker_id"] = worker_id
    payload["schema_version"] = 2
    sidecar = _upload_sidecar(path)
    temporary = sidecar.with_suffix(sidecar.suffix + ".partial")
    temporary.write_text(json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True))
    temporary.replace(sidecar)
    return sidecar


def _read_pending_upload(sidecar: Path) -> tuple[Path, ReplayMetadata, str]:
    try:
        payload = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read pending replay metadata {sidecar}") from error
    expected = {
        "ended_at",
        "game_number",
        "git_sha",
        "player_code",
        "policy_sha256",
        "reservation_id",
        "result",
        "schema_version",
        "started_at",
        "actual_stage",
        "worker_id",
    }
    if not isinstance(payload, dict) or set(payload) != expected or payload.get("schema_version") != 2:
        raise RuntimeError(f"pending replay metadata has the wrong schema: {sidecar}")
    replay = Path(str(sidecar).removesuffix(".upload.json"))
    try:
        metadata = ReplayMetadata(
            reservation_id=payload["reservation_id"],
            player_code=payload["player_code"],
            game_number=payload["game_number"],
            actual_stage=payload["actual_stage"],
            result=payload["result"],
            policy_sha256=payload["policy_sha256"],
            git_sha=payload["git_sha"],
            started_at=datetime.fromisoformat(payload["started_at"]),
            ended_at=datetime.fromisoformat(payload["ended_at"]),
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"pending replay metadata contains invalid values: {sidecar}") from error
    worker_id = payload["worker_id"]
    if not isinstance(worker_id, str) or not worker_id:
        raise RuntimeError(f"pending replay metadata contains invalid values: {sidecar}")
    return replay, metadata, worker_id


def _sidecar_queue(endpoint: QueueEndpoint, worker_id: str) -> RemoteQueue:
    session_id, separator, slot = worker_id.rpartition("/slot-")
    if not separator or not session_id or not slot.isdigit():
        raise RuntimeError(f"pending replay metadata has invalid worker_id {worker_id!r}")
    return RemoteQueue(endpoint, session_id)


def _complete_pending_upload(sidecar: Path, endpoint: QueueEndpoint) -> None:
    replay, metadata, worker_id = _read_pending_upload(sidecar)
    uploaded = upload_replay(replay, metadata)
    with closing(_sidecar_queue(endpoint, worker_id)) as queue:
        queue.record_replay(
            metadata.reservation_id,
            worker_id,
            metadata.game_number,
            key=uploaded.key,
            sha256=uploaded.sha256,
            size=uploaded.size,
            etag=uploaded.etag,
        )
    logger.info(
        "replay uploaded reservation={} game={} key={} size={} etag={}",
        metadata.reservation_id,
        metadata.game_number,
        uploaded.key,
        uploaded.size,
        uploaded.etag,
    )
    replay.unlink()
    sidecar.unlink()


def _drain_pending_uploads(root: Path, endpoint: QueueEndpoint) -> None:
    for sidecar in sorted(root.rglob("*.slp.upload.json")):
        try:
            _complete_pending_upload(sidecar, endpoint)
        except Exception as error:  # Replay failures are isolated from active games.
            logger.warning(
                "pending replay upload failed: {}: {}: {}",
                type(error).__name__,
                error,
                sidecar,
            )


def _retry_pending_uploads(root: Path, endpoint: QueueEndpoint, next_attempt: float) -> float:
    now = time.monotonic()
    if now < next_attempt:
        return next_attempt
    _drain_pending_uploads(root, endpoint)
    return now + _PENDING_UPLOAD_RETRY_SECONDS


def _heartbeat(
    store: RunnerQueue,
    job_id: str,
    worker_id: str,
    done: threading.Event,
    abort: _StopEvent,
) -> None:
    next_heartbeat = time.monotonic() + 5.0
    while not done.wait(0.1):
        if abort.is_set():
            try:
                store.forfeit_service_failure(job_id, worker_id)
            except InvalidTransitionError, SessionEndedError:
                pass
            except QueueUnavailableError as error:
                logger.bind(job=job_id, event="forfeit").warning("job forfeit failed: {}", error)
            return
        if time.monotonic() < next_heartbeat:
            continue
        try:
            store.heartbeat(job_id, worker_id)
        except QueueUnavailableError as error:
            logger.bind(job=job_id, event="heartbeat").warning("job heartbeat failed: {}", error)
            continue
        except InvalidTransitionError:
            return
        next_heartbeat = time.monotonic() + 5.0


@dataclass(frozen=True, slots=True)
class SessionStatus:
    path: Path
    policy_sha256: str
    slots: int

    def __call__(self) -> RunnerStatus:
        try:
            status = read_runner_status(self.path)
        except OSError, ValueError:
            now = time.time()
            statuses = tuple(_unavailable_slot_status(slot, now, starting=True) for slot in range(self.slots))
            status = aggregate_runner_status(
                self.policy_sha256,
                statuses,
                now,
                model_inference_p95_ms=None,
                batch_wait_p95_ms=None,
            )
        _SESSION_HEARTBEAT.set(time.time())
        _HEALTHY_SLOTS.set(status.healthy_slots)
        return status


class _LivePolicySettings:
    """Receive pushed policy settings off the frame thread."""

    def __init__(self, store: RemoteQueue, job: Job, worker_id: str) -> None:
        self._store = store
        self._job_id = job.id
        self._worker_id = worker_id
        self._revision = job.policy_revision
        self._values = (job.choices.desired_return, job.choices.temperature)
        # Set once the queue cancels, expires, or reassigns the job, so a
        # connect wait frees the slot without waiting for its own timeout.
        self.released = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._receive, daemon=True)

    def _apply(self, raw: object) -> bool:
        if not isinstance(raw, str):
            raise QueueProtocolError("live settings message must be text")
        try:
            message = json.loads(raw)
        except json.JSONDecodeError as error:
            raise QueueProtocolError("live settings message is not JSON") from error
        if not isinstance(message, dict) or not isinstance(message.get("type"), str):
            raise QueueProtocolError("live settings message has the wrong shape")
        if message["type"] == "released":
            if set(message) != {"type"}:
                raise QueueProtocolError("released message fields changed")
            self.released.set()
            return False
        if message["type"] != "settings" or set(message) != {
            "type",
            "revision",
            "desired_return",
            "temperature",
        }:
            raise QueueProtocolError("settings message fields changed")
        revision = message["revision"]
        desired_return = message["desired_return"]
        temperature = message["temperature"]
        if type(revision) is not int or revision < 0:
            raise QueueProtocolError("settings revision must be a non-negative integer")
        if desired_return is not None and (
            isinstance(desired_return, bool) or not isinstance(desired_return, (int, float))
        ):
            raise QueueProtocolError("desired_return must be a number or null")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise QueueProtocolError("temperature must be a number")
        if revision >= self._revision:
            self._values = (
                None if desired_return is None else float(desired_return),
                float(temperature),
            )
            self._revision = revision
        return True

    def _receive(self) -> None:
        backoff = 0.25
        reconnecting = False
        while not self._stop.is_set():
            try:
                if reconnecting:
                    job = self._store.get_worker_job(self._job_id, self._worker_id)
                    if job.status in TERMINAL_STATUSES:
                        self.released.set()
                        return
                with self._store.connect_live(self._job_id, self._worker_id) as connection:
                    reconnecting = True
                    backoff = 0.25
                    while not self._stop.is_set():
                        try:
                            message = connection.recv(timeout=0.25)
                        except TimeoutError:
                            continue
                        if not self._apply(message):
                            return
            except InvalidTransitionError, SessionEndedError:
                self.released.set()
                return
            except QueueUnavailableError as error:
                logger.bind(job=self._job_id, event="live_settings").warning("live settings unavailable: {}", error)
            except (ConnectionClosed, OSError) as error:
                logger.bind(job=self._job_id, event="live_settings").warning("live settings disconnected: {}", error)
            if self._stop.wait(backoff):
                return
            backoff = min(backoff * 2, 4.0)

    def current(self) -> tuple[float | None, float]:
        return self._values

    def __enter__(self) -> _LivePolicySettings:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join()


class _ReservationLive:
    """Mark a reservation live when the netplay countdown completes."""

    def __init__(self, config: SlotConfig, store: RunnerQueue, job: Job, health: _SlotHealthReporter) -> None:
        self.config = config
        self.store = store
        self.job = job
        self.health = health
        self.live = False

    def __call__(self) -> None:
        self.store.mark_playing(self.job.id, self.config.worker_id)
        self.health.playing()
        self.live = True
        logger.info(
            "reservation {} live on slot {} game={} delay={}",
            self.job.id,
            self.config.slot,
            self.job.game_count + 1,
            self.job.choices.online_delay,
        )

    def reset(self, job: Job) -> None:
        self.job = job
        self.live = False


def _write_match_measurement(
    config: SlotConfig,
    job: Job,
    result: PlayResult,
    timing: FrameTiming,
    health: _SlotHealthReporter,
    started_at: datetime,
    ended_at: datetime,
) -> Path | None:
    directory = config.measurement_dir
    if directory is None:
        return None
    if len(result.inference_source_frames) != len(result.inference_seconds) or len(result.plan_decisions) != len(
        result.inference_source_frames
    ):
        raise ValueError("match inference samples lack source frame IDs or plan decisions")
    directory.mkdir(parents=True, exist_ok=True)
    elapsed = (ended_at - started_at).total_seconds()
    counters = health.status().chunk_health
    payload = {
        "schema_version": 3,
        "reservation_id": job.id,
        "game_number": job.game_count + 1,
        "slot": config.slot,
        "worker_id": config.worker_id,
        "stream_id": config.stream_id,
        "generation": result.generation,
        "source_git_sha": config.git_sha,
        "graphics_backend": config.graphics_backend,
        "policy_bundle_sha256": config.policy_sha256,
        "checkpoint_sha256": config.checkpoint_sha256,
        "character": job.choices.character,
        "player_identity": job.choices.imitation,
        "desired_return": job.choices.desired_return,
        "temperature": job.choices.temperature,
        "requested_stage": job.choices.requested_stage,
        "timing": asdict(timing),
        "observation_mode": "first_seen_speculative",
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "total_elapsed_seconds": elapsed,
        "connection_countdown_seconds": result.connection_countdown_seconds,
        "gameplay_seconds": result.wall_seconds,
        "match_end_seconds": result.match_end_seconds,
        "frame_ids": result.trajectory.frame_id.tolist(),
        "frames": len(result.trajectory),
        "game_fps": result.game_fps,
        "inference_source_frames": result.inference_source_frames,
        "inference_seconds": result.inference_seconds,
        "frame_interval_seconds": result.frame_interval_seconds,
        "dolphin_step_seconds": result.dolphin_step_seconds,
        "schedule": None if counters is None else asdict(counters),
        "schedule_events": [asdict(event) for event in result.schedule_events],
        "plan_decisions": [asdict(decision) for decision in result.plan_decisions],
        "controller_submission_gaps": None if counters is None else counters.submission_gaps,
        "transport_correction_frames": result.transport_correction_frames,
    }
    path = directory / f"{job.id}-game-{job.game_count + 1}.json"
    pending = path.with_suffix(".json.tmp")
    pending.write_text(json.dumps(payload, allow_nan=False, sort_keys=True))
    pending.replace(path)
    return path


def _write_match_failure(
    config: SlotConfig,
    job: Job,
    timing: FrameTiming,
    health: _SlotHealthReporter,
    started_at: datetime,
    progress: NetplayProgress,
    error: BaseException,
) -> None:
    directory = config.measurement_dir
    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    counters = health.status().chunk_health
    payload = {
        "schema_version": 2,
        "reservation_id": job.id,
        "game_number": job.game_count + 1,
        "slot": config.slot,
        "worker_id": config.worker_id,
        "source_git_sha": config.git_sha,
        "graphics_backend": config.graphics_backend,
        "policy_bundle_sha256": config.policy_sha256,
        "checkpoint_sha256": config.checkpoint_sha256,
        "character": job.choices.character,
        "player_identity": job.choices.imitation,
        "desired_return": job.choices.desired_return,
        "temperature": job.choices.temperature,
        "requested_stage": job.choices.requested_stage,
        "timing": asdict(timing),
        "observation_mode": "first_seen_speculative",
        "started_at": started_at.isoformat(),
        "failed_at": datetime.now(UTC).isoformat(),
        "failure": f"{type(error).__name__}: {error}",
        "progress": asdict(progress),
        "schedule": None if counters is None else asdict(counters),
        "controller_submission_gaps": None if counters is None else counters.submission_gaps,
    }
    path = directory / f"{job.id}-game-{job.game_count + 1}-failure.json"
    pending = path.with_suffix(".json.tmp")
    pending.write_text(json.dumps(payload, allow_nan=False, sort_keys=True))
    pending.replace(path)


def _run_reservation(
    config: SlotConfig,
    store: RemoteQueue,
    policy: InferenceClient,
    runtime: RuntimeConfig,
    job: Job,
    stop: _StopEvent,
    health: _SlotHealthReporter,
    timing: FrameTiming,
) -> None:
    replay_dir = config.replay_dir / f"slot-{config.slot}"
    replay_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat,
        args=(store, job.id, config.worker_id, heartbeat_stop, stop),
        daemon=True,
    )
    heartbeat.start()
    live = _ReservationLive(config, store, job, health)
    logger.info(
        "reservation {} starting on slot {} player={} character={} delay={} first_stage=random "
        "requested_stage={} imitate={} game={}",
        job.id,
        config.slot,
        job.player_code,
        job.choices.character,
        job.choices.online_delay,
        job.choices.requested_stage,
        job.choices.imitation,
        job.game_count + 1,
    )

    store.mark_connecting(job.id, config.worker_id, config.bot_connect_code)
    health.configure_schedule(timing)
    health.connecting(job.choices.online_delay)
    try:
        with (
            _LivePolicySettings(store, job, config.worker_id) as settings,
            NetplaySession(
                config.iso_path,
                dolphin_path=config.dolphin_path,
                user_json_path=config.user_json,
                online_delay=job.choices.online_delay,
                replay_dir=replay_dir,
                slippi_port=config.slippi_port,
                step_timeout_seconds=FRAME_STALL_SECONDS,
                connect_timeout_seconds=IDLE_TIMEOUT_SECONDS,
                connect_abandoned=settings.released.is_set,
                realtime=True,
                graphics_backend=config.graphics_backend,
                stream_output=config.stream_output,
            ) as session,
        ):
            rematch = False
            while not stop.is_set():
                if config.stream_output:
                    assert config.stream_state_path is not None
                    write_stream_state(
                        config.stream_state_path,
                        GameStreamState(
                            job.choices.character,
                            job.choices.imitation,
                            job.choices.desired_return,
                            job.game_count + 1,
                        ),
                    )
                previous = frozenset(replay_dir.rglob("*.slp"))
                started_at = datetime.now(UTC)
                result = run_netplay_match(
                    session,
                    _setup(job, rematch=rematch),
                    policy,
                    runtime,
                    timing,
                    player_identity=None if job.choices.imitation == "MASKED" else job.choices.imitation,
                    policy_settings=settings.current,
                    max_frames=config.max_frames,
                    rematch=rematch,
                    on_live=live,
                    on_failure=partial(_write_match_failure, config, job, timing, health, started_at),
                    observer=health,
                    schedule_observer=health,
                    stream_id=config.stream_id,
                )
                ended_at = datetime.now(UTC)
                _write_match_measurement(config, job, result, timing, health, started_at, ended_at)
                replay_end = read_new_replay_end(replay_dir, previous)
                if replay_end.method is EndMethod.NO_CONTEST:
                    store.mark_no_contest(job.id, config.worker_id)
                    logger.info(
                        "reservation {} ended by no contest on slot {} replay={}",
                        job.id,
                        config.slot,
                        replay_end.path,
                    )
                    return
                if not replay_end.completed:
                    raise RuntimeError(f"netplay replay ended via {replay_end.method.name}: {replay_end.path}")
                replay = replay_end.path
                actual_stage = _stage_name(result.stage)
                human_result = _human_result(result)
                limit_ms = timing.inference_allowance_frames * 1000 / 60
                if result.inference_p95_ms >= limit_ms:
                    logger.warning(
                        "reservation {} missed the delay-{} policy deadline: p95={:.1f}ms limit={:.1f}ms",
                        job.id,
                        job.choices.online_delay,
                        result.inference_p95_ms,
                        limit_ms,
                    )
                next_status = store.finish_game(
                    job.id,
                    config.worker_id,
                    game_number=job.game_count + 1,
                    actual_stage=actual_stage,
                    result=human_result,
                )
                logger.info(
                    "reservation {} game={} complete frames={} wall={:.1f}s fps={:.1f} "
                    "frame_p95={:.1f}ms frame_p99={:.1f}ms dolphin_p95={:.1f}ms dolphin_p99={:.1f}ms "
                    "policy_p95={:.1f}ms policy_p99={:.1f}ms stage={} human_result={} corrections={}",
                    job.id,
                    job.game_count + 1,
                    len(result.trajectory),
                    result.wall_seconds,
                    result.game_fps,
                    result.frame_interval_p95_ms,
                    result.frame_interval_p99_ms,
                    result.dolphin_step_p95_ms,
                    result.dolphin_step_p99_ms,
                    result.inference_p95_ms,
                    result.inference_p99_ms,
                    actual_stage,
                    human_result,
                    result.transport_correction_frames,
                )
                metadata = ReplayMetadata(
                    reservation_id=job.id,
                    player_code=job.player_code,
                    game_number=job.game_count + 1,
                    actual_stage=actual_stage,
                    result=human_result,
                    policy_sha256=config.policy_sha256,
                    git_sha=config.git_sha,
                    started_at=started_at,
                    ended_at=ended_at,
                )
                sidecar = _write_pending_upload(replay, metadata, config.worker_id)
                if config.publish_replays:
                    try:
                        _complete_pending_upload(sidecar, config.queue_endpoint)
                    except Exception as error:  # R2 availability must not end a session.
                        logger.warning("replay upload deferred: {}: {}", type(error).__name__, error)
                if next_status is JobStatus.COMPLETE:
                    return
                health.idle()
                while not stop.is_set():
                    session.park_menu()
                    try:
                        job = store.get_worker_job(job.id, config.worker_id)
                    except InvalidTransitionError:
                        return
                    if job.status is JobStatus.REMATCH_READY:
                        rematch = True
                        live.reset(job)
                        health.configure_schedule(timing)
                        health.connecting(job.choices.online_delay)
                        break
                    if job.status in TERMINAL_STATUSES:
                        return
                    if job.status is not JobStatus.REMATCH_WAIT:
                        raise RuntimeError(f"unexpected reservation status {job.status.value}")
    except ConnectAbandoned:
        logger.info("reservation {} released before its match went live", job.id)
    except FrameTimeout:
        raise _RecoverableRuntimeError("frame_stream_stalled") from None
    except TimeoutError:
        if not live.live:
            with suppress(InvalidTransitionError):
                store.mark_no_show(job.id, config.worker_id)
            return
        raise
    finally:
        if config.stream_output:
            assert config.stream_state_path is not None
            write_stream_state(config.stream_state_path, IdleStreamState())
        with suppress(InferenceUnavailable):
            policy.close_match()
        heartbeat_stop.set()
        heartbeat.join(timeout=1.0)


def _slot_worker(
    config: SlotConfig,
    spec: PolicySpec,
    runtime: RuntimeConfig,
    connection: Connection,
    stop: _StopEvent,
    draining: _StopEvent,
    schedules: tuple[FrameTiming, ...],
    context_frames: int,
    profiles: tuple[PreparedInferenceProfile, ...],
) -> None:
    # The supervisor owns terminal signals. A SIGINT in Event.wait() can kill
    # a spawned worker while it holds the event lock and deadlock shutdown.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if config.display is not None:
        os.environ["DISPLAY"] = config.display
    if config.stream_output:
        os.environ.update(dict(config.pulse_environment))
    else:
        os.environ.pop("PULSE_SERVER", None)
        os.environ.pop("PULSE_SINK", None)
    next_upload_attempt = 0.0
    with (
        connection,
        closing(RemoteQueue(config.queue_endpoint, config.session_id)) as store,
        _SlotHealthReporter(config.slot, config.status_path) as health,
        closing(
            InferenceClient(spec, context_frames, connection, stop, {p.fixed_prefix_frames: p for p in profiles})
        ) as policy,
    ):
        while not stop.is_set():
            health.idle()
            if draining.is_set():
                return
            if config.publish_replays:
                next_upload_attempt = _retry_pending_uploads(
                    config.replay_dir / f"slot-{config.slot}",
                    config.queue_endpoint,
                    next_upload_attempt,
                )
            try:
                job = store.claim_next(config.worker_id)
            except SessionEndedError:
                raise
            except InvalidTransitionError as error:
                logger.bind(slot=config.slot, event="claim").error("slot stopped claiming: {}", error)
                return
            if job is None:
                # Empty claims still use the edge request budget. Stop remains interruptible.
                stop.wait(5.0)
                continue
            logger.info("slot {} claimed reservation {}", config.slot, job.id)
            timing = next(s for s in schedules if s.physical_delay_frames == job.choices.online_delay)
            _handle_reservation(config, store, policy, runtime, job, stop, health, timing)


def _handle_reservation(
    config: SlotConfig,
    store: RemoteQueue,
    policy: InferenceClient,
    runtime: RuntimeConfig,
    job: Job,
    stop: _StopEvent,
    health: _SlotHealthReporter,
    timing: FrameTiming,
) -> None:
    try:
        _run_reservation(config, store, policy, runtime, job, stop, health, timing)
    except DolphinConnectionLost as error:
        health.recovering("dolphin_connection_lost")
        logger.error("reservation {}: {}", job.id, error)
        with suppress(InvalidTransitionError):
            store.forfeit_service_failure(job.id, config.worker_id)
    except NoUsableActionPlan as error:
        health.recovering("no_usable_action_plan")
        logger.error("reservation {}: {}", job.id, error)
        with suppress(InvalidTransitionError):
            store.forfeit_service_failure(job.id, config.worker_id)
    except InferenceUnavailable as error:
        health.recovering("inference_engine_lost")
        logger.error("reservation {}: {}", job.id, error)
        with suppress(InvalidTransitionError):
            store.forfeit_service_failure(job.id, config.worker_id)
        raise
    except SessionEndedError:
        raise
    except QueueUnavailableError as error:
        health.recovering("queue_unavailable")
        logger.error("reservation {} lost queue access: {}", job.id, error)
        raise
    except _RecoverableRuntimeError as error:
        logger.error(
            "reservation {} degraded on slot {}: {}; retrying with a fresh Dolphin",
            job.id,
            config.slot,
            error,
        )
        health.recovering(str(error))
        with suppress(InvalidTransitionError):
            store.fail(job.id, config.worker_id, "runtime_degraded", retryable=True)
        stop.wait(config.recovery_cooldown_seconds)
    except (KeyError, ValueError) as error:
        logger.error("reservation {} rejected: {}: {}", job.id, type(error).__name__, error)
        with suppress(InvalidTransitionError):
            store.fail(job.id, config.worker_id, type(error).__name__.lower(), retryable=False)
    except Exception as error:  # A reservation cannot kill a long-running slot.
        logger.exception("reservation {} failed: {}", job.id, type(error).__name__)
        with suppress(InvalidTransitionError):
            store.fail(job.id, config.worker_id, type(error).__name__.lower(), retryable=True)


def _slot_process(
    context: SpawnContext,
    config: SlotConfig,
    ready: _EngineReady,
    runtime: RuntimeConfig,
    connection: Connection,
    stop: _StopEvent,
    draining: _StopEvent,
) -> BaseProcess:
    return context.Process(
        target=_slot_worker,
        args=(
            config,
            ready.spec,
            runtime,
            connection,
            stop,
            draining,
            _NETPLAY_TIMINGS,
            ready.context_frames,
            ready.profiles,
        ),
        name=f"hal-netplay-slot-{config.slot}",
    )


def _start_slot_process(process: BaseProcess, child_connection: Connection) -> None:
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        process.start()
    finally:
        child_connection.close()
        signal.signal(signal.SIGINT, previous)


def _invalidate_slot_connection(connection: Connection, stream_id: int) -> bool:
    """Wait for the standby pipe's reset barrier before admitting a replacement."""
    token = secrets.token_hex(8)
    deadline = time.monotonic() + _SLOT_RESET_TIMEOUT_SECONDS
    fd: int | None = None
    was_blocking: bool | None = None
    success = False
    try:
        fd = connection.fileno()
        was_blocking = os.get_blocking(fd)
        # A partial control response must fail the reset without blocking the supervisor.
        os.set_blocking(fd, False)
        connection.send(StreamInvalidate(stream_id, token))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not connection.poll(remaining):
                break
            response = connection.recv()
            if isinstance(response, StreamInvalidated):
                success = response == StreamInvalidated(stream_id, token)
            break
    except (EOFError, OSError, ValueError, pickle.UnpicklingError) as error:
        logger.warning("could not reset failed netplay slot stream {}: {}", stream_id, error)
    finally:
        if fd is not None and was_blocking is not None:
            try:
                os.set_blocking(fd, was_blocking)
            except OSError:
                success = False
    return success


def _stop_slot_process(process: BaseProcess) -> None:
    if process.pid is None:
        return
    deadline = time.monotonic() + 2.0
    if process.is_alive():
        with suppress(ProcessLookupError):
            process.terminate()
    process.join(timeout=min(0.75, max(0.0, deadline - time.monotonic())))
    if process.is_alive():
        with suppress(ProcessLookupError):
            process.kill()
    process.join(timeout=max(0.0, deadline - time.monotonic()))
    if process.is_alive():
        raise RuntimeError(f"netplay slot {process.name} could not terminate within two seconds")


def _replace_failed_slot(
    context: SpawnContext,
    failed: BaseProcess,
    config: SlotConfig,
    ready: _EngineReady,
    runtime: RuntimeConfig,
    connection: Connection,
    stop: _StopEvent,
    draining: _StopEvent,
    *,
    restart_allowed: bool,
    capacity: int,
    processes: list[BaseProcess],
) -> tuple[SlotConfig, BaseProcess] | None:
    failed.join(timeout=0)
    logger.warning(
        "netplay slot {} exited code={} worker={}",
        config.slot,
        failed.exitcode,
        config.worker_id,
    )
    config.status_path.unlink(missing_ok=True)
    if config.stream_output:
        assert config.stream_state_path is not None
        write_stream_state(config.stream_state_path, IdleStreamState())
    if not restart_allowed:
        logger.error("netplay slot {} replacement failed; capacity reduced", config.slot)
        return None
    if not _invalidate_slot_connection(connection, config.stream_id):
        connection.close()
        logger.error("netplay slot {} standby connection could not reset the route; capacity reduced", config.slot)
        return None
    replacement = replace(config, stream_id=config.stream_id + capacity)
    process: BaseProcess | None = None
    try:
        process = _slot_process(context, replacement, ready, runtime, connection, stop, draining)
        processes.append(process)
        _start_slot_process(process, connection)
    except (OSError, RuntimeError) as error:
        if process is not None:
            _stop_slot_process(process)
        connection.close()
        logger.error("netplay slot {} replacement could not start: {}", config.slot, error)
        return None
    logger.info(
        "netplay slot {} restarted worker={} stream={} after isolated failure",
        config.slot,
        replacement.worker_id,
        replacement.stream_id,
    )
    return replacement, process


def _await_engine_ready(
    status: Connection,
    process: BaseProcess,
    shutdown: _ShutdownFlag,
    *,
    deadline: float,
) -> _EngineReady:
    while time.monotonic() < deadline:
        if shutdown.requested:
            raise _ShutdownRequested("runner shutdown requested during inference preparation")
        if status.poll(min(0.25, max(0.0, deadline - time.monotonic()))):
            try:
                message = status.recv()
            except EOFError as error:
                raise RuntimeError("inference process closed during preparation") from error
            if isinstance(message, _EngineReady):
                return message
            if isinstance(message, _EngineFailure):
                raise RuntimeError(f"inference preparation failed: {message.reason}")
            raise RuntimeError("inference preparation returned an unknown status")
        if not process.is_alive():
            raise RuntimeError(f"inference process exited during preparation: {process.exitcode}")
    raise TimeoutError("inference preparation did not finish before its deadline")


def _write_budget_record(config: RunnerConfig, ready: _EngineReady, policy_sha256: str) -> None:
    payload = {
        "schema_version": 4,
        "git_sha": config.git_sha,
        "bundle_sha256": policy_sha256,
        "checkpoint_sha256": ready.checkpoint_sha256,
        "capability_version": ready.capability_version,
        "qualified_capacity": len(config.user_jsons),
        "profile_checks": [asdict(check) for check in ready.budgets],
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hardware": ready.hardware,
        "platform": platform.platform(),
        "batch_wait_seconds": config.batch_wait_seconds,
        "preparation_timeout_seconds": config.preparation_timeout_seconds,
        "compiled": config.compiled,
        "history_mode": "kv_cache",
        "sampling_seeds": ready.sampling_seeds,
        "libmelee": melee.version.__version__,
    }
    budget_path = config.status_path.with_suffix(".budget.json")
    budget_path.write_text(json.dumps(payload, allow_nan=False, indent=2))


def _terminate_processes(processes: Sequence[BaseProcess], stop: _StopEvent) -> None:
    stop.set()
    started = tuple(process for process in processes if process.pid is not None)
    deadline = time.monotonic() + 2.0
    grace = min(deadline, time.monotonic() + 0.25)
    for process in started:
        process.join(timeout=max(0.0, grace - time.monotonic()))
    for process in started:
        if process.is_alive():
            with suppress(ProcessLookupError):
                process.terminate()
    term = min(deadline, time.monotonic() + 0.75)
    for process in started:
        process.join(timeout=max(0.0, term - time.monotonic()))
    for process in started:
        if process.is_alive():
            with suppress(ProcessLookupError):
                process.kill()
    for process in started:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    survivors = tuple(process.name for process in started if process.is_alive())
    if survivors:
        raise RuntimeError(f"netplay process termination exceeded two seconds: {survivors}")


def _open_generation_pipes(
    context: _PipeContext, slots: int
) -> tuple[Connection, Connection, dict[int, Connection], dict[int, Connection]]:
    status_receive: Connection | None = None
    status_send: Connection | None = None
    parents: dict[int, Connection] = {}
    children: dict[int, Connection] = {}
    try:
        status_receive, status_send = context.Pipe(duplex=False)
        for slot in range(slots):
            parents[slot], children[slot] = context.Pipe()
    except BaseException:
        for connection in (*parents.values(), *children.values()):
            connection.close()
        if status_receive is not None:
            status_receive.close()
        if status_send is not None:
            status_send.close()
        raise
    return status_receive, status_send, parents, children


def _run_generation(
    config: RunnerConfig,
    bot_connect_codes: tuple[str, ...],
    policy_sha256: str,
    shutdown: _ShutdownFlag,
    *,
    recovery_deadline: float | None,
    session_client: RunnerClient | None = None,
    reporter: SessionReporter | None = None,
    drain_timeout_seconds: float = 900.0,
    displays: tuple[str, ...] = (),
    pulse_environment: tuple[tuple[str, str], ...] = (),
    stream_supervisor: StreamSupervisor | None = None,
) -> None:
    context = mp.get_context("spawn")
    generation_id = secrets.token_hex(8)
    worker_ids = tuple(slot_worker_id(config.session_id, slot) for slot in range(len(config.user_jsons)))
    capacity = len(config.user_jsons)
    stop = context.Event()
    draining = context.Event()
    status_receive, status_send, parent_connections, child_connections = _open_generation_pipes(context, 2 * capacity)
    processes: list[BaseProcess] = []
    slot_processes: dict[int, BaseProcess | None] = {}
    slot_configs: dict[int, SlotConfig] = {}
    replacement_deadlines: dict[int, tuple[float, float]] = {}
    replacement_not_before: dict[int, float] = {}
    restart_counts: dict[int, int] = {}
    slot_status_paths = tuple(_slot_status_path(config.status_path, slot) for slot in range(len(config.user_jsons)))
    try:
        if config.measurement_dir is not None:
            _write_engine_audit_record(
                config.measurement_dir,
                generation_id,
                "started",
                {
                    "schema_version": 1,
                    "phase": "started",
                    "engine_generation_id": generation_id,
                    "source_git_sha": config.git_sha,
                    "policy_bundle_sha256": policy_sha256,
                },
            )
        config.status_path.unlink(missing_ok=True)
        for path in slot_status_paths:
            path.unlink(missing_ok=True)
        engine_config = _InferenceProcessConfig(
            config.policy,
            config.device,
            config.seed,
            config.compiled,
            len(config.user_jsons),
            config.batch_wait_seconds,
            generation_id,
            config.git_sha,
            policy_sha256,
            config.measurement_dir,
            tuple((slot + capacity, slot) for slot in range(capacity)),
        )
        gpu_process = context.Process(
            target=_gpu_inference_process,
            args=(engine_config, parent_connections, status_send, stop),
            name="hal-netplay-inference",
        )
        processes.append(gpu_process)
        gpu_process.start()
        status_send.close()
        for connection in parent_connections.values():
            connection.close()
        startup_deadline = time.monotonic() + config.preparation_timeout_seconds
        if recovery_deadline is not None:
            startup_deadline = min(startup_deadline, recovery_deadline)
        ready = _await_engine_ready(status_receive, gpu_process, shutdown, deadline=startup_deadline)
        if (
            len(ready.budgets) != len(_NETPLAY_TIMINGS)
            or tuple(check.timings[0] for check in ready.budgets) != _NETPLAY_TIMINGS
        ):
            raise RuntimeError("inference process did not qualify both declared netplay profiles")
        _write_budget_record(config, ready, policy_sha256)
        runtime = RuntimeConfig(1, (2, 3), replan_interval_frames=4)
        status_started_at = time.time()
        _write_status(
            config.status_path,
            policy_sha256=policy_sha256,
            slot_paths=slot_status_paths,
            started_at=status_started_at,
            model_inference_p95_ms=None,
            batch_wait_p95_ms=None,
        )
        for slot, (user_json, bot_connect_code, slippi_port) in enumerate(
            zip(config.user_jsons, bot_connect_codes, config.slippi_ports, strict=True)
        ):
            slot_config = SlotConfig(
                slot=slot,
                worker_id=worker_ids[slot],
                stream_id=slot,
                queue_endpoint=config.queue_endpoint,
                session_id=config.session_id,
                user_json=user_json,
                bot_connect_code=bot_connect_code,
                slippi_port=slippi_port,
                iso_path=config.iso_path,
                dolphin_path=config.dolphin_path,
                replay_dir=config.replay_dir,
                status_path=slot_status_paths[slot],
                policy_sha256=policy_sha256,
                checkpoint_sha256=ready.checkpoint_sha256,
                git_sha=config.git_sha,
                graphics_backend=config.graphics_backend,
                max_frames=config.max_frames,
                measurement_dir=config.measurement_dir,
                publish_replays=config.publish_replays,
                display=displays[slot] if displays else None,
                stream_output=config.wants_stream and slot == 0,
                stream_state_path=config.status_path.with_suffix(".stream.json")
                if config.wants_stream and slot == 0
                else None,
                pulse_environment=pulse_environment if config.wants_stream and slot == 0 else (),
            )
            process = _slot_process(context, slot_config, ready, runtime, child_connections[slot], stop, draining)
            processes.append(process)
            _start_slot_process(process, child_connections[slot])
            slot_processes[slot] = process
            slot_configs[slot] = slot_config
            restart_counts[slot] = 0
        logger.info(
            "netplay runner ready generation={} slots={} checkpoint={} capacity={}",
            generation_id,
            len(config.user_jsons),
            ready.checkpoint_sha256,
            len(config.user_jsons),
        )
        last_pulse = time.monotonic()
        next_status = 0.0
        model_p95_ms: float | None = None
        batch_wait_p95_ms: float | None = None
        previous_state: RunnerState | None = None
        drain_deadline: float | None = None
        while True:
            session_state = reporter.state() if reporter is not None else None
            if stream_supervisor is not None:
                stream_supervisor.set_grant(None if session_state is None else session_state.stream)
            remote_draining = session_state is not None and session_state.draining
            if (shutdown.requested or remote_draining) and not draining.is_set():
                if shutdown.requested and not remote_draining and session_client is not None:
                    session_client.drain(config.session_id)
                draining.set()
                drain_deadline = time.monotonic() + drain_timeout_seconds
                logger.bind(session=config.session_id, event="drain").info("runner is draining")
            if shutdown.forced or (drain_deadline is not None and time.monotonic() >= drain_deadline):
                logger.bind(session=config.session_id, event="drain").warning("runner drain was forced")
                stop.set()
                raise RuntimeError("runner drain was forced")
            if draining.is_set() and all(
                process is None or not process.is_alive() for process in slot_processes.values()
            ):
                return
            if status_receive.poll(0.25):
                try:
                    message = status_receive.recv()
                except EOFError as error:
                    raise _EngineLost("inference process closed its health connection") from error
                if isinstance(message, _EngineFailure):
                    raise _EngineLost(message.reason)
                if not isinstance(message, _EnginePulse):
                    raise _EngineLost("inference process returned an unknown health message")
                last_pulse = time.monotonic()
                model_p95_ms = message.model_inference_p95_ms
                batch_wait_p95_ms = message.batch_wait_p95_ms
            if not gpu_process.is_alive():
                raise _EngineLost(f"inference process exited with code {gpu_process.exitcode}")
            if time.monotonic() - last_pulse >= _ENGINE_PROGRESS_TIMEOUT_SECONDS:
                raise _EngineLost("inference process made no progress for one second")
            for slot, (deadline, started_at) in tuple(replacement_deadlines.items()):
                process = slot_processes[slot]
                if process is None or not process.is_alive():
                    continue
                try:
                    slot_status = read_slot_status(slot_status_paths[slot])
                except ValueError:
                    slot_status = None
                if (
                    slot_status is not None
                    and slot_status.updated_at >= started_at
                    and slot_status.state in (SlotState.IDLE, SlotState.CONNECTING, SlotState.PLAYING)
                ):
                    del replacement_deadlines[slot]
                elif time.monotonic() >= deadline:
                    logger.error("netplay slot {} replacement did not become ready within startup grace", slot)
                    _stop_slot_process(process)
            for slot, process in tuple(slot_processes.items()):
                if process is None or process.is_alive():
                    continue
                if not draining.is_set():
                    not_before = replacement_not_before.setdefault(
                        slot, time.monotonic() + _SLOT_RESTART_LEASE_GRACE_SECONDS
                    )
                    if time.monotonic() < not_before:
                        continue
                replacement_not_before.pop(slot, None)
                replacement_deadlines.pop(slot, None)
                restart_started_at = time.time()
                replacement = _replace_failed_slot(
                    context,
                    process,
                    slot_configs[slot],
                    ready,
                    runtime,
                    child_connections[slot + capacity],
                    stop,
                    draining,
                    restart_allowed=not draining.is_set() and restart_counts[slot] == 0,
                    capacity=capacity,
                    processes=processes,
                )
                if replacement is None:
                    slot_processes[slot] = None
                    continue
                slot_configs[slot], slot_processes[slot] = replacement
                restart_counts[slot] += 1
                replacement_deadlines[slot] = (
                    time.monotonic() + SLOT_STARTUP_GRACE_SECONDS,
                    restart_started_at,
                )
            if slot_processes and all(process is None for process in slot_processes.values()):
                raise RuntimeError("all netplay slots are unavailable")
            if time.monotonic() >= next_status:
                status = _write_status(
                    config.status_path,
                    policy_sha256=policy_sha256,
                    slot_paths=slot_status_paths,
                    started_at=status_started_at,
                    model_inference_p95_ms=model_p95_ms,
                    batch_wait_p95_ms=batch_wait_p95_ms,
                )
                if status.state is not previous_state:
                    logger.info(
                        "netplay service state={} healthy_slots={}/{} game_fps={} model_p95_ms={} batch_wait_p95_ms={}",
                        status.state.value,
                        status.healthy_slots,
                        status.slots,
                        status.game_fps,
                        status.model_inference_p95_ms,
                        status.batch_wait_p95_ms,
                    )
                    previous_state = status.state
                next_status = time.monotonic() + 0.5
    finally:
        try:
            _terminate_processes(processes, stop)
        finally:
            for connection in (
                *parent_connections.values(),
                *child_connections.values(),
                status_receive,
                status_send,
            ):
                connection.close()
            config.status_path.unlink(missing_ok=True)
            for path in slot_status_paths:
                path.unlink(missing_ok=True)
            if config.wants_stream:
                write_stream_state(config.status_path.with_suffix(".stream.json"), IdleStreamState())


def run(
    config: RunnerConfig,
    *,
    session_client: RunnerClient | None = None,
    reporter: SessionReporter | None = None,
    drain_timeout_seconds: float = 900.0,
) -> None:
    """Admit matches only after a separate GPU process qualifies each profile."""
    config.status_path.parent.mkdir(parents=True, exist_ok=True)
    config.status_path.unlink(missing_ok=True)
    bot_connect_codes = _bot_connect_codes(config.user_jsons)
    policy_sha256 = _sha256(config.policy)
    shutdown = _ShutdownFlag()
    previous_int = signal.signal(signal.SIGINT, shutdown)
    previous_term = signal.signal(signal.SIGTERM, shutdown)
    recovery_deadline: float | None = None
    try:
        with ExitStack() as resources:
            displays: tuple[str, ...] = ()
            pulse_environment: tuple[tuple[str, str], ...] = ()
            stream_supervisor: StreamSupervisor | None = None
            if config.display_base is not None:
                display_group = resources.enter_context(
                    DisplayGroup(
                        len(config.user_jsons),
                        config.display_base,
                        config.wants_stream,
                        stream_display=config.stream_display,
                    )
                )
                displays = display_group.displays
            if config.wants_stream:
                if session_client is None or reporter is None:
                    raise ValueError("streaming requires a session client and reporter")
                state_path = config.status_path.with_suffix(".stream.json")
                write_stream_state(state_path, IdleStreamState())
                pulse = resources.enter_context(PulseAudio(config.status_path.parent / "pulse"))
                pulse_environment = tuple(pulse.environment.items())
                stream_supervisor = resources.enter_context(
                    StreamSupervisor(
                        displays[0],
                        state_path,
                        config.status_path.with_suffix(".overlay.txt"),
                        session_client.queue_depth,
                        pulse.environment,
                        bandwidth_test=config.twitch_bandwidth_test,
                        slot_status_path=_slot_status_path(config.status_path, 0),
                    )
                )
                stream_supervisor.set_grant(reporter.state().stream)
            for attempt in range(2):
                try:
                    _run_generation(
                        config,
                        bot_connect_codes,
                        policy_sha256,
                        shutdown,
                        recovery_deadline=recovery_deadline,
                        session_client=session_client,
                        reporter=reporter,
                        drain_timeout_seconds=drain_timeout_seconds,
                        displays=displays,
                        pulse_environment=pulse_environment,
                        stream_supervisor=stream_supervisor,
                    )
                    return
                except _ShutdownRequested:
                    return
                except _EngineLost as error:
                    if attempt == 1 or shutdown.requested:
                        raise RuntimeError("netplay inference recovery failed; service remains unavailable") from error
                    recovery_deadline = time.monotonic() + _ENGINE_RECOVERY_TIMEOUT_SECONDS
                    logger.error("netplay inference lost: {}; preparing one replacement process", error)
    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)
        config.status_path.unlink(missing_ok=True)


def _download_session_assets(started: StartedSession, *, local_assets: Path | None) -> tuple[Path, tuple[Path, ...]]:
    remote = None
    if local_assets is None:
        remote = r2.client()
        source = R2Source(remote, r2.bucket())
    else:
        source = LocalSource(local_assets)
    cache = AssetCache(Path("~/.cache/hal-netplay").expanduser(), source)
    try:
        policy = cache.get(PinnedAsset(started.policy.bundle_r2_key, started.policy.bundle_sha256))
        accounts = tuple(cache.get(PinnedAsset(grant.r2_key, grant.sha256)) for grant in started.accounts)
    finally:
        if remote is not None:
            remote.close()
    artifact = read_action_sequence_artifact(policy)
    if artifact.vocabulary.sha256 != started.policy.vocabulary_sha256:
        raise ValueError(
            f"policy vocabulary {artifact.vocabulary.sha256} does not match {started.policy.vocabulary_sha256}"
        )
    for path, grant in zip(accounts, started.accounts, strict=True):
        observed = _bot_connect_code(path)
        if observed != grant.connect_code:
            raise ValueError(f"account {path} has connect code {observed}, not leased {grant.connect_code}")
    return policy, accounts


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="hal-netplay-runner")
    parser.add_argument("--slots", type=int, required=True)
    parser.add_argument("--slippi-port", type=int, default=51441)
    parser.add_argument("--local-assets", type=Path)
    parser.add_argument("--drain-timeout", type=float, default=900.0)
    parser.add_argument("--metrics-port", type=int, default=9100)
    parser.add_argument("--display-base", type=int, default=100)
    parser.add_argument("--stream-display", default=os.environ.get("HAL_NETPLAY_STREAM_DISPLAY"))
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--twitch-bandwidth-test", action="store_true")
    parser.add_argument("--graphics-backend", choices=("Vulkan", "OGL"), default="Vulkan")
    parser.add_argument("--replay-dir", type=Path, default=Path("runs/netplay/replays"))
    parser.add_argument("--status-path", type=Path, default=Path("runs/netplay/runner-status.json"))
    parser.add_argument("--git-sha", default=os.environ.get("HAL_GIT_SHA"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--compiled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch-wait-ms", type=float, default=0.5)
    parser.add_argument("--preparation-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--max-frames", type=int, default=54_000)
    args = parser.parse_args(argv)
    if args.git_sha is None:
        parser.error("set HAL_GIT_SHA or pass --git-sha")
    if not 1 <= args.slots <= 16:
        parser.error("--slots must be in [1, 16]")
    if not math.isfinite(args.drain_timeout) or args.drain_timeout <= 0:
        parser.error("--drain-timeout must be finite and positive")
    if not 1 <= args.slippi_port <= 65_535 or args.slippi_port + args.slots - 1 > 65_535:
        parser.error("--slippi-port range is invalid")
    if not 1 <= args.metrics_port <= 65_535:
        parser.error("--metrics-port must be in [1, 65535]")
    if args.display_base < 0 or args.display_base + args.slots > 65_535:
        parser.error("--display-base range is invalid")
    if args.stream and not args.stream_display:
        parser.error("streaming requires --stream-display with a dedicated NVIDIA Xorg display")

    start_http_server(args.metrics_port, addr="127.0.0.1")
    iso_path = ensure(ISO)
    dolphin_path = ensure(NETPLAY_EMULATOR)
    endpoint = runner_endpoint(os.environ)
    client = RunnerClient(endpoint)
    session_id = new_session_id()
    try:
        policy_config = client.active_policy()
        try:
            for attempt in range(2):
                try:
                    started = client.start_session(
                        session_id=session_id,
                        host=socket.gethostname(),
                        bundle_sha256=policy_config.bundle_sha256,
                        git_sha=args.git_sha,
                        slots=args.slots,
                        wants_stream=args.stream,
                    )
                    break
                except QueueUnavailableError:
                    if attempt == 1:
                        raise
            else:
                raise AssertionError("session start loop always starts or raises")
        except InvalidTransitionError as error:
            logger.bind(session=session_id, event="session_start").error("session start refused: {}", error)
            raise
        except QueueProtocolError:
            with suppress(Exception):
                client.end_session(session_id)
            raise
        status_path = args.status_path.resolve()
        status = SessionStatus(status_path, started.policy.bundle_sha256, args.slots)
        try:
            with SessionReporter(client, session_id, status) as reporter:
                policy, user_jsons = _download_session_assets(started, local_assets=args.local_assets)
                config = RunnerConfig(
                    queue_endpoint=endpoint,
                    session_id=session_id,
                    policy=policy,
                    user_jsons=user_jsons,
                    slippi_ports=tuple(args.slippi_port + slot for slot in range(args.slots)),
                    iso_path=iso_path,
                    dolphin_path=dolphin_path,
                    graphics_backend=args.graphics_backend,
                    replay_dir=args.replay_dir.resolve(),
                    status_path=status_path,
                    git_sha=args.git_sha,
                    device=args.device,
                    seed=args.seed,
                    compiled=args.compiled,
                    batch_wait_seconds=args.batch_wait_ms / 1000,
                    preparation_timeout_seconds=args.preparation_timeout_seconds,
                    max_frames=args.max_frames,
                    display_base=args.display_base,
                    wants_stream=args.stream,
                    stream_display=args.stream_display,
                    twitch_bandwidth_test=args.twitch_bandwidth_test,
                )
                run(
                    config,
                    session_client=client,
                    reporter=reporter,
                    drain_timeout_seconds=args.drain_timeout,
                )
        finally:
            client.end_session(session_id)
    finally:
        client.close()


if __name__ == "__main__":
    main()
