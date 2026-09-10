"""Long-running netplay reservation workers and inference engine."""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import signal
import threading
import time
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Protocol

import melee
import torch
from loguru import logger

from hal.eval.match_summary import summarize_trajectory
from hal.eval.play import PlayResult
from hal.eval.play import require_completed_replay
from hal.eval.play import run_netplay_match
from hal.inference.api import PolicySpec
from hal.inference.api import RuntimeConfig
from hal.inference.checkpoints import resolve_checkpoint
from hal.inference.loader import load_policy
from hal.netplay_service.domain import TERMINAL_STATUSES
from hal.netplay_service.domain import Job
from hal.netplay_service.domain import JobStatus
from hal.netplay_service.domain import validate_player_code
from hal.netplay_service.inference import ContinuousBatcher
from hal.netplay_service.inference import RemotePolicy
from hal.netplay_service.inference import ServingArena
from hal.netplay_service.inference import ServingArenaDescriptor
from hal.netplay_service.queue import InvalidTransitionError
from hal.netplay_service.queue import QueueStore
from hal.netplay_service.replays import ReplayMetadata
from hal.netplay_service.replays import upload_replay
from hal.paths import ISO_PATH
from hal.paths import NETPLAY_EMULATOR_PATH
from hal.sim.netplay import NetplaySession
from hal.sim.netplay import NetplaySetup

_PENDING_UPLOAD_RETRY_SECONDS = 60.0


class _StopEvent(Protocol):
    def is_set(self) -> bool: ...

    def wait(self, timeout: float | None = None) -> bool: ...


@dataclass(frozen=True, slots=True)
class SlotConfig:
    """Cold configuration for one long-running Dolphin worker."""

    slot: int
    worker_id: str
    stream_id: int
    database: Path
    user_json: Path
    bot_connect_code: str
    slippi_port: int
    iso_path: Path
    dolphin_path: Path
    replay_dir: Path
    policy_sha256: str
    git_sha: str
    max_frames: int = 54_000


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    """Validated deployment configuration for one GPU runner."""

    database: Path
    policy: Path
    user_jsons: tuple[Path, ...]
    slippi_ports: tuple[int, ...]
    iso_path: Path
    dolphin_path: Path
    replay_dir: Path
    status_path: Path
    git_sha: str
    device: str = "cuda"
    seed: int | None = None
    compiled: bool = False
    batch_wait_seconds: float = 0.0005
    max_frames: int = 54_000

    def __post_init__(self) -> None:
        if not self.user_jsons:
            raise ValueError("runner needs at least one Slippi account")
        if len(self.user_jsons) != len(self.slippi_ports):
            raise ValueError("runner needs one Slippi port per account slot")
        if len(set(self.slippi_ports)) != len(self.slippi_ports):
            raise ValueError("runner Slippi ports must be unique")
        if any(not 1 <= port <= 65_535 for port in self.slippi_ports):
            raise ValueError("runner Slippi ports must be in [1, 65535]")
        if not self.git_sha or "/" in self.git_sha:
            raise ValueError("runner git_sha must be a non-empty identifier")
        if self.max_frames < 6:
            raise ValueError("runner max_frames must be at least 6")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_status(path: Path, *, policy_sha256: str, slots: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(
            {
                "policy_sha256": policy_sha256,
                "schema_version": 1,
                "slots": slots,
                "updated_at": time.time(),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    temporary.replace(path)


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


def _write_pending_upload(path: Path, metadata: ReplayMetadata) -> Path:
    payload = asdict(metadata)
    payload["started_at"] = metadata.started_at.astimezone(UTC).isoformat()
    payload["ended_at"] = metadata.ended_at.astimezone(UTC).isoformat()
    payload["schema_version"] = 1
    sidecar = _upload_sidecar(path)
    temporary = sidecar.with_suffix(sidecar.suffix + ".partial")
    temporary.write_text(json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True))
    temporary.replace(sidecar)
    return sidecar


def _read_pending_upload(sidecar: Path) -> tuple[Path, ReplayMetadata]:
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
    }
    if not isinstance(payload, dict) or set(payload) != expected or payload.get("schema_version") != 1:
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
    return replay, metadata


def _complete_pending_upload(sidecar: Path, store: QueueStore) -> None:
    replay, metadata = _read_pending_upload(sidecar)
    uploaded = upload_replay(replay, metadata)
    store.record_replay(
        metadata.reservation_id,
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


def _drain_pending_uploads(root: Path, store: QueueStore) -> None:
    for sidecar in sorted(root.rglob("*.slp.upload.json")):
        try:
            _complete_pending_upload(sidecar, store)
        except Exception as error:  # Replay failures are isolated from active games.
            logger.warning(
                "pending replay upload failed: {}: {}: {}",
                type(error).__name__,
                error,
                sidecar,
            )


def _retry_pending_uploads(root: Path, store: QueueStore, next_attempt: float) -> float:
    now = time.monotonic()
    if now < next_attempt:
        return next_attempt
    _drain_pending_uploads(root, store)
    return now + _PENDING_UPLOAD_RETRY_SECONDS


def _heartbeat(store: QueueStore, job_id: str, worker_id: str, stop: threading.Event) -> None:
    while not stop.wait(5.0):
        try:
            store.heartbeat(job_id, worker_id, lease_seconds=20.0)
        except InvalidTransitionError:
            return


def _run_reservation(
    config: SlotConfig,
    store: QueueStore,
    policy: RemotePolicy,
    runtime: RuntimeConfig,
    job: Job,
    stop: _StopEvent,
) -> None:
    replay_dir = config.replay_dir / f"slot-{config.slot}"
    replay_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat,
        args=(store, job.id, config.worker_id, heartbeat_stop),
        daemon=True,
    )
    heartbeat.start()
    live = False
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

    def mark_live() -> None:
        nonlocal live
        store.mark_playing(job.id, config.worker_id)
        live = True
        logger.info(
            "reservation {} live on slot {} game={} delay={}",
            job.id,
            config.slot,
            job.game_count + 1,
            job.choices.online_delay,
        )

    store.mark_connecting(job.id, config.worker_id, config.bot_connect_code, timeout_seconds=60.0)
    try:
        with NetplaySession(
            config.iso_path,
            dolphin_path=config.dolphin_path,
            user_json_path=config.user_json,
            online_delay=job.choices.online_delay,
            replay_dir=replay_dir,
            slippi_port=config.slippi_port,
            connect_timeout_seconds=60.0,
        ) as session:
            rematch = False
            while not stop.is_set():
                previous = frozenset(replay_dir.rglob("*.slp"))
                started_at = datetime.now(UTC)
                result = run_netplay_match(
                    session,
                    _setup(job, rematch=rematch),
                    policy,
                    runtime,
                    player_identity=job.choices.imitation,
                    max_frames=config.max_frames,
                    rematch=rematch,
                    on_live=mark_live,
                    stream_id=config.stream_id,
                )
                ended_at = datetime.now(UTC)
                replay = require_completed_replay(replay_dir, previous)
                actual_stage = _stage_name(result.stage)
                human_result = _human_result(result)
                limit_ms = 33.3 if job.choices.online_delay == 2 else 16.7
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
                    actual_stage=actual_stage,
                    result=human_result,
                )
                logger.info(
                    "reservation {} game={} complete frames={} wall={:.1f}s stage={} human_result={} "
                    "policy_p95={:.1f}ms corrections={}",
                    job.id,
                    job.game_count + 1,
                    len(result.trajectory),
                    result.wall_seconds,
                    actual_stage,
                    human_result,
                    result.inference_p95_ms,
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
                sidecar = _write_pending_upload(replay, metadata)
                try:
                    _complete_pending_upload(sidecar, store)
                except Exception as error:  # R2 availability must not end a session.
                    logger.warning("replay upload deferred: {}: {}", type(error).__name__, error)
                if next_status is JobStatus.COMPLETE:
                    return
                while not stop.is_set():
                    session.park_menu()
                    try:
                        job = store.get_worker_job(job.id, config.worker_id)
                    except InvalidTransitionError:
                        return
                    if job.status is JobStatus.REMATCH_READY:
                        rematch = True
                        live = False
                        break
                    if job.status in TERMINAL_STATUSES:
                        return
                    if job.status is not JobStatus.REMATCH_WAIT:
                        raise RuntimeError(f"unexpected reservation status {job.status.value}")
    except TimeoutError:
        if not live:
            with suppress(InvalidTransitionError):
                store.mark_no_show(job.id, config.worker_id)
            return
        raise
    finally:
        heartbeat_stop.set()
        heartbeat.join(timeout=1.0)


def _slot_worker(
    config: SlotConfig,
    descriptor: ServingArenaDescriptor,
    spec: PolicySpec,
    runtime: RuntimeConfig,
    connection: Connection,
    stop: _StopEvent,
) -> None:
    # The supervisor owns terminal signals. A SIGINT in Event.wait() can kill
    # a spawned worker while it holds the event lock and deadlock shutdown.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    store = QueueStore(config.database)
    next_upload_attempt = 0.0
    with connection, ServingArena.attach(descriptor) as arena:
        policy = RemotePolicy(spec, runtime, arena, connection, config.slot)
        while not stop.is_set():
            next_upload_attempt = _retry_pending_uploads(
                config.replay_dir / f"slot-{config.slot}",
                store,
                next_upload_attempt,
            )
            job = store.claim_next(config.worker_id, lease_seconds=20.0)
            if job is None:
                stop.wait(0.25)
                continue
            logger.info("slot {} claimed reservation {}", config.slot, job.id)
            try:
                _run_reservation(config, store, policy, runtime, job, stop)
            except (KeyError, ValueError) as error:
                logger.error("reservation {} rejected: {}: {}", job.id, type(error).__name__, error)
                with suppress(InvalidTransitionError):
                    store.fail(job.id, config.worker_id, type(error).__name__.lower(), retryable=False)
            except Exception as error:  # A reservation cannot kill a long-running slot.
                logger.exception("reservation {} failed: {}", job.id, type(error).__name__)
                with suppress(InvalidTransitionError):
                    store.fail(job.id, config.worker_id, type(error).__name__.lower(), retryable=True)


def _start_slot_process(process: BaseProcess, child_connection: Connection) -> None:
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        process.start()
    finally:
        child_connection.close()
        signal.signal(signal.SIGINT, previous)


def run(config: RunnerConfig) -> None:
    """Load and prepare one policy, then supervise fixed Dolphin slots."""
    bot_connect_codes = _bot_connect_codes(config.user_jsons)
    runtime = RuntimeConfig(max_batch_size=len(config.user_jsons), transport_delays=(2, 3))
    started = time.perf_counter()
    logger.info(
        "loading netplay policy {} on {} mode={} slots={} delays={} display={}",
        config.policy,
        config.device,
        "compiled" if config.compiled else "eager",
        len(config.user_jsons),
        runtime.transport_delays,
        os.environ.get("DISPLAY", "unset"),
    )
    policy = load_policy(
        config.policy,
        device=config.device,
        seed=config.seed,
        compiled=config.compiled,
    )
    policy.prepare(runtime)
    logger.info("netplay policy ready after {:.2f}s", time.perf_counter() - started)
    stop = mp.get_context("spawn").Event()
    thread_stop = threading.Event()
    shutdown_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal shutdown_requested
        shutdown_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    context = mp.get_context("spawn")
    parent_connections: dict[int, Connection] = {}
    child_connections: dict[int, Connection] = {}
    for slot in range(len(config.user_jsons)):
        parent, child = context.Pipe()
        parent_connections[slot] = parent
        child_connections[slot] = child

    policy_sha256 = _sha256(config.policy)
    processes: list[BaseProcess] = []
    with ServingArena.create(
        len(config.user_jsons),
        max(runtime.transport_delays),
        policy.spec.required_observation_fields,
    ) as arena:
        for slot, (user_json, bot_connect_code, slippi_port) in enumerate(
            zip(config.user_jsons, bot_connect_codes, config.slippi_ports, strict=True)
        ):
            slot_config = SlotConfig(
                slot=slot,
                worker_id=f"slot-{slot}",
                stream_id=slot,
                database=config.database,
                user_json=user_json,
                bot_connect_code=bot_connect_code,
                slippi_port=slippi_port,
                iso_path=config.iso_path,
                dolphin_path=config.dolphin_path,
                replay_dir=config.replay_dir,
                policy_sha256=policy_sha256,
                git_sha=config.git_sha,
                max_frames=config.max_frames,
            )
            process = context.Process(
                target=_slot_worker,
                args=(
                    slot_config,
                    arena.descriptor,
                    policy.spec,
                    runtime,
                    child_connections[slot],
                    stop,
                ),
                name=f"hal-netplay-slot-{slot}",
            )
            _start_slot_process(process, child_connections[slot])
            processes.append(process)

        engine_error: list[BaseException] = []
        batcher = ContinuousBatcher(
            policy,
            runtime,
            arena,
            parent_connections,
            batch_wait_seconds=config.batch_wait_seconds,
        )

        def serve() -> None:
            try:
                with torch.compiler.set_stance("fail_on_recompile"):
                    batcher.serve(thread_stop)
            except BaseException as error:
                engine_error.append(error)
                stop.set()

        engine = threading.Thread(target=serve, name="hal-netplay-inference", daemon=True)
        engine.start()
        logger.info("netplay runner ready slots={} policy_sha256={}", len(processes), policy_sha256)
        try:
            while not shutdown_requested:
                if engine_error:
                    raise RuntimeError("netplay inference engine failed") from engine_error[0]
                failed = [process for process in processes if not process.is_alive()]
                if failed:
                    raise RuntimeError(f"netplay slot process exited with code {failed[0].exitcode}")
                _write_status(
                    config.status_path,
                    policy_sha256=policy_sha256,
                    slots=len(config.user_jsons),
                )
                time.sleep(0.5)
        finally:
            stop.set()
            thread_stop.set()
            for process in processes:
                process.join(timeout=10.0)
                if process.is_alive():
                    process.terminate()
            engine.join(timeout=2.0)
            for connection in (*parent_connections.values(), *child_connections.values()):
                connection.close()
            config.status_path.unlink(missing_ok=True)
            logger.info("netplay runner stopped")
        if engine_error:
            raise RuntimeError("netplay inference engine failed") from engine_error[0]


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item).expanduser().resolve() for item in value.split(",") if item)
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError("HAL_NETPLAY_USER_JSONS must list existing files")
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="hal-netplay-runner")
    parser.add_argument("policy")
    parser.add_argument("--database", type=Path, default=Path("runs/netplay/queue.sqlite3"))
    parser.add_argument("--user-jsons", default=os.environ.get("HAL_NETPLAY_USER_JSONS"))
    parser.add_argument("--slippi-ports", default="51441,51442")
    parser.add_argument("--iso-path", type=Path, default=Path(ISO_PATH))
    parser.add_argument("--dolphin-path", type=Path, default=Path(NETPLAY_EMULATOR_PATH))
    parser.add_argument("--replay-dir", type=Path, default=Path("runs/netplay/replays"))
    parser.add_argument("--status-path", type=Path)
    parser.add_argument("--git-sha", default=os.environ.get("HAL_GIT_SHA"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--compiled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-frames", type=int, default=54_000)
    args = parser.parse_args(argv)
    if args.user_jsons is None:
        parser.error("set HAL_NETPLAY_USER_JSONS or pass --user-jsons")
    if args.git_sha is None:
        parser.error("set HAL_GIT_SHA or pass --git-sha")
    user_jsons = _paths(args.user_jsons)
    ports = tuple(int(value) for value in args.slippi_ports.split(","))
    policy = resolve_checkpoint(args.policy)
    database = args.database.resolve()
    run(
        RunnerConfig(
            database=database,
            policy=policy,
            user_jsons=user_jsons,
            slippi_ports=ports,
            iso_path=args.iso_path.resolve(),
            dolphin_path=args.dolphin_path.resolve(),
            replay_dir=args.replay_dir.resolve(),
            status_path=(args.status_path or database.parent / "runner-status.json").resolve(),
            git_sha=args.git_sha,
            device=args.device,
            seed=args.seed,
            compiled=args.compiled,
            max_frames=args.max_frames,
        )
    )


if __name__ == "__main__":
    main()
