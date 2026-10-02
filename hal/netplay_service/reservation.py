"""One reservation on one runner slot: the reporting link and the Slippi loop."""

import json
import threading
import time
from collections.abc import Callable
from contextlib import closing
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Protocol

import melee
from loguru import logger
from peppi_py.game import EndMethod

from hal.eval.match_summary import summarize_trajectory
from hal.eval.netplay import DolphinConnectionLost
from hal.eval.netplay import NoUsableActionPlan
from hal.eval.netplay import PausedTooLong
from hal.eval.netplay import run_netplay_match
from hal.eval.replays import ReplayEnd
from hal.eval.replays import read_new_replay_end
from hal.eval.results import NetplayProgress
from hal.eval.results import PlayResult
from hal.eval.scheduling import ActionScheduler
from hal.eval.scheduling import FrameTiming
from hal.inference.api import RuntimeConfig
from hal.inference.client import InferenceClient
from hal.inference.client import InferenceUnavailable
from hal.netplay_service.domain import CONNECT_TIMEOUT_SECONDS
from hal.netplay_service.domain import CONNECTION_PROBE_SECONDS
from hal.netplay_service.domain import IDLE_TIMEOUT_SECONDS
from hal.netplay_service.domain import LOCK_HOLD_CAP_SECONDS
from hal.netplay_service.domain import LOCK_HOLD_SECONDS
from hal.netplay_service.domain import PAUSE_TIMEOUT_SECONDS
from hal.netplay_service.domain import REPORT_INTERVAL_SECONDS
from hal.netplay_service.domain import EndReason
from hal.netplay_service.domain import FinishedGame
from hal.netplay_service.domain import GameResult
from hal.netplay_service.domain import Job
from hal.netplay_service.domain import JobStatus
from hal.netplay_service.domain import Observed
from hal.netplay_service.domain import Phase
from hal.netplay_service.domain import Settings
from hal.netplay_service.domain import WindDown
from hal.netplay_service.health import FRAME_STALL_SECONDS
from hal.netplay_service.health import SlotStatus
from hal.netplay_service.queue_client import QueueEndpoint
from hal.netplay_service.queue_client import QueueUnavailableError
from hal.netplay_service.queue_client import RemoteQueue
from hal.netplay_service.queue_contract import InvalidTransitionError
from hal.netplay_service.queue_contract import QueueError
from hal.netplay_service.queue_contract import RunnerQueue
from hal.netplay_service.queue_contract import SessionEndedError
from hal.netplay_service.replays import ReplayMetadata
from hal.netplay_service.replays import upload_replay
from hal.netplay_service.stream import GameStreamState
from hal.netplay_service.stream import IdleStreamState
from hal.netplay_service.stream import write_stream_state
from hal.sim.netplay import ConnectAbandoned
from hal.sim.netplay import DirectMenuDriver
from hal.sim.netplay import DirectSelection
from hal.sim.netplay import NetplaySession
from hal.sim.netplay import NetplaySetup
from hal.sim.netplay import PlayerDisconnected
from hal.sim.netplay import PlayerIdle
from hal.sim.netplay import PlayerNoShow
from hal.sim.session import FrameTimeout

if TYPE_CHECKING:
    from hal.netplay_service.runner import SlotConfig


_PENDING_UPLOAD_RETRY_SECONDS = 60.0


class StopEvent(Protocol):
    def is_set(self) -> bool: ...

    def wait(self, timeout: float | None = None) -> bool: ...


class SlotHealth(Protocol):
    def idle(self) -> None: ...

    def connecting(self, delay: int) -> None: ...

    def playing(self) -> None: ...

    def recovering(self, reason: str) -> None: ...

    def configure_schedule(self, schedule: FrameTiming) -> None: ...

    def observe_frame(self, frame_id: int, dolphin_step_seconds: float) -> None: ...

    def observe_schedule(self, schedule: ActionScheduler) -> None: ...

    def observe_policy(self, seconds: float) -> None: ...

    def status(self) -> SlotStatus: ...


class RecoverableRuntimeError(RuntimeError):
    pass


class ReservationLink:
    """Report snapshots in the background and keep the Worker's latest answer.

    The main loop only reads desired state and writes observations; it never waits
    on the network. A refused report means the Worker no longer gives this slot the
    job, so the link stops and the loop aborts at its next check.
    """

    def __init__(
        self,
        queue: RunnerQueue,
        job: Job,
        worker_id: str,
        *,
        interval_seconds: float = REPORT_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._queue = queue
        self._job_id = job.id
        self._worker_id = worker_id
        self._interval = interval_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._seq = 0
        self._job = job
        self._history: dict[int, Settings] = {job.settings.revision: job.settings}
        self._phase = Phase.BOOTING
        self._deadline: float | None = None
        self._bot_code: str | None = None
        self._locked: int | None = None
        # Keep the complete local history; RemoteQueue sends only unacknowledged results.
        self._games: list[FinishedGame] = list(job.games)
        self._released = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"reservation-{job.id}", daemon=True)

    def __enter__(self) -> ReservationLink:
        self._send()
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()

    def _snapshot(self) -> Observed:
        with self._lock:
            self._seq += 1
            left = None if self._deadline is None else max(0.0, self._deadline - self._clock())
            return Observed(
                seq=self._seq,
                phase=self._phase,
                phase_seconds_left=left,
                bot_code=self._bot_code,
                seen_revision=self._job.settings.revision,
                locked_revision=self._locked,
                finished_games=tuple(self._games),
            )

    def _send(self) -> None:
        if self._released:
            return
        try:
            job = self._queue.report(self._job_id, self._worker_id, self._snapshot())
        except QueueUnavailableError as error:
            # The lease tolerates a few lost reports; the next tick tries again.
            logger.bind(job=self._job_id, event="report").warning("reservation report failed: {}", error)
            return
        except InvalidTransitionError:
            with self._lock:
                self._released = True
            return
        except QueueError as error:
            # A refused or unreadable report will not succeed on retry; stop so the
            # loop aborts and the Worker's lease rule takes the job back.
            logger.bind(job=self._job_id, event="report").error("reservation report refused: {}", error)
            with self._lock:
                self._released = True
            return
        with self._lock:
            self._job = job
            self._history[job.settings.revision] = job.settings
            if job.status is not JobStatus.ASSIGNED:
                self._released = True

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._send()
            if self._released:
                return

    def set_phase(self, phase: Phase, deadline: float | None) -> None:
        with self._lock:
            self._phase, self._deadline = phase, deadline

    def set_bot_code(self, code: str) -> None:
        with self._lock:
            self._bot_code = code

    def lock(self, revision: int | None) -> None:
        with self._lock:
            self._locked = revision

    def add_game(self, game: FinishedGame) -> None:
        with self._lock:
            self._games.append(game)

    def settings(self) -> Settings:
        with self._lock:
            return self._job.settings

    def settings_at(self, revision: int) -> Settings:
        with self._lock:
            return self._history[revision]

    def wind_down(self) -> WindDown | None:
        with self._lock:
            return self._job.wind_down

    def lock_requests(self) -> int:
        with self._lock:
            return self._job.lock_requests

    def released(self) -> bool:
        with self._lock:
            return self._released

    def should_abort(self) -> bool:
        with self._lock:
            return self._released or self._job.wind_down is not None

    def end(self, reason: EndReason, *, retryable: bool = False) -> None:
        """Flush the final snapshot so the last game is recorded, then end the job."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        self._send()
        if self.released():
            return
        with suppress(InvalidTransitionError):
            self._queue.end(self._job_id, self._worker_id, reason, retryable=retryable)
        with self._lock:
            self._released = True


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


def _game_result(replay: ReplayEnd, result: PlayResult) -> GameResult:
    if replay.method is EndMethod.NO_CONTEST:
        return GameResult.NO_CONTEST
    if not replay.completed:
        raise RuntimeError(f"netplay replay ended via {replay.method.name}: {replay.path}")
    human_result = _human_result(result)
    return GameResult.NO_CONTEST if human_result == "tie" else GameResult(human_result)


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


def retry_pending_uploads(root: Path, endpoint: QueueEndpoint, next_attempt: float) -> float:
    now = time.monotonic()
    if now < next_attempt:
        return next_attempt
    _drain_pending_uploads(root, endpoint)
    return now + _PENDING_UPLOAD_RETRY_SECONDS


def _write_match_measurement(
    config: SlotConfig,
    job: Job,
    settings: Settings,
    game_number: int,
    result: PlayResult,
    timing: FrameTiming,
    health: SlotHealth,
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
        "game_number": game_number,
        "slot": config.slot,
        "worker_id": config.worker_id,
        "stream_id": config.stream_id,
        "generation": result.generation,
        "source_git_sha": config.git_sha,
        "graphics_backend": config.graphics_backend,
        "policy_bundle_sha256": config.policy_sha256,
        "checkpoint_sha256": config.checkpoint_sha256,
        "character": settings.character,
        "player_identity": settings.imitation,
        "desired_return": settings.desired_return,
        "temperature": settings.temperature,
        "requested_stage": settings.stage,
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
    path = directory / f"{job.id}-game-{game_number}.json"
    pending = path.with_suffix(".json.tmp")
    pending.write_text(json.dumps(payload, allow_nan=False, sort_keys=True))
    pending.replace(path)
    return path


def _write_match_failure(
    config: SlotConfig,
    job: Job,
    settings: Settings,
    game_number: int,
    timing: FrameTiming,
    health: SlotHealth,
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
        "game_number": game_number,
        "slot": config.slot,
        "worker_id": config.worker_id,
        "source_git_sha": config.git_sha,
        "graphics_backend": config.graphics_backend,
        "policy_bundle_sha256": config.policy_sha256,
        "checkpoint_sha256": config.checkpoint_sha256,
        "character": settings.character,
        "player_identity": settings.imitation,
        "desired_return": settings.desired_return,
        "temperature": settings.temperature,
        "requested_stage": settings.stage,
        "timing": asdict(timing),
        "observation_mode": "first_seen_speculative",
        "started_at": started_at.isoformat(),
        "failed_at": datetime.now(UTC).isoformat(),
        "failure": f"{type(error).__name__}: {error}",
        "progress": asdict(progress),
        "schedule": None if counters is None else asdict(counters),
        "controller_submission_gaps": None if counters is None else counters.submission_gaps,
    }
    path = directory / f"{job.id}-game-{game_number}-failure.json"
    pending = path.with_suffix(".json.tmp")
    pending.write_text(json.dumps(payload, allow_nan=False, sort_keys=True))
    pending.replace(path)


def _selection(link: ReservationLink) -> DirectSelection:
    settings = link.settings()
    stage = melee.Stage.RANDOM_STAGE if settings.stage is None else melee.Stage[settings.stage]
    return DirectSelection(
        settings.revision,
        melee.Character[settings.character],
        0,
        stage,
        settings.imitation,
    )


def run_reservation(
    config: SlotConfig,
    store: RemoteQueue,
    policy: InferenceClient,
    runtime: RuntimeConfig,
    job: Job,
    stop: StopEvent,
    health: SlotHealth,
    timing: FrameTiming,
) -> None:
    replay_dir = config.replay_dir / f"slot-{config.slot}"
    replay_dir.mkdir(parents=True, exist_ok=True)
    with ReservationLink(store, job, config.worker_id) as link:
        link.set_phase(Phase.BOOTING, None)
        link.set_bot_code(config.bot_connect_code)
        driver: DirectMenuDriver

        def mirror() -> None:
            link.set_phase(Phase(driver.phase), driver.deadline)
            link.lock(None if driver.locked is None else driver.locked.revision)

        driver = DirectMenuDriver(
            opponent_code=job.player_code,
            selection=lambda: _selection(link),
            lock_requests=link.lock_requests,
            connect_timeout_seconds=CONNECT_TIMEOUT_SECONDS,
            idle_timeout_seconds=IDLE_TIMEOUT_SECONDS,
            hold_seconds=LOCK_HOLD_SECONDS,
            hold_cap_seconds=LOCK_HOLD_CAP_SECONDS,
            probe_interval_seconds=CONNECTION_PROBE_SECONDS,
            on_change=mirror,
        )
        health.configure_schedule(timing)
        games_played = len(job.games)
        try:
            with NetplaySession(
                config.iso_path,
                dolphin_path=config.dolphin_path,
                user_json_path=config.user_json,
                online_delay=job.online_delay,
                replay_dir=replay_dir,
                slippi_port=config.slippi_port,
                step_timeout_seconds=FRAME_STALL_SECONDS,
                connect_timeout_seconds=CONNECT_TIMEOUT_SECONDS + IDLE_TIMEOUT_SECONDS,
                connect_abandoned=link.should_abort,
                realtime=True,
                graphics_backend=config.graphics_backend,
                stream_output=config.stream_output,
                menu_driver=driver,
            ) as session:
                while not stop.is_set():
                    driver.begin_game()
                    mirror()
                    health.connecting(job.online_delay)
                    current = link.settings()
                    game_number = games_played + 1
                    if config.stream_output:
                        assert config.stream_state_path is not None
                        write_stream_state(
                            config.stream_state_path,
                            GameStreamState(
                                current.character,
                                current.imitation,
                                current.desired_return,
                                game_number,
                            ),
                        )
                    previous = frozenset(replay_dir.rglob("*.slp"))
                    started_at = datetime.now(UTC)

                    def locked_settings() -> Settings:
                        if driver.locked is None:
                            raise RuntimeError("netplay reached a game before HAL locked its selection")
                        return link.settings_at(driver.locked.revision)

                    def on_live() -> None:
                        driver.connected = True
                        locked = driver.locked
                        if locked is None:
                            raise RuntimeError("netplay reached a game before HAL locked its selection")
                        settings = link.settings_at(locked.revision)
                        if session.ego_character != locked.character:
                            actual = None if session.ego_character is None else session.ego_character.name
                            raise RuntimeError(f"local player selected {actual}, expected {locked.character.name}")
                        link.lock(settings.revision)
                        link.set_phase(Phase.IN_GAME, None)
                        health.playing()

                    def player_identity() -> str | None:
                        identity = locked_settings().imitation
                        return None if identity == "MASKED" else identity

                    def on_failure(
                        progress: NetplayProgress,
                        error: BaseException,
                        game: int = game_number,
                        started: datetime = started_at,
                    ) -> None:
                        _write_match_failure(
                            config,
                            job,
                            locked_settings(),
                            game,
                            timing,
                            health,
                            started,
                            progress,
                            error,
                        )

                    result = run_netplay_match(
                        session,
                        NetplaySetup(
                            melee.Character[current.character],
                            job.player_code,
                            local_code=config.bot_connect_code,
                        ),
                        policy,
                        runtime,
                        timing,
                        player_identity=player_identity,
                        policy_settings=lambda: (
                            link.settings().desired_return,
                            link.settings().temperature,
                        ),
                        max_frames=config.max_frames,
                        rematch=games_played > len(job.games),
                        on_live=on_live,
                        on_failure=on_failure,
                        pause_seconds=PAUSE_TIMEOUT_SECONDS,
                        on_pause=lambda paused: link.set_phase(
                            Phase.PAUSED if paused else Phase.IN_GAME,
                            time.monotonic() + PAUSE_TIMEOUT_SECONDS if paused else None,
                        ),
                        observer=health,
                        schedule_observer=health,
                        stream_id=config.stream_id,
                    )
                    ended_at = datetime.now(UTC)
                    replay_end = read_new_replay_end(replay_dir, previous)
                    actual_stage = _stage_name(result.stage)
                    game_result = _game_result(replay_end, result)
                    settings = locked_settings()
                    games_played = game_number
                    link.add_game(FinishedGame(game_number, actual_stage, game_result))
                    _write_match_measurement(
                        config,
                        job,
                        settings,
                        game_number,
                        result,
                        timing,
                        health,
                        started_at,
                        ended_at,
                    )
                    limit_ms = timing.inference_allowance_frames * 1000 / 60
                    if result.inference_p95_ms >= limit_ms:
                        logger.warning(
                            "reservation {} missed the delay-{} policy deadline: p95={:.1f}ms limit={:.1f}ms",
                            job.id,
                            job.online_delay,
                            result.inference_p95_ms,
                            limit_ms,
                        )
                    if (
                        actual_stage == "POKEMON_STADIUM"
                        and settings.stage == "POKEMON_STADIUM"
                        and replay_end.is_frozen_ps is False
                    ):
                        logger.warning("reservation {} played Pokémon Stadium without Frozen Stadium", job.id)
                    logger.info(
                        "reservation {} game={} complete frames={} wall={:.1f}s fps={:.1f} stage={} human_result={}",
                        job.id,
                        game_number,
                        len(result.trajectory),
                        result.wall_seconds,
                        result.game_fps,
                        actual_stage,
                        game_result.value,
                    )
                    metadata = ReplayMetadata(
                        reservation_id=job.id,
                        player_code=job.player_code,
                        game_number=game_number,
                        actual_stage=actual_stage,
                        result=game_result.value,
                        policy_sha256=config.policy_sha256,
                        git_sha=config.git_sha,
                        started_at=started_at,
                        ended_at=ended_at,
                    )
                    sidecar = _write_pending_upload(replay_end.path, metadata, config.worker_id)
                    if config.publish_replays:
                        try:
                            _complete_pending_upload(sidecar, config.queue_endpoint)
                        except Exception as error:  # R2 availability must not end a session.
                            logger.warning("replay upload deferred: {}: {}", type(error).__name__, error)
                    if link.wind_down() is not None:
                        reason = (
                            EndReason.PLAYER_CANCELED if link.wind_down() is WindDown.PLAYER else EndReason.YIELDED
                        )
                        link.end(reason)
                        return
                    health.idle()
        except PlayerNoShow:
            link.end(EndReason.NO_SHOW)
        except PlayerDisconnected:
            link.end(EndReason.PLAYER_DISCONNECTED)
        except PlayerIdle, PausedTooLong:
            link.end(EndReason.IDLE_TIMEOUT)
        except ConnectAbandoned:
            if not link.released():
                wind_down = link.wind_down()
                if wind_down is None:
                    raise RuntimeError("connection was abandoned without release or wind-down") from None
                reason = EndReason.PLAYER_CANCELED if wind_down is WindDown.PLAYER else EndReason.YIELDED
                link.end(reason)
        except QueueUnavailableError, SessionEndedError:
            raise
        except InferenceUnavailable:
            health.recovering("inference_engine_lost")
            link.end(EndReason.SERVICE_FAILURE, retryable=True)
            raise
        except (KeyError, ValueError) as error:
            logger.error("reservation {} rejected: {}: {}", job.id, type(error).__name__, error)
            link.end(EndReason.SERVICE_FAILURE)
        except DolphinConnectionLost as error:
            health.recovering("dolphin_connection_lost")
            logger.error("reservation {}: {}", job.id, error)
            link.end(EndReason.SERVICE_FAILURE, retryable=True)
        except NoUsableActionPlan as error:
            health.recovering("no_usable_action_plan")
            logger.error("reservation {}: {}", job.id, error)
            link.end(EndReason.SERVICE_FAILURE, retryable=True)
        except FrameTimeout:
            health.recovering("frame_stream_stalled")
            link.end(EndReason.SERVICE_FAILURE, retryable=True)
        except RecoverableRuntimeError as error:
            health.recovering(str(error))
            link.end(EndReason.SERVICE_FAILURE, retryable=True)
            stop.wait(config.recovery_cooldown_seconds)
        except Exception as error:  # A reservation cannot kill a long-running slot.
            health.recovering(type(error).__name__.lower())
            logger.exception("reservation {} failed: {}", job.id, type(error).__name__)
            link.end(EndReason.SERVICE_FAILURE, retryable=True)
        finally:
            if config.stream_output:
                assert config.stream_state_path is not None
                write_stream_state(config.stream_state_path, IdleStreamState())
            with suppress(InferenceUnavailable):
                policy.close_match()
