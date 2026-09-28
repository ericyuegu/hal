"""HTTP API for the HAL netplay queue."""

import argparse
import asyncio
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import Depends
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.security import HTTPBearer
from loguru import logger
from prometheus_client import CollectorRegistry
from prometheus_client import Counter
from prometheus_client import Gauge
from prometheus_client import generate_latest
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from starlette.middleware.base import RequestResponseEndpoint
from starlette.middleware.trustedhost import TrustedHostMiddleware

from hal.inference.bundle import read_policy_manifest
from hal.netplay_service.domain import CHARACTERS
from hal.netplay_service.domain import IDLE_TIMEOUT_SECONDS
from hal.netplay_service.domain import IMITATIONS
from hal.netplay_service.domain import STAGES
from hal.netplay_service.domain import Choice
from hal.netplay_service.domain import Job
from hal.netplay_service.domain import MatchChoices
from hal.netplay_service.health import RUNNER_HEARTBEAT_MAX_AGE_SECONDS
from hal.netplay_service.health import TARGET_GAME_FPS
from hal.netplay_service.health import RunnerState
from hal.netplay_service.health import RunnerStatus
from hal.netplay_service.health import read_runner_status
from hal.netplay_service.queue import ActiveJobError
from hal.netplay_service.queue import AuthenticationError
from hal.netplay_service.queue import InvalidTransitionError
from hal.netplay_service.queue import QueueStore


@dataclass(frozen=True, slots=True)
class ApiConfig:
    database: Path
    capacity: int = 2
    allowed_origins: tuple[str, ...] = ("http://localhost:3000",)
    allowed_hosts: tuple[str, ...] = ("localhost", "127.0.0.1")
    runner_status: Path | None = None
    supported_delays: tuple[int, ...] = (2, 3)
    masked_identity: bool = False

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be positive")
        if not self.allowed_origins or "*" in self.allowed_origins:
            raise ValueError("allowed_origins must be explicit")
        if not self.allowed_hosts:
            raise ValueError("allowed_hosts must be non-empty")

    @classmethod
    def from_env(cls, database: str | Path | None = None) -> ApiConfig:
        origins = tuple(
            filter(None, os.environ.get("HAL_NETPLAY_ALLOWED_ORIGINS", "http://localhost:3000").split(","))
        )
        hosts = tuple(filter(None, os.environ.get("HAL_NETPLAY_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")))
        policy_path = os.environ.get("HAL_NETPLAY_POLICY")
        manifest = read_policy_manifest(policy_path) if policy_path else None
        return cls(
            database=Path(database or os.environ.get("HAL_NETPLAY_DATABASE", "runs/netplay/queue.sqlite3")),
            capacity=int(os.environ.get("HAL_NETPLAY_CAPACITY", "2")),
            allowed_origins=origins,
            allowed_hosts=hosts,
            runner_status=(Path(value).resolve() if (value := os.environ.get("HAL_NETPLAY_RUNNER_STATUS")) else None),
            supported_delays=(2,) if manifest is not None and manifest.backend == "o59-history-decoder" else (2, 3),
            masked_identity=manifest is not None and manifest.backend == "o59-history-decoder",
        )


class ChoiceResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    value: str
    label: str


class OptionsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    characters: tuple[ChoiceResponse, ...]
    imitations: tuple[ChoiceResponse, ...]
    stages: tuple[ChoiceResponse, ...]
    online_delays: tuple[int, ...] = (2, 3)
    desired_return_range: tuple[float, float] = (0.0, 40.0)
    default_desired_return: float = 20.0
    temperature_range: tuple[float, float] = (0.8, 1.1)
    default_temperature: float = 1.0
    max_games: int = 5
    no_show_seconds: int = IDLE_TIMEOUT_SECONDS
    rematch_seconds: int = IDLE_TIMEOUT_SECONDS


class CreateJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    player_code: str = Field(min_length=3, max_length=13)
    character: str
    imitation: str
    online_delay: int
    desired_return: float | None = Field(default=20.0, ge=0, le=40)
    temperature: float = Field(default=1.0, ge=0.8, le=1.1)


class UpdatePolicyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    desired_return: float | None = Field(default=None, ge=0, le=40)
    temperature: float | None = Field(default=None, ge=0.8, le=1.1)


class RematchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    character: str
    imitation: str
    stage: str


class JobResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    player_code: str
    character: str
    imitation: str
    online_delay: int
    desired_return: float | None
    temperature: float
    policy_revision: int
    requested_stage: str | None
    status: str
    queue_position: int | None
    attempt: int
    game_count: int
    connect_code: str | None
    actual_stage: str | None
    last_result: str | None
    error_code: str | None
    connect_deadline: float | None
    rematch_deadline: float | None
    cancel_after_game: bool


class CreatedJobResponse(JobResponse):
    token: str


class CapacityResponse(BaseModel):
    capacity: int
    healthy_slots: int
    active: int
    queued: int
    service_status: str
    service_message: str
    target_fps: float
    game_fps: float | None
    frame_interval_p95_ms: float | None
    dolphin_step_p95_ms: float | None
    policy_round_trip_p95_ms: float | None
    model_inference_p95_ms: float | None
    batch_wait_p95_ms: float | None
    recoveries: int


class _RunnerUnavailableError(RuntimeError):
    pass


def _choices(values: tuple[Choice, ...]) -> tuple[ChoiceResponse, ...]:
    return tuple(ChoiceResponse(value=choice.value, label=choice.label) for choice in values)


def _job(job: Job) -> JobResponse:
    return JobResponse(
        id=job.id,
        player_code=job.player_code,
        character=job.choices.character,
        imitation=job.choices.imitation,
        online_delay=job.choices.online_delay,
        desired_return=job.choices.desired_return,
        temperature=job.choices.temperature,
        policy_revision=job.policy_revision,
        requested_stage=job.choices.requested_stage,
        status=job.status.value,
        queue_position=job.queue_position,
        attempt=job.attempt,
        game_count=job.game_count,
        connect_code=job.connect_code,
        actual_stage=job.actual_stage,
        last_result=job.last_result,
        error_code=job.error_code,
        connect_deadline=job.connect_deadline,
        rematch_deadline=job.rematch_deadline,
        cancel_after_game=job.cancel_after_game,
    )


def _runner_health(config: ApiConfig) -> RunnerStatus:
    if config.runner_status is None:
        return RunnerStatus(
            state=RunnerState.READY,
            message="Game servers are ready.",
            policy_sha256="0" * 64,
            slots=config.capacity,
            healthy_slots=config.capacity,
            target_fps=TARGET_GAME_FPS,
            game_fps=None,
            frame_interval_p95_ms=None,
            dolphin_step_p95_ms=None,
            policy_round_trip_p95_ms=None,
            model_inference_p95_ms=None,
            batch_wait_p95_ms=None,
            recoveries=0,
            updated_at=time.time(),
        )
    try:
        status = read_runner_status(config.runner_status)
    except ValueError as error:
        raise _RunnerUnavailableError("runner status is unavailable") from error
    age = time.time() - status.updated_at
    if not -RUNNER_HEARTBEAT_MAX_AGE_SECONDS <= age <= RUNNER_HEARTBEAT_MAX_AGE_SECONDS:
        raise _RunnerUnavailableError("runner heartbeat is stale")
    if status.slots != config.capacity:
        raise _RunnerUnavailableError(
            f"runner has {status.slots} slots, but the API is configured for {config.capacity}"
        )
    return status


def _capacity(config: ApiConfig, queue: QueueStore) -> CapacityResponse:
    try:
        status = _runner_health(config)
    except _RunnerUnavailableError:
        return CapacityResponse(
            capacity=config.capacity,
            healthy_slots=0,
            active=queue.active_count(),
            queued=queue.queue_depth(),
            service_status=RunnerState.UNAVAILABLE.value,
            service_message="Game servers are unavailable. Try again shortly.",
            target_fps=TARGET_GAME_FPS,
            game_fps=None,
            frame_interval_p95_ms=None,
            dolphin_step_p95_ms=None,
            policy_round_trip_p95_ms=None,
            model_inference_p95_ms=None,
            batch_wait_p95_ms=None,
            recoveries=0,
        )
    return CapacityResponse(
        capacity=config.capacity,
        healthy_slots=status.healthy_slots,
        active=queue.active_count(),
        queued=queue.queue_depth(),
        service_status=status.state.value,
        service_message=status.message,
        target_fps=status.target_fps,
        game_fps=status.game_fps,
        frame_interval_p95_ms=status.frame_interval_p95_ms,
        dolphin_step_p95_ms=status.dolphin_step_p95_ms,
        policy_round_trip_p95_ms=status.policy_round_trip_p95_ms,
        model_inference_p95_ms=status.model_inference_p95_ms,
        batch_wait_p95_ms=status.batch_wait_p95_ms,
        recoveries=status.recoveries,
    )


_BEARER = HTTPBearer(auto_error=False)


def _require_job_token(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_BEARER)]) -> str:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="job token is required")
    return credentials.credentials


class _ApiLifecycle:
    """Own the queue background task, route handlers, and per-app metrics."""

    def __init__(self, config: ApiConfig, queue: QueueStore) -> None:
        self.config = config
        self.queue = queue
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "hal_netplay_http_requests_total", "HTTP requests", ("method", "path", "status"), registry=self.registry
        )
        self.queue_gauge = Gauge("hal_netplay_queue_depth", "Queued reservations", registry=self.registry)
        self.active_gauge = Gauge("hal_netplay_active_reservations", "Active reservations", registry=self.registry)
        self.healthy_slots_gauge = Gauge("hal_netplay_healthy_slots", "Healthy Dolphin slots", registry=self.registry)
        self.game_fps_gauge = Gauge("hal_netplay_game_fps", "Lowest recent Dolphin frame rate", registry=self.registry)
        self.frame_p95_gauge = Gauge(
            "hal_netplay_frame_interval_p95_ms",
            "Highest recent p95 Dolphin frame interval",
            registry=self.registry,
        )
        self.dolphin_p95_gauge = Gauge(
            "hal_netplay_dolphin_step_p95_ms",
            "Highest recent p95 blocking Dolphin step",
            registry=self.registry,
        )
        self.policy_p95_gauge = Gauge(
            "hal_netplay_policy_round_trip_p95_ms",
            "Highest recent p95 worker-to-policy round trip",
            registry=self.registry,
        )
        self.model_p95_gauge = Gauge(
            "hal_netplay_model_inference_p95_ms",
            "Recent p95 model inference time",
            registry=self.registry,
        )
        self.batch_wait_p95_gauge = Gauge(
            "hal_netplay_batch_wait_p95_ms",
            "Recent p95 request-coalescing wait",
            registry=self.registry,
        )
        self.recoveries_gauge = Gauge(
            "hal_netplay_slot_recoveries", "Automatic Dolphin slot recoveries", registry=self.registry
        )

    async def reap(self) -> None:
        while True:
            await asyncio.sleep(1)
            expired = await asyncio.to_thread(self.queue.reap_expired)
            if expired:
                logger.info("expired {} stale netplay reservations", expired)

    @asynccontextmanager
    async def lifespan(self, _app: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(self.reap())
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def secure_requests(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                length = int(content_length)
            except ValueError:
                return Response(status_code=400)
            if length < 0:
                return Response(status_code=400)
            if length > 16 * 1024:
                return Response(status_code=413)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        self.requests.labels(request.method, request.url.path, response.status_code).inc()
        return response

    def live(self) -> dict[str, str]:
        return {"status": "ok"}

    def ready(self) -> dict[str, str]:
        self.queue.queue_depth()
        try:
            _runner_health(self.config)
        except _RunnerUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return {"status": "ready"}

    def metrics(self) -> Response:
        capacity_status = _capacity(self.config, self.queue)
        self.queue_gauge.set(capacity_status.queued)
        self.active_gauge.set(capacity_status.active)
        self.healthy_slots_gauge.set(capacity_status.healthy_slots)
        self.game_fps_gauge.set(capacity_status.game_fps or 0.0)
        self.frame_p95_gauge.set(capacity_status.frame_interval_p95_ms or 0.0)
        self.dolphin_p95_gauge.set(capacity_status.dolphin_step_p95_ms or 0.0)
        self.policy_p95_gauge.set(capacity_status.policy_round_trip_p95_ms or 0.0)
        self.model_p95_gauge.set(capacity_status.model_inference_p95_ms or 0.0)
        self.batch_wait_p95_gauge.set(capacity_status.batch_wait_p95_ms or 0.0)
        self.recoveries_gauge.set(capacity_status.recoveries)
        return Response(generate_latest(self.registry), media_type=CONTENT_TYPE_LATEST)

    def options(self) -> OptionsResponse:
        return OptionsResponse(
            characters=_choices(CHARACTERS),
            imitations=_choices(IMITATIONS if self.config.masked_identity else IMITATIONS[1:]),
            stages=_choices(STAGES),
            online_delays=self.config.supported_delays,
        )

    def capacity(self) -> CapacityResponse:
        return _capacity(self.config, self.queue)

    def create_job(self, body: CreateJobRequest) -> CreatedJobResponse:
        if body.online_delay not in self.config.supported_delays:
            raise HTTPException(status_code=422, detail="online delay is unsupported by this policy")
        if body.imitation == "MASKED" and not self.config.masked_identity:
            raise HTTPException(status_code=422, detail="masked identity is unsupported by this policy")
        try:
            status = _runner_health(self.config)
        except _RunnerUnavailableError as error:
            raise HTTPException(status_code=503, detail="Game servers are unavailable. Try again shortly.") from error
        if status.healthy_slots == 0:
            raise HTTPException(status_code=503, detail=status.message)
        try:
            credentials = self.queue.create_job(
                body.player_code,
                MatchChoices(
                    body.character,
                    body.imitation,
                    body.online_delay,
                    desired_return=body.desired_return,
                    temperature=body.temperature,
                ),
            )
        except ActiveJobError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        logger.info(
            "reservation {} queued player={} position={} character={} delay={} imitate={}",
            credentials.job.id,
            credentials.job.player_code,
            credentials.job.queue_position,
            credentials.job.choices.character,
            credentials.job.choices.online_delay,
            credentials.job.choices.imitation,
        )
        return CreatedJobResponse(**_job(credentials.job).model_dump(), token=credentials.token)

    def get_job(self, job_id: str, job_token: Annotated[str, Depends(_require_job_token)]) -> JobResponse:
        try:
            return _job(self.queue.get_job(job_id, job_token))
        except AuthenticationError as error:
            raise HTTPException(status_code=404, detail="job not found") from error

    def update_policy(
        self, job_id: str, body: UpdatePolicyRequest, job_token: Annotated[str, Depends(_require_job_token)]
    ) -> JobResponse:
        if not body.model_fields_set:
            raise HTTPException(status_code=422, detail="provide desired_return or temperature")
        try:
            current = self.queue.get_job(job_id, job_token)
            target = (
                body.desired_return if "desired_return" in body.model_fields_set else current.choices.desired_return
            )
            sampling = body.temperature if "temperature" in body.model_fields_set else current.choices.temperature
            if sampling is None:
                raise ValueError("temperature cannot be null")
            return _job(self.queue.update_policy(job_id, job_token, desired_return=target, temperature=sampling))
        except AuthenticationError as error:
            raise HTTPException(status_code=404, detail="job not found") from error
        except InvalidTransitionError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    def cancel_job(self, job_id: str, job_token: Annotated[str, Depends(_require_job_token)]) -> JobResponse:
        try:
            job = self.queue.cancel(job_id, job_token)
        except AuthenticationError as error:
            raise HTTPException(status_code=404, detail="job not found") from error
        logger.info("reservation {} cancel requested status={}", job.id, job.status.value)
        return _job(job)

    def rematch(
        self, job_id: str, body: RematchRequest, job_token: Annotated[str, Depends(_require_job_token)]
    ) -> JobResponse:
        try:
            job = self.queue.request_rematch(
                job_id,
                job_token,
                character=body.character,
                imitation=body.imitation,
                stage=body.stage,
            )
        except AuthenticationError as error:
            raise HTTPException(status_code=404, detail="job not found") from error
        except InvalidTransitionError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        logger.info(
            "reservation {} rematch ready character={} stage={} imitate={}",
            job.id,
            job.choices.character,
            job.choices.requested_stage,
            job.choices.imitation,
        )
        return _job(job)


def create_app(config: ApiConfig, store: QueueStore | None = None) -> FastAPI:
    queue = QueueStore(config.database) if store is None else store
    api = _ApiLifecycle(config, queue)
    app = FastAPI(title="HAL Netplay", version="1", docs_url=None, redoc_url=None, lifespan=api.lifespan)
    app.state.queue = queue
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(config.allowed_hosts))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
        max_age=600,
    )
    app.middleware("http")(api.secure_requests)
    app.get("/health/live", include_in_schema=False)(api.live)
    app.get("/health/ready", include_in_schema=False)(api.ready)
    app.get("/metrics", include_in_schema=False)(api.metrics)
    app.get("/v1/options", response_model=OptionsResponse)(api.options)
    app.get("/v1/capacity", response_model=CapacityResponse)(api.capacity)
    app.post("/v1/jobs", response_model=CreatedJobResponse, status_code=201)(api.create_job)
    app.get("/v1/jobs/{job_id}", response_model=JobResponse)(api.get_job)
    app.patch("/v1/jobs/{job_id}/policy", response_model=JobResponse)(api.update_policy)
    app.delete("/v1/jobs/{job_id}", response_model=JobResponse)(api.cancel_job)
    app.post("/v1/jobs/{job_id}/rematch", response_model=JobResponse)(api.rematch)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(prog="hal-netplay-api")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--host", default=os.environ.get("HAL_NETPLAY_API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("HAL_NETPLAY_API_PORT", "8080")))
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(create_app(ApiConfig.from_env(args.database)), host=args.host, port=args.port, proxy_headers=True)


if __name__ == "__main__":
    main()
