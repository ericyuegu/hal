"""Clients for the netplay queue Worker's runner and admin routes."""

import re
import time
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Final
from typing import cast

import httpx
from loguru import logger

from hal.netplay_service.domain import Job
from hal.netplay_service.domain import JobStatus
from hal.netplay_service.domain import MatchChoices
from hal.netplay_service.domain import validate_player_code
from hal.netplay_service.domain import validate_stage
from hal.netplay_service.queue_contract import InvalidTransitionError
from hal.netplay_service.queue_contract import QueueError
from hal.netplay_service.queue_contract import SessionEndedError

# Every runner and admin route is idempotent at the Worker (a repeat after a lost
# response returns the first result), so any request may be retried. A 4xx is final.
RETRY_DELAYS_SECONDS: Final[tuple[float, ...]] = (0.25, 0.5, 1.0, 2.0, 4.0)
_TIMEOUT: Final[httpx.Timeout] = httpx.Timeout(10.0, connect=5.0)
_WORKER_ID: Final[re.Pattern[str]] = re.compile(r"(?P<session>[A-Za-z0-9_-]+)/slot-(?P<slot>[0-9]+)")
_JOB_FIELDS: Final[frozenset[str]] = frozenset(
    (
        "id",
        "player_code",
        "character",
        "imitation",
        "online_delay",
        "desired_return",
        "temperature",
        "policy_revision",
        "requested_stage",
        "status",
        "queue_position",
        "attempt",
        "game_count",
        "connect_code",
        "actual_stage",
        "last_result",
        "error_code",
        "connect_deadline",
        "rematch_deadline",
        "cancel_after_game",
    )
)


class QueueRejectedError(QueueError):
    """The queue refused a request with a 4xx other than 409 or 410."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"queue refused the request ({status}): {detail}")
        self.status = status
        self.detail = detail


class QueueUnavailableError(QueueError):
    """Connection errors or 5xx outlasted every retry."""


class QueueProtocolError(QueueError):
    """The queue returned a body this client does not understand."""


@dataclass(frozen=True, slots=True)
class QueueEndpoint:
    """Connection settings; each process builds its own client from them."""

    url: str
    token: str = field(repr=False)
    access_client_id: str | None = None
    access_client_secret: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.url.startswith(("https://", "http://")) or self.url.endswith("/"):
            raise ValueError("HAL_NETPLAY_API_URL must be an http(s) origin without a trailing slash")
        if not self.token:
            raise ValueError("the queue bearer token must be non-empty")
        if (self.access_client_id is None) != (self.access_client_secret is None):
            raise ValueError("set both CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET, or neither")
        if self.url.startswith("https://") and self.access_client_id is None:
            raise ValueError("a public queue URL needs Cloudflare Access service-token credentials")

    def headers(self) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.token}"}
        if self.access_client_id is not None and self.access_client_secret is not None:
            headers["CF-Access-Client-Id"] = self.access_client_id
            headers["CF-Access-Client-Secret"] = self.access_client_secret
        return headers


def _endpoint(environment: Mapping[str, str], token: str, access_id: str, access_secret: str) -> QueueEndpoint:
    missing = [name for name in ("HAL_NETPLAY_API_URL", token) if not environment.get(name)]
    url = environment.get("HAL_NETPLAY_API_URL", "")
    if url.startswith("https://"):
        missing += [name for name in (access_id, access_secret) if not environment.get(name)]
    if missing:
        raise ValueError(f"set {', '.join(missing)}")
    return QueueEndpoint(
        url,
        environment[token],
        environment.get(access_id) or None,
        environment.get(access_secret) or None,
    )


def runner_endpoint(environment: Mapping[str, str]) -> QueueEndpoint:
    return _endpoint(environment, "HAL_NETPLAY_RUNNER_TOKEN", "CF_ACCESS_CLIENT_ID", "CF_ACCESS_CLIENT_SECRET")


def admin_endpoint(environment: Mapping[str, str]) -> QueueEndpoint:
    """The admin tool has its own Access service token, so one .env can also hold a runner's."""
    return _endpoint(
        environment,
        "HAL_NETPLAY_ADMIN_TOKEN",
        "HAL_NETPLAY_ADMIN_ACCESS_CLIENT_ID",
        "HAL_NETPLAY_ADMIN_ACCESS_CLIENT_SECRET",
    )


def slot_worker_id(session_id: str, slot: int) -> str:
    """Match the Worker's `workerId(session, slot)`."""
    return f"{session_id}/slot-{slot}"


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text or response.reason_phrase
    if isinstance(body, dict) and isinstance(body.get("detail"), str):
        return body["detail"]
    return response.text


def _checked(response: httpx.Response) -> httpx.Response:
    # A 3xx is an error too: Cloudflare Access answers a rejected service token with a login redirect.
    if 200 <= response.status_code < 300:
        return response
    detail = _detail(response)
    if response.status_code == 409:
        raise InvalidTransitionError(detail)
    if response.status_code == 410:
        raise SessionEndedError(detail)
    raise QueueRejectedError(response.status_code, detail)


class _Api:
    """One HTTP client with bounded retries and status-code mapping."""

    def __init__(self, endpoint: QueueEndpoint, client: httpx.Client | None, sleep: Callable[[float], None]) -> None:
        self.endpoint = endpoint
        self._client = httpx.Client(base_url=endpoint.url, timeout=_TIMEOUT) if client is None else client
        self._sleep = sleep

    def close(self) -> None:
        self._client.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        body: object = None,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        merged = {**self.endpoint.headers(), **(headers or {})}
        delays: tuple[float | None, ...] = (*RETRY_DELAYS_SECONDS, None)
        for attempt, delay in enumerate(delays, start=1):
            try:
                response = self._client.request(method, path, json=body, headers=merged, params=params)
            except httpx.TransportError as error:
                failure = f"{type(error).__name__}: {error}"
            else:
                if response.status_code < 500:
                    return _checked(response)
                failure = f"HTTP {response.status_code}"
            if delay is None:
                raise QueueUnavailableError(f"{method} {path} failed after {attempt} attempts: {failure}")
            logger.bind(event="queue_retry").warning(
                "queue request {} {} failed: {}; retrying in {}s", method, path, failure, delay
            )
            self._sleep(delay)
        raise AssertionError("the retry loop always returns or raises")


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected a string, got {value!r}")
    return value


def _optional_text(value: object) -> str | None:
    return None if value is None else _text(value)


def _integer(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"expected an integer, got {value!r}")
    return value


def _optional_integer(value: object) -> int | None:
    return None if value is None else _integer(value)


def _number(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"expected a number, got {value!r}")
    return float(value)


def _optional_number(value: object) -> float | None:
    return None if value is None else _number(value)


def _boolean(value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"expected a boolean, got {value!r}")
    return value


def parse_job(payload: object) -> Job:
    """Read one job body; the player and runner routes return the same shape."""
    if not isinstance(payload, dict) or set(payload) != _JOB_FIELDS:
        raise QueueProtocolError("job response fields changed")
    fields = cast(dict[str, object], payload)
    try:
        return Job(
            id=_text(fields["id"]),
            player_code=validate_player_code(_text(fields["player_code"])),
            choices=MatchChoices(
                character=_text(fields["character"]),
                imitation=_text(fields["imitation"]),
                online_delay=_integer(fields["online_delay"]),
                requested_stage=_optional_text(fields["requested_stage"]),
                desired_return=_optional_number(fields["desired_return"]),
                temperature=_number(fields["temperature"]),
            ),
            status=JobStatus(_text(fields["status"])),
            queue_position=_optional_integer(fields["queue_position"]),
            attempt=_integer(fields["attempt"]),
            game_count=_integer(fields["game_count"]),
            connect_code=_optional_text(fields["connect_code"]),
            actual_stage=_optional_text(fields["actual_stage"]),
            last_result=_optional_text(fields["last_result"]),
            error_code=_optional_text(fields["error_code"]),
            connect_deadline=_optional_number(fields["connect_deadline"]),
            rematch_deadline=_optional_number(fields["rematch_deadline"]),
            cancel_after_game=_boolean(fields["cancel_after_game"]),
            policy_revision=_integer(fields["policy_revision"]),
        )
    except (TypeError, ValueError) as error:
        raise QueueProtocolError(f"job response contains invalid values: {error}") from error


def _json(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError as error:
        raise QueueProtocolError(f"queue response is not JSON: {response.text[:200]}") from error


class RemoteQueue:
    """The `RunnerQueue` for one session, over the Worker's runner routes."""

    def __init__(
        self,
        endpoint: QueueEndpoint,
        session_id: str,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.endpoint = endpoint
        self.session_id = session_id
        self._api = _Api(endpoint, client, sleep)

    def close(self) -> None:
        self._api.close()

    def _slot(self, worker_id: str) -> int:
        match = _WORKER_ID.fullmatch(worker_id)
        if match is None or match["session"] != self.session_id:
            raise ValueError(f"worker {worker_id!r} does not belong to session {self.session_id}")
        return int(match["slot"])

    def _slot_headers(self, worker_id: str) -> dict[str, str]:
        return {"X-HAL-Session": self.session_id, "X-HAL-Slot": str(self._slot(worker_id))}

    def _transition(self, job_id: str, worker_id: str, action: str, body: object = None) -> Job:
        response = self._api.request(
            "POST", f"/v1/runner/jobs/{job_id}/{action}", body=body, headers=self._slot_headers(worker_id)
        )
        return parse_job(_json(response))

    def claim_next(self, worker_id: str) -> Job | None:
        response = self._api.request(
            "POST", f"/v1/runner/sessions/{self.session_id}/claim", body={"slot": self._slot(worker_id)}
        )
        return None if response.status_code == 204 else parse_job(_json(response))

    def heartbeat(self, job_id: str, worker_id: str) -> None:
        self._transition(job_id, worker_id, "heartbeat")

    def mark_connecting(self, job_id: str, worker_id: str, connect_code: str) -> None:
        self._transition(job_id, worker_id, "connecting", {"connect_code": validate_player_code(connect_code)})

    def mark_playing(self, job_id: str, worker_id: str) -> None:
        self._transition(job_id, worker_id, "playing")

    def mark_no_show(self, job_id: str, worker_id: str) -> None:
        self._transition(job_id, worker_id, "no-show")

    def mark_no_contest(self, job_id: str, worker_id: str) -> None:
        self._transition(job_id, worker_id, "no-contest")

    def finish_game(
        self, job_id: str, worker_id: str, *, game_number: int, actual_stage: str, result: str
    ) -> JobStatus:
        body = {"game_number": game_number, "actual_stage": validate_stage(actual_stage), "result": result}
        return self._transition(job_id, worker_id, "finish-game", body).status

    def fail(self, job_id: str, worker_id: str, error_code: str, *, retryable: bool) -> JobStatus:
        return self._transition(job_id, worker_id, "fail", {"error_code": error_code, "retryable": retryable}).status

    def forfeit_service_failure(self, job_id: str, worker_id: str) -> None:
        self._transition(job_id, worker_id, "forfeit")

    def record_replay(
        self, job_id: str, worker_id: str, game_number: int, *, key: str, sha256: str, size: int, etag: str
    ) -> None:
        body = {"game_number": game_number, "key": key, "sha256": sha256, "size": size, "etag": etag}
        self._transition(job_id, worker_id, "replay", body)

    def get_worker_job(self, job_id: str, worker_id: str) -> Job:
        response = self._api.request("GET", f"/v1/runner/jobs/{job_id}", headers=self._slot_headers(worker_id))
        return parse_job(_json(response))
