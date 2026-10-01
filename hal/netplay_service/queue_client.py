"""Clients for the netplay queue Worker's runner and admin routes."""

import re
import secrets
import threading
import time
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Final
from typing import cast
from urllib.parse import urlsplit

import httpx
from loguru import logger

from hal.netplay_service.domain import EndReason
from hal.netplay_service.domain import FinishedGame
from hal.netplay_service.domain import GameResult
from hal.netplay_service.domain import Job
from hal.netplay_service.domain import JobStatus
from hal.netplay_service.domain import Observed
from hal.netplay_service.domain import Phase
from hal.netplay_service.domain import PolicyConfig
from hal.netplay_service.domain import Settings
from hal.netplay_service.domain import WindDown
from hal.netplay_service.domain import validate_player_code
from hal.netplay_service.health import RunnerStatus
from hal.netplay_service.queue_contract import InvalidTransitionError
from hal.netplay_service.queue_contract import QueueError
from hal.netplay_service.queue_contract import SessionEndedError

# Every runner and admin route is idempotent at the Worker (a repeat after a lost
# response returns the first result), so any request may be retried. A 4xx is final.
# Must equal RUNNER_PROTOCOL_VERSION in web/netplay-api/src/domain.ts; the two change
# together with any change to a runner route's request or response shape, because
# parse_job refuses a job body with any field added or removed.
RUNNER_PROTOCOL_VERSION: Final = 3
RETRY_DELAYS_SECONDS: Final[tuple[float, ...]] = (0.25, 0.5, 1.0, 2.0, 4.0)
_TIMEOUT: Final[httpx.Timeout] = httpx.Timeout(10.0, connect=5.0)
# The bearer token travels in the clear over http, so http is only for a Worker on this machine.
_LOOPBACK_HOSTS: Final[frozenset[str]] = frozenset(("127.0.0.1", "localhost"))
_WORKER_ID: Final[re.Pattern[str]] = re.compile(r"(?P<session>[A-Za-z0-9_-]+)/slot-(?P<slot>[0-9]+)")
_JOB_FIELDS: Final[frozenset[str]] = frozenset(
    (
        "id",
        "player_code",
        "online_delay",
        "status",
        "end_reason",
        "queue_position",
        "attempt",
        "settings",
        "observed",
        "phase_deadline",
        "games",
        "wind_down",
        "lock_requests",
    )
)
_SETTINGS_FIELDS: Final[frozenset[str]] = frozenset(
    ("revision", "character", "imitation", "stage", "desired_return", "temperature")
)
_OBSERVED_FIELDS: Final[frozenset[str]] = frozenset(("seq", "phase", "bot_code", "seen_revision", "locked_revision"))
_GAME_FIELDS: Final[frozenset[str]] = frozenset(("number", "stage", "result"))


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
        if self.url.startswith("http://") and urlsplit(self.url).hostname not in _LOOPBACK_HOSTS:
            raise ValueError("HAL_NETPLAY_API_URL may use http only for 127.0.0.1 or localhost; use https")
        if not self.token:
            raise ValueError("the queue bearer token must be non-empty")
        if (self.access_client_id is None) != (self.access_client_secret is None):
            raise ValueError("set both the Access client id and secret, or neither")
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
    if bool(environment.get(access_id)) != bool(environment.get(access_secret)):
        raise ValueError(f"set both {access_id} and {access_secret}, or neither")
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
                failure = f"HTTP {response.status_code}: {_detail(response)}"
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


def _object(payload: object, fields: frozenset[str], name: str) -> dict[str, object]:
    if not isinstance(payload, dict) or set(payload) != fields:
        raise QueueProtocolError(f"{name} response fields changed")
    return cast(dict[str, object], payload)


def parse_job(payload: object) -> Job:
    """Read one job body; the player and runner routes return the same shape."""
    if not isinstance(payload, dict) or set(payload) != _JOB_FIELDS:
        raise QueueProtocolError("job response fields changed")
    fields = cast(dict[str, object], payload)
    try:
        settings = _object(fields["settings"], _SETTINGS_FIELDS, "settings")
        observed = None if fields["observed"] is None else _object(fields["observed"], _OBSERVED_FIELDS, "observed")
        games = fields["games"]
        if not isinstance(games, list):
            raise TypeError("games must be a list")
        return Job(
            id=_text(fields["id"]),
            player_code=validate_player_code(_text(fields["player_code"])),
            online_delay=_integer(fields["online_delay"]),
            status=JobStatus(_text(fields["status"])),
            end_reason=None if fields["end_reason"] is None else EndReason(_text(fields["end_reason"])),
            queue_position=_optional_integer(fields["queue_position"]),
            attempt=_integer(fields["attempt"]),
            settings=Settings(
                revision=_integer(settings["revision"]),
                character=_text(settings["character"]),
                imitation=_text(settings["imitation"]),
                stage=_optional_text(settings["stage"]),
                desired_return=_optional_number(settings["desired_return"]),
                temperature=_number(settings["temperature"]),
            ),
            phase=None if observed is None else Phase(_text(observed["phase"])),
            phase_deadline=_optional_number(fields["phase_deadline"]),
            games=tuple(
                FinishedGame(_integer(game["number"]), _text(game["stage"]), GameResult(_text(game["result"])))
                for game in (_object(item, _GAME_FIELDS, "game") for item in games)
            ),
            wind_down=None if fields["wind_down"] is None else WindDown(_text(fields["wind_down"])),
            lock_requests=_integer(fields["lock_requests"]),
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

    def claim_next(self, worker_id: str) -> Job | None:
        response = self._api.request(
            "POST", f"/v1/runner/sessions/{self.session_id}/claim", body={"slot": self._slot(worker_id)}
        )
        return None if response.status_code == 204 else parse_job(_json(response))

    def finish_pairing(self, job_id: str, worker_id: str, attempt: int) -> None:
        slot = self._slot(worker_id)
        self._api.request(
            "POST",
            f"/v1/runner/sessions/{self.session_id}/pairing-finished",
            body={"slot": slot, "job_id": job_id, "attempt": attempt},
        )

    def report(self, job_id: str, worker_id: str, observed: Observed) -> Job:
        response = self._api.request(
            "POST",
            f"/v1/runner/jobs/{job_id}/report",
            body=observed.to_payload(),
            headers=self._slot_headers(worker_id),
        )
        return parse_job(_json(response))

    def end(self, job_id: str, worker_id: str, reason: EndReason, *, retryable: bool) -> Job:
        response = self._api.request(
            "POST",
            f"/v1/runner/jobs/{job_id}/end",
            body={"reason": reason.value, "retryable": retryable},
            headers=self._slot_headers(worker_id),
        )
        return parse_job(_json(response))

    def record_replay(
        self, job_id: str, worker_id: str, game_number: int, *, key: str, sha256: str, size: int, etag: str
    ) -> None:
        body = {"game_number": game_number, "key": key, "sha256": sha256, "size": size, "etag": etag}
        self._api.request("POST", f"/v1/runner/jobs/{job_id}/replay", body=body, headers=self._slot_headers(worker_id))

    def get_worker_job(self, job_id: str, worker_id: str) -> Job:
        response = self._api.request("GET", f"/v1/runner/jobs/{job_id}", headers=self._slot_headers(worker_id))
        return parse_job(_json(response))


@dataclass(frozen=True, slots=True)
class Account:
    connect_code: str
    r2_key: str
    sha256: str


@dataclass(frozen=True, slots=True)
class Pairing:
    slot: int
    job_id: str
    attempt: int


@dataclass(frozen=True, slots=True)
class AccountGrant:
    slot: int
    connect_code: str
    r2_key: str
    sha256: str


@dataclass(frozen=True, slots=True)
class StartedSession:
    session_id: str
    policy: PolicyConfig
    accounts: tuple[AccountGrant, ...]


@dataclass(frozen=True, slots=True)
class StreamGrant:
    slot: int
    key: str


@dataclass(frozen=True, slots=True)
class SessionState:
    draining: bool
    stream: StreamGrant | None = None


def new_session_id() -> str:
    return secrets.token_urlsafe(12)


def _grant(payload: object) -> AccountGrant:
    grant = _object(payload, frozenset(("slot", "connect_code", "r2_key", "sha256")), "account grant")
    try:
        return AccountGrant(
            slot=_integer(grant["slot"]),
            connect_code=validate_player_code(_text(grant["connect_code"])),
            r2_key=_text(grant["r2_key"]),
            sha256=_text(grant["sha256"]),
        )
    except (TypeError, ValueError) as error:
        raise QueueProtocolError(f"account grant contains invalid values: {error}") from error


class RunnerClient:
    """Session routes: active policy, start, status heartbeat, drain, and end."""

    def __init__(
        self,
        endpoint: QueueEndpoint,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._api = _Api(endpoint, client, sleep)

    def close(self) -> None:
        self._api.close()

    def active_policy(self) -> PolicyConfig:
        payload = _json(self._api.request("GET", "/v1/runner/policy"))
        try:
            return PolicyConfig.from_payload(payload)
        except ValueError as error:
            raise QueueProtocolError(f"active policy is invalid: {error}") from error

    def start_session(
        self, *, session_id: str, host: str, bundle_sha256: str, git_sha: str, slots: int, wants_stream: bool
    ) -> StartedSession:
        body = {
            "protocol_version": RUNNER_PROTOCOL_VERSION,
            "session_id": session_id,
            "host": host,
            "bundle_sha256": bundle_sha256,
            "git_sha": git_sha,
            "slots": slots,
            "stream": wants_stream,
        }
        payload = _object(
            _json(self._api.request("POST", "/v1/runner/sessions", body=body)),
            frozenset(("session_id", "accounts", "policy")),
            "session start",
        )
        accounts = payload["accounts"]
        if not isinstance(accounts, list):
            raise QueueProtocolError("session start accounts must be a list")
        grants = tuple(_grant(item) for item in accounts)
        if tuple(grant.slot for grant in grants) != tuple(range(slots)):
            raise QueueProtocolError(f"session start must assign every slot: {grants}")
        if payload["session_id"] != session_id:
            raise QueueProtocolError(f"session start returned {payload['session_id']!r}, not {session_id!r}")
        try:
            policy = PolicyConfig.from_payload(payload["policy"])
        except ValueError as error:
            raise QueueProtocolError(f"session start contains invalid values: {error}") from error
        # A repeated start returns the policy active now, which may be a newer bundle than this runner loaded.
        if policy.bundle_sha256 != bundle_sha256:
            raise QueueProtocolError(
                f"session start returned policy bundle {policy.bundle_sha256}, not the requested {bundle_sha256}"
            )
        return StartedSession(session_id, policy, grants)

    def report_status(self, session_id: str, status: RunnerStatus) -> SessionState:
        payload = _object(
            _json(self._api.request("POST", f"/v1/runner/sessions/{session_id}/status", body=status.to_payload())),
            frozenset(("draining", "stream")),
            "status",
        )
        try:
            raw_stream = payload["stream"]
            stream = None
            if raw_stream is not None:
                grant = _object(raw_stream, frozenset(("slot", "key")), "stream grant")
                slot = _integer(grant["slot"])
                key = _text(grant["key"])
                if slot != 0:
                    raise ValueError("stream grant slot must be 0")
                if not key:
                    raise ValueError("stream grant key must be non-empty")
                stream = StreamGrant(slot, key)
            return SessionState(draining=_boolean(payload["draining"]), stream=stream)
        except (TypeError, ValueError) as error:
            raise QueueProtocolError(f"status response is invalid: {error}") from error

    def queue_depth(self) -> int:
        payload = _json(self._api.request("GET", "/v1/capacity"))
        if not isinstance(payload, dict):
            raise QueueProtocolError("capacity response must be an object")
        capacity = cast(dict[str, object], payload)
        try:
            depth = _integer(capacity.get("queued"))
        except TypeError as error:
            raise QueueProtocolError(f"capacity response is invalid: {error}") from error
        if depth < 0:
            raise QueueProtocolError("capacity queued must be non-negative")
        return depth

    def pairing(self, session_id: str) -> Pairing | None:
        payload = _object(
            _json(self._api.request("GET", f"/v1/runner/sessions/{session_id}/pairing")),
            frozenset(("pairing",)),
            "pairing",
        )
        if payload["pairing"] is None:
            return None
        value = _object(payload["pairing"], frozenset(("slot", "job_id", "attempt")), "pairing")
        try:
            pairing = Pairing(_integer(value["slot"]), _text(value["job_id"]), _integer(value["attempt"]))
            if pairing.slot < 0 or not pairing.job_id or pairing.attempt < 1:
                raise ValueError("invalid pairing identity")
            return pairing
        except (TypeError, ValueError) as error:
            raise QueueProtocolError(f"pairing response is invalid: {error}") from error

    def finish_pairing(self, session_id: str, pairing: Pairing) -> None:
        self._api.request(
            "POST",
            f"/v1/runner/sessions/{session_id}/pairing-finished",
            body={"slot": pairing.slot, "job_id": pairing.job_id, "attempt": pairing.attempt},
        )

    def drain(self, session_id: str) -> None:
        self._api.request("POST", f"/v1/runner/sessions/{session_id}/drain")

    def end_session(self, session_id: str) -> int:
        payload = _object(
            _json(self._api.request("DELETE", f"/v1/runner/sessions/{session_id}")), frozenset(("failed",)), "end"
        )
        try:
            return _integer(payload["failed"])
        except TypeError as error:
            raise QueueProtocolError(f"end response is invalid: {error}") from error


class SessionReporter:
    """Report runner status in the background; the report is also the session heartbeat.

    Enter it right after the session starts. The Worker ends a session after 30 s
    without a report, and downloads and qualification can take longer than that.
    """

    def __init__(
        self,
        client: RunnerClient,
        session_id: str,
        status: Callable[[], RunnerStatus],
        *,
        interval_seconds: float = 2.0,
    ) -> None:
        self._client = client
        self._session_id = session_id
        self._status = status
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._state: SessionState | None = None
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, name=f"session-{session_id}", daemon=True)

    def __enter__(self) -> SessionReporter:
        # The first report is synchronous and strict, so startup fails fast on any queue error.
        self._state = self._client.report_status(self._session_id, self._status())
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()

    def state(self) -> SessionState:
        with self._lock:
            if self._error is not None:
                raise self._error
            assert self._state is not None, "enter the reporter before reading its state"
            return self._state

    def _report(self) -> None:
        try:
            state = self._client.report_status(self._session_id, self._status())
        except QueueUnavailableError as error:
            # The Worker tolerates 30 s of silence; the next tick tries again.
            logger.bind(event="session_status").warning("session status report failed: {}", error)
            return
        except Exception as error:
            # Hand the failure to the owner thread, which re-raises it from state(); a daemon
            # thread that died here would otherwise leave state() returning a stale state.
            with self._lock:
                self._error = error
            self._stop.set()
            return
        with self._lock:
            self._state = state

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self._report()


class AdminClient:
    """Admin routes: policy, accounts, pause, status, and events."""

    def __init__(
        self,
        endpoint: QueueEndpoint,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._api = _Api(endpoint, client, sleep)

    def close(self) -> None:
        self._api.close()

    def put_policy(self, config: PolicyConfig) -> None:
        self._api.request("PUT", "/v1/admin/policy", body=config.to_payload())

    def put_accounts(self, accounts: Sequence[Account]) -> None:
        body = [{"connect_code": a.connect_code, "r2_key": a.r2_key, "sha256": a.sha256} for a in accounts]
        self._api.request("PUT", "/v1/admin/accounts", body=body)

    def set_paused(self, paused: bool) -> None:
        self._api.request("POST", "/v1/admin/pause" if paused else "/v1/admin/resume")

    def status(self) -> dict[str, object]:
        payload = _json(self._api.request("GET", "/v1/admin/status"))
        if not isinstance(payload, dict):
            raise QueueProtocolError("admin status must be an object")
        return cast(dict[str, object], payload)

    def events(self, *, job: str | None, session: str | None, since: float | None) -> list[dict[str, object]]:
        params = {
            name: value
            for name, value in (("job", job), ("session", session), ("since", None if since is None else str(since)))
            if value is not None
        }
        payload = _object(
            _json(self._api.request("GET", "/v1/admin/events", params=params)), frozenset(("events",)), "events"
        )
        events = payload["events"]
        if not isinstance(events, list) or not all(isinstance(event, dict) for event in events):
            raise QueueProtocolError("admin events must be a list of objects")
        return cast(list[dict[str, object]], events)
