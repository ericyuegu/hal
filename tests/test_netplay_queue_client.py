import json

import httpx
import pytest

from hal.netplay_service.domain import JobStatus
from hal.netplay_service.queue_client import RETRY_DELAYS_SECONDS
from hal.netplay_service.queue_client import QueueEndpoint
from hal.netplay_service.queue_client import QueueProtocolError
from hal.netplay_service.queue_client import QueueRejectedError
from hal.netplay_service.queue_client import QueueUnavailableError
from hal.netplay_service.queue_client import RemoteQueue
from hal.netplay_service.queue_client import admin_endpoint
from hal.netplay_service.queue_client import parse_job
from hal.netplay_service.queue_client import runner_endpoint
from hal.netplay_service.queue_client import slot_worker_id
from hal.netplay_service.queue_contract import InvalidTransitionError
from hal.netplay_service.queue_contract import SessionEndedError

ENDPOINT = QueueEndpoint("https://20xx.xyz", "runner-token", "cf-id", "cf-secret")
WORKER = slot_worker_id("sess", 1)


def _job_body(**changes: object) -> dict[str, object]:
    body: dict[str, object] = {
        "id": "job-1",
        "player_code": "CRYO#610",
        "character": "FOX",
        "imitation": "IBDW#0",
        "online_delay": 2,
        "desired_return": 20,
        "temperature": 1,
        "policy_revision": 0,
        "requested_stage": None,
        "status": "leased",
        "queue_position": None,
        "attempt": 1,
        "game_count": 0,
        "connect_code": None,
        "actual_stage": None,
        "last_result": None,
        "error_code": None,
        "connect_deadline": None,
        "rematch_deadline": None,
        "cancel_after_game": False,
    }
    body.update(changes)
    return body


class _Script:
    """Answer requests in order with scripted responses or transport errors."""

    def __init__(self, *outcomes: httpx.Response | httpx.TransportError) -> None:
        self.requests: list[httpx.Request] = []
        self._outcomes = list(outcomes)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, httpx.TransportError):
            raise outcome
        return outcome


def _queue(script: _Script, sleeps: list[float]) -> RemoteQueue:
    client = httpx.Client(base_url=ENDPOINT.url, transport=httpx.MockTransport(script))
    return RemoteQueue(ENDPOINT, "sess", client=client, sleep=sleeps.append)


def test_claim_sends_auth_access_and_slot_and_maps_204_to_none() -> None:
    script = _Script(httpx.Response(204))
    assert _queue(script, []).claim_next(WORKER) is None
    request = script.requests[0]
    assert (request.method, request.url.path) == ("POST", "/v1/runner/sessions/sess/claim")
    assert json.loads(request.content) == {"slot": 1}
    assert request.headers["Authorization"] == "Bearer runner-token"
    assert request.headers["CF-Access-Client-Id"] == "cf-id"
    assert request.headers["CF-Access-Client-Secret"] == "cf-secret"


def test_job_routes_name_the_session_and_slot() -> None:
    script = _Script(httpx.Response(200, json=_job_body(status="rematch_wait", game_count=1)))
    status = _queue(script, []).finish_game("job-1", WORKER, game_number=1, actual_stage="BATTLEFIELD", result="win")
    assert status is JobStatus.REMATCH_WAIT
    request = script.requests[0]
    assert request.url.path == "/v1/runner/jobs/job-1/finish-game"
    assert (request.headers["X-HAL-Session"], request.headers["X-HAL-Slot"]) == ("sess", "1")
    assert json.loads(request.content) == {"game_number": 1, "actual_stage": "BATTLEFIELD", "result": "win"}


def test_connection_errors_and_5xx_retry_with_backoff_then_succeed() -> None:
    sleeps: list[float] = []
    script = _Script(
        httpx.ConnectError("refused"),
        httpx.Response(502),
        httpx.Response(200, json=_job_body(status="connecting", connect_code="HAL#1")),
    )
    _queue(script, sleeps).mark_connecting("job-1", WORKER, "HAL#1")
    assert sleeps == [0.25, 0.5]
    assert len(script.requests) == 3


def test_retries_are_bounded() -> None:
    sleeps: list[float] = []
    script = _Script(*(httpx.Response(503) for _ in range(len(RETRY_DELAYS_SECONDS) + 1)))
    with pytest.raises(QueueUnavailableError, match="after 6 attempts: HTTP 503"):
        _queue(script, sleeps).heartbeat("job-1", WORKER)
    assert sleeps == list(RETRY_DELAYS_SECONDS)


@pytest.mark.parametrize(
    ("status", "error"),
    [(409, InvalidTransitionError), (410, SessionEndedError), (422, QueueRejectedError), (401, QueueRejectedError)],
)
def test_4xx_is_not_retried_and_maps_to_errors(status: int, error: type[Exception]) -> None:
    sleeps: list[float] = []
    script = _Script(httpx.Response(status, json={"detail": "worker does not own this job"}))
    with pytest.raises(error, match="worker does not own this job"):
        _queue(script, sleeps).mark_playing("job-1", WORKER)
    assert sleeps == []
    assert len(script.requests) == 1


def test_redirect_is_rejected_not_parsed() -> None:
    script = _Script(httpx.Response(302, headers={"Location": "https://20xx.cloudflareaccess.com/login"}))
    with pytest.raises(QueueRejectedError, match=r"\(302\)"):
        _queue(script, []).claim_next(WORKER)
    assert len(script.requests) == 1


def test_session_end_is_a_lost_lease() -> None:
    assert issubclass(SessionEndedError, InvalidTransitionError)


def test_worker_from_another_session_is_rejected_before_a_request() -> None:
    script = _Script()
    with pytest.raises(ValueError, match="does not belong to session sess"):
        _queue(script, []).mark_no_show("job-1", slot_worker_id("other", 0))
    assert script.requests == []


def test_parse_job_rejects_drift() -> None:
    job = parse_job(_job_body(desired_return=None))
    assert job.choices.desired_return is None and job.status is JobStatus.LEASED
    with pytest.raises(QueueProtocolError, match="fields changed"):
        parse_job({**_job_body(), "lease_owner": "x"})
    with pytest.raises(QueueProtocolError, match="invalid values"):
        parse_job(_job_body(status="paused"))
    with pytest.raises(QueueProtocolError, match="invalid values"):
        parse_job(_job_body(attempt=True))


def test_endpoint_rejects_misconfiguration_and_hides_secrets() -> None:
    with pytest.raises(ValueError, match="trailing slash"):
        QueueEndpoint("https://20xx.xyz/", "t", "id", "secret")
    with pytest.raises(ValueError, match="both"):
        QueueEndpoint("https://20xx.xyz", "t", "id")
    with pytest.raises(ValueError, match="HAL_NETPLAY_RUNNER_TOKEN"):
        runner_endpoint({"HAL_NETPLAY_API_URL": "http://127.0.0.1:8787"})
    with pytest.raises(ValueError, match="CF_ACCESS_CLIENT_ID"):
        runner_endpoint({"HAL_NETPLAY_API_URL": "https://20xx.xyz", "HAL_NETPLAY_RUNNER_TOKEN": "t"})
    with pytest.raises(ValueError, match="HAL_NETPLAY_ADMIN_ACCESS_CLIENT_ID"):
        admin_endpoint({"HAL_NETPLAY_API_URL": "https://20xx.xyz", "HAL_NETPLAY_ADMIN_TOKEN": "t"})
    local = runner_endpoint(
        {"HAL_NETPLAY_API_URL": "http://127.0.0.1:8787", "HAL_NETPLAY_RUNNER_TOKEN": "dev", "CF_ACCESS_CLIENT_ID": ""}
    )
    assert local.headers() == {"Authorization": "Bearer dev"}
    admin = admin_endpoint(
        {
            "HAL_NETPLAY_API_URL": "https://20xx.xyz",
            "HAL_NETPLAY_ADMIN_TOKEN": "a",
            "HAL_NETPLAY_ADMIN_ACCESS_CLIENT_ID": "admin-id",
            "HAL_NETPLAY_ADMIN_ACCESS_CLIENT_SECRET": "admin-secret",
            "CF_ACCESS_CLIENT_ID": "runner-id",
            "CF_ACCESS_CLIENT_SECRET": "runner-secret",
        }
    )
    assert admin.headers()["CF-Access-Client-Id"] == "admin-id"
    assert "cf-secret" not in repr(ENDPOINT) and "runner-token" not in repr(ENDPOINT)
