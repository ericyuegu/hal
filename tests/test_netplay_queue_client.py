import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from hal.netplay_service.domain import CHARACTERS
from hal.netplay_service.domain import IMITATIONS
from hal.netplay_service.domain import STAGES
from hal.netplay_service.domain import EndReason
from hal.netplay_service.domain import FinishedGame
from hal.netplay_service.domain import GameResult
from hal.netplay_service.domain import Observed
from hal.netplay_service.domain import Phase
from hal.netplay_service.domain import PolicyConfig
from hal.netplay_service.domain import Settings
from hal.netplay_service.health import RunnerStatus
from hal.netplay_service.health import SlotState
from hal.netplay_service.health import SlotStatus
from hal.netplay_service.health import aggregate_runner_status
from hal.netplay_service.queue_client import RETRY_DELAYS_SECONDS
from hal.netplay_service.queue_client import RUNNER_PROTOCOL_VERSION
from hal.netplay_service.queue_client import Account
from hal.netplay_service.queue_client import AdminClient
from hal.netplay_service.queue_client import Pairing
from hal.netplay_service.queue_client import QueueEndpoint
from hal.netplay_service.queue_client import QueueProtocolError
from hal.netplay_service.queue_client import QueueRejectedError
from hal.netplay_service.queue_client import QueueUnavailableError
from hal.netplay_service.queue_client import RemoteQueue
from hal.netplay_service.queue_client import RunnerClient
from hal.netplay_service.queue_client import SessionReporter
from hal.netplay_service.queue_client import SessionState
from hal.netplay_service.queue_client import StreamGrant
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
        "online_delay": 2,
        "status": "assigned",
        "end_reason": None,
        "queue_position": None,
        "attempt": 1,
        "settings": {
            "revision": 2,
            "character": "FALCO",
            "imitation": "MANG#0",
            "stage": None,
            "desired_return": 20.0,
            "temperature": 1.0,
        },
        "observed": {
            "seq": 3,
            "phase": "in_game",
            "bot_code": "HAL#9000",
            "seen_revision": 2,
            "locked_revision": 2,
        },
        "phase_deadline": None,
        "games": [{"number": 1, "stage": "BATTLEFIELD", "result": "win"}],
        "wind_down": None,
        "lock_requests": 0,
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


def test_report_posts_the_snapshot_with_slot_headers() -> None:
    script = _Script(httpx.Response(200, json=_job_body()))
    observed = Observed(
        4,
        Phase.CHARACTER_SELECT,
        3.0,
        "HAL#9000",
        2,
        1,
        (FinishedGame(1, "BATTLEFIELD", GameResult.WIN),),
    )
    job = _queue(script, []).report("job-1", WORKER, observed)
    assert job.id == "job-1"
    request = script.requests[0]
    assert request.url.path == "/v1/runner/jobs/job-1/report"
    assert (request.headers["X-HAL-Session"], request.headers["X-HAL-Slot"]) == ("sess", "1")
    assert json.loads(request.content) == observed.to_payload()


def test_connection_errors_and_5xx_retry_with_backoff_then_succeed() -> None:
    sleeps: list[float] = []
    script = _Script(
        httpx.ConnectError("refused"),
        httpx.Response(502),
        httpx.Response(200, json=_job_body()),
    )
    observed = Observed(1, Phase.BOOTING, None, None, 1, None, ())
    _queue(script, sleeps).report("job-1", WORKER, observed)
    assert sleeps == [0.25, 0.5]
    assert len(script.requests) == 3


def test_retries_are_bounded() -> None:
    sleeps: list[float] = []
    script = _Script(
        *(
            httpx.Response(503, json={"detail": "no policy has been published"})
            for _ in range(len(RETRY_DELAYS_SECONDS) + 1)
        )
    )
    with pytest.raises(QueueUnavailableError, match="after 6 attempts: HTTP 503: no policy has been published$"):
        _queue(script, sleeps).report("job-1", WORKER, Observed(1, Phase.BOOTING, None, None, 1, None, ()))
    assert sleeps == list(RETRY_DELAYS_SECONDS)


@pytest.mark.parametrize(
    ("status", "error"),
    [(409, InvalidTransitionError), (410, SessionEndedError), (422, QueueRejectedError), (401, QueueRejectedError)],
)
def test_4xx_is_not_retried_and_maps_to_errors(status: int, error: type[Exception]) -> None:
    sleeps: list[float] = []
    script = _Script(httpx.Response(status, json={"detail": "worker does not own this job"}))
    with pytest.raises(error, match="worker does not own this job"):
        _queue(script, sleeps).end("job-1", WORKER, EndReason.NO_SHOW, retryable=False)
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
        _queue(script, []).end("job-1", slot_worker_id("other", 0), EndReason.NO_SHOW, retryable=False)
    assert script.requests == []


def test_parse_job_reads_protocol_3() -> None:
    job = parse_job(_job_body())
    assert job.settings == Settings(2, "FALCO", "MANG#0", None, 20.0, 1.0)
    assert job.phase is Phase.IN_GAME
    assert job.games == (FinishedGame(1, "BATTLEFIELD", GameResult.WIN),)


def test_parse_job_rejects_drift() -> None:
    with pytest.raises(QueueProtocolError, match="fields changed"):
        parse_job({**_job_body(), "lease_owner": "x"})
    with pytest.raises(QueueProtocolError, match="invalid values"):
        parse_job(_job_body(status="playing"))
    with pytest.raises(QueueProtocolError, match="invalid values"):
        parse_job(_job_body(attempt=True))


def test_end_maps_409_to_invalid_transition() -> None:
    script = _Script(httpx.Response(409, json={"detail": "worker does not own this job"}))
    with pytest.raises(InvalidTransitionError):
        _queue(script, []).end("job-1", WORKER, EndReason.NO_SHOW, retryable=False)


@pytest.mark.parametrize(
    "url", ["http://20xx.xyz", "http://10.0.0.61:8787", "http://localhost.20xx.xyz", "http://127.0.0.1@20xx.xyz"]
)
def test_endpoint_refuses_http_to_another_host(url: str) -> None:
    with pytest.raises(ValueError, match="http only for 127.0.0.1 or localhost"):
        QueueEndpoint(url, "t")


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1", "http://127.0.0.1:8787", "http://localhost", "http://localhost:8787"]
)
def test_endpoint_accepts_http_to_this_machine(url: str) -> None:
    assert QueueEndpoint(url, "t").headers() == {"Authorization": "Bearer t"}


@pytest.mark.parametrize(
    ("endpoint", "token", "access_id", "access_secret"),
    [
        (runner_endpoint, "HAL_NETPLAY_RUNNER_TOKEN", "CF_ACCESS_CLIENT_ID", "CF_ACCESS_CLIENT_SECRET"),
        (
            admin_endpoint,
            "HAL_NETPLAY_ADMIN_TOKEN",
            "HAL_NETPLAY_ADMIN_ACCESS_CLIENT_ID",
            "HAL_NETPLAY_ADMIN_ACCESS_CLIENT_SECRET",
        ),
    ],
)
def test_environment_errors_name_the_variables_read(
    endpoint: Callable[[dict[str, str]], QueueEndpoint], token: str, access_id: str, access_secret: str
) -> None:
    both = f"set both {access_id} and {access_secret}, or neither"
    with pytest.raises(ValueError, match=f"^set {access_secret}$"):
        endpoint({"HAL_NETPLAY_API_URL": "https://20xx.xyz", token: "t", access_id: "id"})
    with pytest.raises(ValueError, match=f"^{both}$"):
        endpoint({"HAL_NETPLAY_API_URL": "http://127.0.0.1:8787", token: "t", access_id: "id"})
    with pytest.raises(ValueError, match=f"^{both}$"):
        endpoint({"HAL_NETPLAY_API_URL": "http://127.0.0.1:8787", token: "t", access_secret: "secret"})


def test_endpoint_rejects_misconfiguration_and_hides_secrets() -> None:
    with pytest.raises(ValueError, match="trailing slash"):
        QueueEndpoint("https://20xx.xyz/", "t", "id", "secret")
    with pytest.raises(ValueError, match="both the Access client id and secret"):
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


POLICY = PolicyConfig(
    bundle_sha256="a" * 64,
    bundle_r2_key=f"netplay/policies/{'a' * 64}.halpolicy",
    vocabulary_sha256="b" * 64,
    characters=CHARACTERS,
    imitations=IMITATIONS,
    stages=STAGES,
    online_delays=(2, 3),
    desired_return_range=(0.0, 40.0),
    default_desired_return=20.0,
    temperature_range=(0.8, 1.1),
    default_temperature=1.0,
    masked_identity=False,
)


def _runner(script: _Script) -> RunnerClient:
    return RunnerClient(ENDPOINT, client=httpx.Client(base_url=ENDPOINT.url, transport=httpx.MockTransport(script)))


def _admin(script: _Script) -> AdminClient:
    return AdminClient(ENDPOINT, client=httpx.Client(base_url=ENDPOINT.url, transport=httpx.MockTransport(script)))


def test_start_session_reads_policy_and_one_account_per_slot() -> None:
    grant = {"slot": 0, "connect_code": "BOT0#1", "r2_key": "netplay/accounts/x.json", "sha256": "c" * 64}
    script = _Script(
        httpx.Response(
            201, json={"session_id": "session-00000001", "accounts": [grant], "policy": POLICY.to_payload()}
        )
    )
    started = _runner(script).start_session(
        session_id="session-00000001",
        host="box",
        bundle_sha256="a" * 64,
        git_sha="d" * 40,
        slots=1,
        wants_stream=False,
    )
    assert started.session_id == "session-00000001" and started.policy == POLICY
    assert started.accounts[0].connect_code == "BOT0#1"
    assert json.loads(script.requests[0].content) == {
        "protocol_version": RUNNER_PROTOCOL_VERSION,
        "session_id": "session-00000001",
        "host": "box",
        "bundle_sha256": "a" * 64,
        "git_sha": "d" * 40,
        "slots": 1,
        "stream": False,
    }


def test_runner_protocol_matches_the_worker() -> None:
    domain = (Path(__file__).parents[1] / "web/netplay-api/src/domain.ts").read_text()
    assert f"export const RUNNER_PROTOCOL_VERSION = {RUNNER_PROTOCOL_VERSION};" in domain


def test_start_session_rejects_a_missing_or_misnumbered_account() -> None:
    body = {"session_id": "session-00000001", "accounts": [], "policy": POLICY.to_payload()}
    with pytest.raises(QueueProtocolError, match="assign every slot"):
        _runner(_Script(httpx.Response(201, json=body))).start_session(
            session_id="session-00000001",
            host="box",
            bundle_sha256="a" * 64,
            git_sha="d" * 40,
            slots=1,
            wants_stream=False,
        )


def test_start_session_rejects_a_different_session_id() -> None:
    grant = {"slot": 0, "connect_code": "BOT0#1", "r2_key": "netplay/accounts/x.json", "sha256": "c" * 64}
    body = {"session_id": "sess", "accounts": [grant], "policy": POLICY.to_payload()}
    with pytest.raises(QueueProtocolError, match="returned 'sess'"):
        _runner(_Script(httpx.Response(201, json=body))).start_session(
            session_id="session-00000001",
            host="box",
            bundle_sha256="a" * 64,
            git_sha="d" * 40,
            slots=1,
            wants_stream=False,
        )


def _runner_status() -> RunnerStatus:
    slot = SlotStatus(0, SlotState.IDLE, None, None, None, None, None, 0, time.time())
    return aggregate_runner_status("a" * 64, (slot,), time.time(), model_inference_p95_ms=None, batch_wait_p95_ms=None)


def test_status_report_returns_the_session_state() -> None:
    script = _Script(httpx.Response(200, json={"draining": True, "stream": None}))
    assert _runner(script).report_status("sess", _runner_status()) == SessionState(draining=True)
    assert json.loads(script.requests[0].content)["schema_version"] == 5
    assert script.requests[0].url.path == "/v1/runner/sessions/sess/status"


def test_status_report_returns_a_stream_grant() -> None:
    script = _Script(httpx.Response(200, json={"draining": False, "stream": {"slot": 0, "key": "live_x"}}))
    assert _runner(script).report_status("sess", _runner_status()) == SessionState(
        draining=False, stream=StreamGrant(slot=0, key="live_x")
    )


@pytest.mark.parametrize(
    "grant",
    (
        {"slot": 1, "key": "live_x"},
        {"slot": 0, "key": ""},
        {"slot": 0, "key": "live_x", "extra": True},
        "live_x",
    ),
)
def test_status_report_refuses_an_invalid_stream_grant(grant: object) -> None:
    script = _Script(httpx.Response(200, json={"draining": False, "stream": grant}))
    with pytest.raises(QueueProtocolError, match="stream grant|status response"):
        _runner(script).report_status("sess", _runner_status())


def test_runner_reads_public_queue_depth() -> None:
    script = _Script(httpx.Response(200, json={"queued": 3, "capacity": 2}))
    assert _runner(script).queue_depth() == 3
    assert script.requests[0].url.path == "/v1/capacity"


@pytest.mark.parametrize("queued", (-1, 1.5, True, None))
def test_runner_refuses_invalid_queue_depth(queued: object) -> None:
    with pytest.raises(QueueProtocolError, match="capacity"):
        _runner(_Script(httpx.Response(200, json={"queued": queued}))).queue_depth()


def _wait_for(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 5
    while not condition():
        assert time.monotonic() < deadline
        time.sleep(0.01)


def test_reporter_keeps_reporting_through_outages_and_stops_on_session_end() -> None:
    script = _Script(
        httpx.Response(200, json={"draining": False, "stream": None}),
        *(httpx.Response(503) for _ in range(6)),
        httpx.Response(200, json={"draining": True, "stream": None}),
        httpx.Response(410, json={"detail": "session has ended"}),
    )
    client = RunnerClient(
        ENDPOINT,
        client=httpx.Client(base_url=ENDPOINT.url, transport=httpx.MockTransport(script)),
        sleep=lambda _: None,
    )
    # Hold the first background report until the entry state has been checked.
    first_state_checked = threading.Event()
    calls: list[None] = []

    def status() -> RunnerStatus:
        calls.append(None)
        if len(calls) > 1:
            assert first_state_checked.wait(5)
        return _runner_status()

    with SessionReporter(client, "sess", status, interval_seconds=0.01) as reporter:
        assert reporter.state() == SessionState(draining=False)
        first_state_checked.set()

        def ended() -> bool:
            try:
                reporter.state()
            except SessionEndedError:
                return True
            return False

        # One report retries through six 503s, the next sees the drain, the last sees the end.
        _wait_for(ended)
    assert len(script.requests) == 9


def test_reporter_hands_a_status_failure_to_the_owner() -> None:
    script = _Script(httpx.Response(200, json={"draining": False, "stream": None}))
    calls: list[None] = []

    def status() -> RunnerStatus:
        calls.append(None)
        if len(calls) > 1:
            raise RuntimeError("slot status is unreadable")
        return _runner_status()

    with SessionReporter(_runner(script), "sess", status, interval_seconds=0.01) as reporter:

        def failed() -> bool:
            try:
                reporter.state()
            except RuntimeError:
                return True
            return False

        _wait_for(failed)
        with pytest.raises(RuntimeError, match="slot status is unreadable"):
            reporter.state()
    assert len(script.requests) == 1 and len(calls) == 2


def test_start_session_rejects_a_different_policy_bundle() -> None:
    grant = {"slot": 0, "connect_code": "BOT0#1", "r2_key": "netplay/accounts/x.json", "sha256": "c" * 64}
    body = {"session_id": "session-00000001", "accounts": [grant], "policy": POLICY.to_payload()}
    with pytest.raises(QueueProtocolError, match=f"policy bundle {'a' * 64}, not the requested {'e' * 64}"):
        _runner(_Script(httpx.Response(201, json=body))).start_session(
            session_id="session-00000001",
            host="box",
            bundle_sha256="e" * 64,
            git_sha="d" * 40,
            slots=1,
            wants_stream=False,
        )


def test_active_policy_drain_and_end() -> None:
    script = _Script(
        httpx.Response(200, json=POLICY.to_payload()),
        httpx.Response(200, json={"draining": True}),
        httpx.Response(200, json={"failed": 2}),
    )
    client = _runner(script)
    assert client.active_policy() == POLICY
    client.drain("sess")
    assert client.end_session("sess") == 2
    assert [(r.method, r.url.path) for r in script.requests] == [
        ("GET", "/v1/runner/policy"),
        ("POST", "/v1/runner/sessions/sess/drain"),
        ("DELETE", "/v1/runner/sessions/sess"),
    ]


def test_admin_routes() -> None:
    script = _Script(
        httpx.Response(200, json=POLICY.to_payload()),
        httpx.Response(200, json=[]),
        httpx.Response(200, json={"paused": True}),
        httpx.Response(200, json={"events": [{"kind": "job_created"}]}),
    )
    admin = _admin(script)
    admin.put_policy(POLICY)
    admin.put_accounts([Account("BOT0#1", "netplay/accounts/x.json", "c" * 64)])
    admin.set_paused(True)
    assert admin.events(job="j1", session=None, since=12.5) == [{"kind": "job_created"}]
    assert json.loads(script.requests[1].content) == [
        {"connect_code": "BOT0#1", "r2_key": "netplay/accounts/x.json", "sha256": "c" * 64}
    ]
    assert script.requests[2].url.path == "/v1/admin/pause"
    assert dict(script.requests[3].url.params) == {"job": "j1", "since": "12.5"}


def test_pairing_cleanup_retries_with_the_same_attempt() -> None:
    script = _Script(httpx.Response(503), httpx.Response(200, json={}))
    sleeps: list[float] = []
    _queue(script, sleeps).finish_pairing("job-1", WORKER, 2)
    assert sleeps == [0.25]
    assert len(script.requests) == 2
    for request in script.requests:
        assert request.url.path == "/v1/runner/sessions/sess/pairing-finished"
        assert json.loads(request.content) == {"slot": 1, "job_id": "job-1", "attempt": 2}


@pytest.mark.parametrize(
    "body, expected",
    [(None, None), ({"slot": 0, "job_id": "job", "attempt": 1}, Pairing(0, "job", 1))],
)
def test_runner_reads_pairing(body: object, expected: Pairing | None) -> None:
    script = _Script(httpx.Response(200, json={"pairing": body}))
    with httpx.Client(base_url=ENDPOINT.url, transport=httpx.MockTransport(script)) as http:
        client = RunnerClient(ENDPOINT, client=http)
        assert client.pairing("sess") == expected


@pytest.mark.parametrize("field,value", [("slot", -1), ("job_id", ""), ("attempt", 0)])
def test_runner_rejects_invalid_pairing(field: str, value: object) -> None:
    body = {"slot": 0, "job_id": "job", "attempt": 1, field: value}
    script = _Script(httpx.Response(200, json={"pairing": body}))
    with httpx.Client(base_url=ENDPOINT.url, transport=httpx.MockTransport(script)) as http:
        client = RunnerClient(ENDPOINT, client=http)
        with pytest.raises(QueueProtocolError, match="invalid pairing identity"):
            client.pairing("sess")
