"""RemoteQueue, RunnerClient, and SessionReporter against the real queue Worker under `wrangler dev`."""

import json
import os
import time
from collections.abc import Callable
from collections.abc import Iterator

import httpx
import pytest

from hal.netplay_service.domain import CHARACTERS
from hal.netplay_service.domain import IMITATIONS
from hal.netplay_service.domain import STAGES
from hal.netplay_service.domain import JobStatus
from hal.netplay_service.domain import PolicyConfig
from hal.netplay_service.health import RunnerStatus
from hal.netplay_service.health import SlotState
from hal.netplay_service.health import SlotStatus
from hal.netplay_service.health import aggregate_runner_status
from hal.netplay_service.local_worker import DEV_ADMIN_TOKEN
from hal.netplay_service.local_worker import DEV_RUNNER_TOKEN
from hal.netplay_service.local_worker import WORKER_PROJECT
from hal.netplay_service.local_worker import free_port
from hal.netplay_service.local_worker import local_worker
from hal.netplay_service.queue_client import Account
from hal.netplay_service.queue_client import AdminClient
from hal.netplay_service.queue_client import QueueEndpoint
from hal.netplay_service.queue_client import RemoteQueue
from hal.netplay_service.queue_client import RunnerClient
from hal.netplay_service.queue_client import SessionReporter
from hal.netplay_service.queue_client import StartedSession
from hal.netplay_service.queue_client import new_session_id
from hal.netplay_service.queue_client import slot_worker_id
from hal.netplay_service.queue_contract import SessionEndedError

pytestmark = pytest.mark.integration

POLICY = PolicyConfig(
    bundle_sha256="0" * 64,
    bundle_r2_key="netplay/policies/integration.halpolicy",
    vocabulary_sha256="1" * 64,
    characters=CHARACTERS,
    imitations=IMITATIONS,
    stages=STAGES,
    online_delays=(2, 3),
    desired_return_range=(-20.0, 140.0),
    default_desired_return=20.0,
    temperature_range=(0.8, 1.1),
    default_temperature=1.0,
    masked_identity=False,
)
ACCOUNTS = [Account(f"BOT{index}#1", f"netplay/accounts/{index}.json", str(index) * 64) for index in range(4)]


@pytest.fixture(scope="module")
def worker_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    if not (WORKER_PROJECT / "node_modules" / ".bin" / "wrangler").is_file():
        message = f"run `npm ci` in {WORKER_PROJECT}"
        if os.environ.get("HAL_REQUIRE_INTEGRATION") == "1":
            pytest.fail(message)
        pytest.skip(message)
    with local_worker(tmp_path_factory.mktemp("worker"), port=free_port()) as url:
        yield url


@pytest.fixture
def admin(worker_url: str) -> Iterator[AdminClient]:
    client = AdminClient(QueueEndpoint(worker_url, DEV_ADMIN_TOKEN))
    client.put_policy(POLICY)
    client.put_accounts(ACCOUNTS)
    yield client
    client.close()


def _status(state: SlotState = SlotState.IDLE) -> RunnerStatus:
    now = time.time()
    slot = SlotStatus(0, state, None, None, None, None, None, 0, now)
    return aggregate_runner_status(
        POLICY.bundle_sha256, (slot,), now, model_inference_p95_ms=None, batch_wait_p95_ms=None
    )


class _LoseFirstResponse(httpx.BaseTransport):
    """Deliver each matching request, and lose the response to the first one."""

    def __init__(self, matches: Callable[[httpx.Request], bool]) -> None:
        self._inner = httpx.HTTPTransport()
        self._matches = matches
        self.lost = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self._inner.handle_request(request)
        if self.lost == 0 and self._matches(request):
            self.lost += 1
            response.read()
            response.close()
            raise httpx.ReadError("response lost by the test", request=request)
        return response

    def close(self) -> None:
        self._inner.close()


def _lossy(url: str, method: str, suffix: str) -> tuple[httpx.Client, _LoseFirstResponse]:
    transport = _LoseFirstResponse(lambda r: r.method == method and r.url.path.endswith(suffix))
    return httpx.Client(base_url=url, transport=transport), transport


def _start(sessions: RunnerClient) -> StartedSession:
    return sessions.start_session(
        session_id=new_session_id(),
        host="integration",
        bundle_sha256=POLICY.bundle_sha256,
        git_sha="a" * 40,
        slots=1,
        wants_stream=False,
    )


def _create(url: str, player_code: str) -> tuple[str, str]:
    created = httpx.post(
        f"{url}/v1/jobs",
        json={
            "player_code": player_code,
            "character": "FOX",
            "imitation": "MASTER",
            "online_delay": 2,
            "desired_return": 120,
        },
        timeout=10,
    )
    assert created.status_code == 201, created.text
    return created.json()["id"], created.json()["token"]


def _player_job(url: str, job_id: str, token: str) -> dict[str, object]:
    response = httpx.get(f"{url}/v1/jobs/{job_id}", headers={"Authorization": f"Bearer {token}"}, timeout=10)
    return response.json()


def _live_sessions(admin: AdminClient) -> list[dict[str, object]]:
    sessions = admin.status()["sessions"]
    assert isinstance(sessions, list)
    return [row for row in sessions if row["ended_at"] is None]


def test_lost_responses_never_duplicate_sessions_jobs_or_games(worker_url: str, admin: AdminClient) -> None:
    endpoint = QueueEndpoint(worker_url, DEV_RUNNER_TOKEN)
    sleeps: list[float] = []

    live_before = {row["id"] for row in _live_sessions(admin)}
    start_client, start_loss = _lossy(worker_url, "POST", "/v1/runner/sessions")
    started = _start(RunnerClient(endpoint, client=start_client, sleep=sleeps.append))
    assert start_loss.lost == 1
    assert {row["id"] for row in _live_sessions(admin)} - live_before == {started.session_id}
    accounts = admin.status()["accounts"]
    assert isinstance(accounts, list)
    assert [row["connect_code"] for row in accounts if row["session_id"] == started.session_id] == [
        started.accounts[0].connect_code
    ]
    sessions = RunnerClient(endpoint)
    sessions.report_status(started.session_id, _status())

    first_id, first_token = _create(worker_url, "CRYO#610")
    second_id, second_token = _create(worker_url, "OTHER#1")
    worker = slot_worker_id(started.session_id, 0)
    claim_client, claim_loss = _lossy(worker_url, "POST", "/claim")
    claimed = RemoteQueue(endpoint, started.session_id, client=claim_client, sleep=sleeps.append).claim_next(worker)
    assert claim_loss.lost == 1
    assert claimed is not None and (claimed.id, claimed.attempt) == (first_id, 1)
    assert claimed.choices.imitation == "MASTER"
    assert claimed.choices.desired_return == 120.0
    assert _player_job(worker_url, second_id, second_token)["status"] == "queued"

    queue = RemoteQueue(endpoint, started.session_id)
    queue.mark_connecting(first_id, worker, started.accounts[0].connect_code)
    playing_client, playing_loss = _lossy(worker_url, "POST", "/playing")
    RemoteQueue(endpoint, started.session_id, client=playing_client, sleep=sleeps.append).mark_playing(
        first_id, worker
    )
    assert playing_loss.lost == 1
    assert queue.get_worker_job(first_id, worker).status is JobStatus.PLAYING
    finish = {"game_number": 1, "actual_stage": "BATTLEFIELD", "result": "win"}
    assert queue.finish_game(first_id, worker, **finish) is JobStatus.REMATCH_WAIT
    assert queue.finish_game(first_id, worker, **finish) is JobStatus.REMATCH_WAIT
    assert queue.get_worker_job(first_id, worker).game_count == 1

    end_client, end_loss = _lossy(worker_url, "DELETE", f"/v1/runner/sessions/{started.session_id}")
    assert RunnerClient(endpoint, client=end_client, sleep=sleeps.append).end_session(started.session_id) == 1
    assert end_loss.lost == 1
    assert sleeps == [0.25, 0.25, 0.25, 0.25]
    first = _player_job(worker_url, first_id, first_token)
    assert (first["status"], first["error_code"]) == ("failed", "service_generation_aborted")
    httpx.delete(f"{worker_url}/v1/jobs/{second_id}", headers={"Authorization": f"Bearer {second_token}"}, timeout=10)


def test_live_socket_pushes_settings_and_release(worker_url: str, admin: AdminClient) -> None:
    endpoint = QueueEndpoint(worker_url, DEV_RUNNER_TOKEN)
    sessions = RunnerClient(endpoint)
    started = _start(sessions)
    sessions.report_status(started.session_id, _status())
    job_id, token = _create(worker_url, "LIVE#1")
    worker = slot_worker_id(started.session_id, 0)
    queue = RemoteQueue(endpoint, started.session_id)
    assert queue.claim_next(worker) is not None
    with queue.connect_live(job_id, worker) as socket:
        assert json.loads(socket.recv(timeout=5)) == {
            "type": "settings",
            "revision": 0,
            "desired_return": 120,
            "temperature": 1,
        }
        httpx.patch(
            f"{worker_url}/v1/jobs/{job_id}/policy",
            headers={"Authorization": f"Bearer {token}"},
            json={"desired_return": 140},
            timeout=10,
        ).raise_for_status()
        assert json.loads(socket.recv(timeout=5)) == {
            "type": "settings",
            "revision": 1,
            "desired_return": 140,
            "temperature": 1,
        }
        assert queue.get_worker_job(job_id, worker).choices.desired_return == 140.0
        rejected = httpx.patch(
            f"{worker_url}/v1/jobs/{job_id}/policy",
            headers={"Authorization": f"Bearer {token}"},
            json={"desired_return": 141},
            timeout=10,
        )
        assert rejected.status_code == 422
        httpx.delete(f"{worker_url}/v1/jobs/{job_id}", headers={"Authorization": f"Bearer {token}"}, timeout=10)
        assert json.loads(socket.recv(timeout=5)) == {"type": "released"}
    sessions.end_session(started.session_id)


def test_reporter_keeps_a_starting_session_alive_while_a_silent_one_ends(worker_url: str, admin: AdminClient) -> None:
    endpoint = QueueEndpoint(worker_url, DEV_RUNNER_TOKEN)
    sessions = RunnerClient(endpoint)
    starting = _start(sessions)
    silent = _start(sessions)
    sessions.report_status(silent.session_id, _status())
    with SessionReporter(sessions, starting.session_id, lambda: _status(SlotState.STARTING)) as reporter:
        deadline = time.monotonic() + 60
        while silent.session_id in {row["id"] for row in _live_sessions(admin)}:
            assert time.monotonic() < deadline, "the Worker did not end a session that stopped reporting"
            time.sleep(1)
        assert starting.session_id in {row["id"] for row in _live_sessions(admin)}
        assert reporter.state().draining is False
    with pytest.raises(SessionEndedError):
        sessions.report_status(silent.session_id, _status())
    assert sessions.end_session(starting.session_id) == 0
