import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hal.netplay_service.api import ApiConfig
from hal.netplay_service.api import create_app
from hal.netplay_service.health import RunnerState
from hal.netplay_service.health import RunnerStatus
from hal.netplay_service.health import write_runner_status
from hal.netplay_service.queue import QueueStore


def _client(tmp_path: Path) -> tuple[TestClient, QueueStore]:
    store = QueueStore(tmp_path / "queue.sqlite3")
    app = create_app(
        ApiConfig(
            tmp_path / "queue.sqlite3",
            allowed_origins=("https://play.example",),
            allowed_hosts=("testserver",),
        ),
        store,
    )
    return TestClient(app), store


def _request() -> dict[str, object]:
    return {
        "player_code": "CRYO#610",
        "character": "FOX",
        "imitation": "IBDW#0",
        "online_delay": 2,
    }


def _capacity(*, active: int = 0, queued: int = 0) -> dict[str, object]:
    return {
        "capacity": 2,
        "healthy_slots": 2,
        "active": active,
        "queued": queued,
        "service_status": "ready",
        "service_message": "Game servers are ready.",
        "target_fps": 60.0,
        "game_fps": None,
        "frame_interval_p95_ms": None,
        "dolphin_step_p95_ms": None,
        "policy_round_trip_p95_ms": None,
        "model_inference_p95_ms": None,
        "batch_wait_p95_ms": None,
        "recoveries": 0,
    }


def _runner_status(
    state: RunnerState,
    *,
    healthy_slots: int,
    updated_at: float | None = None,
) -> RunnerStatus:
    return RunnerStatus(
        state=state,
        message="Game servers are ready." if state is RunnerState.READY else "Gameplay is degraded.",
        policy_sha256="a" * 64,
        slots=2,
        healthy_slots=healthy_slots,
        target_fps=60.0,
        game_fps=55.0 if state is RunnerState.DEGRADED else None,
        frame_interval_p95_ms=20.0 if state is RunnerState.DEGRADED else None,
        dolphin_step_p95_ms=5.0 if state is RunnerState.DEGRADED else None,
        policy_round_trip_p95_ms=10.0 if state is RunnerState.DEGRADED else None,
        model_inference_p95_ms=8.0 if state is RunnerState.DEGRADED else None,
        batch_wait_p95_ms=0.5 if state is RunnerState.DEGRADED else None,
        recoveries=1 if state is RunnerState.DEGRADED else 0,
        updated_at=time.time() if updated_at is None else updated_at,
    )


def test_options_and_capacity_are_public(tmp_path: Path) -> None:
    client, _store = _client(tmp_path)
    with client:
        options = client.get("/v1/options")
        capacity = client.get("/v1/capacity")

    assert options.status_code == 200
    assert options.json()["online_delays"] == [2, 3]
    assert len(options.json()["characters"]) == 26
    assert capacity.json() == _capacity()


def test_capacity_separates_queued_and_active_reservations(tmp_path: Path) -> None:
    client, store = _client(tmp_path)
    with client:
        assert client.post("/v1/jobs", json=_request()).status_code == 201
        assert client.get("/v1/capacity").json() == _capacity(queued=1)
        assert store.claim_next("slot-0") is not None
        assert client.get("/v1/capacity").json() == _capacity(active=1)


def test_readiness_requires_a_fresh_runner_heartbeat(tmp_path: Path) -> None:
    status = tmp_path / "runner.json"
    app = create_app(
        ApiConfig(
            tmp_path / "queue.sqlite3",
            allowed_origins=("https://play.example",),
            allowed_hosts=("testserver",),
            runner_status=status,
        )
    )
    with TestClient(app) as client:
        assert client.get("/health/ready").status_code == 503
        write_runner_status(status, _runner_status(RunnerState.READY, healthy_slots=2, updated_at=0.0))
        assert client.get("/health/ready").status_code == 503
        write_runner_status(status, _runner_status(RunnerState.READY, healthy_slots=2))
        assert client.get("/health/ready").status_code == 200


def test_capacity_exposes_degraded_gameplay_and_recovery(tmp_path: Path) -> None:
    status_path = tmp_path / "runner.json"
    write_runner_status(status_path, _runner_status(RunnerState.DEGRADED, healthy_slots=1))
    app = create_app(
        ApiConfig(
            tmp_path / "queue.sqlite3",
            allowed_origins=("https://play.example",),
            allowed_hosts=("testserver",),
            runner_status=status_path,
        )
    )

    with TestClient(app) as client:
        response = client.get("/v1/capacity")

    assert response.status_code == 200
    assert response.json() == {
        "capacity": 2,
        "healthy_slots": 1,
        "active": 0,
        "queued": 0,
        "service_status": "degraded",
        "service_message": "Gameplay is degraded.",
        "target_fps": 60.0,
        "game_fps": 55.0,
        "frame_interval_p95_ms": 20.0,
        "dolphin_step_p95_ms": 5.0,
        "policy_round_trip_p95_ms": 10.0,
        "model_inference_p95_ms": 8.0,
        "batch_wait_p95_ms": 0.5,
        "recoveries": 1,
    }


def test_queue_rejects_new_jobs_while_every_slot_recovers(tmp_path: Path) -> None:
    status_path = tmp_path / "runner.json"
    write_runner_status(status_path, _runner_status(RunnerState.RECOVERING, healthy_slots=0))
    app = create_app(
        ApiConfig(
            tmp_path / "queue.sqlite3",
            allowed_origins=("https://play.example",),
            allowed_hosts=("testserver",),
            runner_status=status_path,
        )
    )

    with TestClient(app) as client:
        capacity = client.get("/v1/capacity")
        ready = client.get("/health/ready")
        created = client.post("/v1/jobs", json=_request())

    assert capacity.json()["service_status"] == "recovering"
    assert capacity.json()["healthy_slots"] == 0
    assert ready.status_code == 200
    assert created.status_code == 503


def test_create_poll_and_cancel_job(tmp_path: Path) -> None:
    client, _store = _client(tmp_path)
    with client:
        created = client.post("/v1/jobs", json=_request())
        assert created.status_code == 201
        values = created.json()
        headers = {"Authorization": f"Bearer {values['token']}"}
        status = client.get(f"/v1/jobs/{values['id']}", headers=headers)
        canceled = client.delete(f"/v1/jobs/{values['id']}", headers=headers)

    assert status.json()["queue_position"] == 1
    assert canceled.json()["status"] == "canceled"
    assert "token" not in status.json()


def test_job_credentials_do_not_reveal_existence(tmp_path: Path) -> None:
    client, _store = _client(tmp_path)
    with client:
        created = client.post("/v1/jobs", json=_request()).json()
        missing = client.get(f"/v1/jobs/{created['id']}")
        wrong = client.get(
            f"/v1/jobs/{created['id']}",
            headers={"Authorization": "Bearer wrong"},
        )

    assert missing.status_code == 401
    assert wrong.status_code == 404


def test_active_player_and_request_size_errors_are_bounded(tmp_path: Path) -> None:
    client, _store = _client(tmp_path)
    with client:
        assert client.post("/v1/jobs", json=_request()).status_code == 201
        duplicate = client.post("/v1/jobs", json=_request())
        oversized = client.post(
            "/v1/jobs",
            content=b"x" * 20_000,
            headers={"content-type": "application/json"},
        )

    assert duplicate.status_code == 409
    assert oversized.status_code == 413


def test_cors_allows_only_configured_frontend(tmp_path: Path) -> None:
    client, _store = _client(tmp_path)
    with client:
        allowed = client.options(
            "/v1/jobs",
            headers={
                "Origin": "https://play.example",
                "Access-Control-Request-Method": "POST",
            },
        )
        denied = client.options(
            "/v1/jobs",
            headers={
                "Origin": "https://evil.example",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert allowed.headers["access-control-allow-origin"] == "https://play.example"
    assert "access-control-allow-origin" not in denied.headers


def test_capacity_polling_stress_closes_all_database_connections(
    tmp_path: Path,
) -> None:
    descriptor_directory = Path("/proc/self/fd")
    if not descriptor_directory.is_dir():
        pytest.skip("file descriptor accounting requires procfs")
    before = len(tuple(descriptor_directory.iterdir()))
    client, _store = _client(tmp_path)
    with client:
        for _ in range(2_000):
            assert client.get("/v1/capacity").status_code == 200
    after = len(tuple(descriptor_directory.iterdir()))

    assert after <= before + 2
