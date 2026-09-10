from pathlib import Path

from fastapi.testclient import TestClient

from hal.netplay_service.api import ApiConfig
from hal.netplay_service.api import create_app
from hal.netplay_service.queue import QueueStore


def _client(tmp_path: Path) -> tuple[TestClient, QueueStore, str]:
    store = QueueStore(tmp_path / "queue.sqlite3")
    invite = store.create_invite("tester")
    app = create_app(
        ApiConfig(
            tmp_path / "queue.sqlite3",
            allowed_origins=("https://play.example",),
            allowed_hosts=("testserver",),
        ),
        store,
    )
    return TestClient(app), store, invite


def _request(invite: str) -> dict[str, object]:
    return {
        "invite_code": invite,
        "player_code": "CRYO#610",
        "character": "FOX",
        "imitation": "IBDW#0",
        "online_delay": 2,
    }


def test_options_and_capacity_are_public(tmp_path: Path) -> None:
    client, _store, _invite = _client(tmp_path)
    with client:
        options = client.get("/v1/options")
        capacity = client.get("/v1/capacity")

    assert options.status_code == 200
    assert options.json()["online_delays"] == [2, 3]
    assert len(options.json()["characters"]) == 26
    assert capacity.json() == {"capacity": 2, "active": 0, "queued": 0}


def test_create_poll_and_cancel_job(tmp_path: Path) -> None:
    client, _store, invite = _client(tmp_path)
    with client:
        created = client.post("/v1/jobs", json=_request(invite))
        assert created.status_code == 201
        values = created.json()
        headers = {"Authorization": f"Bearer {values['token']}"}
        status = client.get(f"/v1/jobs/{values['id']}", headers=headers)
        canceled = client.delete(f"/v1/jobs/{values['id']}", headers=headers)

    assert status.json()["queue_position"] == 1
    assert canceled.json()["status"] == "canceled"
    assert "token" not in status.json()


def test_job_credentials_do_not_reveal_existence(tmp_path: Path) -> None:
    client, _store, invite = _client(tmp_path)
    with client:
        created = client.post("/v1/jobs", json=_request(invite)).json()
        missing = client.get(f"/v1/jobs/{created['id']}")
        wrong = client.get(
            f"/v1/jobs/{created['id']}",
            headers={"Authorization": "Bearer wrong"},
        )

    assert missing.status_code == 401
    assert wrong.status_code == 404


def test_invite_and_active_player_errors_are_bounded(tmp_path: Path) -> None:
    client, _store, invite = _client(tmp_path)
    with client:
        invalid = _request("x" * 24)
        assert client.post("/v1/jobs", json=invalid).status_code == 403
        assert client.post("/v1/jobs", json=_request(invite)).status_code == 201
        duplicate = client.post("/v1/jobs", json=_request(invite))
        oversized = client.post(
            "/v1/jobs",
            content=b"x" * 20_000,
            headers={"content-type": "application/json"},
        )

    assert duplicate.status_code == 409
    assert oversized.status_code == 413


def test_cors_allows_only_configured_frontend(tmp_path: Path) -> None:
    client, _store, _invite = _client(tmp_path)
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
