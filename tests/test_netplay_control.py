import time
from contextlib import closing
from dataclasses import replace
from pathlib import Path

from hal.netplay_service.control import Observation
from hal.netplay_service.control import QueueControl
from hal.netplay_service.health import RunnerStatus
from hal.netplay_service.health import SlotState
from hal.netplay_service.health import SlotStatus
from hal.netplay_service.health import aggregate_runner_status
from hal.netplay_service.health import write_slot_status
from hal.netplay_service.queue_client import QueueEndpoint


def test_supervisor_does_not_renew_missing_frozen_or_recovering_slots(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    health = SlotStatus(0, SlotState.PLAYING, 60, 17, 1, 20, None, 0, time.time())

    def status() -> RunnerStatus:
        return aggregate_runner_status(
            "a" * 64, (health,), time.time(), model_inference_p95_ms=None, batch_wait_p95_ms=None
        )

    with closing(QueueControl(QueueEndpoint("http://127.0.0.1:1", "test"), "session", status, path)) as control:
        control._observations[0] = Observation("job", 2, 40, time.monotonic(), "snapshot")
        assert control._progress() == []
        slot_path = tmp_path / "status.json.slot-0.json"
        write_slot_status(slot_path, health)
        assert control._progress() == [{"slot": 0, "job_id": "job", "attempt": 2, "seq": 40}]
        write_slot_status(slot_path, replace(health, updated_at=time.time() - 6))
        assert control._progress() == []
        write_slot_status(slot_path, replace(health, state=SlotState.RECOVERING))
        assert control._progress() == []
        write_slot_status(slot_path, health)
        control._observations[0] = replace(control._observations[0], received_at=time.monotonic() - 7)
        assert control._progress() == []


def test_expired_control_connection_ends_the_local_session(tmp_path: Path) -> None:
    health = SlotStatus(0, SlotState.IDLE, None, None, None, None, None, 0, time.time())

    def status() -> RunnerStatus:
        return aggregate_runner_status(
            "a" * 64, (health,), time.time(), model_inference_p95_ms=None, batch_wait_p95_ms=None
        )

    with closing(
        QueueControl(QueueEndpoint("http://127.0.0.1:1", "test"), "session", status, tmp_path / "status")
    ) as control:
        control._last_health = time.monotonic() - 31
        code, body = control.dispatch("POST", "/v1/runner/sessions/session/status", {}, None)
        assert code == 410
        assert body == {"detail": "session control heartbeat expired"}
