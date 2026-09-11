import json
from pathlib import Path

import pytest

from hal.netplay_service.health import FRAME_STALL_SECONDS
from hal.netplay_service.health import RUNNER_HEARTBEAT_MAX_AGE_SECONDS
from hal.netplay_service.health import RunnerState
from hal.netplay_service.health import RuntimeHealth
from hal.netplay_service.health import SlotState
from hal.netplay_service.health import SlotStatus
from hal.netplay_service.health import aggregate_runner_status
from hal.netplay_service.health import read_runner_status
from hal.netplay_service.health import read_slot_status
from hal.netplay_service.health import write_runner_status
from hal.netplay_service.health import write_slot_status

_POLICY_SHA256 = "a" * 64


def _slot(
    slot: int,
    state: SlotState,
    *,
    fps: float | None = None,
    frame_p95_ms: float | None = None,
    dolphin_p95_ms: float | None = None,
    policy_p95_ms: float | None = None,
    reason: str | None = None,
    recoveries: int = 0,
    updated_at: float = 100.0,
) -> SlotStatus:
    return SlotStatus(
        slot=slot,
        state=state,
        game_fps=fps,
        frame_interval_p95_ms=frame_p95_ms,
        dolphin_step_p95_ms=dolphin_p95_ms,
        policy_round_trip_p95_ms=policy_p95_ms,
        reason=reason,
        recoveries=recoveries,
        updated_at=updated_at,
    )


def _drive(monitor: RuntimeHealth, *, fps: float, frames: int, policy_seconds: float) -> float:
    now = 0.0
    monitor.begin(2, now)
    for frame_id in range(1, frames + 1):
        monitor.observe_policy(policy_seconds, now)
        now += 1.0 / fps
        monitor.observe_frame(frame_id, 0.005, now)
    return now


def test_runtime_health_stays_ready_for_ten_minutes_at_60_fps() -> None:
    monitor = RuntimeHealth()
    now = _drive(monitor, fps=60.0, frames=36_000, policy_seconds=0.010)

    status = monitor.snapshot(now)

    assert status.reason is None
    assert status.recovery_required is False
    assert status.game_fps == pytest.approx(60.0)
    assert status.frame_interval_p95_ms == pytest.approx(1_000.0 / 60.0)
    assert status.dolphin_step_p95_ms == pytest.approx(5.0)
    assert status.policy_round_trip_p95_ms == pytest.approx(10.0)


def test_low_fps_stays_degraded_without_recovery() -> None:
    monitor = RuntimeHealth()
    now = _drive(monitor, fps=58.0, frames=1_200, policy_seconds=0.010)
    status = monitor.snapshot(now)

    assert status.reason == "low_frame_rate"
    assert status.recovery_required is False


def test_frame_stutter_stays_degraded_without_recovery() -> None:
    monitor = RuntimeHealth()
    monitor.begin(2, 0.0)
    now = 0.0
    for frame_id in range(1, 1_201):
        monitor.observe_policy(0.010, now)
        now += 0.030 if frame_id % 10 == 0 else 0.015
        monitor.observe_frame(frame_id, 0.004, now)

    status = monitor.snapshot(now)

    assert status.reason == "frame_stutter"
    assert status.frame_interval_p95_ms == pytest.approx(30.0)
    assert status.recovery_required is False


def test_policy_deadline_degrades_without_restarting_dolphin() -> None:
    monitor = RuntimeHealth()
    monitor.begin(3, 0.0)
    now = 0.0
    for frame_id in range(1, 1_201):
        monitor.observe_policy(0.020, now)
        now += 1.0 / 60.0
        monitor.observe_frame(frame_id, 0.005, now)

    status = monitor.snapshot(now)

    assert status.reason == "slow_inference"
    assert status.recovery_required is False


def test_ten_second_frame_stall_requires_immediate_recovery() -> None:
    monitor = RuntimeHealth()
    monitor.begin(2, 10.0)

    status = monitor.snapshot(10.0 + FRAME_STALL_SECONDS)

    assert status.reason == "frame_stream_stalled"
    assert status.recovery_required is True


def test_runtime_health_survives_repeated_recovery_cycles() -> None:
    monitor = RuntimeHealth()
    now = 0.0

    for _ in range(64):
        monitor.begin(2, now)
        now += FRAME_STALL_SECONDS + 0.01
        status = monitor.snapshot(now)
        assert status.recovery_required is True
        monitor.finish()
        now += 0.1


def test_runner_status_uses_worst_slot_and_round_trips(tmp_path: Path) -> None:
    slots = (
        _slot(0, SlotState.PLAYING, fps=60.0, frame_p95_ms=17.0, dolphin_p95_ms=5.0, policy_p95_ms=10.0),
        _slot(
            1,
            SlotState.DEGRADED,
            fps=58.0,
            frame_p95_ms=22.0,
            dolphin_p95_ms=8.0,
            policy_p95_ms=20.0,
            reason="low_frame_rate",
            recoveries=2,
        ),
    )
    status = aggregate_runner_status(
        _POLICY_SHA256,
        slots,
        101.0,
        model_inference_p95_ms=11.0,
        batch_wait_p95_ms=0.5,
    )
    path = tmp_path / "runner.json"
    write_runner_status(path, status)

    assert read_runner_status(path) == status
    assert status.state is RunnerState.DEGRADED
    assert status.healthy_slots == 1
    assert status.game_fps == 58.0
    assert status.frame_interval_p95_ms == 22.0
    assert status.dolphin_step_p95_ms == 8.0
    assert status.policy_round_trip_p95_ms == 20.0
    assert status.model_inference_p95_ms == 11.0
    assert status.batch_wait_p95_ms == 0.5
    assert status.recoveries == 2


def test_slot_status_round_trip_rejects_schema_drift(tmp_path: Path) -> None:
    path = tmp_path / "slot.json"
    status = _slot(0, SlotState.IDLE)
    write_slot_status(path, status)
    assert read_slot_status(path) == status

    payload = status.to_payload()
    payload["extra"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="wrong schema"):
        read_slot_status(path)


def test_runner_heartbeat_limit_is_shorter_than_a_queue_lease() -> None:
    assert 0 < RUNNER_HEARTBEAT_MAX_AGE_SECONDS < 20.0
