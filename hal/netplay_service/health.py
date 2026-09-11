"""Runtime health records for the production netplay service."""

import json
import math
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Final
from typing import cast

TARGET_GAME_FPS: Final[float] = 60.0
MIN_GAME_FPS: Final[float] = 59.0
FRAME_INTERVAL_P95_LIMIT_MS: Final[float] = 20.0
MIN_HEALTH_FRAMES: Final[int] = 120
HEALTH_WINDOW_FRAMES: Final[int] = 300
FRAME_STALL_SECONDS: Final[float] = 10.0
SLOT_HEARTBEAT_MAX_AGE_SECONDS: Final[float] = 3.0
SLOT_STARTUP_GRACE_SECONDS: Final[float] = 30.0
RUNNER_HEARTBEAT_MAX_AGE_SECONDS: Final[float] = 5.0
POLICY_DEADLINES_SECONDS: Final[dict[int, float]] = {2: 0.0333, 3: 0.0167}
_SLOT_SCHEMA_VERSION: Final[int] = 1
_RUNNER_SCHEMA_VERSION: Final[int] = 2


class SlotState(StrEnum):
    STARTING = "starting"
    IDLE = "idle"
    CONNECTING = "connecting"
    PLAYING = "playing"
    DEGRADED = "degraded"
    RECOVERING = "recovering"


class RunnerState(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    game_fps: float | None
    frame_interval_p95_ms: float | None
    dolphin_step_p95_ms: float | None
    policy_round_trip_p95_ms: float | None
    reason: str | None
    recovery_required: bool


def _p95(values: tuple[float, ...]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


class RuntimeHealth:
    """Measure one game's rolling frame and policy timing."""

    def __init__(self) -> None:
        self._frame_times: deque[float] = deque(maxlen=HEALTH_WINDOW_FRAMES + 1)
        self._dolphin_step_seconds: deque[float] = deque(maxlen=HEALTH_WINDOW_FRAMES)
        self._policy_seconds: deque[float] = deque(maxlen=HEALTH_WINDOW_FRAMES)
        self._active = False
        self._delay: int | None = None
        self._last_frame_at: float | None = None
        self._next_assessment = 0.0
        self._snapshot = RuntimeSnapshot(None, None, None, None, None, False)

    def begin(self, delay: int, now: float) -> None:
        if delay not in POLICY_DEADLINES_SECONDS:
            raise ValueError(f"runtime health does not support delay {delay}")
        _finite_non_negative(now, "runtime health clock")
        self._frame_times.clear()
        self._dolphin_step_seconds.clear()
        self._policy_seconds.clear()
        self._active = True
        self._delay = delay
        self._last_frame_at = now
        self._next_assessment = now
        self._snapshot = RuntimeSnapshot(None, None, None, None, None, False)

    def finish(self) -> None:
        self._frame_times.clear()
        self._dolphin_step_seconds.clear()
        self._policy_seconds.clear()
        self._active = False
        self._delay = None
        self._last_frame_at = None
        self._snapshot = RuntimeSnapshot(None, None, None, None, None, False)

    def observe_frame(self, frame_id: int, dolphin_step_seconds: float, now: float) -> RuntimeSnapshot:
        self._require_active()
        if not isinstance(frame_id, int) or isinstance(frame_id, bool):
            raise ValueError("runtime health frame_id must be an integer")
        _finite_non_negative(dolphin_step_seconds, "Dolphin step latency")
        _finite_non_negative(now, "runtime health clock")
        self._last_frame_at = now
        if frame_id >= 0:
            self._frame_times.append(now)
            self._dolphin_step_seconds.append(dolphin_step_seconds)
        return self._assess_if_due(now)

    def observe_policy(self, seconds: float, now: float) -> RuntimeSnapshot:
        self._require_active()
        _finite_non_negative(seconds, "policy round-trip latency")
        _finite_non_negative(now, "runtime health clock")
        self._policy_seconds.append(seconds)
        return self._assess_if_due(now)

    def snapshot(self, now: float) -> RuntimeSnapshot:
        _finite_non_negative(now, "runtime health clock")
        self._snapshot = self._assess(now)
        self._next_assessment = now + 1.0
        return self._snapshot

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("runtime health observation started before the game")

    def _assess_if_due(self, now: float) -> RuntimeSnapshot:
        if now < self._next_assessment:
            return self._snapshot
        return self.snapshot(now)

    def _assess(self, now: float) -> RuntimeSnapshot:
        frame_times = tuple(self._frame_times)
        frame_seconds = tuple(later - earlier for earlier, later in pairwise(frame_times))
        elapsed = frame_times[-1] - frame_times[0] if len(frame_times) > 1 else 0.0
        game_fps = len(frame_seconds) / elapsed if elapsed > 0 else None
        frame_p95 = _p95(frame_seconds)
        dolphin_p95 = _p95(tuple(self._dolphin_step_seconds))
        policy_p95 = _p95(tuple(self._policy_seconds))

        stalled = self._active and self._last_frame_at is not None and now - self._last_frame_at >= FRAME_STALL_SECONDS
        cadence_reason = self._cadence_reason(game_fps, frame_p95)

        slow_inference = (
            self._delay is not None
            and len(self._policy_seconds) >= MIN_HEALTH_FRAMES
            and policy_p95 is not None
            and policy_p95 > POLICY_DEADLINES_SECONDS[self._delay]
        )
        reason = "frame_stream_stalled" if stalled else cadence_reason
        if reason is None and slow_inference:
            reason = "slow_inference"
        return RuntimeSnapshot(
            game_fps=game_fps,
            frame_interval_p95_ms=None if frame_p95 is None else 1_000.0 * frame_p95,
            dolphin_step_p95_ms=None if dolphin_p95 is None else 1_000.0 * dolphin_p95,
            policy_round_trip_p95_ms=None if policy_p95 is None else 1_000.0 * policy_p95,
            reason=reason,
            recovery_required=stalled,
        )

    def _cadence_reason(self, game_fps: float | None, frame_p95: float | None) -> str | None:
        if len(self._frame_times) < MIN_HEALTH_FRAMES:
            return None
        if frame_p95 is not None and 1_000.0 * frame_p95 > FRAME_INTERVAL_P95_LIMIT_MS:
            return "frame_stutter"
        if game_fps is not None and game_fps < MIN_GAME_FPS:
            return "low_frame_rate"
        return None


@dataclass(frozen=True, slots=True)
class SlotStatus:
    slot: int
    state: SlotState
    game_fps: float | None
    frame_interval_p95_ms: float | None
    dolphin_step_p95_ms: float | None
    policy_round_trip_p95_ms: float | None
    reason: str | None
    recoveries: int
    updated_at: float

    def __post_init__(self) -> None:
        if self.slot < 0:
            raise ValueError("slot must be non-negative")
        _validate_optional_timings(self)
        if self.reason is not None and not self.reason:
            raise ValueError("slot health reason must be non-empty or None")
        if self.recoveries < 0:
            raise ValueError("slot recoveries must be non-negative")
        _finite_non_negative(self.updated_at, "slot updated_at")

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["schema_version"] = _SLOT_SCHEMA_VERSION
        payload["state"] = self.state.value
        return payload

    @classmethod
    def from_payload(cls, payload: object) -> SlotStatus:
        expected = {"schema_version", *cls.__dataclass_fields__}
        values = _payload(payload, expected, _SLOT_SCHEMA_VERSION, "slot")
        try:
            return cls(
                slot=_integer(values["slot"], "slot"),
                state=SlotState(_string(values["state"], "state")),
                game_fps=_optional_number(values["game_fps"], "game_fps"),
                frame_interval_p95_ms=_optional_number(values["frame_interval_p95_ms"], "frame_interval_p95_ms"),
                dolphin_step_p95_ms=_optional_number(values["dolphin_step_p95_ms"], "dolphin_step_p95_ms"),
                policy_round_trip_p95_ms=_optional_number(
                    values["policy_round_trip_p95_ms"], "policy_round_trip_p95_ms"
                ),
                reason=_optional_string(values["reason"], "reason"),
                recoveries=_integer(values["recoveries"], "recoveries"),
                updated_at=_number(values["updated_at"], "updated_at"),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("slot status contains invalid values") from error


@dataclass(frozen=True, slots=True)
class RunnerStatus:
    state: RunnerState
    message: str
    policy_sha256: str
    slots: int
    healthy_slots: int
    target_fps: float
    game_fps: float | None
    frame_interval_p95_ms: float | None
    dolphin_step_p95_ms: float | None
    policy_round_trip_p95_ms: float | None
    model_inference_p95_ms: float | None
    batch_wait_p95_ms: float | None
    recoveries: int
    updated_at: float

    def __post_init__(self) -> None:
        if not self.message:
            raise ValueError("runner status message must be non-empty")
        if len(self.policy_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.policy_sha256
        ):
            raise ValueError("runner policy_sha256 must be lowercase hexadecimal")
        if self.slots < 1 or not 0 <= self.healthy_slots <= self.slots:
            raise ValueError("runner slot counts are invalid")
        if not math.isfinite(self.target_fps) or self.target_fps <= 0:
            raise ValueError("runner target_fps must be finite and positive")
        _validate_optional_timings(self)
        if self.recoveries < 0:
            raise ValueError("runner recoveries must be non-negative")
        _finite_non_negative(self.updated_at, "runner updated_at")

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["schema_version"] = _RUNNER_SCHEMA_VERSION
        payload["state"] = self.state.value
        return payload

    @classmethod
    def from_payload(cls, payload: object) -> RunnerStatus:
        expected = {"schema_version", *cls.__dataclass_fields__}
        values = _payload(payload, expected, _RUNNER_SCHEMA_VERSION, "runner")
        try:
            return cls(
                state=RunnerState(_string(values["state"], "state")),
                message=_string(values["message"], "message"),
                policy_sha256=_string(values["policy_sha256"], "policy_sha256"),
                slots=_integer(values["slots"], "slots"),
                healthy_slots=_integer(values["healthy_slots"], "healthy_slots"),
                target_fps=_number(values["target_fps"], "target_fps"),
                game_fps=_optional_number(values["game_fps"], "game_fps"),
                frame_interval_p95_ms=_optional_number(values["frame_interval_p95_ms"], "frame_interval_p95_ms"),
                dolphin_step_p95_ms=_optional_number(values["dolphin_step_p95_ms"], "dolphin_step_p95_ms"),
                policy_round_trip_p95_ms=_optional_number(
                    values["policy_round_trip_p95_ms"], "policy_round_trip_p95_ms"
                ),
                model_inference_p95_ms=_optional_number(values["model_inference_p95_ms"], "model_inference_p95_ms"),
                batch_wait_p95_ms=_optional_number(values["batch_wait_p95_ms"], "batch_wait_p95_ms"),
                recoveries=_integer(values["recoveries"], "recoveries"),
                updated_at=_number(values["updated_at"], "updated_at"),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("runner status contains invalid values") from error


def aggregate_runner_status(
    policy_sha256: str,
    slots: tuple[SlotStatus, ...],
    now: float,
    *,
    model_inference_p95_ms: float | None,
    batch_wait_p95_ms: float | None,
) -> RunnerStatus:
    if not slots:
        raise ValueError("runner status needs at least one slot")
    healthy_states = {SlotState.IDLE, SlotState.CONNECTING, SlotState.PLAYING}
    healthy_slots = sum(status.state in healthy_states for status in slots)
    if any(status.state is SlotState.DEGRADED for status in slots):
        state = RunnerState.DEGRADED
        message = "Gameplay performance is below the 60 FPS target."
    elif healthy_slots == len(slots):
        state = RunnerState.READY
        message = "Game servers are ready."
    elif healthy_slots == 0:
        state = RunnerState.RECOVERING
        message = "Game servers are recovering."
    else:
        state = RunnerState.DEGRADED
        message = "One game server is recovering; other slots remain available."
    game_fps = tuple(status.game_fps for status in slots if status.game_fps is not None)
    frame_p95 = tuple(status.frame_interval_p95_ms for status in slots if status.frame_interval_p95_ms is not None)
    dolphin_p95 = tuple(status.dolphin_step_p95_ms for status in slots if status.dolphin_step_p95_ms is not None)
    policy_p95 = tuple(
        status.policy_round_trip_p95_ms for status in slots if status.policy_round_trip_p95_ms is not None
    )
    return RunnerStatus(
        state=state,
        message=message,
        policy_sha256=policy_sha256,
        slots=len(slots),
        healthy_slots=healthy_slots,
        target_fps=TARGET_GAME_FPS,
        game_fps=min(game_fps) if game_fps else None,
        frame_interval_p95_ms=max(frame_p95) if frame_p95 else None,
        dolphin_step_p95_ms=max(dolphin_p95) if dolphin_p95 else None,
        policy_round_trip_p95_ms=max(policy_p95) if policy_p95 else None,
        model_inference_p95_ms=model_inference_p95_ms,
        batch_wait_p95_ms=batch_wait_p95_ms,
        recoveries=sum(status.recoveries for status in slots),
        updated_at=now,
    )


def read_slot_status(path: Path) -> SlotStatus:
    return SlotStatus.from_payload(_read_json(path))


def read_runner_status(path: Path) -> RunnerStatus:
    return RunnerStatus.from_payload(_read_json(path))


def write_slot_status(path: Path, status: SlotStatus) -> None:
    _write_json(path, status.to_payload())


def write_runner_status(path: Path, status: RunnerStatus) -> None:
    _write_json(path, status.to_payload())


def _validate_optional_timings(value: object) -> None:
    for name in (
        "game_fps",
        "frame_interval_p95_ms",
        "dolphin_step_p95_ms",
        "policy_round_trip_p95_ms",
        "model_inference_p95_ms",
        "batch_wait_p95_ms",
    ):
        if hasattr(value, name) and (number := getattr(value, name)) is not None:
            _finite_non_negative(number, name)


def _payload(payload: object, expected: set[str], version: int, label: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError(f"{label} status has the wrong schema")
    values = cast(Mapping[str, object], payload)
    if values["schema_version"] != version:
        raise ValueError(f"{label} status has an unsupported schema version")
    return values


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read netplay health status {path}") from error


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True))
    temporary.replace(path)


def _finite_non_negative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def _number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _optional_number(value: object, name: str) -> float | None:
    return None if value is None else _number(value, name)


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _optional_string(value: object, name: str) -> str | None:
    return None if value is None else _string(value, name)
