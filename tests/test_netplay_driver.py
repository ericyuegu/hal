"""Netplay frame timing and fixed matchup conditioning without Dolphin."""

import time
from collections import deque
from dataclasses import replace
from unittest.mock import Mock

import melee
import pytest

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import ControllerAction
from hal.eval import netplay
from hal.eval import observations
from hal.eval.results import NetplayProgress
from hal.eval.scheduling import ActionScheduler
from hal.eval.scheduling import FrameTiming
from hal.inference.api import PolicyInput
from hal.inference.api import PolicySpec
from hal.inference.api import PredictionRequest
from hal.inference.api import RuntimeConfig
from hal.inference.api import action_plan
from hal.inference.client import InferenceUnavailable
from hal.sim.netplay import NetplaySetup
from hal.sim.session import FrameTimeout


def _frame(frame_id: int, applied: ControllerAction) -> dict:
    pre = {
        "joystick": {"x": applied.main_x, "y": applied.main_y},
        "cstick": {"x": applied.c_x, "y": applied.c_y},
        "triggers_physical": {"l": applied.trigger_l, "r": applied.trigger_r},
        "buttons_physical": applied.buttons,
    }
    return {
        "id": frame_id,
        "stage": 31,
        "ports": {
            1: {"leader": {"pre": pre, "post": {"character": 19 if frame_id >= 2 else 1}}},
            2: {"leader": {"pre": pre, "post": {"character": 22}}},
        },
    }


class _Session:
    online_delay = 2
    realtime = True
    ego_port = 1
    opponent_port = 2

    def __init__(self) -> None:
        self.frame_id = -2
        self.queue = deque((NEUTRAL_CONTROLLER_ACTION,) * 2)
        self.submitted: list[tuple[int, ControllerAction]] = []
        self.frame_times: list[float] = []

    def start_match(self, _setup, *, on_countdown_frame, on_countdown_observation):
        for frame_id in (-2, -1):
            self.frame_id = frame_id
            frame = _frame(frame_id, NEUTRAL_CONTROLLER_ACTION)
            on_countdown_observation(frame)
            self.submit(on_countdown_frame(frame))
            self.queue.popleft()
        self.frame_id = 0
        return _frame(0, NEUTRAL_CONTROLLER_ACTION)

    def start_rematch(self, setup, *, on_countdown_frame, on_countdown_observation):
        self.frame_id = -2
        self.queue = deque((NEUTRAL_CONTROLLER_ACTION,) * 2)
        return self.start_match(
            setup,
            on_countdown_frame=on_countdown_frame,
            on_countdown_observation=on_countdown_observation,
        )

    def submit(self, action: ControllerAction) -> None:
        self.submitted.append((self.frame_id, action))
        self.queue.append(action)

    def read_frames(self) -> tuple[list[dict], bool]:
        self.frame_id += 1
        applied = self.queue.popleft()
        self.frame_times = [time.perf_counter()]
        return [_frame(self.frame_id, applied)], self.frame_id < 6


class _Client:
    spec = PolicySpec("fake", "fake", ("stage", "p1_character"), (2,))
    context_frames = 8
    last_latency = 0.001

    def __init__(self) -> None:
        self.requests: list[PredictionRequest] = []
        self.pending: PredictionRequest | None = None
        self.generation = 0
        self.active = False

    @property
    def busy(self) -> bool:
        return self.pending is not None

    def start_match(self, stream_id: int, prefix_frames: int) -> int:
        assert stream_id == 0 and prefix_frames in (3, 4)
        assert not self.active
        self.generation += 1
        self.active = True
        return self.generation

    def close_match(self) -> None:
        self.active = False

    def submit(self, request: PredictionRequest) -> None:
        assert self.active
        self.requests.append(request)
        self.pending = request

    def poll(self):
        if self.pending is None:
            return None
        request = self.pending
        self.pending = None
        return action_plan(request, (ControllerAction(0.5, 0, 0, 0, 0, 0, 0),) * (8 - len(request.fixed_actions)))


def test_netplay_countdown_frame_targets_and_constant_character(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        observations,
        "flatten_canonical_frame",
        lambda frame: {
            "stage": frame["_matchup"]["stage"],
            "p1_character": frame["_matchup"]["character"][1],
        },
    )
    monkeypatch.setattr(netplay.Trajectory, "from_capture", lambda frames, _ports: frames)
    session = _Session()
    client = _Client()

    result = netplay.run_netplay_match(
        session,
        NetplaySetup(melee.Character.FOX, "TEST#1"),
        client,
        RuntimeConfig(1, (2,)),
        FrameTiming(2, 2, 4, 2, 8),
        max_frames=10,
    )

    assert client.requests[0].source_frame == -2
    assert client.requests[0].fixed_actions == (NEUTRAL_CONTROLLER_ACTION,) * 4
    assert any(request.source_frame == 0 for request in client.requests)
    assert all(item.observation["p1_character"] == 1 for request in client.requests for item in request.observations)
    assert any(action.main_x == 0.5 for _, action in session.submitted)
    assert result.ego_port == 1
    assert result.stage == 31
    assert result.connection_countdown_seconds >= 0
    assert result.match_end_seconds >= 0
    assert len(result.inference_source_frames) == len(result.inference_seconds)
    assert len(result.plan_decisions) == len(result.inference_source_frames)
    assert result.generation == client.generation
    assert all(decision.request_generation == result.generation for decision in result.plan_decisions)
    assert tuple(decision.request_source_frame for decision in result.plan_decisions) == result.inference_source_frames
    assert set(result.inference_source_frames) <= {request.source_frame for request in client.requests}
    assert result.schedule_events[0].phase == "countdown"
    assert all(
        event.phase == ("countdown" if event.choice_frame < 0 else "gameplay") for event in result.schedule_events
    )
    assert result.schedule_events[-1].submission_gaps == 0


def test_countdown_input_mask_ends_at_negative_45() -> None:
    scheduler = ActionScheduler(FrameTiming(2, 2, 4, 2, 8), 8, 1)
    moved = ControllerAction(0.5, 0, 0, 0, 0, 0, 0)
    for frame_id in (-46, -45):
        scheduler.submitted[frame_id] = moved
        scheduler.observe(PolicyInput(0, frame_id, 1, {}, NEUTRAL_CONTROLLER_ACTION))
    assert scheduler.transport_corrections == 1


def test_one_frame_handoff_submits_first_tail_at_next_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        observations,
        "flatten_canonical_frame",
        lambda frame: {
            "stage": frame["_matchup"]["stage"],
            "p1_character": frame["_matchup"]["character"][1],
        },
    )
    monkeypatch.setattr(netplay.Trajectory, "from_capture", lambda frames, _ports: frames)
    session = _Session()
    client = _Client()

    netplay.run_netplay_match(
        session,
        NetplaySetup(melee.Character.FOX, "TEST#1"),
        client,
        RuntimeConfig(1, (2,)),
        FrameTiming(2, 1, 3, 4, 8),
        max_frames=10,
    )

    assert client.requests[0].source_frame == -2
    assert len(client.requests[0].fixed_actions) == 3
    assert (-1, ControllerAction(0.5, 0, 0, 0, 0, 0, 0)) in session.submitted


def test_rematch_rejects_previous_generation_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        observations,
        "flatten_canonical_frame",
        lambda frame: {
            "stage": frame["_matchup"]["stage"],
            "p1_character": frame["_matchup"]["character"][1],
        },
    )
    monkeypatch.setattr(netplay.Trajectory, "from_capture", lambda frames, _ports: frames)
    session = _Session()
    client = _Client()
    setup = NetplaySetup(melee.Character.FOX, "TEST#1")
    runtime = RuntimeConfig(1, (2,))
    timing = FrameTiming(2, 2, 4, 2, 8)

    netplay.run_netplay_match(session, setup, client, runtime, timing, max_frames=10)
    second = netplay.run_netplay_match(session, setup, client, runtime, timing, rematch=True, max_frames=10)

    assert {request.generation for request in client.requests} == {1, 2}
    assert any(request.generation == 2 and request.source_frame == -2 for request in client.requests)
    assert second.trajectory[0]["id"] == 0


def test_netplay_rejects_missing_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        observations,
        "flatten_canonical_frame",
        lambda frame: {
            "stage": frame["_matchup"]["stage"],
            "p1_character": frame["_matchup"]["character"][1],
        },
    )

    class SkippingSession(_Session):
        def read_frames(self) -> tuple[list[dict], bool]:
            frames, in_game = super().read_frames()
            self.frame_id += 1
            return [_frame(self.frame_id, NEUTRAL_CONTROLLER_ACTION)], in_game

    with pytest.raises(RuntimeError, match="skipped observation frame 1 before 2"):
        netplay.run_netplay_match(
            SkippingSession(),
            NetplaySetup(melee.Character.FOX, "TEST#1"),
            _Client(),
            RuntimeConfig(1, (2,)),
            FrameTiming(2, 2, 4, 2, 8),
            max_frames=10,
        )


def test_dolphin_timeout_has_separate_failure_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        observations,
        "flatten_canonical_frame",
        lambda frame: {
            "stage": frame["_matchup"]["stage"],
            "p1_character": frame["_matchup"]["character"][1],
        },
    )

    class TimedOutSession(_Session):
        def read_frames(self) -> tuple[list[dict], bool]:
            raise FrameTimeout("no frames")

    session = TimedOutSession()
    client = _Client()
    failures: list[tuple[NetplayProgress, BaseException]] = []

    def capture(progress: NetplayProgress, error: BaseException) -> None:
        failures.append((progress, error))

    with pytest.raises(netplay.DolphinConnectionLost, match="Dolphin connection lost"):
        netplay.run_netplay_match(
            session,
            NetplaySetup(melee.Character.FOX, "TEST#1"),
            client,
            RuntimeConfig(1, (2,)),
            FrameTiming(2, 2, 4, 2, 8),
            max_frames=10,
            on_failure=capture,
        )
    assert session.submitted[-1][1] == NEUTRAL_CONTROLLER_ACTION
    assert not client.active
    progress, error = failures[0]
    assert isinstance(error, netplay.DolphinConnectionLost)
    assert progress.observed_frame_ids == (-2, -1, 0)
    assert progress.stream_id == 0 and progress.generation == 1
    assert progress.schedule_events[0].phase == "countdown"


def test_valid_late_plans_without_an_initial_plan_end_neutral_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        observations,
        "flatten_canonical_frame",
        lambda frame: {
            "stage": frame["_matchup"]["stage"],
            "p1_character": frame["_matchup"]["character"][1],
        },
    )
    clock = [0.0]
    monkeypatch.setattr(netplay.time, "monotonic", lambda: clock[0])

    class LateClient(_Client):
        ready = False

        def poll(self):
            if not self.ready:
                return None
            self.ready = False
            return super().poll()

    client = LateClient()
    generation = client.start_match(0, 3)
    schedule = ActionScheduler(FrameTiming(2, 1, 3, 4, 8), client.context_frames, generation)
    lifecycle = netplay._NetplayLifecycle(
        _Session(),
        client,
        schedule,
        player_identity=None,
        policy_settings=None,
        observer=None,
        schedule_observer=None,
        stream_id=0,
    )

    for frame_id in range(-2, 11):
        clock[0] = max(0.0, (frame_id - 4) * 0.3)
        lifecycle.advance(_frame(frame_id, NEUTRAL_CONTROLLER_ACTION))
        client.ready = frame_id in (4, 10)
        lifecycle.choose(frame_id)
    assert schedule.last_accepted_source_frame is None
    assert schedule.last_consumed_source_frame == 4
    assert schedule.deadline_misses >= 2
    assert schedule.exhausted_chunks == 0
    assert (
        len(lifecycle.plan_decisions)
        == len(lifecycle.inference_seconds)
        == len(lifecycle.inference_source_frames)
        == 2
    )
    assert all(not decision.accepted for decision in lifecycle.plan_decisions)
    assert tuple(decision.request_source_frame for decision in lifecycle.plan_decisions) == tuple(
        lifecycle.inference_source_frames
    )
    assert lifecycle.progress().plan_decisions == tuple(lifecycle.plan_decisions)

    clock[0] = 2.1
    lifecycle.advance(_frame(11, NEUTRAL_CONTROLLER_ACTION))
    with pytest.raises(netplay.NoUsableActionPlan, match="no usable action plan for two seconds"):
        lifecycle.choose(11)


@pytest.mark.parametrize("failure", ["malformed", "no_active_request"])
def test_invalid_response_has_no_valid_plan_trace(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    monkeypatch.setattr(
        observations,
        "flatten_canonical_frame",
        lambda frame: {
            "stage": frame["_matchup"]["stage"],
            "p1_character": frame["_matchup"]["character"][1],
        },
    )

    class MalformedClient(_Client):
        def poll(self):
            plan = super().poll()
            return replace(plan, sequence=plan.sequence + 1) if plan is not None and failure == "malformed" else plan

    client = MalformedClient()
    generation = client.start_match(0, 3)
    schedule = ActionScheduler(FrameTiming(2, 1, 3, 4, 8), client.context_frames, generation)
    lifecycle = netplay._NetplayLifecycle(
        _Session(),
        client,
        schedule,
        player_identity=None,
        policy_settings=None,
        observer=None,
        schedule_observer=None,
        stream_id=0,
    )
    lifecycle.advance(_frame(-2, NEUTRAL_CONTROLLER_ACTION))
    lifecycle.choose(-2)
    lifecycle.advance(_frame(-1, NEUTRAL_CONTROLLER_ACTION))
    if failure == "no_active_request":
        schedule.request = None
    lifecycle.choose(-1)
    assert schedule.inference_failed
    assert schedule.last_decision is None
    assert lifecycle.plan_decisions == lifecycle.inference_seconds == lifecycle.inference_source_frames == []


def test_no_usable_plan_releases_stream_and_release_failure_stops_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        netplay._NetplayLifecycle,
        "run",
        Mock(side_effect=netplay.NoUsableActionPlan("no usable plan")),
    )
    setup = NetplaySetup(melee.Character.FOX, "TEST#1")
    runtime = RuntimeConfig(1, (2,))
    timing = FrameTiming(2, 1, 3, 4, 8)
    client = _Client()
    with pytest.raises(netplay.NoUsableActionPlan, match="no usable plan"):
        netplay.run_netplay_match(_Session(), setup, client, runtime, timing)
    assert not client.active

    class FailedReleaseClient(_Client):
        def close_match(self) -> None:
            self.active = False
            raise InferenceUnavailable("stream release failed")

    failed_client = FailedReleaseClient()
    with pytest.raises(InferenceUnavailable, match="stream release failed"):
        netplay.run_netplay_match(_Session(), setup, failed_client, runtime, timing)
    assert not failed_client.active


def test_failed_countdown_keeps_primary_error_when_stream_release_fails() -> None:
    class FailedSession(_Session):
        def start_match(self, *_args: object, **_kwargs: object) -> dict:
            raise RuntimeError("countdown failed")

    class FailedReleaseClient(_Client):
        def close_match(self) -> None:
            self.active = False
            raise InferenceUnavailable("release timed out")

    def failed_capture(_progress: NetplayProgress, _error: BaseException) -> None:
        raise OSError("diagnostic write failed")

    client = FailedReleaseClient()
    with pytest.raises(RuntimeError, match="countdown failed") as raised:
        netplay.run_netplay_match(
            FailedSession(),
            NetplaySetup(melee.Character.FOX, "TEST#1"),
            client,
            RuntimeConfig(1, (2,)),
            FrameTiming(2, 1, 3, 4, 8),
            on_failure=failed_capture,
        )
    assert not client.active
    assert any("diagnostic write failed" in note for note in raised.value.__notes__)
