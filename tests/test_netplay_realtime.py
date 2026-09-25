"""Frame scheduling and delivery regressions without an emulator or GPU."""

import threading
import time
from dataclasses import replace
from multiprocessing import Pipe
from types import SimpleNamespace
from unittest.mock import Mock

import melee
import pytest

from hal.controller import NEUTRAL_CONTROLLER_ACTION as NEUTRAL
from hal.controller import ControllerAction
from hal.inference.api import PolicyInput
from hal.inference.api import PolicySpec
from hal.inference.chunks import ChunkRequest
from hal.inference.chunks import TimingSchedule
from hal.inference.chunks import chunk_response
from hal.inference.chunks import contiguous_horizons
from hal.inference.chunks import latency_frames
from hal.netplay_service.calibration import Measurement
from hal.netplay_service.calibration import select_schedule
from hal.netplay_service.chunks import EngineLost
from hal.netplay_service.chunks import RemoteChunkPolicy
from hal.netplay_service.health import ChunkHealth
from hal.netplay_service.health import RuntimeHealth
from hal.netplay_service.health import SlotState
from hal.netplay_service.health import SlotStatus
from hal.netplay_service.schedule import FrameSchedule
from hal.sim.netplay import NetplaySession


def action(value: float) -> ControllerAction:
    return ControllerAction(value, 0, 0, 0, 0, 0, 0)


def observation(frame: int, applied: ControllerAction = NEUTRAL) -> PolicyInput:
    return PolicyInput(0, frame, 1, {}, applied, (NEUTRAL,) * 2)


def scheduler() -> FrameSchedule:
    result = FrameSchedule(TimingSchedule(1, 2, 8), 8, 1)
    result.observe(observation(0))
    return result


def test_single_buffer_and_offset_one_contract() -> None:
    timing = TimingSchedule(1, 2, 8)
    assert (timing.budget_frames, timing.prefix_frames, timing.replan_frames, timing.reserve_frames) == (2, 4, 2, 2)
    schedule = scheduler()
    schedule.submitted[1] = action(0.1)
    schedule.submitted[2] = action(0.2)
    schedule.planned.update({3: action(0.3), 4: action(0.4)})
    request = schedule.begin_request()
    assert request is not None
    assert request.forced_prefix == tuple(action(i / 10) for i in range(1, 5))
    assert schedule.begin_request() is None
    response = chunk_response(request, (action(0.5), action(0.6), action(0.7), action(0.8)))
    assert response.actions[4].target_frame == 5
    assert schedule.receive(response)
    schedule.handoff(1)
    assert schedule.request is request
    assert schedule.submit(1) == action(0.4)
    schedule.handoff(2)
    assert schedule.request is None
    assert schedule.submit(2) == action(0.5)


def test_late_suffix_retains_previous_reserve_and_records_conditioning_mismatch() -> None:
    schedule = scheduler()
    schedule.planned = {frame: action(-0.5) for frame in range(1, 10)}
    request = schedule.begin_request()
    assert request is not None
    for frame in range(5):
        if frame:
            schedule.observe(observation(frame, action(-0.5)))
        assert schedule.submit(frame) == action(-0.5)
    response = chunk_response(request, (action(0.5), action(0.6), action(0.7), action(0.8)))
    schedule.receive(response)
    schedule.observe(observation(5, action(-0.5)))
    schedule.handoff(5)
    assert schedule.deadline_misses == 3
    assert schedule.prefix_mismatches == 3
    assert schedule.submit(5) == action(0.8)
    assert schedule.submit(6) == action(-0.5)  # unused old tail survives the handoff
    next_request = schedule.begin_request()
    assert next_request is not None and next_request.source_frame == 5
    assert next_request.context[-1].applied_action == action(-0.5)


def test_exhaustion_keeps_playing_neutral_and_requests_latest_context() -> None:
    schedule = scheduler()
    request = schedule.begin_request()
    assert request is not None
    for frame in range(15):
        if frame:
            schedule.observe(observation(frame))
        schedule.submit(frame)
    schedule.receive(chunk_response(request, (action(0.5),) * 4))
    schedule.handoff(14)
    assert schedule.exhausted_chunks == 1
    assert schedule.neutral_fallback_frames > 0
    latest = schedule.begin_request()
    assert latest is not None and latest.source_frame == 14
    assert [item.frame_id for item in latest.context] == list(range(7, 15))
    assert schedule.submit(15) == NEUTRAL


def test_countdown_rollback_and_rematch_identity() -> None:
    schedule = FrameSchedule(TimingSchedule(1, 2, 8), 8, 1)
    for frame in range(-5, 1):
        assert schedule.observe(observation(frame))
    assert not schedule.observe(observation(-1))
    request = schedule.begin_request()
    assert request is not None
    response = chunk_response(request, (NEUTRAL,) * 4)
    assert not schedule.receive(replace(response, generation=0))
    rematch = FrameSchedule(schedule.timing, 8, 2)
    rematch.observe(observation(0))
    rematch.begin_request()
    assert not rematch.receive(response)
    assert rematch.submit(0) == NEUTRAL


def test_observation_gap_counts_missed_submissions_and_rebuilds_context() -> None:
    schedule = scheduler()
    schedule.submit(0)
    schedule.observe(observation(3, action(0.5)))
    schedule.submit(3)
    assert len(schedule.history) == 1
    assert schedule.transport_corrections == 1
    assert schedule.deadline_misses == 2


def test_engine_loss_drains_plan_before_neutral() -> None:
    schedule = scheduler()
    schedule.planned = {4: action(0.4), 5: action(0.5)}
    schedule.fail_engine()
    assert schedule.begin_request() is None
    assert not schedule.drained(3)
    assert schedule.submit(1) == action(0.4)
    assert schedule.submit(2) == action(0.5)
    assert schedule.submit(3) == NEUTRAL
    assert schedule.drained(5)


def test_calibration_rounding_shape_and_longest_feasible_horizon() -> None:
    assert latency_frames(1 / 60) == 1
    assert latency_frames(1 / 60 + 1e-9) == 2
    assert contiguous_horizons((1, 2, 3, 4, 8, 12)) == (1, 2, 3, 4)
    samples = (
        Measurement(6, 4, (0.010,) * 200),
        Measurement(8, 4, (0.012,) * 200),
        Measurement(12, 5, (0.020,) * 200),
        Measurement(12, 3, (0.010,) * 200),  # measured shape cannot cover latency
    )
    assert select_schedule(samples, 2) == TimingSchedule(1, 2, 8)
    with pytest.raises(RuntimeError, match="unavailable"):
        select_schedule((Measurement(4, 3, (0.030,) * 200),), 2)
    with pytest.raises(ValueError, match="horizon"):
        TimingSchedule(1, 3, 6)


def test_libmelee_read_only_step_does_not_flush_but_default_does() -> None:
    console = melee.Console.__new__(melee.Console)
    controller = Mock()
    console.controllers = [controller]
    console._frametimestamp = time.time()
    console._temp_gamestate = melee.GameState()
    console._slippstream = Mock()
    console._slippstream.dispatch.return_value = None
    console._polling_mode = True
    console._polling_timeout = 0
    assert console.step(flush_controllers=False) is None
    controller.flush.assert_not_called()
    assert console.step() is None
    controller.flush.assert_called_once()


def test_observation_drain_has_no_implicit_writes(tmp_path, monkeypatch) -> None:
    session = NetplaySession(
        "iso", dolphin_path="dolphin", user_json_path="user", online_delay=2, replay_dir=tmp_path, realtime=True
    )
    session._console = Mock()
    session._controller = Mock()
    session._last_frame_id = 0
    states = [SimpleNamespace(menu_state=melee.Menu.IN_GAME, frame=frame) for frame in (1, 2, 3)]
    session._console.step.side_effect = [*states, None]
    monkeypatch.setattr("hal.sim.netplay.canonical_frame", lambda state: {"id": state.frame})
    frames, in_game = session.read_frames()
    assert in_game and [frame["id"] for frame in frames] == [1, 2, 3]
    assert len(session.frame_times) == 3
    session._controller.flush.assert_not_called()
    assert all(call.kwargs == {"flush_controllers": False} for call in session._console.step.call_args_list)


def test_delivery_does_not_block_worker_and_engine_loss_is_explicit() -> None:
    parent, child = Pipe()
    lost = threading.Event()
    client = RemoteChunkPolicy(PolicySpec("test", "test", (), (2,)), 8, (TimingSchedule(1, 2, 8),), child, lost)
    request = ChunkRequest(0, 1, 0, 0, (observation(0),), (NEUTRAL,) * 4)
    gate = threading.Event()

    def engine() -> None:
        received = parent.recv()
        gate.wait(2)
        parent.send(chunk_response(received, (action(0.5),) * 4))

    thread = threading.Thread(target=engine)
    thread.start()
    try:
        client.submit(request)
        for _ in range(100):
            assert client.poll() is None
        assert client.busy
        with pytest.raises(RuntimeError, match="outstanding"):
            client.submit(request)
        gate.set()
        deadline = time.monotonic() + 2
        response = None
        while response is None and time.monotonic() < deadline:
            response = client.poll()
            time.sleep(0.001)
        assert response is not None and response.source_frame == 0
        lost.set()
        with pytest.raises(EngineLost):
            client.poll()
    finally:
        gate.set()
        thread.join(2)
        parent.close()
        child.close()


def test_calibrated_health_requires_five_affected_and_five_clean_seconds() -> None:
    monitor = RuntimeHealth()
    timing = TimingSchedule(1, 2, 8)
    monitor.chunk_health = ChunkHealth(timing)
    monitor.begin(2, 0)
    for frame in range(1, 361):
        now = frame / 60
        monitor.observe_chunks(ChunkHealth(timing, deadline_misses=frame), now)
        monitor.observe_frame(frame, 0.001, now)
        status = monitor.snapshot(now)
        if now < 5:
            assert status.reason is None
    assert status.reason == "chunk_deadlines_missed"
    assert not status.recovery_required
    for frame in range(361, 781):
        now = frame / 60
        monitor.observe_frame(frame, 0.001, now)
        status = monitor.snapshot(now)
        if now < 11:
            assert status.reason is not None
    assert status.reason is None
    record = SlotStatus(0, SlotState.PLAYING, 60, 16.7, 1, 10, None, 0, 13, ChunkHealth(timing, deadline_misses=360))
    assert SlotStatus.from_payload(record.to_payload()) == record


def test_full_request_calibration_counts_warmups_and_measurements_for_each_delay() -> None:
    from hal.inference.api import RuntimeConfig
    from hal.netplay_service.calibration import measure_shape

    class Policy:
        spec = PolicySpec("test", "test", (), (2, 3))
        context_frames = 4
        supported_horizons = (8,)
        calls = 0
        resets = 0
        prefixes = set()

        def prepare_chunks(self, runtime, horizon, prefix):
            assert (runtime.max_batch_size, horizon, prefix) == (2, 8, 5)

        def reset_chunks(self):
            self.resets += 1

        def warmup_context(self, stream, source, delay):
            return tuple(
                PolicyInput(stream, frame, 1, {}, NEUTRAL, (NEUTRAL,) * delay)
                for frame in range(source - 3, source + 1)
            )

        def plan_chunks(self, requests):
            self.calls += len(requests)
            self.prefixes.update(len(request.forced_prefix) for request in requests)
            return tuple(
                chunk_response(request, (NEUTRAL,) * (8 - len(request.forced_prefix))) for request in requests
            )

    policy = Policy()
    result = measure_shape(policy, RuntimeConfig(2, (2, 3)), 8, 5, 0.001)
    assert policy.calls == 2 * 2 * (20 + 200)
    assert len(result.seconds) == 400
    assert policy.prefixes == {4, 5}
    assert policy.resets == 3


def test_worker_drains_chunk_and_flushes_neutral_on_confirmed_engine_loss(monkeypatch) -> None:
    import hal.eval.realtime as realtime
    from hal.inference.api import RuntimeConfig

    class Client:
        schedules = (TimingSchedule(1, 2, 8),)
        context_frames = 8
        busy = False
        last_latency = 0.010
        request = None
        delivered = False
        frame = 0

        def start_match(self):
            return 1

        def submit(self, request):
            self.request = request
            self.busy = True

        def poll(self):
            if self.frame >= 3:
                raise EngineLost("confirmed loss")
            if self.request is not None and not self.delivered:
                self.delivered = True
                self.busy = False
                return chunk_response(self.request, (action(0.5),) * 4)
            return None

    client = Client()

    def frame(frame_id):
        return {
            "id": frame_id,
            "stage": 31,
            "ports": {
                port: {
                    "leader": {
                        "pre": {
                            "joystick": {"x": 0.0, "y": 0.0},
                            "cstick": {"x": 0.0, "y": 0.0},
                            "triggers_physical": {"l": 0.0, "r": 0.0},
                            "buttons_physical": 0,
                        },
                        "post": {"character": 1},
                    }
                }
                for port in (1, 2)
            },
        }

    class Session:
        realtime = True
        online_delay = 2
        ego_port = 1
        opponent_port = 2
        submitted = []
        frame_times = []

        def start_match(self, *_args, **_kwargs):
            return frame(0)

        def submit(self, action):
            self.submitted.append((client.frame, action))

        def read_frames(self):
            client.frame += 1
            self.frame_times = [time.perf_counter()]
            return [frame(client.frame)], True

    session = Session()
    monkeypatch.setattr(realtime, "flatten_canonical_frame", lambda _frame: {})
    with pytest.raises(EngineLost, match="forfeit"):
        realtime.run_realtime_match(session, Mock(), client, RuntimeConfig(1, (2,)))
    assert client.frame == 8
    assert any(selected == action(0.5) for _, selected in session.submitted)
    assert session.submitted[-1] == (8, NEUTRAL)


@pytest.mark.parametrize("flush", [False, True])
@pytest.mark.parametrize("event", ["GAME_START", "FRAME_BOOKEND"])
def test_libmelee_read_only_mode_also_suppresses_event_side_effects(flush, event) -> None:
    from melee.slippstream import EventType

    console = melee.Console.__new__(melee.Console)
    console.controllers = [Mock()]
    console._flush_controllers = flush
    console._events_this_frame = []
    console.eventsize = {getattr(EventType, event).value: 1}
    console._Console__game_start = Mock()
    console._Console__frame_bookend = Mock()
    console._frame = 5
    console.skip_rollback_frames = True
    console.rollback_resolution = "first"
    console.blocking_input = True
    state = melee.GameState()
    state.frame = 5
    console._Console__handle_slippstream_events(bytes([getattr(EventType, event).value]), state)
    assert console.controllers[0].flush.call_count == int(flush)
    assert console.controllers[0].release_all.call_count == int(flush and event == "GAME_START")


def test_neutral_fallback_counts_reserved_startup_and_exhausted_tail_once() -> None:
    schedule = scheduler()
    request = schedule.begin_request()
    assert request is not None
    schedule.submit(0)
    schedule.submit(1)
    assert schedule.neutral_fallback_frames == 2
    schedule.receive(chunk_response(request, (NEUTRAL,) * 4))
    schedule.handoff(2)
    for frame in range(2, 6):
        schedule.submit(frame)
    assert schedule.neutral_fallback_frames == 2  # these neutral actions came from a usable plan
    schedule.submit(6)
    schedule.submit(7)
    assert schedule.neutral_fallback_frames == 4
    assert schedule.exhausted_chunks == 1
