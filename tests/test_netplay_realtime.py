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
from hal.eval.qualification import latency_frames
from hal.eval.qualification import select_frame_timing
from hal.eval.scheduling import ActionScheduler
from hal.eval.scheduling import FrameTiming
from hal.inference.api import PolicyInput
from hal.inference.api import PolicySpec
from hal.inference.api import PredictionRequest
from hal.inference.api import action_plan
from hal.inference.api import contiguous_horizons
from hal.inference.benchmark import LatencyMeasurement
from hal.inference.worker import InferenceClient
from hal.inference.worker import InferenceUnavailable
from hal.inference.worker import InferenceWorker
from hal.inference.worker import WorkerFailure
from hal.inference.worker import start_inference_worker
from hal.netplay_service.health import ChunkHealth
from hal.netplay_service.health import RuntimeHealth
from hal.netplay_service.health import SlotState
from hal.netplay_service.health import SlotStatus
from hal.sim.netplay import NetplaySession


def action(value: float) -> ControllerAction:
    return ControllerAction(value, 0, 0, 0, 0, 0, 0)


def observation(frame: int, applied: ControllerAction = NEUTRAL) -> PolicyInput:
    return PolicyInput(0, frame, 1, {}, applied, (NEUTRAL,) * 2)


def scheduler() -> ActionScheduler:
    result = ActionScheduler(FrameTiming(2, 2, 2, 8), 8, 1)
    result.observe(observation(0))
    return result


def test_single_buffer_and_offset_one_contract() -> None:
    timing = FrameTiming(2, 2, 2, 8)
    assert (
        timing.thinking_allowance_frames,
        timing.fixed_prefix_frames,
        timing.replan_interval_frames,
        timing.reserve_frames,
    ) == (2, 4, 2, 2)
    schedule = scheduler()
    schedule.submitted[1] = action(0.1)
    schedule.submitted[2] = action(0.2)
    schedule.planned.update({3: action(0.3), 4: action(0.4)})
    request = schedule.request_plan()
    assert request is not None
    assert request.fixed_actions == tuple(action(i / 10) for i in range(1, 5))
    assert schedule.request_plan() is None
    response = action_plan(request, (action(0.5), action(0.6), action(0.7), action(0.8)))
    assert response.actions[0].target_frame == 5
    assert schedule.accept_plan(response)
    schedule.apply_ready_plan(1)
    assert schedule.request is request
    assert schedule.action_to_submit(1) == action(0.4)
    schedule.apply_ready_plan(2)
    assert schedule.request is None
    assert schedule.action_to_submit(2) == action(0.5)


def test_synchronous_plan_can_start_at_first_available_input_frame() -> None:
    timing = FrameTiming(2, 0, 2, 4)
    schedule = ActionScheduler(timing, 8, 1)
    schedule.observe(observation(0))
    request = schedule.request_plan()
    assert request is not None
    assert len(request.fixed_actions) == 2
    assert schedule.accept_plan(action_plan(request, (action(0.5), action(0.6))))
    schedule.apply_ready_plan(0)
    assert schedule.action_to_submit(0) == action(0.5)


@pytest.mark.parametrize("arrival_frame", (5, 6))
def test_consecutive_plans_send_only_new_observations_and_keep_reserve(arrival_frame: int) -> None:
    timing = FrameTiming(2, 1, 4, 8)
    schedule = ActionScheduler(timing, 16, 1)
    schedule.observe(observation(0))
    first = schedule.request_plan()
    assert first is not None
    assert [item.frame_id for item in first.observations] == [0]
    assert [item.target_frame for item in action_plan(first, (NEUTRAL,) * 5).actions] == [4, 5, 6, 7, 8]
    schedule.action_to_submit(0)
    schedule.observe(observation(1))
    assert schedule.accept_plan(action_plan(first, tuple(action(value / 10) for value in range(4, 9))))
    schedule.apply_ready_plan(1)
    for frame in range(1, 5):
        if frame > 1:
            schedule.observe(observation(frame))
        schedule.action_to_submit(frame)
    second = schedule.request_plan()
    assert second is not None
    assert [item.frame_id for item in second.observations] == [1, 2, 3, 4]
    assert second.fixed_actions == (action(0.5), action(0.6), action(0.7))
    second_plan = action_plan(second, tuple(action(value / 10) for value in range(1, 6)))
    assert [item.target_frame for item in second_plan.actions] == [8, 9, 10, 11, 12]
    for frame in range(5, arrival_frame + 1):
        schedule.observe(observation(frame))
        if frame < arrival_frame:
            schedule.action_to_submit(frame)
    assert schedule.accept_plan(second_plan)
    schedule.apply_ready_plan(arrival_frame)
    expected = action(0.1) if arrival_frame == 5 else action(0.2)
    assert schedule.action_to_submit(arrival_frame) == expected
    assert schedule.submitted[8] == (action(0.1) if arrival_frame == 5 else action(0.8))
    assert schedule.deadline_misses == arrival_frame - 5
    for frame in range(arrival_frame + 1, 9):
        schedule.observe(observation(frame))
    third = schedule.request_plan()
    assert third is not None
    assert [item.frame_id for item in third.observations] == [5, 6, 7, 8]


def test_fixed_action_mismatch_uses_actual_observation() -> None:
    schedule = scheduler()
    request = schedule.request_plan()
    assert request is not None
    schedule.observe(observation(1, action(0.5)))
    schedule.observe(observation(2))
    assert schedule.accept_plan(action_plan(request, (NEUTRAL,) * 4))
    schedule.apply_ready_plan(2)
    assert schedule.prefix_mismatches == 1


def test_late_suffix_retains_previous_reserve_and_records_conditioning_mismatch() -> None:
    schedule = scheduler()
    schedule.planned = {frame: action(-0.5) for frame in range(1, 10)}
    request = schedule.request_plan()
    assert request is not None
    for frame in range(5):
        if frame:
            schedule.observe(observation(frame, action(-0.5)))
        assert schedule.action_to_submit(frame) == action(-0.5)
    response = action_plan(request, (action(0.5), action(0.6), action(0.7), action(0.8)))
    schedule.accept_plan(response)
    schedule.observe(observation(5, action(-0.5)))
    schedule.apply_ready_plan(5)
    assert schedule.deadline_misses == 3
    assert schedule.prefix_mismatches == 3
    assert schedule.action_to_submit(5) == action(0.8)
    assert schedule.action_to_submit(6) == action(-0.5)  # unused old tail survives the handoff
    next_request = schedule.request_plan()
    assert next_request is not None and next_request.source_frame == 5
    assert next_request.observations[-1].applied_action == action(-0.5)


def test_exhaustion_does_not_retry_with_a_truncated_history() -> None:
    schedule = scheduler()
    request = schedule.request_plan()
    assert request is not None
    for frame in range(15):
        if frame:
            schedule.observe(observation(frame))
        schedule.action_to_submit(frame)
    schedule.accept_plan(action_plan(request, (action(0.5),) * 4))
    schedule.apply_ready_plan(14)
    assert schedule.exhausted_chunks == 1
    assert schedule.neutral_fallback_frames > 0
    with pytest.raises(RuntimeError, match="unreported observations"):
        schedule.request_plan()
    assert schedule.action_to_submit(15) == NEUTRAL


def test_countdown_rollback_and_rematch_identity() -> None:
    schedule = ActionScheduler(FrameTiming(2, 2, 2, 8), 8, 1)
    for frame in range(-5, 1):
        assert schedule.observe(observation(frame))
    assert not schedule.observe(observation(-1))
    request = schedule.request_plan()
    assert request is not None
    response = action_plan(request, (NEUTRAL,) * 4)
    assert not schedule.accept_plan(replace(response, generation=0))
    rematch = ActionScheduler(schedule.timing, 8, 2)
    rematch.observe(observation(0))
    rematch.request_plan()
    assert not rematch.accept_plan(response)
    assert rematch.action_to_submit(0) == NEUTRAL


def test_observation_gap_rejects_without_changing_pending_plan() -> None:
    schedule = scheduler()
    request = schedule.request_plan()
    assert request is not None
    submitted = schedule.action_to_submit(0)
    with pytest.raises(ValueError, match="consecutive"):
        schedule.observe(observation(3, action(0.5)))
    assert len(schedule.history) == 1
    assert schedule.request is request
    assert schedule.submitted[3] == submitted
    assert schedule.transport_corrections == 0


def test_engine_loss_drains_plan_before_neutral() -> None:
    schedule = scheduler()
    schedule.planned = {4: action(0.4), 5: action(0.5)}
    schedule.fail_inference()
    assert schedule.request_plan() is None
    assert not schedule.drained(3)
    assert schedule.action_to_submit(1) == action(0.4)
    assert schedule.action_to_submit(2) == action(0.5)
    assert schedule.action_to_submit(3) == NEUTRAL
    assert schedule.drained(5)


def test_calibration_rounding_shape_and_longest_feasible_horizon() -> None:
    assert latency_frames(1 / 60) == 1
    assert latency_frames(1 / 60 + 1e-9) == 2
    assert contiguous_horizons((1, 2, 3, 4, 8, 12)) == (1, 2, 3, 4)
    samples = (
        LatencyMeasurement(6, 4, (0.010,) * 200),
        LatencyMeasurement(8, 4, (0.020,) * 200),
        LatencyMeasurement(12, 5, (0.020,) * 200),
        LatencyMeasurement(12, 3, (0.020,) * 200),  # measured shape cannot cover latency
    )
    assert select_frame_timing(samples, 2) == FrameTiming(2, 2, 2, 8)
    with pytest.raises(RuntimeError, match="unavailable"):
        select_frame_timing((LatencyMeasurement(4, 3, (0.030,) * 200),), 2)
    with pytest.raises(ValueError, match="horizon"):
        FrameTiming(3, 2, 2, 6)


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
    client = InferenceClient(PolicySpec("test", "test", (), (2,)), 8, child, lost)
    request = PredictionRequest(0, 1, 0, 0, (observation(0),), (NEUTRAL,) * 4)
    gate = threading.Event()

    def engine() -> None:
        received = parent.recv()
        gate.wait(2)
        parent.send(action_plan(received, (action(0.5),) * 4))

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
        with pytest.raises(InferenceUnavailable):
            client.poll()
    finally:
        gate.set()
        thread.join(2)
        parent.close()
        child.close()


def test_worker_rejects_invalid_plan_and_notifies_client() -> None:
    parent, child = Pipe()
    policy = Mock()
    request = PredictionRequest(0, 1, 0, 0, (observation(0),), (NEUTRAL,) * 4)
    policy.predict.return_value = (action_plan(request, (NEUTRAL,)),)
    worker = InferenceWorker(policy, 8, {0: parent}, batch_wait_seconds=0)
    try:
        child.send(request)
        with pytest.raises(ValueError, match="prediction horizon"):
            worker.serve_batch((parent,), 0)
        assert isinstance(child.recv(), WorkerFailure)
    finally:
        parent.close()
        child.close()


def test_worker_validates_entire_batch_before_sending_any_plan() -> None:
    first_parent, first_child = Pipe()
    second_parent, second_child = Pipe()
    first = PredictionRequest(0, 1, 0, 0, (observation(0),), (NEUTRAL,) * 4)
    second = PredictionRequest(1, 1, 0, 0, (replace(observation(0), stream_id=1),), (NEUTRAL,) * 4)
    policy = Mock()
    policy.predict.return_value = (action_plan(first, (NEUTRAL,) * 4), action_plan(second, (NEUTRAL,)))
    worker = InferenceWorker(policy, 8, {0: first_parent, 1: second_parent}, batch_wait_seconds=0)
    try:
        first_child.send(first)
        second_child.send(second)
        with pytest.raises(ValueError, match="prediction horizon"):
            worker.serve_batch((first_parent, second_parent), 0)
        assert isinstance(first_child.recv(), WorkerFailure)
        assert isinstance(second_child.recv(), WorkerFailure)
    finally:
        first_parent.close()
        first_child.close()
        second_parent.close()
        second_child.close()


def test_delivery_thread_start_failure_clears_outstanding_request(monkeypatch: pytest.MonkeyPatch) -> None:
    parent, child = Pipe()
    client = InferenceClient(PolicySpec("test", "test", (), (2,)), 8, child, threading.Event())

    def fail_start(_thread: threading.Thread) -> None:
        raise RuntimeError("thread capacity exhausted")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    try:
        with pytest.raises(InferenceUnavailable, match="could not start"):
            client.submit(PredictionRequest(0, 1, 0, 0, (observation(0),), (NEUTRAL,) * 4))
        assert not client.busy
    finally:
        parent.close()
        child.close()


def test_inference_worker_scope_closes_client_on_caller_error() -> None:
    policy = Mock()
    policy.spec = PolicySpec("test", "test", (), (2,))
    policy.context_frames = 8
    with pytest.raises(RuntimeError, match="caller failed"), start_inference_worker(policy, 8, 0) as client:
        raise RuntimeError("caller failed")
    assert client.connection.closed


def test_calibrated_health_requires_five_affected_and_five_clean_seconds() -> None:
    monitor = RuntimeHealth()
    timing = FrameTiming(2, 2, 2, 8)
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


def test_incremental_benchmark_counts_warmups_and_measurements_for_each_delay() -> None:
    from hal.inference.api import RuntimeConfig
    from hal.inference.benchmark import measure_prediction_shape

    class Policy:
        spec = PolicySpec("test", "test", (), (2, 3))
        context_frames = 4
        supported_horizons = (8,)
        calls = 0
        resets = 0
        prefixes = set()
        sources = []
        observation_counts = []
        reset_flags = []

        def prepare_prediction(self, runtime, horizon, prefix):
            assert (runtime.max_batch_size, horizon, prefix) == (2, 8, 5)

        def reset_prediction(self):
            self.resets += 1

        def predict(self, requests):
            self.calls += len(requests)
            self.prefixes.update(len(request.fixed_actions) for request in requests)
            self.sources.append(requests[0].source_frame)
            self.observation_counts.append(len(requests[0].observations))
            self.reset_flags.append(requests[0].observations[0].reset)
            return tuple(action_plan(request, (NEUTRAL,) * (8 - len(request.fixed_actions))) for request in requests)

    policy = Policy()
    result = measure_prediction_shape(policy, RuntimeConfig(2, (2, 3), replan_interval_frames=1), 8, 5, 0.001)
    assert policy.calls == 2 * 2 * (20 + 200)
    assert len(result.seconds) == 400
    assert policy.prefixes == {4, 5}
    assert policy.sources[:3] == [4, 5, 6]
    assert policy.observation_counts[:3] == [4, 1, 1]
    assert policy.reset_flags[:3] == [True, False, False]
    assert policy.resets == 3


def test_benchmark_does_not_reset_policy_while_worker_is_still_running(monkeypatch: pytest.MonkeyPatch) -> None:
    import hal.inference.benchmark as benchmark
    from hal.inference.api import PredictionPolicy
    from hal.inference.api import RuntimeConfig

    policy = Mock(spec=PredictionPolicy)
    policy.spec = PolicySpec("test", "test", (), (2,))
    policy.context_frames = 8
    ready = threading.Event()
    release = threading.Event()
    threads: list[threading.Thread] = []

    def blocked_serve(_worker: InferenceWorker, _stop: threading.Event) -> None:
        threads.append(threading.current_thread())
        ready.set()
        release.wait(10)

    def failed_submit(_client: InferenceClient, _request: PredictionRequest) -> None:
        assert ready.wait(1)
        policy.reset_prediction.reset_mock()
        raise RuntimeError("request failed")

    monkeypatch.setattr(benchmark.InferenceWorker, "serve", blocked_serve)
    monkeypatch.setattr(benchmark.InferenceClient, "submit", failed_submit)
    try:
        with pytest.raises(RuntimeError, match="worker did not stop; policy cannot be reused"):
            benchmark.measure_prediction_shape(policy, RuntimeConfig(1, (2,)), 8, 3, 0)
        policy.reset_prediction.assert_not_called()
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=2)
            assert not thread.is_alive()


def test_worker_drains_chunk_and_flushes_neutral_on_confirmed_engine_loss(monkeypatch) -> None:
    import hal.eval.netplay as netplay
    from hal.inference.api import RuntimeConfig

    class Client:
        spec = PolicySpec("test", "test", (), (2,))
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
                raise InferenceUnavailable("confirmed loss")
            if self.request is not None and not self.delivered:
                self.delivered = True
                self.busy = False
                return action_plan(self.request, (action(0.5),) * 4)
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
    monkeypatch.setattr(netplay, "policy_input_from_frame", lambda frame, **kwargs: observation(int(frame["id"])))
    with pytest.raises(InferenceUnavailable, match="forfeit"):
        netplay.run_netplay_match(session, Mock(), client, RuntimeConfig(1, (2,)), FrameTiming(2, 2, 2, 8))
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
    request = schedule.request_plan()
    assert request is not None
    schedule.action_to_submit(0)
    schedule.action_to_submit(1)
    assert schedule.neutral_fallback_frames == 2
    schedule.accept_plan(action_plan(request, (NEUTRAL,) * 4))
    schedule.apply_ready_plan(2)
    for frame in range(2, 6):
        schedule.action_to_submit(frame)
    assert schedule.neutral_fallback_frames == 2  # these neutral actions came from a usable plan
    schedule.action_to_submit(6)
    schedule.action_to_submit(7)
    assert schedule.neutral_fallback_frames == 4
    assert schedule.exhausted_chunks == 1


@pytest.mark.parametrize("missing", [False, True])
def test_realtime_request_preserves_required_fields_and_omits_unused_fields(missing: bool) -> None:
    from hal.eval.netplay import DolphinConnectionLost
    from hal.eval.netplay import run_netplay_match
    from hal.inference.api import RuntimeConfig
    from hal.sim.netplay import NetplaySetup
    from hal.sim.session import FrameTimeout
    from hal.training.canonical import flatten_canonical_frame

    frame = {
        "id": 0,
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
                    "post": {"character": 1, "percent": 12.0},
                }
            }
            for port in (1, 2)
        },
    }
    required = ("missing",) if missing else ("p1_percent", "p2_percent", "stage")
    client = Mock(spec=InferenceClient)
    client.spec = PolicySpec("test", "test", required, (2,))
    client.context_frames = 8
    client.start_match.return_value = 1
    client.busy = False
    client.poll.return_value = None
    session = Mock(spec=NetplaySession)
    session.online_delay = 2
    session.realtime = True
    session.ego_port = 1
    session.opponent_port = 2
    session.start_match.return_value = frame
    session.read_frames.side_effect = FrameTimeout("test stops after the first request")
    with pytest.raises(KeyError if missing else DolphinConnectionLost):
        run_netplay_match(
            session,
            NetplaySetup(melee.Character.FOX, "TEST#1"),
            client,
            RuntimeConfig(1, (2,)),
            FrameTiming(2, 2, 2, 8),
        )
    if missing:
        client.submit.assert_not_called()
        return
    request = client.submit.call_args.args[0]
    flat = flatten_canonical_frame({**frame, "_matchup": {"stage": 31, "character": {1: 1, 2: 1}}})
    assert len(flat) > len(required)
    assert request.observations[0].observation == {name: flat[name] for name in required}
