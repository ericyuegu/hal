"""Frame scheduling and delivery regressions without an emulator or GPU."""

import threading
import time
from dataclasses import replace
from multiprocessing import Pipe
from types import SimpleNamespace
from unittest.mock import Mock

import melee
import pytest

import hal.inference.benchmark as benchmark
from hal.controller import NEUTRAL_CONTROLLER_ACTION as NEUTRAL
from hal.controller import ControllerAction
from hal.eval.scheduling import ActionScheduler
from hal.eval.scheduling import FrameTiming
from hal.inference.api import PolicyInput
from hal.inference.api import PolicySpec
from hal.inference.api import PredictionRequest
from hal.inference.api import PreparedInferenceProfile
from hal.inference.api import action_plan
from hal.inference.client import InferenceClient
from hal.inference.client import InferenceUnavailable
from hal.inference.client import StreamAck
from hal.inference.client import StreamAdmission
from hal.inference.client import WorkerFailure
from hal.inference.engine import InferenceEngine as InferenceWorker
from hal.inference.engine import start_inference_worker
from hal.netplay_service.health import ChunkHealth
from hal.netplay_service.health import RuntimeHealth
from hal.netplay_service.health import SlotState
from hal.netplay_service.health import SlotStatus
from hal.sim.netplay import NetplaySession

PROFILE = PreparedInferenceProfile("test-delay-2", "a" * 64, "kv_cache", 8, 4, (1, 2, 4), 2)


def action(value: float) -> ControllerAction:
    return ControllerAction(value, 0, 0, 0, 0, 0, 0)


def observation(frame: int, applied: ControllerAction = NEUTRAL) -> PolicyInput:
    return PolicyInput(0, frame, 1, {}, applied)


def scheduler() -> ActionScheduler:
    result = ActionScheduler(FrameTiming(2, 2, 4, 2, 8), 8, 1)
    result.observe(observation(0))
    return result


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
    client = InferenceClient(PolicySpec("test", "test", (), (2,)), 8, child, lost, {4: PROFILE})
    parent.send(StreamAck(0, 1, "admitted"))
    client.start_match(0, 4)
    assert parent.recv() == StreamAdmission(0, 1, PROFILE)
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
    worker = InferenceWorker({PROFILE: policy}, {0: parent}, batch_wait_seconds=0)
    try:
        child.send(StreamAdmission(0, 1, PROFILE))
        worker.serve_batch((parent,), 0)
        assert child.recv() == StreamAck(0, 1, "admitted")
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
    worker = InferenceWorker({PROFILE: policy}, {0: first_parent, 1: second_parent}, batch_wait_seconds=0)
    try:
        first_child.send(StreamAdmission(0, 1, PROFILE))
        second_child.send(StreamAdmission(1, 1, PROFILE))
        worker.serve_batch((first_parent, second_parent), 0)
        assert first_child.recv() == StreamAck(0, 1, "admitted")
        assert second_child.recv() == StreamAck(1, 1, "admitted")
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
    client = InferenceClient(PolicySpec("test", "test", (), (2,)), 8, child, threading.Event(), {4: PROFILE})

    def fail_start(_thread: threading.Thread) -> None:
        raise RuntimeError("thread capacity exhausted")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    try:
        with pytest.raises(InferenceUnavailable, match="could not start"):
            client.start_match(0, 4)
        assert not client.busy
    finally:
        parent.close()
        child.close()


def test_inference_worker_scope_closes_client_on_caller_error() -> None:
    policy = Mock()
    policy.spec = PolicySpec("test", "test", (), (2,))
    policy.context_frames = 8
    with pytest.raises(RuntimeError, match="caller failed"), start_inference_worker(policy, PROFILE, 0) as client:
        raise RuntimeError("caller failed")
    assert client.connection.closed


def test_calibrated_health_requires_five_affected_and_five_clean_seconds() -> None:
    monitor = RuntimeHealth()
    timing = FrameTiming(2, 2, 4, 2, 8)
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


def test_incremental_benchmark_uses_one_declared_prefix_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    from hal.inference.api import RuntimeConfig
    from hal.inference.benchmark import measure_prediction_shape

    class Policy:
        spec = PolicySpec("test", "test", (), (2, 3))
        checkpoint_sha256 = "a" * 64
        history_mode = "kv_cache"
        prepared_update_shapes = (1, 2, 4)
        context_frames = 4
        supported_horizons = (8,)
        calls = 0
        freezes = 0
        resets = 0
        prefixes = set()
        sources = []
        observation_counts = []
        reset_flags = []

        def prepare_prediction(self, runtime, horizon, prefix):
            assert (runtime.max_batch_size, horizon, prefix) == (2, 8, 5)

        def validate_prepared_profile(self, profile):
            assert (profile.checkpoint_sha256, profile.execution_mode, profile.fixed_prefix_frames) == (
                self.checkpoint_sha256,
                self.history_mode,
                5,
            )

        def reset_prediction(self):
            self.resets += 1

        def freeze_after_warmup(self) -> None:
            assert self.calls == 2 * benchmark.WARMUP_CALLS
            self.freezes += 1

        def predict(self, requests):
            self.calls += len(requests)
            self.prefixes.update(len(request.fixed_actions) for request in requests)
            self.sources.append(requests[0].source_frame)
            self.observation_counts.append(len(requests[0].observations))
            self.reset_flags.append(requests[0].observations[0].reset)
            return tuple(action_plan(request, (NEUTRAL,) * (8 - len(request.fixed_actions))) for request in requests)

        def release_stream(self, _stream_id):
            pass

    policy = Policy()
    monkeypatch.setattr(benchmark, "freeze_inference_runtime", policy.freeze_after_warmup)
    result = measure_prediction_shape(policy, RuntimeConfig(2, (3,), replan_interval_frames=1), 8, 5, 0.0005)
    assert policy.calls == 2 * (20 + 200)
    assert policy.freezes == 1
    assert len(result.seconds) == 200
    assert policy.prefixes == {5}
    assert policy.sources[:3] == [4, 5, 6]
    assert policy.observation_counts[:3] == [4, 1, 1]
    assert policy.reset_flags[:3] == [True, False, False]
    assert policy.resets == 2
    with pytest.raises(ValueError, match="one transport delay"):
        measure_prediction_shape(policy, RuntimeConfig(2, (2, 3)), 8, 5, 0.0005)


def test_benchmark_does_not_reset_policy_while_worker_is_still_running(monkeypatch: pytest.MonkeyPatch) -> None:
    import hal.inference.benchmark as benchmark
    from hal.inference.api import PredictionPolicy
    from hal.inference.api import RuntimeConfig

    policy = Mock(spec=PredictionPolicy)
    policy.spec = PolicySpec("test", "test", (), (2,))
    policy.context_frames = 8
    policy.checkpoint_sha256 = "a" * 64
    policy.history_mode = "kv_cache"
    policy.prepared_update_shapes = (1, 2, 4)
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

    monkeypatch.setattr(benchmark.InferenceEngine, "serve", blocked_serve)
    monkeypatch.setattr(benchmark.InferenceClient, "start_match", lambda _client, _stream_id, _prefix: 1)
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
        release_attempted = False

        def start_match(self, _stream_id, _prefix_frames):
            return 1

        def submit(self, request):
            self.request = request
            self.busy = True

        def close_match(self) -> None:
            self.release_attempted = True
            raise InferenceUnavailable("confirmed loss during release")

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
        netplay.run_netplay_match(session, Mock(), client, RuntimeConfig(1, (2,)), FrameTiming(2, 2, 4, 2, 8))
    assert client.frame == 8
    assert client.release_attempted
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
    schedule.accept_plan(action_plan(request, (NEUTRAL,) * 4), 2)
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
    from hal.representation.observations import flatten_canonical_frame
    from hal.sim.netplay import NetplaySetup
    from hal.sim.session import FrameTimeout

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
            FrameTiming(2, 2, 4, 2, 8),
        )
    if missing:
        client.submit.assert_not_called()
        return
    request = client.submit.call_args.args[0]
    flat = flatten_canonical_frame({**frame, "_matchup": {"stage": 31, "character": {1: 1, 2: 1}}})
    assert len(flat) > len(required)
    assert request.observations[0].observation == {name: flat[name] for name in required}
