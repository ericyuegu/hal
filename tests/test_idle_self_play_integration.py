"""Real local Dolphin cancellation and telemetry, without another model instance."""

import json
import os
import signal
import threading
import time
from contextlib import closing
from contextlib import suppress
from multiprocessing import Pipe
from pathlib import Path
from unittest.mock import Mock

import melee
import pytest

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.eval import self_play
from hal.inference.api import PolicySpec
from hal.inference.api import PreparedInferenceProfile
from hal.inference.api import action_plan
from hal.inference.client import InferenceClient
from hal.inference.engine import InferenceEngine
from hal.netplay_service.control import QueueControl
from hal.netplay_service.domain import CHARACTERS
from hal.netplay_service.domain import STAGES
from hal.netplay_service.graphics import PanelRenderer
from hal.netplay_service.queue_client import QueueEndpoint
from hal.netplay_service.self_play import DemoSelection
from hal.netplay_service.stream import GameStreamState
from hal.netplay_service.telemetry import TelemetryPublisher
from hal.netplay_service.telemetry import TelemetryReceiver
from hal.paths import ISO_PATH
from hal.paths import NETPLAY_EMULATOR_PATH
from hal.sim.netplay import tested_dolphin_version as dolphin_version
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.session import Session
from hal.sim.session import SessionCanceled

pytestmark = pytest.mark.integration


@pytest.fixture
def prerequisites() -> None:
    missing = [str(path) for path in (Path(ISO_PATH), Path(NETPLAY_EMULATOR_PATH)) if not Path(path).is_file()]
    if not os.environ.get("DISPLAY"):
        missing.append("DISPLAY (run under xvfb-run)")
    if missing:
        if os.environ.get("HAL_REQUIRE_INTEGRATION") == "1":
            pytest.fail(f"missing integration fixture: {missing}")
        pytest.skip(f"missing integration fixture: {missing}")


@pytest.mark.parametrize(
    "characters",
    [
        (melee.Character.SHEIK, melee.Character.NESS),
        (melee.Character.NESS, melee.Character.SHEIK),
        (melee.Character.SHEIK, melee.Character.SHEIK),
        (melee.Character.ZELDA, melee.Character.ZELDA),
    ],
)
def test_local_selection_preserves_sheik_and_zelda_identity(
    prerequisites, tmp_path: Path, characters: tuple[melee.Character, ...]
) -> None:
    cancel = threading.Event()
    with Session(
        ISO_PATH,
        dolphin_path=NETPLAY_EMULATOR_PATH,
        slippi_port=51841,
        replay_dir=tmp_path,
        blocking_input=True,
        polling_mode=True,
        canceled=cancel.is_set,
        disable_audio=True,
        gfx_backend="OGL",
        emulation_speed=0,
        dolphin_version=dolphin_version(str(NETPLAY_EMULATOR_PATH)),
        dolphin_batch=True,
    ) as session:
        frame = session.start_match(
            Matchup(
                melee.Stage.FOUNTAIN_OF_DREAMS,
                tuple(PlayerSetup(port, character) for port, character in enumerate(characters, 1)),
            )
        )
        for _ in range(120):
            for port, character in enumerate(characters, 1):
                assert frame["ports"][port]["leader"]["post"]["character"] == character.value
            frame, playing = session.step({1: NEUTRAL_CONTROLLER_ACTION, 2: NEUTRAL_CONTROLLER_ACTION})
            assert playing


@pytest.mark.parametrize("index", range(len(CHARACTERS)), ids=[choice.value for choice in CHARACTERS])
def test_complete_published_catalog_starts_on_both_ports(prerequisites, tmp_path: Path, index: int) -> None:
    chosen = DemoSelection(
        (CHARACTERS[index].value, CHARACTERS[(index - 3) % len(CHARACTERS)].value),
        ("MASTER", "MASTER"),
        STAGES[(index + 4) % len(STAGES)].value,
    )
    matchup = chosen.matchup()
    cancel = threading.Event()
    with Session(
        ISO_PATH,
        dolphin_path=NETPLAY_EMULATOR_PATH,
        slippi_port=51841,
        replay_dir=tmp_path,
        blocking_input=True,
        polling_mode=True,
        canceled=cancel.is_set,
        disable_audio=True,
        gfx_backend="OGL",
        emulation_speed=0,
        dolphin_version=dolphin_version(str(NETPLAY_EMULATOR_PATH)),
        dolphin_batch=True,
    ) as session:
        frame = session.start_match(matchup)
        assert frame["stage"] == matchup.stage.value
        for _ in range(150):
            for player in matchup.players:
                assert frame["ports"][player.port]["leader"]["post"]["character"] == player.character.value
            frame, playing = session.step({1: NEUTRAL_CONTROLLER_ACTION, 2: NEUTRAL_CONTROLLER_ACTION})
            assert playing


@pytest.mark.parametrize("phase", ["startup", "menus", "play"])
def test_local_session_cancellation_releases_dolphin_within_one_second(
    prerequisites, tmp_path: Path, phase: str
) -> None:
    cancel = threading.Event()
    pid: list[int] = []
    canceled_at: list[float] = []
    timer: threading.Timer | None = None

    def cancel_now() -> None:
        canceled_at.append(time.monotonic())
        cancel.set()

    def started(value: int) -> None:
        nonlocal timer
        pid.append(value)
        if phase == "menus":
            timer = threading.Timer(0.5, cancel_now)
            timer.start()

    if phase == "startup":
        cancel_now()
    try:
        with (
            pytest.raises(SessionCanceled),
            Session(
                ISO_PATH,
                dolphin_path=NETPLAY_EMULATOR_PATH,
                slippi_port=51841,
                replay_dir=tmp_path,
                blocking_input=True,
                polling_mode=True,
                canceled=cancel.is_set,
                on_dolphin_started=started,
                dolphin_version=dolphin_version(str(NETPLAY_EMULATOR_PATH)),
                dolphin_batch=True,
                disable_audio=True,
                gfx_backend="OGL",
                step_timeout_seconds=5,
            ) as session,
        ):
            session.start_match(
                Matchup(
                    melee.Stage.BATTLEFIELD,
                    (PlayerSetup(1, melee.Character.FOX), PlayerSetup(2, melee.Character.FALCO)),
                )
            )
            for _ in range(130):
                session.step({1: NEUTRAL_CONTROLLER_ACTION, 2: NEUTRAL_CONTROLLER_ACTION})
            cancel_now()
            session.step({1: NEUTRAL_CONTROLLER_ACTION, 2: NEUTRAL_CONTROLLER_ACTION})
        assert time.monotonic() - canceled_at[-1] < 1
        if pid:
            with pytest.raises(ProcessLookupError):
                os.killpg(pid[0], 0)
    finally:
        if timer is not None:
            timer.cancel()


@pytest.mark.parametrize("cancel_frame", [90, 5400])
def test_two_inference_connections_and_live_overlay_release_on_cancellation(
    prerequisites, tmp_path: Path, cancel_frame: int
) -> None:
    profile = PreparedInferenceProfile("local-test", "a" * 64, "window", 8, 4, (1, 2, 4), 2)
    policy = Mock()
    requests = []

    def predict(batch):
        requests.extend(batch)
        return tuple(action_plan(request, (NEUTRAL_CONTROLLER_ACTION,) * 4, state_value=42.6) for request in batch)

    policy.predict.side_effect = predict
    pairs = (Pipe(), Pipe())
    engine = InferenceEngine({profile: policy}, {4: pairs[0][0], 5: pairs[1][0]}, batch_wait_seconds=0)
    engine_stop = threading.Event()
    engine_thread = threading.Thread(target=engine.serve, args=(engine_stop,), daemon=True)
    engine_thread.start()
    cancel = threading.Event()
    control = QueueControl(QueueEndpoint("http://127.0.0.1:1", "local"), "session", Mock(), tmp_path / "status")
    control._connected = True
    control._state = {"draining": False, "stream": {"slot": 0, "key": "private"}}
    control._capacity = {"queued": 0, "active": 0}
    control._idle_since = time.monotonic() - 2
    assert control.acquire_demo(cancel)
    dolphin_pids: list[int] = []
    clients = tuple(
        InferenceClient(PolicySpec("test", "test", (), (2,)), 128, pair[1], engine_stop, {4: profile})
        for pair in pairs
    )
    state = GameStreamState("FOX", "MASTER", 120, 1, "integration", 4, 1, "self-play", 1)
    rendered = []
    renderer_stop = threading.Event()
    receiver = TelemetryReceiver(tmp_path / "socket")

    def render() -> None:
        renderer = PanelRenderer()
        while not renderer_stop.wait(1 / 60):
            inputs, value = receiver.poll(state, time.monotonic())
            renderer.render(state, inputs, value, "").save(tmp_path / "frame.png")
            rendered.append(value)

    render_thread = threading.Thread(target=render, daemon=True)
    render_thread.start()
    times: list[float] = []
    latencies: list[float] = []

    def frame(frame_id: int, _step: float) -> None:
        if frame_id >= 0:
            times.append(time.monotonic())
        if frame_id >= cancel_frame:
            control._capacity["queued"] = 1
            control.refresh_demo()
            assert control._claim("/claim", {"slot": 0}) == (204, None)
            assert cancel.is_set() and control._demo_held

    try:
        with (
            pytest.raises(SessionCanceled),
            closing(TelemetryPublisher(tmp_path / "socket", tmp_path / "state", state)) as publisher,
            Session(
                ISO_PATH,
                dolphin_path=NETPLAY_EMULATOR_PATH,
                slippi_port=51841,
                replay_dir=tmp_path / "replays",
                blocking_input=True,
                polling_mode=True,
                canceled=cancel.is_set,
                on_dolphin_started=dolphin_pids.append,
                disable_audio=True,
                gfx_backend="OGL",
                dolphin_version=dolphin_version(str(NETPLAY_EMULATOR_PATH)),
                dolphin_batch=True,
            ) as session,
        ):
            self_play.run_match(
                session,
                Matchup(
                    melee.Stage.BATTLEFIELD,
                    (PlayerSetup(1, melee.Character.FOX), PlayerSetup(2, melee.Character.FALCO)),
                ),
                (clients[0], clients[1]),
                (4, 5),
                ("MASTER", "MASTER"),
                publisher,
                frame,
                lambda: None,
                max_frames=cancel_frame + 256,
                observe_policy=latencies.append,
            )
        assert not engine._routes
        assert dolphin_pids
        with pytest.raises(ProcessLookupError):
            os.killpg(dolphin_pids[0], 0)
        assert control._demo_held
        control.release_demo()
        forward = Mock(return_value=(204, None))
        control._forward = forward
        control._claim("/claim", {"slot": 0})
        forward.assert_called_once()
        assert set(request.stream_id for request in requests) == {4, 5}
        assert all(
            item.desired_return == 120 and item.temperature == 1
            for request in requests
            for item in request.observations
        )
        assert any(value == 42.6 for value in rendered)
        assert (len(times) - 1) / (times[-1] - times[0]) >= 50
        ordered = sorted(latencies)
        metrics = {
            "inference": "synthetic neutral policy; no GPU model",
            "emulator_fps": (len(times) - 1) / (times[-1] - times[0]),
            "inference_p50_ms": ordered[len(ordered) // 2] * 1000,
            "inference_p99_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))] * 1000,
            "overlay_frames": len(rendered),
        }
        (tmp_path / "metrics.json").write_text(json.dumps(metrics))
        print(json.dumps(metrics))
    finally:
        control.close()
        renderer_stop.set()
        render_thread.join(timeout=2)
        receiver.close()
        engine_stop.set()
        for client in clients:
            client.close()
        engine_thread.join(timeout=2)
        for pair in pairs:
            for connection in pair:
                connection.close()


def test_cancellable_local_session_reaches_the_next_match_menu(prerequisites, tmp_path: Path) -> None:
    from dataclasses import replace

    cancel = threading.Event()
    with Session(
        ISO_PATH,
        dolphin_path=NETPLAY_EMULATOR_PATH,
        slippi_port=51841,
        replay_dir=tmp_path,
        blocking_input=True,
        polling_mode=True,
        canceled=cancel.is_set,
        disable_audio=True,
        gfx_backend="OGL",
        emulation_speed=0,
        step_timeout_seconds=5,
        dolphin_version=dolphin_version(str(NETPLAY_EMULATOR_PATH)),
        dolphin_batch=True,
    ) as session:
        session.start_match(
            Matchup(
                melee.Stage.BATTLEFIELD,
                (PlayerSetup(1, melee.Character.FOX), PlayerSetup(2, melee.Character.FALCO)),
            )
        )
        walk_offstage = replace(NEUTRAL_CONTROLLER_ACTION, main_x=1)
        for _ in range(6000):
            _, playing = session.step({1: walk_offstage, 2: NEUTRAL_CONTROLLER_ACTION})
            if not playing:
                break
        else:
            pytest.fail("local match did not end after four self-destructs")


def _hung_dolphin_worker(pid, ready) -> None:
    from hal.netplay_service.self_play import DolphinStarted

    os.setsid()
    with Session(
        ISO_PATH,
        dolphin_path=NETPLAY_EMULATOR_PATH,
        slippi_port=51841,
        blocking_input=True,
        polling_mode=True,
        on_dolphin_started=DolphinStarted(pid),
        disable_audio=True,
        dolphin_version=dolphin_version(str(NETPLAY_EMULATOR_PATH)),
        dolphin_batch=True,
        gfx_backend="OGL",
    ) as session:
        session.start_match(
            Matchup(
                melee.Stage.BATTLEFIELD, (PlayerSetup(1, melee.Character.FOX), PlayerSetup(2, melee.Character.FALCO))
            )
        )
        ready.set()
        signal.pause()


def _demo_with_slow_interpreter_shutdown(*args) -> None:
    import atexit

    from hal.netplay_service.self_play import worker

    atexit.register(time.sleep, 3)
    worker(*args)


@pytest.mark.parametrize("phase", ["startup", "menus", "play"])
def test_spawned_demo_releases_resources_before_interpreter_finalizers(
    prerequisites, tmp_path: Path, phase: str
) -> None:
    import multiprocessing as mp

    from hal.netplay_service.self_play import DemoConfig

    context = mp.get_context("spawn")
    pairs = (context.Pipe(), context.Pipe())
    profile = PreparedInferenceProfile("local-test", "a" * 64, "window", 8, 4, (1, 2, 4), 2)
    policy = Mock()
    policy.predict.side_effect = lambda batch: tuple(
        action_plan(request, (NEUTRAL_CONTROLLER_ACTION,) * 4, state_value=0.0) for request in batch
    )
    engine = InferenceEngine({profile: policy}, {4: pairs[0][0], 5: pairs[1][0]}, batch_wait_seconds=0)
    engine_stop = context.Event()
    engine_thread = threading.Thread(target=engine.serve, args=(engine_stop,), daemon=True)
    engine_thread.start()
    cancel = context.Event()
    dolphin_pid = context.RawValue("i", 0)
    state_path = tmp_path / "status.stream.json"
    config = DemoConfig(
        Path(ISO_PATH),
        Path(NETPLAY_EMULATOR_PATH),
        os.environ["DISPLAY"],
        state_path,
        51841,
        0,
        1,
        "shutdown-test",
        (4, 5),
        54_000,
        (),
        "a" * 40,
        "a" * 64,
        "b" * 64,
        (1,),
    )
    if phase == "startup":
        cancel.set()
    process = context.Process(
        target=_demo_with_slow_interpreter_shutdown,
        args=(
            config,
            PolicySpec("test", "test", (), (2,)),
            128,
            (profile,),
            (pairs[0][1], pairs[1][1]),
            cancel,
            engine_stop,
            dolphin_pid,
        ),
    )
    try:
        process.start()
        deadline = time.monotonic() + 30
        record_path = tmp_path / "self-play-shutdown-test.json"
        while time.monotonic() < deadline:
            if phase == "startup" and record_path.exists():
                break
            if phase == "menus" and dolphin_pid.value:
                break
            health_path = state_path.with_suffix(".self-play.json")
            if phase == "play" and health_path.exists():
                health = json.loads(health_path.read_text())
                if health["game_fps"] is not None:
                    break
            if not process.is_alive():
                pytest.fail(f"demo exited before {phase}: {process.exitcode}")
            time.sleep(0.01)
        else:
            pytest.fail(f"demo did not reach {phase}")
        started = time.monotonic()
        cancel.set()
        process.join(timeout=0.6)
        assert not process.is_alive(), "resource release waited for interpreter finalizers"
        assert time.monotonic() - started < 0.6
        assert process.exitcode == 0
        assert not engine._routes
        record = json.loads(record_path.read_text())
        assert record["canceled_at"] <= record["worker_finished_at"]
        if dolphin_pid.value:
            with pytest.raises(ProcessLookupError):
                os.killpg(dolphin_pid.value, 0)
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
        if dolphin_pid.value:
            with suppress(ProcessLookupError):
                os.killpg(dolphin_pid.value, signal.SIGKILL)
        engine_stop.set()
        engine_thread.join(timeout=2)
        for pair in pairs:
            for connection in pair:
                connection.close()


def test_hung_demo_watchdog_reaps_separate_dolphin_group(prerequisites, tmp_path: Path, monkeypatch) -> None:
    import multiprocessing as mp

    from hal.netplay_service import runner
    from hal.netplay_service.control import QueueControl
    from hal.netplay_service.queue_client import QueueEndpoint

    context = mp.get_context("spawn")
    ready = context.Event()
    config = runner.RunnerConfig(
        QueueEndpoint("http://127.0.0.1:1", "local"),
        "session",
        tmp_path / "policy",
        (tmp_path / "account0", tmp_path / "account1"),
        (51841, 51842),
        Path(ISO_PATH),
        Path(NETPLAY_EMULATOR_PATH),
        tmp_path / "replays",
        tmp_path / "status",
        "a" * 40,
        wants_stream=True,
        display_base=100,
        stream_display=os.environ["DISPLAY"],
        idle_self_play=True,
    )
    with closing(QueueControl(config.queue_endpoint, "session", Mock(), tmp_path / "status")) as control:
        control._connected = True
        control._state = {"draining": False, "stream": {"slot": 0, "key": "private"}}
        control._capacity = {"queued": 0, "active": 0, "healthy_slots": 2}
        control._idle_since = time.monotonic() - 2
        process_context = Mock(wraps=context)
        demo = runner._IdleDemo(
            process_context, config, Mock(), control, {slot: Mock() for slot in range(4, 8)}, 2, (), "a" * 64
        )
        process_context.Process.side_effect = lambda **_kwargs: context.Process(
            target=_hung_dolphin_worker, args=(demo.dolphin_pid, ready)
        )
        invalidate = Mock(return_value=True)
        monkeypatch.setattr(runner, "_invalidate_slot_connection", invalidate)
        try:
            demo.tick(threading.Event(), False)
            process = demo.process
            assert ready.wait(30)
            dolphin_pid = demo.dolphin_pid.value
            control._capacity["queued"] = 1
            started = time.monotonic()
            while demo.process is not None and time.monotonic() - started < 1:
                demo.tick(threading.Event(), False)
                time.sleep(0.005)
            assert demo.process is None
            assert demo.disabled and not control._demo_held
            assert invalidate.call_count == 2
            assert not runner._IdleDemo._group_alive(dolphin_pid)
            assert not process.is_alive()
        finally:
            if demo.process is not None:
                demo.close(threading.Event())
