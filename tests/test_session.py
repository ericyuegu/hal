"""Fast unit tests for ``Session`` menu navigation, without a real Dolphin.

The menu-nav loop streams gamestates fast under FFW, so the per-poll
``step_timeout_seconds`` never catches a menu that simply never reaches
IN_GAME. ``_navigate_to_live`` carries its own wall-clock cap so a logical
menu hang surfaces as a clean ``TimeoutError`` instead of spinning forever
(``start_match`` callers already log-and-continue on that).
"""

import multiprocessing
import signal
import time
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import call

import melee
import pytest
from melee.slippstream import SlippstreamClient

import hal.sim.session as session_module
from hal.controller import ControllerAction
from hal.sim.session import DolphinGraphicsBackend
from hal.sim.session import Matchup
from hal.sim.session import Session
from hal.sim.session import set_dolphin_stream_output


class _FakeGameState:
    def __init__(self, menu_state: melee.Menu, stage: melee.Stage = melee.Stage.FINAL_DESTINATION) -> None:
        self.menu_state = menu_state
        self.stage = stage  # Session._canonical reads gamestate.stage for the live stage

    def to_canonical_dict(self) -> dict:
        return {"menu": self.menu_state}


def _session(start_timeout: float) -> Session:
    s = Session(iso_path="unused.iso", dolphin_path="unused", start_timeout_seconds=start_timeout)
    s._console = object()  # non-None so the context-manager guard passes
    return s


def test_polling_waits_for_one_frame_before_flushing_again(monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs: dict[str, object] = {}

    class Console:
        def __init__(self, **values: object) -> None:
            kwargs.update(values)

    monkeypatch.setattr(session_module.melee, "Console", Console)
    monkeypatch.setattr(session_module.atexit, "register", lambda _callback: None)
    session = Session(
        iso_path="unused.iso",
        dolphin_path="unused",
        polling_mode=True,
        step_timeout_seconds=7.5,
    )

    session._boot()

    assert kwargs["polling_timeout"] == 7.5


@pytest.mark.parametrize("gfx_backend", ["Vulkan", "OGL"])
def test_session_configures_graphics_backend_native_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gfx_backend: DolphinGraphicsBackend
) -> None:
    kwargs: dict[str, object] = {}
    version = melee.console.DolphinVersion(False, "3.6.4", melee.console.DolphinBuild.NETPLAY)
    original_get_version = melee.console.get_dolphin_version

    class Console:
        def __init__(self, **values: object) -> None:
            assert melee.console.get_dolphin_version("unused") is version
            kwargs.update(values)

        def _get_dolphin_config_path(self) -> str:
            return str(tmp_path)

    monkeypatch.setattr(session_module.melee, "Console", Console)
    monkeypatch.setattr(session_module.atexit, "register", lambda _callback: None)
    session = Session(
        iso_path="unused.iso",
        dolphin_path="unused",
        gfx_backend=gfx_backend,
        internal_resolution_scale=2,
        dolphin_version=version,
    )

    session._boot()

    assert kwargs["gfx_backend"] == gfx_backend
    assert "EFBScale = 2" in (tmp_path / "GFX.ini").read_text()
    assert melee.console.get_dolphin_version is original_get_version


def test_session_rejects_unsupported_graphics_backend() -> None:
    with pytest.raises(ValueError, match="unsupported Dolphin graphics backend"):
        Session(iso_path="unused.iso", dolphin_path="unused", gfx_backend="Null")  # type: ignore[arg-type]


def test_stream_output_preserves_unrelated_dolphin_settings(tmp_path: Path) -> None:
    (tmp_path / "Dolphin.ini").write_text("[DSP]\nVolume = 75\n[Display]\nKeep = yes\n")
    (tmp_path / "GFX.ini").write_text("[Settings]\nMSAA = 4\n")

    class Console:
        def _get_dolphin_config_path(self) -> str:
            return str(tmp_path)

    set_dolphin_stream_output(Console(), True)  # type: ignore[arg-type]

    dolphin = session_module._CaseSensitiveConfigParser(interpolation=None)
    dolphin.read(tmp_path / "Dolphin.ini")
    gfx = session_module._CaseSensitiveConfigParser(interpolation=None)
    gfx.read(tmp_path / "GFX.ini")
    assert dolphin.get("DSP", "Volume") == "75"
    assert dolphin.get("Display", "Keep") == "yes"
    assert dolphin.get("DSP", "Backend") == "Pulse"
    assert gfx.get("Settings", "MSAA") == "4"
    assert gfx.get("Settings", "EFBScale") == "4"


def test_navigate_to_live_times_out_when_menu_never_goes_live() -> None:
    s = _session(0.05)
    s._step_blocking = lambda: _FakeGameState(melee.Menu.MAIN_MENU)  # type: ignore[method-assign]
    s._drive_menus = lambda gamestate: None  # type: ignore[method-assign]

    t0 = time.monotonic()
    with pytest.raises(TimeoutError, match="did not reach IN_GAME"):
        s._navigate_to_live()
    # The cap must actually fire — a regression that drops it would spin here.
    assert time.monotonic() - t0 < 5.0


def test_navigate_to_live_returns_on_live_menu() -> None:
    s = _session(5.0)
    seq = iter(
        [
            _FakeGameState(melee.Menu.MAIN_MENU),
            _FakeGameState(melee.Menu.MAIN_MENU),
            _FakeGameState(melee.Menu.IN_GAME),
        ]
    )
    s._step_blocking = lambda: next(seq)  # type: ignore[method-assign]
    s._drive_menus = lambda gamestate: None  # type: ignore[method-assign]
    s._validate_live_characters = lambda gamestate: None  # type: ignore[method-assign]

    # _navigate_to_live returns the canonical dict augmented with the live stage.
    assert s._navigate_to_live() == {"menu": melee.Menu.IN_GAME, "stage": int(melee.Stage.FINAL_DESTINATION.value)}


def test_step_reports_latency_only_after_controller_pipe_flush(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class Controller:
        def flush(self) -> None:
            events.append("flush")

    session = _session(5.0)
    controller = Controller()
    session._instrument_controller_flush(1, controller)  # type: ignore[arg-type]
    session._controllers = {1: controller}  # type: ignore[dict-item]
    monkeypatch.setattr(session_module, "apply_inputs", lambda _controller, _inputs: events.append("apply"))

    def advance() -> _FakeGameState:
        events.append("advance")
        controller.flush()
        events.append("receive")
        return _FakeGameState(melee.Menu.IN_GAME)

    session._step_blocking = advance  # type: ignore[method-assign]
    inputs = ControllerAction(
        main_x=0.0,
        main_y=0.0,
        c_x=0.0,
        c_y=0.0,
        trigger_l=0.0,
        trigger_r=0.0,
        buttons=0,
    )
    session.step({1: inputs}, on_inputs_flushed=lambda: events.append("ack"))

    assert events == ["apply", "advance", "flush", "ack", "receive"]


def test_parent_bound_spawn_uses_exec_wrapper_without_preexec(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], tuple[object, ...], dict[str, object]]] = []
    sentinel = object()

    def popen(command: list[str], *args: object, **kwargs: object) -> object:
        calls.append((command, args, kwargs))
        return sentinel

    monkeypatch.setattr(session_module, "_PARENT_BOUND_POPEN_ORIGINAL", popen)

    result = session_module._spawn_parent_bound(["dolphin", "-e", "game.iso"], env={"A": "B"})

    assert result is sentinel
    assert calls == [
        (
            [session_module.sys.executable, "-m", "hal.sim.pdeathsig_exec", "dolphin", "-e", "game.iso"],
            (),
            {"env": {"A": "B"}, "start_new_session": True},
        )
    ]
    assert "preexec_fn" not in calls[0][2]


def test_parent_bound_spawn_adds_dolphin_batch_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    def popen(command: list[str], *_args: object, **_kwargs: object) -> object:
        commands.append(command)
        return object()

    monkeypatch.setattr(session_module, "_PARENT_BOUND_POPEN_ORIGINAL", popen)
    monkeypatch.setattr(session_module, "_PARENT_BOUND_DOLPHIN_BATCH", True)

    session_module._spawn_parent_bound(["dolphin", "-e", "game.iso"])

    assert commands == [
        [session_module.sys.executable, "-m", "hal.sim.pdeathsig_exec", "dolphin", "-b", "-e", "game.iso"]
    ]


def test_teardown_signals_the_whole_dolphin_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    process = Mock(pid=123)
    process.poll.return_value = None
    signals = Mock()
    monkeypatch.setattr(session_module.os, "killpg", signals)
    console = Mock(_process=process, controllers=[])

    session_module.teardown_console(console, None)

    assert signals.call_args_list == [call(123, signal.SIGTERM), call(123, signal.SIGKILL)]
    process.terminate.assert_not_called()
    process.kill.assert_not_called()


def test_replay_repair_failure_does_not_mask_body_error_or_retain_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    class Console:
        _process = None

        def stop(self) -> None:
            pass

    repair_calls = 0

    def fail_repair(_replay_dir) -> None:
        nonlocal repair_calls
        repair_calls += 1
        raise OSError("repair failed")

    session = Session(iso_path="unused.iso", dolphin_path="unused", replay_dir=tmp_path)

    def boot() -> None:
        session._console = Console()  # type: ignore[assignment]
        session._controllers = {1: object()}  # type: ignore[dict-item]
        session._menu_helpers = {1: object()}  # type: ignore[dict-item]
        session._pending_flush_ports = {1}
        session._inputs_flushed_callback = lambda: None
        session._stage_select_steps = 17
        session._matchup = Matchup(stage=melee.Stage.FINAL_DESTINATION, players=())

    monkeypatch.setattr(session, "_boot", boot)
    monkeypatch.setattr(session_module, "finalize_replay_dir", fail_repair)

    with pytest.raises(RuntimeError, match="body failed"), session:
        raise RuntimeError("body failed")

    assert repair_calls == 1
    assert session._console is None
    assert session._controllers == {}
    assert session._menu_helpers == {}
    assert session._pending_flush_ports == set()
    assert session._inputs_flushed_callback is None
    assert session._stage_select_steps == 0
    assert session._matchup is None

    session._teardown()
    assert repair_calls == 1


def test_teardown_kills_dolphin_when_console_stop_raises() -> None:
    class Process:
        terminated = False
        killed = False

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float) -> None:
            assert timeout > 0

    class Console:
        def __init__(self, process: Process) -> None:
            self._process = process

        def stop(self) -> None:
            raise AssertionError("worker never started")

    process = Process()
    session = Session(iso_path="unused.iso", dolphin_path="unused")
    session._console = Console(process)  # type: ignore[assignment]

    session._teardown()

    assert process.terminated
    assert process.killed
    assert session._console is None


def test_teardown_disconnects_controllers_and_breaks_console_references() -> None:
    controller = Mock()
    console = Mock(_process=None, controllers=[controller])

    session_module.teardown_console(console, None)

    controller.disconnect.assert_called_once_with()
    assert console.controllers == []
    console.stop.assert_called_once_with()


def _fill_slippstream_pipe(sender: Connection, ready: Event, ignore_term: bool) -> None:
    if ignore_term:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.set()
    sender.send_bytes(b"x" * 1_048_576)


@pytest.mark.parametrize("ignore_term", [False, True])
def test_teardown_reaps_slippstream_receiver_with_full_pipe(ignore_term: bool) -> None:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    ready = context.Event()
    worker = context.Process(target=_fill_slippstream_pipe, args=(sender, ready, ignore_term))
    client = SlippstreamClient.__new__(SlippstreamClient)
    client._buffer = receiver
    client._shutdown = context.Event()
    client._worker = worker
    client.running = True
    console = Mock(_process=None, controllers=[], _slippstream=client)
    console.stop.side_effect = client.shutdown
    try:
        worker.start()
        assert ready.wait(timeout=10)
        worker.join(timeout=0.1)
        assert worker.is_alive()

        started = time.monotonic()
        session_module.teardown_console(console, None)

        assert time.monotonic() - started < 5
        assert client._worker is None
        assert receiver.closed
        assert not client.running
        console.stop.assert_called_once_with()
        session_module.teardown_console(console, None)
    finally:
        if not worker._closed:
            if worker.is_alive():
                worker.kill()
            worker.join(timeout=5)
            worker.close()
        sender.close()
        receiver.close()


def test_teardown_closes_slippstream_before_connect() -> None:
    client = SlippstreamClient()
    console = Mock(_process=None, controllers=[], _slippstream=client)
    console.stop.side_effect = client.shutdown

    session_module.teardown_console(console, None)

    assert client._worker is None
    assert client._buffer.closed
    assert not client.running


def test_slippstream_shutdown_rejects_untested_libmelee(monkeypatch: pytest.MonkeyPatch) -> None:
    client = SlippstreamClient()
    with monkeypatch.context() as patch:
        patch.setattr(melee.version, "__version__", "0.48.0")
        with pytest.raises(RuntimeError, match="untested libmelee"):
            session_module._stop_slippstream(client)
    session_module._stop_slippstream(client)
