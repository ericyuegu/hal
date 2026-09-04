"""Fast unit tests for ``Session`` menu navigation, without a real Dolphin.

The menu-nav loop streams gamestates fast under FFW, so the per-poll
``step_timeout_seconds`` never catches a menu that simply never reaches
IN_GAME. ``_navigate_to_live`` carries its own wall-clock cap so a logical
menu hang surfaces as a clean ``TimeoutError`` instead of spinning forever
(``start_match`` callers already log-and-continue on that).
"""

import time

import melee
import pytest

import hal.sim.session as session_module
from hal.sim.inputs import ControllerInputsValue
from hal.sim.session import Matchup
from hal.sim.session import Session


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
    inputs = ControllerInputsValue(
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
            {"env": {"A": "B"}},
        )
    ]
    assert "preexec_fn" not in calls[0][2]


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
