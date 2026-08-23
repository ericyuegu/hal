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
