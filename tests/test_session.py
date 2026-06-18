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

    # _navigate_to_live returns the canonical dict augmented with the live stage.
    assert s._navigate_to_live() == {"menu": melee.Menu.IN_GAME, "stage": int(melee.Stage.FINAL_DESTINATION.value)}
