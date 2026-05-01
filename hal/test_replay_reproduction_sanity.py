from __future__ import annotations

import os
from pathlib import Path

import melee
import pytest
from melee.controller import fix_analog_stick

from hal.local_paths import EMULATOR_PATH
from hal.local_paths import ISO_PATH
from notebooks.replay_reproduction_sanity import DIGITAL_BUTTONS
from notebooks.replay_reproduction_sanity import ReplayControllerSender
from notebooks.replay_reproduction_sanity import TimeoutConfig
from notebooks.replay_reproduction_sanity import pipe_value_for_axis_raw
from notebooks.replay_reproduction_sanity import pipe_value_for_trigger_raw
from notebooks.replay_reproduction_sanity import run_reproduction
from notebooks.replay_reproduction_sanity import trigger_raw_from_processed


class FakeController:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def press_button(self, button: melee.Button) -> None:
        self.calls.append(("press_button", button))

    def release_button(self, button: melee.Button) -> None:
        self.calls.append(("release_button", button))

    def tilt_analog(self, button: melee.Button, x: float, y: float) -> None:
        self.calls.append(("tilt_analog", button, x, y))

    def press_shoulder(self, button: melee.Button, amount: float) -> None:
        self.calls.append(("press_shoulder", button, amount))

    def flush(self) -> None:
        self.calls.append(("flush",))


def controller_state(
    *,
    buttons: set[melee.Button] | None = None,
    main: tuple[float, float] = (0.5, 0.5),
    raw_main: tuple[int, int] = (0, 0),
    c: tuple[float, float] = (0.5, 0.5),
    left: float = 0.0,
    r: float = 0.0,
) -> melee.ControllerState:
    state = melee.ControllerState()
    for button in DIGITAL_BUTTONS:
        state.button[button] = button in (buttons or set())
    state.main_stick = main
    state.raw_main_stick = raw_main
    state.c_stick = c
    state.l_shoulder = left
    state.r_shoulder = r
    return state


def assert_pipe_call(controller: FakeController, op: str, button: melee.Button, *args: float) -> None:
    matches = [c for c in controller.calls if c[0] == op and c[1] is button]
    assert matches, f"missing {op} for {button}"
    last = matches[-1]
    for got, want in zip(last[2:], args, strict=True):
        assert abs(got - want) < 1e-6, f"{op}({button}) got {last[2:]} want {args}"


def test_sender_emits_button_transitions_analogs_and_one_flush_per_frame() -> None:
    controller = FakeController()
    sender = ReplayControllerSender({1: controller})  # type: ignore[arg-type]

    sender.send_frame(
        {
            1: controller_state(
                buttons={melee.Button.BUTTON_A, melee.Button.BUTTON_L},
                main=(0.00625, 0.6875),
                c=(0.1, 0.9),
                left=0.4,
                r=0.8,
            )
        }
    )

    assert ("press_button", melee.Button.BUTTON_A) in controller.calls
    assert ("press_button", melee.Button.BUTTON_L) in controller.calls
    assert_pipe_call(
        controller,
        "tilt_analog",
        melee.Button.BUTTON_MAIN,
        fix_analog_stick(0.00625),
        fix_analog_stick(0.6875),
    )
    assert_pipe_call(
        controller,
        "tilt_analog",
        melee.Button.BUTTON_C,
        fix_analog_stick(0.1),
        fix_analog_stick(0.9),
    )
    assert_pipe_call(
        controller,
        "press_shoulder",
        melee.Button.BUTTON_L,
        pipe_value_for_trigger_raw(trigger_raw_from_processed(0.4)),
    )
    assert_pipe_call(
        controller,
        "press_shoulder",
        melee.Button.BUTTON_R,
        pipe_value_for_trigger_raw(trigger_raw_from_processed(0.8)),
    )
    assert controller.calls.count(("flush",)) == 1


def test_sender_holds_buttons_without_duplicate_press_and_releases_changes() -> None:
    controller = FakeController()
    sender = ReplayControllerSender({1: controller})  # type: ignore[arg-type]

    sender.send_frame({1: controller_state(buttons={melee.Button.BUTTON_A, melee.Button.BUTTON_B})})
    sender.send_frame({1: controller_state(buttons={melee.Button.BUTTON_A, melee.Button.BUTTON_Z})})

    assert controller.calls.count(("press_button", melee.Button.BUTTON_A)) == 1
    assert controller.calls.count(("press_button", melee.Button.BUTTON_B)) == 1
    assert controller.calls.count(("release_button", melee.Button.BUTTON_B)) == 1
    assert controller.calls.count(("press_button", melee.Button.BUTTON_Z)) == 1
    assert controller.calls.count(("flush",)) == 2


def test_pipe_value_round_trip_matches_dolphin_quantization() -> None:
    import math

    for raw in range(-127, 128):
        v = pipe_value_for_axis_raw(raw)
        assert 0.0 <= v <= 1.0
        assert math.floor((v - 0.5) * 254) == raw

    for raw in range(0, 256):
        v = pipe_value_for_trigger_raw(raw)
        assert 0.0 <= v <= 1.0
        assert int(v * 255) == raw


def test_trigger_inverse_recovers_processed_value() -> None:
    for processed in (0.0, 0.1, 0.25, 0.4, 0.5, 0.75, 1.0):
        raw = trigger_raw_from_processed(processed)
        recovered = raw / 0x8C
        assert abs(recovered - processed) < 1.0 / 0x8C


@pytest.mark.integration
def test_replay_reproduction_sanity_dev_prefix() -> None:
    if os.getenv("HAL_RUN_REPLAY_REPRODUCTION_INTEGRATION") != "1":
        pytest.skip("set HAL_RUN_REPLAY_REPRODUCTION_INTEGRATION=1 to run Dolphin replay reproduction")

    replay_dir = Path.home() / "data" / "ssbm" / "dev"
    replay_paths = sorted(replay_dir.glob("*.slp")) if replay_dir.exists() else []
    missing = [path for path in (Path(EMULATOR_PATH), Path(ISO_PATH)) if not path.exists()]
    if missing or not replay_paths:
        pytest.skip("requires HAL_EMULATOR_PATH, HAL_ISO_PATH, and at least one ~/data/ssbm/dev/*.slp")

    results = run_reproduction(
        replay_path=replay_paths[0],
        emulator_path=Path(EMULATOR_PATH),
        iso_path=Path(ISO_PATH),
        modes=("normal", "ffw"),
        prefix_frames=300,
        start_frame=0,
        stop_on_mismatch=True,
        debug_dir=Path("/tmp/replay_reproduction_sanity_pytest"),
        timeouts=TimeoutConfig(menu_s=120.0, first_ingame_s=60.0, frame_s=10.0),
    )

    assert all(result.compared_frames == 300 for result in results)
    assert all(not result.mismatches for result in results)
