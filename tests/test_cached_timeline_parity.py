"""The cached local scheduler keeps the frozen control's applied-action timing."""

import hashlib
import json
from pathlib import Path

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import ControllerAction
from hal.controller import controller_action_wire_values
from hal.eval.scheduling import ActionScheduler
from hal.eval.scheduling import FrameTiming
from hal.inference.api import PolicyInput
from hal.inference.api import action_plan
from hal.sim.inputs import ActionTransport
from hal.wire import BUTTON_BITS

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "o59"
_CONTROL_SOURCE_SHA256 = "7a4c5d72b8966beceaa4c95556171fa335fb49c3530ab10b6ae82a31c4212147"
_CONTROL_RECORD_SHA256 = "f95257e9410801a93b80cf55879d9d978ebc1ca7fcc79d25db03484cac5b397c"
_CONTROL_GIT_SHA = "d7454f9d1a136f7c745d2d4478af22e19cd65faf"


def _planned_action(target: int) -> ControllerAction:
    return ControllerAction(
        (target % 5 - 2) / 2,
        (target % 7 - 3) / 3,
        (target % 3 - 1) / 2,
        0.0,
        (target % 4) / 3,
        0.0,
        BUTTON_BITS["a"] if target % 2 else BUTTON_BITS["b"],
    )


def _action(record: dict[str, float | int]) -> ControllerAction:
    return ControllerAction(
        float(record["main_x"]),
        float(record["main_y"]),
        float(record["c_x"]),
        float(record["c_y"]),
        float(record["trigger_l"]),
        float(record["trigger_r"]),
        int(record["buttons"]),
    )


def test_cached_prefix_two_applies_the_frozen_control_actions_at_the_same_frames() -> None:
    source = (Path(__file__).parents[1] / "archive/scripts/capture_cached_timeline_control.py").read_bytes()
    raw = (_FIXTURE_DIR / "cached_timeline_control.json").read_bytes()
    assert hashlib.sha256(source).hexdigest() == _CONTROL_SOURCE_SHA256
    assert hashlib.sha256(raw).hexdigest() == _CONTROL_RECORD_SHA256
    control = json.loads(raw)
    assert control["source_git_sha"] == _CONTROL_GIT_SHA
    assert control["capture_source_sha256"] == _CONTROL_SOURCE_SHA256

    scheduler = ActionScheduler(FrameTiming(0, 0, 2, 2, 4), 256, 1)
    transport = ActionTransport(0)
    applied = NEUTRAL_CONTROLLER_ACTION
    request_index = 0
    for frame in range(20):
        scheduler.observe(PolicyInput(7, frame, 1, {}, applied, reset=frame == 0))
        request = scheduler.request_plan() if frame <= 14 else None
        if request is not None:
            reference = control["requests"][request_index]
            assert request.source_frame == reference["source_frame"]
            assert [controller_action_wire_values(action) for action in request.fixed_actions] == [
                controller_action_wire_values(_action(record)) for record in reference["fixed_actions"]
            ]
            tail = (_planned_action(frame + 3), _planned_action(frame + 4))
            assert [controller_action_wire_values(action) for action in tail] == [
                controller_action_wire_values(_action(record)) for record in reference["generated"]
            ]
            assert scheduler.accept_plan(action_plan(request, tail), frame)
            request_index += 1
        applied = transport.submit(scheduler.action_to_submit(frame))
        expected = control["frames"][frame]
        assert expected["target_frame"] == frame + 1
        assert controller_action_wire_values(applied) == controller_action_wire_values(_action(expected["action"]))
    assert request_index == len(control["requests"]) == 8
