"""Controller-frame scheduling contracts shared by local play and netplay."""

from dataclasses import replace

import pytest

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import ControllerAction
from hal.eval.scheduling import ActionScheduler
from hal.eval.scheduling import FrameTiming
from hal.eval.scheduling import PlanProtocolError
from hal.inference.api import FrameAction
from hal.inference.api import PolicyInput
from hal.inference.api import action_plan


def _action(x: float) -> ControllerAction:
    return ControllerAction(x, 0.0, 0.0, 0.0, 0.0, 0.0, 0)


def _observation(frame: int, applied: ControllerAction = NEUTRAL_CONTROLLER_ACTION) -> PolicyInput:
    return PolicyInput(8127, frame, 1, {}, applied)


def test_declared_profiles_keep_physical_delay_separate_from_prefix() -> None:
    official = FrameTiming(0, 0, 2, 2, 4)
    local = FrameTiming(0, 0, 0, 2, 4)
    netplay_2 = FrameTiming(2, 1, 3, 4, 8)
    netplay_3 = FrameTiming(3, 1, 4, 4, 8)
    assert tuple(timing.reserve_frames for timing in (official, local, netplay_2, netplay_3)) == (0, 2, 1, 0)
    with pytest.raises(ValueError, match="fixed prefix"):
        FrameTiming(2, 1, 2, 4, 8)


def test_async_request_carries_bounded_local_coalescing_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hal.eval.scheduling.time.monotonic", lambda: 10.0)
    online = ActionScheduler(FrameTiming(2, 1, 3, 4, 8), 16, 1)
    online.observe(_observation(0))
    request = online.request_plan()
    assert request is not None and request.deadline_monotonic == 10.0 + 1 / 60

    official = ActionScheduler(FrameTiming(0, 0, 2, 2, 4), 16, 1)
    official.observe(_observation(0))
    request = official.request_plan()
    assert request is not None and request.deadline_monotonic is None


def test_local_zero_delay_zero_prefix_targets_next_frame() -> None:
    schedule = ActionScheduler(FrameTiming(0, 0, 0, 2, 4), 8, 1)
    schedule.observe(_observation(10))
    request = schedule.request_plan()
    assert request is not None and request.fixed_actions == ()
    plan = action_plan(request, (_action(0.5),) * 4)
    assert tuple(item.target_frame for item in plan.actions) == (11, 12, 13, 14)
    assert schedule.accept_plan(plan, 10)
    assert schedule.action_to_submit(10) == _action(0.5)


@pytest.mark.parametrize("malformed", ["identity", "target"])
def test_matching_active_response_rejects_boolean_frame_identity(malformed: str) -> None:
    schedule = ActionScheduler(FrameTiming(0, 0, 0, 2, 4), 8, 1)
    schedule.observe(_observation(0))
    request = schedule.request_plan()
    assert request is not None
    plan = action_plan(request, (NEUTRAL_CONTROLLER_ACTION,) * 4)
    if malformed == "identity":
        plan = replace(plan, source_frame=False)
    else:
        plan = replace(plan, actions=(FrameAction(True, plan.actions[0].action), *plan.actions[1:]))
    with pytest.raises(PlanProtocolError, match="malformed"):
        schedule.accept_plan(plan, 0)
    assert schedule.inference_failed


def test_deadline_equality_accepts_and_one_frame_late_rejects_whole_tail() -> None:
    schedule = ActionScheduler(FrameTiming(2, 1, 3, 4, 8), 16, 7)
    schedule.observe(_observation(0))
    request = schedule.request_plan()
    assert request is not None
    schedule.action_to_submit(0)
    schedule.observe(_observation(1))
    first = action_plan(request, tuple(_action(index / 10) for index in range(4, 9)))
    assert schedule.accept_plan(first, 1)
    assert schedule.last_consumed_source_frame == 0
    for frame in range(1, 5):
        if frame > 1:
            schedule.observe(_observation(frame))
        schedule.action_to_submit(frame)
    second_request = schedule.request_plan()
    assert second_request is not None
    assert tuple(item.frame_id for item in second_request.observations) == (1, 2, 3, 4)
    assert tuple(
        item.target_frame for item in action_plan(second_request, (NEUTRAL_CONTROLLER_ACTION,) * 5).actions
    ) == (
        8,
        9,
        10,
        11,
        12,
    )
    schedule.observe(_observation(5))
    schedule.action_to_submit(5)
    schedule.observe(_observation(6))
    prior = dict(schedule.planned)
    assert not schedule.accept_plan(action_plan(second_request, (_action(0.9),) * 5), 6)
    assert schedule.deadline_misses == 1
    assert schedule.planned == prior
    assert schedule.last_consumed_source_frame == 4
    assert schedule.last_accepted_source_frame == 0
    retry = schedule.request_plan()
    assert retry is not None
    assert tuple(item.frame_id for item in retry.observations) == (5, 6)


def test_prefix_mismatch_rejects_tail_at_controller_wire_precision() -> None:
    schedule = ActionScheduler(FrameTiming(0, 0, 2, 2, 4), 8, 1)
    schedule.observe(_observation(0))
    schedule.planned[1] = _action(0.5)
    request = schedule.request_plan()
    assert request is not None
    schedule.submitted[1] = _action(0.519)
    prior = dict(schedule.planned)
    assert not schedule.accept_plan(action_plan(request, (_action(0.7),) * 2), 1)
    assert schedule.prefix_mismatches == 1
    assert schedule.planned == prior
    assert schedule.last_consumed_source_frame == 0
    assert schedule.last_accepted_source_frame is None


def test_malformed_active_response_fails_protocol_and_releases_request() -> None:
    schedule = ActionScheduler(FrameTiming(0, 0, 0, 2, 4), 8, 3)
    schedule.observe(_observation(0))
    request = schedule.request_plan()
    assert request is not None
    invalid = replace(action_plan(request, (NEUTRAL_CONTROLLER_ACTION,) * 4), generation=2)
    with pytest.raises(PlanProtocolError, match="malformed"):
        schedule.accept_plan(invalid, 0)
    assert schedule.inference_failed
    assert schedule.request is None
    assert schedule.last_consumed_source_frame is None


def test_controller_submission_gaps_have_their_own_counter() -> None:
    schedule = ActionScheduler(FrameTiming(0, 0, 0, 2, 4), 8, 1)
    schedule.action_to_submit(1)
    schedule.action_to_submit(3)
    assert schedule.submission_gaps == 1
    assert schedule.deadline_misses == 0
