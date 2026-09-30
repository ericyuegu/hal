"""Declared netplay prefixes map trained action heads to submit-able frames."""

import pytest

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.eval.scheduling import ActionScheduler
from hal.eval.scheduling import FrameTiming
from hal.inference.api import PolicyInput
from hal.inference.api import action_plan
from hal.models.action_sequence import ActionSequenceConfig


@pytest.mark.parametrize(
    ("timing", "first_target", "tail_length"),
    [
        (FrameTiming(2, 1, 3, 4, 8), 4, 5),
        (FrameTiming(3, 1, 4, 4, 8), 5, 4),
    ],
)
def test_netplay_timing_uses_distinct_prepared_prefix_shapes(
    timing: FrameTiming, first_target: int, tail_length: int
) -> None:
    assert ActionSequenceConfig().head_offsets[:8] == tuple(range(1, 9))
    scheduler = ActionScheduler(timing, 256, generation=1)
    scheduler.observe(PolicyInput(9001, 0, 1, {}, NEUTRAL_CONTROLLER_ACTION))
    request = scheduler.request_plan()
    assert request is not None
    assert len(request.fixed_actions) == timing.fixed_prefix_frames
    plan = action_plan(request, (NEUTRAL_CONTROLLER_ACTION,) * tail_length, state_value=0.0)
    assert tuple(action.target_frame for action in plan.actions) == tuple(range(first_target, 9))
    assert scheduler.accept_plan(plan, choice_frame=1)
