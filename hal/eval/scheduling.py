"""Schedule predicted controller actions against observed game frames."""

from collections import deque
from dataclasses import dataclass

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import ControllerAction
from hal.inference.api import ActionPlan
from hal.inference.api import PolicyInput
from hal.inference.api import PredictionRequest
from hal.inference.api import validate_action_plan
from hal.sim.inputs import controller_actions_match


@dataclass(frozen=True, slots=True)
class FrameTiming:
    """Frame offsets measured from the latest observation in a plan request."""

    input_delay_frames: int
    thinking_allowance_frames: int
    replan_interval_frames: int
    prediction_horizon_frames: int

    def __post_init__(self) -> None:
        values = (
            self.input_delay_frames,
            self.thinking_allowance_frames,
            self.replan_interval_frames,
            self.prediction_horizon_frames,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
            raise ValueError("frame timing values must be non-negative integers")
        if self.replan_interval_frames < 1:
            raise ValueError("replan interval must be positive")
        if self.prediction_horizon_frames < self.fixed_prefix_frames + self.replan_interval_frames:
            raise ValueError("prediction horizon must cover the fixed prefix and replan interval")

    @property
    def fixed_prefix_frames(self) -> int:
        return self.input_delay_frames + self.thinking_allowance_frames

    @property
    def reserve_frames(self) -> int:
        return self.prediction_horizon_frames - self.fixed_prefix_frames - self.replan_interval_frames


class ActionScheduler:
    """Own one controlled port's future actions and inference deadlines."""

    def __init__(self, timing: FrameTiming, context_frames: int, generation: int) -> None:
        if context_frames < timing.replan_interval_frames or generation < 0:
            raise ValueError("invalid history capacity or match generation")
        self.timing = timing
        self.generation = generation
        self.history: deque[PolicyInput] = deque(maxlen=context_frames)
        self.planned: dict[int, ControllerAction] = {}
        self.submitted: dict[int, ControllerAction] = {}
        self._fallback_targets: set[int] = set()
        self._has_plan = False
        self._exhausted = False
        self.request: PredictionRequest | None = None
        self.ready: ActionPlan | None = None
        self.sequence = 0
        self.last_accepted_source_frame: int | None = None
        self.next_request_frame: int | None = None
        self.last_submission_frame: int | None = None
        self.deadline_misses = 0
        self.prefix_mismatches = 0
        self.transport_corrections = 0
        self.exhausted_chunks = 0
        self.neutral_fallback_frames = 0
        self.inference_failed = False

    def observe(self, item: PolicyInput) -> bool:
        """Ignore rollback duplicates and reject missing frames within a generation."""
        if self.history and item.frame_id <= self.history[-1].frame_id:
            return False
        if self.history and item.frame_id != self.history[-1].frame_id + 1:
            raise ValueError("observed game frames must be consecutive within one inference generation")
        expected = self.submitted.get(item.frame_id, NEUTRAL_CONTROLLER_ACTION)
        # Slippi masks pads before -45; these frames still belong in model history.
        if item.frame_id >= -45 and not controller_actions_match(expected, item.applied_action):
            self.transport_corrections += 1
        self.history.append(item)
        cutoff = item.frame_id - max(self.history.maxlen or 1, self.timing.prediction_horizon_frames)
        self._fallback_targets = {frame for frame in self._fallback_targets if frame >= cutoff}
        for schedule in (self.planned, self.submitted):
            for frame in tuple(schedule):
                if frame < cutoff:
                    del schedule[frame]
        return True

    def request_plan(self) -> PredictionRequest | None:
        if self.inference_failed or self.request is not None or not self.history:
            return None
        item = self.history[-1]
        if self.next_request_frame is not None and item.frame_id < self.next_request_frame:
            return None
        if self.last_accepted_source_frame is None:
            observations = tuple(self.history)
        else:
            observations = tuple(
                observed for observed in self.history if observed.frame_id > self.last_accepted_source_frame
            )
            if not observations or observations[0].frame_id != self.last_accepted_source_frame + 1:
                raise RuntimeError("unreported observations exceed the history capacity")
        fixed_actions = []
        for frame in range(item.frame_id + 1, item.frame_id + self.timing.fixed_prefix_frames + 1):
            action = self.submitted.get(frame, self.planned.get(frame, NEUTRAL_CONTROLLER_ACTION))
            if frame not in self.submitted and frame not in self.planned:
                self._fallback_targets.add(frame)
            fixed_actions.append(action)
            # These future submissions cannot change while inference is pending.
            self.planned[frame] = action
        self.request = PredictionRequest(
            item.stream_id,
            self.generation,
            self.sequence,
            item.frame_id,
            observations,
            tuple(fixed_actions),
        )
        self.sequence += 1
        return self.request

    def accept_plan(self, plan: ActionPlan) -> bool:
        request = self.request
        if request is None or (plan.generation, plan.sequence) != (self.generation, request.sequence):
            return False
        validate_action_plan(request, plan, self.timing.prediction_horizon_frames)
        self.last_accepted_source_frame = request.source_frame
        self.ready = plan
        return True

    def apply_ready_plan(self, frame: int) -> None:
        request, plan = self.request, self.ready
        if request is None or plan is None or frame < request.source_frame + self.timing.thinking_allowance_frames:
            return
        tail_start = request.source_frame + self.timing.fixed_prefix_frames + 1
        first = max(tail_start, frame + self.timing.input_delay_frames + 1)
        skipped = max(0, first - tail_start)
        self.deadline_misses += min(skipped, len(plan.actions))
        actual = {item.frame_id: item.applied_action for item in self.history}
        for offset, action in enumerate(request.fixed_actions, 1):
            target = request.source_frame + offset
            executed = actual.get(target, self.submitted.get(target))
            if executed is not None and not controller_actions_match(executed, action):
                self.prefix_mismatches += 1
        for item in plan.actions:
            if item.target_frame < first:
                executed = actual.get(item.target_frame, self.submitted.get(item.target_frame))
                if executed is not None and not controller_actions_match(executed, item.action):
                    self.prefix_mismatches += 1
            else:
                self.planned[item.target_frame] = item.action
                self._fallback_targets.discard(item.target_frame)
        if first > request.source_frame + self.timing.prediction_horizon_frames:
            self.exhausted_chunks += 1
        else:
            self._has_plan = True
            self._exhausted = False
        self.next_request_frame = max(request.source_frame + self.timing.replan_interval_frames, frame)
        self.request = None
        self.ready = None
        while len(self.history) > 1 and self.history[0].frame_id <= request.source_frame:
            self.history.popleft()

    def action_to_submit(self, frame: int) -> ControllerAction:
        if self.last_submission_frame is not None and frame <= self.last_submission_frame:
            raise ValueError("controller submission must advance the game frame")
        target = frame + self.timing.input_delay_frames + 1
        action = self.planned.get(target)
        if action is None:
            action = NEUTRAL_CONTROLLER_ACTION
            self._fallback_targets.add(target)
        if target in self._fallback_targets:
            self.neutral_fallback_frames += 1
            if self._has_plan and not self._exhausted:
                self.exhausted_chunks += 1
                self._exhausted = True
        if self.last_submission_frame is not None:
            self.deadline_misses += frame - self.last_submission_frame - 1
        self.submitted[target] = action
        self.last_submission_frame = frame
        return action

    def fail_inference(self) -> None:
        self.inference_failed = True
        self.request = None
        self.ready = None

    def drained(self, frame: int) -> bool:
        return self.inference_failed and frame >= max(self.planned, default=frame)
