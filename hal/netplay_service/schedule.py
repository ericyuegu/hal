"""A game-frame schedule owned by the Dolphin worker, independent of inference."""

from collections import deque

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import ControllerAction
from hal.inference.api import PolicyInput
from hal.inference.chunks import ChunkRequest
from hal.inference.chunks import ChunkResponse
from hal.inference.chunks import TimingSchedule
from hal.inference.chunks import validate_chunk_response
from hal.sim.inputs import controller_actions_match


class FrameSchedule:
    def __init__(self, timing: TimingSchedule, context_frames: int, generation: int) -> None:
        if context_frames < 1 or generation < 0:
            raise ValueError("invalid history capacity or match generation")
        self.timing = timing
        self.generation = generation
        self.history: deque[PolicyInput] = deque(maxlen=context_frames)
        self.planned: dict[int, ControllerAction] = {}
        self.submitted: dict[int, ControllerAction] = {}
        self._fallback_targets: set[int] = set()
        self._has_chunk = False
        self._exhausted = False
        self.request: ChunkRequest | None = None
        self.ready: ChunkResponse | None = None
        self.sequence = 0
        self.next_request_frame: int | None = None
        self.last_submission_frame: int | None = None
        self.deadline_misses = 0
        self.prefix_mismatches = 0
        self.transport_corrections = 0
        self.exhausted_chunks = 0
        self.neutral_fallback_frames = 0
        self.engine_failed = False

    def observe(self, item: PolicyInput) -> bool:
        """Keep actual observations; duplicate rollback emissions do not advance time."""
        if self.history and item.frame_id <= self.history[-1].frame_id:
            return False
        if self.history and item.frame_id != self.history[-1].frame_id + 1:
            self.history.clear()
        expected = self.submitted.get(item.frame_id, NEUTRAL_CONTROLLER_ACTION)
        # Slippi masks pads before -45; these frames still belong in model history.
        if item.frame_id >= -45 and not controller_actions_match(expected, item.applied_action):
            self.transport_corrections += 1
        self.history.append(item)
        cutoff = item.frame_id - max(self.history.maxlen or 1, self.timing.horizon)
        self._fallback_targets = {frame for frame in self._fallback_targets if frame >= cutoff}
        for schedule in (self.planned, self.submitted):
            for frame in tuple(schedule):
                if frame < cutoff:
                    del schedule[frame]
        return True

    def begin_request(self) -> ChunkRequest | None:
        if self.engine_failed or self.request is not None or not self.history:
            return None
        item = self.history[-1]
        if self.next_request_frame is not None and item.frame_id < self.next_request_frame:
            return None
        prefix = []
        for frame in range(item.frame_id + 1, item.frame_id + self.timing.prefix_frames + 1):
            action = self.submitted.get(frame, self.planned.get(frame, NEUTRAL_CONTROLLER_ACTION))
            if frame not in self.submitted and frame not in self.planned:
                self._fallback_targets.add(frame)
            prefix.append(action)
            # These future submissions cannot change while this request is outstanding.
            self.planned[frame] = action
        self.request = ChunkRequest(
            item.stream_id, self.generation, self.sequence, item.frame_id, tuple(self.history), tuple(prefix)
        )
        self.sequence += 1
        return self.request

    def receive(self, response: ChunkResponse) -> bool:
        request = self.request
        if request is None or (response.generation, response.sequence) != (self.generation, request.sequence):
            return False
        validate_chunk_response(request, response, self.timing.horizon)
        self.ready = response
        return True

    def handoff(self, frame: int) -> None:
        request, response = self.request, self.ready
        if request is None or response is None or frame < request.source_frame + self.timing.budget_frames:
            return
        first = max(request.source_frame + self.timing.prefix_frames + 1, frame + self.timing.transport_frames + 1)
        skipped = max(0, first - (request.source_frame + self.timing.prefix_frames + 1))
        self.deadline_misses += min(skipped, self.timing.horizon - self.timing.prefix_frames)
        actual = {item.frame_id: item.applied_action for item in self.history}
        for item in response.actions:
            if item.target_frame < first:
                executed = actual.get(item.target_frame, self.submitted.get(item.target_frame))
                if executed is not None and not controller_actions_match(executed, item.action):
                    self.prefix_mismatches += 1
            else:
                self.planned[item.target_frame] = item.action
                self._fallback_targets.discard(item.target_frame)
        if first > request.source_frame + self.timing.horizon:
            self.exhausted_chunks += 1
        else:
            self._has_chunk = True
            self._exhausted = False
        self.next_request_frame = max(request.source_frame + self.timing.replan_frames, frame)
        self.request = None
        self.ready = None

    def submit(self, frame: int) -> ControllerAction:
        if self.last_submission_frame is not None and frame <= self.last_submission_frame:
            raise ValueError("controller submission must advance the game frame")
        target = frame + self.timing.transport_frames + 1
        action = self.planned.get(target)
        if action is None:
            action = NEUTRAL_CONTROLLER_ACTION
            self._fallback_targets.add(target)
        if target in self._fallback_targets:
            self.neutral_fallback_frames += 1
            if self._has_chunk and not self._exhausted:
                self.exhausted_chunks += 1
                self._exhausted = True
        if self.last_submission_frame is not None:
            self.deadline_misses += frame - self.last_submission_frame - 1
        self.submitted[target] = action
        self.last_submission_frame = frame
        return action

    def fail_engine(self) -> None:
        self.engine_failed = True
        self.request = None
        self.ready = None

    def drained(self, frame: int) -> bool:
        return self.engine_failed and frame >= max(self.planned, default=frame)
