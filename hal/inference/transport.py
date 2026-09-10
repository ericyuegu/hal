"""Explicit controller transport queue shared by local and online play."""

from collections import deque
from collections.abc import Iterable

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import ControllerAction


class ActionTransport:
    """Track actions committed for future game states."""

    def __init__(self, delay_frames: int) -> None:
        if not isinstance(delay_frames, int) or isinstance(delay_frames, bool) or delay_frames < 0:
            raise ValueError("delay_frames must be a non-negative integer")
        self.delay_frames = delay_frames
        self._pending: deque[ControllerAction] = deque()
        self.reset()

    @property
    def pending(self) -> tuple[ControllerAction, ...]:
        return tuple(self._pending)

    def reset(self, actions: Iterable[ControllerAction] | None = None) -> None:
        values = tuple(actions) if actions is not None else (NEUTRAL_CONTROLLER_ACTION,) * self.delay_frames
        if len(values) != self.delay_frames:
            raise ValueError(f"transport reset needs {self.delay_frames} actions, got {len(values)}")
        self._pending = deque(values)

    def submit(self, action: ControllerAction) -> ControllerAction:
        """Commit one output and return the action due in the next state."""
        if not self.delay_frames:
            return action
        due = self._pending.popleft()
        self._pending.append(action)
        return due
