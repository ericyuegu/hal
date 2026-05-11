"""Per-port input producers.

A ``ControllerSource`` is a callable: given the current frame index and the
last observed gamestate, return the inputs to punch this frame, or ``None``
if the port is driven internally (CPU bot or physical hardware).

``last_gamestate`` is for closed-loop policies (a ``ModelControllerSource``
needs to see the current observation). Replay-style sources ignore it.

The drive loop in ``drive.py`` does not care which subclass it gets — only
the protocol matters.
"""

from collections.abc import Sequence
from typing import Literal
from typing import Protocol

import attrs
import numpy as np

from hal.emulator.controller_io import ControllerInputs
from hal.emulator.controller_io import ControllerInputsValue
from hal.emulator.controller_io import MdsControllerView


class ControllerSource(Protocol):
    """One frame of inputs for one port, or ``None`` if internally driven."""

    def __call__(self, frame_index: int, last_gamestate: dict | None) -> ControllerInputs | None: ...


@attrs.define(slots=True)
class MdsControllerSource:
    """Replay an MDS-recorded port of inputs.

    Reuses one ``MdsControllerView`` and advances ``frame_idx`` in place — no
    per-frame allocation on the hot path.
    """

    columns: dict[str, np.ndarray]
    port_prefix: Literal["p1", "p2"]
    _view: MdsControllerView = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        self._view = MdsControllerView(columns=self.columns, port_prefix=self.port_prefix, frame_idx=0)

    def __call__(self, frame_index: int, last_gamestate: dict | None) -> ControllerInputs | None:
        n = len(self.columns[f"{self.port_prefix}_main_stick_x"])
        if frame_index >= n:
            return None
        self._view.frame_idx = frame_index
        return self._view


class InternalControllerSource:
    """Sentinel: this port is driven inside Melee (CPU bot or physical human).

    ``drive`` skips ``apply_inputs`` for any port that returns ``None``.
    """

    def __call__(self, frame_index: int, last_gamestate: dict | None) -> ControllerInputs | None:
        return None


@attrs.define(slots=True)
class ScriptedControllerSource:
    """Fixed-sequence playback. After the sequence is exhausted, returns
    neutral resting state."""

    sequence: Sequence[ControllerInputs]
    _neutral: ControllerInputs = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        self._neutral = ControllerInputsValue(
            main_x=0.0, main_y=0.0, c_x=0.0, c_y=0.0, trigger_l=0.0, trigger_r=0.0, buttons=0
        )

    def __call__(self, frame_index: int, last_gamestate: dict | None) -> ControllerInputs | None:
        if frame_index < len(self.sequence):
            return self.sequence[frame_index]
        return self._neutral
