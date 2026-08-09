"""Per-port input producers.

A ``ControllerSource`` is a callable: given the current frame index and the
last observed gamestate, return the inputs to punch this frame, or ``None``
if the port is driven internally (CPU bot or physical hardware).

``last_gamestate`` is for closed-loop policies (a ``ModelControllerSource``
needs to see the current observation). Replay-style sources ignore it.

The drive loop in ``loop.py`` does not care which subclass it gets — only
the protocol matters.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Literal
from typing import Protocol
from typing import runtime_checkable

import numpy as np

from hal.sim.inputs import ControllerInputs
from hal.sim.inputs import ControllerInputsValue
from hal.wire import BUTTON_BITS


@runtime_checkable
class ControllerSource(Protocol):
    """One frame of inputs for one port, or ``None`` if internally driven."""

    def __call__(self, frame_index: int, last_gamestate: dict | None) -> ControllerInputs | None: ...


@dataclass(slots=True)
class MDSControllerSource:
    """Replay an MDS-recorded port of inputs.

    Resolve all input arrays once. Pack the button mask once. Each call changes
    only the current frame index and returns this object as the input view.

    Frame alignment: drive's ``captured[0]`` is the gamestate returned by
    ``start_match`` — its slp pre-frame inputs are already locked in by the
    menu-to-game transition and cannot be replayed. Iteration ``t`` punches
    inputs that produce ``captured[t+1]``, which corresponds to slp
    ``pre[t+1]``. So we index forward by one: replay iteration ``t`` reads
    ``columns[t+1]``, matching the same input that record iteration ``t``
    sent. Without this shift, replay lags record by one frame — invisible
    on a neutral sequence but a real bit-exact failure on varying inputs.
    """

    columns: dict[str, np.ndarray]
    port_prefix: Literal["p1", "p2"]
    _main_x: np.ndarray = field(init=False, repr=False)
    _main_y: np.ndarray = field(init=False, repr=False)
    _c_x: np.ndarray = field(init=False, repr=False)
    _c_y: np.ndarray = field(init=False, repr=False)
    _trigger_l: np.ndarray = field(init=False, repr=False)
    _trigger_r: np.ndarray = field(init=False, repr=False)
    _buttons: np.ndarray = field(init=False, repr=False)
    _frame_idx: int = field(default=-1, init=False, repr=False)

    def __post_init__(self) -> None:
        prefix = self.port_prefix
        arrays = {
            "main_x": np.asarray(self.columns[f"{prefix}_main_stick_x"]),
            "main_y": np.asarray(self.columns[f"{prefix}_main_stick_y"]),
            "c_x": np.asarray(self.columns[f"{prefix}_c_stick_x"]),
            "c_y": np.asarray(self.columns[f"{prefix}_c_stick_y"]),
            "trigger_l": np.asarray(self.columns[f"{prefix}_trigger_l"]),
            "trigger_r": np.asarray(self.columns[f"{prefix}_trigger_r"]),
        }
        lengths = {len(values) for values in arrays.values()}
        if len(lengths) != 1:
            shapes = {name: values.shape for name, values in arrays.items()}
            raise ValueError(f"MDS controller columns have different lengths: {shapes}")
        frames = lengths.pop()
        buttons = np.zeros(frames, dtype=np.uint16)
        for name, bit in BUTTON_BITS.items():
            values = np.asarray(self.columns[f"{prefix}_button_{name}"])
            if values.shape != (frames,):
                raise ValueError(f"{prefix}_button_{name} has shape {values.shape}; expected {(frames,)}")
            if not np.isin(values, (0, 1)).all():
                raise ValueError(f"{prefix}_button_{name} must contain only 0 or 1")
            buttons |= values.astype(np.uint16) * np.uint16(bit)
        self._main_x = arrays["main_x"]
        self._main_y = arrays["main_y"]
        self._c_x = arrays["c_x"]
        self._c_y = arrays["c_y"]
        self._trigger_l = arrays["trigger_l"]
        self._trigger_r = arrays["trigger_r"]
        self._buttons = buttons

    def __call__(self, frame_index: int, last_gamestate: dict | None) -> ControllerInputs | None:
        next_idx = frame_index + 1
        if next_idx >= len(self._main_x):
            return None
        self._frame_idx = next_idx
        return self

    def _at(self, values: np.ndarray) -> float:
        if self._frame_idx < 0:
            raise RuntimeError("MDS controller input was read before its first frame")
        return float(values[self._frame_idx])

    @property
    def main_x(self) -> float:
        return self._at(self._main_x)

    @property
    def main_y(self) -> float:
        return self._at(self._main_y)

    @property
    def c_x(self) -> float:
        return self._at(self._c_x)

    @property
    def c_y(self) -> float:
        return self._at(self._c_y)

    @property
    def trigger_l(self) -> float:
        return self._at(self._trigger_l)

    @property
    def trigger_r(self) -> float:
        return self._at(self._trigger_r)

    @property
    def buttons(self) -> int:
        if self._frame_idx < 0:
            raise RuntimeError("MDS controller input was read before its first frame")
        return int(self._buttons[self._frame_idx])


@dataclass(frozen=True, slots=True)
class InternalControllerSource:
    """Sentinel: this port is driven inside Melee (CPU bot or physical human).

    ``drive`` skips ``apply_inputs`` for any port that returns ``None``.
    """

    def __call__(self, frame_index: int, last_gamestate: dict | None) -> ControllerInputs | None:
        return None


_NEUTRAL_INPUTS: ControllerInputs = ControllerInputsValue(
    main_x=0.0, main_y=0.0, c_x=0.0, c_y=0.0, trigger_l=0.0, trigger_r=0.0, buttons=0
)


@dataclass(slots=True)
class ScriptedControllerSource:
    """Fixed-sequence playback. After the sequence is exhausted, returns
    neutral resting state."""

    sequence: Sequence[ControllerInputs]
    _neutral: ControllerInputs = field(default=_NEUTRAL_INPUTS, init=False)

    def __call__(self, frame_index: int, last_gamestate: dict | None) -> ControllerInputs | None:
        if frame_index < len(self.sequence):
            return self.sequence[frame_index]
        return self._neutral


def demo_sequence(n_frames: int, *, port: Literal["p1", "p2"]) -> list[ControllerInputs]:
    """Non-trivial controller sequence for round-trip tests.

    Exercises every input axis the wire path can carry: main-stick excursions,
    c-stick smashes in all four quadrants, staggered button press/release
    boundaries (A, B, X, Z, L), and an analog trigger ramp. ``port`` flips
    several phases so the two ports drive asymmetric inputs — catches
    port-mapping or per-port carry-over bugs that symmetric scripts hide.

    Inputs are deterministic; for bit-exact replay tests, record with one
    instance and replay from the resulting MDS row, not from this sequence.
    """
    p2 = port == "p2"
    sign = -1.0 if p2 else 1.0
    out: list[ControllerInputs] = []
    for t in range(n_frames):
        main_x = main_y = c_x = c_y = trigger_l = trigger_r = 0.0
        buttons = 0
        if 30 <= t < 60:
            # Hold main stick down-and-toward-center; sign flips per port.
            main_x = 0.75 * sign
            main_y = -0.5
        elif 60 <= t < 90:
            # C-stick smashes through all four quadrants, 7 frames per quadrant.
            quadrant = (t - 60) // 7
            c_x = (1.0, 0.0, -1.0, 0.0)[quadrant % 4]
            c_y = (0.0, 1.0, 0.0, -1.0)[quadrant % 4]
        elif 90 <= t < 120:
            # Staggered button stagger. Each press is 3 frames, 4-frame gap.
            phase = (t - 90) % 7
            if phase < 3:
                buttons = (BUTTON_BITS["a"], BUTTON_BITS["b"], BUTTON_BITS["x"], BUTTON_BITS["z"])[((t - 90) // 7) % 4]
        elif 120 <= t < 150:
            # Trigger ramp 0 -> 1 -> 0 over 30 frames. trigger_l on p1, trigger_r on p2
            # so asymmetric per port.
            ramp = 1.0 - abs(((t - 120) - 15) / 15.0)
            if p2:
                trigger_r = ramp
            else:
                trigger_l = ramp
        elif 150 <= t < 180:
            # Z + L combo bursts of 5 frames every 10.
            if (t - 150) % 10 < 5:
                buttons = BUTTON_BITS["z"] | BUTTON_BITS["l"]
        out.append(
            ControllerInputsValue(
                main_x=main_x,
                main_y=main_y,
                c_x=c_x,
                c_y=c_y,
                trigger_l=trigger_l,
                trigger_r=trigger_r,
                buttons=buttons,
            )
        )
    return out
