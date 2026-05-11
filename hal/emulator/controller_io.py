"""Per-frame controller-input value objects + libmelee setter dispatch.

The ``ControllerInputs`` Protocol is the structural contract consumed by
``apply_inputs``. Two impls satisfy it:

- ``MdsControllerView``: zero-copy view aliasing a column-dict and a frame
  index. Used by ``MdsControllerSource`` on the per-frame hot path.
- ``ControllerInputsValue``: ``attrs.frozen(slots=True)`` value object for
  sources that produce inputs from scratch (model output, scripted, .slp
  random-access).

Both expose primitive-typed properties only; ``apply_inputs`` is duck-typed.

Wire-protocol notes:

- libmelee's pipe protocol carries floats: "SET MAIN x y", "SET L amount". For
  bit-exact playback we want the int8 raw byte the original game saw, not the
  post-processed logical stick value. peppi exposes both; we prefer ``raw_*``
  when the slp version recorded it (raw_x ≥ 1.2.0; raw_y ≥ 3.15.0; raw c-stick
  ≥ 3.17.0). Construct the Controller with ``fix_analog_inputs=False`` so the
  setter writes our wire-format byte directly without re-quantizing.
- When raw bytes are not available (older slps), we fall back to feeding
  ``peppi.joystick.x/y`` (the logical [-1, 1] value) via the same wire format.
  This is lossy near non-neutral stick values: peppi's logical value is
  many-to-one over raw int8, so the recovered raw can be off by ±1, which
  cascades into ~0.0025/frame physics drift. The diff reports this case
  explicitly; it's a slp-version limitation, not a code bug.
"""

from typing import Literal
from typing import Protocol
from typing import runtime_checkable

import attrs
import melee
import numpy as np

# slp pre.buttons_physical bitmask. Mirrors hal/data/schema.BUTTON_BITS so
# MDS columns and live punches agree on bit layout. Order matters only for
# diagnostics; press_button / release_button are commutative within a frame.
BUTTON_BIT_TO_MELEE: tuple[tuple[int, melee.enums.Button], ...] = (
    (0x0100, melee.enums.Button.BUTTON_A),
    (0x0200, melee.enums.Button.BUTTON_B),
    (0x0400, melee.enums.Button.BUTTON_X),
    (0x0800, melee.enums.Button.BUTTON_Y),
    (0x0010, melee.enums.Button.BUTTON_Z),
    (0x0020, melee.enums.Button.BUTTON_R),
    (0x0040, melee.enums.Button.BUTTON_L),
    (0x1000, melee.enums.Button.BUTTON_START),
    (0x0008, melee.enums.Button.BUTTON_D_UP),
)


# MDS sentinel for "this frame's raw byte is unavailable" (slp version too
# old to record it). Mirrors hal.data.extract._mask_value for int8 columns —
# np.iinfo(int8).min = -128. ``apply_inputs`` checks the sentinel and falls
# back to the normalized-float path for that axis.
RAW_BYTE_MASK: int = -128


@runtime_checkable
class ControllerInputs(Protocol):
    """Structural protocol for one frame of controller state for one port."""

    main_x: float
    main_y: float
    c_x: float
    c_y: float
    trigger_l: float
    trigger_r: float
    buttons: int  # uint16 bitmask matching BUTTON_BIT_TO_MELEE
    raw_main_x: int  # int8 from slp (>= 1.2.0), or RAW_BYTE_MASK if unrecorded
    raw_main_y: int  # int8 from slp (>= 3.15.0), or RAW_BYTE_MASK if unrecorded


@attrs.frozen(slots=True)
class ControllerInputsValue:
    """Concrete value object satisfying ControllerInputs.

    Used by sources that produce inputs from scratch (model output, scripted
    sequences, .slp random-access). For MDS playback prefer ``MdsControllerView``
    — it aliases the underlying NumPy arrays without copying.
    """

    main_x: float
    main_y: float
    c_x: float
    c_y: float
    trigger_l: float
    trigger_r: float
    buttons: int
    raw_main_x: int = RAW_BYTE_MASK
    raw_main_y: int = RAW_BYTE_MASK


# Mutable so callers can advance ``frame_idx`` in place without reallocating
# (the hot path for batched parallel emulators). Frozen would force one
# instance per frame per port.
@attrs.define(slots=True)
class MdsControllerView:
    """Zero-copy view over MDS columns at a given frame index.

    Field access reads ``columns[f"{port_prefix}_{name}"][frame_idx]`` — no per-
    field copy beyond the NumPy 0-d scalar Python wraps it in. ``buttons`` is
    re-derived from the 9 single-bit columns each access; this is cheap (9
    indexes + bit-or ≈ ns) and keeps the schema unchanged. If profiling later
    flags this as hot, add a packed ``button_mask`` column at extract time.
    """

    columns: dict[str, np.ndarray]
    port_prefix: Literal["p1", "p2"]
    frame_idx: int

    @property
    def main_x(self) -> float:
        return float(self.columns[f"{self.port_prefix}_main_stick_x"][self.frame_idx])

    @property
    def main_y(self) -> float:
        return float(self.columns[f"{self.port_prefix}_main_stick_y"][self.frame_idx])

    @property
    def c_x(self) -> float:
        return float(self.columns[f"{self.port_prefix}_c_stick_x"][self.frame_idx])

    @property
    def c_y(self) -> float:
        return float(self.columns[f"{self.port_prefix}_c_stick_y"][self.frame_idx])

    @property
    def trigger_l(self) -> float:
        return float(self.columns[f"{self.port_prefix}_trigger_l_physical"][self.frame_idx])

    @property
    def trigger_r(self) -> float:
        return float(self.columns[f"{self.port_prefix}_trigger_r_physical"][self.frame_idx])

    @property
    def buttons(self) -> int:
        mask = 0
        for bit, suffix in _BUTTON_BIT_AND_COL_SUFFIX:
            if self.columns[f"{self.port_prefix}_button_{suffix}"][self.frame_idx]:
                mask |= bit
        return mask

    @property
    def raw_main_x(self) -> int:
        return int(self.columns[f"{self.port_prefix}_main_stick_raw_x"][self.frame_idx])

    @property
    def raw_main_y(self) -> int:
        return int(self.columns[f"{self.port_prefix}_main_stick_raw_y"][self.frame_idx])


# Pre-resolved (bit, MDS-column-suffix) pairs to keep the hot-path button
# decode in MdsControllerView.buttons free of attribute access on the enum.
_BUTTON_BIT_AND_COL_SUFFIX: tuple[tuple[int, str], ...] = (
    (0x0100, "a"),
    (0x0200, "b"),
    (0x0400, "x"),
    (0x0800, "y"),
    (0x0010, "z"),
    (0x0020, "r"),
    (0x0040, "l"),
    (0x1000, "start"),
    (0x0008, "d_up"),
)


def apply_inputs(controller: melee.Controller, src: ControllerInputs) -> None:
    """Punch one frame of inputs into a libmelee Controller.

    Setters write directly to the named pipe; ``Console.step()`` flushes — do
    not call ``flush()`` here. The button loop unconditionally presses or
    releases every button this frame so we don't carry stale state from a
    previous source.

    Stick path: prefer raw int8 bytes per-axis (bit-exact) when the slp
    recorded them; otherwise feed peppi's logical [-1, 1] value via
    ``tilt_analog`` with our own wire-format math. The Controller must be
    constructed with ``fix_analog_inputs=False`` so libmelee doesn't re-process
    the wire float we just composed.
    """
    main_x_wire = _stick_axis_wire(src.raw_main_x, src.main_x)
    main_y_wire = _stick_axis_wire(src.raw_main_y, src.main_y)
    controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, main_x_wire, main_y_wire)

    # MDS doesn't store raw c-stick bytes today (slp >= 3.17 would have them);
    # fall back to the logical-to-wire path. C-stick is mostly used as a
    # binary smash-direction indicator, so the ±1 raw round-trip risk is small.
    controller.tilt_analog(
        melee.enums.Button.BUTTON_C,
        _logical_to_wire(src.c_x),
        _logical_to_wire(src.c_y),
    )
    controller.press_shoulder(melee.enums.Button.BUTTON_L, src.trigger_l)
    controller.press_shoulder(melee.enums.Button.BUTTON_R, src.trigger_r)

    buttons = src.buttons
    for bit, button in BUTTON_BIT_TO_MELEE:
        if buttons & bit:
            controller.press_button(button)
        else:
            controller.release_button(button)


def _stick_axis_wire(raw_byte: int, logical: float) -> float:
    """Pick the per-axis wire float: raw byte if recorded, else logical fallback."""
    if raw_byte != RAW_BYTE_MASK:
        return _raw_byte_to_wire(raw_byte)
    return _logical_to_wire(logical)


def _raw_byte_to_wire(raw: int) -> float:
    """Convert int8 raw byte to wire float that round-trips through Dolphin's
    pipe-input parser. Mirrors libmelee.controller.fix_analog_stick's wire
    fudge (+0.1) so Dolphin's rounding lands at exactly ``raw``."""
    return (raw + 0.1) / 254.0 + 0.5


def _logical_to_wire(logical: float) -> float:
    """Convert peppi-logical [-1, 1] stick value to wire format. Equivalent
    to the tilt_analog_unit + fix_analog_stick path; recomputed here so
    ``Controller(fix_analog_inputs=False)`` doesn't double-process it."""
    raw = round(logical * 80)
    return _raw_byte_to_wire(raw)
