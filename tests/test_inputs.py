"""Unit tests for the controller-input wire path (no Dolphin)."""

import melee
import numpy as np

from hal.sim.inputs import ControllerInputsValue
from hal.sim.inputs import MdsControllerView
from hal.sim.inputs import _stick_axis_wire
from hal.wire import MASK_INT8
from hal.wire import raw_byte_to_wire


def _minimal_columns(prefix: str, *, with_raw: bool) -> dict[str, np.ndarray]:
    """Single-frame column dict satisfying MdsControllerView property reads."""
    cols: dict[str, np.ndarray] = {
        f"{prefix}_main_stick_x": np.array([0.5], dtype=np.float32),
        f"{prefix}_main_stick_y": np.array([-0.25], dtype=np.float32),
        f"{prefix}_c_stick_x": np.array([0.0], dtype=np.float32),
        f"{prefix}_c_stick_y": np.array([0.0], dtype=np.float32),
        f"{prefix}_trigger_l_physical": np.array([0.0], dtype=np.float32),
        f"{prefix}_trigger_r_physical": np.array([0.0], dtype=np.float32),
    }
    for b in ("a", "b", "x", "y", "z", "r", "l", "start", "d_up"):
        cols[f"{prefix}_button_{b}"] = np.array([0], dtype=np.int32)
    if with_raw:
        cols[f"{prefix}_main_stick_raw_x"] = np.array([42], dtype=np.int8)
        cols[f"{prefix}_main_stick_raw_y"] = np.array([-30], dtype=np.int8)
        cols[f"{prefix}_c_stick_raw_x"] = np.array([80], dtype=np.int8)
        cols[f"{prefix}_c_stick_raw_y"] = np.array([-80], dtype=np.int8)
    return cols


def test_view_returns_raw_bytes_when_present() -> None:
    cols = _minimal_columns("p1", with_raw=True)
    view = MdsControllerView(columns=cols, port_prefix="p1", frame_idx=0)
    assert view.raw_main_x == 42
    assert view.raw_main_y == -30
    assert view.raw_c_x == 80
    assert view.raw_c_y == -80


def test_view_falls_back_to_mask_when_raw_columns_absent() -> None:
    cols = _minimal_columns("p1", with_raw=False)
    view = MdsControllerView(columns=cols, port_prefix="p1", frame_idx=0)
    assert view.raw_main_x == MASK_INT8
    assert view.raw_main_y == MASK_INT8
    assert view.raw_c_x == MASK_INT8
    assert view.raw_c_y == MASK_INT8


def test_stick_axis_wire_uses_raw_when_present() -> None:
    w = _stick_axis_wire(raw_byte=42, logical=0.5)
    assert w == raw_byte_to_wire(42)


def test_stick_axis_wire_falls_back_to_logical_at_mask() -> None:
    w = _stick_axis_wire(raw_byte=MASK_INT8, logical=0.5)
    # Shift peppi [-1, 1] → libmelee [0, 1].
    assert w == melee.controller.fix_analog_stick((0.5 + 1.0) / 2.0)


def test_controller_inputs_value_defaults_raw_to_mask() -> None:
    v = ControllerInputsValue(main_x=0.0, main_y=0.0, c_x=0.0, c_y=0.0, trigger_l=0.0, trigger_r=0.0, buttons=0)
    assert v.raw_main_x == MASK_INT8
    assert v.raw_main_y == MASK_INT8
    assert v.raw_c_x == MASK_INT8
    assert v.raw_c_y == MASK_INT8
