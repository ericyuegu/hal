"""Check the arithmetic used to turn profiler shapes into hardware work."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("o65_floor", Path(__file__).parents[1] / "scripts/profile_o65_floor.py")
assert _SPEC is not None and _SPEC.loader is not None
floor = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = floor
_SPEC.loader.exec_module(floor)
Gemm = floor.Gemm
gemm_shape = floor.gemm_shape


def test_gemm_shape_uses_recorded_mm_and_bmm_dimensions() -> None:
    mm = gemm_shape([[2048, 256], [256, 768]])
    assert mm == Gemm(1, 2048, 768, 256)
    assert mm.flops == 2 * 2048 * 768 * 256
    assert mm.bytes == 2 * (2048 * 256 + 256 * 768 + 2048 * 768)
    assert gemm_shape([[8, 32, 64], [8, 64, 256]]) == Gemm(8, 32, 256, 64)


@pytest.mark.parametrize("shapes", [[], [[1], [1]], [[2, 3], [4, 5]], [[1, 2, 3], [2, 3, 4]], [[0, 3], [3, 4]]])
def test_gemm_shape_rejects_incomplete_or_incompatible_records(shapes: list[list[int]]) -> None:
    with pytest.raises(ValueError):
        gemm_shape(shapes)


def test_profiler_parser_keeps_dtype_strides_and_addmm_bias_position(tmp_path: Path) -> None:
    path = tmp_path / "trace.json"
    event = {
        "cat": "cpu_op",
        "name": "aten::addmm",
        "args": {
            "Input Dims": [[4], [2, 3], [3, 4]],
            "Input type": ["float", "float", "float"],
            "Input Strides": [[1], [3, 1], [1, 3]],
        },
    }
    path.write_text(json.dumps({"traceEvents": [event, event]}))
    shape = Gemm(1, 2, 4, 3, "fp32", (3, 1), (1, 3))
    assert floor.read_gemm_shapes(path) == {shape: 2}
    assert shape.bytes == 4 * (2 * 3 + 3 * 4 + 2 * 4)
    event["args"]["Input type"] = ["double"] * 3
    path.write_text(json.dumps({"traceEvents": [event]}))
    with pytest.raises(ValueError, match="unqualified"):
        floor.read_gemm_shapes(path)


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"format": "old"},
        {"decisions": True},
        {"actor_slots": 8},
        {"extra": 0},
        {"tick_layouts": [{"buffer": 0, "anchor": None, "batch": -1}]},
        {"optimizer_groups": []},
        {"critic_parameters": [[0]]},
    ],
)
def test_layout_rejects_schema_and_shape_changes(tmp_path: Path, change: dict[str, object]) -> None:
    raw: dict[str, object] = {
        "format": "hal.o65.floor-layout.v1",
        "tick_layouts": [{"buffer": 0, "anchor": None, "batch": 8}],
        "actor_steps": 1,
        "actor_windows": 32,
        "actor_slots": 16,
        "reference_windows": 32,
        "reference_slots": 64,
        "frames": 256,
        "decisions": 192,
        "matches": 1,
        "critic_parameters": [[8, 8]],
        "optimizer_groups": [{"use_muon": True, "logical_splits": 1, "shapes": [[8, 8]]}],
    }
    raw.update(change)
    path = tmp_path / "layout.json"
    path.write_text(json.dumps(raw))
    if change:
        with pytest.raises(ValueError):
            floor.read_layout(path)
    else:
        assert floor.read_layout(path)["tick_layouts"][0]["batch"] == 8
