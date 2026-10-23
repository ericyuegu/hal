"""Attribute asynchronous GPU kernels by CPU launch, not GPU execution time."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from scripts.o65_actor_trace import stage_seconds
finally:
    sys.path.pop(0)


def test_actor_trace_uses_the_launch_phase_for_late_kernels(tmp_path: Path) -> None:
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"name": "o65_actor/trunk_forward", "ts": 0, "dur": 10},
                    {"name": "o65_actor/decoder_forward", "ts": 10, "dur": 10},
                    {"name": "aten::mm", "cat": "cpu_op", "ts": 5, "args": {"External id": 1}},
                    {"name": "gemm", "cat": "kernel", "ts": 15, "dur": 3, "args": {"External id": 1}},
                ]
            }
        )
    )
    assert stage_seconds(trace) == {"trunk_forward": 3e-6}


def test_actor_trace_rejects_unattributed_work(tmp_path: Path) -> None:
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"name": "gemm", "cat": "kernel", "ts": 15, "dur": 3, "args": {"External id": 1}},
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="no CPU launch"):
        stage_seconds(trace)
