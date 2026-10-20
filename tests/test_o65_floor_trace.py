"""Check complete CUDA attribution and nested phase accounting."""

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "o65_floor_trace", Path(__file__).parents[1] / "scripts/analyze_o65_floor.py"
)
assert _SPEC is not None and _SPEC.loader is not None
trace = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = trace
_SPEC.loader.exec_module(trace)


def test_interval_union_and_nested_phase_boundaries() -> None:
    assert trace.union_ns([(2, 5), (4, 8), (10, 12)]) == 8
    ranges = [trace.Range(0, 100, 1, "o65/actor"), trace.Range(20, 30, 1, "o65/actor/muon")]
    assert trace.phase_at(20, ranges) == "muon"
    assert trace.phase_at(30, ranges) == "actor"
    assert trace.phase_at(100, ranges) == "outside"
    with pytest.raises(ValueError, match="negative"):
        trace.union_ns([(5, 4)])


@pytest.mark.parametrize("missing", [False, True])
def test_complete_trace_attributes_graph_nodes_and_rejects_missing_api(tmp_path: Path, missing: bool) -> None:
    path = tmp_path / "trace.sqlite"
    pid, tid = 3 << 24, (3 << 24) | 3
    db = sqlite3.connect(path)
    with db:
        db.execute("CREATE TABLE StringIds (id, value)")
        db.executemany(
            "INSERT INTO StringIds VALUES (?,?)", [(1, "cudaGraphLaunch"), (2, "kernel"), (3, "cudaMemcpyAsync")]
        )
        db.execute("CREATE TABLE NVTX_EVENTS (start,end,globalTid,text,textId)")
        db.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,NULL)",
            [
                (0, 1000, tid, "o65/rollout"),
                (1000, 2000, tid, "o65/preparation"),
                (1100, 1400, tid, "o65/reference"),
                (2000, 2300, tid, "o65/critic"),
                (2300, 3000, tid, "o65/actor"),
                (2400, 2500, tid, "o65/actor/muon"),
                (100, 200, tid, "hal.o65.TransformerPolicy.sample_tick/B32/append2/R4"),
            ],
        )
        db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start,end,globalTid,nameId,correlationId)")
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)",
            [(150, 160, tid, 1, 1), (80, 85, tid, 3, 2), (2410, 2420, tid, 1, 3)],
        )
        db.execute("""CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start,end,globalPid,correlationId,
                      demangledName,gridX,gridY,gridZ,blockX,blockY,blockZ,graphId,graphNodeId)""")
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (170, 190, pid, 99 if missing else 1, 2, 8, 1, 1, 128, 1, 1, 1, 11),
                (2450, 2480, pid, 3, 2, 1, 1, 1, 32, 1, 1, 2, 12),
            ],
        )
        db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start,end,globalPid,correlationId,bytes)")
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (90,95,?,?,32)", (pid, 2))
    db.close()
    if missing:
        with pytest.raises(ValueError, match="no launching API"):
            trace.analyze(path, tmp_path / "out")
        return
    result = trace.analyze(path, tmp_path / "out")
    assert result["tick_kernel_count"] == 1
    assert result["gpu_busy_seconds"] == pytest.approx(55e-9)
    assert result["phases"]["rollout"]["gpu_busy_seconds"] == pytest.approx(25e-9)
    assert result["phases"]["preparation"]["nvtx_host_seconds"] == pytest.approx(700e-9)
    assert result["phases"]["actor"]["nvtx_host_seconds"] == pytest.approx(600e-9)
    assert result["phases"]["muon"]["counts"] == {"graph_kernels": 1}
    assert "grid_x" in (tmp_path / "out/tick-kernels.csv").read_text()
