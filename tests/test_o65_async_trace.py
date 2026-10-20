import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "o65_async_trace", Path(__file__).parents[1] / "scripts/analyze_o65_async.py"
)
assert _SPEC is not None and _SPEC.loader is not None
analyze_o65_async = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = analyze_o65_async
_SPEC.loader.exec_module(analyze_o65_async)


@pytest.mark.parametrize("missing", [False, True])
def test_tick_gpu_union_uses_launch_tick_and_worker_thread(tmp_path: Path, missing: bool) -> None:
    path = tmp_path / "trace.sqlite"
    pid, tid = 3 << 24, (3 << 24) | 3
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE StringIds (id,value)")
        db.executemany("INSERT INTO StringIds VALUES (?,?)", [(1, "cudaGraphLaunch"), (2, "cudaLaunchKernel")])
        db.execute("CREATE TABLE NVTX_EVENTS (start,end,text,textId)")
        db.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?,?,?,NULL)",
            [(100, 200, "o65/rollout/boundary_tick"), (200, 300, "o65/rollout/tick")],
        )
        db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start,globalTid,correlationId,nameId)")
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)",
            [(120, tid, 1, 1), (140, tid + 1, 2, 2), (200, tid, 3, 1)],
        )
        db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start,end,globalPid,correlationId)")
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?)",
            [(180, 220, pid, 1), (190, 230, pid, 99 if missing else 2), (250, 270, pid, 3)],
        )
        db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start,end,globalPid,correlationId)")
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (210,240,?,1)", (pid,))
    if missing:
        with pytest.raises(ValueError, match="no launching API"):
            analyze_o65_async.analyze(path)
        return
    report = analyze_o65_async.analyze(path)
    assert report["ticks"] == [
        {
            "tick": 0,
            "boundary": True,
            "host_seconds": 100e-9,
            "gpu_busy_seconds": 60e-9,
            "kernels": 2,
            "eager_launches": 1,
            "graph_launches": 1,
        },
        {
            "tick": 1,
            "boundary": False,
            "host_seconds": 100e-9,
            "gpu_busy_seconds": 20e-9,
            "kernels": 1,
            "eager_launches": 0,
            "graph_launches": 1,
        },
    ]


def test_device_union_rejects_negative_intervals() -> None:
    with pytest.raises(ValueError, match="negative"):
        analyze_o65_async.busy_seconds([(3, 1)])
