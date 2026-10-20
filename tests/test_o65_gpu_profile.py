import json
import os
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from test_065_loop import exp as gpu_profile


@pytest.mark.integration
def test_nvtx_phase_contains_its_completed_gpu_work(tmp_path: Path) -> None:
    nsys = os.environ.get("HAL_NSYS_BINARY") or shutil.which("nsys")
    if nsys is None:
        pytest.skip("requires optional Nsight Systems")
    if not torch.cuda.is_available():
        if os.environ.get("HAL_REQUIRE_INTEGRATION") == "1":
            pytest.fail("phase attribution requires CUDA")
        pytest.skip("requires CUDA")
    script = tmp_path / "phase.py"
    script.write_text(
        "import json, sys, torch\n"
        "from pathlib import Path\n"
        "from test_065_loop import exp\n"
        "x = torch.randn(2048, 2048, device='cuda')\n"
        "y = torch.empty_like(x)\n"
        "for _ in range(2): torch.mm(x, x, out=y)\n"
        "torch.cuda.synchronize()\n"
        "graph = torch.cuda.CUDAGraph()\n"
        "with torch.cuda.graph(graph):\n"
        "    for _ in range(64): torch.mm(x, x, out=y)\n"
        "recorder = exp.UpdateProfile(Path(sys.argv[1]), torch.device('cuda'), traces=False)\n"
        "torch.cuda.profiler.start()\n"
        "with recorder.phase('preparation'):\n"
        "    with recorder.phase('reference'): graph.replay()\n"
        "torch.cuda.profiler.stop()\n"
    )
    report = tmp_path / "phase"
    subprocess.run(
        [
            nsys,
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--cuda-graph-trace=node",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            f"--output={report}",
            sys.executable,
            str(script),
            str(tmp_path),
        ],
        check=True,
        timeout=60,
    )
    database = tmp_path / "phase.sqlite"
    subprocess.run(
        [nsys, "export", "--type=sqlite", f"--output={database}", str(report) + ".nsys-rep"], check=True, timeout=60
    )
    db = sqlite3.connect(database)
    try:
        end = db.execute("SELECT end FROM NVTX_EVENTS WHERE text='o65/reference'").fetchone()[0]
        kernel_end = db.execute("SELECT MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
        assert end >= kernel_end
    finally:
        db.close()


def test_trace_counts_device_work_once_and_merges_overlap(tmp_path: Path) -> None:
    path = tmp_path / "trace.json"
    path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"cat": "user_annotation", "name": "o65_profile/test", "ts": 0, "dur": 200},
                    {"cat": "kernel", "name": "gemm", "ts": 10, "dur": 10},
                    {"cat": "kernel", "name": "gemm", "ts": 15, "dur": 10},
                    {
                        "cat": "gpu_memcpy",
                        "name": "Memcpy HtoD (Pinned -> Device)",
                        "ts": 30,
                        "dur": 5,
                        "args": {"bytes": 64},
                    },
                    {
                        "cat": "gpu_memcpy",
                        "name": "Memcpy DtoH (Device -> Pinned)",
                        "ts": 30,
                        "dur": 3,
                        "args": {"bytes": 4},
                    },
                    {"cat": "cpu_op", "name": "aten::matmul", "ts": 0, "dur": 100},
                    {"cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 2, "dur": 1},
                    {"cat": "cuda_runtime", "name": "cudaGraphLaunch", "ts": 3, "dur": 1},
                    {"cat": "cuda_runtime", "name": "cudaEventSynchronize", "ts": 4, "dur": 1},
                ]
            }
        )
    )
    result = gpu_profile.summarize_trace(path)
    assert result["device_busy_seconds"] == pytest.approx(20e-6)
    assert result["kernel_seconds"] == pytest.approx(20e-6)
    assert result["kernel_count"] == 2
    assert result["runtime_launch_count"] == 2
    assert result["runtime_launch_seconds"] == pytest.approx(2e-6)
    assert result["top15_kernel_share"] == 1
    assert result["copies"] == {"h2d_count": 1, "h2d_bytes": 64, "d2h_count": 1, "d2h_bytes": 4}
    assert result["synchronizations"] == {"cudaEventSynchronize": 1}


def test_copy_without_bytes_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "trace.json"
    path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"cat": "user_annotation", "name": "o65_profile/test", "ts": 0, "dur": 200},
                    {"cat": "gpu_memcpy", "name": "Memcpy HtoD", "ts": 0},
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="byte count"):
        gpu_profile.summarize_trace(path)


def test_sample_counts_calls_and_work_without_cuda(tmp_path: Path) -> None:
    recorder = gpu_profile.UpdateProfile(tmp_path, torch.device("cpu"), traces=False)
    for count in (4, 2, 1):
        with recorder.sample("chunk", work=count):
            pass
    assert recorder.calls == {"outside/chunk": 3}
    assert recorder.work == {"outside/chunk": 7}
    assert recorder.samples == []
    recorder.write({"test": 1})
    assert json.loads((tmp_path / "timed.json").read_text())["calls"] == {"outside/chunk": 3}


@pytest.mark.parametrize("count", [0, -1])
def test_invalid_warm_count_is_rejected(count: int) -> None:
    with pytest.raises(ValueError, match="warm update"):
        gpu_profile.ProfileConfig(count)


def test_profile_rejects_training_side_effects(tmp_path: Path) -> None:
    import dataclasses

    from test_065_loop import exp

    cli = exp.GpuProfileCli(tmp_path / "final.pt", tmp_path / "data", tmp_path / "output", cpus=(0,))
    native = cli.native()
    for change in ({"wandb_project": "oops"}, {"resume": tmp_path}):
        with pytest.raises(ValueError, match="profiling requires"):
            dataclasses.replace(native, **change)


def test_nsys_rejection_cleans_up_partially_attached_recorder(tmp_path: Path) -> None:
    from test_065_loop import trainer

    run = trainer(groups=1, anchor_rate=0)
    with pytest.raises(TypeError, match="TransformerPolicy"):
        gpu_profile.profile_nsys(run, tmp_path)
    assert run.profile is None and run.engine.profile is None


def test_profile_rejects_cold_groups_and_misreported_update_count(tmp_path: Path) -> None:
    import dataclasses

    cli = gpu_profile.GpuProfileCli(tmp_path / "final.pt", tmp_path / "data", tmp_path / "output", cpus=(0,))
    with pytest.raises(ValueError, match="warm every group"):
        dataclasses.replace(cli, warm_updates=3).native()
    with pytest.raises(ValueError, match="two measured updates"):
        dataclasses.replace(cli.native(), updates=7)


def test_measurement_syncs_are_excluded_by_annotations(tmp_path: Path) -> None:
    path = tmp_path / "syncs.json"
    path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"cat": "user_annotation", "name": "o65_profile/test", "ts": 10, "dur": 100},
                    {"cat": "kernel", "name": "gemm", "ts": 40, "dur": 1},
                    {"cat": "user_annotation", "name": "o65_profile_sync", "ts": 20, "dur": 10},
                    {"cat": "cuda_runtime", "name": "cudaDeviceSynchronize", "ts": 21, "dur": 5},
                    {"cat": "cuda_runtime", "name": "cudaStreamSynchronize", "ts": 50, "dur": 5},
                    {"cat": "cuda_runtime", "name": "cudaDeviceSynchronize", "ts": 120, "dur": 5},
                ]
            }
        )
    )
    result = gpu_profile.summarize_trace(path)
    assert result["synchronizations"] == {"cudaStreamSynchronize": 1}
    assert result["measurement_synchronizations_excluded"] == {"cudaDeviceSynchronize": 2}


def test_estimates_scale_by_work_and_recompute_top_kernels(tmp_path: Path) -> None:
    timed = gpu_profile.UpdateProfile(tmp_path, torch.device("cpu"), traces=False)
    traced = gpu_profile.UpdateProfile(tmp_path, torch.device("cpu"), traces=True)
    timed.calls["student/trunk_chunk"] = 3
    timed.work["student/trunk_chunk"] = 40
    timed.work["student/trunk_tokens"] = 40
    timed.phases["student"] = {"wall_seconds": 2}
    traced.samples = [
        {
            "site": "student/trunk_chunk",
            "work": 10,
            "device_busy_seconds": 0.1,
            "kernel_seconds_by_name": {"gemm": 0.1},
            "kernel_count": 2,
            "runtime_launch_count": 2,
            "runtime_launch_seconds": 0.002,
            "copies": {"h2d_count": 1, "h2d_bytes": 4},
            "synchronizations": {"cudaStreamSynchronize": 1},
        }
    ]
    estimate = gpu_profile.estimate_profile(timed, traced)
    site = estimate["sites"]["student/trunk_chunk"]
    assert site["scale"] == 4 and site["kernel_count"] == 6
    assert site["runtime_launch_seconds"] == pytest.approx(0.006)
    assert site["device_busy_seconds"] == pytest.approx(0.4)
    assert site["synchronizations_excluding_injected"] == {"cudaStreamSynchronize": 3}
    assert site["copies"] == {"h2d_count": 3, "h2d_bytes": 16}
    assert estimate["flops"]["student"] == 2 * 9_400_000 * 40


def test_profile_disabled_preserves_next_batch_and_update(tmp_path: Path) -> None:
    import contextlib

    from test_065_loop import exp
    from test_065_loop import trainer

    class HostOnlyProfile(gpu_profile.UpdateProfile):
        @contextlib.contextmanager
        def phase(self, name: str) -> Iterator[None]:
            previous = self.phase_name
            self.phase_name = name
            try:
                yield
            finally:
                self.phase_name = previous

    control = trainer(groups=1, anchor_rate=0)
    treatment = trainer(groups=1, anchor_rate=0)
    recorder = HostOnlyProfile(tmp_path, torch.device("cpu"), traces=False)
    treatment.profile = treatment.engine.profile = recorder
    for _ in range(2):
        control_batch, _ = control.run_update()
        treatment_batch, _ = treatment.run_update()
        assert exp._batch_hash(control_batch) == exp._batch_hash(treatment_batch)
        assert exp._state_hash(control.policy.model.state_dict()) == exp._state_hash(
            treatment.policy.model.state_dict()
        )
        assert exp._state_hash(control.actor_optimizer.state_dict()) == exp._state_hash(
            treatment.actor_optimizer.state_dict()
        )
        assert exp._state_hash(control.critic_optimizer.state_dict()) == exp._state_hash(
            treatment.critic_optimizer.state_dict()
        )
    assert recorder.calls["actor/step"] > 0


def test_missing_device_activity_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"cat": "user_annotation", "name": "o65_profile/test", "ts": 0, "dur": 1},
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="no CUDA kernels"):
        gpu_profile.summarize_trace(path)


def test_profile_requires_dense_window_sites_and_used_bootstraps() -> None:
    sites = {"rollout/tick", "rollout/boundary_tick", "reference/windows", "critic/fit", "actor/step"}
    gpu_profile.validate_profile_sites(sites, bootstrap=False)
    with pytest.raises(ValueError, match="preparation/bootstrap"):
        gpu_profile.validate_profile_sites(sites, bootstrap=True)
    gpu_profile.validate_profile_sites(sites | {"preparation/bootstrap"}, bootstrap=True)
    with pytest.raises(ValueError, match="reference/windows"):
        gpu_profile.validate_profile_sites(
            (sites - {"reference/windows"}) | {"reference/trunk_chunk", "reference/decoder_chunk"}, bootstrap=False
        )
