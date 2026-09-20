"""Compiler-cache persistence for short O59 benchmark iterations."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts/benchmark_o59_modal.py"
_SPEC = importlib.util.spec_from_file_location("o59_benchmark_launcher_test", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
launcher = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = launcher
_SPEC.loader.exec_module(launcher)


def test_compiler_cache_round_trip_and_corruption_rejection(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "kernel.py").write_text("kernel source\n")
    (cache / "triton").mkdir()
    (cache / "triton" / "kernel.cubin").write_bytes(b"compiled kernel")
    saved = tmp_path / "saved"
    saved.mkdir()
    launcher.save_compiler_cache(cache, saved)
    restored = tmp_path / "restored"
    launcher.restore_compiler_cache(saved, restored)
    assert (restored / "kernel.py").read_text() == "kernel source\n"
    assert (restored / "triton" / "kernel.cubin").read_bytes() == b"compiled kernel"
    with (saved / "compiler-cache.tar.gz").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        launcher.restore_compiler_cache(saved, tmp_path / "corrupt")
    assert not (tmp_path / "corrupt").exists()


def test_compiler_cache_rejects_unknown_format(tmp_path: Path) -> None:
    (tmp_path / "compiler-cache.json").write_text(json.dumps({"format": "unknown"}))
    with pytest.raises(ValueError, match="unsupported compiler cache"):
        launcher.restore_compiler_cache(tmp_path, tmp_path / "restored")


@pytest.mark.parametrize("seconds", [0, 86101])
def test_invalid_budget_fails_before_launch(seconds: int) -> None:
    with pytest.raises(ValueError, match="budget_seconds"):
        launcher.main(launcher.StartArgs(budget_seconds=seconds))


def _job(case="baseline"):
    return launcher.Job("o59-benchmark-job-v2", "a" * 32, "b" * 40, "c" * 64, case, None, None, None, None)


def test_queue_runs_serially_and_finish_follows_prior_jobs() -> None:
    import time
    from dataclasses import asdict

    first = _job()
    second = launcher.Job(**{**asdict(first), "job_id": "d" * 32})
    messages = iter([asdict(first), asdict(second), {"finish": True}])
    executed, records = [], []

    def execute(job):
        executed.append(job.job_id)
        return 0 if job == second else 1

    launcher.serve_jobs(
        receive=lambda timeout: next(messages), execute=execute, record=records.append, deadline=time.monotonic() + 30
    )
    assert executed == [first.job_id, second.job_id]
    assert [record["status"] for record in records] == ["failed", "complete"]


def test_source_and_bank_hashes_are_required(tmp_path: Path) -> None:
    from dataclasses import replace

    launcher.validate_job(_job())
    with pytest.raises(ValueError, match="bank and reference"):
        launcher.validate_job(_job("reuse-history"))
    artifact = tmp_path / "bank.pt"
    artifact.write_bytes(b"bank")
    assert launcher.checked_artifact(tmp_path, "bank.pt", launcher.file_sha256(artifact)) == artifact
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        launcher.checked_artifact(tmp_path, "bank.pt", "0" * 64)
    with pytest.raises(ValueError, match="invalid bank path"):
        launcher.validate_job(replace(_job(), bank="../bank.pt", bank_sha256="a" * 64))


def test_child_failure_and_deadline_preserve_reports(tmp_path: Path) -> None:
    import os
    import time

    calls = []
    exit_code = launcher.run_child(
        [sys.executable, "-c", "raise SystemExit(7)"],
        cwd=tmp_path,
        env=os.environ.copy(),
        log_path=tmp_path / "failed.log",
        deadline=time.monotonic() + 30,
        persist=lambda: calls.append(True),
    )
    assert exit_code == 7 and calls
    with pytest.raises(TimeoutError, match="deadline"):
        launcher.run_child(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            cwd=tmp_path,
            env=os.environ.copy(),
            log_path=tmp_path / "timeout.log",
            deadline=time.monotonic() + 0.1,
            persist=lambda: calls.append(True),
        )
    assert len(calls) >= 2
    assert (tmp_path / "failed.log").exists()


def test_expired_queue_deadline_does_not_consume_job() -> None:
    import time

    received = []
    with pytest.raises(TimeoutError, match="deadline"):
        launcher.serve_jobs(
            receive=lambda timeout: received.append(timeout),
            execute=lambda job: 0,
            record=lambda result: None,
            deadline=time.monotonic() - 1,
        )
    assert received == []


def test_failed_parent_does_not_leave_child_process_group(tmp_path: Path) -> None:
    import os
    import time

    code = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); open('pid','w').write(str(p.pid)); raise SystemExit(1)"
    assert (
        launcher.run_child(
            [sys.executable, "-c", code],
            cwd=tmp_path,
            env=os.environ.copy(),
            log_path=tmp_path / "child.log",
            deadline=time.monotonic() + 10,
            persist=lambda: None,
        )
        == 1
    )
    pid = int((tmp_path / "pid").read_text())
    state = Path(f"/proc/{pid}/stat")
    for _ in range(100):
        if not state.exists() or state.read_text().split()[2] == "Z":
            break
        time.sleep(0.01)
    else:
        pytest.fail("compiler child survived the benchmark parent")


def test_cancellation_records_job_and_stops_queue() -> None:
    import time
    from dataclasses import asdict

    messages = iter([asdict(_job()), {"finish": True}])
    records = []

    def interrupt(job):
        raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError):
        launcher.serve_jobs(
            receive=lambda timeout: next(messages),
            execute=interrupt,
            record=records.append,
            deadline=time.monotonic() + 30,
        )
    assert records[0]["status"] == "cancelled"
    assert next(messages) == {"finish": True}
