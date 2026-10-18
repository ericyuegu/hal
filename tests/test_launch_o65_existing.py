import contextlib
import importlib.util
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "launch_o65_existing", Path(__file__).parents[1] / "scripts/launch_o65_existing.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_existing_job_bounds_training_and_upload_without_vm_mutations() -> None:
    script = _MODULE.job_script("a" * 40, "hal-o65-test")
    start = _MODULE.start_command("hal-o65-test", ".staging.sh")
    stop = _MODULE.ssh_command("sudo systemctl stop hal-o65-test.service")
    for text in (script, start, " ".join(stop)):
        assert (
            re.search(r"shutdown|poweroff|instances\s+(stop|delete|create)|max-run-duration|startup-script", text)
            is None
        )
    assert "systemd-run" in start
    assert "RuntimeMaxSec=85500s" in start
    assert "TimeoutStopSec=900s" in start
    assert "KillSignal=SIGKILL" in start
    assert "KillMode=control-group" in start
    assert "ExecStopPost=/bin/bash /var/lib/hal-o65/hal-o65-test/job.sh finish" in start
    assert "810s /bin/bash" in script
    assert "720s docker run" in script
    assert "60s /bin/bash" in script
    assert "o65/pilot-results/$HAL_RUN_NAME" in script
    assert "git rev-parse HEAD" in script
    assert "uv sync --locked" in script
    assert "--checkpoint-retention compact" in script
    assert "42949672960" in script
    assert "flock -n 9" in script
    # The driver hashes its absolute output path into the W&B run id.
    assert 'result_root="/hal-output/$HAL_RUN_NAME"' in script
    assert '"$output:/hal-output/$HAL_RUN_NAME"' in script
    assert "-e HAL_GIT_SHA -e HAL_RUN_NAME" in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    runtime = script.split("<<'HAL_RUNTIME'\n", 1)[1].split("\nHAL_RUNTIME", 1)[0]
    subprocess.run(["bash", "-n"], input=runtime, text=True, check=True)


@pytest.mark.parametrize("name", ["../run", "hal-o65-a;evil", "hal-o65-A", "hal-netplay-runner"])
def test_existing_job_rejects_unsafe_names(name: str) -> None:
    with pytest.raises(ValueError, match="run name"):
        _MODULE.job_script("a" * 40, name)
    with pytest.raises(ValueError, match="run name"):
        _MODULE.main(_MODULE.Args(action="stop", run_name=name, dry_run=True))


def test_stop_requires_original_name_and_needs_no_source_checkout(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(ValueError, match="stop requires --run-name"):
        _MODULE.main(_MODULE.Args(action="stop", dry_run=True))
    with pytest.raises(ValueError, match="full HAL commit SHA"):
        _MODULE.job_script("HEAD", "hal-o65-test")
    _MODULE.main(_MODULE.Args(action="stop", run_name="hal-o65-original", dry_run=True))
    assert "systemctl stop hal-o65-original.service" in capsys.readouterr().out


@pytest.mark.parametrize(
    "stage,run_code,finish_code",
    [
        ("ok", 0, 0),
        ("training", 3, 0),
        ("cap", -signal.SIGKILL, 0),
        ("stop", -signal.SIGKILL, 0),
        ("upload", 0, 5),
        ("upload-timeout", 0, 124),
        ("netplay", 1, 0),
        ("netplay-container", 1, 0),
        ("disk", 1, 0),
        ("runtime", 1, 0),
        ("secret", 7, 0),
        ("inspection", 6, 0),
        ("image", 2, 125),
        ("prepare", 9, 9),
    ],
)
def test_systemd_post_stop_uploads_after_exit_failure_or_kill(
    tmp_path: Path, stage: str, run_code: int, finish_code: int
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    events = tmp_path / "events"
    for command, body in {
        "gcloud": (
            'if [[ "$FAIL_STAGE" == secret && ! -f "$TEST_STATE/secret-failed" ]]; then\n'
            ' touch "$TEST_STATE/secret-failed"; exit 7; fi\n'
            'printf "test-credential"\n'
        ),
        "nvidia-smi": "true\n",
        "install": (
            'if [[ "$FAIL_STAGE" == prepare && "$*" == *control/hal* ]]; then exit 9; fi\nexec /usr/bin/install "$@"\n'
        ),
        "df": 'printf "Avail\\n%s\\n" "${DISK_BYTES:-50000000000}"\n',
        "systemctl": 'if [[ "$FAIL_STAGE" == netplay ]]; then echo active; else echo inactive; fi\n',
        "docker": (
            'echo "docker $*" >> "$EVENTS"\n'
            'if [[ "$1" == ps && "$FAIL_STAGE" == inspection ]]; then exit 6; fi\n'
            'if [[ "$1" == image && "$FAIL_STAGE" == image ]]; then exit 2; fi\n'
            'if [[ "$1" == info ]]; then\n'
            ' if [[ "$FAIL_STAGE" == runtime ]]; then echo \'{"runc":{}}\'; else echo \'{"nvidia":{}}\'; fi\n'
            'elif [[ "$1" == ps && "$FAIL_STAGE" == netplay-container ]]; then echo hal-netplay-overlay;\n'
            'elif [[ "$1" == container ]]; then test -f "$TEST_STATE/${@: -1}"; exit $?;\n'
            'elif [[ "$1" == rm ]]; then rm -f "$TEST_STATE/${@: -1}";\n'
            'elif [[ "$1" == run ]]; then\n'
            ' if [[ "$*" == *results.py* ]]; then\n'
            '  touch "$TEST_STATE/hal-o65-test-upload"\n'
            '  echo upload >> "$EVENTS"\n'
            '  if [[ "$FAIL_STAGE" == upload ]]; then exit 5; fi\n'
            '  if [[ "$FAIL_STAGE" == upload-timeout ]]; then exit 124; fi\n'
            '  if [[ "$FAIL_STAGE" == image ]]; then exit 125; fi\n'
            " else\n"
            '  touch "$TEST_STATE/hal-o65-test-training"\n'
            '  if [[ "$FAIL_STAGE" == training ]]; then exit 3; fi\n'
            '  if [[ "$FAIL_STAGE" == cap || "$FAIL_STAGE" == stop ]]; then\n'
            '   echo ready >> "$EVENTS"; exec sleep 20; fi\n'
            " fi\n"
            "fi\n"
        ),
    }.items():
        path = bin_dir / command
        path.write_text("#!/bin/bash\n" + body)
        path.chmod(0o755)
    root = tmp_path / "jobs"
    job = root / "hal-o65-test"
    job.mkdir(parents=True)
    netplay = tmp_path / "netplay"
    netplay.write_text("keep")
    script = _MODULE.job_script("a" * 40, "hal-o65-test").replace("/var/lib/hal-o65", str(root))
    script_path = job / "job.sh"
    script_path.write_text(script)
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "EVENTS": str(events),
        "TEST_STATE": str(tmp_path),
        "FAIL_STAGE": stage,
    }
    if stage == "disk":
        environment["DISK_BYTES"] = "1000000000"
    with (tmp_path / "console").open("w+") as console:
        process = subprocess.Popen(
            ["bash", str(script_path), "run"], env=environment, stdout=console, stderr=console, start_new_session=True
        )
        try:
            if stage in ("cap", "stop"):
                deadline = time.monotonic() + 10
                while not events.exists() or "ready" not in events.read_text():
                    if process.poll() is not None or time.monotonic() > deadline:
                        pytest.fail("training did not start")
                    time.sleep(0.01)
                # systemd kills the service cgroup, then invokes ExecStopPost separately.
                os.killpg(process.pid, signal.SIGKILL)
            assert process.wait(timeout=10) == run_code
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        result = "timeout" if stage == "cap" else "success" if run_code == 0 else "exit-code"
        after = subprocess.run(
            ["bash", str(script_path), "finish"],
            env={
                **environment,
                "SERVICE_RESULT": result,
                "EXIT_CODE": "killed" if run_code < 0 else "exited",
                "EXIT_STATUS": str(abs(run_code)),
            },
            stdout=console,
            stderr=console,
            timeout=10,
        )
        assert after.returncode == finish_code
        console.seek(0)
        assert "test-credential" not in console.read()
    lines = events.read_text().splitlines()
    assert lines.count("upload") == (0 if stage == "prepare" else 1)
    if stage in ("ok", "training", "cap", "stop", "upload", "upload-timeout"):
        assert lines.index("docker rm -f hal-o65-test-training") < lines.index("upload")
    if stage != "prepare":
        assert "docker rm -f hal-o65-test-upload" in lines
    assert not any("hal-netplay-" in line and "rm -f" in line for line in lines)
    assert not (job / "scratch").exists()
    assert not (job / "control/job.env").exists()
    assert not (job / "control/job.env.partial").exists()
    assert (job / "output/service-result").read_text().strip() == result
    assert (job / "output/upload-exit-code").read_text().strip() == str(finish_code)
    assert netplay.read_text() == "keep"
