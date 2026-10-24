"""Preserve explicit training batch choices in the command sent to the B200."""

import dataclasses
import io
import shlex
import subprocess
import sys
from pathlib import Path
from typing import cast

import modal
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from scripts import o65_devbox as devbox
    from scripts.o65_devbox import _run
    from scripts.o65_devbox import bench
finally:
    sys.path.pop(0)


@dataclasses.dataclass(frozen=True, slots=True)
class Process:
    returncode: int
    stdout: tuple[str, ...] = ()

    def wait(self) -> None:
        pass


class Sandbox:
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def exec(self, *command: str, workdir: str) -> Process:
        self.commands.append(command)
        # Stop before download; this test only examines the benchmark command.
        return Process(0 if len(self.commands) <= 2 else 1)


@pytest.mark.parametrize("windows,microbatches", [(None, None), (64, 1), (512, 8)])
def test_bench_preserves_training_batch_choices(windows: int | None, microbatches: int | None) -> None:
    sandbox = Sandbox()
    assert (
        bench(
            cast(modal.Sandbox, sandbox),
            64,
            512,
            full=True,
            windows_per_step=windows,
            actor_microbatches=microbatches,
        )
        == 0
    )
    command = sandbox.commands[1][2]
    if windows is None:
        assert "--training.windows-per-step" not in command
    else:
        assert f"--training.windows-per-step {windows} " in command
    if microbatches is None:
        assert "--training.actor-microbatches" not in command
    else:
        assert f"--training.actor-microbatches {microbatches} " in command
    assert "--profiling.rollout-only" not in command


def test_bench_input_capture_is_explicit_and_requires_training() -> None:
    sandbox = Sandbox()
    with pytest.raises(ValueError, match="requires --full"):
        bench(cast(modal.Sandbox, sandbox), 64, 512, full=False, capture_training_inputs=True)
    assert not sandbox.commands
    assert bench(cast(modal.Sandbox, sandbox), 64, 512, full=True, capture_training_inputs=True) == 0
    assert "--profiling.capture-training-inputs" in sandbox.commands[1][2]


def test_bench_update_gradient_capture_keeps_the_full_quoted_path() -> None:
    sandbox = Sandbox()
    path = Path("/cache/gradient inputs")
    with pytest.raises(ValueError, match="requires --full"):
        bench(cast(modal.Sandbox, sandbox), 64, 512, full=False, update_gradient_inputs=path)
    with pytest.raises(ValueError, match="must be absolute"):
        bench(cast(modal.Sandbox, sandbox), 64, 512, full=True, update_gradient_inputs=Path("relative"))
    assert not sandbox.commands
    assert bench(cast(modal.Sandbox, sandbox), 64, 512, full=True, update_gradient_inputs=path) == 0
    arguments = shlex.split(sandbox.commands[1][2])
    assert arguments[arguments.index("--profiling.update-gradient-inputs") + 1] == str(path)


@pytest.mark.parametrize("groups", [1, 4])
def test_bench_warms_all_groups_and_revisits_them(groups: int) -> None:
    sandbox = Sandbox()
    assert bench(cast(modal.Sandbox, sandbox), 64, 512, full=True, groups=groups) == 0
    arguments = shlex.split(sandbox.commands[1][2])
    for option, expected in (("--groups", groups), ("--updates", groups + 2), ("--profiling.warm-updates", groups)):
        assert int(arguments[arguments.index(option) + 1]) == expected


@pytest.mark.parametrize("groups", [0, -1, True])
def test_bench_rejects_invalid_group_count(groups: int) -> None:
    sandbox = Sandbox()
    with pytest.raises(ValueError, match="groups must be"):
        bench(cast(modal.Sandbox, sandbox), 64, 512, full=True, groups=groups)
    assert not sandbox.commands


def test_bench_rejects_stale_trainer_source_before_launch() -> None:
    class StaleSandbox(Sandbox):
        def exec(self, *command: str, workdir: str) -> Process:
            self.commands.append(command)
            return Process(1)

    sandbox = StaleSandbox()
    with pytest.raises(ValueError, match="remote trainer source differs"):
        bench(cast(modal.Sandbox, sandbox), 64, 512, full=True)
    assert len(sandbox.commands) == 1
    assert "sha256sum experiments/065_selfplay_ppo.py" in sandbox.commands[0][2]


def test_bench_can_keep_compiler_files_and_reports_off_a_full_shared_volume() -> None:
    sandbox = Sandbox()
    assert (
        bench(cast(modal.Sandbox, sandbox), 64, 512, full=True, cache_root=Path("/tmp/train-cache"), compile_threads=2)
        == 0
    )
    command = sandbox.commands[1][2]
    for path in (
        "TMPDIR=/tmp/train-cache/tmp",
        "TORCHINDUCTOR_CACHE_DIR=/tmp/train-cache/inductor",
        "TRITON_CACHE_DIR=/tmp/train-cache/triton",
        "--output /tmp/train-cache/runs/",
        "TORCHINDUCTOR_COMPILE_THREADS=2",
    ):
        assert path in command


@pytest.mark.parametrize("decisions", [1, 16, "all"])
def test_bench_preserves_decision_milestone(decisions: int | str) -> None:
    sandbox = Sandbox()
    assert bench(cast(modal.Sandbox, sandbox), 64, 512, full=True, decisions_per_window=decisions) == 0
    assert f"--training.decisions-per-window {decisions} " in sandbox.commands[1][2]


@pytest.mark.parametrize("decisions", [0, 17, True, "invalid"])
def test_bench_rejects_invalid_decision_milestone(decisions: int | str) -> None:
    sandbox = Sandbox()
    with pytest.raises(ValueError, match="decisions per window"):
        bench(cast(modal.Sandbox, sandbox), 64, 512, full=True, decisions_per_window=decisions)
    assert not sandbox.commands


def test_exec_streams_stderr_for_every_command_in_a_shell_script() -> None:
    sandbox = Sandbox()
    _run(cast(modal.Sandbox, sandbox), "printf diagnostic >&2\nfalse")
    result = subprocess.run(sandbox.commands[0], capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert result.stdout == "diagnostic"
    assert result.stderr == ""


def test_bench_downloads_startup_memory_when_capture_fails_before_the_main_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "bench-fixed-m64-full"
    remote = f"/cache/runs/{name}-startup/phase-memory.json"
    payload = b'{"memory/startup_actor_peak_allocated_bytes": 123}'

    class Filesystem:
        def read_bytes(self, path: str) -> bytes:
            assert path == remote
            return payload

    class Result:
        def __init__(self, code: int, output: str = "") -> None:
            self.returncode = code
            self.stdout = io.StringIO(output)
            self.stderr = io.StringIO()

        def wait(self) -> None:
            pass

    class FailedCapture:
        filesystem = Filesystem()

        def exec(self, *command: str, workdir: str) -> Result:
            script = command[-1]
            if "sha256sum" in script:
                return Result(0)
            if "env " in script:
                return Result(137)
            if "test -d" in script:
                return Result(0 if "-startup" in script else 1)
            assert "find " in script
            return Result(0, remote + "\0")

    monkeypatch.setattr(devbox, "LOCAL_RESULTS", tmp_path)
    monkeypatch.setattr(devbox.time, "strftime", lambda *_args: "fixed")
    assert bench(cast(modal.Sandbox, FailedCapture()), 64, 512, full=True) == 137
    assert (tmp_path / "runs" / name / "startup/phase-memory.json").read_bytes() == payload
