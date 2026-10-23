"""Preserve explicit training batch choices in the command sent to the B200."""

import dataclasses
import sys
from pathlib import Path
from typing import cast

import modal
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
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
        return Process(0 if len(self.commands) == 1 else 1)


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
    command = sandbox.commands[0][2]
    if windows is None:
        assert "--training.windows-per-step" not in command
    else:
        assert f"--training.windows-per-step {windows} " in command
    if microbatches is None:
        assert "--training.actor-microbatches" not in command
    else:
        assert f"--training.actor-microbatches {microbatches} " in command
    assert "--profiling.rollout-only" not in command
