"""The development benchmark can use writable storage outside a full volume."""

import io
import shlex
import subprocess
from pathlib import Path
from typing import cast

import modal
import pytest

from scripts import o65_devbox


class Process:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode
        self.stdout = io.StringIO()

    def wait(self) -> int:
        return self.returncode


class Sandbox:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def exec(self, shell: str, flag: str, command: str, *, workdir: str) -> Process:
        assert (shell, flag, workdir) == ("bash", "-c", str(o65_devbox.launch.REMOTE_ROOT))
        self.commands.append(command)
        return Process(int(command.startswith("{\ntest -d ")))


@pytest.mark.parametrize("cache_root", [Path("/cache"), Path("/tmp/tick cache")])
def test_benchmark_uses_declared_cache_root(cache_root: Path) -> None:
    sandbox = Sandbox()
    assert (
        o65_devbox.bench(cast(modal.Sandbox, sandbox), 64, 128, full=False, windows_per_step=64, cache_root=cache_root)
        == 0
    )
    arguments = shlex.split(sandbox.commands[0])
    for name, relative in (("TMPDIR", "tmp"), ("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TRITON_CACHE_DIR", "triton")):
        assert f"{name}={cache_root / relative}" in arguments
    output = Path(arguments[arguments.index("--output") + 1])
    assert output.parent == cache_root / "runs"
    assert shlex.split(sandbox.commands[1])[1:4] == ["test", "-d", str(output)]


def test_run_captures_stderr_from_every_command() -> None:
    sandbox = Sandbox()
    assert o65_devbox._run(cast(modal.Sandbox, sandbox), "printf first >&2\nprintf second") == 0
    result = subprocess.run(["bash", "-c", sandbox.commands[0]], capture_output=True, text=True, check=True)
    assert result.stdout == "firstsecond"
    assert result.stderr == ""
