import base64
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("hal_gce_run", Path(__file__).parents[1] / "scripts" / "gce_run.py")
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_start_script_refuses_a_second_job_and_passes_the_command_encoded() -> None:
    command = "torchrun train.py --comment 'two words'"
    script = _MODULE.start_script("abc123", command, follow=True)
    lines = script.splitlines()
    assert lines.index(next(line for line in lines if "is-active" in line)) < lines.index(
        next(line for line in lines if "systemd-run" in line)
    )
    assert base64.b64encode(command.encode()).decode() in script
    assert "two words" not in script
    assert "tail -n +1" in script
    subprocess.run(["bash", "-n", "-c", script], check=True)


def test_detached_start_does_not_follow_the_log() -> None:
    script = _MODULE.start_script("abc123", "true", follow=False)
    assert "tail" not in script
    assert "started hal-job on abc123" in script


def test_exactly_one_action_is_required() -> None:
    with pytest.raises(SystemExit, match="exactly one"):
        _MODULE.main(_MODULE.Args(cmd=["true"], attach=True))
    with pytest.raises(SystemExit, match="exactly one"):
        _MODULE.main(_MODULE.Args())
