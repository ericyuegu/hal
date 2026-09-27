"""Opt-in failure qualification for an isolated idle production-path runner."""

import importlib.util
import os
import signal
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "hal_idle_runner_faults", Path(__file__).parent / "fixtures" / "o59" / "idle_runner_faults.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_fault_injection_rejects_a_reused_or_unowned_pid() -> None:
    with pytest.raises(RuntimeError, match="identity changed"):
        _MODULE._signal_owned(os.getpid(), -1, signal.SIGSTOP)


def test_status_poll_waits_for_absent_file_but_rejects_corruption(tmp_path: Path) -> None:
    path = tmp_path / "runner-status.json"
    assert not _MODULE._ready(path)
    path.write_text("{broken")
    with pytest.raises(ValueError, match="cannot read netplay health status"):
        _MODULE._ready(path)


def test_gpu_owner_uses_local_device_handles(tmp_path: Path) -> None:
    descriptors = tmp_path / "123" / "fd"
    descriptors.mkdir(parents=True)
    (descriptors / "3").symlink_to("/dev/nvidiactl")
    (descriptors / "4").symlink_to("/dev/nvidia-caps/nvidia-cap1")
    assert not _MODULE._has_gpu_device(123, tmp_path)
    (descriptors / "5").symlink_to("/dev/nvidia0")
    assert _MODULE._has_gpu_device(123, tmp_path)
    assert not _MODULE._has_gpu_device(456, tmp_path)


@pytest.mark.integration
def test_idle_runner_detects_hang_recovers_once_then_stays_unavailable() -> None:
    if os.environ.get("HAL_REQUIRE_IDLE_NETPLAY_FAULTS") != "1":
        pytest.skip("set HAL_REQUIRE_IDLE_NETPLAY_FAULTS=1 on the qualification GPU")
    required = ("HAL_NETPLAY_POLICY", "HAL_NETPLAY_IDLE_ACCOUNT", "HAL_NETPLAY_IDLE_OUTPUT")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        pytest.fail(f"idle runner qualification requires {', '.join(missing)}")
    output = Path(os.environ["HAL_NETPLAY_IDLE_OUTPUT"])
    bundle = Path(os.environ["HAL_NETPLAY_POLICY"])
    account = Path(os.environ["HAL_NETPLAY_IDLE_ACCOUNT"])
    report = _MODULE.qualify_idle_runner_faults(output, bundle, account, slippi_port=51461)
    assert report["failure"] is None
    assert not report["status_file_exists"]
    assert report["source_files_unchanged"]
    assert not report["cleanup"]["remaining_descendants"]
