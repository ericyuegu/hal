"""Session always reaps the Dolphin subprocess, even when shutdown unwinds
through a libmelee failure or the Python parent dies before __exit__ runs.

Pins three failure modes that have orphaned Dolphin on UDP 51441 in the past:

* normal ``with``-block exit (the happy path)
* libmelee's ``Console.stop()`` raises during teardown (we observed
  ``AssertionError: can only join a started process`` from
  ``slippstream.shutdown()`` when the worker never started)
* the Python parent is SIGKILL'd before its atexit/`__exit__` cleanup runs
  (PR_SET_PDEATHSIG path).
"""

import multiprocessing as mp

# Match the project convention used in test_roundtrip.py: forkserver + torch
# + libmelee imports are flaky together, so force plain fork.
if mp.get_start_method(allow_none=True) != "fork":
    mp.set_start_method("fork", force=True)

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Never

import pytest

from hal.paths import EMULATOR_PATH
from hal.paths import ISO_PATH as _ISO_PATH

ISO_PATH = Path(_ISO_PATH)
DOLPHIN_PATH = Path(EMULATOR_PATH)


def _missing_prerequisite(message: str) -> Never:
    if os.environ.get("HAL_REQUIRE_INTEGRATION") == "1":
        pytest.fail(message)
    pytest.skip(message)


def _check_prereqs() -> None:
    if not ISO_PATH.is_file():
        _missing_prerequisite(f"ISO missing at {ISO_PATH}; run `python -m hal.scripts.fetch --name ssbm.ciso`")
    if not DOLPHIN_PATH.is_file():
        _missing_prerequisite(
            f"Dolphin AppRun missing at {DOLPHIN_PATH}; run `python -m hal.scripts.fetch --name dolphin-exiai`"
        )


def _slippi_port_holders() -> list[int]:
    """PIDs holding UDP 51441 right now. Empty list means no orphaned Dolphin."""
    try:
        result = subprocess.run(
            ["lsof", "-iUDP:51441", "-Fp"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError, subprocess.TimeoutExpired:
        _missing_prerequisite("lsof not available")
    return [int(line[1:]) for line in result.stdout.splitlines() if line.startswith("p")]


def _spawn_session_then(action_src: str) -> subprocess.Popen:
    """Launch a sidecar Python that opens a Session and runs ``action_src``
    after printing READY. We read its stdout to know when to act on it."""
    script = f"""
import sys
import melee
from hal.paths import EMULATOR_PATH, ISO_PATH
from hal.sim.session import Matchup, PlayerSetup, Session

matchup = Matchup(
    stage=melee.Stage.FINAL_DESTINATION,
    players=(
        PlayerSetup(port=1, character=melee.Character.FOX, cpu_level=0),
        PlayerSetup(port=2, character=melee.Character.FOX, cpu_level=9),
    ),
)
with Session(iso_path=ISO_PATH, dolphin_path=EMULATOR_PATH, blocking_input=True) as s:
    s.start_match(matchup)
    print('READY', flush=True)
    {action_src}
"""
    return subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _wait_ready(proc: subprocess.Popen, timeout: float = 180.0) -> bool:
    """Block until child prints READY (Dolphin booted + match started) or dies."""
    deadline = time.monotonic() + timeout
    assert proc.stdout is not None
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                return False
            time.sleep(0.1)
            continue
        if line.strip() == "READY":
            return True
    return False


@pytest.mark.integration
def test_normal_exit_kills_dolphin() -> None:
    """Happy path: ``with Session(...)`` cleanup leaves no orphaned Dolphin."""
    _check_prereqs()
    proc = _spawn_session_then("import time; time.sleep(1)")
    try:
        assert _wait_ready(proc), proc.stdout.read() if proc.stdout else ""
        proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    time.sleep(1.0)  # let kernel reap
    assert _slippi_port_holders() == []


@pytest.mark.integration
def test_sigkill_parent_kills_dolphin() -> None:
    """PR_SET_PDEATHSIG path: SIGKILL the Python parent while a Session is
    open. The kernel must deliver SIGKILL to the Dolphin child so it doesn't
    keep UDP 51441 bound across runs."""
    _check_prereqs()
    proc = _spawn_session_then("import time; time.sleep(120)")
    try:
        assert _wait_ready(proc), proc.stdout.read() if proc.stdout else ""
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
    time.sleep(2.0)  # PDEATHSIG delivery + reaping
    assert _slippi_port_holders() == []
