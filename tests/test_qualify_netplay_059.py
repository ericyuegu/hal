"""The qualification harness must not leave service descendants running."""

import importlib.util
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "hal_qualify_netplay_059", Path(__file__).parents[1] / "scripts" / "qualify_netplay_059.py"
)
assert _SPEC is not None and _SPEC.loader is not None
qualify_netplay_059 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = qualify_netplay_059
_SPEC.loader.exec_module(qualify_netplay_059)


def _spawn_idle_descendant(connection: Connection) -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    connection.send(child.pid)
    connection.close()
    while True:
        time.sleep(1)


def test_shutdown_finds_and_terminates_runner_descendants() -> None:
    parent, child = mp.get_context("spawn").Pipe(duplex=False)
    process = mp.get_context("spawn").Process(target=_spawn_idle_descendant, args=(child,))
    process.start()
    child.close()
    descendant_pid = parent.recv()
    parent.close()
    try:
        shutdown = qualify_netplay_059._terminate(process, {})
        assert not process.is_alive()
        assert descendant_pid in shutdown.forced_descendants
        assert not shutdown.remaining_descendants
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
        identity = qualify_netplay_059._process_identity(descendant_pid)
        if identity is not None and identity[0] != "Z":
            os.kill(descendant_pid, signal.SIGKILL)


def test_qualifier_requires_a_display_before_creating_run_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    output = tmp_path / "qualification"
    config = qualify_netplay_059.QualificationConfig(
        output=output,
        bundle=tmp_path / "bundle.hal",
        bot_account=tmp_path / "bot.json",
        peer_account=tmp_path / "peer.json",
        delay=2,
        desired_return=19.976,
        minimum_games=10,
        minimum_gameplay_seconds=1800,
        smoke_frames=300,
        bot_slippi_port=51451,
        peer_slippi_port=51452,
    )
    with pytest.raises(RuntimeError, match="xvfb-run"):
        qualify_netplay_059.qualify(config)
    assert not output.exists()


def test_source_manifest_covers_all_maintained_runtime_modules() -> None:
    hashes = qualify_netplay_059._source_hashes()
    assert "hal/eval/qualification.py" in hashes
    assert "hal/inference/observation_history.py" in hashes
    assert "scripts/qualify_netplay_059.py" in hashes
