import json
import threading
import time
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock

import pytest

from hal.netplay_service import overlay
from hal.netplay_service import stream
from hal.netplay_service.control import QueueControl
from hal.netplay_service.queue_client import QueueEndpoint
from hal.netplay_service.queue_client import StreamGrant
from hal.netplay_service.stream_lease import StreamLease
from hal.netplay_service.stream_lease import read_stream_lease
from hal.netplay_service.stream_lease import write_stream_lease


def _wait_for(condition) -> None:
    deadline = time.monotonic() + 3
    while not condition():
        assert time.monotonic() < deadline
        time.sleep(0.005)


def test_stream_lease_is_private_versioned_and_expires(tmp_path: Path) -> None:
    path = tmp_path / "lease.json"
    lease = StreamLease(100.0, 3, StreamGrant(0, "private-key"))
    write_stream_lease(path, lease)
    assert path.stat().st_mode & 0o777 == 0o600
    assert "private-key" not in repr(lease)
    assert read_stream_lease(path) == lease
    assert lease.current_grant(129.9) == lease.grant
    assert lease.current_grant(130) is None
    assert lease.current_grant(99) is None
    raw = json.loads(path.read_text())
    raw["schema_version"] = 2
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="schema"):
        read_stream_lease(path)


@pytest.mark.parametrize("updated,depth", [(float("nan"), 0), (100, -1), (100, True), (True, 1)])
def test_stream_lease_rejects_invalid_values(updated: float, depth: int) -> None:
    with pytest.raises(ValueError):
        StreamLease(updated, depth, None)


def test_capacity_updates_cannot_renew_stream_authority(tmp_path: Path) -> None:
    path = tmp_path / "lease.json"
    with closing(
        QueueControl(
            QueueEndpoint("http://127.0.0.1:1", "local"),
            "session",
            Mock(),
            tmp_path / "status",
            stream_lease_path=path,
        )
    ) as control:
        control._connected = True
        control._state = {"draining": False, "stream": {"slot": 0, "key": "private-key"}}
        control._last_health = time.monotonic() - 31
        control._capacity = {"queued": 5}
        control._publish_stream_lease()
        lease = read_stream_lease(path)
        assert lease.queue_depth == 5
        assert lease.current_grant(time.time()) is None
        control._last_health = time.monotonic()
        control._publish_stream_lease()
        assert read_stream_lease(path).current_grant(time.time()) == StreamGrant(0, "private-key")
    assert not path.exists()


def test_overlay_storage_failure_does_not_disconnect_gameplay(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.touch()
    control = QueueControl(
        QueueEndpoint("http://127.0.0.1:1", "local"),
        "session",
        Mock(),
        tmp_path / "status",
        stream_lease_path=blocker / "lease.json",
    )
    try:
        control._connected = True
        control._state = {"draining": False, "stream": None}
        control._publish_stream_lease()
        assert control._connected
        assert control.state().draining is False
    finally:
        control._stream_lease_path = None
        control.close()


def test_overlay_can_restart_without_replacing_game_state_or_stream_lease(tmp_path: Path, monkeypatch) -> None:
    status = tmp_path / "status.json"
    state = status.with_suffix(".stream.json")
    lease = status.with_suffix(".stream-lease.json")
    stream.write_stream_state(state, stream.GameStreamState("FOX", "MASTER", 20, 4))
    write_stream_lease(lease, StreamLease(time.time(), 2, StreamGrant(0, "private-key")))
    original_state = state.read_bytes()
    original_lease = lease.read_bytes()
    studios: list[Mock] = []

    def studio(*_args):
        result = Mock()
        result.stats.return_value = {"stream": {"outputActive": True}}
        studios.append(result)
        return result

    monkeypatch.setattr(stream.obs, "ObsStudio", studio)
    for iteration in range(2):
        stop = threading.Event()
        thread = threading.Thread(target=overlay.run, args=(status, ":90", stop))
        thread.start()
        try:
            _wait_for(
                lambda iteration=iteration: (
                    len(studios) == iteration + 1 and studios[-1].start.call_count == 1 and studios[-1].update.called
                )
            )
            studios[-1].start.assert_called_once_with("private-key", bandwidth_test=False)
        finally:
            stop.set()
            thread.join(timeout=3)
        assert not thread.is_alive()
        studios[-1].close.assert_called_once()
        assert state.read_bytes() == original_state
        assert lease.read_bytes() == original_lease
    assert len(studios) == 2
