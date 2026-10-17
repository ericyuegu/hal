import ctypes
import threading
import time
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock

import pytest

from hal.netplay_service import runner
from hal.netplay_service.control import QueueControl
from hal.netplay_service.domain import CHARACTER_VALUES
from hal.netplay_service.domain import IMITATION_VALUES
from hal.netplay_service.domain import STAGE_VALUES
from hal.netplay_service.queue_client import QueueEndpoint
from hal.netplay_service.self_play import selection
from hal.sim.session import Session
from hal.sim.session import SessionCanceled
from hal.sim.session import step_blocking


def _control(path: Path) -> QueueControl:
    control = QueueControl(QueueEndpoint("http://127.0.0.1:1", "local"), "session", Mock(), path)
    control._connected = True
    control._state = {"draining": False, "stream": {"slot": 0, "key": "private"}}
    control._capacity = {"queued": 0, "active": 0, "healthy_slots": 2}
    return control


@pytest.mark.parametrize(
    "reason", ["queued", "active", "drain", "disconnect", "lease", "stale", "assignment", "pairing"]
)
def test_demo_requires_global_idle_fresh_control_and_lease(tmp_path: Path, reason: str) -> None:
    with closing(_control(tmp_path / "status")) as control:
        cancel = threading.Event()
        assert not control.acquire_demo(cancel)
        control._idle_since = time.monotonic() - 1.01
        assert control.acquire_demo(cancel)
        with control._changed:
            if reason in ("queued", "active"):
                control._capacity[reason] = 1
            elif reason == "drain":
                control._state["draining"] = True
            elif reason == "disconnect":
                control._connected = False
            elif reason == "lease":
                control._state["stream"] = None
            elif reason == "stale":
                control._last_health = time.monotonic() - 21
            elif reason == "assignment":
                control._assignments[1] = "human"
            else:
                control._pairing = {"slot": 1}
            control._notify_work()
        assert cancel.is_set()
        assert control._demo_held
        control.release_demo()
        assert not control.acquire_demo(threading.Event())


def test_human_claim_waits_for_demo_cleanup_and_wakes_without_poll_delay(tmp_path: Path) -> None:
    with closing(_control(tmp_path / "status")) as control:
        cancel = threading.Event()
        control._idle_since = time.monotonic() - 2
        assert control.acquire_demo(cancel)
        # Empty human workers cannot interrupt an otherwise eligible demo.
        assert control._claim("/claim", {"slot": 0}) == (204, None)
        assert not cancel.is_set()
        control._capacity["queued"] = 1
        forward = Mock(return_value=(204, None))
        control._forward = forward
        assert control._claim("/claim", {"slot": 0}) == (204, None)
        assert cancel.is_set()
        forward.assert_not_called()
        epoch = control._work_epoch
        woke = threading.Event()

        def wait() -> None:
            code, data = control.dispatch("POST", "/v1/runner/sessions/session/wait", {"after": epoch}, None)
            assert code == 200 and data["epoch"] > epoch
            woke.set()

        thread = threading.Thread(target=wait)
        thread.start()
        control.release_demo()
        assert woke.wait(0.25)
        thread.join()
        control._claim("/claim", {"slot": 0})
        forward.assert_called_once()


def _demo(tmp_path: Path, control: QueueControl, monkeypatch) -> tuple[runner._IdleDemo, Mock]:
    config = runner.RunnerConfig(
        QueueEndpoint("http://127.0.0.1:1", "local"),
        "session",
        tmp_path / "policy",
        (tmp_path / "account0", tmp_path / "account1"),
        (51441, 51442),
        tmp_path / "iso",
        tmp_path / "dolphin",
        tmp_path / "replays",
        tmp_path / "status",
        "a" * 40,
        wants_stream=True,
        display_base=100,
        stream_display=":90",
        idle_self_play=True,
    )
    process = Mock(pid=987654, exitcode=None)
    process.is_alive.return_value = True
    context = Mock()
    context.Event.return_value = threading.Event()
    context.RawValue.return_value = ctypes.c_int(0)
    context.Process.return_value = process
    demo = runner._IdleDemo(context, config, Mock(), control, {slot: Mock() for slot in range(4, 8)}, 2, (), "a" * 64)
    monkeypatch.setattr(demo, "_group_alive", lambda _pid: False)
    control._idle_since = time.monotonic() - 2
    demo.tick(threading.Event(), False)
    process.start.assert_called_once()
    assert control._demo_held
    return demo, process


def test_human_arrival_holds_gate_until_release(tmp_path: Path, monkeypatch) -> None:
    with closing(_control(tmp_path / "status")) as control:
        demo, process = _demo(tmp_path, control, monkeypatch)
        control._capacity["queued"] = 1
        demo.tick(threading.Event(), False)
        assert demo.cancel.is_set() and control._demo_held
        assert demo.process is process
        process.is_alive.return_value = False
        process.exitcode = 0
        demo.tick(threading.Event(), False)
        assert demo.process is None
        assert not control._demo_held
        assert not demo.disabled
        assert demo.context.Process.call_args.kwargs["target"] is runner.self_play.worker
        assert len(demo.context.Process.call_args.kwargs["args"][4]) == 2


def test_hung_demo_is_killed_invalidated_and_disabled_before_human_release(tmp_path: Path, monkeypatch) -> None:
    with closing(_control(tmp_path / "status")) as control:
        demo, process = _demo(tmp_path, control, monkeypatch)
        control._capacity["queued"] = 1
        demo.tick(threading.Event(), False)
        demo.cancel_started = time.monotonic() - 0.61

        def kill(_pid, _signal) -> None:
            process.is_alive.return_value = False
            process.exitcode = -9

        monkeypatch.setattr(runner.os, "killpg", kill)
        invalidate = Mock(return_value=True)
        monkeypatch.setattr(runner, "_invalidate_slot_connection", invalidate)
        demo.tick(threading.Event(), False)
        assert invalidate.call_count == 2
        assert [call.args[1] for call in invalidate.call_args_list] == [4, 5]
        assert demo.disabled and demo.process is None
        assert not control._demo_held
        control._capacity["queued"] = 0
        control._idle_since = time.monotonic() - 2
        demo.tick(threading.Event(), False)
        demo.context.Process.assert_called_once()


def test_failed_inference_cleanup_keeps_human_gate_closed(tmp_path: Path, monkeypatch) -> None:
    with closing(_control(tmp_path / "status")) as control:
        demo, process = _demo(tmp_path, control, monkeypatch)
        process.is_alive.return_value = False
        process.exitcode = 1
        monkeypatch.setattr(runner, "_invalidate_slot_connection", Mock(return_value=False))
        with pytest.raises(runner._EngineLost, match="cleanup"):
            demo.tick(threading.Event(), False)
        assert control._demo_held and demo.disabled


def test_repeated_handoffs_restart_only_after_a_new_idle_second(tmp_path: Path, monkeypatch) -> None:
    with closing(_control(tmp_path / "status")) as control:
        demo, process = _demo(tmp_path, control, monkeypatch)
        for game in range(1, 6):
            assert demo.game_number == game
            control._capacity["queued"] = 1
            demo.tick(threading.Event(), False)
            process.exitcode = 0
            process.is_alive.return_value = False
            demo.tick(threading.Event(), False)
            assert not control._demo_held
            control._capacity["queued"] = 0
            demo.tick(threading.Event(), False)
            assert demo.process is None
            control._idle_since = time.monotonic() - 2
            process.is_alive.return_value = True
            demo.tick(threading.Event(), False)
        assert demo.context.Process.call_count == 6


def test_rotation_is_recorded_seed_deterministic_and_published() -> None:
    first = [selection(55, index) for index in range(30)]
    assert first == [selection(55, index) for index in range(30)]
    assert len({item.stage for item in first}) > 3
    for item in first:
        assert set(item.characters) <= CHARACTER_VALUES
        assert set(item.imitations) <= IMITATION_VALUES
        assert item.stage in STAGE_VALUES


def test_session_cancellation_is_checked_before_each_frame_poll() -> None:
    console = Mock()
    cancel = threading.Event()
    console.step.side_effect = lambda: cancel.set()
    with pytest.raises(SessionCanceled):
        step_blocking(console, 60, cancel.is_set)
    console.step.assert_called_once()
    with pytest.raises(ValueError, match="polling"):
        Session("/iso", dolphin_path="/dolphin", canceled=cancel.is_set)


def test_cancellation_interrupts_initial_receiver_handshake(monkeypatch) -> None:
    from hal.sim.session import connect_cancellable

    cancel = threading.Event()
    client = Mock()

    def poll(_timeout: float) -> bool:
        cancel.set()
        return False

    client._buffer.poll.side_effect = poll
    with pytest.raises(SessionCanceled, match="during connection"):
        connect_cancellable(client, cancel.is_set, 60)
    client._worker.start.assert_called_once()
    client._buffer.recv.assert_not_called()
    monkeypatch.setattr("melee.version.__version__", "0.48.0")
    with pytest.raises(RuntimeError, match="untested"):
        connect_cancellable(client, cancel.is_set, 60)
