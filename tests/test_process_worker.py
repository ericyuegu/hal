"""Focused tests for the spawned Session worker protocol."""

import multiprocessing as mp
import threading
from collections.abc import Mapping

import melee
import numpy as np
import pytest

import hal.sim.worker as worker_module
from hal.sim.inputs import ControllerInputs
from hal.sim.ipc import ArenaSpec
from hal.sim.ipc import ControlMessage
from hal.sim.ipc import MessageType
from hal.sim.ipc import ResultArena
from hal.sim.ipc import ResultSpec
from hal.sim.ipc import RolloutArena
from hal.sim.ipc import receive_control
from hal.sim.ipc import result_shm_name
from hal.sim.ipc import send_control
from hal.sim.rollout import PolicyRuntimeSpec
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup


def _post() -> dict:
    return {
        "position": {"x": 1.0, "y": 2.0},
        "percent": 0.0,
        "shield": 60.0,
        "stock": 4,
        "direction": 1.0,
        "action": 14,
        "jumps_used": 0,
        "airborne": 0,
        "hurtbox_state": 0,
        "hitlag_left": 0.0,
    }


def _live_frame(frame_id: int) -> dict:
    return {
        "id": frame_id,
        "start": {"random_seed": frame_id},
        "ports": {port: {"leader": {"post": _post()}} for port in (1, 2)},
        "items": [],
        "stage": int(melee.Stage.FINAL_DESTINATION.value),
    }


class _TerminalFrameSession:
    """Ends with a menu frame that has no live player post fields."""

    def __init__(self, **_kwargs: object) -> None:
        self.steps = 0

    def __enter__(self) -> _TerminalFrameSession:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def start_match(self, _matchup: Matchup) -> dict:
        return _live_frame(0)

    def step(self, inputs: Mapping[int, ControllerInputs]) -> tuple[dict, bool]:
        assert set(inputs) == {1, 2}
        self.steps += 1
        if self.steps == 1:
            return _live_frame(1), True
        return {"id": 2, "start": {"random_seed": 2}, "ports": {}}, False


def test_worker_keeps_terminal_frame_without_publishing_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """A normal game end is trajectory data, but it is not a model observation."""
    monkeypatch.setattr(worker_module, "Session", _TerminalFrameSession)
    runtime = PolicyRuntimeSpec(
        context_frames=8,
        prediction_frames=4,
        execution_stride=4,
        committed_frames=0,
        action_dim=14,
    )
    arena_spec = ArenaSpec(
        workers=2,
        ring_capacity=runtime.raw_ring_capacity,
        prediction_frames=runtime.prediction_frames,
        action_dim=runtime.action_dim,
        action_token_groups=0,
    )
    matchup = Matchup(
        stage=melee.Stage.FINAL_DESTINATION,
        players=(
            PlayerSetup(port=1, character=melee.Character.FOX),
            PlayerSetup(port=2, character=melee.Character.FOX),
        ),
    )
    parent, child = mp.Pipe(duplex=True)
    send_buffer = bytearray(64)
    receive_buffer = bytearray(64)
    with RolloutArena.create(arena_spec) as arena:
        thread = threading.Thread(
            target=worker_module._session_worker,
            args=(0, child, arena.descriptor, {}, matchup, (1, 2), (0, 1), runtime, 10, False),
        )
        thread.start()
        requests: list[ControlMessage] = []
        for _ in range(2):
            message, receive_buffer = receive_control(parent, receive_buffer)
            assert message.message_type is MessageType.PLAN_REQUEST
            requests.append(message)
            arena.plan_actions[message.task_id, message.plan_slot].fill(0.0)
        for message in requests:
            send_buffer = send_control(
                parent,
                ControlMessage(
                    message_type=MessageType.PLAN_READY,
                    worker_id=0,
                    task_id=message.task_id,
                    auxiliary_sequence=message.auxiliary_sequence,
                    count=runtime.prediction_frames,
                    plan_slot=message.plan_slot,
                    port_or_slot=message.port_or_slot,
                ),
                send_buffer,
            )

        ready, receive_buffer = receive_control(parent, receive_buffer)
        assert ready.message_type is MessageType.RESULT_READY
        assert ready.count == 3
        result = ResultArena.attach(
            result_shm_name(arena.descriptor.name, 0),
            ResultSpec(frames=ready.count, segments=ready.auxiliary_sequence, ports=2),
        )
        try:
            np.testing.assert_array_equal(result.frame_id, [0, 1, 2])
            assert np.isnan(result.post[:, :, -1]).all()
        finally:
            result.close()
        send_control(parent, ControlMessage(message_type=MessageType.RESULT_RELEASED, worker_id=0), send_buffer)
        thread.join(timeout=5.0)
        assert not thread.is_alive()
    parent.close()
