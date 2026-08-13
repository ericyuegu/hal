"""Failure handling for the spawned closed-loop inference broker."""

import multiprocessing as mp
import os
import time

import melee
import numpy as np
import pytest

import hal.sim.process_vec as process_vec
from hal.sim.process_vec import ProcessVecTelemetry
from hal.sim.rollout import ObservationRow
from hal.sim.rollout import PolicyRuntimeSpec
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.vec import VecMatch


class _UnusedPolicy:
    runtime_spec = PolicyRuntimeSpec(
        context_frames=4,
        prediction_frames=4,
        execution_stride=4,
        committed_frames=0,
        action_dim=14,
    )

    def plan_rows(self, rows):
        raise AssertionError(f"failed workers must not reach policy inference: {rows}")


def _match() -> VecMatch:
    return VecMatch(
        matchup=Matchup(
            stage=melee.Stage.FINAL_DESTINATION,
            players=(
                PlayerSetup(port=1, character=melee.Character.FOX),
                PlayerSetup(port=2, character=melee.Character.FOX, cpu_level=9),
            ),
        ),
        model_ports=(1,),
    )


def _silent_worker(*_args) -> None:
    while True:
        time.sleep(10)


def _abrupt_worker(*_args) -> None:
    os._exit(7)


def _use_forked_fake_worker(monkeypatch: pytest.MonkeyPatch, target) -> None:
    context = mp.get_context("fork")
    monkeypatch.setattr(process_vec.mp, "get_context", lambda _method: context)
    monkeypatch.setattr(process_vec, "_session_worker", target)


def test_silent_worker_is_named_killed_and_reaped(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_forked_fake_worker(monkeypatch, _silent_worker)
    telemetry = ProcessVecTelemetry()
    started = time.monotonic()
    result = process_vec.drive_process_vec(
        [{}],
        [_match()],
        _UnusedPolicy(),
        max_frames=8,
        worker_timeout_seconds=0.1,
        telemetry=telemetry,
    )
    assert time.monotonic() - started < 2.0
    assert result == [[]]
    assert telemetry.failed_workers == 1
    assert telemetry.timed_out_workers == 1


def test_abrupt_worker_exit_is_local_to_that_boot(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_forked_fake_worker(monkeypatch, _abrupt_worker)
    telemetry = ProcessVecTelemetry()
    for _ in range(2):
        result = process_vec.drive_process_vec(
            [{}],
            [_match()],
            _UnusedPolicy(),
            max_frames=8,
            worker_timeout_seconds=1.0,
            telemetry=telemetry,
        )
        assert result == [[]]
    assert telemetry.failed_workers == 2
    assert telemetry.timed_out_workers == 0
    assert telemetry.total_seconds > 0


def test_worker_timeout_must_be_positive_and_finite() -> None:
    for value in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="worker_timeout_seconds"):
            process_vec.drive_process_vec(
                [{}],
                [_match()],
                _UnusedPolicy(),
                max_frames=8,
                worker_timeout_seconds=value,
            )


def test_policy_fault_capsule_round_trips_host_snapshot(tmp_path) -> None:
    class Policy(_UnusedPolicy):
        def fault_snapshot(self):
            return {"ctx_pad": [3], "rng_counters": [[1, 0, "buttons", 4]]}, {
                "floats": np.arange(6, dtype=np.float32).reshape(1, 1, 6),
                "cats": np.zeros((0, 1, 6), dtype=np.int64),
            }

    rows = {
        process_vec.Slot(0, 1): [
            ObservationRow(frame_id=12, flat={}, action=np.zeros(14, dtype=np.float32), reset=True)
        ]
    }
    try:
        raise RuntimeError("kernel failed")
    except RuntimeError:
        path = process_vec._write_policy_fault_capsule(tmp_path, Policy(), rows, plan_call=7)
    assert path is not None and path.is_file()
    payload = __import__("json").loads(path.read_text())
    assert payload["plan_call"] == 7
    assert payload["requests"][0]["frame_ids"] == [12]
    assert "kernel failed" in payload["exception"]
    with np.load(path.with_suffix(".npz")) as arrays:
        np.testing.assert_array_equal(arrays["floats"], np.arange(6, dtype=np.float32).reshape(1, 1, 6))
