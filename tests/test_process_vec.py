"""Failure handling for the spawned closed-loop inference broker."""

import multiprocessing as mp
import os
import time
from collections.abc import Mapping
from collections.abc import Sequence
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import melee
import numpy as np
import pytest

import hal.sim.process_vec as process_vec
from hal.eval.harness import SessionConfig
from hal.eval.harness import _session_kwargs
from hal.paths import EMULATOR_PATH
from hal.paths import ISO_PATH
from hal.sim.ipc import LIVE_FLOAT_COLUMNS
from hal.sim.ipc import LIVE_INT_COLUMNS
from hal.sim.ipc import ControlMessage
from hal.sim.ipc import MessageType
from hal.sim.ipc import RolloutArena
from hal.sim.ipc import send_control
from hal.sim.process_vec import ProcessVecTelemetry
from hal.sim.rollout import ObservationRow
from hal.sim.rollout import PolicyRuntimeSpec
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.vec import Slot
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


class _NeutralPolicy:
    runtime_spec = _UnusedPolicy.runtime_spec

    def plan_rows(self, rows: Mapping[Slot, Sequence[ObservationRow]]) -> dict[Slot, np.ndarray]:
        shape = (self.runtime_spec.prediction_frames, self.runtime_spec.action_dim)
        return {slot: np.zeros(shape, dtype=np.float32) for slot in rows}


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


def _ready_worker(
    worker_id: int,
    connection: Connection,
    arena_descriptor: Any,
    session_kwargs: dict[str, Any],
    _matchup: Matchup,
    model_ports: tuple[int, ...],
    arena_slots: tuple[int, ...],
    runtime: PolicyRuntimeSpec,
    _max_frames: int,
    _instant_restart: bool,
) -> None:
    starts = session_kwargs["starts"]
    starts.put((worker_id, time.monotonic()))
    time.sleep(session_kwargs["ready_delay"])
    arena = RolloutArena.attach(arena_descriptor)
    try:
        flat = {name: 0.0 for name in LIVE_FLOAT_COLUMNS}
        flat.update({name: 0 for name in LIVE_INT_COLUMNS})
        for port, arena_slot in zip(model_ports, arena_slots, strict=True):
            arena.write_observation(
                arena_slot,
                1,
                0,
                flat,
                np.zeros(runtime.action_dim, dtype=np.float32),
                reset=True,
            )
            send_control(
                connection,
                ControlMessage(
                    message_type=MessageType.PLAN_REQUEST,
                    worker_id=worker_id,
                    task_id=arena_slot,
                    sequence=1,
                    count=1,
                    port_or_slot=port,
                ),
            )
        while True:
            time.sleep(10)
    finally:
        arena.close()


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
        startup_timeout_seconds=0.1,
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


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_startup_parallelism_must_be_a_positive_integer(value: object) -> None:
    with pytest.raises(ValueError, match="startup_parallelism"):
        process_vec.drive_process_vec(
            [{}],
            [_match()],
            _UnusedPolicy(),
            max_frames=8,
            startup_parallelism=value,
        )


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_startup_timeout_must_be_positive_and_finite(value: float) -> None:
    with pytest.raises(ValueError, match="startup_timeout_seconds"):
        process_vec.drive_process_vec(
            [{}],
            [_match()],
            _UnusedPolicy(),
            max_frames=8,
            startup_timeout_seconds=value,
        )


def test_startup_admission_waits_for_ready_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    context = mp.get_context("fork")
    _use_forked_fake_worker(monkeypatch, _ready_worker)
    starts = context.Queue()
    kwargs = [{"starts": starts, "ready_delay": 0.08} for _ in range(5)]

    with pytest.raises(process_vec.PolicyExecutionError):
        process_vec.drive_process_vec(
            kwargs,
            [_match() for _ in kwargs],
            _UnusedPolicy(),
            max_frames=8,
            startup_parallelism=2,
            startup_timeout_seconds=1.0,
        )

    launched = dict(starts.get(timeout=1.0) for _ in kwargs)
    first_batch = min(launched[0], launched[1])
    second_batch = min(launched[2], launched[3])
    assert second_batch - first_batch >= 0.05
    assert launched[4] - second_batch >= 0.05


@pytest.mark.integration
def test_bounded_startup_with_real_dolphins() -> None:
    iso_path = Path(ISO_PATH)
    dolphin_path = Path(EMULATOR_PATH)
    if not iso_path.is_file():
        pytest.skip(f"ISO missing at {iso_path}")
    if not dolphin_path.is_file():
        pytest.skip(f"Dolphin missing at {dolphin_path}")
    matches = [_match() for _ in range(4)]
    cfg = SessionConfig(
        iso_path=iso_path,
        dolphin_path=dolphin_path,
        use_exi_inputs=True,
        enable_ffw=True,
        emulation_speed=0.0,
        blocking_input=True,
        step_timeout_seconds=30.0,
        tmp_home_directory=True,
    )
    kwargs = [_session_kwargs(cfg, slippi_port=52000 + worker, replay_dir=None) for worker in range(len(matches))]

    results = process_vec.drive_process_vec(
        kwargs,
        matches,
        _NeutralPolicy(),
        max_frames=120,
        startup_parallelism=2,
        startup_timeout_seconds=120.0,
    )

    assert all(len(boots) == 1 for boots in results)
    assert all(len(boots[0]) == 120 for boots in results)


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


def test_worker_cohort_assignment_is_balanced_and_keeps_ports_together() -> None:
    ids = process_vec._worker_cohort_ids(10, 4)
    sizes = [ids.count(cohort) for cohort in range(4)]
    assert max(sizes) - min(sizes) == 1
    slots = {Slot(worker, port) for worker in range(10) for port in (1, 2)}
    pending = set(slots)
    first = process_vec._next_ready_cohort(slots, pending, ids)
    assert first
    assert {ids[slot.match] for slot in first} == {0}
    assert all(Slot(slot.match, 3 - slot.port) in first for slot in first)


def test_one_cohort_preserves_the_all_live_slot_gate() -> None:
    slots = {Slot(worker, port) for worker in range(4) for port in (1, 2)}
    ids = process_vec._worker_cohort_ids(4, 1)
    ordered = sorted(slots, key=lambda slot: (slot.match, slot.port))
    assert process_vec._next_ready_cohort(slots, set(ordered[:-1]), ids) == []
    assert process_vec._next_ready_cohort(slots, set(slots), ids) == ordered


def test_cohort_latency_starts_at_first_worker_preprocessing() -> None:
    acknowledged = [(0, 11, 1, 300), (1, 12, 1, 100), (2, 13, 1, 200)]
    assert process_vec._cohort_latency_start_ns(acknowledged) == 100


def test_empty_cohort_latency_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        process_vec._cohort_latency_start_ns([])


@pytest.mark.parametrize("cohort_count", [1, 2, 3, 4])
def test_cohorts_preserve_slot_keyed_outputs_with_mixed_resets(cohort_count: int) -> None:
    slots = [Slot(worker, port) for worker in range(8) for port in (1, 2)]
    cohort_ids = process_vec._worker_cohort_ids(8, cohort_count)

    class SlotKeyedPolicy:
        def __init__(self, seed: int) -> None:
            self.seed = seed
            self.streams = {}

        def plan_rows(self, rows):
            out = {}
            for slot, slot_rows in rows.items():
                slot_seed = self.seed + slot.match * 8 + slot.port
                if slot not in self.streams or any(row.reset for row in slot_rows):
                    self.streams[slot] = np.random.default_rng(slot_seed)
                noise = self.streams[slot].integers(0, 2**31, dtype=np.int64)
                out[slot] = int(noise) + sum(row.frame_id for row in slot_rows)
            return out

    serial = SlotKeyedPolicy(91)
    cohort = SlotKeyedPolicy(91)
    for step in range(6):
        rows = {
            slot: [
                ObservationRow(
                    frame_id=step - 20 if step == 3 and slot.match % 3 == 0 else step,
                    flat={"worker": slot.match},
                    action=np.full(14, step, dtype=np.float32),
                    reset=step == 3 and slot.match % 3 == 0,
                )
            ]
            for slot in slots
        }
        expected = serial.plan_rows(rows)
        actual = {}
        for cohort_id in range(cohort_count):
            group = {slot: rows[slot] for slot in slots if cohort_ids[slot.match] == cohort_id}
            actual.update(cohort.plan_rows(group))
        assert actual == expected


@pytest.mark.parametrize("n_workers,cohort_count", [(0, 1), (4, 0), (4, 5), (4, True)])
def test_invalid_worker_cohorts_are_rejected(n_workers: int, cohort_count: int) -> None:
    with pytest.raises(ValueError, match="cohort_count|n_workers"):
        process_vec._worker_cohort_ids(n_workers, cohort_count)
