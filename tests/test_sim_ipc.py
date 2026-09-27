import dataclasses
import uuid
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pytest

from hal.eval.harness import resolve_parallelism
from hal.sim import ipc
from hal.sim.ipc import CONTROL_SIZE
from hal.sim.ipc import LIVE_COLUMN_DTYPES
from hal.sim.ipc import LIVE_FLOAT_COLUMNS
from hal.sim.ipc import LIVE_INT_COLUMNS
from hal.sim.ipc import ArenaSpec
from hal.sim.ipc import ControlMessage
from hal.sim.ipc import MessageType
from hal.sim.ipc import ResultArena
from hal.sim.ipc import ResultSpec
from hal.sim.ipc import RolloutArena
from hal.sim.ipc import live_layout_hash
from hal.sim.ipc import result_shm_name
from hal.sim.rollout import PolicyRuntimeSpec
from hal.sim.rollout import covering_power_of_two
from hal.sim.rollout import nearest_power_of_two


def test_nearest_power_of_two_selects_the_larger_power_on_a_tie() -> None:
    assert [nearest_power_of_two(n) for n in (1, 2, 3, 6, 12, 20, 24, 32, 96)] == [
        1,
        2,
        4,
        8,
        16,
        16,
        32,
        32,
        128,
    ]
    assert covering_power_of_two(96) == 128
    with pytest.raises(ValueError):
        nearest_power_of_two(0)


def test_eval_parallelism_uses_affinity_bucket_and_validates_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hal.eval.harness.usable_cpus", lambda: 12)
    assert resolve_parallelism(96, None) == 16
    assert resolve_parallelism(7, None) == 7
    assert resolve_parallelism(96, 32) == 32
    assert resolve_parallelism(96, 30) == 30
    with pytest.raises(ValueError, match=">= 1"):
        resolve_parallelism(96, 0)


def test_policy_runtime_spec_validates_generic_schedule() -> None:
    spec = PolicyRuntimeSpec(
        context_frames=128,
        prediction_frames=20,
        execution_stride=6,
        committed_frames=3,
        action_dim=14,
    )
    assert spec.raw_ring_capacity == 128
    with pytest.raises(ValueError, match="committed_frames"):
        dataclasses.replace(spec, committed_frames=15)
    with pytest.raises(ValueError, match="execution_stride"):
        dataclasses.replace(spec, execution_stride=21)


def test_control_message_has_a_fixed_versioned_wire_format() -> None:
    message = ControlMessage(
        message_type=MessageType.PLAN_REQUEST,
        worker_id=7,
        flags=3,
        task_generation=11,
        task_id=13,
        sequence=17,
        auxiliary_sequence=19,
        count=5,
        plan_slot=1,
        port_or_slot=2,
        status_code=23,
    )
    payload = message.pack()
    assert len(payload) == CONTROL_SIZE
    assert ControlMessage.unpack(payload) == message
    with pytest.raises(ValueError, match="64 bytes"):
        ControlMessage.unpack(payload[:-1])
    bad = bytearray(payload)
    bad[-1] = 1
    with pytest.raises(ValueError, match="reserved"):
        ControlMessage.unpack(bad)


def test_live_layout_is_numeric_and_stable() -> None:
    assert LIVE_COLUMN_DTYPES
    assert set(LIVE_FLOAT_COLUMNS).isdisjoint(LIVE_INT_COLUMNS)
    assert set(LIVE_FLOAT_COLUMNS) | set(LIVE_INT_COLUMNS) == set(LIVE_COLUMN_DTYPES)
    assert all(dtype.kind in ("f", "i", "u") for dtype in LIVE_COLUMN_DTYPES.values())
    assert len(live_layout_hash()) == 64


@pytest.mark.parametrize(
    ("shape", "expected_bytes"),
    (((1, 256, 4, 14), 156352), ((2, 8, 6, 3), 9408), ((3, 257, 8, 14), 472448)),
)
def test_controller_arena_retains_existing_byte_layout(shape: tuple[int, int, int, int], expected_bytes: int) -> None:
    assert RolloutArena.required_bytes(ArenaSpec(*shape)) == expected_bytes


@pytest.mark.parametrize("arena_type", (RolloutArena, ResultArena))
def test_partial_arena_creation_closes_and_unlinks(
    monkeypatch: pytest.MonkeyPatch, arena_type: type[RolloutArena] | type[ResultArena]
) -> None:
    opened: list[SharedMemory] = []
    view_count = 0
    original_view = arena_type._view

    def record_memory(
        name: str | None = None, create: bool = False, size: int = 0, *, track: bool = True
    ) -> SharedMemory:
        shm = SharedMemory(name=name, create=create, size=size, track=track)
        opened.append(shm)
        return shm

    def fail_second_view(
        self: RolloutArena | ResultArena, at: int, shape: tuple[int, ...], dtype: np.dtype
    ) -> tuple[np.ndarray, int]:
        nonlocal view_count
        view_count += 1
        if view_count == 2:
            raise MemoryError("injected view allocation failure")
        return original_view(self, at, shape, dtype)

    monkeypatch.setattr(ipc, "SharedMemory", record_memory)
    monkeypatch.setattr(arena_type, "_view", fail_second_view)
    with pytest.raises(MemoryError, match="injected"):
        if arena_type is RolloutArena:
            RolloutArena.create(ArenaSpec(1, 8, 4, 14))
        else:
            ResultArena.create(f"hal-test-{uuid.uuid4().hex}", ResultSpec(8, 1, 2))
    assert len(opened) == 1
    assert opened[0].buf is None
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=opened[0].name, create=False, track=False)


@pytest.mark.parametrize("arena_type", (RolloutArena, ResultArena))
def test_partial_arena_attachment_closes_handle_and_preserves_owner(
    monkeypatch: pytest.MonkeyPatch, arena_type: type[RolloutArena] | type[ResultArena]
) -> None:
    if arena_type is RolloutArena:
        owner = RolloutArena.create(ArenaSpec(1, 8, 4, 14))
    else:
        owner = ResultArena.create(f"hal-test-{uuid.uuid4().hex}", ResultSpec(8, 1, 2))
    opened: list[SharedMemory] = []
    original_view = arena_type._view
    view_count = 0

    def record_memory(
        name: str | None = None, create: bool = False, size: int = 0, *, track: bool = True
    ) -> SharedMemory:
        shm = SharedMemory(name=name, create=create, size=size, track=track)
        opened.append(shm)
        return shm

    def fail_second_view(
        self: RolloutArena | ResultArena, at: int, shape: tuple[int, ...], dtype: np.dtype
    ) -> tuple[np.ndarray, int]:
        nonlocal view_count
        view_count += 1
        if view_count == 2:
            raise MemoryError("injected view allocation failure")
        return original_view(self, at, shape, dtype)

    monkeypatch.setattr(ipc, "SharedMemory", record_memory)
    monkeypatch.setattr(arena_type, "_view", fail_second_view)
    try:
        with pytest.raises(MemoryError, match="injected"):
            if isinstance(owner, RolloutArena):
                RolloutArena.attach(owner.descriptor)
            else:
                ResultArena.attach(owner._shm.name, owner.spec)
        assert len(opened) == 1
        assert opened[0].buf is None
        assert owner._shm.buf is not None
        attached = SharedMemory(name=owner._shm.name, create=False, track=False)
        attached.close()
    finally:
        owner.close()
        owner._shm.unlink()


def test_shared_arena_round_trips_observation_and_plan_without_pickle() -> None:
    spec = ArenaSpec(workers=2, ring_capacity=8, prediction_frames=6, action_dim=3)
    arena = RolloutArena.create(spec)
    attached = RolloutArena.attach(arena.descriptor)
    try:
        flat = {
            name: float(i) if dtype.kind == "f" else i for i, (name, dtype) in enumerate(LIVE_COLUMN_DTYPES.items())
        }
        action = np.array([0.25, -0.5, 1.0], dtype=np.float32)
        attached.write_observation(1, 10, 123, flat, action, reset=True)

        view, got_action, reset = arena.observation(1, 10)
        assert reset
        assert int(arena.obs_frame_id[1, 10 % spec.ring_capacity]) == 123
        np.testing.assert_array_equal(got_action, action)
        for name, expected in flat.items():
            assert view[name] == expected

        plan = np.arange(18, dtype=np.float32).reshape(6, 3)
        arena.plan_actions[1, 1] = plan
        np.testing.assert_array_equal(attached.plan_actions[1, 1], plan)

        # A later sequence at the same physical index makes the old row invalid.
        attached.write_observation(1, 18, 124, flat, action, reset=False)
        with pytest.raises(ValueError, match="overwritten or torn"):
            arena.observation(1, 10)
    finally:
        attached.close()
        arena.close()
        arena.unlink()


def test_exact_result_slab_round_trips_and_parent_unlinks() -> None:
    spec = ResultSpec(frames=5, segments=2, ports=2)
    name = result_shm_name("unit_test_arena", 3)
    writer = ResultArena.create(name, spec)
    writer.frame_id[:] = np.arange(5)
    writer.random_seed[:] = np.arange(5) + 10
    writer.segment_start[:] = (0, 2)
    writer.segment_length[:] = (2, 3)
    writer.post[:] = np.arange(writer.post.size).reshape(writer.post.shape)
    reader = ResultArena.attach(name, spec)
    try:
        np.testing.assert_array_equal(reader.frame_id, np.arange(5))
        np.testing.assert_array_equal(reader.segment_length, (2, 3))
        np.testing.assert_array_equal(reader.post, writer.post)
    finally:
        reader.close()
        writer.close()
