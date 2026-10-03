"""Native evaluator frame timing and trace alignment."""

from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from hal.data.schema import MDS_PER_FRAME_DTYPES
from hal.eval.native_rollout import first_difference
from hal.eval.native_rollout import replay_inputs
from hal.eval.native_rollout import run_episode
from hal.inference.action_sequence_artifact import REQUIRED_OBSERVATION_FIELDS
from hal.sim.native import FRAME_DTYPE
from hal.sim.native import OBSERVATION_FIELDS
from hal.sim.native import NativeFrameBatch
from hal.sim.native import NativeMatch
from hal.sim.native import NativePlayer
from hal.sim.rollout import ObservationRow
from hal.sim.rollout import PolicyRuntimeSpec
from hal.sim.rollout import Slot


class FakeSimulator:
    def __init__(self, *, terminate_at: int | None = None, packed: bool = False) -> None:
        self.frame = -123
        self.terminate_at = terminate_at
        self.actions: list[np.ndarray] = []
        self._records = np.zeros(1, dtype=FRAME_DTYPE) if packed else None
        if self._records is None:
            self._columns = {
                name: np.zeros(1, dtype=MDS_PER_FRAME_DTYPES[name]) for name in REQUIRED_OBSERVATION_FIELDS
            }
            self._submitted = np.zeros((1, 2, 14), dtype=np.float32)
            self._wire = np.zeros((1, 2, 7), dtype=np.int16)
        else:
            words = self._records["columns"]
            self._columns = {
                name: words[:, index].view(np.dtype(MDS_PER_FRAME_DTYPES[name]))
                for index, name in enumerate(OBSERVATION_FIELDS)
            }
            self._submitted = self._records["applied_action"]
            self._wire = self._records["wire_inputs"]

    def _result(self, *, reset: bool) -> NativeFrameBatch:
        for values in self._columns.values():
            values[0] = self.frame
        if self._records is not None:
            self._records["frame_id"][0] = self.frame
            self._records["reward"][0] = (self.frame, -self.frame)
            self._records["terminated"][0] = not reset and self.frame == self.terminate_at
            self._records["truncated"][0] = False
            self._records["reset"][0] = reset
        return NativeFrameBatch(
            columns=self._columns,
            frame_id=self._records["frame_id"]
            if self._records is not None
            else np.array([self.frame], dtype=np.int32),
            applied_action=self._submitted,
            wire_inputs=self._wire,
            reward=self._records["reward"]
            if self._records is not None
            else np.array([[self.frame, -self.frame]], dtype=np.float32),
            terminated=self._records["terminated"]
            if self._records is not None
            else np.array([not reset and self.frame == self.terminate_at]),
            truncated=self._records["truncated"] if self._records is not None else np.array([False]),
            reset=self._records["reset"] if self._records is not None else np.array([reset]),
            records=self._records,
        )

    def reset(self, configs: Sequence[NativeMatch], mask: NDArray[np.bool_]) -> NativeFrameBatch:
        assert len(configs) == 1 and mask.tolist() == [True]
        self.frame = -123
        self.actions.clear()
        self._submitted.fill(0)
        self._wire.fill(0)
        return self._result(reset=True)

    def step(self, actions: NDArray[np.float32], mask: NDArray[np.bool_]) -> NativeFrameBatch:
        assert actions.shape == (1, 2, 14) and mask.tolist() == [True]
        self.frame += 1
        self.actions.append(actions.copy())
        self._submitted[:] = actions
        self._wire[:, :, :4] = np.rint(actions[:, :, :4] * 80).astype(np.int16)
        return self._result(reset=False)


class FakePolicy:
    runtime_spec = PolicyRuntimeSpec(16, 4, 2, 2, 14, observed_actions=False)

    def __init__(self) -> None:
        self.ingested: list[list[int]] = []

    def plan_rows(self, rows: Mapping[Slot, Sequence[ObservationRow]]) -> Mapping[Slot, np.ndarray]:
        ids = [row.frame_id for row in rows[Slot(0, 1)]]
        self.ingested.append(ids)
        result = {}
        for slot in rows:
            plan = np.zeros((4, 14), dtype=np.float32)
            plan[:, 0] = (slot.port * 10 + len(self.ingested) + np.arange(4)) / 100
            result[slot] = plan
        return result


MATCH = NativeMatch(25, (NativePlayer(1, 2), NativePlayer(2, 20)), 0)


def test_two_step_cadence_and_borrowed_arrays() -> None:
    simulator = FakeSimulator(packed=True)
    policy = FakePolicy()
    trace = run_episode(simulator, policy, MATCH, max_frames=5)
    assert policy.ingested == [[-123], [-122, -121], [-120, -119]]
    np.testing.assert_array_equal(trace.arrays["frame_id"], np.arange(-123, -117))
    assert trace.arrays["submitted"].shape == (6, 2, 14)
    assert trace.arrays["wire_inputs"].shape == (6, 2, 7)
    np.testing.assert_array_equal(trace.arrays["submitted"][0], 0)
    assert trace.arrays["reset"].tolist() == [True, False, False, False, False, False]
    assert trace.stop_reason == "frame_budget"
    assert len([name for name in trace.arrays if name in REQUIRED_OBSERVATION_FIELDS]) == 71
    assert trace.arrays["p1_position_x"][0] == -123
    assert trace.arrays["p1_position_x"][-1] == -118
    assert first_difference(trace, trace) is None
    difference, _seconds = replay_inputs(simulator, MATCH, trace)
    assert difference is None


def test_packed_trace_matches_typed_column_trace_byte_for_byte() -> None:
    fallback = run_episode(FakeSimulator(), FakePolicy(), MATCH, max_frames=5)
    packed_simulator = FakeSimulator(packed=True)
    packed = run_episode(packed_simulator, FakePolicy(), MATCH, max_frames=5)
    assert len(packed.arrays) == 78
    assert first_difference(fallback, packed) is None
    for name in fallback.arrays:
        assert packed.arrays[name].dtype == fallback.arrays[name].dtype
        assert packed.arrays[name].shape == fallback.arrays[name].shape
    packed_simulator.reset((MATCH,), np.array([True]))
    assert first_difference(fallback, packed) is None


def test_packed_trace_npz_preserves_all_array_bytes(tmp_path: Path) -> None:
    trace = run_episode(FakeSimulator(packed=True), FakePolicy(), MATCH, max_frames=2)
    path = tmp_path / "trace.npz"
    trace.save(path)
    with np.load(path) as saved:
        assert set(saved.files) == set(trace.arrays)
        for name, values in trace.arrays.items():
            assert saved[name].dtype == values.dtype
            assert saved[name].shape == values.shape
            assert saved[name].tobytes() == values.tobytes()


def test_trace_preserves_float_bits_in_both_storage_paths() -> None:
    class PayloadSimulator(FakeSimulator):
        def _result(self, *, reset: bool) -> NativeFrameBatch:
            frame = super()._result(reset=reset)
            bits = np.uint32(0x80000000 if reset else 0x7FC00001)
            frame.columns["p1_position_x"][0] = bits.view(np.float32)
            return frame

    for packed in (False, True):
        trace = run_episode(PayloadSimulator(packed=packed), FakePolicy(), MATCH, max_frames=2)
        assert trace.arrays["p1_position_x"].view(np.uint32).tolist() == [
            0x80000000,
            0x7FC00001,
            0x7FC00001,
        ]


def test_native_terminal_stops_mid_chunk() -> None:
    simulator = FakeSimulator(terminate_at=-120, packed=True)
    trace = run_episode(simulator, FakePolicy(), MATCH, max_frames=10)
    assert trace.steps == 3
    assert trace.stop_reason == "native_terminated"
    assert trace.arrays["terminated"].tolist() == [False, False, False, True]


def test_exact_difference_reports_wire_input_row() -> None:
    simulator = FakeSimulator()
    reference = run_episode(simulator, FakePolicy(), MATCH, max_frames=2)
    modified = {name: values.copy() for name, values in reference.arrays.items()}
    modified["wire_inputs"][1, 0, 0] += 1
    from hal.eval.native_rollout import EpisodeResult

    repeat = EpisodeResult(modified, 0.0, 0.0, 0.0, reference.stop_reason)
    assert first_difference(reference, repeat) == ("wire_inputs", 1)


def test_byte_comparison_detects_signed_zero_and_nan_payload() -> None:
    from hal.eval.native_rollout import EpisodeResult

    positive = EpisodeResult({"field": np.array([0.0], dtype=np.float32)}, 0, 0, 0, "frame_budget")
    negative = EpisodeResult({"field": np.array([-0.0], dtype=np.float32)}, 0, 0, 0, "frame_budget")
    assert first_difference(positive, negative) == ("field", 0)
    nan_one = EpisodeResult({"field": np.array([0x7FC00001], dtype=np.uint32).view(np.float32)}, 0, 0, 0, "x")
    nan_two = EpisodeResult({"field": np.array([0x7FC00002], dtype=np.uint32).view(np.float32)}, 0, 0, 0, "x")
    assert first_difference(nan_one, nan_two) == ("field", 0)
