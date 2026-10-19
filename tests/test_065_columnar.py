"""Packed frame storage, queue and checkpoint identity contracts."""

import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch

from hal.wire import ACTION_DIM

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "hal_o65_columnar_capture", os.environ.get("HAL_O65_CAPTURE_SOURCE", str(ROOT / "experiments/065_selfplay_ppo.py"))
)
assert _SPEC is not None and _SPEC.loader is not None
SOURCE_SHA256 = hashlib.sha256(Path(str(_SPEC.origin)).read_bytes()).hexdigest()
exp = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = exp
_SPEC.loader.exec_module(exp)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_windows_gather_ragged_histories_and_tails(device: str) -> None:
    import numpy as np

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    store = exp.FrameStore(8, 3, torch.device(device))
    generator = torch.Generator().manual_seed(65)
    store.records.copy_(torch.randint(0, 256, store.records.shape, dtype=torch.uint8, generator=generator).to(device))
    store.actions.copy_(torch.rand(store.actions.shape, generator=generator).to(device))
    table = exp.DenseRollout(64, store)
    meta = table.integers[:3]
    meta[:, exp.LANE] = torch.tensor([2, 0, 1], device=device)
    meta[:, exp.START] = torch.tensor([3, 0, 5], device=device)
    meta[:, exp.LENGTH] = torch.tensor([4, 2, 3], device=device)
    meta[:, exp.SIDE] = torch.tensor([1, 0, 1], device=device)
    meta[:, exp.IDENTITY] = torch.tensor([8, 9, 10], device=device)
    ids = torch.tensor([2, 0, 1], device=device)
    rows = exp.TickInputs(table, ids)
    for tail in (None, 2):
        actual = rows.packed(5, tail)
        expected_records = torch.zeros(3, 5, store.records.shape[-1], device=device, dtype=torch.uint8)
        expected_actions = torch.zeros(3, 5, 2, ACTION_DIM, device=device)
        lengths = []
        for index, row in enumerate(meta[ids]):
            length = int(row[exp.LENGTH]) if tail is None else min(tail, int(row[exp.LENGTH]))
            start = int(row[exp.START] + row[exp.LENGTH] - length)
            lane = int(row[exp.LANE])
            expected_records[index, :length] = store.records[start : start + length, lane]
            expected_actions[index, :length] = store.actions[start : start + length, lane]
            lengths.append(length)
        torch.testing.assert_close(actual[0], expected_records, rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected_actions, rtol=0, atol=0)
        np.testing.assert_array_equal(actual[2].cpu(), [1, 1, 0])
        np.testing.assert_array_equal(actual[3].cpu(), [10, 8, 9])
        np.testing.assert_array_equal(actual[4].cpu(), lengths)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_native_scatter_uses_batched_strided_coordinates(device: str) -> None:
    import numpy as np

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    raw = np.zeros(3, dtype=exp.native.FRAME_DTYPE)
    raw["frame_id"] = [-123, 1, 257]
    raw["applied_action"][:, 0, 0] = [0.25, -0.5, 0.75]
    raw["reward"] = [[1.25, -2.5], [2.75, -3.5], [4.25, -6.0]]
    raw["terminated"] = [False, True, False]
    raw["truncated"] = [True, False, False]
    columns = raw["columns"].copy()
    columns.flags.writeable = False
    batch = exp.ShardFrames(
        torch.from_numpy(raw.view(np.uint8).reshape(3, -1).copy()).to(device),
        torch.from_numpy(raw["applied_action"].copy()).to(device),
        columns,
        raw["reward"].copy(),
        raw["terminated"].copy(),
        raw["truncated"].copy(),
        raw["applied_action"].copy(),
    )
    store = exp.FrameStore(5, 3, torch.device(device))
    coordinates = torch.tensor([[1, 2, 1], [4, 0, 2], [2, 1, 0]], device=device)
    positions, lanes, sources = coordinates.unbind(1)
    actual_rewards = store.write(positions, lanes, batch, sources)
    for position, lane, source in ((1, 2, 1), (4, 0, 2), (2, 1, 0)):
        torch.testing.assert_close(
            store.records[position, lane].cpu(),
            torch.from_numpy(raw[source : source + 1].view(np.uint8)),
            rtol=0,
            atol=0,
        )
        np.testing.assert_array_equal(store.actions[position, lane].cpu(), raw["applied_action"][source])
        np.testing.assert_array_equal(store.rewards[position, lane].cpu(), raw["reward"][source])
        np.testing.assert_array_equal(
            store.flags[position, lane].cpu(), [raw["terminated"][source], raw["truncated"][source]]
        )
    np.testing.assert_array_equal(actual_rewards.cpu(), raw["reward"][[1, 2, 0]])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_action_ring_wrap_and_cadence(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    ring = exp.ActionRing(1, 6, torch.device(device))
    ring.reset(0, (2, 2))
    meta = torch.zeros(2, 18, device=device, dtype=torch.long)
    meta[:, exp.SIDE] = torch.tensor([0, 1], device=device)
    meta[:, exp.REPLAN] = torch.tensor([4, 2], device=device)
    actions = torch.arange(32, device=device).reshape(2, 4, 4)
    controllers = torch.zeros(2, 4, ACTION_DIM, device=device)
    ring.commit(meta, torch.ones(2, device=device, dtype=torch.bool), actions, controllers, count=4)
    ring.head += 2
    ring.length_device -= 2
    meta[:, exp.REPLAN] = 2
    ring.commit(meta, torch.ones(2, device=device, dtype=torch.bool), actions + 100, controllers, count=2)
    left = torch.tensor([0], device=device)
    expected = torch.cat((actions[0], actions[0, :2] + 100))
    torch.testing.assert_close(ring.read(left, left, 6)[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(ring.length_device.cpu(), torch.tensor([[6, 4]]))


def test_pool_rejects_old_formats_and_owns_cpu_queue_snapshots() -> None:
    from test_065_loop import trainer

    run = trainer(groups=1, anchor_rate=0, development=True)
    run.run_update()
    saved = run.engine.snapshot()
    with pytest.raises(ValueError, match="snapshot identity"):
        run.engine.restore_pool(dataclasses.replace(saved, format="hal.o65.rollout_pool.v2"))
    queue = saved.groups[0].queue.clone()
    run.run_update()
    torch.testing.assert_close(saved.groups[0].queue, queue, rtol=0, atol=0)
    run.engine.restore_pool(saved)


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_old_training_checkpoint_is_rejected_before_loading(version: int, tmp_path: Path) -> None:
    from test_065_loop import PROVENANCE
    from test_065_loop import exp as loop_exp
    from test_065_loop import trainer

    run = trainer(groups=1, anchor_rate=0)
    run.run_update()
    path = loop_exp.save_checkpoint(run, tmp_path, PROVENANCE)
    manifest = json.loads(path.read_text())
    manifest["format"] = f"hal.o65.training.v{version}"
    manifest["version"] = version
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsupported checkpoint identity"):
        loop_exp.restore_checkpoint(run, path, PROVENANCE)


def test_native_frame_batch_rejects_nonfinite_rewards_and_conflicting_end_flags() -> None:
    import numpy as np

    columns = np.zeros((1, len(exp.native.OBSERVATION_FIELDS)), dtype=np.uint32)
    columns.flags.writeable = False
    frame = exp.ShardFrames(
        torch.zeros(1, exp.native.FRAME_DTYPE.itemsize, dtype=torch.uint8),
        torch.zeros(1, 2, ACTION_DIM),
        columns,
        np.zeros((1, 2), dtype=np.float32),
        np.zeros(1, dtype=np.bool_),
        np.zeros(1, dtype=np.bool_),
        np.zeros((1, 2, ACTION_DIM), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="finite"):
        dataclasses.replace(frame, rewards=np.array([[0, np.nan]], dtype=np.float32))
    with pytest.raises(ValueError, match="time-out"):
        dataclasses.replace(frame, terminal=np.ones(1, dtype=np.bool_), timeout=np.ones(1, dtype=np.bool_))
    with pytest.raises(ValueError, match="schema"):
        dataclasses.replace(frame, rewards=np.zeros((1, 2), dtype=np.float64))


def test_pool_rejects_changed_observation_identity() -> None:
    from test_065_loop import trainer

    run = trainer(groups=1)
    saved = run.engine.snapshot()
    with pytest.raises(ValueError, match="schema mismatch"):
        run.engine.restore_pool(
            dataclasses.replace(saved, groups=(dataclasses.replace(saved.groups[0], fields=("wrong",)),))
        )
