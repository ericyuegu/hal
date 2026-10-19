"""Native proxy parity fixture and bounded complete-update measurements.

Run this module under the shared GPU lock. Recording requires an explicit output;
the checked-in fixture is never updated by pytest.
"""

import argparse
import contextlib
import cProfile
import dataclasses
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from hal.wire import ACTION_DIM

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).with_name("fixtures") / "o65_columnar.json"
_SPEC = importlib.util.spec_from_file_location(
    "hal_o65_columnar_capture", os.environ.get("HAL_O65_CAPTURE_SOURCE", str(ROOT / "experiments/065_selfplay_ppo.py"))
)
assert _SPEC is not None and _SPEC.loader is not None
SOURCE_SHA256 = hashlib.sha256(Path(str(_SPEC.origin)).read_bytes()).hexdigest()
exp = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = exp
_SPEC.loader.exec_module(exp)


class CaptureShard(exp.NativeSimulatorShard):
    def __init__(
        self, data_dir: Path, match_ids: tuple[int, ...], *, cpu: int, device: torch.device, max_frame: int
    ) -> None:
        super().__init__(data_dir, match_ids, cpu=cpu, device=device, max_frame=max_frame)
        self.record_digest = hashlib.sha256()
        self.action_digest = hashlib.sha256()

    def _frames(self, frame: Any) -> Any:
        self.record_digest.update(frame.records.tobytes())
        self.action_digest.update(exp.native_controller_readback(frame.applied_action).tobytes())
        return super()._frames(frame)


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def fingerprints(run: Any, batch: Any) -> dict[str, str]:
    # Hash frames once per decision's last observation, in decision order. This
    # includes both sides and catches padding, stale storage, and reset readback.
    records = torch.stack([r.state.frames[-1].device_record for r in batch.rollout.records])
    actions = torch.stack([r.state.frames[-1].device_actions for r in batch.rollout.records])
    return {
        "records": tensor_hash(records),
        "processed_actions": tensor_hash(actions),
        "sampled_factors": tensor_hash(batch.decisions.sampled_indices_JF),
        "behavior_log_probs": tensor_hash(batch.decisions.behavior_log_probs_JF),
        "rewards": exp._state_hash([r.rewards for r in batch.rollout.records]),
        "batch": exp._batch_hash(batch),
        "advantages": tensor_hash(batch.advantages),
        "values": tensor_hash(batch.values),
        "old_log_probs": tensor_hash(batch.old.log_probs),
        "reference_log_probs": tensor_hash(batch.reference.log_probs),
        "actor": exp._state_hash(tuple(run.policy.actor_parameters())),
        "critic": exp._state_hash(tuple(run.policy.critic_parameters())),
    }


def capture(
    matches: int, output: Path, *, performance: bool, verify_resume: bool = False, warmup: bool = False
) -> dict[str, object]:
    if matches not in (16, 64):
        raise ValueError("local capture requires 16 or 64 matches")
    if performance and verify_resume:
        raise ValueError("resume verification requires a deterministic parity capture")
    if warmup and not performance:
        raise ValueError("warmup measurement requires performance mode")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(not performance)
    torch.manual_seed(65)
    checkpoint = exp.load_evaluation_checkpoint(ROOT / "runs/proxy/final.pt", device=torch.device("cuda"))
    checkpoint.model.requires_grad_(True)
    checkpoint.model.percentile_head.requires_grad_(False)
    schedule = exp.development_config(checkpoint.model.head_offsets)
    policy = exp.TransformerPolicy(checkpoint.model, schedule, checkpoint.statistics, compiled=True)
    sampling = exp.read_native_sampler(ROOT / "experiments/o65/pilot-sampler.json")
    training = exp.TrainConfig(critic_warmup_updates=10000 if warmup else 0, deterministic=not performance)
    source = Path(os.environ.get("HAL_MSL_SOURCE", "~/src/melee-sim-light")).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as resources:
        shards = []
        cpus = sorted(os.sched_getaffinity(0))[:4]
        for index, cpu in enumerate(cpus):
            ids = tuple(range(index * matches // 4, (index + 1) * matches // 4))
            shard = (exp.NativeSimulatorShard if performance else CaptureShard)(
                source / "data", ids, cpu=cpu, device=policy.device, max_frame=28800 if performance else 0
            )
            resources.callback(shard.close)
            shards.append(shard)
        engine = exp.RolloutEngine(
            (tuple(shards),),
            policy,
            exp.EpisodeSampler(sampling, schedule, exp.CounterRng(65)),
            {0: policy.frozen_copy()},
        )
        engine.initialize()
        run = exp.Trainer(engine, training)
        rows = []
        for _ in range(3 if performance else 2):
            policies = (policy, *engine.anchors.values())
            before = sum(sum(size * count for size, count in p.sample_group_counts.items()) for p in policies)
            batch, metrics = run.run_update()
            after = sum(sum(size * count for size, count in p.sample_group_counts.items()) for p in policies)
            metrics["systems/decode_calls_per_tick"] = (after - before) / (schedule.rollout_frames / 2)
            row = metrics if performance else fingerprints(run, batch)
            if not performance:
                row["all_frame_records"] = exp._state_hash([s.record_digest.hexdigest() for s in shards])
                row["all_processed_actions"] = exp._state_hash([s.action_digest.hexdigest() for s in shards])
            rows.append(row)
            if not performance:
                assert batch.rollout.resets > 0 and batch.rollout.timeouts > 0
            del batch
        if performance:
            profiler = cProfile.Profile()
            profiler.runcall(engine.collect, run.update)
            profiler.dump_stats(output / "rollout.prof")
        if verify_resume:
            manifests = checkpoint.provenance["data_manifests"]
            provenance = {
                "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
                "data_manifest_sha256": manifests["ranked-anonymized-1-policy-world-v8"],
                "simulator_sha256": shards[0].shard.library_sha256,
                "adapter_sha256": exp.native_bridge.library_sha256(),
            }
            exp.verify_checkpoint_resume(run, output, provenance)
        result = {
            "matches": matches,
            "performance": performance,
            "source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_file_sha256": SOURCE_SHA256,
            "configuration": {
                "training": dataclasses.asdict(training),
                "sampling": dataclasses.asdict(sampling),
                "schedule": dataclasses.asdict(schedule),
                "cpus": cpus,
                "groups": [[shard.match_ids for shard in shards]],
                "max_frame": shards[0].max_frame,
                "compiled": True,
            },
            "environment": exp._environment(),
            "checkpoint": checkpoint.checkpoint_sha256,
            "sampler": exp.sha256(ROOT / "experiments/o65/pilot-sampler.json"),
            "native_data": exp.sha256(source / "data/raw/manifest.json"),
            "simulator": shards[0].shard.library_sha256,
            "adapter": exp.native_bridge.library_sha256(),
            "rows": rows,
        }
        (output / "capture.json").write_text(json.dumps(result, indent=2, sort_keys=True))
        return result


@pytest.mark.integration
@pytest.mark.parametrize("matches", [16, 64])
def test_native_columnar_exact_parity(matches: int, tmp_path: Path) -> None:
    expected = json.loads(FIXTURE.read_text())[str(matches)]
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        actual = capture(matches, tmp_path, performance=False)
    finally:
        torch.use_deterministic_algorithms(previous)
    # Normalize tuples as JSON does for the persisted fixture.
    actual = json.loads(json.dumps(actual))
    for name in (
        "configuration",
        "environment",
        "checkpoint",
        "sampler",
        "native_data",
        "simulator",
        "adapter",
        "rows",
    ):
        assert actual[name] == expected[name], name


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_windows_gather_ragged_histories_and_tails(device: str) -> None:
    import numpy as np

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    store = exp.FrameStore(8, 3, torch.device(device), exp.native.OBSERVATION_FIELDS)
    generator = torch.Generator().manual_seed(65)
    store.records.copy_(torch.randint(0, 256, store.records.shape, dtype=torch.uint8, generator=generator).to(device))
    store.actions.copy_(torch.rand(store.actions.shape, generator=generator).to(device))
    table = exp.DecisionTable(3, store)
    table.count = 3
    table.meta[:, exp.LANE] = [2, 0, 1]
    table.meta[:, exp.START] = [3, 0, 5]
    table.meta[:, exp.LENGTH] = [4, 2, 3]
    table.meta[:, exp.SIDE] = [1, 0, 1]
    table.meta[:, exp.IDENTITY] = [8, 9, 10]
    table.integers.copy_(torch.from_numpy(table.meta).to(device))
    rows = table.rows([2, 0, 1])
    for tail in (None, 2):
        actual = rows.packed(5, tail)
        expected_records = torch.zeros(3, 5, store.records.shape[-1], device=device, dtype=torch.uint8)
        expected_actions = torch.zeros(3, 5, 2, ACTION_DIM, device=device)
        lengths = []
        for index, meta in enumerate(rows.meta):
            length = int(meta[exp.LENGTH]) if tail is None else min(tail, int(meta[exp.LENGTH]))
            start = int(meta[exp.START] + meta[exp.LENGTH] - length)
            lane = int(meta[exp.LANE])
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
    batch = exp.FrameBatch(
        torch.from_numpy(raw.view(np.uint8).reshape(3, -1).copy()).to(device),
        torch.from_numpy(raw["applied_action"].copy()).to(device),
        columns,
        raw["reward"].copy(),
        raw["terminated"].copy(),
        raw["truncated"].copy(),
        raw["applied_action"].copy(),
    )
    store = exp.FrameStore(5, 3, torch.device(device), exp.native.OBSERVATION_FIELDS)
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
    import numpy as np

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    ring = exp.ActionRing(1, 6, torch.device(device))
    ring.reset(0, (2, 2))
    lanes = np.array([0, 0], dtype=np.int64)
    sides = np.array([0, 1], dtype=np.int64)
    actions = torch.arange(32, device=device).reshape(2, 4, 4)
    ring.append(lanes, sides, actions, np.array([4, 2], dtype=np.int64))
    left = torch.tensor([0], device=device)
    for _ in range(2):
        ring.consume(np.array([0], dtype=np.int64))
    ring.append(lanes, sides, actions + 100, np.array([2, 2], dtype=np.int64))
    expected = torch.cat((actions[0], actions[0, :2] + 100))
    torch.testing.assert_close(ring.read(left, left, 6)[0], expected, rtol=0, atol=0)
    np.testing.assert_array_equal(ring.length, [[6, 4]])
    np.testing.assert_array_equal(ring.length_device.cpu(), ring.length)
    with pytest.raises(ValueError, match="capacity"):
        ring.append(lanes, sides, actions, np.array([2, 2], dtype=np.int64))
    for _ in range(4):
        ring.consume(np.array([0], dtype=np.int64))
    with pytest.raises(ValueError, match="uncommitted"):
        ring.consume(np.array([0], dtype=np.int64))


def test_pool_rejects_old_formats_and_owns_cpu_queue_snapshots() -> None:
    from test_065_loop import trainer

    run = trainer(groups=1, anchor_rate=0, development=True)
    run.run_update()
    saved = run.engine.snapshot()
    with pytest.raises(ValueError, match="snapshot identity"):
        run.engine.restore_pool(dataclasses.replace(saved, format="hal.o65.rollout_pool.v1"))
    queue = saved.groups[0].queue.clone()
    run.run_update()
    torch.testing.assert_close(saved.groups[0].queue, queue, rtol=0, atol=0)
    run.engine.restore_pool(saved)


def test_rollout_keeps_contexts_in_tensors_until_diagnostics_request_them() -> None:
    from test_065_loop import exp as loop_exp
    from test_065_loop import transformer_trainer

    run = transformer_trainer("cpu", development=True)
    rollout = run.engine.collect(0)
    assert isinstance(rollout.records, loop_exp.DecisionRecords)
    table = rollout.records.states.table
    store = table.store
    assert store._host is None and not store._views
    for host, device in (
        (table.duration, table.duration_device),
        (table.remaining, table.remaining_device),
        (table.terminal, table.terminal_device),
        (table.boundary, table.boundary_device),
        (table.following, table.following_device),
    ):
        torch.testing.assert_close(device[: table.count].cpu(), torch.from_numpy(host[: table.count]), rtol=0, atol=0)
    batch = run.prepare(rollout)
    run.fit(batch)
    assert store._host is None and not store._views


@pytest.mark.parametrize("version", [1, 2, 3])
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
    frame = exp.FrameBatch(
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


def test_explicit_packed_states_keep_their_device_readback() -> None:
    from test_065_loop import transformer_trainer

    run = transformer_trainer("cpu")
    state = run.engine._state(0, 0, 0, 0)
    frame = state.frames[0]
    readback = torch.zeros(2, ACTION_DIM)
    readback[0, 0] = 0.75
    readback[0, 6] = 1
    frame = dataclasses.replace(frame, device_actions=readback)
    state = dataclasses.replace(state, frames=(frame,))
    expected, expected_actions = run.policy._contexts((state,))
    actual, actual_actions = run.policy._contexts(run.policy._columnar_states((state,)))
    assert expected["ego_main_stick_x"].item() == 0.75
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    torch.testing.assert_close(actual_actions, expected_actions, rtol=0, atol=0)


def test_capture_rejects_resume_with_default_algorithms(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="deterministic parity"):
        capture(16, tmp_path, performance=True, verify_resume=True)


def test_capture_rejects_warmup_in_golden_parity_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires performance mode"):
        capture(16, tmp_path, performance=False, warmup=True)


def test_frame_store_rejects_a_changed_observation_schema() -> None:
    from test_065_loop import trainer

    run = trainer(groups=1, anchor_rate=0)
    history = run.engine.matches[0].history
    frame = dataclasses.replace(history[0], columns={"different_column": 0.0})
    with pytest.raises(ValueError, match="schema changed"):
        history.store.write_frame(0, 0, frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--performance", action="store_true")
    parser.add_argument("--verify-resume", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()
    capture(
        args.matches, args.output, performance=args.performance, verify_resume=args.verify_resume, warmup=args.warmup
    )
