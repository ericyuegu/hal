"""Two-process tests of the actual PPO update and distributed state boundary."""

import dataclasses
import json
import threading
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from hal.training import self_play_distributed as distributed
from hal.training.self_play_flops import phase_metrics


def _worker(rank: int, rendezvous: str, directory: str) -> None:
    # Import the existing deterministic simulator seam inside each fresh process.
    from test_065_loop import PROVENANCE
    from test_065_loop import FakeShard
    from test_065_loop import exp
    from test_065_loop import trainer

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        run = trainer(groups=1, length=67 if rank == 0 else 1, delay=8)
        engine = exp.RolloutEngine(
            (
                (
                    FakeShard((2 * rank,), length=67 if rank == 0 else 1),
                    FakeShard((2 * rank + 1,), length=67 if rank == 0 else 1),
                ),
            ),
            run.policy,
            run.engine.sampler,
            {0: run.reference},
        )
        engine.initialize()
        run = exp.Trainer(engine, dataclasses.replace(run.config, actor_microbatches=2))
        initial = tuple(p.detach().clone() for p in run.policy.model.parameters())
        frozen = tuple(p.detach().clone() for p in run.reference.model.parameters())
        for _ in range(2):
            batch, metrics = run.run_update()
            # Rank one has no executed decisions, but participates in every step.
            assert bool(batch.selected.any()) == (rank == 0)
            peers = distributed.gather_objects((run.actor_steps, run.scheduler.last_epoch, len(batch.steps)))
            assert len(set(peers)) == 1
            assert len(batch.steps) > 0
            distributed.check_parameters(tuple(run.policy.model.parameters()))
            assert metrics["data/decisions"] > len(batch.values)
        assert any(
            not torch.equal(before, after)
            for before, after in zip(initial, run.policy.model.parameters(), strict=True)
        )
        assert all(
            torch.equal(before, after) for before, after in zip(frozen, run.reference.model.parameters(), strict=True)
        )
        assert run.slot_updates[1] == 2
        for parameter, anchor in zip(
            run.policy.model.parameters(), run.engine.anchors[1].model.parameters(), strict=True
        ):
            assert torch.equal(parameter, anchor)
        run.engine.behavior_tracker.state[:, 0] = rank + 3
        behavior = run.engine.behavior_tracker.state.copy()
        checkpoint = exp.save_checkpoint(run, Path(directory) / "checkpoint", PROVENANCE)
        # Rank-local state must survive the distributed checkpoint.
        local_cursor = run.engine.sampler.cursor
        local_rng = torch.get_rng_state().clone()
        run.engine.sampler.cursor += rank + 1
        torch.manual_seed(777 + rank)
        run.engine.behavior_tracker.state[:, 0] = 0
        exp.restore_checkpoint(run, checkpoint, PROVENANCE)
        np.testing.assert_array_equal(run.engine.behavior_tracker.state, behavior)
        assert run.engine.sampler.cursor == local_cursor
        assert torch.equal(torch.get_rng_state(), local_rng)
        exp.check_resume_process(run, checkpoint, Path(directory) / "resume-controls", PROVENANCE, prepare=True)
        boundary = exp._resume_state_hashes(run)
        started, release = threading.Event(), threading.Event()
        write = exp.write_checkpoint_shards

        def delayed_write(snapshot: exp.CheckpointSnapshot, target: Path, local_rank: int) -> list[dict[str, str]]:
            assert threading.current_thread() is not threading.main_thread()
            started.set()
            assert release.wait(30)
            return write(snapshot, target, local_rank)

        # Training must advance while immutable checkpoint bytes await background I/O.
        with patch.object(exp, "write_checkpoint_shards", delayed_write):
            worker = exp.DistributedCheckpointWorker(Path(directory) / "async", PROVENANCE, None)
            try:
                worker.submit(run)
                assert started.wait(10)
                run.run_update()
                worker.check()
                assert not (Path(directory) / "async/update-00000002.json").exists()
            finally:
                release.set()
                worker.close()
        after = exp._update_tensors(run)
        exp.restore_checkpoint(run, Path(directory) / "async/update-00000002.json", PROVENANCE)
        assert exp._resume_state_hashes(run) == boundary
        run.run_update()
        assert exp._compare_resume_updates(after, after, exp._update_tensors(run)).passed
        distributed.check_parameters(tuple(run.policy.model.parameters()))

        # Unequal local counts must give the gradient of the merged objective.
        parameter = torch.nn.Parameter(torch.tensor(2.0))
        x = torch.tensor([1.0, 3.0] if rank == 0 else [5.0])
        count = distributed.sum_tensor(torch.tensor(x.numel()))
        ((parameter * x).square().sum() * 2 / count).backward()
        distributed.average_gradients((parameter,))
        assert parameter.grad is not None
        assert float(parameter.grad) == pytest.approx(4 * (1 + 9 + 25) / 3)

        values = torch.tensor([1.0, 2.0] if rank == 0 else [8.0])
        targets = torch.tensor([2.0, 5.0] if rank == 0 else [9.0])
        totals = distributed.MetricTotals(torch.device("cpu"))
        totals.total("count", values.numel())
        totals.mean("mean", values)
        totals.std("std", values)
        totals.explained_variance("ev", values, targets)
        totals.ratio("gameplay/win_rate", 1 if rank == 0 else 2, 2 if rank == 0 else 7)
        totals.ratio("policy/kl_per_frame", float(values.sum()), values.numel())
        totals.total("behavior/student_grab_count", 1 if rank == 0 else 9)
        totals.total("behavior/student_active_frames_count", 3600 if rank == 0 else 360)
        totals.ratio("behavior/student_grab_per_min", (1 if rank == 0 else 9) * 3600, 3600 if rank == 0 else 360)
        merged = totals.reduce(
            {
                "systems/actor_seconds": 2.0 if rank == 0 else 4.0,
                "systems/rollout_seconds": 1.0 if rank == 0 else 3.0,
                "systems/preparation_seconds": 1.0,
                "systems/critic_seconds": 1.0,
                "data/trained_decisions": float(values.numel()),
                "memory/first_ppo_actor_peak_allocated_bytes": 100.0 + rank,
            }
        )
        merged.update(
            phase_metrics(
                (1e12 * (rank + 1), 2e12 * (rank + 1), 3e12, 4e12),
                10 + rank,
                values.numel(),
                merged,
                device=torch.device("cpu"),
                peak=10e12,
            )
        )
        if rank == 0:
            (Path(directory) / "metrics.json").write_text(json.dumps(merged))
        assert distributed.sum_tensor(torch.tensor(1.0)).item() == 2
        with pytest.raises(ValueError, match="world size mismatch"):
            distributed.validate_world_size("1")
        if rank == 1:
            with torch.no_grad():
                next(run.policy.model.parameters()).flatten()[0].add_(0.25)
        with pytest.raises(RuntimeError, match="diverged"):
            distributed.check_parameters(tuple(run.policy.model.parameters()))
        (Path(directory) / f"rank{rank}.json").write_text(
            json.dumps({"steps": run.actor_steps, "updates": run.update})
        )
    finally:
        dist.destroy_process_group()


def _fresh_resume_worker(rank: int, rendezvous: str, directory: str) -> None:
    from test_065_loop import PROVENANCE
    from test_065_loop import FakeShard
    from test_065_loop import exp
    from test_065_loop import trainer

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        run = trainer(groups=1, length=67 if rank == 0 else 1, delay=8)
        engine = exp.RolloutEngine(
            (
                (
                    FakeShard((2 * rank,), length=67 if rank == 0 else 1),
                    FakeShard((2 * rank + 1,), length=67 if rank == 0 else 1),
                ),
            ),
            run.policy,
            run.engine.sampler,
            {0: run.reference},
        )
        engine.initialize()
        run = exp.Trainer(engine, dataclasses.replace(run.config, actor_microbatches=2))
        manifest = Path(directory) / "checkpoint/update-00000002.json"
        exp.check_resume_process(run, manifest, Path(directory) / "resume-controls", PROVENANCE, prepare=False)
        assert run.update == 2
        exp.run_training(run, 5, Path(directory) / "continued", PROVENANCE)
        assert run.update == 5
        final = Path(directory) / "continued/checkpoints/update-00000005.json"
        exp.restore_checkpoint(run, final, PROVENANCE)
        assert run.update == 5
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def gloo_result(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("o65-gloo")
    mp.spawn(_worker, args=(f"file://{directory / 'rendezvous'}", str(directory)), nprocs=2, join=True)
    mp.spawn(
        _fresh_resume_worker, args=(f"file://{directory / 'fresh-rendezvous'}", str(directory)), nprocs=2, join=True
    )
    return directory


def test_two_rank_ppo_steps_parameters_drift_and_resume(gloo_result: Path) -> None:
    rows = [json.loads((gloo_result / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert rows[0] == rows[1]
    assert rows[0]["updates"] == 3 and rows[0]["steps"] >= 3
    manifest = json.loads(next((gloo_result / "checkpoint").glob("update-*.json")).read_text())
    assert manifest["version"] == 12 and manifest["environment"]["world_size"] == "2"
    assert {shard["role"] for shard in manifest["shards"]} == {
        "training",
        "rank1",
        "simulator_0_0",
        "simulator_0_1",
        "rank1_simulator_0_0",
        "rank1_simulator_0_1",
    }


def test_rank_zero_metrics_equal_merged_single_process(gloo_result: Path) -> None:
    merged = json.loads((gloo_result / "metrics.json").read_text())
    values, targets = torch.tensor([1.0, 2.0, 8.0]), torch.tensor([2.0, 5.0, 9.0])
    assert merged["count"] == 3
    assert merged["mean"] == pytest.approx(float(values.mean()))
    assert merged["std"] == pytest.approx(float(values.std(unbiased=False)))
    assert merged["ev"] == pytest.approx(
        float(1 - (targets - values).var(unbiased=False) / targets.var(unbiased=False))
    )
    assert merged["gameplay/win_rate"] == pytest.approx(3 / 9)
    assert merged["policy/kl_per_frame"] == pytest.approx(11 / 3)
    assert merged["behavior/student_grab_count"] == 10
    assert merged["behavior/student_active_frames_count"] == 3960
    assert merged["behavior/student_grab_per_min"] == pytest.approx(36000 / 3960)
    assert merged["memory/first_ppo_actor_peak_allocated_bytes"] == 101
    assert merged["flops/actor_total"] == 6e12
    assert merged["flops/actor_per_decision"] == 2e12
    assert merged["throughput/actor_tflops"] == 1.5
    assert merged["mfu/actor"] == pytest.approx(6e12 / (4 * 2 * 10e12))
    assert merged["flops/rollout_per_match_frame"] == pytest.approx(3e12 / 21)


def test_fresh_process_resume_matches_next_batch_and_update_on_every_rank(gloo_result: Path) -> None:
    for rank in range(2):
        report = json.loads((gloo_result / "resume-controls" / f"rank{rank}-result.json").read_text())
        assert report["passed"]
        assert report["rank"] == rank and report["world_size"] == 2
        assert len(set(report["next_batch_sha256"])) == 1
        assert len(set(report["next_scheduler_sha256"])) == 1
        assert not (gloo_result / "resume-controls" / f"rank{rank}-controls.pt").exists()
    metrics = [json.loads(line) for line in (gloo_result / "continued/metrics.jsonl").read_text().splitlines()]
    assert [row["global_step"] for row in metrics] == [3, 4, 5]
    manifest = json.loads((gloo_result / "continued/checkpoints/update-00000005.json").read_text())
    assert manifest["update"] == 5
    assert len(manifest["shards"]) == 6


def test_resume_rejects_different_world_size(gloo_result: Path) -> None:
    from test_065_loop import PROVENANCE
    from test_065_loop import exp
    from test_065_loop import trainer

    manifest = next((gloo_result / "checkpoint").glob("update-*.json"))
    with pytest.raises(ValueError, match="world size mismatch"):
        exp.restore_checkpoint(trainer(groups=1), manifest, PROVENANCE)


def test_split_cpus_is_disjoint_and_rejects_unequal_partitions() -> None:
    assert distributed.split_cpus((2, 4, 6, 8), 2, 0) == (2, 4)
    assert distributed.split_cpus((2, 4, 6, 8), 2, 1) == (6, 8)
    with pytest.raises(ValueError, match="evenly"):
        distributed.split_cpus((2, 4, 6), 2, 0)


@pytest.mark.parametrize("operation", ["write_checkpoint_shards", "write_checkpoint_manifest"])
def test_background_distributed_write_failure_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    from test_065_loop import PROVENANCE
    from test_065_loop import exp
    from test_065_loop import trainer

    run = trainer(groups=1)
    run.run_update()

    def fail(*args: object) -> None:
        raise OSError("checkpoint disk failure")

    monkeypatch.setattr(exp, operation, fail)
    worker = exp.DistributedCheckpointWorker(tmp_path, PROVENANCE, None)
    worker.submit(run)
    with pytest.raises(OSError, match="checkpoint disk failure"):
        worker.close()
    assert run.update == 1
    assert not list(tmp_path.glob("update-*.json"))


@pytest.mark.parametrize("version", [10, 11])
def test_resume_rejects_formats_without_combined_state(tmp_path: Path, version: int) -> None:
    from test_065_loop import PROVENANCE
    from test_065_loop import exp
    from test_065_loop import trainer

    run = trainer(groups=1)
    run.run_update()
    path = exp.save_checkpoint(run, tmp_path, PROVENANCE)
    manifest = json.loads(path.read_text())
    manifest.update(format=f"hal.o65.training.v{version}", version=version)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checkpoint format"):
        exp.restore_checkpoint(run, path, PROVENANCE)


def test_measurement_runs_ppo_without_checkpoint_publication(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import os

    from test_065_loop import PROVENANCE
    from test_065_loop import exp
    from test_065_loop import trainer

    run = trainer(groups=1)
    config = exp.NativeCli(
        checkpoint=tmp_path / "bc.pt",
        sampler=tmp_path / "sampler.json",
        output=tmp_path / "measurement",
        data_dir=tmp_path,
        cpus=(min(os.sched_getaffinity(0)),),
        updates=2,
        measurement=True,
        training=run.config,
    )
    exp.run_measurement(run, config, PROVENANCE, {})
    rows = [json.loads(line) for line in (config.output / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[-1]["train/actor_steps"] > 0
    assert not (config.output / "checkpoints").exists()
    assert len([line for line in capsys.readouterr().out.splitlines() if '"measurement": true' in line]) == 2
