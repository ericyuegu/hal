"""Two-process tests of the actual PPO update and distributed state boundary."""

import dataclasses
import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from hal.training import self_play_distributed as distributed


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
        checkpoint = exp.save_checkpoint(run, Path(directory) / "checkpoint", PROVENANCE)
        # Rank-local state must survive the distributed checkpoint.
        local_cursor = run.engine.sampler.cursor
        local_rng = torch.get_rng_state().clone()
        run.engine.sampler.cursor += rank + 1
        torch.manual_seed(777 + rank)
        exp.restore_checkpoint(run, checkpoint, PROVENANCE)
        assert run.engine.sampler.cursor == local_cursor
        assert torch.equal(torch.get_rng_state(), local_rng)
        # A next update after resume still has identical optimizer progress.
        run.run_update()
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
        merged = totals.reduce({})
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


@pytest.fixture(scope="module")
def gloo_result(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("o65-gloo")
    mp.spawn(_worker, args=(f"file://{directory / 'rendezvous'}", str(directory)), nprocs=2, join=True)
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
