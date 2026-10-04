"""Two-process Gloo checks for experiment 060's DDP and resume contracts."""

import copy
import importlib.util
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Any
from typing import cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

_SPEC = importlib.util.spec_from_file_location(
    "hal_experiment_060_distributed",
    Path(__file__).parents[1] / "experiments" / "060_compute_optimal_action_sequence.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _rank_checkpoint_record(cfg: Any, rank: int) -> dict[str, object]:
    calibration = _MODULE.ReturnCalibration(window_count=_MODULE.CALIBRATION_WINDOWS // 2)
    return {
        "rank": rank,
        "seed": _MODULE.rank_seed(cfg.seed, rank),
        "partition_sha256": _MODULE.data_partition_hashes(cfg)[rank],
        "loader": {"next_batch_token": f"rank-{rank}-batch-2"},
        "identity_masker": {"rank": rank},
        "return_masker": {"rank": rank},
        "prefix_sampler": {"rank": rank},
        "return_calibration": calibration.state_dict(),
        "rng": _MODULE.rng_state(),
    }


def _gloo_worker(rank: int, rendezvous: str, result_dir: str) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        cfg = _MODULE.TrainConfig()
        object_group = dist.new_group(backend="gloo")
        context = _MODULE.DistributedContext(rank, rank, 2, torch.device("cpu"), object_group)
        selection = _MODULE.data_selection(cfg, rank=rank, world_size=2)
        assert selection.sha256 == _MODULE.data_partition_hashes(cfg)[rank]

        raw_model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            raw_model.weight.fill_(1.0)
        model = DistributedDataParallel(raw_model)
        model(torch.tensor([[float(rank + 1)]])).sum().backward()
        gradient = raw_model.weight.grad
        assert gradient is not None
        assert torch.allclose(gradient, torch.tensor([[1.5]]))

        reduced = torch.tensor(float(rank + 1))
        _MODULE._all_reduce_mean(reduced)
        assert float(reduced) == 1.5

        with torch.no_grad():
            raw_model.weight.fill_(0.25)
        optimizer = torch.optim.SGD(raw_model.parameters(), lr=0.1, momentum=0.9)
        torch.manual_seed(1000 + rank)
        optimizer.zero_grad()
        first_batch = torch.rand(4, 1)
        model(first_batch).square().mean().backward()
        optimizer.step()

        saved_model = copy.deepcopy(raw_model.state_dict())
        saved_optimizer = copy.deepcopy(optimizer.state_dict())
        local_record = _rank_checkpoint_record(cfg, rank)
        gathered = _MODULE.gather_rank_checkpoint_states(local_record, context)
        checkpoint: dict[str, object] | None = None
        if rank == 0:
            assert gathered is not None
            checkpoint = {
                "model": saved_model,
                "optimizer": saved_optimizer,
                "distributed": {
                    "contract": _MODULE.distributed_training_contract(cfg),
                    "ranks": list(gathered),
                },
            }
        payload: list[object] = [checkpoint]
        dist.broadcast_object_list(payload, src=0)
        checkpoint = cast(dict[str, object], payload[0])
        local = _MODULE.rank_resume_state(checkpoint, cfg, context)
        assert local["loader"] == {"next_batch_token": f"rank-{rank}-batch-2"}

        _MODULE.restore_rng(cast(dict[str, object], local["rng"]))
        next_batch = torch.rand(4, 1)
        optimizer.zero_grad()
        model(next_batch).square().mean().backward()
        optimizer.step()
        reference_weight = raw_model.weight.detach().clone()

        resumed_raw_model = nn.Linear(1, 1, bias=False)
        resumed_raw_model.load_state_dict(cast(dict[str, torch.Tensor], checkpoint["model"]))
        resumed_optimizer = torch.optim.SGD(resumed_raw_model.parameters(), lr=0.1, momentum=0.9)
        resumed_optimizer.load_state_dict(cast(dict[str, object], checkpoint["optimizer"]))
        resumed_model = DistributedDataParallel(resumed_raw_model)
        _MODULE.restore_rng(cast(dict[str, object], local["rng"]))
        resumed_batch = torch.rand(4, 1)
        assert torch.equal(resumed_batch, next_batch)
        resumed_optimizer.zero_grad()
        resumed_model(resumed_batch).square().mean().backward()
        resumed_optimizer.step()
        assert torch.equal(resumed_raw_model.weight, reference_weight)
        Path(result_dir, f"rank-{rank}.txt").write_text("passed\n")
    finally:
        dist.destroy_process_group()


def test_two_process_gradient_metrics_partitions_and_exact_resume(tmp_path: Path) -> None:
    rendezvous = tmp_path / "gloo-rendezvous"
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_gloo_worker, args=(rank, str(rendezvous), str(tmp_path))) for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=90)
        assert process.exitcode == 0
    assert [Path(tmp_path, f"rank-{rank}.txt").read_text() for rank in range(2)] == ["passed\n", "passed\n"]
