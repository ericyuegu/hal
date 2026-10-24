"""Cover underflowed action probabilities and a failed distributed worker."""

import contextlib
import dataclasses
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_065_loop import exp
from test_065_loop import trainer

from hal.models.replay_actor import ActorTargets
from hal.models.replay_actor import actor_loss
from hal.models.replay_actor import validate_actor_targets
from hal.models.window_replay import ReplayOutput
from hal.training import self_play_distributed as distributed


def test_normalization_error_preserves_the_failing_distribution() -> None:
    logs = torch.tensor([0.2, 0.3]).log()
    with pytest.raises(ValueError, match="policy log-probs must be normalized; row=0, log_mass=.*values="):
        exp.categorical_kl(logs, torch.tensor([0.4, 0.6]).log())


@pytest.mark.parametrize("tail", [(-110.0, -102.58578491210938), (-102.58578491210938, -110.0)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_rollout_and_actor_kl_keep_finite_log_tails(tail: tuple[float, float], device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    logits = torch.tensor([0.2, -0.4, tail[0]], requires_grad=True, device=device)
    reference_logits = torch.tensor([-0.1, 0.3, tail[1]], requires_grad=True, device=device)
    p = logits.log_softmax(-1)
    q = reference_logits.log_softmax(-1)
    if device == "cpu":
        assert bool((p.exp() == 0).any()) != bool((q.exp() == 0).any())
    run = trainer(device, groups=1)
    batch = run.prepare(run.engine.collect(0))
    count = len(batch.old.log_probs)
    old = dataclasses.replace(batch.old, log_distributions=(p.expand(count, 4, 3),) * 4)
    reference = dataclasses.replace(batch.old, log_distributions=(q.expand(count, 4, 3),) * 4)
    expected = (p.double().exp() * (p.double() - q.double())).sum()
    rollout_kl = exp.factor_kl(old, reference, torch.ones(count, 16, dtype=torch.bool, device=device))
    torch.testing.assert_close(rollout_kl.double(), expected.expand(count, 16), atol=1e-6, rtol=1e-6)

    replay = run.policy.window_output(batch.windows)
    shape = replay.log_probs_WKF.shape[:2]
    logs = p[0].expand(*shape, 16)
    mask = torch.ones_like(logs, dtype=torch.bool)
    output = ReplayOutput(logs, (p.expand(*shape, 4, 3),) * 4, mask, replay.features)
    targets = ActorTargets(
        logs.detach(),
        p.new_zeros(shape),
        mask.any(-1),
        p.new_ones(shape),
        (q.expand(*shape, 4, 3),) * 4,
        p.new_tensor(0.0),
        p.new_tensor(1.0),
        p.new_tensor(float(mask.any(-1).sum())),
    )
    validate_actor_targets(output, targets)
    loss = actor_loss(output, targets)
    torch.testing.assert_close(loss.kl_loss.double(), 16 * expected, atol=1e-6, rtol=1e-6)
    expected_gradient = torch.autograd.grad(16 * expected, logits, retain_graph=True)[0]
    loss.loss.backward()
    assert logits.grad is not None and bool(torch.isfinite(logits.grad).all())
    torch.testing.assert_close(logits.grad, expected_gradient, atol=1e-6, rtol=1e-6)
    assert reference_logits.grad is None


def _failed_worker(rank: int, rendezvous: str, directory: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    with contextlib.ExitStack() as resources, distributed.fail_fast():
        resources.callback(time.sleep, 45)
        dist.barrier()
        if rank == 1:
            raise ValueError("injected numerical failure")
        dist.all_reduce(torch.ones(1))
    (Path(directory) / f"cleaned-{rank}").touch()


def test_one_worker_failure_terminates_peers_before_cleanup(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    started = time.monotonic()
    with pytest.raises(mp.ProcessExitedException, match="exit code 1"):
        mp.spawn(_failed_worker, args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path)), nprocs=2, join=True)
    assert time.monotonic() - started < 20
    assert "injected numerical failure" in capfd.readouterr().err
    assert not list(tmp_path.glob("cleaned-*"))


def test_single_process_error_preserves_exception_and_cleanup() -> None:
    cleaned: list[bool] = []
    with (
        pytest.raises(ValueError, match="original error"),
        contextlib.ExitStack() as resources,
        distributed.fail_fast(),
    ):
        resources.callback(cleaned.append, True)
        raise ValueError("original error")
    assert cleaned == [True]
