"""Single-node collectives outside self-play CUDA graphs."""

import contextlib
import math
import os
from collections.abc import Iterator
from collections.abc import Sequence
from datetime import timedelta
from typing import Literal

import torch
import torch.distributed as dist
from torch import Tensor


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def split_cpus(cpus: tuple[int, ...], processes: int, local_rank: int) -> tuple[int, ...]:
    if processes < 1 or not 0 <= local_rank < processes or len(cpus) % processes or len(cpus) < processes:
        raise ValueError("shard CPUs must divide evenly across local ranks")
    if len(set(cpus)) != len(cpus):
        raise ValueError("shard CPUs must be distinct")
    count = len(cpus) // processes
    return cpus[local_rank * count : (local_rank + 1) * count]


@contextlib.contextmanager
def process_group(device: str) -> Iterator[torch.device]:
    size = int(os.environ.get("WORLD_SIZE", "1"))
    if size == 1:
        yield torch.device(device)
        return
    local = int(os.environ["LOCAL_RANK"])
    if (
        device != "cuda"
        or size < 2
        or int(os.environ["LOCAL_WORLD_SIZE"]) != size
        or int(os.environ["RANK"]) != local
        or not 0 <= local < size <= torch.cuda.device_count()
    ):
        raise ValueError("O65 distributed training requires one torchrun node with one CUDA device per rank")
    torch.cuda.set_device(local)
    selected = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=selected, timeout=timedelta(minutes=30))
    try:
        yield selected
    finally:
        dist.destroy_process_group()


def sum_tensor(value: Tensor) -> Tensor:
    result = value.detach().clone()
    if world_size() > 1:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def maximum(value: int, device: torch.device) -> int:
    result = torch.tensor(value, device=device, dtype=torch.int64)
    if world_size() > 1:
        dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return int(result)


def gather_objects[T](value: T) -> list[T]:
    if world_size() == 1:
        return [value]
    result: list[T] = [value for _ in range(world_size())]
    dist.all_gather_object(result, value)
    return result


@torch.no_grad()
def average_gradients(parameters: Sequence[torch.nn.Parameter], *, bucket_bytes: int = 32 * 1024 * 1024) -> None:
    """Average only the supplied trainable parameters, preserving graph gradient storage."""
    if world_size() == 1:
        return
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise ValueError("gradient synchronization requires trainable parameters only")
    buckets: list[list[Tensor]] = []
    count = 0
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        gradient = parameter.grad
        if gradient.is_sparse:
            raise ValueError("O65 gradient synchronization requires dense gradients")
        size = gradient.numel() * gradient.element_size()
        if not buckets or count + size > bucket_bytes or buckets[-1][0].dtype != gradient.dtype:
            buckets.append([])
            count = 0
        buckets[-1].append(gradient)
        count += size
    for gradients in buckets:
        flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world_size())
        offset = 0
        for gradient in gradients:
            gradient.copy_(flat[offset : offset + gradient.numel()].view_as(gradient))
            offset += gradient.numel()


@torch.no_grad()
def check_parameters(parameters: Sequence[Tensor]) -> None:
    """Compare full-tensor sums and norms plus samples, failing on any rank's drift."""
    if world_size() == 1:
        return
    checks = []
    for parameter in parameters:
        value = parameter.detach().float().reshape(-1)
        checks.append(
            torch.stack(
                (
                    value.sum(dtype=torch.float64),
                    torch.linalg.vector_norm(value).double(),
                    value[:: max(1, value.numel() // 32)].double().sum(),
                )
            )
        )
    checksum = torch.stack(checks)
    low, high = checksum.clone(), checksum.clone()
    dist.all_reduce(low, op=dist.ReduceOp.MIN)
    dist.all_reduce(high, op=dist.ReduceOp.MAX)
    if not bool(torch.isfinite(checksum).all()) or not torch.equal(low, high):
        raise RuntimeError("O65 parameters diverged across ranks")


def validate_world_size(saved: str) -> None:
    if saved != str(world_size()):
        raise ValueError(f"resume world size mismatch: checkpoint={saved}, current={world_size()}")


class MetricTotals:
    """Reduce sufficient statistics before computing rates, spreads and variance."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.rows: dict[str, tuple[str, list[float]]] = {}

    def ratio(self, name: str, total: float, count: float, *, kind: Literal["mean", "rmse"] = "mean") -> float:
        self.rows[name] = (kind, [total, count, 0.0, 0.0, 0.0])
        return total / count if count else math.nan

    def total(self, name: str, value: float) -> float:
        self.rows[name] = ("sum", [value, 1.0, 0.0, 0.0, 0.0])
        return value

    def mean(self, name: str, values: Tensor) -> float:
        return self.ratio(name, float(values.double().sum()), values.numel())

    def std(self, name: str, values: Tensor) -> float:
        x = values.detach().double()
        self.rows[name] = ("std", [float(x.sum()), x.numel(), float(x.square().sum()), 0.0, 0.0])
        return float(x.std(unbiased=False)) if x.numel() else math.nan

    def explained_variance(self, name: str, values: Tensor, targets: Tensor) -> float:
        target = targets.detach().double()
        error = target - values.detach().double()
        self.rows[name] = (
            "ev",
            [
                float(target.sum()),
                target.numel(),
                float(target.square().sum()),
                float(error.sum()),
                float(error.square().sum()),
            ],
        )
        variance = float(target.var(unbiased=False)) if target.numel() else 0.0
        return float(1 - error.var(unbiased=False) / variance) if variance else math.nan

    def reduce(self, metrics: dict[str, float]) -> dict[str, float]:
        if world_size() == 1:
            return metrics
        schemas = gather_objects({name: kind for name, (kind, _) in self.rows.items()})
        schema: dict[str, str] = {}
        for item in schemas:
            for name, kind in item.items():
                if name in schema and schema[name] != kind:
                    raise ValueError(f"inconsistent metric reduction for {name}")
                schema[name] = kind
        names = sorted(schema)
        packed = (
            sum_tensor(
                torch.tensor(
                    [self.rows.get(name, (schema[name], [0.0] * 5))[1] for name in names],
                    dtype=torch.float64,
                    device=self.device,
                )
            )
            .cpu()
            .tolist()
        )
        result: dict[str, float] = {}
        for name, (total, count, squares, error, error_squares) in zip(names, packed, strict=True):
            if not count:
                continue
            kind = schema[name]
            value = total / count
            if kind == "sum":
                value = total
            elif kind == "std":
                value = math.sqrt(max(0.0, squares / count - value**2))
            elif kind == "ev":
                variance = max(0.0, squares / count - value**2)
                value = 1 - max(0.0, error_squares / count - (error / count) ** 2) / variance if variance else math.nan
            elif kind == "rmse":
                value = math.sqrt(value)
            result[name] = value
        # System timings describe the slowest rank. Counts add; synchronized
        # optimizer diagnostics use a mean. Domain rates above use their counts.
        others = gather_objects({name: value for name, value in metrics.items() if name not in schema})
        for name in sorted({name for row in others for name in row}):
            values = [row[name] for row in others if name in row]
            if name in ("policy/ratio_p01", "policy/ratio_p99", "policy/old_log_ratio_p99"):
                # Quantiles cannot be combined from rank-local quantiles.
                if name in others[0]:
                    result[f"{name}_rank0"] = others[0][name]
                continue
            reduction: Literal["sum", "max", "mean"] = "mean"
            if name.startswith(("systems/", "memory/")):
                reduction = "max"
            if name.startswith(("data/identity_", "data/character_", "data/stage_")) or name in (
                "data/resets",
                "data/timeouts",
                "data/decisions",
                "data/trained_decisions",
                "systems/rollout_frames",
                "systems/reset_idle_frames",
                "critic/monte_carlo_episode_count",
                "critic/bootstrap_context_count",
            ):
                reduction = "sum"
            result[name] = (
                sum(values) if reduction == "sum" else max(values) if reduction == "max" else sum(values) / len(values)
            )
        if "systems/rollout_frames" in result:
            result["systems/match_frames_per_second"] = (
                result["systems/rollout_frames"] / result["systems/update_seconds"]
            )
            result["systems/simulator_fps"] = result["systems/rollout_frames"] / result["systems/rollout_seconds"]
        return result
