"""Explicit episode conditioning ranges for self-play policies."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ConditioningRanges:
    log_lambda: tuple[float, float]
    delay: tuple[int, int]
    replan: tuple[int, int]

    def __post_init__(self) -> None:
        for name, bounds in (("log_lambda", self.log_lambda), ("delay", self.delay), ("replan", self.replan)):
            if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or bounds[0] > bounds[1]:
                raise ValueError(f"invalid {name} range")
        if any(isinstance(x, bool) or not isinstance(x, int) for x in (*self.delay, *self.replan)):
            raise ValueError("delay and replan bounds must be integers")
        if self.delay[0] < 0 or self.replan[0] < 1:
            raise ValueError("delay must be nonnegative and replan must be positive")

    def normalize(self, log_lambda: Tensor, delay: Tensor, replan: Tensor, *, validate: bool = True) -> Tensor:
        if log_lambda.shape != delay.shape or delay.shape != replan.shape:
            raise ValueError("episode conditioning shapes must agree")
        values = (log_lambda, delay, replan)
        ranges = (self.log_lambda, self.delay, self.replan)
        normalized: list[Tensor] = []
        for value, (low, high) in zip(values, ranges, strict=True):
            if (
                validate
                and not torch.compiler.is_compiling()
                and bool((~torch.isfinite(value) | (value < low) | (value > high)).any())
            ):
                raise ValueError("episode conditioning is outside its declared range")
            normalized.append(
                torch.zeros_like(value, dtype=torch.float32)
                if low == high
                else 2 * (value.float() - low) / (high - low) - 1
            )
        return torch.stack(normalized, dim=-1)
