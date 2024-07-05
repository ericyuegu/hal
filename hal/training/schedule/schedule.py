from typing import Protocol
from typing import Sequence
from typing import Union

import attr
import numpy as np
import torch

from hal.training.schedule.stage import LRStage

DEFAULT_LR_STAGES: Sequence[LRStage] = (
    LRStage(end=0.0625, scalar=1),
    LRStage(end=0.75, scalar=1),
    LRStage(end=0.875, scalar=0.1),
    LRStage(end=0.9375, scalar=0.01),
)


class Schedule(Protocol):
    def __call__(self, progress: float) -> float:
        raise NotImplementedError


@attr.s(auto_attribs=True, frozen=True)
class SchedulePieceWiseCos(Schedule):
    """A cosine annealing curve schedule."""

    stages: Sequence[LRStage] = DEFAULT_LR_STAGES
    initial: float = 1.0

    def __call__(self, progress: float) -> float:
        if progress > self.stages[-1].end:
            return self.stages[-1].scalar

        prev_stage = curr_stage = LRStage(end=0.0, scalar=self.initial)
        for stage in self.stages:
            curr_stage = stage
            if progress <= stage.end:
                break
            prev_stage = stage

        lam = 0.5 * (1 + np.cos(np.pi * (progress - prev_stage.end) / (curr_stage.end - prev_stage.end)))
        return lam * prev_stage.scalar + (1 - lam) * curr_stage.scalar


@attr.s(auto_attribs=True, frozen=True)
class SchedulePieceWiseCosWarmup(SchedulePieceWiseCos):
    """A cosine annealing curve schedule with warmup."""

    initial: float = 0.0


@attr.s(auto_attribs=True, frozen=True)
class LearningRateChanger:
    """Base class for learning rate schedulers / changers."""

    opt: torch.optim.Optimizer

    def set_opt_lr(self, lr: Union[float, torch.Tensor]) -> float:
        for params in self.opt.param_groups:
            params["lr"] = lr
        return lr


@attr.s(auto_attribs=True, frozen=True)
class LearningRatePieceWiseCos(LearningRateChanger):
    """Use a cosine annealing curve in the LR decay stages for continuity."""

    opt: torch.optim.Optimizer
    base_lr: float
    schedule: Schedule = SchedulePieceWiseCosWarmup()

    def __call__(self, progress: float) -> float:
        return self.set_opt_lr(self.base_lr * self.schedule(progress))
