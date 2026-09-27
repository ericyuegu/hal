"""Supervised batches and their tensor lifetime operations."""

from dataclasses import dataclass

import torch
from torch import Tensor

from hal.representation.features import Context


@dataclass(frozen=True, slots=True)
class TrainBatch:
    """One supervised example batch: a Context plus the action chunk to predict."""

    context: Context
    target: Tensor  # [B, L_chunk, d_action]
    replay_ids: tuple[str, ...] | None = None

    def to(self, device: str | torch.device) -> TrainBatch:
        return TrainBatch(
            context=self.context.to(device),
            target=self.target.to(device, non_blocking=True),
            replay_ids=self.replay_ids,
        )

    def pin_memory(self) -> TrainBatch:
        return TrainBatch(
            context=self.context.pin_memory(), target=self.target.pin_memory(), replay_ids=self.replay_ids
        )

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        """Keep every tensor's allocation owned until ``stream`` finishes using it."""
        tensors = [*self.context.features.values(), self.context.ctx_pad, self.target]
        for tensor in tensors:
            tensor.record_stream(stream)
