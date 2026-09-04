"""Deterministic categorical sampling for batched policy evaluation."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from hal.training.features import Context

_UINT64_MASK = (1 << 64) - 1


def sample_categorical(
    logits: Tensor,
    *,
    argmax: bool,
    uniform: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample class indices, optionally from caller-provided uniforms."""
    values = logits.float()
    if argmax:
        return values.argmax(dim=-1)
    probabilities = F.softmax(values, dim=-1)
    if uniform is None:
        return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
    if uniform.shape != probabilities.shape[:-1]:
        raise ValueError(f"uniform shape {tuple(uniform.shape)} != batch shape {tuple(probabilities.shape[:-1])}")
    uniform = uniform.to(device=probabilities.device, dtype=probabilities.dtype)
    return (probabilities.cumsum(-1) < uniform[..., None]).sum(-1).clamp_max(probabilities.shape[-1] - 1)


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return value ^ (value >> 31)


class SlotGroupRng:
    """Counter RNG keyed by evaluation slot, generation, and output group."""

    def __init__(self, seed: int, group_names: Sequence[str]) -> None:
        names = tuple(group_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("group_names must be non-empty and unique")
        self.seed = seed & _UINT64_MASK
        self.group_index = {name: index for index, name in enumerate(names)}
        self.generations: dict[int, int] = {}
        self.counters: dict[tuple[int, int, str], int] = {}
        self.slot_ids: tuple[int, ...] = ()
        self.device = torch.device("cpu")

    def begin(self, context: Context) -> None:
        if context.slot_ids is None:
            raise ValueError("slot-keyed sampling needs slot_ids")
        slot_ids = tuple(int(value) for value in context.slot_ids.tolist())
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError(f"slot_ids must be unique, got {slot_ids}")
        resets = (
            (False,) * len(slot_ids)
            if context.reset is None
            else tuple(bool(value) for value in context.reset.tolist())
        )
        for slot_id, reset in zip(slot_ids, resets, strict=True):
            if slot_id not in self.generations:
                self.generations[slot_id] = 0
            elif reset:
                self.generations[slot_id] += 1
            generation = self.generations[slot_id]
            if reset or not any(key[:2] == (slot_id, generation) for key in self.counters):
                for name in self.group_index:
                    self.counters[(slot_id, generation, name)] = 0
        self.slot_ids = slot_ids
        self.device = context.slot_ids.device

    def uniforms(self, group: str) -> Tensor:
        try:
            group_index = self.group_index[group]
        except KeyError as error:
            raise ValueError(f"unknown group {group!r}") from error
        values: list[float] = []
        group_key = _splitmix64(group_index + 1)
        for slot_id in self.slot_ids:
            generation = self.generations[slot_id]
            key = (slot_id, generation, group)
            counter = self.counters[key]
            mixed = self.seed ^ _splitmix64(slot_id) ^ _splitmix64(generation) ^ group_key ^ _splitmix64(counter)
            values.append(((_splitmix64(mixed) >> 11) + 0.5) / (1 << 53))
            self.counters[key] = counter + 1
        return torch.tensor(values, device=self.device)

    def state(self) -> tuple[tuple[int, int, str, int], ...]:
        return tuple(sorted((*key, value) for key, value in self.counters.items()))
