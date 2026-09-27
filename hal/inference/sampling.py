"""Independent counter-based categorical sampling streams."""

from collections.abc import Sequence

import torch
from torch import Tensor

_UINT64_MASK = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return value ^ (value >> 31)


class StreamGroupRng:
    """Counter RNG keyed by inference stream, generation, and output group."""

    def __init__(self, seed: int, group_names: Sequence[str]) -> None:
        names = tuple(group_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("group_names must be non-empty and unique")
        self.seed = seed & _UINT64_MASK
        self.index_by_group = {name: index for index, name in enumerate(names)}
        self.generations: dict[int, int] = {}
        self.counters: dict[tuple[int, int, str], int] = {}
        self.stream_ids: tuple[int, ...] = ()
        self.device = torch.device("cpu")

    def begin(
        self,
        stream_ids: Sequence[int],
        generations: Sequence[int],
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        stream_ids = tuple(stream_ids)
        generations = tuple(generations)
        if len(generations) != len(stream_ids):
            raise ValueError("sampling requires one generation per stream")
        if len(set(stream_ids)) != len(stream_ids) or any(type(value) is not int for value in stream_ids):
            raise ValueError("sampling stream IDs must be unique integers")
        if any(type(value) is not int or value < 0 for value in generations):
            raise ValueError("sampling generations must be nonnegative integers")
        for stream_id, generation in zip(stream_ids, generations, strict=True):
            if self.generations.get(stream_id) != generation:
                self.release(stream_id)
                self.generations[stream_id] = generation
                for name in self.index_by_group:
                    self.counters[(stream_id, generation, name)] = 0
        self.stream_ids = stream_ids
        self.device = torch.device(device)

    def release(self, stream_id: int) -> None:
        """Drop all sampling storage for a completed stream."""
        self.generations.pop(stream_id, None)
        self.counters = {key: value for key, value in self.counters.items() if key[0] != stream_id}
        self.stream_ids = tuple(value for value in self.stream_ids if value != stream_id)

    def restore(
        self,
        *,
        generations: Sequence[tuple[int, int]],
        counters: Sequence[tuple[int, int, str, int]],
    ) -> None:
        generation_by_stream = dict(generations)
        if len(generation_by_stream) != len(generations) or any(
            type(stream_id) is not int or type(generation) is not int or generation < 0
            for stream_id, generation in generations
        ):
            raise ValueError("invalid sampling generations")
        restored: dict[tuple[int, int, str], int] = {}
        for stream_id, generation, group, count in counters:
            key = (stream_id, generation, group)
            if (
                generation_by_stream.get(stream_id) != generation
                or group not in self.index_by_group
                or type(count) is not int
                or count < 0
                or key in restored
            ):
                raise ValueError("invalid sampling counters")
            restored[key] = count
        for stream_id, generation in generations:
            present = {name for row, version, name in restored if (row, version) == (stream_id, generation)}
            if present != self.index_by_group.keys():
                raise ValueError("sampling counters must contain every group for a stream")
        self.generations = generation_by_stream
        self.counters = restored
        self.stream_ids = ()

    def uniforms(self, group: str, active: Sequence[bool] | None = None) -> Tensor:
        """Draw once for each active stream and leave inactive counters unchanged."""
        try:
            index_by_group = self.index_by_group[group]
        except KeyError as error:
            raise ValueError(f"unknown group {group!r}") from error
        mask = (True,) * len(self.stream_ids) if active is None else tuple(active)
        if len(mask) != len(self.stream_ids):
            raise ValueError(f"active mask has {len(mask)} rows, expected {len(self.stream_ids)}")
        values: list[float] = []
        group_key = _splitmix64(index_by_group + 1)
        for stream_id, enabled in zip(self.stream_ids, mask, strict=True):
            if not enabled:
                values.append(0.5)
                continue
            generation = self.generations[stream_id]
            key = (stream_id, generation, group)
            counter = self.counters[key]
            mixed = self.seed ^ _splitmix64(stream_id) ^ _splitmix64(generation) ^ group_key ^ _splitmix64(counter)
            values.append(((_splitmix64(mixed) >> 11) + 0.5) / (1 << 53))
            self.counters[key] = counter + 1
        return torch.tensor(values, device=self.device)

    def state(self) -> tuple[tuple[int, int, str, int], ...]:
        return tuple(sorted((*key, value) for key, value in self.counters.items()))
