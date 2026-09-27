"""The stable full-context validation path used by experiment 059."""

import functools
import hashlib
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from typing import cast

import numpy as np
import torch
from streaming import Stream
from streaming import StreamingDataset
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

from hal.data.feature_stats import FeatureStats
from hal.data.policy_world_schema import decode_policy_world_replay
from hal.data.schema import check_schema_version
from hal.data.streaming_compat import patch_streaming
from hal.representation.features import ExtraColumns
from hal.representation.features import FeatureProjection
from hal.streams import StreamSource
from hal.training.batches import TrainBatch
from hal.training.replay_windows import ReplayLabels
from hal.training.replay_windows import Window
from hal.training.replay_windows import choose_chunk_starts
from hal.training.replay_windows import collate_windows
from hal.training.replay_windows import make_window
from hal.training.replay_windows import stable_window_rng
from hal.training.replay_windows import train_batch_from_columns


@dataclass(frozen=True, slots=True)
class ValidationWindowIdentity:
    replay_id: str
    start: int
    ego_prefix: str
    window_sha256: str


def _window_digest(window: Mapping[str, np.ndarray | np.integer]) -> str:
    digest = hashlib.sha256()
    for name in sorted(window):
        value = np.asarray(window[name])
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(str(value.shape).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


class _ValidationWindows(IterableDataset[Window]):
    """Yield one replay-stable, full-context window per policy-world row."""

    def __init__(
        self,
        mds: StreamingDataset,
        *,
        context_length: int,
        chunk_length: int,
        seed: int,
        schema_version: int,
        projection: FeatureProjection,
        replay_labels: ReplayLabels,
    ) -> None:
        self.mds = mds
        self.context_length = context_length
        self.chunk_length = chunk_length
        self.seed = seed
        self.schema_version = schema_version
        self.projection = projection
        self.replay_labels = replay_labels
        self.epoch = 0
        self.identities: list[ValidationWindowIdentity] = []

    def __iter__(self) -> Iterator[Window]:
        epoch = self.epoch
        self.epoch += 1
        self.identities.clear()
        for compact in self.mds:
            replay_id = str(compact["replay_id"])
            labels = self.replay_labels(compact)
            sample = cast(dict[str, np.ndarray], decode_policy_world_replay(compact))
            frame = sample["frame"]
            frame_count = len(frame)
            for name, value in labels.items():
                array = np.asarray(value)
                if array.ndim < 1 or len(array) != frame_count:
                    raise ValueError(
                        f"replay label {name!r} has length {len(array) if array.ndim else 0}, expected {frame_count}"
                    )
                if name in sample:
                    raise ValueError(f"replay label {name!r} collides with a decoded column")
                sample[name] = array
            check_schema_version(sample, expected=self.schema_version)
            rng = stable_window_rng(self.seed, epoch, replay_id)
            starts = choose_chunk_starts(
                frame_count,
                self.context_length,
                self.chunk_length,
                1,
                rng,
                require_full_context=True,
            )
            if len(starts) != 1:
                raise ValueError(
                    f"full-context window requires at least {self.context_length + self.chunk_length} "
                    f"frames, got {frame_count} for replay {replay_id!r}"
                )
            start = int(starts[0]) - self.context_length
            ego_prefix = "p1" if rng.random() < 0.5 else "p2"
            window = make_window(
                sample,
                ego_prefix=ego_prefix,
                start=start,
                pad=0,
                length=self.context_length + self.chunk_length,
                projection=self.projection,
            )
            window["ctx_pad"] = np.int64(0)
            self.identities.append(ValidationWindowIdentity(replay_id, start, ego_prefix, _window_digest(window)))
            yield window


type BatchTransform = Callable[[list[Window], TrainBatch], object]


def _collate_validation(
    windows: list[Window],
    *,
    stats: dict[str, FeatureStats],
    context_length: int,
    extra: ExtraColumns,
    projection: FeatureProjection,
    batch_transform: BatchTransform,
) -> object:
    batch = train_batch_from_columns(
        collate_windows(windows),
        stats=stats,
        L_ctx=context_length,
        extra=extra,
        projection=projection,
    )
    return batch_transform(windows, batch)


def make_validation_replay_loader(
    *,
    sources: Sequence[StreamSource],
    stats: dict[str, FeatureStats],
    context_length: int,
    chunk_length: int,
    batch_size: int,
    seed: int,
    cache_limit: str | int,
    schema_version: int,
    extra: ExtraColumns,
    projection: FeatureProjection,
    replay_labels: ReplayLabels,
    batch_transform: BatchTransform,
) -> DataLoader[Any]:
    """Preserve 059's ordered Mosaic py1e validation cohort and collation."""
    if not sources or len({source.name for source in sources}) != len(sources):
        raise ValueError("validation sources must be non-empty and unique")
    if context_length < 1 or chunk_length < 1 or batch_size < 1:
        raise ValueError("validation window and batch geometry must be positive")
    patch_streaming()
    streams = [
        Stream(
            remote=source.remote,
            local=str(source.local_root),
            split="val",
            repeat=1,
            download_retry=2,
        )
        for source in sources
    ]
    mds = StreamingDataset(
        streams=streams,
        cache_limit=cache_limit,
        predownload=8 * batch_size,
        batch_size=batch_size,
        shuffle=True,
        shuffle_algo="py1e",
        shuffle_block_size=8192,
        shuffle_seed=seed,
    )
    windows = _ValidationWindows(
        mds,
        context_length=context_length,
        chunk_length=chunk_length,
        seed=seed,
        schema_version=schema_version,
        projection=projection,
        replay_labels=replay_labels,
    )
    collate = functools.partial(
        _collate_validation,
        stats=stats,
        context_length=context_length,
        extra=extra,
        projection=projection,
        batch_transform=batch_transform,
    )
    return DataLoader(
        windows,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate,
        persistent_workers=False,
        prefetch_factor=None,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
        timeout=0,
        in_order=False,
    )
