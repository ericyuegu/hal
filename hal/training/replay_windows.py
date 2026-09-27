"""Replay window sampling and conversion shared by 059 loaders."""

import hashlib
from collections.abc import Callable
from collections.abc import Mapping

import numpy as np
import torch

from hal.data.feature_stats import FeatureStats
from hal.representation.features import Context
from hal.representation.features import ExtraColumns
from hal.representation.features import FeatureProjection
from hal.representation.features import preprocess
from hal.representation.features import stack_actions
from hal.training.batches import TrainBatch

type ReplayRow = dict[str, np.ndarray | int | str]
type ReplayTransform = Callable[[ReplayRow], ReplayRow]
type ReplayLabels = Callable[[Mapping[str, object]], dict[str, np.ndarray]]
type Window = dict[str, np.ndarray | np.integer]


def relabel_ego(window: dict[str, np.ndarray], ego_prefix: str) -> dict[str, np.ndarray]:
    """Rename p1_*/p2_* keys to ego_*/opp_* based on `ego_prefix`."""
    opp_prefix = "p2" if ego_prefix == "p1" else "p1"
    rel: dict[str, np.ndarray] = {}
    for k, v in window.items():
        if k.startswith(f"{ego_prefix}_"):
            rel[f"ego_{k[3:]}"] = v
        elif k.startswith(f"{opp_prefix}_"):
            rel[f"opp_{k[3:]}"] = v
        else:
            rel[k] = v
    return rel


def make_window(
    sample: dict,
    *,
    ego_prefix: str,
    start: int,
    pad: int,
    length: int,
    projection: FeatureProjection | None,
) -> Window:
    relative = relabel_ego(sample, ego_prefix)
    if projection is not None:
        relative = {k: v for k, v in relative.items() if k in projection.columns}
    else:
        relative.pop("schema_version", None)
    stop = start + length
    out: Window = {}
    for name, values in relative.items():
        real = values[max(0, start) : stop]
        if pad:
            front = np.zeros((pad, *values.shape[1:]), dtype=values.dtype)
            real = np.concatenate([front, real], axis=0)
        out[name] = real
    return out


def choose_chunk_starts(
    T: int,
    L_ctx: int,
    L_chunk: int,
    K: int,
    rng: np.random.Generator,
    *,
    require_full_context: bool = False,
) -> np.ndarray:
    """Up to ``K`` chunk-start positions in ``[1, T - L_chunk]`` whose windows
    ``[cs - L_ctx, cs + L_chunk)`` are pairwise non-overlapping.

    Reading a whole replay off disk to emit a single ~``L_ctx+L_chunk`` window
    wastes ~99% of the bytes read; emitting ``K`` windows amortizes that read.
    The positions are stratified into ``k`` equal lanes (one window per lane,
    randomly placed within it) so the windows spread across the episode and stay
    distinct rather than clustering. ``k`` clamps to what the episode can fit, so
    short replays yield fewer than ``K``. ``K=1`` reduces to a single window drawn
    uniformly over the full range (the historical behavior)."""
    L = L_ctx + L_chunk
    cs_lo, cs_hi = (L_ctx if require_full_context else 1), T - L_chunk
    span = cs_hi - cs_lo + 1
    if span < 1:
        return np.empty(0, dtype=np.int64)
    k = min(K, max(1, span // L))
    stride = span // k
    lane_lo = cs_lo + np.arange(k) * stride
    # Each lane's window must end ``L`` before the next lane starts (non-overlap);
    # the last lane has no successor, so it can range to the end of the episode.
    lane_hi = lane_lo + (stride - L)
    lane_hi[-1] = cs_hi
    lane_hi = np.maximum(lane_hi, lane_lo)
    return lane_lo + rng.integers(0, lane_hi - lane_lo + 1)


def stable_window_rng(seed: int, epoch: int, replay_id: str) -> np.random.Generator:
    """Return a process-independent RNG for one replay in one Mosaic epoch."""
    digest = hashlib.blake2b(replay_id.encode(), digest_size=8).digest()
    identity = int.from_bytes(digest, "little")
    return np.random.default_rng((seed, epoch, identity & 0xFFFFFFFF, identity >> 32))


def collate_windows(batch: list[dict]) -> dict[str, np.ndarray]:
    """Stack a list of ``[seq]`` per-sample windows into ``[B, seq]`` columns."""
    keys = batch[0].keys()
    return {k: np.stack([s[k] for s in batch]) for k in keys}


def collate_train_batch(
    batch: list[dict],
    *,
    stats: dict[str, FeatureStats],
    L_ctx: int,
    extra: ExtraColumns | None = None,
    projection: FeatureProjection | None = None,
) -> TrainBatch:
    """Worker-side collate: stack → ``preprocess`` → split ``[ctx | chunk]``.

    The window the sampler yields is laid out ``[ctx | chunk]`` over
    ``seq = L_ctx + L_chunk`` frames. Context features are the first ``L_ctx``
    frames; the target action chunk is the remaining frames sliced off the
    stacked ego-action channels at ``[L_ctx :]``. Returns a fully-tensorized
    ``TrainBatch`` so the training loop does no reshaping — just ``.to(device)``.

    ``extra`` is the experiment's column routing beyond the built-in feature
    tables (see ``features.ExtraColumns``); the closed-loop policy must carry the
    same one so both observation paths build the same token.
    """
    return train_batch_from_columns(
        collate_windows(batch),
        stats=stats,
        L_ctx=L_ctx,
        extra=extra,
        projection=projection,
    )


def train_batch_from_columns(
    columns: Mapping[str, np.ndarray],
    *,
    stats: dict[str, FeatureStats],
    L_ctx: int,
    extra: ExtraColumns | None = None,
    projection: FeatureProjection | None = None,
) -> TrainBatch:
    """Convert already-stacked ``[B, seq]`` columns into a training batch."""
    ctx_pad = torch.from_numpy(columns["ctx_pad"].astype(np.int64, copy=False))
    feats = preprocess(columns, stats, extra=extra, projection=projection)
    actions = stack_actions(feats)
    context_features = {k: v[:, :L_ctx] for k, v in feats.items()}
    target = actions[:, L_ctx:]
    return TrainBatch(Context(features=context_features, ctx_pad=ctx_pad), target=target)
