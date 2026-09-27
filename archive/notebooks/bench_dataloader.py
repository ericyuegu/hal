"""Measure loader latency and replay diversity without running a model."""

import hashlib
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from hal.training.dataloader import make_loader
from hal.training.ego_stats import load_consolidated_stats
from hal.training.features import BASE_ACTION_PROJECTION
from hal.training.features import TrainBatch
from hal.training.replay_reservoir import make_reservoir_loader


@dataclass(frozen=True, slots=True)
class Args:
    source: Path
    compact: Path
    mode: Literal["all", "v7", "compact-k2", "compact-reservoir"] = "all"
    source_remote: str | None = None
    compact_remote: str | None = None
    batches: int = 32
    batch_size: int = 128
    num_workers: int = 4
    prefetch_factor: int = 2
    predownload: int = 512
    cache_limit_gb: int = 128
    L_ctx: int = 256
    L_chunk: int = 13
    source_windows_per_replay: int = 2
    reservoir_windows_per_replay: int = 4
    reservoir_capacity: int = 256
    seed: int = 0
    schema_version: int = 7


def _tensor_bytes(batch: TrainBatch) -> int:
    tensors = [*batch.context.features.values(), batch.context.ctx_pad, batch.target]
    return sum(value.numel() * value.element_size() for value in tensors)


def _batch_hash(batch: TrainBatch) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for name, value in sorted(batch.context.features.items()):
        digest.update(name.encode())
        digest.update(value.numpy().tobytes())
    digest.update(batch.context.ctx_pad.numpy().tobytes())
    digest.update(batch.target.numpy().tobytes())
    return digest.hexdigest()


def _cache_bytes(root: Path) -> dict[str, int]:
    sizes = {"raw": 0, "compressed": 0}
    if not root.exists():
        return sizes
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.endswith(".mds"):
            sizes["raw"] += path.stat().st_size
        elif path.name.endswith(".mds.zstd"):
            sizes["compressed"] += path.stat().st_size
    return sizes


def _measure(name: str, loader, batches: int, cache_root: Path) -> dict[str, object]:
    cache_before = _cache_bytes(cache_root)
    iterator = iter(loader)
    durations = []
    sizes = []
    diversity = []
    seen_replays: set[str] = set()
    last_seen: dict[str, int] = {}
    reuse_gaps = []
    hashes = []
    hash_seconds = 0.0
    for _ in range(batches):
        start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        durations.append(time.perf_counter() - start)
        sizes.append(batch.target.shape[0])
        if batch.replay_ids is not None:
            diversity.append(len(set(batch.replay_ids)))
            for replay_id in batch.replay_ids:
                if replay_id in last_seen:
                    reuse_gaps.append(len(durations) - 1 - last_seen[replay_id])
                last_seen[replay_id] = len(durations) - 1
                seen_replays.add(replay_id)
        hash_start = time.perf_counter()
        hashes.append(_batch_hash(batch))
        hash_seconds += time.perf_counter() - hash_start
        if not torch.isfinite(batch.target).all():
            raise ValueError(f"{name}: target contains a nonfinite value")
    if not durations:
        raise RuntimeError(f"{name}: loader yielded no batches")
    cache_after = _cache_bytes(cache_root)
    return {
        "name": name,
        "batches": len(durations),
        "samples": sum(sizes),
        "first_batch_seconds": durations[0],
        "later_batch_seconds_mean": statistics.fmean(durations[1:]) if len(durations) > 1 else None,
        "samples_per_second": sum(sizes) / sum(durations),
        "batch_tensor_mb": _tensor_bytes(batch) / 1e6,
        "distinct_replays_min": min(diversity) if diversity else None,
        "distinct_replays_total": len(seen_replays) if diversity else None,
        "replay_reuse_gap_min": min(reuse_gaps) if reuse_gaps else None,
        "replay_reuse_gap_median": statistics.median(reuse_gaps) if reuse_gaps else None,
        "replay_reuse_gap_two_fraction": (
            sum(gap == 2 for gap in reuse_gaps) / len(reuse_gaps) if reuse_gaps else None
        ),
        "batch_hashes": hashes,
        "batch_hash_seconds": hash_seconds,
        "cache_raw_bytes_added": cache_after["raw"] - cache_before["raw"],
        "cache_compressed_bytes_added": cache_after["compressed"] - cache_before["compressed"],
    }


def main(args: Args) -> None:
    stats = load_consolidated_stats(args.source / "stats.json")
    common = dict(
        split="train",
        stats=stats,
        L_ctx=args.L_ctx,
        L_chunk=args.L_chunk,
        batch_size=args.batch_size,
        seed=args.seed,
        shuffle_block_size=2_000,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        schema_version=args.schema_version,
        projection=BASE_ACTION_PROJECTION,
    )
    loaders = [
        (
            "v7",
            args.source,
            lambda: make_loader(
                data_root=str(args.source),
                remote=args.source_remote,
                cache_limit=f"{args.cache_limit_gb}gb",
                predownload=args.predownload,
                windows_per_replay=args.source_windows_per_replay,
                **common,
            ),
        ),
        (
            "compact-k2",
            args.compact,
            lambda: make_loader(
                data_root=str(args.compact),
                remote=args.compact_remote,
                cache_limit=f"{args.cache_limit_gb}gb",
                predownload=args.predownload,
                windows_per_replay=args.source_windows_per_replay,
                compact=True,
                **common,
            ),
        ),
        (
            "compact-reservoir",
            args.compact,
            lambda: make_reservoir_loader(
                data_root=str(args.compact),
                remote=args.compact_remote,
                cache_limit=f"{args.cache_limit_gb}gb",
                predownload=args.predownload,
                windows_per_replay=args.reservoir_windows_per_replay,
                reservoir_capacity=args.reservoir_capacity,
                **common,
            ),
        ),
    ]
    for name, cache_root, make in loaders:
        if args.mode != "all" and name != args.mode:
            continue
        loader = make()
        print(json.dumps(_measure(name, loader, args.batches, cache_root), sort_keys=True), flush=True)
        del loader


if __name__ == "__main__":
    main(tyro.cli(Args))
