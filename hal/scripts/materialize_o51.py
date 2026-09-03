"""Materialize O51's four disjoint, training-optimized MDS bands."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro
from streaming import MDSWriter
from streaming import StreamingDataset
from tqdm import tqdm

from hal import r2
from hal import streams
from hal.data.feature_stats import dump_finalized_stats
from hal.data.o51 import DEFAULT_BAND_ROOT
from hal.data.o51 import DEFAULT_O48_INDEX
from hal.data.o51 import DEFAULT_O48_REMOTE
from hal.data.o51 import TIER_SCALES
from hal.data.o51 import build_nested_corpus
from hal.data.o51 import read_o48_inventory
from hal.data.o51 import write_band_manifests
from hal.data.o51_schema import O51_MDS_COLUMNS
from hal.data.o51_schema import O51_RETURN_SUFFIX
from hal.training.ego_stats import load_consolidated_mixture_stats
from hal.training.o51_data import encode_o51_replay
from hal.training.player_identity import ReplayPlayerLookup
from hal.training.player_identity import load_player_identity_sidecar

DEFAULT_PLAYER_SIDECAR = Path("data/processed/player-identity-v1/professional-code-v1.jsonl.gz")
DEFAULT_PLAYER_SIDECAR_REMOTE = "s3://hal/processed/player-identity-v1/professional-code-v1.jsonl.gz"
DEFAULT_SHARD_SIZE = 256 * 2**20


@dataclass(frozen=True, slots=True)
class MaterializeArgs:
    inventory: Path = DEFAULT_O48_INDEX
    inventory_remote: str = DEFAULT_O48_REMOTE
    output: Path = DEFAULT_BAND_ROOT
    output_remote: str | None = None
    player_sidecar: Path = DEFAULT_PLAYER_SIDECAR
    player_sidecar_remote: str = DEFAULT_PLAYER_SIDECAR_REMOTE
    cache_limit: str = "2500gb"
    predownload: int = 2048
    shard_size: int = DEFAULT_SHARD_SIZE
    gamma: float = 0.99618
    damage_shaping: float = 1.0
    win_reward: float = 50.0
    stock_value: float = 120.0
    max_rows_per_band: int | None = None


def _ensure_s3_artifact(path: Path, remote: str) -> Path:
    """Download one missing immutable sidecar with an atomic local publish."""
    if path.is_file():
        return path
    if not remote.startswith("s3://"):
        raise ValueError(f"expected an s3:// artifact URI, got {remote!r}")
    bucket, separator, key = remote.removeprefix("s3://").partition("/")
    if not separator or not bucket or not key:
        raise ValueError(f"invalid S3 artifact URI {remote!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        r2.client().download_file(bucket, key, str(temporary))
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _dataset(source: streams.StreamSource, args: MaterializeArgs) -> StreamingDataset:
    remote, local = source.for_split("train")
    return StreamingDataset(
        remote=remote,
        local=str(local),
        batch_size=32,
        shuffle=False,
        cache_limit=args.cache_limit,
        predownload=args.predownload,
        download_retry=8,
    )


def _remote_parts(remote: str) -> tuple[str, str]:
    if not remote.startswith("s3://"):
        raise ValueError(f"expected an s3:// output URI, got {remote!r}")
    bucket, separator, prefix = remote.removeprefix("s3://").partition("/")
    if not separator or not bucket or not prefix:
        raise ValueError(f"invalid S3 output URI {remote!r}")
    return bucket, prefix.rstrip("/")


def _ensure_empty_remote(remote: str) -> None:
    bucket, prefix = _remote_parts(remote)
    response = r2.client().list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/", MaxKeys=1)
    if response.get("KeyCount", len(response.get("Contents", ()))):
        raise FileExistsError(f"O51 remote output already exists: {remote}")


def _bridge_streaming_endpoint() -> None:
    """Give Mosaic's uploader the standard R2 endpoint used by HAL."""
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    if not endpoint:
        raise RuntimeError("remote O51 output requires AWS_ENDPOINT_URL")
    os.environ.setdefault("S3_ENDPOINT_URL", endpoint)


def _writer_output(path: Path, remote: str | None, scale: int) -> str | tuple[str, str]:
    if remote is None:
        return str(path)
    return str(path), f"{remote.rstrip('/')}/band-{scale}/train"


def _shard_summary(shards: list[dict[str, Any]]) -> tuple[int, int, int]:
    return (
        sum(int(shard["raw_data"]["bytes"]) for shard in shards),
        max((int(shard["samples"]) for shard in shards), default=0),
        len(shards),
    )


def _upload_artifacts(root: Path, remote: str, paths: list[Path]) -> None:
    bucket, prefix = _remote_parts(remote)
    client = r2.client()
    for path in paths:
        relative = path.relative_to(root).as_posix()
        client.upload_file(str(path), bucket, f"{prefix}/{relative}")


def materialize_o51(args: MaterializeArgs) -> dict[str, Any]:
    """Build local bands in hash order and return their complete audit."""
    if args.predownload < 1 or args.shard_size < 1:
        raise ValueError("predownload and shard_size must be positive")
    if args.max_rows_per_band is not None and args.max_rows_per_band < 1:
        raise ValueError("max_rows_per_band must be positive")
    train_paths = {scale: args.output / f"band-{scale}" / "train" for scale in TIER_SCALES}
    existing = [path for path in train_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"O51 output already exists: {existing[0]}")
    if args.output_remote is not None:
        _ensure_empty_remote(args.output_remote)
        _bridge_streaming_endpoint()

    inventory_path = _ensure_s3_artifact(args.inventory, args.inventory_remote)
    sidecar_path = _ensure_s3_artifact(args.player_sidecar, args.player_sidecar_remote)
    inventory = read_o48_inventory(inventory_path)
    corpus = build_nested_corpus(inventory, strict_official=args.max_rows_per_band is None)
    manifests = write_band_manifests(corpus, args.output)
    sidecar = load_player_identity_sidecar(sidecar_path)
    player_lookup = ReplayPlayerLookup(sidecar.by_replay)
    source_datasets: dict[str, StreamingDataset] = {}
    source_report: dict[str, Counter[str]] = {source.name: Counter() for source in streams.POLICY_WORLD_V7_SOURCES}
    source_band_report: dict[str, dict[int, Counter[str]]] = {
        source.name: {scale: Counter() for scale in TIER_SCALES} for source in streams.POLICY_WORLD_V7_SOURCES
    }
    band_report: dict[int, Counter[str]] = {scale: Counter() for scale in TIER_SCALES}

    with ExitStack() as stack:
        writers = {
            scale: stack.enter_context(
                MDSWriter(
                    out=_writer_output(path, args.output_remote, scale),
                    keep_local=args.output_remote is None,
                    columns=O51_MDS_COLUMNS,
                    compression="zstd",
                    hashes=["md5", "sha256"],
                    size_limit=args.shard_size,
                    exist_ok=False,
                )
            )
            for scale, path in train_paths.items()
        }
        for scale in TIER_SCALES:
            entries = corpus.bands[scale].entries
            if args.max_rows_per_band is not None:
                entries = entries[: args.max_rows_per_band]
            for entry in tqdm(entries, desc=f"O51 band {scale}", unit="replay"):
                source = streams.BY_NAME[entry.source]
                dataset = source_datasets.get(entry.source)
                if dataset is None:
                    dataset = _dataset(source, args)
                    source_datasets[entry.source] = dataset
                compact = dataset[entry.row]
                replay_id = str(compact["replay_id"])
                frames = int(np.asarray(compact["num_frames"]).item())
                if replay_id != entry.replay_id or frames != entry.frames:
                    raise ValueError(
                        f"{entry.source} row {entry.row}: inventory says {(entry.replay_id, entry.frames)}, "
                        f"MDS says {(replay_id, frames)}"
                    )
                encoded = encode_o51_replay(
                    compact,
                    player_labels=player_lookup(compact),
                    gamma=args.gamma,
                    damage_shaping=args.damage_shaping,
                    win_reward=args.win_reward,
                    stock_value=args.stock_value,
                )
                writers[scale].write(encoded)

                p1_valid = np.asarray(encoded[f"p1_{O51_RETURN_SUFFIX}_valid"], dtype=np.uint8)
                p2_valid = np.asarray(encoded[f"p2_{O51_RETURN_SUFFIX}_valid"], dtype=np.uint8)
                terminal = bool(p1_valid.all() and p2_valid.all())
                identities = sum(int(encoded[f"p{port}_player_id"]) != 0 for port in (1, 2))
                counts = Counter(
                    replays=1,
                    frames=frames,
                    potential_targets=2 * (frames - 1),
                    terminal_replays=terminal,
                    truncated_replays=not terminal,
                    identity_sides=identities,
                    awr_eligible_sides=2 * terminal,
                    awr_eligible_frames=int(p1_valid.sum()) + int(p2_valid.sum()),
                )
                source_report[entry.source].update(counts)
                source_band_report[entry.source][scale].update(counts)
                band_report[scale].update(counts)

    raw_sizes: dict[int, int] = {}
    largest_shard_samples: dict[int, int] = {}
    shard_counts: dict[int, int] = {}
    for scale, writer in writers.items():
        raw_sizes[scale], largest_shard_samples[scale], shard_counts[scale] = _shard_summary(writer.shards)
    u0 = band_report[1]
    u0_ineligible_fraction = 1 - u0["terminal_replays"] / max(u0["replays"], 1)
    normalization_path = args.output / "normalization-stats.json"
    # Match O50's natural replay sampling, using the content-unique full-tier
    # counts for the fixed mixture of O50's per-source statistics.
    normalization_weights = [
        float(source_report[source.name]["replays"]) for source in streams.POLICY_WORLD_V7_SOURCES
    ]
    normalization = load_consolidated_mixture_stats(
        [source.local_root / "stats.json" for source in streams.POLICY_WORLD_V7_SOURCES],
        normalization_weights,
        expected_mds_schema_version=7,
    )
    dump_finalized_stats(normalization_path, normalization, mds_schema_version=7)
    normalization_sha256 = hashlib.sha256(normalization_path.read_bytes()).hexdigest()
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "o51-nested-v1",
        "corpus_hash": corpus.corpus_hash,
        "band_manifests": [path.relative_to(args.output).as_posix() for path in manifests],
        "band_hashes": {scale: corpus.bands[scale].sha256 for scale in TIER_SCALES},
        "tier_hashes": {scale: corpus.tiers[scale].sha256 for scale in TIER_SCALES},
        "bands": {scale: dict(band_report[scale]) for scale in TIER_SCALES},
        "sources": {name: dict(values) for name, values in source_report.items()},
        "source_bands": {
            name: {scale: dict(values) for scale, values in bands.items()}
            for name, bands in source_band_report.items()
        },
        "raw_band_bytes": raw_sizes,
        "largest_shard_samples": largest_shard_samples,
        "shards": shard_counts,
        "u0_awr_ineligible_fraction": u0_ineligible_fraction,
        "terminal_only_proxy_required": u0_ineligible_fraction > 0.05,
        "normalization_stats": str(normalization_path),
        "normalization_stats_sha256": normalization_sha256,
        "normalization_weighting": "content-unique-replays",
        "player_sidecar_sha256": sidecar.sha256,
        "player_vocabulary_sha256": sidecar.vocabulary.sha256,
        "parameters": {
            "gamma": args.gamma,
            "damage_shaping": args.damage_shaping,
            "win_reward": args.win_reward,
            "stock_value": args.stock_value,
        },
    }
    metadata = args.output / "materialization.json"
    metadata.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.output_remote is not None:
        _upload_artifacts(
            args.output,
            args.output_remote,
            [*manifests, normalization_path, metadata],
        )
    return report


def main(args: MaterializeArgs) -> None:
    print(json.dumps(materialize_o51(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(MaterializeArgs))
