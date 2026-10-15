"""Restore the same replay queries with the earlier models' 128-frame inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor

from hal.data import mds
from hal.training import dataloader
from hal.training import ego_stats
from hal.training import features

REFERENCE_SHA256 = "ce30744586fcc6bd5c828dc13f9c1fa930b014465ec157ee5ab506646359f6e2"
STATS_SHA256 = "031dd0e35a7201b299933ec999615f8657dd3e616e0efd3d78b156b1920e0bef"
CONTEXT_LENGTH = 128
WINDOW_COUNT = 256


@dataclass(frozen=True, slots=True)
class WindowIdentity:
    shard: int
    row: int
    start: int
    length: int
    ego: str
    raw_sha256: str
    first_frame: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_source_window(source: Mapping[str, object], start: int, stop: int) -> str:
    """Use the original cohort's names/dtypes/shapes/bytes hash convention."""
    digest = hashlib.sha256()
    for name in sorted(source):
        value = np.asarray(source[name])
        if value.ndim:
            value = np.ascontiguousarray(value[start:stop])
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(json.dumps(value.shape, separators=(",", ":")).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def context_window(source: dict[str, object], identity: WindowIdentity) -> dataloader.Window:
    """Verify the old full window, then retain its final 128 context frames."""
    if identity.ego not in ("p1", "p2") or identity.start < 0 or identity.length != 260:
        raise ValueError("Unsupported original window identity")
    if int(np.asarray(source["schema_version"])) != 7:
        raise ValueError("The raw replay is not canonical schema 7")
    frame = np.asarray(source["frame"])
    if len(frame) < identity.start + identity.length:
        raise ValueError("The source replay does not cover the recorded window")
    if int(frame[identity.start]) != identity.first_frame:
        raise ValueError("The first raw frame differs from the recorded window")
    actual_hash = hash_source_window(source, identity.start, identity.start + identity.length)
    if actual_hash != identity.raw_sha256:
        raise ValueError("The original 260-frame raw window hash differs")
    selected_frame = frame[identity.start + 128 : identity.start + 256]
    if not np.all(np.diff(selected_frame) == 1):
        raise ValueError("The selected context contains a frame gap or restart")
    window = dataloader.make_window(
        source,
        ego_prefix=identity.ego,
        start=identity.start + 128,
        pad=0,
        length=CONTEXT_LENGTH,
        projection=features.BASE_ACTION_PROJECTION,
    )
    window["ctx_pad"] = np.asarray(0, dtype=np.int64)
    return window


def hash_context(context: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(context):
        value = context[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(tuple(value.shape), separators=(",", ":")).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def prepare_inputs(
    reference_path: Path, stats_path: Path, scratch: Path
) -> tuple[dict[str, Tensor], dict[str, object]]:
    if sha256_file(reference_path) != REFERENCE_SHA256 or sha256_file(stats_path) != STATS_SHA256:
        raise ValueError("The frozen reference or ranked statistics hash differs")
    reference = json.loads(reference_path.read_text())
    cohort = reference["cohort"]
    records: list[dict[str, object]] = cohort["windows"]
    if len(records) != WINDOW_COUNT or len({row["replay_id"] for row in records}) != WINDOW_COUNT:
        raise ValueError("The reference must have exactly 256 distinct replays")
    root = Path(cohort["source_root"])
    manifest = Path(cohort["source_manifest"])
    if sha256_file(root / "val/index.json") != cohort["source_index_sha256"]:
        raise ValueError("The source MDS index hash differs")
    if sha256_file(manifest) != cohort["source_manifest_sha256"]:
        raise ValueError("The source manifest hash differs")
    identities = [
        WindowIdentity(
            shard=cast(int, row["shard"]),
            row=cast(int, row["shard_row"]),
            start=cast(int, row["start"]),
            length=cast(int, row["length"]),
            ego=cast(str, row["ego"]),
            raw_sha256=cast(str, row["raw_schema7_window_sha256"]),
            first_frame=cast(int, row["frame_start"]),
        )
        for row in records
    ]
    windows: dict[int, dataloader.Window] = {}
    restored: dict[int, dict[str, object]] = {}
    infos = mds.read_shard_index(root, "val")
    for shard_index, info in enumerate(infos):
        selected = [index for index, identity in enumerate(identities) if identity.shard == shard_index]
        if not selected:
            continue
        with mds.open_shard(root, "val", info, scratch) as reader:
            for index in selected:
                identity = identities[index]
                source = dict(reader[identity.row])
                windows[index] = context_window(source, identity)
                restored[index] = {
                    **records[index],
                    "context_start": identity.start + 128,
                    "context_length": CONTEXT_LENGTH,
                    "query_frame": int(np.asarray(source["frame"])[identity.start + 255]),
                    "raw_128_frame_sha256": hash_source_window(source, identity.start + 128, identity.start + 256),
                }
        print(f"Restored shard {shard_index}: {len(selected)} exact replay windows", flush=True)
    if sorted(windows) != list(range(WINDOW_COUNT)):
        raise ValueError("Some recorded replay windows were not restored")
    stats = ego_stats.load_consolidated_stats(stats_path)
    batch = dataloader.train_batch_from_columns(
        dataloader.collate_windows([windows[index] for index in range(WINDOW_COUNT)]),
        stats=stats,
        L_ctx=CONTEXT_LENGTH,
        projection=features.BASE_ACTION_PROJECTION,
    )
    context = batch.context.features
    for name, value in context.items():
        if value.shape != (WINDOW_COUNT, CONTEXT_LENGTH) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid prepared input {name}: shape={tuple(value.shape)}")
    if torch.count_nonzero(batch.context.ctx_pad).item():
        raise ValueError("The restored contexts must not have padding")
    for index in range(WINDOW_COUNT):
        restored[index]["prepared_context_sha256"] = hash_context(
            {key: value[index] for key, value in context.items()}
        )
    return context, {
        "schema_version": 1,
        "reference_path": str(reference_path.resolve()),
        "reference_sha256": REFERENCE_SHA256,
        "original_cohort_sha256": cohort["package_sha256"],
        "source_manifest_sha256": cohort["source_manifest_sha256"],
        "source_index_sha256": cohort["source_index_sha256"],
        "stats_path": str(stats_path.resolve()),
        "stats_sha256": STATS_SHA256,
        "projection": "BASE_ACTION_PROJECTION",
        "feature_names": sorted(context),
        "context_length": CONTEXT_LENGTH,
        "query_token_position": CONTEXT_LENGTH - 1,
        "all_original_raw_window_hashes_match": True,
        "prepared_context_sha256": hash_context(context),
        "windows": [restored[index] for index in range(WINDOW_COUNT)],
        "generator_sha256": sha256_file(Path(__file__).resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=False)
    context, metadata = prepare_inputs(args.reference, args.stats, args.scratch)
    package_path = output / "inputs.pt"
    with package_path.open("xb") as target:
        torch.save({"context_features": context, "metadata_json": json.dumps(metadata, sort_keys=True)}, target)
    metadata["package_sha256"] = sha256_file(package_path)
    with (output / "inputs.json").open("x") as target:
        json.dump(metadata, target, indent=2, sort_keys=True, allow_nan=False)
        target.write("\n")
    print(json.dumps({"output": str(output), "package_sha256": metadata["package_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
