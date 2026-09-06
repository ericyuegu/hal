"""Audit an R2-staged MDS dataset and publish it immutably."""

import datetime as dt
import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import tyro
from loguru import logger

from hal import r2
from hal.data.bounded_writer import rclone_copyto


def _objects(prefix: str) -> dict[str, dict[str, Any]]:
    rows = json.loads(r2.run_rclone("lsjson", prefix, "--recursive", "--files-only", "--hash"))
    return {str(row["Path"]): row for row in rows}


def _cat_json(path: str) -> dict[str, Any]:
    return json.loads(r2.run_rclone("cat", path))


def audit(prefix: str) -> dict[str, Any]:
    objects = _objects(prefix)
    required = {"manifest.jsonl", "stats.json", "projection.json", "failures.materialize.jsonl"}
    missing = sorted(required - objects.keys())
    if missing:
        raise ValueError(f"{prefix}: missing required objects {missing}")

    rows_by_split: dict[str, int] = {}
    shard_count = 0
    for split in ("train", "val", "test"):
        index_name = f"{split}/index.json"
        if index_name not in objects:
            raise ValueError(f"{prefix}: missing {index_name}")
        index = _cat_json(f"{prefix.rstrip('/')}/{index_name}")
        rows_by_split[split] = sum(int(shard["samples"]) for shard in index["shards"])
        for shard in index["shards"]:
            zip_info = shard.get("zip_data")
            data_info = zip_info if zip_info is not None else shard.get("raw_data")
            if data_info is None:
                raise ValueError(f"{prefix}: {split} shard has no stored data object")
            name = f"{split}/{data_info['basename']}"
            remote = objects.get(name)
            if remote is None:
                raise ValueError(f"{prefix}: index references missing object {name}")
            if int(remote["Size"]) != int(data_info["bytes"]):
                raise ValueError(f"{prefix}: size mismatch for {name}")
            expected_md5 = data_info.get("hashes", {}).get("md5")
            actual_md5 = (remote.get("Hashes") or {}).get("md5") or (remote.get("Hashes") or {}).get("MD5")
            if not expected_md5 or not actual_md5:
                raise ValueError(f"{prefix}: missing MD5 evidence for {name}")
            if expected_md5.lower() != actual_md5.lower():
                raise ValueError(f"{prefix}: MD5 mismatch for {name}")
            shard_count += 1

    projection = _cat_json(f"{prefix.rstrip('/')}/projection.json")
    projected = {name: int(value) for name, value in projection["rows"].items()}
    if projected != rows_by_split:
        raise ValueError(f"{prefix}: projection rows {projected} != indexes {rows_by_split}")
    with tempfile.TemporaryDirectory(prefix="hal-audit-") as temp:
        manifest = Path(temp) / "manifest.jsonl"
        r2.copy_file(f"{prefix.rstrip('/')}/manifest.jsonl", manifest)
        manifest_rows: dict[str, list[int]] = {"train": [], "val": [], "test": []}
        with manifest.open() as handle:
            for line in handle:
                row = json.loads(line)
                annotation = row.get("annotation")
                if annotation is None:
                    raise ValueError(f"{prefix}: manifest contains an unannotated row")
                manifest_rows[annotation["split"]].append(int(annotation["mds_row_idx"]))
        for split, indices in manifest_rows.items():
            if sorted(indices) != list(range(rows_by_split[split])):
                raise ValueError(f"{prefix}: {split} manifest row indexes are not contiguous")
    return {
        "rows": rows_by_split,
        "shards": shard_count,
        "objects": len(objects),
        "bytes": sum(int(row["Size"]) for row in objects.values()),
        "failures": int(projection.get("failures", 0)),
        "object_hashes": {
            name: {
                "bytes": int(row["Size"]),
                "md5": (row.get("Hashes") or {}).get("md5") or (row.get("Hashes") or {}).get("MD5"),
            }
            for name, row in sorted(objects.items())
        },
    }


def publish_mds(
    staging: str,
    final: str,
    *,
    purge_staging: bool = True,
    audit_fn: Callable[[str], dict[str, Any]] | None = None,
) -> None:
    if not staging.startswith("r2:") or not final.startswith("r2:"):
        raise ValueError("publish_mds requires r2: staging and final prefixes")
    if _objects(final):
        raise FileExistsError(f"final prefix is not empty: {final}")
    run_audit = audit if audit_fn is None else audit_fn
    before = run_audit(staging)
    r2.run_rclone("copy", staging, final, "--immutable", "--server-side-across-configs")
    after = run_audit(final)
    if after != before:
        raise ValueError(f"published audit differs: staging={before}, final={after}")

    success = {
        "published_at": dt.datetime.now(dt.UTC).isoformat(),
        "staging": staging,
        "final": final,
        **after,
    }
    with tempfile.TemporaryDirectory(prefix="hal-publish-") as temp:
        marker = Path(temp) / "_SUCCESS"
        marker.write_text(json.dumps(success, indent=2, sort_keys=True) + "\n")
        rclone_copyto(marker, f"{final.rstrip('/')}/_SUCCESS")
    if purge_staging:
        r2.run_rclone("purge", staging)
    logger.info(f"published {staging} -> {final}: {after}")


if __name__ == "__main__":
    tyro.cli(publish_mds)
