"""Audit an R2-staged MDS dataset and publish it immutably."""

import datetime as dt
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

from hal.data.bounded_writer import rclone_copyto
from hal.data.feature_stats import STATS_SCHEMA_VERSION
from hal.data.policy_world_schema import POLICY_WORLD_MDS_COLUMNS
from hal.data.policy_world_schema import POLICY_WORLD_SCHEMA_VERSION
from hal.data.schema import MDS_COLUMNS
from hal.data.schema import SCHEMA_VERSION


def _run(*args: str) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"{' '.join(args)} failed: {detail}")
    return result.stdout


def _objects(prefix: str) -> dict[str, dict[str, Any]]:
    rows = json.loads(_run("rclone", "lsjson", prefix, "--recursive", "--files-only", "--hash"))
    return {str(row["Path"]): row for row in rows}


def _cat_json(path: str) -> dict[str, Any]:
    return json.loads(_run("rclone", "cat", path))


def audit(prefix: str) -> dict[str, Any]:
    objects = _objects(prefix)
    required = {"manifest.jsonl", "stats.json", "failures.materialize.jsonl"}
    missing = sorted(required - objects.keys())
    if missing:
        raise ValueError(f"{prefix}: missing required objects {missing}")

    rows_by_split: dict[str, int] = {}
    shard_count = 0
    stored_format: str | None = None
    for split in ("train", "val", "test"):
        index_name = f"{split}/index.json"
        if index_name not in objects:
            raise ValueError(f"{prefix}: missing {index_name}")
        index = _cat_json(f"{prefix.rstrip('/')}/{index_name}")
        rows_by_split[split] = sum(int(shard["samples"]) for shard in index["shards"])
        for shard in index["shards"]:
            columns = dict(zip(shard["column_names"], shard["column_encodings"], strict=True))
            if columns == MDS_COLUMNS:
                shard_format = "full"
            elif columns == POLICY_WORLD_MDS_COLUMNS:
                shard_format = "policy-world"
            else:
                raise ValueError(f"{prefix}: {split} has an unsupported stored column schema")
            if stored_format is not None and shard_format != stored_format:
                raise ValueError(f"{prefix}: splits use different stored column schemas")
            stored_format = shard_format
            zip_info = shard.get("zip_data")
            if zip_info is None:
                raise ValueError(f"{prefix}: {split} contains an uncompressed shard")
            name = f"{split}/{zip_info['basename']}"
            remote = objects.get(name)
            if remote is None:
                raise ValueError(f"{prefix}: index references missing object {name}")
            if int(remote["Size"]) != int(zip_info["bytes"]):
                raise ValueError(f"{prefix}: size mismatch for {name}")
            expected_md5 = zip_info.get("hashes", {}).get("md5")
            actual_md5 = (remote.get("Hashes") or {}).get("md5") or (remote.get("Hashes") or {}).get("MD5")
            if not expected_md5 or not actual_md5:
                raise ValueError(f"{prefix}: missing MD5 evidence for {name}")
            if expected_md5.lower() != actual_md5.lower():
                raise ValueError(f"{prefix}: MD5 mismatch for {name}")
            shard_count += 1

    if stored_format is None:
        raise ValueError(f"{prefix}: no shard establishes the stored column schema")
    stats = _cat_json(f"{prefix.rstrip('/')}/stats.json")
    if (
        stats.get("schema_version") != STATS_SCHEMA_VERSION
        or stats.get("mds_schema_version") != SCHEMA_VERSION
        or stats.get("split") != "train"
    ):
        raise ValueError(f"{prefix}: stats schema or split differs from the current data contract")
    failure_lines = _run("rclone", "cat", f"{prefix.rstrip('/')}/failures.materialize.jsonl")
    failures = sum(bool(line.strip()) for line in failure_lines.splitlines())
    if stored_format == "policy-world":
        if "projection.json" not in objects:
            raise ValueError(f"{prefix}: policy-world corpus is missing projection.json")
        projection = _cat_json(f"{prefix.rstrip('/')}/projection.json")
        projected = {name: int(value) for name, value in projection["rows"].items()}
        if (
            projection.get("policy_world_schema_version") != POLICY_WORLD_SCHEMA_VERSION
            or projection.get("source_schema_version") != SCHEMA_VERSION
            or projection.get("columns") != POLICY_WORLD_MDS_COLUMNS
            or projected != rows_by_split
            or int(projection.get("failures", -1)) != failures
        ):
            raise ValueError(f"{prefix}: projection differs from the stored policy-world rows")
    elif "projection.json" in objects:
        raise ValueError(f"{prefix}: full replay corpus has an unexpected projection.json")
    with tempfile.TemporaryDirectory(prefix="hal-audit-") as temp:
        manifest = Path(temp) / "manifest.jsonl"
        _run("rclone", "copyto", f"{prefix.rstrip('/')}/manifest.jsonl", str(manifest))
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
        "failures": failures,
    }


def publish_mds(staging: str, final: str, *, purge_staging: bool = True) -> None:
    if not staging.startswith("r2:") or not final.startswith("r2:"):
        raise ValueError("publish_mds requires r2: staging and final prefixes")
    if _objects(final):
        raise FileExistsError(f"final prefix is not empty: {final}")
    before = audit(staging)
    _run("rclone", "copy", staging, final, "--immutable", "--server-side-across-configs")
    after = audit(final)
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
        _run("rclone", "purge", staging)
    logger.info(f"published {staging} -> {final}: {after}")
