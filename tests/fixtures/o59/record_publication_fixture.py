"""Verify and record the immutable 059 full-MDS publication fixture."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def _objects(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["Path"]): row for row in json.loads(path.read_text())}


def main(run_root: Path, output: Path) -> None:
    staging = _objects(run_root / "staging-objects.json")
    final = _objects(run_root / "final-objects.json")
    marker_path = run_root / "_SUCCESS"
    marker = json.loads(marker_path.read_text())
    if marker["staging"] == marker["final"] or not marker["final"].endswith("/final"):
        raise ValueError("publication marker does not identify separate staging and final prefixes")
    if set(final) != set(staging) | {"_SUCCESS"}:
        raise ValueError("final object names differ from staging plus _SUCCESS")

    local_objects: dict[str, dict[str, int | str]] = {}
    for name, remote in sorted(staging.items()):
        local = run_root / "mds" / name
        if not local.is_file():
            raise FileNotFoundError(local)
        md5 = _md5(local)
        size = local.stat().st_size
        if size != remote["Size"] or md5 != remote["Hashes"]["md5"]:
            raise ValueError(f"staging differs from local MDS object {name}")
        published = final[name]
        if size != published["Size"] or md5 != published["Hashes"]["md5"]:
            raise ValueError(f"final differs from local MDS object {name}")
        local_objects[name] = {"bytes": size, "md5": md5, "sha256": _sha256(local)}

    if _md5(marker_path) != final["_SUCCESS"]["Hashes"]["md5"]:
        raise ValueError("final success marker differs from retained local marker")
    if marker["objects"] != len(staging) or marker["bytes"] != sum(
        int(item["bytes"]) for item in local_objects.values()
    ):
        raise ValueError("publication marker disagrees with staged object count or bytes")
    if marker["rows"] != {"train": 2, "val": 2, "test": 2} or marker["failures"]:
        raise ValueError("publication marker does not describe the prepared fixture")

    checkout = Path(__file__).resolve().parents[3]
    inputs = {
        "source_archive": checkout / "data/raw/dev.7z",
        "index": run_root / "index.jsonl",
        "eligible_paths": run_root / "eligible-paths.txt",
        "selected_paths": run_root / "paths.txt",
        "selected_replays": run_root / "paths.jsonl",
    }
    modules = (
        "hal/data/index_builder.py",
        "hal/data/replay_selection.py",
        "hal/data/mds_materialization.py",
        "hal/data/mds_publication.py",
        "tests/fixtures/o59/prepare_publication_fixture.py",
    )
    source_sha = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    report = {
        "source_git_sha": source_sha,
        "purpose": "small .slp-derived full-MDS publication fixture; not the 44-source 059 corpus",
        "source_file_sha256": {name: _sha256(path) for name, path in inputs.items()},
        "candidate_module_sha256": {name: _sha256(checkout / name) for name in modules},
        "local_objects": local_objects,
        "staging_prefix": marker["staging"],
        "final_prefix": marker["final"],
        "staging_object_count": len(staging),
        "final_object_count": len(final),
        "success_marker": marker,
        "success_marker_sha256": _sha256(marker_path),
        "audit_before_marker": {
            "rows": marker["rows"],
            "shards": marker["shards"],
            "objects": marker["objects"],
            "bytes": marker["bytes"],
            "failures": marker["failures"],
        },
        "audit_after_marker": {
            "rows": marker["rows"],
            "shards": marker["shards"],
            "objects": len(final),
            "bytes": sum(int(item["Size"]) for item in final.values()),
            "failures": marker["failures"],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"final": marker["final"], "objects_verified": len(staging)}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.run_root, args.output)
