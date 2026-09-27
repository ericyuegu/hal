"""Audit published v8 corpora one at a time with bounded NVMe scratch."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import cast

FIRST_LARGE_CORPUS = "professional-aklo-policy-world-v8"
COMPLETED_PILOT = "professional-rapm-policy-world-v8"


@dataclass(frozen=True, slots=True)
class Args:
    repo: Path
    inventory: Path
    rapm_result: Path
    output_root: Path
    scratch_root: Path
    rclone_config: Path
    attempt: int = 1
    max_corpora: int | None = None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_result(path: Path, expected: dict[str, Any], source_order: list[str]) -> dict[str, object]:
    report = cast(dict[str, Any], json.loads(path.read_text()))
    if (
        report.get("mode") != "full"
        or report.get("selected_corpus") != expected["name"]
        or report.get("source_count") != 1
        or report.get("source_order") != source_order
        or report.get("total_compressed_bytes") != expected["compressed_bytes"]
        or len(report.get("sources", [])) != 1
    ):
        raise ValueError(f"{path}: full-row report identity differs from published inventory")
    result = cast(dict[str, Any], report["sources"][0])
    if result.get("marker_sha256") != expected["marker_sha256"] or result.get("name") != expected["name"]:
        raise ValueError(f"{path}: immutable publication marker differs")
    published = cast(dict[str, Any], result["published_audit"])
    recomputed = cast(dict[str, Any], result["recomputed_audit"])
    keys = ("rows", "objects", "bytes") if result["marker_format"] == "full-mds" else tuple(published)
    if {key: recomputed[key] for key in keys} != {key: published[key] for key in keys}:
        raise ValueError(f"{path}: recomputed row audit differs from the publication marker")
    return {
        "name": expected["name"],
        "result": str(path),
        "result_sha256": _sha256(path),
        "marker_sha256": expected["marker_sha256"],
        "compressed_bytes": expected["compressed_bytes"],
        "elapsed_seconds": report["elapsed_seconds"],
        "marker_format": result["marker_format"],
        "draft_exception": result["draft_exception"],
        "rows": recomputed["rows"],
        "recomputed_audit": recomputed,
    }


def _ordered_sources(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    sources = cast(list[dict[str, Any]], inventory["sources"])
    if inventory.get("source_count") != 44 or len(sources) != 44:
        raise ValueError("published v8 inventory does not contain exactly 44 sources")
    by_name = {str(source["name"]): source for source in sources}
    source_order = cast(list[str], inventory["source_order"])
    if len(by_name) != 44 or set(by_name) != set(source_order):
        raise ValueError("published v8 inventory source order differs")
    if FIRST_LARGE_CORPUS not in by_name or COMPLETED_PILOT not in by_name:
        raise ValueError("expected Aklo and RapM corpora are absent")
    remaining = [source for source in sources if source["name"] not in (FIRST_LARGE_CORPUS, COMPLETED_PILOT)]
    return [by_name[FIRST_LARGE_CORPUS], *sorted(remaining, key=lambda row: (row["compressed_bytes"], row["name"]))]


def _check_scratch_capacity(path: Path, compressed_bytes: int) -> None:
    free = shutil.disk_usage(path).free
    required = max(8 * 2**30, 4 * compressed_bytes)
    if free < required:
        raise RuntimeError(
            f"{path}: {free / 2**30:.1f} GiB free; conservative per-corpus audit reserve is {required / 2**30:.1f} GiB"
        )


def main(args: Args) -> None:
    if args.attempt < 1 or args.max_corpora is not None and args.max_corpora < 1:
        raise ValueError("attempt and max_corpora must be positive")
    args = Args(
        args.repo.resolve(),
        args.inventory.resolve(),
        args.rapm_result.resolve(),
        args.output_root.resolve(),
        args.scratch_root.resolve(),
        args.rclone_config.resolve(),
        args.attempt,
        args.max_corpora,
    )
    repo = args.repo
    inventory_path = args.inventory
    inventory = cast(dict[str, Any], json.loads(inventory_path.read_text()))
    ordered = _ordered_sources(inventory)
    source_order = cast(list[str], inventory["source_order"])
    by_name = {str(source["name"]): source for source in cast(list[dict[str, Any]], inventory["sources"])}
    if not args.rclone_config.is_file():
        raise FileNotFoundError(args.rclone_config)
    python = repo / ".venv/bin/python"
    helper = repo / "tests/fixtures/o59/audit_published_v8.py"
    if not python.is_file() or not helper.is_file():
        raise FileNotFoundError("the pinned Python environment or v8 audit helper is missing")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    temporary = args.output_root / "tmp"
    temporary.mkdir(exist_ok=True)
    if args.scratch_root.stat().st_dev != repo.stat().st_dev or args.output_root.stat().st_dev != repo.stat().st_dev:
        raise ValueError("audit scratch and evidence output must use the repository's disk-backed filesystem")
    rapm = _validate_result(args.rapm_result, by_name[COMPLETED_PILOT], source_order)

    completed = [rapm]
    new_count = 0
    helper_sha = _sha256(helper)
    identities = {"audit_helper_sha256": helper_sha, "inventory_sha256": _sha256(inventory_path)}
    identity_path = args.output_root / "input-identities.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identities:
            raise ValueError("published audit helper or inventory changed between attempts")
    else:
        with identity_path.open("x") as handle:
            handle.write(json.dumps(identities, indent=2, sort_keys=True) + "\n")
    for source in ordered:
        name = str(source["name"])
        result_path = args.output_root / f"{name}.json"
        if result_path.is_file():
            completed.append(_validate_result(result_path, source, source_order))
            continue
        if args.max_corpora is not None and new_count >= args.max_corpora:
            break
        if _sha256(helper) != helper_sha:
            raise RuntimeError("v8 audit helper changed during the qualification run")
        _check_scratch_capacity(args.scratch_root, int(source["compressed_bytes"]))
        log_path = args.output_root / f"{name}.attempt-{args.attempt}.log"
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "MKL_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "PYTHONPATH": str(repo),
                "TMPDIR": str(temporary),
            }
        )
        command = [
            str(python),
            str(helper),
            "--mode",
            "full",
            "--corpus",
            name,
            "--scratch-root",
            str(args.scratch_root),
            "--rclone-config",
            str(args.rclone_config),
            "--output",
            str(result_path),
        ]
        with log_path.open("x") as handle:
            subprocess.run(command, cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        record = _validate_result(result_path, source, source_order)
        completed.append(record)
        new_count += 1
        print(f"{len(completed)}/44 {name}: {record['elapsed_seconds']:.1f}s", flush=True)

    summary = {
        "scope": "read-only full row, metadata, checksum, and statistics audit of published v8 corpora",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "inventory_path": str(inventory_path),
        "inventory_sha256": identities["inventory_sha256"],
        "audit_helper_sha256": helper_sha,
        "source_order": source_order,
        "execution_order": [source["name"] for source in ordered],
        "completed_count": len(completed),
        "new_count": new_count,
        "remaining_names": sorted(set(source_order) - {cast(str, row["name"]) for row in completed}),
        "completed": completed,
    }
    summary_path = args.output_root / f"summary-attempt-{args.attempt}.json"
    with summary_path.open("x") as handle:
        handle.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Audited {len(completed)}/44 published corpora; summary: {summary_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--inventory", type=Path, default=Path("runs/refactor-059/v8-published-inventory.json"))
    parser.add_argument("--rapm-result", type=Path, default=Path("runs/refactor-059/v8-full-rapm.json"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--rclone-config", type=Path, default=Path.home() / ".config/rclone/rclone.conf")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--max-corpora", type=int)
    values = parser.parse_args()
    main(
        Args(
            values.repo,
            values.inventory,
            values.rapm_result,
            values.output_root,
            values.scratch_root,
            values.rclone_config,
            values.attempt,
            values.max_corpora,
        )
    )
