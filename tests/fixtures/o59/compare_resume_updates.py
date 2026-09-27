"""Compare all saved fields from the control and candidate resume captures."""

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import fields
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read_capture(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads((root / "report.json").read_text())
    snapshot_path = root / "next-update.pt"
    if _sha256(snapshot_path) != report["snapshot_sha256"]:
        raise ValueError(f"{root}: capture content differs from its report")
    snapshot = torch.load(snapshot_path, map_location="cpu", mmap=True, weights_only=False)
    return snapshot, report


def _equal_tensor_with_paired_nans(expected: torch.Tensor, actual: torch.Tensor) -> bool:
    if torch.equal(expected, actual):
        return True
    if not expected.is_floating_point():
        return False
    missing = torch.isnan(expected)
    if not bool(missing.any()) or not torch.equal(missing, torch.isnan(actual)):
        return False
    return torch.equal(expected[~missing], actual[~missing])


def _compare(expected: Any, actual: Any, path: str, mismatches: list[str]) -> int:
    if type(expected) is not type(actual):
        mismatches.append(f"{path}: type changed")
        return 1
    if isinstance(expected, torch.Tensor):
        if (
            expected.dtype != actual.dtype
            or expected.shape != actual.shape
            or not _equal_tensor_with_paired_nans(expected, actual)
        ):
            mismatches.append(f"{path}: tensor changed")
        return 1
    if isinstance(expected, np.ndarray):
        if (
            expected.dtype != actual.dtype
            or expected.shape != actual.shape
            or not (
                np.array_equal(expected, actual)
                or (np.issubdtype(expected.dtype, np.floating) and np.array_equal(expected, actual, equal_nan=True))
            )
        ):
            mismatches.append(f"{path}: array changed")
        return 1
    if isinstance(expected, Mapping):
        if expected.keys() != actual.keys():
            mismatches.append(f"{path}: mapping keys changed")
        return sum(
            _compare(value, actual[key], f"{path}/{key}", mismatches)
            for key, value in expected.items()
            if key in actual
        )
    if isinstance(expected, (tuple, list)):
        if len(expected) != len(actual):
            mismatches.append(f"{path}: sequence length changed")
        return sum(
            _compare(left, right, f"{path}/{idx}", mismatches)
            for idx, (left, right) in enumerate(zip(expected, actual, strict=False))
        )
    if is_dataclass(expected) and not isinstance(expected, type):
        return sum(
            _compare(getattr(expected, field.name), getattr(actual, field.name), f"{path}/{field.name}", mismatches)
            for field in fields(expected)
        )
    if expected != actual:
        mismatches.append(f"{path}: value changed")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    expected, control_report = _read_capture(args.control)
    actual, candidate_report = _read_capture(args.candidate)
    mismatches: list[str] = []
    leaves = _compare(expected, actual, "snapshot", mismatches)
    report = {
        "status": "passed" if not mismatches else "failed",
        "comparison": "exact non-NaN tensor/array values, paired NaNs, dtypes, shapes, and recursive saved records",
        "leaf_count": leaves,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "control": control_report,
        "candidate": candidate_report,
        "comparison_source_sha256": _sha256(Path(__file__)),
    }
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "mismatches"}), flush=True)
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
