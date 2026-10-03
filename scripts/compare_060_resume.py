"""Compare uninterrupted and interrupted/resumed experiment 060 checkpoints."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated
from typing import Any
from typing import cast

import numpy as np
import torch
import tyro


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right, equal_nan=True)
    if isinstance(left, Mapping):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, Sequence) and not isinstance(left, str | bytes):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right, strict=True))
    return bool(left == right)


@dataclass(frozen=True, slots=True)
class Args:
    uninterrupted: Annotated[Path, tyro.conf.Positional]
    resumed: Annotated[Path, tyro.conf.Positional]
    output: Annotated[Path, tyro.conf.Positional]


def main(args: Args) -> None:
    if args.output.exists():
        raise FileExistsError(f"immutable comparison output already exists: {args.output}")
    control = torch.load(args.uninterrupted, map_location="cpu", weights_only=False)
    resumed = torch.load(args.resumed, map_location="cpu", weights_only=False)
    if control["step"] != resumed["step"] or control["cfg"] != resumed["cfg"]:
        raise ValueError("checkpoints do not represent the same update and treatment")
    control_provenance = cast(dict[str, object], control.get("provenance"))
    resumed_provenance = cast(dict[str, object], resumed.get("provenance"))
    if not isinstance(control_provenance, dict) or control_provenance != resumed_provenance:
        raise ValueError("checkpoint provenance differs")
    control_distributed = cast(dict[str, object], control.get("distributed"))
    resumed_distributed = cast(dict[str, object], resumed.get("distributed"))
    if not isinstance(control_distributed, dict) or not isinstance(resumed_distributed, dict):
        raise ValueError("both checkpoints must contain experiment 060 distributed state")
    if control_distributed.get("contract") != resumed_distributed.get("contract"):
        raise ValueError("distributed checkpoint contracts differ")
    control_ranks = control_distributed.get("ranks")
    resumed_ranks = resumed_distributed.get("ranks")
    if not isinstance(control_ranks, list) or not isinstance(resumed_ranks, list):
        raise ValueError("distributed checkpoints do not contain rank state")
    if len(control_ranks) != len(resumed_ranks):
        raise ValueError("distributed checkpoints contain different rank counts")

    rank_fields = (
        "partition_sha256",
        "loader",
        "identity_masker",
        "return_masker",
        "prefix_sampler",
        "return_calibration",
        "rng",
    )
    rank_comparisons = []
    for rank, (control_rank, resumed_rank) in enumerate(zip(control_ranks, resumed_ranks, strict=True)):
        if not isinstance(control_rank, dict) or not isinstance(resumed_rank, dict):
            raise ValueError("checkpoint rank state is invalid")
        control_rank = cast(dict[str, object], control_rank)
        resumed_rank = cast(dict[str, object], resumed_rank)
        fields = {name: _equal(control_rank.get(name), resumed_rank.get(name)) for name in rank_fields}
        rank_comparisons.append({"rank": rank, **fields})
    comparison = {
        "model": _equal(control["model"], resumed["model"]),
        "optimizer": _equal(control["opt"], resumed["opt"]),
        "scheduler": _equal(control["sched"], resumed["sched"]),
        "combined_return_calibration": _equal(
            control["return_calibration"],
            resumed["return_calibration"],
        ),
        "ranks": rank_comparisons,
    }
    rank_exact = all(all(value for name, value in item.items() if name != "rank") for item in rank_comparisons)
    passed = (
        all(
            cast(bool, comparison[name]) for name in ("model", "optimizer", "scheduler", "combined_return_calibration")
        )
        and rank_exact
    )
    record = {
        "schema_version": 1,
        "experiment_id": "060_compute_optimal_action_sequence_v2",
        "git_sha": control_provenance.get("git_sha"),
        "command": sys.argv,
        "update": int(control["step"]) + 1,
        "uninterrupted_checkpoint_sha256": _sha256(args.uninterrupted),
        "resumed_checkpoint_sha256": _sha256(args.resumed),
        "comparison": comparison,
        "next_rank_local_batch_exact": rank_exact,
        "next_optimizer_update_exact": passed,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"resume comparison failed: {comparison}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(tyro.cli(Args))
