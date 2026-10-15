"""Measure checkpoint matrix spectra and changes without rebuilding the model."""

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch

from hal.training import physical_shard_loader


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_state(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Load tensors with PyTorch's restricted unpickler and reject unknown roots."""
    allowed = (physical_shard_loader.PhysicalRow, physical_shard_loader.RingSlotDescriptor)
    allowed_names = {f"{value.__module__}.{value.__qualname__}" for value in allowed}
    unsafe = set(torch.serialization.get_unsafe_globals_in_checkpoint(path)) - allowed_names
    if unsafe:
        raise ValueError(f"checkpoint needs explicitly reviewed serialization globals: {sorted(unsafe)}")
    with torch.serialization.safe_globals(list(allowed)):
        raw = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    if not isinstance(raw, Mapping) or not isinstance(raw.get("model"), Mapping):
        raise ValueError(f"{path}: expected a HAL checkpoint with a model mapping")
    state: dict[str, torch.Tensor] = {}
    for name, tensor in raw["model"].items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{path}: model state must map strings to tensors")
        state[name] = tensor
    metadata: dict[str, object] = {}
    for name in (
        "step",
        "checkpoint_step",
        "wandb_run_id",
        "wandb_id",
        "cfg",
        "git_sha",
        "experiment_id",
        "source_checkpoint_sha256",
        "source_r2_etag",
        "source_uri",
        "source_manifest_sha256",
        "source_selection_sha256",
        "data_protocol",
    ):
        if name in raw:
            metadata[name] = raw[name]
    return state, metadata


def logical_matrices(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    matrices = {}
    for name, tensor in state.items():
        if tensor.ndim != 2 or not tensor.is_floating_point() or not name.endswith("weight"):
            continue
        matrices[name] = tensor
        if name.endswith(("c_attn.weight", "qkv.weight")):
            if tensor.shape[0] != 3 * tensor.shape[1]:
                raise ValueError(f"{name}: unsupported fused QKV shape {tuple(tensor.shape)}")
            for part, value in zip(("q", "k", "v"), tensor.chunk(3, dim=0), strict=True):
                matrices[f"{name}/{part}"] = value
    return matrices


def matrix_metrics(matrix: torch.Tensor) -> dict[str, float | int]:
    if matrix.ndim != 2 or not torch.isfinite(matrix).all():
        raise ValueError("spectral analysis requires a finite matrix")
    values = matrix.float()
    singular = torch.linalg.svdvals(values).double()
    squared_norm = singular.square().sum()
    norm = squared_norm.sqrt()
    spectral = singular[0]
    nonzero = singular > 0
    if not bool(nonzero.any()):
        return {
            "rows": matrix.shape[0],
            "columns": matrix.shape[1],
            "frobenius": 0.0,
            "rms": 0.0,
            "spectral": 0.0,
            "stable_rank": 0.0,
            "effective_rank": 0.0,
            "top1_energy": 0.0,
            "top4_energy": 0.0,
        }
    probability = singular[nonzero] / singular.sum()
    return {
        "rows": matrix.shape[0],
        "columns": matrix.shape[1],
        "frobenius": float(norm),
        "rms": float(norm / math.sqrt(matrix.numel())),
        "spectral": float(spectral),
        "stable_rank": float(squared_norm / spectral.square()),
        "effective_rank": float((-(probability * probability.log()).sum()).exp()),
        "top1_energy": float(spectral.square() / squared_norm),
        "top4_energy": float(singular[:4].square().sum() / squared_norm),
    }


def matrix_change(before: torch.Tensor, after: torch.Tensor) -> dict[str, float | None]:
    """Separate net radial and directional changes between saved checkpoints."""
    if before.shape != after.shape:
        raise ValueError("matrix shapes differ across checkpoints")
    left = before.double().flatten()
    right = after.double().flatten()
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if left_norm == 0 or right_norm == 0:
        return {"norm_ratio": None, "angle_degrees": None, "relative_difference": None}
    cosine = torch.dot(left, right) / (left_norm * right_norm)
    return {
        "norm_ratio": float(right_norm / left_norm),
        "angle_degrees": math.degrees(math.acos(float(cosine.clamp(-1, 1)))),
        "relative_difference": float(torch.linalg.vector_norm(right - left) / left_norm),
    }


def inspect_checkpoint(label: str, path: Path, device: str) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    state, metadata = checkpoint_state(path)
    matrices = logical_matrices(state)
    rows = []
    for name, matrix in matrices.items():
        row: dict[str, object] = {"name": name, **matrix_metrics(matrix.to(device))}
        rows.append(row)
    result: dict[str, object] = {
        "label": label,
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "metadata": metadata,
        "matrices": rows,
    }
    print(f"{label}: {len(rows)} matrix measurements; SHA-256 {result['sha256']}", flush=True)
    return result, matrices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, help="label=/absolute/checkpoint.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--analysis-git-sha", help="Explicit source revision when the worker has no Git directory")
    args = parser.parse_args()
    analysis_git_sha = (
        args.analysis_git_sha or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    )
    if re.fullmatch(r"[0-9a-f]{40}", analysis_git_sha) is None:
        raise ValueError("analysis source revision must be a full Git SHA")
    torch.set_num_threads(4)
    records = []
    previous: dict[str, torch.Tensor] | None = None
    previous_label: str | None = None
    for specification in args.checkpoint:
        label, separator, source = specification.partition("=")
        if not separator or re.fullmatch(r"[A-Za-z0-9_.-]+", label) is None:
            raise ValueError(f"invalid checkpoint specification {specification!r}")
        record, matrices = inspect_checkpoint(label, Path(source), args.device)
        if previous is not None:
            if matrices.keys() != previous.keys():
                raise ValueError("checkpoints do not have the same logical matrices")
            rows = cast(list[dict[str, object]], record["matrices"])
            for row in rows:
                name = cast(str, row["name"])
                row["change_from_previous"] = matrix_change(previous[name], matrices[name])
            record["previous_label"] = previous_label
        records.append(record)
        previous, previous_label = matrices, label
    result = {
        "schema_version": 1,
        "analysis_git_sha": analysis_git_sha,
        "script_sha256": file_sha256(Path(__file__)),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "checkpoints": records,
        "limitations": [
            "Net changes between checkpoints are not per-update learning rates.",
            "Raw parameter norms and spectra depend on parameterization.",
            "Fused QKV and its logical submatrices are both reported; do not sum them together.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
