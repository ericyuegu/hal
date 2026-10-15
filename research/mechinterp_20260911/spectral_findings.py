"""Summarize the six measured checkpoints without conflating norms and failure."""

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True, slots=True)
class Matrix:
    name: str
    frobenius: float
    stable_rank: float
    spectral: float
    norm_ratio: float | None
    angle_degrees: float | None


@dataclass(frozen=True, slots=True)
class Checkpoint:
    optimizer: str
    step: int
    matrices: tuple[Matrix, ...]


def mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("expected an object with string keys")
    return cast(Mapping[str, object], value)


def number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("expected a finite number")
    return float(value)


def load_checkpoints(path: Path) -> tuple[Checkpoint, ...]:
    data = mapping(json.loads(path.read_bytes()))
    result = []
    for optimizer in ("muon", "adamw"):
        records = mapping(data[optimizer])["checkpoints"]
        if not isinstance(records, list):
            raise ValueError("checkpoint collection must be a list")
        for item in records:
            record = mapping(item)
            checkpoint_step = mapping(record["metadata"])["checkpoint_step"]
            if isinstance(checkpoint_step, bool) or not isinstance(checkpoint_step, int) or checkpoint_step < 0:
                raise ValueError("checkpoint step must be a non-negative integer")
            step = checkpoint_step + 1
            if record["label"] != f"{optimizer}_{step}":
                raise ValueError("checkpoint boundary and label disagree")
            rows = record["matrices"]
            if not isinstance(rows, list):
                raise ValueError("matrix collection must be a list")
            matrices = []
            for value in rows:
                row = mapping(value)
                name = row["name"]
                if not isinstance(name, str):
                    raise ValueError("matrix name must be a string")
                change = mapping(row["change_from_previous"]) if "change_from_previous" in row else None
                matrices.append(
                    Matrix(
                        name,
                        number(row["frobenius"]),
                        number(row["stable_rank"]),
                        number(row["spectral"]),
                        number(change["norm_ratio"]) if change is not None else None,
                        number(change["angle_degrees"]) if change is not None else None,
                    )
                )
            if len({matrix.name for matrix in matrices}) != len(matrices):
                raise ValueError("duplicate matrix measurement")
            result.append(Checkpoint(optimizer, step, tuple(matrices)))
    if len({(record.optimizer, record.step) for record in result}) != len(result):
        raise ValueError("duplicate checkpoint")
    return tuple(result)


def matrix_group(name: str) -> str | None:
    if name.startswith("trunk.blocks."):
        for suffix, label in (
            ("/q", "Trunk Q"),
            ("/k", "Trunk K"),
            ("/v", "Trunk V"),
            ("attn.c_proj.weight", "Attention output"),
            ("mlp.c_fc.weight", "MLP up"),
            ("mlp.c_proj.weight", "MLP down"),
        ):
            if name.endswith(suffix):
                return label
    if name.startswith("temporal.blocks.") and name.endswith("/q"):
        return "Temporal Q"
    if name.startswith("temporal.outputs."):
        if name.endswith("up.weight"):
            return "Action hidden"
        if name.endswith("down.weight"):
            return "Action readout"
    if name.startswith("temporal.trunk_outputs."):
        return "Trunk readout"
    return None


def group_summary(checkpoint: Checkpoint) -> dict[str, object]:
    groups: dict[str, list[Matrix]] = {}
    for matrix in checkpoint.matrices:
        group = matrix_group(matrix.name)
        if group is not None:
            groups.setdefault(group, []).append(matrix)
    result: dict[str, object] = {}
    for group, matrices in groups.items():
        row: dict[str, object] = {"matrices": len(matrices)}
        metrics = {
            "frobenius": [matrix.frobenius for matrix in matrices],
            "stable_rank": [matrix.stable_rank for matrix in matrices],
            "spectral": [matrix.spectral for matrix in matrices],
            "norm_ratio": [matrix.norm_ratio for matrix in matrices],
            "angle_degrees": [matrix.angle_degrees for matrix in matrices],
        }
        for field, values in metrics.items():
            if any(value is None for value in values):
                row[field] = None
            else:
                quantiles = np.quantile([number(value) for value in values], (0.25, 0.5, 0.75))
                row[field] = dict(zip(("q25", "median", "q75"), quantiles.tolist(), strict=True))
        result[group] = row
    return result


def plot(checkpoints: tuple[Checkpoint, ...], output: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    colors = {"muon": "#b64937", "adamw": "#216c9a"}
    for optimizer in colors:
        records = sorted((record for record in checkpoints if record.optimizer == optimizer), key=lambda c: c.step)
        steps = [record.step / 1000 for record in records]
        for axis, use_rank, label in (
            (axes[0, 0], False, "Trunk Q Frobenius norm"),
            (axes[0, 1], True, "Trunk Q stable rank"),
        ):
            values = np.asarray(
                [
                    [
                        matrix.stable_rank if use_rank else matrix.frobenius
                        for matrix in record.matrices
                        if matrix_group(matrix.name) == "Trunk Q"
                    ]
                    for record in records
                ]
            )
            lower, middle, upper = np.quantile(values, (0.25, 0.5, 0.75), axis=1)
            axis.plot(steps, middle, "o-", color=colors[optimizer], label=optimizer)
            axis.fill_between(steps, lower, upper, color=colors[optimizer], alpha=0.15)
            axis.set(xlabel="Optimizer updates (thousands)", ylabel=label)
            axis.legend(frameon=False)

    for axis, step, title in (
        (axes[1, 0], 57344, "Early → slump (24,576 → 57,344)"),
        (axes[1, 1], 131072, "Late improvement (98,304 → 131,072)"),
    ):
        checkpoint = next(record for record in checkpoints if record.optimizer == "muon" and record.step == step)
        palette = {"Q/K": "#b64937", "V/output": "#216c9a", "MLP": "#85803e", "Readouts": "#673d80"}
        used: set[str] = set()
        for matrix in checkpoint.matrices:
            group = matrix_group(matrix.name)
            if group in {"Trunk Q", "Trunk K"}:
                label = "Q/K"
            elif group in {"Trunk V", "Attention output"}:
                label = "V/output"
            elif group in {"MLP up", "MLP down"}:
                label = "MLP"
            elif group in {"Action readout", "Trunk readout"}:
                label = "Readouts"
            else:
                continue
            axis.scatter(
                matrix.angle_degrees,
                matrix.norm_ratio,
                s=20,
                color=palette[label],
                alpha=0.65,
                label=label if label not in used else None,
            )
            used.add(label)
        axis.axhline(1, color="0.6", linewidth=0.8, linestyle="--")
        axis.set(xlabel="Net matrix angle (degrees)", ylabel="Frobenius norm / preceding checkpoint", title=title)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Muon scale growth is real; rank collapse and a norm-only failure marker are not supported")
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"matrix_geometry.{suffix}", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).with_name("artifacts") / "spectra" / "historical_spectra.json"
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("artifacts") / "spectra")
    args = parser.parse_args()
    checkpoints = load_checkpoints(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoints": {f"{record.optimizer}_{record.step}": group_summary(record) for record in checkpoints},
        "interpretation": [
            "Shading and quantiles describe matrix variation, not sampling uncertainty.",
            "Angles are net differences between saved checkpoints, not actual per-update angular learning rates.",
            "Raw norms and spectra depend on parameterization; these are descriptive associations.",
            "Readouts have different row counts; do not interpret their pooled rank as a common capacity limit.",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot(checkpoints, args.output)


if __name__ == "__main__":
    main()
