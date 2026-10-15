"""Locate each head's unique maximum-weight key in the saved Muon example."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import subprocess
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

DETAIL_SHA256 = "ce30744586fcc6bd5c828dc13f9c1fa930b014465ec157ee5ab506646359f6e2"
EXAMPLE_SHA256 = "1518b5581f7beda9a431d2f4c9c7e22c6712c62a350a8fdf135f37c7d8ec38be"
DISPLAY_MODELS = (
    (1, "7dgv85x8", "Half-rate Muon continuation"),
    (0, "p1fyyp1z", "Original Muon"),
)


@dataclasses.dataclass(frozen=True, slots=True)
class LayerPositions:
    layer: int
    key_positions: tuple[int, ...]
    peak_weights: tuple[float, ...]
    distinct_positions: int
    head_pairs_same_position: int


def summarize_positions(weights_lhk: npt.NDArray[np.float32]) -> tuple[LayerPositions, ...]:
    """Describe one query's unique maximum over keys for every layer and head.

    Tied maxima are rejected: an arbitrary tie-break could create or remove
    apparent agreement between heads. Key order is preserved, not reversed.
    """
    if weights_lhk.dtype != np.float32 or weights_lhk.ndim != 3 or min(weights_lhk.shape) < 1:
        raise ValueError("weights must be a nonempty float32 [layer, head, key] array")
    if not np.isfinite(weights_lhk).all() or (weights_lhk < 0).any():
        raise ValueError("attention weights must be finite and non-negative")
    if not np.allclose(weights_lhk.sum(axis=-1, dtype=np.float64), 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("attention weights must sum to one over keys")
    maxima_lh = weights_lhk.max(axis=-1)
    if ((weights_lhk == maxima_lh[..., None]).sum(axis=-1) != 1).any():
        raise ValueError("unique maxima are required; tied positions cannot be collapsed to one marker")
    positions_lh = weights_lhk.argmax(axis=-1)
    layers = []
    for layer, positions_h in enumerate(positions_lh):
        _, counts = np.unique(positions_h, return_counts=True)
        layers.append(
            LayerPositions(
                layer=layer,
                key_positions=tuple(int(position) for position in positions_h),
                peak_weights=tuple(float(weight) for weight in maxima_lh[layer]),
                distinct_positions=len(counts),
                head_pairs_same_position=int((counts * (counts - 1) // 2).sum()),
            )
        )
    return tuple(layers)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a mapping with string keys")
    return cast(Mapping[str, object], value)


def build_payload(detail_path: Path, example_path: Path) -> dict[str, object]:
    """Derive positions from exactly the two previously validated checkpoints."""
    if _sha256(detail_path) != DETAIL_SHA256 or _sha256(example_path) != EXAMPLE_SHA256:
        raise ValueError("saved attention artifacts differ from their frozen identities")
    detail = _mapping(json.loads(detail_path.read_bytes()), "attention detail")
    if detail.get("schema_version") != 1:
        raise ValueError("unsupported attention detail schema")
    checkpoints = detail.get("checkpoints")
    cohort = _mapping(detail.get("cohort"), "cohort")
    windows = cohort.get("windows")
    if not isinstance(checkpoints, list) or len(checkpoints) != 2:
        raise ValueError("the saved comparison must contain exactly two checkpoints")
    if not isinstance(windows, list) or len(windows) != 256:
        raise ValueError("the saved cohort must retain all 256 window records")
    models = []
    with np.load(example_path, allow_pickle=False) as archive:
        for index, run_id, label in DISPLAY_MODELS:
            checkpoint = _mapping(checkpoints[index], "checkpoint")
            if (
                checkpoint.get("run_id") != run_id
                or checkpoint.get("checkpoint_step") != 57343
                or checkpoint.get("boundary_step") != 57344
            ):
                raise ValueError("saved checkpoint identity does not match the display mapping")
            provenance = _mapping(checkpoint.get("provenance"), "checkpoint provenance")
            array_key = f"checkpoint_{index:02d}__weights"
            weights_lhk = archive[array_key]
            if weights_lhk.shape != (16, 16, 256):
                raise ValueError(f"unexpected saved attention shape: {weights_lhk.shape}")
            layers = summarize_positions(weights_lhk)
            positions = np.asarray([layer.key_positions for layer in layers])
            models.append(
                {
                    "run_id": run_id,
                    "label": label,
                    "boundary_step": 57344,
                    "model_export_sha256": provenance["model_export_sha256"],
                    "source_array_key": array_key,
                    "layers": [dataclasses.asdict(layer) for layer in layers],
                    "distinct_positions_all_layers": int(np.unique(positions).size),
                    "heads_with_tied_maxima": 0,
                }
            )
    return {
        "schema_version": 1,
        "scope": {
            "example_index": 0,
            "layer_count": 16,
            "head_count": 16,
            "key_count": 256,
            "query_token_position": 255,
            "key_position_order": "0 is oldest; 255 is most recent input token",
            "frames_before_query": "255 - key_position",
            "head_pairs_per_layer": 120,
            "averaging": "none; one saved input and query position",
            "example": dict(_mapping(windows[0], "example zero")),
        },
        "source": {
            "detail_path": str(detail_path.resolve()),
            "detail_sha256": DETAIL_SHA256,
            "example_path": str(example_path.resolve()),
            "example_sha256": EXAMPLE_SHA256,
            "cohort_sha256": cohort["package_sha256"],
            "measurement_precision": detail["precision"],
        },
        "models": models,
    }


def main() -> None:
    study = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=study / "artifacts/attention_muon_control")
    parser.add_argument("--output", type=Path, default=study / "artifacts/attention_head_positions")
    args = parser.parse_args()
    output_path = args.output / "head_positions.json"
    archive_path = args.output / "head_positions_source.tar"
    if output_path.exists() or archive_path.exists():
        raise FileExistsError("refusing to overwrite a derived artifact or source snapshot")
    plan_path = study / "attention_head_positions_plan.md"
    test_path = study.parents[1] / "tests/test_mechinterp_head_positions.py"
    sources = (Path(__file__).resolve(), plan_path, test_path)
    if any(not path.is_file() for path in sources):
        raise FileNotFoundError("the analysis source, plan, and focused tests must exist before derivation")
    payload = build_payload(args.source / "attention_detail.json", args.source / "attention_example0.npz")
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=study, capture_output=True, text=True, check=True
    ).stdout.strip()
    args.output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "x") as archive:
        for path in sources:
            archive.add(path, arcname=str(path.relative_to(study.parents[1])))
    payload["derivation"] = {
        "git_sha": git_sha,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "source_sha256": _sha256(Path(__file__)),
        "plan_sha256": _sha256(plan_path),
        "source_archive_sha256": _sha256(archive_path),
        "random_sampling": False,
        "new_model_execution": False,
    }
    with output_path.open("x") as output:
        output.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
