"""Safely export the matched half-Muon checkpoint for attention analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from typing import cast

import torch
from torch import Tensor

from hal.training import physical_shard_loader

_EXPECTED_EXPERIMENT_ID: Final[str] = "050_scaled_temporal_awr_v5"
_EXPECTED_DATA_PROTOCOL: Final[str] = "o50-replay-ring-v8"
_EXPECTED_SOURCE_SELECTION_SHA256: Final[str] = "2593361352b92e705be3fbeae1b4e9bb1a3c9f1787cd713014a7a95b7df62477"
_EXPECTED_LOADER_SOURCE_SHA256: Final[str] = "c098ac84193b1c83c473689317ef757be46f47c9d5edb018bd6ed7095ee75e35"
_EXPECTED_UNSAFE_GLOBALS: Final[frozenset[str]] = frozenset(
    {
        "hal.training.physical_shard_loader.PhysicalRow",
        "hal.training.physical_shard_loader.RingSlotDescriptor",
    }
)


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """Immutable identity of one trusted source checkpoint."""

    boundary_step: int
    bytes: int
    etag: str
    parent_checkpoint_sha256: str
    run_id: str
    sha256: str
    uri: str


MATCHED_HALF_MUON: Final[SourceIdentity] = SourceIdentity(
    boundary_step=57_344,
    bytes=1_779_666_698,
    etag='"9f4346d792838a73cb5fe6303758dfc5-213"',
    parent_checkpoint_sha256="6d8855b51b0ec06c633b8990788947ae242c86953f0cd1a2a95e396f041f6d5d",
    run_id="7dgv85x8",
    sha256="89c941d64f6407eed34bc6b9ad3ccc1466aace67d48f0ef45978bcb7575a4b08",
    uri="r2://hal/runs/o50-p1fyyp1z-u24576-muon-half/checkpoints/step-0057344.pt",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _string_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a mapping with string keys")
    return cast(Mapping[str, object], value)


def export_model_only(source: Path, output: Path, identity: SourceIdentity = MATCHED_HALF_MUON) -> None:
    """Restricted-load, validate, and export one trusted checkpoint."""
    if output.exists() or output.with_suffix(".metadata.json").exists():
        raise FileExistsError(f"refusing to overwrite model export or metadata: {output}")
    if source.stat().st_size != identity.bytes or _sha256_file(source) != identity.sha256:
        raise ValueError("source checkpoint bytes differ from the trusted identity")
    unsafe_globals = frozenset(torch.serialization.get_unsafe_globals_in_checkpoint(source))
    if unsafe_globals != _EXPECTED_UNSAFE_GLOBALS:
        raise ValueError(f"source checkpoint has unexpected pickle globals: {sorted(unsafe_globals)}")
    loader_source = Path(physical_shard_loader.__file__).resolve()
    if _sha256_file(loader_source) != _EXPECTED_LOADER_SOURCE_SHA256:
        raise ValueError("allowlisted loader class source differs from the audited file")

    safe_globals: list[Callable[..., object] | tuple[Callable[..., object], str]] = [
        physical_shard_loader.PhysicalRow,
        physical_shard_loader.RingSlotDescriptor,
    ]
    with torch.serialization.safe_globals(safe_globals):
        raw = _string_mapping(
            torch.load(source, map_location="cpu", weights_only=True, mmap=True),
            "source checkpoint",
        )
    checkpoint_step = raw.get("step")
    config = _string_mapping(raw.get("cfg"), "checkpoint config")
    loader = _string_mapping(raw.get("loader"), "checkpoint loader")
    model = _string_mapping(raw.get("model"), "checkpoint model")
    if checkpoint_step != identity.boundary_step - 1 or raw.get("wandb_id") != identity.run_id:
        raise ValueError("checkpoint step or W&B identity differs")
    if (
        config.get("experiment_id") != _EXPECTED_EXPERIMENT_ID
        or config.get("muon_lr_multiplier") != 0.5
        or config.get("parent_wandb_id") != "p1fyyp1z"
        or config.get("parent_checkpoint_name") != "checkpoints/step-0024576.pt"
        or config.get("parent_checkpoint_sha256") != identity.parent_checkpoint_sha256
    ):
        raise ValueError("checkpoint treatment or parent lineage differs")
    source_manifests = _string_mapping(loader.get("source_manifest_sha256"), "source manifests")
    if (
        len(source_manifests) != 44
        or loader.get("data_protocol") != _EXPECTED_DATA_PROTOCOL
        or loader.get("source_selection_sha256") != _EXPECTED_SOURCE_SELECTION_SHA256
    ):
        raise ValueError("checkpoint data identity differs")
    if not model or any(not isinstance(value, Tensor) for value in model.values()):
        raise TypeError("checkpoint model state must contain only tensors")
    tensor_model = cast(Mapping[str, Tensor], model)

    payload = {
        "cfg": dict(config),
        "checkpoint_step": checkpoint_step,
        "data_protocol": loader["data_protocol"],
        "model": dict(tensor_model),
        "source_manifest_sha256": dict(source_manifests),
        "source_selection_sha256": loader["source_selection_sha256"],
        "source_checkpoint_sha256": identity.sha256,
        "source_r2_etag": identity.etag,
        "source_uri": identity.uri,
        "wandb_id": identity.run_id,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(f"{output.suffix}.partial")
    torch.save(payload, partial)
    os.replace(partial, output)
    metadata = {
        "boundary_step": identity.boundary_step,
        "bytes": output.stat().st_size,
        "checkpoint_step": checkpoint_step,
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "data_protocol": loader["data_protocol"],
        "experiment_id": config["experiment_id"],
        "loader_source_sha256": _EXPECTED_LOADER_SOURCE_SHA256,
        "model_bytes": sum(value.numel() * value.element_size() for value in tensor_model.values()),
        "model_tensors": len(model),
        "output": output.name,
        "output_sha256": _sha256_file(output),
        "parent_checkpoint_sha256": identity.parent_checkpoint_sha256,
        "parent_wandb_id": config["parent_wandb_id"],
        "raw_cfg": dict(config),
        "run_id": identity.run_id,
        "safe_pickle_globals": sorted(_EXPECTED_UNSAFE_GLOBALS),
        "source_checkpoint_bytes": identity.bytes,
        "source_checkpoint_sha256": identity.sha256,
        "source_identity": asdict(identity),
        "source_manifest_sha256": dict(source_manifests),
        "source_r2_etag": identity.etag,
        "source_selection_sha256": loader["source_selection_sha256"],
        "source_uri": identity.uri,
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export_model_only(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
