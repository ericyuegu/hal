"""Focused tests for the restricted half-Muon checkpoint exporter."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from hal.training import physical_shard_loader
from research.mechinterp_20260911 import export_muon_control


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_export_model_only_restricts_globals_and_preserves_identity(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    output = tmp_path / "7dgv85x8-step-0057344.model.pt"
    locator = physical_shard_loader.PhysicalRow(source="source", shard=1, row=2)
    descriptor = physical_shard_loader.RingSlotDescriptor(
        slot=0,
        locator=locator,
        epoch=3,
        replay_checksum=4,
    )
    config = {
        "experiment_id": "050_scaled_temporal_awr_v5",
        "muon_lr_multiplier": 0.5,
        "parent_wandb_id": "p1fyyp1z",
        "parent_checkpoint_name": "checkpoints/step-0024576.pt",
        "parent_checkpoint_sha256": "p" * 64,
    }
    torch.save(
        {
            "cfg": config,
            "loader": {
                "data_protocol": "o50-replay-ring-v8",
                "ring": (descriptor,),
                "source_manifest_sha256": {f"source-{index}": "m" * 64 for index in range(44)},
                "source_selection_sha256": export_muon_control._EXPECTED_SOURCE_SELECTION_SHA256,
            },
            "model": {"weight": torch.ones(2, 2)},
            "step": 57_343,
            "wandb_id": "7dgv85x8",
        },
        source,
    )
    identity = export_muon_control.SourceIdentity(
        boundary_step=57_344,
        bytes=source.stat().st_size,
        etag='"etag"',
        parent_checkpoint_sha256="p" * 64,
        run_id="7dgv85x8",
        sha256=_sha256(source),
        uri="r2://hal/run/checkpoint.pt",
    )

    export_muon_control.export_model_only(source, output, identity)

    raw = torch.load(output, map_location="cpu", weights_only=True)
    metadata = json.loads(output.with_suffix(".metadata.json").read_text())
    assert set(raw) == {
        "cfg",
        "checkpoint_step",
        "data_protocol",
        "model",
        "source_checkpoint_sha256",
        "source_manifest_sha256",
        "source_r2_etag",
        "source_selection_sha256",
        "source_uri",
        "wandb_id",
    }
    assert raw["checkpoint_step"] == 57_343
    assert raw["source_checkpoint_sha256"] == identity.sha256
    assert metadata["output_sha256"] == _sha256(output)
    assert metadata["safe_pickle_globals"] == sorted(export_muon_control._EXPECTED_UNSAFE_GLOBALS)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_muon_control.export_model_only(source, output, identity)
