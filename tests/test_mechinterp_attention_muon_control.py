"""Focused identity and exact-parity tests for the Muon attention control."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from research.mechinterp_20260911 import attention_detail
from research.mechinterp_20260911 import attention_muon_control as MODULE


def _metadata_fixture(tmp_path: Path) -> tuple[Path, MODULE.CheckpointSpec, dict[str, str]]:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"control-model")
    source_manifests = {"source-policy-world-v8": "a" * 64}
    spec = replace(
        MODULE.CHECKPOINT_SPECS[0],
        model_filename=model_path.name,
        model_export_sha256=MODULE._sha256_file(model_path),
        source_checkpoint_sha256="b" * 64,
        source_selection_sha256="c" * 64,
    )
    metadata = {
        "run_id": spec.run_id,
        "checkpoint_step": spec.checkpoint_step,
        "boundary_step": spec.boundary_step,
        "experiment_id": spec.experiment_id,
        "source_checkpoint_sha256": spec.source_checkpoint_sha256,
        "source_selection_sha256": spec.source_selection_sha256,
        "output_sha256": spec.model_export_sha256,
        "bytes": model_path.stat().st_size,
        "source_manifest_sha256": source_manifests,
        "raw_cfg": {
            "experiment_id": spec.experiment_id,
            "source_names": list(source_manifests),
        },
    }
    model_path.with_suffix(".metadata.json").write_text(json.dumps(metadata))
    return model_path, spec, source_manifests


def test_fixed_cohort_hash_is_the_public_64_hex_identity() -> None:
    assert MODULE.FIXED_COHORT_SHA256 == MODULE.attention_detail.FIXED_COHORT_SHA256
    assert len(MODULE.FIXED_COHORT_SHA256) == 64


@pytest.mark.parametrize("field", ["run_id", "checkpoint_step", "source_selection_sha256"])
def test_checkpoint_metadata_rejects_identity_mismatch(tmp_path: Path, field: str) -> None:
    model_path, spec, source_manifests = _metadata_fixture(tmp_path)
    metadata_path = model_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text())
    metadata[field] = "wrong" if field != "checkpoint_step" else spec.checkpoint_step + 1
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match=field):
        MODULE.validate_checkpoint_metadata(model_path, spec, source_manifests)


def test_checkpoint_metadata_rejects_source_manifest_mismatch(tmp_path: Path) -> None:
    model_path, spec, source_manifests = _metadata_fixture(tmp_path)
    metadata_path = model_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text())
    metadata["source_manifest_sha256"]["source-policy-world-v8"] = "d" * 64
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="source manifests"):
        MODULE.validate_checkpoint_metadata(model_path, spec, source_manifests)


def _summary_and_rows() -> tuple[
    attention_detail.AttentionSummary,
    NDArray[np.float32],
    dict[str, NDArray[np.float64] | NDArray[np.int64]],
]:
    weights = np.array([[[[0.2, 0.8]]]], dtype=np.float32)
    summary = MODULE.attention_detail.summarize_attention(weights)
    rows = {
        "entropy_nats": summary.per_head.mean_entropy_nats,
        "effective_context": summary.per_head.mean_exp_entropy,
        "row_max_weight": summary.per_head.mean_row_max_weight,
        "weighted_lag": summary.per_head.mean_weighted_lag,
        "current_frame_weight": summary.per_head.current_frame_weight,
        "argmax_lag": summary.per_head.most_frequent_argmax_lag,
    }
    return summary, weights[0], rows


def _write_parity_artifacts(
    tmp_path: Path,
    summary: attention_detail.AttentionSummary,
    example_weights: NDArray[np.float32],
    rows: dict[str, NDArray[np.float64] | NDArray[np.int64]],
) -> tuple[Path, Path]:
    detail_path = tmp_path / "attention_detail.json"
    detail_path.write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "run_id": MODULE.CHECKPOINT_SPECS[0].run_id,
                        "checkpoint_step": MODULE.CHECKPOINT_SPECS[0].checkpoint_step,
                        "summary": MODULE.summary_to_dict(summary),
                    }
                ]
            }
        )
    )
    example_path = tmp_path / "attention_example0.npz"
    arrays: dict[str, np.ndarray] = {"checkpoint_01__weights": example_weights}
    arrays.update({f"checkpoint_01__{name}": value for name, value in rows.items()})
    np.savez_compressed(example_path, **arrays)
    return detail_path, example_path


def test_parent_parity_requires_exact_summary_and_example(tmp_path: Path) -> None:
    summary, example_weights, rows = _summary_and_rows()
    detail_path, example_path = _write_parity_artifacts(tmp_path, summary, example_weights, rows)

    MODULE.validate_parent_parity(summary, example_weights, rows, detail_path, example_path)

    changed = MODULE.summary_to_dict(summary)
    changed["global"]["mean_entropy_nats"] += 1e-12
    detail_path.write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "run_id": MODULE.CHECKPOINT_SPECS[0].run_id,
                        "checkpoint_step": MODULE.CHECKPOINT_SPECS[0].checkpoint_step,
                        "summary": changed,
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="summary differs exactly"):
        MODULE.validate_parent_parity(summary, example_weights, rows, detail_path, example_path)


def test_parent_parity_rejects_exact_raw_example_mismatch(tmp_path: Path) -> None:
    summary, example_weights, rows = _summary_and_rows()
    detail_path, example_path = _write_parity_artifacts(tmp_path, summary, example_weights, rows)
    with np.load(example_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["checkpoint_01__weights"] = example_weights.copy()
    arrays["checkpoint_01__weights"][0, 0, 0] += np.float32(1e-3)
    np.savez_compressed(example_path, **arrays)

    with pytest.raises(ValueError, match="example field weights differs exactly"):
        MODULE.validate_parent_parity(summary, example_weights, rows, detail_path, example_path)


def _fake_measurement(experiment_id: str, geometry: int) -> MODULE.CheckpointMeasurement:
    summary, example_weights, rows = _summary_and_rows()
    return MODULE.CheckpointMeasurement(
        spec=MODULE.CHECKPOINT_SPECS[0],
        provenance={
            "resolved_config": {
                "experiment_id": experiment_id,
                "architecture": {"d_model": geometry},
            }
        },
        summary=summary,
        example_weights=example_weights,
        example_rows=rows,
        trace_max_abs_error=0.0,
        probability_sum_max_abs_error=0.0,
        metadata_sha256="a" * 64,
    )


def test_control_config_pair_allows_experiment_id_only_difference() -> None:
    MODULE.validate_control_config_pair(
        (
            _fake_measurement("050_scaled_temporal_awr_v4", 1024),
            _fake_measurement("050_scaled_temporal_awr_v5", 1024),
        )
    )


def test_control_config_pair_rejects_inference_difference() -> None:
    with pytest.raises(ValueError, match="inference configurations differ"):
        MODULE.validate_control_config_pair(
            (
                _fake_measurement("050_scaled_temporal_awr_v4", 1024),
                _fake_measurement("050_scaled_temporal_awr_v5", 256),
            )
        )
