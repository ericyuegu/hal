"""Focused schema tests for the historical O50 attention artifacts."""

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import cast

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "research" / "mechinterp_20260911" / "attention_findings.py"
_SPEC = importlib.util.spec_from_file_location("attention_findings", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
attention_findings = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = attention_findings
_SPEC.loader.exec_module(attention_findings)


def _matrix(base: float) -> list[list[float]]:
    return [[base + layer * 0.01 + head * 0.001 for head in range(16)] for layer in range(16)]


def _artifact_payload(*, run_id: str, step: int, optimizer: str = "muon") -> dict[str, object]:
    entropy = _matrix(0.1)
    return {
        "schema_version": 2,
        "checkpoint": {
            "backend_config": {
                "experiment_id": "050_scaled_temporal_awr_v6"
                if optimizer == "adamw"
                else "050_scaled_temporal_awr_v4",
                "prediction_frames": 4,
                "architecture": {"L_ctx": 256, "n_layers": 16, "n_heads": 16},
            },
            "checkpoint_step": step - 1,
            "source_uri": "r2://run/all-adamw/checkpoint.pt"
            if optimizer == "adamw"
            else "r2://run/muon/checkpoint.pt",
            "wandb_id": run_id,
        },
        "cohort_package": {"sha256": "a" * 64, "windows": 256},
        "data": {"seed": 20260911, "windows": [{} for _ in range(256)]},
        "nll_units": {
            "group_nll": "natural-log nats per controller group per predicted frame, then averaged over four frames"
        },
        "attention": {
            "entropy_nats": entropy,
            "effective_context": [[math.exp(value) for value in row] for row in entropy],
            "mean_lag": _matrix(64.0),
            "last_4_mass": _matrix(0.2),
            "last_16_mass": _matrix(0.5),
            "distant_64_plus_mass": _matrix(0.3),
            "current_mass": _matrix(0.1),
        },
        "history_retained_frames": {
            str(retained): {
                "replay_paired_horizon_group_mean_nll_delta": {
                    "mean": 0.01 if retained < 256 else 0.0,
                    "bootstrap_95": [0.0, 0.02] if retained < 256 else [0.0, 0.0],
                }
            }
            for retained in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        },
    }


def _write_history(tmp_path: Path, payloads: list[dict[str, object]]) -> Path:
    source = tmp_path / "historical"
    for index, payload in enumerate(payloads):
        directory = source / f"model-{index}"
        directory.mkdir(parents=True)
        (directory / "results.json").write_text(json.dumps(payload))
    return source


def _intervention_payload(entropy_early: list[list[float]], entropy_slump: list[list[float]]) -> dict[str, object]:
    def treatment(kind: str, entropy: list[list[float]]) -> dict[str, object]:
        return {
            "kind": kind,
            "attention": {
                "metrics": {"entropy_nats": entropy},
                "traced_trunk_max_abs_error": 0.0,
            },
        }

    return {
        "schema_version": 1,
        "cohort": {"data": {"seed": 20260911, "windows": [{} for _ in range(256)]}},
        "interventions": {
            "early_identity": treatment("identity_early", entropy_early),
            "slump_identity": treatment("identity_slump", entropy_slump),
        },
    }


def _corrected_payload(record: object, *, legacy_sha256: str) -> dict[str, object]:
    record = cast(attention_findings.AttentionArtifact, record)
    entropy = [[value + 0.001 for value in row] for row in record.entropy_nats]
    return {
        "schema_version": 1,
        "attention": {
            "entropy_nats": entropy,
            "effective_context": [[math.exp(value) for value in row] for row in entropy],
            "mean_lag": [list(row) for row in record.mean_lag],
            "last_4_mass": _matrix(0.2),
            "last_16_mass": _matrix(0.5),
            "distant_64_plus_mass": _matrix(0.3),
            "current_mass": _matrix(0.1),
        },
        "checkpoint": {
            "wandb_id": record.run_id,
            "checkpoint_step": record.checkpoint_step,
            "model_export_sha256": "b" * 64,
        },
        "cohort": {"sha256": record.cohort_sha256, "windows": record.cohort_windows},
        "controls": {
            "attention_probability_sum_max_abs_error": 0.0,
            "traced_trunk_max_abs_error": 0.0,
        },
        "legacy_functional_artifact": {"sha256": legacy_sha256},
        "precision": {
            "qk_score_dtype": "torch.float32",
            "score_and_softmax_autocast_enabled": False,
            "softmax_output_dtype": "torch.float32",
        },
    }


def test_load_historical_extracts_layer_head_matrices(tmp_path: Path) -> None:
    source = _write_history(
        tmp_path,
        [
            _artifact_payload(run_id="muon", step=24576),
            _artifact_payload(run_id="adam", step=57344, optimizer="adamw"),
        ],
    )

    records = attention_findings.load_historical(source, expected_count=2)

    assert [record.optimizer for record in records] == ["Muon", "AdamW"]
    assert records[0].update == 24576
    assert len(records[0].entropy_nats) == 16
    assert len(records[0].entropy_nats[0]) == 16
    assert records[0].mean_entropy_nats == pytest.approx(0.1825)
    assert records[0].history[-1].delta_nll == 0.0


def test_load_historical_rejects_attention_shape(tmp_path: Path) -> None:
    payload = _artifact_payload(run_id="muon", step=24576)
    attention = cast(dict[str, object], payload["attention"])
    attention["entropy_nats"] = [[0.1]]
    source = _write_history(tmp_path, [payload])

    with pytest.raises(ValueError, match="16 layers"):
        attention_findings.load_historical(source, expected_count=1)


def test_load_historical_rejects_mixed_cohorts(tmp_path: Path) -> None:
    first = _artifact_payload(run_id="one", step=24576)
    second = _artifact_payload(run_id="two", step=57344)
    cohort = cast(dict[str, object], second["cohort_package"])
    cohort["sha256"] = "b" * 64
    source = _write_history(tmp_path, [first, second])

    with pytest.raises(ValueError, match="one replay cohort"):
        attention_findings.load_historical(source, expected_count=2)


def test_load_historical_rejects_non_natural_log_units(tmp_path: Path) -> None:
    payload = _artifact_payload(run_id="muon", step=24576)
    units = cast(dict[str, object], payload["nll_units"])
    units["group_nll"] = "bits per frame"
    source = _write_history(tmp_path, [payload])

    with pytest.raises(ValueError, match="natural-log nats"):
        attention_findings.load_historical(source, expected_count=1)


def test_intervention_identity_trace_matches_historical_heatmaps(tmp_path: Path) -> None:
    early = _artifact_payload(run_id="muon", step=24576)
    slump = _artifact_payload(run_id="muon", step=57344)
    source = _write_history(tmp_path, [early, slump])
    historical = attention_findings.load_historical(source, expected_count=2)
    early_attention = cast(dict[str, object], early["attention"])
    slump_attention = cast(dict[str, object], slump["attention"])
    entropy = cast(list[list[float]], early_attention["entropy_nats"])
    slump_entropy = cast(list[list[float]], slump_attention["entropy_nats"])
    intervention = tmp_path / "interventions.json"
    intervention.write_text(json.dumps(_intervention_payload(entropy, slump_entropy)))

    trace = attention_findings.load_intervention_trace(intervention, historical)

    assert trace.cohort_windows == 256
    assert trace.early_max_trace_error == 0.0
    assert trace.slump_max_trace_error == 0.0


def test_intervention_identity_trace_rejects_mismatch(tmp_path: Path) -> None:
    early = _artifact_payload(run_id="muon", step=24576)
    slump = _artifact_payload(run_id="muon", step=57344)
    source = _write_history(tmp_path, [early, slump])
    historical = attention_findings.load_historical(source, expected_count=2)
    early_attention = cast(dict[str, object], early["attention"])
    slump_attention = cast(dict[str, object], slump["attention"])
    entropy = cast(list[list[float]], early_attention["entropy_nats"])
    slump_entropy = cast(list[list[float]], slump_attention["entropy_nats"])
    mismatch = [row[:] for row in entropy]
    mismatch[0][0] += 0.01
    intervention = tmp_path / "interventions.json"
    intervention.write_text(json.dumps(_intervention_payload(mismatch, slump_entropy)))

    with pytest.raises(ValueError, match="entropy trace differs"):
        attention_findings.load_intervention_trace(intervention, historical)


def test_corrected_attention_joins_checkpoint_cohort_and_legacy_sha(tmp_path: Path) -> None:
    early = _artifact_payload(run_id="muon", step=24576)
    slump = _artifact_payload(run_id="adam", step=57344, optimizer="adamw")
    source = _write_history(tmp_path, [early, slump])
    historical = attention_findings.load_historical(source, expected_count=2)
    reaudits = tmp_path / "reaudits"
    reaudits.mkdir()
    for record in historical:
        payload = _corrected_payload(record, legacy_sha256=record.source_sha256)
        (reaudits / f"{record.run_id}-{record.checkpoint_step}.json").write_text(json.dumps(payload))

    corrected = attention_findings.load_corrected_attention(reaudits, historical)

    assert [item.run_id for item in corrected] == [record.run_id for record in historical]
    assert corrected[0].mean_entropy_nats == pytest.approx(historical[0].mean_entropy_nats + 0.001)
    expected_payload = _corrected_payload(historical[0], legacy_sha256="a" * 64)
    expected_attention = cast(dict[str, object], expected_payload["attention"])
    expected_effective = cast(list[list[float]], expected_attention["effective_context"])
    assert corrected[0].mean_effective_context == pytest.approx(sum(sum(row) for row in expected_effective) / 256)
    assert corrected[0].score_dtype == corrected[0].softmax_dtype == "torch.float32"


def test_corrected_attention_uses_corrected_mean_lag(tmp_path: Path) -> None:
    payload = _artifact_payload(run_id="muon", step=24576)
    source = _write_history(tmp_path, [payload])
    historical = attention_findings.load_historical(source, expected_count=1)
    reaudits = tmp_path / "reaudits"
    reaudits.mkdir()
    corrected = _corrected_payload(historical[0], legacy_sha256=historical[0].source_sha256)
    attention = cast(dict[str, object], corrected["attention"])
    attention["mean_lag"] = [[value + 1.0 for value in row] for row in historical[0].mean_lag]
    result = reaudits / "result.json"
    result.write_text(json.dumps(corrected))

    loaded = attention_findings.load_corrected_attention(reaudits, historical)

    assert loaded[0].mean_lag_frames == pytest.approx(historical[0].mean_lag_frames + 1.0)
    trace = attention_findings.InterventionTrace("a" * 64, 256, 0.0, 0.0)
    summary = attention_findings._summary(historical, trace, _SCRIPT, loaded)
    checkpoint = cast(list[dict[str, object]], summary["checkpoints"])[0]
    assert checkpoint["mean_lag_frames"] == pytest.approx(historical[0].mean_lag_frames + 1.0)


def test_corrected_attention_rejects_wrong_legacy_sha(tmp_path: Path) -> None:
    payload = _artifact_payload(run_id="muon", step=24576)
    source = _write_history(tmp_path, [payload])
    historical = attention_findings.load_historical(source, expected_count=1)
    reaudits = tmp_path / "reaudits"
    reaudits.mkdir()
    corrected = _corrected_payload(historical[0], legacy_sha256="c" * 64)
    (reaudits / "result.json").write_text(json.dumps(corrected))

    with pytest.raises(ValueError, match="old artifact"):
        attention_findings.load_corrected_attention(reaudits, historical)


def test_corrected_attention_rejects_precision_or_cohort_mismatch(tmp_path: Path) -> None:
    payload = _artifact_payload(run_id="muon", step=24576)
    source = _write_history(tmp_path, [payload])
    historical = attention_findings.load_historical(source, expected_count=1)
    reaudits = tmp_path / "reaudits"
    reaudits.mkdir()
    corrected = _corrected_payload(historical[0], legacy_sha256=historical[0].source_sha256)
    cast(dict[str, object], corrected["precision"])["softmax_output_dtype"] = "torch.bfloat16"
    (reaudits / "result.json").write_text(json.dumps(corrected))

    with pytest.raises(ValueError, match="dtypes"):
        attention_findings.load_corrected_attention(reaudits, historical)

    cast(dict[str, object], corrected["precision"])["softmax_output_dtype"] = "torch.float32"
    cast(dict[str, object], corrected["cohort"])["sha256"] = "d" * 64
    (reaudits / "result.json").write_text(json.dumps(corrected))
    with pytest.raises(ValueError, match="cohort"):
        attention_findings.load_corrected_attention(reaudits, historical)
