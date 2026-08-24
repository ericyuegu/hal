"""Tests for experiment-039 result masking and native-latency joins."""

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_039_capacity.py"
_SPEC = importlib.util.spec_from_file_location("analyze_039_capacity", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
analysis = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = analysis
_SPEC.loader.exec_module(analysis)


def _run() -> SimpleNamespace:
    config = {
        "phase": "cooldown",
        "model_family": "scaled",
        "n_layers": 5,
        "d_model": 320,
        "target_processed_positions": 2**26,
        "unique_data_divisor": 1,
        "unique_replays": 100,
        "episode_hash": "abc",
        "unique_loss_positions": 1_000,
        "trunk_parameters": 6_000_000,
        "decoder_parameters": 1_000_000,
        "total_parameters": 7_000_000,
        "adam_tau_scaling": "powerlines",
        "muon_lr": 0.04,
        "adam_lr": 8.5e-4,
        "muon_weight_decay": 0.1,
        "adam_weight_decay": 0.05,
    }
    summary = {
        "training/processed_positions": 2**26,
        "training/D_over_U_positions": 1.0,
        "training/D_over_N": 9.5,
        "training/adam_tau": 1.0,
        "training/flops": 123.0,
        "training/flops_formula": "formula",
        "training/wall_seconds": 90.0,
        "training/cumulative_wall_seconds": 90.0,
        "training/incremental_run_wall_seconds": 10.0,
        "training/prefix_wall_seconds": 80.0,
        "val/nll": 2.0,
        "eval_d1/valid_latency_bucket": 1.0,
        "eval_d1/stock_lcb": 0.4,
        "eval_d1/mean_stock_difference": 0.6,
        "eval_d1/latency_p95_ms": 10.0,
        "eval_d2/valid_latency_bucket": 0.0,
        "eval_d2/stock_lcb": 9.0,
        "eval_d2/mean_stock_difference": 9.0,
        "eval_d2/latency_p95_ms": 40.0,
    }
    return SimpleNamespace(
        id="run-id",
        name="run-name",
        url="https://wandb.invalid/run-id",
        state="finished",
        created_at="2026-01-01",
        config=config,
        summary=summary,
    )


def test_terminal_row_masks_invalid_closed_loop_cells() -> None:
    row = analysis._terminal_row(_run())

    assert row is not None and row["endpoint_complete"]
    assert row["d1_stock_lcb"] == 0.4
    assert math.isnan(row["d2_stock_lcb"])
    assert row["d2_latency_p95_ms"] == 40.0
    assert row["training_time_seconds"] == 90.0
    assert row["incremental_branch_time_seconds"] == 10.0


def test_exact_isoflop_endpoint_is_not_assigned_to_power_of_two_matrix(tmp_path: Path) -> None:
    run = _run()
    target = 350_981_584
    run.config |= {
        "n_layers": 3,
        "d_model": 192,
        "target_processed_positions": target,
        "total_parameters": 2_064_111,
    }
    run.summary["training/processed_positions"] = target
    row = analysis._terminal_row(run)

    assert row is not None
    assert row["model"] == "L3"
    assert row["D_exp"] is None
    assert row["endpoint_kind"] == "exact-isoflop"

    args = analysis.Args(output_dir=str(tmp_path))
    missing = analysis._save_matrix_artifacts(pd.DataFrame([row]), args, tmp_path)

    matrix = pd.read_csv(tmp_path / "nx_d_d1.csv", index_col="model")
    assert "L3" not in matrix.index
    assert all(len(cells) == len(analysis.MODEL_ORDER) * len(analysis.D_EXPONENTS) for cells in missing.values())


def test_native_curve_uses_local_valid_delay_and_masked_performance() -> None:
    row = analysis._terminal_row(_run())
    assert row is not None
    frame = pd.DataFrame([row])
    latency = {
        "L5": {
            "native_delay": 1,
            "latency": {
                "1": {
                    "latency_p50_ms": 8.0,
                    "latency_p95_ms": 10.0,
                    "sustained_inference_rows_per_s": 800.0,
                }
            },
        }
    }

    joined = analysis._join_native_latency(frame, latency).iloc[0]

    assert joined["native_delay"] == 1
    assert joined["native_latency_p95_ms"] == 10.0
    assert joined["native_stock_lcb"] == 0.4


def test_latency_frontier_selects_largest_valid_model(tmp_path: Path) -> None:
    latency = {}
    for model, parameters, valid in (("L5", 7_000_000, 1.0), ("L7", 18_000_000, 1.0), ("L10", 50_000_000, 0.0)):
        latency[model] = {
            "total_parameters": parameters,
            "latency": {
                str(delay): {
                    "valid_bucket": valid,
                    "latency_p95_ms": 12.0,
                    "deadline_misses": 0.0 if valid else 1.0,
                }
                for delay in analysis.DELAYS
            },
        }

    frontier = analysis._save_latency_frontier(latency, tmp_path)

    assert set(frontier["largest_model"]) == {"L7"}
    assert (tmp_path / "latency_capacity_frontier.csv").is_file()
    assert (tmp_path / "latency_capacity_frontier.json").is_file()


def test_load_latency_requires_explicit_end_to_end_boundaries(tmp_path: Path) -> None:
    payload = {
        "schema_version": analysis.LATENCY_ARTIFACT_SCHEMA,
        "latency_start_boundary": analysis.LATENCY_START_BOUNDARY,
        "latency_end_boundary": analysis.LATENCY_END_BOUNDARY,
        "model_family": "scaled",
        "L": 5,
    }
    artifact = tmp_path / "L5.json"
    artifact.write_text(json.dumps(payload))

    assert analysis._load_latency(tmp_path)["L5"] == payload

    payload["latency_start_boundary"] = "latest_worker_observation_preprocessing"
    artifact.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unsupported latency artifact"):
        analysis._load_latency(tmp_path)
