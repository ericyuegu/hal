"""Focused tests for the standalone Experiment 039 scaling audit."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE_PATH = Path(__file__).parents[1] / "research" / "mechinterp_20260911" / "scaling_audit.py"
SPEC = importlib.util.spec_from_file_location("mechinterp_scaling_audit", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load {MODULE_PATH}")
scaling_audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scaling_audit
SPEC.loader.exec_module(scaling_audit)


def _run_row(
    run_id: str,
    *,
    model: str = "L5",
    processed: float = 2**26,
    nll: float = 2.0,
    excluded: bool = False,
    phase: str = "cooldown",
) -> dict[str, object]:
    return {
        "record_type": "run",
        "run_id": run_id,
        "model": model,
        "model_family": "scaled",
        "L": 5,
        "phase": phase,
        "endpoint_exact": True,
        "analysis_excluded": excluded,
        "analysis_exclusion_reason": "bad" if excluded else None,
        "wandb_state": "finished",
        "target_processed_positions": processed,
        "processed_positions": processed,
        "validation_nll": nll,
        "validation_primary_nll": nll / 1.5,
        "validation_auxiliary_nll": nll / 3.0,
        "training_flops": 6.0 * 30_000_000.0 * processed,
        "total_parameters": 7_000_000,
        "trunk_parameters": 6_000_000,
        "decoder_parameters": 1_000_000,
        "unique_data_divisor": 1,
        "unique_loss_positions": 2_000_000_000,
        "unique_replays": 100_000,
        "episode_hash": "episode",
    }


def test_select_completed_runs_marks_divergence_and_excludes_failed_rows() -> None:
    raw = pd.DataFrame(
        [
            _run_row("first", nll=2.0),
            _run_row("second", processed=2**27, nll=2.02),
            _run_row("excluded", processed=2**28, nll=1.8, excluded=True),
            {
                **_run_row("prefix", processed=2**29, nll=1.7, phase="prefix"),
                "endpoint_exact": False,
            },
        ]
    )

    result = scaling_audit.select_completed_runs(raw)

    assert result["run_id"].tolist() == ["first", "second"]
    assert result["diverged"].tolist() == [False, True]


def test_distinct_positions_is_bounded_and_monotone() -> None:
    values = scaling_audit.distinct_positions(np.asarray([1.0, 10.0, 100.0]), 1_000.0)

    assert np.all(np.diff(values) > 0)
    assert np.all(values < np.asarray([1.0, 10.0, 100.0]))


def test_attach_match_metrics_uses_active_rows_and_recomputes_rate() -> None:
    evaluations = pd.DataFrame(
        [
            {
                "evaluation_key": "run:d1",
                "eval_net_stock_per_min": 1.0,
                "eval_net_dmg_per_min": 10.0,
                "protocol_hash_present": True,
            }
        ]
    )
    raw = pd.DataFrame(
        [
            {
                "record_type": "match",
                "evaluation_key": "run:d1",
                "match_active_frames": 60.0,
                "match_boot_index": 0,
                "match_stock_difference": 1.0 / 60.0,
                "match_damage_difference": 10.0 / 60.0,
            },
            {
                "record_type": "match",
                "evaluation_key": "run:d1",
                "match_active_frames": 60.0,
                "match_boot_index": 1,
                "match_stock_difference": 1.0 / 60.0,
                "match_damage_difference": 10.0 / 60.0,
            },
            {
                "record_type": "match",
                "evaluation_key": "run:d1",
                "match_active_frames": 0.0,
                "match_boot_index": 2,
                "match_stock_difference": 100.0,
                "match_damage_difference": 100.0,
            },
            {
                "record_type": "match",
                "evaluation_key": "incomplete:d1",
                "match_active_frames": 60.0,
                "match_boot_index": 3,
                "match_stock_difference": 100.0,
                "match_damage_difference": 100.0,
            },
        ]
    )

    result, matches = scaling_audit.attach_match_metrics(evaluations, raw, bootstrap_resamples=20)

    assert len(matches) == 2
    assert result.loc[0, "active_matches"] == 2
    assert result.loc[0, "recomputed_net_stock_per_min"] == pytest.approx(1.0)
    assert result.loc[0, "recomputed_net_dmg_per_min"] == pytest.approx(10.0)


def test_correlation_table_excludes_diverged_rows_by_default() -> None:
    frame = pd.DataFrame(
        {
            "has_active_matches": [True, True, True, True],
            "matched_delay": [True, True, True, True],
            "diverged": [False, False, False, True],
            "delay_frames": [4.0, 4.0, 4.0, 4.0],
            "model": ["L3", "L5", "L7", "L10"],
            "processed_positions": [1.0, 2.0, 3.0, 4.0],
            "run_id": ["a", "b", "c", "d"],
            "validation_nll": [3.0, 2.0, 1.0, 0.0],
            "validation_primary_nll": [3.0, 2.0, 1.0, 0.0],
            "validation_auxiliary_nll": [3.0, 2.0, 1.0, 0.0],
            "recomputed_net_stock_per_min": [0.0, 1.0, 2.0, 3.0],
        }
    )

    healthy = scaling_audit.correlation_table(frame, include_diverged=False, bootstrap_resamples=20)
    all_rows = scaling_audit.correlation_table(frame, include_diverged=True, bootstrap_resamples=20)

    healthy_fixed = healthy.loc[healthy["scope"].eq("fixed_delay")]
    all_fixed = all_rows.loc[all_rows["scope"].eq("fixed_delay")]
    assert healthy_fixed["n_evaluations"].unique().tolist() == [3]
    assert all_fixed["n_evaluations"].unique().tolist() == [4]


def test_scaling_fit_recovers_synthetic_curve() -> None:
    n_eff = np.asarray([1, 2, 4, 8, 16, 32], dtype=float) * 1e6
    processed = np.asarray([1, 4, 2, 8, 3, 12], dtype=float) * 1e7
    target = 1.5 + 20.0 * n_eff**-0.4 + 30.0 * processed**-0.3
    frame = pd.DataFrame(
        {
            "model": [f"L{index}" for index in range(len(n_eff))],
            "training_flops": 6.0 * n_eff * processed,
            "processed_positions": processed,
            "validation_nll": target,
            "n_eff": n_eff,
            "unique_loss_positions": 1e12,
        }
    )

    fit = scaling_audit.fit_scaling_law(frame, max_starts=36, target_compute=1e18)

    assert fit.asymptote == pytest.approx(1.5, abs=0.02)
    assert fit.capacity_exponent == pytest.approx(0.4, abs=0.03)
    assert fit.data_exponent == pytest.approx(0.3, abs=0.03)
    assert fit.predicted_nll > 0


def test_load_raw_rejects_missing_persisted_schema(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    path.write_text("record_type\nrun\n")

    with pytest.raises(ValueError, match="missing persisted columns"):
        scaling_audit.load_raw(path)


def test_load_raw_column_predicate_preserves_list_selection(tmp_path: Path) -> None:
    columns = tuple(reversed(sorted(scaling_audit.REQUIRED_COLUMNS)))
    frame = pd.DataFrame({name: [index, index + 1] for index, name in enumerate(columns)})
    frame["unused_config_json"] = ["{}", "{}"]
    path = tmp_path / "raw.csv"
    frame.to_csv(path, index=False)

    result = scaling_audit.load_raw(path)

    pd.testing.assert_frame_equal(result, frame.drop(columns="unused_config_json"))
