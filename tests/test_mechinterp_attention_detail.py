"""Focused tests for detailed attention metric definitions."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).parents[1] / "research/mechinterp_20260911/attention_detail.py"
SPEC = importlib.util.spec_from_file_location("attention_detail", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_one_hot_current_and_oldest_have_zero_entropy_and_expected_lag() -> None:
    weights = np.zeros((2, 1, 1, 4), dtype=np.float32)
    weights[0, 0, 0, -1] = 1.0
    weights[1, 0, 0, 0] = 1.0

    summary = MODULE.summarize_attention(weights)

    assert summary.global_metrics.mean_entropy_nats == pytest.approx(0.0)
    assert summary.global_metrics.mean_exp_entropy == pytest.approx(1.0)
    assert summary.global_metrics.mean_row_max_weight == pytest.approx(1.0)
    assert summary.global_metrics.mean_weighted_lag == pytest.approx(1.5)
    assert summary.global_metrics.current_frame_weight == pytest.approx(0.5)
    assert summary.global_metrics.most_frequent_argmax_lag == 0
    assert summary.global_metrics.argmax_lag_frequency == pytest.approx(0.5)
    assert summary.global_metrics.mean_attention_by_lag.tolist() == pytest.approx([0.5, 0.0, 0.0, 0.5])


def test_uniform_attention_has_log_entropy_and_four_effective_frames() -> None:
    weights = np.full((3, 2, 2, 4), 0.25, dtype=np.float64)

    summary = MODULE.summarize_attention(weights)

    assert summary.global_metrics.mean_entropy_nats == pytest.approx(np.log(4.0))
    assert summary.global_metrics.mean_exp_entropy == pytest.approx(4.0)
    assert summary.global_metrics.mean_row_max_weight == pytest.approx(0.25)
    assert summary.global_metrics.mean_weighted_lag == pytest.approx(1.5)
    assert summary.global_metrics.current_frame_weight == pytest.approx(0.25)
    assert summary.global_metrics.mean_attention_by_lag.tolist() == pytest.approx([0.25] * 4)


def test_effective_context_is_average_exp_entropy_not_exp_average_entropy() -> None:
    weights = np.zeros((2, 1, 1, 4), dtype=np.float64)
    weights[0, 0, 0] = [0.5, 0.5, 0.0, 0.0]
    weights[1, 0, 0, -1] = 1.0

    summary = MODULE.summarize_attention(weights)

    expected_mean_exp = (2.0 + 1.0) / 2.0
    exp_mean_entropy = np.exp(np.log(2.0) / 2.0)
    assert summary.global_metrics.mean_exp_entropy == pytest.approx(expected_mean_exp)
    assert summary.global_metrics.mean_exp_entropy != pytest.approx(exp_mean_entropy)


def test_moving_one_hot_peaks_keep_row_entropy_zero_but_broaden_global_distribution() -> None:
    weights = np.zeros((4, 1, 1, 4), dtype=np.float32)
    for replay in range(4):
        weights[replay, 0, 0, replay] = 1.0

    summary = MODULE.summarize_attention(weights)

    assert summary.global_metrics.mean_entropy_nats == pytest.approx(0.0)
    assert summary.global_metrics.mean_exp_entropy == pytest.approx(1.0)
    assert summary.global_metrics.mean_attention_by_lag.tolist() == pytest.approx([0.25] * 4)
    assert summary.global_metrics.argmax_lag_histogram.tolist() == [1, 1, 1, 1]


def test_nonuniform_metrics_report_peak_and_nearest_global_lag_weight() -> None:
    weights = np.array([[[[0.1, 0.2, 0.3, 0.4]]]], dtype=np.float64)

    summary = MODULE.summarize_attention(weights)

    assert summary.global_metrics.mean_row_max_weight == pytest.approx(0.4)
    assert summary.global_metrics.current_frame_weight == pytest.approx(0.4)
    assert summary.global_metrics.mean_weighted_lag == pytest.approx(1.0)
    assert summary.global_metrics.nearest_global_mean_lag == 1
    assert summary.global_metrics.mean_weight_at_nearest_global_mean_lag == pytest.approx(0.3)
    assert summary.global_metrics.most_frequent_argmax_lag == 0
    assert summary.global_metrics.peak_weight_quantiles["p100"] == pytest.approx(0.4)


@pytest.mark.parametrize(
    "weights",
    [
        np.zeros((1, 1, 1), dtype=np.float32),
        np.zeros((0, 1, 1, 4), dtype=np.float32),
        np.zeros((1, 0, 1, 4), dtype=np.float32),
        np.array([[[[0.5, -0.5]]]], dtype=np.float32),
        np.array([[[[np.nan, 1.0]]]], dtype=np.float32),
        np.array([[[[0.2, 0.2]]]], dtype=np.float32),
    ],
)
def test_invalid_shape_or_probabilities_are_rejected(weights: np.ndarray) -> None:
    with pytest.raises(ValueError):
        MODULE.summarize_attention(weights)
