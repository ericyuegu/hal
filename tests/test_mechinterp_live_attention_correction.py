"""Tests for the archived live-attention lag correction."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).parents[1] / "research/mechinterp_20260911/live_attention_correction.py"
SPEC = importlib.util.spec_from_file_location("live_attention_correction", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_mean_lag_uses_sum_over_keys_for_first_and_current_positions() -> None:
    first = np.zeros((1, 1, 4), dtype=np.float32)
    first[..., 0] = 1.0
    current = np.zeros((1, 1, 4), dtype=np.float32)
    current[..., -1] = 1.0

    assert MODULE.mean_lag_from_weights(first) == pytest.approx(3.0)
    assert MODULE.mean_lag_from_weights(current) == pytest.approx(0.0)


def test_mean_lag_rejects_non_probability_weights() -> None:
    weights = np.zeros((1, 1, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="sum to one"):
        MODULE.mean_lag_from_weights(weights)


@pytest.mark.parametrize("shape", [(0, 1, 4), (1, 0, 4)])
def test_mean_lag_rejects_empty_layer_or_head_axis(shape: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError, match=r"shape \[layer, head, key\]"):
        MODULE.mean_lag_from_weights(np.empty(shape, dtype=np.float32))
