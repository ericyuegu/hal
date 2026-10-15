"""Correct the archived live-attention mean-lag reduction."""

from __future__ import annotations

import numpy as np


def mean_lag_from_weights(weights: np.ndarray) -> float:
    """Return mean key lag after reducing each attention distribution.

    Args:
        weights: Attention probabilities with shape ``[layer, head, key]``.

    Returns:
        Mean lag in context frames, where the current key has lag zero.

    Raises:
        ValueError: If the array is not a finite probability tensor.
    """
    if weights.ndim != 3 or min(weights.shape) < 1:
        raise ValueError(f"attention weights must have shape [layer, head, key], got {weights.shape}")
    values = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("attention weights must be finite and non-negative")
    if not np.allclose(values.sum(axis=-1), 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError("attention weights must sum to one over keys")
    lags = np.arange(values.shape[-1] - 1, -1, -1, dtype=np.float64)
    return float(np.sum(values * lags, axis=-1).mean())
