"""Frame-local damage and stock events shared by storage and training."""

import numpy as np

from hal.wire import MASK_INT32


def stock_loss_events(stock: np.ndarray) -> np.ndarray:
    """Return 1 on frames where a known stock count decreases."""
    ids = np.asarray(stock).astype(np.int64)
    known = ids != MASK_INT32
    out = np.zeros(ids.shape, dtype=np.float32)
    out[1:] = ((ids[1:] < ids[:-1]) & known[1:] & known[:-1]).astype(np.float32)
    return out


def match_point_events(stock: np.ndarray) -> np.ndarray:
    """Return 1 when a known stock count decreases to zero."""
    ids = np.asarray(stock).astype(np.int64)
    return stock_loss_events(stock) * (ids == 0).astype(np.float32)


def damage_taken(percent: np.ndarray) -> np.ndarray:
    """Return positive per-frame percent changes; treat invalid values as resets."""
    values = np.asarray(percent, dtype=np.float32)
    out = np.zeros(values.shape, dtype=np.float32)
    out[1:] = np.maximum(values[1:] - values[:-1], 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
