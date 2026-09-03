"""Materialization and loader labels for O51 replay bands."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from hal.data.o51_schema import O51_MDS_SCHEMA_VERSION
from hal.data.o51_schema import O51_RETURN_SUFFIX
from hal.data.policy_world_schema import POLICY_WORLD_MDS_COLUMNS
from hal.data.policy_world_schema import POLICY_WORLD_SCHEMA_VERSION
from hal.training import returns as returns_lib
from hal.training.player_identity import MASKED_PLAYER_ID


def _scalar_int(source: Mapping[str, object], name: str) -> int:
    value = np.asarray(source[name])
    if value.shape:
        raise ValueError(f"{name} must be scalar, got {value.shape}")
    return int(value.item())


def _player_scalar(labels: Mapping[str, np.ndarray], name: str, frames: int) -> int:
    values = np.asarray(labels[name])
    if values.shape != (frames,) or not np.all(values == values[0]):
        raise ValueError(f"{name} must be one replay-wide player ID")
    value = int(values[0])
    if value < MASKED_PLAYER_ID:
        raise ValueError(f"{name} has invalid player ID {value}")
    return value


def encode_o51_replay(
    compact: Mapping[str, object],
    *,
    player_labels: Mapping[str, np.ndarray],
    gamma: float,
    damage_shaping: float,
    win_reward: float,
    stock_value: float,
) -> dict[str, object]:
    """Attach both ports' returns, masks, and scalar identities to one row."""
    if _scalar_int(compact, "policy_world_schema_version") != POLICY_WORLD_SCHEMA_VERSION:
        raise ValueError("O51 materialization requires policy-world-v7 rows")
    frames = _scalar_int(compact, "num_frames")
    if frames < 1:
        raise ValueError(f"num_frames must be positive, got {frames}")
    returns = returns_lib.compact_policy_returns(
        compact,
        gamma=gamma,
        damage_shaping=damage_shaping,
        win_reward=win_reward,
        stock_value=stock_value,
        suffix=O51_RETURN_SUFFIX,
    )
    out = {name: compact[name] for name in POLICY_WORLD_MDS_COLUMNS}
    out["o51_schema_version"] = O51_MDS_SCHEMA_VERSION
    for port in ("p1", "p2"):
        values = np.asarray(returns[f"{port}_{O51_RETURN_SUFFIX}"], dtype=np.float32)
        valid = np.asarray(returns[f"{port}_{O51_RETURN_SUFFIX}_valid"], dtype=np.bool_)
        if values.shape != (frames,) or valid.shape != (frames,):
            raise ValueError(f"{port} return labels have the wrong shape")
        if not np.isfinite(values[valid]).all() or not np.isnan(values[~valid]).all():
            raise ValueError(f"{port} return values do not match their validity mask")
        out[f"{port}_{O51_RETURN_SUFFIX}"] = values
        out[f"{port}_{O51_RETURN_SUFFIX}_valid"] = valid.astype(np.uint8)
        out[f"{port}_player_id"] = _player_scalar(player_labels, f"{port}_player_id", frames)
    return out


def o51_replay_labels(compact: Mapping[str, object]) -> dict[str, np.ndarray]:
    """Return full return arrays and scalar IDs for window-local expansion."""
    if _scalar_int(compact, "o51_schema_version") != O51_MDS_SCHEMA_VERSION:
        raise ValueError("row is not an O51 training replay")
    frames = _scalar_int(compact, "num_frames")
    labels: dict[str, np.ndarray] = {}
    for port in ("p1", "p2"):
        returns = np.asarray(compact[f"{port}_{O51_RETURN_SUFFIX}"], dtype=np.float32)
        valid = np.asarray(compact[f"{port}_{O51_RETURN_SUFFIX}_valid"], dtype=np.uint8)
        if returns.shape != (frames,) or valid.shape != (frames,) or not np.isin(valid, (0, 1)).all():
            raise ValueError(f"stored {port} O51 labels are structurally invalid")
        selected = valid.astype(bool)
        if not np.isfinite(returns[selected]).all() or not np.isnan(returns[~selected]).all():
            raise ValueError(f"stored {port} O51 returns do not match their mask")
        labels[f"{port}_{O51_RETURN_SUFFIX}"] = returns
        labels[f"{port}_{O51_RETURN_SUFFIX}_valid"] = selected
        player_id = _scalar_int(compact, f"{port}_player_id")
        if player_id < MASKED_PLAYER_ID:
            raise ValueError(f"stored {port} player ID is invalid")
        labels[f"{port}_player_id"] = np.asarray(player_id, dtype=np.int32)
    return labels
