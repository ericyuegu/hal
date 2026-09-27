"""Observation columns and tensors shared by training and inference."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
from torch import Tensor

from hal.data.feature_stats import FeatureStats
from hal.data.feature_stats import consolidate_key
from hal.wire import ACTION_CHANNELS
from hal.wire import ACTION_DIM
from hal.wire import ITEM_SLOTS
from hal.wire import item_column
from hal.wire import mask_value

FLOAT_FEATURES: tuple[str, ...] = (
    "position_x",
    "position_y",
    "percent",
    "shield",
    "direction",
    "hitlag_left",
)


CAT_FEATURES: dict[str, tuple[int, int]] = {
    "action": (512, 64),
    "stock": (5, 2),
    "jumps_used": (9, 2),
    "hurtbox_state": (4, 2),
    "airborne": (2, 1),
}


FLOAT_TRANSFORMS: Final[tuple[str, ...]] = ("standardize", "minmax")


@dataclass(frozen=True, slots=True)
class ExtraColumns:
    """Additional column routing declared by the model representation."""

    floats: Mapping[str, str]
    cats: Mapping[str, tuple[int, int] | None]

    def __post_init__(self) -> None:
        unknown = sorted(set(self.floats.values()) - set(FLOAT_TRANSFORMS))
        if unknown:
            raise ValueError(f"unknown float transform(s) {unknown}; expected one of {FLOAT_TRANSFORMS}")
        both = sorted(set(self.floats) & set(self.cats))
        if both:
            raise ValueError(f"{both} declared as both a float and a categorical column")


@dataclass(frozen=True, slots=True)
class FeatureProjection:
    columns: frozenset[str]


ITEM_COLUMNS: Final[ExtraColumns] = ExtraColumns(
    floats={
        # Item position and velocity, in the same raw game units as the player
        # positions. Standardize: projectile speeds are heavy-tailed.
        "pos_x": "standardize",
        "pos_y": "standardize",
        "vel_x": "standardize",
        "vel_y": "standardize",
    },
    cats={
        "type": (256, 16),
        # Per-item engine state. Its meaning depends on the item type, so a model
        # reads it together with ``type``.
        "state": (256, 4),
    },
)


ITEM_FLOATS: Final[tuple[str, ...]] = tuple(ITEM_COLUMNS.floats)


ITEM_CAT_VOCABS: Final[dict[str, int]] = {
    name: spec[0] for name, spec in ITEM_COLUMNS.cats.items() if spec is not None
}


ITEM_PRESENCE_SUFFIX: Final[str] = "pos_x"


ITEM_PROBE_COLUMN: Final[str] = item_column(0, ITEM_PRESENCE_SUFFIX)


ITEM_INPUT_COLUMNS: Final[frozenset[str]] = frozenset(
    item_column(slot, suffix) for slot in range(ITEM_SLOTS) for suffix in (*ITEM_COLUMNS.floats, *ITEM_COLUMNS.cats)
)


NO_EXTRA_COLUMNS: Final[ExtraColumns] = ExtraColumns(floats={}, cats={})


BASE_PLAYER_PREFIXES: Final[tuple[str, ...]] = ("ego", "ego_nana", "opp_nana", "opp")


BASE_ACTION_PROJECTION: Final[FeatureProjection] = FeatureProjection(
    columns=frozenset(
        {"stage", "ego_character", "opp_character"}
        | {f"{prefix}_{name}" for prefix in BASE_PLAYER_PREFIXES for name in (*FLOAT_FEATURES, *CAT_FEATURES)}
        | {f"ego_{channel}" for channel in ACTION_CHANNELS}
    ),
)


BASE_ITEMS_PROJECTION: Final[FeatureProjection] = FeatureProjection(
    columns=BASE_ACTION_PROJECTION.columns | ITEM_INPUT_COLUMNS,
)


ITEM_PLAYER_COLUMNS: Final[ExtraColumns] = ExtraColumns(
    floats=ITEM_COLUMNS.floats,
    cats={**ITEM_COLUMNS.cats, "player_id": None},
)


ITEM_PLAYER_PROJECTION: Final[FeatureProjection] = FeatureProjection(
    columns=BASE_ITEMS_PROJECTION.columns | {"ego_player_id"},
)


NEUTRAL_ACTION = np.zeros(ACTION_DIM, dtype=np.float32)


_STICK_TRIGGER_SUFFIXES = (
    "main_stick_x",
    "main_stick_y",
    "c_stick_x",
    "c_stick_y",
    "trigger_l",
    "trigger_r",
)


@dataclass(frozen=True, slots=True)
class Context:
    """The observed gamestate the model conditions on. Built identically by the
    train dataloader and the closed-loop driver, so the model never branches on
    which.

    ``features`` carries per-feature columns at length ``L_ctx`` (normalized
    floats + their mask sidecars + int64 categorical ids + raw stick/trigger/
    button channels, including the ego's own controller history). ``ctx_pad``
    hides each sample's not-yet-filled leftmost context positions from attention.

    Deliberately neutral: any already-committed action prefix an RTC experiment
    conditions on is part of the predicted chunk (at train) or supplied to the
    inference integrator (at eval), not carried here.
    """

    features: dict[str, Tensor]
    ctx_pad: Tensor  # [B] int64

    @property
    def batch(self) -> int:
        return next(iter(self.features.values())).shape[0]

    def to(self, device: str | torch.device) -> Context:
        return Context(
            features={k: v.to(device, non_blocking=True) for k, v in self.features.items()},
            ctx_pad=self.ctx_pad.to(device, non_blocking=True),
        )

    def pin_memory(self) -> Context:
        # Page-lock the collated tensors so the DataLoader pin thread enables the
        # async (``non_blocking``) host→device copy in ``to``. Called by torch's
        # pin_memory machinery when the loader has ``pin_memory=True``.
        return Context(
            features={k: v.pin_memory() for k, v in self.features.items()},
            ctx_pad=self.ctx_pad.pin_memory(),
        )


def _has_suffix(name: str, suffixes: Mapping[str, object] | tuple[str, ...]) -> bool:
    return any(name.endswith(f"_{suffix}") for suffix in suffixes)


def feature_kind(name: str, extra: ExtraColumns = NO_EXTRA_COLUMNS) -> str:
    if name == "frame":
        return "drop"
    # Extra floats resolve BEFORE any categorical: ``velocities_self_x_ground`` also
    # ends with the ``ground`` categorical's suffix.
    if _has_suffix(name, extra.floats):
        return "float"
    # Global stage + per-player character: int categoricals joined from the replay
    # manifest (not in the per-frame MDS). Inert unless those columns are present.
    if name == "stage" or name.endswith("_character"):
        return "cat"
    if _has_suffix(name, CAT_FEATURES) or _has_suffix(name, extra.cats):
        return "cat"
    if "_button_" in name:
        return "button"
    if _has_suffix(name, _STICK_TRIGGER_SUFFIXES):
        return "stick_trigger"
    if _has_suffix(name, FLOAT_FEATURES):
        return "float"
    return "drop"


def mask_sentinel_positions(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind == "f":
        return np.isnan(arr)
    return arr == mask_value(arr.dtype)


def _normalize(arr: np.ndarray, s: FeatureStats) -> np.ndarray:
    if s.max == s.min:
        return np.zeros_like(arr, dtype=np.float32)
    return (2.0 * (arr - s.min) / (s.max - s.min) - 1.0).astype(np.float32)


def _standardize(arr: np.ndarray, s: FeatureStats) -> np.ndarray:
    if s.std == 0:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - s.mean) / s.std).astype(np.float32)


def float_feature_transform(name: str, extra: ExtraColumns) -> str:
    """Normalization for one float column. ``extra`` declares it per suffix; otherwise
    percent + position standardize — their dataset max (percent ~507, off the 0-160
    decision range) squashes min-max into a sliver — and every other float min-maxes."""
    for suffix, transform in extra.floats.items():
        if name.endswith(f"_{suffix}"):
            return transform
    return "standardize" if ("position" in name or "percent" in name) else "minmax"


def preprocess(
    batch: Mapping[str, np.ndarray],
    feature_stats: dict[str, FeatureStats],
    *,
    extra: ExtraColumns | None = None,
    projection: FeatureProjection | None = None,
) -> dict[str, Tensor]:
    """Normalize selected columns, preserving controller values and missing-data masks."""
    routing = NO_EXTRA_COLUMNS if extra is None else extra
    out: dict[str, Tensor] = {}
    for name, arr in batch.items():
        if projection is not None and name not in projection.columns:
            continue
        kind = feature_kind(name, routing)
        if kind == "drop":
            continue
        mask = mask_sentinel_positions(arr)
        if kind == "button" or kind == "stick_trigger":
            x = np.where(mask, 0.0, arr).astype(np.float32)
        elif kind == "cat":
            x = np.where(mask, 0, arr).astype(np.int64)
        elif kind == "float":
            s = feature_stats[consolidate_key(name)]
            transform = float_feature_transform(name, routing)
            x = _standardize(arr, s) if transform == "standardize" else _normalize(arr, s)
            x = np.where(mask, 0.0, x)
        else:
            raise AssertionError(f"unhandled kind {kind} for {name}")
        out[name] = torch.from_numpy(np.ascontiguousarray(x))
        if kind == "float" and mask.any():
            out[f"{name}_mask"] = torch.from_numpy(np.ascontiguousarray(mask.astype(np.float32)))
    return out


def stack_actions(batch: dict[str, Tensor]) -> Tensor:
    """Stack ego action channels in canonical order → ``[B, L, ACTION_DIM]`` over
    whatever sequence length the batch carries (full window at train; L_ctx at
    inference)."""
    return torch.stack([batch[f"ego_{ch}"] for ch in ACTION_CHANNELS], dim=-1)
