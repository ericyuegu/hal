"""Observation/action codec shared across experiments.

Owns the single source of truth for two wires:

* MDS columns ↔ model-ready tensors (the per-feature routing + normalization
  below), and
* a 15-channel action vector ↔ :class:`ControllerInputsValue` (the inference
  output bridge).

Plus the typed model I/O value objects :class:`Context` and :class:`TrainBatch`.

Kept **side-effect-free** (no module-level CUDA / device probing) so that
forkserver-spawned DataLoader workers can re-import it to run :func:`preprocess`
in-process — the same constraint that keeps ``dataloader.py`` importable there.

Tensor-dim names (docstrings):
    B           = batch
    L           = sequence length carried by the batch (window at train, L_ctx at inference)
    L_ctx       = context length
    L_chunk     = predicted chunk length
    d_action    = action vector dim (A_DIM)
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from hal.data.stats import FeatureStats
from hal.sim.inputs import ControllerInputsValue
from hal.training.stats import consolidate_key
from hal.wire import BUTTON_BITS
from hal.wire import mask_value

A_DIM = 14  # 4 sticks + 2 triggers + 8 buttons (START excluded — see ACTION_CHANNELS)

# Continuous gamestate features (normalized via FeatureStats). Sticks, triggers,
# buttons and categoricals are routed separately below.
FLOAT_FEATURES: tuple[str, ...] = (
    "position_x",
    "position_y",
    "percent",
    "shield",
    "direction",
    "hitlag_left",
)

# (vocab_size, embed_dim) per categorical feature. action vocab rounded up
# from libmelee's ~395 to a safe power-of-two-ish.
CAT_FEATURES: dict[str, tuple[int, int]] = {
    "action": (512, 32),
    "stock": (5, 8),
    "jumps_used": (9, 8),
    "hurtbox_state": (4, 8),
    "airborne": (2, 4),
}

# Canonical ordering of the 14-channel ego action vector. Matches
# ControllerInputsValue field order for sticks/triggers and BUTTON_BITS for
# buttons. Used as model target and to construct ControllerInputsValue at
# inference — must stay in lockstep with _BUTTON_ORDER / action_vec_to_controller.
# START is deliberately absent: pressing it opens Melee's pause menu, which drops
# the match out of IN_GAME and ends the rollout, so the policy can never emit it.
# Excluding it from the action space spares the model a target dim it can't use;
# round-trip replay of recorded inputs uses a different path and keeps START.
ACTION_CHANNELS: tuple[str, ...] = (
    "main_stick_x",
    "main_stick_y",
    "c_stick_x",
    "c_stick_y",
    "trigger_l_physical",
    "trigger_r_physical",
    "button_a",
    "button_b",
    "button_x",
    "button_y",
    "button_z",
    "button_r",
    "button_l",
    "button_d_up",
)
assert len(ACTION_CHANNELS) == A_DIM

_BUTTON_ORDER = ("a", "b", "x", "y", "z", "r", "l", "d_up")
# raw_* dropped (we use logical sticks); nana_* skipped for now; trigger_logical
# redundant with the physical l/r channels already consumed.
_DROP_PATTERNS = ("_raw_", "_nana_", "_trigger_logical")

NEUTRAL_ACTION = np.zeros(A_DIM, dtype=np.float32)

_STICK_TRIGGER_SUFFIXES = (
    "main_stick_x",
    "main_stick_y",
    "c_stick_x",
    "c_stick_y",
    "trigger_l_physical",
    "trigger_r_physical",
)


# %%
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


@dataclass(frozen=True, slots=True)
class TrainBatch:
    """One supervised example batch: a Context plus the action chunk to predict."""

    context: Context
    target: Tensor  # [B, L_chunk, d_action]

    def to(self, device: str | torch.device) -> TrainBatch:
        return TrainBatch(context=self.context.to(device), target=self.target.to(device, non_blocking=True))


# %%
def _classify(name: str) -> str:
    if name == "frame":
        return "drop"
    if any(p in name for p in _DROP_PATTERNS):
        return "drop"
    if any(name.endswith(f"_{c}") for c in CAT_FEATURES):
        return "cat"
    if "_button_" in name:
        return "button"
    if any(name.endswith(f"_{s}") for s in _STICK_TRIGGER_SUFFIXES):
        return "stick_trigger"
    if any(name.endswith(f"_{f}") for f in FLOAT_FEATURES):
        return "float"
    return "drop"


def _is_masked(arr: np.ndarray) -> np.ndarray:
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


def preprocess(
    batch: dict[str, np.ndarray],
    feature_stats: dict[str, FeatureStats],
) -> dict[str, Tensor]:
    """Tokenizer-style per-feature sanitization + per-float mask sidecars.

    Operates on either single-sample ``[L]`` arrays or batched ``[B, L]`` — the
    numpy ops broadcast either way and ``torch.from_numpy`` preserves the shape.
    Sticks/triggers/buttons keep their raw ranges (they are the action target);
    only FLOAT_FEATURES are normalized. Columns the classifier drops (``frame``,
    ``ctx_pad``, raw/nana/trigger_logical) are not returned.
    """
    out: dict[str, Tensor] = {}
    for name, arr in batch.items():
        kind = _classify(name)
        if kind == "drop":
            continue
        mask = _is_masked(arr)
        if kind == "button" or kind == "stick_trigger":
            x = np.where(mask, 0.0, arr).astype(np.float32)
        elif kind == "cat":
            x = np.where(mask, 0, arr).astype(np.int64)
        elif kind == "float":
            s = feature_stats[consolidate_key(name)]
            x = _standardize(arr, s) if "position" in name else _normalize(arr, s)
            x = np.where(mask, 0.0, x)
        else:
            raise AssertionError(f"unhandled kind {kind} for {name}")
        out[name] = torch.from_numpy(np.ascontiguousarray(x))
        if kind == "float" and mask.any():
            out[f"{name}_mask"] = torch.from_numpy(np.ascontiguousarray(mask.astype(np.float32)))
    return out


def stack_actions(batch: dict[str, Tensor]) -> Tensor:
    """Stack ego action channels in canonical order → ``[B, L, A_DIM]`` over
    whatever sequence length the batch carries (full window at train; L_ctx at
    inference)."""
    return torch.stack([batch[f"ego_{ch}"] for ch in ACTION_CHANNELS], dim=-1)


def action_vec_to_controller(a: np.ndarray) -> ControllerInputsValue:
    """One A_DIM-vector → one ControllerInputsValue. Buttons threshold at 0.5.
    START is not in the action space (see ACTION_CHANNELS), so the policy can
    never press it."""
    a = np.asarray(a).reshape(-1)
    buttons = 0
    for i, name in enumerate(_BUTTON_ORDER):
        if a[6 + i] > 0.5:
            buttons |= BUTTON_BITS[name]
    return ControllerInputsValue(
        main_x=float(np.clip(a[0], -1.0, 1.0)),
        main_y=float(np.clip(a[1], -1.0, 1.0)),
        c_x=float(np.clip(a[2], -1.0, 1.0)),
        c_y=float(np.clip(a[3], -1.0, 1.0)),
        trigger_l=float(np.clip(a[4], 0.0, 1.0)),
        trigger_r=float(np.clip(a[5], 0.0, 1.0)),
        buttons=int(buttons),
    )
