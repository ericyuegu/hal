"""Observation/action codec shared across experiments.

Owns the single source of truth for two wires:

* MDS columns ↔ model-ready tensors (the per-feature routing + normalization
  below, plus the derived spatial block), and
* a 14-channel action vector ↔ :class:`ControllerInputsValue` (the inference
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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
from melee import Stage
from melee.stages import BLASTZONES
from melee.stages import EDGE_POSITION
from torch import Tensor

from hal.data.feature_stats import FeatureStats
from hal.policy import INCLUDED_STAGES
from hal.sim.inputs import action_vec_to_controller as action_vec_to_controller
from hal.training.ego_stats import consolidate_key
from hal.wire import A_DIM
from hal.wire import ACTION_CHANNELS
from hal.wire import ACTION_DIM as _ACTION_DIM
from hal.wire import ITEM_SLOTS
from hal.wire import VELOCITY_COMPONENTS
from hal.wire import item_column
from hal.wire import mask_value

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

# Default categorical table sizes. New experiments can widen the action table.
CAT_FEATURES: dict[str, tuple[int, int]] = {
    "action": (512, 64),
    "stock": (5, 2),
    "jumps_used": (9, 2),
    "hurtbox_state": (4, 2),
    "airborne": (2, 1),
}

# Normalizations a float column can take: see _standardize / _normalize.
FLOAT_TRANSFORMS: Final[tuple[str, ...]] = ("standardize", "minmax")


@dataclass(frozen=True, slots=True)
class ExtraColumns:
    """MDS column suffixes a consumer routes ON TOP OF the two tables above.

    Schema v6 widened the per-frame post block. A model built before that must keep
    its exact input width, so :func:`preprocess` routes the new columns only for the
    consumer that asks for them: with no ``extra`` it drops them, exactly as it did
    before the columns existed. The consumer declares the routing once and passes it
    down BOTH observation paths (train collate and closed-loop replan), so the two
    can never disagree about which columns a model sees.

    ``floats`` maps a suffix to its normalization (one of :data:`FLOAT_TRANSFORMS`).
    ``cats`` maps a suffix to the ``(vocab, embed_dim)`` of the table the model builds
    for it, or to ``None`` when the model indexes an embedding it already owns. The
    tables themselves stay model-side; this declares only the routing and the shapes.
    """

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
    """Raw, ego-relative columns needed by one model."""

    columns: frozenset[str]
    derive_spatial: bool = True


# Schema v6's additions to the per-player post block, less the ones no model reads yet
# (the five raw ``state_flags`` bytes and the global ``item{0..3}_*`` slots).
V6_PLAYER_COLUMNS: Final[ExtraColumns] = ExtraColumns(
    floats={
        # Engine velocities. Self-velocity has separate air and ground slots (``airborne``
        # selects the live one) and knockback is carried apart from it; the engine sums
        # them for the frame's displacement. Heavy-tailed — one knockback spike dwarfs
        # locomotion — so they standardize rather than min-max.
        **{f"velocities_{component}": "standardize" for component in VELOCITY_COMPONENTS},
        # Frames the character has spent in the current action (the engine's own counter).
        "state_age": "standardize",
        # Multiplexed by action state: hitstun frames remaining DURING hitstun states, an
        # unrelated per-state counter elsewhere. Stored raw, so a consumer that reads it
        # must gate on ``action`` — this table only fixes its scale.
        "misc_as": "standardize",
    },
    cats={
        "l_cancel": (3, 2),  # 0 = not applicable, 1 = successful, 2 = unsuccessful
        # Ground/platform id the character stands on. Fifteen ids, max 54, over all six
        # included stages (398 ranked-anonymized-1 val replays), plus the u16 "no ground"
        # sentinel 65535 while airborne — that one clamps into the last row and owns it.
        "ground": (64, 4),
        # Live, transform-aware character id (libmelee internal space). No table of its
        # own: it is the per-frame form of the character-SELECT pick, so a model indexes
        # its existing character embedding with it.
        "character_live": None,
    },
)

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
"""Schema v6's GLOBAL projectile block, ``item{0..3}_*``.

Slots are ordered by ascending spawn id, so a slot keeps its item until an OLDER
item despawns.

``owner`` is deliberately NOT routed. It holds a physical libmelee port (1..4),
and the MDS does not record which ports p1/p2 occupy, so the column cannot be made
ego-relative. It stays dropped until a schema v8 stores those ports.

``type`` is the raw u16 item id: the ``melee.enums.ProjectileType`` values sit in
[6, 210] and 255 means unknown, so a 256-row table covers the space.

Suffix collisions (verified): ``pos_x``, ``pos_y``, ``vel_x``, ``vel_y`` and ``type``
match item columns only. ``state`` also matches ``*_hurtbox_state``, which
:func:`_classify` routes as "cat" through :data:`CAT_FEATURES` in the same clause —
same kind, so no column is misrouted.
"""

ITEM_INPUT_COLUMNS: Final[frozenset[str]] = frozenset(
    item_column(slot, suffix) for slot in range(ITEM_SLOTS) for suffix in (*ITEM_COLUMNS.floats, *ITEM_COLUMNS.cats)
)
"""The raw item columns a model reads: every slot crossed with every routed suffix."""

_NO_EXTRA: Final[ExtraColumns] = ExtraColumns(floats={}, cats={})

# Re-export the policy action wire. Training callers that imported these names
# from this module continue to work while data code imports hal.wire directly.
ACTION_DIM = _ACTION_DIM

BASE_PLAYER_PREFIXES: Final[tuple[str, ...]] = ("ego", "ego_nana", "opp_nana", "opp")
BASE_ACTION_PROJECTION: Final[FeatureProjection] = FeatureProjection(
    columns=frozenset(
        {"stage", "ego_character", "opp_character"}
        | {f"{prefix}_{name}" for prefix in BASE_PLAYER_PREFIXES for name in (*FLOAT_FEATURES, *CAT_FEATURES)}
        | {f"ego_{channel}" for channel in ACTION_CHANNELS}
    ),
    derive_spatial=False,
)

BASE_ITEMS_PROJECTION: Final[FeatureProjection] = FeatureProjection(
    columns=BASE_ACTION_PROJECTION.columns | ITEM_INPUT_COLUMNS,
    derive_spatial=False,
)
""":data:`BASE_ACTION_PROJECTION` plus the projectile block."""

_BUTTON_ORDER = tuple(channel.removeprefix("button_") for channel in ACTION_CHANNELS[6:])

NEUTRAL_ACTION = np.zeros(A_DIM, dtype=np.float32)

_STICK_TRIGGER_SUFFIXES = (
    "main_stick_x",
    "main_stick_y",
    "c_stick_x",
    "c_stick_y",
    "trigger_l",
    "trigger_r",
)


# %%
# --- Derived spatial features -------------------------------------------------
#
# The MDS (schema v5) stores absolute positions but neither velocities nor stage
# geometry, so a model has to infer both through attention. :func:`derive_spatial`
# hands it that state, computed on the fly from the raw columns — no MDS
# re-materialization, and one code path for train and closed-loop eval because
# both funnel through :func:`preprocess`.
#
# Ledge and blastzone geometry come straight from libmelee's stage tables
# (``melee.stages.EDGE_POSITION`` / ``melee.stages.BLASTZONES``, sourced from
# Magus420's frame-data thread), so there is no second geometry table to drift.
# An id that is neither a known stage nor ``Stage.NO_STAGE`` fails loud.
#
# Values stay in RAW game units and are scaled by the fixed constants below.
# Dataset statistics are deliberately NOT used: a distance is a physical quantity
# whose scale is known a priori, and a stats-derived scale would silently change
# the feature when the dataset changes.

# Positions span roughly ±275 (x) / ±355 (y) and blastzone spans reach ~510, so a
# 1/100 scale lands every offset/distance in about [-3, 5].
_SPATIAL_POS_SCALE: Final[float] = 1.0 / 100.0
# Per-frame position deltas: ordinary locomotion is 0-4 units/frame and heavy
# knockback ~30, so 1/10 keeps normal motion visible without crushing hits. The
# rare respawn teleport is a genuine large value and is left unclipped.
_SPATIAL_DPOS_SCALE: Final[float] = 1.0 / 10.0

# The ledge lip sits at y = 0 on every included stage (the main platform surface).
_LEDGE_Y: Final[float] = 0.0

_SPATIAL_PLAYERS: Final[tuple[str, ...]] = ("ego", "opp")

# (name, scale) for the ego/opp-symmetric block, in token order.
_SHARED_SPATIAL: Final[tuple[tuple[str, float], ...]] = (
    ("rel_dx", _SPATIAL_POS_SCALE),  # opp_x - ego_x
    ("rel_dy", _SPATIAL_POS_SCALE),  # opp_y - ego_y
    ("rel_dist", _SPATIAL_POS_SCALE),  # euclidean
    ("rel_dx_ego_facing", _SPATIAL_POS_SCALE),  # +ve => opponent is in front of ego
    ("rel_dx_opp_facing", _SPATIAL_POS_SCALE),  # +ve => ego is in front of the opponent
)

# (suffix, scale) emitted per player in _SPATIAL_PLAYERS, in token order.
_PLAYER_SPATIAL: Final[tuple[tuple[str, float], ...]] = (
    ("ledge_dx", _SPATIAL_POS_SCALE),  # |x| - edge; +ve => past the ledge, out over the void
    ("ledge_dy", _SPATIAL_POS_SCALE),  # y - ledge height; pairs with ledge_dx at ONE scale
    ("offstage", 1.0),  # flag: |x| > edge
    ("blast_left", _SPATIAL_POS_SCALE),  # x - left blastzone   (+ve => inside)
    ("blast_right", _SPATIAL_POS_SCALE),  # right blastzone - x
    ("blast_top", _SPATIAL_POS_SCALE),  # top blastzone - y
    ("blast_bottom", _SPATIAL_POS_SCALE),  # y - bottom blastzone
    ("dpos_x", _SPATIAL_DPOS_SCALE),  # x[t] - x[t-1]
    ("dpos_y", _SPATIAL_DPOS_SCALE),  # y[t] - y[t-1]
)

SPATIAL_FEATURES: Final[tuple[str, ...]] = tuple(name for name, _ in _SHARED_SPATIAL) + tuple(
    f"{player}_{suffix}" for player in _SPATIAL_PLAYERS for suffix, _ in _PLAYER_SPATIAL
)

# ``1.0`` = invalid, matching the ``{feature}_mask`` sidecar convention. Both are
# emitted unconditionally so the model's input width never depends on the data.
# ``spatial_mask`` covers the whole per-frame block; ``spatial_dpos_mask`` is the
# stricter two-frame validity the finite differences need.
SPATIAL_MASKS: Final[tuple[str, ...]] = ("spatial_mask", "spatial_dpos_mask")

SPATIAL_COLUMNS: Final[tuple[str, ...]] = SPATIAL_FEATURES + SPATIAL_MASKS

# The compute-effective subset (8 of 25 columns), in token order. Offstage / ledge /
# bottom-blastzone address the measured failure budget (91% of deaths at the bottom
# blastzone); the two relative channels cover spacing; the frame validity mask stays.
# Dropped: ``ledge_dy`` (linear in ``position_y``), the side and top blast margins (a
# small share of deaths), and the finite-difference ``dpos_*`` proxy with its second
# mask — schema v6 stores the engine's own velocities, which supersede it.
SPATIAL_COLUMNS_LEAN: Final[tuple[str, ...]] = (
    "ego_offstage",
    "opp_offstage",
    "ego_ledge_dx",
    "opp_ledge_dx",
    "ego_blast_bottom",
    "rel_dx_ego_facing",
    "rel_dy",
    "spatial_mask",
)
assert set(SPATIAL_COLUMNS_LEAN) <= set(SPATIAL_COLUMNS)

_SPATIAL_SCALES: Final[dict[str, float]] = {
    **{name: scale for name, scale in _SHARED_SPATIAL},
    **{f"{player}_{suffix}": scale for player in _SPATIAL_PLAYERS for suffix, scale in _PLAYER_SPATIAL},
}

# Raw columns the block is a pure function of. ``stage`` doubles as the gate: a
# batch without it predates matchup conditioning and gets no spatial block.
_SPATIAL_GATE: Final[str] = "stage"
_SPATIAL_INPUTS: Final[tuple[str, ...]] = (
    _SPATIAL_GATE,
    *(f"{player}_{column}" for player in _SPATIAL_PLAYERS for column in ("position_x", "position_y", "direction")),
)


def _stage_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense ``stage id -> geometry`` lookups over the included stages: ``(known,
    edge_x, blastzones)`` where ``blastzones`` rows are ``(left, right, top, bottom)``."""
    size = max(stage.value for stage in INCLUDED_STAGES) + 1
    known = np.zeros(size, dtype=bool)
    edge_x = np.zeros(size, dtype=np.float32)
    blastzones = np.zeros((size, 4), dtype=np.float32)
    for stage in INCLUDED_STAGES:
        known[stage.value] = True
        edge_x[stage.value] = EDGE_POSITION[stage]
        blastzones[stage.value] = BLASTZONES[stage]
    return known, edge_x, blastzones


_STAGE_KNOWN, _STAGE_EDGE_X, _STAGE_BLASTZONES = _stage_tables()


def _stage_known(stage: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(ids, known)`` for a raw stage column. ``Stage.NO_STAGE`` (0) is the
    zero-filled cold-start pad and reads as not-known; any OTHER unmapped id is a
    schema or matchup bug and raises rather than falling back to some geometry."""
    ids = np.asarray(stage).astype(np.int64)
    in_range = (ids >= 0) & (ids < _STAGE_KNOWN.shape[0])
    known = in_range & _STAGE_KNOWN[np.where(in_range, ids, 0)]
    unmapped = ~known & (ids != Stage.NO_STAGE.value)
    if unmapped.any():
        raise ValueError(
            f"stage id(s) {sorted(set(ids[unmapped].tolist()))} have no libmelee ledge/blastzone geometry; "
            f"expected one of {tuple(stage.name for stage in INCLUDED_STAGES)} "
            f"or {Stage.NO_STAGE.value} (the zero-filled cold-start pad)"
        )
    return ids, known


def derive_spatial(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Engineered relative / stage-geometry / velocity features for one ``[L]`` or
    ``[B, L]`` batch of RAW MDS columns, keyed by :data:`SPATIAL_COLUMNS`.

    Both consumers left-pad a not-yet-real frame with an ALL-ZERO row (the train
    window's cold start, the closed-loop rolling buffer's fill and its post
    instant-restart refill), and a zero row is indistinguishable from a legitimate
    0.0 position by value — but not by ``stage``, which reads ``Stage.NO_STAGE``
    exactly on those rows and a real stage id on every observed one. That single
    rule reproduces ``ctx_pad`` identically on both paths without either consumer
    having to plumb it in, and it is what keeps a finite difference from running
    across the pad→real boundary (position ``ctx_pad`` has no predecessor, so its
    delta is masked, not a spurious jump from the origin).

    A frame is *observed* iff its stage is known AND every ego/opp position and
    direction it reads is unmasked; the block is all-or-nothing per frame. A delta
    is valid iff its frame and the one before it are both observed. Invalid entries
    are zeroed and flagged in :data:`SPATIAL_MASKS`.

    Velocities are finite differences of stored positions — a proxy for the engine's
    true per-frame velocity, which schema v5 does not carry; they differ from it
    wherever the engine clamps or teleports (respawns, ledge snaps).
    """
    missing = [name for name in _SPATIAL_INPUTS if name not in batch]
    if missing:
        raise ValueError(f"derive_spatial needs raw columns {missing}, which the batch does not carry")

    ids, known = _stage_known(batch[_SPATIAL_GATE])
    observed = known
    for name in _SPATIAL_INPUTS[1:]:
        observed = observed & ~_is_masked(np.asarray(batch[name]))
    dpos_valid = np.zeros(observed.shape, dtype=bool)
    dpos_valid[..., 1:] = observed[..., 1:] & observed[..., :-1]

    def column(name: str) -> np.ndarray:
        """Raw float column with unobserved frames zeroed, so no NaN enters the arithmetic."""
        return np.where(observed, np.asarray(batch[name], dtype=np.float32), 0.0)

    def delta(values: np.ndarray) -> np.ndarray:
        out = np.zeros_like(values)
        out[..., 1:] = values[..., 1:] - values[..., :-1]
        return out

    safe_ids = np.where(known, ids, 0)
    edge_x = _STAGE_EDGE_X[safe_ids]
    blastzones = _STAGE_BLASTZONES[safe_ids]

    position = {
        player: (column(f"{player}_position_x"), column(f"{player}_position_y")) for player in _SPATIAL_PLAYERS
    }
    ego_x, ego_y = position["ego"]
    opp_x, opp_y = position["opp"]
    rel_dx, rel_dy = opp_x - ego_x, opp_y - ego_y
    raw: dict[str, np.ndarray] = {
        "rel_dx": rel_dx,
        "rel_dy": rel_dy,
        "rel_dist": np.hypot(rel_dx, rel_dy),
        "rel_dx_ego_facing": rel_dx * np.sign(column("ego_direction")),
        "rel_dx_opp_facing": -rel_dx * np.sign(column("opp_direction")),
    }
    for player, (x, y) in position.items():
        raw[f"{player}_ledge_dx"] = np.abs(x) - edge_x
        raw[f"{player}_ledge_dy"] = y - _LEDGE_Y
        raw[f"{player}_offstage"] = (np.abs(x) > edge_x).astype(np.float32)
        raw[f"{player}_blast_left"] = x - blastzones[..., 0]
        raw[f"{player}_blast_right"] = blastzones[..., 1] - x
        raw[f"{player}_blast_top"] = blastzones[..., 2] - y
        raw[f"{player}_blast_bottom"] = y - blastzones[..., 3]
        raw[f"{player}_dpos_x"] = delta(x)
        raw[f"{player}_dpos_y"] = delta(y)

    out = {
        name: (
            np.where(dpos_valid if name.endswith(("_dpos_x", "_dpos_y")) else observed, raw[name], 0.0) * scale
        ).astype(np.float32)
        for name, scale in _SPATIAL_SCALES.items()
    }
    out["spatial_mask"] = (~observed).astype(np.float32)
    out["spatial_dpos_mask"] = (~dpos_valid).astype(np.float32)
    return out


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
    # Optional closed-loop metadata. Training leaves these unset. Evaluation uses them for
    # slot-keyed sampling and for a fresh random stream after a match reset.
    slot_ids: Tensor | None = None  # [B] int64
    reset: Tensor | None = None  # [B] bool

    @property
    def batch(self) -> int:
        return next(iter(self.features.values())).shape[0]

    def to(self, device: str | torch.device) -> Context:
        return Context(
            features={k: v.to(device, non_blocking=True) for k, v in self.features.items()},
            ctx_pad=self.ctx_pad.to(device, non_blocking=True),
            slot_ids=None if self.slot_ids is None else self.slot_ids.to(device, non_blocking=True),
            reset=None if self.reset is None else self.reset.to(device, non_blocking=True),
        )

    def pin_memory(self) -> Context:
        # Page-lock the collated tensors so the DataLoader pin thread enables the
        # async (``non_blocking``) host→device copy in ``to``. Called by torch's
        # pin_memory machinery when the loader has ``pin_memory=True``.
        return Context(
            features={k: v.pin_memory() for k, v in self.features.items()},
            ctx_pad=self.ctx_pad.pin_memory(),
            slot_ids=None if self.slot_ids is None else self.slot_ids.pin_memory(),
            reset=None if self.reset is None else self.reset.pin_memory(),
        )


@dataclass(frozen=True, slots=True)
class TrainBatch:
    """One supervised example batch: a Context plus the action chunk to predict."""

    context: Context
    target: Tensor  # [B, L_chunk, d_action]
    replay_ids: tuple[str, ...] | None = None

    def to(self, device: str | torch.device) -> TrainBatch:
        return TrainBatch(
            context=self.context.to(device),
            target=self.target.to(device, non_blocking=True),
            replay_ids=self.replay_ids,
        )

    def pin_memory(self) -> TrainBatch:
        return TrainBatch(
            context=self.context.pin_memory(), target=self.target.pin_memory(), replay_ids=self.replay_ids
        )

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        """Keep every tensor's allocation owned until ``stream`` finishes using it."""
        tensors = [*self.context.features.values(), self.context.ctx_pad, self.target]
        if self.context.slot_ids is not None:
            tensors.append(self.context.slot_ids)
        if self.context.reset is not None:
            tensors.append(self.context.reset)
        for tensor in tensors:
            tensor.record_stream(stream)


# %%
def _has_suffix(name: str, suffixes: Mapping[str, object] | tuple[str, ...]) -> bool:
    return any(name.endswith(f"_{suffix}") for suffix in suffixes)


def _classify(name: str, extra: ExtraColumns = _NO_EXTRA) -> str:
    if name == "frame":
        return "drop"
    # The spatial block is computed on the fly with its own fixed scalings; routing
    # it as a "float" would look up dataset stats it has no entry for. Seeing one as
    # an INPUT column means the MDS started materializing it — a schema change, not
    # something to silently normalize, so preprocess raises on this kind.
    if name in SPATIAL_COLUMNS:
        return "derived"
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


def _float_transform(name: str, extra: ExtraColumns) -> str:
    """Normalization for one float column. ``extra`` declares it per suffix; otherwise
    percent + position standardize — their dataset max (percent ~507, off the 0-160
    decision range) squashes min-max into a sliver — and every other float min-maxes."""
    for suffix, transform in extra.floats.items():
        if name.endswith(f"_{suffix}"):
            return transform
    return "standardize" if ("position" in name or "percent" in name) else "minmax"


def preprocess(
    batch: dict[str, np.ndarray],
    feature_stats: dict[str, FeatureStats],
    *,
    extra: ExtraColumns | None = None,
    projection: FeatureProjection | None = None,
) -> dict[str, Tensor]:
    """Tokenizer-style per-feature sanitization + per-float mask sidecars.

    Operates on either single-sample ``[L]`` arrays or batched ``[B, L]`` — the
    numpy ops broadcast either way and ``torch.from_numpy`` preserves the shape.
    Sticks/triggers/buttons keep their native ranges (they are the action
    target); only FLOAT_FEATURES are normalized. nana follower columns are
    gamestate-only (float/cat), masked for non-Ice-Climbers players. Columns the
    classifier drops (``frame``, ``schema_version``, ``ctx_pad``) are not returned.

    Batches carrying the matchup-conditioning ``stage`` column additionally get the
    derived :data:`SPATIAL_COLUMNS` block (see :func:`derive_spatial`); it is
    emitted unconditionally from here so train and closed-loop eval share one code
    path, and models that do not want it simply never read those keys.

    ``extra`` (see :class:`ExtraColumns`) routes columns beyond the built-in tables —
    the schema-v6 post block. Without it those columns are dropped, so every model
    built before v6 keeps its exact input width.
    """
    routing = _NO_EXTRA if extra is None else extra
    out: dict[str, Tensor] = {}
    for name, arr in batch.items():
        if projection is not None and name not in projection.columns:
            continue
        kind = _classify(name, routing)
        if kind == "drop":
            continue
        if kind == "derived":
            raise ValueError(
                f"{name!r} arrived as an input column, but the spatial block is derived on the fly by "
                "derive_spatial; materializing it into the MDS needs a schema bump and this derivation removed"
            )
        mask = _is_masked(arr)
        if kind == "button" or kind == "stick_trigger":
            x = np.where(mask, 0.0, arr).astype(np.float32)
        elif kind == "cat":
            x = np.where(mask, 0, arr).astype(np.int64)
        elif kind == "float":
            s = feature_stats[consolidate_key(name)]
            transform = _float_transform(name, routing)
            x = _standardize(arr, s) if transform == "standardize" else _normalize(arr, s)
            x = np.where(mask, 0.0, x)
        else:
            raise AssertionError(f"unhandled kind {kind} for {name}")
        out[name] = torch.from_numpy(np.ascontiguousarray(x))
        if kind == "float" and mask.any():
            out[f"{name}_mask"] = torch.from_numpy(np.ascontiguousarray(mask.astype(np.float32)))
    if _SPATIAL_GATE in batch and (projection is None or projection.derive_spatial):
        for name, value in derive_spatial(batch).items():
            out[name] = torch.from_numpy(np.ascontiguousarray(value))
    return out


def stack_actions(batch: dict[str, Tensor]) -> Tensor:
    """Stack ego action channels in canonical order → ``[B, L, A_DIM]`` over
    whatever sequence length the batch carries (full window at train; L_ctx at
    inference)."""
    return torch.stack([batch[f"ego_{ch}"] for ch in ACTION_CHANNELS], dim=-1)
