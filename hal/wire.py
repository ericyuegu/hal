"""Cross-layer source of truth for slp-native wire conventions.

Imported by both ``hal.data`` (extract / schema / index / filter scripts)
and ``hal.sim`` (inputs, trajectory, session). Anything declared
here is the canonical encoding shared across offline-dataset and online-
emulator code — no other module should re-state what's defined here.

See CLAUDE.md (Controller data model) for the logical-only
controller representation and the peppi → MDS → libmelee → Dolphin data flow.
"""

from collections.abc import Sequence
from typing import Any
from typing import Final

import melee
import numpy as np
import peppi_py.game
from numpy.typing import DTypeLike

# ---------------------------------------------------------------------------
# Policy action wire
# ---------------------------------------------------------------------------

# Canonical ordering of the action vector used by policy datasets and models.
# START is not included: a policy must not open the pause menu during a match.
ACTION_CHANNELS: Final[tuple[str, ...]] = (
    "main_stick_x",
    "main_stick_y",
    "c_stick_x",
    "c_stick_y",
    "trigger_l",
    "trigger_r",
    "button_a",
    "button_b",
    "button_x",
    "button_y",
    "button_z",
    "button_r",
    "button_l",
    "button_d_up",
)
ACTION_DIM: Final[int] = len(ACTION_CHANNELS)
# Compatibility name used by existing model code.
A_DIM: Final[int] = ACTION_DIM

# ---------------------------------------------------------------------------
# Player / port conventions
# ---------------------------------------------------------------------------

# MDS column prefixes for the two players we track per replay (1v1 only).
PLAYER_PREFIXES: Final[tuple[str, str]] = ("p1", "p2")

# All libmelee ports. slp/peppi use 0..3; libmelee uses 1..4.
VALID_LIBMELEE_PORTS: Final[tuple[int, int, int, int]] = (1, 2, 3, 4)


def peppi_port_to_libmelee(peppi_port: peppi_py.game.Port | int) -> int:
    """peppi Port enum (or 0..3 int) -> libmelee port (1..4)."""
    return int(getattr(peppi_port, "value", peppi_port)) + 1


def libmelee_port_to_peppi(port: int) -> int:
    """Inverse of ``peppi_port_to_libmelee`` (returns peppi 0..3 int)."""
    return port - 1


# ---------------------------------------------------------------------------
# Frame conventions
# ---------------------------------------------------------------------------

# Slippi-standard "first in-game frame" id (post-2-second countdown). This is
# a frame_id (peppi's signed counter), not an array index.
GAME_START_FRAME: Final[int] = -123


def dedupe_keep_idx(frame_ids: Sequence[int]) -> np.ndarray:
    """Indices keeping the LAST row per ``frame_id`` — rollback consolidation.

    peppi-py emits one row per recorded slp state including rollback
    corrections, so the same ``frame_id`` can repeat 2-3 times. The final
    occurrence is the engine's committed value. Returned indices are
    ascending so frame order is preserved.
    """
    seen: set[int] = set()
    keep: list[int] = []
    for i in range(len(frame_ids) - 1, -1, -1):
        f = int(frame_ids[i])
        if f in seen:
            continue
        seen.add(f)
        keep.append(i)
    keep.reverse()
    return np.asarray(keep, dtype=np.int64)


# ---------------------------------------------------------------------------
# Button bits (slp pre.buttons_physical)
# ---------------------------------------------------------------------------

# Slp-native bitmasks per the Slippi spec. Single declaration; the MDS column
# names, the libmelee press/release dispatch, and the bit-decode in
# MDS controller playback derives its packed mask from this dict once.
BUTTON_BITS: Final[dict[str, int]] = {
    "a": 0x0100,
    "b": 0x0200,
    "x": 0x0400,
    "y": 0x0800,
    "z": 0x0010,
    "r": 0x0020,
    "l": 0x0040,
    "start": 0x1000,
    "d_up": 0x0008,
}


def slp_button_to_melee(name: str) -> melee.enums.Button:
    """Map an MDS button column suffix (a, b, ..., d_up) to libmelee's enum."""
    return getattr(melee.enums.Button, f"BUTTON_{name.upper()}")


# ---------------------------------------------------------------------------
# Mask sentinels (per-dtype "field unavailable" values)
# ---------------------------------------------------------------------------

# NaN propagates through arithmetic and is the NumPy idiom for missing float
# data. DETECT WITH ``np.isnan(arr)`` — ``arr == nan`` is always False because
# ``nan != nan``. Equality-based detection silently misses every masked entry.
MASK_FLOAT: Final[float] = float("nan")

# Signed int sentinels round-trip through equality normally.
MASK_INT8: Final[int] = -128
MASK_INT16: Final[int] = -(1 << 15)

# int32 uses INT32_MAX (not min) — historical choice; shipped manifests rely
# on it as the int32 sentinel.
MASK_INT32: Final[int] = (1 << 31) - 1

MASK_UINT8: Final[int] = (1 << 8) - 1


def mask_value(dtype: DTypeLike) -> float | int:
    """Dtype-appropriate mask sentinel for an unavailable column or scalar.

    - floats -> ``MASK_FLOAT`` (NaN)
    - signed int < 4 bytes -> ``np.iinfo(dtype).min`` (e.g. int8 -> -128)
    - signed int >= 4 bytes -> ``MASK_INT32``
    - unsigned int -> ``np.iinfo(dtype).max``
    """
    np_dtype = np.dtype(dtype)
    if np.issubdtype(np_dtype, np.floating):
        return MASK_FLOAT
    info = np.iinfo(np_dtype)
    if np_dtype.kind == "i":
        return info.min if np_dtype.itemsize < 4 else MASK_INT32
    return info.max


# ---------------------------------------------------------------------------
# Analog deadzones (Melee input processing)
# ---------------------------------------------------------------------------

# Melee ignores trigger bytes below 43 (of the 140 that means full press), so
# slp physical values under 43/140 are resting-hardware jitter with zero game
# effect (~26% of human frame-shoulder samples). ``extract`` zeroes them so the
# stored per-shoulder trigger is the game-causal signal, mirroring how the slp
# logical stick is already post-deadzone. Pinned empirically: the slp logical
# trigger engages at exactly 43/140 = 0.30714 across human replays.
TRIGGER_DEADZONE: Final[float] = 43.0 / 140.0


# ---------------------------------------------------------------------------
# Stage / character bridges (slp-native int -> libmelee enum)
# ---------------------------------------------------------------------------


def slp_stage_to_libmelee(slp_stage_id: int) -> melee.Stage:
    """slp-native stage id -> ``melee.Stage`` enum.

    Footgun: the two value spaces disagree (e.g. Fountain of Dreams is slp 2
    but ``melee.Stage.FOUNTAIN_OF_DREAMS.value`` = 8). Always go through this.
    """
    stage = melee.enums.to_internal_stage(slp_stage_id)
    if stage is melee.Stage.NO_STAGE:
        raise ValueError(f"unknown slp stage id {slp_stage_id}")
    return stage


# slp "External Character ID" (game-start block) -> libmelee internal Character.
# Two distinct id spaces: the slp start block stores Melee's external (character-
# select) id (Fox=2, Falco=20); libmelee's Character enum is the internal/in-game
# id (Fox=1, Falco=22) reported in every post-frame. They are NOT equal and must
# never be cast into each other. (libmelee's own ``enums.to_internal`` is yet a
# THIRD, cursor-slot numbering — also not this map.) This map exists ONLY to
# normalize the external id at the two peppi reads (``extract_replay`` /
# ``extract_index_entry``) into the internal Character value the index, MDS,
# model, filter, and sim all speak. Anchors verified against post-frame internal
# ids in real replays; the full table is Melee's canonical external id list.
_SLP_EXTERNAL_TO_CHARACTER: Final[dict[int, melee.Character]] = {
    0: melee.Character.CPTFALCON,
    1: melee.Character.DK,
    2: melee.Character.FOX,
    3: melee.Character.GAMEANDWATCH,
    4: melee.Character.KIRBY,
    5: melee.Character.BOWSER,
    6: melee.Character.LINK,
    7: melee.Character.LUIGI,
    8: melee.Character.MARIO,
    9: melee.Character.MARTH,
    10: melee.Character.MEWTWO,
    11: melee.Character.NESS,
    12: melee.Character.PEACH,
    13: melee.Character.PIKACHU,
    14: melee.Character.POPO,  # Ice Climbers; Nana is the follower and has no external id
    15: melee.Character.JIGGLYPUFF,
    16: melee.Character.SAMUS,
    17: melee.Character.YOSHI,
    18: melee.Character.ZELDA,
    19: melee.Character.SHEIK,
    20: melee.Character.FALCO,
    21: melee.Character.YLINK,
    22: melee.Character.DOC,
    23: melee.Character.ROY,
    24: melee.Character.PICHU,
    25: melee.Character.GANONDORF,
}


def slp_character_to_libmelee(slp_character_id: int) -> melee.Character:
    """slp external (character-select) character id -> ``melee.Character`` enum.

    The only external→internal conversion site. Applied at the two peppi reads so
    everything downstream (index, MDS, model, filter, sim) speaks the internal
    Character value.
    """
    char = _SLP_EXTERNAL_TO_CHARACTER.get(slp_character_id)
    if char is None:
        raise ValueError(f"unknown slp character id {slp_character_id}")
    return char


# Character name -> libmelee internal Character value (the id space the index/MDS
# now store). Derived from the selectable-character map so it contains exactly the
# characters that can appear as a start-block pick: NANA (the Ice Climbers follower,
# never a start-block character) and non-playable enum members (wireframes, Giga
# Bowser, sandbag, unknown) are excluded. Used by the filter CLI to resolve
# ``--characters FOX`` against the stored internal ids.
CHARACTERS_BY_NAME: Final[dict[str, int]] = {c.name: int(c.value) for c in _SLP_EXTERNAL_TO_CHARACTER.values()}


# ---------------------------------------------------------------------------
# Post-frame field naming
# ---------------------------------------------------------------------------

# Components of the post-frame ``velocities`` block. ``self_x_air`` and
# ``self_x_ground`` are separate engine slots (which one is live is selected by
# ``airborne``); knockback is carried separately from self-velocity and the two
# are summed to get the frame's actual displacement.
VELOCITY_COMPONENTS: Final[tuple[str, ...]] = (
    "self_x_air",
    "self_x_ground",
    "self_y",
    "knockback_x",
    "knockback_y",
)

# ``state_flags`` is a 5-byte raw bitfield on the action state. Stored undecoded
# (one int column per byte); the per-bit meanings are a decode layer we leave to
# consumers rather than baking a second, drift-prone vocabulary in here.
N_STATE_FLAG_BYTES: Final[int] = 5

# Accessor paths for the post suffixes that are NOT a bare same-named field of
# the post block. Each path is valid against BOTH peppi's ``Post`` (attribute,
# then index into a tuple-of-arrays) and libmelee's canonical post dict (key,
# then tuple index), so one declaration drives the offline and online readers.
_POST_FIELD_PATHS: Final[dict[str, tuple[str | int, ...]]] = {
    "position_x": ("position", "x"),
    "position_y": ("position", "y"),
    **{f"velocities_{c}": ("velocities", c) for c in VELOCITY_COMPONENTS},
    **{f"state_flags_{i}": ("state_flags", i) for i in range(N_STATE_FLAG_BYTES)},
    # The engine's live, transform-aware character (already the libmelee
    # INTERNAL id — no conversion). Renamed because the MDS already spends
    # ``p{1,2}_character`` on the per-replay character-SELECT pick.
    "character_live": ("character",),
}

# MDS column suffixes for the per-frame post block. Names match peppi-py's
# (renamed) ``Post`` dataclass and libmelee's canonical ``Post`` 1:1 except for
# the entries in ``_POST_FIELD_PATHS`` (nested blocks, the indexed state-flag
# bytes, the renamed live character), so a single suffix addresses both sides
# via ``post_field_path``.
POST_FIELD_SUFFIXES: Final[tuple[str, ...]] = (
    "position_x",
    "position_y",
    "percent",
    "shield",
    "stock",
    "direction",
    "action",
    "hitlag_left",
    "jumps_used",
    "airborne",
    "hurtbox_state",
    "character_live",
    "state_age",
    "misc_as",
    "l_cancel",
    "ground",
    *(f"state_flags_{i}" for i in range(N_STATE_FLAG_BYTES)),
    *(f"velocities_{c}" for c in VELOCITY_COMPONENTS),
)


def post_field_path(suffix: str) -> tuple[str | int, ...]:
    """Accessor path for one ``POST_FIELD_SUFFIXES`` entry.

    Walk it with ``getattr``/index against peppi's SoA ``Post`` or with
    key/index against libmelee's canonical post dict — the path is the same
    either way, which is what keeps ``extract``, ``Trajectory`` and the
    closed-loop observation reading identical fields.
    """
    return _POST_FIELD_PATHS.get(suffix, (suffix,))


def canonical_post_field(post: dict, suffix: str) -> float:
    """Read one ``POST_FIELD_SUFFIXES`` value from a libmelee canonical post dict
    (the shape ``Session.step`` yields). A field absent on this slp/build — or
    nested under a block this slp version never recorded — comes back as
    ``MASK_FLOAT`` (NaN), the same mask convention ``Trajectory.from_slp`` uses.

    Shared by ``sim.trajectory.from_capture`` and
    ``training.canonical.flatten_canonical_frame`` so the two never drift.
    """
    value: Any = post
    for step in post_field_path(suffix):
        value = value.get(step) if isinstance(value, dict) else value[step]
        if value is None:
            return MASK_FLOAT
    return float(value)


# ---------------------------------------------------------------------------
# Items / projectiles
# ---------------------------------------------------------------------------

# Item slots stored per frame. Melee tracks up to 15 live items; 4 covers the
# 1v1 projectile load (lasers, needles, turnips, bombs, arrows) at a bounded
# column cost. Absent slots carry the dtype mask sentinel.
ITEM_SLOTS: Final[int] = 4

# Accessor paths for one item's stored fields, valid against BOTH peppi's SoA
# ``Item`` and libmelee's canonical item dict — same contract as
# ``_POST_FIELD_PATHS``. Ordering of this dict is the column order.
ITEM_FIELD_PATHS: Final[dict[str, tuple[str, ...]]] = {
    "type": ("type",),
    "state": ("state",),
    "pos_x": ("position", "x"),
    "pos_y": ("position", "y"),
    "vel_x": ("velocity", "x"),
    "vel_y": ("velocity", "y"),
    "owner": ("owner",),
}
ITEM_FIELD_SUFFIXES: Final[tuple[str, ...]] = tuple(ITEM_FIELD_PATHS)

# Slot assignment is by ASCENDING SPAWN ID (``Item.id``, the engine's
# monotonically increasing spawn counter). That makes the slot ordering
# deterministic, identical for the offline (peppi) and online (libmelee)
# readers, and stable frame-to-frame: an item keeps its slot until an OLDER
# item despawns. Overflow past ``ITEM_SLOTS`` drops the newest items.
ITEM_SPAWN_ID_FIELD: Final[str] = "id"

# The slp records -1 as the owner of an unowned item (most stage/neutral items).
ITEM_OWNER_NONE: Final[int] = -1


def item_column(slot: int, suffix: str) -> str:
    """MDS column name for one item slot's field. Items are global state, not
    per-player, so they carry no ``p{1,2}`` prefix."""
    return f"item{slot}_{suffix}"


def item_owner_to_libmelee_port(owner: int) -> int:
    """slp item owner -> libmelee port (1..4); ``ITEM_OWNER_NONE`` passes through.

    The raw slp value is a peppi 0..3 port, while every other port the MDS and
    index store is already libmelee 1..4. Normalizing here keeps ``owner``
    directly comparable to the ports p1/p2 were assigned from.
    """
    if owner == ITEM_OWNER_NONE:
        return ITEM_OWNER_NONE
    if not 0 <= owner < len(VALID_LIBMELEE_PORTS):
        raise ValueError(f"item owner {owner} is neither {ITEM_OWNER_NONE} (unowned) nor a peppi port 0..3")
    return peppi_port_to_libmelee(owner)


def canonical_item_field(item: dict, suffix: str) -> float:
    """Read one ``ITEM_FIELD_SUFFIXES`` value from a libmelee canonical item dict.
    Absent on this slp/build -> ``MASK_FLOAT``, matching ``canonical_post_field``."""
    value: Any = item
    for step in ITEM_FIELD_PATHS[suffix]:
        value = value.get(step) if isinstance(value, dict) else value[step]
        if value is None:
            return MASK_FLOAT
    if suffix == "owner":
        return float(item_owner_to_libmelee_port(int(value)))
    return float(value)


def canonical_item_columns(items: Sequence[dict] | None) -> dict[str, float]:
    """Flat ``item{k}_*`` columns for one frame of a libmelee canonical frame's
    ``items`` list, ordered by ascending spawn id (see ``ITEM_SPAWN_ID_FIELD``).

    Every slot is emitted every frame: slots past the live-item count — and all
    of them when the slp/build predates item events (``items is None``) — carry
    ``MASK_FLOAT``, so the online column set matches the MDS unconditionally.
    """
    out = {item_column(slot, s): MASK_FLOAT for slot in range(ITEM_SLOTS) for s in ITEM_FIELD_SUFFIXES}
    if not items:
        return out
    live = sorted(items, key=lambda it: it[ITEM_SPAWN_ID_FIELD])[:ITEM_SLOTS]
    for slot, item in enumerate(live):
        for suffix in ITEM_FIELD_SUFFIXES:
            out[item_column(slot, suffix)] = canonical_item_field(item, suffix)
    return out
