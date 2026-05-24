"""Cross-layer source of truth for slp-native wire conventions.

Imported by both ``hal.data`` (extract / schema / index / filter scripts)
and ``hal.sim`` (inputs, trajectory, session). Anything declared
here is the canonical encoding shared across offline-dataset and online-
emulator code — no other module should re-state what's defined here.

See CLAUDE.md (Architecture) for terminology (raw / wire / logical / physical)
and the data-flow diagram (peppi → MDS → controller view → libmelee → Dolphin).
"""

from collections.abc import Sequence
from typing import Final

import melee
import numpy as np
import peppi_py.game
from numpy.typing import DTypeLike

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
# MdsControllerView all derive from this dict at import time.
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
# Analog stick wire format (Dolphin pipe protocol)
# ---------------------------------------------------------------------------

# Delegate to libmelee's public ``raw_byte_to_wire`` so the wire-format math
# (and the +0.1 fudge) lives in exactly one place — the same module that owns
# ``fix_analog_stick`` and the Controller pipe writer.
raw_byte_to_wire = melee.controller.raw_byte_to_wire


def wire_to_raw_byte(wire: float) -> int:
    """Inverse: recover the int8 byte that a wire float deserializes to in
    Dolphin's parser. Useful for round-trip validation."""
    return int((wire - 0.5) * 254.0 - 0.1)


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


def slp_character_to_libmelee(slp_character_id: int) -> melee.Character:
    """slp-native character id -> ``melee.Character`` enum.

    The two value spaces *happen* to coincide today (pinned by
    ``tests/test_wire_bridges.py``); the bridge exists anyway so intent
    is explicit and a future divergence shows up as a localized failure.
    """
    return melee.Character(slp_character_id)


# Standard cast — slp-native ids 0..26 are the playable characters; higher ids
# (WIRE_FRAME, MASTER_HAND, ...) are non-selectable. Derived rather than
# enumerated so any libmelee enum update flows through.
CHARACTERS_BY_NAME: Final[dict[str, int]] = {c.name: int(c.value) for c in melee.Character if 0 <= int(c.value) <= 26}


# ---------------------------------------------------------------------------
# Post-frame field naming
# ---------------------------------------------------------------------------

# MDS column suffixes for the per-frame post block. Names match peppi-py's
# (renamed) ``Post`` dataclass and libmelee's canonical ``Post`` 1:1, so a
# single suffix is all that's needed to address both.
#
# Special case that consumers handle at call sites (not encoded in this list):
#   - ``position_x`` / ``position_y``: peppi nests them under
#     ``post.position.{x,y}``.
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
)


def canonical_post_field(post: dict, suffix: str) -> float:
    """Read one ``POST_FIELD_SUFFIXES`` value from a libmelee canonical post dict
    (the shape ``Session.step`` yields). ``position_{x,y}`` are nested under
    ``post['position']``; a field absent on this slp/build comes back as
    ``MASK_FLOAT`` (NaN), the same mask convention ``Trajectory.from_slp`` uses.

    Shared by ``sim.trajectory.from_capture`` and
    ``training.canonical.flatten_canonical_frame`` so the two never drift.
    """
    if suffix == "position_x":
        return float(post["position"]["x"])
    if suffix == "position_y":
        return float(post["position"]["y"])
    value = post.get(suffix)
    return float(value) if value is not None else MASK_FLOAT
