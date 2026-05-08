"""peppi-py-based per-frame extractor.

`extract_replay(path) -> dict[str, np.ndarray] | None` returns one ndarray per
column in `MDS_PER_FRAME_DTYPES`, length=frame_count, ready to feed into
`MDSWriter.write`. Returns None on any unrecoverable parse failure.

Key choices:

- **No mutation.** Sticks stay peppi-native [-1, 1]. Direction stays
  peppi-native -1.0/1.0. Ego/opponent perspective is a runtime concern.
- **Buttons** are unpacked from `pre.buttons_physical` bitmask (slp spec).
- **Raw analog bytes** (slp version-gated) are filled with NP_MASK_VALUE
  when peppi reports None.
- **action_frame** is derived as a 1-indexed run-length counter on
  `post.state` — this matches libmelee's `action_frame` convention bit-for-bit
  (verified in notebooks/peppi_vs_libmelee.py).
- **Nana**: when a player isn't Ice Climbers, all `*_nana_*` columns are
  filled with the mask sentinel.
- **Frame range**: peppi's `frames.id` includes pre-game (negative) frames.
  We trim to `id >= -123` (Slippi standard "start of game" frame) to match
  libmelee's IN_GAME / SUDDEN_DEATH filter.
"""

import numpy as np
import peppi_py
from loguru import logger
from numpy.typing import DTypeLike

from hal.constants import NP_MASK_VALUE
from hal.data.schema import MDS_PER_FRAME_DTYPES
from hal.data.schema import PLAYER_PREFIXES

# slp Slippi-standard "first in-game frame" id (post-2-second countdown).
GAME_START_FRAME: int = -123

# peppi pre.buttons_physical bitmask per slp spec.
BUTTON_BITS: dict[str, int] = {
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

# peppi-py character id for Nana (the follower); slp-native.
NANA_CHARACTER_ID: int = 11

# Game post-frame fields we read off `port.leader.post`.
_GAMESTATE_PEPPI_FIELDS: tuple[tuple[str, str | None], ...] = (
    ("position_x", "position.x"),
    ("position_y", "position.y"),
    ("percent", "percent"),
    ("shield", "shield"),
    ("stock", "stocks"),
    ("direction", "direction"),
    ("action", "state"),
    # action_frame derived
    ("hitlag_left", "hitlag"),
    ("jumps_used", "jumps"),
    ("airborne", "airborne"),
    ("hurtbox_state", "hurtbox_state"),
)


def _mask_value(dtype: DTypeLike) -> float | int:
    """Dtype-appropriate mask sentinel.

    NP_MASK_VALUE (= INT32_MAX) doesn't fit in int8/uint8, so we pick a
    per-dtype "out of normal range" value:
      - floats: NaN
      - signed int: dtype min  (e.g. -128 for int8, INT_MIN for int32)
      - unsigned int: dtype max (e.g. 255 for uint8)
    """
    np_dtype = np.dtype(dtype)
    if np.issubdtype(np_dtype, np.floating):
        return float("nan")
    info = np.iinfo(np_dtype)
    if np_dtype.kind == "i":
        return info.min if np_dtype.itemsize < 4 else NP_MASK_VALUE
    return info.max  # unsigned


def _arr_to_np(arr: object, dtype: DTypeLike, length: int) -> np.ndarray:
    """Convert a pyarrow Array (or None) to numpy with mask substitution.

    None whole-column means peppi didn't parse this field for this slp
    version — fill with the dtype's mask sentinel. None scalars within a
    column (e.g. peppi.hitlag for one frame) get the same treatment.
    """
    mask = _mask_value(dtype)
    if arr is None:
        return np.full(length, mask, dtype=dtype)
    raw = arr.to_pylist()
    return np.array([v if v is not None else mask for v in raw], dtype=dtype)


def _resolve(obj: object, dotted: str) -> object:
    cur: object = obj
    for part in dotted.split("."):
        cur = getattr(cur, part)
        if cur is None:
            return None
    return cur


def _action_frame_from_state(state_arr: object) -> np.ndarray:
    """1-indexed run-length on post.state — matches libmelee's action_frame."""
    if state_arr is None:
        return np.array([], dtype=np.int32)
    states = state_arr.to_pylist()
    out = np.empty(len(states), dtype=np.int32)
    prev: int | None = None
    counter = 0
    for i, s in enumerate(states):
        if s != prev:
            counter = 1
            prev = s
        else:
            counter += 1
        out[i] = counter
    return out


def _unpack_buttons(physical: object, length: int) -> dict[str, np.ndarray]:
    if physical is None:
        return {b: np.zeros(length, dtype=np.int32) for b in BUTTON_BITS}
    bits = np.array([v if v is not None else 0 for v in physical.to_pylist()], dtype=np.int32)
    return {b: ((bits & mask) != 0).astype(np.int32) for b, mask in BUTTON_BITS.items()}


def _peppi_port_index(g: object) -> dict[int, int]:
    """libmelee-style port (1..4) -> peppi port array index (0..3)."""
    out: dict[int, int] = {}
    for i, pl in enumerate(g.start.players):
        port_value = int(getattr(pl.port, "value", pl.port))
        out[port_value + 1] = i
    return out


def _extract_player(leader: object, prefix: str, frame_slice: slice, length: int) -> dict[str, np.ndarray]:
    """Pull the gamestate + controller columns for one port's leader (main char)."""
    out: dict[str, np.ndarray] = {}
    pre = leader.pre
    post = leader.post

    # Gamestate from post-frame block
    for col_suffix, peppi_path in _GAMESTATE_PEPPI_FIELDS:
        arr = _resolve(post, peppi_path) if peppi_path else None
        dtype = MDS_PER_FRAME_DTYPES[f"{prefix}_{col_suffix}"]
        out[f"{prefix}_{col_suffix}"] = _arr_to_np(arr, dtype, length)[frame_slice]

    out[f"{prefix}_action_frame"] = _action_frame_from_state(post.state)[frame_slice]

    # Buttons
    for b, arr in _unpack_buttons(pre.buttons_physical, length).items():
        out[f"{prefix}_button_{b}"] = arr[frame_slice]

    # Sticks (peppi-native [-1,1])
    out[f"{prefix}_main_stick_x"] = _arr_to_np(pre.joystick.x, np.float32, length)[frame_slice]
    out[f"{prefix}_main_stick_y"] = _arr_to_np(pre.joystick.y, np.float32, length)[frame_slice]
    out[f"{prefix}_c_stick_x"] = _arr_to_np(pre.cstick.x, np.float32, length)[frame_slice]
    out[f"{prefix}_c_stick_y"] = _arr_to_np(pre.cstick.y, np.float32, length)[frame_slice]

    # Triggers
    out[f"{prefix}_trigger_logical"] = _arr_to_np(pre.triggers, np.float32, length)[frame_slice]
    tp = pre.triggers_physical
    out[f"{prefix}_trigger_l_physical"] = _arr_to_np(tp.l if tp is not None else None, np.float32, length)[frame_slice]
    out[f"{prefix}_trigger_r_physical"] = _arr_to_np(tp.r if tp is not None else None, np.float32, length)[frame_slice]

    # Raw analog bytes (slp-version gated)
    out[f"{prefix}_main_stick_raw_x"] = _arr_to_np(pre.raw_analog_x, np.int8, length)[frame_slice]
    out[f"{prefix}_main_stick_raw_y"] = _arr_to_np(pre.raw_analog_y, np.int8, length)[frame_slice]

    return out


def _extract_nana(
    follower: object | None, prefix: str, frame_slice: slice, length: int, frame_count: int
) -> dict[str, np.ndarray]:
    """Nana columns: gamestate only (no controller). Mask if no follower."""
    nana_cols: dict[str, np.ndarray] = {}
    if follower is None:
        for col, dtype in MDS_PER_FRAME_DTYPES.items():
            if not col.startswith(f"{prefix}_nana_"):
                continue
            nana_cols[col] = np.full(frame_count, _mask_value(dtype), dtype=dtype)
        return nana_cols

    post = follower.post
    for col_suffix, peppi_path in _GAMESTATE_PEPPI_FIELDS:
        arr = _resolve(post, peppi_path) if peppi_path else None
        dtype = MDS_PER_FRAME_DTYPES[f"{prefix}_nana_{col_suffix}"]
        nana_cols[f"{prefix}_nana_{col_suffix}"] = _arr_to_np(arr, dtype, length)[frame_slice]

    nana_cols[f"{prefix}_nana_action_frame"] = _action_frame_from_state(post.state)[frame_slice]
    return nana_cols


def extract_replay(replay_path: str) -> dict[str, np.ndarray] | None:
    """Parse a slp file and return per-frame ndarrays keyed by MDS column name.

    Returns None if peppi can't parse the file or the start block is missing
    expected players. Caller logs.
    """
    try:
        g = peppi_py.read_slippi(str(replay_path), skip_frames=False)
    except Exception as e:
        logger.debug(f"peppi failed for {replay_path}: {e}")
        return None

    if g.frames is None or g.frames.id is None:
        logger.debug(f"empty frames for {replay_path}")
        return None

    ids = g.frames.id.to_pylist()
    raw_length = len(ids)
    # Trim to the first in-game frame onward.
    start_idx = 0
    for i, fid in enumerate(ids):
        if fid >= GAME_START_FRAME:
            start_idx = i
            break
    frame_slice = slice(start_idx, raw_length)
    out_length = raw_length - start_idx

    if out_length <= 0:
        logger.debug(f"no in-game frames for {replay_path}")
        return None

    port_idx_by_libmelee = _peppi_port_index(g)
    # Map p1/p2 to the two lowest occupied ports (in port-ascending order).
    # Replays on ports (3, 4) — common in tournament setups — would otherwise
    # be silently dropped. We still require exactly two players (1v1).
    occupied_ports = sorted(port_idx_by_libmelee)
    if len(occupied_ports) != len(PLAYER_PREFIXES):
        logger.debug(f"{replay_path}: {len(occupied_ports)} players; expected {len(PLAYER_PREFIXES)} (1v1)")
        return None

    sample: dict[str, np.ndarray] = {
        "frame": np.array(ids[start_idx:], dtype=np.int32),
    }

    for prefix, port in zip(PLAYER_PREFIXES, occupied_ports):
        peppi_idx = port_idx_by_libmelee[port]
        port_data = g.frames.ports[peppi_idx]
        sample.update(_extract_player(port_data.leader, prefix, frame_slice, raw_length))
        sample.update(_extract_nana(port_data.follower, prefix, frame_slice, raw_length, out_length))

    # Sanity: every column has the expected length.
    bad = [(k, v.shape[0]) for k, v in sample.items() if v.shape[0] != out_length]
    if bad:
        logger.debug(f"{replay_path}: column length mismatch {bad[:3]}")
        return None

    return sample
