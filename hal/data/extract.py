"""peppi-py-based per-frame extractor.

``extract_replay(path) -> dict[str, np.ndarray] | None`` returns one ndarray
per column in ``MDS_PER_FRAME_DTYPES`` ready to feed into ``MDSWriter.write``.
Returns ``None`` on any unrecoverable parse failure.

See CLAUDE.md (Architecture → Conventions) for the rules this module assumes:
no value mutation (peppi-native ranges), bitmask button unpacking, dtype-
specific mask sentinels for slp-version-unavailable fields, and the post-
countdown frame-id trim at ``wire.GAME_START_FRAME``.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import peppi_py
from loguru import logger
from numpy.typing import DTypeLike
from peppi_py.frame import Data
from peppi_py.frame import Post
from peppi_py.game import Game

from hal.data.schema import MDS_PER_FRAME_DTYPES
from hal.wire import BUTTON_BITS
from hal.wire import CHARACTERS_BY_NAME
from hal.wire import GAME_START_FRAME
from hal.wire import PLAYER_PREFIXES
from hal.wire import POST_FIELD_SUFFIXES
from hal.wire import mask_value
from hal.wire import peppi_port_to_libmelee

NANA_CHARACTER_ID: int = CHARACTERS_BY_NAME["NANA"]


def _arr_to_np(arr: Any, dtype: DTypeLike, length: int) -> np.ndarray:
    """Convert a pyarrow Array (or None) to numpy with mask substitution.

    None whole-column means peppi didn't parse this field for this slp
    version — fill with the dtype's mask sentinel. None scalars within a
    column (e.g. peppi.hitlag for one frame) get the same treatment.
    """
    mask = mask_value(dtype)
    if arr is None:
        return np.full(length, mask, dtype=dtype)
    return _list_to_np(arr.to_pylist(), dtype, length)


def _list_to_np(values: Sequence[Any] | None, dtype: DTypeLike, length: int) -> np.ndarray:
    mask = mask_value(dtype)
    if values is None:
        return np.full(length, mask, dtype=dtype)
    return np.array([v if v is not None else mask for v in values], dtype=dtype)


def _resolve(obj: Any, dotted: str) -> Any:
    cur: Any = obj
    for part in dotted.split("."):
        cur = getattr(cur, part)
        if cur is None:
            return None
    return cur


def _action_frame_from_states(actions: list[int] | None) -> np.ndarray:
    """1-indexed run-length on post.action — matches libmelee's action_frame.

    Returns an empty array when ``actions`` is None; the caller's per-column
    length sanity check then drops the replay.
    """
    if actions is None:
        return np.array([], dtype=np.int32)
    out = np.empty(len(actions), dtype=np.int32)
    prev: int | None = None
    counter = 0
    for i, a in enumerate(actions):
        if a != prev:
            counter = 1
            prev = a
        else:
            counter += 1
        out[i] = counter
    return out


def _gamestate_arrays(post: Post, key_prefix: str, frame_slice: slice, length: int) -> dict[str, np.ndarray]:
    """Pull every gamestate column for one post block under ``key_prefix``.

    Shared by leader (``p1``/``p2``) and follower (``p1_nana``/``p2_nana``).
    ``post.action`` is materialized once and reused for the ``action`` column
    and ``action_frame`` derivation.
    """
    out: dict[str, np.ndarray] = {}
    for suffix in POST_FIELD_SUFFIXES:
        if suffix == "action":
            continue  # paired with action_frame below
        col = f"{key_prefix}_{suffix}"
        dtype = MDS_PER_FRAME_DTYPES[col]
        if suffix in ("position_x", "position_y"):
            # peppi nests these as post.position.x / post.position.y
            value = _resolve(post, f"position.{suffix[-1]}")
        else:
            value = getattr(post, suffix)
        out[col] = _arr_to_np(value, dtype, length)[frame_slice]

    action_arr = post.action
    actions: list[int] | None = action_arr.to_pylist() if action_arr is not None else None
    action_col = f"{key_prefix}_action"
    out[action_col] = _list_to_np(actions, MDS_PER_FRAME_DTYPES[action_col], length)[frame_slice]
    out[f"{key_prefix}_action_frame"] = _action_frame_from_states(actions)[frame_slice]
    return out


def _unpack_buttons(physical: Any, length: int) -> dict[str, np.ndarray]:
    if physical is None:
        return {b: np.zeros(length, dtype=np.int32) for b in BUTTON_BITS}
    bits = np.array([v if v is not None else 0 for v in physical.to_pylist()], dtype=np.int32)
    return {b: ((bits & mask) != 0).astype(np.int32) for b, mask in BUTTON_BITS.items()}


def _peppi_idx_by_libmelee_port(g: Game) -> dict[int, int]:
    """Map libmelee port (1..4) to peppi's port-array index (0..3)."""
    return {peppi_port_to_libmelee(pl.port): i for i, pl in enumerate(g.start.players)}


def _extract_player(leader: Data, prefix: str, frame_slice: slice, length: int) -> dict[str, np.ndarray]:
    """Pull the gamestate + controller columns for one port's leader (main char)."""
    pre = leader.pre
    out = _gamestate_arrays(leader.post, prefix, frame_slice, length)

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

    # Raw analog bytes (slp-version gated). Main stick: x >= 1.2.0, y >= 3.15.0.
    # C-stick: both axes >= 3.17.0. Older slps backfill with the int8 mask
    # sentinel so apply_inputs falls back to the lossy logical-to-wire path.
    out[f"{prefix}_main_stick_raw_x"] = _arr_to_np(pre.raw_analog_x, np.int8, length)[frame_slice]
    out[f"{prefix}_main_stick_raw_y"] = _arr_to_np(pre.raw_analog_y, np.int8, length)[frame_slice]
    out[f"{prefix}_c_stick_raw_x"] = _arr_to_np(pre.raw_analog_cstick_x, np.int8, length)[frame_slice]
    out[f"{prefix}_c_stick_raw_y"] = _arr_to_np(pre.raw_analog_cstick_y, np.int8, length)[frame_slice]

    return out


def _extract_nana(follower: Data | None, prefix: str, frame_slice: slice, length: int) -> dict[str, np.ndarray]:
    """Nana columns: gamestate only (no controller). Mask if no follower."""
    nana_prefix = f"{prefix}_nana"
    if follower is None:
        out_length = frame_slice.stop - frame_slice.start
        return {
            col: np.full(out_length, mask_value(dtype), dtype=dtype)
            for col, dtype in MDS_PER_FRAME_DTYPES.items()
            if col.startswith(f"{nana_prefix}_")
        }
    return _gamestate_arrays(follower.post, nana_prefix, frame_slice, length)


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
    start_idx = next((i for i, fid in enumerate(ids) if fid >= GAME_START_FRAME), raw_length)
    frame_slice = slice(start_idx, raw_length)
    out_length = raw_length - start_idx

    if out_length <= 0:
        logger.debug(f"no in-game frames for {replay_path}")
        return None

    peppi_idx_by_libmelee_port = _peppi_idx_by_libmelee_port(g)
    # Map p1/p2 to the two lowest occupied libmelee ports (in ascending order).
    # Replays on ports (3, 4) — common in tournament setups — would otherwise
    # be silently dropped. We still require exactly two players (1v1).
    occupied_libmelee_ports = sorted(peppi_idx_by_libmelee_port)
    if len(occupied_libmelee_ports) != len(PLAYER_PREFIXES):
        logger.debug(f"{replay_path}: {len(occupied_libmelee_ports)} players; expected {len(PLAYER_PREFIXES)} (1v1)")
        return None

    sample: dict[str, np.ndarray] = {
        "frame": np.array(ids[start_idx:], dtype=np.int32),
    }

    for prefix, port in zip(PLAYER_PREFIXES, occupied_libmelee_ports, strict=True):
        peppi_idx = peppi_idx_by_libmelee_port[port]
        port_data = g.frames.ports[peppi_idx]
        sample.update(_extract_player(port_data.leader, prefix, frame_slice, raw_length))
        sample.update(_extract_nana(port_data.follower, prefix, frame_slice, raw_length))

    # Sanity: every column has the expected length.
    bad = [(k, v.shape[0]) for k, v in sample.items() if v.shape[0] != out_length]
    if bad:
        logger.debug(f"{replay_path}: column length mismatch {bad[:3]}")
        return None

    return sample
