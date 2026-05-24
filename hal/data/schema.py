"""Per-frame MDS schema.

Defines the columns written into MDS shards (one ndarray per column, length =
replay frame count). Per-replay scalars (``slp_version``, ``stage``, etc.)
live in ``hal.data.index.ReplayIndexEntry``. Slp-native vocabulary
(button bits, mask sentinels, player prefixes) lives in ``hal.wire``.

See CLAUDE.md (Architecture) for naming, mask sentinels, and slp-version gating.
"""

import numpy as np
from numpy.typing import DTypeLike

from hal.wire import BUTTON_BITS

# Bump on any breaking change to MDS_PER_FRAME_DTYPES (column add/remove/dtype
# change) or to the extraction semantics that produce them. Consumers verify
# the version matches before reading; mismatch is a hard error.
#
# 3: drop the per-player action_frame column. It was a 1-indexed run-length on
#    the action id, but the closed-loop policy feeds the engine's state_age
#    (0-indexed, resets within a constant action) — the two never matched, so
#    the column was a train/inference skew on a model input.
# 2: add raw_analog_cstick_x/y columns (slp >= 3.17) for bit-exact c-stick
#    replay.
# 1: initial introduction of the version field.
SCHEMA_VERSION: int = 3


def _gamestate_columns(prefix: str) -> dict[str, DTypeLike]:
    """Post-frame block fields that are 1:1 mappable from peppi."""
    return {
        f"{prefix}_position_x": np.float32,
        f"{prefix}_position_y": np.float32,
        f"{prefix}_percent": np.float32,
        f"{prefix}_shield": np.float32,
        f"{prefix}_stock": np.int32,
        f"{prefix}_direction": np.float32,
        f"{prefix}_action": np.int32,
        f"{prefix}_hitlag_left": np.float32,  # peppi reports None for slp < ~3.8.0; masked
        f"{prefix}_jumps_used": np.int32,
        f"{prefix}_airborne": np.int32,
        f"{prefix}_hurtbox_state": np.int32,  # 0=vulnerable, 1=invulnerable, 2=intangible
    }


def _controller_columns(prefix: str) -> dict[str, DTypeLike]:
    """Pre-frame block fields. Action[t] -> state[t+1] alignment."""
    cols: dict[str, DTypeLike] = {f"{prefix}_button_{b}": np.int32 for b in BUTTON_BITS}
    cols.update(
        {
            f"{prefix}_main_stick_x": np.float32,
            f"{prefix}_main_stick_y": np.float32,
            f"{prefix}_c_stick_x": np.float32,
            f"{prefix}_c_stick_y": np.float32,
            # `trigger_logical` is peppi's smoothed analog value (single channel,
            # used by training as the analog-shoulder feature). `trigger_l/r_physical`
            # are the slp-native per-shoulder bytes used by the emulator wire path.
            f"{prefix}_trigger_logical": np.float32,
            f"{prefix}_trigger_l_physical": np.float32,
            f"{prefix}_trigger_r_physical": np.float32,
            f"{prefix}_main_stick_raw_x": np.int8,
            f"{prefix}_main_stick_raw_y": np.int8,
            f"{prefix}_c_stick_raw_x": np.int8,  # slp >= 3.17.0; mask sentinel otherwise
            f"{prefix}_c_stick_raw_y": np.int8,
        }
    )
    return cols


def _nana_columns(prefix: str) -> dict[str, DTypeLike]:
    """Nana follower (Ice Climbers). Filled with mask sentinel for non-IC players.
    Nana has no controller — only gamestate."""
    return {f"{prefix}_nana_{k.removeprefix(prefix + '_')}": v for k, v in _gamestate_columns(prefix).items()}


MDS_PER_FRAME_DTYPES: dict[str, DTypeLike] = {
    "frame": np.int32,
    **_gamestate_columns("p1"),
    **_controller_columns("p1"),
    **_nana_columns("p1"),
    **_gamestate_columns("p2"),
    **_controller_columns("p2"),
    **_nana_columns("p2"),
}

MDS_DTYPE_STR_BY_COLUMN: dict[str, str] = {
    name: f"ndarray:{np.dtype(dtype).name}" for name, dtype in MDS_PER_FRAME_DTYPES.items()
}
