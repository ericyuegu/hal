"""Per-frame MDS schema.

Per-replay scalars (slp_version, stage, character, etc.) live in
`hal.data.manifest.ReplayIndexEntry`, NOT here. This module defines only the
columns written into the MDS shards — one ndarray per column, length =
replay frame count.

Conventions
-----------

- **Sticks (`*_stick_x/y`)**: peppi-native floats in `[-1, 1]` with neutral=0.
  This is *different* from the old libmelee-style `[0, 1]` with neutral=0.5.
  The data layer never mutates values, so feature encoders that previously
  assumed `[0, 1]` need to update.
- **Triggers (`trigger_*`)**: floats in `[0, 1]` (peppi and libmelee agree).
- **Direction (`direction`)**: float, -1.0 (left) or 1.0 (right) — peppi's
  signed scalar; supersedes the old boolean `facing`.
- **Buttons (`button_*`)**: 0/1 ints, decoded from peppi's
  `pre.buttons_physical` bitmask.
- **Raw analog bytes (`*_raw_*`)**: int8 / uint8 from the slp pre-frame block,
  for closed-loop bit-exact controller reproduction. Gated by slp version
  (raw_x ≥ 1.2.0; raw_y ≥ 3.15.0); when unavailable the column is filled
  with `NP_MASK_VALUE` and consumers must short-circuit via the manifest's
  `slp_version`.

Controller alignment: row `t` stores `(state[t], action[t])` where `action[t]`
is the input recorded as producing frame `t+1`. With peppi this is the
`pre_*` columns at frame `t` — no manual shift needed.

Ego/opponent: never mutated here. Sampler picks `ego_port ∈ {1,2}` and feature
encoders look up `p{ego_port}_*` / `p{3-ego_port}_*` directly.
"""

import numpy as np
from numpy.typing import DTypeLike

PLAYER_PREFIXES: tuple[str, ...] = ("p1", "p2")

# peppi pre.buttons_physical bitmask per slp spec. Single source of truth for
# the set of buttons exposed as MDS columns and the bitmask used to decode them.
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
        f"{prefix}_action_frame": np.int32,
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
            f"{prefix}_trigger_logical": np.float32,
            f"{prefix}_trigger_l_physical": np.float32,
            f"{prefix}_trigger_r_physical": np.float32,
            f"{prefix}_main_stick_raw_x": np.int8,
            f"{prefix}_main_stick_raw_y": np.int8,
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
