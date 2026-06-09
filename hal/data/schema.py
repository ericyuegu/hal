"""Per-frame MDS schema.

Defines the columns written into MDS shards (one ndarray per column, length =
replay frame count, plus the scalar ``schema_version``). Per-replay scalars
(``slp_version``, ``stage``, etc.) live in ``hal.data.index.ReplayIndexEntry``.
Slp-native vocabulary (button bits, mask sentinels, player prefixes) lives in
``hal.wire``.

See CLAUDE.md (Controller data model) for the logical-only
controller representation, mask sentinels, and naming.
"""

import numpy as np
from numpy.typing import DTypeLike

from hal.wire import BUTTON_BITS

# Bump on any breaking change to MDS_COLUMNS (column add/remove/dtype change)
# or to the extraction semantics that produce them. Consumers verify the
# version matches before reading; mismatch is a hard error.
#
# 4: (a) re-add the global ``stage`` + per-player ``p{1,2}_character`` columns
#    (dropped at v3's predecessor) as per-replay constants broadcast across
#    frames, so the policy can condition on matchup again. ``stage`` is stored
#    as the libmelee ``Stage`` value (not slp-native) so it matches the
#    closed-loop obs without a second translation; ``character`` slp-native id
#    already equals the libmelee ``Character`` value.
#    (b) logical-only controller block: drop the raw stick byte columns and the
#    fused ``trigger_logical``; rename ``trigger_{l,r}_physical`` →
#    ``trigger_{l,r}`` with sub-deadzone values zeroed (wire.TRIGGER_DEADZONE).
#    The pre-frame block now is the model action space and round-trips
#    game-exactly through ``apply_inputs``.
#    (c) add the per-row ``schema_version`` scalar so the bare streaming read
#    path fails loud on stale caches instead of a cryptic frombuffer error.
# 3: drop the per-player action_frame column. It was a 1-indexed run-length on
#    the action id, but the closed-loop policy feeds the engine's state_age
#    (0-indexed, resets within a constant action) — the two never matched, so
#    the column was a train/inference skew on a model input.
# 2: add raw_analog_cstick_x/y columns (slp >= 3.17) for bit-exact c-stick
#    replay.
# 1: initial introduction of the version field.
SCHEMA_VERSION: int = 4


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
            # Sticks are slp-logical (post-deadzone, [-1, 1] on the 1/80 grid);
            # triggers are per-shoulder ([0, 1] on the 1/140 grid, zeroed below
            # wire.TRIGGER_DEADZONE). Game-causal values only — this block is
            # the model action space and what apply_inputs feeds back.
            f"{prefix}_main_stick_x": np.float32,
            f"{prefix}_main_stick_y": np.float32,
            f"{prefix}_c_stick_x": np.float32,
            f"{prefix}_c_stick_y": np.float32,
            f"{prefix}_trigger_l": np.float32,
            f"{prefix}_trigger_r": np.float32,
        }
    )
    return cols


def _nana_columns(prefix: str) -> dict[str, DTypeLike]:
    """Nana follower (Ice Climbers). Filled with mask sentinel for non-IC players.
    Nana has no controller — only gamestate."""
    return {f"{prefix}_nana_{k.removeprefix(prefix + '_')}": v for k, v in _gamestate_columns(prefix).items()}


# ``stage`` + ``p{1,2}_character`` are per-replay constants broadcast across frames
# (not in peppi's per-frame post block) — see extract.broadcast and SCHEMA_VERSION 4.
MDS_PER_FRAME_DTYPES: dict[str, DTypeLike] = {
    "frame": np.int32,
    "stage": np.int32,
    "p1_character": np.int32,
    **_gamestate_columns("p1"),
    **_controller_columns("p1"),
    **_nana_columns("p1"),
    "p2_character": np.int32,
    **_gamestate_columns("p2"),
    **_controller_columns("p2"),
    **_nana_columns("p2"),
}

MDS_DTYPE_STR_BY_COLUMN: dict[str, str] = {
    name: f"ndarray:{np.dtype(dtype).name}" for name, dtype in MDS_PER_FRAME_DTYPES.items()
}

# Full writer spec: the per-frame ndarrays plus a scalar row version, so the
# bare StreamingDataset read path (no manifest in sight) can fail loud on a
# version mismatch instead of crashing in frombuffer on a stale cache.
MDS_COLUMNS: dict[str, str] = {"schema_version": "int", **MDS_DTYPE_STR_BY_COLUMN}


def check_schema_version(sample: dict) -> None:
    """Assert one MDS row was materialized at this code's ``SCHEMA_VERSION``.

    Call on the first row read from any split. Rows written before the scalar
    existed (< v4) have no ``schema_version`` key at all.
    """
    found = sample.get("schema_version")
    if found != SCHEMA_VERSION:
        raise ValueError(
            f"MDS row schema_version={found!r} != SCHEMA_VERSION={SCHEMA_VERSION}. "
            "Re-materialize the dataset (and wipe stale local split caches)."
        )
