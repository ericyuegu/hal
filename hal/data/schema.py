"""Per-frame MDS schema.

Defines the columns written into MDS shards (one ndarray per column, length =
replay frame count, plus the scalar ``schema_version``). Per-replay scalars
(``slp_version``, ``stage``, etc.) live in ``hal.data.index.ReplayIndexEntry``.
Slp-native vocabulary (button bits, mask sentinels, player prefixes) lives in
``hal.wire``.

See CLAUDE.md (Controller data model) for the logical-only
controller representation, mask sentinels, and naming.
"""

from enum import IntEnum

import numpy as np
from numpy.typing import DTypeLike

from hal.wire import BUTTON_BITS
from hal.wire import ITEM_SLOTS
from hal.wire import N_STATE_FLAG_BYTES
from hal.wire import POST_FIELD_SUFFIXES
from hal.wire import VELOCITY_COMPONENTS
from hal.wire import item_column

# Bump on any breaking change to MDS_COLUMNS (column add/remove/dtype change)
# or to the extraction semantics that produce them. Consumers verify the
# version matches before reading; mismatch is a hard error.
#
# 7: add the per-player ``p{1,2}_rank`` columns — the ranked-ladder tier of each
#    player, a per-replay constant broadcast across frames like ``stage`` and
#    ``p{1,2}_character``. The value is the ``Rank`` IntEnum, read from the slp
#    start block's netplay name ("Platinum Player" / "Diamond Player" /
#    "Master Player"). Any other name is ``Rank.UNKNOWN`` (0), so a non-ranked
#    corpus reads loud instead of silently claiming a tier.
# 6: widen the per-frame post block and add global item (projectile) slots.
#    (a) per-player: the ``velocities`` block (5 channels), ``misc_as``,
#    ``state_age``, the 5 raw ``state_flags`` bytes, ``l_cancel``, ``ground``,
#    and ``character_live``. All are ``wire.POST_FIELD_SUFFIXES`` entries, so
#    the nana follower block picks them up too.
#    (b) global: ``item{0..3}_*`` — the K=4 lowest-spawn-id live items, with
#    ``wire.ITEM_FIELD_SUFFIXES`` fields each.
#    ``character_live`` is the engine's live, transform-aware character, so a
#    Sheik<->Zelda transform is now visible per frame; the broadcast
#    ``p{1,2}_character`` character-SELECT pick is unchanged and still stale
#    across transforms. ``state_age`` is the engine-truth replacement for the
#    ``action_frame`` column dropped at v3: v3 removed a RECONSTRUCTED
#    run-length that was off-by-one against the engine, this stores the
#    engine's own counter, which is exactly what the closed-loop policy reads.
# 5: ``p{1,2}_character`` is normalized to the libmelee ``Character`` value
#    (Fox=1) at the peppi read, matching how ``stage`` is already stored; it
#    was previously the slp external/character-select id (Fox=2). The index
#    ``ReplayIndexEntry.players[*].character`` is normalized the same way.
# 4: (a) re-add the global ``stage`` + per-player ``p{1,2}_character`` columns
#    (dropped at v3's predecessor) as per-replay constants broadcast across
#    frames, so the policy can condition on matchup again. ``stage`` is stored
#    as the libmelee ``Stage`` value (not slp-native) so it matches the
#    closed-loop obs without a second translation. (``character`` was stored as
#    the slp external id here; normalized to the libmelee value at v5.)
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
SCHEMA_VERSION: int = 7


class Rank(IntEnum):
    """Ranked-ladder tier of one player, as stored in ``p{1,2}_rank``.

    ``UNKNOWN`` is 0 so an unread or non-ranked tier is loud (a consumer that
    weights by rank raises on it) instead of quietly reading as the lowest tier.
    """

    UNKNOWN = 0
    PLATINUM = 1
    DIAMOND = 2
    MASTER = 3


# The three netplay display names the ranked-anonymized corpus carries, one per
# tier. Anonymization replaces the player's real display name with these exact
# strings, so the match is exact: a real netplay name that merely CONTAINS
# "master" is a name, not a tier.
_RANK_BY_PLAYER_NAME: dict[str, Rank] = {
    "Platinum Player": Rank.PLATINUM,
    "Diamond Player": Rank.DIAMOND,
    "Master Player": Rank.MASTER,
}


def rank_from_player_name(name: str | None) -> Rank:
    """Netplay display name -> ``Rank``. Anything unrecognized is ``UNKNOWN``."""
    if name is None:
        return Rank.UNKNOWN
    return _RANK_BY_PLAYER_NAME.get(name, Rank.UNKNOWN)


# Storage dtype per ``wire.POST_FIELD_SUFFIXES`` entry. Every suffix must appear
# here; ``_gamestate_columns`` iterates the wire tuple and raises at import on a
# suffix with no declared dtype. Fields peppi reports as None for older slp
# versions get the dtype's mask sentinel at extract, not a default value.
_POST_FIELD_DTYPES: dict[str, DTypeLike] = {
    "position_x": np.float32,
    "position_y": np.float32,
    "percent": np.float32,
    "shield": np.float32,
    "stock": np.int32,
    "direction": np.float32,
    "action": np.int32,
    "hitlag_left": np.float32,  # peppi reports None for slp < ~3.8.0; masked
    "jumps_used": np.int32,
    "airborne": np.int32,
    "hurtbox_state": np.int32,  # 0=vulnerable, 1=invulnerable, 2=intangible
    # Live, transform-aware character id (libmelee INTERNAL space, as the engine
    # reports it). Distinct from the broadcast ``p{1,2}_character`` select-pick.
    "character_live": np.int32,
    # Frames the character has spent in the current ``action``. The engine's own
    # counter, 0-indexed and reset within a constant action id — this is what the
    # closed-loop policy already reads, and the reason the reconstructed
    # ``action_frame`` column was dropped at v3.
    "state_age": np.float32,
    # Multiplexed by action state: during hitstun action states it is the number
    # of hitstun frames remaining; in other states the engine reuses the slot for
    # unrelated per-state counters (e.g. charge/entry timers), sometimes as a raw
    # int bit pattern rather than a meaningful float. Stored as the raw f32 so
    # every interpretation is recoverable; consumers must gate on ``action``.
    "misc_as": np.float32,
    # 0=not applicable, 1=successful L-cancel, 2=unsuccessful.
    "l_cancel": np.int32,
    # Ground/platform id the character is standing on (meaningless while airborne).
    "ground": np.int32,
    **{f"state_flags_{i}": np.int32 for i in range(N_STATE_FLAG_BYTES)},
    **{f"velocities_{c}": np.float32 for c in VELOCITY_COMPONENTS},
}

# Per-item-slot storage dtypes. peppi's native widths (u16 type, u8 state, i8
# owner) are widened to int32 so every categorical column in the schema shares
# one mask sentinel (``wire.MASK_INT32``).
_ITEM_FIELD_DTYPES: dict[str, DTypeLike] = {
    "type": np.int32,
    "state": np.int32,
    "pos_x": np.float32,
    "pos_y": np.float32,
    "vel_x": np.float32,
    "vel_y": np.float32,
    "owner": np.int32,  # libmelee port 1..4, or wire.ITEM_OWNER_NONE when unowned
}


def _gamestate_columns(prefix: str) -> dict[str, DTypeLike]:
    """Post-frame block fields, one per ``wire.POST_FIELD_SUFFIXES`` entry."""
    return {f"{prefix}_{suffix}": _POST_FIELD_DTYPES[suffix] for suffix in POST_FIELD_SUFFIXES}


def _item_columns() -> dict[str, DTypeLike]:
    """Global item (projectile) slots. Slot k holds the k-th lowest live spawn
    id this frame; unfilled slots are masked. See ``wire.ITEM_SPAWN_ID_FIELD``
    for the ordering contract the online reader shares."""
    return {
        item_column(slot, suffix): dtype for slot in range(ITEM_SLOTS) for suffix, dtype in _ITEM_FIELD_DTYPES.items()
    }


def _controller_columns(prefix: str) -> dict[str, DTypeLike]:
    """Pre-frame block fields. Same-row alignment: row t's controller is the input active
    during frame t and its consequence lands in row t's post-state (a Y press and its
    KNEE_BEND jumpsquat share a row) — identical to the closed-loop ``(post_i, pre_i)``
    pairing in ``hal/training/closed_loop.py``. Verified in ``notebooks/alignment_probe.py``."""
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


# ``stage``, ``p{1,2}_character`` and ``p{1,2}_rank`` are per-replay constants broadcast
# across frames (not in peppi's per-frame post block) — see extract.broadcast and
# SCHEMA_VERSION 5/7. ``stage`` and ``p{1,2}_character`` are libmelee enum values (via
# slp_stage_to_libmelee / slp_character_to_libmelee); ``p{1,2}_rank`` is the ``Rank``
# enum. ``item{k}_*`` is global per-frame state, hence no player prefix.
MDS_PER_FRAME_DTYPES: dict[str, DTypeLike] = {
    "frame": np.int32,
    "stage": np.int32,
    **_item_columns(),
    "p1_character": np.int32,
    "p1_rank": np.uint8,
    **_gamestate_columns("p1"),
    **_controller_columns("p1"),
    **_nana_columns("p1"),
    "p2_character": np.int32,
    "p2_rank": np.uint8,
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


def check_schema_version(sample: dict, *, expected: int = SCHEMA_VERSION) -> None:
    """Assert one MDS row was materialized at the schema version the consumer targets.

    Call on the first row read from any split. Rows written before the scalar
    existed (< v4) have no ``schema_version`` key at all. ``expected`` defaults
    to this code's ``SCHEMA_VERSION``; a consumer that deliberately reads an
    older materialization must declare that version explicitly (visible in its
    config) — a mismatch always raises.
    """
    found = sample.get("schema_version")
    if found != expected:
        raise ValueError(
            f"MDS row schema_version={found!r} != expected={expected} (code SCHEMA_VERSION={SCHEMA_VERSION}). "
            "Re-materialize the dataset (and wipe stale local split caches), or declare the version this "
            "consumer targets."
        )
