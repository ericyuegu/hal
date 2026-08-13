"""Pin the slp-id ↔ libmelee-enum bridges in ``hal.wire``.

Two facts that ARCHITECTURE used to cite from a notebook now live here:

- Character ids do NOT identity-map. slp start-block ids are EXTERNAL/CSS ids
  (Fox=2, Falco=20); libmelee's ``Character`` enum is internal (Fox=1,
  Falco=22). Reading one as the other silently miscasts every character. The
  index/MDS store the internal value; the single external→internal conversion
  is ``wire.slp_character_to_libmelee`` at the peppi read. Anchors below are
  verified against post-frame internal ids in real replays.
- Stage ids do NOT identity-map (slp 2 = Fountain of Dreams; libmelee
  ``Stage.FOUNTAIN_OF_DREAMS.value`` = 8). All stage conversion must go
  through ``wire.slp_stage_to_libmelee``.
"""

import melee
import pytest

from hal import wire

# Tournament-legal slp-native stage ids. The slp ↔ libmelee id spaces disagree
# (e.g. slp 2 = Fountain of Dreams, libmelee.Stage.FOUNTAIN_OF_DREAMS.value=8);
# this table is the witness set.
_LEGAL_STAGES_BY_NAME: dict[str, int] = {
    "FOUNTAIN_OF_DREAMS": 2,
    "POKEMON_STADIUM": 3,
    "YOSHIS_STORY": 8,
    "DREAMLAND": 28,
    "BATTLEFIELD": 31,
    "FINAL_DESTINATION": 32,
}


# slp EXTERNAL id -> libmelee internal Character. Anchors empirically verified
# against post-frame internal character ids across 399 mang0 replays.
_EXTERNAL_TO_CHARACTER_ANCHORS: dict[int, melee.Character] = {
    0: melee.Character.CPTFALCON,
    1: melee.Character.DK,
    2: melee.Character.FOX,
    8: melee.Character.MARIO,
    9: melee.Character.MARTH,
    14: melee.Character.POPO,  # Ice Climbers
    15: melee.Character.JIGGLYPUFF,
    19: melee.Character.SHEIK,
    20: melee.Character.FALCO,
    25: melee.Character.GANONDORF,
}


def test_slp_external_character_maps_to_internal() -> None:
    """slp start-block ids are external/CSS ids; the bridge must translate them
    to libmelee's internal Character enum (NOT reinterpret the integer)."""
    for slp_id, char in _EXTERNAL_TO_CHARACTER_ANCHORS.items():
        assert wire.slp_character_to_libmelee(slp_id) is char


def test_external_fox_is_not_read_as_internal() -> None:
    """Regression witness: the old bridge did ``melee.Character(slp_id)``, reading
    external Fox (2) as internal CPTFALCON. The two must not be conflated."""
    assert wire.slp_character_to_libmelee(2) is melee.Character.FOX
    assert wire.slp_character_to_libmelee(2) is not melee.Character.CPTFALCON


def test_external_to_character_map_is_injective() -> None:
    """No two external ids may map to the same Character — otherwise the single
    external→internal conversion at the peppi read would be ambiguous."""
    chars = list(wire._SLP_EXTERNAL_TO_CHARACTER.values())
    assert len(chars) == len(set(chars))


def test_characters_by_name_are_internal_ids() -> None:
    """filter.py resolves ``--characters`` via CHARACTERS_BY_NAME against the
    stored (now internal/libmelee) ids, so the table must live in internal space."""
    assert wire.CHARACTERS_BY_NAME["FOX"] == 1
    assert wire.CHARACTERS_BY_NAME["FALCO"] == 22
    assert wire.CHARACTERS_BY_NAME["MARTH"] == 18
    assert wire.CHARACTERS_BY_NAME["CPTFALCON"] == 2


def test_slp_character_to_libmelee_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown slp character id"):
        wire.slp_character_to_libmelee(99)


def test_character_wire_bridge_round_trips_every_selectable_character() -> None:
    for external in range(26):
        character = wire.slp_character_to_libmelee(external)
        assert wire.libmelee_character_to_slp(character) == external


def test_stage_ids_do_not_identity_map() -> None:
    """Fountain of Dreams is the canonical witness that slp and libmelee stage ids disagree."""
    fod_slp_id = _LEGAL_STAGES_BY_NAME["FOUNTAIN_OF_DREAMS"]
    fod_libmelee = wire.slp_stage_to_libmelee(fod_slp_id)
    assert fod_libmelee is melee.Stage.FOUNTAIN_OF_DREAMS
    assert fod_libmelee.value != fod_slp_id, (
        "slp and libmelee stage id spaces have collapsed; the footgun in "
        "wire.slp_stage_to_libmelee no longer exists and the docs should be updated."
    )


def test_legal_stages_all_resolve() -> None:
    """Every tournament-legal slp stage id has a libmelee enum on the other side."""
    for name, slp_id in _LEGAL_STAGES_BY_NAME.items():
        libmelee_stage = wire.slp_stage_to_libmelee(slp_id)
        assert libmelee_stage is not melee.Stage.NO_STAGE, name


def test_unknown_stage_raises() -> None:
    with pytest.raises(ValueError, match="unknown slp stage id"):
        wire.slp_stage_to_libmelee(9999)


# --- canonical_post_field: shared libmelee-post-dict → POST_FIELD_SUFFIXES value ---


def _canonical_post(**overrides: object) -> dict:
    post = {
        "character": 22,
        "position": {"x": 1.5, "y": -2.5},
        "percent": 12.0,
        "shield": 60.0,
        "stock": 4,
        "direction": 1.0,
        "action": 14,
        "jumps_used": 0,
        "airborne": 1,
        "hurtbox_state": 0,
        "hitlag_left": 0.0,
        "state_age": 8.0,
        "state_flags": (0, 1, 2, 3, 4),
        "misc_as": 5.0,
        "l_cancel": 1,
        "ground": 7,
        "velocities": {
            "self_x_air": 1.0,
            "self_x_ground": 2.0,
            "self_y": 3.0,
            "knockback_x": 4.0,
            "knockback_y": 5.0,
        },
    }
    post.update(overrides)
    return post


def test_canonical_post_field_nests_position() -> None:
    post = _canonical_post()
    assert wire.canonical_post_field(post, "position_x") == 1.5
    assert wire.canonical_post_field(post, "position_y") == -2.5


def test_canonical_post_field_nests_velocities() -> None:
    """The velocities block is nested exactly like position — one suffix has to
    address peppi's ``post.velocities.self_y`` and libmelee's dict alike."""
    post = _canonical_post()
    for component, expected in (("self_x_air", 1.0), ("self_x_ground", 2.0), ("self_y", 3.0)):
        assert wire.canonical_post_field(post, f"velocities_{component}") == expected


def test_canonical_post_field_indexes_state_flags() -> None:
    """state_flags is a 5-byte tuple; each byte is its own suffix/column, stored raw."""
    post = _canonical_post()
    read = [wire.canonical_post_field(post, f"state_flags_{i}") for i in range(wire.N_STATE_FLAG_BYTES)]
    assert read == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_canonical_post_field_character_live_reads_post_character() -> None:
    """``character_live`` is the post block's own (live, transform-aware) character.
    It is renamed only because the MDS spends ``p{1,2}_character`` on the per-replay
    character-SELECT pick, which the post block does not carry."""
    assert wire.canonical_post_field(_canonical_post(character=19), "character_live") == 19.0
    assert wire.post_field_path("character_live") == ("character",)


def test_canonical_post_field_absent_nested_block_masks_every_component() -> None:
    """slp < 3.5 has no velocities block at all; every component must mask rather
    than raise or read a neighbouring field."""
    import numpy as np

    post = _canonical_post()
    del post["velocities"]
    assert all(np.isnan(wire.canonical_post_field(post, f"velocities_{c}")) for c in wire.VELOCITY_COMPONENTS)


def test_canonical_post_field_absent_optional_is_nan() -> None:
    import numpy as np

    post = _canonical_post()
    del post["jumps_used"]
    assert np.isnan(wire.canonical_post_field(post, "jumps_used"))


def test_canonical_post_field_preserves_genuine_zero() -> None:
    assert wire.canonical_post_field(_canonical_post(jumps_used=0), "jumps_used") == 0.0


def test_canonical_post_field_covers_every_post_suffix() -> None:
    """Every POST_FIELD_SUFFIXES entry is resolvable from a full canonical post —
    so from_capture and flatten_canonical_frame can both just loop the tuple."""
    import numpy as np

    post = _canonical_post()
    for suffix in wire.POST_FIELD_SUFFIXES:
        value = wire.canonical_post_field(post, suffix)
        assert not np.isnan(value), f"{suffix} masked on a fully-populated post block"


# --- items: shared spawn-id slot ordering between the offline and online readers ---


def _canonical_item(spawn_id: int, **overrides: object) -> dict:
    item = {
        "type": 55,
        "state": 0,
        "direction": -1.0,
        "velocity": {"x": -5.0, "y": 0.0},
        "position": {"x": 10.0, "y": 20.0},
        "damage": 0,
        "timer": 83.0,
        "id": spawn_id,
        "misc": (0, 0, 0, 0),
        "owner": 0,
        "instance_id": None,
    }
    item.update(overrides)
    return item


def test_canonical_item_columns_orders_slots_by_ascending_spawn_id() -> None:
    """Slot order is the spawn-id order, NOT the order libmelee happened to
    append the items — that's what makes the columns comparable frame to frame
    and identical to what the offline extractor writes."""
    items = [_canonical_item(9, type=1), _canonical_item(4, type=2), _canonical_item(7, type=3)]
    cols = wire.canonical_item_columns(items)
    assert [cols[wire.item_column(k, "type")] for k in range(3)] == [2.0, 3.0, 1.0]


def test_canonical_item_columns_masks_unfilled_slots() -> None:
    import numpy as np

    cols = wire.canonical_item_columns([_canonical_item(1)])
    assert not np.isnan(cols[wire.item_column(0, "pos_x")])
    for slot in range(1, wire.ITEM_SLOTS):
        assert all(np.isnan(cols[wire.item_column(slot, s)]) for s in wire.ITEM_FIELD_SUFFIXES)


def test_canonical_item_columns_no_items_is_all_masked() -> None:
    """``items is None`` (slp/build predates item events) and an empty list both
    emit the full column set, fully masked — the online obs shape never varies."""
    import numpy as np

    for items in (None, []):
        cols = wire.canonical_item_columns(items)
        assert set(cols) == {wire.item_column(k, s) for k in range(wire.ITEM_SLOTS) for s in wire.ITEM_FIELD_SUFFIXES}
        assert all(np.isnan(v) for v in cols.values())


def test_canonical_item_columns_overflow_drops_newest() -> None:
    """More live items than slots: the oldest ITEM_SLOTS survive, so an item's
    slot only moves when an OLDER item despawns."""
    items = [_canonical_item(spawn_id, type=spawn_id) for spawn_id in range(wire.ITEM_SLOTS + 3)]
    cols = wire.canonical_item_columns(items)
    assert [cols[wire.item_column(k, "type")] for k in range(wire.ITEM_SLOTS)] == [
        float(k) for k in range(wire.ITEM_SLOTS)
    ]


def test_canonical_item_owner_is_normalized_to_libmelee_port() -> None:
    """slp records the owner as a peppi 0..3 port; the stored value is the
    libmelee 1..4 port every other port in the MDS/index uses."""
    cols = wire.canonical_item_columns([_canonical_item(1, owner=0)])
    assert cols[wire.item_column(0, "owner")] == 1.0
    assert wire.item_owner_to_libmelee_port(3) == 4


def test_unowned_item_owner_passes_through() -> None:
    cols = wire.canonical_item_columns([_canonical_item(1, owner=wire.ITEM_OWNER_NONE)])
    assert cols[wire.item_column(0, "owner")] == float(wire.ITEM_OWNER_NONE)


def test_item_owner_rejects_out_of_range_port() -> None:
    with pytest.raises(ValueError, match="item owner"):
        wire.item_owner_to_libmelee_port(4)
