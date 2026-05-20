"""Pin the slp-id ↔ libmelee-enum bridges in ``hal.wire``.

Two facts that ARCHITECTURE used to cite from a notebook now live here:

- Character ids identity-map between slp and libmelee today. If a future
  libmelee update reorders the enum, this test fails loudly instead of
  silently miscasting characters at the controller-injection boundary.
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


def test_character_ids_identity_map_today() -> None:
    """Every standard-cast slp character id round-trips to the same libmelee enum value."""
    for name, slp_id in wire.CHARACTERS_BY_NAME.items():
        libmelee_char = wire.slp_character_to_libmelee(slp_id)
        assert libmelee_char.value == slp_id, (
            f"{name}: slp id {slp_id} → libmelee {libmelee_char!r} with value "
            f"{libmelee_char.value}. The two id spaces have diverged; update the bridge."
        )


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
