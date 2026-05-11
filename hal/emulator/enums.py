"""slp-native ↔ libmelee enum bridges.

`ReplayIndexEntry.stage` and `PlayerEntry.character` are slp-native ids (the
values the .slp file actually records). libmelee's `Stage` enum disagrees with
slp-native for stage ids; `Character` agrees. Both helpers raise on unknown
ids — silent fallback would mask data corruption.
"""

import melee


def slp_stage_to_libmelee(slp_stage_id: int) -> melee.Stage:
    stage = melee.enums.to_internal_stage(slp_stage_id)
    if stage is melee.Stage.NO_STAGE:
        raise ValueError(f"unknown slp stage id {slp_stage_id}")
    return stage


def slp_character_to_libmelee(slp_character_id: int) -> melee.Character:
    return melee.Character(slp_character_id)
