"""Slippi stage and character identifiers at the replay decoding boundary."""

from typing import Final

import melee


def slp_stage_to_libmelee(slp_stage_id: int) -> melee.Stage:
    """slp-native stage id -> ``melee.Stage`` enum.

    Footgun: the two value spaces disagree (e.g. Fountain of Dreams is slp 2
    but ``melee.Stage.FOUNTAIN_OF_DREAMS.value`` = 8). Always go through this.
    """
    stage = melee.enums.to_internal_stage(slp_stage_id)
    if stage is melee.Stage.NO_STAGE:
        raise ValueError(f"unknown slp stage id {slp_stage_id}")
    return stage


_SLP_EXTERNAL_TO_CHARACTER: Final[dict[int, melee.Character]] = {
    0: melee.Character.CPTFALCON,
    1: melee.Character.DK,
    2: melee.Character.FOX,
    3: melee.Character.GAMEANDWATCH,
    4: melee.Character.KIRBY,
    5: melee.Character.BOWSER,
    6: melee.Character.LINK,
    7: melee.Character.LUIGI,
    8: melee.Character.MARIO,
    9: melee.Character.MARTH,
    10: melee.Character.MEWTWO,
    11: melee.Character.NESS,
    12: melee.Character.PEACH,
    13: melee.Character.PIKACHU,
    14: melee.Character.POPO,  # Ice Climbers; Nana is the follower and has no external id
    15: melee.Character.JIGGLYPUFF,
    16: melee.Character.SAMUS,
    17: melee.Character.YOSHI,
    18: melee.Character.ZELDA,
    19: melee.Character.SHEIK,
    20: melee.Character.FALCO,
    21: melee.Character.YLINK,
    22: melee.Character.DOC,
    23: melee.Character.ROY,
    24: melee.Character.PICHU,
    25: melee.Character.GANONDORF,
}


def slp_character_to_libmelee(slp_character_id: int) -> melee.Character:
    """slp external (character-select) character id -> ``melee.Character`` enum.

    The only external→internal conversion site. Applied at the two peppi reads so
    everything downstream (index, MDS, model, filter, sim) speaks the internal
    Character value.
    """
    char = _SLP_EXTERNAL_TO_CHARACTER.get(slp_character_id)
    if char is None:
        raise ValueError(f"unknown slp character id {slp_character_id}")
    return char


CHARACTERS_BY_NAME: Final[dict[str, int]] = {c.name: int(c.value) for c in _SLP_EXTERNAL_TO_CHARACTER.values()}
