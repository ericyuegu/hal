"""Resolve public netplay choices at the libmelee boundary."""

import melee

from hal.netplay_service.domain import CHARACTERS
from hal.netplay_service.domain import STAGES
from hal.netplay_service.domain import validate_character
from hal.netplay_service.domain import validate_stage

# Resolve the complete catalog on import so incompatible releases fail before
# a reservation or a later random demo reaches an unsupported enum name.
_CHARACTERS = {
    choice.value: melee.Character.YLINK if choice.value == "YOUNG_LINK" else melee.Character[choice.value]
    for choice in CHARACTERS
}
_STAGES = {choice.value: melee.Stage[choice.value] for choice in STAGES}


def character_from_choice(value: str) -> melee.Character:
    return _CHARACTERS[validate_character(value)]


def stage_from_choice(value: str) -> melee.Stage:
    return _STAGES[validate_stage(value)]
