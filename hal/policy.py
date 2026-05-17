"""Project policy: which characters, stages, ports, and player labels
participate in HAL training. Pure project decisions.

Cross-layer slp/wire vocabulary (button bits, mask sentinels, stage/character
bridges, post-frame field naming) lives in ``hal.wire``.
"""

from typing import Final
from typing import Literal

from melee import Character
from melee import Stage

Player = Literal["p1", "p2"]
PLAYER_1_PORT: Final[int] = 1
PLAYER_2_PORT: Final[int] = 2


def get_opponent(player: Player) -> Player:
    return "p2" if player == "p1" else "p1"


INCLUDED_STAGES: Final[tuple[Stage, ...]] = (
    Stage.FINAL_DESTINATION,
    Stage.BATTLEFIELD,
    Stage.POKEMON_STADIUM,
    Stage.DREAMLAND,
    Stage.FOUNTAIN_OF_DREAMS,
    Stage.YOSHIS_STORY,
)

INCLUDED_CHARACTERS: Final[tuple[Character, ...]] = (
    Character.MARIO,
    Character.FOX,
    Character.CPTFALCON,
    Character.DK,
    Character.KIRBY,
    Character.BOWSER,
    Character.LINK,
    Character.SHEIK,
    Character.NESS,
    Character.PEACH,
    Character.POPO,
    Character.NANA,
    Character.PIKACHU,
    Character.SAMUS,
    Character.YOSHI,
    Character.JIGGLYPUFF,
    Character.MEWTWO,
    Character.LUIGI,
    Character.MARTH,
    Character.ZELDA,
    Character.YLINK,
    Character.DOC,
    Character.FALCO,
    Character.PICHU,
    Character.GAMEANDWATCH,
    Character.GANONDORF,
    Character.ROY,
)
