from enum import Enum
from typing import Literal, Final, Tuple

# Gamestate
STAGES: Final[Tuple[str, ...]] = (
    "final_destination",
    "battlefield",
    "pokemon_stadium",
    "dreamland",
    "fountain_of_dreams",
    "yoshis_story"
)
CHARACTERS: Final[Tuple[str, ...]] = (
    "mario",
    "fox",
    "cptfalcon",
    "dk",
    "kirby",
    "bowser",
    "link",
    "sheik",
    "ness",
    "peach",
    "popo",
    "pikachu",
    "samus",
    "yoshi",
    "jigglypuff",
    "mewtwo",
    "luigi",
    "marth",
    "zelda",
    "ylink",
    "doc",
    "falco",
    "pichu",
    "gameandwatch",
    "ganondorf",
    "roy"
)

class Character(Enum):
    """"""
    MARIO = 0x00
    FOX = 0x01
    CPTFALCON = 0x02
    DK = 0x03
    KIRBY = 0x04
    BOWSER = 0x05
    LINK = 0x06
    SHEIK = 0x07
    NESS = 0x08
    PEACH = 0x09
    POPO = 0x0a
    PIKACHU = 0x0c
    SAMUS = 0x0d
    YOSHI = 0x0e
    JIGGLYPUFF = 0x0f
    MEWTWO = 0x10
    LUIGI = 0x11
    MARTH = 0x12
    ZELDA = 0x13
    YLINK = 0x14
    DOC = 0x15
    FALCO = 0x16
    PICHU = 0x17
    GAMEANDWATCH = 0x18
    GANONDORF = 0x19
    ROY = 0x1a


# Evaluation
DEVICES = Literal["cpu", "cuda", "mps"]
EVAL_MODE = Literal["cpu", "model"]
EVAL_STAGES = Literal["all", "fd", "bf", "ps", "dl", "fod", "ys"]
