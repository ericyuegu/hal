from typing import Dict
from typing import Final
from typing import Literal
from typing import Tuple

import numpy as np
from melee import Action
from melee import Character
from melee import Stage

VALID_PLAYERS: Final[Tuple[str, str]] = ("p1", "p2")
Player = Literal["p1", "p2"]
PLAYER_1_PORT: Final[int] = 1
PLAYER_2_PORT: Final[int] = 2


def get_opponent(player: Player) -> Player:
    return "p2" if player == "p1" else "p1"


###################
# Gamestate      #
###################

INCLUDED_STAGES: Tuple[str, ...] = (
    "FINAL_DESTINATION",
    "BATTLEFIELD",
    "POKEMON_STADIUM",
    "DREAMLAND",
    "FOUNTAIN_OF_DREAMS",
    "YOSHIS_STORY",
)
IDX_BY_STAGE: Dict[Stage, int] = {
    stage: i for i, stage in enumerate(stage for stage in Stage if stage.name in INCLUDED_STAGES)
}
IDX_BY_STAGE_STR: Dict[str, int] = {stage.name: i for stage, i in IDX_BY_STAGE.items()}
STAGE_BY_IDX: Dict[int, str] = {i: stage.name for stage, i in IDX_BY_STAGE.items()}

INCLUDED_CHARACTERS: Tuple[str, ...] = (
    "MARIO",
    "FOX",
    "CPTFALCON",
    "DK",
    "KIRBY",
    "BOWSER",
    "LINK",
    "SHEIK",
    "NESS",
    "PEACH",
    "POPO",
    "PIKACHU",
    "SAMUS",
    "YOSHI",
    "JIGGLYPUFF",
    "MEWTWO",
    "LUIGI",
    "MARTH",
    "ZELDA",
    "YLINK",
    "DOC",
    "FALCO",
    "PICHU",
    "GAMEANDWATCH",
    "GANONDORF",
    "ROY",
)
IDX_BY_CHARACTER: Dict[Character, int] = {
    char: i for i, char in enumerate(char for char in Character if char.name in INCLUDED_CHARACTERS)
}
IDX_BY_CHARACTER_STR: Dict[str, int] = {char.name: i for char, i in IDX_BY_CHARACTER.items()}
CHARACTER_BY_IDX: Dict[int, str] = {i: char.name for char, i in IDX_BY_CHARACTER.items()}

IDX_BY_ACTION: Dict[Action, int] = {action: i for i, action in enumerate(Action)}
ACTION_BY_IDX: Dict[int, str] = {i: action.name for action, i in IDX_BY_ACTION.items()}

INCLUDED_BUTTONS: Tuple[str, ...] = (
    "BUTTON_A",
    "BUTTON_B",
    "BUTTON_X",
    "BUTTON_Z",
    "BUTTON_L",
    "NO_BUTTON",
)


###################
# Embeddings      #
###################

REPLAY_UUID: Tuple[str] = ("replay_uuid",)
FRAME: Tuple[str] = ("frame",)
STAGE: Tuple[str, ...] = ("stage",)
PLAYER_INPUT_FEATURES_TO_EMBED: Tuple[str, ...] = ("character", "action")
PLAYER_INPUT_FEATURES_TO_NORMALIZE: Tuple[str, ...] = (
    "percent",
    "stock",
    "facing",
    "invulnerable",
    "jumps_left",
    "on_ground",
)
PLAYER_INPUT_FEATURES_TO_INVERT_AND_NORMALIZE: Tuple[str, ...] = ("shield_strength",)
PLAYER_POSITION: Tuple[str, ...] = (
    "position_x",
    "position_y",
)
# Optional input features
PLAYER_ACTION_FRAME_FEATURES: Tuple[str, ...] = (
    "action_frame",
    "hitlag_left",
    "hitstun_left",
)
PLAYER_SPEED_FEATURES: Tuple[str, ...] = (
    "speed_air_x_self",
    "speed_y_self",
    "speed_x_attack",
    "speed_y_attack",
    "speed_ground_x_self",
)
PLAYER_ECB_FEATURES: Tuple[str, ...] = (
    "ecb_bottom_x",
    "ecb_bottom_y",
    "ecb_top_x",
    "ecb_top_y",
    "ecb_left_x",
    "ecb_left_y",
    "ecb_right_x",
    "ecb_right_y",
)
# Target features
TARGET_FEATURES_TO_ONE_HOT_ENCODE: Tuple[str, ...] = (
    "a",
    "b",
    "x",
    "z",
    "l",
    "no_button",
)

STICK_XY_CLUSTER_CENTERS_V0: np.ndarray = np.array(
    [
        [0.5, 0.5],
        [1.0, 0.5],
        [0.0, 0.5],
        [0.50, 0.0],
        [0.50, 1.0],
        [0.50, 0.25],
        [0.50, 0.75],
        [0.75, 0.5],
        [0.25, 0.5],
        [0.15, 0.15],
        [0.85, 0.15],
        [0.85, 0.85],
        [0.15, 0.85],
        [0.28, 0.93],
        [0.28, 0.07],
        [0.72, 0.07],
        [0.72, 0.93],
        [0.07, 0.28],
        [0.07, 0.72],
        [0.93, 0.72],
        [0.93, 0.28],
    ]
)

STICK_XY_CLUSTER_CENTERS_V1: np.ndarray = np.array(
    [
        [0.0, 0.5],
        [0.00625001, 0.5],
        [0.01249999, 0.5],
        [0.01875004, 0.5],
        [0.02510793, 0.35311657],
        [0.03005841, 0.65799654],
        [0.03299065, 0.3349865],
        [0.03376986, 0.5],
        [0.03587092, 0.6727351],
        [0.04220961, 0.31305742],
        [0.04386523, 0.6916493],
        [0.05431645, 0.71487],
        [0.05444592, 0.28490368],
        [0.05454144, 0.5],
        [0.07482443, 0.24709114],
        [0.07801791, 0.7576855],
        [0.09531481, 0.5],
        [0.09535429, 0.78292215],
        [0.09985004, 0.20878187],
        [0.11363883, 0.8066825],
        [0.11647762, 0.18972751],
        [0.13243222, 0.16891299],
        [0.13258195, 0.5],
        [0.1446668, 0.8444289],
        [0.14478308, 0.15668231],
        [0.1494113, 0.15145782],
        [0.15049767, 0.8531804],
        [0.1637692, 0.13653028],
        [0.16442652, 0.3181435],
        [0.1646255, 0.67829835],
        [0.17135878, 0.8696089],
        [0.17299859, 0.5],
        [0.19738317, 0.10951868],
        [0.21023673, 0.8977858],
        [0.22221868, 0.5],
        [0.23655486, 0.08284256],
        [0.27612394, 0.06041595],
        [0.2787372, 0.5],
        [0.29497877, 0.942798],
        [0.2966504, 0.70870084],
        [0.2990079, 0.29176182],
        [0.31974578, 0.03904507],
        [0.3340432, 0.5],
        [0.5, 0.0],
        [0.5, 0.00624999],
        [0.5, 0.01249999],
        [0.5, 0.02083922],
        [0.5, 0.02813881],
        [0.5, 0.04200239],
        [0.5, 0.05247641],
        [0.5, 0.05691455],
        [0.5, 0.10909294],
        [0.5, 0.12218902],
        [0.5, 0.14451367],
        [0.5, 0.2021173],
        [0.5, 0.21594848],
        [0.5, 0.23224661],
        [0.5, 0.2689559],
        [0.5, 0.31208524],
        [0.5, 0.34421578],
        [0.5, 0.5],
        [0.5, 0.6469974],
        [0.5, 0.6555226],
        [0.5, 0.688049],
        [0.5, 0.72464114],
        [0.5, 0.7410333],
        [0.5, 0.7666579],
        [0.5, 0.7844166],
        [0.5, 0.8157609],
        [0.5, 0.8251362],
        [0.5, 0.8408744],
        [0.5, 0.8499187],
        [0.5, 0.8823752],
        [0.5, 0.9084008],
        [0.5, 0.95979774],
        [0.5, 0.9710111],
        [0.5, 1.0],
        [0.6659568, 0.5],
        [0.6802542, 0.03904507],
        [0.7009921, 0.29176182],
        [0.7033496, 0.70870084],
        [0.70502126, 0.942798],
        [0.7212628, 0.5],
        [0.72387606, 0.06041595],
        [0.76344514, 0.08284256],
        [0.7777813, 0.5],
        [0.7897633, 0.8977858],
        [0.80261683, 0.10951868],
        [0.8270014, 0.5],
        [0.82864124, 0.8696089],
        [0.83537453, 0.67829835],
        [0.8355735, 0.3181435],
        [0.8362308, 0.13653028],
        [0.8495023, 0.8531804],
        [0.8505887, 0.15145782],
        [0.8552169, 0.15668231],
        [0.8553332, 0.8444289],
        [0.86741805, 0.5],
        [0.8675678, 0.16891299],
        [0.8835224, 0.18972751],
        [0.8863612, 0.8066825],
        [0.90014994, 0.20878187],
        [0.90464574, 0.78292215],
        [0.9046852, 0.5],
        [0.9219821, 0.7576855],
        [0.92517555, 0.24709114],
        [0.94545853, 0.5],
        [0.9455541, 0.28490368],
        [0.94568354, 0.71487],
        [0.9561348, 0.6916493],
        [0.9577904, 0.31305742],
        [0.9641291, 0.6727351],
        [0.96623015, 0.5],
        [0.96700937, 0.3349865],
        [0.9699416, 0.65799654],
        [0.9748921, 0.35311657],
        [1.0, 0.5],
    ],
    dtype=np.float32,
)
