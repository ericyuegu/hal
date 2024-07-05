import numpy as np
from melee import Action
from melee import Button
from melee import Character
from melee import Stage

###################
# Gamestate      #
###################

EXCLUDED_STAGES: tuple[str, ...] = ("NO_STAGE", "RANDOM_STAGE")
IDX_BY_STAGE: dict[Stage, int] = {
    stage: i for i, stage in enumerate(stage for stage in Stage if stage.name not in EXCLUDED_STAGES)
}
STAGE_BY_IDX: dict[int, str] = {i: stage.name for stage, i in IDX_BY_STAGE.items()}

EXCLUDED_CHARACTERS: tuple[str, ...] = (
    "NANA",
    "WIREFRAME_MALE",
    "WIREFRAME_FEMALE",
    "GIGA_BOWSER",
    "SANDBAG",
    "UNKNOWN_CHARACTER",
)
IDX_BY_CHARACTER: dict[Character, int] = {
    char: i for i, char in enumerate(char for char in Character if char.name not in EXCLUDED_CHARACTERS)
}
CHARACTER_BY_IDX: dict[int, str] = {i: char.name for char, i in IDX_BY_CHARACTER.items()}

IDX_BY_ACTION: dict[Action, int] = {action: i for i, action in enumerate(Action)}
ACTION_BY_IDX: dict[int, str] = {i: action.name for action, i in IDX_BY_ACTION.items()}

EXCLUDED_BUTTONS: tuple[str, ...] = (
    "BUTTON_D_DOWN",
    "BUTTON_D_LEFT",
    "BUTTON_D_RIGHT",
)
IDX_BY_BUTTON: dict[Button, int] = {
    button: i for i, button in enumerate(button for button in Button if button.name not in EXCLUDED_BUTTONS)
}
BUTTON_BY_IDX: dict[int, str] = {i: button.name for button, i in IDX_BY_BUTTON.items()}


###################
# Embeddings      #
###################

MAIN_STICK_XY_CLUSTER_CENTERS_V0: np.ndarray = np.array(
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
