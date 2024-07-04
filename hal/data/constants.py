from melee import Action
from melee import Character
from melee import Stage

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
