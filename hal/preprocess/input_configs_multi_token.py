from functools import partial

from hal.preprocess.input_config import InputConfig
from hal.preprocess.registry import InputConfigRegistry
from hal.preprocess.target_configs import fine_main_analog_shoulder
from hal.preprocess.transformations import cast_int32
from hal.preprocess.transformations import concat_controller_inputs
from hal.preprocess.transformations import invert_and_normalize
from hal.preprocess.transformations import normalize
from hal.preprocess.transformations import standardize

DEFAULT_HEAD_NAME = "gamestate"


def baseline_controller_fine_main_analog_shoulder() -> InputConfig:
    """
    Baseline input features, fine-grained controller inputs.

    Separate embedding heads for stage, character, & action.
    No platforms, no projectiles.
    """

    player_features = (
        "character",
        "action",
        "percent",
        "stock",
        "facing",
        "invulnerable",
        "jumps_left",
        "on_ground",
        "shield_strength",
        "position_x",
        "position_y",
    )

    return InputConfig(
        player_features=player_features,
        transformation_by_feature_name={
            # Shared/embedded features are passed unchanged, to be embedded by model
            "stage": cast_int32,
            "character": cast_int32,
            "action": cast_int32,
            # Normalized player features
            "percent": normalize,
            "stock": normalize,
            "facing": normalize,
            "invulnerable": normalize,
            "jumps_left": normalize,
            "on_ground": normalize,
            "shield_strength": invert_and_normalize,
            "position_x": standardize,
            "position_y": standardize,
            "controller": partial(concat_controller_inputs, target_config=fine_main_analog_shoulder()),
        },
        frame_offsets_by_input={
            "controller": -1,
        },
        grouped_feature_names_by_head={
            "stage": ("stage",),
            "ego_character": ("ego_character",),
            "opponent_character": ("opponent_character",),
            "ego_action": ("ego_action",),
            "opponent_action": ("opponent_action",),
            "controller": ("controller",),
        },
        input_shapes_by_head={
            DEFAULT_HEAD_NAME: (2 * 9,),  # 2x for ego and opponent
            "controller": (fine_main_analog_shoulder().target_size,),
        },
    )


InputConfigRegistry.register(
    "baseline_controller_fine_main_analog_shoulder", baseline_controller_fine_main_analog_shoulder()
)
