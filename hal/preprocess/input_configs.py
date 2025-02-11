from hal.constants import INCLUDED_BUTTONS
from hal.constants import SHOULDER_CLUSTER_CENTERS_V0
from hal.constants import STICK_XY_CLUSTER_CENTERS_V0
from hal.constants import STICK_XY_CLUSTER_CENTERS_V1
from hal.preprocess.input_config import InputConfig
from hal.preprocess.registry import InputConfigRegistry
from hal.preprocess.transformations import cast_int32
from hal.preprocess.transformations import concat_controller_inputs_coarse
from hal.preprocess.transformations import concat_controller_inputs_fine
from hal.preprocess.transformations import concat_controller_inputs_fine_shoulder
from hal.preprocess.transformations import invert_and_normalize
from hal.preprocess.transformations import normalize
from hal.preprocess.transformations import standardize

DEFAULT_HEAD_NAME = "gamestate"


def inputs_v0() -> InputConfig:
    """
    Baseline input features.

    Separate embedding heads for stage, character, & action.
    No controller, no platforms, no projectiles.
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
            "frame": cast_int32,
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
        },
        frame_offsets_by_feature={},
        grouped_feature_names_by_head={
            "stage": ("stage",),
            "ego_character": ("ego_character",),
            "opponent_character": ("opponent_character",),
            "ego_action": ("ego_action",),
            "opponent_action": ("opponent_action",),
        },
        input_shapes_by_head={
            DEFAULT_HEAD_NAME: (2 * 9 + 1,),  # 2x for ego and opponent + 1 for frame
        },
    )


def inputs_v0_controller() -> InputConfig:
    """
    Baseline input features, controller inputs.

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
            # Ego controller inputs
            "controller": concat_controller_inputs_coarse,
        },
        frame_offsets_by_feature={
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
            DEFAULT_HEAD_NAME: (2 * 9 + 2 * len(STICK_XY_CLUSTER_CENTERS_V0) + len(INCLUDED_BUTTONS),),
        },
    )


def inputs_v1() -> InputConfig:
    """
    Baseline input features + action frame.

    Separate embedding heads for stage, character, & action.
    No controller, no platforms, no projectiles.
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
        "action_frame",
    )

    return InputConfig(
        player_features=player_features,
        transformation_by_feature_name={
            # Shared/embedded features are passed unchanged, to be embedded by model
            "frame": cast_int32,
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
            "action_frame": normalize,
        },
        frame_offsets_by_feature={},
        grouped_feature_names_by_head={
            "stage": ("stage",),
            "ego_character": ("ego_character",),
            "opponent_character": ("opponent_character",),
            "ego_action": ("ego_action",),
            "opponent_action": ("opponent_action",),
        },
        input_shapes_by_head={
            DEFAULT_HEAD_NAME: (2 * 10 + 1,),  # 2x for ego and opponent + 1 for frame
        },
    )


def inputs_v1_controller() -> InputConfig:
    """
    Baseline input features + action frame, controller inputs.

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
        "action_frame",
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
            "action_frame": normalize,
            # Ego controller inputs
            "controller": concat_controller_inputs_coarse,
        },
        frame_offsets_by_feature={
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
            DEFAULT_HEAD_NAME: (2 * 10 + 2 * len(STICK_XY_CLUSTER_CENTERS_V0) + len(INCLUDED_BUTTONS),),
        },
    )


def inputs_v2_controller() -> InputConfig:
    """
    Baseline input features, finer-grained clusters for controller inputs.

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
            # Ego controller inputs
            "controller": concat_controller_inputs_fine,
        },
        frame_offsets_by_feature={
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
            DEFAULT_HEAD_NAME: (2 * 9 + 2 * len(STICK_XY_CLUSTER_CENTERS_V1) + len(INCLUDED_BUTTONS),),
        },
    )


def inputs_v3_controller() -> InputConfig:
    """
    Baseline input features, finer-grained clusters for controller inputs.

    Predicting analog shoulder presses.

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
            # Ego controller inputs
            "controller": concat_controller_inputs_fine_shoulder,
        },
        frame_offsets_by_feature={
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
            DEFAULT_HEAD_NAME: (
                2 * 9
                + 2 * len(STICK_XY_CLUSTER_CENTERS_V1)
                + len(INCLUDED_BUTTONS)
                + len(SHOULDER_CLUSTER_CENTERS_V0),
            ),
        },
    )


InputConfigRegistry.register("inputs_v0", inputs_v0())
InputConfigRegistry.register("inputs_v0_controller", inputs_v0_controller())
InputConfigRegistry.register("inputs_v1", inputs_v1())
InputConfigRegistry.register("inputs_v1_controller", inputs_v1_controller())
InputConfigRegistry.register("inputs_v2_controller", inputs_v2_controller())
InputConfigRegistry.register("inputs_v3_controller", inputs_v3_controller())
