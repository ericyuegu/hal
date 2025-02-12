from functools import partial

import attr

from hal.preprocess.input_config import InputConfig
from hal.preprocess.registry import InputConfigRegistry
from hal.preprocess.target_configs import baseline_coarse
from hal.preprocess.target_configs import baseline_fine
from hal.preprocess.transformations import cast_int32
from hal.preprocess.transformations import concat_controller_inputs
from hal.preprocess.transformations import invert_and_normalize
from hal.preprocess.transformations import normalize
from hal.preprocess.transformations import standardize

DEFAULT_HEAD_NAME = "gamestate"


def baseline() -> InputConfig:
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
        frame_offsets_by_input={},
        grouped_feature_names_by_head={
            "stage": ("stage",),
            "ego_character": ("ego_character",),
            "opponent_character": ("opponent_character",),
            "ego_action": ("ego_action",),
            "opponent_action": ("opponent_action",),
        },
        input_shapes_by_head={
            DEFAULT_HEAD_NAME: (2 * 9,),  # 2x for ego and opponent
        },
    )


def baseline_controller() -> InputConfig:
    """
    Baseline input features, coarse controller inputs.

    Separate embedding heads for stage, character, & action.
    No platforms, no projectiles.
    """

    base_config = baseline()
    config = attr.evolve(
        base_config,
        transformation_by_feature_name={
            **base_config.transformation_by_feature_name,
            "controller": partial(concat_controller_inputs, target_config=baseline_coarse()),
        },
        frame_offsets_by_input={
            **base_config.frame_offsets_by_input,
            "controller": -1,
        },
        grouped_feature_names_by_head={
            **base_config.grouped_feature_names_by_head,
            "controller": ("controller",),
        },
        input_shapes_by_head={
            **base_config.input_shapes_by_head,
            "controller": (baseline_coarse().target_size,),
        },
    )
    return config


def baseline_controller_fine() -> InputConfig:
    """
    Baseline input features, fine-grained controller inputs.

    Separate embedding heads for stage, character, & action.
    No platforms, no projectiles.
    """

    base_config = baseline()
    config = attr.evolve(
        base_config,
        transformation_by_feature_name={
            **base_config.transformation_by_feature_name,
            "controller": partial(concat_controller_inputs, target_config=baseline_fine()),
        },
        frame_offsets_by_input={
            **base_config.frame_offsets_by_input,
            "controller": -1,
        },
        grouped_feature_names_by_head={
            **base_config.grouped_feature_names_by_head,
            "controller": ("controller",),
        },
        input_shapes_by_head={
            **base_config.input_shapes_by_head,
            "controller": (baseline_fine().target_size,),
        },
    )
    return config


def baseline_action_frame() -> InputConfig:
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
        frame_offsets_by_input={},
        grouped_feature_names_by_head={
            "stage": ("stage",),
            "ego_character": ("ego_character",),
            "opponent_character": ("opponent_character",),
            "ego_action": ("ego_action",),
            "opponent_action": ("opponent_action",),
        },
        input_shapes_by_head={
            DEFAULT_HEAD_NAME: (2 * 10,),  # 2x for ego and opponent
        },
    )


def baseline_action_frame_controller() -> InputConfig:
    """
    Baseline input features + action frame, controller inputs.

    Separate embedding heads for stage, character, & action.
    No platforms, no projectiles.
    """
    # TODO
    ...


InputConfigRegistry.register("baseline", baseline())
InputConfigRegistry.register("baseline_controller", baseline_controller())
InputConfigRegistry.register("baseline_controller_fine", baseline_controller_fine())
InputConfigRegistry.register("baseline_action_frame", baseline_action_frame())
InputConfigRegistry.register("baseline_action_frame_controller", baseline_action_frame_controller())
