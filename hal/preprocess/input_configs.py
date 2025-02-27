from functools import partial

import attr

from hal.preprocess.input_config import InputConfig
from hal.preprocess.registry import InputConfigRegistry
from hal.preprocess.target_configs import baseline_coarse
from hal.preprocess.target_configs import baseline_fine
from hal.preprocess.target_configs import baseline_finer
from hal.preprocess.target_configs import fine_main_analog_shoulder
from hal.preprocess.target_configs import fine_main_coarser_cstick
from hal.preprocess.target_configs import fine_orig_buttons
from hal.preprocess.target_configs import fine_orig_buttons_one_hot_shoulder_one_hot
from hal.preprocess.target_configs import separate_digital_shoulders_analog_shoulder_one_hot
from hal.preprocess.transformations import cast_int32
from hal.preprocess.transformations import concat_controller_inputs
from hal.preprocess.transformations import invert_and_normalize
from hal.preprocess.transformations import normalize
from hal.preprocess.transformations import normalize_and_embed_fourier
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
            # TODO handle Nana
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


def baseline_controller_finer() -> InputConfig:
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
            "controller": partial(concat_controller_inputs, target_config=baseline_finer()),
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
            "controller": (baseline_finer().target_size,),
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


def fourier_xy() -> InputConfig:
    """
    Baseline input features + controller inputs + Fourier-transformed x/y positions.

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
            "position_x": partial(normalize_and_embed_fourier, dim=4),
            "position_y": partial(normalize_and_embed_fourier, dim=4),
            # Target features
            "controller": partial(concat_controller_inputs, target_config=baseline_coarse()),
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
            # TODO handle Nana
        },
        input_shapes_by_head={
            DEFAULT_HEAD_NAME: (2 * 7 + (2 * 2 * 4),),  # 2x for ego and opponent
            "controller": (baseline_coarse().target_size,),
        },
    )


def baseline_controller_fine_main_analog_shoulder() -> InputConfig:
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
            "controller": partial(concat_controller_inputs, target_config=fine_main_analog_shoulder()),
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
            "controller": (fine_main_analog_shoulder().target_size,),
        },
    )
    return config


def baseline_fine_main_coarser_cstick() -> InputConfig:
    """
    Baseline input features, fine-grained controller inputs, coarser c-stick.

    Separate embedding heads for stage, character, & action.
    No platforms, no projectiles.
    """

    base_config = baseline()
    config = attr.evolve(
        base_config,
        transformation_by_feature_name={
            **base_config.transformation_by_feature_name,
            "controller": partial(concat_controller_inputs, target_config=fine_main_coarser_cstick()),
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
            "controller": (fine_main_coarser_cstick().target_size,),
        },
    )
    return config


def baseline_fine_orig_buttons() -> InputConfig:
    """
    Baseline input features, fine-grained controller inputs, original buttons.

    Separate embedding heads for stage, character, & action.
    No platforms, no projectiles.
    """

    base_config = baseline()
    config = attr.evolve(
        base_config,
        transformation_by_feature_name={
            **base_config.transformation_by_feature_name,
            "controller": partial(concat_controller_inputs, target_config=fine_orig_buttons()),
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
            "controller": (fine_orig_buttons().target_size,),
        },
    )
    return config


def baseline_fine_orig_buttons_one_hot_no_shoulder() -> InputConfig:
    """
    Baseline input features, fine-grained controller inputs, original buttons.

    Separate embedding heads for stage, character, & action.
    No platforms, no projectiles.
    """

    base_config = baseline()
    config = attr.evolve(
        base_config,
        transformation_by_feature_name={
            **base_config.transformation_by_feature_name,
            "controller": partial(
                concat_controller_inputs, target_config=fine_orig_buttons_one_hot_shoulder_one_hot()
            ),
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
            "controller": (fine_orig_buttons_one_hot_shoulder_one_hot().target_size,),
        },
    )
    return config


def baseline_separate_digital_shoulders_analog_shoulder_one_hot() -> InputConfig:
    """
    Baseline input features, fine-grained controller inputs, original buttons.

    Separate embedding heads for stage, character, & action.
    No platforms, no projectiles.
    """

    base_config = baseline()
    config = attr.evolve(
        base_config,
        transformation_by_feature_name={
            **base_config.transformation_by_feature_name,
            "controller": partial(
                concat_controller_inputs, target_config=separate_digital_shoulders_analog_shoulder_one_hot()
            ),
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
            "controller": (separate_digital_shoulders_analog_shoulder_one_hot().target_size,),
        },
    )
    return config


InputConfigRegistry.register("baseline", baseline())
InputConfigRegistry.register("baseline_controller", baseline_controller())
InputConfigRegistry.register("baseline_controller_fine", baseline_controller_fine())
InputConfigRegistry.register("baseline_action_frame", baseline_action_frame())
InputConfigRegistry.register("baseline_action_frame_controller", baseline_action_frame_controller())
InputConfigRegistry.register("fourier_xy", fourier_xy())
InputConfigRegistry.register(
    "baseline_controller_fine_main_analog_shoulder", baseline_controller_fine_main_analog_shoulder()
)
InputConfigRegistry.register("baseline_controller_finer", baseline_controller_finer())
InputConfigRegistry.register("baseline_fine_main_coarser_cstick", baseline_fine_main_coarser_cstick())
InputConfigRegistry.register("baseline_fine_orig_buttons", baseline_fine_orig_buttons())
InputConfigRegistry.register(
    "baseline_fine_orig_buttons_one_hot_no_shoulder", baseline_fine_orig_buttons_one_hot_no_shoulder()
)
InputConfigRegistry.register(
    "separate_digital_shoulders_analog_shoulder_one_hot", baseline_separate_digital_shoulders_analog_shoulder_one_hot()
)
