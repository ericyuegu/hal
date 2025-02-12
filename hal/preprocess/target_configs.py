from hal.constants import INCLUDED_BUTTONS
from hal.constants import SHOULDER_CLUSTER_CENTERS_V0
from hal.constants import STICK_XY_CLUSTER_CENTERS_V0
from hal.constants import STICK_XY_CLUSTER_CENTERS_V1
from hal.preprocess.registry import TargetConfig
from hal.preprocess.registry import TargetConfigRegistry
from hal.preprocess.transformations import encode_buttons_one_hot
from hal.preprocess.transformations import encode_c_stick_one_hot_coarse
from hal.preprocess.transformations import encode_c_stick_one_hot_fine
from hal.preprocess.transformations import encode_main_stick_one_hot_coarse
from hal.preprocess.transformations import encode_main_stick_one_hot_fine
from hal.preprocess.transformations import encode_shoulder_one_hot_coarse
from hal.preprocess.transformations import pass_through


def baseline_coarse() -> TargetConfig:
    return TargetConfig(
        transformation_by_target={
            "main_stick": encode_main_stick_one_hot_coarse,
            "c_stick": encode_c_stick_one_hot_coarse,
            "buttons": encode_buttons_one_hot,
        },
        frame_offsets_by_target={
            "main_stick": 0,
            "c_stick": 0,
            "buttons": 0,
        },
        target_shapes_by_head={
            "main_stick": (len(STICK_XY_CLUSTER_CENTERS_V0),),
            "c_stick": (len(STICK_XY_CLUSTER_CENTERS_V0),),
            "buttons": (len(INCLUDED_BUTTONS),),
        },
    )


def coarse_shoulder() -> TargetConfig:
    return TargetConfig(
        transformation_by_target={
            "main_stick": encode_main_stick_one_hot_coarse,
            "c_stick": encode_c_stick_one_hot_coarse,
            "shoulder": encode_shoulder_one_hot_coarse,
            "buttons": encode_buttons_one_hot,
        },
        frame_offsets_by_target={
            "main_stick": 0,
            "c_stick": 0,
            "shoulder": 0,
            "buttons": 0,
        },
        target_shapes_by_head={
            "main_stick": (len(STICK_XY_CLUSTER_CENTERS_V0),),
            "c_stick": (len(STICK_XY_CLUSTER_CENTERS_V0),),
            "shoulder": (len(SHOULDER_CLUSTER_CENTERS_V0),),
            "buttons": (len(INCLUDED_BUTTONS),),
        },
    )


def baseline_fine() -> TargetConfig:
    return TargetConfig(
        transformation_by_target={
            "main_stick": encode_main_stick_one_hot_fine,
            "c_stick": encode_c_stick_one_hot_fine,
            "buttons": encode_buttons_one_hot,
        },
        frame_offsets_by_target={
            "main_stick": 0,
            "c_stick": 0,
            "buttons": 0,
        },
        target_shapes_by_head={
            "main_stick": (len(STICK_XY_CLUSTER_CENTERS_V1),),
            "c_stick": (len(STICK_XY_CLUSTER_CENTERS_V1),),
            "buttons": (len(INCLUDED_BUTTONS),),
        },
    )


def gaussian_coarse() -> TargetConfig:
    return TargetConfig(
        transformation_by_target={
            "main_stick": pass_through,
            "c_stick": pass_through,
            "buttons": encode_buttons_one_hot,
        },
        frame_offsets_by_target={
            "main_stick": 0,
            "c_stick": 0,
            "buttons": 0,
        },
        target_shapes_by_head={
            "main_stick": (len(STICK_XY_CLUSTER_CENTERS_V0),),
            "c_stick": (len(STICK_XY_CLUSTER_CENTERS_V0),),
            "buttons": (len(INCLUDED_BUTTONS),),
        },
        reference_points=STICK_XY_CLUSTER_CENTERS_V0,
        sigma=0.08,
    )


def gaussian_fine() -> TargetConfig:
    return TargetConfig(
        transformation_by_target={
            "main_stick": pass_through,
            "c_stick": pass_through,
            "buttons": encode_buttons_one_hot,
        },
        frame_offsets_by_target={
            "main_stick": 0,
            "c_stick": 0,
            "buttons": 0,
        },
        target_shapes_by_head={
            "main_stick": (len(STICK_XY_CLUSTER_CENTERS_V1),),
            "c_stick": (len(STICK_XY_CLUSTER_CENTERS_V1),),
            "buttons": (len(INCLUDED_BUTTONS),),
        },
        reference_points=STICK_XY_CLUSTER_CENTERS_V1,
        sigma=0.05,
    )


TargetConfigRegistry.register("baseline_coarse", baseline_coarse())
TargetConfigRegistry.register("coarse_shoulder", coarse_shoulder())
TargetConfigRegistry.register("baseline_fine", baseline_fine())
TargetConfigRegistry.register("gaussian_coarse", gaussian_coarse())
TargetConfigRegistry.register("gaussian_fine", gaussian_fine())
