from hal.constants import INCLUDED_BUTTONS_NO_SHOULDER
from hal.constants import SHOULDER_CLUSTER_CENTERS_V0
from hal.constants import STICK_XY_CLUSTER_CENTERS_V0_1
from hal.constants import STICK_XY_CLUSTER_CENTERS_V2
from hal.preprocess.registry import TargetConfig
from hal.preprocess.registry import TargetConfigRegistry
from hal.preprocess.transformations import encode_buttons_one_hot_no_shoulder
from hal.preprocess.transformations import encode_c_stick_one_hot_coarser
from hal.preprocess.transformations import encode_main_stick_one_hot_fine
from hal.preprocess.transformations import encode_shoulder_one_hot_coarse


def next_and_12_frames() -> TargetConfig:
    return TargetConfig(
        transformation_by_target={
            "main_stick_1": encode_main_stick_one_hot_fine,
            "c_stick_1": encode_c_stick_one_hot_coarser,
            "buttons_1": encode_buttons_one_hot_no_shoulder,
            "shoulder_1": encode_shoulder_one_hot_coarse,
            "main_stick_12": encode_main_stick_one_hot_fine,
            "c_stick_12": encode_c_stick_one_hot_coarser,
            "buttons_12": encode_buttons_one_hot_no_shoulder,
            "shoulder_12": encode_shoulder_one_hot_coarse,
        },
        frame_offsets_by_target={
            "main_stick_1": 0,
            "c_stick_1": 0,
            "buttons_1": 0,
            "shoulder_1": 0,
            "main_stick_12": 11,
            "c_stick_12": 11,
            "buttons_12": 11,
            "shoulder_12": 11,
        },
        target_shapes_by_head={
            "main_stick_1": (len(STICK_XY_CLUSTER_CENTERS_V2),),
            "c_stick_1": (len(STICK_XY_CLUSTER_CENTERS_V0_1),),
            "buttons_1": (len(INCLUDED_BUTTONS_NO_SHOULDER),),
            "shoulder_1": (len(SHOULDER_CLUSTER_CENTERS_V0),),
            "main_stick_12": (len(STICK_XY_CLUSTER_CENTERS_V2),),
            "c_stick_12": (len(STICK_XY_CLUSTER_CENTERS_V0_1),),
            "buttons_12": (len(INCLUDED_BUTTONS_NO_SHOULDER),),
            "shoulder_12": (len(SHOULDER_CLUSTER_CENTERS_V0),),
        },
    )


TargetConfigRegistry.register("next_and_12_frames", next_and_12_frames())
