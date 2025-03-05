from hal.constants import INCLUDED_BUTTONS
from hal.constants import SHOULDER_CLUSTER_CENTERS_V2
from hal.constants import STICK_XY_CLUSTER_CENTERS_V0_1
from hal.constants import STICK_XY_CLUSTER_CENTERS_V2
from hal.preprocess.registry import TargetConfig
from hal.preprocess.registry import TargetConfigRegistry
from hal.preprocess.transformations import encode_buttons_one_hot_early_release
from hal.preprocess.transformations import encode_c_stick_one_hot_coarser
from hal.preprocess.transformations import encode_main_stick_one_hot_fine
from hal.preprocess.transformations import encode_shoulder_one_hot
from hal.preprocess.transformations import get_returns


def multi_token(frames: tuple[int, ...]) -> TargetConfig:
    transformation_by_target = {
        "main_stick": encode_main_stick_one_hot_fine,
        "c_stick": encode_c_stick_one_hot_coarser,
        "buttons": encode_buttons_one_hot_early_release,
        "shoulder": encode_shoulder_one_hot,
    }
    target_shapes_by_head = {
        "main_stick": (len(STICK_XY_CLUSTER_CENTERS_V2),),
        "c_stick": (len(STICK_XY_CLUSTER_CENTERS_V0_1),),
        "buttons": (len(INCLUDED_BUTTONS),),
        "shoulder": (len(SHOULDER_CLUSTER_CENTERS_V2),),
    }

    return TargetConfig(
        transformation_by_target={
            f"{k}_{frame}": transformation_by_target[k] for k in transformation_by_target for frame in frames
        },
        frame_offsets_by_target={f"{k}_{frame}": frame - 1 for k in transformation_by_target for frame in frames},
        target_shapes_by_head={
            f"{k}_{frame}": target_shapes_by_head[k] for k in target_shapes_by_head for frame in frames
        },
        multi_token_heads=frames,
    )


def multi_token_value(frames: tuple[int, ...]) -> TargetConfig:
    transformation_by_target = {
        "main_stick": encode_main_stick_one_hot_fine,
        "c_stick": encode_c_stick_one_hot_coarser,
        "buttons": encode_buttons_one_hot_early_release,
        "shoulder": encode_shoulder_one_hot,
        "returns": get_returns,
    }
    target_shapes_by_head = {
        "main_stick": (len(STICK_XY_CLUSTER_CENTERS_V2),),
        "c_stick": (len(STICK_XY_CLUSTER_CENTERS_V0_1),),
        "buttons": (len(INCLUDED_BUTTONS),),
        "shoulder": (len(SHOULDER_CLUSTER_CENTERS_V2),),
        "returns": (1,),
    }

    return TargetConfig(
        transformation_by_target={
            f"{k}_{frame}": transformation_by_target[k] for k in transformation_by_target for frame in frames
        },
        frame_offsets_by_target={
            f"{k}_{frame}": frame - 1 for k in transformation_by_target for frame in frames if k != "returns"
        },
        target_shapes_by_head={
            f"{k}_{frame}": target_shapes_by_head[k] for k in target_shapes_by_head for frame in frames
        },
        multi_token_heads=frames,
    )


TargetConfigRegistry.register("frame_1_and_12", multi_token((1, 12)))
TargetConfigRegistry.register("frame_1_12_18", multi_token((1, 12, 18)))
TargetConfigRegistry.register("frame_1_and_12_value", multi_token_value((1, 12)))
