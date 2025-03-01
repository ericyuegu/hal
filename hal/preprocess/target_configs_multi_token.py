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


def frame_1_and_12() -> TargetConfig:
    frames = (1, 12)

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


TargetConfigRegistry.register("frame_1_and_12", frame_1_and_12())
