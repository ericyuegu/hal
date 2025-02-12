from hal.preprocess.postprocess_config import PostprocessConfig
from hal.preprocess.registry import PostprocessConfigRegistry
from hal.preprocess.transformations import sample_buttons
from hal.preprocess.transformations import sample_c_stick_coarse
from hal.preprocess.transformations import sample_c_stick_fine
from hal.preprocess.transformations import sample_main_stick_coarse
from hal.preprocess.transformations import sample_main_stick_fine
from hal.preprocess.transformations import sample_shoulder


def baseline_coarse() -> PostprocessConfig:
    return PostprocessConfig(
        transformation_by_controller_input={
            "main_stick": sample_main_stick_coarse,
            "c_stick": sample_c_stick_coarse,
            "button": sample_buttons,
        }
    )


def baseline_fine() -> PostprocessConfig:
    return PostprocessConfig(
        transformation_by_controller_input={
            "main_stick": sample_main_stick_fine,
            "c_stick": sample_c_stick_fine,
            "button": sample_buttons,
        }
    )


def baseline_coarse_shoulder() -> PostprocessConfig:
    return PostprocessConfig(
        transformation_by_controller_input={
            "main_stick": sample_main_stick_coarse,
            "c_stick": sample_c_stick_coarse,
            "button": sample_buttons,
            "shoulder": sample_shoulder,
        }
    )


PostprocessConfigRegistry.register("baseline_coarse", baseline_coarse())
PostprocessConfigRegistry.register("baseline_fine", baseline_fine())
PostprocessConfigRegistry.register("baseline_coarse_shoulder", baseline_coarse_shoulder())
