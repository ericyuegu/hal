from hal.preprocess.postprocess_config import PostprocessConfig
from hal.preprocess.registry import PostprocessConfigRegistry
from hal.preprocess.transformations import sample_analog_shoulder
from hal.preprocess.transformations import sample_analog_shoulder_coarse
from hal.preprocess.transformations import sample_c_stick_coarse
from hal.preprocess.transformations import sample_c_stick_coarser
from hal.preprocess.transformations import sample_c_stick_fine
from hal.preprocess.transformations import sample_digital_shoulder_l
from hal.preprocess.transformations import sample_digital_shoulder_r
from hal.preprocess.transformations import sample_main_stick_coarse
from hal.preprocess.transformations import sample_main_stick_fine
from hal.preprocess.transformations import sample_main_stick_finer
from hal.preprocess.transformations import sample_original_single_button_no_shoulder
from hal.preprocess.transformations import sample_single_button
from hal.preprocess.transformations import sample_single_button_no_shoulder
from hal.preprocess.transformations import threshold_independent_buttons


def baseline_coarse() -> PostprocessConfig:
    return PostprocessConfig(
        transformation_by_controller_input={
            "main_stick": sample_main_stick_coarse,
            "c_stick": sample_c_stick_coarse,
            "buttons": sample_single_button,
        }
    )


def baseline_fine() -> PostprocessConfig:
    return PostprocessConfig(
        transformation_by_controller_input={
            "main_stick": sample_main_stick_fine,
            "c_stick": sample_c_stick_fine,
            "buttons": sample_single_button,
        }
    )


def baseline_coarse_shoulder() -> PostprocessConfig:
    return PostprocessConfig(
        transformation_by_controller_input={
            "main_stick": sample_main_stick_coarse,
            "c_stick": sample_c_stick_coarse,
            "buttons": sample_single_button,
            "shoulder": sample_analog_shoulder_coarse,
        }
    )


def fine_main_analog_shoulder() -> PostprocessConfig:
    return PostprocessConfig(
        transformation_by_controller_input={
            "main_stick": sample_main_stick_fine,
            "c_stick": sample_c_stick_coarser,
            "buttons": sample_single_button_no_shoulder,
            "shoulder": sample_analog_shoulder_coarse,
        }
    )


def baseline_finer() -> PostprocessConfig:
    return PostprocessConfig(
        transformation_by_controller_input={
            "main_stick": sample_main_stick_finer,
            "c_stick": sample_c_stick_coarser,
            "buttons": sample_single_button,
        }
    )


def fine_main_coarser_cstick() -> PostprocessConfig:
    return PostprocessConfig(
        transformation_by_controller_input={
            "main_stick": sample_main_stick_fine,
            "c_stick": sample_c_stick_coarser,
            "buttons": sample_single_button,
        }
    )


def fine_orig_buttons() -> PostprocessConfig:
    # independent buttons
    return PostprocessConfig(
        transformation_by_controller_input={
            "main_stick": sample_main_stick_fine,
            "c_stick": sample_c_stick_coarser,
            "buttons": threshold_independent_buttons,
        }
    )


def fine_orig_buttons_one_hot_shoulder_one_hot() -> PostprocessConfig:
    return PostprocessConfig(
        transformation_by_controller_input={
            "main_stick": sample_main_stick_fine,
            "c_stick": sample_c_stick_coarser,
            "buttons": sample_original_single_button_no_shoulder,
            "shoulder": sample_analog_shoulder_coarse,
        }
    )


def separate_digital_shoulders_analog_shoulder_one_hot() -> PostprocessConfig:
    return PostprocessConfig(
        transformation_by_controller_input={
            "main_stick": sample_main_stick_fine,
            "c_stick": sample_c_stick_coarser,
            "buttons": sample_original_single_button_no_shoulder,
            "analog_shoulder": sample_analog_shoulder,
            "shoulder_l": sample_digital_shoulder_l,
            "shoulder_r": sample_digital_shoulder_r,
        }
    )


PostprocessConfigRegistry.register("baseline_coarse", baseline_coarse())
PostprocessConfigRegistry.register("baseline_fine", baseline_fine())
PostprocessConfigRegistry.register("baseline_coarse_shoulder", baseline_coarse_shoulder())
PostprocessConfigRegistry.register("fine_main_analog_shoulder", fine_main_analog_shoulder())
PostprocessConfigRegistry.register("baseline_finer", baseline_finer())
PostprocessConfigRegistry.register("fine_main_coarser_cstick", fine_main_coarser_cstick())
PostprocessConfigRegistry.register("fine_orig_buttons", fine_orig_buttons())
PostprocessConfigRegistry.register(
    "fine_orig_buttons_one_hot_shoulder_one_hot", fine_orig_buttons_one_hot_shoulder_one_hot()
)
PostprocessConfigRegistry.register(
    "separate_digital_shoulders_analog_shoulder_one_hot", separate_digital_shoulders_analog_shoulder_one_hot()
)
