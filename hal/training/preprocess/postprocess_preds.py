import torch
from tensordict import TensorDict

from hal.constants import STICK_XY_CLUSTER_CENTERS_V0
from hal.training.preprocess.registry import PredPostprocessingRegistry


@PredPostprocessingRegistry.register("targets_v0")
def model_predictions_to_controller_inputs_v0(pred: TensorDict) -> TensorDict:
    """
    Reverse the one-hot encoding of buttons and analog stick x, y values for a given player.
    """
    # Decode main stick and c-stick
    main_stick_cluster_idx = torch.argmax(pred["main_stick"], dim=-1)
    main_stick_x, main_stick_y = torch.split(
        torch.tensor(STICK_XY_CLUSTER_CENTERS_V0[main_stick_cluster_idx]), 1, dim=-1
    )

    c_stick_cluster_idx = torch.argmax(pred["c_stick"], dim=-1)
    c_stick_x, c_stick_y = torch.split(torch.tensor(STICK_XY_CLUSTER_CENTERS_V0[c_stick_cluster_idx]), 1, dim=-1)

    # Decode buttons
    one_hot_buttons = pred["buttons"]
    button_a, button_b, jump, button_z, shoulder, no_button = torch.split(one_hot_buttons, 1, dim=-1)

    return TensorDict(
        {
            "main_stick_x": main_stick_x,
            "main_stick_y": main_stick_y,
            "c_stick_x": c_stick_x,
            "c_stick_y": c_stick_y,
            "button_a": button_a,
            "button_b": button_b,
            "button_x": jump,
            "button_z": button_z,
            "button_l": shoulder,
            "button_none": no_button,
        }
    )
