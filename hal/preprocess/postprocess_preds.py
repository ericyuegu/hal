import torch
from tensordict import TensorDict

from hal.constants import SHOULDER_CLUSTER_CENTERS_V0
from hal.constants import STICK_XY_CLUSTER_CENTERS_V0
from hal.constants import STICK_XY_CLUSTER_CENTERS_V1
from hal.preprocess.registry import PredPostprocessingRegistry


@PredPostprocessingRegistry.register("preds_v0")
def model_predictions_to_controller_inputs_v0(pred_C: TensorDict, temperature: float = 1.0) -> TensorDict:
    """
    Sample using temperature from the predicted distribution.
    """
    # Decode x, y from joint categorical distribution
    main_stick_probs = torch.softmax(pred_C["main_stick"] / temperature, dim=-1)
    main_stick_cluster_idx = torch.multinomial(main_stick_probs, num_samples=1)
    main_stick_x, main_stick_y = torch.split(
        torch.tensor(STICK_XY_CLUSTER_CENTERS_V0[main_stick_cluster_idx]), 1, dim=-1
    )

    c_stick_probs = torch.softmax(pred_C["c_stick"] / temperature, dim=-1)
    c_stick_cluster_idx = torch.multinomial(c_stick_probs, num_samples=1)
    c_stick_x, c_stick_y = torch.split(torch.tensor(STICK_XY_CLUSTER_CENTERS_V0[c_stick_cluster_idx]), 1, dim=-1)

    # Decode buttons
    button_probs = torch.softmax(pred_C["buttons"] / temperature, dim=-1)
    button_idx = torch.multinomial(button_probs, num_samples=1)

    return TensorDict(
        {
            "main_stick_x": main_stick_x,
            "main_stick_y": main_stick_y,
            "c_stick_x": c_stick_x,
            "c_stick_y": c_stick_y,
            "button": button_idx,
        }
    )


@PredPostprocessingRegistry.register("preds_v0_greedy")
def model_predictions_to_controller_inputs_v0_greedy(pred: TensorDict) -> TensorDict:
    """
    Argmax the main stick and c-stick clusters, and the buttons.
    """
    # Decode main stick and c-stick
    main_stick_cluster_idx = torch.argmax(pred["main_stick"], dim=-1, keepdim=True)
    main_stick_x, main_stick_y = torch.split(
        torch.tensor(STICK_XY_CLUSTER_CENTERS_V0[main_stick_cluster_idx]), 1, dim=-1
    )

    c_stick_cluster_idx = torch.argmax(pred["c_stick"], dim=-1, keepdim=True)
    c_stick_x, c_stick_y = torch.split(torch.tensor(STICK_XY_CLUSTER_CENTERS_V0[c_stick_cluster_idx]), 1, dim=-1)

    # Decode buttons
    one_hot_buttons = pred["buttons"]
    button_idx = torch.argmax(one_hot_buttons, dim=-1, keepdim=True)

    return TensorDict(
        {
            "main_stick_x": main_stick_x,
            "main_stick_y": main_stick_y,
            "c_stick_x": c_stick_x,
            "c_stick_y": c_stick_y,
            "button": button_idx,
        }
    )


@PredPostprocessingRegistry.register("preds_v1")
def model_predictions_to_controller_inputs_v1(pred_C: TensorDict, temperature: float = 1.0) -> TensorDict:
    """
    Analog shoulder presses.
    """
    # Decode x, y from joint categorical distribution
    main_stick_probs = torch.softmax(pred_C["main_stick"] / temperature, dim=-1)
    main_stick_cluster_idx = torch.multinomial(main_stick_probs, num_samples=1)
    main_stick_x, main_stick_y = torch.split(
        torch.tensor(STICK_XY_CLUSTER_CENTERS_V1[main_stick_cluster_idx]), 1, dim=-1
    )

    c_stick_probs = torch.softmax(pred_C["c_stick"] / temperature, dim=-1)
    c_stick_cluster_idx = torch.multinomial(c_stick_probs, num_samples=1)
    c_stick_x, c_stick_y = torch.split(torch.tensor(STICK_XY_CLUSTER_CENTERS_V1[c_stick_cluster_idx]), 1, dim=-1)

    # Decode buttons
    button_probs = torch.softmax(pred_C["buttons"] / temperature, dim=-1)
    button_idx = torch.multinomial(button_probs, num_samples=1)

    # Decode shoulder
    shoulder_probs = torch.softmax(pred_C["shoulder"] / temperature, dim=-1)
    shoulder_idx = torch.multinomial(shoulder_probs, num_samples=1)
    shoulder_x = torch.tensor(SHOULDER_CLUSTER_CENTERS_V0[shoulder_idx]).unsqueeze(-1)

    return TensorDict(
        {
            "main_stick_x": main_stick_x,
            "main_stick_y": main_stick_y,
            "c_stick_x": c_stick_x,
            "c_stick_y": c_stick_y,
            "button": button_idx,
            "shoulder": shoulder_x,
        }
    )
