import torch
from tensordict import TensorDict

from hal.constants import INCLUDED_BUTTONS
from hal.constants import STICK_XY_CLUSTER_CENTERS_V0


def model_predictions_to_controller_inputs_v0_temp(preds: TensorDict) -> dict[str, torch.Tensor]:
    main_stick_indices = preds["main_stick"].argmax(dim=-1)
    c_stick_indices = preds["c_stick"].argmax(dim=-1)
    button_indices = preds["buttons"].argmax(dim=-1)

    stick_centers = torch.tensor(STICK_XY_CLUSTER_CENTERS_V0, device=main_stick_indices.device)
    main_stick = stick_centers[main_stick_indices]
    c_stick = stick_centers[c_stick_indices]

    result = {
        "main_stick_x": main_stick[:, 0],
        "main_stick_y": main_stick[:, 1],
        "c_stick_x": c_stick[:, 0],
        "c_stick_y": c_stick[:, 1],
    }
    for i, button in enumerate(INCLUDED_BUTTONS):
        key = "button_none" if button == "NO_BUTTON" else f"button_{button.removeprefix('BUTTON_').lower()}"
        result[key] = (button_indices == i).to(torch.int64)
    return result
