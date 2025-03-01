from functools import partial

import torch
from tensordict import TensorDict

from hal.constants import INCLUDED_BUTTONS
from hal.constants import SHOULDER_CLUSTER_CENTERS_V2
from hal.constants import STICK_XY_CLUSTER_CENTERS_V0_1
from hal.constants import STICK_XY_CLUSTER_CENTERS_V2
from hal.preprocess.postprocess_config import PostprocessConfig
from hal.preprocess.registry import PostprocessConfigRegistry


def sample_main_stick_fine(pred_C: TensorDict, temperature: float = 1.0, frame: int = 1) -> tuple[float, float]:
    main_stick_probs = torch.softmax(pred_C[f"main_stick_{frame}"] / temperature, dim=-1)
    main_stick_cluster_idx = torch.multinomial(main_stick_probs, num_samples=1)
    main_stick_x, main_stick_y = torch.split(
        torch.tensor(STICK_XY_CLUSTER_CENTERS_V2[main_stick_cluster_idx]), 1, dim=-1
    )

    return main_stick_x.item(), main_stick_y.item()


def sample_c_stick_coarser(pred_C: TensorDict, temperature: float = 1.0, frame: int = 1) -> tuple[float, float]:
    c_stick_probs = torch.softmax(pred_C[f"c_stick_{frame}"] / temperature, dim=-1)
    c_stick_cluster_idx = torch.multinomial(c_stick_probs, num_samples=1)
    c_stick_x, c_stick_y = torch.split(torch.tensor(STICK_XY_CLUSTER_CENTERS_V0_1[c_stick_cluster_idx]), 1, dim=-1)

    return c_stick_x.item(), c_stick_y.item()


def sample_single_button(pred_C: TensorDict, temperature: float = 1.0, frame: int = 1) -> list[str]:
    button_probs = torch.softmax(pred_C[f"buttons_{frame}"] / temperature, dim=-1)
    button_idx = int(torch.multinomial(button_probs, num_samples=1).item())
    button = INCLUDED_BUTTONS[button_idx]
    return [button]


def sample_analog_shoulder(pred_C: TensorDict, temperature: float = 1.0, frame: int = 1) -> float:
    shoulder_input = pred_C.get(f"analog_shoulder_{frame}", pred_C[f"shoulder_{frame}"])
    shoulder_probs = torch.softmax(shoulder_input / temperature, dim=-1)
    shoulder_idx = int(torch.multinomial(shoulder_probs, num_samples=1).item())
    shoulder = SHOULDER_CLUSTER_CENTERS_V2[shoulder_idx]
    return shoulder


BASELINE_TRANSFORMATION_BY_CONTROLLER_INPUT = {
    "main_stick": sample_main_stick_fine,
    "c_stick": sample_c_stick_coarser,
    "buttons": sample_single_button,
    "shoulder": sample_analog_shoulder,
}


def frame_1() -> PostprocessConfig:
    return PostprocessConfig(
        transformation_by_controller_input={
            k: partial(v, frame=1) for k, v in BASELINE_TRANSFORMATION_BY_CONTROLLER_INPUT.items()
        }
    )


def frame_12() -> PostprocessConfig:
    return PostprocessConfig(
        transformation_by_controller_input={
            k: partial(v, frame=12) for k, v in BASELINE_TRANSFORMATION_BY_CONTROLLER_INPUT.items()
        }
    )


PostprocessConfigRegistry.register("frame_1", frame_1())
PostprocessConfigRegistry.register("frame_12", frame_12())
