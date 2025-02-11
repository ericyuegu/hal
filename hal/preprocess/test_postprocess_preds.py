import pytest
import torch
from tensordict import TensorDict

from hal.constants import STICK_XY_CLUSTER_CENTERS_V0
from hal.training.preprocess.postprocess_preds import model_predictions_to_controller_inputs_v0_temp


@pytest.fixture
def mock_predictions():
    seq_len = 100
    stick_clusters = len(STICK_XY_CLUSTER_CENTERS_V0)
    button_categories = 6

    # Create one-hot encodings
    main_stick = torch.eye(stick_clusters)[torch.randint(0, stick_clusters, (seq_len,))]
    c_stick = torch.eye(stick_clusters)[torch.randint(0, stick_clusters, (seq_len,))]
    buttons = torch.eye(button_categories)[torch.randint(0, button_categories, (seq_len,))]

    return TensorDict(
        {
            "main_stick": main_stick,
            "c_stick": c_stick,
            "buttons": buttons,
        }
    )


def test_model_predictions_to_controller_inputs(mock_predictions) -> None:
    result = model_predictions_to_controller_inputs_v0_temp(mock_predictions)

    # Check if all expected keys are present
    expected_keys = [
        "main_stick_x",
        "main_stick_y",
        "c_stick_x",
        "c_stick_y",
        "button_a",
        "button_b",
        "button_x",
        "button_z",
        "button_l",
        "button_none",
    ]
    assert all(key in result for key in expected_keys)

    # Check shapes
    seq_len = mock_predictions["main_stick"].shape[0]
    for key, value in result.items():
        assert value.shape == (seq_len,)

    # Check if stick values are within the valid range [-1, 1]
    for stick in ["main_stick", "c_stick"]:
        assert torch.all(result[f"{stick}_x"] >= -1) and torch.all(result[f"{stick}_x"] <= 1)
        assert torch.all(result[f"{stick}_y"] >= -1) and torch.all(result[f"{stick}_y"] <= 1)

    # Check if button values are binary (0 or 1)
    button_keys = ["button_a", "button_b", "button_x", "button_z", "button_l", "button_none"]
    for key in button_keys:
        assert torch.all((result[key] == 0) | (result[key] == 1))

    # Check if main_stick and c_stick values are from STICK_XY_CLUSTER_CENTERS_V0
    for stick in ["main_stick", "c_stick"]:
        x_values = result[f"{stick}_x"]
        y_values = result[f"{stick}_y"]
        xy_pairs = torch.stack((x_values, y_values), dim=1)

        for xy_pair in xy_pairs:
            assert any(torch.allclose(xy_pair, torch.tensor(center)) for center in STICK_XY_CLUSTER_CENTERS_V0)

    # Check if exactly one button is pressed at each time step
    button_sum = torch.sum(torch.stack([result[key] for key in button_keys]), dim=0)
    assert torch.all(button_sum == 1)


if __name__ == "__main__":
    pytest.main([__file__])
