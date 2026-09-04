import pytest
import torch

from hal.training.controller_codec import BUTTONS_GROUP
from hal.training.controller_codec import TRIGGERS_GROUP
from hal.training.controller_codec import DiscreteControllerCodec
from hal.wire import ACTION_CHANNELS
from hal.wire import ACTION_DIM


def test_quantized_controller_is_a_fixed_point() -> None:
    codec = DiscreteControllerCodec(embed_dim=16)
    actions = torch.zeros(3, 5, ACTION_DIM)
    actions[..., ACTION_CHANNELS.index("main_stick_x")] = 0.25
    actions[..., ACTION_CHANNELS.index("c_stick_y")] = -0.75

    indices = codec.quantize(actions)

    assert torch.equal(codec.quantize(codec.dequantize(indices)), indices)


def test_click_canonicalization_requires_a_full_trigger() -> None:
    codec = DiscreteControllerCodec(embed_dim=16)
    actions = torch.zeros(1, ACTION_DIM)
    actions[..., ACTION_CHANNELS.index("button_l")] = 1.0

    indices = codec.quantize(actions)
    decoded = codec.dequantize(indices)

    assert decoded[..., ACTION_CHANNELS.index("trigger_l")].item() == 1.0
    assert not codec.button_mask(indices[..., TRIGGERS_GROUP])[0, indices[0, BUTTONS_GROUP]]


def test_codec_keeps_checkpoint_parameter_names() -> None:
    names = tuple(DiscreteControllerCodec(embed_dim=8).state_dict())

    assert names == (
        "main_centers",
        "c_centers",
        "trigger_centers",
        "button_valid_for_trigger",
        "class_embeddings.buttons.weight",
        "class_embeddings.main_stick.weight",
        "class_embeddings.c_stick.weight",
        "class_embeddings.triggers.weight",
        "semantic_projections.buttons.weight",
        "semantic_projections.main_stick.weight",
        "semantic_projections.c_stick.weight",
        "semantic_projections.triggers.weight",
    )


def test_codec_rejects_the_wrong_wire_width() -> None:
    with pytest.raises(ValueError, match="channels"):
        DiscreteControllerCodec.canonicalize(torch.zeros(2, ACTION_DIM - 1))
