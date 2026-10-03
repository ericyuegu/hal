import numpy as np
import pytest
import torch

from hal.models import controller_codec
from hal.models.controller_codec import BUTTONS_GROUP
from hal.models.controller_codec import TRIGGERS_GROUP
from hal.models.controller_codec import DiscreteControllerCodec
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


def test_stick_centers_live_in_target_space():
    for centers in (controller_codec.STICK_CLUSTER_CENTERS_MAIN, controller_codec.STICK_CLUSTER_CENTERS_C):
        assert centers.ndim == 2 and centers.shape[1] == 2
        assert centers.min() >= -1.0 and centers.max() <= 1.0
    # neutral + the four full cardinals are present in both sets (canonical [-1,1] coords)
    for centers in (controller_codec.STICK_CLUSTER_CENTERS_MAIN, controller_codec.STICK_CLUSTER_CENTERS_C):
        for pt in ([0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]):
            assert (centers == torch.tensor(pt)).all(dim=1).any()


def test_pruned81_matches_the_notebook_center_order_and_coordinates() -> None:
    original = controller_codec.STICK_CLUSTER_CENTERS_MAIN.numpy()
    core = np.delete(original[:25], np.array((18, 20, 22, 24)), axis=0)
    middle = []
    for turn in range(4):
        for angle in (30, 45, 60):
            theta = np.deg2rad(turn * 90 + angle)
            middle.append((0.75 * np.cos(theta), 0.75 * np.sin(theta)))
    quadrant = np.linspace(np.deg2rad(17), np.deg2rad(73), 11)
    rim_angles = np.concatenate((np.deg2rad(np.arange(4) * 90), *(quadrant + turn * np.pi / 2 for turn in range(4))))
    rim = np.stack((np.cos(rim_angles), np.sin(rim_angles)), axis=-1).astype(np.float32)
    expected = np.concatenate((core, np.asarray(middle, dtype=np.float32), rim))

    actual = controller_codec.STICK_CLUSTER_CENTERS_MAIN_PRUNED81

    assert actual.shape == (81, 2)
    assert torch.equal(actual, torch.from_numpy(expected))
    assert torch.equal(actual[:21], torch.from_numpy(core))


def test_pruned81_keeps_partial_axes_and_full_rim_cardinals() -> None:
    centers = controller_codec.STICK_CLUSTER_CENTERS_MAIN_PRUNED81
    for radius in (0.35, 0.5, 0.675, 0.85):
        for point in ((radius, 0.0), (-radius, 0.0), (0.0, radius), (0.0, -radius)):
            assert (centers == torch.tensor(point)).all(dim=1).any()
    cardinals = torch.tensor(((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)))
    assert torch.allclose(centers[33:37], cardinals, atol=1e-7, rtol=0)
    assert torch.equal(centers[33:37].abs().amax(dim=1), torch.ones(4))
    assert not (centers.abs() == 79 / 80).any()
    for point in ((0.6, 0.6), (-0.6, 0.6), (-0.6, -0.6), (0.6, -0.6)):
        assert not (centers == torch.tensor(point)).all(dim=1).any()


def test_pruned81_codec_uses_an_81_class_main_stick_vocabulary() -> None:
    codec = DiscreteControllerCodec(embed_dim=8, main_stick_layout="pruned81")

    assert codec.main_stick_layout == "pruned81"
    assert codec.group_vocabs == (256, 81, 9, 25)
    assert codec.class_embeddings["main_stick"].num_embeddings == 81
    assert torch.equal(codec.main_centers, controller_codec.STICK_CLUSTER_CENTERS_MAIN_PRUNED81)

    indices = torch.zeros(81, 4, dtype=torch.long)
    indices[:, controller_codec.MAIN_STICK_GROUP] = torch.arange(81)
    assert torch.equal(codec.quantize(codec.dequantize(indices)), indices)


def test_codec_rejects_an_unknown_main_stick_layout() -> None:
    with pytest.raises(ValueError, match="unknown main-stick layout"):
        DiscreteControllerCodec(embed_dim=8, main_stick_layout="unknown")  # type: ignore[arg-type]


def test_c_stick_set_is_the_compact_nine_point_set():
    # c-stick is 95% neutral, the rest on the rim cardinals/diagonals: neutral + 4 cardinals
    # + 4 full diagonals, distinct from the larger main pose set.
    c = controller_codec.STICK_CLUSTER_CENTERS_C
    assert c.shape == (9, 2)
    for pt in ([0.7, 0.7], [-0.7, 0.7], [0.7, -0.7], [-0.7, -0.7]):
        assert (c == torch.tensor(pt)).all(dim=1).any()
    assert controller_codec.STICK_CLUSTER_CENTERS_MAIN.shape[0] > c.shape[0]


def test_stick_centers_are_unique():
    for centers in (
        controller_codec.STICK_CLUSTER_CENTERS_MAIN,
        controller_codec.STICK_CLUSTER_CENTERS_MAIN_PRUNED81,
        controller_codec.STICK_CLUSTER_CENTERS_C,
    ):
        assert torch.unique(centers, dim=0).shape[0] == centers.shape[0]


def test_nearest_cluster_is_identity_on_the_centers():
    for centers in (controller_codec.STICK_CLUSTER_CENTERS_MAIN, controller_codec.STICK_CLUSTER_CENTERS_C):
        idx = controller_codec.nearest_cluster(centers, centers)
        assert torch.equal(idx, torch.arange(centers.shape[0]))
        assert torch.equal(controller_codec.cluster_to_xy(idx, centers), centers)


def test_trigger_centers_are_sorted_in_unit_interval():
    t = controller_codec.TRIGGER_CENTERS
    assert t.ndim == 1 and t.shape[0] == 5
    assert t.min() >= 0.0 and t.max() <= 1.0
    assert torch.equal(t, t.sort().values)  # ascending, the 0/analog/1 spread
    assert t[0].item() == 0.0 and t[-1].item() == 1.0


def test_nearest_center_round_trips_the_centers():
    t = controller_codec.TRIGGER_CENTERS
    idx = controller_codec.nearest_center(t, t)
    assert torch.equal(idx, torch.arange(t.shape[0]))
    assert torch.equal(controller_codec.center_to_value(idx, t), t)


def test_nearest_center_snaps_to_closest_and_preserves_shape():
    t = controller_codec.TRIGGER_CENTERS  # (0.0, 0.35, 0.6, 0.85, 1.0)
    x = torch.tensor([[0.02, 0.99], [0.34, 0.62]])  # [..., 2] (per-shoulder), arbitrary batch dims
    idx = controller_codec.nearest_center(x, t)
    assert idx.shape == x.shape
    assert torch.equal(idx, torch.tensor([[0, 4], [1, 2]]))
    recon = controller_codec.center_to_value(idx, t)
    assert torch.equal(recon, torch.tensor([[0.0, 1.0], [0.35, 0.6]]))


def test_combo_round_trips_every_id():
    # pack(unpack(i)) == i for all 256, and unpack is the bit-decomposition of i.
    ids = torch.arange(controller_codec.N_BUTTON_COMBOS)
    bits = controller_codec.combo_to_buttons(ids)
    assert bits.shape == (256, 8)
    assert torch.isin(bits, torch.tensor([0.0, 1.0])).all()
    assert torch.equal(controller_codec.buttons_to_combo(bits), ids)


def test_combo_bit_k_is_button_channel_k():
    # bit k of the combo id must be button channel k (ACTION_CHANNELS / _BUTTON_ORDER order):
    # a single press on channel k yields combo id 2**k.
    for k in range(8):
        b = torch.zeros(8)
        b[k] = 1.0
        assert controller_codec.buttons_to_combo(b).item() == (1 << k)


def test_combo_handles_arbitrary_batch_dims():
    b = (torch.rand(3, 4, 8) > 0.5).float()
    combo = controller_codec.buttons_to_combo(b)
    assert combo.shape == (3, 4)
    assert torch.equal(controller_codec.combo_to_buttons(combo), b)
