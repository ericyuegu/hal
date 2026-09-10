"""Contracts for the O53 fixed polar controller experiment."""

import copy
import importlib.util
import math
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from hal.training.controller_codec import DiscreteControllerCodec as CartesianControllerCodec


def _load():
    path = Path(__file__).resolve().parents[2] / "experiments" / "053_polar_controller.py"
    name = "test_exp053"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exp = _load()


def _tiny_cfg(**changes):
    arch = {
        **asdict(exp.Architecture()),
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 4,
        "L_ctx": 8,
        "temporal_d_model": 32,
        "temporal_layers": 1,
        "temporal_heads": 4,
        "temporal_ff_dim": 64,
        "group_head_dim": 32,
        "value_hidden_dim": 16,
        "item_hidden_dim": 8,
        "item_dim": 5,
    }
    return exp.TrainConfig(
        arch=exp.Architecture(**arch),
        batch_size=2,
        compile_trunk=False,
        compile_temporal=False,
        inference_mode="eager",
        num_workers=0,
        push_to_r2=False,
        **changes,
    )


def _neutral_actions(rows: int) -> torch.Tensor:
    return torch.zeros(rows, exp.A_DIM)


def test_fixed_polar_class_counts_and_parameter_contract() -> None:
    codec = exp.DiscreteControllerCodec(32)

    assert exp.CONTROLLER_GROUP_VOCABS == (256, 85, 13, 25)
    assert codec.main_label_radii.shape == (85,)
    assert codec.c_label_radii.shape == (13,)
    assert exp.proxy_config().arch.parameter_count_contract["total"] == 14_490_994
    assert exp.proxy_config().adam_lr == 0.0017


def test_radius_boundaries_angle_wrapping_and_outer_clamping() -> None:
    codec = exp.DiscreteControllerCodec(8)
    actions = _neutral_actions(5)
    actions[0, 0] = 22 / 80
    actions[1, 0] = 23 / 80
    actions[2, 0] = -1.0
    actions[2, 1] = 0.0
    actions[3, 0] = -1.0
    actions[3, 1] = -0.0
    actions[4, 0:2] = 1.0

    labels = codec.quantize(actions)[:, exp.MAIN_STICK_GROUP]

    assert labels[0].item() == 0
    assert labels[1].item() == 3
    assert labels[2].item() == labels[3].item() == 21
    assert 21 <= labels[4].item() < 85
    assert codec.main_label_radii[labels[4]].item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("group", "vocab"),
    ((exp.MAIN_STICK_GROUP, 85), (exp.C_STICK_GROUP, 13)),
)
def test_every_polar_class_is_a_quantize_dequantize_fixed_point(group: int, vocab: int) -> None:
    codec = exp.DiscreteControllerCodec(8)
    indices = torch.zeros(vocab, exp.CONTROLLER_GROUP_COUNT, dtype=torch.long)
    indices[:, group] = torch.arange(vocab)

    reconstructed = codec.dequantize(indices)
    requantized = codec.quantize(reconstructed)

    torch.testing.assert_close(requantized[:, group], indices[:, group])


def test_polar_semantics_use_radius_sine_cosine_and_zero_origin() -> None:
    codec = exp.DiscreteControllerCodec(8)

    origin = codec.semantic_values("main_stick", torch.tensor(0))
    positive_x_inner = codec.semantic_values("main_stick", torch.tensor(3))
    positive_x_outer = codec.semantic_values("main_stick", torch.tensor(53))
    positive_y_c = codec.semantic_values("c_stick", torch.tensor(11))

    torch.testing.assert_close(origin, torch.tensor([0.0, 0.0, 0.0]))
    torch.testing.assert_close(positive_x_inner, torch.tensor([23 / 80, 0.0, 1.0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(positive_x_outer, torch.tensor([1.0, 0.0, 1.0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(positive_y_c, torch.tensor([1.0, 1.0, 0.0]), atol=1e-6, rtol=0)
    assert codec.semantic_projections["main_stick"].in_features == 3
    assert codec.semantic_projections["c_stick"].in_features == 3


def test_button_and_trigger_codec_is_unchanged() -> None:
    polar = exp.DiscreteControllerCodec(8)
    cartesian = CartesianControllerCodec(8)
    generator = torch.Generator().manual_seed(17)
    actions = torch.rand(256, exp.A_DIM, generator=generator)
    actions[:, :4] = 2.0 * actions[:, :4] - 1.0
    actions[:, 6:] = (actions[:, 6:] > 0.7).float()

    polar_indices = polar.quantize(actions)
    cartesian_indices = cartesian.quantize(actions)

    torch.testing.assert_close(polar_indices[:, exp.BUTTONS_GROUP], cartesian_indices[:, exp.BUTTONS_GROUP])
    torch.testing.assert_close(polar_indices[:, exp.TRIGGERS_GROUP], cartesian_indices[:, exp.TRIGGERS_GROUP])
    torch.testing.assert_close(polar.trigger_centers, cartesian.trigger_centers)
    torch.testing.assert_close(polar.button_valid_for_trigger, cartesian.button_valid_for_trigger)
    torch.testing.assert_close(
        polar.dequantize(polar_indices)[:, 4:],
        cartesian.dequantize(cartesian_indices)[:, 4:],
    )


def test_temporal_decoder_has_four_separate_group_heads() -> None:
    model = exp.GPT(_tiny_cfg())

    assert tuple(model.temporal.outputs) == exp.CONTROLLER_GROUP_NAMES
    assert tuple(model.temporal.trunk_outputs) == exp.CONTROLLER_GROUP_NAMES
    assert len({id(head) for head in model.temporal.outputs.values()}) == 4
    assert model.temporal.outputs["main_stick"].down.out_features == 85
    assert model.temporal.outputs["c_stick"].down.out_features == 13


def test_checkpoint_identity_rejects_o52() -> None:
    state = exp._checkpoint_config(exp.proxy_config())
    state["experiment_id"] = "052_adamw_temporal_awr_v1"

    with pytest.raises(ValueError, match="checkpoint experiment_id"):
        exp.config_from_state(state)


def test_checkpoint_and_run_tag_record_polar_treatment() -> None:
    cfg = exp.proxy_config()
    state = exp._checkpoint_config(cfg)
    tag = exp.model_tag(cfg)

    assert state["experiment_id"] == "053_polar_controller_v1"
    assert exp.config_from_state(state) == cfg
    assert "polar053" in tag
    assert "fixed-polar-sticks" in tag
    assert "alr0.0017" in tag
    assert cfg.arch.parameter_count_contract["total"] == 14_490_994


def test_validation_summary_keeps_separate_stick_mse() -> None:
    cfg = exp.proxy_config()
    values = {
        "loss_unweighted": 1.0,
        "temporal_loss_near_unweighted": 2.0,
        "temporal_loss_far_unweighted": 3.0,
        "exact_frame_acc": 0.1,
        "dense_four_sequence_acc": 0.2,
        "change_f1": 0.3,
        "sampled_transition_rate": 0.4,
        "main_stick_quantization_mse": 0.005,
        "c_stick_quantization_mse": 0.006,
    }
    for name in exp.CONTROLLER_GROUP_NAMES:
        values[f"rollout_nll_o04_{name}"] = 1.0
        values[f"exposure_gap_o04_{name}"] = 0.0

    summary = exp._validation_wandb_metrics(values, cfg)

    assert summary["main_stick_quantization_mse"] == 0.005
    assert summary["c_stick_quantization_mse"] == 0.006


def test_tiny_train_state_restores_the_next_update() -> None:
    cfg = _tiny_cfg(adam_lr=8.5e-4)
    torch.manual_seed(7)
    model = exp.GPT(cfg)
    optimizer = exp.make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, exp.lr_schedule(cfg))
    batch = exp.synthetic_awr_batch(cfg, torch.device("cpu"))

    exp.train_step(
        model,
        batch,
        cfg,
        step=0,
        update=1,
        valid_prefixes=cfg.batch_size * (cfg.arch.L_ctx - cfg.arch.direct_loss_start),
        trunk_fn=model.forward,
        temporal_fn=model.temporal.teacher_forced_nll_with_diagnostics,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    model_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    scheduler_state = copy.deepcopy(scheduler.state_dict())

    resumed = exp.GPT(cfg)
    resumed.load_state_dict(model_state)
    resumed_optimizer = exp.make_optimizer(resumed, cfg)
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, exp.lr_schedule(cfg))
    resumed_scheduler.load_state_dict(scheduler_state)

    for candidate_model, candidate_optimizer, candidate_scheduler in (
        (model, optimizer, scheduler),
        (resumed, resumed_optimizer, resumed_scheduler),
    ):
        exp.train_step(
            candidate_model,
            batch,
            cfg,
            step=1,
            update=2,
            valid_prefixes=cfg.batch_size * (cfg.arch.L_ctx - cfg.arch.direct_loss_start),
            trunk_fn=candidate_model.forward,
            temporal_fn=candidate_model.temporal.teacher_forced_nll_with_diagnostics,
            optimizer=candidate_optimizer,
            scheduler=candidate_scheduler,
        )

    for actual, expected in zip(model.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
    assert math.isfinite(float(resumed_optimizer.param_groups[0]["lr"]))
