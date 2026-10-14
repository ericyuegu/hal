import importlib.util
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

_SPEC = importlib.util.spec_from_file_location(
    "hal_experiment_062",
    Path(__file__).parents[1] / "experiments" / "062_production_bc_action_sequence.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _small_config(stick_features: str = "polar") -> Any:
    return replace(
        _MODULE.TrainConfig(),
        arch=replace(
            _MODULE.Architecture(),
            d_model=32,
            n_layers=1,
            n_heads=4,
            L_ctx=8,
            temporal_d_model=32,
            temporal_layers=1,
            temporal_heads=4,
            temporal_ff_dim=64,
            group_head_dim=16,
            value_hidden_dim=16,
        ),
        stick_features=stick_features,
        local_batch_size=2,
        microbatch_size=2,
        prediction_frames=3,
        player_vocab_size=2,
        compile_trunk=False,
        compile_temporal=False,
    )


@pytest.mark.parametrize(("arm", "batch", "ranks"), [("b512", 512, 2), ("b1024", 1024, 4), ("b2048", 2048, 8)])
def test_batch_arms_match_in_positions_without_accumulation(arm: str, batch: int, ranks: int) -> None:
    cfg = _MODULE.proxy_config_for_arm(arm)
    _MODULE.validate_config(cfg)
    assert (cfg.batch_size, cfg.world_size) == (batch, ranks)
    assert cfg.local_batch_size == cfg.microbatch_size == 256
    positions = cfg.supervised_positions_per_update
    assert cfg.max_steps * positions == 2**29
    assert cfg.warmup_steps * positions == cfg.val_every * positions == 2**26
    assert cfg.decay_start_update + cfg.decay_duration == cfg.max_steps
    assert cfg.decay_duration * 4 == cfg.max_steps
    assert cfg.stick_features == "polar" and not cfg.state_prediction


def test_stick_control_differs_only_by_stick_features() -> None:
    treatment = _MODULE.proxy_config_for_arm("b1024")
    control = _MODULE.proxy_config_for_arm("b1024_xy")
    _MODULE.validate_config(control)
    assert replace(control, stick_features="polar") == treatment


def test_production_state_head_and_off_arm_configs_are_rejected() -> None:
    cfg = _MODULE.proxy_config_for_arm("b1024")
    with pytest.raises(ValueError, match="production waits"):
        _MODULE.validate_config(_MODULE.TrainConfig())
    with pytest.raises(ValueError, match="omits the state head"):
        _MODULE.validate_config(replace(cfg, state_prediction=True))
    with pytest.raises(ValueError, match="arms are frozen"):
        _MODULE.validate_config(replace(cfg, warmup_updates=4096))
    with pytest.raises(ValueError, match="requires a proxy arm"):
        _MODULE.proxy_config_for_arm("none")


def test_polar_stick_features_are_radius_sine_cosine_and_zero_at_neutral() -> None:
    codec = _MODULE.PolarStickCodec(8, main_stick_layout="legacy65")
    plain = _MODULE.DiscreteControllerCodec(8, main_stick_layout="legacy65")
    for name, count in (("main_stick", 65), ("c_stick", 9)):
        indices = torch.arange(count)
        xy = plain.semantic_values(name, indices)
        values = codec.semantic_values(name, indices)
        radius = xy.norm(dim=-1)
        neutral = radius <= _MODULE.STICK_NEUTRAL_RADIUS
        assert int(neutral.sum()) == 1
        assert torch.equal(values[neutral], torch.zeros_like(values[neutral]))
        torch.testing.assert_close(values[~neutral, 0], radius[~neutral])
        torch.testing.assert_close(values[~neutral, 1] * radius[~neutral], xy[~neutral, 1])
        torch.testing.assert_close(values[~neutral, 2] * radius[~neutral], xy[~neutral, 0])
    for name in ("buttons", "triggers"):
        indices = torch.arange(4)
        assert torch.equal(codec.semantic_values(name, indices), plain.semantic_values(name, indices))


def test_stick_arms_share_initialization_except_stick_projections() -> None:
    torch.manual_seed(5)
    polar = _MODULE.make_model(_small_config("polar"))
    after_polar = torch.rand(1)
    torch.manual_seed(5)
    plain = _MODULE.make_model(_small_config("xy"))
    after_plain = torch.rand(1)
    assert torch.equal(after_polar, after_plain)
    assert polar.codec is polar.temporal.codec
    assert isinstance(polar.codec, _MODULE.PolarStickCodec)
    plain_parameters = dict(plain.named_parameters())
    for name, parameter in polar.named_parameters():
        expected = plain_parameters.pop(name)
        if name in ("codec.semantic_projections.main_stick.weight", "codec.semantic_projections.c_stick.weight"):
            assert parameter.shape == (*expected.shape[:-1], 3)
            continue
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    assert not plain_parameters


def test_proxy_parameter_contract_counts_the_polar_features() -> None:
    torch.manual_seed(0)
    polar = _MODULE.subsystem_parameter_counts(_MODULE.make_model(_MODULE.proxy_config_for_arm("b1024")))
    torch.manual_seed(0)
    plain = _MODULE.subsystem_parameter_counts(_MODULE.make_model(_MODULE.proxy_config_for_arm("b1024_xy")))
    assert polar["total"] - plain["total"] == 64
    assert polar["temporal_decoder"] - plain["temporal_decoder"] == 64
    assert polar["state_head"] == plain["state_head"] == 0


def test_policy_loss_is_unweighted_behavior_cloning_after_awr_would_start() -> None:
    cfg = _small_config()
    torch.manual_seed(1)
    model = _MODULE.make_model(cfg)
    batch = _MODULE.synthetic_awr_batch(cfg, torch.device("cpu"))
    batch = replace(batch, returns=torch.randn_like(batch.returns) * 500)
    trunk, temporal = _MODULE._training_functions(model, cfg)
    _, _, metrics = _MODULE.microbatch_loss(
        model,
        batch,
        cfg,
        step=10 * _MODULE.AWRCalibration.start_update,
        valid_prefixes=4,
        trunk_fn=trunk,
        temporal_fn=temporal,
        prefix_positions=torch.tensor([[4, 7], [5, 6]]),
    )
    weight_keys = [name for name in metrics if "weight" in name and ("mean" in name or "max" in name)]
    assert weight_keys
    for name in weight_keys:
        if name.endswith(("_mean", "_max")) and "correlation" not in name:
            assert math.isclose(float(metrics[name]), 1.0, rel_tol=0, abs_tol=1e-6), name
