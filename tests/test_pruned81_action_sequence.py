"""Scientific contracts for experiment 061."""

import importlib.util
import sys
from dataclasses import asdict
from dataclasses import fields
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _load_experiment(name: str) -> ModuleType:
    path = Path(__file__).parents[1] / "experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"hal_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_O60 = _load_experiment("060_compute_optimal_action_sequence")
_O61 = _load_experiment("061_pruned81_action_sequence")


def test_pruned81_preserves_action_only_config_apart_from_controller_layout() -> None:
    control = _O60.TrainConfig()
    treatment = _O61.TrainConfig()

    control_architecture = {item.name: getattr(control.arch, item.name) for item in fields(control.arch)}
    treatment_architecture = {item.name: getattr(treatment.arch, item.name) for item in fields(treatment.arch)}
    changed = {
        name: (control_architecture[name], treatment_architecture[name])
        for name in control_architecture
        if control_architecture[name] != treatment_architecture[name]
    }

    assert changed == {"main_stick_layout": ("legacy65", "pruned81")}
    control_values = asdict(control)
    treatment_values = asdict(treatment)
    # 061 retains the action-only objective and short-delay protocol from before
    # 060 added future states and moved evaluation to a 21-frame delay.
    assert control_values.pop("state_prediction") is True
    assert control_values.pop("state_loss_weight") == 1.0
    assert (control_values.pop("prediction_frames"), treatment_values.pop("prediction_frames")) == (23, 4)
    assert (control_values.pop("delay_frames"), treatment_values.pop("delay_frames")) == (21, 2)
    treatment_values["arch"]["main_stick_layout"] = "legacy65"
    assert control_values == treatment_values


def test_pruned81_production_parameter_and_compute_contracts() -> None:
    cfg = _O61.TrainConfig()
    _O61.validate_config(cfg)
    with torch.device("meta"):
        model = _O61.make_model(cfg)
    counts = _O61.subsystem_parameter_counts(model)

    assert model.codec.group_vocabs == (256, 81, 9, 25)
    assert counts["return_conditioner"] == 0
    assert "state_head" not in counts
    assert counts["total"] == 193_393_341
    assert _O61.compute_equivalent_parameter_count(cfg, counts) == 3_953_839_361
    assert cfg.max_steps == 714_752
    assert cfg.supervised_positions_per_update == 16_384
    assert cfg.target_positions == 11_710_496_768


def test_pruned81_proxy_keeps_the_depth_treatment() -> None:
    # 060's proxy moved to a 50M single-GPU state-head ablation; 061 keeps the 15M depth treatment.
    treatment = _O61.proxy_config()
    assert (treatment.arch.n_layers, treatment.arch.temporal_layers) == (6, 8)
    assert treatment.arch.head_offsets == tuple(range(1, 31))
    assert not treatment.return_conditioning
    assert treatment.target_positions == 2**30
    with torch.device("meta"):
        model = _O61.make_model(treatment)
    counts = _O61.subsystem_parameter_counts(model)
    assert counts["total"] == 14_702_397
    assert _O61.compute_equivalent_parameter_count(treatment, counts) == 306_237_953


def test_pruned81_checkpoint_rejects_the_legacy_layout() -> None:
    values = _O61._checkpoint_config(_O61.TrainConfig())
    values["architecture"]["main_stick_layout"] = "legacy65"

    with pytest.raises(ValueError, match="parameter contract"):
        _O61.config_from_state(values)
