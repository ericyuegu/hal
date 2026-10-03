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


def test_pruned81_is_the_only_production_treatment_change() -> None:
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
    assert asdict(control.awr) == asdict(treatment.awr)
    assert control.source_names == treatment.source_names
    assert control.target_positions == treatment.target_positions


def test_pruned81_production_parameter_and_compute_contracts() -> None:
    cfg = _O61.TrainConfig()
    _O61.validate_config(cfg)
    with torch.device("meta"):
        model = _O61.make_model(cfg)
    counts = _O61.subsystem_parameter_counts(model)

    assert model.codec.group_vocabs == (256, 81, 9, 25)
    assert counts["total"] == 121_941_261
    assert _O61.compute_equivalent_parameter_count(cfg, counts) == 2_424_485_409
    assert cfg.max_steps == 497_664
    assert cfg.supervised_positions_per_update == 16_384
    assert cfg.target_positions == 8_153_726_976


def test_pruned81_proxy_matches_the_depth_treatment() -> None:
    control = _O60.proxy_config()
    treatment = _O61.proxy_config()
    assert (control.arch.n_layers, control.arch.temporal_layers) == (6, 8)
    assert (treatment.arch.n_layers, treatment.arch.temporal_layers) == (6, 8)
    assert control.target_positions == treatment.target_positions == 2**30
    with torch.device("meta"):
        model = _O61.make_model(treatment)
    counts = _O61.subsystem_parameter_counts(model)
    assert counts["total"] == 14_186_557
    assert _O61.compute_equivalent_parameter_count(treatment, counts) == 260_109_057


def test_pruned81_checkpoint_rejects_the_legacy_layout() -> None:
    values = _O61._checkpoint_config(_O61.TrainConfig())
    values["architecture"]["main_stick_layout"] = "legacy65"

    with pytest.raises(ValueError, match="parameter contract"):
        _O61.config_from_state(values)
