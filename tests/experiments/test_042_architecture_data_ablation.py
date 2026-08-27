"""Contracts for the experiment 042 architecture and data ablations."""

import importlib.util
import math
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import torch


def _load(name: str, filename: str) -> ModuleType:
    """Load a numeric experiment module by path."""
    path = Path(__file__).resolve().parents[2] / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp = _load("test_exp042", "042_architecture_data_ablation.py")


def _small_cfg(**overrides) -> exp.TrainConfig:
    """Return a CPU-sized configuration with production semantics."""
    architecture_values = {
        **asdict(exp.ARCHITECTURE),
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 4,
        "L_ctx": 4,
        "temporal_d_model": 32,
        "temporal_layers": 1,
        "temporal_heads": 4,
        "temporal_ff_dim": 64,
        "group_head_dim": 64,
    }
    config_values = {
        "batch_size": 2,
        "target_loss_positions": 16,
        "warmup_steps": 1,
        "compile_trunk": False,
        "compile_temporal": False,
        "num_workers": 0,
        "push_to_r2": False,
        "inference_mode": "eager",
        "wandb_log_code": False,
    }
    for name, value in overrides.items():
        if name in architecture_values:
            architecture_values[name] = value
        else:
            config_values[name] = value
    return exp.TrainConfig(
        arch=exp.Architecture(**architecture_values),
        **config_values,
    )


def test_production_geometry_budget_and_cadences_match_026() -> None:
    cfg = exp.TrainConfig()

    assert asdict(cfg.arch) == {
        "d_model": 384,
        "n_layers": 8,
        "n_heads": 6,
        "attn_window": 0,
        "L_ctx": 128,
        "sample_chunk_length": 20,
        "head_offsets": (1, 2, 3, 4, 5, 6, 9, 12, 16, 20),
        "temporal_d_model": 128,
        "temporal_layers": 2,
        "temporal_heads": 4,
        "temporal_ff_dim": 256,
        "group_head_dim": 256,
        "action_embed_dim": 16,
        "offset_embed_dim": 16,
        "action_vocab": 1024,
        "action_state_embed_dim": 48,
        "char_vocab": 32,
        "char_dim": 8,
        "stage_vocab": 32,
        "stage_dim": 4,
    }
    assert cfg.max_steps == 16_384
    assert cfg.batch_size == 512
    assert cfg.warmup_steps == 500
    assert (cfg.val_every, cfg.ckpt_every, cfg.eval_every) == (1024, 1024, 4096)
    assert (cfg.eval_n_matchups, cfg.final_eval_n_matchups) == (32, 96)
    assert cfg.final_diagnostic_n_matchups == 32
    assert cfg.eval_max_parallel == 32


def test_treatments_change_only_the_training_corpus() -> None:
    arch_only = exp.TrainConfig(treatment="arch-only")
    full_mix = exp.TrainConfig(treatment="full-mix")

    differing = {name for name in asdict(arch_only) if getattr(arch_only, name) != getattr(full_mix, name)}
    assert differing == {"treatment"}
    assert arch_only.source_names == ("ranked-anonymized-1-policy-v7",)
    assert arch_only.replay_format == "policy"
    assert arch_only.cache_limit_gb == 160
    assert len(full_mix.source_names) == 44
    assert len(set(full_mix.source_names)) == 44
    assert full_mix.source_names.count("professional-zain-policy-world-v7") == 1
    assert full_mix.replay_format == "policy-world"
    assert full_mix.cache_limit_gb == 1792


def test_loader_keeps_base_observations_and_native_repeat_one_mix() -> None:
    stats = {}
    arch_kwargs = exp.loader_kwargs(exp.TrainConfig(), stats)
    mix_kwargs = exp.loader_kwargs(exp.TrainConfig(treatment="full-mix"), stats)

    assert arch_kwargs["extra"] is None
    assert arch_kwargs["projection"] is exp.BASE_ACTION_PROJECTION
    assert len(arch_kwargs["sources"]) == 1
    assert len(mix_kwargs["sources"]) == 44
    assert "source_weights" not in mix_kwargs


def test_model_is_041_architecture_without_projectiles_or_critic() -> None:
    model = exp.GPT(_small_cfg())

    assert isinstance(model.observation_encoder, exp.SwiGLU)
    assert not hasattr(model, "item_encoder")
    assert not hasattr(model, "value_head")
    assert not hasattr(model.temporal, "trunk_outputs")
    assert all(isinstance(head, exp.LinearActionHead) for head in model.temporal.outputs.values())


def test_objective_is_026_primary_four_plus_auxiliary_mean() -> None:
    nll = torch.arange(1, 11, dtype=torch.float32).view(1, 1, 10, 1)
    nll = nll.expand(1, 1, 10, exp.N_GROUPS)
    valid = torch.ones(1, 1, dtype=torch.bool)

    primary, auxiliary, total = exp.temporal_objective_parts(
        nll,
        valid_prefixes=1,
        valid=valid,
    )

    expected_primary = 4 * torch.arange(1, 5, dtype=torch.float32).mean()
    expected_auxiliary = 4 * torch.arange(5, 11, dtype=torch.float32).mean()
    torch.testing.assert_close(primary, expected_primary)
    torch.testing.assert_close(auxiliary, expected_auxiliary)
    torch.testing.assert_close(total, expected_primary + expected_auxiliary)


def test_optimizer_matches_026_partition_and_hyperparameters() -> None:
    cfg = _small_cfg()
    model = exp.GPT(cfg)
    optimizer = exp.make_optimizer(model, cfg)
    muon_group = next(group for group in optimizer.param_groups if group["use_muon"])
    adam_groups = [group for group in optimizer.param_groups if not group["use_muon"]]
    muon_ids = {id(parameter) for parameter in muon_group["params"]}

    assert muon_group["lr"] == 0.02
    assert muon_group["weight_decay"] == 0.01
    assert all(group["lr"] == 8.5e-4 for group in adam_groups)
    assert {group["weight_decay"] for group in adam_groups} == {0.0, 0.01}
    assert not muon_ids & {id(parameter) for parameter in model.temporal.parameters()}
    assert muon_ids == {id(parameter) for parameter in model.trunk.blocks.parameters() if parameter.ndim >= 2}


def test_cosine_schedule_matches_026_warmup_and_floor() -> None:
    cfg = exp.TrainConfig()
    schedule = exp.lr_schedule(cfg)

    assert schedule(0) == 0.0
    assert math.isclose(schedule(cfg.warmup_steps), 1.0)
    assert math.isclose(schedule(cfg.max_steps), cfg.lr_floor_ratio)


def test_checkpoint_config_round_trips_each_treatment() -> None:
    for treatment in ("arch-only", "full-mix"):
        cfg = exp.TrainConfig(treatment=treatment)
        restored = exp.config_from_state(exp._checkpoint_config(cfg))

        assert restored == cfg
        assert restored.source_names == cfg.source_names
