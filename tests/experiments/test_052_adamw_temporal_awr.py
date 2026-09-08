"""Contracts for the O52 all-AdamW sweep."""

import copy
import importlib.util
import sys
from dataclasses import asdict
from dataclasses import fields
from pathlib import Path

import pytest
import torch


def _load():
    path = Path(__file__).resolve().parents[2] / "experiments" / "052_adamw_temporal_awr.py"
    name = "test_exp052"
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


def test_proxy_and_rank1_data_identity_are_fixed() -> None:
    cfg = exp.TrainConfig()
    proxy = exp.proxy_config()

    assert proxy.arch.parameter_count_contract["total"] == 14_480_922
    assert proxy.target_positions == 2**30
    assert proxy.max_steps == 16_384
    assert cfg.source_names == ("ranked-anonymized-1-policy-world-v8",)
    assert cfg.train_replays == 112_188
    assert cfg.train_frames == 1_203_888_017
    assert cfg.replay_slots == 65_536
    assert cfg.replay_slots <= cfg.train_replays
    assert exp.data_selection(cfg).sha256 == cfg.selection_sha256
    assert exp.source_manifest_sha256(cfg) == {
        "ranked-anonymized-1-policy-world-v8": "b97eab90e761bcf2bf03b48981f0ab6acc1ac3057157c58ae0c5a72c76c43bd8"
    }


def test_config_rejects_any_other_source() -> None:
    cfg = exp.TrainConfig(source_names=("ranked-anonymized-2-policy-world-v8",))

    with pytest.raises(ValueError, match="requires only ranked-anonymized-1"):
        exp.validate_config(cfg)


def test_every_parameter_uses_adamw_exactly_once() -> None:
    cfg = _tiny_cfg()
    model = exp.GPT(cfg)
    roles = exp.optimizer_roles(model, cfg)
    optimizer = exp.make_optimizer(model, cfg)
    memberships = [parameter for group in optimizer.param_groups for parameter in group["params"]]

    assert set(roles) == dict(model.named_parameters()).keys()
    assert len(memberships) == len({id(parameter) for parameter in memberships})
    assert {id(parameter) for parameter in memberships} == {id(parameter) for parameter in model.parameters()}
    assert isinstance(optimizer, torch.optim.AdamW)
    assert all("momentum" not in group for group in optimizer.param_groups)
    assert roles["trunk.blocks.0.attn.c_attn.weight"].lr_kind == "hidden"
    assert roles["temporal.blocks.0.qkv.weight"].lr_kind == "hidden"
    assert roles["value_head.up.weight"].lr_kind == "hidden"


def test_adamw_state_restores_the_next_update() -> None:
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
    hidden_state = resumed_optimizer.state[resumed.trunk.blocks[0].attn.c_attn.weight]
    assert set(hidden_state) == {"exp_avg", "exp_avg_sq", "step"}


def test_checkpoint_and_run_names_record_the_treatment() -> None:
    cfg = exp.proxy_config()
    state = exp._checkpoint_config(cfg)
    tag = exp.model_tag(cfg)

    assert state["experiment_id"] == "052_adamw_temporal_awr_v1"
    assert exp.config_from_state(state) == cfg
    assert "all-adamw" in tag
    assert "alr0.000425" in tag
    assert "muon" not in {field.name for field in fields(exp.TrainConfig)}
    assert cfg.automatic_evaluation is False


def test_proxy_smoke_uses_the_sweep_model(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = {}
    monkeypatch.setattr(exp, "load_stats", lambda _cfg: {})
    monkeypatch.setattr(exp, "train", lambda cfg, _stats, **kwargs: observed.update(cfg=cfg, kwargs=kwargs))

    exp.main(exp.TrainArgs(proxy=True, smoke=True, stop_after_update=100))

    assert observed["cfg"] == exp.proxy_config()
    assert observed["kwargs"]["proxy"] is True
    assert observed["kwargs"]["smoke"] is True
    assert observed["kwargs"]["stop_after_update"] == 100
