"""The single-seed O52 BC ablation preserves initialization and supervision."""

import ast
import hashlib
import importlib.util
import json
import random
import sys
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from hal.inference.backends.temporal_awr.model import O50Config

ROOT = Path(__file__).resolve().parents[2]


def load_experiment(filename: str) -> ModuleType:
    path = ROOT / "experiments" / filename
    name = "o60_test_exp_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bc = load_experiment("060_plain_bc.py")
awr = load_experiment("052_adamw_temporal_awr.py")


def tiny_config(module: ModuleType):
    arch = replace(
        module.proxy_config().arch,
        d_model=32,
        n_layers=1,
        n_heads=4,
        L_ctx=8,
        temporal_d_model=32,
        temporal_layers=1,
        temporal_heads=4,
        temporal_ff_dim=64,
        group_head_dim=32,
        value_hidden_dim=16,
        item_hidden_dim=8,
        item_dim=5,
    )
    return replace(module.proxy_config(), arch=arch, batch_size=2, compile_trunk=False, compile_temporal=False)


def assert_batch_equal(left, right) -> None:
    assert left.batch.replay_ids == right.batch.replay_ids
    assert left.context.features.keys() == right.context.features.keys()
    for name in left.context.features:
        torch.testing.assert_close(left.context.features[name], right.context.features[name], rtol=0, atol=0)
    for actual, expected in (
        (left.context.ctx_pad, right.context.ctx_pad),
        (left.target, right.target),
        (left.returns, right.returns),
        (left.eligible, right.eligible),
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_control_source_and_resolved_configuration() -> None:
    source = (ROOT / "experiments/052_adamw_temporal_awr.py").read_bytes()
    assert hashlib.sha256(source).hexdigest() == "d0e75291dc3e0c322aea0cdd8b57573c5fec20b2c843b71943b8e3777f247b69"
    control = asdict(replace(awr.proxy_config(), adam_lr=0.0017))
    treatment = asdict(bc.proxy_config())
    control["awr"]["value_loss_weight"] = 0.0
    assert treatment == control
    assert bc.proxy_config().max_steps == 16_384
    assert bc.proxy_config().warmup_steps == 512
    assert not bc.proxy_config().automatic_evaluation
    state = bc._checkpoint_config(bc.proxy_config())
    assert state["experiment_id"] == "060_plain_bc_v1"
    assert bc.config_from_state(state) == bc.proxy_config()
    with pytest.raises(ValueError, match="experiment_id"):
        bc.config_from_state(awr._checkpoint_config(awr.proxy_config()))
    with pytest.raises(ValueError):
        bc.validate_config(replace(bc.proxy_config(), awr=bc.AWRCalibration(value_loss_weight=1.0)))


def test_full_initialization_is_bit_identical_including_allocated_critic() -> None:
    torch.manual_seed(0)
    control = awr.GPT(replace(awr.proxy_config(), adam_lr=0.0017))
    control_rng = torch.get_rng_state()
    torch.manual_seed(0)
    treatment = bc.GPT(bc.proxy_config())
    assert sum(p.numel() for p in treatment.parameters()) == 14_480_922
    assert control.state_dict().keys() == treatment.state_dict().keys()
    for name, expected in control.state_dict().items():
        torch.testing.assert_close(treatment.state_dict()[name], expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), control_rng)
    assert (
        bc.optimizer_roles(treatment, bc.proxy_config()).keys()
        == awr.optimizer_roles(control, awr.proxy_config()).keys()
    )


def test_batch_construction_and_loader_configuration_are_unchanged() -> None:
    names = {
        "_collate_o52_batch",
        "_make_train_loader",
        "_make_loaders",
        "cache_validation",
        "prepared_targets",
        "IdentityMasker",
        "DeviceBatchPrefetcher",
        "data_selection",
    }
    definitions = []
    for filename in ("052_adamw_temporal_awr.py", "060_plain_bc.py"):
        tree = ast.parse((ROOT / "experiments" / filename).read_text())
        definitions.append({node.name: ast.dump(node) for node in tree.body if getattr(node, "name", None) in names})
    assert definitions[0] == definitions[1]
    batches = []
    for module in (awr, bc):
        cfg = tiny_config(module)
        torch.manual_seed(cfg.seed)
        module.GPT(cfg)
        masker = module.IdentityMasker(cfg.seed ^ 0x0521D, cfg.identity_dropout)
        batches.append([masker(module.synthetic_awr_batch(cfg, torch.device("cpu"))) for _ in range(3)])
    for left, right in zip(*batches, strict=True):
        assert_batch_equal(left, right)


@pytest.mark.parametrize("step", [0, 4095, 4096, 4097, 16383])
def test_exact_bc_loss_and_no_critic_gradient_at_every_phase(step: int) -> None:
    cfg = tiny_config(bc)
    torch.manual_seed(0)
    model = bc.GPT(cfg)
    batch = bc.synthetic_awr_batch(cfg, torch.device("cpu"))
    batch.context.ctx_pad[0] = cfg.arch.L_ctx - 1
    batch.returns.fill_(1e20)
    batch.eligible[0] = False
    history, targets, valid = bc.prepared_targets(model, batch)
    prefixes = int(valid.sum())
    hidden = model.forward(batch.context.features, batch.context.ctx_pad, None)[:, cfg.arch.direct_loss_start :]
    nll = model.temporal.teacher_forced_nll(hidden, history, targets).float()
    joint = nll.sum(-1).masked_fill(~valid[..., None], 0)
    expected = (joint[..., :6].sum() / (prefixes * 6) + 0.5 * joint[..., 6:].sum() / (prefixes * 4)) / 1.5
    loss, _, metrics = bc.microbatch_loss(
        model,
        batch,
        cfg,
        step=step,
        valid_prefixes=prefixes,
        trunk_fn=model.forward,
        temporal_fn=model.temporal.teacher_forced_nll,
    )
    torch.testing.assert_close(loss, expected, rtol=1e-7, atol=1e-6)
    assert metrics["awr/active"].item() == 0
    assert metrics["awr/weight_mean"].item() == metrics["awr/weight_max"].item() == 1
    loss.backward()
    assert all(parameter.grad is None for parameter in model.value_head.parameters())
    assert any(parameter.grad is not None for parameter in model.trunk.parameters())
    before = {name: p.detach().clone() for name, p in model.value_head.named_parameters()}
    optimizer = bc.make_optimizer(model, cfg)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    for name, parameter in model.value_head.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_checkpoint_restores_rng_next_batch_and_next_optimizer_update(tmp_path: Path) -> None:
    cfg = tiny_config(bc)
    torch.manual_seed(0)
    model = bc.GPT(cfg)
    optimizer = bc.make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, bc.lr_schedule(cfg))
    masker = bc.IdentityMasker(cfg.seed ^ 0x0521D, cfg.identity_dropout)

    def update(candidate, opt, sched, batch, step: int) -> None:
        bc.train_step(
            candidate,
            batch,
            cfg,
            step=step,
            update=step + 1,
            valid_prefixes=cfg.batch_size * (cfg.arch.L_ctx - cfg.arch.direct_loss_start),
            trunk_fn=candidate.forward,
            temporal_fn=candidate.temporal.teacher_forced_nll,
            optimizer=opt,
            scheduler=sched,
        )

    update(model, optimizer, scheduler, masker(bc.synthetic_awr_batch(cfg, torch.device("cpu"))), 0)
    snapshot = bc.save_boundary_checkpoint(
        tmp_path,
        update=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        uploader=None,
        milestone=False,
        wandb_id=None,
        actual_loss_positions=8,
        loader_state={"cursor": 1},
        identity_masker_state=masker.state_dict(),
    )
    expected_random = (random.random(), np.random.random())
    expected_batch = masker(bc.synthetic_awr_batch(cfg, torch.device("cpu")))
    update(model, optimizer, scheduler, expected_batch, 1)
    saved = torch.load(snapshot, weights_only=False)
    restored = bc.GPT(cfg)
    restored.load_state_dict(saved["model"])
    restored_opt = bc.make_optimizer(restored, cfg)
    restored_sched = torch.optim.lr_scheduler.LambdaLR(restored_opt, bc.lr_schedule(cfg))
    restored_opt.load_state_dict(saved["opt"])
    restored_sched.load_state_dict(saved["sched"])
    restored_masker = bc.IdentityMasker(cfg.seed ^ 0x0521D, cfg.identity_dropout)
    restored_masker.load_state_dict(saved["identity_masker"])
    bc.restore_rng_state(saved["rng"])
    assert (random.random(), np.random.random()) == expected_random
    actual_batch = restored_masker(bc.synthetic_awr_batch(cfg, torch.device("cpu")))
    assert_batch_equal(actual_batch, expected_batch)
    update(restored, restored_opt, restored_sched, actual_batch, 1)
    for actual, expected in zip(restored.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert restored_sched.state_dict() == scheduler.state_dict()
    for parameter, expected_parameter in zip(restored.parameters(), model.parameters(), strict=True):
        for name, value in optimizer.state[expected_parameter].items():
            torch.testing.assert_close(restored_opt.state[parameter][name], value, rtol=0, atol=0)


def test_portable_runtime_accepts_distinct_bc_identity() -> None:
    cfg = bc.proxy_config()
    state = bc._checkpoint_config(cfg)
    portable = O50Config.from_checkpoint(state, player_code_bytes=1, stats_sha256="a" * 64)
    assert portable.experiment_id == "060_plain_bc_v1"
    assert portable.amp_dtype == "bfloat16"


def test_production_rejects_retuning() -> None:
    bc.validate_control_invariants(bc.proxy_config())
    with pytest.raises(ValueError, match="invariants"):
        bc.validate_control_invariants(replace(bc.proxy_config(), seed=1))
    with pytest.raises(ValueError, match="invariants"):
        bc.validate_control_invariants(replace(bc.proxy_config(), adam_lr=0.001))


def test_evaluation_protocol_matches_persisted_corrected_control() -> None:
    values = json.loads((ROOT / "experiments/o60/control-match-rows.json").read_text())["protocol"]
    protocol = bc.EvalProtocol(**values)
    bc.validate_control_protocol(protocol)
    for field, value in (("seed", 1), ("inference_compile_mode", "default"), ("max_frames", 100)):
        with pytest.raises(ValueError, match="protocol changed"):
            bc.validate_control_protocol(replace(protocol, **{field: value}))


def test_paired_bootstrap_preserves_pairs_and_difference_direction() -> None:
    comparison = load_experiment("o60/compare.py")
    minutes = np.linspace(1, 2, 96)
    control = np.column_stack((3 * minutes, 100 * minutes, minutes))
    treatment = np.column_stack((2 * minutes, 80 * minutes, minutes))
    result = comparison.paired_comparison(control, treatment)
    assert result["seed"] == 60
    assert result["resamples"] == 2000
    for name, expected in (("net_stocks_per_min", 1), ("net_damage_per_min", 20)):
        assert result["metrics"][name]["difference"] == pytest.approx(expected)
        assert result["metrics"][name]["ci95"] == pytest.approx([expected, expected])
    identical = comparison.paired_comparison(control, control)
    assert identical["metrics"]["net_stocks_per_min"]["ci95"] == [0, 0]


def test_comparison_rejects_incomplete_or_changed_evaluation(tmp_path: Path) -> None:
    comparison = load_experiment("o60/compare.py")
    rows = json.loads((ROOT / "experiments/o60/control-match-rows.json").read_text())
    metrics = json.loads((ROOT / "experiments/o60/control-metrics.json").read_text())
    (tmp_path / "match_rows.json").write_text(json.dumps(rows))
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    _, boots = comparison.load_boots(tmp_path)
    assert boots.shape == (96, 3)
    metrics["completed_boots"] = 95
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="incomplete"):
        comparison.load_boots(tmp_path)
    metrics["completed_boots"] = 96
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    rows["protocol"]["transport_semantics"] = "truncation"
    (tmp_path / "match_rows.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="protocol differs"):
        comparison.load_boots(tmp_path)
