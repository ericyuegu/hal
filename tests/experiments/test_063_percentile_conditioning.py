"""Scientific contracts for the three percentile treatments."""

import copy
import functools
import importlib.util
import json
import sys
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any
from typing import cast

import numpy as np
import pytest
import torch

import hal.training.buffered_mds_replay_loader as replay_loader
from hal.training import returns as returns_lib


def _load(name: str) -> ModuleType:
    path = Path(__file__).parents[2] / "experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"hal_contract_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


control = _load("059_muon_action_sequence")
exp = _load("063_percentile_conditioning")
ARMS = ("return60", "bootstrap30", "match")


def _cfg(arm="return60", *, length=128, batch_size=2):
    cfg = exp.proxy_config(arm)
    arch = replace(
        cfg.arch,
        d_model=32,
        n_layers=1,
        n_heads=2,
        L_ctx=length,
        temporal_d_model=32,
        temporal_layers=2,
        temporal_heads=2,
        temporal_ff_dim=128,
        group_head_dim=32,
        value_hidden_dim=16,
        item_hidden_dim=8,
        item_dim=5,
    )
    return replace(
        cfg,
        arch=arch,
        batch_size=batch_size,
        amp_dtype="float32",
        compile_trunk=False,
        compile_temporal=False,
        inference_mode="eager",
        num_workers=0,
        push_to_r2=False,
    )


def _batch(cfg, device="cpu"):
    batch = exp.synthetic_awr_batch(cfg, torch.device(device))
    values = torch.linspace(-20, 50, cfg.arch.L_ctx, device=device).expand(cfg.batch_size, -1).clone()
    return replace(
        batch,
        batch=replace(batch.batch, replay_ids=tuple(f"replay-{i}" for i in range(cfg.batch_size))),
        returns=values,
        future_return=values / 2,
    )


def _control_config(cfg):
    values = asdict(cfg)
    values.pop("arm")
    values["arch"] = control.Architecture(**values["arch"])
    values["awr"] = control.AWRCalibration(**values["awr"])
    return control.TrainConfig(**values)


def _control_batch(batch):
    return control.ReturnBatch(
        batch.batch, batch.returns, batch.eligible, batch.future_return, batch.available, batch.condition_present
    )


@pytest.mark.parametrize("arm", ARMS)
def test_config_matches_059_apart_from_percentile_fields(arm):
    values = asdict(exp.TrainConfig(arm=arm))
    assert values.pop("arm") == arm
    assert values == asdict(control.TrainConfig())
    values = asdict(exp.proxy_config(arm))
    values.pop("arm")
    assert values == asdict(control.proxy_config())
    assert arm in exp.model_tag(exp.proxy_config(arm))
    exp.validate_config(exp.proxy_config(arm))
    with pytest.raises(TypeError):
        exp.TrainConfig()


@pytest.mark.parametrize("module,arm", [(control, None), *[(exp, arm) for arm in ARMS]])
def test_ranked_subset_has_pinned_selection_and_rejects_other_subsets(module, arm):
    cfg = module.proxy_config() if arm is None else module.proxy_config(arm)
    full = module.data_selection(cfg)
    assert full.sha256 == cfg.selection_sha256
    ranked = replace(cfg, source_names=("ranked-anonymized-1-policy-world-v8",))
    module.validate_config(ranked)
    selection = module.data_selection(ranked)
    assert selection.sha256 == ranked.selection_sha256
    assert selection.row_count == ranked.train_replays
    assert full.row_count >= cfg.replay_slots == 131_072
    assert cfg.replay_phase_block_batches == 25
    assert ranked.replay_slots == 112_128 <= selection.row_count
    assert ranked.replay_slots % ranked.batch_size == 0
    assert (
        ranked.replay_slots // ranked.batch_size - ranked.replay_phase_block_batches + 1
        >= ranked.minimum_replay_gap_batches
    )
    with pytest.raises(ValueError, match="sources|selection"):
        module.validate_config(replace(cfg, source_names=cfg.source_names[1:]))
    with pytest.raises(ValueError, match="selection"):
        module.data_selection(replace(cfg, source_names=cfg.source_names[1:]))


@pytest.mark.parametrize("target", [0.0, -1000.0, 1000.0])
def test_hl_gauss_mass_and_peak(target):
    values = torch.tensor([[target]])
    mass = exp.hl_gauss_target(values)
    torch.testing.assert_close(mass.sum(-1), torch.ones(1, 1))
    index = torch.bucketize(exp.symlog(values), exp.EDGES[1:-1])
    assert torch.equal(mass.argmax(-1), index)
    assert mass.min() >= 0
    assert mass.shape == (1, 1, 65)


def test_hl_gauss_clamps_and_ignores_unavailable_nan():
    assert torch.equal(exp.hl_gauss_target(torch.tensor([1e5])), exp.hl_gauss_target(torch.tensor([1000.0])))
    logits = torch.zeros(1, 2, 65, requires_grad=True)
    loss = exp.hl_gauss_loss(logits, torch.tensor([[float("nan"), 0.0]]), torch.tensor([[False, True]]))
    torch.testing.assert_close(loss, torch.tensor(65.0).log())
    loss.backward()
    assert torch.equal(logits.grad[:, 0], torch.zeros(1, 65))
    empty = exp.hl_gauss_loss(logits, torch.full((1, 2), float("nan")), torch.zeros(1, 2, dtype=torch.bool))
    assert empty == 0 and torch.isfinite(empty)


@pytest.mark.parametrize("proxy,total,head", [(False, 247944126, 1081921), (True, 17133246, 73921)])
def test_parameter_optimizer_and_flops_contracts(proxy, total, head):
    cfg = exp.proxy_config("return60") if proxy else exp.TrainConfig(arm="return60")
    with torch.device("meta"):
        model = exp.make_model(cfg)
    counts = exp.subsystem_parameter_counts(model)
    assert counts["total"] == total
    assert counts["percentile_head"] == head
    baseline = _control_config(cfg)
    with torch.device("meta"):
        baseline_model = control.make_model(baseline)
    baseline_counts = control.subsystem_parameter_counts(baseline_model)
    assert exp.approximate_training_flops_per_update(cfg, counts) == (
        control.approximate_training_flops_per_update(baseline, baseline_counts) + 6 * 512 * 128 * head
    )
    roles = exp.optimizer_roles(model, cfg)
    assert roles.keys() == dict(model.named_parameters()).keys()
    for name in ("up.weight", "down.weight", "down.bias"):
        assert roles[f"percentile_head.{name}"] == roles[f"value_head.{name}"]
    small = exp.make_model(_cfg())
    optimizer = exp.make_optimizer(small, _cfg())
    members = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(members) == len({id(p) for p in members}) == len(list(small.parameters()))


def test_head_is_detached_and_preserves_control_parameters_and_rng():
    cfg = _cfg()
    torch.manual_seed(19)
    baseline = control.make_model(_control_config(cfg))
    rng = torch.get_rng_state().clone()
    torch.manual_seed(19)
    model = exp.make_model(cfg)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, model.state_dict()[name]), name
    batch = _batch(cfg)
    hidden = model(batch.context.features, batch.context.ctx_pad, None)
    features = exp.decoder_rmsnorm(hidden[:, cfg.arch.direct_loss_start :]).detach()
    loss = exp.hl_gauss_loss(model.percentile_head(features), batch.future_return[:, 64:], batch.available[:, 64:])
    loss.backward()
    assert all(
        p.grad is None or torch.count_nonzero(p.grad) == 0
        for n, p in model.named_parameters()
        if not n.startswith("percentile_head.")
    )
    assert all(p.grad is not None for p in model.percentile_head.parameters())


def test_bootstrap_recovers_true_return_and_availability():
    cfg = _cfg("bootstrap30", length=256)
    batch = _batch(cfg)
    target, available = exp.percentile_target("bootstrap30", batch, batch.returns[:, 128:])
    assert torch.equal(target[:, :-30], batch.returns[:, 128:-30])
    assert available[:, :-30].all() and not available[:, -30:].any()
    batch.eligible[:, 200] = False
    _, available = exp.percentile_target("bootstrap30", batch, batch.returns[:, 128:])
    assert not available[:, 170 - 128].any()
    assert not available[:, 200 - 128].any()


def test_bootstrap_zero_value_is_exact_thirty_frame_reward():
    cfg = _cfg("bootstrap30", length=256)
    replay = {
        "p1_stock": np.full(300, 4),
        "p2_stock": np.full(300, 4),
        "p1_percent": np.zeros(300),
        "p2_percent": np.zeros(300),
    }
    replay["p2_percent"][140:160] = np.arange(20) * 3
    replay["p2_stock"][160:] = 3
    replay["p1_percent"][178:190] = np.arange(12) * 2
    replay["p1_stock"][190:] = 3
    reward = returns_lib.frame_reward(
        replay, ego="p1", opp="p2", damage_shaping=1.0, win_reward=50.0, stock_value=120.0
    )
    full = returns_lib.discounted_return(reward, cfg.awr.gamma)
    batch = replace(_batch(cfg), returns=torch.from_numpy(full[1:257]).expand(2, -1))
    target, available = exp.percentile_target("bootstrap30", batch, torch.zeros(2, 128))
    expected = torch.tensor(
        [sum(cfg.awr.gamma**k * float(reward[t + 1 + k]) for k in range(30)) for t in range(128, 226)]
    )
    torch.testing.assert_close(target[:, :-30], expected.expand(2, -1), atol=5e-5, rtol=1e-5)
    assert available[:, :-30].all()


@pytest.mark.parametrize(
    "arm,target_name,mask_name", [("return60", "future_return", "available"), ("match", "returns", "eligible")]
)
def test_targets_use_their_own_columns_and_masks(arm, target_name, mask_name):
    cfg = _cfg(arm)
    batch = _batch(cfg)
    batch.available[:, 70] = False
    batch.eligible[:, 80] = False
    target, available = exp.percentile_target(arm, batch, batch.returns[:, 64:])
    assert torch.equal(target, getattr(batch, target_name)[:, 64:])
    assert torch.equal(available, getattr(batch, mask_name)[:, 64:])


def test_labels_are_uniform_for_a_known_distribution():
    generator = torch.Generator().manual_seed(19)
    probabilities = torch.arange(1, 66).float().square()
    probabilities /= probabilities.sum()
    k = torch.multinomial(probabilities, 100_000, replacement=True, generator=generator)
    centers = (exp.EDGES[1:] + exp.EDGES[:-1]) / 2
    targets = centers[k].sign() * centers[k].abs().expm1()
    u = exp.percentile_label(probabilities.log().expand(100_000, -1), targets, generator)
    quantiles = torch.quantile(u, torch.arange(1, 10).float() / 10)
    torch.testing.assert_close(quantiles, torch.arange(1, 10).float() / 10, atol=0.01, rtol=0)
    assert not u.requires_grad


def test_ties_fill_the_probability_mass():
    probabilities = torch.zeros(65)
    probabilities[32], probabilities[64] = 0.9, 0.1
    u = exp.percentile_label(
        probabilities.log().expand(10_000, -1), torch.zeros(10_000), torch.Generator().manual_seed(7)
    )
    assert u.min() >= 0 and u.max() <= 0.9
    assert u.min() < 0.001 and u.max() > 0.899


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_labeler_resume(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    labeler = exp.PercentileLabeler(0, device)
    logits = torch.randn(2, 128, 65, device=device)
    targets = torch.zeros(2, 128, device=device)
    labeler(logits, targets)
    state = labeler.state_dict()
    expected = labeler(logits, targets)
    restored = exp.PercentileLabeler(9, device)
    restored.load_state_dict(state)
    assert torch.equal(restored(logits, targets), expected)


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("step", [4095, 4096])
def test_condition_warmup_dropout_and_prefix_values(arm, step):
    cfg = _cfg(arm)
    batch = _batch(cfg)
    batch.condition_present[:, ::3] = False
    batch.available[:, ::7] = False
    batch.eligible[:, ::5] = False
    model = exp.make_model(cfg)
    positions = torch.arange(64, 96).expand(2, -1)
    labeler = exp.PercentileLabeler(0, "cpu")
    expected_labeler = exp.PercentileLabeler(0, "cpu")
    captured = []

    def temporal(*args):
        captured.append((args[-2].clone(), args[-1].clone()))
        return model.temporal.teacher_forced_nll_with_diagnostics(*args)

    with torch.no_grad():
        hidden = model(batch.context.features, batch.context.ctx_pad, None)
        features = exp.decoder_rmsnorm(hidden[:, 64:]).detach()
        target, available = exp.percentile_target(arm, batch, model.value_head(features).squeeze(-1))
        expected_u = expected_labeler(model.percentile_head(features), torch.where(available, target, 0.0))
    loss, _, metrics = exp.microbatch_loss(
        model,
        batch,
        cfg,
        step=step,
        valid_prefixes=64,
        trunk_fn=model.forward,
        temporal_fn=temporal,
        prefix_positions=positions,
        percentile_labeler=labeler,
    )
    assert torch.isfinite(loss)
    torch.testing.assert_close(captured[0][0], exp.condition_value(expected_u[:, :32]))
    expected_present = batch.condition_present[:, 64:] & available & (step >= 4096)
    assert torch.equal(captured[0][1], expected_present[:, :32])
    torch.testing.assert_close(metrics["percentile/conditioned_fraction"], expected_present.float().mean())


def test_match_dropout_does_not_use_return60_availability():
    batch = _batch(_cfg("match"))
    batch.available.fill_(False)
    masked = exp.ReturnMasker(3, 0.0)(batch)
    assert masked.condition_present.all()
    assert not control.ReturnMasker(3, 0.0)(_control_batch(batch)).condition_present.any()


@pytest.mark.parametrize("step", [0, 4096])
def test_loss_and_value_gradient_match_059_plus_percentile(step):
    cfg = _cfg()
    baseline_cfg = _control_config(cfg)
    torch.manual_seed(5)
    model = exp.make_model(cfg)
    torch.manual_seed(5)
    baseline = control.make_model(baseline_cfg)
    batch = _batch(cfg)
    positions = torch.arange(64, 96).expand(2, -1)
    common = dict(step=step, valid_prefixes=64, prefix_positions=positions)
    loss, nll, metrics = exp.microbatch_loss(
        model,
        batch,
        cfg,
        trunk_fn=model.forward,
        temporal_fn=model.temporal.teacher_forced_nll_with_diagnostics,
        percentile_labeler=exp.PercentileLabeler(0, "cpu"),
        **common,
    )
    expected, expected_nll, expected_metrics = control.microbatch_loss(
        baseline,
        _control_batch(batch),
        baseline_cfg,
        trunk_fn=baseline.forward,
        temporal_fn=baseline.temporal.teacher_forced_nll_with_diagnostics,
        **common,
    )
    torch.testing.assert_close(loss, expected + metrics["percentile/loss"])
    torch.testing.assert_close(nll, expected_nll)
    torch.testing.assert_close(metrics["value/loss"], expected_metrics["value/loss"])
    loss.backward()
    expected.backward()
    for name, parameter in baseline.named_parameters():
        if name.startswith("temporal.return_conditioner."):
            continue
        torch.testing.assert_close(parameter.grad, dict(model.named_parameters())[name].grad, msg=name)


@pytest.mark.parametrize("arm", ARMS)
def test_protocol_rejects_other_versions_and_arms(arm):
    cfg = exp.proxy_config(arm)
    protocol = exp.conditioning_protocol(cfg)
    assert protocol["version"] == 2 and protocol["arm"] == arm
    assert protocol["warmup_update"] == 4097 and protocol["labeler_seed"] == 0x063AB
    calibration = exp.ReturnCalibration(window_count=exp.CALIBRATION_WINDOWS)
    state = {
        "conditioning_protocol": protocol,
        "percentile_labeler": exp.PercentileLabeler(0, "cpu").state_dict(),
        "return_calibration": calibration.state_dict(),
        "return_masker": exp.ReturnMasker(0, 0.2).state_dict(),
    }
    exp.validate_conditioning_state(state, cfg)
    for version in [1, 3]:
        with pytest.raises(ValueError, match="protocol"):
            exp.validate_conditioning_state({**state, "conditioning_protocol": {**protocol, "version": version}}, cfg)
    for other in set(ARMS) - {arm}:
        with pytest.raises(ValueError, match="protocol"):
            exp.validate_conditioning_state(state, replace(cfg, arm=other))


def test_experiment_checkpoints_reject_each_other(tmp_path):
    for reader, writer, cfg in [(exp, control, control.proxy_config()), (control, exp, exp.proxy_config("return60"))]:
        path = tmp_path / f"{writer.__name__}.pt"
        torch.save({"cfg": writer._checkpoint_config(cfg)}, path)
        with pytest.raises(ValueError, match="checkpoint"):
            reader.load_checkpoint(str(path), device="cpu")


@pytest.mark.parametrize("name,value", [("u50", 0.0), ("u75", 60.0), ("u90", 96.0), ("unconditioned", None)])
def test_eval_targets_and_evidence_directory(name, value, tmp_path):
    assert (
        exp.desired_return_for_target(name) == pytest.approx(value)
        if value is not None
        else exp.desired_return_for_target(name) is None
    )
    cfg = _cfg()
    model = exp.make_model(cfg)
    protocol = exp._eval_protocol(
        cfg, model, n_matchups=1, checkpoint_sha256="a" * 64, desired_return=exp.desired_return_for_target(name)
    )
    assert protocol.return_target == name
    exp._write_eval_evidence(tmp_path, [], {}, protocol)
    exp._validate_eval_directory(tmp_path, protocol)
    for other in {"u50", "u75", "u90", "unconditioned"} - {name}:
        with pytest.raises(ValueError, match="different protocol"):
            exp._validate_eval_directory(
                tmp_path, replace(protocol, return_target=other, desired_return=exp.desired_return_for_target(other))
            )
    assert json.loads((tmp_path / "match_rows.json").read_text())["protocol"]["return_target"] == name


def _equal(left, right, path=""):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right), (
            f"{path}: max absolute difference {float((left.float() - right.float()).abs().max())}"
        )
    elif isinstance(left, Mapping):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key], f"{path}/{key}")
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _equal(a, b, path)
    else:
        assert left == right


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("device,compiled", [("cpu", False), ("cuda", False), ("cuda", True)])
def test_training_change_and_next_update_resume(tmp_path, monkeypatch, request, arm, device, compiled):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if compiled:
        # Isolate state restoration from Inductor's nondeterministic embedding reductions.
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        request.addfinalizer(
            functools.partial(
                torch.use_deterministic_algorithms,
                torch.are_deterministic_algorithms_enabled(),
                warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            )
        )
        torch.use_deterministic_algorithms(True)
    cfg = replace(
        _cfg(arm, batch_size=1),
        amp_dtype="bfloat16" if device == "cuda" else "float32",
        compile_trunk=compiled,
        compile_temporal=compiled,
    )

    def setup(saved=None):
        model = exp.make_model(cfg).to(device)
        optimizer = exp.make_optimizer(model, cfg)
        scheduler = exp.LearningRateScheduler(optimizer, exp.lr_schedule(cfg))
        sampler = exp.PrefixSampler(7, device)
        labeler = exp.PercentileLabeler(cfg.seed, device)
        if saved is not None:
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            sampler.load_state_dict(saved["prefix"])
            labeler.load_state_dict(saved["percentile_labeler"])
            exp.restore_rng(saved["rng"])
        trunk_fn, temporal_fn = exp._training_functions(model, cfg)
        if compiled:
            exp._compile_synthetic_forward_backward(model, cfg, step=4095, trunk_fn=trunk_fn, temporal_fn=temporal_fn)
        return model, optimizer, scheduler, sampler, labeler, trunk_fn, temporal_fn

    def update(state, batch, step):
        model, optimizer, scheduler, sampler, labeler, trunk_fn, temporal_fn = state
        result = exp.train_step(
            model,
            batch,
            cfg,
            step=step,
            update=step + 1,
            valid_prefixes=32,
            trunk_fn=trunk_fn,
            temporal_fn=temporal_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            prefix_sampler=sampler,
            percentile_labeler=labeler,
        )
        return {
            "model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "scheduler": scheduler.state_dict(),
            "prefix": sampler.state_dict(),
            "percentile_labeler": labeler.state_dict(),
            "metrics": {name: value.detach().cpu().clone() for name, value in result.metrics.items()},
            "gradient_norm": result.gradient_norm.detach().cpu().clone(),
        }

    original = setup()
    batch = _batch(cfg, device)
    update(original, batch, 4095)
    model, optimizer, scheduler, sampler, labeler, *_ = original
    saved = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "prefix": sampler.state_dict(),
        "percentile_labeler": labeler.state_dict(),
        "rng": exp.rng_state(),
    }
    path = tmp_path / "resume.pt"
    torch.save(saved, path)
    next_batch = replace(batch, returns=batch.returns + 5, future_return=batch.future_return + 2)
    next_batch.context.features["ego_position_x"].fill_(0.25)
    expected = update(original, next_batch, 4096)
    restored = setup(torch.load(path, weights_only=False))
    actual = update(restored, next_batch, 4096)
    _equal(expected, actual)


def test_calibration_metrics_report_deciles_and_empty_selection():
    u = (torch.arange(1000).float() + 0.5).reshape(2, 500) / 1000
    selected = torch.ones_like(u, dtype=torch.bool)
    metrics = exp.percentile_metrics(u, selected, selected, torch.tensor(1.0))
    assert metrics["percentile/calibration_error"] < 1e-6
    assert metrics["percentile/max_decile_drift"] < 0.001
    for index in range(1, 10):
        assert abs(float(metrics[f"percentile/u_p{index * 10:02d}"]) - index / 10) < 0.001
    empty = exp.percentile_metrics(u, ~selected, ~selected, torch.tensor(0.0))
    assert all(torch.isfinite(value) and value == 0 for value in empty.values())


def test_boundary_checkpoint_includes_labeler_state(tmp_path):
    cfg = _cfg()
    model = exp.make_model(cfg)
    optimizer = exp.make_optimizer(model, cfg)
    scheduler = exp.LearningRateScheduler(optimizer, exp.lr_schedule(cfg))
    labeler = exp.PercentileLabeler(0, "cpu")
    labeler(torch.zeros(2, 128, 65), torch.zeros(2, 128))
    saved = labeler.state_dict()
    path = exp.save_boundary_checkpoint(
        tmp_path,
        update=4096,
        model=model,
        calibration=exp.ReturnCalibration(window_count=exp.CALIBRATION_WINDOWS),
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        corpus=exp.Corpus(sources=(), player_sidecar=Path("unused"), from_huggingface=False),
        uploader=None,
        milestone=True,
        wandb_id=None,
        actual_loss_positions=0,
        loader_state={},
        return_masker_state=exp.ReturnMasker(0, 0.2).state_dict(),
        percentile_labeler_state=saved,
    )
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["conditioning_protocol"] == exp.conditioning_protocol(cfg)
    _equal(checkpoint["percentile_labeler"], saved)
    restored = exp.PercentileLabeler(99, "cpu")
    restored.load_state_dict(checkpoint["percentile_labeler"])
    assert torch.equal(
        restored(torch.zeros(1, 1, 65), torch.zeros(1, 1)), labeler(torch.zeros(1, 1, 65), torch.zeros(1, 1))
    )


def _step_snapshot(model, result, sampler) -> dict[str, object]:
    return {
        "parameters": {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()},
        "gradients": {
            name: parameter.grad.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        },
        "metrics": {name: value.detach().cpu().clone() for name, value in result.metrics.items()},
        "gradient_norm": result.gradient_norm.detach().cpu().clone(),
        "sampler": sampler.generator.get_state().cpu(),
    }


class _ResumeReplayAdapter:
    def __init__(self) -> None:
        self.manifests = {"source": replay_loader.SourceManifest("source", (65,))}

    def prepare_shard(self, task: replay_loader.ShardTask) -> None:
        del task

    def _generation(self, row: int, epoch: int, windows: int) -> tuple[str, tuple[dict[str, np.ndarray], ...]]:
        return f"replay-{row}", tuple(
            {"value": np.asarray([epoch * 1000 + row * windows + ordinal], dtype=np.float32)}
            for ordinal in range(windows)
        )

    def decode_chunk(
        self, request: replay_loader._DecodeChunkRequest, task: replay_loader.ShardTask, **options: object
    ) -> replay_loader.DecodedChunk:
        windows = cast(int, options["windows_per_generation"])
        generations = [self._generation(row, request.epoch, windows) for row in request.rows]
        return replay_loader.DecodedChunk(
            request=request,
            task=task,
            replay_ids=tuple(value[0] for value in generations),
            locators=tuple(replay_loader.PhysicalRow(task.source, task.shard, row) for row in request.rows),
            columns={"value": np.stack([[window["value"] for window in value[1]] for value in generations])},
            windows_per_generation=windows,
        )

    def decode_generations(
        self, task: replay_loader.ShardTask, requests: Sequence[tuple[int, int]], **options: object
    ) -> Mapping[tuple[int, int], tuple[str, tuple[dict[str, np.ndarray], ...]]]:
        del task
        return {
            request: self._generation(*request, cast(int, options["windows_per_generation"])) for request in requests
        }


def _resume_labels(row: Mapping[str, object]) -> dict[str, np.ndarray]:
    del row
    return {}


def _resume_collate(
    replay_ids: tuple[str, ...], columns: Mapping[str, np.ndarray], *, cfg: exp.TrainConfig
) -> exp.ReturnBatch:
    batch = _batch(cfg)
    values = torch.from_numpy(columns["value"].copy()).expand(-1, cfg.arch.L_ctx).clone()
    batch.context.features["ego_position_x"] = values / 100
    batch.context.features["ego_player_id"].fill_(17)
    available = values.long().remainder(3) != 0
    return replace(
        batch,
        batch=replace(batch.batch, replay_ids=replay_ids),
        returns=values,
        future_return=values + 1,
        available=available,
        condition_present=available.clone(),
    )


def _resume_loader(cfg: exp.TrainConfig) -> replay_loader.BufferedMDSReplayLoader[exp.ReturnBatch]:
    import functools

    selection = exp.PhysicalShardSelection.from_sources((exp.SourceRowSelection("source", 65),))
    return replay_loader.BufferedMDSReplayLoader(
        selection=selection,
        adapter=_ResumeReplayAdapter(),
        tasks=(replay_loader.ShardTask("source", 0, 0, 65),),
        data_protocol=cfg.data_protocol,
        source_manifest_sha256={"source": "a" * 64},
        labels=_resume_labels,
        projection=None,
        batch_transform=functools.partial(_resume_collate, cfg=cfg),
        batch_size=cfg.batch_size,
        replay_slots=64,
        seed=cfg.seed,
        num_workers=0,
        context_length=cfg.arch.L_ctx,
        chunk_length=cfg.arch.sample_chunk_length,
        windows_per_generation=8,
        replay_phase_block_batches=8,
        schema_version=7,
        reserved_disk_bytes=0,
        pin_memory=False,
    )


@pytest.mark.parametrize("arm", ARMS)
def test_drained_prefetch_restores_loader_masks_and_optimizer_update(tmp_path: Path, arm) -> None:
    from contextlib import ExitStack

    cfg = _cfg(arm, batch_size=8)
    cfg = replace(cfg, arch=replace(cfg.arch, L_ctx=128))
    torch.manual_seed(71)

    def setup(resources: ExitStack, saved: dict[str, Any] | None = None) -> tuple[Any, ...]:
        model = exp.make_model(cfg)
        calibration = exp.ReturnCalibration(window_count=exp.CALIBRATION_WINDOWS)
        optimizer = exp.make_optimizer(model, cfg)
        scheduler = exp.LearningRateScheduler(optimizer, exp.lr_schedule(cfg))
        sampler = exp.PrefixSampler(7, "cpu")
        identity = exp.IdentityMasker(11, 0.5)
        masker = exp.ReturnMasker(13, 0.2)
        labeler = exp.PercentileLabeler(cfg.seed, "cpu")
        loader = _resume_loader(cfg)
        resources.callback(loader.close)
        if saved is not None:
            loader.load_state_dict(saved["loader"])
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            sampler.load_state_dict(saved["prefix"])
            identity.load_state_dict(saved["identity"])
            masker.load_state_dict(saved["return_masker"])
            calibration.load_state_dict(saved["calibration"])
            labeler.load_state_dict(saved["percentile_labeler"])
            exp.restore_rng(saved["rng"])
        prefetch = exp.DeviceBatchPrefetcher(
            loader, cfg, "cpu", identity, return_masker=masker, calibration=calibration
        )
        resources.callback(prefetch.close)
        return model, optimizer, scheduler, sampler, identity, masker, loader, calibration, labeler, prefetch

    def update(state: tuple[Any, ...], batch: exp.ReturnBatch, index: int) -> dict[str, object]:
        model, optimizer, scheduler, sampler, *_ = state
        result = exp.train_step(
            model,
            batch,
            cfg,
            step=4096 + index,
            update=index + 1,
            valid_prefixes=8 * 32,
            trunk_fn=model.forward,
            temporal_fn=model.temporal.teacher_forced_nll_with_diagnostics,
            optimizer=optimizer,
            scheduler=scheduler,
            prefix_sampler=sampler,
            percentile_labeler=state[-2],
            prefix_validated_on_cpu=True,
        )
        return _step_snapshot(model, result, sampler)

    with ExitStack() as resources:
        original = setup(resources)
        model, optimizer, scheduler, sampler, identity, masker, loader, calibration, labeler, prefetch = original
        for index in range(5):
            batch, _ = prefetch.next()
            update(original, batch, index)
            if index < 4:
                prefetch.fill_lookahead(1)
                prefetch.stage_next()
        assert prefetch.drained
        path = tmp_path / "resume.pt"
        torch.save(
            {
                "loader": loader.state_dict(),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "prefix": sampler.state_dict(),
                "identity": identity.state_dict(),
                "return_masker": masker.state_dict(),
                "calibration": calibration.state_dict(),
                "percentile_labeler": labeler.state_dict(),
                "rng": exp.rng_state(),
            },
            path,
        )
        prefetch.fill_lookahead(1)
        prefetch.stage_next()
        expected_batch, _ = prefetch.next()
        expected = update(original, expected_batch, 5)
        restored = setup(resources, torch.load(path, weights_only=False))
        actual_batch, _ = restored[-1].next()
        assert exp._return_batch_sha256(actual_batch) == exp._return_batch_sha256(expected_batch)
        actual = update(restored, actual_batch, 5)
        _equal(expected, actual)
        assert restored[-3].state_dict() == calibration.state_dict()
        _equal(restored[-2].state_dict(), labeler.state_dict())
        assert restored[-1].drained


@pytest.mark.parametrize("target", ("u50", "u75", "u90", "unconditioned"))
def test_checkpoint_evaluation_uses_percentile_instead_of_raw_calibration(tmp_path, monkeypatch, target):
    cfg = exp.proxy_config("return60")
    with torch.device("meta"):
        model = exp.make_model(cfg)
    calibration = exp.ReturnCalibration(window_count=exp.CALIBRATION_WINDOWS)
    # An incomplete calibration is sufficient: percentile evaluation must never
    # read the control's positive raw-return quantile.
    state = {"step": 16383, "return_calibration": calibration.state_dict()}
    monkeypatch.setattr(exp, "load_checkpoint", lambda *args, **kwargs: (model, cfg, {}, state))
    captured = []

    def evaluate(*args, **kwargs):
        captured.append(kwargs["desired_return"])
        return {"scheduled_boots": 96.0, "completed_boots": 96.0, "boots": 96.0}

    monkeypatch.setattr(exp, "eval_vs_cpu", evaluate)
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"identity-only")
    exp.eval_checkpoint(
        str(path),
        return_target=target,
        output_name=f"eval-{target}",
        wandb_namespace="eval" if target == "u75" else f"eval_{target}",
    )
    assert captured == [exp.desired_return_for_target(target)]


@pytest.mark.parametrize("arm", ARMS)
def test_offline_validation_uses_percentile_inputs_without_advancing_training_rng(monkeypatch, arm):
    cfg = _cfg(arm, batch_size=1)
    batch = _batch(cfg)
    model = exp.make_model(cfg)
    captured = []
    original = model.temporal.teacher_forced_logits_by_group

    def teacher(*args):
        captured.append((args[-2].clone(), args[-1].clone()))
        return original(*args)

    monkeypatch.setattr(model.temporal, "teacher_forced_logits_by_group", teacher)
    generator_state = torch.get_rng_state().clone()
    with torch.no_grad():
        hidden = model(batch.context.features, batch.context.ctx_pad, None)
        features = exp.decoder_rmsnorm(hidden[:, 64:]).detach()
        target, available = exp.percentile_target(arm, batch, model.value_head(features).squeeze(-1))
        u = exp.PercentileLabeler(cfg.seed, "cpu")(
            model.percentile_head(features), torch.where(available, target, 0.0)
        )
    metrics = exp.val_metrics(model, [batch], cfg)
    assert metrics and model.training
    torch.testing.assert_close(captured[0][0], exp.condition_value(u))
    assert torch.equal(captured[0][1], available)
    assert torch.equal(generator_state, torch.get_rng_state())
