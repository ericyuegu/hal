"""Contracts for production percentile conditioning and the denser offsets."""

import copy
import functools
import importlib.util
import inspect
import sys
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _load(name: str) -> ModuleType:
    path = Path(__file__).parents[2] / "experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"hal_contract_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


previous = _load("063_percentile_conditioning")
exp = _load("064_production_percentile")


def _cfg():
    cfg = exp.proxy_config("bootstrap30")
    return replace(
        cfg,
        arch=replace(
            cfg.arch,
            d_model=32,
            n_layers=1,
            n_heads=2,
            L_ctx=128,
            temporal_d_model=32,
            temporal_layers=2,
            temporal_heads=2,
            temporal_ff_dim=128,
            group_head_dim=32,
            value_hidden_dim=16,
            item_hidden_dim=8,
            item_dim=5,
        ),
        batch_size=1,
        amp_dtype="float32",
        compile_trunk=False,
        compile_temporal=False,
        inference_mode="eager",
        num_workers=0,
        push_to_r2=False,
    )


def _batch(cfg, device="cpu"):
    batch = exp.synthetic_awr_batch(cfg, torch.device(device))
    returns = torch.linspace(-20, 50, cfg.arch.L_ctx, device=device)[None, :]
    return replace(batch, returns=returns, future_return=returns / 2)


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for name in left:
            _equal(left[name], right[name])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("proxy", [False, True])
def test_configuration_changes_only_the_action_offsets(proxy):
    cfg = exp.proxy_config("bootstrap30") if proxy else exp.TrainConfig(arm="bootstrap30")
    baseline = previous.proxy_config("bootstrap30") if proxy else previous.TrainConfig(arm="bootstrap30")
    values = asdict(cfg)
    values["arch"]["sample_chunk_length"] = 28
    values["arch"]["head_offsets"] = (*range(1, 13), 16, 20, 24, 28)
    assert values == asdict(baseline)
    assert cfg.arch.head_offsets == (*range(1, 25), 27, 30)
    assert cfg.arch.sample_chunk_length == 30
    assert cfg.minimum_replay_frames == 293
    exp.validate_config(cfg)
    if not proxy:
        assert (cfg.arch.d_model, cfg.arch.n_layers, cfg.arch.temporal_layers) == (1024, 12, 6)
        assert (cfg.stable_updates, cfg.max_steps) == (98_304, 131_072)
        assert len(cfg.source_names) == 44
    with pytest.raises(ValueError, match="architecture|offsets"):
        exp.validate_config(replace(cfg, arch=replace(cfg.arch, sample_chunk_length=28)))


@pytest.mark.parametrize("proxy,total", [(False, 247_944_158), (True, 17_133_278)])
def test_model_optimizer_and_compute_account_for_all_offsets(proxy, total):
    cfg = exp.proxy_config("bootstrap30") if proxy else exp.TrainConfig(arm="bootstrap30")
    baseline_cfg = previous.proxy_config("bootstrap30") if proxy else previous.TrainConfig(arm="bootstrap30")
    with torch.device("meta"):
        model = exp.make_model(cfg)
        baseline = previous.make_model(baseline_cfg)
    counts = exp.subsystem_parameter_counts(model)
    baseline_counts = previous.subsystem_parameter_counts(baseline)
    assert counts == cfg.arch.parameter_count_contract
    assert counts["total"] == total == baseline_counts["total"] + 32
    assert model.temporal.offset_embedding.num_embeddings == 31
    assert exp.optimizer_roles(model, cfg).keys() == dict(model.named_parameters()).keys()
    expected = (
        6
        * cfg.batch_size
        * (
            cfg.arch.L_ctx * (counts["trunk"] + counts["other"])
            + 128 * (counts["value_head"] + counts["percentile_head"])
            + 32 * (counts["trunk_skip_heads"] + counts["return_conditioner"])
            + 32 * 26 * (counts["temporal_decoder"] + counts["group_heads"])
        )
    )
    assert exp.approximate_training_flops_per_update(cfg, counts) == expected


def test_offset_gradients_keep_next_frame_share_and_awr_only_on_first_six():
    offsets = torch.arange(2, 25, dtype=torch.float32)
    dense = 0.664 * offsets.reciprocal() / offsets.reciprocal().sum()
    coefficients = torch.cat((torch.tensor([0.33]), dense, torch.tensor([0.004, 0.002])))
    torch.testing.assert_close(torch.tensor(exp.OFFSET_LOSS_WEIGHTS), coefficients)
    assert coefficients.sum() == pytest.approx(1.0)
    assert coefficients[0] == pytest.approx(0.33)
    assert coefficients[1:24].sum() == pytest.approx(0.664)
    assert (coefficients[:-1] > coefficients[1:]).all()
    nll = torch.ones(2, 26, 4, requires_grad=True)
    weights = torch.tensor([2.0, 4.0])
    near, far, loss = exp.temporal_objective_parts(nll, weights, valid_prefixes=1, valid=torch.tensor([True, False]))
    assert float(near.detach()) == pytest.approx(float(4 * 2 * coefficients[:6].sum()))
    assert float(far.detach()) == pytest.approx(float(4 * coefficients[6:].sum()))
    loss.backward()
    expected = coefficients.clone()
    expected[:6] *= 2
    torch.testing.assert_close(nll.grad[0], expected[:, None].expand(26, 4))
    assert torch.count_nonzero(nll.grad[1]) == 0


def test_targets_align_every_offset_through_frame_30():
    cfg = _cfg()
    model = exp.make_model(cfg)
    batch = _batch(cfg)
    batch.target[:, :, 0] = torch.arange(30) % 2
    history, targets, valid = exp.prepared_targets(model, batch)
    full = model.codec.quantize(torch.cat((exp.stack_actions(batch.context.features), batch.target), dim=1))
    assert targets.shape == (1, 64, 26, 4)
    assert valid.all()
    torch.testing.assert_close(history, full[:, 64:128])
    for index, offset in enumerate(cfg.arch.head_offsets):
        torch.testing.assert_close(targets[:, :, index], full[:, 64 + offset : 128 + offset])
    torch.testing.assert_close(targets[:, -1, -1], full[:, -1])


def test_percentile_and_value_targets_preserve_063_behavior():
    cfg = _cfg()
    batch = _batch(cfg)
    value = torch.linspace(0, 2, 64)[None, :]
    baseline_batch = previous.ReturnBatch(
        batch.batch, batch.returns, batch.eligible, batch.future_return, batch.available, batch.condition_present
    )
    _equal(
        exp.percentile_target("bootstrap30", batch, value),
        previous.percentile_target("bootstrap30", baseline_batch, value),
    )
    logits = torch.linspace(-2, 2, 65).expand(1, 64, -1)
    _equal(exp.hl_gauss_target(value), previous.hl_gauss_target(value))
    _equal(exp.PercentileLabeler(0, "cpu")(logits, value), previous.PercentileLabeler(0, "cpu")(logits, value))


def test_checkpoint_identity_and_offset_weights_are_validated():
    cfg = exp.TrainConfig(arm="bootstrap30")
    values = exp._checkpoint_config(cfg)
    assert exp.config_from_state(values) == cfg
    with pytest.raises(ValueError, match="experiment_id"):
        exp.config_from_state(previous._checkpoint_config(previous.TrainConfig(arm="bootstrap30")))
    with pytest.raises(ValueError, match="experiment_id"):
        previous.config_from_state(values)
    protocol = exp.conditioning_protocol(cfg)
    assert protocol["head_offsets"] == list(cfg.arch.head_offsets)
    assert protocol["offset_loss_weights"] == list(exp.OFFSET_LOSS_WEIGHTS)
    protocol["offset_loss_weights"][0] = 0.1
    with pytest.raises(ValueError, match="conditioning protocol"):
        exp.validate_conditioning_state({"conditioning_protocol": protocol}, cfg)


def test_u95_is_the_default_for_all_gameplay_entry_points():
    assert exp.EvalArgs(checkpoint="unused.pt").return_target == "u95"
    for function in (exp.eval_checkpoint, exp.run_closed_loop_evaluation, exp.spawn_closed_loop_evaluation):
        assert inspect.signature(function).parameters["return_target"].default == "u95"
    assert exp.desired_return_for_target("u95") == pytest.approx(108.0)
    assert exp.desired_return_for_target("unconditioned") is None


def test_validation_batch_size_preserves_teacher_forced_scores():
    cfg = replace(_cfg(), batch_size=2)
    torch.manual_seed(31)
    model = exp.make_model(cfg)
    batch = exp.synthetic_awr_batch(cfg, torch.device("cpu"))
    batch.context.features["ego_position_x"][1].fill_(0.25)
    batch.target[1, :, 0] = torch.arange(cfg.arch.sample_chunk_length) % 2

    def row(index):
        selected = slice(index, index + 1)
        context = replace(
            batch.context,
            features={name: value[selected] for name, value in batch.context.features.items()},
            ctx_pad=batch.context.ctx_pad[selected],
        )
        return replace(
            batch,
            batch=replace(batch.batch, context=context, target=batch.target[selected]),
            returns=batch.returns[selected],
            eligible=batch.eligible[selected],
            future_return=batch.future_return[selected],
            available=batch.available[selected],
            condition_present=batch.condition_present[selected],
        )

    combined = exp.val_metrics(model, [batch], cfg)
    separate = exp.val_metrics(model, [row(0), row(1)], cfg)
    keys = {name for name in combined if name.startswith("nll_o")}
    keys |= {"loss_unweighted", "temporal_loss_near_unweighted", "temporal_loss_far_unweighted"}
    for name in keys:
        assert combined[name] == pytest.approx(separate[name], rel=1e-6, abs=1e-5), name


@pytest.mark.parametrize("arm", ["return60", "bootstrap30", "match"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_validation_scores_match_all_prefix_reference(arm, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cfg = replace(_cfg(), arm=arm, batch_size=2, amp_dtype="bfloat16" if device == "cuda" else "float32")
    torch.manual_seed(43)
    model = exp.make_model(cfg).to(device)
    batch = exp.synthetic_awr_batch(cfg, torch.device("cpu"))
    batch.context.ctx_pad[1] = 16
    batch.context.features["ego_position_x"][1] = torch.linspace(-1, 1, cfg.arch.L_ctx)
    batch.target[1, :, 0] = torch.arange(cfg.arch.sample_chunk_length) % 2
    batch.returns.copy_(torch.linspace(-100, 100, cfg.arch.L_ctx)[None, :])
    batch.future_return.copy_(batch.returns / 2)
    batch.eligible[1, -1] = False
    batch.available[1, -1] = False
    model.eval()
    with torch.no_grad():
        for projection in model.temporal.return_conditioner.projections:
            projection.weight.normal_(std=0.01)
            projection.bias.normal_(std=0.01)
        device_batch = batch.to(device)
        history, targets, _valid = exp.prepared_targets(model, device_batch)
        with exp.amp_context(cfg, torch.device(device)):
            hidden = model.forward_dense(device_batch.context.features, device_batch.context.ctx_pad, None)
        features = exp.decoder_rmsnorm(hidden[:, cfg.arch.direct_loss_start :]).float()
        value = model.value_head(features).squeeze(-1)
        target, available = exp.percentile_target(arm, device_batch, value)
        labeler = exp.PercentileLabeler(cfg.seed, device)
        conditions = exp.condition_value(labeler(model.percentile_head(features), torch.where(available, target, 0.0)))
        positions = torch.arange(cfg.arch.direct_loss_start, cfg.arch.L_ctx, device=device).expand(cfg.batch_size, -1)
        with exp.amp_context(cfg, torch.device(device)):
            logits = model.temporal.teacher_forced_logits_by_group(
                hidden, device_batch.context.ctx_pad, positions, history, targets, conditions, available
            )
            all_nll = model.temporal.nll_from_logits(logits, targets)
        expected = exp.nll_mean_metrics(all_nll[:, -1].double().mean(0).cpu(), cfg.arch.head_offsets)
    actual = exp.val_metrics(model, [batch], cfg)
    keys = {name for name in expected if name.startswith("nll_o")}
    keys |= {"loss_unweighted", "temporal_loss_near_unweighted", "temporal_loss_far_unweighted"}
    for name in keys:
        assert actual[name] == pytest.approx(
            expected[name], abs=0.01 if device == "cuda" else 1e-5, rel=0.002 if device == "cuda" else 1e-6
        ), name


@pytest.mark.parametrize("device,compiled", [("cpu", False), ("cuda", False), ("cuda", True)])
def test_next_update_is_exact_after_resume_with_26_offsets(tmp_path, monkeypatch, request, device, compiled):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if compiled:
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
        _cfg(),
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
        return copy.deepcopy(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "prefix": sampler.state_dict(),
                "percentile_labeler": labeler.state_dict(),
                "metrics": {name: value.detach().cpu() for name, value in result.metrics.items()},
                "gradient_norm": result.gradient_norm.detach().cpu(),
            }
        )

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
    next_batch = replace(batch, returns=batch.returns + 5)
    next_batch.context.features["ego_position_x"].fill_(0.25)
    expected = update(original, next_batch, 4096)
    restored = setup(torch.load(path, weights_only=False))
    _equal(expected, update(restored, next_batch, 4096))
