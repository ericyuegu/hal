"""Contracts for the O55 history cross-attention ablation."""

import copy
import importlib.util
import json
import sys
from dataclasses import asdict
from dataclasses import fields
from pathlib import Path

import pytest
import torch


def _load():
    path = Path(__file__).resolve().parents[2] / "experiments" / "055_history_cross_attention.py"
    name = "test_exp055"
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
        "temporal_layers": 4,
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
    assert proxy.target_windows == 2**23
    assert proxy.max_steps == 16_384
    assert cfg.source_names == ("ranked-anonymized-1-policy-world-v8",)
    assert cfg.train_replays == 112_188
    assert cfg.train_frames == 1_203_888_017
    assert cfg.val_n_samples == 1024
    assert (cfg.eval_n_matchups, cfg.final_eval_n_matchups) == (96, 96)
    assert cfg.replay_slots == 112_128
    assert cfg.replay_slots <= cfg.train_replays
    assert cfg.replay_slots // cfg.batch_size - cfg.replay_phase_block_batches + 1 == 195
    assert cfg.minimum_replay_gap_batches == 195
    assert exp.data_selection(cfg).sha256 == cfg.selection_sha256
    assert exp.source_manifest_sha256(cfg) == {
        "ranked-anonymized-1-policy-world-v8": "b97eab90e761bcf2bf03b48981f0ab6acc1ac3057157c58ae0c5a72c76c43bd8"
    }
    assert exp.assert_protocol_diversity(96) == (
        58,
        13,
        14,
        "a2202b353e3e769f2ab25e673226ef29fb6f949f4391c2b9f3003afdc7ce3c15",
    )


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


@pytest.mark.parametrize(
    ("variant", "stored", "active"),
    (
        ("baseline", 14_480_922, 14_480_922),
        ("add-history", 14_579_226, 14_579_226),
        ("remove-state-bias", 14_579_226, 14_546_458),
        ("replace-attention", 14_874_138, 14_579_226),
    ),
)
def test_proxy_parameter_counts_distinguish_stored_and_active(variant: str, stored: int, active: int) -> None:
    cfg = exp.proxy_config()
    cfg = exp.replace(cfg, decoder_variant=variant)
    model = exp.GPT(cfg)
    counts = exp.subsystem_parameter_counts(model)

    assert counts["total"] == stored
    assert exp.active_parameter_count(cfg, counts) == active


def _decoder_inputs(cfg):
    generator = torch.Generator().manual_seed(71)
    hidden = torch.randn(
        cfg.batch_size,
        cfg.arch.L_ctx,
        cfg.arch.d_model,
        generator=generator,
    )
    observed = torch.zeros(
        cfg.batch_size,
        1,
        exp.CONTROLLER_GROUP_COUNT,
        dtype=torch.long,
    )
    targets = torch.zeros(
        cfg.batch_size,
        1,
        len(cfg.arch.head_offsets),
        exp.CONTROLLER_GROUP_COUNT,
        dtype=torch.long,
    )
    ctx_pad = torch.zeros(cfg.batch_size, dtype=torch.long)
    return hidden, ctx_pad, observed, targets


def test_training_targets_only_the_final_context_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _tiny_cfg()
    model = exp.GPT(cfg)
    batch = exp.synthetic_awr_batch(cfg, torch.device("cpu"))
    actions = exp.stack_actions(batch.context.features)
    full = model.codec.quantize(torch.cat((actions, batch.target), dim=1))
    expected = torch.stack(
        [full[:, cfg.arch.L_ctx - 1 + offset] for offset in cfg.arch.head_offsets],
        dim=1,
    )
    quantized_shapes = []
    original_quantize = exp.DiscreteControllerCodec.quantize

    def record_quantized_shape(codec, action_chunk):
        quantized_shapes.append(tuple(action_chunk.shape))
        return original_quantize(codec, action_chunk)

    monkeypatch.setattr(exp.DiscreteControllerCodec, "quantize", record_quantized_shape)
    observed, targets, valid = exp.prepared_targets(model, batch)

    assert observed.shape == (cfg.batch_size, 1, exp.CONTROLLER_GROUP_COUNT)
    assert targets.shape == (
        cfg.batch_size,
        1,
        len(cfg.arch.head_offsets),
        exp.CONTROLLER_GROUP_COUNT,
    )
    assert valid.shape == (cfg.batch_size, 1)
    assert valid.all()
    assert quantized_shapes == [(cfg.batch_size, cfg.arch.sample_chunk_length + 1, exp.A_DIM)]
    torch.testing.assert_close(observed[:, 0], full[:, cfg.arch.L_ctx - 1])
    torch.testing.assert_close(targets[:, 0], expected)


def test_baseline_does_not_build_a_history_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _tiny_cfg()
    model = exp.GPT(cfg)
    hidden, ctx_pad, observed, targets = _decoder_inputs(cfg)

    def fail(*_args):
        raise AssertionError("baseline built a history mask")

    monkeypatch.setattr(exp.HistoryCrossAttention, "memory_mask", staticmethod(fail))

    model.temporal.teacher_forced_nll(hidden, ctx_pad, observed, targets)


def test_cross_attention_keeps_projected_dtype_under_autocast() -> None:
    cfg = _tiny_cfg(decoder_variant="add-history")
    model = exp.GPT(cfg)
    hidden, _, _, _ = _decoder_inputs(cfg)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        key, value = model.temporal.history_cross_attentions[0].project_memory(hidden)

    assert key.dtype == torch.bfloat16
    assert value.dtype == torch.bfloat16


def test_common_parameters_have_identical_initialization_across_arms() -> None:
    cfg = _tiny_cfg()
    torch.manual_seed(83)
    baseline = exp.GPT(cfg)
    baseline_state = baseline.state_dict()

    first_cross_attention = None
    for variant in ("add-history", "remove-state-bias", "replace-attention"):
        torch.manual_seed(83)
        candidate = exp.GPT(exp.replace(cfg, decoder_variant=variant))
        candidate_state = candidate.state_dict()
        for name, expected in baseline_state.items():
            torch.testing.assert_close(candidate_state[name], expected, rtol=0, atol=0)
        current_cross_attention = {
            name: value
            for name, value in candidate_state.items()
            if name.startswith("temporal.history_cross_attentions.0.")
        }
        if first_cross_attention is None:
            first_cross_attention = current_cross_attention
        else:
            for name, expected in first_cross_attention.items():
                torch.testing.assert_close(current_cross_attention[name], expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "variant",
    ("baseline", "add-history", "remove-state-bias", "replace-attention"),
)
def test_batched_teacher_forcing_matches_stepwise_decoding(variant: str) -> None:
    cfg = _tiny_cfg(decoder_variant=variant)
    torch.manual_seed(89)
    model = exp.GPT(cfg)
    hidden, ctx_pad, observed, targets = _decoder_inputs(cfg)

    teacher = model.temporal.teacher_forced_logits_by_group(hidden, ctx_pad, observed, targets)
    stepwise = model.temporal.forced_stepwise_logits(hidden, ctx_pad, observed[:, 0], targets[:, 0])

    for depth, frame in enumerate(stepwise):
        for name in exp.CONTROLLER_GROUP_NAMES:
            torch.testing.assert_close(teacher[name][:, 0, depth], frame[name], rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    "variant",
    ("baseline", "add-history", "remove-state-bias", "replace-attention"),
)
def test_left_padding_does_not_change_decoder_logits(variant: str) -> None:
    cfg = _tiny_cfg(decoder_variant=variant)
    torch.manual_seed(97)
    model = exp.GPT(cfg)
    hidden, _, observed, targets = _decoder_inputs(cfg)
    pad = 3

    padded = model.temporal.teacher_forced_logits_by_group(
        hidden, torch.full((cfg.batch_size,), pad), observed, targets
    )
    stripped = model.temporal.teacher_forced_logits_by_group(
        hidden[:, pad:], torch.zeros(cfg.batch_size, dtype=torch.long), observed, targets
    )

    for name in exp.CONTROLLER_GROUP_NAMES:
        torch.testing.assert_close(padded[name], stripped[name], rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize(
    "variant",
    ("baseline", "add-history", "remove-state-bias", "replace-attention"),
)
def test_eager_inference_runs_each_decoder_arm(variant: str) -> None:
    cfg = _tiny_cfg(decoder_variant=variant)
    torch.manual_seed(99)
    model = exp.GPT(cfg).eval()
    context = exp.synthetic_context(cfg, cfg.batch_size, torch.device("cpu"))
    inference = exp.BF16Inference(model, cfg, compiled=False)

    actions = inference.decode(context, cfg.prediction_frames, argmax=True)

    assert actions.shape == (cfg.batch_size, cfg.prediction_frames, exp.A_DIM)
    assert torch.isfinite(actions).all()


@pytest.mark.parametrize(
    ("variant", "uses_history"),
    (
        ("baseline", False),
        ("add-history", True),
        ("remove-state-bias", True),
        ("replace-attention", True),
    ),
)
def test_only_cross_attention_arms_depend_on_earlier_trunk_states(variant: str, uses_history: bool) -> None:
    cfg = _tiny_cfg(decoder_variant=variant)
    torch.manual_seed(101)
    model = exp.GPT(cfg)
    hidden, ctx_pad, observed, targets = _decoder_inputs(cfg)
    changed = hidden.clone()
    changed[:, 0] += 3.0

    before = model.temporal.teacher_forced_logits_by_group(hidden, ctx_pad, observed, targets)
    after = model.temporal.teacher_forced_logits_by_group(changed, ctx_pad, observed, targets)
    largest_change = max(
        float((before[name] - after[name]).detach().abs().max()) for name in before if name != "buttons"
    )

    assert (largest_change > 1e-7) is uses_history


@pytest.mark.parametrize(
    "variant",
    ("baseline", "add-history", "remove-state-bias", "replace-attention"),
)
def test_gradients_follow_the_registered_decoder_path(variant: str) -> None:
    cfg = _tiny_cfg(decoder_variant=variant)
    torch.manual_seed(103)
    model = exp.GPT(cfg)
    hidden, ctx_pad, observed, targets = _decoder_inputs(cfg)

    loss = model.temporal.teacher_forced_nll(hidden, ctx_pad, observed, targets).mean()
    assert torch.isfinite(loss)
    loss.backward()

    state_width = cfg.arch.d_model
    state_gradient = model.temporal.token_projection.weight.grad[:, :state_width]
    assert (bool(state_gradient.abs().max() > 0)) is (variant in ("baseline", "add-history"))
    for block in model.temporal.blocks:
        self_gradient = block.qkv.weight.grad
        assert (self_gradient is not None) is (variant != "replace-attention")
        if self_gradient is not None:
            assert self_gradient.abs().max() > 0
    if variant == "baseline":
        assert not model.temporal.history_cross_attentions
    else:
        expected_cross_layers = cfg.arch.temporal_layers if variant == "replace-attention" else 1
        assert len(model.temporal.history_cross_attentions) == expected_cross_layers
        for cross_attention in model.temporal.history_cross_attentions:
            cross_gradient = cross_attention.query.weight.grad
            assert cross_gradient is not None
            assert cross_gradient.abs().max() > 0


def _match_row(boot: int, net_stocks: int, *, ego_character: int = 1):
    return exp.MatchRow(
        ego_character=ego_character,
        opp_character=2,
        stage=3,
        boot_index=boot,
        match_ordinal=0,
        active_frames=3600,
        total_frames=3723,
        damage_dealt=0.0,
        damage_taken=0.0,
        stocks_taken=max(net_stocks, 0),
        stocks_lost=max(-net_stocks, 0),
    )


def test_paired_analysis_resamples_matched_boots() -> None:
    control = [_match_row(boot, 0) for boot in range(4)]
    treatment = [_match_row(boot, 1) for boot in range(4)]

    result = exp.paired_net_stock_delta(control, treatment, bootstrap_resamples=2000, seed=11)

    assert result == {
        "boots": 4.0,
        "control_net_stock_per_min": 0.0,
        "treatment_net_stock_per_min": 1.0,
        "net_stock_delta": 1.0,
        "net_stock_delta_ci_lo": 1.0,
        "net_stock_delta_ci_hi": 1.0,
    }


def test_paired_analysis_rejects_a_changed_matchup() -> None:
    control = [_match_row(0, 0)]
    treatment = [_match_row(0, 1, ego_character=4)]

    with pytest.raises(ValueError, match="matchup differs"):
        exp.paired_net_stock_delta(control, treatment)


def test_match_row_loader_rejects_protocol_schema_drift(tmp_path: Path) -> None:
    evidence = tmp_path / "match_rows.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": exp._MATCH_ROW_SCHEMA_VERSION,
                "protocol": {},
                "rows": [],
            }
        )
    )

    with pytest.raises(ValueError, match="protocol fields differ"):
        exp._load_match_row_evidence(evidence)


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
        valid_prefixes=cfg.batch_size,
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
            valid_prefixes=cfg.batch_size,
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
    run_name = exp.make_run_name(
        "055_history_cross_attention",
        tag,
        "ranked-anonymized-1/policy-world-v8",
        "o55-d-replace-attention",
    )

    assert state["experiment_id"] == "055_history_cross_attention_v1"
    assert exp.config_from_state(state) == cfg
    assert "all-adamw" in tag
    assert "baseline-final-prefix" in tag
    assert "alr0.0017" in tag
    assert "ranked-anon-1" in run_name
    assert len(run_name.encode()) <= 255
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
