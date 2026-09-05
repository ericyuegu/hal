"""O51 parameterization, optimization, and launch contracts."""

import gc
import importlib.util
import math
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch


def _load() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "experiments" / "051_muon_parameterization.py"
    spec = importlib.util.spec_from_file_location("test_exp051", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp = _load()


@pytest.mark.parametrize(
    "command",
    ["train", "audit-data", "loader-benchmark"],
)
def test_nested_config_command_help_is_parseable(command: str) -> None:
    script = Path(exp.__file__).resolve()
    result = subprocess.run(
        [sys.executable, str(script), command, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_full_run_does_not_require_throughput_preflight_evidence(monkeypatch) -> None:
    cfg = exp.config_for(
        "mid",
        target_positions=8 * exp.D0,
        tier_scale=8,
        push_to_r2=False,
    )
    observed: dict[str, object] = {}

    monkeypatch.setattr(exp, "load_stats", lambda _cfg: {})

    def train(selected_cfg, _stats, **_kwargs) -> None:
        observed["tier_scale"] = selected_cfg.tier_scale
        observed["module_file"] = exp.__file__

    monkeypatch.setattr(exp, "train", train)

    exp._run_train(exp.TrainArgs(cfg=cfg))

    assert observed == {"tier_scale": 8, "module_file": exp.__file__}


def test_prepared_o51_data_starts_iterator_before_background_next(monkeypatch) -> None:
    events: list[tuple[str, int]] = []
    batch = object()

    class Loader:
        def __iter__(self):
            events.append(("iter", threading.get_ident()))
            return self

        def __next__(self):
            events.append(("next", threading.get_ident()))
            return batch

    loader = Loader()
    monkeypatch.setattr(exp, "_make_loaders", lambda *_args: (loader, []))
    sidecar = SimpleNamespace(by_replay={})

    prepared = exp._prepare_training_data(exp.TrainConfig(), {}, sidecar, None)
    try:
        assert prepared.first_batch_future.result() is batch
    finally:
        prepared.executor.shutdown(wait=True)

    assert events[0] == ("iter", threading.main_thread().ident)
    assert events[1][0] == "next"
    assert events[1][1] != threading.main_thread().ident


def test_prefetcher_applies_parent_transforms_once_per_batch() -> None:
    cfg = _tiny_cfg()
    batches = [
        exp.synthetic_awr_batch(cfg, torch.device("cpu")),
        exp.synthetic_awr_batch(cfg, torch.device("cpu")),
    ]
    transformed: list[exp.AWRBatch] = []

    def transform(batch):
        transformed.append(batch)
        return batch

    prefetcher = exp.DeviceBatchPrefetcher(batches, cfg, "cpu", transform)
    try:
        prefetcher.next()
        prefetcher.start_preload()
        prefetcher.finish_preload()
        prefetcher.next()
    finally:
        prefetcher.close()

    assert transformed == batches


def test_loader_benchmark_measures_direct_batches(monkeypatch) -> None:
    cfg = exp.TrainConfig()
    replay_ids = tuple(f"replay-{index}" for index in range(cfg.batch_size))
    train_batch = exp.TrainBatch(
        exp.Context({}, torch.zeros(cfg.batch_size, dtype=torch.int64)),
        torch.empty(cfg.batch_size, 0, 0),
        replay_ids,
    )
    batch = exp.AWRBatch(
        train_batch,
        torch.empty(cfg.batch_size, 0),
        torch.empty(cfg.batch_size, 0, dtype=torch.bool),
    )

    class Loader:
        source_sample_counts = {"source": cfg.batch_size}

        def __iter__(self):
            while True:
                yield batch

    sidecar = type("Sidecar", (), {"by_replay": {}})()
    monkeypatch.setattr(exp, "load_stats", lambda _cfg: {})
    monkeypatch.setattr(exp, "load_identity_sidecar", lambda _cfg: sidecar)
    monkeypatch.setattr(exp, "_make_train_loader", lambda *_args: Loader())
    monkeypatch.setattr(exp, "data_selection", lambda _cfg: object())
    report = exp.benchmark_loader(cfg, warmup_batches=1, measured_batches=2)

    assert report["loader_only_windows_per_s"] > 0
    assert report["distinct_replays"] == cfg.batch_size
    assert report["within_batch_unique"] is True
    assert report["cooldown_batches"] == 0
    assert report["identity_coverage_fraction"] == 1.0
    assert report["slot_frequency_spread"] == 0
    assert report["repeat_floor_passed"] is True
    assert report["steady_state_turnover_passed"] is True
    assert report["shuffle_passed"] is True


@pytest.mark.parametrize("level", ["base", "proxy", "mid", "large"])
def test_fresh_train_selects_requested_model_level(monkeypatch, level: str) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(exp, "load_stats", lambda _cfg: {})

    def train(selected_cfg, _stats, **_kwargs) -> None:
        observed["level"] = exp.model_level(selected_cfg.arch)

    monkeypatch.setattr(exp, "train", train)

    exp._run_train(
        exp.TrainArgs(
            cfg=exp.config_for("mid", push_to_r2=False),
            level=level,
            smoke=True,
            stop_after_update=1,
        )
    )

    assert observed == {"level": level}


def test_resume_rejects_conflicting_model_level(monkeypatch) -> None:
    cfg = exp.config_for("base", push_to_r2=False)
    monkeypatch.setattr(
        exp,
        "load_for_resume",
        lambda *_args, **_kwargs: {"cfg": exp._checkpoint_config(cfg)},
    )

    with pytest.raises(SystemExit, match="does not match resumed base"):
        exp._run_train(exp.TrainArgs(resume="run", level="large"))


def _tiny_cfg(**changes: object):
    architecture = replace(
        exp.ARCHITECTURE,
        d_model=32,
        n_layers=2,
        n_heads=4,
        L_ctx=8,
        temporal_d_model=32,
        temporal_layers=2,
        temporal_heads=4,
        temporal_ff_dim=64,
        group_head_dim=16,
        value_hidden_dim=16,
        item_hidden_dim=8,
        item_dim=5,
    )
    return exp.TrainConfig(
        arch=architecture,
        target_positions=256,
        batch_size=2,
        compile_trunk=False,
        compile_temporal=False,
        inference_mode="eager",
        num_workers=0,
        push_to_r2=False,
        **changes,
    )


def test_all_four_model_sizes_are_exact_without_allocating_weights() -> None:
    for level, expected in exp.EXPECTED_PARAMETER_COUNTS.items():
        with torch.device("meta"):
            model = exp.Policy(exp.config_for(level))
        assert exp.subsystem_parameter_counts(model)["total"] == expected
        del model
        gc.collect()


def test_optimizer_roles_cover_every_tensor_by_semantics() -> None:
    cfg = exp.config_for("base")
    with torch.device("meta"):
        model = exp.Policy(cfg)
    roles = exp.optimizer_roles(model, cfg)

    assert set(roles) == dict(model.named_parameters()).keys()
    assert roles["trunk.blocks.0.attn.c_attn.weight"] == exp.OptimizerRole("muon", "hidden", True, logical_splits=3)
    assert roles["temporal.blocks.0.qkv.weight"] == exp.OptimizerRole("muon", "hidden", True, logical_splits=3)
    assert roles["temporal.token_projection.weight"].optimizer == "muon"
    assert roles["temporal.token_projection.bias"].optimizer == "adamw"
    assert roles["temporal.outputs.buttons.up.weight"].optimizer == "muon"
    assert roles["temporal.outputs.buttons.down.weight"].lr_kind == "output"
    assert roles["value_head.up.weight"].logical_splits == 2
    for prefix in (
        "codec.semantic_projections.",
        "observation_encoder.",
        "player_projection.",
        "temporal.group_condition.",
    ):
        assert all(role.optimizer == "adamw" for name, role in roles.items() if name.startswith(prefix))

    optimizer = exp.make_optimizer(model, cfg)
    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(grouped) == len({id(parameter) for parameter in grouped}) == len(roles)
    muon_groups = [group for group in optimizer.param_groups if group["use_muon"]]
    assert {group["muon_scale_clamp_min_one"] for group in muon_groups} == {False}
    assert {group["logical_splits"] for group in muon_groups} >= {1, 2, 3}
    assert optimizer._adam_diagnostic_names == {}


def test_stability_signals_run_only_on_the_logging_cadence() -> None:
    cfg = _tiny_cfg()
    model = exp.Policy(cfg)
    batch = exp.synthetic_awr_batch(cfg, torch.device("cpu"))
    _trunk_fn, temporal_fn = exp._training_functions(model, cfg)

    assert temporal_fn.__name__ == "teacher_forced_nll"
    assert exp._training_diagnostics(model, batch, cfg, 24) == {}
    metrics = exp._training_diagnostics(model, batch, cfg, 25)
    assert {
        "stability/action_pre_norm_rms",
        "stability/centered_logit_abs_p999",
        "stability/uncentered_button_logit_abs_p999",
    } <= metrics.keys()
    assert all(math.isfinite(value) for value in metrics.values())


@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_depth_rule_applies_to_training_and_cached_inference(alpha: float) -> None:
    base_trunk = exp.depth_rule("trunk", 8, alpha)
    base_temporal = exp.depth_rule("temporal", 2, alpha)
    assert base_trunk == exp.DepthRule(attention=0.25, mlp=1.0)
    assert base_temporal == exp.DepthRule(attention=0.5, mlp=1.0)

    deep = exp.depth_rule("trunk", 16, alpha)
    assert deep.attention == pytest.approx(0.25 * 2**-alpha)
    assert deep.mlp == pytest.approx(2**-alpha)

    cfg = _tiny_cfg(depth_alpha=alpha)
    block = exp.TemporalBlock(cfg).eval()
    values = torch.randn(3, 5, cfg.arch.temporal_d_model)
    full = block(values)
    cache = None
    cached = []
    for index in range(values.shape[1]):
        output, cache = block.forward_step(values[:, index], cache)
        cached.append(output)
    torch.testing.assert_close(torch.stack(cached, dim=1), full, rtol=2e-5, atol=2e-5)
    rule = exp.depth_rule("temporal", cfg.arch.temporal_layers, alpha)
    assert (block.scale, block.mlp_scale) == pytest.approx((rule.attention, rule.mlp))


@pytest.mark.parametrize("hidden_multiplier", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("readout", ["zero", "mup-normal"])
def test_six_initialization_arms_only_zero_designated_readouts(
    hidden_multiplier: float,
    readout: str,
) -> None:
    torch.manual_seed(7)
    cfg = _tiny_cfg(hidden_std_multiplier=hidden_multiplier, readout_init=readout)
    model = exp.Policy(cfg)
    final_weights = [module.weight for module, _ in exp._final_readouts(model)]

    if readout == "zero":
        assert all(torch.count_nonzero(weight) == 0 for weight in final_weights)
    else:
        assert all(torch.count_nonzero(weight) > 0 for weight in final_weights)
    assert torch.count_nonzero(model.player_projection.weight) == 0
    assert torch.count_nonzero(model.temporal.blocks[0].qkv.weight) > 0
    assert all(torch.count_nonzero(module.weight) > 0 for module in model.temporal.group_condition.values())


def test_hidden_initialization_has_no_depth_factor_and_mup_readout_scales_by_inverse_width() -> None:
    torch.manual_seed(13)
    half = exp.Policy(_tiny_cfg(hidden_std_multiplier=0.5))
    torch.manual_seed(13)
    full = exp.Policy(_tiny_cfg(hidden_std_multiplier=1.0))
    torch.testing.assert_close(
        2 * half.temporal.blocks[0].qkv.weight,
        full.temporal.blocks[0].qkv.weight,
    )
    assert exp.mup_readout_std(256, 128) == pytest.approx(exp.mup_readout_std(128, 128) / 2)


def test_centering_preserves_policy_and_removes_each_group_common_mode() -> None:
    logits = torch.randn(4, 7) * 9 + 100
    centered = exp.center_class_logits(logits)

    torch.testing.assert_close(centered.mean(dim=-1), torch.zeros(4), atol=2e-5, rtol=0)
    torch.testing.assert_close(centered.softmax(-1), logits.softmax(-1))
    assert torch.equal(centered.argmax(-1), logits.argmax(-1))


def test_training_nll_skips_centering_without_changing_loss_or_gradient() -> None:
    decoder = exp.Policy(_tiny_cfg(readout_init="mup-normal")).temporal
    batch, positions = 2, 3
    hidden = torch.randn(batch, positions, decoder.trunk_width, requires_grad=True)
    observed = torch.zeros(batch, positions, exp.N_GROUPS, dtype=torch.long)
    targets = torch.zeros(
        batch,
        positions,
        len(decoder.head_offsets),
        exp.N_GROUPS,
        dtype=torch.long,
    )

    raw_nll = decoder.teacher_forced_nll(hidden, observed, targets)
    raw_gradient = torch.autograd.grad(raw_nll.sum(), hidden, retain_graph=True)[0]
    centered_logits = decoder.teacher_forced_logits_by_group(hidden, observed, targets)
    centered_nll = decoder.nll_from_logits(centered_logits, targets)
    centered_gradient = torch.autograd.grad(centered_nll.sum(), hidden)[0]

    torch.testing.assert_close(raw_nll, centered_nll)
    torch.testing.assert_close(raw_gradient, centered_gradient, atol=2e-5, rtol=2e-5)


def test_batch_duration_lr_beta_epsilon_and_decay_formulas() -> None:
    defaults = exp.TrainConfig()
    assert defaults.adam_eps == 1e-12
    runtime_fields = {field.name for field in exp.fields(exp.TrainConfig)}
    assert not runtime_fields.intersection(
        {
            "cache_limit_gb",
            "shuffle_block_size",
            "predownload",
            "download_retry",
            "windows_per_replay",
            "reservoir_capacity",
            "replay_cooldown_batches",
            "replay_pack_batch_size",
            "loader_prefetch_factor",
            "shuffle_algo",
        }
    )
    assert exp.REPLAY_SLOTS_BY_TIER == {
        1: 114_688,
        2: 131_072,
        4: 131_072,
        8: 131_072,
    }
    assert defaults.num_workers == 24
    assert exp.scaled_adam_lr(4.25e-4, batch_multiplier=1, duration_multiplier=1) == 4.25e-4
    assert exp.scaled_adam_lr(
        4.25e-4,
        batch_multiplier=2,
        duration_multiplier=8,
        fan_in_multiplier=4,
        output=True,
    ) == pytest.approx(4.25e-4 * math.sqrt(2 / 8) / 4)
    assert exp.scaled_adam_betas((0.9, 0.95), batch_multiplier=2, duration_multiplier=4) == pytest.approx(
        (0.95, 0.975)
    )
    assert exp.scaled_adam_epsilon(1e-12, batch_multiplier=2, duration_multiplier=8) == pytest.approx(2e-12)

    fixed = exp.config_for("base", target_positions=4 * exp.D0, tier_scale=4)
    assert exp.muon_lr_multiplier(fixed) == 1
    inverse = replace(fixed, muon_duration_scaling="inverse-sqrt")
    assert exp.muon_lr_multiplier(inverse) == pytest.approx(0.5)
    batch_scaled = replace(inverse, batch_size=1024, muon_batch_scaling="sqrt")
    assert exp.muon_lr_multiplier(batch_scaled) == pytest.approx(math.sqrt(2) / 2)
    muon_role = exp.OptimizerRole("muon", "hidden", True)
    assert exp._role_lr(muon_role, exp.config_for("base", muon_lr=0.014)) == pytest.approx(0.014)
    assert exp._role_lr(muon_role, exp.config_for("proxy", muon_lr=0.014)) == pytest.approx(0.014)
    adam_decay, muon_decay = exp.scaled_weight_decays(batch_scaled)
    assert adam_decay == pytest.approx(batch_scaled.adam_weight_decay * math.sqrt(2 / 4))
    assert muon_decay == pytest.approx(batch_scaled.muon_weight_decay * 2 / (4 * math.sqrt(2 / 4)))


@pytest.mark.parametrize(
    ("scale", "updates", "warmup"),
    [(1, 16_384, 512), (2, 32_768, 1024), (4, 65_536, 2048), (8, 131_072, 4096)],
)
def test_position_schedule_and_nested_endpoint_are_exact(scale: int, updates: int, warmup: int) -> None:
    cfg = exp.config_for("proxy", target_positions=scale * exp.D0, tier_scale=scale)
    exp.validate_config(cfg)
    assert (cfg.max_steps, cfg.warmup_steps) == (updates, warmup)
    assert exp.lr_schedule(cfg)(0) == pytest.approx(1 / warmup)
    assert exp.lr_schedule(cfg)(cfg.max_steps - 1) == pytest.approx(1 / 170)
    with pytest.raises(ValueError, match="matched nested"):
        exp.validate_config(replace(cfg, tier_scale=1 if scale != 1 else 2))


def test_o50_task_awr_and_validation_cohort_remain_frozen() -> None:
    cfg = exp.config_for("base")
    with pytest.raises(ValueError, match="architecture fields"):
        exp.validate_config(replace(cfg, arch=replace(cfg.arch, head_offsets=(1, 2, 3, 4, 5, 6))))
    with pytest.raises(ValueError, match="AWR objective"):
        exp.validate_config(replace(cfg, awr=replace(cfg.awr, beta=cfg.awr.beta / 2)))
    with pytest.raises(ValueError, match="task or evaluation"):
        exp.validate_config(replace(cfg, val_n_samples=cfg.val_n_samples // 2))
    with pytest.raises(ValueError, match="fixed base Adam"):
        exp.validate_config(replace(cfg, adam_eps=1e-8))


def test_production_loader_and_compilation_choices_use_supported_values() -> None:
    cfg = exp.config_for("base")
    exp.validate_production_config(cfg)
    with pytest.raises(ValueError, match="num_workers"):
        exp.validate_production_config(replace(cfg, num_workers=7))
    with pytest.raises(ValueError, match="compiled"):
        exp.validate_production_config(replace(cfg, compile_temporal=False))


def test_checkpoint_identity_prevents_o50_or_changed_schedule_resume() -> None:
    cfg = exp.config_for("base", target_positions=exp.D0 // 8, tier_scale=1)
    state = exp._checkpoint_config(cfg)
    assert exp.config_from_state(state) == cfg

    wrong_experiment = {**state, "experiment_id": "050_scaled_temporal_awr_v1"}
    with pytest.raises(ValueError, match="experiment_id"):
        exp.config_from_state(wrong_experiment)
    changed_schedule = {**state, "max_steps": int(state["max_steps"]) + 1}
    with pytest.raises(ValueError, match="not derivable"):
        exp.config_from_state(changed_schedule)


def test_older_o51_checkpoint_is_rejected_for_resume_and_evaluation() -> None:
    cfg = exp.config_for("base")
    legacy = exp._checkpoint_config(cfg)
    legacy["experiment_id"] = "051_muon_parameterization_v9"

    with pytest.raises(ValueError, match="experiment_id"):
        exp.config_from_state(legacy)
    with pytest.raises(ValueError, match="experiment_id"):
        exp._config_from_eval_state(legacy)


def test_arm_guard_rejects_or_pauses_exact_threshold_violations() -> None:
    nonfinite = exp.arm_decision(
        {"train/loss": math.nan},
        post_warmup_clip_fraction=0.0,
    )
    reject = exp.arm_decision(
        {"stability/centered_logit_abs_p999": 65.0},
        post_warmup_clip_fraction=0.0,
    )
    pause = exp.arm_decision(
        {
            "stability/centered_logit_abs_p999": 4.0,
            "stability/uncentered_button_logit_abs_p999": 129.0,
        },
        post_warmup_clip_fraction=0.0,
    )
    assert nonfinite == exp.ArmDecision("reject", ("non-finite metrics: ['train/loss']",))
    assert reject.status == "reject"
    assert pause.status == "pause"

    guard = exp._ArmGuard(warmup_updates=10, final_update=20)
    assert guard.observe({"global_step": 10, "stability/centered_logit_abs_p999": 1.0}).status == "pass"
    decision = guard.observe(
        {
            "global_step": 20,
            "stability/centered_logit_abs_p999": 1.0,
            "optimizer/clip_fraction": 0.11,
        }
    )
    assert decision.status == "reject"

    warmup_growth = exp._ArmGuard(warmup_updates=10, final_update=100)
    for update in range(1, 10):
        assert (
            warmup_growth.observe(
                {
                    "global_step": update,
                    "stability/centered_logit_abs_p999": float(update),
                }
            ).status
            == "pass"
        )
    assert (
        warmup_growth.observe(
            {
                "global_step": 10,
                "stability/centered_logit_abs_p999": 10.0,
            }
        ).status
        == "pass"
    )
    for update in range(11, 14):
        assert (
            warmup_growth.observe(
                {
                    "global_step": update,
                    "stability/centered_logit_abs_p999": 40.0,
                }
            ).status
            == "pass"
        )
    assert (
        warmup_growth.observe(
            {
                "global_step": 14,
                "stability/centered_logit_abs_p999": 40.0,
            }
        ).status
        == "reject"
    )

    endpoint_clipping = exp._ArmGuard(warmup_updates=10, final_update=30)
    assert (
        endpoint_clipping.observe(
            {
                "global_step": 10,
                "stability/centered_logit_abs_p999": 1.0,
            }
        ).status
        == "pass"
    )
    assert (
        endpoint_clipping.observe(
            {
                "global_step": 20,
                "optimizer/clip_fraction": 1.0,
                "stability/centered_logit_abs_p999": 1.0,
            }
        ).status
        == "pass"
    )
    assert endpoint_clipping.observe({"global_step": 30}).status == "reject"

    alternating = exp._ArmGuard(warmup_updates=0, final_update=100)
    assert (
        alternating.observe(
            {
                "global_step": 1,
                "stability/centered_logit_abs_p999": 1.0,
                "stability/action_pre_norm_rms": 1.0,
            }
        ).status
        == "pass"
    )
    for update in range(2, 8):
        centered, rms = (4.0, 1.0) if update % 2 == 0 else (1.0, 4.0)
        assert (
            alternating.observe(
                {
                    "global_step": update,
                    "stability/centered_logit_abs_p999": centered,
                    "stability/action_pre_norm_rms": rms,
                }
            ).status
            == "pass"
        )


def test_wandb_log_applies_the_arm_guard_without_patching_wandb(monkeypatch) -> None:
    logged: list[dict[str, object]] = []
    run = SimpleNamespace(summary={}, log_code=lambda **_kwargs: None)

    def rebound_log(values: dict[str, object], *_args: object, **_kwargs: object) -> None:
        logged.append(values)

    def init(**_kwargs: object) -> None:
        monkeypatch.setattr(exp.wandb, "log", rebound_log)
        monkeypatch.setattr(exp.wandb, "run", run)

    monkeypatch.setattr(exp.wandb, "init", init)
    monkeypatch.setattr(exp.wandb, "define_metric", lambda *_args, **_kwargs: None)
    exp._init_wandb(exp.config_for("base"), "guard-test", None)

    with pytest.raises(RuntimeError, match="raw button-logit"):
        exp._log_wandb(
            {
                "global_step": 25,
                "stability/centered_logit_abs_p999": 1.0,
                "stability/uncentered_button_logit_abs_p999": 129.0,
            },
            exp._ArmGuard(warmup_updates=0, final_update=100),
        )
    assert logged[-1]["global_step"] == 25


def test_large_promotion_gate_requires_every_condition() -> None:
    evidence = exp.PromotionEvidence(3, 0.10, 0.001, 5.0, 1.01, 1.0)
    exp.validate_large_promotion(evidence)
    with pytest.raises(ValueError, match="paired"):
        exp.validate_large_promotion(replace(evidence, paired_90_lower_bound=0.0))
    with pytest.raises(ValueError, match="non-finite"):
        exp.validate_large_promotion(replace(evidence, stock_per_min_gain=math.nan))
