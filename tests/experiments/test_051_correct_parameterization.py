"""O51 parameterization, optimization, and launch contracts."""

import gc
import importlib.util
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch

from hal.training.o51_data import CorpusSelection
from hal.training.o51_data import SourceSlice
from hal.training.o51_data import TierSelection


def _load() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "experiments" / "051_correct_parameterization.py"
    spec = importlib.util.spec_from_file_location("test_exp051", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp = _load()


@pytest.mark.parametrize(
    "command",
    ["train", "audit-data", "preflight", "loader-benchmark", "collect-soak"],
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


def test_full_tier_smoke_can_collect_preflight_data_without_prior_evidence(monkeypatch) -> None:
    cfg = exp.config_for(
        "mid",
        target_positions=8 * exp.D0,
        tier_scale=8,
        push_to_r2=False,
    )
    original_o50_file = exp._o50.__file__
    observed: dict[str, object] = {}

    monkeypatch.setattr(exp, "load_stats", lambda _cfg: {})
    monkeypatch.setattr(
        exp,
        "_require_launch_evidence",
        lambda *_args: pytest.fail("smoke collection requested prior preflight evidence"),
    )

    def train(selected_cfg, _stats, **_kwargs) -> None:
        observed["tier_scale"] = selected_cfg.tier_scale
        observed["module_file"] = exp._o50.__file__

    monkeypatch.setattr(exp._o50, "train", train)

    exp._run_train(exp.TrainArgs(cfg=cfg, smoke=True, stop_after_update=1))

    assert observed == {"tier_scale": 8, "module_file": exp.__file__}
    assert exp._o50.__file__ == original_o50_file


def test_loader_benchmark_measures_direct_batches(monkeypatch) -> None:
    cfg = exp.TrainConfig()
    replay_ids = tuple(f"replay-{index}" for index in range(cfg.batch_size))
    train_batch = exp._o50.TrainBatch(
        exp._o50.Context({}, torch.zeros(cfg.batch_size, dtype=torch.int64)),
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
    monkeypatch.setattr(exp._o50, "load_identity_sidecar", lambda _cfg: sidecar)
    monkeypatch.setattr(exp, "_make_train_loader", lambda *_args: Loader())
    monkeypatch.setattr(exp, "data_selection", lambda _cfg: object())
    monkeypatch.setattr(exp, "preflight_fingerprint", lambda *_args: "f" * 64)

    report = exp.benchmark_loader(cfg, warmup_batches=1, measured_batches=2)

    assert report["loader_only_windows_per_s"] > 0
    assert report["distinct_replays"] == cfg.batch_size
    assert report["within_batch_unique"] is True
    assert report["cooldown_passed"] is False


@pytest.mark.parametrize("level", ["base", "proxy", "mid", "large"])
def test_fresh_train_selects_requested_model_level(monkeypatch, level: str) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(exp, "load_stats", lambda _cfg: {})

    def train(selected_cfg, _stats, **_kwargs) -> None:
        observed["level"] = exp.model_level(selected_cfg.arch)

    monkeypatch.setattr(exp._o50, "train", train)

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
        exp._o50,
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
            model = exp.GPT(exp.config_for(level))
        assert exp.subsystem_parameter_counts(model)["total"] == expected
        del model
        gc.collect()


def test_optimizer_roles_cover_every_tensor_by_semantics() -> None:
    cfg = exp.config_for("base")
    with torch.device("meta"):
        model = exp.GPT(cfg)
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
    assert {group["muon_scale_mode"] for group in muon_groups} == {"o51"}
    assert {group["logical_splits"] for group in muon_groups} >= {1, 2, 3}
    assert optimizer._adam_diagnostic_names == {}


def test_stability_signals_run_only_on_the_logging_cadence() -> None:
    cfg = _tiny_cfg()
    model = exp.GPT(cfg)
    batch = exp._o50.synthetic_awr_batch(cfg, torch.device("cpu"))
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
    model = exp.GPT(cfg)
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
    half = exp.GPT(_tiny_cfg(hidden_std_multiplier=0.5))
    torch.manual_seed(13)
    full = exp.GPT(_tiny_cfg(hidden_std_multiplier=1.0))
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
    decoder = exp.GPT(_tiny_cfg(readout_init="mup-normal")).temporal
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
    assert defaults.cache_limit_gb == 1700
    assert defaults.replay_cooldown_batches == 16
    assert defaults.reservoir_capacity == 17 * defaults.batch_size
    assert replace(defaults, batch_size=1024).reservoir_capacity == 17_408
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


def test_production_loader_and_compilation_choices_come_from_preflight_grid() -> None:
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


def test_arm_guard_rejects_or_pauses_exact_threshold_violations() -> None:
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


def test_wandb_guard_is_installed_after_init_rebinds_log(monkeypatch) -> None:
    logged: list[dict[str, object]] = []
    run = SimpleNamespace(summary={}, log_code=lambda **_kwargs: None)

    def rebound_log(values: dict[str, object], *_args: object, **_kwargs: object) -> None:
        logged.append(values)

    def init(**_kwargs: object) -> None:
        monkeypatch.setattr(exp._o50.wandb, "log", rebound_log)
        monkeypatch.setattr(exp._o50.wandb, "run", run)

    monkeypatch.setattr(exp._o50.wandb, "init", init)
    monkeypatch.setattr(exp._o50.wandb, "define_metric", lambda *_args, **_kwargs: None)
    exp._init_wandb(exp.config_for("base"), "guard-test", None)

    with pytest.raises(RuntimeError, match="raw button-logit"):
        exp._o50.wandb.log(
            {
                "global_step": 25,
                "stability/centered_logit_abs_p999": 1.0,
                "stability/uncentered_button_logit_abs_p999": 129.0,
            }
        )
    assert logged[-1]["global_step"] == 25


def test_large_promotion_gate_requires_every_condition() -> None:
    evidence = exp.PromotionEvidence(3, 0.10, 0.001, 5.0, 1.01, 1.0)
    exp.validate_large_promotion(evidence)
    with pytest.raises(ValueError, match="paired"):
        exp.validate_large_promotion(replace(evidence, paired_90_lower_bound=0.0))
    with pytest.raises(ValueError, match="non-finite"):
        exp.validate_large_promotion(replace(evidence, stock_per_min_gain=math.nan))


def _selection() -> CorpusSelection:
    source_names = [source.name for source in exp.streams.POLICY_WORLD_V7_SOURCES]
    tiers = {}
    for scale in exp.TIER_SCALES:
        tiers[scale] = TierSelection(
            scale=scale,
            sources=tuple(SourceSlice(source=name, stop=scale) for name in source_names),
            potential_targets=2 * scale * len(source_names),
            sha256=f"{scale:x}" * 64,
        )
    return CorpusSelection(
        corpus_hash="a" * 64,
        source_manifest_sha256={name: "b" * 64 for name in source_names},
        tiers=tiers,
    )


def _passing_preflight(cfg, selection):
    return exp.PreflightReport(
        fingerprint=exp.preflight_fingerprint(cfg, selection),
        synthetic_mfu=0.15,
        compute_only_updates_per_s=10.0,
        loader_only_windows_per_s=1250.0,
        raw_bytes_per_window=35.0,
        o50_raw_bytes_per_window=100.0,
        peak_memory_fraction=0.949,
        graph_gap_fraction=0.05,
        optimizer_time_fraction=0.10,
        disk_capacity_bytes=2 * 2**40,
        exact_resume=True,
        memory_passed=True,
        shuffle_passed=True,
        telemetry={name: 0.0 for name in exp.REQUIRED_PREFLIGHT_TELEMETRY},
    )


def test_preflight_enforces_every_launch_and_cache_gate(monkeypatch) -> None:
    cfg = exp.config_for("base")
    selection = _selection()
    monkeypatch.setattr(exp, "data_selection", lambda _cfg: selection)
    report = _passing_preflight(cfg, selection)

    assert exp.preflight_failures(cfg, report) == ()
    assert exp.preflight_failures(cfg, replace(report, synthetic_mfu=0.0)) == ()
    assert "telemetry" in " ".join(exp.preflight_failures(cfg, replace(report, telemetry={})))
    assert "invalid" in " ".join(exp.preflight_failures(cfg, replace(report, synthetic_mfu=math.nan)))
    assert "not boolean" in " ".join(
        exp.preflight_failures(cfg, replace(report, exact_resume="false"))  # type: ignore[arg-type]
    )
    assert "256 GiB" in " ".join(
        exp.preflight_failures(
            cfg,
            replace(report, disk_capacity_bytes=1900 * 2**30),
        )
    )
    exp.validate_shuffle_config("py1e", 4096)

    mid_cfg = exp.config_for("mid")
    mid_report = replace(report, fingerprint=exp.preflight_fingerprint(mid_cfg, selection))
    assert "55M synthetic compiled MFU" in " ".join(
        exp.preflight_failures(mid_cfg, replace(mid_report, synthetic_mfu=0.149))
    )


def test_soak_is_a_separate_full_u8_gate(monkeypatch) -> None:
    cfg = exp.config_for("mid", target_positions=8 * exp.D0, tier_scale=8)
    selection = _selection()
    monkeypatch.setattr(exp, "data_selection", lambda _cfg: selection)
    report = exp.SoakReport(
        **exp.asdict(_passing_preflight(cfg, selection)),
        tier_scale=8,
        end_to_end_updates_per_s=9.0,
        gpu_windows_per_s=1000.0,
        loader_wait_mean_fraction=0.049,
        loader_wait_p95_fraction=0.099,
        end_to_end_mfu=0.135,
        soak_seconds=7200.0,
        judged_seconds=1800.0,
    )

    assert exp.soak_failures(cfg, report) == ()
    assert "two hours" in " ".join(exp.soak_failures(cfg, replace(report, soak_seconds=7199.0)))
    assert "end-to-end MFU" in " ".join(exp.soak_failures(cfg, replace(report, end_to_end_mfu=0.134)))
    assert "full U8" in " ".join(exp.soak_failures(cfg, replace(report, tier_scale=4)))
    assert "mean loader wait" in " ".join(exp.soak_failures(cfg, replace(report, loader_wait_mean_fraction=0.05)))


def test_soak_collector_uses_only_the_final_judgment_window(monkeypatch) -> None:
    cfg = exp.config_for("mid", target_positions=8 * exp.D0, tier_scale=8)
    selection = _selection()
    monkeypatch.setattr(exp, "data_selection", lambda _cfg: selection)
    preflight = _passing_preflight(cfg, selection)
    history = []
    for index, elapsed in enumerate(range(0, 7201, 900)):
        history.append(
            {
                **{name: 1.0 for name in exp.REQUIRED_PREFLIGHT_TELEMETRY},
                "global_step": index * 100,
                "progress/elapsed_s": elapsed,
                "throughput/update_s": 0.1,
                "throughput/mfu_wall_clock": 0.01 if elapsed < 5400 else 0.14,
                "loader/wait_s": 0.004,
            }
        )

    report = exp.collect_soak_report(cfg, preflight, history)

    assert report.end_to_end_mfu == pytest.approx(0.14)
    assert report.end_to_end_updates_per_s == pytest.approx(10.0)
    assert report.gpu_windows_per_s == pytest.approx(10 * cfg.batch_size)
    assert report.loader_wait_mean_fraction == pytest.approx(0.04)
    assert report.soak_seconds == 7200
    assert report.judged_seconds == 1800


def test_preflight_fingerprint_binds_direct_prefixes() -> None:
    cfg = exp.TrainConfig()
    selection = _selection()
    changed_source = replace(selection.tier(8).sources[0], stop=9)
    changed_tier = replace(
        selection.tier(8),
        sources=(changed_source, *selection.tier(8).sources[1:]),
    )
    changed = replace(selection, tiers={**selection.tiers, 8: changed_tier})

    assert exp.preflight_fingerprint(cfg, selection) != exp.preflight_fingerprint(cfg, changed)
