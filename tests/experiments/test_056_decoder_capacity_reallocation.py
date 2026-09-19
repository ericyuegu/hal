"""Contracts for the O56 decoder-capacity reallocation."""

import copy
import importlib.util
import json
import sys
from dataclasses import asdict
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load():
    path = Path(__file__).resolve().parents[2] / "experiments" / "056_decoder_capacity_reallocation.py"
    name = "test_exp056"
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

    assert proxy.arch.parameter_count_contract == {
        "trunk": 8_650_752,
        "temporal_decoder": 1_638_576,
        "group_heads": 306_851,
        "value_head": 65_665,
        "other": 861_382,
        "total": 11_523_226,
    }
    assert proxy.target_positions == 2**30
    assert proxy.max_steps == 16_384
    assert proxy.adam_lr == 1.7e-3
    assert cfg.source_names == ("ranked-anonymized-1-policy-world-v8",)
    assert cfg.train_replays == 112_188
    assert cfg.train_frames == 1_203_888_017
    assert cfg.val_n_samples == 1024
    assert cfg.replay_slots == 112_128
    assert cfg.replay_slots <= cfg.train_replays
    assert cfg.replay_slots // cfg.batch_size - cfg.replay_phase_block_batches + 1 == 195
    assert cfg.minimum_replay_gap_batches == 195
    assert exp.data_selection(cfg).sha256 == cfg.selection_sha256
    assert exp.source_manifest_sha256(cfg) == {
        "ranked-anonymized-1-policy-world-v8": "b97eab90e761bcf2bf03b48981f0ab6acc1ac3057157c58ae0c5a72c76c43bd8"
    }


def test_capacity_control_and_effective_training_flops_are_fixed() -> None:
    cfg = exp.proxy_config()
    counts = exp.subsystem_parameter_counts(exp.GPT(cfg))
    control_counts = {
        "trunk": 12_582_912,
        "temporal_decoder": 768_752,
        "group_heads": 202_211,
        "value_head": 65_665,
        "other": 861_382,
    }

    assert exp.inference_parameter_uses(control_counts) == 17_328_146
    assert exp.inference_parameter_uses(counts) == 17_293_842
    assert exp.inference_parameter_uses(counts) / exp.inference_parameter_uses(control_counts) == pytest.approx(
        0.9980203230026688
    )
    control_flops = exp.approximate_training_flops_per_update(cfg, control_counts)
    treatment_flops = exp.approximate_training_flops_per_update(cfg, counts)
    assert control_flops == 14_416_825_417_728
    assert treatment_flops == 15_156_197_326_848
    assert treatment_flops / control_flops == pytest.approx(1.0512853480358313)


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

    assert state["experiment_id"] == "056_decoder_capacity_reallocation_v1"
    assert exp.config_from_state(state) == cfg
    assert "all-adamw" in tag
    assert "alr0.0017" in tag
    assert "muon" not in {field.name for field in fields(exp.TrainConfig)}
    assert cfg.automatic_evaluation is False


def test_control_checkpoint_identity_requires_explicit_eval_permission() -> None:
    cfg = exp.proxy_config()
    state = exp._checkpoint_config(cfg)
    state["experiment_id"] = "052_adamw_temporal_awr_v1"

    with pytest.raises(ValueError, match="052_adamw_temporal_awr_v1"):
        exp.config_from_state(state)

    assert exp.config_from_state(state, allow_control_checkpoint=True) == cfg


def test_control_checkpoint_permission_rejects_unknown_experiments() -> None:
    state = exp._checkpoint_config(exp.proxy_config())
    state["experiment_id"] = "unknown_experiment"

    with pytest.raises(ValueError, match="unknown_experiment"):
        exp.config_from_state(state, allow_control_checkpoint=True)


def test_proxy_smoke_uses_the_treatment_model(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = {}
    monkeypatch.setattr(exp, "load_stats", lambda _cfg: {})
    monkeypatch.setattr(exp, "train", lambda cfg, _stats, **kwargs: observed.update(cfg=cfg, kwargs=kwargs))

    exp.main(exp.TrainArgs(proxy=True, smoke=True, stop_after_update=100))

    assert observed["cfg"] == exp.proxy_config()
    assert observed["kwargs"]["proxy"] is True
    assert observed["kwargs"]["smoke"] is True
    assert observed["kwargs"]["stop_after_update"] == 100


def test_fresh_training_requires_the_fixed_proxy_arm() -> None:
    with pytest.raises(SystemExit, match="only the fixed proxy treatment"):
        exp.main(exp.TrainArgs())


def test_eval_protocol_versions_conditioned_transport() -> None:
    cfg = _tiny_cfg()
    protocol = exp._eval_protocol(
        cfg,
        exp.GPT(cfg),
        n_matchups=1,
        checkpoint_sha256="0" * 64,
        max_parallel=1,
    )

    assert exp._MATCH_ROW_SCHEMA_VERSION == 7
    assert protocol.transport_semantics == "conditioned_pending_actions_v1"
    assert protocol.pending_prefix_conditioned
    assert protocol.evaluation_protocol_version == 2
    assert not protocol.forced_prefix_consumes_sampling_draws


def test_eval_checkpoint_routes_through_portable_conditioned_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = exp.proxy_config()
    checkpoint = tmp_path / "final.pt"
    observed = {}
    source = SimpleNamespace(
        source_sha256="a" * 64,
        checkpoint_config=exp._checkpoint_config(cfg),
        step=16_383,
        wandb_id="treatment-run",
        player_id=lambda _identity: 4,
    )
    monkeypatch.setattr(exp, "load_o50_checkpoint", lambda path, *, device: source)

    def eval_vs_cpu(loaded_source, loaded_cfg, **kwargs):
        observed.update(source=loaded_source, cfg=loaded_cfg, kwargs=kwargs)
        return {"boots": 1.0, "crashed": 0.0}

    monkeypatch.setattr(exp, "eval_vs_cpu", eval_vs_cpu)
    monkeypatch.setattr(exp, "require_complete_eval", lambda _metrics, _expected: None)

    metrics = exp.eval_checkpoint(
        str(checkpoint),
        n_matchups=1,
        eager=True,
    )

    assert metrics == {"boots": 1.0, "crashed": 0.0}
    assert observed["source"] is source
    assert observed["cfg"] == cfg
    assert observed["kwargs"]["ego_player_id"] == 0
    assert observed["kwargs"]["ego_player_code"] is None
    assert observed["kwargs"]["replay_dir"] == tmp_path / "eval_replays_conditioned_v1"


def test_timing_eval_loads_checkpoint_once_and_runs_all_conditioned_horizons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = exp.proxy_config()
    checkpoint = tmp_path / "checkpoints" / "step-0016384.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    source = SimpleNamespace(
        source_sha256="a" * 64,
        checkpoint_config=exp._checkpoint_config(cfg) | {"experiment_id": exp._CONTROL_EXPERIMENT_ID},
        config=SimpleNamespace(experiment_id=exp._CONTROL_EXPERIMENT_ID),
        step=16_383,
        wandb_id="control-run",
    )
    loads: list[Path] = []
    calls: list[dict[str, object]] = []

    def load(path, *, device):
        loads.append(Path(path))
        return source

    def evaluate(loaded_source, loaded_cfg, **kwargs):
        assert loaded_source is source
        assert loaded_cfg == cfg
        calls.append(kwargs)
        return {
            "scheduled_boots": 96.0,
            "completed_boots": 96.0,
            "boots": 96.0,
            "crashed": 0.0,
        }

    monkeypatch.setattr(exp, "load_o50_checkpoint", load)
    monkeypatch.setattr(exp, "eval_vs_cpu", evaluate)
    monkeypatch.setattr(exp, "analyze_timing_evidence", lambda root, **_kwargs: {"root": str(root)})

    result = exp.eval_timing_checkpoint(
        str(checkpoint),
        upload_run=None,
        shared_wandb=False,
        expected_checkpoint_sha256="a" * 64,
    )

    assert loads == [checkpoint]
    assert [(call["prediction_frames"], call["replan_interval_frames"]) for call in calls] == [
        (3, 1),
        (4, 2),
        (5, 3),
        (6, 4),
    ]
    assert all(call["delay_frames"] == 2 and call["max_parallel"] == 32 for call in calls)
    assert result["root"] == str(tmp_path / exp._TIMING_EVIDENCE_NAME)


def test_zero_delay_timing_eval_loads_checkpoint_once_and_runs_h1_through_h3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = exp.proxy_config()
    checkpoint = tmp_path / "checkpoints" / "step-0016384.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    source = SimpleNamespace(
        source_sha256="a" * 64,
        checkpoint_config=exp._checkpoint_config(cfg) | {"experiment_id": exp._CONTROL_EXPERIMENT_ID},
        config=SimpleNamespace(experiment_id=exp._CONTROL_EXPERIMENT_ID),
        step=16_383,
        wandb_id="control-run",
    )
    loads: list[Path] = []
    calls: list[dict[str, object]] = []

    def load(path, *, device):
        loads.append(Path(path))
        return source

    def evaluate(loaded_source, loaded_cfg, **kwargs):
        assert loaded_source is source
        assert loaded_cfg == cfg
        calls.append(kwargs)
        return {
            "scheduled_boots": 96.0,
            "completed_boots": 96.0,
            "boots": 96.0,
            "crashed": 0.0,
        }

    monkeypatch.setattr(exp, "load_o50_checkpoint", load)
    monkeypatch.setattr(exp, "eval_vs_cpu", evaluate)
    monkeypatch.setattr(exp, "analyze_zero_delay_timing_evidence", lambda root, **_kwargs: {"root": str(root)})

    result = exp.eval_zero_delay_timing_checkpoint(
        str(checkpoint),
        upload_run=None,
        shared_wandb=False,
        expected_checkpoint_sha256="a" * 64,
    )

    assert loads == [checkpoint]
    assert [(call["prediction_frames"], call["replan_interval_frames"]) for call in calls] == [
        (1, 1),
        (2, 2),
        (3, 3),
    ]
    assert all(call["delay_frames"] == 0 and call["max_parallel"] == 32 for call in calls)
    assert result["root"] == str(tmp_path / exp._ZERO_DELAY_TIMING_EVIDENCE_NAME)


def test_paired_behavior_delta_resamples_matched_boots() -> None:
    control = [exp.BehaviorBoot((1, 2), 1, 1, 1.0, 0.0, 0.0, 1.0) for _ in range(96)]
    treatment = [exp.BehaviorBoot((1, 2), 2, 0, 1.0, 1.0, 10.0, 1.0) for _ in range(96)]
    sample = torch.arange(96).repeat(2000, 1).numpy()

    failed = exp._paired_ratio_delta(
        control,
        treatment,
        "failed_wavedash_attempts_per_min",
        sample=sample,
    )
    success = exp._paired_ratio_delta(control, treatment, "wavedash_success_rate", sample=sample)

    assert failed == {
        "control": 1.0,
        "treatment": 0.0,
        "delta": -1.0,
        "delta_ci_lo": -1.0,
        "delta_ci_hi": -1.0,
    }
    assert success == {
        "control": 0.5,
        "treatment": 1.0,
        "delta": 0.5,
        "delta_ci_lo": 0.5,
        "delta_ci_hi": 0.5,
    }


def test_timing_analysis_pairs_every_horizon_against_h4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_cfg()
    protocols = {
        horizon: asdict(
            exp._eval_protocol(
                cfg,
                exp.GPT(cfg),
                n_matchups=96,
                checkpoint_sha256="a" * 64,
                max_parallel=32,
                prediction_frames=horizon,
                delay_frames=2,
                replan_interval_frames=replan,
            )
        )
        for horizon, replan in exp._TIMING_CONFIGS
    }

    def horizon_for(path: Path) -> int:
        return int(path.name.rsplit("h", maxsplit=1)[1])

    monkeypatch.setattr(
        exp,
        "_load_eval_evidence",
        lambda path: (protocols[horizon_for(path)], []),
    )
    monkeypatch.setattr(
        exp,
        "_behavior_boots",
        lambda _path, protocol, _rows: {
            boot: exp.BehaviorBoot(
                (boot % 13, boot % 14),
                int(protocol["prediction_frames"]),
                6 - int(protocol["prediction_frames"]),
                1.0,
                float(int(protocol["prediction_frames"]) - 4),
                float(10 * (int(protocol["prediction_frames"]) - 4)),
                1.0,
            )
            for boot in range(96)
        },
    )
    monkeypatch.setattr(
        exp,
        "_load_eval_metrics",
        lambda path: {"decode_p95_ms": float(horizon_for(path)), "boots": 96.0},
    )
    monkeypatch.setattr(exp, "_eval_evidence_sha256", lambda path: str(horizon_for(path)) * 64)

    result = exp.analyze_timing_evidence(tmp_path, bootstrap_resamples=2_000, seed=0)

    assert result["checkpoint_sha256"] == "a" * 64
    assert result["evaluations"]["h6"]["decode_p95_ms"] == 6.0
    comparisons = result["comparisons"]
    assert set(comparisons) == {"h3_minus_h4", "h5_minus_h4", "h6_minus_h4"}
    assert comparisons["h6_minus_h4"]["metrics"]["net_stock_per_min"]["delta"] == 2.0


def test_timing_analysis_rejects_a_changed_invariant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_cfg()
    protocols = {
        horizon: asdict(
            exp._eval_protocol(
                cfg,
                exp.GPT(cfg),
                n_matchups=96,
                checkpoint_sha256="a" * 64,
                max_parallel=32,
                prediction_frames=horizon,
                delay_frames=2,
                replan_interval_frames=replan,
            )
        )
        for horizon, replan in exp._TIMING_CONFIGS
    }
    protocols[3]["seed"] = int(protocols[3]["seed"]) + 1

    def load(path: Path):
        horizon = int(path.name.rsplit("h", maxsplit=1)[1])
        return protocols[horizon], []

    monkeypatch.setattr(exp, "_load_eval_evidence", load)
    monkeypatch.setattr(exp, "_behavior_boots", lambda *_args: {})
    monkeypatch.setattr(exp, "_load_eval_metrics", lambda *_args: {})
    monkeypatch.setattr(exp, "_eval_evidence_sha256", lambda *_args: "0" * 64)

    with pytest.raises(ValueError, match="outside horizon and replan interval"):
        exp.analyze_timing_evidence(tmp_path)


def test_zero_delay_timing_analysis_pairs_h1_and_h3_against_h2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_cfg()
    protocols = {
        horizon: asdict(
            exp._eval_protocol(
                cfg,
                exp.GPT(cfg),
                n_matchups=96,
                checkpoint_sha256="a" * 64,
                max_parallel=32,
                prediction_frames=horizon,
                delay_frames=0,
                replan_interval_frames=replan,
            )
        )
        for horizon, replan in exp._ZERO_DELAY_TIMING_CONFIGS
    }

    def horizon_for(path: Path) -> int:
        return int(path.name.rsplit("h", maxsplit=1)[1])

    monkeypatch.setattr(exp, "_load_eval_evidence", lambda path: (protocols[horizon_for(path)], []))
    monkeypatch.setattr(
        exp,
        "_behavior_boots",
        lambda _path, protocol, _rows: {
            boot: exp.BehaviorBoot(
                (boot % 13, boot % 14),
                int(protocol["prediction_frames"]),
                4 - int(protocol["prediction_frames"]),
                1.0,
                float(int(protocol["prediction_frames"]) - 2),
                float(10 * (int(protocol["prediction_frames"]) - 2)),
                1.0,
            )
            for boot in range(96)
        },
    )
    monkeypatch.setattr(
        exp,
        "_load_eval_metrics",
        lambda path: {"decode_p95_ms": float(horizon_for(path)), "boots": 96.0},
    )
    monkeypatch.setattr(exp, "_eval_evidence_sha256", lambda path: str(horizon_for(path)) * 64)

    result = exp.analyze_zero_delay_timing_evidence(tmp_path, bootstrap_resamples=2_000, seed=0)

    assert result["checkpoint_sha256"] == "a" * 64
    assert result["reference_horizon"] == 2
    assert result["evaluations"]["h3"]["decode_p95_ms"] == 3.0
    comparisons = result["comparisons"]
    assert set(comparisons) == {"h1_minus_h2", "h3_minus_h2"}
    assert comparisons["h3_minus_h2"]["metrics"]["net_stock_per_min"]["delta"] == 1.0


def test_zero_delay_timing_analysis_rejects_a_changed_invariant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_cfg()
    protocols = {
        horizon: asdict(
            exp._eval_protocol(
                cfg,
                exp.GPT(cfg),
                n_matchups=96,
                checkpoint_sha256="a" * 64,
                max_parallel=32,
                prediction_frames=horizon,
                delay_frames=0,
                replan_interval_frames=replan,
            )
        )
        for horizon, replan in exp._ZERO_DELAY_TIMING_CONFIGS
    }
    protocols[1]["cpu_level"] = int(protocols[1]["cpu_level"]) - 1

    def load(path: Path):
        horizon = int(path.name.rsplit("h", maxsplit=1)[1])
        return protocols[horizon], []

    monkeypatch.setattr(exp, "_load_eval_evidence", load)
    monkeypatch.setattr(exp, "_behavior_boots", lambda *_args: {})
    monkeypatch.setattr(exp, "_load_eval_metrics", lambda *_args: {})
    monkeypatch.setattr(exp, "_eval_evidence_sha256", lambda *_args: "0" * 64)

    with pytest.raises(ValueError, match="outside horizon and replan interval"):
        exp.analyze_zero_delay_timing_evidence(tmp_path)


def test_evidence_loader_rejects_an_incomplete_boot_set(tmp_path: Path) -> None:
    protocol = {item.name: None for item in fields(exp.EvalProtocol)}
    protocol["n_matchups"] = 96
    row = exp.MatchRow(
        ego_character=1,
        opp_character=2,
        stage=3,
        boot_index=0,
        match_ordinal=0,
        active_frames=3600,
        total_frames=3723,
        damage_dealt=0.0,
        damage_taken=0.0,
        stocks_taken=0,
        stocks_lost=0,
    )
    (tmp_path / "match_rows.json").write_text(
        json.dumps(
            {
                "schema_version": exp._MATCH_ROW_SCHEMA_VERSION,
                "protocol": protocol,
                "rows": [row.as_dict()],
            }
        )
    )
    (tmp_path / "metrics.json").write_text("{}")

    with pytest.raises(ValueError, match="every boot 0..95"):
        exp._load_eval_evidence(tmp_path)


def test_behavior_analysis_rejects_missing_replays(tmp_path: Path) -> None:
    rows = [
        exp.MatchRow(
            ego_character=1,
            opp_character=2,
            stage=3,
            boot_index=boot,
            match_ordinal=0,
            active_frames=3600,
            total_frames=3723,
            damage_dealt=0.0,
            damage_taken=0.0,
            stocks_taken=0,
            stocks_lost=0,
        )
        for boot in range(96)
    ]

    with pytest.raises(ValueError, match="has 0 replays for 1 match rows"):
        exp._behavior_boots(tmp_path, {"ego_port": 1}, rows)


def test_character_identity_allows_only_sheik_zelda_transformations() -> None:
    sheik = int(exp.melee.Character.SHEIK.value)
    zelda = int(exp.melee.Character.ZELDA.value)
    fox = int(exp.melee.Character.FOX.value)

    assert exp._same_character_identity(sheik, zelda)
    assert exp._same_character_identity(zelda, sheik)
    assert exp._same_character_identity(fox, fox)
    assert not exp._same_character_identity(sheik, fox)


def test_behavior_analysis_ignores_zero_behavior_tail_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = []
    for boot in range(96):
        boot_dir = tmp_path / f"boot_{boot:03d}"
        boot_dir.mkdir()
        (boot_dir / "active.slp").touch()
        rows.append(
            exp.MatchRow(
                ego_character=1,
                opp_character=2,
                stage=3,
                boot_index=boot,
                match_ordinal=0,
                active_frames=3600,
                total_frames=3723,
                damage_dealt=10.0,
                damage_taken=5.0,
                stocks_taken=1,
                stocks_lost=0,
            )
        )
    (tmp_path / "boot_000" / "tail.slp").touch()
    rows.append(
        exp.MatchRow(
            ego_character=1,
            opp_character=2,
            stage=3,
            boot_index=0,
            match_ordinal=1,
            active_frames=1,
            total_frames=50,
            damage_dealt=0.0,
            damage_taken=0.0,
            stocks_taken=0,
            stocks_lost=0,
        )
    )
    rows.sort(key=lambda row: (row.boot_index, row.match_ordinal))

    def frames_for(path: str):
        tail = path.endswith("tail.slp")
        return SimpleNamespace(
            players=[
                SimpleNamespace(port=1, is_cpu=False, character=1, tail=tail),
                SimpleNamespace(port=2, is_cpu=True, character=2, tail=tail),
            ]
        )

    monkeypatch.setattr(exp, "read_replay_tolerant", lambda path: str(path))
    monkeypatch.setattr(exp, "behavior_frames", frames_for)
    monkeypatch.setattr(
        exp,
        "active_mask",
        lambda player, _opponent, _frames: torch.zeros(60) if player.tail else torch.ones(60),
    )
    monkeypatch.setattr(
        exp,
        "movement",
        lambda _player, _active: SimpleNamespace(wavedashes=1, failed_wavedash_attempts=2),
    )

    boots = exp._behavior_boots(tmp_path, {"ego_port": 1}, rows)

    assert len(boots) == 96
    assert boots[0].wavedashes == 1
    assert boots[0].failed_wavedash_attempts == 2
    assert boots[0].behavior_minutes == pytest.approx(1.0 / 60.0)
    assert boots[0].gameplay_minutes == pytest.approx(3601.0 / 3600.0)

    monkeypatch.setattr(exp, "active_mask", lambda *_args: torch.zeros(60))
    with pytest.raises(ValueError, match="boot 0 has no active behavior"):
        exp._behavior_boots(tmp_path, {"ego_port": 1}, rows)


def test_behavior_analysis_rejects_replay_that_tolerant_reader_cannot_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        exp.MatchRow(
            ego_character=1,
            opp_character=2,
            stage=3,
            boot_index=boot,
            match_ordinal=0,
            active_frames=3600,
            total_frames=3723,
            damage_dealt=0.0,
            damage_taken=0.0,
            stocks_taken=0,
            stocks_lost=0,
        )
        for boot in range(96)
    ]
    for boot in range(96):
        boot_dir = tmp_path / f"boot_{boot:03d}"
        boot_dir.mkdir()
        (boot_dir / "match.slp").touch()
    monkeypatch.setattr(exp, "read_replay_tolerant", lambda _path: None)

    with pytest.raises(ValueError, match="is unreadable after final-frame repair"):
        exp._behavior_boots(tmp_path, {"ego_port": 1}, rows)
