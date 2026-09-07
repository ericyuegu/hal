"""Frozen contracts for the O50 production program."""

import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path

import modal
import pytest
import torch

from hal.training.features import A_DIM
from hal.training.features import TrainBatch


def _load():
    path = Path(__file__).resolve().parents[2] / "experiments" / "050_scaled_temporal_awr.py"
    name = "test_exp050"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exp = _load()


def _tiny_cfg():
    arch = {
        **asdict(exp.ARCHITECTURE),
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
    )


def test_frozen_geometry_schedule_and_accounting() -> None:
    cfg = exp.TrainConfig()
    assert cfg.arch.L_ctx == 256
    assert exp.DIRECT_LOSS_START == 128
    assert cfg.arch.head_offsets == (1, 2, 3, 4, 5, 6, 9, 12, 16, 20)
    assert cfg.batch_size == 512
    assert cfg.num_workers == 24
    assert cfg.cache_limit_bytes == 2_000_000_000_000
    assert exp.loader_kwargs(cfg, {})["cache_limit"] == 2_000_000_000_000
    assert cfg.shuffle_block_size == 8192
    assert cfg.predownload == 8192
    assert cfg.max_steps == 2**17
    assert cfg.warmup_steps == 4096
    assert cfg.target_positions == 8 * exp.D0
    assert (cfg.hidden_std_multiplier, cfg.readout_init, cfg.depth_alpha) == (0.5, "mup-normal", 0.5)
    assert (cfg.muon_lr, cfg.adam_lr, cfg.muon_weight_decay, cfg.adam_weight_decay) == (
        0.014,
        4.25e-4,
        1e-4,
        1e-4,
    )
    assert (cfg.eval_every, cfg.eval_n_matchups, cfg.final_eval_n_matchups) == (8192, 96, 96)
    assert (cfg.prediction_frames, cfg.delay_frames, cfg.replan_interval_frames) == (4, 2, 2)
    schedule = exp.lr_schedule(cfg)
    assert schedule(0) == pytest.approx(1 / 4096)
    assert schedule(4095) == 1.0
    assert schedule(cfg.max_steps - 1) == pytest.approx(1 / 170)
    updates = exp.closed_loop_evaluation_updates(cfg.max_steps, cfg.eval_every)
    assert updates == tuple(range(8192, 131_073, 8192))
    assert updates.count(cfg.max_steps) == 1


def test_awr_activates_on_update_4097() -> None:
    assert exp.AWR_START_UPDATE > 4096
    assert exp.AWR_START_UPDATE <= 4097
    advantage = torch.tensor([[0.0, 199.5]])
    eligible = torch.ones_like(advantage, dtype=torch.bool)
    inactive, _ = exp.advantage_weights(advantage, eligible, beta=199.5, weight_max=3.5, active=False)
    active, stats = exp.advantage_weights(advantage, eligible, beta=199.5, weight_max=3.5, active=True)
    assert torch.equal(inactive, torch.ones_like(inactive))
    assert active.mean() == pytest.approx(1.0)
    assert stats["weight_max"] <= 3.5


def test_prepared_targets_keep_only_the_suffix() -> None:
    cfg = _tiny_cfg()
    model = exp.GPT(cfg)
    context = exp.synthetic_context(cfg, 2, torch.device("cpu"))
    target = torch.zeros(2, cfg.arch.sample_chunk_length, A_DIM)
    history, targets, valid = exp.prepared_targets(model, TrainBatch(context, target))
    # The production split is position 128; this tiny fixture uses the same midpoint contract.
    assert exp.DIRECT_LOSS_START == 128
    assert history.shape[1] == 4
    assert targets.shape[1] == 4
    assert valid.shape[1] == 4


def test_identity_masker_is_window_wide_and_resumable() -> None:
    cfg = _tiny_cfg()
    context = exp.synthetic_context(cfg, 2, torch.device("cpu"))
    context.features["ego_player_id"].fill_(7)
    batch = exp.AWRBatch(
        TrainBatch(context, torch.zeros(2, cfg.arch.sample_chunk_length, A_DIM)),
        torch.zeros(2, cfg.arch.L_ctx),
        torch.ones(2, cfg.arch.L_ctx, dtype=torch.bool),
    )
    first = exp.IdentityMasker(17, 0.5)
    state = first.state_dict()
    assert state["generator"].device.type == "cpu"
    expected = first(batch).context.features["ego_player_id"]
    resumed = exp.IdentityMasker(999, 0.5)
    resumed.load_state_dict(state)
    actual = resumed(batch).context.features["ego_player_id"]
    assert torch.equal(actual, expected)
    assert all(torch.unique(row).numel() == 1 for row in actual)


def test_prefetcher_applies_identity_mask_once_per_batch() -> None:
    cfg = _tiny_cfg()
    batches = [
        exp.synthetic_awr_batch(cfg, torch.device("cpu")),
        exp.synthetic_awr_batch(cfg, torch.device("cpu")),
    ]
    transformed = []

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


def test_parameter_contract_records_action_embedding_width_32() -> None:
    assert exp.ARCHITECTURE.action_embed_dim == 32
    assert exp.EXPECTED_PARAMETER_COUNTS["total"] == 216_496_794


def test_model_tag_names_the_actual_head_architecture() -> None:
    tag = exp.model_tag(exp.TrainConfig())
    assert "nonlinear-head-trunk-skip" in tag
    assert "o51-parameterized-mwd0.0001-awd0.0001" in tag
    assert "linear-head-no-skip" not in tag


def test_o51_initialization_depth_and_effective_rates() -> None:
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = exp.GPT(cfg)
    hidden = model.temporal.blocks[0].qkv.weight
    assert hidden.std().item() == pytest.approx(0.5 / hidden.shape[1] ** 0.5, rel=0.08)
    output = model.temporal.outputs["buttons"].down.weight
    assert output.std().item() == pytest.approx(exp.mup_readout_std(output.shape[1], 128), rel=0.08)
    assert exp.depth_rule("trunk", 16, 0.5).mlp == pytest.approx(1 / 2**0.5)
    production = exp.TrainConfig()
    assert exp.scaled_adam_betas(production) == pytest.approx((0.9875, 0.99375))
    assert exp.scaled_adam_epsilon(production) == pytest.approx(8**0.5 * 1e-12)
    assert exp._role_lr(exp.OptimizerRole("adamw", "input", False), production) == pytest.approx(4.25e-4 / 8**0.5)
    assert exp._role_lr(exp.OptimizerRole("adamw", "output", True, fan_in_multiplier=4), production) == pytest.approx(
        4.25e-4 / 8**0.5 / 4
    )


def test_optimizer_roles_cover_every_parameter_and_split_qkv() -> None:
    cfg = _tiny_cfg()
    model = exp.GPT(cfg)
    roles = exp.optimizer_roles(model, cfg)
    assert set(roles) == dict(model.named_parameters()).keys()
    assert roles["temporal.blocks.0.qkv.weight"].logical_splits == 3
    assert roles["value_head.up.weight"].logical_splits == 2
    optimizer = exp.make_optimizer(model, cfg)
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimized == {id(parameter) for parameter in model.parameters()}
    muon_groups = [group for group in optimizer.param_groups if group["use_muon"]]
    assert {group["logical_splits"] for group in muon_groups} == {1, 2, 3}
    assert all(group["muon_scale_clamp_min_one"] is False for group in muon_groups)


def test_v8_corpus_and_checkpoint_identity() -> None:
    cfg = exp.TrainConfig()
    assert len(cfg.source_names) == 44
    assert sum(exp.streams.POLICY_WORLD_V8_TRAIN_REPLAYS.values()) == exp.TRAIN_REPLAYS == 1_295_370
    assert sum(exp.streams.POLICY_WORLD_V8_TRAIN_FRAMES.values()) == exp.TRAIN_FRAMES == 13_266_364_175
    state = exp._checkpoint_config(cfg)
    assert state["experiment_id"] == "050_scaled_temporal_awr_v3"
    state["experiment_id"] = "050_scaled_temporal_awr_v2"
    with pytest.raises(ValueError, match="experiment_id"):
        exp.config_from_state(state)


def test_training_loader_uses_ordered_delivery(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class Sidecar:
        by_replay = {}

    class FakeResumableLoader:
        streaming_dataset = range(exp.TRAIN_REPLAYS)

    def make_loader(**kwargs):
        calls.append(kwargs)
        return FakeResumableLoader() if kwargs.get("resumable") else object()

    monkeypatch.setattr(exp, "load_identity_sidecar", lambda _cfg: Sidecar())
    monkeypatch.setattr(exp, "make_loader", make_loader)
    monkeypatch.setattr(exp, "ResumableStreamingDataLoader", FakeResumableLoader)
    monkeypatch.setattr(exp, "cache_validation", lambda _loader, _n_samples: [])

    exp._make_loaders(exp.TrainConfig(), {})

    assert calls[0]["in_order"] is True


def test_closed_loop_spawn_uses_launcher_app(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class Evaluator:
        def spawn(self, *args):
            calls.append(args)
            return "call-id"

    monkeypatch.setenv("HAL_MODAL_APP_NAME", "hal-test")
    monkeypatch.setattr(modal.Function, "from_name", lambda app, name: Evaluator())
    assert exp.spawn_closed_loop_evaluation("run", 8192, "a" * 64, 96) == "call-id"
    assert calls == [("run", 8192, "a" * 64, 96)]


def test_eval_rejects_checkpoint_before_loading_when_hash_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(exp, "load_checkpoint", lambda *_args, **_kwargs: pytest.fail("loaded bad checkpoint"))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        exp.eval_checkpoint(str(checkpoint), expected_checkpoint_sha256="0" * 64)


def test_training_is_the_primary_wandb_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    init_kwargs = {}
    definitions = []
    run = type("Run", (), {"summary": {}})()

    monkeypatch.setattr(exp.wandb, "init", lambda **kwargs: init_kwargs.update(kwargs))
    monkeypatch.setattr(exp.wandb, "run", run)
    monkeypatch.setattr(exp.wandb, "define_metric", lambda *args, **kwargs: definitions.append((args, kwargs)))

    exp._init_wandb(exp.TrainConfig(wandb_log_code=False), "run", None)

    settings = init_kwargs["settings"]
    assert settings.mode == "shared"
    assert settings.x_label == "training"
    assert settings.x_primary is True
    assert (("eval/*",), {"step_metric": "global_step", "summary": "none"}) in definitions
    assert (
        ("eval/checkpoint_step",),
        {"step_metric": "global_step", "summary": "max"},
    ) in definitions


def test_shared_eval_logging_keeps_out_of_order_checkpoint_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    init_kwargs = []
    logs = []
    finished = []

    class SharedRun:
        def log(self, values) -> None:
            logs.append(values)

        def finish(self) -> None:
            finished.append(True)

    def init(**kwargs):
        init_kwargs.append(kwargs)
        return SharedRun()

    monkeypatch.setattr(exp.wandb, "init", init)
    exp._log_shared_eval_metrics(
        "train-id",
        16_384,
        {"boots": 96.0, "crashed": 0.0, "net_stock_per_min": 0.5},
    )
    exp._log_shared_eval_metrics(
        "train-id",
        8192,
        {"boots": 96.0, "crashed": 0.0, "net_stock_per_min": 0.25},
    )

    assert [kwargs["id"] for kwargs in init_kwargs] == ["train-id", "train-id"]
    for kwargs in init_kwargs:
        settings = kwargs["settings"]
        assert settings.mode == "shared"
        assert settings.x_primary is False
        assert settings.x_update_finish_state is False
        assert settings.x_disable_stats is True
    assert logs == [
        {
            "global_step": 16_384,
            "eval/checkpoint_step": 16_384,
            "eval/boots": 96.0,
            "eval/crashed": 0.0,
            "eval/net_stock_per_min": 0.5,
        },
        {
            "global_step": 8192,
            "eval/checkpoint_step": 8192,
            "eval/boots": 96.0,
            "eval/crashed": 0.0,
            "eval/net_stock_per_min": 0.25,
        },
    ]
    assert finished == [True, True]
