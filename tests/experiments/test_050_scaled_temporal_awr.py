"""Frozen contracts for the O50 production program."""

import importlib.util
import json
import os
import socket
import sys
import threading
import time
from dataclasses import asdict
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hal.training.features import A_DIM
from hal.training.features import TrainBatch
from hal.training.player_identity import ReplayPlayerLookup


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


def test_schedule_reaches_the_derived_training_boundary() -> None:
    cfg = exp.TrainConfig()
    assert cfg.max_steps == 2**17
    assert cfg.warmup_steps == 4096
    schedule = exp.lr_schedule(cfg)
    assert schedule(0) == pytest.approx(1 / 4096)
    assert schedule(4095) == 1.0
    assert schedule(cfg.max_steps - 1) == pytest.approx(1 / 170)
    updates = exp.closed_loop_evaluation_updates(cfg.max_steps, cfg.eval_every)
    assert updates == tuple(range(8192, 131_073, 8192))
    assert updates.count(cfg.max_steps) == 1


def test_awr_activates_on_update_4097() -> None:
    assert exp.AWRCalibration.start_update == 4097
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
    assert exp.Architecture().direct_loss_start == 128
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


def test_inference_reads_identity_from_the_runtime_context() -> None:
    cfg = _tiny_cfg()
    model = exp.GPT(cfg)
    inference = exp.BF16Inference(model, cfg, bucket=2, compiled=False)
    context = exp._condition_ego_player(exp.synthetic_context(cfg, 2, torch.device("cpu")), 7)
    captured = {}

    def trunk(features, _ctx_pad, _observed):
        captured["player_id"] = features["ego_player_id"].clone()
        return torch.zeros(2, cfg.arch.L_ctx, cfg.arch.d_model)

    def decoder(_hidden, _observed, _uniforms):
        return torch.zeros(2, cfg.prediction_frames, 4, dtype=torch.long)

    inference._trunks[2] = trunk
    inference._decoders[(2, cfg.prediction_frames)] = decoder
    inference.decode(context, cfg.prediction_frames)

    assert torch.equal(captured["player_id"], torch.full((2, cfg.arch.L_ctx), 7))


@pytest.mark.parametrize(
    ("delay", "replan", "expected"),
    [
        (2, 2, [0.0, 0.0, 3.0, 4.0]),
        (0, 1, [1.0, 11.0, 21.0, 31.0]),
    ],
)
def test_truncation_policy_executes_the_selected_timing_slice(
    monkeypatch: pytest.MonkeyPatch,
    delay: int,
    replan: int,
    expected: list[float],
) -> None:
    calls = 0

    def predict(_context, committed):
        nonlocal calls
        assert committed is None
        values = np.zeros((1, 4, exp.A_DIM), dtype=np.float32)
        values[0, :, 0] = 10 * calls + np.arange(1, 5)
        calls += 1
        return values

    policy = exp.DelayedTruncationPolicy(
        predict_chunk=predict,
        stats={},
        L_ctx=8,
        L_chunk=4,
        s=replan,
        d=0,
        delay_frames=delay,
        device="cpu",
    )
    slot = exp.Slot(0, 1)
    state = SimpleNamespace(reset_pending=True, last_action=None)
    policy._slots[slot] = state
    policy._ingest = lambda _live, _obs: None

    def context(_live):
        state.reset_pending = False
        return object()

    policy._context = context
    monkeypatch.setattr(exp, "action_vec_to_controller", lambda action: action)

    actual = [float(policy(frame, {slot: {}})[slot][0]) for frame in range(4)]

    assert actual == expected


def test_deployment_timing_rejects_an_unavailable_prediction_slice() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        exp._validate_deployment_timing(4, delay_frames=2, replan_interval_frames=3)


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
        prefetcher.fill_lookahead(1)
        prefetcher.stage_next()
        prefetcher.next()
    finally:
        prefetcher.close()

    assert transformed == batches


def test_four_batch_lookahead_drains_at_every_state_boundary() -> None:
    cfg = _tiny_cfg()
    batch = exp.synthetic_awr_batch(cfg, torch.device("cpu"))
    calls = []

    class Loader:
        def __iter__(self):
            return self

        def __next__(self):
            calls.append(threading.get_ident())
            return batch

    prefetcher = exp.DeviceBatchPrefetcher(Loader(), cfg, "cpu")
    queue_depths = []
    try:
        for update in range(1, 9):
            prefetcher.next()
            prefetcher.fill_lookahead(0 if update == 8 else 8 - update)
            queue_depths.append(prefetcher.queue_depth)
            if update < 8:
                prefetcher.stage_next()
        assert prefetcher.drained
    finally:
        prefetcher.close()

    assert len(calls) == 8
    assert len(set(calls)) == 1
    assert calls[0] != threading.main_thread().ident
    assert max(queue_depths) == 4
    assert queue_depths[-1] == 0


def test_prefetcher_reports_ready_separately_from_submitted_batches() -> None:
    cfg = _tiny_cfg()
    batch = exp.synthetic_awr_batch(cfg, torch.device("cpu"))
    blocked = threading.Event()
    release = threading.Event()
    calls = 0

    class Loader:
        def __iter__(self):
            return self

        def __next__(self):
            nonlocal calls
            calls += 1
            if calls == 2:
                blocked.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("test did not release the CPU loader")
            return batch

    prefetcher = exp.DeviceBatchPrefetcher(Loader(), cfg, "cpu")
    try:
        prefetcher.next()
        prefetcher.fill_lookahead(4)
        assert blocked.wait(timeout=5)
        assert prefetcher.submitted_batches == 4
        assert prefetcher.ready_batches == 0
        release.set()
        deadline = time.monotonic() + 5
        while prefetcher.ready_batches != 4 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert prefetcher.ready_batches == 4
    finally:
        release.set()
        prefetcher.close()


def test_update_timer_excludes_checkpoint_gap_inside_a_metrics_window() -> None:
    now = 0.0

    def clock() -> float:
        return now

    timer = exp._UpdateTimer(clock)
    for index in range(25):
        timer.start()
        now += 2.0
        timer.finish()
        if index == 22:
            now += 1_000.0

    assert timer.stats_and_reset(25) == (2.0, 2.0)


def test_prepared_data_starts_workers_before_background_next(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []
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
    sidecar = type("Sidecar", (), {"by_replay": {}})()

    prepared = exp._prepare_training_data(exp.TrainConfig(), {}, sidecar, None)
    try:
        assert prepared.first_batch_future.result() is batch
    finally:
        prepared.resources.close()

    assert events[0] == ("iter", threading.main_thread().ident)
    assert events[1][0] == "next"
    assert events[1][1] != threading.main_thread().ident


def test_parameter_contract_records_action_embedding_width_32() -> None:
    architecture = exp.Architecture()
    assert architecture.action_embed_dim == 32
    assert architecture.parameter_count_contract["total"] == 216_496_794


def test_proxy_is_the_o51_15m_16_layer_d0_treatment() -> None:
    cfg = exp.proxy_config()
    assert cfg.arch.n_layers == 16
    assert (cfg.max_steps, cfg.warmup_steps) == (16_384, 512)
    assert exp.closed_loop_evaluation_updates(cfg.max_steps, cfg.eval_every) == (8192, 16_384)
    assert exp.subsystem_parameter_counts(exp.GPT(cfg)) == cfg.arch.parameter_count_contract


def test_proxy_cli_selects_the_frozen_treatment(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = {}

    monkeypatch.setattr(exp, "load_stats", lambda _cfg: {})

    def train(cfg, _stats, **kwargs) -> None:
        observed["cfg"] = cfg
        observed["proxy"] = kwargs["proxy"]

    monkeypatch.setattr(exp, "train", train)

    exp.main(exp.TrainArgs(proxy=True))

    assert observed == {"cfg": exp.proxy_config(), "proxy": True}


def test_half_muon_fork_keeps_parent_wandb_id_only_as_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "production-source"
    checkpoint_name = "checkpoints/step-0024576.pt"
    checkpoint_path = tmp_path / "runs" / source / checkpoint_name
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"parent checkpoint")
    parent_config = exp._checkpoint_config(exp.TrainConfig())
    parent_config["experiment_id"] = "050_scaled_temporal_awr_v4"
    for name in (
        "muon_lr_multiplier",
        "continuation_diagnostics",
        "parent_run_name",
        "parent_checkpoint_name",
        "parent_checkpoint_sha256",
        "parent_wandb_id",
    ):
        parent_config.pop(name)
    resume_state = {
        "cfg": parent_config,
        "sched": {"last_epoch": 24_576},
        "step": 24_575,
        "wandb_id": "p1fyyp1z",
    }
    observed = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(exp, "load_for_resume", lambda *_args, **_kwargs: resume_state)
    monkeypatch.setattr(exp, "load_stats", lambda _cfg: {})
    monkeypatch.setattr(exp, "_remote_run_exists", lambda _run: False)

    def train(cfg, _stats, **kwargs) -> None:
        observed["cfg"] = cfg
        observed["kwargs"] = kwargs

    monkeypatch.setattr(exp, "train", train)

    exp.main(
        exp.TrainArgs(
            resume=source,
            resume_checkpoint=checkpoint_name,
            resume_as="o50-p1fyyp1z-u24576-muon-half",
            resume_muon_lr_multiplier=0.5,
        )
    )

    cfg = observed["cfg"]
    assert cfg.muon_lr_multiplier == 0.5
    assert cfg.continuation_diagnostics is True
    assert cfg.parent_wandb_id == "p1fyyp1z"
    assert cfg.parent_checkpoint_sha256 == exp.checkpoint_sha256(checkpoint_path)
    assert observed["kwargs"]["resume_state"]["wandb_id"] is None
    assert observed["kwargs"]["fork_from_parent"] is True


def test_diagnostic_scalars_move_to_cpu_before_stacking() -> None:
    class DeviceScalar:
        def __init__(self, value: float) -> None:
            self.value = value

        def detach(self) -> DeviceScalar:
            return self

        def float(self) -> DeviceScalar:
            return self

        def cpu(self) -> torch.Tensor:
            return torch.tensor(self.value)

    metrics = {"cpu": DeviceScalar(1.0), "cuda": DeviceScalar(2.0)}

    assert exp._download_scalar_metrics(metrics, 24_576) == {"cpu": 1.0, "cuda": 2.0}


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
    assert cfg.train_replays == 1_295_370
    assert cfg.train_frames == 13_266_364_175
    selection = exp.data_selection(cfg)
    assert selection.row_count == cfg.train_replays
    assert selection.sha256 == cfg.selection_sha256
    assert set(exp.streams.POLICY_WORLD_V8_TRAIN_MANIFEST_SHA256) == set(cfg.source_names)
    state = exp._checkpoint_config(cfg)
    assert exp.config_from_state(state) == cfg
    assert {"max_steps", "warmup_steps"} <= state.keys()
    assert not {"max_steps", "warmup_steps"} & {field.name for field in fields(exp.TrainConfig)}
    assert state["experiment_id"] == "050_scaled_temporal_awr_v5"
    legacy = dict(state)
    legacy["experiment_id"] = "050_scaled_temporal_awr_v4"
    for name in (
        "muon_lr_multiplier",
        "continuation_diagnostics",
        "parent_run_name",
        "parent_checkpoint_name",
        "parent_checkpoint_sha256",
        "parent_wandb_id",
    ):
        legacy.pop(name)
    assert exp.config_from_state(legacy) == cfg
    state["experiment_id"] = "050_scaled_temporal_awr_v3"
    with pytest.raises(ValueError, match="experiment_id"):
        exp.config_from_state(state)


def test_restored_muon_schedule_is_rebased_without_changing_adam() -> None:
    cfg = _tiny_cfg()
    model = exp.GPT(cfg)
    optimizer = exp.make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, exp.lr_schedule(cfg))
    before = [
        (group["use_muon"], group["lr"], group["initial_lr"], scheduler.base_lrs[index])
        for index, group in enumerate(optimizer.param_groups)
    ]

    exp.rebase_restored_muon_schedule(optimizer, scheduler, 0.5)

    for index, (use_muon, lr, initial_lr, base_lr) in enumerate(before):
        expected = 0.5 if use_muon else 1.0
        group = optimizer.param_groups[index]
        assert group["lr"] == pytest.approx(lr * expected)
        assert group["initial_lr"] == pytest.approx(initial_lr * expected)
        assert scheduler.base_lrs[index] == pytest.approx(base_lr * expected)
        assert scheduler.get_last_lr()[index] == pytest.approx(lr * expected)


def test_optimizer_diagnostics_match_direct_parameter_deltas() -> None:
    cfg = _tiny_cfg()
    model = exp.GPT(cfg)
    optimizer = exp.make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, exp.lr_schedule(cfg))
    before = {id(parameter): parameter.detach().clone() for parameter in model.parameters()}
    batch = exp.synthetic_awr_batch(cfg, torch.device("cpu"))
    diagnostics = exp.OptimizerStepDiagnostics.create(model, cfg)

    result = exp.train_step(
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
        optimizer_diagnostics=diagnostics,
    )

    gradient_square = torch.zeros(())
    for subsystem, parameters in exp.parameter_subsystems(model).items():
        direct_update_l2 = (
            torch.stack([(before[id(parameter)] - parameter).float().square().sum() for parameter in parameters])
            .sum()
            .sqrt()
        )
        prefix = f"diagnostics/optimizer/{subsystem}/all"
        torch.testing.assert_close(result.optimizer_diagnostics[f"{prefix}/update_l2"], direct_update_l2)
        gradient_square += result.optimizer_diagnostics[f"{prefix}/grad_l2_pre_clip"].square()
    torch.testing.assert_close(gradient_square.sqrt(), result.gradient_norm)


def test_fixed_diagnostics_are_centered_and_checkpointable() -> None:
    cfg = _tiny_cfg()
    model = exp.GPT(cfg)
    batch = exp.synthetic_awr_batch(cfg, torch.device("cpu")).batch
    batch = exp.TrainBatch(
        exp.Context(
            {name: value.repeat(4, 1) for name, value in batch.context.features.items()},
            batch.context.ctx_pad.repeat(4),
        ),
        batch.target.repeat(4, 1, 1),
    )
    validation = [batch]
    cpu_rng = torch.get_rng_state()

    tracker, metrics = exp.FixedDiagnosticTracker.create(model, validation, cfg, 24_576)

    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert all(torch.isfinite(value) for value in metrics.values())
    assert metrics["diagnostics/fixed_policy_kl/from_parent/total_nats"] == 0
    assert 0 <= metrics["diagnostics/attention/trunk/mean"] <= 1
    assert 0 <= metrics["diagnostics/attention/temporal/mean"] <= 1
    restored = exp.FixedDiagnosticTracker.from_state(tracker.state_dict())
    assert restored.batch_sha256 == tracker.batch_sha256
    for name in exp.CONTROLLER_GROUP_NAMES:
        torch.testing.assert_close(
            restored.baseline_log_probabilities[name],
            tracker.baseline_log_probabilities[name],
        )
    with torch.no_grad():
        button_head = exp.cast(exp.NonlinearActionHead, model.temporal.outputs["buttons"])
        button_head.down.bias[0].add_(0.5)
    changed = restored.measure(model, cfg, 28_672)
    assert changed["diagnostics/fixed_policy_kl/from_parent/total_nats"] > 0
    assert changed["diagnostics/fixed_policy_kl/from_previous/total_nats"] > 0
    assert restored.previous_update == 28_672


def test_fixed_legal_logits_ignore_common_class_shifts() -> None:
    cfg = _tiny_cfg()
    model = exp.GPT(cfg)
    batch = exp.synthetic_awr_batch(cfg, torch.device("cpu")).batch
    before_log_probabilities, before_metrics, _valid = exp.fixed_policy_diagnostics(model, batch, cfg)
    with torch.no_grad():
        for head in model.temporal.outputs.values():
            cast_head = exp.cast(exp.NonlinearActionHead, head)
            cast_head.down.bias.add_(7)

    after_log_probabilities, after_metrics, _valid = exp.fixed_policy_diagnostics(model, batch, cfg)

    for name in exp.CONTROLLER_GROUP_NAMES:
        torch.testing.assert_close(after_log_probabilities[name], before_log_probabilities[name])
        prefix = f"diagnostics/fixed_logits/{name}"
        torch.testing.assert_close(after_metrics[f"{prefix}/rms"], before_metrics[f"{prefix}/rms"])
        torch.testing.assert_close(after_metrics[f"{prefix}/abs_p999"], before_metrics[f"{prefix}/abs_p999"])


def test_training_constructs_the_physical_shard_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = {}

    class Adapter:
        def __init__(self, selection, *, download_retry):
            observed["adapter"] = (selection, download_retry)
            self.manifests = {source.source: (source.stop,) for source in selection.sources}

        def validate_manifests(self, **kwargs):
            observed["manifest_validation"] = kwargs

    class Loader:
        required_disk_bytes = 10
        disk_free_bytes = 20
        minimum_replay_gap_batches = 200

        @classmethod
        def __class_getitem__(cls, _item):
            return cls

        def __init__(self, **kwargs):
            observed["loader"] = kwargs
            self.source_sample_counts = kwargs["selection"].row_counts_by_source()

        def close(self):
            observed["closed"] = True

    monkeypatch.setattr(exp, "MDSStorageAdapter", Adapter)
    monkeypatch.setattr(exp, "build_shard_plan", lambda *_args: (object(),))
    monkeypatch.setattr(exp, "PhysicalShardReplayLoader", Loader)

    loader = exp._make_train_loader(exp.TrainConfig(), {}, ReplayPlayerLookup({}))

    assert loader is not None
    kwargs = observed["loader"]
    cfg = exp.TrainConfig()
    assert kwargs["data_protocol"] == cfg.data_protocol
    assert kwargs["replay_slots"] == 131_072
    assert kwargs["windows_per_generation"] == 8
    assert kwargs["replay_phase_block_batches"] == 25
    assert kwargs["num_workers"] == 24
    assert kwargs["materialization_threads"] == cfg.raw_shard_materialization_threads
    assert kwargs["source_manifest_sha256"] == exp.streams.POLICY_WORLD_V8_TRAIN_MANIFEST_SHA256


def test_training_functions_log_compile_calls(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Trunk:
        attn_path = "varlen_flash"

        def resolve_attention(self, device: str) -> None:
            assert device == "cuda"

    class Temporal:
        def teacher_forced_nll_with_diagnostics(self) -> None:
            pass

    class Model:
        trunk = Trunk()
        temporal = Temporal()

        def forward(self) -> None:
            pass

    compiled = []

    def compile_function(function, **kwargs):
        compiled.append((function, kwargs))
        return function

    monkeypatch.setattr(exp, "DEVICE", "cuda")
    monkeypatch.setattr(exp.torch, "compile", compile_function)

    exp._training_functions(Model(), exp.TrainConfig())

    assert len(compiled) == 2
    output = capsys.readouterr().out
    assert "[compile] calling torch.compile for trunk" in output
    assert "[compile] calling torch.compile for temporal model" in output


def test_validation_cache_logs_progress_and_completion(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _tiny_cfg()
    batch = exp.synthetic_awr_batch(cfg, torch.device("cpu"))
    monkeypatch.setattr(exp, "_STARTUP_LOG_INTERVAL_S", 0.0)

    validation = exp.cache_validation([batch, batch], 2 * cfg.batch_size)

    assert len(validation) == 2
    assert all(item is batch for item in validation)
    output = capsys.readouterr().out
    assert "[validation] caching 4 samples" in output
    assert "[validation] cached 2/4 samples" in output
    assert "[validation] cache complete: 4 samples" in output


def test_compile_warmup_logs_start_heartbeat_and_completion(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Model:
        def train(self) -> None:
            pass

        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none

    class Loss:
        def backward(self) -> None:
            time.sleep(0.03)

    monkeypatch.setattr(exp, "DEVICE", "cuda")
    monkeypatch.setattr(exp, "_STARTUP_LOG_INTERVAL_S", 0.01)
    monkeypatch.setattr(exp.torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(exp.torch.cuda, "set_rng_state_all", lambda _state: None)
    monkeypatch.setattr(exp.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(exp.torch.compiler, "cudagraph_mark_step_begin", lambda: None)
    monkeypatch.setattr(exp, "synthetic_awr_batch", lambda _cfg, _device: object())
    monkeypatch.setattr(exp, "microbatch_loss", lambda *_args, **_kwargs: (Loss(), None, None))

    exp._compile_synthetic_forward_backward(
        Model(),
        exp.TrainConfig(),
        step=0,
        trunk_fn=lambda: None,
        temporal_fn=lambda: None,
    )

    output = capsys.readouterr().out
    assert "[compile] starting synthetic forward/backward to trigger lazy compilation" in output
    assert "[compile] lazy compilation still running" in output
    assert "[compile] lazy compilation complete" in output


def test_training_rejects_insufficient_disk() -> None:
    loader = type("Loader", (), {"required_disk_bytes": 11, "disk_free_bytes": 10})()
    with pytest.raises(RuntimeError, match="do not fit"):
        exp._require_loader_disk(loader)


def test_legacy_training_configuration_and_resume_option_are_removed() -> None:
    config_fields = {field.name for field in fields(exp.TrainConfig)}
    assert not config_fields.intersection(
        {"shuffle_block_size", "predownload", "loader_timeout_s", "cache_limit_bytes"}
    )
    assert "resume_predownload" not in {field.name for field in fields(exp.TrainArgs)}


def test_boundary_state_resumes_next_batch_update_and_identity_dropout() -> None:
    cfg = _tiny_cfg()

    class Loader:
        def __init__(self) -> None:
            self.cursor = 0

        def __iter__(self):
            return self

        def __next__(self):
            batch = exp.synthetic_awr_batch(cfg, torch.device("cpu"))
            batch.context.features["ego_player_id"].fill_(self.cursor + 3)
            batch.target.fill_(self.cursor + 1)
            self.cursor += 1
            return batch

        def state_dict(self):
            return {"cursor": self.cursor}

        def load_state_dict(self, state):
            self.cursor = state["cursor"]

    loader = Loader()
    masker = exp.IdentityMasker(23, 0.5)
    prefetcher = exp.DeviceBatchPrefetcher(loader, cfg, "cpu", masker)
    try:
        for update in range(1, 9):
            prefetcher.next()
            prefetcher.fill_lookahead(0 if update == 8 else 8 - update)
            if update < 8:
                prefetcher.stage_next()
        assert prefetcher.drained
        loader_state = loader.state_dict()
        masker_state = masker.state_dict()
    finally:
        prefetcher.close()

    expected_prefetcher = exp.DeviceBatchPrefetcher(loader, cfg, "cpu", masker)
    try:
        expected_batch, expected_prefixes = expected_prefetcher.next()
    finally:
        expected_prefetcher.close()

    restored_loader = Loader()
    restored_loader.load_state_dict(loader_state)
    restored_masker = exp.IdentityMasker(999, 0.5)
    restored_masker.load_state_dict(masker_state)
    restored_prefetcher = exp.DeviceBatchPrefetcher(restored_loader, cfg, "cpu", restored_masker)
    try:
        actual_batch, actual_prefixes = restored_prefetcher.next()
    finally:
        restored_prefetcher.close()

    assert actual_prefixes == expected_prefixes
    torch.testing.assert_close(actual_batch.target, expected_batch.target)
    torch.testing.assert_close(
        actual_batch.context.features["ego_player_id"],
        expected_batch.context.features["ego_player_id"],
    )
    assert restored_masker.state_dict()["forced"] == masker.state_dict()["forced"]

    def optimizer_update(batch):
        weight = torch.nn.Parameter(torch.tensor(0.25))
        optimizer = torch.optim.SGD([weight], lr=0.01)
        identity = batch.context.features["ego_player_id"].float().mean()
        loss = (weight * (batch.target.float().mean() + identity)).square()
        loss.backward()
        optimizer.step()
        return weight.detach()

    torch.testing.assert_close(optimizer_update(actual_batch), optimizer_update(expected_batch))


def test_closed_loop_spawn_uses_launcher_broker(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher, experiment = socket.socketpair()
    monkeypatch.setenv("HAL_MODAL_EVAL_FD", str(experiment.fileno()))
    requests = []

    def serve() -> None:
        with launcher, launcher.makefile("rb") as lines:
            requests.append(json.loads(lines.readline()))
            os.write(launcher.fileno(), b'{"function_call_id":"fc-eval"}\n')

    server = threading.Thread(target=serve)
    server.start()
    try:
        assert exp.spawn_closed_loop_evaluation("run", 8192, "a" * 64, 96) == "fc-eval"
    finally:
        experiment.close()
        server.join()
    assert requests == [
        {
            "run_name": "run",
            "update": 8192,
            "expected_checkpoint_sha256": "a" * 64,
            "n_matchups": 96,
        }
    ]


def test_eval_rejects_checkpoint_before_loading_when_hash_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(exp, "load_checkpoint", lambda *_args, **_kwargs: pytest.fail("loaded bad checkpoint"))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        exp.eval_checkpoint(str(checkpoint), expected_checkpoint_sha256="0" * 64)


def test_cody_fox_d0r1_protocol_records_the_complete_treatment() -> None:
    cfg = _tiny_cfg()
    model = exp.GPT(cfg)

    protocol = exp._eval_protocol(
        cfg,
        model,
        n_matchups=96,
        checkpoint_sha256="a" * 64,
        max_parallel=3,
        fixed_ego_character=exp.melee.Character.FOX,
        ego_player_id=8092,
        ego_player_code="IBDW#0",
        delay_frames=0,
        replan_interval_frames=1,
    )

    assert protocol.fixed_ego_character == int(exp.melee.Character.FOX.value)
    assert protocol.ego_player_id == 8092
    assert protocol.ego_player_code == "IBDW#0"
    assert protocol.opponent_identity_conditioned is False
    assert (protocol.oriented_pairs, protocol.ego_characters, protocol.cpu_characters) == (14, 1, 14)
    assert (protocol.delay_frames, protocol.replan_interval_frames) == (0, 1)


def test_eval_resolves_exact_checkpoint_player_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    cfg = exp.TrainConfig()
    vocabulary = exp.PlayerVocabulary(("AA#1", "IBDW#0"))
    encoded = torch.from_numpy(exp.vocabulary_buffer(vocabulary))
    model = SimpleNamespace(get_buffer=lambda _name: encoded)
    observed = {}
    monkeypatch.setattr(exp, "checkpoint_sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(exp, "load_checkpoint", lambda _path: (model, cfg, {}, {"step": 8191}))

    def evaluate(_model, _stats, _cfg, **kwargs):
        observed.update(kwargs)
        return {"scheduled_boots": 96.0, "completed_boots": 96.0, "boots": 96.0}

    monkeypatch.setattr(exp, "eval_vs_cpu", evaluate)

    exp.eval_checkpoint(
        str(checkpoint),
        player_code=" IBDW#0 ",
        fixed_ego_character=exp.melee.Character.FOX,
        delay_frames=0,
        replan_interval_frames=1,
    )

    assert observed["ego_player_id"] == vocabulary.id_for_code("IBDW#0")
    assert observed["ego_player_code"] == "IBDW#0"
    assert observed["fixed_ego_character"] is exp.melee.Character.FOX
    assert (observed["delay_frames"], observed["replan_interval_frames"]) == (0, 1)


def test_uploaded_eval_variant_requires_distinct_artifact_and_metric_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    cfg = exp.TrainConfig()
    vocabulary = exp.PlayerVocabulary(("IBDW#0",))
    encoded = torch.from_numpy(exp.vocabulary_buffer(vocabulary))
    model = SimpleNamespace(get_buffer=lambda _name: encoded)
    monkeypatch.setattr(exp, "checkpoint_sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(exp, "load_checkpoint", lambda _path: (model, cfg, {}, {"step": 8191}))

    with pytest.raises(ValueError, match="output_name"):
        exp.eval_checkpoint(str(checkpoint), upload_run="run", player_code="IBDW#0")
    with pytest.raises(ValueError, match="W&B namespace"):
        exp.eval_checkpoint(
            str(checkpoint),
            output_name="cody",
            shared_wandb=True,
            player_code="IBDW#0",
        )


def test_eval_overrides_do_not_mutate_checkpoint_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    cfg = exp.TrainConfig()
    observed = {}
    monkeypatch.setattr(exp, "checkpoint_sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(exp, "load_checkpoint", lambda _path: (object(), cfg, {}, {"step": 0}))

    def evaluate(_model, _stats, actual_cfg, **kwargs):
        observed["cfg"] = actual_cfg
        observed["kwargs"] = kwargs
        return {"scheduled_boots": 7.0, "completed_boots": 7.0, "boots": 7.0}

    monkeypatch.setattr(exp, "eval_vs_cpu", evaluate)

    exp.eval_checkpoint(str(checkpoint), n_matchups=7, eager=True, max_parallel=3)

    assert observed["cfg"] is cfg
    assert observed["kwargs"]["eager"] is True
    assert observed["kwargs"]["max_parallel"] == 3
    assert cfg.inference_mode == "compiled"
    assert cfg.eval_max_parallel == 32


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


def test_shared_variant_metrics_use_the_requested_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    logs = []

    class SharedRun:
        def log(self, values) -> None:
            logs.append(values)

        def finish(self) -> None:
            pass

    monkeypatch.setattr(exp.wandb, "init", lambda **_kwargs: SharedRun())

    exp._log_shared_eval_metrics(
        "train-id",
        8192,
        {"boots": 96.0, "net_stock_per_min": 0.25},
        namespace="eval_cody_fox_d0r1",
    )

    assert logs == [
        {
            "global_step": 8192,
            "eval/checkpoint_step": 8192,
            "eval_cody_fox_d0r1/boots": 96.0,
            "eval_cody_fox_d0r1/net_stock_per_min": 0.25,
        }
    ]
