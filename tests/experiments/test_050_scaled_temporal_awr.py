"""Frozen contracts for the O50 production program."""

import importlib.util
import sys
import threading
import time
from dataclasses import asdict
from dataclasses import fields
from pathlib import Path

import modal
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
    assert state["experiment_id"] == "050_scaled_temporal_awr_v4"
    state["experiment_id"] = "050_scaled_temporal_awr_v3"
    with pytest.raises(ValueError, match="experiment_id"):
        exp.config_from_state(state)


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
