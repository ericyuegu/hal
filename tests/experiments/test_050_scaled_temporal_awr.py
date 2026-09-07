"""Frozen contracts for the O50 production program."""

import hashlib
import importlib.util
import json
import sys
import threading
from dataclasses import asdict
from dataclasses import fields
from pathlib import Path

import modal
import numpy as np
import pytest
import torch

from hal.data.policy_schema import PACKED_STATE_SUFFIXES
from hal.data.policy_schema import pack_player_state
from hal.data.policy_world_schema import POLICY_WORLD_MDS_COLUMNS
from hal.data.policy_world_schema import POLICY_WORLD_SCHEMA_VERSION
from hal.training import returns as returns_lib
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
        **changes,
    )


def test_frozen_geometry_schedule_and_accounting() -> None:
    cfg = exp.TrainConfig()
    assert cfg.arch.L_ctx == 256
    assert exp.DIRECT_LOSS_START == 128
    assert cfg.arch.head_offsets == (1, 2, 3, 4, 5, 6, 9, 12, 16, 20)
    assert cfg.batch_size == 512
    assert cfg.num_workers == 24
    assert exp.REPLAY_SLOTS == 131_072
    assert exp.WINDOWS_PER_GENERATION == 8
    assert exp.REPLAY_PHASE_BLOCK_BATCHES == 25
    assert exp.MIN_REPLAY_GAP_BATCHES == 200
    assert exp.RESERVED_DISK_BYTES == 256 * 2**30
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
    assert exp.ARCHITECTURE.action_embed_dim == 32
    assert exp.EXPECTED_PARAMETER_COUNTS["total"] == 216_496_794


def test_proxy_is_the_o51_15m_16_layer_d0_treatment() -> None:
    cfg = exp.proxy_config()
    assert cfg.arch == exp.PROXY_ARCHITECTURE
    assert cfg.arch.n_layers == 16
    assert (cfg.max_steps, cfg.warmup_steps) == (16_384, 512)
    assert exp.closed_loop_evaluation_updates(cfg.max_steps, cfg.eval_every) == (8192, 16_384)
    assert exp.subsystem_parameter_counts(exp.GPT(cfg)) == exp.PROXY_PARAMETER_COUNTS
    exp.validate_proxy_config(cfg)
    with pytest.raises(ValueError, match="production config"):
        exp.validate_production_config(cfg)


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
    assert sum(exp.streams.POLICY_WORLD_V8_TRAIN_REPLAYS.values()) == exp.TRAIN_REPLAYS == 1_295_370
    assert sum(exp.streams.POLICY_WORLD_V8_TRAIN_FRAMES.values()) == exp.TRAIN_FRAMES == 13_266_364_175
    selection = exp.data_selection(cfg)
    assert selection.row_count == exp.TRAIN_REPLAYS
    assert selection.sha256 == exp.V8_SELECTION_SHA256
    assert exp._canonical_selection_sha256(selection.sources) == exp.V8_SELECTION_SHA256
    assert set(exp.SOURCE_MANIFEST_SHA256) == set(cfg.source_names)
    state = exp._checkpoint_config(cfg)
    assert state["experiment_id"] == "050_scaled_temporal_awr_v4"
    state["experiment_id"] = "050_scaled_temporal_awr_v3"
    with pytest.raises(ValueError, match="experiment_id"):
        exp.config_from_state(state)


def _manifest_payload(source: str, *, schema_value: str = "value") -> bytes:
    rows = exp.streams.POLICY_WORLD_V8_TRAIN_REPLAYS[source]
    return json.dumps(
        {
            "version": exp.V8_MDS_INDEX_VERSION,
            "shards": [
                {
                    "samples": rows,
                    "column_names": ["column"],
                    "column_encodings": [schema_value],
                    "column_sizes": [None],
                }
            ],
        },
        separators=(",", ":"),
    ).encode()


def test_v8_manifest_rejects_hash_schema_and_row_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    source = exp._DEFAULT_SOURCE_NAMES[0]
    payload = _manifest_payload(source)
    monkeypatch.setitem(exp.SOURCE_MANIFEST_SHA256, source, hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(
        exp,
        "V8_MDS_SCHEMA_SHA256",
        exp._manifest_schema_sha256(json.loads(payload)["shards"][0]),
    )
    exp._validate_v8_manifest(source, payload)

    with pytest.raises(ValueError, match="SHA-256"):
        exp._validate_v8_manifest(source, payload + b" ")
    wrong_schema = _manifest_payload(source, schema_value="other")
    monkeypatch.setitem(exp.SOURCE_MANIFEST_SHA256, source, hashlib.sha256(wrong_schema).hexdigest())
    with pytest.raises(ValueError, match="schema"):
        exp._validate_v8_manifest(source, wrong_schema)
    wrong_rows = payload.replace(str(exp.streams.POLICY_WORLD_V8_TRAIN_REPLAYS[source]).encode(), b"1")
    monkeypatch.setitem(exp.SOURCE_MANIFEST_SHA256, source, hashlib.sha256(wrong_rows).hexdigest())
    with pytest.raises(ValueError, match="rows"):
        exp._validate_v8_manifest(source, wrong_rows)


def test_training_constructs_the_physical_shard_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = {}

    class Adapter:
        def __init__(self, selection, *, download_retry):
            observed["adapter"] = (selection, download_retry)
            self.manifests = {source.source: (source.stop,) for source in selection.sources}

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
    monkeypatch.setattr(exp, "_validate_v8_manifests", lambda *_args: None)
    monkeypatch.setattr(exp, "build_shard_plan", lambda *_args: (object(),))
    monkeypatch.setattr(exp, "PhysicalShardReplayLoader", Loader)

    loader = exp._make_train_loader(exp.TrainConfig(), {}, ReplayPlayerLookup({}))

    assert loader is not None
    kwargs = observed["loader"]
    assert kwargs["data_protocol"] == exp.DATA_PROTOCOL
    assert kwargs["replay_slots"] == 131_072
    assert kwargs["windows_per_generation"] == 8
    assert kwargs["replay_phase_block_batches"] == 25
    assert kwargs["num_workers"] == 24
    assert kwargs["source_manifest_sha256"] == exp.SOURCE_MANIFEST_SHA256


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


def _compact_replay(frames: int) -> dict[str, object]:
    compact: dict[str, object] = {}
    for name, encoding in POLICY_WORLD_MDS_COLUMNS.items():
        if encoding == "str":
            compact[name] = "replay-1"
        elif encoding == "int":
            compact[name] = 0
        else:
            compact[name] = np.zeros(frames, dtype=np.dtype(encoding.removeprefix("ndarray:")))
    compact.update(
        policy_world_schema_version=POLICY_WORLD_SCHEMA_VERSION,
        source_schema_version=7,
        replay_id="replay-1",
        num_frames=frames,
    )
    for port in ("p1", "p2"):
        values = {name: np.zeros(frames, dtype=np.int32) for name in PACKED_STATE_SUFFIXES}
        values["stock"].fill(4)
        values["direction"] = np.ones(frames, dtype=np.float32)
        if port == "p2":
            values["stock"][-1] = 0
        compact[f"{port}_state"] = pack_player_state(values)
        compact[f"{port}_percent"] = np.arange(frames, dtype=np.float32)
    return compact


def test_compact_physical_labels_match_full_return_labels() -> None:
    compact = _compact_replay(12)
    expected = returns_lib.compact_policy_returns(
        compact,
        gamma=0.9,
        damage_shaping=1.0,
        win_reward=50.0,
        stock_value=120.0,
        suffix=exp._RETURN_SUFFIX,
    )
    labels = exp.O50ReplayLabels(
        player_lookup=ReplayPlayerLookup({"replay-1": (7, 11)}),
        gamma=0.9,
        damage_shaping=1.0,
        win_reward=50.0,
        stock_value=120.0,
    )(compact)

    for name, values in expected.items():
        np.testing.assert_array_equal(labels[name], values)
    np.testing.assert_array_equal(labels["p1_player_id"], np.asarray(7, dtype=np.int32))
    np.testing.assert_array_equal(labels["p2_player_id"], np.asarray(11, dtype=np.int32))


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
