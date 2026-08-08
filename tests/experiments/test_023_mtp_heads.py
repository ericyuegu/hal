import hashlib
import importlib.util
import inspect
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from hal.data.feature_stats import FeatureStats
from hal.training.dataloader import make_loader
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import Context
from hal.training.features import TrainBatch
from hal.training.features import preprocess

_EXPERIMENT = Path(__file__).resolve().parents[2] / "experiments" / "023_mtp_heads.py"
_DEV_MDS = _EXPERIMENT.parents[1] / "data" / "processed" / "dev" / "mds"


def _load_experiment():
    spec = importlib.util.spec_from_file_location("exp023", _EXPERIMENT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp = _load_experiment()


def _stats() -> dict[str, FeatureStats]:
    keys = (*FLOAT_FEATURES, *(f"nana_{name}" for name in FLOAT_FEATURES))
    return {name: FeatureStats(mean=0.0, std=1.0, min=-1.0, max=1.0) for name in keys}


def _loss_inputs(offset_values: dict[int, float], *, length: int = 4):
    nll = {}
    transition = {}
    for offset, value in offset_values.items():
        for name in exp._GROUP_NAMES:
            nll[(offset, name)] = torch.full((length,), value)
            transition[(offset, name)] = torch.zeros(length, dtype=torch.bool)
    return nll, transition


def test_defaults_match_the_e0_plan() -> None:
    cfg = exp.TrainConfig()
    assert cfg.head_offsets == (1, 5, 9, 13)
    assert cfg.aux_loss_weight == 1.0
    assert (cfg.d_model, cfg.n_layers, cfg.n_heads) == (256, 8, 4)
    assert (cfg.attn_window, cfg.L_ctx, cfg.batch_size) == (0, 256, 512)
    assert cfg.eval_incremental_kv is False
    assert cfg.batch_size * cfg.L_ctx == 131072
    assert (cfg.max_steps, cfg.warmup_steps) == (2**14, 500)
    assert (cfg.muon_lr, cfg.adam_lr, cfg.weight_decay) == (0.02, 8.5e-4, 0.01)
    assert cfg.amp_dtype == "bfloat16"
    assert cfg.compile_trunk is True
    assert cfg.windows_per_replay == 4
    assert cfg.reservoir_capacity == 4096
    assert cfg.predownload == 512
    assert cfg.val_n_samples == 1192
    assert cfg.gradient_diagnostic_batch_size == 64
    assert (cfg.eval_every, cfg.eval_n_matchups, cfg.final_eval_n_matchups) == (4096, 32, 96)
    assert cfg.eval_max_parallel == 32
    assert cfg.eval_max_frames == 7200
    assert cfg.data_root == "data/processed/ranked-anonymized-1/mds-policy-v7"
    assert cfg.compact_data
    assert cfg.action_vocab == 1024
    assert cfg.mds_schema_version == 7
    assert cfg.cache_limit_gb == 128


def test_matched_p1_geometry_keeps_tokens_and_attention_work_close() -> None:
    p0 = exp.TrainConfig()
    p1 = exp.TrainConfig(L_ctx=1024, batch_size=128, attn_window=128)

    exp.validate_config(p1, has_button_combo_counts=False)
    assert p1.batch_size * p1.L_ctx == p0.batch_size * p0.L_ctx == 131072

    p0_edges = p0.batch_size * p0.L_ctx * (p0.L_ctx + 1) // 2
    p1_edges_per_sample = sum(min(position + 1, p1.attn_window) for position in range(p1.L_ctx))
    p1_edges = p1.batch_size * p1_edges_per_sample
    assert (p0_edges, p1_edges) == (16_842_752, 15_736_832)
    assert p1_edges / p0_edges == pytest.approx(0.93434, rel=1e-5)
    assert not p1.eval_incremental_kv


def test_gradient_diagnostics_select_only_the_shared_representation() -> None:
    model = exp.GPT(exp.TrainConfig(d_model=32, n_layers=2, n_heads=2))
    selected = {id(parameter) for parameter in exp._representation_parameters(model)}
    names = {name for name, parameter in model.named_parameters() if id(parameter) in selected}
    expected_prefixes = ("cat_embeds.", "char_emb.", "stage_emb.", "ctx_proj.", "trunk.")

    assert names
    assert names == {name for name, _ in model.named_parameters() if name.startswith(expected_prefixes)}
    assert not any(name.startswith("heads.") for name in names)


def test_compact_config_requires_cooldown_capacity() -> None:
    cfg = exp.TrainConfig(batch_size=8, reservoir_capacity=8)
    with pytest.raises(ValueError, match="twice the micro-batch"):
        exp.validate_config(cfg, has_button_combo_counts=True)


def test_old_checkpoint_config_keeps_512_action_rows() -> None:
    saved = asdict(exp.TrainConfig(d_model=32, n_layers=1, n_heads=2, L_ctx=16))
    saved.pop("action_vocab")
    saved.pop("compact_data")
    cfg = exp._cfg_from_state(saved)
    assert cfg.action_vocab == 512
    assert not cfg.compact_data
    assert exp.GPT(cfg).cat_embeds["action"].num_embeddings == 512


def test_old_validation_batch_cap_loads_the_fixed_sample_default() -> None:
    saved = asdict(exp.TrainConfig())
    saved.pop("val_n_samples")
    saved["val_n_batches"] = 32

    cfg = exp._cfg_from_state(saved)

    assert cfg.val_n_samples == 1192


def test_old_host_scaled_eval_config_loads_the_fixed_parallel_default() -> None:
    saved = asdict(exp.TrainConfig())
    saved.pop("eval_max_parallel")
    saved["eval_parallel_per_cpu"] = 1.0

    cfg = exp._cfg_from_state(saved)

    assert cfg.eval_max_parallel == 32


def test_h2h_reference_download_checks_the_pinned_sha256(tmp_path, monkeypatch) -> None:
    payload = b"fixed P0 checkpoint"
    expected = hashlib.sha256(payload).hexdigest()

    def fake_download(run_name, dest_dir, *, name):
        assert (run_name, name) == ("p0-run", "final.pt")
        dest_dir.mkdir(parents=True)
        checkpoint = dest_dir / name
        checkpoint.write_bytes(payload)
        return checkpoint

    monkeypatch.setattr(exp, "download_latest", fake_download)
    cfg = exp.TrainConfig(
        final_h2h_reference_run="p0-run",
        final_h2h_reference_sha256=expected,
    )

    checkpoint = exp._fetch_h2h_reference(cfg, tmp_path)

    assert checkpoint.read_bytes() == payload


def test_h2h_reference_download_rejects_the_wrong_sha256(tmp_path, monkeypatch) -> None:
    def fake_download(run_name, dest_dir, *, name):
        dest_dir.mkdir(parents=True)
        checkpoint = dest_dir / name
        checkpoint.write_bytes(b"wrong checkpoint")
        return checkpoint

    monkeypatch.setattr(exp, "download_latest", fake_download)
    cfg = exp.TrainConfig(
        final_h2h_reference_run="p0-run",
        final_h2h_reference_sha256="0" * 64,
    )

    with pytest.raises(RuntimeError, match="reference checkpoint SHA-256 mismatch"):
        exp._fetch_h2h_reference(cfg, tmp_path)


@pytest.mark.parametrize("digest", ["0" * 63, "z" * 64])
def test_config_rejects_an_invalid_h2h_reference_sha256(digest: str) -> None:
    cfg = exp.TrainConfig(final_h2h_reference_run="p0-run", final_h2h_reference_sha256=digest)

    with pytest.raises(ValueError, match="64-character hexadecimal"):
        exp.validate_config(cfg, has_button_combo_counts=False)


def test_config_rejects_an_h2h_sha_without_a_reference_run() -> None:
    cfg = exp.TrainConfig(final_h2h_reference_sha256="0" * 64)

    with pytest.raises(ValueError, match="requires final_h2h_reference_run"):
        exp.validate_config(cfg, has_button_combo_counts=False)


def test_eval_parallelism_is_capped_by_config_and_sample_count() -> None:
    cfg = exp.TrainConfig(eval_max_parallel=32)

    assert exp._eval_max_parallel(cfg, 96) == 32
    assert exp._eval_max_parallel(cfg, 12) == 12


def test_misc_action_state_is_not_a_model_feature() -> None:
    batch = {"ego_misc_as": np.array([float("nan"), 3.0], dtype=np.float32)}
    assert preprocess(batch, {}) == {}


def test_finite_gradient_norm_accepts_a_finite_update() -> None:
    model = torch.nn.Linear(2, 1)
    objective = model(torch.ones(1, 2)).sum()
    objective.backward()

    norm = exp._finite_gradient_norm(model, objective.detach(), step=4)

    assert torch.isfinite(norm)


def test_finite_gradient_norm_rejects_nonfinite_loss() -> None:
    model = torch.nn.Linear(2, 1)
    objective = model(torch.ones(1, 2)).sum()
    objective.backward()

    with pytest.raises(FloatingPointError, match="step 4: loss is not finite"):
        exp._finite_gradient_norm(model, torch.tensor(float("nan")), step=4)


def test_replay_overlap_is_unknown_for_the_first_step() -> None:
    assert exp._replay_overlap(None, {"a", "b"}) is None


def test_replay_overlap_counts_reused_ids() -> None:
    assert exp._replay_overlap(frozenset({"a", "b"}), {"b", "c"}) == 1


def test_finite_gradient_norm_rejects_nonfinite_gradient() -> None:
    model = torch.nn.Linear(2, 1)
    objective = model(torch.ones(1, 2)).sum()
    objective.backward()
    model.weight.grad[0, 0] = float("nan")

    with pytest.raises(FloatingPointError, match="step 4: gradients are not finite"):
        exp._finite_gradient_norm(model, objective.detach(), step=4)


def test_data_loading_starts_train_prefetch_without_consuming_a_batch() -> None:
    events = []
    train_batch = object()
    val_batch = TrainBatch(
        context=Context(features={}, ctx_pad=torch.zeros(1, dtype=torch.long)),
        target=torch.zeros(1, 1, A_DIM),
        replay_ids=("val-0",),
    )

    class TrainIterator:
        def __iter__(self):
            return self

        def __next__(self):
            events.append("train_next")
            return train_batch

    class TrainLoader:
        def __iter__(self):
            events.append("train_iter")
            return TrainIterator()

    class ValLoader:
        def __iter__(self):
            events.append("val_iter")
            yield val_batch

    train_iterator, pool, future, started_at = exp._start_data_loading(TrainLoader(), ValLoader(), 1)
    cached, finished_at = future.result(timeout=2)
    pool.shutdown()

    assert events[0] == "train_iter"
    assert "train_next" not in events
    assert cached == [val_batch]
    assert finished_at >= started_at
    assert next(train_iterator) is train_batch


def test_validation_cache_takes_an_exact_sample_count() -> None:
    def batch(start: int, size: int) -> TrainBatch:
        return TrainBatch(
            context=Context(
                features={"x": torch.arange(start, start + size)[:, None]},
                ctx_pad=torch.zeros(size, dtype=torch.long),
            ),
            target=torch.zeros(size, 1, A_DIM),
            replay_ids=tuple(f"val-{i}" for i in range(start, start + size)),
        )

    _, pool, future, _ = exp._start_data_loading([], [batch(0, 3), batch(3, 3)], 5)
    cached, _ = future.result(timeout=2)
    pool.shutdown()

    assert [item.target.shape[0] for item in cached] == [3, 2]
    assert [replay for item in cached for replay in item.replay_ids] == [f"val-{i}" for i in range(5)]
    assert torch.cat([item.context.features["x"] for item in cached]).flatten().tolist() == list(range(5))


@pytest.mark.skipif(not (_DEV_MDS / "train").is_dir(), reason="local dev MDS is not available")
def test_input_projection_preserves_every_consumed_tensor_and_loss() -> None:
    kwargs = dict(
        data_root=str(_DEV_MDS),
        split="train",
        stats=_stats(),
        L_ctx=32,
        L_chunk=2,
        batch_size=2,
        seed=3,
        num_workers=0,
        windows_per_replay=1,
        schema_version=5,
    )
    full = next(iter(make_loader(**kwargs)))
    projected = next(iter(make_loader(**kwargs, projection=exp._INPUT_PROJECTION)))

    assert projected.context.ctx_pad.equal(full.context.ctx_pad)
    assert projected.target.equal(full.target)
    assert set(projected.context.features) < set(full.context.features)
    for name, value in projected.context.features.items():
        assert value.equal(full.context.features[name]), name

    cfg = exp.TrainConfig(
        d_model=32,
        n_layers=1,
        n_heads=2,
        L_ctx=32,
        attn_window=0,
        head_offsets=(1, 2),
        batch_size=2,
        compile_trunk=False,
    )
    model = exp.GPT(cfg).eval()
    full_parts = exp.action_loss(model, full)
    projected_parts = exp.action_loss(model, projected)
    for key in full_parts.nll:
        torch.testing.assert_close(projected_parts.nll[key], full_parts.nll[key])


def test_run_tag_names_attention_and_decode_mode() -> None:
    baseline = exp.TrainConfig()
    swa = exp.TrainConfig(attn_window=128, eval_incremental_kv=True)
    assert "-full-recompute-" in exp._model_tag(baseline)
    assert "-swa128-kv-" in exp._model_tag(swa)


def test_eval_protocol_records_the_actual_model_dtype(tmp_path) -> None:
    cfg = exp.TrainConfig()
    protocol = exp._eval_protocol(
        cfg,
        settings=exp.DecodeSettings(1.0, None, 0, 0.0, False),
        exec_horizon=1,
        default_n_matchups=3,
        model_dtype="torch.float16",
    )
    path = tmp_path / "match_rows.json"
    exp._write_match_rows(path, [], protocol)

    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 2
    assert payload["protocol"]["model_dtype"] == "torch.float16"
    assert payload["protocol"]["eval_incremental_kv"] is False
    assert payload["protocol"]["cpu_level"] == 9
    assert payload["protocol"]["ego_port"] == 1
    assert payload["protocol"]["seed_stage"] == int(exp.PRIOR_SWEEP_SEED_STAGE.value)
    assert len(payload["protocol"]["matchup_schedule_sha256"]) == 64
    assert payload["protocol"]["instant_match_restart"] is True
    assert payload["protocol"]["stage_policy"] == "battlefield_then_random_legal"
    assert payload["protocol"]["completion_policy"] == "finish_in_flight_wave"
    assert payload["protocol"]["active_frame_policy"] == "frame_id_gte_0_exclude_zero_active"
    assert payload["protocol"]["uncertainty_policy"] == "bootstrap_boots_2000"
    assert payload["protocol"]["start_retries"] == 2


def test_eval_protocol_accepts_fixed_parallelism() -> None:
    protocol = exp._eval_protocol(
        exp.TrainConfig(),
        settings=exp.DecodeSettings(1.0, None, 0, 0.0, False),
        exec_horizon=1,
        default_n_matchups=96,
        model_dtype="torch.float16",
        max_parallel=32,
    )

    assert protocol.max_parallel == 32


@pytest.mark.parametrize("max_parallel", [0, 4])
def test_eval_protocol_rejects_invalid_fixed_parallelism(max_parallel) -> None:
    with pytest.raises(ValueError, match="max_parallel must be in"):
        exp._eval_protocol(
            exp.TrainConfig(),
            settings=exp.DecodeSettings(1.0, None, 0, 0.0, False),
            exec_horizon=1,
            default_n_matchups=3,
            model_dtype="torch.float16",
            max_parallel=max_parallel,
        )


def test_eval_sweep_uses_recorded_cpu_protocol(monkeypatch) -> None:
    cfg = exp.TrainConfig()
    protocol = exp._eval_protocol(
        cfg,
        settings=exp.DecodeSettings(1.0, None, 0, 0.0, False),
        exec_horizon=1,
        default_n_matchups=3,
        model_dtype="torch.float16",
    )
    seen = {}

    def fake_sweep(_factory, **kwargs):
        seen.update(kwargs)
        return [], []

    monkeypatch.setattr(exp, "sweep_vs_cpu_prior_with_rows", fake_sweep)
    exp._run_eval_sweep(lambda: object(), protocol=protocol, replay_dir=None, rows_path=None)

    assert seen["cpu_level"] == protocol.cpu_level
    assert seen["ego_port"] == protocol.ego_port
    assert int(seen["seed_stage"].value) == protocol.seed_stage
    assert seen["session_cfg"].instant_match_restart is protocol.instant_match_restart
    assert seen["start_retries"] == protocol.start_retries


def test_eval_sweep_records_decode_telemetry(tmp_path, monkeypatch) -> None:
    cfg = exp.TrainConfig()
    protocol = exp._eval_protocol(
        cfg,
        settings=exp.DecodeSettings(1.0, None, 0, 0.0, False),
        exec_horizon=1,
        default_n_matchups=1,
        model_dtype="torch.float16",
    )
    telemetry = exp.DecodeTelemetry()

    def fake_sweep(factory, **_kwargs):
        policy = factory()
        policy(0, {"a": object(), "b": object()})
        return [], []

    monkeypatch.setattr(exp, "sweep_vs_cpu_prior_with_rows", fake_sweep)
    metrics = exp._run_eval_sweep(
        lambda: lambda _frame_index, _obs: {},
        protocol=protocol,
        replay_dir=None,
        rows_path=tmp_path / "match_rows.json",
        telemetry=telemetry,
    )

    assert metrics["decode_policy_calls"] == 1
    assert metrics["decode_slot_frames"] == 2
    assert metrics["decode_model_forwards"] == 0
    assert metrics["eval_wall_seconds"] >= 0
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics


def test_decode_telemetry_measures_cpu_model_forwards() -> None:
    telemetry = exp.DecodeTelemetry()

    result = telemetry.model_forward(lambda: torch.ones(2), rows=2, device=torch.device("cpu"))
    metrics = telemetry.metrics()

    assert torch.equal(result, torch.ones(2))
    assert metrics["decode_model_forwards"] == 1
    assert metrics["decode_model_rows"] == 2
    assert metrics["decode_model_forward_seconds"] >= 0
    assert metrics["decode_model_forward_ms_per_row"] >= 0


def test_manual_eval_overrides_checkpoint_incremental_mode(tmp_path, monkeypatch) -> None:
    cfg = exp.TrainConfig(
        d_model=32,
        n_layers=2,
        n_heads=2,
        L_ctx=32,
        attn_window=8,
        eval_incremental_kv=False,
        eval_fp16=False,
    )
    model = exp.GPT(cfg).eval()
    seen = {}

    monkeypatch.setattr(exp, "_load_ckpt", lambda _path: (model, cfg, _stats(), {"step": 7}))

    def fake_make_policy(_model, _stats_arg, eval_cfg, **_kwargs):
        seen["cfg"] = eval_cfg
        return object()

    def fake_sweep(factory, *, protocol, **_kwargs):
        factory()
        seen["protocol"] = protocol
        return {}

    monkeypatch.setattr(exp, "make_policy", fake_make_policy)
    monkeypatch.setattr(exp, "_run_eval_sweep", fake_sweep)
    exp.eval_ckpt(
        "checkpoint.pt",
        eval_output_dir=str(tmp_path),
        eval_incremental_kv=True,
        eval_n_matchups=1,
    )

    assert seen["cfg"].eval_incremental_kv is True
    assert seen["protocol"].eval_incremental_kv is True
    assert cfg.eval_incremental_kv is False


def test_manual_eval_rejects_incremental_override_for_full_attention(tmp_path, monkeypatch) -> None:
    cfg = exp.TrainConfig(d_model=32, n_layers=1, n_heads=2, L_ctx=16, attn_window=0, eval_fp16=False)
    model = exp.GPT(cfg).eval()
    monkeypatch.setattr(exp, "_load_ckpt", lambda _path: (model, cfg, _stats(), {"step": 7}))

    with pytest.raises(ValueError, match="needs attn_window > 0"):
        exp.eval_ckpt(
            "checkpoint.pt",
            eval_output_dir=str(tmp_path),
            eval_incremental_kv=True,
            eval_n_matchups=1,
        )


def test_final_decode_uses_the_checkpoint_decode_dtype(monkeypatch) -> None:
    cfg = exp.TrainConfig(d_model=32, n_layers=1, n_heads=2, L_ctx=16, eval_fp16=True)
    model = exp.GPT(cfg)
    monkeypatch.setattr(exp, "DEVICE", "cuda")

    exp._prepare_final_decode_model(model, cfg)

    assert {parameter.dtype for parameter in model.parameters()} == {torch.float16}
    assert model.main_centers.dtype == torch.float32
    assert model.c_centers.dtype == torch.float32
    assert model.trig_centers.dtype == torch.float32


def test_eval_run_downloads_checkpoint_and_uploads_labeled_evidence(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    calls = {}

    def fake_download(run_name, dest_dir, *, name):
        calls["download"] = (run_name, dest_dir, name)
        dest_dir.mkdir(parents=True)
        path = dest_dir / name
        path.touch()
        return path

    def fake_eval(checkpoint, **kwargs):
        calls["eval"] = (checkpoint, kwargs)
        output = Path(kwargs["eval_output_dir"])
        output.mkdir(parents=True)
        (output / "match_rows.json").write_text("{}")
        return {}

    class Uploader:
        def __init__(self, run_name):
            calls["uploader_run"] = run_name

        def upload_tree(self, root, *, base, pattern="*"):
            calls["upload"] = (root, base, pattern)
            return 1

        def close(self):
            calls["closed"] = True

    monkeypatch.setattr(exp, "download_latest", fake_download)
    monkeypatch.setattr(exp, "eval_ckpt", fake_eval)
    monkeypatch.setattr(exp, "BackgroundUploader", Uploader)
    exp.main(
        exp.Args(
            eval_run="p1-run",
            wandb_run_id="wandb-id",
            wandb_label="p2-kv",
            eval_n_matchups=96,
            eval_max_parallel=32,
            eval_decode="kv",
        )
    )

    run_dir = (tmp_path / "runs" / "p1-run").resolve()
    output_dir = run_dir / "manual_evals" / "p2-kv"
    assert calls["download"] == ("p1-run", Path("runs/p1-run/manual_checkpoints"), "final.pt")
    checkpoint, kwargs = calls["eval"]
    assert checkpoint == Path("runs/p1-run/manual_checkpoints/final.pt").as_posix()
    assert kwargs["eval_output_dir"] == str(output_dir)
    assert kwargs["eval_n_matchups"] == 96
    assert kwargs["eval_max_parallel"] == 32
    assert kwargs["eval_incremental_kv"] is True
    assert kwargs["wandb_label"] == "p2-kv"
    assert calls["upload"] == (output_dir, run_dir, "*")
    assert calls["uploader_run"] == "p1-run"
    assert calls["closed"] is True


def test_periodic_eval_upload_keeps_metrics_result_and_log(tmp_path) -> None:
    run_dir = tmp_path / "run"
    replay_dir = run_dir / "replays" / "step_004096"
    replay_dir.mkdir(parents=True)
    paths = (
        replay_dir / "match_rows.json",
        replay_dir / "metrics.json",
        run_dir / "eval_results" / "step_004096.json",
        run_dir / "eval_logs" / "step_004096.log",
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")

    class Uploader:
        def __init__(self) -> None:
            self.files = []

        def upload_tree(self, root, *, base, pattern):
            assert (root, base, pattern) == (replay_dir, run_dir, "*.slp")
            return 47

        def upload(self, path, *, key):
            self.files.append((path, key))

    uploader = Uploader()
    counts = exp._queue_periodic_eval_evidence(
        uploader,
        run_dir=run_dir,
        replay_dir=replay_dir,
        result_path=paths[2],
        log_path=paths[3],
    )

    assert counts == (47, 4)
    assert uploader.files == [(path, str(path.relative_to(run_dir))) for path in paths]


def test_heads_are_independent_linear_projections() -> None:
    cfg = exp.TrainConfig(d_model=32, n_layers=1, n_heads=2, L_ctx=16, attn_window=8)
    model = exp.GPT(cfg)
    assert len(model.heads) == 4
    assert all(isinstance(head, exp.IndependentHead) for head in model.heads)
    assert all(isinstance(head.proj, torch.nn.Linear) for head in model.heads)
    assert not any(name.startswith("value_head.") for name, _ in model.named_parameters())

    h = torch.randn(2, 3, cfg.d_model)
    logits = model.heads[0].logits(h)
    assert {name: value.shape for name, value in logits.items()} == {
        name: (2, 3, vocab) for name, vocab in zip(exp._GROUP_NAMES, exp._GROUP_VOCABS, strict=True)
    }
    assert model.heads[0].proj(h).shape == (2, 3, exp.A_VOCAB)


def _small_head_cfg(*, head_mode: str):
    return exp.TrainConfig(
        d_model=32,
        n_layers=1,
        n_heads=2,
        L_ctx=8,
        attn_window=0,
        head_offsets=(1, 2),
        batch_size=2,
        compile_trunk=False,
        head_mode=head_mode,
        action_mlp_ratio=2,
    )


def _factored_cfg(order=("c_stick", "triggers", "buttons", "main_stick")):
    cfg = _small_head_cfg(head_mode="factored_mlp")
    cfg.action_group_order = order
    cfg.action_condition_dim = 8
    return cfg


def _target_indices(model, batch):
    actions = torch.cat([exp.stack_actions(batch.context.features), batch.target], dim=1)
    quantized = exp._quantize(model, actions)
    length = actions.shape[1] - batch.target.shape[1]
    return tuple(quantized[:, offset : offset + length] for offset in model.head_offsets)


@pytest.mark.parametrize(
    "order",
    [
        ("buttons", "main_stick", "c_stick"),
        ("buttons", "buttons", "c_stick", "triggers"),
        ("buttons", "main_stick", "c_stick", "unknown"),
    ],
)
def test_factored_group_order_must_be_a_permutation(order) -> None:
    cfg = _factored_cfg(order)
    with pytest.raises(ValueError, match="action_group_order must be a permutation"):
        exp.validate_config(cfg, has_button_combo_counts=True)


@pytest.mark.parametrize(
    "order",
    [
        ("c_stick", "triggers", "buttons", "main_stick"),
        ("main_stick", "buttons", "triggers", "c_stick"),
    ],
)
def test_factored_mlp_starts_as_the_exact_state_mlp(order) -> None:
    state_cfg = _small_head_cfg(head_mode="state_mlp")
    factored_cfg = _factored_cfg(order)
    torch.manual_seed(41)
    state_model = exp.GPT(state_cfg).eval()
    torch.manual_seed(41)
    factored_model = exp.GPT(factored_cfg).eval()
    batch = _random_batch(state_cfg)
    targets = _target_indices(factored_model, batch)

    for name, value in state_model.state_dict().items():
        torch.testing.assert_close(value, factored_model.state_dict()[name], rtol=0, atol=0)
    adapter = factored_model.head_adapter
    assert isinstance(adapter, exp.FactoredMLPAdapter)
    assert torch.count_nonzero(adapter.state_proj.weight) > 0
    for projection in adapter.condition_projs.values():
        assert torch.count_nonzero(projection.weight) == 0
    for output_head in adapter.residual_projs:
        for projection in output_head.values():
            assert torch.count_nonzero(projection.weight) == 0
            assert torch.count_nonzero(projection.bias) == 0

    with torch.no_grad():
        state_hidden = state_model(batch.context.features, batch.context.ctx_pad)
        factored_hidden = factored_model(batch.context.features, batch.context.ctx_pad)
        state_logits = state_model.all_group_logits(state_hidden)
        factored_logits = factored_model.all_group_logits(factored_hidden, targets)
    for state_head, factored_head in zip(state_logits, factored_logits, strict=True):
        for name in exp._GROUP_NAMES:
            torch.testing.assert_close(state_head[name], factored_head[name], rtol=0, atol=0)


def test_factored_embeddings_reject_an_out_of_range_class() -> None:
    cfg = _factored_cfg(("buttons", "main_stick", "c_stick", "triggers"))
    model = exp.GPT(cfg)
    batch = _random_batch(cfg)
    hidden = model(batch.context.features, batch.context.ctx_pad)
    targets = list(_target_indices(model, batch))
    targets[0] = targets[0].clone()
    targets[0][..., exp._BUTTONS_G] = exp._GROUP_VOCABS[exp._BUTTONS_G]

    with pytest.raises(AssertionError, match="buttons class index is out of range"):
        model.all_group_logits(hidden, tuple(targets))


def test_factored_gradient_path_opens_in_three_updates() -> None:
    cfg = _factored_cfg()
    model = exp.GPT(cfg)
    adapter = model.head_adapter
    assert isinstance(adapter, exp.FactoredMLPAdapter)
    batch = _random_batch(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    def backward() -> None:
        parts = exp.action_loss(model, batch)
        exp.objective(parts.nll, parts.transition, 1.0, 1.0).backward()

    backward()
    assert exp._parameter_gradient_norm(adapter.residual_projs.parameters()) > 0
    assert exp._parameter_gradient_norm(adapter.condition_projs.parameters()) == 0
    assert exp._parameter_gradient_norm(adapter.action_embeddings.parameters()) == 0
    optimizer.step()
    optimizer.zero_grad()

    backward()
    assert exp._parameter_gradient_norm(adapter.condition_projs.parameters()) > 0
    assert exp._parameter_gradient_norm(adapter.action_embeddings.parameters()) == 0
    optimizer.step()
    optimizer.zero_grad()

    backward()
    assert exp._parameter_gradient_norm(adapter.action_embeddings.parameters()) > 0


def test_factored_teacher_forcing_changes_only_later_groups() -> None:
    cfg = _factored_cfg(("buttons", "main_stick", "c_stick", "triggers"))
    model = exp.GPT(cfg).eval()
    adapter = model.head_adapter
    assert isinstance(adapter, exp.FactoredMLPAdapter)
    for projection in adapter.condition_projs.values():
        torch.nn.init.constant_(projection.weight, 0.2)
    for output_head in adapter.residual_projs:
        for projection in output_head.values():
            torch.nn.init.constant_(projection.weight, 0.1)
    batch = _random_batch(cfg)
    hidden = model(batch.context.features, batch.context.ctx_pad)
    targets = list(_target_indices(model, batch))
    changed = [value.clone() for value in targets]
    changed[0][..., exp._BUTTONS_G] = (changed[0][..., exp._BUTTONS_G] + 1) % exp._GROUP_VOCABS[exp._BUTTONS_G]

    original_logits = model.all_group_logits(hidden, tuple(targets))[0]
    changed_logits = model.all_group_logits(hidden, tuple(changed))[0]

    torch.testing.assert_close(original_logits["buttons"], changed_logits["buttons"], rtol=0, atol=0)
    assert not torch.equal(original_logits["main_stick"], changed_logits["main_stick"])


def test_factored_ancestral_decode_uses_the_declared_order(monkeypatch) -> None:
    cfg = _factored_cfg(("main_stick", "buttons", "triggers", "c_stick"))
    model = exp.GPT(cfg).eval()
    adapter = model.head_adapter
    assert isinstance(adapter, exp.FactoredMLPAdapter)
    calls = []
    original = adapter.group_logits

    def record(head_index, group, base, state_preactivation, prefix, embedded_prefix=None):
        calls.append((group, tuple(prefix)))
        return original(head_index, group, base, state_preactivation, prefix, embedded_prefix)

    monkeypatch.setattr(adapter, "group_logits", record)
    exp.chunk_from_hidden(
        model,
        torch.randn(2, cfg.d_model),
        (1,),
        group_temps=(1.0,) * exp.N_GROUPS,
        argmax=True,
    )

    assert calls == [
        ("main_stick", ()),
        ("buttons", ("main_stick",)),
        ("triggers", ("main_stick", "buttons")),
        ("c_stick", ("main_stick", "buttons", "triggers")),
    ]


def test_slot_group_streams_are_order_invariant_and_reset_local() -> None:
    ids = torch.tensor([11, 22])
    first = Context(
        features={}, ctx_pad=torch.zeros(2, dtype=torch.long), slot_ids=ids, reset=torch.tensor([True, True])
    )
    no_reset = Context(
        features={}, ctx_pad=torch.zeros(2, dtype=torch.long), slot_ids=ids, reset=torch.tensor([False, False])
    )
    reset_one = Context(
        features={}, ctx_pad=torch.zeros(2, dtype=torch.long), slot_ids=ids, reset=torch.tensor([True, False])
    )
    left = exp.SlotGroupRandom(7)
    right = exp.SlotGroupRandom(7)
    left.begin(first)
    right.begin(first)
    left_draws = {name: left.uniforms(name) for name in exp._GROUP_NAMES}
    right_draws = {name: right.uniforms(name) for name in reversed(exp._GROUP_NAMES)}
    for name in exp._GROUP_NAMES:
        torch.testing.assert_close(left_draws[name], right_draws[name], rtol=0, atol=0)
    assert left.state() == right.state()

    left.begin(reset_one)
    right.begin(no_reset)
    for name in exp._GROUP_NAMES:
        left_next = left.uniforms(name)
        right_next = right.uniforms(name)
        torch.testing.assert_close(left_next[1], right_next[1], rtol=0, atol=0)


@pytest.mark.parametrize(
    "order",
    [
        ("c_stick", "triggers", "buttons", "main_stick"),
        ("main_stick", "buttons", "triggers", "c_stick"),
    ],
)
def test_keyed_sampling_matches_e1_at_factored_initialization(order) -> None:
    state_cfg = _small_head_cfg(head_mode="state_mlp")
    factored_cfg = _factored_cfg(order)
    torch.manual_seed(53)
    state_model = exp.GPT(state_cfg).eval()
    torch.manual_seed(53)
    factored_model = exp.GPT(factored_cfg).eval()
    hidden = torch.randn(3, state_cfg.d_model)
    ctx = Context(
        features={},
        ctx_pad=torch.zeros(3, dtype=torch.long),
        slot_ids=torch.tensor([4, 9, 17]),
        reset=torch.tensor([True, True, True]),
    )
    state_streams = exp.SlotGroupRandom(29)
    factored_streams = exp.SlotGroupRandom(29)

    for call in range(12):
        state_streams.begin(ctx)
        factored_streams.begin(ctx)
        state_action = exp.chunk_from_hidden(
            state_model,
            hidden,
            (1,),
            group_temps=(1.0,) * exp.N_GROUPS,
            uniforms=state_streams.uniforms,
        )
        factored_action = exp.chunk_from_hidden(
            factored_model,
            hidden,
            (1,),
            group_temps=(1.0,) * exp.N_GROUPS,
            uniforms=factored_streams.uniforms,
        )
        torch.testing.assert_close(state_action, factored_action, rtol=0, atol=0)
        assert state_streams.state() == factored_streams.state()
        if call == 0:
            ctx = Context(
                features={},
                ctx_pad=ctx.ctx_pad,
                slot_ids=ctx.slot_ids,
                reset=torch.tensor([False, False, False]),
            )


def test_factored_validation_uses_private_rng_and_reports_ancestor_metrics() -> None:
    cfg = _factored_cfg()
    model = exp.GPT(cfg)
    batch = _random_batch(cfg)
    torch.manual_seed(101)
    before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    metrics = exp.val_metrics(model, [batch], cfg)
    gradients = exp.gradient_diagnostics(model, batch, cfg)

    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    if cuda_before is not None:
        for actual, expected in zip(torch.cuda.get_rng_state_all(), cuda_before, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert gradients["grad/state_mlp_residual_norm"] > 0
    assert gradients["grad/factored_condition_norm"] == 0
    assert gradients["grad/factored_embedding_norm"] == 0
    model.eval()
    with torch.no_grad():
        hidden = model(batch.context.features, batch.context.ctx_pad)
        _, valid = exp._multi_offset_targets(batch.context, batch.target, cfg.head_offsets)
        hidden_valid = hidden.reshape(-1, hidden.shape[-1])[valid.reshape(-1)]
        full_actions = torch.cat([exp.stack_actions(batch.context.features), batch.target], dim=1)
        target = exp._quantize(model, full_actions)[:, 1 : 1 + cfg.L_ctx]
        target_valid = target.reshape(-1, exp.N_GROUPS)[valid.reshape(-1)]
        ancestor_logits, _ = exp._ancestor_sampled_logits(
            model,
            hidden_valid,
            model.primary_head_idx,
            exp._factorization_generators(cfg.factorization_diag_seed, hidden.device),
        )
    for group, name in enumerate(exp._GROUP_NAMES):
        expected_nll = (
            torch.nn.functional.cross_entropy(ancestor_logits[name], target_valid[:, group]).item() / exp._LN2
        )
        assert metrics[f"ancestor_nll_off1_{name}"] == pytest.approx(expected_nll)
    for offset in cfg.head_offsets:
        assert f"ancestor_exact_frame_acc_off{offset}" in metrics
        for name in exp._GROUP_NAMES:
            assert f"ancestor_nll_off{offset}_{name}" in metrics
            assert f"ancestor_acc_off{offset}_{name}" in metrics
            assert metrics[f"ancestor_nll_gap_off{offset}_{name}"] == pytest.approx(
                metrics[f"ancestor_nll_off{offset}_{name}"] - metrics[f"nll_off{offset}_{name}"]
            )


@pytest.mark.parametrize(
    ("order", "embedding_count", "total"),
    [
        (("c_stick", "triggers", "buttons", "main_stick"), 9_280, 7_786_110),
        (("main_stick", "buttons", "triggers", "c_stick"), 11_072, 7_787_902),
    ],
)
def test_factored_parameter_counts_match_the_plan(order, embedding_count, total) -> None:
    e1 = exp.GPT(exp.TrainConfig(head_mode="state_mlp", action_mlp_ratio=2))
    e2 = exp.GPT(
        exp.TrainConfig(
            head_mode="factored_mlp",
            action_mlp_ratio=2,
            action_condition_dim=32,
            action_group_order=order,
        )
    )
    counts = exp.parameter_counts(e2)

    assert counts["condition_projection"] == 98_304
    assert counts["action_embedding"] == embedding_count
    assert counts["total"] - exp.parameter_counts(e1)["total"] == 98_304 + embedding_count
    assert counts["total"] == total


@pytest.mark.parametrize(
    "order",
    [
        ("c_stick", "triggers", "buttons", "main_stick"),
        ("main_stick", "buttons", "triggers", "c_stick"),
    ],
)
def test_factored_checkpoint_and_training_step(order) -> None:
    cfg = _factored_cfg(order)
    model = exp.GPT(cfg)
    batch = _random_batch(cfg)
    optimizer = exp.make_optimizer(model, cfg)
    memberships = {id(parameter): group for group in optimizer.param_groups for parameter in group["params"]}
    adapter = model.head_adapter
    assert isinstance(adapter, exp.FactoredMLPAdapter)
    for parameter in (*adapter.condition_projs.parameters(), *adapter.action_embeddings.parameters()):
        assert memberships[id(parameter)]["use_muon"] is False

    parts = exp.action_loss(model, batch)
    loss = exp.objective(parts.nll, parts.transition, cfg.aux_loss_weight, cfg.transition_loss_weight)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)

    saved = model.state_dict()
    loaded_cfg = exp._cfg_from_state(asdict(cfg))
    loaded = exp.GPT(loaded_cfg)
    exp._load_model_state(loaded, saved)
    targets = _target_indices(model, batch)
    with torch.no_grad():
        expected = model.all_group_logits(model(batch.context.features, batch.context.ctx_pad), targets)
        actual = loaded.all_group_logits(loaded(batch.context.features, batch.context.ctx_pad), targets)
    assert loaded_cfg.action_group_order == order
    assert loaded_cfg.action_condition_dim == cfg.action_condition_dim
    for expected_head, actual_head in zip(expected, actual, strict=True):
        for name in exp._GROUP_NAMES:
            torch.testing.assert_close(actual_head[name], expected_head[name], rtol=0, atol=0)


def test_state_mlp_starts_as_the_exact_linear_model() -> None:
    linear_cfg = _small_head_cfg(head_mode="linear")
    mlp_cfg = _small_head_cfg(head_mode="state_mlp")
    torch.manual_seed(23)
    linear = exp.GPT(linear_cfg).eval()
    torch.manual_seed(23)
    mlp = exp.GPT(mlp_cfg).eval()

    linear_state = linear.state_dict()
    mlp_state = mlp.state_dict()
    for name, value in linear_state.items():
        torch.testing.assert_close(value, mlp_state[name], rtol=0, atol=0)

    assert mlp.head_adapter is not None
    assert torch.count_nonzero(mlp.head_adapter.state_proj.weight) > 0
    for output_head in mlp.head_adapter.residual_projs:
        for projection in output_head.values():
            assert torch.count_nonzero(projection.weight) == 0
            assert torch.count_nonzero(projection.bias) == 0

    batch = _random_batch(linear_cfg)
    with torch.no_grad():
        linear_hidden = linear(batch.context.features, batch.context.ctx_pad)
        mlp_hidden = mlp(batch.context.features, batch.context.ctx_pad)
        torch.testing.assert_close(linear_hidden, mlp_hidden, rtol=0, atol=0)
        for linear_logits, mlp_logits in zip(
            linear.all_group_logits(linear_hidden), mlp.all_group_logits(mlp_hidden), strict=True
        ):
            for name in exp._GROUP_NAMES:
                torch.testing.assert_close(linear_logits[name], mlp_logits[name], rtol=0, atol=0)

        linear_action = exp.decode(linear, batch.context, gen=torch.Generator().manual_seed(31))
        mlp_action = exp.decode(mlp, batch.context, gen=torch.Generator().manual_seed(31))
        torch.testing.assert_close(linear_action, mlp_action, rtol=0, atol=0)

    linear_parts = exp.action_loss(linear, batch)
    mlp_parts = exp.action_loss(mlp, batch)
    linear_loss = exp.objective(linear_parts.nll, linear_parts.transition, 1.0, 1.0)
    mlp_loss = exp.objective(mlp_parts.nll, mlp_parts.transition, 1.0, 1.0)
    torch.testing.assert_close(linear_loss, mlp_loss, rtol=0, atol=0)


def test_chunk_logits_compute_the_shared_state_mlp_once() -> None:
    cfg = _small_head_cfg(head_mode="state_mlp")
    model = exp.GPT(cfg).eval()
    assert model.head_adapter is not None
    hidden = torch.randn(3, cfg.d_model)
    calls = 0

    def count_call(_module, _args, _output) -> None:
        nonlocal calls
        calls += 1

    handle = model.head_adapter.state_proj.register_forward_hook(count_call)
    actual = exp.chunk_from_hidden(
        model,
        hidden,
        cfg.head_offsets,
        group_temps=(1.0,) * exp.N_GROUPS,
        argmax=True,
    )
    handle.remove()

    expected = torch.stack(
        [
            exp._sample_action(
                model,
                head_index,
                hidden,
                group_temps=(1.0,) * exp.N_GROUPS,
                btn_support_min=0,
                min_p=0.0,
                click_trigger_fix=False,
                argmax=True,
                gen=None,
            )
            for head_index in range(len(cfg.head_offsets))
        ],
        dim=1,
    )

    assert calls == 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_state_mlp_zero_output_initialization_does_not_kill_learning() -> None:
    cfg = _small_head_cfg(head_mode="state_mlp")
    model = exp.GPT(cfg)
    assert model.head_adapter is not None
    batch = _random_batch(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with torch.no_grad():
        before = {
            name: value.detach().clone()
            for name, value in model.all_group_logits(model(batch.context.features, batch.context.ctx_pad))[0].items()
        }

    first_parts = exp.action_loss(model, batch)
    first_loss = exp.objective(first_parts.nll, first_parts.transition, 1.0, 1.0)
    first_loss.backward()
    for output_head in model.head_adapter.residual_projs:
        for projection in output_head.values():
            assert projection.weight.grad is not None
            assert torch.isfinite(projection.weight.grad).all()
            assert torch.count_nonzero(projection.weight.grad) > 0
    assert model.head_adapter.state_proj.weight.grad is not None
    assert torch.count_nonzero(model.head_adapter.state_proj.weight.grad) == 0

    optimizer.step()
    optimizer.zero_grad()
    with torch.no_grad():
        after = model.all_group_logits(model(batch.context.features, batch.context.ctx_pad))[0]
    assert any(not torch.equal(before[name], after[name]) for name in exp._GROUP_NAMES)

    second_parts = exp.action_loss(model, batch)
    second_loss = exp.objective(second_parts.nll, second_parts.transition, 1.0, 1.0)
    second_loss.backward()

    state_grad = model.head_adapter.state_proj.weight.grad
    assert state_grad is not None
    assert torch.isfinite(state_grad).all()
    assert torch.count_nonzero(state_grad) > 0


@pytest.mark.parametrize("head_weight_decay", [False, True])
def test_state_mlp_parameters_use_the_output_head_adamw_partition(head_weight_decay: bool) -> None:
    cfg = _small_head_cfg(head_mode="state_mlp")
    cfg.head_weight_decay = head_weight_decay
    model = exp.GPT(cfg)
    optimizer = exp.make_optimizer(model, cfg)
    memberships = {id(parameter): group for group in optimizer.param_groups for parameter in group["params"]}

    assert len(memberships) == sum(1 for _ in model.parameters())
    for parameter in exp._output_head_parameters(model):
        group = memberships[id(parameter)]
        assert group["use_muon"] is False
        expected_decay = cfg.weight_decay if head_weight_decay and parameter.ndim >= 2 else 0.0
        assert group["weight_decay"] == expected_decay


def test_state_mlp_checkpoint_round_trip() -> None:
    cfg = _small_head_cfg(head_mode="state_mlp")
    model = exp.GPT(cfg).eval()
    batch = _random_batch(cfg)
    with torch.no_grad():
        expected = model.all_group_logits(model(batch.context.features, batch.context.ctx_pad))

    loaded_cfg = exp._cfg_from_state(asdict(cfg))
    loaded = exp.GPT(loaded_cfg).eval()
    exp._load_model_state(loaded, model.state_dict())
    with torch.no_grad():
        actual = loaded.all_group_logits(loaded(batch.context.features, batch.context.ctx_pad))

    assert loaded_cfg.head_mode == "state_mlp"
    assert loaded_cfg.action_mlp_ratio == 2
    for expected_head, actual_head in zip(expected, actual, strict=True):
        for name in exp._GROUP_NAMES:
            torch.testing.assert_close(expected_head[name], actual_head[name], rtol=0, atol=0)


def test_old_checkpoint_without_head_fields_loads_as_linear() -> None:
    cfg = _small_head_cfg(head_mode="linear")
    model = exp.GPT(cfg).eval()
    saved_cfg = asdict(cfg)
    saved_cfg.pop("head_mode")
    saved_cfg.pop("action_mlp_ratio")

    loaded_cfg = exp._cfg_from_state(saved_cfg)
    loaded = exp.GPT(loaded_cfg).eval()
    exp._load_model_state(loaded, model.state_dict())

    assert loaded_cfg.head_mode == "linear"
    assert loaded.head_adapter is None
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)


def test_state_mlp_parameter_counts_match_the_plan() -> None:
    linear = exp.GPT(exp.TrainConfig(head_mode="linear"))
    mlp = exp.GPT(exp.TrainConfig(head_mode="state_mlp", action_mlp_ratio=2))
    linear_counts = exp.parameter_counts(linear)
    mlp_counts = exp.parameter_counts(mlp)

    assert mlp_counts["state_projection"] == 131_584
    assert mlp_counts["residual_output"] == 728_460
    assert mlp_counts["total"] - linear_counts["total"] == 860_044
    assert mlp_counts["trunk"] == linear_counts["trunk"]
    assert mlp_counts["classifier"] == linear_counts["classifier"]


def test_no_auxiliary_head_returns_the_primary_loss() -> None:
    nll, transition = _loss_inputs({1: 2.0})
    actual = exp.objective(nll, transition, aux_weight=1.0, transition_weight=1.0)
    assert actual.item() == pytest.approx(4 * 2.0)


def test_one_auxiliary_head_gets_the_full_auxiliary_weight() -> None:
    nll, transition = _loss_inputs({1: 2.0, 5: 3.0})
    actual = exp.objective(nll, transition, aux_weight=0.25, transition_weight=1.0)
    assert actual.item() == pytest.approx(4 * 2.0 + 0.25 * 4 * 3.0)


def test_three_auxiliary_heads_share_one_fixed_total_weight() -> None:
    nll, transition = _loss_inputs({1: 2.0, 5: 1.0, 9: 3.0, 13: 5.0})
    actual = exp.objective(nll, transition, aux_weight=0.5, transition_weight=1.0)
    expected = 4 * 2.0 + 0.5 * (4 * 1.0 + 4 * 3.0 + 4 * 5.0) / 3
    assert actual.item() == pytest.approx(expected)


def test_auxiliary_scale_does_not_grow_with_head_count() -> None:
    one_nll, one_transition = _loss_inputs({1: 0.0, 5: 3.0})
    many_nll, many_transition = _loss_inputs({1: 0.0, 5: 3.0, 9: 3.0, 13: 3.0})
    one = exp.objective(one_nll, one_transition, aux_weight=1.0, transition_weight=1.0)
    many = exp.objective(many_nll, many_transition, aux_weight=1.0, transition_weight=1.0)
    torch.testing.assert_close(one, many)


def test_zero_auxiliary_weight_removes_all_auxiliary_loss() -> None:
    nll, transition = _loss_inputs({1: 2.0, 5: 100.0, 9: 200.0})
    actual = exp.objective(nll, transition, aux_weight=0.0, transition_weight=1.0)
    assert actual.item() == pytest.approx(4 * 2.0)


def test_transition_weighting_matches_a_hand_calculation() -> None:
    nll, transition = _loss_inputs({1: 0.0}, length=3)
    for name in exp._GROUP_NAMES:
        nll[(1, name)] = torch.tensor([1.0, 2.0, 7.0])
        transition[(1, name)] = torch.tensor([False, True, False])
    actual = exp.objective(nll, transition, aux_weight=1.0, transition_weight=3.0)
    expected_per_group = (1.0 + 3.0 * 2.0 + 7.0) / (1.0 + 3.0 + 1.0)
    assert actual.item() == pytest.approx(4 * expected_per_group)


def test_e0_has_no_advantage_weight_input() -> None:
    for function in (exp._weighted_mean, exp._offset_objective, exp.objective):
        assert "sample_weight" not in inspect.signature(function).parameters
    fields = exp.TrainConfig.__dataclass_fields__
    assert not any(name.startswith("awr_") for name in fields)
    assert "rank_weights" not in fields


def test_offset_targets_use_the_requested_future_frames() -> None:
    length = 4
    features = {name: torch.zeros(1, length) for name in (f"ego_{ch}" for ch in ACTION_CHANNELS)}
    features["ego_main_stick_x"][0] = torch.tensor([10.0, 11.0, 12.0, 13.0])
    target = torch.zeros(1, 5, len(ACTION_CHANNELS))
    target[0, :, 0] = torch.tensor([14.0, 15.0, 16.0, 17.0, 18.0])
    ctx = Context(features=features, ctx_pad=torch.tensor([1]))

    targets, valid = exp._multi_offset_targets(ctx, target, (1, 3, 5))

    assert targets[1][0, :, 0].tolist() == [11.0, 12.0, 13.0, 14.0]
    assert targets[3][0, :, 0].tolist() == [13.0, 14.0, 15.0, 16.0]
    assert targets[5][0, :, 0].tolist() == [15.0, 16.0, 17.0, 18.0]
    assert valid.tolist() == [[False, True, True, True]]


def test_one_training_step_uses_only_plain_behavior_cloning() -> None:
    cfg = exp.TrainConfig(
        d_model=32,
        n_layers=1,
        n_heads=2,
        L_ctx=8,
        attn_window=0,
        head_offsets=(1, 2),
        batch_size=2,
        max_steps=1,
        warmup_steps=0,
        compile_trunk=False,
        eval_incremental_kv=False,
    )
    batch = _random_batch(cfg)
    model = exp.GPT(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.heads[0].proj.weight.detach().clone()

    parts = exp.action_loss(model, batch)
    loss = exp.objective(parts.nll, parts.transition, cfg.aux_loss_weight, cfg.transition_loss_weight)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert not torch.equal(before, model.heads[0].proj.weight)
    source = inspect.getsource(exp.train).lower()
    for stale_name in ("awr", "rank_weight", "value_head", "critic", "batch.batch"):
        assert stale_name not in source


def _random_batch(cfg, *, batch_size: int = 2) -> TrainBatch:
    gen = torch.Generator().manual_seed(7)
    features = {}
    for prefix in exp._PLAYER_PREFIXES:
        for name in FLOAT_FEATURES:
            features[f"{prefix}_{name}"] = torch.randn(batch_size, cfg.L_ctx, generator=gen)
        for name, (vocab, _) in CAT_FEATURES.items():
            features[f"{prefix}_{name}"] = torch.randint(0, vocab, (batch_size, cfg.L_ctx), generator=gen)
    for channel in ACTION_CHANNELS:
        features[f"ego_{channel}"] = torch.rand(batch_size, cfg.L_ctx, generator=gen)
    for name in ("ego_character", "opp_character", "stage"):
        features[name] = torch.randint(0, 26, (batch_size, cfg.L_ctx), generator=gen)
    return TrainBatch(
        context=Context(features=features, ctx_pad=torch.arange(batch_size)),
        target=torch.rand(batch_size, max(cfg.head_offsets), A_DIM, generator=gen),
    )


def test_policy_telemetry_does_not_change_full_decode() -> None:
    cfg = exp.TrainConfig(
        d_model=32,
        n_layers=1,
        n_heads=2,
        L_ctx=8,
        attn_window=0,
        head_offsets=(1, 2),
        batch_size=2,
        compile_trunk=False,
    )
    model = exp.GPT(cfg).eval()
    ctx = _random_batch(cfg).context
    expected = exp.decode(model, ctx, gen=torch.Generator().manual_seed(11)).numpy()
    telemetry = exp.DecodeTelemetry()
    policy = exp.make_policy(model, _stats(), cfg, device="cpu", decode_seed=11, telemetry=telemetry)

    actual = policy.predict_chunk(ctx, None)

    np.testing.assert_array_equal(actual, expected)
    metrics = telemetry.metrics()
    assert metrics["decode_model_forwards"] == 1
    assert metrics["decode_model_rows"] == 2


def test_checkpoint_decode_parity_covers_rolls_and_mixed_resets() -> None:
    cfg = exp.TrainConfig(
        d_model=32,
        n_layers=2,
        n_heads=2,
        L_ctx=16,
        attn_window=4,
        head_offsets=(1, 2),
        batch_size=2,
        compile_trunk=False,
        require_flex=False,
    )

    result = exp.checkpoint_decode_parity(
        exp.GPT(cfg),
        cfg,
        frames=37,
        slots=3,
        seed=5,
        atol=1e-4,
        rtol=1e-4,
    )

    assert result["passed"]
    assert result["comparisons"] == 111
    assert result["sampled_action_mismatches"] == 0
    assert result["reset_frames"] == {"0": [0], "1": [0, 11], "2": [0, 14, 35]}


def test_parity_run_downloads_checkpoint_and_uploads_evidence(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    calls = {}

    def fake_download(run_name, dest_dir, *, name):
        calls["download"] = (run_name, dest_dir, name)
        dest_dir.mkdir(parents=True)
        path = dest_dir / name
        path.touch()
        return path

    def fake_parity(checkpoint, **kwargs):
        calls["parity"] = (checkpoint, kwargs)
        output = Path(kwargs["output_path"])
        output.parent.mkdir(parents=True)
        output.write_text("{}")
        return {"passed": True}

    class Uploader:
        def __init__(self, run_name):
            calls["uploader_run"] = run_name

        def upload(self, path, *, key):
            calls["upload"] = (path, key)

        def close(self):
            calls["closed"] = True

    monkeypatch.setattr(exp, "download_latest", fake_download)
    monkeypatch.setattr(exp, "run_checkpoint_decode_parity", fake_parity)
    monkeypatch.setattr(exp, "BackgroundUploader", Uploader)
    exp.main(exp.Args(parity_run="p1-run", parity_frames=99, parity_slots=4, parity_seed=7))

    checkpoint, kwargs = calls["parity"]
    run_dir = (tmp_path / "runs" / "p1-run").resolve()
    output = run_dir / "manual_evals" / "p2-parity" / "decode_parity.json"
    assert calls["download"] == ("p1-run", Path("runs/p1-run/manual_checkpoints"), "final.pt")
    assert checkpoint == Path("runs/p1-run/manual_checkpoints/final.pt").as_posix()
    assert kwargs == {"output_path": str(output), "frames": 99, "slots": 4, "seed": 7}
    assert calls["upload"] == (output, "manual_evals/p2-parity/decode_parity.json")
    assert calls["uploader_run"] == "p1-run"
    assert calls["closed"] is True


def test_validation_reports_each_group_at_each_offset() -> None:
    cfg = exp.TrainConfig(
        d_model=32,
        n_layers=1,
        n_heads=2,
        L_ctx=8,
        attn_window=0,
        head_offsets=(1, 2),
        batch_size=2,
        compile_trunk=False,
        eval_incremental_kv=False,
    )

    metrics = exp.val_metrics(exp.GPT(cfg), [_random_batch(cfg)], cfg)

    for offset in cfg.head_offsets:
        assert f"nll_off{offset}" in metrics
        assert f"exact_frame_acc_off{offset}" in metrics
        for group in exp._GROUP_NAMES:
            assert f"nll_off{offset}_{group}" in metrics
            assert f"acc_off{offset}_{group}" in metrics
            assert f"nll_off{offset}_{group}_trans" in metrics
            assert f"nll_off{offset}_{group}_hold" in metrics
            assert f"acc_off{offset}_{group}_trans" in metrics
            assert f"acc_off{offset}_{group}_hold" in metrics
            assert f"pred_change_rate_off{offset}_{group}" in metrics
            assert f"pred_persistence_off{offset}_{group}" in metrics
            assert f"changeF1_off{offset}_{group}" in metrics


def test_state_mlp_validation_reports_residual_and_gradient_norms() -> None:
    cfg = _small_head_cfg(head_mode="state_mlp")
    model = exp.GPT(cfg)
    batch = _random_batch(cfg)

    metrics = exp.val_metrics(model, [batch], cfg)
    gradients = exp.gradient_diagnostics(model, batch, cfg)

    for offset in cfg.head_offsets:
        for group in exp._GROUP_NAMES:
            assert metrics[f"residual_logit_rms_ratio_off{offset}_{group}"] == 0.0
    assert gradients["grad/state_mlp_state_norm"] == 0.0
    assert gradients["grad/state_mlp_residual_norm"] > 0.0


class _FakeRun:
    id = "test-run"

    def __init__(self) -> None:
        self.summary = {}


class _WandbSpy:
    def __init__(self) -> None:
        self.logs: list[dict] = []
        self.run = None

    def init(self, *args, **kwargs):
        self.run = _FakeRun()
        return self.run

    def define_metric(self, *args, **kwargs) -> None:
        return None

    def log(self, payload: dict, *args, **kwargs):
        self.logs.append(payload)


class _Uploader:
    def __init__(self, run_name: str) -> None:
        self.run_name = run_name

    def upload(self, *args, **kwargs) -> None:
        return None

    def upload_tree(self, *args, **kwargs) -> int:
        return 0

    def close(self) -> None:
        return None


@pytest.mark.skipif(not (_DEV_MDS / "train").is_dir(), reason="local dev MDS is not available")
@pytest.mark.parametrize("head_mode", ["linear", "state_mlp"])
def test_train_runs_end_to_end_without_value_or_weight_logs(tmp_path, monkeypatch, capsys, head_mode) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("AWS_BUCKET", "hal-test")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")

    def no_closed_loop(*args, **kwargs):
        raise AssertionError("closed-loop evaluation must be disabled")

    monkeypatch.setattr(exp, "eval_vs_cpu", no_closed_loop)
    spy = _WandbSpy()
    monkeypatch.setattr(exp, "wandb", spy)
    monkeypatch.setattr(exp, "BackgroundUploader", _Uploader)
    cfg = exp.TrainConfig(
        d_model=32,
        n_layers=2,
        n_heads=2,
        L_ctx=64,
        attn_window=16,
        head_offsets=(1, 2),
        batch_size=4,
        grad_accum_steps=1,
        max_steps=1,
        warmup_steps=0,
        compile_trunk=False,
        val_every=0,
        val_n_samples=4,
        gradient_diagnostic_batch_size=2,
        eval_every=0,
        final_eval_n_matchups=0,
        ckpt_every=0,
        data_root=str(_DEV_MDS),
        compact_data=False,
        mds_schema_version=5,
        num_workers=0,
        windows_per_replay=2,
        val_split="train",
        eval_incremental_kv=False,
        head_mode=head_mode,
    )

    exp.train(cfg, _stats(), comment="pytest")

    output = capsys.readouterr().out
    assert "step 0:" in output
    assert "closed-loop eval skipped" in output
    runs = list((tmp_path / "runs").iterdir())
    assert len(runs) == 1
    assert (runs[0] / "final.pt").is_file()
    logged_keys = {key.lower() for payload in spy.logs for key in payload}
    assert max(payload.get("data/train_batches_seen", 0) for payload in spy.logs) == 1
    for stale_name in ("awr", "rank", "value", "critic"):
        assert not any(stale_name in key for key in logged_keys)
