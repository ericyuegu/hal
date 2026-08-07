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
    assert cfg.val_n_batches == 32
    assert cfg.gradient_diagnostic_batch_size == 64
    assert (cfg.eval_every, cfg.eval_n_matchups, cfg.final_eval_n_matchups) == (4096, 32, 96)
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
    val_batch = object()

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
    assert kwargs["eval_incremental_kv"] is True
    assert kwargs["wandb_label"] == "p2-kv"
    assert calls["upload"] == (output_dir, run_dir, "*")
    assert calls["uploader_run"] == "p1-run"
    assert calls["closed"] is True


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
    gen = torch.Generator().manual_seed(7)
    features = {}
    for prefix in exp._PLAYER_PREFIXES:
        for name in FLOAT_FEATURES:
            features[f"{prefix}_{name}"] = torch.randn(2, cfg.L_ctx, generator=gen)
        for name, (vocab, _) in CAT_FEATURES.items():
            features[f"{prefix}_{name}"] = torch.randint(0, vocab, (2, cfg.L_ctx), generator=gen)
    for channel in ACTION_CHANNELS:
        features[f"ego_{channel}"] = torch.rand(2, cfg.L_ctx, generator=gen)
    for name in ("ego_character", "opp_character", "stage"):
        features[name] = torch.randint(0, 26, (2, cfg.L_ctx), generator=gen)
    batch = TrainBatch(
        context=Context(features=features, ctx_pad=torch.tensor([0, 1])),
        target=torch.rand(2, 2, A_DIM, generator=gen),
    )
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
def test_train_runs_end_to_end_without_value_or_weight_logs(tmp_path, monkeypatch, capsys) -> None:
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
        val_n_batches=1,
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
