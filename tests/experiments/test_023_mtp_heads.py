import importlib.util
import inspect
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
    for stale_name in ("awr", "rank", "value", "critic"):
        assert not any(stale_name in key for key in logged_keys)
