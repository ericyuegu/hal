"""Contracts for the sparse game-state flow expert experiment."""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import TrainBatch


def _load(name: str, filename: str):
    path = Path(__file__).resolve().parents[2] / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exp = _load("test_exp029", "029_game_state_flow.py")
exp026 = _load("test_exp026_for_029", "026_temporal_mtp.py")


def _cfg(**overrides):
    values = dict(
        d_model=32,
        n_layers=1,
        n_heads=4,
        L_ctx=4,
        temporal_d_model=32,
        temporal_layers=1,
        temporal_heads=4,
        temporal_ff_dim=64,
        group_head_dim=64,
        state_d_model=32,
        state_layers=1,
        state_heads=4,
        state_ff_dim=64,
        state_action_dim=16,
        state_time_dim=16,
        batch_size=2,
        grad_accum_steps=1,
        reservoir_capacity=4,
        max_steps=2,
        compile_trunk=False,
        compile_temporal=False,
        num_workers=0,
        push_to_r2=False,
    )
    return exp.TrainConfig(**{**values, **overrides})


def _train_batch(cfg, batch_size: int = 2) -> TrainBatch:
    context = exp.synthetic_context(cfg, batch_size, torch.device("cpu"))
    return TrainBatch(context=context, target=torch.zeros(batch_size, cfg.sample_chunk_length, A_DIM))


def _state_batch(cfg, *, valid_role: str | None = None) -> exp.StateBatch:
    batch_size = 2
    shape = (batch_size, len(cfg.state_offsets), len(exp.STATE_ROLES), len(exp.STATE_CONTINUOUS))
    valid = torch.ones(shape, dtype=torch.bool) if valid_role is None else torch.zeros(shape, dtype=torch.bool)
    if valid_role is not None:
        valid[..., exp.STATE_ROLES.index(valid_role), 0] = True
    categorical = {
        f"{role}_{name}": torch.zeros(batch_size, len(cfg.state_offsets), dtype=torch.long)
        for role in exp.STATE_ROLES
        for name in exp.STATE_CATEGORICAL
    }
    categorical_valid = {name: torch.zeros_like(value, dtype=torch.bool) for name, value in categorical.items()}
    return exp.StateBatch(
        batch=_train_batch(cfg, batch_size),
        continuous=torch.zeros(shape),
        continuous_valid=valid,
        categorical=categorical,
        categorical_valid=categorical_valid,
        presence={},
    )


def test_defaults_pin_sparse_offsets_nana_weight_and_reference() -> None:
    cfg = exp.TrainConfig()
    expected = (1, 2, 3, 4, 5, 6, 9, 12, 16, 20)
    assert cfg.head_offsets == cfg.state_offsets == expected
    assert cfg.state_nana_weight == 0.2
    assert cfg.final_h2h_reference_run == exp.REFERENCE_026_RUN
    assert cfg.final_h2h_reference_sha256 == exp.REFERENCE_026_SHA256
    assert cfg.final_h2h_n_configs == 64
    assert cfg.final_h2h_max_parallel == 32


def test_varlen_flash_rejects_float32_training() -> None:
    with pytest.raises(ValueError, match="varlen_flash requires amp_dtype='bfloat16'"):
        exp.validate_config(_cfg(amp_dtype="float32"))


def test_experiment_has_no_numbered_experiment_import() -> None:
    path = Path(exp.__file__)
    tree = ast.parse(path.read_text())
    imported = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imported += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(name and name.startswith("experiments.") for name in imported)


def test_policy_initialization_exactly_matches_026() -> None:
    cfg = _cfg()
    common = {name: getattr(cfg, name) for name in exp026.TrainConfig.__dataclass_fields__ if hasattr(cfg, name)}
    torch.manual_seed(17)
    reference = exp026.GPT(exp026.TrainConfig(**common))
    torch.manual_seed(17)
    policy = exp.TrainingModel(cfg).policy
    assert reference.state_dict().keys() == policy.state_dict().keys()
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(value, policy.state_dict()[name], rtol=0, atol=0)


def test_action_prefix_summary_at_offset_20_depends_on_intermediate_action() -> None:
    expert = exp.GameStateExpert(_cfg()).eval()
    actions = torch.zeros(1, 20, A_DIM)
    before = expert.action_summaries(actions)
    actions[:, 7, 0] = 1
    after = expert.action_summaries(actions)
    torch.testing.assert_close(before[:, :6], after[:, :6])
    assert not torch.equal(before[:, -1], after[:, -1])


def test_categorical_predictions_cannot_see_noised_continuous_targets() -> None:
    cfg = _cfg()
    expert = exp.GameStateExpert(cfg).eval()
    batch_size = 2
    actions = torch.randn(batch_size, cfg.sample_chunk_length, A_DIM)
    context = torch.randn(batch_size, cfg.L_ctx, cfg.d_model)
    ctx_pad = torch.zeros(batch_size, dtype=torch.long)
    time = torch.rand(batch_size)
    shape = (batch_size, len(cfg.state_offsets), expert.continuous_dim)
    _, first, first_presence = expert(torch.randn(shape), actions, time, context, ctx_pad)
    _, second, second_presence = expert(torch.randn(shape), actions, time, context, ctx_pad)
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
    for name in first_presence:
        torch.testing.assert_close(first_presence[name], second_presence[name], rtol=0, atol=0)


def test_sparse_state_targets_are_aligned_from_final_context_boundary(monkeypatch) -> None:
    cfg = _cfg()
    length = cfg.L_ctx + cfg.sample_chunk_length
    features: dict[str, torch.Tensor] = {}
    for role in exp.STATE_ROLES:
        for field in exp.STATE_CONTINUOUS:
            features[f"{role}_{field}"] = torch.arange(length)[None].repeat(2, 1).float()
        for field in exp.STATE_CATEGORICAL:
            features[f"{role}_{field}"] = torch.arange(length)[None].repeat(2, 1).long()
    for channel in ACTION_CHANNELS:
        features[f"ego_{channel}"] = torch.zeros(2, length)
    monkeypatch.setattr(exp, "collate_windows", lambda windows: {})
    monkeypatch.setattr(exp, "preprocess", lambda *args, **kwargs: features)
    batch = exp.collate_state_batch([{}], _train_batch(cfg), stats={}, cfg=cfg, projection=exp.BASE_ACTION_PROJECTION)
    expected = torch.tensor([cfg.L_ctx - 1 + offset for offset in cfg.state_offsets]).float()
    torch.testing.assert_close(batch.continuous[0, :, 0, 0], expected)


def test_nana_continuous_loss_is_exactly_one_fifth_of_leader() -> None:
    cfg = _cfg()
    squared = torch.ones(2, len(cfg.state_offsets), len(exp.STATE_ROLES), len(exp.STATE_CONTINUOUS))

    def values(role: str) -> torch.Tensor:
        valid = torch.zeros_like(squared, dtype=torch.bool)
        valid[..., exp.STATE_ROLES.index(role), 0] = True
        return exp.weighted_continuous_loss(squared, valid, cfg)

    leader = values("ego")
    nana = values("ego_nana")
    torch.testing.assert_close(nana, leader * cfg.state_nana_weight, rtol=1e-5, atol=1e-6)


def test_state_only_gradient_reaches_trunk_but_not_action_decoder() -> None:
    cfg = _cfg()
    model = exp.TrainingModel(cfg)
    batch = _state_batch(cfg)
    history, _, _ = exp.prepared_targets(model.policy, batch.batch)
    hidden = model.policy(batch.batch.context.features, batch.batch.context.ctx_pad, history)
    parts = exp.state_loss(model, batch, hidden, cfg, gen=torch.Generator().manual_seed(3))
    parts.loss.backward()
    assert model.policy.ctx_proj.weight.grad is not None
    assert model.policy.trunk.blocks[0].attn.c_attn.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.policy.temporal.blocks.parameters())
    assert all(parameter.grad is None for parameter in model.policy.temporal.outputs.parameters())
    assert all(parameter.grad is None for parameter in model.policy.temporal.trunk_outputs.parameters())


def test_tail_masks_are_seeded_and_keep_dense_six() -> None:
    cfg = _cfg()
    first = exp.sample_tail_mask(32, cfg, device=torch.device("cpu"), gen=torch.Generator().manual_seed(5))
    second = exp.sample_tail_mask(32, cfg, device=torch.device("cpu"), gen=torch.Generator().manual_seed(5))
    assert torch.equal(first, second)
    assert first[:, :6].all()
    assert (~first[:, 6:]).any()


def test_state_loss_ramp_starts_at_zero_and_reaches_quarter() -> None:
    cfg = _cfg()
    assert exp.state_weight(0, cfg) == 0
    assert exp.state_weight(500, cfg) == pytest.approx(0.125)
    assert exp.state_weight(1000, cfg) == pytest.approx(0.25)
    assert exp.state_weight(2000, cfg) == pytest.approx(0.25)


def test_policy_only_loader_drops_state_expert(tmp_path: Path) -> None:
    cfg = _cfg()
    training = exp.TrainingModel(cfg)
    checkpoint = tmp_path / "final.pt"
    torch.save({"cfg": vars(cfg), "model": training.state_dict(), "step": 2}, checkpoint)
    loaded, loaded_cfg, _, _ = exp.load_checkpoint(str(checkpoint), device="cpu")
    assert isinstance(loaded, exp.GPT)
    assert loaded_cfg == cfg
    assert not any(name.startswith("state_expert") for name, _ in loaded.named_parameters())
    for name, value in training.policy.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name])


def test_unprefixed_026_reference_loads_into_policy(tmp_path: Path) -> None:
    cfg = _cfg()
    policy = exp.GPT(cfg)
    checkpoint = tmp_path / "reference.pt"
    torch.save({"cfg": vars(cfg), "model": policy.state_dict(), "step": 7}, checkpoint)
    loaded, _, _, state = exp.load_026_reference(checkpoint, device="cpu")
    assert state["step"] == 7
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name])


def test_reference_resolution_rejects_wrong_digest(monkeypatch, tmp_path: Path) -> None:
    cfg = _cfg()
    checkpoint = tmp_path / "wrong.pt"
    checkpoint.write_bytes(b"wrong")
    monkeypatch.setattr(exp.Path, "is_file", lambda self: False)
    monkeypatch.setattr(exp, "download_latest", lambda *args, **kwargs: checkpoint)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        exp.resolve_h2h_reference(cfg, tmp_path)


def test_tiny_training_run_reaches_final_evals_and_pinned_h2h(monkeypatch, tmp_path: Path) -> None:
    cfg = _cfg(
        max_steps=1,
        val_every=0,
        eval_every=0,
        ckpt_every=0,
        wandb_log_code=False,
        inference_mode="eager",
    )
    batch = _state_batch(cfg)
    reference = tmp_path / "026-final.pt"
    monkeypatch.setattr(exp, "_make_loaders", lambda cfg, stats: ([batch], [batch]))
    monkeypatch.setattr(exp, "make_run_name", lambda *args: "tiny-029")
    monkeypatch.setattr(exp, "setup_run_dir", lambda name: (tmp_path, tmp_path / "replays"))
    monkeypatch.setattr(exp, "resolve_h2h_reference", lambda cfg, run_dir: reference)
    monkeypatch.setattr(exp, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp, "_checkpoint_sha256", lambda path: "a" * 64)
    evaluations: list[tuple[int, int]] = []
    eval_inferences: dict[int, object] = {}
    inference_models: list[object] = []

    def fake_inference(model, cfg, *, bucket=None, **kwargs):
        inference_models.append(model)
        engine = object()
        eval_inferences[bucket] = engine
        return engine

    monkeypatch.setattr(exp, "BF16Inference", fake_inference)

    def fake_eval(model, stats, cfg, *, n_matchups, replay_dir, exec_horizon=None, inference=None, **kwargs):
        assert model in inference_models
        assert inference in eval_inferences.values()
        evaluations.append((n_matchups, cfg.exec_horizon if exec_horizon is None else exec_horizon))
        return {"net_stock_lcb": 0.0, "net_dmg_lcb": 0.0}

    monkeypatch.setattr(exp, "eval_vs_cpu", fake_eval)
    h2h_calls: list[Path] = []

    def fake_h2h(model, stats, cfg, *, run_dir, reference, uploader, inference):
        assert model in inference_models
        assert inference in eval_inferences.values()
        h2h_calls.append(reference)
        return {}

    monkeypatch.setattr(exp, "final_h2h", fake_h2h)

    class Run:
        id = "test"
        summary: dict[str, object] = {}

    monkeypatch.setattr(exp.wandb, "run", Run())
    monkeypatch.setattr(exp.wandb, "init", lambda **kwargs: None)
    monkeypatch.setattr(exp.wandb, "define_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp.wandb, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp.wandb, "finish", lambda: None)
    exp.train(cfg, {}, comment="tiny")
    assert evaluations == [(cfg.final_eval_n_matchups, 4), (cfg.final_diag_n_matchups, 6)]
    assert len({id(model) for model in inference_models}) == 1
    assert set(eval_inferences) == {
        exp._eval_inference_bucket(cfg, cfg.final_eval_n_matchups),
        cfg.final_h2h_max_parallel,
    }
    assert h2h_calls == [reference]
