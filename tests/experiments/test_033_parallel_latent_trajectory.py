"""Focused contracts for experiment 033's parallel latent trajectory policy."""

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest
import torch

from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import Context
from hal.training.features import TrainBatch

_PATH = Path(__file__).resolve().parents[2] / "experiments" / "033_parallel_latent_trajectory.py"
_SPEC = importlib.util.spec_from_file_location("test_exp033", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
exp = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = exp
_SPEC.loader.exec_module(exp)


def _cfg(**overrides):
    values = dict(
        d_model=32,
        n_layers=1,
        n_heads=4,
        L_ctx=4,
        trajectory_modes=8,
        temporal_d_model=32,
        temporal_layers=1,
        temporal_heads=4,
        temporal_ff_dim=64,
        group_head_dim=64,
        batch_size=2,
        grad_accum_steps=1,
        reservoir_capacity=4,
        warmup_steps=1,
        max_steps=2,
        compile_trunk=False,
        compile_temporal=False,
        num_workers=0,
        push_to_r2=False,
        diagnostic_histories=2,
        trajectory_samples_per_mode=2,
        joint_diagnostic_samples=4,
    )
    return exp.TrainConfig(**{**values, **overrides})


def _actions(batch: int, length: int, generator: torch.Generator) -> torch.Tensor:
    actions = torch.empty(batch, length, A_DIM)
    actions[..., :4] = torch.rand(batch, length, 4, generator=generator) * 2 - 1
    actions[..., 4:6] = torch.rand(batch, length, 2, generator=generator)
    actions[..., 6:] = torch.randint(0, 2, (batch, length, 8), generator=generator).float()
    return actions


def _context(cfg, batch: int = 2, seed: int = 0) -> Context:
    generator = torch.Generator().manual_seed(seed)
    ctx = exp.synthetic_context(cfg, batch, torch.device("cpu"))
    features = dict(ctx.features)
    native = _actions(batch, cfg.L_ctx, generator)
    for channel, values in zip(ACTION_CHANNELS, native.unbind(-1), strict=True):
        features[f"ego_{channel}"] = values
    return Context(
        features=features,
        ctx_pad=torch.tensor([0, 1][:batch]),
        slot_ids=torch.arange(batch),
        reset=torch.ones(batch, dtype=torch.bool),
    )


def _batch(cfg, batch: int = 2, seed: int = 0) -> TrainBatch:
    generator = torch.Generator().manual_seed(seed + 20)
    return TrainBatch(context=_context(cfg, batch, seed), target=_actions(batch, cfg.sample_chunk_length, generator))


def _random_targets(cfg, *prefix: int, generator=None) -> torch.Tensor:
    return torch.stack(
        [torch.randint(vocab, (*prefix, len(cfg.head_offsets)), generator=generator) for vocab in exp.GROUP_VOCABS],
        dim=-1,
    )


def test_defaults_preserve_026_setup_and_select_k8() -> None:
    cfg = exp.TrainConfig()
    assert (cfg.batch_size, cfg.L_ctx, cfg.max_steps, cfg.seed) == (1024, 128, 16_384, 0)
    assert (cfg.d_model, cfg.n_layers, cfg.n_heads) == (384, 8, 6)
    assert cfg.head_offsets == (1, 2, 3, 4, 5, 6, 9, 12, 16, 20)
    assert cfg.trajectory_modes == 8
    assert cfg.sampling_mode == "shared_k"
    assert cfg.attention_backend == "torch_sdpa_flash"


def test_output_shapes_and_forward_needs_only_hidden() -> None:
    cfg = _cfg()
    decoder = exp.ParallelActionDecoder(cfg, exp.StructuredControllerCodec(cfg.action_embed_dim))
    logits = decoder(torch.randn(2, cfg.d_model))
    assert inspect.signature(decoder.forward).parameters.keys() == {"hidden"}
    for group, name in enumerate(exp.GROUP_NAMES):
        assert logits[name].shape == (2, cfg.trajectory_modes, len(cfg.head_offsets), exp.GROUP_VOCABS[group])


def test_targets_cannot_change_forward_logits() -> None:
    cfg = _cfg()
    decoder = exp.ParallelActionDecoder(cfg, exp.StructuredControllerCodec(cfg.action_embed_dim)).eval()
    hidden = torch.randn(2, cfg.d_model)
    first_targets = _random_targets(cfg, 2, generator=torch.Generator().manual_seed(1))
    second_targets = _random_targets(cfg, 2, generator=torch.Generator().manual_seed(2))
    first = decoder.teacher_forced_logits_by_group(hidden, None, first_targets)
    second = decoder.teacher_forced_logits_by_group(hidden, None, second_targets)
    for name in exp.GROUP_NAMES:
        torch.testing.assert_close(first[name], second[name])


def test_attention_is_full_and_k_is_folded_into_the_batch(monkeypatch) -> None:
    cfg = _cfg(temporal_layers=1)
    decoder = exp.ParallelActionDecoder(cfg, exp.StructuredControllerCodec(cfg.action_embed_dim)).eval()
    calls = []
    original = torch.nn.functional.scaled_dot_product_attention

    def recorded(query, key, value, **kwargs):
        calls.append((query.shape, kwargs))
        return original(query, key, value, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", recorded)
    decoder(torch.randn(3, cfg.d_model))
    assert len(calls) == 1
    shape, kwargs = calls[0]
    assert shape[0] == 3 * cfg.trajectory_modes
    assert shape[2] == len(cfg.head_offsets) * exp.N_GROUPS
    assert kwargs == {"is_causal": False}


def test_candidates_are_isolated_batch_items() -> None:
    cfg = _cfg(trajectory_modes=4)
    decoder = exp.ParallelActionDecoder(cfg, exp.StructuredControllerCodec(cfg.action_embed_dim)).eval()
    hidden = torch.randn(2, cfg.d_model)
    before = decoder(hidden)
    with torch.no_grad():
        decoder.mode_embedding.weight[2].add_(10)
    after = decoder(hidden)
    for name in exp.GROUP_NAMES:
        torch.testing.assert_close(before[name][:, :2], after[name][:, :2])
        torch.testing.assert_close(before[name][:, 3:], after[name][:, 3:])
        assert not torch.equal(before[name][:, 2], after[name][:, 2])


def test_k1_best_of_k_is_exact_parallel_cross_entropy() -> None:
    cfg = _cfg(trajectory_modes=1)
    decoder = exp.ParallelActionDecoder(cfg, exp.StructuredControllerCodec(cfg.action_embed_dim))
    targets = _random_targets(cfg, 3, generator=torch.Generator().manual_seed(5))
    logits = decoder(torch.randn(3, cfg.d_model))
    loss, best_k, loss_per_mode, best_nll = exp.best_of_k_loss(logits, targets)
    ordinary = exp.parallel_nll(logits, targets).sum(dim=(-1, -2, -3)).mean()
    torch.testing.assert_close(loss, ordinary)
    assert best_k.eq(0).all()
    torch.testing.assert_close(loss_per_mode[:, 0], best_nll.sum(dim=(-1, -2)))


def test_best_of_k_selects_known_winner() -> None:
    batch, K, T = 2, 4, 3
    targets = torch.zeros(batch, T, exp.N_GROUPS, dtype=torch.long)
    logits = {}
    for group, name in enumerate(exp.GROUP_NAMES):
        values = torch.zeros(batch, K, T, exp.GROUP_VOCABS[group])
        values[:, 3, :, 0] = 20
        logits[name] = values
    _, best_k, loss_per_mode, _ = exp.best_of_k_loss(logits, targets)
    assert best_k.tolist() == [3, 3]
    assert (loss_per_mode[:, 3] < loss_per_mode[:, :3].min(dim=-1).values).all()


@pytest.mark.parametrize("mode", ["shared_k", "per_frame_k", "per_slot_k"])
def test_mode_sharing_tables_have_the_requested_structure(mode: str) -> None:
    uniforms = torch.tensor(
        [
            [[0.1], [0.3], [0.6], [0.9]],
            [[0.6], [0.4], [0.7], [0.8]],
        ]
    )
    modes = exp.mode_indices_from_uniforms(uniforms, 4, mode)[0]
    if mode == "shared_k":
        assert modes.unique().numel() == 1
    elif mode == "per_frame_k":
        assert (modes == modes[:, :1]).all()
        assert modes[:, 0].unique().numel() == 2
    else:
        assert modes.unique().numel() == 4


def test_shared_and_per_slot_match_marginals_but_not_joint_structure() -> None:
    batch, K, T = 8_000, 2, 2
    logits = {}
    for group, name in enumerate(exp.GROUP_NAMES):
        values = torch.full((batch, K, T, exp.GROUP_VOCABS[group]), -20.0)
        values[:, 0, :, 0] = 20
        values[:, 1, :, 1] = 20
        logits[name] = values
    gen = torch.Generator().manual_seed(13)
    tables = {
        mode: exp.sample_parallel_logits(
            logits,
            T,
            sampling_mode=mode,
            argmax=False,
            gen=gen,
        )
        for mode in ("shared_k", "per_slot_k")
    }
    for t in range(T):
        for group in range(exp.N_GROUPS):
            difference = (
                tables["shared_k"][:, t, group].float().mean() - tables["per_slot_k"][:, t, group].float().mean()
            )
            assert abs(float(difference)) < 0.03
    shared_agreement = (tables["shared_k"][:, 0, 0] == tables["shared_k"][:, 0, 1]).float().mean()
    slot_agreement = (tables["per_slot_k"][:, 0, 0] == tables["per_slot_k"][:, 0, 1]).float().mean()
    assert shared_agreement > 0.99
    assert 0.45 < slot_agreement < 0.55


def test_action_loss_and_validation_expose_required_diagnostics() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg)
    batch = _batch(cfg)
    parts = exp.action_loss(model, batch)
    assert parts.nll.shape == (7, len(cfg.head_offsets), exp.N_GROUPS)
    assert parts.loss_per_mode.shape == (7, cfg.trajectory_modes)
    assert torch.isfinite(exp.objective(parts))
    metrics = exp.val_metrics(model, [batch], cfg)
    required = {
        "mode_entropy",
        "effective_modes",
        "fraction_modes_below_1pct",
        "mode_distance_within",
        "mode_distance_between",
        "mode_distance_ratio",
        "sampler_shared_k/same_frame_group_dependence",
        "sampler_per_frame_k/adjacent_frame_dependence",
        "sampler_per_slot_k/separated_frame_dependence",
        "sampler_marginal_tv_shared_vs_per_slot",
        "best_mode/nll_o01_buttons",
        "sampler_shared_k/exact_frame_acc",
    }
    assert required <= metrics.keys()
    assert all(torch.isfinite(torch.tensor(metrics[name])) for name in required)


def test_tiny_k1_training_uses_one_model_and_shared_sampler(monkeypatch, tmp_path: Path) -> None:
    cfg = _cfg(
        trajectory_modes=1,
        max_steps=1,
        val_every=0,
        eval_every=0,
        ckpt_every=0,
        wandb_log_code=False,
        inference_mode="eager",
    )
    batch = _batch(cfg)
    monkeypatch.setattr(exp, "_make_loaders", lambda cfg, stats: ([batch], [batch]))
    monkeypatch.setattr(exp, "make_run_name", lambda *args: "tiny-033-k1")
    monkeypatch.setattr(exp, "setup_run_dir", lambda name: (tmp_path, tmp_path / "replays"))
    monkeypatch.setattr(exp, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp, "_checkpoint_sha256", lambda path: "a" * 64)
    evaluations = []

    class Inference:
        def __init__(self, model, cfg):
            self.model = model
            self.cfg = cfg

    monkeypatch.setattr(exp, "BF16Inference", Inference)

    def fake_eval(model, stats, cfg, *, n_matchups, replay_dir, exec_horizon=None, **kwargs):
        evaluations.append((cfg.sampling_mode, cfg.exec_horizon if exec_horizon is None else exec_horizon))
        return {"net_stock_lcb": 0.0, "net_dmg_lcb": 0.0}

    monkeypatch.setattr(exp, "eval_vs_cpu", fake_eval)

    class Run:
        id = "test"
        summary = {}

    logs = []
    monkeypatch.setattr(exp.wandb, "run", Run())
    monkeypatch.setattr(exp.wandb, "init", lambda **kwargs: None)
    monkeypatch.setattr(exp.wandb, "define_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp.wandb, "log", lambda values: logs.append(values))
    monkeypatch.setattr(exp.wandb, "finish", lambda: None)
    exp.train(cfg, {}, comment="tiny")
    assert evaluations == [("shared_k", 4), ("shared_k", 6)]
    assert any("train/effective_modes" in values for values in logs)
    assert any("val/mode_utilization_histogram" in values for values in logs)
    assert any("eval/shared_k/net_stock_lcb" in values for values in logs)
