"""Contracts for four-frame action-chunk IQL experiment 027."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import Context
from hal.training.features import TrainBatch
from hal.training.features import stack_actions


def _load(name: str, filename: str):
    path = Path(__file__).resolve().parents[2] / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exp026 = _load("test_exp026_for_027", "026_temporal_mtp.py")
exp = _load("test_exp027", "027_temporal_mtp.py")


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
        batch_size=2,
        grad_accum_steps=1,
        reservoir_capacity=4,
        warmup_steps=1,
        max_steps=2,
        compile_trunk=False,
        compile_temporal=False,
        num_workers=0,
        push_to_r2=False,
        iql_q_bins=11,
    )
    return exp.TrainConfig(**{**values, **overrides})


def _actions(batch: int, length: int, generator: torch.Generator) -> torch.Tensor:
    actions = torch.empty(batch, length, A_DIM)
    actions[..., :4] = torch.rand(batch, length, 4, generator=generator) * 2 - 1
    actions[..., 4:6] = torch.rand(batch, length, 2, generator=generator)
    actions[..., 6:] = torch.randint(0, 2, (batch, length, 8), generator=generator).float()
    return actions


def _iql_batch(cfg, seed: int = 0) -> exp.IQLBatch:
    generator = torch.Generator().manual_seed(seed)
    context = exp.synthetic_context(cfg, 2, torch.device("cpu"))
    context = Context(features=context.features, ctx_pad=torch.tensor([0, 1]))
    target = _actions(2, cfg.sample_chunk_length, generator)
    extended = {
        name: torch.cat((value, value[:, -1:].expand(-1, exp.EXECUTED_CHUNK)), dim=1)
        for name, value in context.features.items()
    }
    for channel, values in zip(ACTION_CHANNELS, target[:, : exp.EXECUTED_CHUNK].unbind(-1), strict=True):
        extended[f"ego_{channel}"][:, -exp.EXECUTED_CHUNK :] = values
    return exp.IQLBatch(
        batch=TrainBatch(context=context, target=target),
        extended_context=Context(features=extended, ctx_pad=context.ctx_pad),
        rewards=torch.randn(2, cfg.L_ctx, exp.EXECUTED_CHUNK, generator=generator),
        returns=torch.randn(2, cfg.L_ctx, generator=generator),
    )


def test_defaults_freeze_selected_iql_treatment() -> None:
    cfg = exp.TrainConfig()
    assert cfg.head_offsets == (1, 2, 3, 4, 5, 6, 9, 12, 16, 20)
    assert cfg.exec_horizon == 4
    assert (cfg.iql_tau, cfg.iql_gamma, cfg.iql_beta, cfg.iql_weight_max) == (0.9, 0.99827, 0.8, 5.0)
    assert (cfg.iql_q_min, cfg.iql_q_max, cfg.iql_q_bins) == (-4.0, 4.0, 201)
    assert cfg.iql_q_target_encoding == "two_hot"
    assert (cfg.iql_damage_shaping, cfg.iql_win_reward) == (0.01, 0.5)


def test_027_preserves_every_seeded_026_policy_parameter() -> None:
    common = dict(
        d_model=32,
        n_layers=1,
        n_heads=4,
        L_ctx=4,
        temporal_d_model=32,
        temporal_layers=1,
        temporal_heads=4,
        temporal_ff_dim=64,
        group_head_dim=64,
        batch_size=2,
        reservoir_capacity=4,
        compile_trunk=False,
        compile_temporal=False,
        num_workers=0,
        push_to_r2=False,
    )
    torch.manual_seed(13)
    baseline = exp026.GPT(exp026.TrainConfig(**common))
    torch.manual_seed(13)
    treatment = exp.GPT(exp.TrainConfig(**common))
    treatment_state = treatment.state_dict()
    for name, expected in baseline.state_dict().items():
        torch.testing.assert_close(treatment_state[name], expected)


def test_shaped_reward_and_return_have_exact_one_frame_recurrence() -> None:
    sample = {
        "p1_stock": np.array([4, 4, 3, 3, 0], dtype=np.int32),
        "p2_stock": np.array([4, 3, 3, 0, 0], dtype=np.int32),
        "p1_percent": np.array([0, 10, 10, 30, 0], dtype=np.float32),
        "p2_percent": np.array([0, 0, 25, 25, 0], dtype=np.float32),
    }
    gamma = 0.8
    labeled = exp.label_iql_replay(sample, gamma=gamma, damage_shaping=0.01, win_reward=0.5)
    for port in ("p1", "p2"):
        reward = labeled[f"{port}_iql_reward"]
        returns = labeled[f"{port}_iql_return"]
        np.testing.assert_allclose(returns[:-1], reward[:-1] + gamma * returns[1:], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(labeled["p1_iql_reward"], -labeled["p2_iql_reward"])


def test_collated_chunk_labels_are_exactly_t_plus_one_through_t_plus_four() -> None:
    length = 12
    window = {
        exp.EGO_REWARD_COLUMN: np.arange(length, dtype=np.float32),
        exp.EGO_RETURN_COLUMN: np.arange(length, dtype=np.float32) + 100,
    }
    rewards, returns = exp.aligned_iql_labels([window], L_ctx=4)
    torch.testing.assert_close(
        rewards[0],
        torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]], dtype=torch.float32),
    )
    torch.testing.assert_close(returns[0], torch.tensor([101, 102, 103, 104], dtype=torch.float32))


def test_four_frame_td_target_uses_gamma_four_bootstrap_and_detaches_it() -> None:
    rewards = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    next_value = torch.tensor([[5.0]], requires_grad=True)
    target = exp.chunk_td_target(rewards, next_value, gamma=0.5)
    assert target.item() == pytest.approx(1 + 0.5 * 2 + 0.25 * 3 + 0.125 * 4 + 0.5**4 * 5)
    assert not target.requires_grad


def test_categorical_encoders_are_normalized_and_two_hot_decodes_exactly() -> None:
    model = exp.GPT(_cfg(iql_q_min=-2, iql_q_max=2, iql_q_bins=5))
    target = torch.tensor([-3.0, -1.5, 0.3, 2.5])
    encoded = exp.encode_q_target(target, model, model.cfg)
    torch.testing.assert_close(encoded.sum(-1), torch.ones(4))
    decoded = encoded @ model.q_bin_values
    torch.testing.assert_close(decoded, target.clamp(-2, 2), atol=1e-6, rtol=1e-6)

    hl_cfg = _cfg(iql_q_min=-2, iql_q_max=2, iql_q_bins=5, iql_q_target_encoding="hl_gauss")
    hl_model = exp.GPT(hl_cfg)
    hl = exp.encode_q_target(target, hl_model, hl_cfg)
    torch.testing.assert_close(hl.sum(-1), torch.ones(4), atol=1e-6, rtol=1e-6)
    assert (hl >= 0).all()


def test_decoded_q_stays_float32_under_autocast() -> None:
    model = exp.GPT(_cfg())
    logits = torch.randn(7, model.cfg.iql_q_bins)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        decoded = model.decode_q(logits)
    assert decoded.dtype == torch.float32


def test_extended_causal_pass_leaves_policy_history_states_unchanged() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg).eval()
    batch = _iql_batch(cfg)
    history = model.codec.quantize(stack_actions(batch.batch.context.features))
    extended = model.codec.quantize(stack_actions(batch.extended_context.features))
    short_hidden = model(batch.batch.context.features, batch.batch.context.ctx_pad, history)
    long_hidden = model(batch.extended_context.features, batch.extended_context.ctx_pad, extended)
    torch.testing.assert_close(long_hidden[:, : cfg.L_ctx], short_hidden, atol=2e-5, rtol=2e-5)


def test_q_consumes_teacher_forced_chunk_and_policy_weighting_touches_only_dense_four() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg)
    batch = _iql_batch(cfg, seed=4)
    history, targets, valid = exp.prepared_targets(model, batch.batch)
    extended_actions = model.codec.quantize(stack_actions(batch.extended_context.features))
    extended_hidden = model(batch.extended_context.features, batch.extended_context.ctx_pad, extended_actions)
    hidden = extended_hidden[:, : cfg.L_ctx]
    next_hidden = extended_hidden[:, exp.EXECUTED_CHUNK : cfg.L_ctx + exp.EXECUTED_CHUNK]
    changed = targets.clone()
    changed[..., 0, exp.C_G] = (changed[..., 0, exp.C_G] + 1) % exp.GROUP_VOCABS[exp.C_G]
    assert not torch.equal(
        model.chunk_q_logits(hidden, targets[..., :4, :]),
        model.chunk_q_logits(hidden, changed[..., :4, :]),
    )

    dense_nll = model.temporal.teacher_forced_nll(hidden, history, targets)
    parts = exp.iql_parts(model, hidden, next_hidden, targets, batch.rewards, valid, dense_nll, cfg)
    joint = dense_nll[valid].sum(-1)
    expected = (parts.weight[:, None] * joint[:, :4]).mean() + cfg.aux_loss_weight * joint[:, 4:].mean()
    torch.testing.assert_close(parts.policy_loss, expected)
    assert not parts.weight.requires_grad
    assert not parts.advantage.requires_grad


def test_actor_weight_is_raw_clipped_exponential_without_mean_normalization() -> None:
    advantage = torch.tensor([-0.8, 0.0, 0.8, 8.0])
    weight, clipped = exp.actor_weights(advantage, beta=0.8, weight_max=5.0)
    torch.testing.assert_close(weight[:3], torch.exp(torch.tensor([-1.0, 0.0, 1.0])))
    assert weight[-1].item() == pytest.approx(5.0)
    assert clipped.tolist() == [False, False, False, True]
    assert weight.mean().item() != pytest.approx(1.0)


def test_inference_never_calls_either_critic(monkeypatch) -> None:
    cfg = _cfg(inference_mode="eager")
    model = exp.GPT(cfg).eval()

    def forbidden(*args, **kwargs):
        raise AssertionError("critic reached inference")

    monkeypatch.setattr(model.q_head, "forward", forbidden)
    monkeypatch.setattr(model.value_head, "forward", forbidden)
    context = exp.synthetic_context(cfg, 2, torch.device("cpu"))
    engine = exp.BF16Inference(model, cfg, compiled=False)
    assert engine.decode(context, 4, argmax=True).shape == (2, 4, A_DIM)
    with pytest.raises(ValueError, match="exactly four"):
        engine.decode(context, 6, argmax=True)


def test_decode_canonicalizes_feature_keys_to_one_program_shape() -> None:
    """A context that misses never-fired mask sidecars, in any key order, must reach
    the model as the SAME feature dict the synthetic prewarm context reaches it with.
    Dynamo guards on dict membership and key order, so a divergence makes the first
    real decode of an evaluation recompile — and run kernels the prewarm never proved."""
    cfg = _cfg(inference_mode="eager")
    synthetic = exp.synthetic_context(cfg, 2, torch.device("cpu"))
    reference = exp.canonical_context(synthetic, cfg.observation_bundle)
    assert list(reference.features) == sorted(synthetic.features)

    dropped = [name for name in synthetic.features if name.endswith("_mask") and not name.startswith("ego_nana")][:6]
    scrambled_names = [name for name in reversed(list(synthetic.features)) if name not in dropped]
    scrambled = Context(
        features={name: synthetic.features[name] for name in scrambled_names},
        ctx_pad=synthetic.ctx_pad,
        slot_ids=synthetic.slot_ids,
        reset=synthetic.reset,
    )
    canonical = exp.canonical_context(scrambled, cfg.observation_bundle)
    assert list(canonical.features) == list(reference.features)
    for name in dropped:
        assert torch.equal(canonical.features[name], torch.zeros_like(reference.features[name]))

    model = exp.GPT(cfg).eval()
    engine = exp.BF16Inference(model, cfg, compiled=False)
    generator_a = torch.Generator().manual_seed(3)
    generator_b = torch.Generator().manual_seed(3)
    torch.testing.assert_close(
        engine.decode(scrambled, 4, gen=generator_a), engine.decode(synthetic, 4, gen=generator_b)
    )


def test_fixed_inference_bucket_pads_live_and_degraded_waves_to_one_shape() -> None:
    cfg = _cfg(inference_mode="eager")
    model = exp.GPT(cfg).eval()
    engine = exp.BF16Inference(model, cfg, bucket=8, compiled=False)
    engine.compiled = True  # exercise fixed-bucket routing with eager callables
    seen: list[tuple[int, tuple[int, ...]]] = []
    trunk = engine._trunk

    def record(features, pad, actions):
        seen.append((pad.shape[0], tuple(value.shape[0] for value in features.values())))
        return trunk(features, pad, actions)

    engine._trunk = record
    context = exp.synthetic_context(cfg, 2, torch.device("cpu"))
    assert engine.decode(context, 4, argmax=True).shape == (2, 4, A_DIM)
    assert seen == [(8, (8,) * len(context.features))]

    too_large = exp.synthetic_context(cfg, 9, torch.device("cpu"))
    with pytest.raises(ValueError, match="exceeds fixed compiled bucket 8"):
        engine.decode(too_large, 4, argmax=True)


def test_eval_prewarms_synchronously_before_starting_dolphin(monkeypatch, tmp_path: Path) -> None:
    cfg = _cfg(inference_mode="eager")
    model = exp.GPT(cfg)
    bucket = exp._eval_inference_bucket(cfg, 1)
    engine = exp.BF16Inference(model, cfg, bucket=bucket, compiled=False)
    events: list[str] = []

    def prewarm() -> float:
        events.append("prewarm")
        return 1.25

    def sweep(*args, **kwargs):
        events.append("sweep")
        assert not model.training
        return [], []

    monkeypatch.setattr(engine, "prewarm", prewarm)
    monkeypatch.setattr(exp, "sweep_vs_cpu_prior_with_rows", sweep)
    monkeypatch.setattr(exp, "vs_cpu_metrics", lambda *args, **kwargs: {})
    metrics = exp.eval_vs_cpu(
        model,
        {},
        cfg,
        n_matchups=1,
        replay_dir=tmp_path,
        inference=engine,
    )
    assert events == ["prewarm", "sweep"]
    assert metrics["inference_compile_seconds"] == pytest.approx(1.25)
    assert model.training


def test_checkpoint_config_ignores_removed_inference_bucket_field() -> None:
    cfg = _cfg()
    values = dict(vars(cfg))
    values["inference_buckets"] = (32, 64, 128)
    assert exp.config_from_state(values) == cfg


def test_optimizer_owns_new_critic_parameters_once() -> None:
    model = exp.GPT(_cfg())
    optimizer = exp.make_optimizer(model, model.cfg)
    owned = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(owned) == sum(1 for _ in model.parameters())
    assert len({id(parameter) for parameter in owned}) == len(owned)


def test_validation_emits_required_critic_diagnostics_and_histograms() -> None:
    cfg = _cfg()
    values = exp.val_metrics(exp.GPT(cfg), [_iql_batch(cfg, seed=9)], cfg)
    required = {
        "q_loss",
        "value_loss",
        "loss",
        "objective",
        "q_mean",
        "q_std",
        "q_p05",
        "q_p50",
        "q_p95",
        "value_mean",
        "advantage_mean",
        "actor_weight_mean",
        "actor_weight_median",
        "actor_weight_p99",
        "actor_weight_max",
        "awr_weight_max_frac",
        "td_residual_mean",
        "td_residual_mae",
        "td_residual_rmse",
        "td_residual_p95",
        "corr_q_g",
        "corr_value_g",
        "q_g_mae",
        "value_g_mae",
        "expectile_balance",
        "q_histogram",
        "value_histogram",
        "advantage_histogram",
        "awr_weights",
    }
    assert required <= values.keys()


def test_tiny_training_run_reaches_policy_only_final_evaluation(monkeypatch, tmp_path: Path) -> None:
    cfg = _cfg(
        max_steps=1,
        val_every=0,
        eval_every=0,
        ckpt_every=0,
        wandb_log_code=False,
        inference_mode="eager",
    )
    batch = _iql_batch(cfg, seed=12)
    monkeypatch.setattr(exp, "_make_loaders", lambda cfg, stats: ([batch], [batch]))
    monkeypatch.setattr(exp, "make_run_name", lambda *args: "tiny-027")
    monkeypatch.setattr(exp, "setup_run_dir", lambda name: (tmp_path, tmp_path / "replays"))
    monkeypatch.setattr(exp, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp, "_checkpoint_sha256", lambda path: "a" * 64)
    evaluations: list[tuple[int, object, object]] = []

    def fake_eval(model, stats, cfg, *, n_matchups, replay_dir, checkpoint_sha256, inference=None, **kwargs):
        assert isinstance(inference, exp.BF16Inference)
        assert inference.model is model
        assert inference.bucket == exp._eval_inference_bucket(cfg, n_matchups)
        evaluations.append((n_matchups, model, inference))
        return {"net_stock_lcb": 0.0, "net_dmg_lcb": 0.0}

    monkeypatch.setattr(exp, "eval_vs_cpu", fake_eval)

    class Run:
        id = "test"
        summary: dict[str, object] = {}

    monkeypatch.setattr(exp.wandb, "run", Run())
    monkeypatch.setattr(exp.wandb, "init", lambda **kwargs: None)
    monkeypatch.setattr(exp.wandb, "define_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp.wandb, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp.wandb, "finish", lambda: None)
    exp.train(cfg, {}, comment="tiny")
    assert [n for n, _, _ in evaluations] == [cfg.final_eval_n_matchups]
