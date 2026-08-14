"""Focused contracts for the unified action-token BC treatment."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from hal.training.features import Context

_PATH = Path(__file__).resolve().parents[2] / "experiments" / "032_action_token_bc.py"
_SPEC = importlib.util.spec_from_file_location("test_exp032", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
exp = importlib.util.module_from_spec(_SPEC)
sys.modules["test_exp032"] = exp
_SPEC.loader.exec_module(exp)


def _cfg(**overrides):
    values = dict(
        d_model=32,
        n_layers=1,
        n_heads=4,
        L_ctx=4,
        temporal_d_model=32,
        temporal_layers=2,
        temporal_heads=4,
        temporal_ff_dim=64,
        group_head_dim=64,
        batch_size=2,
        reservoir_capacity=4,
        warmup_steps=1,
        max_steps=2,
        compile_trunk=False,
        compile_temporal=False,
        num_workers=0,
        push_to_r2=False,
        inference_mode="eager",
    )
    return exp.TrainConfig(**{**values, **overrides})


def _context(cfg, batch=2):
    return exp.synthetic_context(cfg, batch, torch.device("cpu"))


def _indices(batch, context, horizon, generator):
    return torch.stack(
        [torch.randint(vocab, (batch, context, horizon), generator=generator) for vocab in exp.GROUP_VOCABS],
        dim=-1,
    )


def test_default_action_token_decoder_matches_reference_parameter_budget() -> None:
    model = exp.GPT(exp.TrainConfig())
    decoder = sum(parameter.numel() for parameter in model.temporal.parameters())
    total = sum(parameter.numel() for parameter in model.parameters())
    assert 676_281 <= decoder <= 747_469
    assert 14_300_387 <= total <= 15_805_691
    assert model.cfg.group_order == ("c_stick", "main_stick", "triggers", "buttons")
    assert exp.model_tag(model.cfg).startswith("atbc032-")
    assert model.value_head.weight.count_nonzero() == 0
    assert model.value_head.bias.count_nonzero() == 0


def test_v4_checkpoint_config_reinstantiates_treatment_model() -> None:
    cfg = exp.TrainConfig(policy_version=13)
    restored = exp.config_from_state(vars(cfg))
    assert (restored.decoder_arch_version, restored.policy_version) == (4, 13)
    assert isinstance(exp.base.GPT(restored), exp.GPT)


def test_every_offset_group_pair_is_one_token_in_frozen_order() -> None:
    model = exp.GPT(_cfg())
    decoder = model.temporal
    assert decoder.token_offsets.tolist()[:8] == [1, 1, 1, 1, 2, 2, 2, 2]
    expected_roles = [exp.GROUP_INDEX[name] for name in exp.GROUP_ORDER] * len(model.head_offsets)
    assert decoder.token_roles.tolist() == expected_roles


def test_teacher_forcing_is_causal_within_and_across_frames() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg).eval()
    hidden = torch.randn(1, cfg.L_ctx, cfg.d_model)
    observed = torch.zeros(1, cfg.L_ctx, exp.N_GROUPS, dtype=torch.long)
    targets = torch.zeros(1, cfg.L_ctx, len(cfg.head_offsets), exp.N_GROUPS, dtype=torch.long)
    changed = targets.clone()
    changed[..., 0, exp.C_G] = 1
    before = model.temporal.teacher_forced_logits_by_group(hidden, observed, targets)
    after = model.temporal.teacher_forced_logits_by_group(hidden, observed, changed)
    torch.testing.assert_close(before["c_stick"][..., 0, :], after["c_stick"][..., 0, :])
    assert not torch.equal(before["main_stick"][..., 0, :], after["main_stick"][..., 0, :])
    assert not torch.equal(before["c_stick"][..., 1, :], after["c_stick"][..., 1, :])


def test_parallel_teacher_forcing_matches_plan_local_cached_steps() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg).eval()
    generator = torch.Generator().manual_seed(17)
    hidden = torch.randn(2, cfg.L_ctx, cfg.d_model, generator=generator)
    observed = torch.zeros(2, cfg.L_ctx, exp.N_GROUPS, dtype=torch.long)
    targets = _indices(2, cfg.L_ctx, len(cfg.head_offsets), generator)
    parallel = model.temporal.teacher_forced_logits_by_group(hidden, observed, targets)
    cached = model.temporal.forced_stepwise_logits(hidden, observed[:, -1], targets[:, -1])
    for depth, frame in enumerate(cached):
        for name in exp.GROUP_NAMES:
            torch.testing.assert_close(parallel[name][:, -1, depth], frame[name], atol=2e-5, rtol=2e-5)


def test_sample_plan_and_score_plan_recompute_exact_factor_statistics() -> None:
    cfg = _cfg(policy_version=9)
    model = exp.GPT(cfg).eval()
    context = _context(cfg)
    uniforms = torch.rand(4, exp.N_GROUPS, 2, generator=torch.Generator().manual_seed(5))
    sampled = model.sample_plan(context, 4, uniforms=uniforms)
    scored = model.score_plan(context, sampled.tokens)
    torch.testing.assert_close(scored.per_factor_logp, sampled.per_factor_logp, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(scored.entropy, sampled.entropy, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(scored.value, sampled.value)
    assert sampled.actions.shape == (2, 4, exp.A_DIM)
    assert sampled.executed_prefix_mask.all()
    assert sampled.policy_version.tolist() == [9, 9]


def test_bf16_cached_score_has_unit_ppo_ratio_and_gradients() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg).eval()
    context = _context(cfg)
    uniforms = torch.rand(4, exp.N_GROUPS, 2, generator=torch.Generator().manual_seed(29))
    executed = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        sampled = model.sample_plan(context, 4, uniforms=uniforms)
        scored = model.score_plan(context, sampled.tokens, executed)
    torch.testing.assert_close(scored.per_factor_logp, sampled.per_factor_logp, atol=0, rtol=0)
    mask = scored.executed_prefix_mask
    old_joint = sampled.per_factor_logp.masked_fill(~mask, 0).sum(dim=(1, 2))
    new_joint = scored.per_factor_logp.masked_fill(~mask, 0).sum(dim=(1, 2))
    ratio = (new_joint - old_joint).exp()
    torch.testing.assert_close(ratio, torch.ones_like(ratio), atol=0, rtol=0)
    (-new_joint.mean()).backward()
    assert model.temporal.outputs["buttons"].down.weight.grad is not None


def test_interrupted_plan_mask_expands_over_factors() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg).eval()
    context = _context(cfg)
    sampled = model.sample_plan(context, 4, argmax=True)
    frame_mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
    scored = model.score_plan(context, sampled.tokens, frame_mask)
    assert scored.executed_prefix_mask.shape == sampled.tokens.shape
    assert torch.equal(scored.executed_prefix_mask[..., 0], frame_mask)
    invalid = frame_mask.clone()
    invalid[0] = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
    with pytest.raises(ValueError, match="contiguous"):
        model.score_plan(context, sampled.tokens, invalid)


def test_train_rewrites_inherited_run_tags(monkeypatch) -> None:
    seen = {}

    def fake_init(*args, **kwargs):
        seen.update(kwargs)

    def fake_train(cfg, stats, **kwargs):
        exp.base.wandb.init(project="hal", tags=["gpt", "026"])

    monkeypatch.setattr(exp.base.wandb, "init", fake_init)
    monkeypatch.setattr(exp, "_base_train", fake_train)
    exp.train(_cfg(), {})
    assert seen["tags"] == ["gpt", "action-token", "autoregressive", "bc", "032"]


def test_slot_keyed_sampling_survives_mixed_slot_resets() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg).eval()
    engine = exp.BF16Inference(model, cfg, compiled=False)
    streams = exp.SlotGroupRandom(123)
    first = _context(cfg)
    engine.decode(first, 4, streams=streams)
    generations = dict(streams.generations)
    second = Context(
        features=first.features,
        ctx_pad=first.ctx_pad,
        slot_ids=first.slot_ids,
        reset=torch.tensor([False, True]),
    )
    result = engine.decode(second, 4, streams=streams)
    assert result.shape == (2, 4, exp.A_DIM)
    assert streams.generations[0] == generations[0]
    assert streams.generations[1] == generations[1] + 1
