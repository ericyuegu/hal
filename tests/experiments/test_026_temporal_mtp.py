"""Contracts for the fast sparse-offset temporal controller experiment."""

import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from hal.eval.cross_stage import conservative_net_lcb
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import SPATIAL_COLUMNS_LEAN
from hal.training.features import V6_PLAYER_COLUMNS
from hal.training.features import Context
from hal.training.features import TrainBatch

_PATH = Path(__file__).resolve().parents[2] / "experiments" / "026_temporal_mtp.py"
_SPEC = importlib.util.spec_from_file_location("test_exp026", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
exp = importlib.util.module_from_spec(_SPEC)
sys.modules["test_exp026"] = exp
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
        grad_accum_steps=1,
        reservoir_capacity=4,
        warmup_steps=1,
        max_steps=2,
        compile_trunk=False,
        compile_temporal=False,
        num_workers=0,
        push_to_r2=False,
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


def test_defaults_freeze_requested_treatment_knobs() -> None:
    cfg = exp.TrainConfig()
    assert (cfg.batch_size, cfg.L_ctx, cfg.max_steps, cfg.seed) == (1024, 128, 16_384, 0)
    assert (cfg.d_model, cfg.n_layers, cfg.n_heads) == (384, 8, 6)
    assert cfg.head_offsets == (1, 2, 3, 4, 5, 6, 9, 12, 16, 20)
    assert (cfg.temporal_d_model, cfg.temporal_layers, cfg.temporal_heads, cfg.temporal_ff_dim) == (128, 2, 4, 256)
    assert cfg.group_order == ("c_stick", "main_stick", "triggers", "buttons")
    assert cfg.inference_buckets == (1, 2, 4, 8, 16, 32)


def test_codec_canonicalizes_clicks_and_builds_semantic_tokens() -> None:
    model = exp.GPT(_cfg())
    actions = torch.zeros(1, 2, A_DIM)
    actions[..., exp.BUTTON_L_CH] = 1
    indices = model.codec.quantize(actions)
    restored = model.codec.dequantize(indices)
    assert restored[..., exp.TRIGGER_L_CH].eq(1).all()
    embedded = model.codec.embed_frame(indices)
    assert embedded.shape == (1, 2, 4 * model.cfg.action_embed_dim)
    for group in model.codec.embed_groups(indices).values():
        torch.testing.assert_close(group.square().mean(-1), torch.ones_like(group[..., 0]), atol=2e-4, rtol=2e-4)


def test_button_mask_excludes_clicks_without_full_corresponding_trigger() -> None:
    codec = exp.StructuredControllerCodec(16)
    button_bits = exp.scoring.combo_to_buttons(torch.arange(exp.GROUP_VOCABS[exp.BUTTONS_G]))
    left_click = button_bits[:, exp.BUTTON_L_CH - 6].bool()
    no_trigger = torch.zeros(1, dtype=torch.long)
    assert codec.button_mask(no_trigger)[0, left_click].all()
    full_pair = torch.tensor([exp.GROUP_VOCABS[exp.TRIG_G] - 1])
    assert not codec.button_mask(full_pair)[0, left_click].any()


def test_selected_offset_alignment_and_padding_boundaries() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg)
    length = cfg.L_ctx + cfg.sample_chunk_length
    indices = torch.stack([torch.arange(length).remainder(vocab) for vocab in exp.GROUP_VOCABS], dim=-1)[None]
    native = model.codec.dequantize(indices)
    context = _context(cfg, batch=1)
    features = dict(context.features)
    for channel, values in zip(ACTION_CHANNELS, native[:, : cfg.L_ctx].unbind(-1), strict=True):
        features[f"ego_{channel}"] = values
    batch = TrainBatch(
        context=Context(features=features, ctx_pad=torch.tensor([2])),
        target=native[:, cfg.L_ctx :],
    )
    targets, valid = exp.chunk_targets(model, batch)
    assert valid.tolist() == [[False, False, True, True]]
    for prefix in range(cfg.L_ctx):
        expected = torch.stack([indices[0, prefix + offset] for offset in cfg.head_offsets])
        torch.testing.assert_close(targets[0, prefix], expected)


def test_action_loss_quantizes_the_combined_sequence_once(monkeypatch) -> None:
    model = exp.GPT(_cfg())
    calls = 0
    original = model.codec.quantize

    def counted(actions):
        nonlocal calls
        calls += 1
        return original(actions)

    monkeypatch.setattr(model.codec, "quantize", counted)
    parts = exp.action_loss(model, _batch(model.cfg))
    assert calls == 1
    assert parts.nll.shape == (7, len(model.head_offsets), exp.N_GROUPS)


def test_objective_uses_exact_primary_plus_weighted_auxiliary_arithmetic() -> None:
    nll = torch.zeros(1, 10, 4)
    nll[:, :4] = 2
    nll[:, 4:] = 3
    parts = exp.ActionLoss(nll=nll, targets=torch.zeros_like(nll, dtype=torch.long))
    assert exp.objective(parts, 0.5).item() == 8 + 0.5 * 12


def test_temporal_target_change_only_reaches_same_frame_later_groups_and_later_heads() -> None:
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


def test_parallel_teacher_forcing_matches_stepwise_logits() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg).eval()
    generator = torch.Generator().manual_seed(7)
    hidden = torch.randn(2, cfg.L_ctx, cfg.d_model, generator=generator)
    observed = torch.stack(
        [torch.randint(vocab, (2, cfg.L_ctx), generator=generator) for vocab in exp.GROUP_VOCABS], dim=-1
    )
    targets = torch.stack(
        [
            torch.randint(vocab, (2, cfg.L_ctx, len(cfg.head_offsets)), generator=generator)
            for vocab in exp.GROUP_VOCABS
        ],
        dim=-1,
    )
    parallel = model.temporal.teacher_forced_logits_by_group(hidden, observed, targets)
    stepwise = model.temporal.forced_stepwise_logits(hidden, observed[:, -1], targets[:, -1])
    for depth, logits in enumerate(stepwise):
        for name in exp.GROUP_NAMES:
            torch.testing.assert_close(parallel[name][:, -1, depth], logits[name], atol=2e-5, rtol=2e-5)


def test_temporal_sdpa_chunks_flattened_batch_without_changing_results(monkeypatch) -> None:
    cfg = _cfg(temporal_layers=1)
    block = exp.TemporalBlock(cfg).eval()
    inputs = torch.randn(7, len(cfg.head_offsets), cfg.temporal_d_model)
    expected = block._forward_chunk(inputs)
    calls: list[int] = []
    original = torch.nn.functional.scaled_dot_product_attention

    def recorded(query, key, value, **kwargs):
        calls.append(query.shape[0])
        return original(query, key, value, **kwargs)

    monkeypatch.setattr(exp, "TEMPORAL_SDPA_BATCH_LIMIT", 3)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", recorded)
    actual = block(inputs)
    assert calls == [3, 3, 1]
    torch.testing.assert_close(actual, expected)


def test_decoder_has_actual_offset_embeddings_and_no_cross_attention() -> None:
    model = exp.GPT(_cfg())
    assert model.temporal.offset_embedding.num_embeddings == model.cfg.sample_chunk_length + 1
    assert not hasattr(model.temporal, "trunk_cross_attention")
    names = [name for name, _ in model.temporal.named_parameters()]
    assert not any("cross_attention" in name for name in names)


def test_rollout_rejects_sparse_tail_work_and_returns_only_executed_frames() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg).eval()
    ctx = _context(cfg)
    hidden = model(ctx.features, ctx.ctx_pad)
    observed = model.codec.quantize(torch.stack([ctx.features[f"ego_{x}"] for x in ACTION_CHANNELS], -1))[:, -1]
    assert model.temporal.sample_indices(hidden, observed, cfg.head_offsets[:4], argmax=True).shape == (2, 4, 4)
    with pytest.raises(ValueError, match="dense"):
        model.temporal.sample_indices(hidden, observed, cfg.head_offsets, argmax=True)


def test_bucket_padding_and_slot_keyed_rng_do_not_change_real_rows() -> None:
    cfg = _cfg(inference_mode="eager")
    model = exp.GPT(cfg).eval()
    engine = exp.BF16Inference(model, cfg, compiled=False)
    ctx = _context(cfg, batch=1, seed=4)
    first_rng = exp.SlotGroupRandom(9)
    first = engine.decode(ctx, 4, streams=first_rng)
    # Force a larger bucket while keeping the same real row and random stream.
    larger_cfg = _cfg(inference_mode="eager", inference_buckets=(2, 4, 8, 16, 32))
    second_engine = exp.BF16Inference(model, larger_cfg, compiled=False)
    second = second_engine.decode(ctx, 4, streams=exp.SlotGroupRandom(9))
    torch.testing.assert_close(first, second)


@pytest.mark.parametrize("bundle", ["base", "v6_lean"])
def test_both_observation_bundles_forward_and_optimize(bundle: str) -> None:
    cfg = _cfg(observation_bundle=bundle)
    model = exp.GPT(cfg)
    batch = _batch(cfg)
    loss = exp.objective(exp.action_loss(model, batch))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.ctx_proj.weight.grad is not None


def test_v6_synthetic_context_routes_every_declared_extra() -> None:
    cfg = _cfg(observation_bundle="v6_lean")
    ctx = exp.synthetic_context(cfg, 1, torch.device("cpu"))
    for prefix in exp._PLAYER_PREFIXES:
        for name in (*V6_PLAYER_COLUMNS.floats, *V6_PLAYER_COLUMNS.cats):
            assert f"{prefix}_{name}" in ctx.features
    assert set(SPATIAL_COLUMNS_LEAN) <= ctx.features.keys()


def test_protocol_schedule_is_frozen_and_diverse() -> None:
    assert exp.assert_protocol_diversity(32)[:3] == (26, 8, 8)
    assert exp.assert_protocol_diversity(96)[:3] == (58, 13, 14)


def test_conservative_lcb_matches_backfill_formula() -> None:
    got = conservative_net_lcb(10.0, 4.0, (8.04, 11.96), (3.02, 4.98))
    expected = 6.0 - 1.645 * ((1.96 + 0.98) / 1.96)
    assert got == pytest.approx(expected)


def test_checkpoint_config_round_trip_and_optimizer_owns_every_parameter() -> None:
    cfg = _cfg()
    restored = exp.config_from_state(asdict(cfg))
    assert restored == cfg
    model = exp.GPT(cfg)
    optimizer = exp.make_optimizer(model, cfg)
    owned = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(owned) == sum(1 for _ in model.parameters())
    assert len({id(parameter) for parameter in owned}) == len(owned)


def test_tiny_training_run_reaches_final_validation_and_both_evaluations(monkeypatch, tmp_path: Path) -> None:
    cfg = _cfg(
        max_steps=1,
        val_every=0,
        eval_every=0,
        ckpt_every=0,
        wandb_log_code=False,
        inference_mode="eager",
    )
    batch = _batch(cfg)
    monkeypatch.setattr(exp, "_make_loaders", lambda cfg, stats: ([batch], [batch]))
    monkeypatch.setattr(exp, "make_run_name", lambda *args: "tiny-026")
    monkeypatch.setattr(exp, "setup_run_dir", lambda name: (tmp_path, tmp_path / "replays"))
    monkeypatch.setattr(exp, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp, "_checkpoint_sha256", lambda path: "a" * 64)
    evaluations: list[tuple[int, int]] = []

    def fake_eval(model, stats, cfg, *, n_matchups, replay_dir, exec_horizon=None, checkpoint_sha256):
        evaluations.append((n_matchups, cfg.exec_horizon if exec_horizon is None else exec_horizon))
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
    assert evaluations == [(cfg.final_eval_n_matchups, 4), (cfg.final_diag_n_matchups, 6)]
