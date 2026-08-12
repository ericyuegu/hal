"""Contracts for the one-hot controller encoding ablation (028)."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from hal.training import scoring
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import Context
from hal.training.features import TrainBatch

_PATH = Path(__file__).resolve().parents[2] / "experiments" / "028_onehot_controller.py"
_SPEC = importlib.util.spec_from_file_location("test_exp028", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
exp = importlib.util.module_from_spec(_SPEC)
sys.modules["test_exp028"] = exp
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


def test_defaults_match_the_control_run_except_the_ablation_axis() -> None:
    cfg = exp.TrainConfig()
    assert cfg.btn_min_count == 0  # the bare run is the one-axis arm; the vocab cut is opt-in
    assert (cfg.L_ctx, cfg.max_steps, cfg.seed) == (128, 16_384, 0)
    assert (cfg.d_model, cfg.n_layers, cfg.n_heads) == (384, 8, 6)
    assert cfg.head_offsets == (1, 2, 3, 4, 5, 6, 9, 12, 16, 20)
    assert cfg.group_order == ("c_stick", "main_stick", "triggers", "buttons")


def test_codec_owns_zero_parameters() -> None:
    for min_count in (0, 100):
        codec = exp.StructuredControllerCodec(min_count)
        assert sum(parameter.numel() for parameter in codec.parameters()) == 0


def test_group_embedding_is_a_plain_one_hot() -> None:
    codec = exp.StructuredControllerCodec(100)
    indices = torch.stack([torch.arange(4).remainder(width) for width in codec.group_widths], dim=-1)
    frame = codec.embed_frame(indices)
    assert frame.shape == (4, codec.frame_width)
    assert ((frame == 0.0) | (frame == 1.0)).all()
    start = 0
    for name in exp.GROUP_NAMES:
        width = codec.group_widths[exp.GROUP_INDEX[name]]
        block = frame[:, start : start + width]
        assert (block.sum(-1) == 1.0).all()
        assert torch.equal(block.argmax(-1), indices[:, exp.GROUP_INDEX[name]])
        start += width
    assert start == codec.frame_width


def test_model_widths_follow_the_codec() -> None:
    for min_count, k, d_in in ((0, 256, 643), (100, 52, 439)):
        model = exp.GPT(exp.TrainConfig(btn_min_count=min_count))
        assert model.codec.group_widths == (k, 65, 9, 25)
        assert model.ctx_proj.in_features == d_in
        assert model.temporal.token_projection.in_features == 384 + model.codec.frame_width + 16
        film = {name: linear.in_features for name, linear in model.temporal.group_condition.items()}
        assert film == {"main_stick": 9, "triggers": 74, "buttons": 99}
        for name in exp.GROUP_NAMES:
            width = model.codec.group_widths[exp.GROUP_INDEX[name]]
            assert model.temporal.outputs[name].down.out_features == width
            assert model.temporal.trunk_outputs[name].out_features == width


def test_quantize_dequantize_is_a_fixed_point_in_class_space() -> None:
    generator = torch.Generator().manual_seed(3)
    actions = _actions(8, 16, generator)
    for min_count in (0, 100):
        codec = exp.StructuredControllerCodec(min_count)
        indices = codec.quantize(actions)
        assert int(indices[..., exp.BUTTONS_G].max()) < codec.group_widths[exp.BUTTONS_G]
        again = codec.quantize(codec.dequantize(indices))
        assert torch.equal(indices, again)


def test_rare_combo_quantizes_to_its_nearest_supported_class() -> None:
    codec = exp.StructuredControllerCodec(100)
    support = scoring.btn_combo_support(100)
    rare = int((~support).nonzero()[0])
    actions = torch.zeros(1, A_DIM)
    actions[0, 6:] = scoring.combo_to_buttons(torch.tensor(rare))
    cls = codec.quantize(actions)[0, exp.BUTTONS_G]
    landed = int(codec.class_to_combo[cls])
    assert bool(support[landed])
    assert landed == int(scoring.btn_combo_remap(100)[rare])
    # a supported combo keeps its exact bits through the round trip
    frequent = int(support.nonzero()[1])  # a supported combo with real presses
    actions[0, 6:] = scoring.combo_to_buttons(torch.tensor(frequent))
    restored = codec.dequantize(codec.quantize(codec.canonicalize(actions)))
    assert torch.equal(restored[0, 6:], scoring.combo_to_buttons(torch.tensor(frequent)))


def test_every_quantized_frame_is_valid_against_its_own_trigger_class() -> None:
    # Exhaustive: every raw combo x every trigger pair must produce a (trigger, button)
    # pair the validity mask allows, or its CE target would be -inf. This is the assertion
    # that fails first if a regenerated BTN_COMBO_COUNTS lets the remap add a click.
    triggers = torch.cartesian_prod(scoring.TRIGGER_CENTERS, scoring.TRIGGER_CENTERS)
    for min_count in (0, 100):
        codec = exp.StructuredControllerCodec(min_count)
        combos = torch.arange(scoring.N_BUTTON_COMBOS)
        actions = torch.zeros(scoring.N_BUTTON_COMBOS, triggers.shape[0], A_DIM)
        actions[..., 4:6] = triggers[None, :, :]
        actions[..., 6:] = scoring.combo_to_buttons(combos)[:, None, :]
        indices = codec.quantize(actions)
        valid = codec.button_valid_for_trigger[indices[..., exp.TRIG_G], indices[..., exp.BUTTONS_G]]
        assert valid.all()


def test_button_mask_excludes_clicks_without_full_corresponding_trigger() -> None:
    for min_count in (0, 100):
        codec = exp.StructuredControllerCodec(min_count)
        button_bits = scoring.combo_to_buttons(codec.class_to_combo)
        left_click = button_bits[:, exp.BUTTON_L_CH - 6].bool()
        assert left_click.any()
        no_trigger = torch.zeros(1, dtype=torch.long)
        assert codec.button_mask(no_trigger)[0, left_click].all()
        full_pair = torch.tensor([exp.GROUP_VOCABS[exp.TRIG_G] - 1])
        assert not codec.button_mask(full_pair)[0, left_click].any()


def test_parallel_teacher_forcing_matches_stepwise_logits() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg).eval()
    generator = torch.Generator().manual_seed(7)
    hidden = torch.randn(2, cfg.L_ctx, cfg.d_model, generator=generator)
    observed = torch.stack(
        [torch.randint(width, (2, cfg.L_ctx), generator=generator) for width in model.codec.group_widths], dim=-1
    )
    targets = torch.stack(
        [
            torch.randint(width, (2, cfg.L_ctx, len(cfg.head_offsets)), generator=generator)
            for width in model.codec.group_widths
        ],
        dim=-1,
    )
    parallel = model.temporal.teacher_forced_logits_by_group(hidden, observed, targets)
    stepwise = model.temporal.forced_stepwise_logits(hidden, observed[:, -1], targets[:, -1])
    for depth, logits in enumerate(stepwise):
        for name in exp.GROUP_NAMES:
            torch.testing.assert_close(parallel[name][:, -1, depth], logits[name], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("min_count", [0, 100])
def test_action_loss_and_sampling_smoke(min_count: int) -> None:
    cfg = _cfg(btn_min_count=min_count)
    model = exp.GPT(cfg)
    parts = exp.action_loss(model, _batch(cfg))
    assert parts.nll.shape == (7, len(model.head_offsets), exp.N_GROUPS)
    assert torch.isfinite(parts.nll).all()
    exp.objective(parts, cfg.aux_loss_weight).backward()
    model = model.eval()
    observed = torch.stack([torch.randint(width, (2, cfg.L_ctx)) for width in model.codec.group_widths], dim=-1)[:, -1]
    with torch.no_grad():
        frames = model.temporal.sample_indices(
            torch.randn(2, cfg.L_ctx, cfg.d_model), observed, cfg.head_offsets[:4], argmax=True
        )
    assert frames.shape == (2, 4, exp.N_GROUPS)
    for group, width in enumerate(model.codec.group_widths):
        assert int(frames[..., group].max()) < width
    # sampled frames never violate the trigger-click physical constraint
    native = model.codec.dequantize(frames)
    clicked_l = native[..., exp.BUTTON_L_CH] > 0.5
    assert (native[..., exp.TRIGGER_L_CH][clicked_l] == 1.0).all()


def test_optimizer_owns_every_parameter() -> None:
    for min_count in (0, 100):
        model = exp.GPT(_cfg(btn_min_count=min_count))
        optimizer = exp.make_optimizer(model, _cfg(btn_min_count=min_count))
        owned = sum(len(group["params"]) for group in optimizer.param_groups)
        assert owned == sum(1 for _ in model.parameters())


def test_checkpoint_arch_fields_pin_the_ablation_knob() -> None:
    assert "btn_min_count" in exp._CHECKPOINT_ARCH_FIELDS
    assert "action_embed_dim" not in exp._CHECKPOINT_ARCH_FIELDS
