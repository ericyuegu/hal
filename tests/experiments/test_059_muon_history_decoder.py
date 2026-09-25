"""Focused contracts for O59's sampled wide history decoder."""

import importlib.util
import random
import sys
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np
import pytest
import torch

import hal.training.physical_shard_loader as replay_loader
from hal.inference.backends.history_decoder.model import GPT as ServingGPT
from hal.training.features import NEUTRAL_ACTION
from hal.training.features import stack_actions


def _load():
    path = Path(__file__).resolve().parents[2] / "experiments" / "059_muon_history_decoder.py"
    name = "test_exp059"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exp = _load()


def _step_snapshot(model, result, sampler) -> dict[str, object]:
    return {
        "parameters": {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()},
        "gradients": {
            name: parameter.grad.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        },
        "metrics": {name: value.detach().cpu().clone() for name, value in result.metrics.items()},
        "gradient_norm": result.gradient_norm.detach().cpu().clone(),
        "sampler": sampler.generator.get_state().cpu(),
        "conditioning": model.return_calibration.state_dict(),
    }


def _assert_nested_equal(expected: object, actual: object) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(expected, actual)
    elif isinstance(expected, Mapping):
        assert isinstance(actual, Mapping)
        assert expected.keys() == actual.keys()
        for key in expected:
            _assert_nested_equal(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected)) and len(expected) == len(actual)
        for left, right in zip(expected, actual, strict=True):
            _assert_nested_equal(left, right)
    else:
        assert expected == actual


def _assert_step_close(expected: dict[str, object], actual: dict[str, object]) -> None:
    _assert_nested_equal(expected["sampler"], actual["sampler"])
    _assert_nested_equal(expected["conditioning"], actual["conditioning"])
    for field, (rtol, atol) in {"parameters": (2e-5, 2e-6), "gradients": (0.01, 2e-4)}.items():
        left = cast(Mapping[str, torch.Tensor], expected[field])
        right = cast(Mapping[str, torch.Tensor], actual[field])
        assert left.keys() == right.keys()
        for name in left:
            torch.testing.assert_close(left[name], right[name], rtol=rtol, atol=atol, msg=f"{field}/{name}")
    left_metrics = cast(Mapping[str, torch.Tensor], expected["metrics"])
    right_metrics = cast(Mapping[str, torch.Tensor], actual["metrics"])
    assert left_metrics.keys() == right_metrics.keys()
    for name in left_metrics:
        torch.testing.assert_close(left_metrics[name], right_metrics[name], rtol=0.01, atol=2e-4, msg=name)


def _tiny_cfg(*, batch_size=2, **changes):
    arch = {
        **asdict(exp.Architecture()),
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 4,
        "L_ctx": 8,
        "temporal_d_model": 32,
        "temporal_layers": 2,
        "temporal_heads": 4,
        "temporal_ff_dim": 128,
        "group_head_dim": 32,
        "value_hidden_dim": 16,
        "item_hidden_dim": 8,
        "item_dim": 5,
    }
    return exp.TrainConfig(
        arch=exp.Architecture(**arch),
        batch_size=batch_size,
        compile_trunk=False,
        compile_temporal=False,
        inference_mode="eager",
        num_workers=0,
        push_to_r2=False,
        **changes,
    )


def test_production_contract_and_full_run_adam_scaling() -> None:
    cfg = exp.TrainConfig()

    assert cfg.max_steps == 131_072
    assert cfg.warmup_steps == 4_096
    assert cfg.policy_prefixes_per_update == 2**14
    assert cfg.value_prefixes_per_update == 2**16
    assert cfg.arch.temporal_ff_dim == 4 * cfg.arch.temporal_d_model
    assert exp.proxy_config().arch.temporal_ff_dim == 4 * exp.proxy_config().arch.temporal_d_model
    assert cfg.arch.head_offsets == (*range(1, 13), 16, 20, 24, 28)
    assert cfg.minimum_replay_frames == 291
    assert exp.scaled_adam_betas(cfg) == pytest.approx((0.9875, 0.99375))
    assert exp.scaled_adam_epsilon(cfg) == pytest.approx(8**0.5 * 1e-12)


def test_wsd_is_flat_when_stable_training_is_extended() -> None:
    stable = exp.lr_schedule(exp.TrainConfig())
    assert stable(0) == pytest.approx(1 / 4096)
    assert stable(4095) == 1
    assert stable(98_303) == 1
    assert stable(200_000) == 1

    fork = replace(
        exp.TrainConfig(),
        decay_start_update=98_304,
        decay_duration=32_768,
        parent_run_name="parent",
        parent_checkpoint_name="checkpoints/step-0098304.pt",
        parent_checkpoint_sha256="a" * 64,
        parent_wandb_id="wandb-parent",
    )
    schedule = exp.lr_schedule(fork)
    assert schedule(98_303) == 1
    assert schedule(114_687) == pytest.approx((1 + 1 / 170) / 2)
    assert schedule(131_071) == pytest.approx(1 / 170)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_prefix_sampling_is_distinct_uniform_and_exactly_resumable(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for device prefix RNG resume")
    sampler = exp.PrefixSampler(7, device)
    ctx_pad = torch.tensor([0, 128], device=device)
    first = sampler.sample(ctx_pad, length=256, suffix_start=128)
    state = sampler.state_dict()
    expected_next = sampler.sample(ctx_pad, length=256, suffix_start=128)

    # Check both ordinary CPU state and a checkpoint loaded with map_location.
    for state_device in ("cpu", device):
        restored = exp.PrefixSampler(999, device)
        restored.load_state_dict({"generator": state["generator"].to(state_device)})
        actual_next = restored.sample(ctx_pad, length=256, suffix_start=128)
        assert torch.equal(actual_next, expected_next)

    assert first.shape == (2, 32)
    assert torch.all(first[:, 1:] > first[:, :-1])
    assert int(first.min()) >= 128 and int(first.max()) < 256


def test_global_rng_state_is_exactly_resumable() -> None:
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    state = exp.rng_state()
    expected = (random.random(), float(np.random.random()), torch.rand(4))

    exp.restore_rng(state)
    actual = (random.random(), float(np.random.random()), torch.rand(4))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_exact_offset_coefficients_and_dense_awr_gather() -> None:
    nll = torch.ones(2, 32, 16, 1)
    weights = torch.full((2, 32), 2.0)
    valid = torch.ones(2, 32, dtype=torch.bool)
    near, far, total = exp.temporal_objective_parts(
        nll,
        weights,
        valid_prefixes=64,
        valid=valid,
    )

    assert near == pytest.approx(1.7)
    assert far == pytest.approx(0.15)
    assert total == pytest.approx(1.85)
    assert sum(exp.OFFSET_LOSS_WEIGHTS) == pytest.approx(1.0)

    advantage = torch.arange(2 * 128, dtype=torch.float32).view(2, 128)
    eligible = torch.ones_like(advantage, dtype=torch.bool)
    dense, _ = exp.advantage_weights(advantage, eligible, beta=199.5, weight_max=3.5)
    positions = torch.tensor([[0, 17, 127], [4, 63, 100]])
    rows = torch.arange(2)[:, None]
    assert torch.equal(dense[rows, positions], dense.gather(1, positions))


def test_history_attention_is_causal_and_has_backward_parity() -> None:
    torch.manual_seed(3)
    cfg = _tiny_cfg()
    decoder = exp.GPT(cfg).temporal
    hidden = torch.randn(2, 8, 32, requires_grad=True)
    ctx_pad = torch.tensor([0, 2])
    prefixes = torch.tensor([[4, 7], [4, 7]])
    observed = torch.zeros(2, 2, exp.CONTROLLER_GROUP_COUNT, dtype=torch.long)
    targets = torch.zeros(2, 2, 16, exp.CONTROLLER_GROUP_COUNT, dtype=torch.long)

    output = decoder.teacher_forced_states(
        hidden, ctx_pad, prefixes, observed, targets, torch.zeros(2, 2), torch.ones(2, 2, dtype=torch.bool)
    )
    output[:, 0].sum().backward()
    assert hidden.grad is not None
    assert torch.equal(hidden.grad[0, 5:], torch.zeros_like(hidden.grad[0, 5:]))
    assert torch.equal(hidden.grad[1, :2], torch.zeros_like(hidden.grad[1, :2]))

    changed = hidden.detach().clone()
    changed[:, 5:] += 100
    isolated = decoder.teacher_forced_states(
        changed, ctx_pad, prefixes, observed, targets, torch.zeros(2, 2), torch.ones(2, 2, dtype=torch.bool)
    )
    torch.testing.assert_close(output[:, 0], isolated[:, 0])


@pytest.mark.parametrize("norm_eps", [1e-6, 1e-5])
def test_action_head_forward_and_diagnostics_have_gradient_parity(norm_eps: float) -> None:
    torch.manual_seed(5)
    head = exp.NonlinearActionHead(32, 32, 8, norm_eps=norm_eps)
    inputs = (torch.randn(2, 32) * 1e-3).requires_grad_()
    direct = head(inputs)
    diagnostic, _ = head.forward_with_input(inputs)
    parameters = (inputs, *head.parameters())
    direct_gradients = torch.autograd.grad(direct.square().sum(), parameters)
    diagnostic_gradients = torch.autograd.grad(diagnostic.square().sum(), parameters)

    torch.testing.assert_close(direct, diagnostic, rtol=0, atol=0)
    for actual, expected in zip(direct_gradients, diagnostic_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_action_group_conditioning_uses_adaptive_rmsnorm_and_cumulative_prefix() -> None:
    torch.manual_seed(17)
    cfg = _tiny_cfg(batch_size=2)
    decoder = exp.GPT(cfg).temporal.eval()
    assert exp.CONTROLLER_DECODE_ORDER == ("c_stick", "main_stick", "triggers", "buttons")
    states = torch.randn(2, 3, cfg.arch.temporal_d_model, requires_grad=True)
    embedded = {
        name: torch.randn(2, 3, cfg.arch.action_embed_dim, requires_grad=True) for name in exp.CONTROLLER_DECODE_ORDER
    }
    for position, name in enumerate(exp.CONTROLLER_DECODE_ORDER):
        head = decoder.outputs[name]
        normalized = head.normalize(states)
        actual = decoder.group_features(states, name, embedded)
        if position == 0:
            torch.testing.assert_close(actual, normalized)
            continue
        condition = decoder.group_condition[name]
        assert condition.in_features == position * cfg.arch.action_embed_dim
        prefix = torch.cat([embedded[earlier] for earlier in exp.CONTROLLER_DECODE_ORDER[:position]], dim=-1)
        raw_scale, shift = condition(prefix).chunk(2, dim=-1)
        expected = normalized * (1 + torch.tanh(raw_scale)) + shift
        torch.testing.assert_close(actual, expected)

    with torch.no_grad():
        decoder.group_condition["buttons"].weight.fill_(0.01)
    decoder.group_features(states, "buttons", embedded).square().sum().backward()
    assert states.grad is not None and torch.count_nonzero(states.grad) > 0
    for name in exp.CONTROLLER_DECODE_ORDER[:-1]:
        assert embedded[name].grad is not None and torch.count_nonzero(embedded[name].grad) > 0
    assert embedded["buttons"].grad is None


def test_teacher_forced_action_head_does_not_renormalize_after_film() -> None:
    torch.manual_seed(29)
    cfg = _tiny_cfg(batch_size=1)
    decoder = exp.GPT(cfg).temporal.eval()
    condition = decoder.group_condition["buttons"]
    with torch.no_grad():
        condition.weight.zero_()
        condition.bias[: cfg.arch.temporal_d_model].fill_(torch.atanh(torch.tensor(0.25)).item())
        condition.bias[cfg.arch.temporal_d_model :].fill_(0.5)
    hidden = torch.randn(1, 8, cfg.arch.d_model)
    ctx_pad = torch.tensor([1])
    prefixes = torch.tensor([[7]])
    observed = torch.zeros(1, 1, exp.CONTROLLER_GROUP_COUNT, dtype=torch.long)
    targets = torch.zeros(1, 1, len(cfg.arch.head_offsets), exp.CONTROLLER_GROUP_COUNT, dtype=torch.long)
    return_value = torch.zeros(1, 1)
    present = torch.ones(1, 1, dtype=torch.bool)
    states = decoder.teacher_forced_states(hidden, ctx_pad, prefixes, observed, targets, return_value, present)
    expected = decoder.outputs["buttons"].normalize(states) * 1.25 + 0.5
    _logits, button_values, projections = decoder._teacher_forced_outputs(
        hidden, ctx_pad, prefixes, observed, targets, return_value, present
    )
    head_input, _combined_logits, _mask = button_values
    projected_input, _up, _down, head_output, *_trunk = projections["buttons"]
    torch.testing.assert_close(head_input, expected)
    torch.testing.assert_close(projected_input, expected)
    torch.testing.assert_close(head_output, decoder.outputs["buttons"].project(expected))
    assert not torch.allclose(head_input, decoder.outputs["buttons"].normalize(expected))


@pytest.mark.parametrize("button_input_scale", [1.0, 1e-3])
def test_teacher_forced_and_stepwise_decoding_match(button_input_scale: float) -> None:
    torch.manual_seed(5)
    cfg = _tiny_cfg(batch_size=1)
    decoder = exp.GPT(cfg).temporal.eval()
    assert decoder.outputs["buttons"].norm_eps == 1e-5
    assert all(head.norm_eps == 1e-6 for head in decoder.trunk_outputs.values())
    if button_input_scale != 1.0:
        # Small conditioned states expose normalization differences at the head.
        condition = decoder.group_condition["buttons"]
        with torch.no_grad():
            condition.weight.zero_()
            condition.bias.zero_()
            condition.bias[: cfg.arch.temporal_d_model].fill_(
                torch.atanh(torch.tensor(button_input_scale - 1.0)).item()
            )
    hidden = torch.randn(1, 8, 32)
    ctx_pad = torch.tensor([2])
    prefix = torch.tensor([[7]])
    observed = torch.zeros(1, 1, exp.CONTROLLER_GROUP_COUNT, dtype=torch.long)
    targets = torch.zeros(1, 1, 16, exp.CONTROLLER_GROUP_COUNT, dtype=torch.long)

    batched = decoder.teacher_forced_logits_by_group(
        hidden, ctx_pad, prefix, observed, targets, torch.zeros(1, 1), torch.ones(1, 1, dtype=torch.bool)
    )
    stepwise = decoder.forced_stepwise_logits(
        hidden, observed[:, 0], targets[:, 0], torch.zeros(1), torch.ones(1, dtype=torch.bool), ctx_pad=ctx_pad
    )
    for depth in range(16):
        for name in exp.CONTROLLER_GROUP_NAMES:
            torch.testing.assert_close(batched[name][:, 0, depth], stepwise[depth][name], atol=2e-5, rtol=2e-5)


def test_live_decode_preserves_committed_action_prefix() -> None:
    torch.manual_seed(11)
    cfg = _tiny_cfg(batch_size=1)
    decoder = exp.GPT(cfg).temporal.eval()
    hidden = torch.randn(1, 8, 32)
    observed = torch.zeros(1, exp.CONTROLLER_GROUP_COUNT, dtype=torch.long)
    committed = torch.zeros(1, 2, exp.CONTROLLER_GROUP_COUNT, dtype=torch.long)
    committed[:, 0, exp.TRIGGERS_GROUP] = 1

    decoded = decoder.sample_indices(
        hidden,
        observed,
        tuple(range(1, 5)),
        torch.zeros(1),
        torch.ones(1, dtype=torch.bool),
        argmax=True,
        ctx_pad=torch.tensor([2]),
        forced_prefix=committed,
    )

    assert torch.equal(decoded[:, :2], committed)


def test_optimizer_membership_and_logical_splits_are_complete() -> None:
    cfg = _tiny_cfg()
    model = exp.GPT(cfg)
    roles = exp.optimizer_roles(model, cfg)
    optimizer = exp.make_optimizer(model, cfg)
    members = [parameter for group in optimizer.param_groups for parameter in group["params"]]

    assert set(roles) == dict(model.named_parameters()).keys()
    assert len(members) == len({id(parameter) for parameter in members})
    assert {id(parameter) for parameter in members} == {id(parameter) for parameter in model.parameters()}
    assert roles["temporal.blocks.0.qkv.weight"].logical_splits == 3
    assert roles["temporal.history_attention.key_value.weight"].logical_splits == 2
    assert roles["temporal.trunk_outputs.buttons.up.weight"].optimizer == "muon"
    assert roles["temporal.trunk_outputs.buttons.up.weight"].lr_kind == "hidden"
    assert roles["temporal.trunk_outputs.buttons.down.weight"].optimizer == "adamw"
    assert roles["temporal.trunk_outputs.buttons.down.weight"].lr_kind == "output"
    for name in exp.CONTROLLER_DECODE_ORDER[1:]:
        weight = roles[f"temporal.group_condition.{name}.weight"]
        bias = roles[f"temporal.group_condition.{name}.bias"]
        assert (weight.optimizer, weight.lr_kind, weight.decay) == ("adamw", "input", True)
        assert (bias.optimizer, bias.lr_kind, bias.decay) == ("adamw", "vector", False)


def test_trunk_skip_uses_matching_nonlinear_heads_and_parameter_contract() -> None:
    cfg = exp.proxy_config()
    model = exp.GPT(cfg)

    for name in exp.CONTROLLER_GROUP_NAMES:
        decoder_head = model.temporal.outputs[name]
        trunk_head = model.temporal.trunk_outputs[name]
        assert isinstance(decoder_head, exp.NonlinearActionHead)
        assert isinstance(trunk_head, exp.NonlinearActionHead)
        assert trunk_head.up.weight.shape == decoder_head.up.weight.shape
        assert trunk_head.down.weight.shape == decoder_head.down.weight.shape
    assert exp.subsystem_parameter_counts(model) == cfg.arch.parameter_count_contract
    assert "dual-nonlinear-head" in exp.model_tag(cfg)


def _live_inputs():
    from hal.data.feature_stats import FeatureStats
    from hal.training.ego_stats import consolidate_key
    from hal.training.features import feature_kind

    fields = exp.BASE_ITEMS_PROJECTION.columns - {f"ego_{name}" for name in exp.ACTION_CHANNELS}
    flat = {}
    for name in fields:
        canonical = (
            f"p1_{name[4:]}" if name.startswith("ego_") else f"p2_{name[4:]}" if name.startswith("opp_") else name
        )
        flat[canonical] = 0.0
    stats = {
        consolidate_key(name): FeatureStats(mean=0.0, std=1.0, min=-1.0, max=1.0)
        for name in fields
        if feature_kind(name, exp.ITEM_COLUMNS) == "float"
    }
    return flat, stats


@pytest.mark.parametrize(("delay", "stride"), [(2, 2), (0, 2), (1, 2), (3, 1)])
def test_make_policy_plan_rows_bootstrap_continue_and_reset(delay: int, stride: int) -> None:
    from hal.sim.rollout import ObservationRow
    from hal.sim.vec import Slot

    torch.manual_seed(19)
    cfg = _tiny_cfg(batch_size=1)
    model = exp.GPT(cfg).eval()
    flat, stats = _live_inputs()
    policy = exp.make_policy(
        model, stats, cfg, decode_seed=3, device="cpu", delay_frames=delay, replan_interval_frames=stride
    )
    slot = Slot(0, 2)
    neutral = exp.NEUTRAL_ACTION.copy()
    first = policy.plan_rows({slot: [ObservationRow(10, flat, neutral, reset=True)]})[slot]
    assert policy.runtime_spec.committed_frames == delay
    assert policy.runtime_spec.execution_stride == stride
    np.testing.assert_array_equal(first[:delay], np.broadcast_to(neutral, (delay, len(neutral))))
    first_copy = first.copy()
    rows = [ObservationRow(11 + index, flat, first[index].copy()) for index in range(stride)]
    second = policy.plan_rows({slot: rows})[slot]
    np.testing.assert_array_equal(second[:delay], first_copy[stride : stride + delay])
    np.testing.assert_array_equal(first, first_copy)
    reset = policy.plan_rows({slot: [ObservationRow(-123, flat, neutral, reset=True)]})[slot]
    np.testing.assert_array_equal(reset[:delay], first_copy[:delay])


@pytest.mark.parametrize("delay", [0, 1, 2, 3])
def test_traced_and_normal_decoding_match_actions_draws_and_padding(delay: int) -> None:
    torch.manual_seed(23)
    cfg = _tiny_cfg(batch_size=3)
    model = exp.GPT(cfg).eval()
    engine = exp.BF16Inference(model, cfg, compiled=False)
    normal_rng = exp.SlotGroupRng(41, exp.CONTROLLER_GROUP_NAMES)
    trace_rng = exp.SlotGroupRng(41, exp.CONTROLLER_GROUP_NAMES)
    ctx = exp.synthetic_context(cfg, 3, torch.device("cpu"))
    ctx = replace(ctx, ctx_pad=torch.tensor([0, 2, 6]), slot_ids=torch.tensor([1, 2, 9]))
    prefix = torch.zeros(3, delay, len(exp.ACTION_CHANNELS))
    prefix[..., 0] = 0.333
    for resets in ([True, True, True], [False, True, False]):
        ctx = replace(ctx, reset=torch.tensor(resets))
        actions = engine.decode(ctx, 4, streams=normal_rng, committed=prefix)
        traced = engine.decode_with_trace(ctx, 4, streams=trace_rng, committed=prefix)
        assert torch.equal(actions, traced.actions)
        assert normal_rng.state() == trace_rng.state()
        assert torch.equal(traced.indices[:, :delay], model.codec.quantize(prefix))
        assert torch.all(traced.uniforms[:delay] == 0.5)
        assert all(values.shape[:2] == (3, 4) for values in traced.logits)
    assert {counter for slot, generation, group, counter in normal_rng.state() if slot == 1} == {2 * (4 - delay)}
    assert set(engine._decoders) == {(4, 4, delay)}
    assert set(engine._trace_decoders) == {(4, 4, delay)}


@pytest.mark.parametrize("traced", [False, True])
@pytest.mark.parametrize("shape", [(1, 2), (2, 2, 14), (1, 2, 13), (1, 5, 14)])
def test_decode_rejects_invalid_commitment_before_advancing_rng(traced: bool, shape: tuple[int, ...]) -> None:
    cfg = _tiny_cfg(batch_size=1)
    engine = exp.BF16Inference(exp.GPT(cfg).eval(), cfg, compiled=False)
    rng = exp.SlotGroupRng(7, exp.CONTROLLER_GROUP_NAMES)
    context = exp.synthetic_context(cfg, 1, torch.device("cpu"))
    decode = engine.decode_with_trace if traced else engine.decode
    with pytest.raises(ValueError, match="committed actions"):
        decode(context, 4, streams=rng, committed=torch.zeros(shape))
    assert rng.state() == ()
    assert engine._trunks == {}


def test_normal_decode_keeps_logits_out_of_the_decoder_path() -> None:
    cfg = _tiny_cfg(batch_size=1)
    engine = exp.BF16Inference(exp.GPT(cfg).eval(), cfg, compiled=False)
    context = exp.synthetic_context(cfg, 1, torch.device("cpu"))
    generator = torch.Generator().manual_seed(5)
    engine.decode(context, 4, gen=generator, committed=torch.zeros(1, 2, len(exp.ACTION_CHANNELS)))
    expected = torch.Generator().manual_seed(5)
    for _ in range(2 * exp.CONTROLLER_GROUP_COUNT):
        torch.rand(1, generator=expected)
    assert torch.equal(generator.get_state(), expected.get_state())
    assert engine._trace_decoders == {}


def test_prewarm_cache_includes_resolved_delay() -> None:
    cfg = _tiny_cfg()
    engine = exp.BF16Inference(exp.GPT(cfg).eval(), cfg, compiled=False)
    for delay in (0, 1, 2, 3):
        assert engine.prewarm(3, 4, committed_frames=delay) == 0
    assert engine._warmed == {(4, 4, delay) for delay in range(4)}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for live recompilation verification")
def test_resolved_delay_prewarm_avoids_live_recompilation() -> None:
    cfg = _tiny_cfg(batch_size=3)
    # Compiled flex attention requires at least 16 channels per head.
    cfg = replace(cfg, arch=replace(cfg.arch, temporal_heads=2))
    model = exp.GPT(cfg).eval().cuda()
    with torch.no_grad():
        for projection in model.temporal.return_conditioner.projections:
            projection.weight.normal_(std=0.01)
            projection.bias.normal_(std=0.01)
        for condition in model.temporal.group_condition.values():
            condition.bias.normal_(std=0.1)
    engine = exp.BF16Inference(model, cfg, compiled=True, bucket=4, desired_return=120.0)
    context = exp.synthetic_context(cfg, 3, torch.device("cuda"))
    for delay in (0, 1, 2, 3):
        engine.prewarm(3, 4, committed_frames=delay)
        for desired_return in (180.0, None):
            engine.desired_return = desired_return
            with torch.compiler.set_stance("fail_on_recompile"):
                engine.decode(context, 4, committed=torch.zeros(3, delay, len(exp.ACTION_CHANNELS), device="cuda"))
        assert engine.prewarm(3, 4, committed_frames=delay) == 0
    context = replace(
        context, slot_ids=torch.arange(3, device="cuda"), reset=torch.ones(3, dtype=torch.bool, device="cuda")
    )
    committed = torch.zeros(3, 2, len(exp.ACTION_CHANNELS), device="cuda")
    engine.decode_with_trace(context, 4, streams=exp.SlotGroupRng(41, exp.CONTROLLER_GROUP_NAMES), committed=committed)
    for desired_return in (120.0, None):
        engine.desired_return = desired_return
        normal_rng = exp.SlotGroupRng(41, exp.CONTROLLER_GROUP_NAMES)
        traced_rng = exp.SlotGroupRng(41, exp.CONTROLLER_GROUP_NAMES)
        with torch.compiler.set_stance("fail_on_recompile"):
            actions = engine.decode(context, 4, streams=normal_rng, committed=committed)
            traced = engine.decode_with_trace(context, 4, streams=traced_rng, committed=committed)
        assert torch.equal(actions, traced.actions)
        assert normal_rng.state() == traced_rng.state()


def test_target_gradient_attribution_matches_autograd_and_fixed_coefficients() -> None:
    torch.manual_seed(29)
    logits = torch.randn(2, 3, 16, exp.CONTROLLER_GROUP_COUNT, 7, requires_grad=True)
    targets = torch.randint(7, logits.shape[:-1])
    nll = -logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
    weights = torch.tensor([[0.2, 1.3, 2.7], [1.7, 0.4, 3.1]])
    valid = torch.tensor([[True, False, True], [False, True, True]])
    near, far, total = exp.temporal_objective_parts(nll, weights, valid_prefixes=6, valid=valid)
    joint = torch.where(valid[..., None], nll.sum(-1), 0)
    expected_near = (joint[..., 0] * weights * 0.5).sum() / 6
    expected_near += (joint[..., 1:6] * weights[..., None] * 0.07).sum() / 6
    expected_far = (joint[..., 6:] * 0.015).sum() / 6
    torch.testing.assert_close(near, expected_near)
    torch.testing.assert_close(far, expected_far)
    torch.testing.assert_close(total, expected_near + expected_far)
    total.backward()
    target_grad = logits.grad.gather(-1, targets[..., None]).squeeze(-1).abs().sum((0, 1))
    metrics = exp.projection_local_target_gradient_l1(
        nll, weights, offsets=exp.Architecture.head_offsets, valid_prefixes=6, valid=valid
    )
    prefix = "diagnostics/projection_local/target_logit_grad_l1"
    for depth, offset in enumerate(exp.Architecture.head_offsets):
        for group, name in enumerate(exp.CONTROLLER_GROUP_NAMES):
            torch.testing.assert_close(metrics[f"{prefix}/o{offset:02d}/{name}"], target_grad[depth, group])
        torch.testing.assert_close(metrics[f"{prefix}/by_offset/o{offset:02d}"], target_grad[depth].sum())
    for group, name in enumerate(exp.CONTROLLER_GROUP_NAMES):
        torch.testing.assert_close(metrics[f"{prefix}/by_action_group/{name}"], target_grad[:, group].sum())
    torch.testing.assert_close(metrics[f"{prefix}/total"], target_grad.sum())
    means = nll.detach().mean((0, 1))
    unweighted = exp.nll_mean_metrics(means, exp.Architecture.head_offsets)
    expected = (means.sum(-1) * torch.tensor([0.5, *([0.07] * 5), *([0.015] * 10)])).sum() / np.log(2)
    assert unweighted["loss_unweighted"] == pytest.approx(float(expected))


def test_awr_normalization_uses_dense_eligible_prefixes_before_sampling() -> None:
    advantage = torch.tensor([[0.0, 100.0, -70.0, 900.0], [20.0, -150.0, 70.0, 200.0]])
    eligible = torch.tensor([[True, True, False, True], [True, False, True, True]])
    valid = torch.tensor([[True, True, True, True], [True, True, True, False]])
    weights, _ = exp.advantage_weights(advantage, eligible, beta=199.5, weight_max=3.5, valid=valid)
    selected = eligible & valid
    raw = torch.exp(advantage / 199.5).clamp(max=3.5)
    expected = torch.where(selected, raw / raw[selected].mean(), 1.0)
    torch.testing.assert_close(weights, expected)
    positions = torch.tensor([[0, 3], [0, 2]])
    sampled = weights.gather(1, positions)
    torch.testing.assert_close(sampled, expected.gather(1, positions))
    assert not torch.isclose(sampled.mean(), torch.tensor(1.0))


def test_checkpoint_config_keeps_auxiliary_field_at_one() -> None:
    cfg = exp.TrainConfig()
    payload = exp._checkpoint_config(cfg)
    assert payload["awr_calibration"]["auxiliary_loss_weight"] == 1.0
    restored = exp.config_from_state(payload)
    assert restored == cfg
    exp.validate_config(restored)
    with pytest.raises(ValueError, match="checkpoint compatibility"):
        exp.validate_config(replace(cfg, awr=replace(cfg.awr, auxiliary_loss_weight=0.5)))


def test_evaluation_persists_emulator_and_inference_metrics_separately(tmp_path, monkeypatch) -> None:
    import json

    from hal.sim.rollout import ObservationRow
    from hal.sim.vec import Slot

    class Clock:
        seconds = 0.0

        def perf_counter(self):
            return self.seconds

    clock = Clock()
    cfg = _tiny_cfg(batch_size=1)
    model = exp.GPT(cfg)
    flat, stats = _live_inputs()
    engine = exp.BF16Inference(model, cfg, compiled=False)
    warmed = []

    def prewarm(rows, horizon, *, committed_frames):
        warmed.append((rows, horizon, committed_frames))
        clock.seconds += 30
        return 30.0

    def decode(context, horizon, **kwargs):
        assert kwargs["committed"].shape == (1, 1, len(exp.ACTION_CHANNELS))
        clock.seconds += 0.25
        return torch.zeros(1, horizon, len(exp.ACTION_CHANNELS))

    rows = [
        exp.MatchRow(1, 2, 31, 0, 0, 100, 123, 1.0, 2.0, 0, 1),
        exp.MatchRow(1, 2, 31, 0, 1, 150, 177, 3.0, 4.0, 1, 0),
    ]

    def sweep(factory, *, process_telemetry, **kwargs):
        assert kwargs["max_parallel"] == 1
        policy = factory()
        assert policy.runtime_spec.committed_frames == 1
        policy.plan_rows({Slot(0, 1): [ObservationRow(0, flat, exp.NEUTRAL_ACTION, reset=True)]})
        process_telemetry.plan_calls = 1
        process_telemetry.plan_rows = 1
        process_telemetry.inference_latency_ms.append(300.0)
        clock.seconds += 3.75
        return [], rows

    monkeypatch.setattr(exp, "time", clock)
    monkeypatch.setattr(engine, "prewarm", prewarm)
    monkeypatch.setattr(engine, "decode", decode)
    monkeypatch.setattr(exp, "sweep_vs_cpu_prior_with_rows", sweep)
    monkeypatch.setattr(exp, "default_session_cfg", lambda *args, **kwargs: None)
    metrics = exp.eval_vs_cpu(
        model, stats, cfg, n_matchups=1, replay_dir=tmp_path, inference=engine, max_parallel=1, delay_frames=1
    )
    assert model.training
    assert warmed == [(1, 4, 1)]
    assert metrics["eval_total_wall_seconds"] == 34.0
    assert metrics["eval_wall_seconds"] == 4.0
    assert metrics["inference_compile_seconds"] == 30.0
    assert metrics["captured_emulator_frames"] == 300.0
    assert metrics["emulator_fps"] == 75.0
    assert metrics["decode_predicted_actions_per_s"] == 16.0
    assert metrics["decode_p95_ms"] == 250.0
    assert metrics["broker_inference_latency_p95_ms"] == 300.0
    assert metrics["broker_plan_rows"] == 1.0
    assert "decode_executed_frames_per_s" not in metrics
    assert "neutral_action_fraction" not in metrics
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics
    selected = exp._eval_wandb_metrics(metrics)
    assert selected["emulator_fps"] == 75.0
    assert selected["broker_inference_latency_p95_ms"] == 300.0
    assert "decode_executed_frames_per_s" in exp.DecodeTelemetry().metrics()


@pytest.mark.parametrize("batch_size", [512, 1024, 2048])
def test_cpu_validated_prefix_sampling_preserves_draws_and_rng(batch_size: int) -> None:
    padding = torch.arange(batch_size) % 129
    checked = exp.PrefixSampler(7, "cpu")
    trusted = exp.PrefixSampler(7, "cpu")
    for _ in range(3):
        expected = checked.sample(padding, length=256, suffix_start=128)
        actual = trusted.sample(padding, length=256, suffix_start=128, validated_on_cpu=True)
        assert torch.equal(expected, actual)
        assert torch.equal(checked.generator.get_state(), trusted.generator.get_state())


def test_direct_prefix_sampling_rejects_short_suffix_before_drawing() -> None:
    sampler = exp.PrefixSampler(7, "cpu")
    before = sampler.generator.get_state()
    with pytest.raises(ValueError, match="32 real suffix"):
        sampler.sample(torch.tensor([225]), length=256, suffix_start=128)
    assert torch.equal(before, sampler.generator.get_state())


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_training_validation_change_and_next_update_resume_are_exact(
    active: bool, device: str, tmp_path: Path
) -> None:
    import copy

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for device next-update resume")
    cfg = _tiny_cfg(batch_size=1)
    cfg = replace(
        cfg,
        arch=replace(cfg.arch, L_ctx=128, n_heads=2, temporal_heads=2),
        amp_dtype="bfloat16" if device == "cuda" else "float32",
    )
    torch.manual_seed(13)
    batch = _return_batch(cfg).to(device)
    batch.context.features["ego_player_id"].fill_(17)
    batch = replace(batch, batch=replace(batch.batch, context=replace(batch.context, slot_ids=None, reset=None)))
    checked_model = exp.GPT(cfg).to(device)
    trusted_model = copy.deepcopy(checked_model)

    def setup(model):
        optimizer = exp.make_optimizer(model, cfg)
        scheduler = exp.LambdaLR(optimizer, exp.lr_schedule(cfg))
        sampler = exp.PrefixSampler(7, device)
        return optimizer, scheduler, sampler

    def update(model, state, index, trusted, selected_batch):
        optimizer, scheduler, sampler = state
        return exp.train_step(
            model,
            selected_batch,
            cfg,
            step=(4096 if active else 0) + index,
            update=index + 1,
            valid_prefixes=32,
            trunk_fn=model.forward,
            temporal_fn=model.temporal.teacher_forced_nll_with_diagnostics,
            optimizer=optimizer,
            scheduler=scheduler,
            prefix_sampler=sampler,
            prefix_validated_on_cpu=trusted,
        )

    checked = setup(checked_model)
    trusted = setup(trusted_model)
    for index in range(2):
        left = update(checked_model, checked, index, False, batch)
        right = update(trusted_model, trusted, index, True, batch)
        _assert_step_close(
            _step_snapshot(checked_model, left, checked[2]),
            _step_snapshot(trusted_model, right, trusted[2]),
        )
    identity_masker = exp.IdentityMasker(11, 0.5)
    return_masker = exp.ReturnMasker(13, 0.2)
    first_masked = return_masker(identity_masker(batch.to("cpu")))
    trusted_model.return_calibration.observe(first_masked)
    checkpoint = tmp_path / "training-resume.pt"
    next_batch = copy.deepcopy(batch)
    next_batch.context.features["ego_position_x"].fill_(0.5)
    next_batch = replace(next_batch, returns=next_batch.returns + 1)
    assert exp._train_batch_sha256(next_batch.batch) != exp._train_batch_sha256(batch.batch)
    torch.save(
        {
            "model": trusted_model.state_dict(),
            "optimizer": trusted[0].state_dict(),
            "scheduler": trusted[1].state_dict(),
            "prefix": trusted[2].state_dict(),
            "rng": exp.rng_state(),
            "batch": exp._return_batch_state(next_batch),
            "identity_masker": identity_masker.state_dict(),
            "return_masker": return_masker.state_dict(),
            "calibration": trusted_model.return_calibration.state_dict(),
        },
        checkpoint,
    )
    expected_batch = return_masker(identity_masker(next_batch.to("cpu")))
    trusted_model.return_calibration.observe(expected_batch)
    expected = update(trusted_model, trusted, 2, True, expected_batch.to(device))
    expected_state = _step_snapshot(trusted_model, expected, trusted[2])
    saved = torch.load(checkpoint, weights_only=False)
    restored_model = exp.GPT(cfg).to(device)
    restored = setup(restored_model)
    restored_model.load_state_dict(saved["model"])
    restored[0].load_state_dict(saved["optimizer"])
    restored[1].load_state_dict(saved["scheduler"])
    restored[2].load_state_dict(saved["prefix"])
    exp.restore_rng(saved["rng"])
    restored_batch = exp._return_batch_from_state(saved["batch"]).to(device)
    assert exp._train_batch_sha256(restored_batch.batch) == exp._train_batch_sha256(next_batch.batch)
    assert torch.equal(restored_batch.returns, next_batch.returns)
    assert torch.equal(restored_batch.eligible, next_batch.eligible)
    restored_identity = exp.IdentityMasker(999, 0.5)
    restored_return = exp.ReturnMasker(999, 0.2)
    restored_identity.load_state_dict(saved["identity_masker"])
    restored_return.load_state_dict(saved["return_masker"])
    restored_model.return_calibration.load_state_dict(saved["calibration"])
    restored_batch = restored_return(restored_identity(restored_batch.to("cpu")))
    restored_model.return_calibration.observe(restored_batch)
    assert exp._return_batch_sha256(restored_batch) == exp._return_batch_sha256(expected_batch)
    assert torch.equal(restored_return.generator.get_state(), return_masker.generator.get_state())
    assert torch.equal(restored_identity.generator.get_state(), identity_masker.generator.get_state())
    actual = update(restored_model, restored, 2, True, restored_batch.to(device))
    _assert_nested_equal(expected_state, _step_snapshot(restored_model, actual, restored[2]))


def test_compile_mode_is_versioned_and_preserved_on_resume() -> None:
    cfg = replace(exp.TrainConfig(), train_compile_mode="max-autotune")
    exp.validate_config(cfg)
    payload = exp._checkpoint_config(cfg)
    assert payload["experiment_id"] == "059_muon_history_decoder_v5"
    assert payload["checkpoint_format_version"] == 4
    assert exp.config_from_state(payload).train_compile_mode == "max-autotune"
    for version in (1, 2, 3):
        with pytest.raises(ValueError, match="checkpoint format version"):
            exp.config_from_state({**payload, "checkpoint_format_version": version})
    for previous in (
        "059_muon_history_decoder_v1",
        "059_muon_history_decoder_v2",
        "059_muon_history_decoder_v3",
        "059_muon_history_decoder_v4",
    ):
        with pytest.raises(ValueError, match="checkpoint experiment_id"):
            exp.config_from_state({**payload, "experiment_id": previous})
    with pytest.raises(ValueError, match="unsupported training compile mode"):
        exp.validate_config(replace(cfg, train_compile_mode="unsupported"))


def _return_batch(cfg, *, values=None, available=None):
    batch = exp.synthetic_awr_batch(cfg, torch.device("cpu"))
    context = replace(batch.context, slot_ids=None, reset=None)
    if values is None:
        values = torch.arange(cfg.batch_size * cfg.arch.L_ctx).reshape(cfg.batch_size, cfg.arch.L_ctx).float()
    if available is None:
        available = torch.ones_like(values, dtype=torch.bool)
    return replace(
        batch,
        batch=replace(batch.batch, context=context, replay_ids=tuple(f"replay-{i}" for i in range(cfg.batch_size))),
        future_return=values,
        available=available,
        condition_present=available.clone(),
    )


def test_return_recipe_and_production_parameter_count() -> None:
    cfg = exp.TrainConfig()
    assert (cfg.arch.n_layers, cfg.arch.temporal_layers, cfg.arch.return_embed_dim) == (12, 6, 128)
    assert (exp.proxy_config().arch.n_layers, exp.proxy_config().arch.temporal_layers) == (12, 6)
    assert (cfg.awr.beta, cfg.awr.weight_max, cfg.awr.gamma) == (150.0, 10.0, 0.99855)
    assert cfg.return_conditioning and cfg.return_dropout == 0.2
    with torch.device("meta"):
        model = exp.GPT(cfg)
    assert exp.subsystem_parameter_counts(model)["total"] == 246_862_205
    roles = exp.optimizer_roles(model, cfg)
    for name in dict(model.named_parameters()):
        if "return_conditioner" in name:
            assert roles[name].optimizer == "adamw"
            assert roles[name].lr_kind == ("input" if name.endswith("weight") else "vector")


def test_future_return_horizon_terminal_and_unavailable_tail() -> None:
    sample = {
        f"p{port}_{field}": np.full(65, value, dtype=np.float32)
        for port in (1, 2)
        for field, value in (("stock", 4), ("percent", 0))
    }
    sample["p2_percent"][1:] += 2
    sample["p2_percent"][60:] += 3
    sample["p2_percent"][61:] += 5
    result = exp.future_return_labels(sample)
    assert result["p1_return60"][0] == pytest.approx(2 + 3 * 0.99855**59)
    assert result["p1_return60"][1] == pytest.approx(3 * 0.99855**58 + 5 * 0.99855**59)
    assert result["p1_return60_valid"].sum() == 5
    assert np.isnan(result["p1_return60"][5:]).all()
    np.testing.assert_array_equal(result["p1_return60"], -result["p2_return60"])
    sample["p2_stock"][:] = 1
    sample["p2_stock"][10:] = 0
    result = exp.future_return_labels(sample)
    assert result["p1_return60"][0] == pytest.approx(2 + 170 * 0.99855**9)
    assert result["p1_return60_valid"].all()
    assert (result["p1_return60"][10:] == 0).all()
    sample["p2_stock"][:] = 4
    sample["mc_terminated"] = np.asarray(True)
    assert exp.future_return_labels(sample)["p1_return60_valid"].all()


def test_collation_keeps_awr_next_frame_and_conditioning_current_position() -> None:
    cfg = _tiny_cfg()
    raw = _return_batch(cfg).batch
    windows = []
    for row in range(cfg.batch_size):
        values = np.arange(cfg.arch.L_ctx + 28, dtype=np.float32) + row * 100
        windows.append(
            {
                "ego_awr_return": values,
                "ego_awr_return_valid": np.ones_like(values, dtype=bool),
                "ego_return60": values + 1000,
                "ego_return60_valid": np.ones_like(values, dtype=bool),
            }
        )
    batch = exp.collate_awr_batch(windows, raw, L_ctx=cfg.arch.L_ctx)
    torch.testing.assert_close(batch.returns[:, 0], torch.tensor([1.0, 101.0]))
    torch.testing.assert_close(batch.future_return[:, 0], torch.tensor([1000.0, 1100.0]))
    assert not any("return" in name for name in batch.context.features)
    restored = exp._return_batch_from_state(exp._return_batch_state(batch))
    assert exp._return_batch_sha256(restored) == exp._return_batch_sha256(batch)
    assert exp._return_batch_sha256(restored.slice(1)) == exp._return_batch_sha256(batch.slice(1))
    with pytest.raises(ValueError, match="return batch fields"):
        exp._return_batch_from_state(exp._train_batch_state(raw))


def test_zero_initialization_is_neutral_and_preserves_rng() -> None:
    cfg = _tiny_cfg()
    torch.manual_seed(101)
    enabled = exp.GPT(cfg)
    enabled_rng = torch.get_rng_state()
    torch.manual_seed(101)
    disabled = exp.GPT(replace(cfg, return_conditioning=False))
    assert torch.equal(enabled_rng, torch.get_rng_state())
    for name, parameter in enabled.named_parameters():
        assert torch.equal(parameter, dict(disabled.named_parameters())[name])
    conditioner = enabled.temporal.return_conditioner
    state = torch.get_rng_state()
    with torch.random.fork_rng(devices=[]):
        exp.ReturnConditioner(cfg)
    assert torch.equal(state, torch.get_rng_state())
    values = torch.tensor([float("nan"), 0.0, 120.0], requires_grad=True)
    modulation = conditioner(values, torch.tensor([False, True, True]))
    for tensor in modulation:
        assert torch.equal(tensor, torch.zeros_like(tensor))
    sum(tensor.sum() for tensor in modulation).backward()
    assert torch.isfinite(values.grad).all()
    for parameter in conditioner.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    with torch.no_grad():
        for projection in conditioner.projections:
            projection.bias.fill_(0.2)
            projection.weight.fill_(0.1)
    modulation = conditioner(values.detach(), torch.tensor([False, True, True]))
    assert torch.count_nonzero(modulation[0][0]) == 0
    assert torch.count_nonzero(modulation[0][1]) > 0
    disabled.temporal.return_conditioner.load_state_dict(conditioner.state_dict())
    assert (
        torch.count_nonzero(disabled.temporal.return_conditioner(values.detach(), torch.ones(3, dtype=torch.bool))[0])
        == 0
    )


def test_nonzero_conditioned_parallel_stepwise_and_traced_parity() -> None:
    cfg = _tiny_cfg(batch_size=2)
    torch.manual_seed(91)
    model = exp.GPT(cfg).eval()
    with torch.no_grad():
        for projection in model.temporal.return_conditioner.projections:
            projection.weight.normal_(std=0.02)
            projection.bias.normal_(std=0.03)
    hidden = torch.randn(2, 8, 32)
    observed = torch.zeros(2, 1, 4, dtype=torch.long)
    targets = torch.zeros(2, 1, 16, 4, dtype=torch.long)
    values = torch.tensor([[180.0], [float("nan")]])
    present = torch.tensor([[True], [False]])
    parallel = model.temporal.teacher_forced_logits_by_group(
        hidden, torch.tensor([0, 2]), torch.tensor([[7], [7]]), observed, targets, values, present
    )
    stepwise = model.temporal.forced_stepwise_logits(
        hidden, observed[:, 0], targets[:, 0], values[:, 0], present[:, 0], ctx_pad=torch.tensor([0, 2])
    )
    for index, logits in enumerate(stepwise):
        for name in exp.CONTROLLER_GROUP_NAMES:
            torch.testing.assert_close(parallel[name][:, 0, index], logits[name], rtol=2e-5, atol=2e-5)
    engine = exp.BF16Inference(model, cfg, desired_return=180.0, compiled=False)
    context = exp.synthetic_context(cfg, 2, torch.device("cpu"))
    context = replace(context, slot_ids=torch.tensor([1, 3]), reset=torch.ones(2, dtype=torch.bool))
    committed = torch.zeros(2, 2, exp.A_DIM)
    normal = engine.decode(context, 4, streams=exp.SlotGroupRng(1, exp.CONTROLLER_GROUP_NAMES), committed=committed)
    traced = engine.decode_with_trace(
        context, 4, streams=exp.SlotGroupRng(1, exp.CONTROLLER_GROUP_NAMES), committed=committed
    )
    assert torch.equal(normal, traced.actions)


def test_masks_and_partial_calibration_resume_independently() -> None:
    cfg = _tiny_cfg()
    batch = _return_batch(cfg)
    masker = exp.ReturnMasker(17, 0.2)
    global_rng = torch.get_rng_state()
    masker(batch)
    state = masker.state_dict()
    expected = masker(batch)
    restored = exp.ReturnMasker(999, 0.2)
    restored.load_state_dict(state)
    assert torch.equal(expected.condition_present, restored(batch).condition_present)
    assert torch.equal(global_rng, torch.get_rng_state())
    calibration = exp.ReturnCalibration()
    absent = replace(batch, condition_present=torch.zeros_like(batch.available))
    calibration.observe(absent)
    assert calibration.values == [7.0, 15.0]
    saved = calibration.state_dict()
    resumed = exp.ReturnCalibration()
    resumed.load_state_dict(saved)
    resumed.observe(absent)
    calibration.observe(absent)
    assert resumed.state_dict() == calibration.state_dict()
    with pytest.raises(ValueError, match="identity or targets"):
        exp.ReturnCalibration().load_state_dict({**saved, "sha256": "0" * 64})
    with pytest.raises(ValueError, match="incomplete"):
        resumed.targets()


def test_calibration_freezes_exactly_at_consumed_window_limit() -> None:
    cfg = _tiny_cfg(batch_size=257)
    batch = _return_batch(cfg)
    calibration = exp.ReturnCalibration()
    for _ in range(256):
        calibration.observe(batch)
    assert len(calibration.values) == 65_536
    state = calibration.state_dict()
    calibration.observe(replace(batch, future_return=batch.future_return + 1e6))
    assert state == calibration.state_dict()
    assert calibration.targets()[2] == pytest.approx(np.quantile(calibration.values, 0.9, method="linear"))


def test_prefetch_calibrates_only_consumed_batches() -> None:
    cfg = _tiny_cfg()
    cfg = replace(cfg, arch=replace(cfg.arch, L_ctx=128))
    batch = _return_batch(cfg)
    calibration = exp.ReturnCalibration()
    prefetch = exp.DeviceBatchPrefetcher(
        [batch],
        cfg,
        "cpu",
        exp.IdentityMasker(3, 1.0),
        return_masker=exp.ReturnMasker(5, 1.0),
        calibration=calibration,
    )
    try:
        assert calibration.values == []
        consumed, _ = prefetch.next()
        assert calibration.values == [127.0, 255.0]
        assert not consumed.condition_present.any()
        assert consumed.available.all()
        assert not consumed.context.features["ego_player_id"].any()
        prefetch.fill_lookahead(1)
        prefetch.stage_next()
        assert len(calibration.values) == 2
    finally:
        prefetch.close()


def test_production_run_name_fits_filesystem_component_limit() -> None:
    name = exp.make_run_name(
        "059_muon_history_decoder", exp.model_tag(exp.TrainConfig()), "policy-world-v8", "v5-b512"
    )
    assert len(name.encode()) <= 255
    assert "o59v5" in name and "wmax10" in name


class _ResumeReplayAdapter:
    def __init__(self) -> None:
        self.manifests = {"source": replay_loader.SourceManifest("source", (65,))}

    def prepare_shard(self, task: replay_loader.ShardTask) -> None:
        del task

    def _generation(self, row: int, epoch: int, windows: int) -> tuple[str, tuple[dict[str, np.ndarray], ...]]:
        return f"replay-{row}", tuple(
            {"value": np.asarray([epoch * 1000 + row * windows + ordinal], dtype=np.float32)}
            for ordinal in range(windows)
        )

    def decode_chunk(
        self, request: replay_loader._DecodeChunkRequest, task: replay_loader.ShardTask, **options: object
    ) -> replay_loader.DecodedChunk:
        windows = cast(int, options["windows_per_generation"])
        generations = [self._generation(row, request.epoch, windows) for row in request.rows]
        return replay_loader.DecodedChunk(
            request=request,
            task=task,
            replay_ids=tuple(value[0] for value in generations),
            locators=tuple(replay_loader.PhysicalRow(task.source, task.shard, row) for row in request.rows),
            columns={"value": np.stack([[window["value"] for window in value[1]] for value in generations])},
            windows_per_generation=windows,
            generations_per_replay=1,
        )

    def decode_generations(
        self, task: replay_loader.ShardTask, requests: Sequence[tuple[int, int]], **options: object
    ) -> Mapping[tuple[int, int], tuple[str, tuple[dict[str, np.ndarray], ...]]]:
        del task
        return {
            request: self._generation(*request, cast(int, options["windows_per_generation"])) for request in requests
        }


def _resume_labels(row: Mapping[str, object]) -> dict[str, np.ndarray]:
    del row
    return {}


def _resume_collate(
    replay_ids: tuple[str, ...], columns: Mapping[str, np.ndarray], *, cfg: exp.TrainConfig
) -> exp.ReturnBatch:
    batch = _return_batch(cfg)
    values = torch.from_numpy(columns["value"].copy()).expand(-1, cfg.arch.L_ctx).clone()
    batch.context.features["ego_position_x"] = values / 100
    batch.context.features["ego_player_id"].fill_(17)
    available = values.long().remainder(3) != 0
    return replace(
        batch,
        batch=replace(batch.batch, replay_ids=replay_ids),
        returns=values,
        future_return=values + 1,
        available=available,
        condition_present=available.clone(),
    )


def _resume_loader(cfg: exp.TrainConfig) -> replay_loader.PhysicalShardReplayLoader[exp.ReturnBatch]:
    import functools

    selection = exp.PhysicalShardSelection.from_sources((exp.SourceRowSelection("source", 65),))
    return replay_loader.PhysicalShardReplayLoader(
        selection=selection,
        adapter=_ResumeReplayAdapter(),
        tasks=(replay_loader.ShardTask("source", 0, 0, 65),),
        data_protocol=cfg.data_protocol,
        source_manifest_sha256={"source": "a" * 64},
        labels=_resume_labels,
        projection=None,
        batch_transform=functools.partial(_resume_collate, cfg=cfg),
        batch_size=cfg.batch_size,
        replay_slots=64,
        seed=cfg.seed,
        num_workers=0,
        context_length=cfg.arch.L_ctx,
        chunk_length=cfg.arch.sample_chunk_length,
        windows_per_generation=8,
        replay_phase_block_batches=8,
        schema_version=7,
        reserved_disk_bytes=0,
        pin_memory=False,
    )


def test_drained_prefetch_restores_loader_masks_and_optimizer_update(tmp_path: Path) -> None:
    from contextlib import ExitStack

    cfg = _tiny_cfg(batch_size=8, amp_dtype="float32")
    cfg = replace(cfg, arch=replace(cfg.arch, L_ctx=128))
    torch.manual_seed(71)

    def setup(resources: ExitStack, saved: dict[str, Any] | None = None) -> tuple[Any, ...]:
        model = exp.GPT(cfg)
        optimizer = exp.make_optimizer(model, cfg)
        scheduler = exp.LambdaLR(optimizer, exp.lr_schedule(cfg))
        sampler = exp.PrefixSampler(7, "cpu")
        identity = exp.IdentityMasker(11, 0.5)
        masker = exp.ReturnMasker(13, 0.2)
        loader = _resume_loader(cfg)
        resources.callback(loader.close)
        if saved is not None:
            loader.load_state_dict(saved["loader"])
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            sampler.load_state_dict(saved["prefix"])
            identity.load_state_dict(saved["identity"])
            masker.load_state_dict(saved["return_masker"])
            model.return_calibration.load_state_dict(saved["calibration"])
            exp.restore_rng(saved["rng"])
        prefetch = exp.DeviceBatchPrefetcher(
            loader, cfg, "cpu", identity, return_masker=masker, calibration=model.return_calibration
        )
        resources.callback(prefetch.close)
        return model, optimizer, scheduler, sampler, identity, masker, loader, prefetch

    def update(state: tuple[Any, ...], batch: exp.ReturnBatch, index: int) -> dict[str, object]:
        model, optimizer, scheduler, sampler, *_ = state
        result = exp.train_step(
            model,
            batch,
            cfg,
            step=4096 + index,
            update=index + 1,
            valid_prefixes=8 * 32,
            trunk_fn=model.forward,
            temporal_fn=model.temporal.teacher_forced_nll_with_diagnostics,
            optimizer=optimizer,
            scheduler=scheduler,
            prefix_sampler=sampler,
            prefix_validated_on_cpu=True,
        )
        return _step_snapshot(model, result, sampler)

    with ExitStack() as resources:
        original = setup(resources)
        model, optimizer, scheduler, sampler, identity, masker, loader, prefetch = original
        for index in range(5):
            batch, _ = prefetch.next()
            update(original, batch, index)
            if index < 4:
                prefetch.fill_lookahead(1)
                prefetch.stage_next()
        assert prefetch.drained
        path = tmp_path / "resume.pt"
        torch.save(
            {
                "loader": loader.state_dict(),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "prefix": sampler.state_dict(),
                "identity": identity.state_dict(),
                "return_masker": masker.state_dict(),
                "calibration": model.return_calibration.state_dict(),
                "rng": exp.rng_state(),
            },
            path,
        )
        prefetch.fill_lookahead(1)
        prefetch.stage_next()
        expected_batch, _ = prefetch.next()
        expected = update(original, expected_batch, 5)
        restored = setup(resources, torch.load(path, weights_only=False))
        actual_batch, _ = restored[-1].next()
        assert exp._return_batch_sha256(actual_batch) == exp._return_batch_sha256(expected_batch)
        actual = update(restored, actual_batch, 5)
        _assert_nested_equal(expected, actual)
        assert restored[-1].drained


def test_conditioned_and_unconditioned_evidence_cannot_share_directory(tmp_path: Path) -> None:
    import json

    cfg = _tiny_cfg()
    model = exp.GPT(cfg)
    protocol = exp._eval_protocol(
        cfg,
        model,
        n_matchups=1,
        checkpoint_sha256="a" * 64,
        desired_return=75.0,
        return_calibration_sha256="b" * 64,
    )
    exp._validate_eval_directory(tmp_path, protocol)
    exp._write_eval_evidence(tmp_path, [], {}, protocol)
    exp._validate_eval_directory(tmp_path, protocol)
    with pytest.raises(ValueError, match="different protocol"):
        exp._validate_eval_directory(tmp_path, replace(protocol, return_target="unconditioned", desired_return=None))
    saved = json.loads((tmp_path / "match_rows.json").read_text())
    assert saved["protocol"]["desired_return"] == 75.0
    assert saved["protocol"]["return_calibration_sha256"] == "b" * 64
    (tmp_path / "match_rows.json").write_text(json.dumps({**saved, "schema_version": 6}))
    with pytest.raises(ValueError, match="different protocol"):
        exp._validate_eval_directory(tmp_path, protocol)


def test_serving_decoder_matches_frozen_experiment() -> None:
    cfg = _tiny_cfg(batch_size=1)
    reference = exp.GPT(cfg).eval()
    serving = ServingGPT(cfg).eval()
    serving.load_state_dict(reference.state_dict(), strict=True)
    context = exp.synthetic_context(cfg, 1, torch.device("cpu"))
    observed = reference.codec.quantize(stack_actions(context.features))
    forced = reference.codec.quantize(torch.as_tensor(NEUTRAL_ACTION).reshape(1, 1, -1).expand(1, 2, -1))
    hidden = torch.randn(1, cfg.arch.L_ctx, cfg.arch.d_model)
    uniforms = torch.rand(4, len(exp.CONTROLLER_GROUP_NAMES), 1)
    target = torch.tensor([20.0])
    present = torch.tensor([True])

    with torch.inference_mode():
        torch.testing.assert_close(
            reference.context_tokens(context.features, observed),
            serving.context_tokens(context.features, observed),
            rtol=0,
            atol=0,
        )
        for temperature in (0.8, 1.0, 1.1):
            args = (hidden, observed[:, -1], reference.head_offsets[:4], target, present)
            expected = reference.temporal.sample_indices(
                *args,
                argmax=False,
                uniforms=uniforms,
                temperature=temperature,
                ctx_pad=context.ctx_pad,
                forced_prefix=forced,
            )
            actual = serving.temporal.sample_indices(
                *args,
                argmax=False,
                uniforms=uniforms,
                temperature=torch.tensor(temperature),
                ctx_pad=context.ctx_pad,
                forced_prefix=forced,
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
