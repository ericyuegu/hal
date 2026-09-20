"""Focused contracts for O59's sampled wide history decoder."""

import importlib.util
import random
import sys
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch


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
        "temporal_ff_dim": 64,
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


def test_prefix_sampling_is_distinct_uniform_and_exactly_resumable() -> None:
    sampler = exp.PrefixSampler(7, "cpu")
    ctx_pad = torch.tensor([0, 128])
    first = sampler.sample(ctx_pad, length=256, suffix_start=128)
    state = sampler.state_dict()
    expected_next = sampler.sample(ctx_pad, length=256, suffix_start=128)

    restored = exp.PrefixSampler(999, "cpu")
    restored.load_state_dict(state)
    actual_next = restored.sample(ctx_pad, length=256, suffix_start=128)

    assert first.shape == (2, 32)
    assert torch.all(first[:, 1:] > first[:, :-1])
    assert int(first.min()) >= 128 and int(first.max()) < 256
    assert torch.equal(actual_next, expected_next)


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
    decoder = exp.CausalTemporalDecoder(cfg, exp.DiscreteControllerCodec(cfg.arch.action_embed_dim))
    hidden = torch.randn(2, 8, 32, requires_grad=True)
    ctx_pad = torch.tensor([0, 2])
    prefixes = torch.tensor([[4, 7], [4, 7]])
    observed = torch.zeros(2, 2, exp.CONTROLLER_GROUP_COUNT, dtype=torch.long)
    targets = torch.zeros(2, 2, 16, exp.CONTROLLER_GROUP_COUNT, dtype=torch.long)

    output = decoder.teacher_forced_states(hidden, ctx_pad, prefixes, observed, targets)
    output[:, 0].sum().backward()
    assert hidden.grad is not None
    assert torch.equal(hidden.grad[0, 5:], torch.zeros_like(hidden.grad[0, 5:]))
    assert torch.equal(hidden.grad[1, :2], torch.zeros_like(hidden.grad[1, :2]))

    changed = hidden.detach().clone()
    changed[:, 5:] += 100
    isolated = decoder.teacher_forced_states(changed, ctx_pad, prefixes, observed, targets)
    torch.testing.assert_close(output[:, 0], isolated[:, 0])


def test_teacher_forced_and_stepwise_decoding_match() -> None:
    torch.manual_seed(5)
    cfg = _tiny_cfg(batch_size=1)
    decoder = exp.CausalTemporalDecoder(cfg, exp.DiscreteControllerCodec(cfg.arch.action_embed_dim)).eval()
    hidden = torch.randn(1, 8, 32)
    ctx_pad = torch.tensor([2])
    prefix = torch.tensor([[7]])
    observed = torch.zeros(1, 1, exp.CONTROLLER_GROUP_COUNT, dtype=torch.long)
    targets = torch.zeros(1, 1, 16, exp.CONTROLLER_GROUP_COUNT, dtype=torch.long)

    batched = decoder.teacher_forced_logits_by_group(hidden, ctx_pad, prefix, observed, targets)
    stepwise = decoder.forced_stepwise_logits(hidden, observed[:, 0], targets[:, 0], ctx_pad=ctx_pad)
    for depth in range(16):
        for name in exp.CONTROLLER_GROUP_NAMES:
            torch.testing.assert_close(batched[name][:, 0, depth], stepwise[depth][name], atol=2e-5, rtol=2e-5)


def test_live_decode_preserves_committed_action_prefix() -> None:
    torch.manual_seed(11)
    cfg = _tiny_cfg(batch_size=1)
    decoder = exp.CausalTemporalDecoder(cfg, exp.DiscreteControllerCodec(cfg.arch.action_embed_dim)).eval()
    hidden = torch.randn(1, 8, 32)
    observed = torch.zeros(1, exp.CONTROLLER_GROUP_COUNT, dtype=torch.long)
    committed = torch.zeros(1, 2, exp.CONTROLLER_GROUP_COUNT, dtype=torch.long)
    committed[:, 0, exp.TRIGGERS_GROUP] = 1

    decoded = decoder.sample_indices(
        hidden,
        observed,
        tuple(range(1, 5)),
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
    engine = exp.BF16Inference(model, cfg, compiled=True, bucket=4)
    context = exp.synthetic_context(cfg, 3, torch.device("cuda"))
    for delay in (0, 1, 2, 3):
        engine.prewarm(3, 4, committed_frames=delay)
        with torch.compiler.set_stance("fail_on_recompile"):
            engine.decode(context, 4, committed=torch.zeros(3, delay, len(exp.ACTION_CHANNELS), device="cuda"))
        assert engine.prewarm(3, 4, committed_frames=delay) == 0


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


def test_scalar_download_preserves_order_and_cpu_values() -> None:
    metrics = {
        "z": torch.tensor(3, dtype=torch.int64),
        "a": torch.tensor(0.125, requires_grad=True),
        "b": torch.tensor(True),
    }
    actual = exp._download_scalar_metrics(metrics, 7)
    assert list(actual) == list(metrics)
    assert actual == {"z": 3.0, "a": 0.125, "b": 1.0}
    assert exp._download_scalar_metrics({}, 7) == {}


@pytest.mark.parametrize(
    "value", [torch.zeros(1), torch.zeros(2), torch.tensor(float("nan")), torch.tensor(float("inf"))]
)
def test_scalar_download_rejects_invalid_metrics(value: torch.Tensor) -> None:
    with pytest.raises(FloatingPointError, match="update 13"):
        exp._download_scalar_metrics({"good": torch.tensor(1.0), "bad": value}, 13)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for mixed-device scalar transfers")
def test_scalar_download_handles_mixed_devices() -> None:
    metrics = {
        "cpu": torch.tensor(1.0),
        "gpu": torch.tensor(2.0, device="cuda"),
        "cpu2": torch.tensor(3.0),
        "gpu2": torch.tensor(4.0, device="cuda"),
    }
    assert exp._download_scalar_metrics(metrics, 1) == {"cpu": 1.0, "gpu": 2.0, "cpu2": 3.0, "gpu2": 4.0}
    assert metrics["cpu"].device.type == "cpu"


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
