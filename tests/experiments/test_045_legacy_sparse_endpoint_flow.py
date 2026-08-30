"""Focused contracts for O45 legacy endpoint flow and inference-time RTC."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from hal.sim.vec import Slot
from hal.training.closed_loop import _SlotState
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import Context
from hal.training.features import TrainBatch


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ROOT = Path(__file__).resolve().parents[2]
exp = _load("test_exp045", _ROOT / "experiments" / "045_legacy_sparse_endpoint_flow.py")
exp038 = _load("test_exp045_parent_038", _ROOT / "experiments" / "038_sparse_endpoint_flow.py")
exp043 = _load("test_exp045_codec_043", _ROOT / "experiments" / "043_legacy_codec.py")


def _cfg(**overrides):
    values = {
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 4,
        "L_ctx": 4,
        "training_prefixes": 4,
        "flow_d_model": 64,
        "flow_layers": 1,
        "flow_heads": 1,
        "flow_ff_dim": 64,
        "flow_time_embed_dim": 16,
        "flow_context_dim": 32,
        "flow_condition_dim": 32,
        "batch_size": 2,
        "reservoir_capacity": 4,
        "warmup_steps": 1,
        "max_steps": 2,
        "compile_trunk": False,
        "compile_flow": False,
        "num_workers": 0,
        "push_to_r2": False,
        "inference_mode": "eager",
        "latency_iterations": 0,
        "validation_solver_contexts": 2,
        "validation_diversity_contexts": 2,
        "validation_noise_samples": 2,
    }
    return exp.TrainConfig(**{**values, **overrides})


def _actions(batch: int, length: int, generator: torch.Generator) -> torch.Tensor:
    actions = torch.empty(batch, length, A_DIM)
    actions[..., :4] = torch.rand(batch, length, 4, generator=generator) * 2 - 1
    actions[..., 4:6] = torch.rand(batch, length, 2, generator=generator)
    actions[..., 6:] = torch.randint(0, 2, (batch, length, 8), generator=generator).float()
    return actions


def _batch(cfg, *, seed: int = 0) -> TrainBatch:
    generator = torch.Generator().manual_seed(seed)
    synthetic = exp.synthetic_context(cfg, cfg.batch_size, torch.device("cpu"))
    features = dict(synthetic.features)
    history = _actions(cfg.batch_size, cfg.L_ctx, generator)
    for channel, values in zip(ACTION_CHANNELS, history.unbind(-1), strict=True):
        features[f"ego_{channel}"] = values
    context = Context(features=features, ctx_pad=torch.zeros(cfg.batch_size, dtype=torch.long))
    return TrainBatch(
        context=context,
        target=_actions(cfg.batch_size, cfg.sample_chunk_length, generator),
    )


def test_primary_configuration_is_exact_and_rejects_drift() -> None:
    cfg = exp.TrainConfig()
    exp.validate_production_config(cfg)
    assert exp.GROUP_VOCABS == (6, 37, 9, 5)
    assert exp.FLOW_DIM == 57
    assert cfg.head_offsets == (1, 2, 3, 4, 5, 6, 9, 12, 16, 20)
    assert cfg.training_prefixes == 32
    assert (cfg.flow_d_model, cfg.flow_layers, cfg.flow_heads, cfg.flow_ff_dim) == (192, 3, 3, 512)
    assert (cfg.execution_stride, cfg.committed_frames, cfg.flow_nfe, cfg.flow_epsilon) == (4, 0, 4, 1e-3)
    assert cfg.codec_version == 2
    assert exp.production_run_name(cfg) == "045-legacy-endpoint-flow-sparse10-nfe4-p4-s4-d0-seed0"
    model = exp.GPT(cfg)
    assert sum(parameter.numel() for parameter in model.parameters()) == 8_865_957
    assert exp.decoder_mac_estimate(cfg)["flow_macs_per_nfe"] == 15_395_888
    assert exp.inference_flops_per_replan(model, cfg) == 1_824_481_664
    with pytest.raises(ValueError, match="production experiment configuration mismatch"):
        exp.validate_production_config(exp.TrainConfig(flow_nfe=8))


def test_dense_16_variant_is_p16_d4_s12_and_keeps_parameter_count() -> None:
    cfg = exp.dense_16_config(exp.TrainConfig())
    exp.validate_production_config(cfg)
    assert cfg.head_offsets == tuple(range(1, 17))
    assert (cfg.execution_stride + cfg.committed_frames, cfg.execution_stride, cfg.committed_frames) == (16, 12, 4)
    assert exp.evaluation_modes(cfg) == ((12, 4),)
    assert sum(parameter.numel() for parameter in exp.GPT(cfg).parameters()) == 8_865_957

    small = exp.dense_16_config(_cfg())
    model = exp.GPT(small)
    _, targets, _ = exp.prepared_targets(model, _batch(small), n_prefixes=small.training_prefixes)
    assert targets.shape == (small.batch_size, small.training_prefixes, 16, exp.N_GROUPS)


def test_sparse_runtime_rejects_d_greater_than_six_minus_s() -> None:
    cfg = _cfg()
    with pytest.raises(ValueError, match=r"D <= 6 - S=2"):
        exp.runtime_prediction_frames(cfg, 4, 3)


def test_observation_trunk_keeps_o38_shapes() -> None:
    cfg = _cfg()
    parent_cfg = exp038.TrainConfig(
        **{name: value for name, value in vars(cfg).items() if hasattr(exp038.TrainConfig, name)}
    )
    torch.manual_seed(17)
    model = exp.GPT(cfg)
    torch.manual_seed(17)
    parent = exp038.GPT(parent_cfg)
    prefixes = ("cat_embeds.", "v6_cat_embeds.", "char_emb.", "stage_emb.", "ctx_proj.", "trunk.")
    ours = {name: value for name, value in model.state_dict().items() if name.startswith(prefixes)}
    theirs = {name: value for name, value in parent.state_dict().items() if name.startswith(prefixes)}
    assert ours.keys() == theirs.keys()
    for name in ours:
        assert ours[name].shape == theirs[name].shape, name


def test_codec_matches_o43_centers_golden_hashes_and_complete_grid() -> None:
    codec = exp.StructuredControllerCodec(16)
    reference = exp043.StructuredControllerCodec(16)
    assert torch.equal(codec.main_centers, reference.main_centers[:37])
    assert torch.equal(codec.c_centers, reference.c_centers)
    assert torch.equal(codec.trigger_centers, reference.trigger_centers[:5])

    grid = torch.cartesian_prod(*(torch.arange(size) for size in exp.GROUP_VOCABS))
    restored = codec.quantize(codec.dequantize(grid[:, None]))[:, 0]
    assert torch.equal(restored, grid)

    axis = torch.arange(-80, 81, dtype=torch.float32) / 80
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack((x.flatten(), y.flatten()), dim=-1)
    actions = torch.zeros(len(points), 1, A_DIM)
    actions[:, 0, :2] = points
    main = codec.quantize(actions)[:, 0, exp.MAIN_G].to(torch.uint8).numpy()
    assert hashlib.sha256(main.tobytes()).hexdigest() == (
        "4034586852c41655a0de9c2c4aafea711ad326196f2e26bf4cf25519d6846018"
    )
    actions.zero_()
    actions[:, 0, 2:4] = points
    c_stick = codec.quantize(actions)[:, 0, exp.C_G].to(torch.uint8).numpy()
    assert hashlib.sha256(c_stick.tobytes()).hexdigest() == (
        "2733621eee35975a28b734a331f8cc82fb4bf171cecd9a5be74d977784762dd8"
    )
    actions = torch.zeros(141, 1, A_DIM)
    actions[:, 0, exp.TRIGGER_L_CH] = torch.arange(141, dtype=torch.float32) / 140
    trigger = codec.quantize(actions)[:, 0, exp.TRIG_G].to(torch.uint8).numpy()
    assert hashlib.sha256(trigger.tobytes()).hexdigest() == (
        "d53bbb5472b9e9e5e3b6641e565ee628d0061fbe6047f4d11344af9ec7356390"
    )


def test_codec_matches_o43_release_and_fused_shoulder_semantics() -> None:
    actions = torch.zeros(1, 5, A_DIM)
    a = ACTION_CHANNELS.index("button_a")
    b = ACTION_CHANNELS.index("button_b")
    actions[0, :, b] = 1
    actions[0, 2:4, a] = 1
    assert exp.legacy_button_classes(actions).tolist() == [[1, 1, 0, 0, 5]]
    torch.testing.assert_close(exp.legacy_button_classes(actions), exp043.legacy_button_classes(actions))

    shoulders = torch.zeros(1, 3, A_DIM)
    shoulders[0, 0, exp.TRIGGER_R_CH] = 0.39
    shoulders[0, 1, exp.TRIGGER_R_CH] = 0.41
    shoulders[0, 2, exp.BUTTON_R_CH] = 1
    codec = exp.StructuredControllerCodec(16)
    indices = codec.quantize(shoulders)
    assert indices[0, :, exp.TRIG_G].tolist() == [1, 1, 0]
    assert indices[0, :, exp.BUTTONS_G].tolist() == [5, 5, 4]
    decoded = codec.dequantize(indices)
    assert decoded[0, :, exp.TRIGGER_L_CH].tolist() == pytest.approx([0.35, 0.35, 0.0])
    assert not decoded[..., exp.TRIGGER_R_CH].any()
    assert decoded[0, :, exp.BUTTON_L_CH].tolist() == [0.0, 0.0, 1.0]
    assert not decoded[..., exp.BUTTON_R_CH].any()


def test_stratified_prefixes_cover_regions_and_always_include_final() -> None:
    pad = torch.tensor([0, 9])
    one = exp.stratified_prefix_indices(pad, 128, 32, generator=torch.Generator().manual_seed(3))
    two = exp.stratified_prefix_indices(pad, 128, 32, generator=torch.Generator().manual_seed(4))
    assert one.shape == (2, 32)
    assert torch.equal(one[:, -1], torch.tensor([127, 127]))
    assert (one >= pad[:, None]).all()
    assert (one[:, :-1] < 127).all()
    assert not torch.equal(one[:, :-1], two[:, :-1])
    available = 127 - pad
    region_ids = torch.arange(31)
    lower = pad[:, None] + available[:, None] * region_ids // 31
    upper = pad[:, None] + available[:, None] * (region_ids + 1) // 31
    assert ((one[:, :-1] >= lower) & (one[:, :-1] < upper)).all()
    cold_start = exp.stratified_prefix_indices(torch.tensor([127]), 128, 32)
    assert cold_start.shape == (1, 32)
    assert torch.equal(cold_start, torch.full((1, 32), 127))


def test_named_flow_state_projection_and_heads_have_exact_shapes() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg)
    rows = 3
    noisy = exp.gaussian_flow_state(rows, len(cfg.head_offsets), device=torch.device("cpu"))
    context_h = torch.randn(rows, cfg.d_model)
    tau = torch.rand(rows)
    logits, gates = model.flow(noisy, context_h, tau)
    assert {name: tuple(value.shape) for name, value in logits.items()} == {
        name: (rows, 10, exp.GROUP_VOCABS[group]) for group, name in enumerate(exp.GROUP_NAMES)
    }
    assert gates.shape == (cfg.flow_layers, 2)
    assert model.flow.group_projections["buttons"].out_features == 64
    assert model.flow.group_projections["main_stick"].out_features == 48
    assert model.flow.group_projections["c_stick"].out_features == 16
    assert model.flow.group_projections["triggers"].out_features == 24


def test_flow_interpolation_uses_one_shared_tau_per_prefix() -> None:
    targets = torch.stack(
        [
            torch.randint(vocab, (2, 10), generator=torch.Generator().manual_seed(group + 1))
            for group, vocab in enumerate(exp.GROUP_VOCABS)
        ],
        dim=-1,
    )
    tau = torch.tensor([0.0, 0.75])
    noisy, noise = exp.noisy_flow_state(targets, tau, generator=torch.Generator().manual_seed(8))
    clean = exp.categorical_endpoint(targets)
    for name in exp.GROUP_NAMES:
        expected = (1.0 - tau[:, None, None]) * noise[name] + tau[:, None, None] * clean[name]
        torch.testing.assert_close(noisy[name], expected)
        torch.testing.assert_close(noisy[name][0], noise[name][0])


def test_adaln_zero_initializes_identity_and_attention_is_bidirectional() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg).eval()
    block = model.flow.blocks[0]
    assert torch.count_nonzero(block.modulation_projection.weight) == 0
    assert torch.count_nonzero(block.modulation_projection.bias) == 0
    d_model = cfg.flow_d_model
    with torch.no_grad():
        block.modulation_projection.bias[2 * d_model : 3 * d_model] = 1.0
    noisy = exp.gaussian_flow_state(1, 10, device=torch.device("cpu"), generator=torch.Generator().manual_seed(1))
    changed = {name: value.clone() for name, value in noisy.items()}
    changed["buttons"][:, 0] += 1.0
    context_h = torch.randn(1, cfg.d_model)
    tau = torch.tensor([0.5])
    before, _ = model.flow(noisy, context_h, tau)
    after, _ = model.flow(changed, context_h, tau)
    assert not torch.allclose(before["buttons"][:, -1], after["buttons"][:, -1])


def test_endpoint_objective_is_equal_group_mean_and_reaches_trunk() -> None:
    cfg = _cfg()
    torch.manual_seed(0)
    model = exp.GPT(cfg)
    with torch.no_grad():
        for block in model.flow.blocks:
            block.modulation_projection.weight.normal_(std=0.01)
    batch = _batch(cfg)
    loss, nll, tau, targets, logits, _ = exp.microbatch_loss(
        model,
        batch,
        cfg,
        trunk_fn=lambda features, pad, actions: model(features, pad, actions),
        flow_fn=model.flow,
        generator=torch.Generator().manual_seed(7),
    )
    torch.testing.assert_close(loss, nll.mean(dim=(0, 1)).mean())
    assert tau.shape == (cfg.batch_size * cfg.training_prefixes,)
    assert targets.shape == (cfg.batch_size * cfg.training_prefixes, 10, exp.N_GROUPS)
    assert logits["buttons"].shape[-1] == 6
    loss.backward()
    assert exp._mean_gradient_norm(model.trunk.parameters()) > 0
    assert exp._mean_gradient_norm(model.flow.parameters()) > 0


def test_solver_uses_exact_left_endpoint_nfe_and_keeps_state_fp32(monkeypatch) -> None:
    cfg = _cfg(flow_nfe=4)
    expert = exp.CategoricalEndpointFlow(cfg)
    times = []
    dtypes = []

    def fake_forward(self, noisy, context_h, tau):
        del context_h
        times.append(float(tau[0]))
        dtypes.append({name: value.dtype for name, value in noisy.items()})
        logits = {name: torch.zeros(noisy[name].shape, dtype=torch.bfloat16) for name in exp.GROUP_NAMES}
        return logits, torch.zeros(cfg.flow_layers, 2)

    monkeypatch.setattr(exp.CategoricalEndpointFlow, "forward", fake_forward)
    context_h = torch.randn(2, cfg.d_model)
    noise = {
        name: torch.randn(2, 10, exp.GROUP_VOCABS[group], dtype=torch.float16)
        for group, name in enumerate(exp.GROUP_NAMES)
    }
    logits, state = expert.solve(context_h, noise)
    assert times == pytest.approx([0.0, 0.333, 0.666, 0.999], abs=1e-6)
    assert all(dtype == torch.float32 for call in dtypes for dtype in call.values())
    assert all(value.dtype == torch.float32 for value in state.values())
    assert all(value.dtype == torch.bfloat16 for value in logits.values())


def test_rtc_clamps_categorical_prefix_at_every_nfe(monkeypatch) -> None:
    cfg = _cfg(flow_nfe=4)
    expert = exp.CategoricalEndpointFlow(cfg)
    committed = torch.tensor([[[1, 2, 3, 4]], [[5, 6, 7, 2]]])
    seen = []
    original = expert.forward

    def record(noisy, context_h, tau):
        seen.append({name: value[:, :1].clone() for name, value in noisy.items()})
        return original(noisy, context_h, tau)

    monkeypatch.setattr(expert, "forward", record)
    context_h = torch.randn(2, cfg.d_model)
    noise = exp.gaussian_flow_state(2, 10, device=torch.device("cpu"))
    _, state = expert.solve(context_h, noise, committed=committed)
    expected = exp.categorical_endpoint(committed)
    assert len(seen) == 4
    for values in seen:
        for name in exp.GROUP_NAMES:
            torch.testing.assert_close(values[name], expected[name])
    for name in exp.GROUP_NAMES:
        torch.testing.assert_close(state[name][:, :1], expected[name])


def test_eager_inference_runs_exactly_four_expert_calls_and_returns_h4(monkeypatch) -> None:
    cfg = _cfg()
    model = exp.GPT(cfg).eval()
    context = exp.synthetic_context(cfg, 2, torch.device("cpu"))
    calls = 0
    original = model.flow.forward

    def counted_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model.flow, "forward", counted_forward)
    inference = exp.BF16Inference(model, cfg, compiled=False, bucket=2)
    actions = inference.decode(context, 4, gen=torch.Generator().manual_seed(11))
    assert calls == 4
    assert actions.shape == (2, 4, A_DIM)
    assert torch.isfinite(actions).all()


def test_h4_d0_matches_ordinary_unconditioned_solver() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg).eval()
    context = exp.synthetic_context(cfg, 2, torch.device("cpu"))
    inference = exp.BF16Inference(model, cfg, compiled=False, bucket=2)
    seed = 19
    actual = inference.decode(context, 4, gen=torch.Generator().manual_seed(seed))

    canonical = exp.canonical_context(context, cfg.observation_bundle)
    observed = model.codec.quantize(exp.stack_actions(canonical.features))
    hidden = model(canonical.features, canonical.ctx_pad, observed)
    noise = exp.gaussian_flow_state(
        2,
        len(cfg.head_offsets),
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(seed),
    )
    logits, _ = model.flow.solve(hidden[:, -1], noise, nfe=cfg.flow_nfe)
    indices = model.flow.argmax_indices(logits)
    expected = model.codec.canonicalize(model.codec.dequantize(indices[:, :4]))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_rtc_returns_exact_raw_prefix_and_caches_by_mode() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg).eval()
    context = exp.synthetic_context(cfg, 2, torch.device("cpu"))
    inference = exp.BF16Inference(model, cfg, compiled=False, bucket=2)
    bootstrap = inference.decode(
        context,
        5,
        execution_stride=4,
        committed_frames=1,
        gen=torch.Generator().manual_seed(3),
    )
    assert bootstrap.shape == (2, 5, A_DIM)
    committed = model.codec.dequantize(torch.tensor([[[1, 2, 3, 4]], [[4, 7, 8, 2]]])).numpy()
    result = inference.decode(
        context,
        5,
        execution_stride=4,
        committed=committed,
        gen=torch.Generator().manual_seed(4),
    )
    assert result.shape == (2, 5, A_DIM)
    torch.testing.assert_close(result[:, :1], torch.from_numpy(committed), rtol=0, atol=0)
    inference.decode(context, 4, execution_stride=4, gen=torch.Generator().manual_seed(4))
    assert set(inference._decoders) == {(2, 0), (2, 1)}


def test_flow_noise_stream_is_keyed_by_slot_not_batch_order() -> None:
    def context(slot_ids):
        return Context(
            features={},
            ctx_pad=torch.zeros(len(slot_ids), dtype=torch.long),
            slot_ids=torch.tensor(slot_ids),
            reset=torch.ones(len(slot_ids), dtype=torch.bool),
        )

    ordered = exp.SlotFlowRandom(9)
    ordered.begin(context([11, 22]))
    whole = ordered.noise(10)

    reversed_rows = exp.SlotFlowRandom(9)
    reversed_rows.begin(context([22, 11]))
    reversed_noise = reversed_rows.noise(10)
    for name in exp.GROUP_NAMES:
        torch.testing.assert_close(whole[name], reversed_noise[name].flip(0), rtol=0, atol=0)


def test_runtime_contracts_handoff_reset_and_rng_advance_by_stride() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg)

    class Inference:
        pass

    h4 = exp.make_policy(model, {}, cfg, execution_stride=4, committed_frames=1, inference=Inference(), device="cpu")
    h1 = exp.make_policy(model, {}, cfg, execution_stride=1, committed_frames=1, inference=Inference(), device="cpu")
    assert (
        h4.runtime_spec.prediction_frames,
        h4.runtime_spec.execution_stride,
        h4.runtime_spec.committed_frames,
    ) == (5, 4, 1)
    assert (
        h1.runtime_spec.prediction_frames,
        h1.runtime_spec.execution_stride,
        h1.runtime_spec.committed_frames,
    ) == (2, 1, 1)

    slot = Slot(3, 1)
    plan = np.arange(5 * A_DIM, dtype=np.float32).reshape(5, A_DIM)
    h4._slots[slot] = _SlotState(pending=plan)
    np.testing.assert_array_equal(h4._committed([slot]), plan[None, 4:5])

    def context(reset: bool) -> Context:
        return Context(
            features={},
            ctx_pad=torch.zeros(1, dtype=torch.long),
            slot_ids=torch.tensor([25]),
            reset=torch.tensor([reset]),
        )

    stream = exp.SlotFlowRandom(7)
    stream.begin(context(True))
    stream.advance(4)
    assert stream.state() == ((25, 0, 4),)
    stream.begin(context(False))
    assert stream.base_frames == (4,)
    stream.begin(context(True))
    assert stream.base_frames == (0,)
    assert stream.state() == ((25, 0, 4), (25, 1, 0))


def test_validation_reports_required_flow_diagnostics() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg)
    values = exp.val_metrics(model, [_batch(cfg)], cfg)
    required = {
        "flow/loss_buttons",
        "flow/loss_main_stick",
        "flow/loss_c_stick",
        "flow/loss_triggers",
        "flow/ce_o20",
        "context/shuffled_delta_ce",
        "noise/unique_first_four_plans",
        "noise/first_four_hamming",
        "noise/all_offsets_hamming",
        "solver/nfe4_ce",
        "solver/nfe8_ce",
        "solver/nfe16_ce",
        "temporal/invalid_trigger_button_rate_before_repair",
    }
    assert required <= values.keys()
    assert all(torch.isfinite(torch.tensor(values[name])) for name in required)


def test_evaluator_parallelism_never_exceeds_available_cpus(monkeypatch) -> None:
    cfg = _cfg(eval_max_parallel=32)
    monkeypatch.setattr(exp, "usable_cpus", lambda: 16)
    assert exp._eval_parallelism(cfg, 96) == 16
    monkeypatch.setattr(exp, "usable_cpus", lambda: 10)
    assert exp._eval_parallelism(cfg, 96) == 8


def test_parent_latency_harness_reports_nfe_and_p99() -> None:
    cfg = _cfg()
    values = exp.benchmark_model(exp.GPT(cfg), cfg, iterations=2, rows=1)
    required = {
        "h4_d0/replan_p50_ms",
        "h4_d1/replan_p95_ms",
        "h1_d1/replan_p99_ms",
        "h4_d0/model_decode_p50_ms",
        "h4_d1/amortized_ms_per_executed_frame",
        "h1_d1/action_expert_flops_per_nfe",
        "h4_d0/action_expert_flops_per_replan",
    }
    assert required <= values.keys()
    assert values["h4_d0/nfe"] == 4
    assert values["h4_d1/decoder_calls_per_replan"] == 4


def test_checkpoint_protocol_round_trip(tmp_path: Path) -> None:
    cfg = _cfg()
    restored = exp.config_from_state(exp._checkpoint_config(cfg))
    assert restored == cfg
    model = exp.GPT(cfg)
    protocol = exp._eval_protocol(
        cfg,
        model,
        n_matchups=96,
        execution_stride=4,
        committed_frames=1,
        checkpoint_sha256="a" * 64,
    )
    exp._write_eval_evidence(tmp_path, [], {"net_stock_lcb": 0.0}, protocol)
    payload = json.loads((tmp_path / "metrics.json").read_text())
    assert payload["protocol"]["nfe"] == 4
    assert payload["protocol"]["future_offsets"] == list(cfg.head_offsets)
    assert payload["protocol"]["endpoint_decode"] == "argmax"
    assert payload["protocol"]["prediction_frames"] == 5
    assert payload["protocol"]["execution_stride"] == 4
    assert payload["protocol"]["committed_frames"] == 1
    assert payload["protocol"]["codec_version"] == 2
    assert payload["protocol"]["vocabularies"] == [6, 37, 9, 5]
    assert payload["protocol"]["hard_categorical_prefix_clamp"] is True


def test_small_training_runs_one_final_closed_loop_eval(monkeypatch, tmp_path: Path) -> None:
    cfg = _cfg(max_steps=1, val_every=0, ckpt_every=0, wandb_log_code=False)
    batch = _batch(cfg)
    monkeypatch.setattr(exp, "_make_loaders", lambda cfg, stats: ([batch], [batch]))
    monkeypatch.setattr(exp, "setup_run_dir", lambda name: (tmp_path, tmp_path / "replays"))
    monkeypatch.setattr(exp, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp, "_checkpoint_sha256", lambda path: "a" * 64)
    monkeypatch.setattr(exp, "val_metrics", lambda model, batches, cfg: {"loss": 0.0})
    monkeypatch.setattr(exp, "BF16Inference", lambda model, cfg: object())
    evaluations = []

    def fake_eval(model, stats, cfg, *, execution_stride, committed_frames, **kwargs):
        del model, stats, kwargs
        evaluations.append((execution_stride, committed_frames))
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
    exp.train(cfg, {}, requested_run_name="tiny-045")
    assert evaluations == [(4, 0), (4, 1), (1, 1)]
    assert any("train/trunk_grad_norm" in values for values in logs)
    assert any("eval/net_stock_lcb" in values and "eval_h4_d0/net_stock_lcb" in values for values in logs)
    assert any("eval_h4_d1/net_stock_lcb" in values for values in logs)
    assert any("eval_h1_d1/net_stock_lcb" in values for values in logs)


def test_experiment_source_is_self_contained() -> None:
    source = (_ROOT / "experiments" / "045_legacy_sparse_endpoint_flow.py").read_text()
    assert "spec_from_file_location" not in source
    assert "experiments/038" not in source
