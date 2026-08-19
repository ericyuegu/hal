"""Integrity contracts for the width by decoder factorial."""

import importlib.util
import shutil
import sys
import time
from dataclasses import asdict
from dataclasses import fields
from pathlib import Path
from typing import cast

import pytest
import torch

from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import Context
from hal.training.features import TrainBatch

_EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _EXPERIMENTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exp026 = _load("test_exp026_for_037", "026_temporal_mtp.py")
exp = _load("test_exp037", "037_width_decoder_factorial.py")


def _cfg(cell: exp.FactorialCell, **overrides) -> exp.TrainConfig:
    values = {
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 4,
        "L_ctx": 4,
        "temporal_d_model": 32,
        "temporal_layers": 2,
        "temporal_heads": 4,
        "temporal_ff_dim": 64,
        "group_head_dim": 64,
        "batch_size": 2,
        "grad_accum_steps": 1,
        "reservoir_capacity": 4,
        "warmup_steps": 1,
        "max_steps": 2,
        "compile_trunk": False,
        "compile_temporal": False,
        "num_workers": 0,
        "push_to_r2": False,
        "inference_mode": "eager",
    }
    values.update(overrides)
    cfg = exp.TrainConfig(**values)
    d_level = cell[1]
    decoder = cell[3]
    return exp.config_for_cell(cfg, f"w{d_level}d{decoder}")


def _actions(batch: int, length: int, generator: torch.Generator) -> torch.Tensor:
    actions = torch.empty(batch, length, A_DIM)
    actions[..., :4] = torch.rand(batch, length, 4, generator=generator) * 2 - 1
    actions[..., 4:6] = torch.rand(batch, length, 2, generator=generator)
    actions[..., 6:] = torch.randint(0, 2, (batch, length, 8), generator=generator).float()
    return actions


def _context(cfg: exp.TrainConfig, batch: int = 2, seed: int = 0) -> Context:
    generator = torch.Generator().manual_seed(seed)
    context = exp.synthetic_context(cfg, batch, torch.device("cpu"))
    features = dict(context.features)
    actions = _actions(batch, cfg.L_ctx, generator)
    for channel, values in zip(ACTION_CHANNELS, actions.unbind(-1), strict=True):
        features[f"ego_{channel}"] = values
    return Context(
        features=features,
        ctx_pad=torch.arange(batch).clamp_max(cfg.L_ctx - 1),
        slot_ids=torch.arange(batch),
        reset=torch.ones(batch, dtype=torch.bool),
    )


def _batch(cfg: exp.TrainConfig, seed: int = 0) -> TrainBatch:
    generator = torch.Generator().manual_seed(seed + 100)
    return TrainBatch(
        context=_context(cfg, seed=seed),
        target=_actions(2, cfg.sample_chunk_length, generator),
        replay_ids=("a" * 32, "b" * 32),
    )


def _base_cfg(cfg: exp.TrainConfig) -> exp026.TrainConfig:
    names = {item.name for item in fields(exp026.TrainConfig)}
    return exp026.TrainConfig(**{name: value for name, value in asdict(cfg).items() if name in names})


def _padded(values: torch.Tensor, vocab: int) -> torch.Tensor:
    return torch.nn.functional.pad(values.float(), (0, max(exp.GROUP_VOCABS) - vocab))


def _diagnostic_decode(model, hidden, observed, offsets, uniforms):
    """Mirror the production sampler while returning logits and probabilities."""
    decoder = model.temporal
    frame_indices = []
    frame_logits = []
    frame_probabilities = []
    if isinstance(decoder, exp.IndependentOffsetDecoder):
        raw = {
            name: logits[:, -1].transpose(0, 1)
            for name, logits in decoder.raw_logits_by_group(hidden, offsets).items()
        }

    previous = observed
    caches = [None] * len(decoder.blocks) if not isinstance(decoder, exp.IndependentOffsetDecoder) else []
    for depth, offset in enumerate(offsets):
        if isinstance(decoder, exp.IndependentOffsetDecoder):
            state = None
            trunk_logits = None
        else:
            trunk = exp.decoder_rmsnorm(hidden[:, -1])
            trunk_logits = {name: decoder.trunk_outputs[name](trunk) for name in exp.GROUP_NAMES}
            offset_tensor = torch.full((hidden.shape[0],), offset, device=hidden.device, dtype=torch.long)
            state = decoder.token_projection(
                torch.cat(
                    (trunk, decoder.codec.embed_frame(previous), decoder.offset_embedding(offset_tensor)),
                    dim=-1,
                )
            )
            next_caches = []
            for block, past in zip(decoder.blocks, caches, strict=True):
                state, present = block.forward_step(state, past)
                next_caches.append(present)
            caches = next_caches
            state = exp.decoder_rmsnorm(state)
        picks = {}
        embedded = {}
        logits_by_group = {}
        probabilities_by_group = {}
        for name in exp.GROUP_ORDER:
            if isinstance(decoder, exp.IndependentOffsetDecoder):
                logits = raw[name][depth]
            else:
                logits = decoder.outputs[name](decoder.group_features(state, name, embedded)) + trunk_logits[name]
            if name == "buttons":
                logits = logits.masked_fill(decoder.codec.button_mask(picks["triggers"]), float("-inf"))
            group = exp.GROUP_INDEX[name]
            pick = exp.sample_categorical(logits, argmax=False, uniform=uniforms[depth, group])
            picks[name] = pick
            if not isinstance(decoder, exp.IndependentOffsetDecoder):
                embedded[name] = decoder.codec.group_embedding(name, pick)
            vocab = exp.GROUP_VOCABS[group]
            logits_by_group[name] = _padded(logits, vocab)
            probabilities_by_group[name] = _padded(torch.softmax(logits.float(), dim=-1), vocab)
        indices = torch.stack([picks[name] for name in exp.GROUP_NAMES], dim=-1)
        frame_indices.append(indices)
        frame_logits.append(torch.stack([logits_by_group[name] for name in exp.GROUP_NAMES]))
        frame_probabilities.append(torch.stack([probabilities_by_group[name] for name in exp.GROUP_NAMES]))
        previous = indices
    return (
        torch.stack(frame_indices, dim=1),
        torch.stack(frame_logits),
        torch.stack(frame_probabilities),
    )


def test_production_cells_freeze_only_width_and_decoder() -> None:
    configs = {cell: exp._production_config(cell) for cell in exp._CELL_GEOMETRY}
    assert {(cfg.batch_size, cfg.max_steps, cfg.L_ctx) for cfg in configs.values()} == {(512, 4_096, 128)}
    assert {(cfg.seed, cfg.eval_seed) for cfg in configs.values()} == {(0, 0)}
    assert {(cfg.d_model, cfg.n_heads) for cfg in configs.values()} == {(256, 4), (384, 6)}
    assert {cfg.decoder_structure for cfg in configs.values()} == {"independent", "causal"}

    ignored = {"d_model", "n_heads", "decoder_structure", "factorial_cell"}
    baseline = asdict(configs["w1d1"])
    for cfg in configs.values():
        assert {name: value for name, value in asdict(cfg).items() if name not in ignored} == {
            name: value for name, value in baseline.items() if name not in ignored
        }
        exp.validate_config(cfg)


def test_production_rejects_drift_and_smoke_allows_only_runtime_knobs() -> None:
    with pytest.raises(ValueError, match="requires geometry"):
        exp.validate_config(exp.TrainConfig(factorial_cell="w0d0"))
    with pytest.raises(ValueError, match="production 037 config changed"):
        exp.validate_config(exp.TrainConfig(seed=1))
    smoke = exp.config_for_cell(exp.TrainConfig(max_steps=8, grad_accum_steps=2), "w0d0")
    exp.validate_config(smoke)
    with pytest.raises(ValueError, match="smoke 037 config changed"):
        exp.validate_config(exp.config_for_cell(exp.TrainConfig(max_steps=8, seed=1), "w0d0"))


def test_w1d1_is_exact_026_parameters_logits_loss_gradients_and_samples() -> None:
    cfg = _cfg("w1d1")
    cfg026 = _base_cfg(cfg)
    torch.manual_seed(17)
    control = exp026.GPT(cfg026).eval()
    torch.manual_seed(17)
    factorial = exp.GPT(cfg).eval()

    assert factorial.state_dict().keys() == control.state_dict().keys()
    for name, expected in control.state_dict().items():
        torch.testing.assert_close(factorial.state_dict()[name], expected, rtol=0, atol=0)

    batch = _batch(cfg, seed=3)
    control_batch = batch
    control_parts = exp026.action_loss(control, control_batch)
    factorial_parts = exp.action_loss(factorial, batch)
    torch.testing.assert_close(factorial_parts.nll, control_parts.nll, rtol=0, atol=0)
    factorial_loss = exp.objective(factorial_parts)
    control_loss = exp026.objective(control_parts)
    torch.testing.assert_close(factorial_loss, control_loss, rtol=0, atol=0)
    factorial_loss.backward()
    control_loss.backward()
    for (name, actual), (control_name, expected) in zip(
        factorial.named_parameters(),
        control.named_parameters(),
        strict=True,
    ):
        assert name == control_name
        if actual.grad is None or expected.grad is None:
            assert actual.grad is expected.grad is None
            continue
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)

    history, targets, _ = exp.prepared_targets(factorial, batch)
    with torch.no_grad():
        hidden = factorial(batch.context.features, batch.context.ctx_pad, history)
        control_hidden = control(batch.context.features, batch.context.ctx_pad, history)
        logits = factorial.temporal.teacher_forced_logits_by_group(hidden, history, targets)
        control_logits = control.temporal.teacher_forced_logits_by_group(control_hidden, history, targets)
        for name in exp.GROUP_NAMES:
            torch.testing.assert_close(logits[name], control_logits[name], rtol=0, atol=0)
        uniforms = torch.rand(4, exp.N_GROUPS, 2, generator=torch.Generator().manual_seed(9))
        actual = factorial.temporal.sample_indices(
            hidden,
            history[:, -1],
            cfg.head_offsets[:4],
            argmax=False,
            uniforms=uniforms,
        )
        expected = control.temporal.sample_indices(
            control_hidden,
            history[:, -1],
            cfg.head_offsets[:4],
            argmax=False,
            uniforms=uniforms,
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_w1d1_optimizer_partition_and_hyperparameters_are_exact_026() -> None:
    cfg = _cfg("w1d1")
    cfg026 = _base_cfg(cfg)
    torch.manual_seed(41)
    control = exp026.GPT(cfg026)
    torch.manual_seed(41)
    factorial = exp.GPT(cfg)
    control_optimizer = exp026.make_optimizer(control, cfg026)
    factorial_optimizer = exp.make_optimizer(factorial, cfg)

    def signature(model, optimizer):
        names = {id(parameter): name for name, parameter in model.named_parameters()}
        return [
            {
                "parameters": [names[id(parameter)] for parameter in group["params"]],
                "lr": group["lr"],
                "momentum": group.get("momentum"),
                "betas": group.get("betas"),
                "eps": group.get("eps"),
                "weight_decay": group["weight_decay"],
                "use_muon": group["use_muon"],
            }
            for group in optimizer.param_groups
        ]

    assert signature(factorial, factorial_optimizer) == signature(control, control_optimizer)


def test_d0_raw_logits_have_no_action_or_cross_position_conditioning() -> None:
    cfg = _cfg("w1d0")
    model = exp.GPT(cfg).eval()
    assert isinstance(model.temporal, exp.IndependentOffsetDecoder)
    assert not hasattr(model.temporal, "group_condition")
    assert not any(isinstance(module, exp.base.TemporalBlock) for module in model.temporal.modules())
    hidden = torch.randn(2, cfg.L_ctx, cfg.d_model, generator=torch.Generator().manual_seed(4))
    observed = torch.zeros(2, cfg.L_ctx, exp.N_GROUPS, dtype=torch.long)
    targets = torch.zeros(2, cfg.L_ctx, len(cfg.head_offsets), exp.N_GROUPS, dtype=torch.long)
    changed_observed = torch.stack(
        [torch.randint(vocab, observed.shape[:-1]) for vocab in exp.GROUP_VOCABS],
        dim=-1,
    )
    changed_targets = torch.stack(
        [torch.randint(vocab, targets.shape[:-1]) for vocab in exp.GROUP_VOCABS],
        dim=-1,
    )

    raw = model.temporal.raw_logits_by_group(hidden)
    before = model.temporal.teacher_forced_logits_by_group(hidden, observed, targets)
    changed_targets[..., exp.TRIG_G] = targets[..., exp.TRIG_G]
    after = model.temporal.teacher_forced_logits_by_group(hidden, changed_observed, changed_targets)
    for name in exp.GROUP_NAMES:
        torch.testing.assert_close(raw[name], model.temporal.raw_logits_by_group(hidden)[name], rtol=0, atol=0)
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)

    changed_trigger = targets.clone()
    changed_trigger[..., exp.TRIG_G] = exp.GROUP_VOCABS[exp.TRIG_G] - 1
    trigger_masked = model.temporal.teacher_forced_logits_by_group(hidden, observed, changed_trigger)
    for name in ("c_stick", "main_stick", "triggers"):
        torch.testing.assert_close(before[name], trigger_masked[name], rtol=0, atol=0)
    assert torch.equal(before["buttons"].isfinite(), trigger_masked["buttons"].isfinite()) is False


@pytest.mark.parametrize(
    ("cell", "total", "decoder"),
    [
        ("w0d0", 7_056_367, 624_707),
        ("w0d1", 7_081_711, 650_051),
        ("w1d0", 15_027_695, 686_531),
        ("w1d1", 15_053_039, 711_875),
    ],
)
def test_parameter_counts_are_preregistered(cell: exp.FactorialCell, total: int, decoder: int) -> None:
    model = exp.GPT(exp._production_config(cell))
    assert sum(parameter.numel() for parameter in model.parameters()) == total
    assert exp.decoder_parameter_count(model) == decoder


@pytest.mark.parametrize("width", ["w0", "w1"])
def test_d0_capacity_is_within_five_percent_and_trunk_initialization_matches(width: str) -> None:
    cfg_d0 = exp._production_config(cast(exp.FactorialCell, f"{width}d0"))
    cfg_d1 = exp._production_config(cast(exp.FactorialCell, f"{width}d1"))
    torch.manual_seed(23)
    d0 = exp.GPT(cfg_d0)
    torch.manual_seed(23)
    d1 = exp.GPT(cfg_d1)
    difference = abs(exp.decoder_parameter_count(d0) - exp.decoder_parameter_count(d1))
    assert difference / exp.decoder_parameter_count(d1) < 0.05
    for (name, actual), (control_name, expected) in zip(
        d0.trunk.named_parameters(),
        d1.trunk.named_parameters(),
        strict=True,
    ):
        assert name == control_name
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_every_d0_decoder_parameter_has_a_finite_gradient_and_optimizer_owner() -> None:
    cfg = _cfg("w1d0")
    model = exp.GPT(cfg)
    loss = exp.objective(exp.action_loss(model, _batch(cfg, seed=8)))
    loss.backward()
    for name, parameter in model.temporal.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name

    optimizer = exp.make_optimizer(model, cfg)
    owned = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(owned) == sum(1 for _ in model.parameters())
    assert len({id(parameter) for parameter in owned}) == len(owned)


def test_all_cells_share_loader_sampling_and_schedule_contracts() -> None:
    configs = [exp._production_config(cell) for cell in exp._CELL_GEOMETRY]
    assert {exp.sampling_contract_sha256(cfg) for cfg in configs} == {
        "4888b73937a3fa31077b6fd3203b5c3b25f1d4070f59257a2e3bd79587fff499"
    }
    loader_keys = {
        "data_root",
        "L_ctx",
        "L_chunk",
        "batch_size",
        "seed",
        "schema_version",
        "shuffle_block_size",
    }
    contracts = [
        {key: value for key, value in exp.loader_kwargs(cfg, {}).items() if key in loader_keys}
        for cfg in configs
    ]
    assert contracts.count(contracts[0]) == 4
    schedules = [[exp.lr_schedule(cfg)(step) for step in (0, 1, 500, 1_024, 2_048, 4_096)] for cfg in configs]
    assert schedules.count(schedules[0]) == 4


def test_actual_train_window_and_validation_order_hashes_are_cell_invariant() -> None:
    batches = [_batch(_cfg("w0d0"), seed=seed) for seed in (1, 2)]
    train_hashes = {exp.train_order_sha256(batches) for _ in exp._CELL_GEOMETRY}
    validation_hashes = {exp.validation_cache_sha256(batches) for _ in exp._CELL_GEOMETRY}
    assert len(train_hashes) == len(validation_hashes) == 1
    changed = [_batch(_cfg("w0d0"), seed=1), _batch(_cfg("w0d0"), seed=3)]
    assert exp.train_order_sha256(changed) not in train_hashes
    assert exp.validation_cache_sha256(changed) not in validation_hashes

    no_ids = [
        TrainBatch(context=batch.context, target=batch.target, replay_ids=None)
        for batch in batches
    ]
    no_id_hash = exp.validation_cache_sha256(no_ids)
    assert no_id_hash != next(iter(validation_hashes))
    assert exp.validation_cache_sha256(no_ids) == no_id_hash
    changed_features = dict(no_ids[0].context.features)
    first_name = sorted(changed_features)[0]
    changed_features[first_name] = changed_features[first_name].clone()
    first_value = changed_features[first_name].view(-1)
    first_value[0] = ~first_value[0] if first_value.dtype == torch.bool else first_value[0] + 1
    changed_context = Context(
        features=changed_features,
        ctx_pad=no_ids[0].context.ctx_pad,
        slot_ids=no_ids[0].context.slot_ids,
        reset=no_ids[0].context.reset,
    )
    feature_changed = [
        TrainBatch(context=changed_context, target=no_ids[0].target),
        no_ids[1],
    ]
    assert exp.validation_cache_sha256(feature_changed) != no_id_hash

    manifests = {
        cell: exp.launch_manifest(cfg, next(iter(validation_hashes)), next(iter(train_hashes)))
        for cell, cfg in ((cell, exp._production_config(cell)) for cell in exp._CELL_GEOMETRY)
    }
    assert len({manifest["config_sha256"] for manifest in manifests.values()}) == 4
    for field in (
        "source_sha256",
        "sampling_contract_sha256",
        "train_order_first_two_batches_sha256",
        "validation_cache_sha256",
        "action_random_contract_sha256",
        "boot_schedule_sha256",
        "named_checkpoints",
        "eval_max_parallel",
        "max_steps",
    ):
        assert len({str(manifest[field]) for manifest in manifests.values()}) == 1


def test_checkpoint_identity_names_and_round_trip_are_frozen() -> None:
    cfg = exp._production_config("w0d0")
    restored = exp.config_from_state(asdict(cfg))
    assert restored == cfg
    assert [
        exp.named_checkpoint_path(Path("runs/x/latest.pt"), step)
        for step in (1_024, 2_048, 3_072)
    ] == [
        Path("runs/x/step_001024.pt"),
        Path("runs/x/step_002048.pt"),
        None,
    ]
    assert exp.named_checkpoint_path(Path("runs/x/final.pt"), 4_096) == Path("runs/x/step_004096.pt")
    wrong = asdict(cfg)
    wrong["experiment_id"] = "026_temporal_mtp"
    with pytest.raises(ValueError, match="experiment_id"):
        exp.config_from_state(wrong)


def test_named_checkpoints_write_and_round_trip_cell_identity(tmp_path: Path) -> None:
    cfg = exp._production_config("w0d0")
    model = exp.GPT(cfg)
    optimizer = exp.make_optimizer(model, cfg)
    scheduler = exp.LambdaLR(optimizer, exp.lr_schedule(cfg))
    latest = tmp_path / "latest.pt"
    for update in (1_024, 2_048, 4_096):
        exp._save_update_checkpoint(
            latest,
            update=update,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            uploader=None,
        )
        named = tmp_path / f"step_{update:06d}.pt"
        state = torch.load(named, map_location="cpu", weights_only=False)
        assert state["step"] == update - 1
        assert exp.config_from_state(state["cfg"]).factorial_cell == "w0d0"
    final = tmp_path / "final.pt"
    shutil.copy2(latest, final)
    assert final.read_bytes() == (tmp_path / "step_004096.pt").read_bytes()


def test_cli_dispatches_custom_trainer_and_applies_cell(monkeypatch) -> None:
    assert exp.base.train is exp.train
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setattr(exp.base, "main", fake_main)
    args = exp.Args(cfg=exp.TrainConfig(max_steps=2), cell="w0d0")
    exp.main(args)
    assert captured["args"].cfg.factorial_cell == "w0d0"
    assert captured["args"].cfg.decoder_structure == "independent"


def test_resume_is_rejected_without_verified_rng_and_loader_cursor() -> None:
    with pytest.raises(RuntimeError, match="restart this cell fresh"):
        exp.train(_cfg("w0d0"), {}, resume_run="interrupted", resume_state={})
    with pytest.raises(SystemExit, match="restart this cell fresh"):
        exp.main(exp.Args(resume="interrupted", cell="w0d0"))


@pytest.mark.parametrize("cell", ["w0d0", "w0d1", "w1d0", "w1d1"])
def test_same_seed_smoke_has_finite_aggregate_group_and_offset_losses(
    cell: exp.FactorialCell,
) -> None:
    torch.manual_seed(53)
    cfg = _cfg(cell)
    model = exp.GPT(cfg)
    parts = exp.action_loss(model, _batch(cfg, seed=53))
    loss = exp.objective(parts)
    metrics = exp.nll_metrics(parts.nll, cfg.head_offsets)

    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    for offset in cfg.head_offsets:
        assert f"nll_o{offset:02d}" in metrics
        for group in exp.GROUP_NAMES:
            assert f"nll_o{offset:02d}_{group}" in metrics


def test_paired_rng_tables_are_cell_invariant_and_do_not_touch_global_rng() -> None:
    cfg = _cfg("w1d0")
    context = _context(cfg)

    def table() -> torch.Tensor:
        stream = exp.SlotGroupRandom(0)
        stream.begin(context)
        return torch.stack(
            [torch.stack([stream.uniforms(name) for name in exp.GROUP_NAMES]) for _ in range(6)]
        )

    state = torch.random.get_rng_state()
    first = table()
    torch.testing.assert_close(torch.random.get_rng_state(), state)
    second = table()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert exp.matchup_diversity(96)[3] == exp026.matchup_diversity(96)[3]


@pytest.mark.parametrize("cell", ["w0d0", "w0d1", "w1d0", "w1d1"])
def test_h4_h6_decode_twice_without_invalid_actions(cell: exp.FactorialCell) -> None:
    cfg = _cfg(cell)
    model = exp.GPT(cfg).eval()
    engine = exp.BF16Inference(model, cfg, compiled=False)
    context = _context(cfg)
    for horizon in (4, 6):
        for _ in range(2):
            actions = engine.decode(context, horizon, streams=exp.SlotGroupRandom(31))
            indices = model.codec.quantize(actions)
            assert actions.shape == (2, horizon, A_DIM)
            assert model.codec.button_valid_for_trigger[
                indices[..., exp.TRIG_G], indices[..., exp.BUTTONS_G]
            ].all()


def test_d0_decoder_has_no_dynamo_graph_breaks() -> None:
    cfg = _cfg("w1d0")
    model = exp.GPT(cfg).eval()
    hidden = torch.randn(2, cfg.L_ctx, cfg.d_model)
    observed = torch.zeros(2, cfg.L_ctx, exp.N_GROUPS, dtype=torch.long)
    targets = torch.zeros(2, cfg.L_ctx, len(cfg.head_offsets), exp.N_GROUPS, dtype=torch.long)
    explanation = torch._dynamo.explain(model.temporal.teacher_forced_nll)(hidden, observed, targets)
    assert explanation.graph_break_count == 0
    assert explanation.graph_count == 1


def test_tiny_d0_latency_smoke_is_finite() -> None:
    cfg = _cfg("w1d0")
    model = exp.GPT(cfg).eval()
    engine = exp.BF16Inference(model, cfg, compiled=False)
    context = _context(cfg)
    engine.decode(context, 4, streams=exp.SlotGroupRandom(5))
    started = time.perf_counter()
    for _ in range(2):
        engine.decode(context, 4, streams=exp.SlotGroupRandom(5))
    elapsed = time.perf_counter() - started
    assert 0 < elapsed < 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA integrity gate")
@pytest.mark.parametrize("cell", ["w1d0", "w1d1"])
def test_cuda_compiled_live_h4_h6_are_fullgraph_and_match_eager(cell: exp.FactorialCell) -> None:
    cfg = _cfg(cell, inference_mode="compiled")
    model = exp.GPT(cfg).cuda().eval()
    context = _context(cfg, batch=32).to("cuda")
    observed = model.codec.quantize(
        torch.stack([context.features[f"ego_{name}"] for name in ACTION_CHANNELS], dim=-1)
    )
    with exp.amp_context(cfg, "cuda"):
        model(context.features, context.ctx_pad, observed)

    def trunk(features, pad, actions):
        return model(features, pad, actions)

    compiled_trunk = torch.compile(trunk, dynamic=False, fullgraph=True)
    with exp.amp_context(cfg, "cuda"):
        hidden = compiled_trunk(context.features, context.ctx_pad, observed)
    for horizon in (4, 6):
        offsets = cfg.head_offsets[:horizon]
        generator = torch.Generator(device="cuda").manual_seed(10_000 + horizon)
        uniforms = torch.rand(horizon, exp.N_GROUPS, 32, device="cuda", generator=generator)

        def decoder(hidden, observed, uniforms, offsets=offsets):
            return model.temporal.sample_indices(
                hidden,
                observed,
                offsets,
                argmax=False,
                uniforms=uniforms,
            )

        def diagnostic(hidden, observed, uniforms, offsets=offsets):
            return _diagnostic_decode(model, hidden, observed, offsets, uniforms)

        compiled_decoder = torch.compile(decoder, dynamic=False, fullgraph=True)
        compiled_diagnostic = torch.compile(diagnostic, dynamic=False, fullgraph=True)
        with exp.amp_context(cfg, "cuda"):
            expected = decoder(hidden, observed[:, -1], uniforms)
            eager_indices, eager_logits, eager_probabilities = diagnostic(hidden, observed[:, -1], uniforms)
            first = compiled_decoder(hidden, observed[:, -1], uniforms)
            second = compiled_decoder(hidden, observed[:, -1], uniforms)
            compiled_indices, compiled_logits, compiled_probabilities = compiled_diagnostic(
                hidden,
                observed[:, -1],
                uniforms,
            )
        torch.testing.assert_close(eager_indices, expected, rtol=0, atol=0)
        torch.testing.assert_close(compiled_indices, first, rtol=0, atol=0)
        torch.testing.assert_close(second, first, rtol=0, atol=0)
        finite = torch.isfinite(eager_logits) & torch.isfinite(compiled_logits)
        max_logit_error = float((compiled_logits[finite] - eager_logits[finite]).abs().max())
        max_probability_error = float((compiled_probabilities - eager_probabilities).abs().max())
        eager_cdf = eager_probabilities.cumsum(dim=-1)
        compiled_cdf = compiled_probabilities.cumsum(dim=-1)
        max_cdf_error = float((compiled_cdf - eager_cdf).abs().max())
        assert max_logit_error <= 0.02
        assert max_probability_error <= 2e-3
        assert max_cdf_error <= 3e-3

        mismatch = first != expected
        mismatch_count = int(mismatch.sum())
        for batch, depth, group in mismatch.nonzero().tolist():
            vocab = exp.GROUP_VOCABS[group]
            uniform = uniforms[depth, group, batch]
            eager_boundaries = eager_cdf[depth, group, batch, :vocab]
            local_cdf_error = (compiled_cdf[depth, group, batch, :vocab] - eager_boundaries).abs().max()
            boundary_distance = (eager_boundaries - uniform).abs().min()
            assert float(boundary_distance) <= float(local_cdf_error) + 2e-6
        print(
            f"[cuda-parity] {cell} H{horizon}: mismatches={mismatch_count}/{first.numel()} "
            f"max_logit={max_logit_error:.3e} max_prob={max_probability_error:.3e} "
            f"max_cdf={max_cdf_error:.3e}",
            flush=True,
        )
        indices = compiled_indices
        assert model.codec.button_valid_for_trigger[
            indices[..., exp.TRIG_G], indices[..., exp.BUTTONS_G]
        ].all()
