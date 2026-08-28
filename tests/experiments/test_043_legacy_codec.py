"""Contracts for O43's exact legacy controller codec."""

import hashlib
import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from hal.sim.inputs import action_vec_to_controller
from hal.sim.process_vec import drive_process_vec
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import Context
from hal.training.features import TrainBatch
from hal.wire import BUTTON_BITS


def _load(name: str, filename: str) -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exp026 = _load("test_exp026_for_043", "026_temporal_mtp.py")
exp = _load("test_exp043", "043_legacy_codec.py")


def _cfg(**overrides) -> exp.TrainConfig:
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
        inference_mode="eager",
        num_workers=0,
        push_to_r2=False,
    )
    return exp.TrainConfig(**{**values, **overrides})


def _native_actions(batch: int, length: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    actions = torch.empty(batch, length, A_DIM)
    actions[..., :4] = torch.rand(batch, length, 4, generator=generator) * 2 - 1
    actions[..., 4:6] = torch.rand(batch, length, 2, generator=generator)
    actions[..., 6:] = torch.randint(0, 2, (batch, length, A_DIM - 6), generator=generator).float()
    return actions


def _batch(cfg: exp.TrainConfig) -> TrainBatch:
    context = exp.synthetic_context(cfg, cfg.batch_size, torch.device("cpu"))
    features = dict(context.features)
    actions = _native_actions(cfg.batch_size, cfg.L_ctx, seed=3)
    for name, values in zip(ACTION_CHANNELS, actions.unbind(-1), strict=True):
        features[f"ego_{name}"] = values
    return TrainBatch(
        context=Context(features=features, ctx_pad=context.ctx_pad),
        target=_native_actions(cfg.batch_size, cfg.sample_chunk_length, seed=4),
    )


def _assert_unused_are_masked(logits: dict[str, torch.Tensor]) -> None:
    for group, name in enumerate(exp.GROUP_NAMES):
        valid = exp.LEGACY_GROUP_VOCABS[group]
        assert torch.isfinite(logits[name][..., :valid]).all()
        assert torch.isneginf(logits[name][..., valid:]).all()


def test_defaults_match_the_named_o26_reference_run() -> None:
    cfg = exp.TrainConfig()
    assert (cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.L_ctx) == (384, 8, 6, 128)
    assert cfg.head_offsets == (1, 2, 3, 4, 5, 6, 9, 12, 16, 20)
    assert (cfg.temporal_d_model, cfg.temporal_layers, cfg.temporal_heads, cfg.temporal_ff_dim) == (128, 2, 4, 256)
    assert (cfg.batch_size, cfg.max_steps, cfg.warmup_steps) == (512, 16_384, 500)
    assert (cfg.cache_limit_gb, cfg.eval_max_parallel) == (160, 32)
    assert cfg.data_root == "data/processed/ranked-anonymized-1/mds-policy-v7"
    assert cfg.replay_format == "policy"
    assert cfg.val_data_root == cfg.data_root
    assert cfg.val_replay_format == "policy"
    assert cfg.group_order == ("c_stick", "main_stick", "triggers", "buttons")
    assert "mtp043-legacy" in exp.model_tag(cfg)


def test_o43_preserves_o26_trainable_architecture_and_initialization() -> None:
    cfg026 = exp026.TrainConfig()
    cfg043 = exp.TrainConfig()
    torch.manual_seed(123)
    baseline = exp026.GPT(cfg026)
    torch.manual_seed(123)
    treatment = exp.GPT(cfg043)

    baseline_parameters = dict(baseline.named_parameters())
    treatment_parameters = dict(treatment.named_parameters())
    assert tuple(treatment_parameters) == tuple(baseline_parameters)
    for name, expected in baseline_parameters.items():
        actual = treatment_parameters[name]
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected), name

    assert tuple(treatment.state_dict()) == tuple(baseline.state_dict())
    assert all(
        treatment.state_dict()[name].shape == baseline.state_dict()[name].shape for name in baseline.state_dict()
    )
    assert sum(parameter.numel() for parameter in treatment.parameters()) == 15_053_039
    assert len(treatment.state_dict()) == 111


def test_historical_centers_and_complete_controller_grid_round_trip() -> None:
    codec = exp.StructuredControllerCodec(16)
    grid = torch.cartesian_prod(*(torch.arange(size) for size in exp.LEGACY_GROUP_VOCABS))
    restored = codec.quantize(codec.dequantize(grid[:, None]))[:, 0]
    assert torch.equal(restored, grid)

    buttons = torch.zeros(5, 4, dtype=torch.long)
    buttons[:, exp.BUTTONS_G] = torch.arange(5)
    decoded = codec.dequantize(buttons)
    assert decoded[:, 6:].sum(-1).tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]
    assert decoded[2, ACTION_CHANNELS.index("button_x")] == 1
    assert decoded[2, ACTION_CHANNELS.index("button_y")] == 0


def test_historical_quantization_matches_pinned_numpy_golden_hashes() -> None:
    codec = exp.StructuredControllerCodec(16)
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
        "ee33ad55936406fa958d3046bb6f966efcfbe7f5eb14fb7be944848e25b3f1e4"
    )


def test_chord_reducer_matches_the_pinned_numpy_implementation() -> None:
    rng = np.random.default_rng(42)
    pressed = rng.integers(0, 2, size=(16, 31, 4), dtype=np.int64)
    actions = torch.zeros(16, 31, A_DIM)
    actions[..., (6, 7, 8, 10)] = torch.from_numpy(pressed).float()

    expected = []
    for sequence in pressed:
        no_button = (sequence.sum(axis=1, keepdims=True) == 0).astype(np.int64)
        buttons = np.concatenate((sequence.copy(), no_button), axis=1)
        row_sums = buttons.sum(axis=1)
        multi = np.flatnonzero(row_sums > 1)
        previous: set[int] = set()
        if len(multi):
            first = int(multi[0])
            previous = set(np.flatnonzero(buttons[first - 1])) if first else set()
        for index in multi:
            current = set(np.flatnonzero(buttons[index]))
            if current == previous:
                buttons[index] = buttons[index - 1]
                continue
            selected = min(current - previous) if current > previous else min(current)
            buttons[index] = 0
            buttons[index, selected] = 1
            previous = current
        buttons[row_sums == 0, -1] = 1
        expected.append(buttons.argmax(axis=1))

    assert np.array_equal(exp.legacy_button_classes(actions).numpy(), np.stack(expected))


def test_held_b_plus_a_reproduces_legacy_release_and_repress() -> None:
    actions = torch.zeros(1, 5, A_DIM)
    a = ACTION_CHANNELS.index("button_a")
    b = ACTION_CHANNELS.index("button_b")
    actions[0, :, b] = 1
    actions[0, 2:4, a] = 1

    codec = exp.StructuredControllerCodec(16)
    indices = codec.quantize(actions)
    assert indices[0, :, exp.BUTTONS_G].tolist() == [1, 1, 0, 0, 1]

    decoded = codec.dequantize(indices)[0].numpy()
    controller = [action_vec_to_controller(frame) for frame in decoded]
    assert [frame.buttons for frame in controller] == [
        BUTTON_BITS["b"],
        BUTTON_BITS["b"],
        BUTTON_BITS["a"],
        BUTTON_BITS["a"],
        BUTTON_BITS["b"],
    ]


def test_v7_shoulder_reconstruction_is_fused_and_decodes_on_l() -> None:
    actions = torch.zeros(1, 3, A_DIM)
    actions[0, 0, exp.TRIGGER_R_CH] = 0.39
    actions[0, 1, exp.TRIGGER_R_CH] = 0.41
    actions[0, 2, exp.BUTTON_R_CH] = 1

    codec = exp.StructuredControllerCodec(16)
    indices = codec.quantize(actions)
    assert indices[0, :, exp.TRIG_G].tolist() == [1, 1, 2]
    decoded = codec.dequantize(indices)
    assert decoded[0, :, exp.TRIGGER_L_CH].tolist() == pytest.approx([0.4, 0.4, 1.0])
    assert not decoded[..., exp.TRIGGER_R_CH].any()
    assert not decoded[..., exp.BUTTON_L_CH].any()
    assert not decoded[..., exp.BUTTON_R_CH].any()


def test_unused_classes_are_masked_in_every_training_and_inference_path() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg).eval()
    hidden = torch.randn(2, cfg.L_ctx, cfg.d_model)
    observed = torch.zeros(2, cfg.L_ctx, exp.N_GROUPS, dtype=torch.long)
    targets = torch.zeros(2, cfg.L_ctx, len(cfg.head_offsets), exp.N_GROUPS, dtype=torch.long)

    parallel = model.temporal.teacher_forced_logits_by_group(hidden, observed, targets)
    _assert_unused_are_masked(parallel)

    for logits in model.temporal.forced_stepwise_logits(hidden, observed[:, -1], targets[:, -1]):
        _assert_unused_are_masked(logits)

    sampled = model.temporal.sample_indices(hidden, observed[:, -1], cfg.head_offsets[:4], argmax=True)
    for group, valid in enumerate(exp.LEGACY_GROUP_VOCABS):
        assert (sampled[..., group] < valid).all()

    rollout_logits, rollout = model.temporal.rollout_conditioned_logits(hidden, observed[:, -1])
    for logits in rollout_logits:
        _assert_unused_are_masked(logits)
    for group, valid in enumerate(exp.LEGACY_GROUP_VOCABS):
        assert (rollout[..., group] < valid).all()


def test_live_sampler_only_receives_real_legacy_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()
    model = exp.GPT(cfg).eval()
    hidden = torch.randn(2, cfg.L_ctx, cfg.d_model)
    observed = torch.zeros(2, exp.N_GROUPS, dtype=torch.long)
    vocab_sizes: list[int] = []
    original = exp.sample_categorical

    def record_vocab(logits: torch.Tensor, **kwargs) -> torch.Tensor:
        vocab_sizes.append(logits.shape[-1])
        return original(logits, **kwargs)

    monkeypatch.setattr(exp, "sample_categorical", record_vocab)
    model.temporal.sample_indices(hidden, observed, cfg.head_offsets[:4], argmax=False)

    expected_per_frame = [exp.LEGACY_GROUP_VOCABS[exp.GROUP_INDEX[name]] for name in exp.GROUP_ORDER]
    assert vocab_sizes == expected_per_frame * 4


def test_masked_output_rows_receive_zero_gradient() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg)
    hidden = torch.randn(2, cfg.L_ctx, cfg.d_model)
    observed = torch.zeros(2, cfg.L_ctx, exp.N_GROUPS, dtype=torch.long)
    targets = torch.zeros(2, cfg.L_ctx, len(cfg.head_offsets), exp.N_GROUPS, dtype=torch.long)
    model.temporal.teacher_forced_nll(hidden, observed, targets).mean().backward()

    for group, name in enumerate(exp.GROUP_NAMES):
        valid = exp.LEGACY_GROUP_VOCABS[group]
        output = model.temporal.outputs[name].down
        assert output.weight.grad is not None and not output.weight.grad[valid:].any()
        assert output.bias.grad is not None and not output.bias.grad[valid:].any()
        trunk = model.temporal.trunk_outputs[name]
        assert trunk.weight.grad is not None and not trunk.weight.grad[valid:].any()


def test_cpu_loss_backward_and_checkpoint_round_trip(tmp_path: Path) -> None:
    cfg = _cfg()
    model = exp.GPT(cfg)
    loss = exp.objective(exp.action_loss(model, _batch(cfg)), cfg.aux_loss_weight)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())

    checkpoint = tmp_path / "state.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = exp.GPT(cfg)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected)


def test_cody_config_uses_policy_world_train_and_fixed_baseline_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(
        data_root="data/processed/professional/cody/mds-policy-world-v7",
        replay_format="policy-world",
        val_n_samples=2,
    )
    batch = _batch(cfg)
    calls: dict[str, dict[str, object]] = {}

    def reservoir_loader(**kwargs):
        calls["train"] = kwargs
        return [batch]

    def validation_loader(**kwargs):
        calls["val"] = kwargs
        return [batch]

    monkeypatch.setattr(exp, "make_reservoir_loader", reservoir_loader)
    monkeypatch.setattr(exp, "make_loader", validation_loader)
    train_loader, val_cache = exp._make_loaders(cfg, {})

    assert list(train_loader) == [batch]
    assert val_cache == [batch]
    assert calls["train"]["replay_format"] == "policy-world"
    assert calls["val"]["data_root"] == "data/processed/ranked-anonymized-1/mds-policy-v7"
    assert calls["val"]["replay_format"] == "policy"


def test_eval_suites_are_separate_and_named_in_wandb(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = _cfg()
    model = exp.GPT(cfg)
    calls: list[tuple[Path, object]] = []

    def fake_eval(*_args, replay_dir, fixed_ego_character, **_kwargs):
        calls.append((replay_dir, fixed_ego_character))
        return {"boots": 2.0}

    monkeypatch.setattr(exp, "eval_vs_cpu", fake_eval)
    suites = exp.eval_suites(
        model,
        {},
        cfg,
        n_matchups=2,
        replay_dir=tmp_path,
        checkpoint_sha256="a" * 64,
        inference=object(),
    )

    assert [path.name for path, _ in calls] == ["char_matchup", "fox"]
    assert calls[0][1] is None
    assert calls[1][1] == exp.melee.Character.FOX
    assert exp.eval_suite_wandb_metrics(suites) == {
        "eval_char_matchup/boots": 2.0,
        "eval_fox/boots": 2.0,
    }


def test_tiny_training_entrypoint_uses_legacy_codec(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = _cfg(
        d_model=64,
        max_steps=1,
        val_every=0,
        eval_every=0,
        ckpt_every=0,
        wandb_log_code=False,
        final_diag_n_matchups=0,
        final_eval_n_matchups=1,
    )
    batch = _batch(cfg)
    monkeypatch.setattr(exp, "_make_loaders", lambda cfg, stats: ([batch], [batch]))
    monkeypatch.setattr(exp, "make_run_name", lambda *args: "tiny-043")
    monkeypatch.setattr(exp, "setup_run_dir", lambda name: (tmp_path, tmp_path / "replays"))
    monkeypatch.setattr(exp, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp, "_checkpoint_sha256", lambda path: "a" * 64)
    monkeypatch.setattr(exp, "BF16Inference", lambda model, cfg: object())
    monkeypatch.setattr(
        exp,
        "eval_vs_cpu",
        lambda model, stats, cfg, **kwargs: {
            "scheduled_boots": 1.0,
            "completed_boots": 1.0,
            "boots": 1.0,
        },
    )

    class Run:
        id = "test"
        summary: dict[str, object] = {}

    monkeypatch.setattr(exp.wandb, "run", Run())
    monkeypatch.setattr(exp.wandb, "init", lambda **kwargs: None)
    monkeypatch.setattr(exp.wandb, "define_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp.wandb, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp.wandb, "finish", lambda **kwargs: None)

    exp.train(cfg, {}, comment="tiny")


def test_o43_evaluation_uses_o41_staggered_spawned_workers() -> None:
    assert inspect.signature(drive_process_vec).parameters["startup_parallelism"].default == 8
    cfg = _cfg()
    policy = exp.make_policy(exp.GPT(cfg), {}, cfg, inference=object(), device="cpu")
    assert callable(policy.plan_rows)
    assert policy.runtime_spec.prediction_frames == cfg.exec_horizon
