"""Contracts for O43's exact legacy controller codec."""

import hashlib
import importlib.util
import inspect
import sys
from dataclasses import asdict
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
    assert cfg.codec_version == 2
    assert cfg.ablation_arm == "A"
    assert cfg.exec_horizon == 1
    assert cfg.final_diag_exec_horizon == 4
    assert cfg.final_eval_n_matchups == cfg.final_diag_n_matchups == 96
    assert cfg.next_frame_loss_share == 0.5
    assert "mtp043-legacy" in exp.model_tag(cfg)
    assert "s1-o1w50" in exp.model_tag(cfg)
    assert exp._planned_inference_programs(cfg) == ((32, 1), (32, 4))


def test_ablation_arms_have_distinct_tags_and_checkpoint_identity() -> None:
    tags = {}
    for arm in ("A", "B", "C", "D"):
        cfg = _cfg(ablation_arm=arm)
        restored = exp.config_from_state(asdict(cfg))
        assert restored == cfg
        tags[arm] = exp.model_tag(cfg)

    assert len(set(tags.values())) == 4
    assert tags["A"].endswith("-base")
    assert tags["B"].endswith("-ablB-film0")
    assert tags["C"].endswith("-ablC-linear-head")
    assert tags["D"].endswith("-ablD-no-trunk-skip")

    legacy = asdict(_cfg())
    legacy.pop("ablation_arm")
    assert exp.config_from_state(legacy).ablation_arm == "A"
    with pytest.raises(ValueError, match="unknown ablation_arm"):
        exp.validate_config(_cfg(ablation_arm="Z"))


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


def test_ablation_arms_preserve_shared_initialization_and_rng_state() -> None:
    def initialized(arm: str) -> tuple[torch.nn.Module, torch.Tensor]:
        torch.manual_seed(123)
        model = exp.GPT(_cfg(ablation_arm=arm))
        return model, torch.get_rng_state()

    baseline, baseline_rng = initialized("A")
    baseline_parameters = dict(baseline.named_parameters())
    for arm in ("B", "C", "D"):
        treatment, treatment_rng = initialized(arm)
        torch.testing.assert_close(treatment_rng, baseline_rng)
        for name, parameter in treatment.named_parameters():
            if arm == "B" and name.startswith("temporal.group_condition."):
                continue
            if name in baseline_parameters:
                assert torch.equal(parameter, baseline_parameters[name]), (arm, name)


def test_zero_initialized_film_keeps_the_o43_equation() -> None:
    cfg = _cfg(ablation_arm="B")
    decoder = exp.GPT(cfg).temporal
    states = torch.randn(2, 3, cfg.temporal_d_model)
    embedded = {name: torch.randn(2, 3, cfg.action_embed_dim) for name in exp.GROUP_ORDER}

    for condition in decoder.group_condition.values():
        torch.testing.assert_close(condition.weight, torch.zeros_like(condition.weight))
        torch.testing.assert_close(condition.bias, torch.zeros_like(condition.bias))
    torch.testing.assert_close(decoder.group_features(states, "buttons", embedded), states)

    condition = decoder.group_condition["buttons"]
    with torch.no_grad():
        condition.bias[: decoder.d_model].fill_(100.0)
        condition.bias[decoder.d_model :].fill_(100.0)
    torch.testing.assert_close(decoder.group_features(states, "buttons", embedded), 2 * states + 100)


def test_linear_head_arm_uses_the_exact_o42_readout() -> None:
    cfg = _cfg(ablation_arm="C")
    decoder = exp.GPT(cfg).temporal
    assert all(isinstance(head, exp.LinearActionHead) for head in decoder.outputs.values())
    assert all(head.output.bias is None for head in decoder.outputs.values())
    assert decoder.trunk_outputs is not None

    head = decoder.outputs["buttons"]
    features = torch.randn(3, 5, cfg.temporal_d_model)
    expected = torch.nn.functional.linear(
        torch.nn.functional.rms_norm(features, (cfg.temporal_d_model,), eps=1e-5),
        head.output.weight,
    )
    torch.testing.assert_close(head(features), expected)


def test_no_skip_arm_removes_only_trunk_output_parameters() -> None:
    decoder = exp.GPT(_cfg(ablation_arm="D")).temporal
    assert decoder.trunk_outputs is None
    assert not any(name.startswith("temporal.trunk_outputs.") for name in exp.GPT(_cfg(ablation_arm="D")).state_dict())


def test_ablation_parameter_and_flop_counts_reflect_the_active_model() -> None:
    reports = {}
    for arm in ("A", "B", "C", "D"):
        cfg = _cfg(ablation_arm=arm)
        counts = exp.subsystem_parameter_counts(exp.GPT(cfg))
        reports[arm] = (counts["total"], exp.approximate_training_flops_per_update(cfg, counts))

    assert reports["B"] == reports["A"]
    assert reports["C"][0] < reports["A"][0]
    assert reports["C"][1] < reports["A"][1]
    assert reports["D"][0] < reports["A"][0]
    assert reports["D"][1] < reports["A"][1]


def test_historical_centers_and_complete_controller_grid_round_trip() -> None:
    assert exp.LEGACY_GROUP_VOCABS == (6, 37, 9, 5)
    assert exp.LEGACY_TRIGGER_CENTERS.tolist() == pytest.approx([0.0, 0.35, 0.6, 0.85, 1.0])
    codec = exp.StructuredControllerCodec(16)
    grid = torch.cartesian_prod(*(torch.arange(size) for size in exp.LEGACY_GROUP_VOCABS))
    restored = codec.quantize(codec.dequantize(grid[:, None]))[:, 0]
    assert torch.equal(restored, grid)

    buttons = torch.zeros(6, 4, dtype=torch.long)
    buttons[:, exp.BUTTONS_G] = torch.arange(6)
    decoded = codec.dequantize(buttons)
    assert decoded[:, 6:].sum(-1).tolist() == [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    assert decoded[2, ACTION_CHANNELS.index("button_x")] == 1
    assert decoded[2, ACTION_CHANNELS.index("button_y")] == 0
    assert decoded[4, ACTION_CHANNELS.index("button_l")] == 1
    assert decoded[4, ACTION_CHANNELS.index("button_r")] == 0


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
        "d53bbb5472b9e9e5e3b6641e565ee628d0061fbe6047f4d11344af9ec7356390"
    )


def test_chord_reducer_matches_the_pinned_numpy_implementation() -> None:
    rng = np.random.default_rng(42)
    pressed = rng.integers(0, 2, size=(16, 31, 5), dtype=np.int64)
    actions = torch.zeros(16, 31, A_DIM)
    channels = tuple(
        ACTION_CHANNELS.index(name) for name in ("button_a", "button_b", "button_x", "button_z", "button_l")
    )
    actions[..., channels] = torch.from_numpy(pressed).float()

    expected = []
    for sequence in pressed:
        no_button = (sequence.sum(axis=1, keepdims=True) == 0).astype(np.int64)
        buttons = np.concatenate((sequence.copy(), no_button), axis=1)
        previous: set[int] = set()
        for index in range(len(buttons)):
            current = set(np.flatnonzero(buttons[index]))
            if current and current == previous:
                buttons[index] = buttons[index - 1]
                continue
            new = current - previous
            selected = min(new) if new else -1
            buttons[index] = 0
            buttons[index, selected] = 1
            previous = current
        expected.append(buttons.argmax(axis=1))

    assert np.array_equal(exp.legacy_button_classes(actions).numpy(), np.stack(expected))


def test_held_b_plus_a_reproduces_legacy_early_release() -> None:
    actions = torch.zeros(1, 5, A_DIM)
    a = ACTION_CHANNELS.index("button_a")
    b = ACTION_CHANNELS.index("button_b")
    actions[0, :, b] = 1
    actions[0, 2:4, a] = 1

    codec = exp.StructuredControllerCodec(16)
    indices = codec.quantize(actions)
    assert indices[0, :, exp.BUTTONS_G].tolist() == [1, 1, 0, 0, 5]

    decoded = codec.dequantize(indices)[0].numpy()
    controller = [action_vec_to_controller(frame) for frame in decoded]
    assert [frame.buttons for frame in controller] == [
        BUTTON_BITS["b"],
        BUTTON_BITS["b"],
        BUTTON_BITS["a"],
        BUTTON_BITS["a"],
        0,
    ]


def test_analog_and_digital_shoulders_remain_separate() -> None:
    actions = torch.zeros(1, 3, A_DIM)
    actions[0, 0, exp.TRIGGER_R_CH] = 0.39
    actions[0, 1, exp.TRIGGER_R_CH] = 0.41
    actions[0, 2, exp.BUTTON_R_CH] = 1

    codec = exp.StructuredControllerCodec(16)
    indices = codec.quantize(actions)
    assert indices[0, :, exp.TRIG_G].tolist() == [1, 1, 0]
    assert indices[0, :, exp.BUTTONS_G].tolist() == [5, 5, 4]
    decoded = codec.dequantize(indices)
    assert decoded[0, :, exp.TRIGGER_L_CH].tolist() == pytest.approx([0.35, 0.35, 0.0])
    assert not decoded[..., exp.TRIGGER_R_CH].any()
    assert decoded[0, :, exp.BUTTON_L_CH].tolist() == [0.0, 0.0, 1.0]
    assert not decoded[..., exp.BUTTON_R_CH].any()


@pytest.mark.parametrize("arm", ("A", "B", "C", "D"))
def test_unused_classes_are_masked_in_every_training_and_inference_path(arm: str) -> None:
    cfg = _cfg(ablation_arm=arm)
    model = exp.GPT(cfg).eval()
    hidden = torch.randn(2, cfg.L_ctx, cfg.d_model)
    observed = torch.zeros(2, cfg.L_ctx, exp.N_GROUPS, dtype=torch.long)
    targets = torch.zeros(2, cfg.L_ctx, len(cfg.head_offsets), exp.N_GROUPS, dtype=torch.long)

    parallel = model.temporal.teacher_forced_logits_by_group(hidden, observed, targets)
    _assert_unused_are_masked(parallel)

    for logits in model.temporal.forced_stepwise_logits(hidden, observed[:, -1], targets[:, -1]):
        _assert_unused_are_masked(logits)

    single = model.temporal.sample_indices(hidden, observed[:, -1], cfg.head_offsets[:1], argmax=True)
    assert single.shape == (2, 1, exp.N_GROUPS)

    sampled = model.temporal.sample_indices(hidden, observed[:, -1], cfg.head_offsets[:4], argmax=True)
    for group, valid in enumerate(exp.LEGACY_GROUP_VOCABS):
        assert (sampled[..., group] < valid).all()

    rollout_logits, rollout = model.temporal.rollout_conditioned_logits(hidden, observed[:, -1])
    for logits in rollout_logits:
        _assert_unused_are_masked(logits)
    for group, valid in enumerate(exp.LEGACY_GROUP_VOCABS):
        assert (rollout[..., group] < valid).all()


@pytest.mark.parametrize("arm", ("A", "B", "C", "D"))
def test_parallel_teacher_forcing_matches_stepwise_logits(arm: str) -> None:
    cfg = _cfg(ablation_arm=arm)
    model = exp.GPT(cfg).eval()
    generator = torch.Generator().manual_seed(7)
    hidden = torch.randn(2, cfg.L_ctx, cfg.d_model, generator=generator)
    observed = torch.zeros(2, cfg.L_ctx, exp.N_GROUPS, dtype=torch.long)
    targets = torch.zeros(2, cfg.L_ctx, len(cfg.head_offsets), exp.N_GROUPS, dtype=torch.long)

    parallel = model.temporal.teacher_forced_logits_by_group(hidden, observed, targets)
    stepwise = model.temporal.forced_stepwise_logits(hidden, observed[:, -1], targets[:, -1])
    for depth, logits in enumerate(stepwise):
        for name in exp.GROUP_NAMES:
            torch.testing.assert_close(parallel[name][:, -1, depth], logits[name], atol=2e-5, rtol=2e-5)


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


@pytest.mark.parametrize("arm", ("A", "B", "C", "D"))
def test_cpu_loss_backward_and_checkpoint_round_trip(tmp_path: Path, arm: str) -> None:
    cfg = _cfg(ablation_arm=arm)
    model = exp.GPT(cfg)
    loss = exp.objective(
        exp.action_loss(model, _batch(cfg)),
        cfg.aux_loss_weight,
        cfg.next_frame_loss_share,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())

    checkpoint = tmp_path / f"state-{arm}.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = exp.GPT(cfg)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected)


def test_next_frame_receives_half_of_total_action_loss_weight() -> None:
    nll = torch.ones(1, 10, exp.N_GROUPS, requires_grad=True)
    parts = exp.ActionLoss(nll=nll, targets=torch.empty(0))

    exp.objective(parts, next_frame_loss_share=0.5).backward()

    coefficient = nll.grad[0, :, 0]
    assert coefficient[0] == pytest.approx(1.0)
    assert coefficient[1:].tolist() == pytest.approx([1 / 9] * 9)
    assert coefficient[0] / coefficient.sum() == pytest.approx(0.5)


def test_none_loss_share_restores_dense_four_plus_auxiliary_weights() -> None:
    nll = torch.ones(1, 10, exp.N_GROUPS, requires_grad=True)
    parts = exp.ActionLoss(nll=nll, targets=torch.empty(0))

    exp.objective(parts, next_frame_loss_share=None).backward()

    coefficient = nll.grad[0, :, 0]
    assert coefficient[:4].tolist() == pytest.approx([1 / 4] * 4)
    assert coefficient[4:].tolist() == pytest.approx([1 / 6] * 6)


def test_cody_config_uses_policy_world_train_and_fixed_baseline_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = "archive:///tmp/cody.zip!game.slp"
    replay_paths = tmp_path / "paths.txt"
    replay_paths.write_text(f"{source_path}\n")
    cfg = _cfg(
        data_root="data/processed/professional/cody/mds-policy-world-v7",
        train_replay_paths=str(replay_paths),
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
    replay_filter = calls["train"]["replay_filter"]
    assert replay_filter(exp.policy_replay_identity(source_path))
    assert not replay_filter(exp.policy_replay_identity("archive:///tmp/cody.zip!other.slp"))
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


def test_eval_can_download_upload_and_backfill(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "final.pt"
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(exp, "_resolve_eval_checkpoint", lambda *_args: checkpoint)
    monkeypatch.setattr(exp, "eval_checkpoint", lambda path, **kwargs: calls.append((path, kwargs)))

    exp.main(
        exp.Args(
            eval="final.pt",
            eval_run="run",
            eval_max_parallel=16,
            eval_output_name="eval_backfill_step_0016384_s1",
            eval_backfill_wandb=True,
        )
    )

    assert calls == [
        (
            str(checkpoint),
            {
                "exec_horizon": None,
                "n_matchups": None,
                "eager": False,
                "max_parallel": 16,
                "output_name": "eval_backfill_step_0016384_s1",
                "upload_run": "run",
                "backfill_wandb": True,
            },
        )
    ]


def test_eval_backfills_both_suites_at_the_checkpoint_step(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "final.pt"
    checkpoint.touch()
    cfg = _cfg(final_eval_n_matchups=2)
    model = exp.GPT(cfg)
    suites = {
        "char_matchup": {"scheduled_boots": 2.0, "completed_boots": 2.0, "boots": 2.0},
        "fox": {"scheduled_boots": 2.0, "completed_boots": 2.0, "boots": 2.0},
    }
    calls: dict[str, object] = {}

    monkeypatch.setattr(exp, "load_checkpoint", lambda _path: (model, cfg, {}, {"step": 16_384, "wandb_id": "id"}))
    monkeypatch.setattr(exp, "BF16Inference", lambda _model, _cfg: "inference")
    monkeypatch.setattr(exp, "_checkpoint_sha256", lambda _path: "a" * 64)

    def fake_eval_suites(*_args, **kwargs):
        calls["eval"] = kwargs
        return suites

    monkeypatch.setattr(exp, "eval_suites", fake_eval_suites)
    monkeypatch.setattr(exp, "_upload_eval_evidence", lambda run, path: calls.update(upload=(run, path)))
    monkeypatch.setattr(
        exp,
        "_backfill_eval_metrics",
        lambda wandb_id, step, values, **kwargs: calls.update(backfill=(wandb_id, step, values, kwargs)),
    )

    result = exp.eval_checkpoint(
        str(checkpoint),
        exec_horizon=4,
        max_parallel=16,
        upload_run="run",
        backfill_wandb=True,
    )

    eval_kwargs = calls["eval"]
    assert eval_kwargs["n_matchups"] == 2
    assert eval_kwargs["inference"] == "inference"
    assert eval_kwargs["exec_horizon"] == 4
    assert eval_kwargs["replay_dir"].name == "eval_backfill_step_0016384_s4"
    assert calls["upload"] == ("run", eval_kwargs["replay_dir"])
    assert calls["backfill"] == ("id", 16_384, suites, {"suffix": "_s4"})
    assert result == suites


def test_horizon_backfill_writes_suffixed_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, float]] = []

    class Run:
        summary: dict[str, object] = {}

    monkeypatch.setattr(exp.wandb, "run", Run())
    monkeypatch.setattr(exp.wandb, "init", lambda **kwargs: None)
    monkeypatch.setattr(exp.wandb, "define_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp.wandb, "log", lambda values: calls.append(values))
    monkeypatch.setattr(exp.wandb, "finish", lambda: None)

    exp._backfill_eval_metrics("id", 16_384, {"fox": {"boots": 96.0}}, suffix="_s4")

    assert calls == [
        {
            "global_step": 16_384,
            "eval/backfilled_s4": 1,
            "eval_fox_s4/boots": 96.0,
        }
    ]
    assert exp.wandb.run.summary["evaluation/backfilled_step_s4"] == 16_384


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
