"""Focused contracts for experiment 037's matched factorization matrix."""

import importlib.util
import json
import sys
from dataclasses import asdict
from dataclasses import fields
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

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
exp = _load("test_exp037", _ROOT / "experiments" / "037_factorization_matrix.py")
exp036 = _load("test_exp037_control_036", _ROOT / "experiments" / "036_advantage_weighted_bc.py")
audit037 = _load("test_audit037", _ROOT / "scripts" / "audit_037_results.py")


def _cfg(cell: str = "D3", **overrides):
    values = {
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 4,
        "L_ctx": 4,
        "temporal_d_model": 32,
        "temporal_layers": 1,
        "temporal_heads": 4,
        "temporal_ff_dim": 64,
        "group_head_dim": 64,
        "value_head_hidden_dim": 32,
        "batch_size": 2,
        "reservoir_capacity": 4,
        "warmup_steps": 1,
        "max_steps": 2,
        "compile_trunk": False,
        "compile_temporal": False,
        "num_workers": 0,
        "push_to_r2": False,
        "inference_mode": "eager",
        "latency_iterations": 0,
    }
    cfg = exp.TrainConfig(**{**values, **overrides})
    return exp.config_for_cell(cell, cfg)


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
    context = Context(features=features, ctx_pad=torch.tensor([0, 1], dtype=torch.long))
    return TrainBatch(
        context=context,
        target=_actions(cfg.batch_size, cfg.sample_chunk_length, generator),
    )


def _awr_batch(cfg, *, seed: int = 0):
    batch = _batch(cfg, seed=seed)
    returns = torch.randn(cfg.batch_size, cfg.L_ctx, generator=torch.Generator().manual_seed(seed + 19))
    return exp.AWRBatch(batch=batch, returns=returns, eligible=torch.ones_like(returns, dtype=torch.bool))


def _decoder_inputs(model, cfg, *, seed: int = 0):
    batch = _awr_batch(cfg, seed=seed)
    history, targets, _ = exp.prepared_targets(model, batch)
    hidden = model(batch.context.features, batch.context.ctx_pad, history)
    return batch, hidden, history, targets


def _cfg036(cfg):
    names = {field.name for field in fields(exp036.TrainConfig)}
    return exp036.TrainConfig(**{name: value for name, value in asdict(cfg).items() if name in names})


def _optimizer_fingerprint(model, cfg):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    optimizer = exp.make_optimizer(model, cfg)
    return tuple(
        (
            bool(group["use_muon"]),
            float(group["weight_decay"]),
            tuple(names[id(parameter)] for parameter in group["params"]),
        )
        for group in optimizer.param_groups
    )


def test_d3_same_seed_actor_parameters_and_outputs_match_036() -> None:
    cfg = _cfg("D3")
    torch.manual_seed(0)
    model = exp.GPT(cfg)
    torch.manual_seed(0)
    control = exp036.GPT(_cfg036(cfg))
    ours = exp.actor_state_dict(model)
    theirs = {name: value for name, value in control.state_dict().items() if not name.startswith("value_head.")}
    assert ours.keys() == theirs.keys()
    for name in ours:
        torch.testing.assert_close(ours[name], theirs[name], rtol=0, atol=0, msg=name)

    batch, hidden, history, targets = _decoder_inputs(model, cfg)
    control_hidden = control(batch.context.features, batch.context.ctx_pad, history)
    torch.testing.assert_close(hidden, control_hidden, rtol=0, atol=0)
    logits = model.temporal.teacher_forced_logits_by_group(hidden, history, targets)
    control_logits = control.temporal.teacher_forced_logits_by_group(control_hidden, history, targets)
    for name in exp.GROUP_NAMES:
        torch.testing.assert_close(logits[name], control_logits[name], rtol=0, atol=0)


def test_d3_loads_compatible_036_actor_weights() -> None:
    cfg = _cfg("D3")
    torch.manual_seed(3)
    control = exp036.GPT(_cfg036(cfg))
    torch.manual_seed(9)
    model = exp.GPT(cfg)
    value_before = {name: value.clone() for name, value in model.value_head.state_dict().items()}
    exp.load_036_actor_weights(model, {"model": control.state_dict()})
    for name, value in exp.actor_state_dict(model).items():
        torch.testing.assert_close(value, control.state_dict()[name], rtol=0, atol=0)
    for name, value in model.value_head.state_dict().items():
        torch.testing.assert_close(value, value_before[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="only with the D3"):
        exp.load_036_actor_weights(exp.GPT(_cfg("D2")), control.state_dict())


def test_every_cell_has_the_same_parameters_and_optimizer_ownership() -> None:
    counts = []
    optimizers = []
    for cell in exp.MATRIX_CELLS:
        cfg = _cfg(cell)
        torch.manual_seed(0)
        model = exp.GPT(cfg)
        counts.append(exp.parameter_counts(model))
        optimizers.append(_optimizer_fingerprint(model, cfg))
    assert all(value == counts[0] for value in counts)
    assert all(value == optimizers[0] for value in optimizers)


def test_every_cell_has_the_same_parameters_receiving_gradients() -> None:
    counts = []
    for cell in exp.MATRIX_CELLS:
        cfg = _cfg(cell)
        torch.manual_seed(0)
        model = exp.GPT(cfg)
        batch = _awr_batch(cfg)
        valid_prefixes = int((cfg.L_ctx - batch.context.ctx_pad).sum())
        loss, _, _ = exp.microbatch_loss(
            model,
            batch,
            cfg,
            step=0,
            valid_prefixes=valid_prefixes,
            trunk_fn=lambda features, pad, actions, model=model: model(features, pad, actions),
            temporal_fn=model.temporal.teacher_forced_nll,
        )
        loss.backward()
        counts.append(exp.parameter_counts(model)["receiving_grad"])
    assert counts[0] > 0
    assert all(count == counts[0] for count in counts)


def test_production_cells_have_exact_names_and_fixed_configuration() -> None:
    expected = {
        "D0": "037-D0-future-independent-group-independent-bc-seed0",
        "D1": "037-D1-future-independent-group-ar-bc-seed0",
        "D2": "037-D2-future-ar-group-independent-bc-seed0",
        "D3": "037-D3-future-ar-group-ar-bc-seed0",
    }
    for cell, name in expected.items():
        cfg = exp.config_for_cell(cell)
        exp.validate_production_config(cfg)
        assert exp.production_run_name(cfg) == name
    with pytest.raises(ValueError, match="production matrix configuration mismatch"):
        exp.validate_production_config(exp.config_for_cell("D0", exp.TrainConfig(batch_size=256)))


def test_value_loss_has_no_gradient_path_to_actor_or_trunk() -> None:
    cfg = _cfg("D3")
    torch.manual_seed(0)
    model = exp.GPT(cfg)
    batch, hidden, _, _ = _decoder_inputs(model, cfg)
    valid = torch.arange(cfg.L_ctx)[None, :] >= batch.context.ctx_pad[:, None]
    value_loss = F.mse_loss(model.value_head(hidden.detach()[valid].float()).squeeze(-1), batch.returns[valid])
    actor = [parameter for name, parameter in model.named_parameters() if not name.startswith("value_head.")]
    gradients = torch.autograd.grad(value_loss, actor, allow_unused=True)
    assert all(gradient is None for gradient in gradients)


def test_future_independent_targets_cannot_cross_offsets() -> None:
    cfg = _cfg("D1")
    torch.manual_seed(0)
    model = exp.GPT(cfg)
    _, hidden, history, targets = _decoder_inputs(model, cfg)
    changed = targets.clone()
    changed[..., 0, :] = torch.stack(
        [torch.remainder(changed[..., 0, group] + 1, vocab) for group, vocab in enumerate(exp.GROUP_VOCABS)],
        dim=-1,
    )
    before = model.temporal.teacher_forced_learned_logits_by_group(hidden, history, targets)
    after = model.temporal.teacher_forced_learned_logits_by_group(hidden, history, changed)
    for name in exp.GROUP_NAMES:
        torch.testing.assert_close(before[name][..., 1:, :], after[name][..., 1:, :], rtol=0, atol=0)


def test_future_ar_has_an_earlier_action_path_but_never_a_later_action_path() -> None:
    cfg = _cfg("D3")
    torch.manual_seed(0)
    model = exp.GPT(cfg)
    _, hidden, history, targets = _decoder_inputs(model, cfg)
    earlier = targets.clone()
    earlier[..., 0, exp.C_G] = (earlier[..., 0, exp.C_G] + 1) % exp.GROUP_VOCABS[exp.C_G]
    states = model.temporal.teacher_forced_states(hidden, history, targets)
    states_earlier = model.temporal.teacher_forced_states(hidden, history, earlier)
    assert not torch.allclose(states[..., 1, :], states_earlier[..., 1, :])

    later = targets.clone()
    later[..., -1, exp.C_G] = (later[..., -1, exp.C_G] + 1) % exp.GROUP_VOCABS[exp.C_G]
    before = model.temporal.teacher_forced_learned_logits_by_group(hidden, history, targets)
    after = model.temporal.teacher_forced_learned_logits_by_group(hidden, history, later)
    for name in exp.GROUP_NAMES:
        torch.testing.assert_close(before[name][..., :-1, :], after[name][..., :-1, :], rtol=0, atol=0)


def test_group_independent_uses_null_inputs_while_group_ar_teacher_forces_ancestors() -> None:
    for cell, expect_change in (("D2", False), ("D3", True)):
        cfg = _cfg(cell)
        torch.manual_seed(0)
        model = exp.GPT(cfg)
        _, hidden, history, targets = _decoder_inputs(model, cfg)
        changed = targets.clone()
        depth = 2
        changed[..., depth, exp.C_G] = (changed[..., depth, exp.C_G] + 1) % exp.GROUP_VOCABS[exp.C_G]
        before = model.temporal.teacher_forced_learned_logits_by_group(hidden, history, targets)["main_stick"]
        after = model.temporal.teacher_forced_learned_logits_by_group(hidden, history, changed)["main_stick"]
        differs = not torch.allclose(before[..., depth, :], after[..., depth, :])
        assert differs is expect_change


def test_group_ar_inference_conditions_on_sampled_ancestors(monkeypatch) -> None:
    cfg = _cfg("D3")
    torch.manual_seed(0)
    model = exp.GPT(cfg)
    _, hidden, history, _ = _decoder_inputs(model, cfg)
    observed = history[:, -1]

    def second_group_logits(first_pick: int) -> torch.Tensor:
        calls = []

        def fake_sample(logits, *, argmax, uniform=None, gen=None):
            del argmax, uniform, gen
            calls.append(logits.detach().clone())
            if len(calls) == 1:
                return torch.full(logits.shape[:-1], first_pick, dtype=torch.long)
            return logits.argmax(dim=-1)

        monkeypatch.setattr(exp, "sample_categorical", fake_sample)
        model.temporal.sample_indices(hidden, observed, (1,), argmax=False)
        return calls[1]

    logits_zero = second_group_logits(0)
    logits_one = second_group_logits(1)
    assert not torch.allclose(logits_zero, logits_one)


@pytest.mark.parametrize("cell,uses_sample", (("D0", False), ("D2", True)))
def test_future_inference_uses_sampled_frames_only_in_ar(cell, uses_sample, monkeypatch) -> None:
    cfg = _cfg(cell)
    torch.manual_seed(0)
    model = exp.GPT(cfg)
    _, hidden, history, _ = _decoder_inputs(model, cfg)
    observed = history[:, -1]
    seen_previous = []
    original = exp.CausalTemporalDecoder._step_features

    def record_previous(self, previous, offsets):
        seen_previous.append(previous.detach().clone())
        return original(self, previous, offsets)

    monkeypatch.setattr(exp.CausalTemporalDecoder, "_step_features", record_previous)
    monkeypatch.setattr(
        exp,
        "sample_categorical",
        lambda logits, **kwargs: torch.ones(logits.shape[:-1], dtype=torch.long),
    )
    model.temporal.sample_indices(hidden, observed, (1, 2), argmax=False)
    torch.testing.assert_close(seen_previous[0], observed)
    expected = torch.ones_like(observed) if uses_sample else observed
    torch.testing.assert_close(seen_previous[1], expected)


def test_legality_mask_is_identical_and_sampled_actions_are_legal() -> None:
    masks = []
    for cell in exp.MATRIX_CELLS:
        cfg = _cfg(cell)
        torch.manual_seed(0)
        model = exp.GPT(cfg)
        masks.append(model.codec.button_valid_for_trigger.clone())
        _, hidden, history, _ = _decoder_inputs(model, cfg)
        sampled = model.temporal.sample_indices(hidden, history[:, -1], (1, 2), argmax=True)
        assert model.codec.button_valid_for_trigger[sampled[..., exp.TRIG_G], sampled[..., exp.BUTTONS_G]].all()
    assert all(torch.equal(mask, masks[0]) for mask in masks)


def test_group_losses_sum_to_joint_nll_and_offset_normalization_matches() -> None:
    objective_values = []
    for cell in exp.MATRIX_CELLS:
        cfg = _cfg(cell)
        torch.manual_seed(0)
        model = exp.GPT(cfg)
        _, hidden, history, targets = _decoder_inputs(model, cfg)
        logits = model.temporal.teacher_forced_logits_by_group(hidden, history, targets)
        losses = model.temporal.teacher_forced_nll(hidden, history, targets)
        manual = []
        for group, name in enumerate(exp.GROUP_NAMES):
            manual.append(
                F.cross_entropy(
                    logits[name].float().reshape(-1, exp.GROUP_VOCABS[group]),
                    targets[..., group].reshape(-1),
                    reduction="none",
                ).view(*targets.shape[:-1])
            )
        torch.testing.assert_close(losses, torch.stack(manual, dim=-1))
        random_nll = torch.arange(7 * 10 * exp.N_GROUPS, dtype=torch.float32).view(7, 10, exp.N_GROUPS) / 100
        objective_values.append(
            exp.advantage_weighted_objective(
                random_nll,
                torch.ones(7),
                scope=cfg.advantage_scope,
                valid_prefixes=7,
                aux_loss_weight=cfg.aux_loss_weight,
            )
        )
    assert all(torch.equal(value, objective_values[0]) for value in objective_values)


def test_checkpoint_round_trip_preserves_flags_and_outputs(tmp_path: Path) -> None:
    cfg = _cfg("D1")
    torch.manual_seed(0)
    model = exp.GPT(cfg).eval()
    _, hidden, history, targets = _decoder_inputs(model, cfg)
    before = model.temporal.teacher_forced_logits_by_group(hidden, history, targets)
    path = tmp_path / "checkpoint.pt"
    torch.save({"cfg": exp._checkpoint_config(cfg), "model": model.state_dict()}, path)
    state = torch.load(path, weights_only=False)
    restored_cfg = exp.config_from_state(state["cfg"])
    assert restored_cfg.future_conditioning == "independent"
    assert restored_cfg.group_conditioning == "autoregressive"
    restored = exp.GPT(restored_cfg).eval()
    restored.load_state_dict(state["model"])
    restored_hidden = restored(_batch(cfg).context.features, _batch(cfg).context.ctx_pad, history)
    after = restored.temporal.teacher_forced_logits_by_group(restored_hidden, history, targets)
    for name in exp.GROUP_NAMES:
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)


def test_sampling_streams_are_keyed_by_slot_frame_and_group() -> None:
    def context(slot_ids, reset):
        return Context(
            features={},
            ctx_pad=torch.zeros(len(slot_ids), dtype=torch.long),
            slot_ids=torch.tensor(slot_ids),
            reset=torch.tensor(reset),
        )

    one = exp.SlotGroupRandom(7)
    one.begin(context([11, 22], [True, True]))
    whole = {(frame, group): one.uniforms(frame, group) for frame in range(6) for group in exp.GROUP_NAMES}
    one.advance(6)

    split = exp.SlotGroupRandom(7)
    split.begin(context([11, 22], [True, True]))
    parts = {(frame, group): split.uniforms(frame, group) for frame in range(2) for group in exp.GROUP_NAMES}
    split.advance(2)
    split.begin(context([22, 11], [False, False]))
    for local_frame in range(4):
        for group in exp.GROUP_NAMES:
            value = split.uniforms(local_frame, group)
            torch.testing.assert_close(value.flip(0), whole[(local_frame + 2, group)], rtol=0, atol=0)
    for key, value in parts.items():
        torch.testing.assert_close(value, whole[key], rtol=0, atol=0)


def test_evaluation_horizon_does_not_modify_checkpoint_configuration(monkeypatch, tmp_path: Path) -> None:
    cfg = _cfg("D3")
    saved = exp._checkpoint_config(cfg)
    model = exp.GPT(cfg)
    state = {"step": 2, "cfg": saved}
    monkeypatch.setattr(exp, "load_checkpoint", lambda path: (model, cfg, {}, state))
    monkeypatch.setattr(exp, "_checkpoint_sha256", lambda path: "a" * 64)
    seen = {}

    def fake_eval(model, stats, runtime_cfg, *, exec_horizon, **kwargs):
        del model, stats, kwargs
        seen["runtime_exec_horizon"] = runtime_cfg.exec_horizon
        seen["requested_horizon"] = exec_horizon
        return {"net_stock_lcb": 0.0}

    monkeypatch.setattr(exp, "eval_vs_cpu", fake_eval)
    exp.eval_checkpoint(str(tmp_path / "final.pt"), exec_horizon=1, eager=True)
    assert seen == {"runtime_exec_horizon": 4, "requested_horizon": 1}
    assert exp._checkpoint_config(cfg) == saved


def test_metrics_artifact_contains_the_horizon_and_protocol_digest(tmp_path: Path) -> None:
    cfg = _cfg("D1")
    model = exp.GPT(cfg)
    protocol = exp._eval_protocol(
        cfg,
        model,
        n_matchups=96,
        exec_horizon=2,
        checkpoint_sha256="b" * 64,
    )
    exp._write_eval_evidence(tmp_path, [], {"net_stock_lcb": 0.25}, protocol)
    payload = json.loads((tmp_path / "metrics.json").read_text())
    assert payload["protocol"]["exec_horizon"] == 2
    assert payload["protocol"]["future_conditioning"] == "independent"
    assert payload["protocol"]["group_conditioning"] == "autoregressive"
    assert payload["protocol"]["protocol_sha256"] == protocol.protocol_sha256
    assert payload["metrics"]["net_stock_lcb"] == 0.25


def test_evaluator_never_starts_more_dolphins_than_cpus(monkeypatch) -> None:
    cfg = _cfg("D3", eval_max_parallel=32)
    monkeypatch.setattr(exp, "usable_cpus", lambda: 16)
    assert exp._eval_parallelism(cfg, 96) == 16
    monkeypatch.setattr(exp, "usable_cpus", lambda: 10)
    assert exp._eval_parallelism(cfg, 96) == 8


def test_finite_learned_logit_audit_keeps_hard_support_separate() -> None:
    cfg = _cfg("D0")
    model = exp.GPT(cfg)
    values = audit037.finite_learned_logit_metrics(exp, model, [_batch(cfg)], cfg)
    assert all(torch.isfinite(torch.tensor(value)) for value in values.values())
    assert values["gap"] == pytest.approx(0.0, abs=1e-6)
    assert 0.0 <= values["target_button_support_rate"] <= 1.0


def test_experiment_source_is_self_contained() -> None:
    source = (_ROOT / "experiments" / "037_factorization_matrix.py").read_text()
    assert "spec_from_file_location" not in source
    assert "experiments/036" not in source


@pytest.mark.parametrize("cell", tuple(exp.MATRIX_CELLS))
def test_small_end_to_end_training_works_for_every_cell(cell, monkeypatch, tmp_path: Path) -> None:
    # CUDA flex attention requires at least 16 channels per head.
    cfg = _cfg(
        cell,
        d_model=64,
        max_steps=1,
        val_every=0,
        eval_every=0,
        ckpt_every=0,
        wandb_log_code=False,
    )
    batch = _awr_batch(cfg)
    monkeypatch.setattr(exp, "_make_loaders", lambda cfg, stats: ([batch], [batch.batch]))
    monkeypatch.setattr(exp, "setup_run_dir", lambda name: (tmp_path, tmp_path / "replays"))
    monkeypatch.setattr(exp, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp, "_checkpoint_sha256", lambda path: "a" * 64)
    monkeypatch.setattr(exp, "BF16Inference", lambda model, cfg: object())
    evaluations = []

    def fake_eval(model, stats, cfg, *, n_matchups, replay_dir, exec_horizon=None, **kwargs):
        del model, stats, n_matchups, replay_dir, kwargs
        evaluations.append(cfg.exec_horizon if exec_horizon is None else exec_horizon)
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
    exp.train(cfg, {}, requested_run_name=f"tiny-{cell}")
    assert evaluations == list(exp.FINAL_EVAL_HORIZONS)
    assert any("train/value_loss" in values for values in logs)
