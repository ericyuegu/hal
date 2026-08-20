"""Focused contracts for experiment 036's advantage-weighted loss treatment."""

import functools
import importlib.util
import math
import sys
from dataclasses import asdict
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import torch

import hal.training.replay_reservoir as replay_reservoir
from hal.training import returns as returns_lib
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import Context
from hal.training.features import FeatureProjection
from hal.training.features import TrainBatch
from hal.training.features import _classify
from hal.training.replay_reservoir import PolicyReplayPackDataset

_PATH = Path(__file__).resolve().parents[2] / "experiments" / "036_advantage_weighted_bc.py"
_SPEC = importlib.util.spec_from_file_location("test_exp036", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
exp = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = exp
_SPEC.loader.exec_module(exp)

# The 026 control, loaded ONLY for comparison tests. The experiment file itself
# is self-contained and never imports another experiment.
_PATH_026 = Path(__file__).resolve().parents[2] / "experiments" / "026_temporal_mtp.py"
_SPEC_026 = importlib.util.spec_from_file_location("test_exp036_control_026", _PATH_026)
assert _SPEC_026 is not None and _SPEC_026.loader is not None
exp026 = importlib.util.module_from_spec(_SPEC_026)
sys.modules[_SPEC_026.name] = exp026
_SPEC_026.loader.exec_module(exp026)


def _cfg(**overrides) -> exp.TrainConfig:
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
        "awr_warmup_steps": 1,
        "max_steps": 2,
        "compile_trunk": False,
        "compile_temporal": False,
        "num_workers": 0,
        "push_to_r2": False,
        "inference_mode": "eager",
    }
    return exp.TrainConfig(**{**values, **overrides})


def _actions(batch: int, length: int, generator: torch.Generator) -> torch.Tensor:
    actions = torch.empty(batch, length, A_DIM)
    actions[..., :4] = torch.rand(batch, length, 4, generator=generator) * 2 - 1
    actions[..., 4:6] = torch.rand(batch, length, 2, generator=generator)
    actions[..., 6:] = torch.randint(0, 2, (batch, length, 8), generator=generator).float()
    return actions


def _batch(cfg, pads: list[int] | None = None, seed: int = 0) -> TrainBatch:
    pads = [0, 1] if pads is None else pads
    batch = len(pads)
    generator = torch.Generator().manual_seed(seed)
    ctx = exp.synthetic_context(cfg, batch, torch.device("cpu"))
    features = dict(ctx.features)
    native = _actions(batch, cfg.L_ctx, generator)
    for channel, values in zip(ACTION_CHANNELS, native.unbind(-1), strict=True):
        features[f"ego_{channel}"] = values
    context = Context(features=features, ctx_pad=torch.tensor(pads, dtype=torch.int64))
    return TrainBatch(context=context, target=_actions(batch, cfg.sample_chunk_length, generator))


def _awr_batch(
    cfg,
    pads: list[int] | None = None,
    eligible_rows: list[bool] | None = None,
    seed: int = 0,
) -> exp.AWRBatch:
    batch = _batch(cfg, pads, seed)
    rows = batch.target.shape[0]
    eligible_rows = [True] * rows if eligible_rows is None else eligible_rows
    generator = torch.Generator().manual_seed(seed + 7)
    returns = torch.randn(rows, cfg.L_ctx, generator=generator)
    eligible = torch.tensor(eligible_rows)[:, None].expand(rows, cfg.L_ctx).clone()
    returns[~eligible] = float("nan")
    return exp.AWRBatch(batch=batch, returns=returns, eligible=eligible)


def _loss_fns(model):
    return dict(
        trunk_fn=lambda features, pad, actions: model(features, pad, actions),
        temporal_fn=model.temporal.teacher_forced_nll,
    )


def _valid_prefixes(cfg, batch) -> int:
    return int((cfg.L_ctx - batch.context.ctx_pad).sum())


# --- alignment and data plumbing -------------------------------------------------


def test_collate_pairs_position_t_with_g_t_plus_1() -> None:
    cfg = _cfg()
    length = cfg.L_ctx + cfg.sample_chunk_length
    windows = [
        {
            exp.EGO_RETURN: np.arange(length, dtype=np.float32),
            exp.EGO_RETURN_VALID: np.ones(length, dtype=np.bool_),
        }
    ]
    collated = exp.collate_awr_batch(windows, _batch(cfg, [0]), L_ctx=cfg.L_ctx)
    np.testing.assert_array_equal(collated.returns.numpy()[0], np.arange(1, cfg.L_ctx + 1, dtype=np.float32))
    assert collated.eligible.all()


def test_return_labels_come_from_the_full_replay_and_add_only_two_columns(monkeypatch) -> None:
    frames = 40
    label_kwargs = dict(gamma=0.9, damage_shaping=0.01, win_reward=0.5, suffix=exp._RETURN_SUFFIX)
    full = {
        "schema_version": 7,
        "frame": np.arange(frames, dtype=np.int32),
        "p1_value": np.arange(frames, dtype=np.float32),
        "p2_value": np.arange(frames, dtype=np.float32) + 1000.0,
        "p1_stock": np.where(np.arange(frames) < 30, 2, 1).astype(np.int32),
        "p2_stock": np.where(np.arange(frames) < frames - 1, 3, 0).astype(np.int32),
        "p1_percent": np.arange(frames, dtype=np.float32) * 0.5,
        "p2_percent": np.arange(frames, dtype=np.float32) * 0.25,
    }

    def fake_full(source):
        del source
        return {name: value.copy() if isinstance(value, np.ndarray) else value for name, value in full.items()}

    def fake_slices(source, ranges):
        del source
        return tuple(
            {name: value[start:stop] if isinstance(value, np.ndarray) else value for name, value in full.items()}
            for start, stop in ranges
        )

    monkeypatch.setattr(replay_reservoir, "decode_policy_replay", fake_full)
    monkeypatch.setattr(replay_reservoir, "decode_policy_replay_slices", fake_slices)
    compact = {"replay_id": "d" * 32, "source_schema_version": 7, "num_frames": frames}
    length = 8 + 4
    common = {"L_ctx": 8, "L_chunk": 4, "seed": 13, "windows_per_replay": 3, "schema_version": 7}
    control_projection = FeatureProjection(frozenset({"ego_value"}), derive_spatial=False)
    widened = FeatureProjection(frozenset({"ego_value", exp.EGO_RETURN, exp.EGO_RETURN_VALID}), derive_spatial=False)
    control = next(iter(PolicyReplayPackDataset([compact], projection=control_projection, **common)))
    labeled_pack = next(
        iter(
            PolicyReplayPackDataset(
                [dict(compact)],
                projection=widened,
                replay_transform=functools.partial(returns_lib.label_replay, **label_kwargs),
                **common,
            )
        )
    )

    assert labeled_pack.replay_id == control.replay_id
    assert len(labeled_pack.windows) == len(control.windows)
    expected_labels = returns_lib.replay_returns(fake_full(None), **label_kwargs)
    for plain, labeled in zip(control.windows, labeled_pack.windows, strict=True):
        assert labeled.keys() == plain.keys() | {exp.EGO_RETURN, exp.EGO_RETURN_VALID}
        for name in plain:
            np.testing.assert_array_equal(labeled[name], plain[name])
        last = float(labeled["ego_value"][-1])
        ego = "p2" if last >= 1000 else "p1"
        start = int(last % 1000) - (length - 1)
        pad = max(0, -start)
        expected = np.concatenate(
            [np.zeros(pad, dtype=np.float32), expected_labels[f"{ego}_awr_return"][max(0, start) : start + length]]
        )
        np.testing.assert_allclose(labeled[exp.EGO_RETURN], expected, atol=1e-6)
        np.testing.assert_array_equal(labeled[exp.EGO_RETURN_VALID][pad:], True)
        np.testing.assert_array_equal(labeled[exp.EGO_RETURN_VALID][:pad], False)
        assert int(labeled["ctx_pad"]) == min(pad, 8)


def test_return_columns_never_reach_the_model_and_loaders_widen_the_projection(monkeypatch) -> None:
    assert _classify(exp.EGO_RETURN) == "drop"
    assert _classify(exp.EGO_RETURN_VALID) == "drop"

    cfg = _cfg()
    captured: dict[str, dict] = {}

    def fake_reservoir(**kwargs):
        captured["train"] = kwargs
        return ["train-loader"]

    def fake_loader(**kwargs):
        captured["val"] = kwargs
        return ["val-loader"]

    monkeypatch.setattr(exp, "make_reservoir_loader", lambda **kwargs: fake_reservoir(**kwargs))
    monkeypatch.setattr(exp, "make_loader", lambda **kwargs: fake_loader(**kwargs))
    monkeypatch.setattr(exp, "cache_validation", lambda loader, n: loader)
    train_loader, val_cache = exp._make_loaders(cfg, stats={})
    assert train_loader == ["train-loader"] and val_cache == ["val-loader"]

    train = captured["train"]
    assert {exp.EGO_RETURN, exp.EGO_RETURN_VALID} <= train["projection"].columns
    assert train["replay_format"] == "policy"
    assert train["replay_transform"].func is returns_lib.label_replay
    assert train["replay_transform"].keywords["gamma"] == cfg.awr_gamma
    assert train["batch_transform"].func is exp.collate_awr_batch
    val = captured["val"]
    assert "replay_transform" not in val
    assert val["batch_size"] == cfg.val_batch_size


def test_valid_rows_flatten_by_the_same_mask_as_the_nll() -> None:
    cfg = _cfg()
    batch = _awr_batch(cfg, pads=[2, 0], eligible_rows=[True, False])
    valid = torch.arange(cfg.L_ctx)[None, :] >= batch.context.ctx_pad[:, None]
    returns, eligible = batch.valid_rows(valid)
    assert returns.shape == eligible.shape == (2 + 4,)
    torch.testing.assert_close(returns[:2], batch.returns[0, 2:])
    assert eligible[:2].all() and not eligible[2:].any()
    assert torch.isnan(returns[2:]).all()


# --- weights ---------------------------------------------------------------------


def test_weights_match_the_hand_computation_and_normalize_to_mean_one() -> None:
    advantage = torch.tensor([0.8, -0.8])
    eligible = torch.ones(2, dtype=torch.bool)
    weight, stats = exp.advantage_weights(advantage, eligible, beta=0.8, weight_max=5.0)
    raw = torch.tensor([math.e, 1 / math.e])
    torch.testing.assert_close(weight, raw / raw.mean())
    assert float(weight.mean()) == pytest.approx(1.0)
    expected_ess = float(raw.sum() ** 2 / (2 * (raw**2).sum()))
    assert float(stats["weight_ess"]) == pytest.approx(expected_ess)
    assert float(stats["weight_clip_frac"]) == 0.0


def test_truncated_rows_keep_unit_weight_and_all_negative_advantages_stay_finite() -> None:
    advantage = torch.tensor([-100.0, -200.0, float("nan")])
    eligible = torch.tensor([True, True, False])
    weight, stats = exp.advantage_weights(advantage, eligible, beta=0.8, weight_max=5.0)
    assert torch.isfinite(weight).all()
    assert float(weight[2]) == 1.0
    assert float(weight[:2].mean()) == pytest.approx(1.0, abs=1e-5)
    assert float(stats["eligible_frac"]) == pytest.approx(2 / 3)


def test_raw_cap_applies_before_normalization_and_never_again_after() -> None:
    advantage = torch.tensor([10.0] + [-10.0] * 9)
    eligible = torch.ones(10, dtype=torch.bool)
    weight, stats = exp.advantage_weights(advantage, eligible, beta=0.8, weight_max=5.0)
    assert float(stats["weight_clip_frac"]) == pytest.approx(0.1)
    assert float(stats["weight_norm_max"]) > 5.0  # normalization can exceed the raw cap
    assert float(weight.max()) == pytest.approx(float(stats["weight_norm_max"]))
    assert float(weight.mean()) == pytest.approx(1.0)


def test_weights_reject_gradient_carrying_advantages_and_nonfinite_eligible_rows() -> None:
    with pytest.raises(ValueError, match="DETACHED"):
        exp.advantage_weights(torch.ones(2, requires_grad=True), torch.ones(2, dtype=torch.bool), beta=1, weight_max=5)
    with pytest.raises(FloatingPointError, match="non-finite"):
        exp.advantage_weights(
            torch.tensor([float("nan"), 0.0]), torch.ones(2, dtype=torch.bool), beta=1.0, weight_max=5.0
        )


def test_no_eligible_rows_fall_back_to_all_ones() -> None:
    weight, stats = exp.advantage_weights(
        torch.full((3,), float("nan")), torch.zeros(3, dtype=torch.bool), beta=0.8, weight_max=5.0
    )
    torch.testing.assert_close(weight, torch.ones(3))
    assert float(stats["eligible_frac"]) == 0.0


# --- objective -------------------------------------------------------------------


def test_unit_weights_reproduce_the_026_objective_exactly() -> None:
    nll = torch.randn(7, 10, exp.N_GROUPS, generator=torch.Generator().manual_seed(3)).abs()
    for scope in ("primary", "all"):
        ours = exp.advantage_weighted_objective(nll, torch.ones(7), scope=scope, valid_prefixes=7, aux_loss_weight=0.5)
        joint = nll.sum(dim=-1)
        expected = joint[:, :4].mean() + 0.5 * joint[:, 4:].mean()
        torch.testing.assert_close(ours, expected)


def test_primary_scope_leaves_the_auxiliary_gradient_unweighted() -> None:
    generator = torch.Generator().manual_seed(4)
    values = torch.randn(5, 10, exp.N_GROUPS, generator=generator).abs()
    weight = torch.rand(5, generator=generator) + 0.5
    weight = weight / weight.mean()

    def gradients(scope: str, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nll = values.clone().requires_grad_()
        exp.advantage_weighted_objective(nll, w, scope=scope, valid_prefixes=5, aux_loss_weight=1.0).backward()
        return nll.grad[:, :4].clone(), nll.grad[:, 4:].clone()

    primary_w, aux_w = gradients("primary", weight)
    primary_1, aux_1 = gradients("primary", torch.ones(5))
    assert not torch.allclose(primary_w, primary_1)
    torch.testing.assert_close(aux_w, aux_1)
    _, aux_all = gradients("all", weight)
    assert not torch.allclose(aux_all, aux_1)


def test_one_weight_multiplies_every_controller_group_at_a_position() -> None:
    weight = torch.tensor([3.0, 0.5])
    for group in range(exp.N_GROUPS):
        nll = torch.zeros(2, 10, exp.N_GROUPS)
        nll[:, :4, group] = 1.0
        value = exp.advantage_weighted_objective(nll, weight, scope="primary", valid_prefixes=2, aux_loss_weight=1.0)
        torch.testing.assert_close(value, torch.tensor(weight.sum().item() * 4 / (2 * 4)))


def test_objective_is_shape_agnostic_across_offset_variants() -> None:
    for n_offsets in (6, 8, 10):
        nll = torch.rand(3, n_offsets, exp.N_GROUPS)
        value = exp.advantage_weighted_objective(
            nll, torch.ones(3), scope="all", valid_prefixes=3, aux_loss_weight=1.0
        )
        assert torch.isfinite(value)
    with pytest.raises(ValueError, match="beyond the dense primary prefix"):
        exp.advantage_weighted_objective(
            torch.rand(3, 4, exp.N_GROUPS), torch.ones(3), scope="all", valid_prefixes=3, aux_loss_weight=1.0
        )
    with pytest.raises(ValueError, match="advantage scope"):
        exp.advantage_weighted_objective(
            torch.rand(3, 10, exp.N_GROUPS), torch.ones(3), scope="auxiliary", valid_prefixes=3, aux_loss_weight=1.0
        )


# --- decoder conditioning --------------------------------------------------------


def test_additive_token_projection_matches_the_concat_reference() -> None:
    cfg = _cfg()
    torch.manual_seed(0)
    decoder = exp.GPT(cfg).temporal
    generator = torch.Generator().manual_seed(1)
    horizon = len(cfg.head_offsets)
    hidden = torch.randn(2, cfg.L_ctx, cfg.d_model, generator=generator)
    previous = torch.stack(
        [torch.randint(0, vocab, (2, cfg.L_ctx, horizon), generator=generator) for vocab in exp.GROUP_VOCABS],
        dim=-1,
    )
    trunk = exp.decoder_rmsnorm(hidden)
    offsets = torch.tensor(cfg.head_offsets)
    ours = decoder._state_bias(trunk)[:, :, None] + decoder._step_features(previous, offsets)
    action = decoder.codec.embed_frame(previous)
    offset = decoder.offset_embedding(offsets).view(1, 1, horizon, -1).expand(2, cfg.L_ctx, -1, -1)
    reference = decoder.token_projection(
        torch.cat((trunk[:, :, None].expand(-1, -1, horizon, -1), action, offset), dim=-1)
    )
    torch.testing.assert_close(ours, reference, atol=1e-5, rtol=1e-5)


def test_zero_initialized_film_is_exact_identity_then_departs() -> None:
    cfg = _cfg()
    torch.manual_seed(0)
    baseline = exp.GPT(cfg)
    torch.manual_seed(0)
    filmed = exp.GPT(exp.TrainConfig(**{**asdict(cfg), "temporal_state_film": True}))
    assert baseline.temporal.state_film is None
    assert "temporal.state_film.weight" in filmed.state_dict()

    batch = _awr_batch(cfg)
    history, targets, _ = exp.prepared_targets(baseline, batch)
    base_nll = baseline.temporal.teacher_forced_nll(
        baseline(batch.context.features, batch.context.ctx_pad, history), history, targets
    )
    film_nll = filmed.temporal.teacher_forced_nll(
        filmed(batch.context.features, batch.context.ctx_pad, history), history, targets
    )
    torch.testing.assert_close(base_nll, film_nll)  # zero-init FiLM is the identity

    with torch.no_grad():
        filmed.temporal.state_film.weight.normal_(std=0.1)
        filmed.temporal.state_film.bias.normal_(std=0.1)
    moved = filmed.temporal.teacher_forced_nll(
        filmed(batch.context.features, batch.context.ctx_pad, history), history, targets
    )
    assert not torch.allclose(base_nll, moved)


def test_film_keeps_parallel_and_stepwise_paths_in_parity() -> None:
    cfg = exp.TrainConfig(**{**asdict(_cfg()), "temporal_state_film": True})
    torch.manual_seed(0)
    model = exp.GPT(cfg)
    with torch.no_grad():
        model.temporal.state_film.weight.normal_(std=0.1)
        model.temporal.state_film.bias.normal_(std=0.1)
    batch = _awr_batch(cfg)
    history, targets, _ = exp.prepared_targets(model, batch)
    hidden = model(batch.context.features, batch.context.ctx_pad, history)
    parallel = model.temporal.teacher_forced_logits(hidden, history, targets)
    stepwise = model.temporal.forced_stepwise_logits(hidden, history[:, -1], targets[:, -1])
    for depth in range(len(cfg.head_offsets)):
        for name in exp.GROUP_NAMES:
            torch.testing.assert_close(parallel[depth][name][:, -1], stepwise[depth][name], atol=1e-5, rtol=1e-4)


# --- the loss seam ---------------------------------------------------------------


def _policy_state(model) -> dict[str, torch.Tensor]:
    return {name: value for name, value in model.state_dict().items() if not name.startswith("value_head.")}


def _control_model(cfg) -> exp026.GPT:
    names = {field.name for field in fields(exp026.TrainConfig)}
    control_cfg = exp026.TrainConfig(**{name: value for name, value in asdict(cfg).items() if name in names})
    return exp026.GPT(control_cfg)


def test_same_seed_policy_parameters_match_026_and_optimizer_owns_the_value_head() -> None:
    cfg = _cfg()
    torch.manual_seed(0)
    model = exp.GPT(cfg)
    torch.manual_seed(0)
    control = _control_model(cfg)
    ours = _policy_state(model)
    theirs = control.state_dict()
    assert ours.keys() == theirs.keys()
    for name in ours:
        torch.testing.assert_close(ours[name], theirs[name], msg=name)

    optimizer = exp.make_optimizer(model, cfg)
    owned = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(owned) == len({id(parameter) for parameter in owned}) == sum(1 for _ in model.parameters())
    muon_ids = {
        id(parameter) for group in optimizer.param_groups if group["use_muon"] for parameter in group["params"]
    }
    assert all(id(parameter) not in muon_ids for parameter in model.value_head.parameters())


def test_warmup_step_is_exact_bc_while_the_value_head_still_learns() -> None:
    cfg = _cfg()
    torch.manual_seed(0)
    model = exp.GPT(cfg)
    torch.manual_seed(0)
    control = _control_model(cfg)
    batch = _awr_batch(cfg, eligible_rows=[True, False])
    prefixes = _valid_prefixes(cfg, batch)

    loss, _, extra = exp.microbatch_loss(model, batch, cfg, step=0, valid_prefixes=prefixes, **_loss_fns(model))
    assert float(extra["awr/active"]) == 0.0
    parts = exp026.action_loss(control, batch.batch)
    expected = exp026.objective(parts, cfg.aux_loss_weight)
    # 036 computes the token projection additively (same weights, reordered float
    # sums), so equality with 026 is near-exact, not bitwise.
    tolerance = dict(atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(loss - extra["train/value_loss"] * cfg.awr_value_loss_weight, expected, **tolerance)

    loss.backward()
    expected.backward()
    control_grads = dict(control.named_parameters())
    for name, parameter in model.named_parameters():
        if name.startswith("value_head."):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert float(parameter.grad.abs().sum()) > 0
        else:
            torch.testing.assert_close(parameter.grad, control_grads[name].grad, msg=name, **tolerance)


def test_value_mse_with_a_detached_trunk_never_moves_policy_parameters() -> None:
    cfg = _cfg()

    def policy_grads(value_loss_weight: float, detach: bool) -> dict[str, torch.Tensor]:
        variant = exp.TrainConfig(
            **{**asdict(cfg), "awr_value_loss_weight": value_loss_weight, "awr_value_detach_trunk": detach}
        )
        torch.manual_seed(0)
        model = exp.GPT(variant)
        batch = _awr_batch(variant)
        loss, _, _ = exp.microbatch_loss(
            model, batch, variant, step=0, valid_prefixes=_valid_prefixes(variant, batch), **_loss_fns(model)
        )
        loss.backward()
        return {
            name: parameter.grad.clone()
            for name, parameter in model.named_parameters()
            if not name.startswith("value_head.") and parameter.grad is not None
        }

    with_value = policy_grads(1.0, detach=True)
    without_value = policy_grads(0.0, detach=True)
    assert with_value.keys() == without_value.keys()
    for name in with_value:
        torch.testing.assert_close(with_value[name], without_value[name], msg=name)

    shared = policy_grads(1.0, detach=False)
    assert any(not torch.allclose(shared[name], without_value[name]) for name in shared)


def test_truncated_batch_takes_ordinary_bc_and_no_value_gradient() -> None:
    cfg = _cfg()
    torch.manual_seed(0)
    model = exp.GPT(cfg)
    batch = _awr_batch(cfg, eligible_rows=[False, False])
    loss, nll, extra = exp.microbatch_loss(
        model, batch, cfg, step=cfg.awr_warmup_steps, valid_prefixes=_valid_prefixes(cfg, batch), **_loss_fns(model)
    )
    assert torch.isfinite(loss)
    assert float(extra["train/value_loss"]) == 0.0
    assert float(extra["train/eligible_frac"]) == 0.0
    loss.backward()
    for parameter in model.value_head.parameters():
        assert parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0
    assert torch.isfinite(nll).all()


def test_advantage_is_detached_from_both_actor_and_value_parameters() -> None:
    cfg = _cfg()
    torch.manual_seed(0)
    model = exp.GPT(cfg)
    batch = _awr_batch(cfg)
    history, targets, valid = exp.prepared_targets(model, batch)
    hidden = model(batch.context.features, batch.context.ctx_pad, history)
    value = model.value_head(hidden.detach()[valid].float()).squeeze(-1)
    returns, _ = batch.valid_rows(valid)
    advantage = (returns - value).detach()
    assert not advantage.requires_grad


def test_offset_variants_pass_validation_and_flow_through_the_loss() -> None:
    for offsets in ((1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 6, 12, 20)):
        cfg = _cfg(head_offsets=offsets)
        exp.validate_config(cfg)
        torch.manual_seed(0)
        model = exp.GPT(cfg)
        batch = _awr_batch(cfg)
        loss, nll, _ = exp.microbatch_loss(
            model, batch, cfg, step=cfg.max_steps - 1, valid_prefixes=_valid_prefixes(cfg, batch), **_loss_fns(model)
        )
        assert nll.shape[1] == len(offsets)
        assert torch.isfinite(loss)


def test_config_gates() -> None:
    with pytest.raises(ValueError, match="advantage_scope"):
        exp.validate_config(_cfg(advantage_scope="offset1"))
    with pytest.raises(ValueError, match="awr_warmup_steps"):
        exp.validate_config(_cfg(awr_warmup_steps=2, max_steps=2))
    with pytest.raises(TypeError, match="AWRBatch"):
        cfg = _cfg()
        model = exp.GPT(cfg)
        exp.microbatch_loss(model, _batch(cfg), cfg, step=0, valid_prefixes=7, **_loss_fns(model))


def test_fixed_policies_are_not_config_fields() -> None:
    removed = {
        "compact_data",
        "decode_temp",
        "decoder_arch_version",
        "final_diag_exec_horizon",
        "grad_accum_steps",
        "group_order",
        "inference_buckets",
        "experiment_id",
        "require_flex",
        "wandb_grad_every",
    }
    assert removed.isdisjoint(field.name for field in fields(exp.TrainConfig))


def test_checkpoint_config_round_trip_and_identity_rejections() -> None:
    cfg = _cfg(advantage_scope="all", awr_beta=1.6)
    checkpoint_cfg = exp._checkpoint_config(cfg)
    assert checkpoint_cfg["experiment_id"] == exp._EXPERIMENT_ID
    restored = exp.config_from_state(checkpoint_cfg)
    assert restored == cfg
    legacy_cfg = {
        **checkpoint_cfg,
        "compact_data": True,
        "decoder_arch_version": 3,
        "grad_accum_steps": 1,
        "group_order": exp.GROUP_ORDER,
        "require_flex": False,
    }
    assert exp.config_from_state(legacy_cfg) == cfg
    values_026 = {name: 0 for name in exp._CHECKPOINT_ARCH_FIELDS}
    with pytest.raises(ValueError, match="not experiment 036"):
        exp.config_from_state(values_026)
    with pytest.raises(ValueError, match="experiment_id"):
        exp.config_from_state({**checkpoint_cfg, "experiment_id": "034_rank_weighted_bc_v1"})


def test_experiment_file_is_self_contained() -> None:
    source = _PATH.read_text()
    assert "importlib" not in source  # experiments never load other experiments
    assert "spec_from_file_location" not in source


def test_return_audit_reports_terminal_fraction_and_beta_sweep(tmp_path: Path) -> None:
    from streaming import MDSWriter

    from hal.data.policy_schema import POLICY_MDS_COLUMNS
    from hal.data.policy_schema import POLICY_SCHEMA_VERSION

    frames = 32
    with MDSWriter(out=str(tmp_path / "val"), columns=POLICY_MDS_COLUMNS, compression="zstd") as writer:
        for index in range(3):
            sample: dict[str, object] = {
                "policy_schema_version": POLICY_SCHEMA_VERSION,
                "source_schema_version": 7,
                "replay_id": f"{index:032x}",
                "num_frames": frames,
                "stage": 2,
                "p1_character": 1,
                "p2_character": 22,
                "p1_nana_present": 0,
                "p2_nana_present": 0,
            }
            for name, encoding in POLICY_MDS_COLUMNS.items():
                if name in sample:
                    continue
                dtype = np.dtype(encoding.removeprefix("ndarray:"))
                sample[name] = np.zeros(1 if "nana" in name else frames, dtype=dtype)
            writer.write(sample)

    cfg = _cfg(data_root=str(tmp_path), mds_schema_version=7)
    report = exp.return_audit(cfg, split="val", max_replays=3)
    assert report["replays"] == 3
    assert report["truncated_frac"] == 0.0  # zero-filled stocks read as an ended game
    assert report["frames"] == 3 * 2 * frames
    assert f"global mean/beta{exp._AUDIT_BETAS[0]:g}" in report


def test_tiny_training_run_flips_awr_active_and_reaches_final_validation(monkeypatch, tmp_path: Path) -> None:
    # Flex attention on CUDA needs head_dim >= 16, so the end-to-end model is
    # slightly wider than the CPU-only unit-test models.
    cfg = _cfg(
        d_model=64,
        max_steps=2,
        awr_warmup_steps=1,
        val_every=0,
        eval_every=0,
        ckpt_every=0,
        wandb_log_code=False,
    )
    batch = _awr_batch(cfg, eligible_rows=[True, False])
    monkeypatch.setattr(exp, "_make_loaders", lambda cfg, stats: ([batch], [batch.batch]))
    monkeypatch.setattr(exp, "make_run_name", lambda *args: "tiny-036")
    monkeypatch.setattr(exp, "setup_run_dir", lambda name: (tmp_path, tmp_path / "replays"))
    monkeypatch.setattr(exp, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp, "_checkpoint_sha256", lambda path: "a" * 64)
    evaluations: list[tuple[int, int]] = []
    inference = object()
    monkeypatch.setattr(exp, "BF16Inference", lambda model, cfg: inference)

    def fake_eval(model, stats, cfg, *, n_matchups, replay_dir, exec_horizon=None, checkpoint_sha256, inference=None):
        evaluations.append((n_matchups, cfg.exec_horizon if exec_horizon is None else exec_horizon))
        return {"net_stock_lcb": 0.0, "net_dmg_lcb": 0.0}

    monkeypatch.setattr(exp, "eval_vs_cpu", fake_eval)

    class Run:
        id = "test"
        summary: dict[str, object] = {}

    logs: list[dict] = []
    monkeypatch.setattr(exp.wandb, "run", Run())
    monkeypatch.setattr(exp.wandb, "init", lambda **kwargs: None)
    monkeypatch.setattr(exp.wandb, "define_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp.wandb, "log", lambda values: logs.append(values))
    monkeypatch.setattr(exp.wandb, "finish", lambda: None)
    exp.train(cfg, {}, comment="tiny")

    active = [entry["awr/active"] for entry in logs if "awr/active" in entry]
    assert active == [0.0, 1.0]
    assert all("train/value_loss" in entry for entry in logs if "awr/active" in entry)
    assert evaluations == [(cfg.final_eval_n_matchups, 4), (cfg.final_diag_n_matchups, 6)]
