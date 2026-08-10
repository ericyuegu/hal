"""Contracts for the chunk-AWR experiment on the 026 stack.

The file pins three groups of properties:
1. Reward/return labels align with the actions the model executes (G_{t+1} rule).
2. The critic, weights, and objective are isolated: the control arm is bit-identical
   to 026, and no advantage signal can leak gradients into the policy.
3. The warm-up gate and the training loop follow the pre-registered protocol.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import Context
from hal.training.features import TrainBatch

_PATH = Path(__file__).resolve().parents[2] / "experiments" / "027_chunk_awr.py"
_SPEC = importlib.util.spec_from_file_location("test_exp027", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
exp = importlib.util.module_from_spec(_SPEC)
sys.modules["test_exp027"] = exp
_SPEC.loader.exec_module(exp)

_PATH_026 = Path(__file__).resolve().parents[2] / "experiments" / "026_temporal_mtp.py"
_SPEC_026 = importlib.util.spec_from_file_location("test_exp026_ref", _PATH_026)
assert _SPEC_026 is not None and _SPEC_026.loader is not None
exp026 = importlib.util.module_from_spec(_SPEC_026)
sys.modules["test_exp026_ref"] = exp026
_SPEC_026.loader.exec_module(exp026)


def _cfg(**overrides):
    # head_dim must reach 16: compiled FlexAttention rejects smaller embeddings on CUDA.
    values = dict(
        d_model=64,
        n_layers=1,
        n_heads=4,
        L_ctx=8,
        temporal_d_model=32,
        temporal_layers=2,
        temporal_heads=4,
        temporal_ff_dim=64,
        group_head_dim=64,
        batch_size=2,
        grad_accum_steps=1,
        reservoir_capacity=4,
        warmup_steps=1,
        max_steps=4,
        awr_warmup_steps=2,
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


def _awr_batch(cfg, batch: int = 2, seed: int = 0, terminal: bool = True) -> exp.AWRBatch:
    generator = torch.Generator().manual_seed(seed + 40)
    base = _batch(cfg, batch, seed)
    horizon = cfg.awr_horizon
    rewards = torch.randn(batch, cfg.L_ctx + horizon, generator=generator) * 0.1
    gamma = cfg.awr_gamma
    length = rewards.shape[1]
    returns = torch.zeros(batch, cfg.L_ctx)
    # Hand-rolled G_{t+1} over the reward slice plus a zero tail beyond it.
    for t in reversed(range(cfg.L_ctx)):
        tail = (
            returns[:, t + 1]
            if t + 1 < cfg.L_ctx
            else rewards[:, t + 1 : length].mul(torch.tensor([gamma**k for k in range(length - t - 1)])).sum(-1)
        )
        returns[:, t] = rewards[:, t] + gamma * tail
    return exp.AWRBatch(
        batch=base,
        returns=returns,
        rewards=rewards,
        terminal=torch.full((batch,), terminal, dtype=torch.bool),
    )


# %% Config


def test_defaults_freeze_the_declared_flight() -> None:
    cfg = exp.TrainConfig()
    assert cfg.batch_size == 512
    assert cfg.awr_mode == "off"
    assert cfg.awr_gamma == pytest.approx(0.99827)
    assert cfg.awr_damage_shaping == pytest.approx(0.01)
    assert cfg.awr_win_reward == pytest.approx(0.5)
    assert cfg.awr_horizon == 4
    assert cfg.awr_beta is None
    assert cfg.awr_beta_grid == (0.05, 0.1, 0.2, 0.4)
    assert cfg.awr_weight_max == pytest.approx(5.0)
    assert cfg.awr_warmup_steps == 2048


def test_validate_config_rejects_bad_awr_settings() -> None:
    with pytest.raises(ValueError, match="grad_accum"):
        exp.validate_config(_cfg(awr_mode="critic", grad_accum_steps=2, batch_size=4))
    with pytest.raises(ValueError, match="awr_horizon"):
        exp.validate_config(_cfg(awr_mode="critic", awr_horizon=3))
    with pytest.raises(ValueError, match="awr_warmup_steps"):
        exp.validate_config(_cfg(awr_mode="critic", awr_warmup_steps=99))
    with pytest.raises(ValueError, match="awr_beta"):
        exp.validate_config(_cfg(awr_mode="critic", awr_beta=-1.0))
    exp.validate_config(_cfg(awr_mode="off"))


# %% Reward labels and alignment


def test_annotations_agree_between_compact_and_decoded_paths() -> None:
    from hal.data.policy_schema import pack_player_state

    frames = 32
    rng = np.random.default_rng(0)
    stock = {"p1": np.repeat([4, 3, 3, 2], 8).astype(np.int32), "p2": np.repeat([4, 4, 1, 0], 8).astype(np.int32)}
    percent = {port: rng.uniform(0, 150, frames).astype(np.float32) for port in ("p1", "p2")}
    decoded = {}
    compact = {"num_frames": frames}
    for port in ("p1", "p2"):
        decoded[f"{port}_stock"] = stock[port]
        decoded[f"{port}_percent"] = percent[port]
        compact[f"{port}_percent"] = percent[port]
        compact[f"{port}_state"] = pack_player_state(
            {
                "action": np.zeros(frames, dtype=np.int32),
                "stock": stock[port],
                "jumps_used": np.zeros(frames, dtype=np.int32),
                "hurtbox_state": np.zeros(frames, dtype=np.int32),
                "airborne": np.zeros(frames, dtype=np.int32),
                "direction": np.zeros(frames, dtype=np.float32),
            }
        )
    cfg = _cfg()
    from_compact = exp.annotate_compact_replay(
        compact, gamma=cfg.awr_gamma, damage_shaping=cfg.awr_damage_shaping, win_reward=cfg.awr_win_reward
    )
    from_decoded = exp.annotate_decoded_replay(
        decoded, gamma=cfg.awr_gamma, damage_shaping=cfg.awr_damage_shaping, win_reward=cfg.awr_win_reward
    )
    assert set(from_compact) == set(from_decoded) == set(exp.AWR_ANNOTATION_COLUMNS)
    for name in from_compact:
        np.testing.assert_allclose(from_compact[name], from_decoded[name], atol=1e-6, err_msg=name)
    assert from_compact["awr_terminal"].min() == 1.0  # p2 lost the last stock


def test_collate_pairs_position_t_with_the_return_after_the_next_action() -> None:
    cfg = _cfg()
    length = cfg.L_ctx + cfg.sample_chunk_length
    horizon = cfg.awr_horizon
    ret = np.arange(100, 100 + length, dtype=np.float32)
    reward = np.arange(length, dtype=np.float32)
    windows = []
    window = {"ctx_pad": np.int64(0), "ego_awr_return": ret, "ego_awr_reward": reward}
    window["awr_terminal"] = np.ones(length, dtype=np.float32)
    windows.append(window)
    base = _batch(cfg, batch=1)
    out = exp.collate_awr_batch(windows, base, L_ctx=cfg.L_ctx, horizon=horizon)
    # returns[t] must be G_{t+1}: the label one frame after the context position.
    np.testing.assert_allclose(out.returns[0].numpy(), ret[1 : cfg.L_ctx + 1])
    # rewards[k] must be r_{k+1}, reaching horizon frames past the context.
    np.testing.assert_allclose(out.rewards[0].numpy(), reward[1 : cfg.L_ctx + horizon + 1])
    assert bool(out.terminal[0])


def test_awr_columns_never_reach_the_model_as_features() -> None:
    from hal.training.features import preprocess

    stacked = {
        "ego_awr_return": np.zeros((1, 4), dtype=np.float32),
        "ego_awr_reward": np.zeros((1, 4), dtype=np.float32),
        "awr_terminal": np.zeros((1, 4), dtype=np.float32),
    }
    out = preprocess(stacked, {}, projection=None)
    assert not out


def test_awr_projection_extends_base_projection_only_with_awr_columns() -> None:
    from hal.training.features import BASE_ACTION_PROJECTION

    added = exp.AWR_PROJECTION.columns - BASE_ACTION_PROJECTION.columns
    assert added == {"ego_awr_return", "ego_awr_reward", "awr_terminal"}


# %% Advantage


def test_chunk_advantage_matches_hand_computation() -> None:
    gamma, horizon = 0.9, 4
    value = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    rewards = torch.arange(1.0, 13.0).view(1, 12)  # r_{t+1} at index t
    valid = torch.ones(1, 8, dtype=torch.bool)
    advantage, eligible = exp.chunk_advantage(value, rewards, valid, gamma=gamma, horizon=horizon)
    t = 1
    expected = (
        rewards[0, t]
        + gamma * rewards[0, t + 1]
        + gamma**2 * rewards[0, t + 2]
        + gamma**3 * rewards[0, t + 3]
        + gamma**4 * value[0, t + 4]
        - value[0, t]
    )
    assert advantage[0, t].item() == pytest.approx(expected.item(), rel=1e-5)
    assert eligible[0, : 8 - horizon].all() and not eligible[0, 8 - horizon :].any()


def test_chunk_advantage_matches_the_return_identity_for_any_value() -> None:
    # A_4[t] - A_MC[t] must equal gamma^4 (V(s_{t+4}) - G_{t+5}) for EVERY value
    # function: the two advantages differ only in how they close the tail. This
    # pins both the reward slice alignment and the bootstrap position.
    cfg = _cfg()
    awr = _awr_batch(cfg)
    gamma, horizon = cfg.awr_gamma, cfg.awr_horizon
    value = torch.randn(awr.returns.shape, generator=torch.Generator().manual_seed(9))
    valid = torch.ones_like(awr.returns, dtype=torch.bool)
    advantage, eligible = exp.chunk_advantage(value, awr.rewards, valid, gamma=gamma, horizon=horizon)
    mc = awr.returns - value
    length = value.shape[1]
    for t in range(length - horizon):
        expected = gamma**horizon * (value[:, t + horizon] - awr.returns[:, t + horizon])
        torch.testing.assert_close(advantage[:, t] - mc[:, t], expected, atol=1e-4, rtol=1e-4)
    assert eligible[:, : length - horizon].all()


def test_chunk_advantage_respects_padding() -> None:
    value = torch.zeros(1, 8)
    rewards = torch.zeros(1, 12)
    valid = torch.tensor([[False, False, True, True, True, True, True, True]])
    _, eligible = exp.chunk_advantage(value, rewards, valid, gamma=0.9, horizon=4)
    assert not eligible[0, :2].any()


# %% Weights


def test_awr_weights_clip_before_exponent_and_normalize_to_mean_one() -> None:
    advantage = torch.tensor([0.0, 0.1, 10.0, -0.2])
    eligible = torch.ones(4, dtype=torch.bool)
    weight, stats = exp.awr_weights(advantage, eligible, beta=0.1, weight_max=5.0)
    raw = torch.exp(torch.clamp(advantage / 0.1, max=np.log(5.0)))
    torch.testing.assert_close(weight, raw / raw.mean(), atol=1e-5, rtol=1e-5)
    assert weight.mean().item() == pytest.approx(1.0, rel=1e-5)
    assert stats["w_raw_clip_frac"] == pytest.approx(1 / 4)
    assert stats["ess_frame"] == pytest.approx(float((weight.sum() ** 2) / (4 * weight.square().sum())), rel=1e-5)


def test_awr_weights_survive_all_large_negative_advantages() -> None:
    advantage = torch.full((6,), -200.0)
    weight, _ = exp.awr_weights(advantage, torch.ones(6, dtype=torch.bool), beta=0.1, weight_max=5.0)
    assert torch.isfinite(weight).all()
    assert weight.mean().item() == pytest.approx(1.0, rel=1e-5)


def test_awr_weights_allow_normalized_weights_above_the_raw_cap() -> None:
    # One capped row among many strong negatives: the raw mean falls below one, so
    # mean-one normalization must push the top weight past the raw cap of 5.
    advantage = torch.tensor([1.0] + [-50.0] * 9)
    weight, stats = exp.awr_weights(advantage, torch.ones(10, dtype=torch.bool), beta=0.1, weight_max=5.0)
    assert weight.max().item() > 5.0
    assert stats["w_norm_max"] == pytest.approx(weight.max().item(), rel=1e-5)


def test_awr_weights_give_ineligible_rows_weight_one() -> None:
    advantage = torch.tensor([2.0, 0.0, -1.0, 0.5])
    eligible = torch.tensor([True, False, True, False])
    weight, _ = exp.awr_weights(advantage, eligible, beta=0.5, weight_max=5.0)
    assert weight[~eligible].eq(1.0).all()
    assert weight[eligible].mean().item() == pytest.approx(1.0, rel=1e-5)


def test_awr_weights_reject_gradients_and_nonfinite_values() -> None:
    eligible = torch.ones(2, dtype=torch.bool)
    with pytest.raises(ValueError, match="detach"):
        exp.awr_weights(torch.zeros(2, requires_grad=True), eligible, beta=0.1, weight_max=5.0)
    with pytest.raises(ValueError, match="finite"):
        exp.awr_weights(torch.tensor([0.0, float("nan")]), eligible, beta=0.1, weight_max=5.0)


def test_awr_weights_with_no_eligible_rows_are_all_one() -> None:
    weight, stats = exp.awr_weights(torch.zeros(3), torch.zeros(3, dtype=torch.bool), beta=0.1, weight_max=5.0)
    assert weight.eq(1.0).all()
    assert stats["ess_frame"] == pytest.approx(1.0)


def test_window_ess_averages_raw_weights_inside_each_window_first() -> None:
    raw = torch.tensor([1.0, 3.0, 2.0, 2.0])
    rows = torch.tensor([0, 0, 1, 1])
    means = torch.tensor([2.0, 2.0])
    expected = float((means.sum() ** 2) / (2 * means.square().sum()))
    assert exp.window_ess(raw, rows, 2) == pytest.approx(expected)


# %% Critic, isolation, and identity


def test_control_model_is_bit_identical_to_026() -> None:
    kwargs = dict(
        d_model=64,
        n_layers=1,
        n_heads=4,
        L_ctx=8,
        temporal_d_model=32,
        temporal_layers=2,
        temporal_heads=4,
        temporal_ff_dim=64,
        group_head_dim=64,
        batch_size=2,
    )
    torch.manual_seed(0)
    ours = exp.GPT(_cfg(awr_mode="off"))
    torch.manual_seed(0)
    reference = exp026.GPT(exp026.TrainConfig(**kwargs))
    ours_params = dict(ours.named_parameters())
    ref_params = dict(reference.named_parameters())
    assert ours_params.keys() == ref_params.keys()
    for name, value in ref_params.items():
        assert torch.equal(ours_params[name], value), name


def test_critic_arm_keeps_every_policy_parameter_identical_to_control() -> None:
    torch.manual_seed(0)
    control = exp.GPT(_cfg(awr_mode="off"))
    torch.manual_seed(0)
    treatment = exp.GPT(_cfg(awr_mode="critic"))
    control_params = dict(control.named_parameters())
    for name, value in treatment.named_parameters():
        if name.startswith("critic."):
            continue
        assert torch.equal(control_params[name], value), name
    assert treatment.critic is not None and control.critic is None
    policy_names = {name for name, _ in treatment.named_parameters() if not name.startswith("critic.")}
    assert {name for name, _ in treatment.policy_parameters()} == policy_names


def test_critic_losses_never_touch_the_trunk_and_heads_get_gradients() -> None:
    cfg = _cfg(awr_mode="critic")
    model = exp.GPT(cfg)
    awr = _awr_batch(cfg)
    history, targets, valid = exp.prepared_targets(model, awr.batch)
    hidden = model(awr.batch.context.features, awr.batch.context.ctx_pad, history)
    losses = exp.critic_losses(model, hidden, awr, targets, valid, cfg)
    total = losses.value + losses.qhat + losses.variance
    total.backward()
    for name, parameter in model.named_parameters():
        if name.startswith("critic."):
            continue
        assert parameter.grad is None or not parameter.grad.abs().any(), name
    grads = [p.grad for p in model.critic.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_qhat_reads_the_chunk_and_value_does_not() -> None:
    cfg = _cfg(awr_mode="critic")
    model = exp.GPT(cfg)
    awr = _awr_batch(cfg)
    history, targets, valid = exp.prepared_targets(model, awr.batch)
    hidden = model(awr.batch.context.features, awr.batch.context.ctx_pad, history)
    chunk = targets[..., : cfg.awr_horizon, :]
    with torch.no_grad():
        base = model.critic(hidden, chunk)
        rolled = model.critic(hidden, chunk.roll(1, dims=0))
    assert not torch.allclose(base.qhat, rolled.qhat)
    torch.testing.assert_close(base.value, rolled.value)


def test_advantage_and_weights_are_detached_from_policy_and_critic() -> None:
    cfg = _cfg(awr_mode="critic")
    model = exp.GPT(cfg)
    awr = _awr_batch(cfg)
    history, targets, valid = exp.prepared_targets(model, awr.batch)
    hidden = model(awr.batch.context.features, awr.batch.context.ctx_pad, history)
    losses = exp.critic_losses(model, hidden, awr, targets, valid, cfg)
    assert not losses.advantage.requires_grad
    assert not losses.weight_source.requires_grad


def test_value_loss_covers_only_valid_rows_of_terminal_replays() -> None:
    cfg = _cfg(awr_mode="critic")
    model = exp.GPT(cfg)
    awr = _awr_batch(cfg, terminal=False)
    history, targets, valid = exp.prepared_targets(model, awr.batch)
    hidden = model(awr.batch.context.features, awr.batch.context.ctx_pad, history)
    losses = exp.critic_losses(model, hidden, awr, targets, valid, cfg)
    assert losses.value.item() == 0.0
    assert losses.value_rows == 0


def test_policy_optimizer_owns_exactly_the_policy_parameters() -> None:
    cfg = _cfg(awr_mode="critic")
    model = exp.GPT(cfg)
    optimizer = exp.make_optimizer(model, cfg)
    owned = {id(p) for group in optimizer.param_groups for p in group["params"]}
    policy = {id(p) for _, p in model.policy_parameters()}
    critic = {id(p) for p in model.critic.parameters()}
    assert owned == policy
    assert not owned & critic
    critic_optimizer = exp.make_critic_optimizer(model, cfg)
    critic_owned = [id(p) for group in critic_optimizer.param_groups for p in group["params"]]
    assert set(critic_owned) == critic and len(critic_owned) == len(critic)


def test_objective_applies_one_weight_to_the_dense_prefix_only() -> None:
    generator = torch.Generator().manual_seed(3)
    nll = torch.rand(6, 10, 4, generator=generator)
    parts = exp.ActionLoss(nll=nll, targets=torch.zeros(6, 10, 4, dtype=torch.long))
    unweighted = exp.objective(parts, 1.0)
    torch.testing.assert_close(exp.objective(parts, 1.0, weight=torch.ones(6)), unweighted)
    weight = torch.tensor([2.0, 0.5, 1.0, 1.0, 0.25, 1.25])
    weighted = exp.objective(parts, 1.0, weight=weight)
    joint = nll.sum(-1)
    expected = (joint[:, :4] * weight[:, None]).mean() + joint[:, 4:].mean()
    torch.testing.assert_close(weighted, expected)
    # The auxiliary term must not move with the weights.
    aux_only = exp.objective(parts, 1.0, weight=torch.zeros(6)) - 0.0
    torch.testing.assert_close(aux_only, joint[:, 4:].mean())


# %% Gate


def test_beta_table_reports_ess_and_clip_per_beta() -> None:
    source = torch.tensor([0.0, 0.2, -0.2, 1.0, -1.0, 0.05])
    rows = torch.tensor([0, 0, 1, 1, 2, 2])
    table = exp.beta_table(source, rows, 3, grid=(0.05, 0.4), weight_max=5.0)
    assert [entry["beta"] for entry in table] == [0.05, 0.4]
    for entry in table:
        assert 0.0 < entry["ess_frame"] <= 1.0
        assert 0.0 < entry["ess_window"] <= 1.0
        assert 0.0 <= entry["clip_frac"] <= 1.0
    # The tighter temperature drives more rows onto the raw cap.
    assert table[0]["clip_frac"] >= table[1]["clip_frac"]


def test_select_beta_picks_the_smallest_passing_value() -> None:
    cfg = _cfg(awr_mode="critic")
    table = [
        {"beta": 0.05, "ess_frame": 0.05, "ess_window": 0.5, "clip_frac": 0.5},
        {"beta": 0.1, "ess_frame": 0.3, "ess_window": 0.15, "clip_frac": 0.1},
        {"beta": 0.2, "ess_frame": 0.4, "ess_window": 0.3, "clip_frac": 0.1},
        {"beta": 0.4, "ess_frame": 0.9, "ess_window": 0.8, "clip_frac": 0.0},
    ]
    assert exp.select_beta(table, cfg) == 0.2
    none_pass = [dict(entry, ess_window=0.0) for entry in table]
    assert exp.select_beta(none_pass, cfg) is None


def test_gate_runs_on_a_tiny_model_and_writes_the_artifact(tmp_path: Path) -> None:
    cfg = _cfg(awr_mode="critic", awr_beta=0.1)
    model = exp.GPT(cfg)
    batches = [_awr_batch(cfg, seed=seed) for seed in range(2)]
    result = exp.run_activation_gate(model, batches, cfg, tmp_path)
    artifact = tmp_path / "awr_gate.json"
    assert artifact.is_file()
    assert result.beta == pytest.approx(0.1)
    assert isinstance(result.passed, bool)
    assert np.isfinite(result.value_corr)
    assert result.table


# %% Loop support


def test_cache_validation_slices_awr_batches_exactly() -> None:
    cfg = _cfg()
    batches = [_awr_batch(cfg, batch=2, seed=seed) for seed in range(3)]
    cached = exp.cache_validation(batches, 3)
    assert sum(item.returns.shape[0] for item in cached) == 3
    assert cached[0].returns.shape[0] == 2 and cached[1].returns.shape[0] == 1
    torch.testing.assert_close(cached[1].rewards, batches[1].rewards[:1])
    torch.testing.assert_close(cached[1].batch.target, batches[1].batch.target[:1])


def test_critic_state_round_trips_through_the_state_dict() -> None:
    cfg = _cfg(awr_mode="critic")
    source = exp.GPT(cfg)
    clone = exp.GPT(cfg)
    clone.load_state_dict(source.state_dict())
    for (name, a), (_, b) in zip(source.named_parameters(), clone.named_parameters(), strict=True):
        assert torch.equal(a, b), name


def test_config_round_trip_keeps_awr_fields() -> None:
    from dataclasses import asdict

    cfg = _cfg(awr_mode="critic", awr_beta=0.2)
    restored = exp.config_from_state(asdict(cfg))
    assert restored == cfg


# %% h2h contract


def test_decode_settings_are_frozen_and_make_policy_rejects_other_knobs() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg)
    settings = exp._decode_settings(model, cfg)
    assert settings.temp == 1.0 and settings.temps is None
    assert settings.btn_support_min == 0 and settings.min_p == 0.0
    assert settings.click_trigger_fix is False
    assert exp._load_ckpt is exp.load_checkpoint
    stats: dict = {}
    with pytest.raises(ValueError, match="temperature"):
        exp.make_policy(model, stats, cfg, decode_temp=0.5)


# %% End to end


def _fake_wandb(monkeypatch) -> None:
    class Run:
        id = "test"
        summary: dict[str, object] = {}

    monkeypatch.setattr(exp.wandb, "run", Run())
    monkeypatch.setattr(exp.wandb, "init", lambda **kwargs: None)
    monkeypatch.setattr(exp.wandb, "define_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp.wandb, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp.wandb, "finish", lambda: None)


def _fake_eval(monkeypatch) -> list[tuple[int, int]]:
    evaluations: list[tuple[int, int]] = []

    def fake(model, stats, cfg, *, n_matchups, replay_dir, exec_horizon=None, checkpoint_sha256):
        evaluations.append((n_matchups, cfg.exec_horizon if exec_horizon is None else exec_horizon))
        return {"net_stock_lcb": 0.0, "net_dmg_lcb": 0.0}

    monkeypatch.setattr(exp, "eval_vs_cpu", fake)
    return evaluations


def _capture_final_state(monkeypatch) -> dict:
    captured: dict = {}

    def fake_save(path, *, step, model, opt, sched, cfg, wandb_id, uploader=None, extra=None):
        state = {name: value.clone() for name, value in model.state_dict().items()}
        captured.setdefault("first_state", state)
        captured.setdefault("first_step", step)
        captured["state"] = state
        captured["extra"] = extra

    monkeypatch.setattr(exp, "save_checkpoint", fake_save)
    monkeypatch.setattr(exp, "_checkpoint_sha256", lambda path: "a" * 64)
    return captured


@pytest.mark.parametrize("mode", ["off", "critic"])
def test_tiny_training_run_reaches_final_validation_and_evaluations(monkeypatch, tmp_path: Path, mode: str) -> None:
    cfg = _cfg(
        awr_mode=mode,
        max_steps=3,
        awr_warmup_steps=1,
        awr_beta=0.1,
        val_every=0,
        eval_every=0,
        ckpt_every=0,
        wandb_log_code=False,
        inference_mode="eager",
    )
    batch = _batch(cfg)
    loop_batch = _awr_batch(cfg) if mode != "off" else batch
    monkeypatch.setattr(exp, "_make_loaders", lambda cfg, stats: ([loop_batch], [loop_batch]))
    monkeypatch.setattr(exp, "make_run_name", lambda *args: f"tiny-027-{mode}")
    monkeypatch.setattr(exp, "setup_run_dir", lambda name: (tmp_path, tmp_path / "replays"))
    _capture_final_state(monkeypatch)
    if mode != "off":
        passing = exp.GateResult(
            passed=True, beta=0.1, value_corr=0.5, qhat_corr=0.5, action_sensitivity=1.5, table=[], reasons=[]
        )
        monkeypatch.setattr(exp, "run_activation_gate", lambda *args, **kwargs: passing)
    evaluations = _fake_eval(monkeypatch)
    _fake_wandb(monkeypatch)
    exp.train(cfg, {}, comment="tiny")
    assert evaluations == [(cfg.final_eval_n_matchups, 4), (cfg.final_diag_n_matchups, 6)]


def test_warmup_policy_update_is_bit_identical_to_the_control(monkeypatch, tmp_path: Path) -> None:
    # Compare the first periodic checkpoint, taken after two updates that are both
    # inside the critic warm-up: every policy parameter must have moved exactly as
    # the control arm moved on the same data.
    states: dict[str, dict] = {}
    for mode in ("off", "critic"):
        cfg = _cfg(
            awr_mode=mode,
            max_steps=3,
            awr_warmup_steps=2,
            awr_beta=0.1,
            val_every=0,
            eval_every=0,
            ckpt_every=1,
            wandb_log_code=False,
            inference_mode="eager",
        )
        batch = _batch(cfg)
        loop_batch = _awr_batch(cfg) if mode != "off" else batch
        monkeypatch.setattr(exp, "_make_loaders", lambda cfg, stats, b=loop_batch: ([b], [b]))
        monkeypatch.setattr(exp, "make_run_name", lambda *args: "tiny-027-warmup")
        monkeypatch.setattr(exp, "setup_run_dir", lambda name, m=mode: (tmp_path / m, tmp_path / m / "replays"))
        captured = _capture_final_state(monkeypatch)
        if mode != "off":
            passing = exp.GateResult(
                passed=True, beta=0.1, value_corr=0.5, qhat_corr=0.5, action_sensitivity=1.5, table=[], reasons=[]
            )
            monkeypatch.setattr(exp, "run_activation_gate", lambda *args, gate=passing, **kwargs: gate)
        _fake_eval(monkeypatch)
        _fake_wandb(monkeypatch)
        exp.train(cfg, {}, comment="tiny")
        assert captured["first_step"] == 1
        states[mode] = captured["first_state"]
    control, treatment = states["off"], states["critic"]
    critic_keys = {name for name in treatment if name.startswith("critic.")}
    assert critic_keys
    assert set(treatment) - critic_keys == set(control)
    for name in control:
        assert torch.equal(control[name], treatment[name]), name


# %% Audit


def test_audit_statistics_from_synthetic_annotations() -> None:
    from hal.training.returns import discounted_returns

    cfg = _cfg(awr_mode="critic")
    rng = np.random.default_rng(0)
    annotations = []
    for index in range(6):
        frames = 200
        reward = rng.normal(0, 0.05, frames).astype(np.float32)
        returns = discounted_returns(reward, cfg.awr_gamma)
        annotations.append(
            {
                "p1_awr_reward": reward,
                "p1_awr_return": returns,
                "p2_awr_reward": -reward,
                "p2_awr_return": discounted_returns(-reward, cfg.awr_gamma),
                "awr_terminal": np.full(frames, float(index % 3 != 0), dtype=np.float32),
            }
        )
    report = exp.audit_from_annotations(annotations, cfg)
    assert report["replays"] == 6
    assert report["terminal_frac"] == pytest.approx(4 / 6)
    assert set(report["return_quantiles"]) == {"p01", "p05", "p25", "p50", "p75", "p95", "p99"}
    for surrogate in ("global_mean", "replay_mean"):
        assert report[f"beta_table_{surrogate}"], surrogate
        for entry in report[f"beta_table_{surrogate}"]:
            assert 0.0 < entry["ess_frame"] <= 1.0


def test_model_tag_names_the_arm() -> None:
    assert "awr027" in exp.model_tag(_cfg(awr_mode="critic"))
    assert "critic" in exp.model_tag(_cfg(awr_mode="critic"))
    assert "off" in exp.model_tag(_cfg(awr_mode="off"))
