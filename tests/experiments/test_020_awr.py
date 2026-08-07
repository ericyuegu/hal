"""020 keeps 016's model and reweights WHICH frames its loss listens to. These tests pin the
four properties that make that a clean A/B, plus the plumbing the reweighting rides on.

1. Only the objective changed. Trunk and action heads are 016's with ``spatial_features`` off;
   the value head is the single addition, and ``awr_enabled=False`` reproduces 016's exact
   unweighted mean NLL.
2. The return is right. Reward fires on stock DROPS, a respawn's percent reset is not damage,
   and G is the discounted sum to the end of the episode.
3. The return reaches the right frames. Labeled onto the replay row, it is windowed, padded and
   ego-relabeled by hal's own sampler, so perturbing one frame's stock moves G at exactly the
   window positions that frame precedes.
4. The weights behave. Clipped, mean 1, ESS as defined, and detached — the policy loss can never
   train the value head.
"""

import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from melee import Stage

from hal.data.feature_stats import FeatureStats
from hal.data.schema import MDS_PER_FRAME_DTYPES
from hal.training.dataloader import WindowDataset
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import Context
from hal.training.features import TrainBatch

_REPO = Path(__file__).resolve().parent.parent.parent
_EXP_DIR = _REPO / "experiments"
_DEV_MDS = _REPO / "data" / "processed" / "dev" / "mds"


def _load_experiment(filename: str):
    spec = importlib.util.spec_from_file_location(filename.split(".")[0], _EXP_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


exp016 = _load_experiment("016_spatial_features.py")
exp020 = _load_experiment("020_awr.py")

_GROUPS = exp020._GROUP_NAMES


def _tiny_cfg(exp, **kwargs):
    defaults = dict(
        d_model=64, n_layers=2, n_heads=2, L_ctx=16, head_offsets=(1, 2), batch_size=2, max_steps=8, warmup_steps=2
    )
    return exp.TrainConfig(**{**defaults, **kwargs})


def _stats() -> dict[str, FeatureStats]:
    keys = (*FLOAT_FEATURES, *(f"nana_{k}" for k in FLOAT_FEATURES))
    return {k: FeatureStats(mean=0.0, std=1.0, min=-1.0, max=1.0) for k in keys}


def _features(exp, batch: int, length: int, gen: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """A synthetic observation batch: zeros, or plausible random values when ``gen`` is given."""

    def randn(*shape: int) -> torch.Tensor:
        return torch.zeros(*shape) if gen is None else torch.randn(*shape, generator=gen)

    features: dict[str, torch.Tensor] = {}
    for prefix in exp._PLAYER_PREFIXES:
        for feat in FLOAT_FEATURES:
            features[f"{prefix}_{feat}"] = randn(batch, length)
        for name, (vocab, _) in CAT_FEATURES.items():
            hi = 1 if gen is None else vocab
            features[f"{prefix}_{name}"] = torch.randint(0, hi, (batch, length), generator=gen)
    for channel in ACTION_CHANNELS:
        features[f"ego_{channel}"] = randn(batch, length)
    for key in ("ego_character", "opp_character", "stage"):
        features[key] = torch.randint(0, 1 if gen is None else 26, (batch, length), generator=gen)
    return features


def _awr_batch(cfg, batch: int = 4, seed: int = 0) -> exp020.AWRBatch:
    gen = torch.Generator().manual_seed(seed)
    ctx = Context(features=_features(exp020, batch, cfg.L_ctx, gen), ctx_pad=torch.tensor([0, 1] * (batch // 2)))
    target = torch.rand(batch, max(cfg.head_offsets), A_DIM, generator=gen) * 2 - 1
    returns = torch.randn(batch, cfg.L_ctx, generator=gen)
    return exp020.AWRBatch(batch=TrainBatch(context=ctx, target=target), returns=returns)


def _model(cfg, seed: int = 7):
    torch.manual_seed(seed)
    return exp020.GPT(cfg).eval()


# --- the frozen recipe -------------------------------------------------------


def test_defaults_are_the_deployed_016_base_recipe() -> None:
    """Read back from the 016-base checkpoint cfg, not from 016's file defaults."""
    cfg = exp020.TrainConfig()
    assert (cfg.batch_size, cfg.max_steps) == (512, 16384)
    assert (cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.L_ctx) == (256, 8, 4, 256)
    assert (cfg.muon_lr, cfg.adam_lr) == (0.02, 8.5e-4)
    assert cfg.head_offsets == (1, 5, 9, 13)
    assert cfg.warmup_steps == 500 and cfg.val_every == 1024 and cfg.ckpt_every == 2048
    assert cfg.windows_per_replay == 4 and cfg.final_eval_n_matchups == 96
    assert (cfg.eval_every, cfg.eval_n_matchups, cfg.eval_timeout_seconds) == (4096, 32, 2700.0)
    assert cfg.mds_schema_version == 5  # ranked-anonymized-1 is materialized at v5
    assert cfg.final_h2h_self_label == "020-awr"
    assert cfg.final_h2h_reference_label == "016-base"


def test_awr_defaults_are_the_audited_recipe() -> None:
    """beta comes from the offline return audit (ESS ~= 44% on the dev MDS at gamma=0.999)."""
    cfg = exp020.TrainConfig()
    assert cfg.awr_enabled is True
    assert cfg.awr_gamma == 0.999
    assert cfg.awr_beta == 0.5
    assert cfg.awr_weight_max == 20.0
    assert cfg.awr_value_loss_weight == 1.0
    assert cfg.awr_damage_shaping == 0.0  # stock events only
    assert cfg.awr_win_reward == 0.0  # every stock equal


def test_loader_kwargs_declare_the_mds_schema_version() -> None:
    """Both splits opt down to the dataset's version explicitly; the loader guard does the rest."""
    kwargs = exp020._loader_kwargs(exp020.TrainConfig(), _stats())
    assert kwargs["schema_version"] == 5


# --- only the objective changed ----------------------------------------------


def test_model_is_016_spatial_off_plus_a_value_head() -> None:
    """Same input width, same trunk AND head parameters from the same seed, same hidden state.
    The value head is the only new tensor."""
    cfg16, cfg20 = _tiny_cfg(exp016, spatial_features=False), _tiny_cfg(exp020)
    torch.manual_seed(7)
    model16 = exp016.GPT(cfg16).eval()
    model20 = _model(cfg20)

    assert model20.ctx_proj.in_features == 374 == model16.ctx_proj.in_features
    state16, state20 = model16.state_dict(), model20.state_dict()
    assert set(state20) - set(state16) == {"value_head.weight", "value_head.bias"}
    assert not set(state16) - set(state20)
    for key, value in state16.items():
        torch.testing.assert_close(state20[key], value, rtol=0, atol=0)
    features = _features(exp020, 2, cfg20.L_ctx)
    ctx_pad = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        torch.testing.assert_close(model16(features, ctx_pad), model20(features, ctx_pad), rtol=0, atol=0)


def test_value_head_shape_and_placement() -> None:
    cfg = _tiny_cfg(exp020)
    model = _model(cfg)
    assert (model.value_head.in_features, model.value_head.out_features) == (cfg.d_model, 1)
    parts = exp020.action_loss(model, _awr_batch(cfg).batch)
    assert parts.value.shape == parts.nll[(1, "buttons")].shape  # one V per scored position
    assert parts.valid.shape == (4, cfg.L_ctx)


def test_awr_disabled_objective_is_the_unweighted_mean_nll() -> None:
    cfg = _tiny_cfg(exp020, awr_enabled=False)
    model = _model(cfg)
    batch = _awr_batch(cfg)
    parts = exp020.action_loss(model, batch.batch)
    obj = exp020.objective(parts.nll, parts.transition, cfg.aux_loss_weight, cfg.transition_loss_weight)
    primary = sum(parts.nll[(1, name)].mean() for name in _GROUPS)
    aux_offsets = sorted({offset for offset, _ in parts.nll if offset != 1})
    auxiliary = torch.stack(
        [sum(parts.nll[(offset, name)].mean() for name in _GROUPS) for offset in aux_offsets]
    ).mean()
    torch.testing.assert_close(obj, primary + cfg.aux_loss_weight * auxiliary)


def test_awr_weight_changes_the_objective_but_not_the_reported_nll() -> None:
    cfg = _tiny_cfg(exp020)
    model = _model(cfg)
    batch = _awr_batch(cfg)
    parts = exp020.action_loss(model, batch.batch)
    weight, _ = exp020.awr_weights(
        batch.valid_returns(parts.valid) - parts.value.detach(), beta=cfg.awr_beta, weight_max=cfg.awr_weight_max
    )
    plain = exp020.objective(parts.nll, parts.transition, cfg.aux_loss_weight, cfg.transition_loss_weight)
    weighted = exp020.objective(parts.nll, parts.transition, cfg.aux_loss_weight, cfg.transition_loss_weight, weight)
    assert weighted.item() != plain.item()
    expected = sum((weight * parts.nll[(1, name)]).sum() / weight.sum() for name in _GROUPS)
    aux_offsets = sorted({offset for offset, _ in parts.nll if offset != 1})
    expected = (
        expected
        + cfg.aux_loss_weight
        * torch.stack([sum(parts.nll[(offset, name)].mean() for name in _GROUPS) for offset in aux_offsets]).mean()
    )
    torch.testing.assert_close(weighted, expected)
    # The logged number is the unweighted one, so it stays comparable to 016 whatever beta does.
    assert exp020.nll_breakdown({name: parts.nll[(1, name)] for name in _GROUPS})["total"] == pytest.approx(
        sum(parts.nll[(1, name)].mean().item() for name in _GROUPS) / math.log(2.0)
    )


# --- the return --------------------------------------------------------------


def _toy_replay(frames: int = 12) -> dict[str, np.ndarray]:
    sample = {
        "p1_stock": np.full(frames, 4, dtype=np.int32),
        "p2_stock": np.full(frames, 4, dtype=np.int32),
        "p1_percent": np.zeros(frames, dtype=np.float32),
        "p2_percent": np.zeros(frames, dtype=np.float32),
    }
    sample["p1_stock"][5:] = 3  # ego (p1) dies on frame 5
    sample["p2_stock"][8:] = 3  # opponent dies on frame 8
    return sample


def test_returns_match_a_hand_computed_toy_episode() -> None:
    """gamma=0.5, ego loses a stock at t=5 and takes one at t=8: G is the exact discounted sum."""
    sample = _toy_replay()
    returns = exp020.replay_returns(sample, gamma=0.5, damage_shaping=0.0, win_reward=0.0)
    ego = returns["p1_awr_return"]
    expected = np.array(
        [-(0.5 ** (5 - t)) + 0.5 ** (8 - t) if t <= 5 else (0.5 ** (8 - t) if t <= 8 else 0.0) for t in range(12)],
        dtype=np.float32,
    )
    np.testing.assert_allclose(ego, expected, rtol=0, atol=1e-6)
    # The opponent's column is the exact mirror: one reward, two viewpoints.
    np.testing.assert_allclose(returns["p2_awr_return"], -expected, rtol=0, atol=1e-6)
    assert ego[5] == pytest.approx(-1 + 0.5**3)
    assert ego[9] == 0.0  # nothing left to happen


def test_stock_increments_and_masked_frames_never_fire() -> None:
    stock = np.array([4, 4, 3, 4, 4, 3], dtype=np.int32)  # a drop, a (impossible) rise, a drop
    np.testing.assert_array_equal(exp020.stock_loss_events(stock), [0, 0, 1, 0, 0, 1])
    masked = np.array([4, np.iinfo(np.int32).max, 3, 3], dtype=np.int32)
    np.testing.assert_array_equal(exp020.stock_loss_events(masked), [0, 0, 0, 0])


def test_percent_reset_at_respawn_is_not_damage() -> None:
    """Damage shaping reads percent RISES only: the 80 -> 0 reset on respawn contributes nothing."""
    frames = 6
    sample = {
        "p1_stock": np.full(frames, 4, dtype=np.int32),
        "p2_stock": np.full(frames, 4, dtype=np.int32),
        "p1_percent": np.zeros(frames, dtype=np.float32),
        "p2_percent": np.array([0.0, 30.0, 80.0, 0.0, 0.0, 10.0], dtype=np.float32),
    }
    reward = exp020.frame_reward(sample, ego="p1", opp="p2", damage_shaping=1.0, win_reward=0.0)
    np.testing.assert_allclose(reward, [0.0, 30.0, 50.0, 0.0, 0.0, 10.0], rtol=0, atol=1e-6)
    # With shaping off the same episode is all zeros: no stock changed hands.
    np.testing.assert_allclose(
        exp020.frame_reward(sample, ego="p1", opp="p2", damage_shaping=0.0, win_reward=0.0), np.zeros(frames)
    )


def test_match_point_fires_only_when_a_stock_count_empties() -> None:
    """The 2->1 drop is an ordinary stock; the 1->0 drop is the match point. A masked frame next
    to a drop suppresses it, exactly as in ``stock_loss_events``."""
    stock = np.array([2, 2, 1, 1, 0, 0], dtype=np.int32)
    np.testing.assert_array_equal(exp020.match_point_events(stock), [0, 0, 0, 0, 1, 0])
    quit_out = np.array([2, 1, 1], dtype=np.int32)  # game ends without reaching 0
    np.testing.assert_array_equal(exp020.match_point_events(quit_out), [0, 0, 0])
    masked = np.array([1, np.iinfo(np.int32).max, 0], dtype=np.int32)
    np.testing.assert_array_equal(exp020.match_point_events(masked), [0, 0, 0])


def test_win_reward_makes_the_last_stock_worth_more() -> None:
    """With win_reward=3 the match-deciding stock pays 1+3=4 and an ordinary stock still pays 1."""
    frames = 8
    sample = {
        "p1_stock": np.full(frames, 2, dtype=np.int32),
        "p2_stock": np.full(frames, 2, dtype=np.int32),
        "p1_percent": np.zeros(frames, dtype=np.float32),
        "p2_percent": np.zeros(frames, dtype=np.float32),
    }
    sample["p2_stock"][3:] = 1  # ordinary stock at t=3
    sample["p2_stock"][6:] = 0  # match point at t=6
    reward = exp020.frame_reward(sample, ego="p1", opp="p2", damage_shaping=0.0, win_reward=3.0)
    np.testing.assert_allclose(reward, [0, 0, 0, 1, 0, 0, 4, 0], rtol=0, atol=1e-7)
    # The loser's viewpoint is the exact mirror.
    mirror = exp020.frame_reward(sample, ego="p2", opp="p1", damage_shaping=0.0, win_reward=3.0)
    np.testing.assert_allclose(mirror, -reward, rtol=0, atol=1e-7)


def test_return_is_the_full_episode_not_the_window() -> None:
    """The reason the label is attached before windowing: at gamma=0.999 a reward 500 frames away
    still carries 60% of its value, so a window-local sum would be badly truncated."""
    frames = 1000
    sample = {
        "p1_stock": np.full(frames, 4, dtype=np.int32),
        "p2_stock": np.full(frames, 4, dtype=np.int32),
        "p1_percent": np.zeros(frames, dtype=np.float32),
        "p2_percent": np.zeros(frames, dtype=np.float32),
    }
    sample["p2_stock"][900:] = 3
    ego = exp020.replay_returns(sample, gamma=0.999, damage_shaping=0.0, win_reward=0.0)["p1_awr_return"]
    assert ego[0] == pytest.approx(0.999**900, rel=1e-5)
    assert ego[899] == pytest.approx(0.999, rel=1e-6)


# --- the return reaches the right frames -------------------------------------


_P1_MARKER, _P2_MARKER = 11.0, 22.0


def _mds_replay(frames: int) -> dict[str, np.ndarray]:
    """A full synthetic MDS row: every schema column, so hal's collate path runs for real. The two
    ports carry different constant percents so a window can report which one the sampler made ego
    (percent is inert here — damage shaping is off)."""
    sample: dict[str, np.ndarray] = {}
    for name, dtype in MDS_PER_FRAME_DTYPES.items():
        sample[name] = np.zeros(frames, dtype=dtype)
    sample["frame"] = np.arange(frames, dtype=np.int32)
    for port in ("p1", "p2"):
        sample[f"{port}_stock"] = np.full(frames, 4, dtype=np.int32)
    sample["p1_percent"] = np.full(frames, _P1_MARKER, dtype=np.float32)
    sample["p2_percent"] = np.full(frames, _P2_MARKER, dtype=np.float32)
    sample["stage"] = np.full(frames, Stage.FINAL_DESTINATION.value, dtype=np.int32)
    sample["schema_version"] = 6
    return sample


def _windows(replay: dict, *, L_ctx: int, L_chunk: int, gamma: float, K: int, seed: int) -> list[dict]:
    """Drive hal's real sampler over an in-memory return-labeled replay stream."""
    labeled = exp020.ReturnLabeledReplays([replay], gamma=gamma, damage_shaping=0.0, win_reward=0.0)
    return list(WindowDataset(labeled, L_ctx, L_chunk, seed=seed, windows_per_replay=K, schema_version=6))


def test_window_threading_aligns_the_return_with_the_context_frames() -> None:
    """Perturb ONE frame's stock and G must change at exactly the window positions whose frame
    PRECEDES it, by exactly gamma^(distance) — the alignment the value target rests on. Checked on
    every window the sampler draws, with the sign following whichever port it made ego."""
    L_ctx, L_chunk, frames, death, gamma = 8, 2, 200, 120, 0.9
    replay = _mds_replay(frames)
    replay["p2_stock"][death:] = 3  # p2 loses a stock at frame 120
    windows = _windows(replay, L_ctx=L_ctx, L_chunk=L_chunk, gamma=gamma, K=8, seed=0)
    assert len(windows) >= 4
    seen_before_event = False
    for window in windows:
        pad = int(window["ctx_pad"])
        ego_is_p1 = float(window["ego_percent"][pad]) == _P1_MARKER
        sign = 1.0 if ego_is_p1 else -1.0  # +1 when the opponent is the one who died
        for position in range(pad, L_ctx):
            frame = int(window["frame"][position])
            expected = sign * gamma ** (death - frame) if frame <= death else 0.0
            assert window[exp020.EGO_RETURN_COLUMN][position] == pytest.approx(expected, rel=1e-5, abs=1e-7)
            seen_before_event |= frame < death
    assert seen_before_event, "no window landed before the event; the alignment was never exercised"


def test_returns_do_not_reach_the_model_as_a_feature() -> None:
    """The label rides the same columns as gamestate, so this pins that ``preprocess`` drops it:
    a return leaking into the observation would let the model read its own future."""
    L_ctx = 8
    replay = _mds_replay(60)
    replay["p2_stock"][40:] = 3
    windows = _windows(replay, L_ctx=L_ctx, L_chunk=2, gamma=0.9, K=2, seed=1)
    batch = exp020.collate_awr_batch(windows, stats=_stats(), L_ctx=L_ctx)
    assert batch.returns.shape == (len(windows), L_ctx)
    assert exp020.EGO_RETURN_COLUMN not in batch.batch.context.features
    assert "opp_awr_return" not in batch.batch.context.features
    assert batch.batch.context.batch == len(windows)


def test_collate_shifts_the_return_to_the_predicted_frame() -> None:
    """``batch.returns[b, t]`` must be the WINDOW column at ``t+1``: the return of the frame the
    offset-1 head predicts, so a reward that lands on the context frame itself (caused by the
    PREVIOUS action) never enters this action's advantage."""
    L_ctx = 8
    replay = _mds_replay(60)
    replay["p2_stock"][40:] = 3
    windows = _windows(replay, L_ctx=L_ctx, L_chunk=2, gamma=0.9, K=3, seed=2)
    batch = exp020.collate_awr_batch(windows, stats=_stats(), L_ctx=L_ctx)
    for b, window in enumerate(windows):
        np.testing.assert_allclose(
            batch.returns[b].numpy(), window[exp020.EGO_RETURN_COLUMN][1 : L_ctx + 1], rtol=0, atol=0
        )


def test_left_padded_positions_carry_a_zero_return() -> None:
    """A window that starts before the episode is left-padded by hal; the return column pads with
    it, and those positions are masked out of the loss by ctx_pad anyway."""
    replay = _mds_replay(20)
    replay["p2_stock"][10:] = 3
    labeled = exp020.ReturnLabeledReplays([replay], gamma=0.9, damage_shaping=0.0, win_reward=0.0)
    sampler = WindowDataset(labeled, 16, 2, seed=3, windows_per_replay=4, schema_version=6)
    padded = [w for w in sampler if int(w["ctx_pad"]) > 0]
    assert padded, "expected at least one cold-start window from a 20-frame replay"
    for window in padded:
        pad = int(window["ctx_pad"])
        np.testing.assert_array_equal(window[exp020.EGO_RETURN_COLUMN][:pad], np.zeros(pad, dtype=np.float32))


# --- the weights -------------------------------------------------------------


def test_weights_are_clipped_normalized_and_detached() -> None:
    advantage = torch.tensor([-3.0, -1.0, 0.0, 1.0, 8.0])
    weight, stats = exp020.awr_weights(advantage, beta=1.0, weight_max=5.0)
    raw = torch.exp(advantage).clamp(max=5.0)
    torch.testing.assert_close(weight, raw / raw.mean())
    assert weight.mean().item() == pytest.approx(1.0, rel=1e-6)
    assert weight.max().item() <= 5.0 / raw.mean().item() + 1e-6
    assert stats["weight_max_frac"] == pytest.approx(1 / 5)  # exp(1)=2.7 is under the cap, exp(8) is over it
    assert not weight.requires_grad


def test_ess_matches_the_definition() -> None:
    advantage = torch.tensor([0.0, 0.0, 0.0, math.log(9.0)])
    weight, stats = exp020.awr_weights(advantage, beta=1.0, weight_max=1e9)
    # raw = [1, 1, 1, 9] -> (sum w)^2 / (n * sum w^2) = 144 / (4 * 84)
    assert stats["ess"] == pytest.approx(144.0 / (4 * 84.0), rel=1e-6)
    assert weight.numel() == 4
    uniform, uniform_stats = exp020.awr_weights(torch.zeros(16), beta=1.0, weight_max=1e9)
    assert uniform_stats["ess"] == pytest.approx(1.0)
    torch.testing.assert_close(uniform, torch.ones(16))


def test_weights_refuse_an_attached_advantage() -> None:
    """A gradient path from the policy loss into V would make AWR self-referential."""
    with pytest.raises(ValueError, match="DETACHED"):
        exp020.awr_weights(torch.zeros(4, requires_grad=True), beta=1.0, weight_max=10.0)


def test_policy_loss_never_trains_the_value_head_and_the_value_loss_does() -> None:
    cfg = _tiny_cfg(exp020)
    model = _model(cfg)
    batch = _awr_batch(cfg)
    parts = exp020.action_loss(model, batch.batch)
    returns = batch.valid_returns(parts.valid)
    weight, _ = exp020.awr_weights(returns - parts.value.detach(), beta=cfg.awr_beta, weight_max=cfg.awr_weight_max)
    policy = exp020.objective(parts.nll, parts.transition, cfg.aux_loss_weight, cfg.transition_loss_weight, weight)
    policy.backward(retain_graph=True)
    assert model.value_head.weight.grad is None or torch.count_nonzero(model.value_head.weight.grad) == 0
    assert torch.count_nonzero(model.ctx_proj.weight.grad) > 0
    F.mse_loss(parts.value, returns).backward()
    assert torch.count_nonzero(model.value_head.weight.grad) > 0


def test_value_detach_trunk_gates_the_trunk_gradient() -> None:
    """Default False: the value MSE trains the shared trunk (the deployed 020 behavior — a second
    axis vs 016). True: the value gradient stops at the head, so the arm is pure reweighting."""
    cfg = _tiny_cfg(exp020)
    batch = _awr_batch(cfg)
    for detach, trunk_moves in ((False, True), (True, False)):
        model = _model(cfg)
        parts = exp020.action_loss(model, batch.batch, value_detach=detach)
        returns = batch.valid_returns(parts.valid)
        F.mse_loss(parts.value, returns).backward()
        assert torch.count_nonzero(model.value_head.weight.grad) > 0
        trunk_grads = [p.grad for p in model.blocks.parameters() if p.grad is not None]
        trunk_moved = any(torch.count_nonzero(g) > 0 for g in trunk_grads)
        assert trunk_moved == trunk_moves, f"value_detach={detach}: trunk gradient present={trunk_moved}"
        ctx_grad = model.ctx_proj.weight.grad
        ctx_moved = ctx_grad is not None and bool(torch.count_nonzero(ctx_grad) > 0)
        assert ctx_moved == trunk_moves, f"value_detach={detach}: ctx_proj gradient present={ctx_moved}"


def test_awr_val_metrics_are_finite() -> None:
    cfg = _tiny_cfg(exp020)
    model = _model(cfg)
    metrics, weights = exp020.awr_val_metrics(model, [_awr_batch(cfg), _awr_batch(cfg, seed=1)], cfg)
    assert set(metrics) == {
        "value_mse",
        "value_mean",
        "return_mean",
        "return_std",
        "awr_ess",
        "awr_weight_max_frac",
    }
    assert all(math.isfinite(v) for v in metrics.values())
    assert metrics["value_mse"] >= 0.0
    assert 0.0 < metrics["awr_ess"] <= 1.0
    assert weights.numel() > 0 and not weights.requires_grad


def test_action_loss_is_bitwise_repeatable() -> None:
    """No RNG in the objective: eval mode disables history dropout and nothing else samples."""
    cfg = _tiny_cfg(exp020)
    model = _model(cfg)
    batch = _awr_batch(cfg)
    first, second = exp020.action_loss(model, batch.batch), exp020.action_loss(model, batch.batch)
    for key, value in first.nll.items():
        torch.testing.assert_close(value, second.nll[key], rtol=0, atol=0)
    torch.testing.assert_close(first.value, second.value, rtol=0, atol=0)


# --- decode is 016's ---------------------------------------------------------


def test_decode_ignores_the_value_head() -> None:
    """Perturbing V must not move a single decoded action: the deployed policy is 016's."""
    cfg = _tiny_cfg(exp020)
    model = _model(cfg)
    ctx = Context(
        features=_features(exp020, 8, cfg.L_ctx, torch.Generator().manual_seed(2)),
        ctx_pad=torch.zeros(8, dtype=torch.long),
    )
    before = exp020.decode(model, ctx, gen=torch.Generator().manual_seed(5))
    with torch.no_grad():
        model.value_head.weight.mul_(100.0).add_(3.0)
    after = exp020.decode(model, ctx, gen=torch.Generator().manual_seed(5))
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_closed_loop_policy_samples() -> None:
    cfg = _tiny_cfg(exp020)
    model = _model(cfg)
    ctx = Context(
        features=_features(exp020, 8, cfg.L_ctx, torch.Generator().manual_seed(3)),
        ctx_pad=torch.zeros(8, dtype=torch.long),
    )
    chunks = [
        exp020.make_policy(model, _stats(), cfg, device="cpu", decode_seed=seed).predict_chunk(ctx, None)
        for seed in (0, 1)
    ]
    assert not (chunks[0] == chunks[1]).all(), "policy decoded greedily"


# --- configuration -----------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("awr_gamma", 1.5, "awr_gamma"),
        ("awr_gamma", 0.0, "awr_gamma"),
        ("awr_beta", 0.0, "awr_beta"),
        ("awr_beta", float("inf"), "awr_beta"),
        ("awr_weight_max", 0.0, "awr_weight_max"),
        ("awr_value_loss_weight", -1.0, "awr_value_loss_weight"),
        ("awr_damage_shaping", -0.5, "awr_damage_shaping"),
    ],
)
def test_validate_config_rejects_bad_awr_knobs(field: str, value: float, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        exp020.validate_config(_tiny_cfg(exp020, **{field: value}), has_button_combo_counts=False)


def test_attach_returns_rejects_a_foreign_loader() -> None:
    """The seam is a WindowDataset; anything else must fail loud rather than silently train
    on batches with no return label."""

    class NotALoader:
        dataset = object()

    with pytest.raises(TypeError, match="WindowDataset"):
        exp020._attach_returns(NotALoader(), exp020.TrainConfig(), _stats())


# --- end to end on real data -------------------------------------------------


@pytest.mark.skipif(not (_DEV_MDS / "train").is_dir(), reason="local dev MDS not materialized")
def test_mini_train_on_dev_mds(tmp_path, monkeypatch, capsys) -> None:
    """Four real training steps over the dev MDS: the loader labels returns, the collate builds
    AWRBatch, the value head fits, and the weights come out with a usable ESS."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("AWS_BUCKET", "hal-test")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://127.0.0.1:9")  # discard port: uploads fail instantly
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setattr(exp020, "eval_vs_cpu", lambda *a, **k: {})  # no Dolphin in a unit test

    class Uploader:
        def __init__(self, _run_name: str) -> None:
            pass

        def upload(self, *args, **kwargs) -> None:
            pass

        def upload_tree(self, *args, **kwargs) -> int:
            return 0

        def close(self) -> None:
            pass

    monkeypatch.setattr(exp020, "BackgroundUploader", Uploader)

    cfg = exp020.TrainConfig(
        d_model=32,
        n_layers=2,
        n_heads=2,
        L_ctx=32,
        head_offsets=(1, 2),
        batch_size=4,
        max_steps=4,
        warmup_steps=1,
        val_every=2,
        val_n_batches=2,
        gradient_diagnostic_batch_size=2,
        eval_every=0,
        ckpt_every=0,
        num_workers=0,
        windows_per_replay=2,
        data_root=str(_DEV_MDS),
        val_split="train",
        mds_schema_version=5,
    )
    stats = {k: FeatureStats(mean=0.0, std=1.0, min=-1.0, max=1.0) for k in _stats()}
    exp020.train(cfg, stats, comment="pytest")
    out = capsys.readouterr().out
    assert "step 3:" in out
    assert (Path(os.getcwd()) / "runs").is_dir()
