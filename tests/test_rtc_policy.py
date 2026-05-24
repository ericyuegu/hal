"""Regression guard for the RTC policy's rolling-buffer alignment.

experiments/001 builds its model context by pairing, at each position, a past
gamestate with the ego action that *produced* it. If the rolling buffers ever
drift out of lockstep, the model would see a frame-shifted observation at
inference that it never saw in training. This pins the invariant.

The experiment filename starts with a digit, so it's loaded by path.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from hal.data.stats import FeatureStats
from hal.sim.vec import Slot

_EXP_PATH = Path(__file__).resolve().parent.parent / "experiments" / "001_flow_matching_rtc_baseline.py"


def _load_experiment():
    spec = importlib.util.spec_from_file_location("exp001", _EXP_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


exp = _load_experiment()

_FLOAT_KEYS = ("position_x", "position_y", "percent", "shield", "direction", "hitlag_left")


def _stats() -> dict[str, FeatureStats]:
    return {k: FeatureStats(mean=0.0, std=1.0, min=-1.0, max=1.0) for k in _FLOAT_KEYS}


def _post(position_x: float) -> dict:
    return {
        "position": {"x": position_x, "y": 0.0},
        "percent": 0.0,
        "shield": 60.0,
        "stock": 4,
        "direction": 1.0,
        "action": 14,
        "jumps_used": 0,
        "airborne": 0,
        "hurtbox_state": 0,
        "hitlag_left": 0.0,
        "state_age": 0.0,
    }


def _obs(call_idx: int, ego_port: int) -> dict:
    """Canonical frame whose EGO position_x is tagged with the call index, so we
    can recover which gamestate landed at each context position."""
    opp_port = 2 if ego_port == 1 else 1
    return {
        "id": call_idx,
        "start": {"random_seed": 0},
        "ports": {
            ego_port: {"leader": {"post": _post(float(call_idx))}},
            opp_port: {"leader": {"post": _post(-1.0)}},
        },
    }


def _build_policy():
    cfg = exp.TrainConfig(
        d_model=16,
        n_layers=1,
        n_heads=2,
        dim_feedforward=16,
        time_emb_dim=8,
        dropout=0.0,
        ego_history_dropout_prob=0.0,
        L_ctx=4,
        L_chunk=4,
        latency_frames=0,
        n_flow_steps=1,
    )
    torch.manual_seed(0)
    model = exp.FlowMatchingPolicy(cfg)
    model.eval()
    policy = exp.FlowMatchingBatchPolicy(
        model=model,
        stats=_stats(),
        L_ctx=cfg.L_ctx,
        L_chunk=cfg.L_chunk,
        n_lat=cfg.latency_frames,
        n_flow_steps=cfg.n_flow_steps,
        device="cpu",
    )
    return cfg, policy


def test_context_pairs_each_gamestate_with_the_action_that_produced_it():
    """At a steady-state replan, the ego action at context position i must be the
    action the policy returned at the call that produced gamestate i (which is
    one frame earlier). Buffers drifting apart would break this."""
    cfg, policy = _build_policy()
    ego_port = 1
    slot = Slot(0, ego_port)

    captured: list[dict[str, np.ndarray]] = []
    real_build = policy._build_stacked_batch
    real_push = policy._push_ego

    def spy_build(live):
        batch = real_build(live)
        captured.append({k: v.copy() for k, v in batch.items()})
        return batch

    returned: list[np.ndarray] = []

    def spy_push(s, a):
        if s == slot:
            returned.append(np.asarray(a, dtype=np.float32).copy())
        real_push(s, a)

    policy._build_stacked_batch = spy_build
    policy._push_ego = spy_push

    for t in range(4 * cfg.L_ctx):
        policy(t, {slot: _obs(t, ego_port)})

    assert captured, "policy never replanned"
    batch = captured[-1]  # last == steady state (bootstrap pad already flushed)
    frames = batch["ego_position_x"][0].astype(int)  # gamestate frame per position
    ego_main_x = batch["ego_main_stick_x"][0]  # raw ego action channel 0

    # The whole window is one contiguous slice of frames.
    assert list(frames) == list(range(int(frames[0]), int(frames[0]) + cfg.L_ctx))
    # Steady state: every position carries a real prior action (no pad left).
    assert int(frames[0]) - 1 >= 0
    for i, f in enumerate(frames):
        assert ego_main_x[i] == pytest.approx(returned[f - 1][0]), f"position {i} (frame {f}) misaligned"
