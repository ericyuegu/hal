"""Regression guard for the closed-loop policy's rolling-buffer alignment.

``RecedingHorizon`` builds its model context by pairing, at each position, a
past gamestate with the ego action that *produced* it. If the rolling buffers
ever drift out of lockstep, the model would see a frame-shifted observation at
inference that it never saw in training. This pins the invariant.

The policy lives in ``hal.training.closed_loop``; the experiment (loaded by path,
since its filename starts with a digit) wires a model into it via ``make_policy``.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from hal.data.stats import FeatureStats
from hal.sim.vec import Slot

_EXP_PATH = Path(__file__).resolve().parent.parent / "experiments" / "002_flow_matching_rtc.py"


def _load_experiment():
    spec = importlib.util.spec_from_file_location("exp002", _EXP_PATH)
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


def _build_policy(*, inference_delay: int = 0, execution_horizon: int = 4):
    cfg = exp.TrainConfig(
        d_model=16,
        n_layers=1,
        n_heads=2,
        dim_feedforward=16,
        time_emb_dim=8,
        dropout=0.0,
        L_ctx=4,
        L_chunk=4,
        inference_delay=inference_delay,
        execution_horizon=execution_horizon,
        n_flow_steps=1,
    )
    torch.manual_seed(0)
    model = exp.FlowMatchingPolicy(cfg)
    model.eval()
    policy = exp.make_policy(model, _stats(), cfg, device="cpu")
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


def test_rtc_commits_previous_chunks_prefix():
    """With d>0 the new chunk is conditioned on the previous chunk's [s : s+d]
    actions, and the integrator pins those d positions to that committed prefix —
    this is what makes the real-time-chunking handoff continuous."""
    d, s = 2, 2
    cfg, policy = _build_policy(inference_delay=d, execution_horizon=s)
    slot = Slot(0, 1)

    committed_seen: list[np.ndarray | None] = []
    pendings: list[np.ndarray] = []
    real_predict = policy.predict_chunk
    real_replan = policy._replan

    def spy_predict(ctx, committed):
        committed_seen.append(None if committed is None else committed.copy())
        return real_predict(ctx, committed)

    def spy_replan(live):
        real_replan(live)
        pendings.append(policy._slots[slot].pending.copy())

    policy.predict_chunk = spy_predict
    policy._replan = spy_replan

    for t in range(4 * s):  # bootstrap + several steady-state replans
        policy(t, {slot: _obs(t, 1)})

    assert committed_seen[0] is None, "bootstrap has no committed prefix"
    assert len(pendings) >= 3
    for i in range(1, len(pendings)):
        prefix = committed_seen[i]
        assert prefix is not None and prefix.shape == (1, d, exp.A_DIM)
        # conditioned on the previous chunk's [s : s+d]
        np.testing.assert_allclose(prefix[0], pendings[i - 1][s : s + d], rtol=1e-5, atol=1e-6)
        # and the integrator pinned the new chunk's first d positions to it
        np.testing.assert_allclose(pendings[i][:d], prefix[0], rtol=1e-5, atol=1e-6)
