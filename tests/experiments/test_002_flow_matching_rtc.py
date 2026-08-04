"""Regression guards for the closed-loop policy's rolling-buffer invariants.

``RecedingHorizon`` builds its model context by pairing, at each position, a
past gamestate with the ego action that *produced* it. If the rolling buffers
ever drift out of lockstep, the model would see a frame-shifted observation at
inference that it never saw in training. This pins the alignment invariant.

It also clears those buffers at an instant-restart match boundary (the frame id
resetting to the pre-game countdown), so a new match never opens on a context
spanning two matches — a window with zero training support. This pins that too.

The policy lives in ``hal.training.closed_loop``; the experiment (loaded by path,
since its filename starts with a digit) wires a model into it via ``make_policy``.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from hal.data.feature_stats import FeatureStats
from hal.sim.vec import Slot
from hal.training.features import ACTION_CHANNELS

_EXP_PATH = Path(__file__).resolve().parent.parent.parent / "experiments" / "002_flow_matching_rtc.py"


def _load_experiment():
    spec = importlib.util.spec_from_file_location("exp002", _EXP_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


exp = _load_experiment()

_FLOAT_KEYS = ("position_x", "position_y", "percent", "shield", "direction", "hitlag_left")


def _named(windows) -> dict[str, np.ndarray]:
    """One stacked replan batch as ``name -> [B, L]``. The stats below are the identity
    on both transforms, so a normalized column still reads as its raw value."""
    n_value = len(windows.layout.value_names)
    named = dict(zip(windows.layout.value_names, windows.floats[:n_value], strict=True))
    named.update(zip(windows.layout.cat_names, windows.cats, strict=True))
    return named


def _stats() -> dict[str, FeatureStats]:
    # The obs bridge emits a (masked) nana follower block, so preprocess needs nana float stats too.
    keys = (*_FLOAT_KEYS, *(f"nana_{k}" for k in _FLOAT_KEYS))
    return {k: FeatureStats(mean=0.0, std=1.0, min=-1.0, max=1.0) for k in keys}


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
    real_build = policy._stack_windows
    real_push = policy._push_ego

    def spy_build(live, length):
        windows = real_build(live, length)
        captured.append({k: v.copy() for k, v in _named(windows).items()})
        return windows

    returned: list[np.ndarray] = []

    def spy_push(s, a):
        if s == slot:
            returned.append(np.asarray(a, dtype=np.float32).copy())
        real_push(s, a)

    policy._stack_windows = spy_build
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

    def spy_replan(live, committed):
        real_replan(live, committed)
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


def _obs_id(*, frame_id: int, tag: float, ego_port: int) -> dict:
    """Canonical frame with an explicit ``id`` (so a drop forces an instant-restart reset)
    and ego ``position_x`` set to ``tag`` (so we can recover which frames survive in the
    post-reset context)."""
    opp_port = 2 if ego_port == 1 else 1
    return {
        "id": frame_id,
        "start": {"random_seed": 0},
        "ports": {
            ego_port: {"leader": {"post": _post(tag)}},
            opp_port: {"leader": {"post": _post(-1.0)}},
        },
    }


def test_buffers_reset_at_instant_restart_match_boundary():
    """Instant-restart plays many matches per Dolphin boot; each new match's frame id resets
    to the pre-game countdown (drops below the prior match's). The policy must clear its
    rolling buffers at that drop, so a match never opens on the previous match's gamestate +
    ego context — a full L_ctx window spanning two matches has zero training support."""
    cfg, policy = _build_policy(execution_horizon=1)  # s == 1 → the policy replans every frame
    slot = Slot(0, 1)

    ctx_pads: list[int] = []
    batches: list[dict[str, np.ndarray]] = []
    real_predict = policy.predict_chunk
    real_build = policy._stack_windows

    def spy_predict(ctx, committed):
        ctx_pads.append(int(ctx.ctx_pad[0]))
        return real_predict(ctx, committed)

    def spy_build(live, length):
        windows = real_build(live, length)
        batches.append({k: v.copy() for k, v in _named(windows).items()})
        return windows

    policy.predict_chunk = spy_predict
    policy._stack_windows = spy_build

    # Match 1: a rising id run long enough to saturate the L_ctx buffer. Match 2: the id
    # drops to a new countdown then rises again — the instant-restart boundary. Each frame's
    # ego position_x is tagged with the call index so the surviving context is identifiable.
    pre_ids = list(range(390, 390 + 2 * cfg.L_ctx))
    post_ids = list(range(-123, -123 + 2 * cfg.L_ctx))
    reset_call = len(pre_ids)  # first post-reset call; with s == 1 also the first post-reset replan
    for t, fid in enumerate(pre_ids + post_ids):
        policy(t, {slot: _obs_id(frame_id=fid, tag=float(t), ego_port=1)})

    # Precondition: the buffer saturated before the reset (ctx_pad hit 0), so the reset is
    # clearing a full two-match-spanning window, not an already-short one.
    assert ctx_pads[reset_call - 1] == 0, "buffer never saturated before the reset"

    # First post-reset replan: only the single post-reset frame is in context.
    assert ctx_pads[reset_call] == cfg.L_ctx - 1

    # Ego history is fully neutral-padded — no ego action carried across the boundary.
    reset_batch = batches[reset_call]
    for ch in ACTION_CHANNELS:
        col = reset_batch[f"ego_{ch}"][0]
        assert np.all(col == 0.0), f"ego channel {ch} not neutral after reset: {col}"

    # Once the buffer refills, every gamestate in context is a post-reset frame (tag >=
    # reset_call); no pre-reset frame ever leaks back in.
    saturated_call = reset_call + cfg.L_ctx - 1
    assert ctx_pads[saturated_call] == 0, "buffer did not refill after the reset"
    tags = batches[saturated_call]["ego_position_x"][0]
    assert np.all(tags >= reset_call), f"a pre-reset frame leaked into post-reset context: {tags}"


def test_async_restart_discards_only_that_slots_pending_chunk():
    """With s>1, a reset between shared boundaries must replan that slot immediately.

    The other boot must execute the remainder of its existing chunk; otherwise instant
    restarts make execution depend on unrelated boots' match lengths and concurrency.
    """
    cfg, policy = _build_policy(execution_horizon=4)
    reset_slot = Slot(0, 1)
    steady_slot = Slot(1, 1)
    calls: list[tuple[int, list[int]]] = []

    def tagged_predict(ctx, committed):
        assert committed is None
        call = len(calls)
        calls.append((ctx.ctx_pad.shape[0], ctx.ctx_pad.tolist()))
        plans = np.zeros((ctx.ctx_pad.shape[0], cfg.L_chunk, exp.A_DIM), dtype=np.float32)
        for row in range(plans.shape[0]):
            plans[row, :, 0] = 0.1 * (call + 1) + 0.01 * row + 0.001 * np.arange(cfg.L_chunk)
        return plans

    policy.predict_chunk = tagged_predict
    policy(
        0,
        {
            reset_slot: _obs_id(frame_id=100, tag=0.0, ego_port=1),
            steady_slot: _obs_id(frame_id=100, tag=0.0, ego_port=1),
        },
    )
    policy(
        1,
        {
            reset_slot: _obs_id(frame_id=101, tag=1.0, ego_port=1),
            steady_slot: _obs_id(frame_id=101, tag=1.0, ego_port=1),
        },
    )
    old_reset_chunk = policy._slots[reset_slot].pending.copy()
    steady_chunk = policy._slots[steady_slot].pending.copy()

    policy(
        2,
        {
            reset_slot: _obs_id(frame_id=-123, tag=2.0, ego_port=1),
            steady_slot: _obs_id(frame_id=102, tag=2.0, ego_port=1),
        },
    )

    assert calls == [(2, [cfg.L_ctx - 1, cfg.L_ctx - 1]), (1, [cfg.L_ctx - 1])]
    reset_state = policy._slots[reset_slot]
    steady_state = policy._slots[steady_slot]
    assert reset_state.offset == 1
    assert steady_state.offset == 3
    assert reset_state.pending[0, 0] != pytest.approx(old_reset_chunk[2, 0])
    assert reset_state.last_action[0] == pytest.approx(reset_state.pending[0, 0])
    assert steady_state.last_action[0] == pytest.approx(steady_chunk[2, 0])
