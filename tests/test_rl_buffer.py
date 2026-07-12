"""Rollout window-assembly + GAE plumbing tests (pure CPU, fast).

Pins the alignment contract M4b depends on: our ``gae_inputs`` reproduces
tianshou's episodic-return math (private ``_gae`` cross-checked against the public
``compute_episodic_return``), the terminated/truncated off-by-ones mask/bootstrap
the right value, windows tile a stream without spanning an episode boundary, the
scatter map round-trips, and a window's stacked features are byte-identical to
``RecedingHorizon._build_stacked_batch`` on the same synthetic history.
"""

import numpy as np
from nets_melee import N_GROUPS
from rollout import FinalizedStream
from rollout import build_windows
from rollout import gae_inputs
from rollout import scatter_gae
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.data import Batch
from tianshou.data import VectorReplayBuffer

from hal.training.closed_loop import RecedingHorizon
from hal.training.closed_loop import _SlotState
from hal.training.features import A_DIM
from hal.wire import POST_FIELD_SUFFIXES

GAMMA = 0.997
LAMBDA = 0.95

_FLAT_PREFIXES = ("p1", "p2", "p1_nana", "p2_nana")


def _flat_frame(i: int, rng: np.random.Generator) -> dict:
    """A flat gamestate frame with the full key set flatten_canonical_frame emits
    (per-player post fields + nana + stage/character), values seeded by frame id."""
    out: dict[str, float] = {}
    for p in _FLAT_PREFIXES:
        for suf in POST_FIELD_SUFFIXES:
            out[f"{p}_{suf}"] = float(rng.standard_normal())
    out["stage"] = 3
    out["p1_character"] = 1
    out["p2_character"] = 2
    out["_seed"] = float(i)  # a benign extra column to prove generic key handling
    return out


def _make_stream(
    T: int,
    *,
    terminated_at: list[int] | None = None,
    truncated_at: list[int] | None = None,
    ego_port: int = 1,
    seed: int = 0,
) -> FinalizedStream:
    rng = np.random.default_rng(seed)
    terminated = np.zeros(T, bool)
    truncated = np.zeros(T, bool)
    for t in terminated_at or []:
        terminated[t] = True
    for t in truncated_at or []:
        truncated[t] = True
    return FinalizedStream(
        ego_port=ego_port,
        matchup=None,
        flat=tuple(_flat_frame(i, rng) for i in range(T + 1)),
        ego_act=rng.standard_normal((T, A_DIM)).astype(np.float32),
        act_idx=rng.integers(0, 5, size=(T, N_GROUPS)).astype(np.int64),
        logp=rng.standard_normal(T).astype(np.float32),
        rew=rng.standard_normal(T).astype(np.float32),
        terminated=terminated,
        truncated=truncated,
    )


# --- (a) GAE cross-check against tianshou's public episodic-return ------------
def test_gae_matches_compute_episodic_return() -> None:
    T = 12
    stream = _make_stream(T, terminated_at=[3, 7], truncated_at=[11], seed=1)
    v = np.random.default_rng(2).standard_normal(T + 1).astype(np.float32)

    ((adv, returns),) = gae_inputs([stream], [v], gamma=GAMMA, gae_lambda=LAMBDA)

    # Feed the same transitions to a real buffer and use tianshou's public path.
    buf = VectorReplayBuffer(T, buffer_num=1)
    ids = np.zeros(1, dtype=np.int64)
    for t in range(T):
        buf.add(
            Batch(
                obs=np.zeros((1, 1), np.float32),
                act=np.zeros((1,), np.float32),
                rew=stream.rew[t : t + 1],
                terminated=stream.terminated[t : t + 1],
                truncated=stream.truncated[t : t + 1],
                obs_next=np.zeros((1, 1), np.float32),
            ),
            buffer_ids=ids,
        )
    batch, indices = buf.sample(0)
    ref_returns, ref_adv = Algorithm.compute_episodic_return(
        batch, buf, indices, v_s_=v[1:], v_s=v[:T], gamma=GAMMA, gae_lambda=LAMBDA
    )
    assert np.allclose(adv, ref_adv, atol=1e-5)
    assert np.allclose(returns, ref_returns, atol=1e-5)


# --- (b) terminated masks the bootstrap; truncated keeps it -------------------
def test_terminated_vs_truncated_bootstrap() -> None:
    # One transition, huge next-state value. Terminated => v_s_ masked to 0, so
    # adv = rew - v_s. Truncated => v_s_ kept, so adv = rew + gamma*v_next - v_s.
    v = np.array([1.0, 100.0], np.float32)
    term = _make_stream(1, terminated_at=[0], seed=3)
    trunc = _make_stream(1, truncated_at=[0], seed=3)  # same rew/values (same seed)
    ((adv_t, _),) = gae_inputs([term], [v], gamma=GAMMA, gae_lambda=LAMBDA)
    ((adv_r, _),) = gae_inputs([trunc], [v], gamma=GAMMA, gae_lambda=LAMBDA)
    rew = float(term.rew[0])
    assert np.isclose(adv_t[0], rew - v[0], atol=1e-5)
    assert np.isclose(adv_r[0], rew + GAMMA * v[1] - v[0], atol=1e-5)


# --- (c) window tiling: coverage, ctx_pad, no boundary span, scatter ----------
def test_short_segment_single_left_padded_window() -> None:
    L_ctx = 16
    T = 5  # 6 frames <= L_ctx -> one window, left-padded
    stream = _make_stream(T, truncated_at=[T - 1], seed=4)
    windows = build_windows(stream, L_ctx)
    assert len(windows) == 1
    w = windows[0]
    assert w.ctx_pad == L_ctx - (T + 1)
    # Real rows sit on the right; pad rows carry frame_pos -1 and valid False.
    assert (w.frame_pos[: w.ctx_pad] == -1).all()
    assert (w.frame_pos[w.ctx_pad :] == np.arange(T + 1)).all()
    assert w.valid.sum() == T  # frames 0..T-1 scored; bootstrap frame T value-only
    assert not w.valid[-1]  # last real row is the bootstrap frame


def test_long_segment_non_overlapping_cover() -> None:
    L_ctx = 8
    T = 20
    stream = _make_stream(T, truncated_at=[T - 1], seed=5)
    windows = build_windows(stream, L_ctx)
    # Every frame 0..T appears exactly once across windows; scored frames 0..T-1 valid once.
    seen = np.concatenate([w.frame_pos[w.frame_pos >= 0] for w in windows])
    assert sorted(seen.tolist()) == list(range(T + 1))
    scored = np.concatenate([w.frame_pos[w.valid] for w in windows])
    assert sorted(scored.tolist()) == list(range(T))
    # Trailing window is a full L_ctx block ending at the last frame T.
    last = windows[-1]
    assert last.ctx_pad == 0
    assert int(last.frame_pos[-1]) == T


def test_no_window_spans_episode_boundary() -> None:
    L_ctx = 8
    T = 20
    stream = _make_stream(T, terminated_at=[6, 13], truncated_at=[T - 1], seed=6)
    windows = build_windows(stream, L_ctx)
    assert windows  # sanity: some windows were produced
    for w in windows:
        real = w.frame_pos[w.frame_pos >= 0]
        a, b = int(real[0]), int(real[-1])
        # No terminated transition strictly inside the window's frame span.
        assert not stream.terminated[a:b].any(), f"window {a}..{b} spans a boundary"


def test_scatter_round_trips_through_frame_pos() -> None:
    L_ctx = 8
    T = 15
    stream = _make_stream(T, terminated_at=[5], truncated_at=[T - 1], seed=7)
    adv = np.arange(T, dtype=np.float32) + 0.5  # distinct per transition
    returns = -(np.arange(T, dtype=np.float32) + 0.5)
    for w in build_windows(stream, L_ctx):
        sw = scatter_gae(w, adv, returns)
        for row in range(L_ctx):
            if sw.valid[row]:
                fp = int(sw.frame_pos[row])
                assert sw.adv[row] == adv[fp]
                assert sw.returns[row] == returns[fp]
            else:  # pad + bootstrap rows stay zero
                assert sw.adv[row] == 0.0 and sw.returns[row] == 0.0


# --- (d) feature parity with RecedingHorizon._build_stacked_batch -------------
def _receding_horizon() -> RecedingHorizon:
    return RecedingHorizon(predict_chunk=lambda ctx, committed: None, stats={}, L_ctx=8, L_chunk=1, s=1, d=0)


def _slot_for(rh: RecedingHorizon, flat_hist: list[dict], ego_hist: list[np.ndarray]):
    from hal.sim.vec import Slot

    slot = Slot(match=0, port=1)
    rh._slots[slot] = _SlotState(flat_hist=list(flat_hist), ego_inputs_hist=list(ego_hist))
    return slot


def _assert_window_matches_stacked(stream: FinalizedStream, L_ctx: int) -> None:
    windows = build_windows(stream, L_ctx)
    last = windows[-1]
    T = stream.n_transitions
    # Reconstruct the replan-time rolling state RecedingHorizon would hold at frame T:
    # the trailing <=L_ctx flat frames and the matching one-short ego history.
    n = int((last.frame_pos >= 0).sum())
    lo = T + 1 - n  # first real frame index in the trailing window
    flat_hist = list(stream.flat[lo : T + 1])
    ego_lo = lo if lo == 0 else lo - 1
    ego_hist = list(stream.ego_act[ego_lo:T])
    rh = _receding_horizon()
    slot = _slot_for(rh, flat_hist, ego_hist)
    ref = rh._build_stacked_batch([slot])  # {k: [1, L_ctx]}
    for k, v in ref.items():
        assert k in last.feats, f"missing feature {k}"
        assert np.array_equal(last.feats[k], v[0]), f"feature {k} mismatch"


def test_parity_short_history() -> None:
    # Cold-start regime: trailing window shorter than L_ctx, ego one-short (front-pad 1).
    _assert_window_matches_stacked(_make_stream(5, truncated_at=[4], seed=8), L_ctx=8)


def test_parity_full_history() -> None:
    # Steady-state regime: trailing window is a full L_ctx block (front-pad 0).
    _assert_window_matches_stacked(_make_stream(20, truncated_at=[19], seed=9), L_ctx=8)
