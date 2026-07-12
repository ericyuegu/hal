"""Melee self-play collector tests.

Pure-CPU tests (no Dolphin) drive :class:`RLBatchPolicy` over scripted obs sequences
through a tiny net and pin the load-bearing contracts: episode segmentation at an
instant-restart ``id`` drop, the terminal bonus landing on the last pre-reset step,
the behavior ``(act_idx, logp)`` recorded matching what the acting policy emitted, the
ego-action alignment (``a_t`` is the ego channel of frame ``t+1``), finite rewards, the
truncated finalize tail, KV-cache reset on a boundary, and sync-mode determinism. The
KEY invariant — the KV-incremental collection log-prob equals the ``forward_full`` PPO
recompute pre-eviction (``ratio_dev_epoch0 ≈ 0``) — is checked directly.

``test_collector_real_dolphin`` (``@pytest.mark.integration``) runs the same collector
against two real warm-started Dolphins and asserts a well-formed ``RolloutIteration``.
"""

import multiprocessing as mp

# libmelee's slippstream client spawns a child via mp.Process; force plain fork on 3.14
# (the other integration tests do the same) — no-op for the pure-CPU unit tests.
if mp.get_start_method(allow_none=True) != "fork":
    mp.set_start_method("fork", force=True)

from pathlib import Path

import numpy as np
import pytest
import torch
from melee_collector import NetActingPolicy
from melee_collector import RLBatchPolicy
from nets_melee import ArchConfig
from nets_melee import FactoredCategorical
from nets_melee import PolicyValueNet
from rl_config import RewardConfig
from rollout import build_windows
from rollout import collate_windows

from hal.sim.vec import Slot
from hal.training.stats import load_consolidated_stats

CFG = ArchConfig(d_model=32, n_layers=2, n_heads=2, L_ctx=16, char_vocab=8, char_dim=4, stage_vocab=8, stage_dim=2)
STATS_PATH = Path("data/processed/ranked-anonymized-1/mds/stats.json")


def _stats() -> dict:
    if not STATS_PATH.is_file():
        pytest.skip(f"dataset stats missing at {STATS_PATH}")
    return load_consolidated_stats(STATS_PATH)


def _post(x: float = 1.0, pct: float = 0.0, stock: float = 4.0) -> dict:
    return {
        "position": {"x": x, "y": 2.0},
        "percent": pct,
        "shield": 60.0,
        "stock": stock,
        "direction": 1.0,
        "action": 14,
        "jumps_used": 0,
        "airborne": 0,
        "hurtbox_state": 0,
        "hitlag_left": 0.0,
    }


def _frame(i: int, *, p1_pct: float = 0.0, p1_stock: float = 4.0, p2_pct: float = 0.0, p2_stock: float = 4.0) -> dict:
    return {
        "id": i,
        "stage": 3,
        "ports": {
            1: {"leader": {"post": _post(1.0, p1_pct, p1_stock)}},
            2: {"leader": {"post": _post(-1.0, p2_pct, p2_stock)}},
        },
        "_matchup": {"stage": 3, "character": {1: 1, 2: 1}},
    }


def _make_policy(
    *,
    n_slots: int = 1,
    refresh_every: int = 4,
    rollout_frames: int = 10_000,
    seed: int = 0,
    recorder: list | None = None,
) -> tuple[RLBatchPolicy, list[Slot], NetActingPolicy]:
    torch.manual_seed(0)
    net = PolicyValueNet(CFG).eval()
    handle: NetActingPolicy | _RecordingHandle = NetActingPolicy(net, n_slots=n_slots, device="cpu", seed=seed)
    if recorder is not None:
        handle = _RecordingHandle(handle, recorder)
    slots = [Slot(0, p) for p in ([1] if n_slots == 1 else [1, 2])]
    pol = RLBatchPolicy(
        handles={"ema": handle},
        assign=lambda s: "ema",
        slots=slots,
        ego_port_of=lambda s: s.port,
        matchup_of=lambda s: {"stage": 3, "character": {1: 1, 2: 1}},
        reward_cfg=RewardConfig(),
        stats=_stats(),
        L_ctx=CFG.L_ctx,
        refresh_every=refresh_every,
        rollout_frames=rollout_frames,
    )
    return pol, slots, net


class _RecordingHandle:
    """Wraps an ActingPolicy, logging every ``step`` output so tests can compare the
    emitted (act_idx, logp) against what the stream recorded."""

    def __init__(self, inner: NetActingPolicy, log: list) -> None:
        self.inner = inner
        self.log = log

    def step(self, rows, ctx):
        act_idx, logp, vecs = self.inner.step(rows, ctx)
        self.log.append((rows.tolist(), act_idx.copy(), logp.copy(), vecs.copy()))
        return act_idx, logp, vecs

    def rebuild(self, rows, windows):
        self.inner.rebuild(rows, windows)

    def reset_slot(self, row):
        self.inner.reset_slot(row)


def _win_then_reset() -> list[dict]:
    """Six in-game frames (ids 0..5) with ego (p1) taking p2's stock at frame 5, then a
    single reset frame (id 0) that opens a fresh match."""
    seq = [_frame(i, p1_pct=i * 2.0, p2_pct=i * 5.0, p2_stock=(4.0 if i < 5 else 1.0)) for i in range(6)]
    seq.append(_frame(0))  # id drops -> instant-restart boundary
    return seq


def test_segments_at_reset_with_terminal_bonus_on_last_pre_reset_step() -> None:
    pol, (slot,), _ = _make_policy()
    for t, f in enumerate(_win_then_reset()):
        pol(t, {slot: f})

    st = pol.state[slot].stream
    # frames 0..5 -> steps 0..4 (each step's pre-obs is the previous frame); the reset
    # discards the post-frame-5 pending action and credits the terminal to step 4.
    assert st.n_recorded == 5
    terminated = np.array(st._terminated)
    assert terminated.tolist() == [False, False, False, False, True]
    # p1 won (p2 lost a stock) -> +win_bonus lands on the last recorded step.
    assert st._rew[-1] > RewardConfig().win_bonus
    assert np.all(np.isfinite(np.array(st._rew)))


def test_kv_cache_resets_on_episode_boundary() -> None:
    pol, (slot,), _ = _make_policy(refresh_every=1000)  # no rebuild — isolate the reset
    handle = pol.handles["ema"]
    for t, f in enumerate(_frame(i) for i in range(4)):
        pol(t, {slot: f})
    assert int(handle.caches.length[0]) == 4
    pol(4, {slot: _frame(0)})  # reset: cache cold-started, then this new frame stepped once
    assert int(handle.caches.length[0]) == 1


def test_recorded_behavior_matches_acting_policy() -> None:
    log: list = []
    pol, (slot,), _ = _make_policy(recorder=log)
    seq = [_frame(i, p1_pct=i, p2_pct=2 * i) for i in range(6)]
    for t, f in enumerate(seq):
        pol(t, {slot: f})

    st = pol.state[slot].stream
    # Action emitted at frame t is recorded when frame t+1 completes it; the last emitted
    # action (frame 5) stays pending. So recorded steps 0..4 == emitted actions 0..4.
    emitted_idx = [entry[1][0] for entry in log]  # per-frame act_idx (n_slots==1)
    emitted_logp = [entry[2][0] for entry in log]
    assert st.n_recorded == len(seq) - 1
    for t in range(st.n_recorded):
        assert np.array_equal(st._act_idx[t], emitted_idx[t])
        assert st._logp[t] == pytest.approx(emitted_logp[t], abs=1e-6)
        assert np.array_equal(st._ego_act[t], log[t][3][0])  # stored ego_act == emitted action vec


def test_collection_logp_matches_ppo_recompute_pre_eviction() -> None:
    """The KV-incremental behavior log-prob equals a full ``forward_full`` recompute over
    the same window — this is ``ratio_dev_epoch0 ≈ 0`` before any KV eviction, the
    invariant that keeps the PPO importance weights honest."""
    pol, (slot,), net = _make_policy(refresh_every=4)
    seq = [_frame(i, p1_pct=0.5 * i, p2_pct=0.7 * i) for i in range(12)]
    for t, f in enumerate(seq):
        pol(t, {slot: f})

    st = pol.state[slot].stream
    st.append_boundary(pol.state[slot].flat_hist[-1])
    windows = build_windows(st.finalize(), CFG.L_ctx, stream_id=0)
    batch = collate_windows(windows, _stats())
    with torch.no_grad():
        dist = FactoredCategorical(net.policy_logits(net.forward_full(batch.context)))
        logp_recompute = dist.log_prob(batch.act_idx)
    v = batch.valid
    assert float((logp_recompute[v] - batch.logp_old[v]).abs().max()) < 1e-4


def test_ego_action_is_channel_of_next_frame_window() -> None:
    """In a built window, the ego action executed at frame ``t`` is the ego controller
    channel of frame ``t+1`` (column ``k`` carries ``a_{k-1}``)."""
    pol, (slot,), _ = _make_policy(refresh_every=1000)
    seq = [_frame(i, p1_pct=0.3 * i, p2_pct=0.4 * i) for i in range(10)]
    for t, f in enumerate(seq):
        pol(t, {slot: f})
    st = pol.state[slot].stream
    st.append_boundary(pol.state[slot].flat_hist[-1])
    fin = st.finalize()
    (window,) = build_windows(fin, CFG.L_ctx, stream_id=0)  # one episode -> one window
    # main_stick_x channel; find the row for frame t+1 via frame_pos and compare to a_t[0].
    ego_col = window.feats["ego_main_stick_x"]
    for row in range(CFG.L_ctx):
        fp = int(window.frame_pos[row])
        if fp >= 1:  # frame fp's ego channel == action executed at frame fp-1
            assert ego_col[row] == pytest.approx(float(fin.ego_act[fp - 1][0]), abs=1e-6)


def test_finalize_truncates_tail_and_keeps_boundary_frame() -> None:
    pol, (slot,), _ = _make_policy(rollout_frames=5)  # finalize once 5 transitions land
    it = None
    for t in range(20):
        pol(t, {slot: _frame(t, p1_pct=t, p2_pct=t)})
        it = pol.take_iteration()
        if it is not None:
            break
    assert it is not None
    (stream,) = it.streams
    assert stream.n_transitions >= 5
    assert len(stream.flat) == stream.n_transitions + 1  # T+1 boundary frame included
    assert bool(stream.truncated[-1])  # tail bootstraps from the boundary value
    assert not bool(stream.terminated[-1])


def test_self_play_both_ports_are_separate_streams() -> None:
    pol, slots, _ = _make_policy(n_slots=2)
    for t in range(8):
        out = pol(t, {s: _frame(t, p1_pct=t, p2_pct=2 * t) for s in slots})
        assert set(out) == set(slots)
    ego_ports = {pol.state[s].ego_port for s in slots}
    assert ego_ports == {1, 2}
    for s in slots:
        assert pol.state[s].stream.n_recorded == 7


def test_sync_mode_determinism() -> None:
    """Two seeded runs of the canned loop produce byte-identical streams."""
    seq = [_frame(i, p1_pct=0.5 * i, p2_pct=0.6 * i) for i in range(10)]

    def run() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pol, (slot,), _ = _make_policy(seed=123)
        for t, f in enumerate(seq):
            pol(t, {slot: f})
        st = pol.state[slot].stream
        return np.array(st._act_idx), np.array(st._logp), np.array(st._rew)

    a_idx0, a_logp0, a_rew0 = run()
    b_idx0, b_logp0, b_rew0 = run()
    assert np.array_equal(a_idx0, b_idx0)
    assert np.array_equal(a_logp0, b_logp0)
    assert np.array_equal(a_rew0, b_rew0)


# --- real-Dolphin integration ------------------------------------------------
WARM_START = Path(
    "runs/260616-004736_012_multi_token_gpt-d256-L8-h4-Lc256-o1.5.9.13_ranked-anon-1_gpt-16k-b1024/final.pt"
)


@pytest.mark.integration
def test_collector_real_dolphin() -> None:
    """Two real warm-started Dolphins, both ports self-play driven, ~1 iteration collected
    end-to-end; assert a well-formed RolloutIteration with finite rewards/logps."""
    import queue
    import threading

    import melee
    from melee_collector import drive_rl
    from nets_melee import load_il_policy

    from hal.eval.harness import _build_session
    from hal.eval.harness import default_session_cfg
    from hal.paths import EMULATOR_PATH
    from hal.paths import ISO_PATH
    from hal.sim.session import Matchup
    from hal.sim.session import PlayerSetup

    if not Path(ISO_PATH).is_file() or not Path(EMULATOR_PATH).is_file():
        pytest.skip("ISO or Dolphin missing")
    if not WARM_START.is_file():
        pytest.skip(f"warm-start checkpoint missing at {WARM_START}")

    net, cfg = load_il_policy(WARM_START)
    net.eval()
    stats = _stats()
    n_boots = 2
    ports = (1, 2)
    slots = [Slot(i, p) for i in range(n_boots) for p in ports]
    handle = NetActingPolicy(net, n_slots=len(slots), device="cpu", temp=1.0, seed=0, max_pos=cfg.L_ctx + 128)

    session_cfg = default_session_cfg(instant_match_restart=True)
    matchups = [
        Matchup(
            stage=melee.Stage.BATTLEFIELD,
            players=(
                PlayerSetup(port=1, character=melee.Character.FOX, cpu_level=0),
                PlayerSetup(port=2, character=melee.Character.FOX, cpu_level=0),
            ),
        )
        for _ in range(n_boots)
    ]
    slot_matchup = {
        Slot(i, p): {
            "stage": int(matchups[i].stage.value),
            "character": {pl.port: int(pl.character.value) for pl in matchups[i].players},
        }
        for i in range(n_boots)
        for p in ports
    }
    pol = RLBatchPolicy(
        handles={"ema": handle},
        assign=lambda s: "ema",
        slots=slots,
        ego_port_of=lambda s: s.port,
        matchup_of=lambda s: slot_matchup[s],
        reward_cfg=RewardConfig(),
        stats=stats,
        L_ctx=cfg.L_ctx,
        refresh_every=64,
        rollout_frames=600,
    )

    def build_sessions():
        sessions = [_build_session(session_cfg, slippi_port=51461 + i, replay_dir=None) for i in range(n_boots)]
        return sessions, matchups, [ports] * n_boots

    q: queue.Queue = queue.Queue(maxsize=1)
    stop = threading.Event()
    drive_rl(
        build_sessions,
        pol,
        n_iterations=1,
        queue_out=q,
        stop=stop,
        on_iteration=lambda: None,
        sync_gate=None,
        progress_every=0,
    )
    it = q.get(timeout=5.0)
    assert it.n_transitions >= 600
    assert len(it.streams) == len(slots)
    for st in it.streams:
        assert len(st.flat) == st.n_transitions + 1
        if st.n_transitions:
            assert np.all(np.isfinite(st.rew))
            assert np.all(np.isfinite(st.logp))
