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

from dataclasses import replace
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
from hal.training.features import A_DIM
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


def _swap_weights_run(*, force_rebuild: bool) -> float:
    """Collect 6 frames, swap the acting net's weights in place (the shape of the
    EMA→act_net copy), optionally ``request_rebuild``, collect 6 more; return the max
    |recompute - recorded| logp over the steps that ride the (re-seeded) cache."""
    pol, (slot,), net = _make_policy(refresh_every=1000)  # periodic rebuild off — isolate the request
    seq = [_frame(i, p1_pct=0.5 * i, p2_pct=0.7 * i) for i in range(12)]
    for t in range(6):
        pol(t, {slot: seq[t]})

    torch.manual_seed(99)
    net.load_state_dict(PolicyValueNet(CFG).state_dict())  # in-place: the handle's net object
    if force_rebuild:
        pol.stepper.request_rebuild()
    for t in range(6, 12):
        pol(t, {slot: seq[t]})

    st = pol.state[slot].stream
    st.append_boundary(pol.state[slot].flat_hist[-1])
    windows = build_windows(st.finalize(), CFG.L_ctx, stream_id=0)
    batch = collate_windows(windows, _stats())
    with torch.no_grad():
        dist = FactoredCategorical(net.policy_logits(net.forward_full(batch.context)))
        logp_recompute = dist.log_prob(batch.act_idx)
    fp = torch.from_numpy(np.stack([w.frame_pos for w in windows]))
    # Frame 6's action is sampled BEFORE the end-of-step rebuild (hybrid by design);
    # steps 7+ decode against whatever K/V state the arm under test left behind.
    post = batch.valid & (fp >= 7)
    assert bool(post.any())
    return float((logp_recompute[post] - batch.logp_old[post]).abs().max())


def test_request_rebuild_reseeds_caches_after_weight_swap() -> None:
    """After a weight swap, ``request_rebuild`` must re-seed the caches so subsequent
    behavior logp matches the new net's full recompute; without it, K/V computed by the
    old weights keep corrupting the decode (the control arm proves the test bites)."""
    assert _swap_weights_run(force_rebuild=True) < 1e-4
    assert _swap_weights_run(force_rebuild=False) > 1e-3


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


def test_prefix_carries_context_across_iteration_boundary() -> None:
    """A stream finalized mid-episode hands the rolling window's frames to its successor
    as burn-in prefix, so the successor's strided-window recompute matches the recorded
    behavior logp exactly pre-eviction — including its head positions, which without the
    prefix recompute from a cold window (the control arm pins that failure mode)."""
    pol, (slot,), net = _make_policy(refresh_every=1000, rollout_frames=6)
    it = None
    for t in range(14):
        pol(t, {slot: _frame(t, p1_pct=0.5 * t, p2_pct=0.7 * t)})
        it = it or pol.take_iteration()
    assert it is not None  # first iteration emitted; the live stream is its successor

    st = pol.state[slot]
    assert st.stream.prefix_flat  # successor carries the pre-boundary context
    st.stream.append_boundary(st.flat_hist[-1])
    fin = st.stream.finalize()
    assert len(fin.prefix_flat) == fin.prefix_ego.shape[0]

    def max_dev(stream) -> float:  # noqa: ANN001
        windows = build_windows(stream, CFG.L_ctx, stride=2, stream_id=0)
        batch = collate_windows(windows, _stats())
        with torch.no_grad():
            dist = FactoredCategorical(net.policy_logits(net.forward_full(batch.context)))
            logp = dist.log_prob(batch.act_idx)
        v = batch.valid
        assert bool(v.any())
        return float((logp[v] - batch.logp_old[v]).abs().max())

    assert max_dev(fin) < 1e-4
    bare = replace(fin, prefix_flat=(), prefix_ego=np.zeros((0, A_DIM), np.float32))
    assert max_dev(bare) > 1e-3  # control: without the prefix the head recompute is cold


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


def test_wave_reboot_flushes_streams_truncated() -> None:
    """A wave reboot must not discard recorded transitions: ``reset_slots`` flushes the
    in-flight stream (truncated tail + boundary) as an orphan that joins the next
    finalized iteration."""
    pol, (slot,), _ = _make_policy(rollout_frames=12)
    for t in range(6):
        pol(t, {slot: _frame(t, p1_pct=t, p2_pct=t)})
    assert pol.state[slot].stream.n_recorded == 5  # in flight at the "reboot"

    pol.reset_slots([slot])
    assert pol.state[slot].stream.n_recorded == 0  # fresh stream post-reboot
    assert pol.state[slot].last_id is None  # sentinel: rebooted slot's first frame can't read as a reset

    it = None
    for t in range(30):  # ids restart at 0 — must NOT trigger a spurious episode reset
        pol(100 + t, {slot: _frame(t, p1_pct=t, p2_pct=t)})
        it = pol.take_iteration()
        if it is not None:
            break
    assert it is not None
    assert it.n_transitions >= 12
    orphan, live = it.streams  # orphans come first in the finalized iteration
    assert orphan.n_transitions == 5  # the pre-reboot transitions survived
    assert bool(orphan.truncated[-1]) and not bool(orphan.terminated[-1])
    assert len(orphan.flat) == orphan.n_transitions + 1
    assert live.n_transitions == it.n_transitions - 5


def test_finalize_adjacent_reset_warns_bonus_lost_and_recovers() -> None:
    """An episode ending on the first frame after an iteration boundary cannot credit its
    terminal bonus (the steps were already emitted truncated) — the collector must log the
    loss loudly and keep collecting consistently, not crash or mis-terminate."""
    from loguru import logger

    pol, (slot,), _ = _make_policy(rollout_frames=5)
    it = None
    t = 0
    while it is None:
        pol(t, {slot: _frame(t, p1_pct=t)})
        it = pol.take_iteration()
        t += 1
    assert pol.state[slot].stream.n_recorded == 0  # fresh stream right after finalize
    assert pol.state[slot].episode_steps > 0  # but the episode is still running
    assert pol.state[slot].stream.prefix_flat  # successor carries the pre-boundary context

    msgs: list[str] = []
    sink = logger.add(lambda m: msgs.append(str(m)), level="WARNING")
    try:
        pol(t, {slot: _frame(0)})  # id drops -> episode boundary on the finalize-adjacent frame
    finally:
        logger.remove(sink)
    assert any("terminal bonus LOST" in m for m in msgs), msgs

    st = pol.state[slot]
    assert st.episode_steps == 0  # new episode
    assert st.stream.n_recorded == 0  # the dropped pending never became a step
    assert st.stream.prefix_flat == ()  # the dead match's context prefix was dropped with it
    pol(t + 1, {slot: _frame(1)})
    pol(t + 2, {slot: _frame(2)})
    assert st.stream.n_recorded == 2  # collection resumed cleanly
    assert not st.stream._terminated[-1]


# --- matchup rotation (A) -----------------------------------------------------
def test_wave_matchups_tile_the_prior_in_nonoverlapping_slices() -> None:
    """Each wave's matchups are the next contiguous ``n_boots`` slice of the prior; because
    ``matchups_for`` is prefix-stable, successive waves tile it with no overlap — so a run
    that reboots waves rotates through the full training distribution, not one fixed slice."""
    from melee_train import wave_matchups

    from hal.eval.matchups import matchups_for

    n = 4
    full = matchups_for(3 * n)  # first 12 prior matchups, in the deterministic prefix order
    for w in range(3):
        wave = wave_matchups(w, n)
        assert len(wave) == n
        got = [(m.players[0].character, m.players[1].character) for m in wave]
        assert got == full[w * n : (w + 1) * n]


def test_reset_slots_rotation_rebinds_fresh_stream_matchup() -> None:
    """The wave-reboot rotation path: ``reset_slots(matchups=...)`` orphans the in-flight
    stream under its OLD matchup and starts the fresh stream on the NEW slice's matchup."""
    pol, (slot,), _ = _make_policy(rollout_frames=100)
    old = pol.state[slot].matchup
    for t in range(4):
        pol(t, {slot: _frame(t, p1_pct=t, p2_pct=t)})
    assert pol.state[slot].stream.n_recorded == 3  # in flight

    new_meta = {slot: {"stage": 2, "character": {1: 22, 2: 1}}}
    pol.reset_slots([slot], matchups=new_meta)
    assert pol._orphans[-1].matchup == old  # orphaned stream keeps the matchup it was collected under
    assert pol.state[slot].matchup == new_meta[slot]  # slot rebound to the new slice
    assert pol.state[slot].stream.matchup == new_meta[slot]  # fresh stream carries it forward


def test_eval_policy_steps_identically_to_collector() -> None:
    """Parity: ``EvalBatchPolicy`` and ``RLBatchPolicy`` hand-duplicate the same three-pass
    stepping (reset detect, hist caps, ``ego_hist[:-1]`` rebuild, refresh clock). Driven with
    the same net/seed over one canned obs sequence (with a reset and several rebuilds), they
    must emit identical actions AND evolve identical KV-cache state — the standing equivalence
    check for the duplicated stepping cores."""
    from melee_eval import EvalBatchPolicy

    stats = _stats()
    torch.manual_seed(0)
    net = PolicyValueNet(CFG).eval()  # one net; each policy gets its own seeded handle over it
    rl_log: list = []
    ev_log: list = []
    rl_handle = _RecordingHandle(NetActingPolicy(net, n_slots=1, device="cpu", seed=0), rl_log)
    ev_handle = _RecordingHandle(NetActingPolicy(net, n_slots=1, device="cpu", seed=0), ev_log)
    slot = Slot(0, 1)
    refresh = 4
    rl_pol = RLBatchPolicy(
        handles={"ema": rl_handle},
        assign=lambda s: "ema",
        slots=[slot],
        ego_port_of=lambda s: s.port,
        matchup_of=lambda s: {"stage": 3, "character": {1: 1, 2: 1}},
        reward_cfg=RewardConfig(),
        stats=stats,
        L_ctx=CFG.L_ctx,
        refresh_every=refresh,
        rollout_frames=10_000,
    )
    ev_pol = EvalBatchPolicy(
        handles={"ema": ev_handle},
        handle_of=lambda s: "ema",
        stats=stats,
        L_ctx=CFG.L_ctx,
        refresh_every=refresh,
    )
    seq = [_frame(i, p1_pct=i, p2_pct=2 * i, p2_stock=(4.0 if i < 5 else 1.0)) for i in range(6)]
    seq.append(_frame(0))  # id drop -> instant-restart reset (both must cold-start the cache)
    # Post-reset frames span several rebuilds AND exceed L_ctx=16, so the ring-buffer
    # eviction path is part of the comparison, not just fresh growth.
    seq += [_frame(i, p1_pct=float(i)) for i in range(1, 25)]

    for t, f in enumerate(seq):
        rl_pol(t, {slot: f})
        ev_pol(t, {slot: f})

    assert len(rl_log) == len(ev_log) == len(seq)
    for (_, a_rl, lp_rl, v_rl), (_, a_ev, lp_ev, v_ev) in zip(rl_log, ev_log, strict=True):
        assert np.array_equal(a_rl, a_ev)  # identical sampled action indices
        assert np.array_equal(v_rl, v_ev)  # identical dequantized action vectors
        np.testing.assert_allclose(lp_rl, lp_ev, rtol=0, atol=0)  # identical behavior log-probs
    rc, ec = rl_handle.inner.caches, ev_handle.inner.caches
    assert torch.equal(rc.length, ec.length)  # cache lengths in lockstep through the reset + rebuilds
    assert torch.equal(rc.write_idx, ec.write_idx)
    assert torch.equal(rc.next_pos, ec.next_pos)
    assert torch.equal(rc.K, ec.K)  # identical K/V ring contents
    assert torch.equal(rc.V, ec.V)


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

    from hal.eval.harness import build_session
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

    def build_boot(i: int, attempt: int):
        return build_session(session_cfg, slippi_port=51461 + (attempt % 8) * n_boots + i, replay_dir=None)

    q: queue.Queue = queue.Queue(maxsize=1)
    stop = threading.Event()
    drive_rl(
        build_boot,
        lambda _w: matchups,
        [ports] * n_boots,
        pol,
        n_iterations=1,
        queue_out=q,
        stop=stop,
        on_iteration=lambda: None,
        sync_gate=None,
        progress_every=0,
    )
    it, cstats = q.get(timeout=5.0)
    assert it.n_transitions >= 600
    assert cstats.frames > 0 and cstats.lockstep_sps > 0
    assert len(it.streams) == len(slots)
    for st in it.streams:
        assert len(st.flat) == st.n_transitions + 1
        if st.n_transitions:
            assert np.all(np.isfinite(st.rew))
            assert np.all(np.isfinite(st.logp))
