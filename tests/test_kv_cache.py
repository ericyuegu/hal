"""KV-cache incremental-decode equivalence tests (pure CPU, fp32, tiny trunk).

The correctness spine of the rollout speedup: incremental decode must match a full
re-forward exactly before eviction, a rebuild must restore that exactness, a reset
slot must behave cold, and mixed-length slots must batch without cross-talk. Tiny
cfg (d_model=32, L=2, heads=2, L_ctx=16) keeps 2.5x-window sweeps instant.
"""

import torch
from kv_cache import SlotCaches
from nets_melee import ArchConfig
from nets_melee import PolicyValueNet

from hal.training.features import ACTION_CHANNELS
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import Context

ATOL = 1e-5
CFG = ArchConfig(d_model=32, n_layers=2, n_heads=2, L_ctx=16, char_vocab=8, char_dim=4, stage_vocab=8, stage_dim=2)
_PREFIXES = ("ego", "ego_nana", "opp_nana", "opp")


def _make_net(seed: int = 0) -> PolicyValueNet:
    torch.manual_seed(seed)
    net = PolicyValueNet(CFG)
    net.eval()
    return net


def _seq_features(B: int, T: int, *, seed: int = 0) -> dict[str, torch.Tensor]:
    """A synthetic per-frame feature sequence [B, T] with every key context_tokens reads
    (floats, categoricals, ego action channels, char/stage). No mask sidecars -> the net
    supplies zeros, matching the no-mask training path."""
    g = torch.Generator().manual_seed(seed)
    feats: dict[str, torch.Tensor] = {}
    for p in _PREFIXES:
        for f in FLOAT_FEATURES:
            feats[f"{p}_{f}"] = torch.randn(B, T, generator=g)
        for name, (vocab, _) in CAT_FEATURES.items():
            feats[f"{p}_{name}"] = torch.randint(0, vocab, (B, T), generator=g)
    for ch in ACTION_CHANNELS:
        feats[f"ego_{ch}"] = torch.randn(B, T, generator=g)
    feats["ego_character"] = torch.randint(0, CFG.char_vocab, (B, T), generator=g)
    feats["opp_character"] = torch.randint(0, CFG.char_vocab, (B, T), generator=g)
    feats["stage"] = torch.randint(0, CFG.stage_vocab, (B, T), generator=g)
    return feats


def _window(feats: dict[str, torch.Tensor], lo: int, hi: int) -> Context:
    return Context(
        features={k: v[:, lo:hi] for k, v in feats.items()},
        ctx_pad=torch.zeros(next(iter(feats.values())).shape[0], dtype=torch.long),
    )


def _frame(feats: dict[str, torch.Tensor], t: int) -> Context:
    return _window(feats, t, t + 1)


def test_exact_growth_matches_full_reforward() -> None:
    # From an empty cache, every incremental step's last-position hidden AND logits equal
    # a full forward over frames [0..t] (pre-eviction: RoPE shift-invariance -> bit-exact).
    net = _make_net()
    feats = _seq_features(1, CFG.L_ctx, seed=1)
    cache = SlotCaches(net, n_slots=1)
    for t in range(CFG.L_ctx):
        h_inc = cache.step_incremental(net, None, _frame(feats, t))
        h_full = net.forward_full(_window(feats, 0, t + 1))[:, -1]
        assert torch.allclose(h_inc, h_full, atol=ATOL), f"hidden mismatch at t={t}: {(h_inc - h_full).abs().max()}"
        lg_inc = net.policy_logits(h_inc)
        lg_full = net.policy_logits(h_full)
        assert torch.allclose(lg_inc, lg_full, atol=ATOL), f"logit mismatch at t={t}"


def test_rebuild_restores_exact_windowed_forward() -> None:
    # Run well past eviction (drift accumulates), rebuild on the trailing window, then one
    # more incremental step == a full forward over that window + the new frame. The rebuild
    # window and the comparison window share their LEFT boundary, so no eviction breaks the
    # equivalence (that shared boundary is exactly what a periodic rebuild buys).
    net = _make_net()
    L = CFG.L_ctx
    N = int(2.5 * L)  # frames 0..N fed; N+1 total
    feats = _seq_features(1, N + 1, seed=2)
    cache = SlotCaches(net, n_slots=1)
    slot = torch.tensor([0])
    for t in range(N):
        cache.step_incremental(net, None, _frame(feats, t))

    win_lo = N - L + 1  # L-1 real frames [win_lo .. N-1]
    cache.rebuild(net, slot, _window(feats, win_lo, N))
    assert int(cache.length[0]) == L - 1 and int(cache.next_pos[0]) == L - 1

    h_inc = cache.step_incremental(net, None, _frame(feats, N))
    h_full = net.forward_full(_window(feats, win_lo, N + 1))[:, -1]
    assert torch.allclose(h_inc, h_full, atol=ATOL), f"post-rebuild mismatch: {(h_inc - h_full).abs().max()}"


def test_drift_diagnostic_between_rebuilds() -> None:
    # Quantify how far post-eviction incremental drifts from the every-frame
    # windowed forward (012's semantics) at eviction depth. Informs refresh_every.
    net = _make_net()
    L = CFG.L_ctx
    N = 2 * L
    feats = _seq_features(1, N, seed=3)
    cache = SlotCaches(net, n_slots=1)
    max_dlogit = 0.0
    with torch.no_grad():
        for t in range(N):
            h_inc = cache.step_incremental(net, None, _frame(feats, t))
            lo = max(0, t - L + 1)
            h_full = net.forward_full(_window(feats, lo, t + 1))[:, -1]
            if t >= L:  # only meaningful once eviction has happened
                d = float((net.policy_logits(h_inc) - net.policy_logits(h_full)).abs().max())
                max_dlogit = max(max_dlogit, d)
    print(f"\n[drift] max |Δlogit| at eviction depth over {N - L} frames: {max_dlogit:.4g}")
    # Post-eviction incremental MUST differ from the every-frame windowed forward
    # (their deep K/V see different histories) — zero drift would mean the
    # diagnostic is measuring nothing (e.g. eviction never engaged).
    assert max_dlogit > 0.0


def test_reset_slot_behaves_cold() -> None:
    # A reset slot decodes identically to a fresh cache from frame 0.
    net = _make_net()
    feats = _seq_features(1, 5, seed=4)
    slot = torch.tensor([0])

    warm = SlotCaches(net, n_slots=1)
    for t in range(5):
        warm.step_incremental(net, slot, _frame(feats, t))
    warm.reset_slot(0)

    fresh = SlotCaches(net, n_slots=1)
    other = _seq_features(1, 3, seed=99)
    for t in range(3):
        h_reset = warm.step_incremental(net, slot, _frame(other, t))
        h_fresh = fresh.step_incremental(net, slot, _frame(other, t))
        assert torch.allclose(h_reset, h_fresh, atol=ATOL), f"reset slot not cold at t={t}"


def test_mixed_length_batch_matches_per_slot() -> None:
    # Three slots primed to DIFFERENT lengths, stepped together, must equal each stepped
    # alone — i.e. the key-padding mask isolates slots of differing cache lengths.
    net = _make_net()
    lengths = (3, 7, 11)
    seqs = [_seq_features(1, n + 1, seed=10 + i) for i, n in enumerate(lengths)]

    together = SlotCaches(net, n_slots=3)
    for i, n in enumerate(lengths):  # prime each slot alone to its length
        for t in range(n):
            together.step_incremental(net, torch.tensor([i]), _frame(seqs[i], t))

    # One joint step over all three slots at their (differing) next frames.
    joint_feats = {
        k: torch.cat([seqs[i][k][:, lengths[i] : lengths[i] + 1] for i in range(3)], dim=0) for k in seqs[0]
    }
    joint_ctx = Context(features=joint_feats, ctx_pad=torch.zeros(3, dtype=torch.long))
    h_joint = together.step_incremental(net, torch.tensor([0, 1, 2]), joint_ctx)

    for i, n in enumerate(lengths):
        solo = SlotCaches(net, n_slots=1)
        for t in range(n):
            solo.step_incremental(net, torch.tensor([0]), _frame(seqs[i], t))
        h_solo = solo.step_incremental(net, torch.tensor([0]), _frame(seqs[i], n))
        assert torch.allclose(h_joint[i : i + 1], h_solo, atol=ATOL), f"slot {i} mismatch in batch"


def test_all_slots_path_mixed_lengths_after_reset() -> None:
    # The slot_ids=None fast path (cache views, no gather) with 3 slots at MIXED
    # lengths — one mid-rollout reset_slot — must match per-slot stepping.
    net = _make_net()
    lengths = [4, 6, 9]
    seqs = [_seq_features(1, max(lengths) + 1, seed=20 + i) for i in range(3)]

    cache = SlotCaches(net, n_slots=3)
    for i, n in enumerate(lengths):
        for t in range(n):
            cache.step_incremental(net, torch.tensor([i]), _frame(seqs[i], t))
    cache.reset_slot(1)  # slot 1 cold-starts: its next joint step is its frame 0
    lengths[1] = 0

    joint_feats = {
        k: torch.cat([seqs[i][k][:, lengths[i] : lengths[i] + 1] for i in range(3)], dim=0) for k in seqs[0]
    }
    joint_ctx = Context(features=joint_feats, ctx_pad=torch.zeros(3, dtype=torch.long))
    h_joint = cache.step_incremental(net, None, joint_ctx)  # all-slots view path

    for i, n in enumerate(lengths):
        solo = SlotCaches(net, n_slots=1)
        for t in range(n):
            solo.step_incremental(net, torch.tensor([0]), _frame(seqs[i], t))
        h_solo = solo.step_incremental(net, torch.tensor([0]), _frame(seqs[i], n))
        assert torch.allclose(h_joint[i : i + 1], h_solo, atol=ATOL), f"slot {i} mismatch on all-slots path"
