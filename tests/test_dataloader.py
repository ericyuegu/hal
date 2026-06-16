"""WindowDataset determinism — the val set must be reproducible across runs,
and train windows must still vary across epochs given a fixed seed."""

import numpy as np

from hal.data.schema import SCHEMA_VERSION
from hal.training.dataloader import WindowDataset
from hal.training.dataloader import _choose_chunk_starts

L_CTX, L_CHUNK = 6, 4
_L = L_CTX + L_CHUNK


def _fake_mds(n_samples: int = 6, length: int = 60) -> list[dict[str, np.ndarray]]:
    """In-memory stand-in for a StreamingDataset: each sample is one replay."""
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "frame": np.arange(length, dtype=np.int32),
            "p1_position_x": np.arange(length, dtype=np.float32),
            "p2_position_x": np.arange(length, dtype=np.float32) + 1000.0,
        }
        for _ in range(n_samples)
    ]


def _fingerprint(sampler: WindowDataset) -> list[tuple[int, str]]:
    """(window start, ego side) per yielded window — observable proxy for the
    sampler's two random draws (start offset + ego_prefix)."""
    out = []
    for w in sampler:
        start = int(w["frame"][0])
        ego_side = "p1" if w["ego_position_x"][0] < 500 else "p2"
        out.append((start, ego_side))
    return out


def test_same_seed_same_windows() -> None:
    """Two fresh samplers with the same seed yield identical windows — this is
    what makes cached val loss comparable across runs."""
    a = _fingerprint(WindowDataset(_fake_mds(), L_CTX, L_CHUNK, seed=0))
    b = _fingerprint(WindowDataset(_fake_mds(), L_CTX, L_CHUNK, seed=0))
    assert a == b


def test_different_seed_different_windows() -> None:
    a = _fingerprint(WindowDataset(_fake_mds(), L_CTX, L_CHUNK, seed=0))
    b = _fingerprint(WindowDataset(_fake_mds(), L_CTX, L_CHUNK, seed=1))
    assert a != b


def test_windows_vary_across_epochs() -> None:
    """A single sampler iterated twice (two epochs) draws different windows, so
    a fixed seed doesn't freeze train augmentation to one window per replay."""
    s = WindowDataset(_fake_mds(), L_CTX, L_CHUNK, seed=0)
    epoch0 = _fingerprint(s)
    epoch1 = _fingerprint(s)
    assert epoch0 != epoch1


def test_window_length_and_ctx_pad() -> None:
    """Every emitted window is exactly L_ctx + L_chunk frames and carries an int
    ctx_pad — the neutral [ctx | chunk] contract, no bridge frames."""
    for w in WindowDataset(_fake_mds(), L_CTX, L_CHUNK, seed=0):
        assert len(w["frame"]) == _L
        assert "ctx_pad" in w


def test_cold_start_floor_skips_too_short() -> None:
    """cs_min=1 needs >=1 real context frame and the L_chunk chunk in-episode, so
    a replay of exactly L_chunk frames (cs_max=0 < 1) yields nothing."""
    assert list(WindowDataset(_fake_mds(n_samples=2, length=L_CHUNK), L_CTX, L_CHUNK, seed=0)) == []
    # one extra frame is enough for a single anchor (cs=1, fully left-padded ctx).
    assert list(WindowDataset(_fake_mds(n_samples=2, length=L_CHUNK + 1), L_CTX, L_CHUNK, seed=0))


def test_choose_chunk_starts_nonoverlapping_and_bounded() -> None:
    """Up to K chunk-starts, stratified with pairwise gap >= _L so the
    [cs - L_ctx, cs + L_chunk) windows never overlap; all in [1, T - L_chunk];
    count clamps to what the episode can fit."""
    rng = np.random.default_rng(0)
    for T in [11, 30, 44, 60, 100, 300]:
        for K in [1, 2, 4, 16]:
            cs = _choose_chunk_starts(T, L_CTX, L_CHUNK, K, rng)
            fit = max(1, (T - L_CHUNK) // _L)
            assert len(cs) == min(K, fit)
            assert (cs >= 1).all() and (cs <= T - L_CHUNK).all()
            assert (np.diff(np.sort(cs)) >= _L).all()


def test_yields_k_nonoverlapping_windows_per_replay() -> None:
    """K=4 over a long replay emits 4 windows whose real (non-pad) frames are
    pairwise disjoint — distinct training examples, not near-duplicate slices."""
    K = 4
    wins = list(WindowDataset(_fake_mds(n_samples=1, length=60), L_CTX, L_CHUNK, seed=0, windows_per_replay=K))
    assert len(wins) == K
    seen: set[int] = set()
    for w in wins:
        real = set(w["frame"][int(w["ctx_pad"]) :].tolist())
        assert real and real.isdisjoint(seen)
        seen |= real


def test_windows_per_replay_clamps_to_short_replay() -> None:
    """A replay too short for K disjoint windows yields only what fits (here 2),
    never overlapping ones to hit the count."""
    wins = list(WindowDataset(_fake_mds(n_samples=1, length=30), L_CTX, L_CHUNK, seed=0, windows_per_replay=4))
    assert len(wins) == 2


def test_windows_per_replay_default_is_one() -> None:
    """Default K=1 keeps the historical one-window-per-replay behavior."""
    wins = list(WindowDataset(_fake_mds(n_samples=3, length=60), L_CTX, L_CHUNK, seed=0))
    assert len(wins) == 3
