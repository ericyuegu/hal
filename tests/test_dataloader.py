"""WindowSampler determinism — the val set must be reproducible across runs,
and train windows must still vary across epochs given a fixed seed."""

import numpy as np

from hal.training.dataloader import WindowSampler

L_CTX, L_CHUNK = 6, 4
_L = L_CTX + L_CHUNK


def _fake_mds(n_samples: int = 6, length: int = 60) -> list[dict[str, np.ndarray]]:
    """In-memory stand-in for a StreamingDataset: each sample is one replay."""
    return [
        {
            "frame": np.arange(length, dtype=np.int32),
            "p1_position_x": np.arange(length, dtype=np.float32),
            "p2_position_x": np.arange(length, dtype=np.float32) + 1000.0,
        }
        for _ in range(n_samples)
    ]


def _fingerprint(sampler: WindowSampler) -> list[tuple[int, str]]:
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
    a = _fingerprint(WindowSampler(_fake_mds(), L_CTX, L_CHUNK, seed=0))
    b = _fingerprint(WindowSampler(_fake_mds(), L_CTX, L_CHUNK, seed=0))
    assert a == b


def test_different_seed_different_windows() -> None:
    a = _fingerprint(WindowSampler(_fake_mds(), L_CTX, L_CHUNK, seed=0))
    b = _fingerprint(WindowSampler(_fake_mds(), L_CTX, L_CHUNK, seed=1))
    assert a != b


def test_windows_vary_across_epochs() -> None:
    """A single sampler iterated twice (two epochs) draws different windows, so
    a fixed seed doesn't freeze train augmentation to one window per replay."""
    s = WindowSampler(_fake_mds(), L_CTX, L_CHUNK, seed=0)
    epoch0 = _fingerprint(s)
    epoch1 = _fingerprint(s)
    assert epoch0 != epoch1
