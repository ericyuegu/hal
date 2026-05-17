"""Unit tests for hal.data.replay_stats — pure-function helpers + fixture probe."""

from pathlib import Path

import pytest

from hal.data.replay_stats import PlayerStats
from hal.data.replay_stats import ReplayStats
from hal.data.replay_stats import _count_input_edges
from hal.data.replay_stats import _count_sds
from hal.data.replay_stats import _damage_dealt
from hal.data.replay_stats import _final_stocks
from hal.data.replay_stats import compute_replay_stats
from hal.paths import DEV_ARCHIVE_PATH
from hal.wire import BUTTON_BITS


class _ArrowLike:
    """Minimal stand-in for a pyarrow Array exposing __len__ and to_pylist()."""

    def __init__(self, values: list[object]) -> None:
        self._values = values

    def __len__(self) -> int:
        return len(self._values)

    def to_pylist(self) -> list[object]:
        return list(self._values)


def test_damage_dealt_sums_positive_deltas() -> None:
    # opponent percent 0 -> 5 -> 12, stocks unchanged: damage = 12.
    assert _damage_dealt(_ArrowLike([0.0, 5.0, 12.0]), _ArrowLike([4, 4, 4])) == pytest.approx(12.0)


def test_damage_dealt_skips_stock_reset_frame() -> None:
    # 50 -> 0 across a stock loss should NOT count as -50 nor as anything.
    # 0 -> 10 on the next frame is +10. Total = 10.
    assert _damage_dealt(
        _ArrowLike([50.0, 0.0, 10.0]),
        _ArrowLike([4, 3, 3]),
    ) == pytest.approx(10.0)


def test_damage_dealt_skips_negative_deltas_without_stock_change() -> None:
    # Healing items / Pichu reverse-damage edge cases — not damage dealt.
    assert _damage_dealt(_ArrowLike([20.0, 18.0, 18.0]), _ArrowLike([4, 4, 4])) == 0.0


def test_damage_dealt_none_arrays_return_zero() -> None:
    assert _damage_dealt(None, _ArrowLike([4])) == 0.0
    assert _damage_dealt(_ArrowLike([1.0]), None) == 0.0


def test_damage_dealt_handles_none_entries() -> None:
    # A None in percent column means peppi didn't have a value; NaN delta excluded.
    assert _damage_dealt(_ArrowLike([0.0, None, 10.0]), _ArrowLike([4, 4, 4])) == 0.0


def test_final_stocks_returns_last_non_none() -> None:
    assert _final_stocks(_ArrowLike([4, 3, 2, 1, None])) == 1
    assert _final_stocks(_ArrowLike([4, 3, 2])) == 2
    assert _final_stocks(None) == 0
    assert _final_stocks(_ArrowLike([])) == 0


def test_count_input_edges_counts_rising_only() -> None:
    a = BUTTON_BITS["a"]
    b = BUTTON_BITS["b"]
    # Frame seq: 0, A, A, A+B, B, 0 -> rising edges: A (frame 1), B (frame 3). Total = 2.
    edges = _count_input_edges(_ArrowLike([0, a, a, a | b, b, 0]))
    assert edges == 2


def test_count_input_edges_ignores_extraneous_bits() -> None:
    # Bits outside BUTTON_BITS (e.g. d_left = 0x0001) are masked away.
    edges = _count_input_edges(_ArrowLike([0x0001, 0x0001 | BUTTON_BITS["a"], 0x0001]))
    assert edges == 1


def test_count_input_edges_none_input() -> None:
    assert _count_input_edges(None) == 0
    assert _count_input_edges(_ArrowLike([])) == 0
    assert _count_input_edges(_ArrowLike([BUTTON_BITS["a"]])) == 0


def test_count_sds_credits_recent_hit() -> None:
    # Hit landed at idx 4 (percent 0->10), death at idx 5 (5 frames after
    # the hit, well inside the 45-frame credit window) → opponent credited.
    sds, early = _count_sds(
        _ArrowLike([0.0, 0.0, 0.0, 0.0, 10.0, 0.0]),
        _ArrowLike([4, 4, 4, 4, 4, 3]),
        _ArrowLike([6, 6, 6, 6, 1, 1]),
        self_port_idx=0,
    )
    assert (sds, early) == (0, 0)


def test_count_sds_stale_hit_is_sd() -> None:
    # Hit landed at idx 1 (0 -> 10), death at idx 250 — 249 frames later,
    # > 180-frame credit window, so this is an SD.
    percent = [0.0, 10.0] + [10.0] * 249
    stock = [4] * 250 + [3]
    lhb = [6] + [1] * 250
    sds, early = _count_sds(_ArrowLike(percent), _ArrowLike(stock), _ArrowLike(lhb), self_port_idx=0)
    assert sds == 1
    assert early == 1


def test_count_sds_self_grab_is_sd() -> None:
    # last_hit_by == self_port_idx ⇒ always an SD, even with a recent hit.
    sds, early = _count_sds(
        _ArrowLike([0.0, 0.0, 10.0, 0.0]),
        _ArrowLike([4, 4, 4, 3]),
        _ArrowLike([6, 6, 0, 0]),  # self at idx 0
        self_port_idx=0,
    )
    assert sds == 1
    assert early == 1


def test_count_sds_early_vs_late_window() -> None:
    # No hits ever; two deaths. Death @ 100 is "early" (gap from frame 0 =
    # 100). Death @ 1000 is "late" (gap from prev death = 900).
    n = 1001
    percent = [0.0] * n
    stock = [4] * 100 + [3] * 900 + [2]
    lhb = [6] * n
    sds, early = _count_sds(_ArrowLike(percent), _ArrowLike(stock), _ArrowLike(lhb), self_port_idx=0)
    assert sds == 2
    assert early == 1


def test_count_sds_no_deaths() -> None:
    sds, early = _count_sds(_ArrowLike([0.0] * 3), _ArrowLike([4, 4, 4]), _ArrowLike([6, 6, 6]), self_port_idx=0)
    assert (sds, early) == (0, 0)


def test_count_sds_none_inputs() -> None:
    assert _count_sds(None, _ArrowLike([4]), _ArrowLike([6]), 0) == (0, 0)
    assert _count_sds(_ArrowLike([0.0]), None, _ArrowLike([6]), 0) == (0, 0)
    assert _count_sds(_ArrowLike([0.0]), _ArrowLike([4]), None, 0) == (0, 0)


def test_player_stats_roundtrip() -> None:
    p = PlayerStats(
        port=1,
        damage_dealt=120.5,
        damage_taken=80.0,
        stocks_remaining=2,
        inputs=900,
        sds=1,
        early_sds=0,
    )
    assert PlayerStats.from_dict(p.to_dict()) == p


def test_replay_stats_roundtrip() -> None:
    rs = ReplayStats(
        players=(
            PlayerStats(port=1, damage_dealt=1.0, damage_taken=2.0, stocks_remaining=3, inputs=4, sds=0, early_sds=0),
            PlayerStats(port=2, damage_dealt=5.0, damage_taken=6.0, stocks_remaining=0, inputs=7, sds=2, early_sds=1),
        )
    )
    assert ReplayStats.from_dict(rs.to_dict()) == rs
    assert rs.for_port(2).damage_dealt == 5.0
    assert rs.for_port(2).sds == 2
    assert rs.for_port(3) is None


@pytest.mark.skipif(
    not Path(DEV_ARCHIVE_PATH).exists(),
    reason=f"dev archive missing at {DEV_ARCHIVE_PATH}; run `python -m hal.scripts.fetch --name dev.7z`",
)
def test_compute_replay_stats_on_fixture(tmp_path: Path) -> None:
    """Smoke test against a real .slp from the dev archive."""
    import peppi_py
    import py7zr

    with py7zr.SevenZipFile(DEV_ARCHIVE_PATH, "r") as z:
        members = [m for m in z.getnames() if m.endswith(".slp")]
        assert members
        first = members[0]
        z.extract(path=tmp_path, targets=[first])
    g = peppi_py.read_slippi(str(tmp_path / first), skip_frames=False)
    rs = compute_replay_stats(g)
    assert rs is not None
    assert len(rs.players) == 2
    assert {p.port for p in rs.players} == {1, 2}
    for p in rs.players:
        assert p.damage_dealt >= 0.0
        assert p.damage_taken >= 0.0
        assert 0 <= p.stocks_remaining <= 4
        assert p.inputs > 0  # any non-CSS-only replay has button presses
        assert p.sds >= 0
        assert 0 <= p.early_sds <= p.sds
