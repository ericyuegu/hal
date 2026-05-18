"""Unit tests for hal.data.replay_stats — pure-function helpers + fixture probe."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from hal.data.replay_stats import PlayerStats
from hal.data.replay_stats import ReplayStats
from hal.data.replay_stats import _count_input_edges
from hal.data.replay_stats import _death_percents
from hal.data.replay_stats import _final_stocks
from hal.data.replay_stats import _percent_gained
from hal.data.replay_stats import compute_replay_stats
from hal.paths import DEV_ARCHIVE_PATH
from hal.wire import BUTTON_BITS
from hal.wire import dedupe_keep_idx


class _ArrowLike:
    """Minimal stand-in for a pyarrow Array exposing __len__ and to_pylist()."""

    def __init__(self, values: Sequence[Any]) -> None:
        self._values = list(values)

    def __len__(self) -> int:
        return len(self._values)

    def to_pylist(self) -> list[Any]:
        return list(self._values)


def test_percent_gained_sums_positive_deltas() -> None:
    # opponent percent 0 -> 5 -> 12, stocks unchanged: damage = 12.
    assert _percent_gained(_ArrowLike([0.0, 5.0, 12.0]), _ArrowLike([4, 4, 4])) == pytest.approx(12.0)


def test_percent_gained_skips_stock_reset_frame() -> None:
    # 50 -> 0 across a stock loss should NOT count as -50 nor as anything.
    # 0 -> 10 on the next frame is +10. Total = 10.
    assert _percent_gained(
        _ArrowLike([50.0, 0.0, 10.0]),
        _ArrowLike([4, 3, 3]),
    ) == pytest.approx(10.0)


def test_percent_gained_skips_negative_deltas_without_stock_change() -> None:
    # Healing items / Pichu reverse-damage edge cases — not damage dealt.
    assert _percent_gained(_ArrowLike([20.0, 18.0, 18.0]), _ArrowLike([4, 4, 4])) == 0.0


def test_percent_gained_none_arrays_return_zero() -> None:
    assert _percent_gained(None, _ArrowLike([4])) == 0.0
    assert _percent_gained(_ArrowLike([1.0]), None) == 0.0


def test_percent_gained_handles_none_entries() -> None:
    # A None in percent column means peppi didn't have a value; NaN delta excluded.
    assert _percent_gained(_ArrowLike([0.0, None, 10.0]), _ArrowLike([4, 4, 4])) == 0.0


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


def test_death_percents_basic() -> None:
    # Stocks 4 -> 4 -> 3: one death at idx 1->2; recorded percent is pct[1] = 80.
    out = _death_percents(_ArrowLike([0.0, 80.0, 0.0]), _ArrowLike([4, 4, 3]))
    assert out == (pytest.approx(80.0),)


def test_death_percents_multiple_stock_losses() -> None:
    # Three deaths: at idx 1->2 (pct=50), idx 3->4 (pct=120), idx 5->6 (pct=80).
    pct = [0.0, 50.0, 0.0, 120.0, 0.0, 80.0, 0.0]
    stk = [4, 4, 3, 3, 2, 2, 1]
    out = _death_percents(_ArrowLike(pct), _ArrowLike(stk))
    assert out == (pytest.approx(50.0), pytest.approx(120.0), pytest.approx(80.0))


def test_death_percents_ignores_none_sentinel_transitions() -> None:
    # Game-end trailer: stocks 4 -> 4 -> None. The 4->None decrement is NOT a death.
    out = _death_percents(_ArrowLike([0.0, 50.0, 50.0]), _ArrowLike([4, 4, None]))
    assert out == ()


def test_death_percents_no_deaths() -> None:
    assert _death_percents(_ArrowLike([0.0, 10.0, 20.0]), _ArrowLike([4, 4, 4])) == ()


def test_death_percents_none_inputs() -> None:
    assert _death_percents(None, _ArrowLike([4, 3])) == ()
    assert _death_percents(_ArrowLike([0.0, 50.0]), None) == ()
    assert _death_percents(_ArrowLike([]), _ArrowLike([])) == ()


def test_dedupe_keep_idx_collapses_rollback_pattern() -> None:
    # Rollback emits each repeated frame_id with a later "correction"; keep the LAST.
    # ids: 2065, 2066, 2065, 2066, 2067, 2066, 2067
    # last-occurrence: 2065@2, 2066@5, 2067@6; sorted ascending -> [2, 5, 6]
    keep = dedupe_keep_idx([2065, 2066, 2065, 2066, 2067, 2066, 2067])
    assert list(keep) == [2, 5, 6]


def test_dedupe_keep_idx_already_unique() -> None:
    keep = dedupe_keep_idx([-123, -122, -121, -120])
    assert list(keep) == [0, 1, 2, 3]


def test_player_stats_roundtrip() -> None:
    p = PlayerStats(
        port=1,
        damage_dealt=120.5,
        damage_taken=80.0,
        stocks_remaining=2,
        inputs=900,
        death_percents=(45.0, 130.0),
    )
    assert PlayerStats.from_dict(p.to_dict()) == p


def test_replay_stats_roundtrip() -> None:
    rs = ReplayStats(
        players=(
            PlayerStats(port=1, damage_dealt=1.0, damage_taken=2.0, stocks_remaining=3, inputs=4, death_percents=()),
            PlayerStats(
                port=2,
                damage_dealt=5.0,
                damage_taken=6.0,
                stocks_remaining=0,
                inputs=7,
                death_percents=(60.0, 95.5, 110.0, 75.0),
            ),
        )
    )
    assert ReplayStats.from_dict(rs.to_dict()) == rs


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
        # Deaths == starting stocks (4) - remaining; bounded by the 4-stock cap.
        assert len(p.death_percents) <= 4
        assert all(dp >= 0.0 for dp in p.death_percents)
