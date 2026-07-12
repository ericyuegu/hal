"""Pure-CPU tests for the self-play RL judge (``melee_eval``).

No Dolphin, no GPU: these pin the parts a biased eval would silently break — winner
attribution under port alternation (ema on p1 in even boots, p2 in odd), the
finite-readings discard, the bootstrap CIs, and the G3 verdict logic — plus the JSON
round-trip. The H2H acting path itself shares the Dolphin-integration-tested collector
machinery, so it is exercised by the live smoke, not here.
"""

import json

import numpy as np
from melee_eval import H2HMatch
from melee_eval import attribute_h2h
from melee_eval import ema_port_for_boot
from melee_eval import g3_vs_cpu_pass
from melee_eval import readout_from_traj
from melee_eval import summarize_h2h
from melee_eval import vs_cpu_net_stock_rate
from melee_eval import write_report

from hal.eval.scoring import MatchSummary
from hal.sim.trajectory import Trajectory


# --- port alternation -------------------------------------------------------
def test_ema_port_alternates_by_boot_parity() -> None:
    assert ema_port_for_boot(0) == 1
    assert ema_port_for_boot(1) == 2
    assert ema_port_for_boot(2) == 1
    assert ema_port_for_boot(3) == 2


# --- winner attribution -----------------------------------------------------
def _summary(*, p1_stocks: int, p2_stocks: int, p1_dmg: float, p2_dmg: float, frames: int = 3600) -> MatchSummary:
    return MatchSummary(
        frames=frames,
        p1_stocks_left=p1_stocks,
        p2_stocks_left=p2_stocks,
        p1_damage_taken=p1_dmg,
        p2_damage_taken=p2_dmg,
    )


def test_attribute_h2h_ema_on_p1() -> None:
    # ema (p1) ends with 3 stocks, il (p2) with 1; ema dealt what p2 received.
    m = attribute_h2h(_summary(p1_stocks=3, p2_stocks=1, p1_dmg=40.0, p2_dmg=120.0), ema_port=1)
    assert m.ema_stocks_left == 3
    assert m.il_stocks_left == 1
    assert m.ema_damage_dealt == 120.0  # p2 received
    assert m.il_damage_dealt == 40.0  # p1 received


def test_attribute_h2h_ema_on_p2_swaps_sides() -> None:
    # SAME raw readings, but ema is now p2 — attribution must flip, not report p1 as ema.
    m = attribute_h2h(_summary(p1_stocks=3, p2_stocks=1, p1_dmg=40.0, p2_dmg=120.0), ema_port=2)
    assert m.ema_stocks_left == 1
    assert m.il_stocks_left == 3
    assert m.ema_damage_dealt == 40.0  # p1 received (dealt by ema=p2)
    assert m.il_damage_dealt == 120.0


# --- trajectory readout + discard -------------------------------------------
def _traj(p1_stock: list[float], p2_stock: list[float], p1_pct: list[float], p2_pct: list[float]) -> Trajectory:
    n = len(p1_stock)
    return Trajectory(
        frame_id=np.arange(n, dtype=np.int64),
        post={
            1: {"stock": np.array(p1_stock, np.float64), "percent": np.array(p1_pct, np.float64)},
            2: {"stock": np.array(p2_stock, np.float64), "percent": np.array(p2_pct, np.float64)},
        },
        random_seed=np.zeros(n, np.int64),
    )


def test_readout_uses_last_finite_stock() -> None:
    # trailing NaN (IN_GAME -> menu) must not zero the deciding stock reading.
    traj = _traj([4, 4, 3, np.nan], [4, 3, 1, np.nan], [0, 20, 40, np.nan], [0, 50, 130, np.nan])
    m = readout_from_traj(traj, ema_port=1)
    assert m is not None
    assert m.ema_stocks_left == 3
    assert m.il_stocks_left == 1


def test_readout_discards_all_nan_match() -> None:
    traj = _traj([np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan])
    assert readout_from_traj(traj, ema_port=1) is None


def test_readout_discards_single_frame() -> None:
    assert readout_from_traj(_traj([4], [4], [0], [0]), ema_port=1) is None


# --- H2H summary + bootstrap CI ---------------------------------------------
def _match(ema_stocks: int, il_stocks: int, ema_dmg: float = 100.0, il_dmg: float = 100.0) -> H2HMatch:
    return H2HMatch(
        ema_stocks_left=ema_stocks,
        il_stocks_left=il_stocks,
        ema_damage_dealt=ema_dmg,
        il_damage_dealt=il_dmg,
        frames=3600,
    )


def test_summary_win_rate_excludes_ties_from_denominator() -> None:
    matches = [_match(3, 1), _match(2, 2), _match(1, 3), _match(4, 0)]  # 2 wins, 1 tie, 1 loss
    s = summarize_h2h(matches, n_discarded=0, seed=0)
    assert s["n_matches"] == 4
    assert s["n_ties"] == 1
    assert s["ema_win_rate"] == 2 / 3  # 2 wins out of 3 decided (tie dropped from denominator)
    assert s["mean_stock_diff"] == (2 + 0 - 2 + 4) / 4


def test_summary_degenerate_all_wins_ci_above_half() -> None:
    matches = [_match(4, 0) for _ in range(20)]
    s = summarize_h2h(matches, n_discarded=0, seed=0)
    assert s["ema_win_rate"] == 1.0
    lo, _ = s["win_rate_ci95"]
    assert lo > 0.5
    assert s["g3_h2h"] == "PASS"  # CI excludes 0.5 and stock diff > 0


def test_summary_5050_ci_straddles_half() -> None:
    matches = [_match(3, 1) for _ in range(50)] + [_match(1, 3) for _ in range(50)]
    s = summarize_h2h(matches, n_discarded=0, seed=0)
    assert s["ema_win_rate"] == 0.5
    lo, hi = s["win_rate_ci95"]
    assert lo < 0.5 < hi
    assert abs(s["mean_stock_diff"]) < 1e-9
    assert s["g3_h2h"] == "FAIL"  # CI straddles 0.5


def test_summary_g3_fail_when_stock_diff_negative_even_if_ci_excludes_half() -> None:
    # ema loses most matches: win-rate CI excludes 0.5 (below it) but stock diff < 0 -> FAIL.
    matches = [_match(0, 4) for _ in range(20)]
    s = summarize_h2h(matches, n_discarded=0, seed=0)
    hi = s["win_rate_ci95"][1]
    assert hi < 0.5  # CI excludes 0.5
    assert s["mean_stock_diff"] < 0
    assert s["g3_h2h"] == "FAIL"


def test_summary_reports_damage_per_min() -> None:
    matches = [_match(3, 1, ema_dmg=180.0, il_dmg=60.0)]  # 1 min match
    s = summarize_h2h(matches, n_discarded=0, seed=0)
    assert s["ema_damage_per_min"] == 180.0
    assert s["il_damage_per_min"] == 60.0


# --- vs-CPU no-regression verdict -------------------------------------------
def _cpu_summary(stocks_taken: int, stocks_lost: int, frames: int = 3600) -> MatchSummary:
    # ego on p1: taken = 4 - p2_left, lost = 4 - p1_left.
    return MatchSummary(
        frames=frames,
        p1_stocks_left=4 - stocks_lost,
        p2_stocks_left=4 - stocks_taken,
        p1_damage_taken=float(stocks_lost) * 100,
        p2_damage_taken=float(stocks_taken) * 100,
    )


def test_vs_cpu_net_stock_rate() -> None:
    summaries = [_cpu_summary(stocks_taken=3, stocks_lost=1)]  # net +2 in 1 minute
    assert vs_cpu_net_stock_rate(summaries) == 2.0


def test_g3_vs_cpu_pass_when_not_worse() -> None:
    # ckpt clearly better than a weak baseline -> PASS (not a regression).
    ckpt = [_cpu_summary(stocks_taken=3, stocks_lost=1) for _ in range(30)]
    assert g3_vs_cpu_pass(ckpt, baseline_net_rate=0.0, seed=0)


def test_g3_vs_cpu_fail_on_regression() -> None:
    # ckpt clearly worse than a strong baseline -> FAIL.
    ckpt = [_cpu_summary(stocks_taken=1, stocks_lost=3) for _ in range(30)]
    assert not g3_vs_cpu_pass(ckpt, baseline_net_rate=2.0, seed=0)


# --- JSON round-trip --------------------------------------------------------
def test_json_round_trip(tmp_path) -> None:
    matches = [_match(3, 1), _match(2, 2), _match(1, 3)]
    report = {
        "mode": "h2h",
        "policy": "ema",
        "git_sha": "deadbeef",
        "temp": 1.0,
        **summarize_h2h(matches, n_discarded=2, seed=0),
    }
    out = tmp_path / "eval.json"
    write_report(out, report)
    loaded = json.loads(out.read_text())
    for key in ("mode", "n_matches", "n_ties", "n_discarded", "ema_win_rate", "win_rate_ci95", "g3_h2h"):
        assert key in loaded
    assert loaded["n_discarded"] == 2
    assert isinstance(loaded["win_rate_ci95"], list) and len(loaded["win_rate_ci95"]) == 2
