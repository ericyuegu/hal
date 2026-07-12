"""Shaped-reward unit tests (pure CPU, fast).

Pins the two load-bearing invariants of ``rewards.py``: the per-step damage terms
sum to ``hal.data.replay_stats.cumulative_damage`` on the same percent/stock
streams (identical NaN + stock-reset gating), and stock/terminal events fire
exactly once with the correct sign.
"""

import math

import numpy as np
import pytest
from rewards import step_reward
from rewards import terminal_bonus
from rl_config import RewardConfig

from hal.data.replay_stats import cumulative_damage

NAN = float("nan")


def _frames(p1_pct: list, p2_pct: list, p1_stk: list, p2_stk: list) -> list[dict]:
    return [
        {"p1_percent": a, "p2_percent": b, "p1_stock": c, "p2_stock": d}
        for a, b, c, d in zip(p1_pct, p2_pct, p1_stk, p2_stk, strict=True)
    ]


def _sum_damage_dealt(frames: list[dict], cfg: RewardConfig, ego_port: int = 1) -> float:
    return sum(step_reward(frames[t], frames[t + 1], ego_port, cfg) for t in range(len(frames) - 1))


# damage_dealt only (=1.0), everything else zeroed → per-step sum must equal
# cumulative_damage on the OPPONENT's (p2) columns for ego port 1.
_DEALT_ONLY = RewardConfig(damage_dealt=1.0, damage_taken=0.0, stock_take=0.0, stock_loss=0.0, win_bonus=0.0)


def test_plain_damage_accumulation() -> None:
    p2_pct = [0.0, 10.0, 24.0, 24.0, 40.0]
    p2_stk = [4.0, 4.0, 4.0, 4.0, 4.0]
    frames = _frames([0.0] * 5, p2_pct, [4.0] * 5, p2_stk)
    got = _sum_damage_dealt(frames, _DEALT_ONLY)
    assert math.isclose(got, cumulative_damage(np.array(p2_pct), np.array(p2_stk)))
    assert math.isclose(got, 40.0)


def test_nan_boundary_frame_skipped() -> None:
    # A masked (NaN) transition frame contributes nothing and does not corrupt the
    # neighbours — the delta into and out of the NaN is dropped, exactly like diff.
    p2_pct = [0.0, 12.0, NAN, 30.0, 45.0]
    p2_stk = [4.0, 4.0, NAN, 4.0, 4.0]
    frames = _frames([0.0] * 5, p2_pct, [4.0] * 5, p2_stk)
    got = _sum_damage_dealt(frames, _DEALT_ONLY)
    assert math.isclose(got, cumulative_damage(np.array(p2_pct), np.array(p2_stk)))
    assert math.isclose(got, 27.0)  # only 0->12 (12) and 30->45 (15) survive the gate


def test_respawn_reset_not_counted_as_damage() -> None:
    # Stock decrement resets percent to 0; the negative delta AND the stock-change
    # frame must both be excluded so a death is not read as a huge heal or damage.
    p2_pct = [0.0, 80.0, 0.0, 15.0]
    p2_stk = [4.0, 4.0, 3.0, 3.0]
    frames = _frames([0.0] * 4, p2_pct, [4.0] * 4, p2_stk)
    got = _sum_damage_dealt(frames, _DEALT_ONLY)
    assert math.isclose(got, cumulative_damage(np.array(p2_pct), np.array(p2_stk)))
    assert math.isclose(got, 80.0 + 15.0)  # the 80->0 reset is dropped


def test_damage_taken_matches_cumulative_on_ego() -> None:
    p1_pct = [0.0, 5.0, 5.0, 22.0]
    p1_stk = [4.0, 4.0, 4.0, 4.0]
    frames = _frames(p1_pct, [0.0] * 4, p1_stk, [4.0] * 4)
    taken_only = RewardConfig(damage_dealt=0.0, damage_taken=1.0, stock_take=0.0, stock_loss=0.0, win_bonus=0.0)
    got = sum(step_reward(frames[t], frames[t + 1], 1, taken_only) for t in range(len(frames) - 1))
    assert math.isclose(-got, cumulative_damage(np.array(p1_pct), np.array(p1_stk)))
    assert math.isclose(got, -22.0)


def test_stock_take_and_loss_fire_once_each() -> None:
    cfg = RewardConfig(damage_dealt=0.0, damage_taken=0.0, stock_take=1.0, stock_loss=1.0, win_bonus=0.0)
    # opp loses a stock at t=1 (+1), ego loses one at t=2 (-1); each event fires once.
    frames = _frames([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [4.0, 4.0, 3.0, 3.0], [4.0, 3.0, 3.0, 3.0])
    rewards = [step_reward(frames[t], frames[t + 1], 1, cfg) for t in range(len(frames) - 1)]
    assert rewards == [1.0, -1.0, 0.0]


def test_stock_events_skip_nan_boundaries() -> None:
    cfg = RewardConfig(damage_dealt=0.0, damage_taken=0.0, stock_take=1.0, stock_loss=1.0, win_bonus=0.0)
    # A NaN stock reading on either side of a decrement must not register a stock event.
    frames = _frames([0.0] * 3, [0.0] * 3, [4.0, 4.0, 4.0], [4.0, NAN, 3.0])
    rewards = [step_reward(frames[t], frames[t + 1], 1, cfg) for t in range(len(frames) - 1)]
    assert rewards == [0.0, 0.0]


def test_terminal_bonus_win_loss_tie_truncation() -> None:
    cfg = RewardConfig(win_bonus=2.0)
    win = {"p1_percent": 0.0, "p2_percent": 0.0, "p1_stock": 2.0, "p2_stock": 0.0}
    loss = {"p1_percent": 0.0, "p2_percent": 0.0, "p1_stock": 0.0, "p2_stock": 3.0}
    tie = {"p1_percent": 0.0, "p2_percent": 0.0, "p1_stock": 1.0, "p2_stock": 1.0}
    assert terminal_bonus(win, 1, cfg, terminated=True) == 2.0
    assert terminal_bonus(loss, 1, cfg, terminated=True) == -2.0
    # Equal-but-depleted stocks = the genuine timeout tie -> legitimate 0.
    assert terminal_bonus(tie, 1, cfg, terminated=True) == 0.0
    # Truncation never pays a bonus, even from a winning frame.
    assert terminal_bonus(win, 1, cfg, terminated=False) == 0.0
    # Winner is port-relative: the loss frame is a WIN for port 2.
    assert terminal_bonus(loss, 2, cfg, terminated=True) == 2.0


def test_terminal_bonus_nan_on_terminated_raises() -> None:
    # A terminated match always has a decidable stock reading on the correct
    # (last pre-reset) frame; NaN there means the caller handed the wrong frame
    # (IN_GAME->menu transition or the next match's first frame) -> fail loud.
    cfg = RewardConfig(win_bonus=2.0)
    nan_frame = {"p1_percent": 0.0, "p2_percent": 0.0, "p1_stock": NAN, "p2_stock": 1.0}
    with pytest.raises(ValueError, match="non-finite stock"):
        terminal_bonus(nan_frame, 1, cfg, terminated=True)
    # Truncation is exempt: the budget can cut anywhere, including on a masked frame.
    assert terminal_bonus(nan_frame, 1, cfg, terminated=False) == 0.0
