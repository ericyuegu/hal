"""Shaped per-step + terminal rewards for Melee self-play RL (pure, torch-free).

The reward for step ``t`` is a function of the transition ``prev -> cur`` (frames
``t`` and ``t+1``): damage the ego dealt/took over that frame and any stock that
changed hands. This is the classic RL off-by-one — the reward credited to the
action chosen at frame ``t`` (which produced frame ``t+1``) is read from the
``(t, t+1)`` frame pair, so ``rollout.py`` stores ``T`` rewards alongside ``T+1``
flat frames and the rewards/terminated/truncated arrays index the transition,
never the frame.

Damage terms reuse the exact gating of :func:`hal.data.replay_stats.cumulative_damage`:
a percent delta counts only when both readings are finite AND the same player's
stock did not change across the pair (a death resets percent to 0, which must not
register as a heal, and a NaN sentinel boundary frame must not register at all).
Summed over a whole episode, the per-step damage-dealt term therefore equals
``cumulative_damage`` on the opponent's columns — pinned by a test.
"""

import math
from collections.abc import Mapping

from rl_config import RewardConfig

_OTHER_PORT: dict[int, int] = {1: 2, 2: 1}


def _opp_port(ego_port: int) -> int:
    opp = _OTHER_PORT.get(ego_port)
    if opp is None:
        raise ValueError(f"ego_port must be 1 or 2, got {ego_port}")
    return opp


def _gated_gain(pct_prev: float, pct_cur: float, stk_prev: float, stk_cur: float) -> float:
    """Positive percent delta over a transition, gated exactly like
    ``cumulative_damage``: both percents finite, stock finite and unchanged, delta
    strictly positive. Anything else contributes 0 (death resets, NaN boundaries)."""
    if not (math.isfinite(pct_prev) and math.isfinite(pct_cur)):
        return 0.0
    if not (math.isfinite(stk_prev) and math.isfinite(stk_cur)) or stk_cur != stk_prev:
        return 0.0
    delta = pct_cur - pct_prev
    return delta if delta > 0.0 else 0.0


def step_reward(
    prev_flat: Mapping[str, float],
    cur_flat: Mapping[str, float],
    ego_port: int,
    cfg: RewardConfig,
) -> float:
    """Shaped reward for the transition ``prev_flat -> cur_flat`` from ego's view.

    Sums four terms: ``+damage_dealt * Δopp_percent`` (opponent hurt),
    ``-damage_taken * Δego_percent`` (ego hurt), ``+stock_take`` when the opponent
    loses a stock over the pair, ``-stock_loss`` when ego loses one. Every term is
    NaN-robust via the ``cumulative_damage`` gate; stock events require both stock
    readings finite and a strict decrement.
    """
    opp = _opp_port(ego_port)
    ego_pct_prev, ego_pct_cur = prev_flat[f"p{ego_port}_percent"], cur_flat[f"p{ego_port}_percent"]
    opp_pct_prev, opp_pct_cur = prev_flat[f"p{opp}_percent"], cur_flat[f"p{opp}_percent"]
    ego_stk_prev, ego_stk_cur = prev_flat[f"p{ego_port}_stock"], cur_flat[f"p{ego_port}_stock"]
    opp_stk_prev, opp_stk_cur = prev_flat[f"p{opp}_stock"], cur_flat[f"p{opp}_stock"]

    reward = 0.0
    reward += cfg.damage_dealt * _gated_gain(opp_pct_prev, opp_pct_cur, opp_stk_prev, opp_stk_cur)
    reward -= cfg.damage_taken * _gated_gain(ego_pct_prev, ego_pct_cur, ego_stk_prev, ego_stk_cur)
    if math.isfinite(opp_stk_prev) and math.isfinite(opp_stk_cur) and opp_stk_cur < opp_stk_prev:
        reward += cfg.stock_take
    if math.isfinite(ego_stk_prev) and math.isfinite(ego_stk_cur) and ego_stk_cur < ego_stk_prev:
        reward -= cfg.stock_loss
    return reward


def terminal_bonus(
    final_flat: Mapping[str, float],
    ego_port: int,
    cfg: RewardConfig,
    *,
    terminated: bool,
) -> float:
    """``+win_bonus`` if ego won, ``-win_bonus`` if lost, 0 on a genuine tie or truncation.

    Frame contract: ``final_flat`` MUST be the last pre-reset frame of the ENDED
    match. Under instant-restart the boundary obs after a terminated episode is
    the NEXT match's first frame (finite, both sides back at full stocks — reads
    as a tie), and an IN_GAME→menu transition frame carries NaN per-port fields;
    handing either one here would silently zero the bonus on every match.

    Only a genuine episode end (``terminated``) can pay the bonus — a rollout cut
    at the frame budget (``terminated=False``) always returns 0 so the learner
    doesn't credit a win/loss it never saw. On a terminated episode the correct
    frame always carries a decidable stock reading, so a non-finite stock raises
    (wrong frame) rather than silently paying 0. Equal stocks pay 0: the genuine
    timeout tie has equal-but-depleted stocks. Equal FULL stocks (4-4) is the
    wrong-frame symptom, but it is finite and indistinguishable from a (never
    occurring in practice) 4-4 timeout, so it cannot be detected here — the frame
    contract above is load-bearing."""
    if not terminated:
        return 0.0
    opp = _opp_port(ego_port)
    ego_stock = final_flat[f"p{ego_port}_stock"]
    opp_stock = final_flat[f"p{opp}_stock"]
    if not (math.isfinite(ego_stock) and math.isfinite(opp_stock)):
        raise ValueError(
            f"terminal_bonus: non-finite stock reading (p{ego_port}_stock={ego_stock}, p{opp}_stock={opp_stock}) "
            "on a terminated episode — final_flat must be the last pre-reset frame of the ended match, not an "
            "IN_GAME→menu transition frame or the next match's first frame"
        )
    if ego_stock > opp_stock:
        return cfg.win_bonus
    if ego_stock < opp_stock:
        return -cfg.win_bonus
    return 0.0
