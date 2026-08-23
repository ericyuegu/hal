"""Reward events and discounted returns for whole replays.

Experiments weight losses or fit value functions against the outcome of a match.
This module owns the shared formulas: per-frame reward events from the stock and
percent columns, the discounted return over a complete replay, and the per-port
labeling a loader ``replay_transform`` applies before window sampling.

The formulas were first written and tested inside experiments 020 and 031. This
module is their single home; experiments import it instead of copying it.
"""

import numpy as np
from scipy.signal import lfilter

from hal.wire import MASK_INT32


def stock_loss_events(stock: np.ndarray) -> np.ndarray:
    """1.0 on each frame whose stock count drops below the frame before it, else 0.0.

    The drop is the event; its size is not (Melee never takes two stocks in one
    frame). The counter only rises between games, so increments are ignored, and
    frame 0 has no predecessor, so it can never fire. A masked sentinel on either
    side of a step suppresses the event instead of reading the sentinel as a drop.
    """
    ids = np.asarray(stock).astype(np.int64)
    known = ids != MASK_INT32
    out = np.zeros(ids.shape, dtype=np.float32)
    out[1:] = ((ids[1:] < ids[:-1]) & known[1:] & known[:-1]).astype(np.float32)
    return out


def match_point_events(stock: np.ndarray) -> np.ndarray:
    """1.0 on the frame a player's last stock is lost (the count drops to 0), else 0.0.

    A subset of :func:`stock_loss_events`: the same drop detection, kept only where
    the new count is zero. A game that ends by quit-out never empties a stock
    count, so it has no event.
    """
    ids = np.asarray(stock).astype(np.int64)
    return stock_loss_events(stock) * (ids == 0).astype(np.float32)


def damage_taken(percent: np.ndarray) -> np.ndarray:
    """Per-frame increase in a player's percent, clipped at >= 0.

    The drop back to 0 on a respawn and the NaN of a masked frame are resets, not
    healing, so only rises count.
    """
    values = np.asarray(percent, dtype=np.float32)
    out = np.zeros(values.shape, dtype=np.float32)
    out[1:] = np.maximum(values[1:] - values[:-1], 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def frame_reward(
    sample: dict,
    *,
    ego: str,
    opp: str,
    damage_shaping: float,
    win_reward: float,
    stock_value: float = 1.0,
) -> np.ndarray:
    """Per-frame reward for the player at port ``ego`` over one whole replay.

    ``+stock_value`` when the opponent loses a stock and ``-stock_value`` when the ego does, both on
    the frame the drop becomes visible; plus ``win_reward`` extra on the
    match-deciding stock; plus ``damage_shaping`` times the percent the opponent
    took minus the percent the ego took on that frame. The sparse stock term is
    the outcome the match is scored on; the shaping terms densify the signal.
    """
    stock = stock_loss_events(sample[f"{opp}_stock"]) - stock_loss_events(sample[f"{ego}_stock"])
    reward = stock if stock_value == 1.0 else stock_value * stock
    if win_reward:
        wins = match_point_events(sample[f"{opp}_stock"]) - match_point_events(sample[f"{ego}_stock"])
        reward = reward + win_reward * wins
    if damage_shaping:
        dealt = damage_taken(sample[f"{opp}_percent"]) - damage_taken(sample[f"{ego}_percent"])
        reward = reward + damage_shaping * dealt
    return reward


def discounted_return(reward: np.ndarray, gamma: float) -> np.ndarray:
    """``G_t = sum_k gamma^k * r_{t+k}`` to the end of the episode.

    The recurrence ``G_t = r_t + gamma * G_{t+1}`` is a first-order IIR filter run
    backward in time, so one reversed ``lfilter`` pass computes it exactly. The
    filter runs in float64 and the result is cast to float32.
    """
    values = np.asarray(reward, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("reward must be one-dimensional")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    out = lfilter([1.0], [1.0, -gamma], values[::-1])[::-1]
    return out.astype(np.float32)


def infer_terminal_replay(sample: dict) -> bool:
    """True when the replay ends at a known episode boundary.

    An explicit scalar ``mc_terminated`` attached by a materializer wins. Without
    it, only an observed final stock count of zero is trusted; an ambiguous
    quit-out is not terminal.
    """
    if "mc_terminated" in sample:
        value = np.asarray(sample["mc_terminated"])
        if value.size != 1:
            raise ValueError("mc_terminated must be scalar")
        return bool(value.reshape(-1)[0])
    return any(int(np.asarray(sample[f"{port}_stock"])[-1]) == 0 for port in ("p1", "p2"))


def replay_returns(
    sample: dict,
    *,
    gamma: float,
    damage_shaping: float,
    win_reward: float,
    stock_value: float = 1.0,
    suffix: str,
    terminated: bool | None = None,
) -> dict[str, np.ndarray]:
    """Both ports' return and validity columns for one replay.

    The keys are ``p{1,2}_<suffix>`` and ``p{1,2}_<suffix>_valid``, both full-length
    per-frame arrays. The columns are named per port, not per role, because the
    sampler picks the ego port after windowing: ``dataloader.relabel_ego`` then
    renames the sampled port's columns to ``ego_<suffix>`` with no extra code.

    A truncated replay (``terminated`` false, or inferred false) has an unknown
    return tail, so it gets all-NaN returns and an all-False mask — never a zero
    tail. Consumers must select by the mask with boolean indexing; a NaN times
    zero is still NaN.
    """
    complete = infer_terminal_replay(sample) if terminated is None else bool(terminated)
    out: dict[str, np.ndarray] = {}
    for port, other in (("p1", "p2"), ("p2", "p1")):
        reward = frame_reward(
            sample,
            ego=port,
            opp=other,
            damage_shaping=damage_shaping,
            win_reward=win_reward,
            stock_value=stock_value,
        )
        returns = discounted_return(reward, gamma) if complete else np.full(reward.shape, np.nan, dtype=np.float32)
        out[f"{port}_{suffix}"] = returns
        out[f"{port}_{suffix}_valid"] = np.full(reward.shape, complete, dtype=np.bool_)
    return out


def label_replay(
    sample: dict,
    *,
    gamma: float,
    damage_shaping: float,
    win_reward: float,
    stock_value: float = 1.0,
    suffix: str,
    terminated: bool | None = None,
) -> dict:
    """Add full-episode return labels to a replay row before the loader makes windows."""
    return {
        **sample,
        **replay_returns(
            sample,
            gamma=gamma,
            damage_shaping=damage_shaping,
            win_reward=win_reward,
            stock_value=stock_value,
            suffix=suffix,
            terminated=terminated,
        ),
    }
