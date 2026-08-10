"""Reward events, discounted returns, and replay return labels for AWR training.

These are the audited 020/022 formulas in one shared home. The reward is scored in
stock units: ``+-1`` per stock, ``win_reward`` extra on the match-deciding stock, and
``damage_shaping`` per percent point. Returns are computed over the WHOLE replay
before window sampling; a return summed inside a training window would be truncated
toward zero exactly where the credit lives.
"""

from collections.abc import Mapping
from typing import Final

import numpy as np
from scipy.signal import lfilter

from hal.wire import mask_value

AWR_REWARD_SUFFIX: Final[str] = "awr_reward"
AWR_RETURN_SUFFIX: Final[str] = "awr_return"


def stock_loss_events(stock: np.ndarray) -> np.ndarray:
    """1.0 on every frame whose stock count DROPS below the frame before it, else 0.0.

    A drop is the event; its size is not (Melee never takes two stocks in one frame).
    Increments are ignored — the counter only rises between games — and frame 0 has no
    predecessor, so it can never fire. A masked sentinel on either side of a step
    suppresses the event rather than reading the sentinel as a huge drop."""
    ids = np.asarray(stock).astype(np.int64)
    known = ids != mask_value(np.int32)
    out = np.zeros(ids.shape, dtype=np.float32)
    out[1:] = ((ids[1:] < ids[:-1]) & known[1:] & known[:-1]).astype(np.float32)
    return out


def match_point_events(stock: np.ndarray) -> np.ndarray:
    """1.0 on the frame a player's LAST stock is lost (the count drops to 0), else 0.0.

    A subset of ``stock_loss_events``: the same drop detection, kept only where the new
    count is zero. A ranked game that ends by quit-out never empties a stock count, so
    it has no event."""
    ids = np.asarray(stock).astype(np.int64)
    return stock_loss_events(stock) * (ids == 0).astype(np.float32)


def damage_taken(percent: np.ndarray) -> np.ndarray:
    """Per-frame INCREASE in a player's percent, clipped at >= 0.

    The drop back to 0 on a respawn (and the masked NaN of an unavailable frame) is a
    reset, not healing, so only rises count."""
    values = np.asarray(percent, dtype=np.float32)
    out = np.zeros(values.shape, dtype=np.float32)
    out[1:] = np.maximum(values[1:] - values[:-1], 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def frame_reward(
    sample: Mapping[str, np.ndarray], *, ego: str, opp: str, damage_shaping: float, win_reward: float
) -> np.ndarray:
    """Per-frame reward for the player at port ``ego`` over one whole replay.

    ``+1`` when the opponent loses a stock, ``-1`` when the ego does (both on the frame
    the drop becomes visible), plus ``win_reward`` extra on the match-deciding stock,
    plus ``damage_shaping`` times the percent the opponent took minus the percent the
    ego took on that frame. The sparse stock term is the outcome the match is scored
    on; the shaping terms densify the signal without redefining it."""
    reward = stock_loss_events(sample[f"{opp}_stock"]) - stock_loss_events(sample[f"{ego}_stock"])
    if win_reward:
        wins = match_point_events(sample[f"{opp}_stock"]) - match_point_events(sample[f"{ego}_stock"])
        reward = reward + win_reward * wins
    if damage_shaping:
        dealt = damage_taken(sample[f"{opp}_percent"]) - damage_taken(sample[f"{ego}_percent"])
        reward = reward + damage_shaping * dealt
    return reward


def discounted_returns(reward: np.ndarray, gamma: float) -> np.ndarray:
    """``G_t = sum_k gamma^k r_{t+k}`` to the END OF THE EPISODE, by one reverse scan.

    The scan is a one-pole IIR filter on the reversed reward — ``y[n] = x[n] +
    gamma*y[n-1]`` — so ``lfilter`` runs it in C with double-precision accumulation."""
    tail = lfilter([1.0], [1.0, -gamma], np.asarray(reward, dtype=np.float64)[::-1])
    return tail[::-1].astype(np.float32)


def replay_reward_columns(
    sample: Mapping[str, np.ndarray], *, gamma: float, damage_shaping: float, win_reward: float
) -> dict[str, np.ndarray]:
    """Both ports' reward AND return columns for one replay, keyed ``p{1,2}_awr_{reward,return}``.

    Named per port rather than per role because the sampler picks the ego port AFTER
    windowing: ``dataloader.relabel_ego`` then renames the right ones to ``ego_awr_*``
    with no AWR-specific code. Both travel through the same windowing and padding,
    which is what makes ``G_t = r_t + gamma * G_{t+1}`` hold position by position on
    the collated arrays."""
    out: dict[str, np.ndarray] = {}
    for port, other in (("p1", "p2"), ("p2", "p1")):
        reward = frame_reward(sample, ego=port, opp=other, damage_shaping=damage_shaping, win_reward=win_reward)
        out[f"{port}_{AWR_REWARD_SUFFIX}"] = reward.astype(np.float32)
        out[f"{port}_{AWR_RETURN_SUFFIX}"] = discounted_returns(reward, gamma)
    return out


def is_terminal(sample: Mapping[str, np.ndarray]) -> bool:
    """True when either port's LAST stock falls inside the replay.

    A terminal replay has a known return tail. A quit-out or a truncated recording
    does not; its rows keep behavior-cloning weight and supply no value target."""
    return bool(match_point_events(sample["p1_stock"]).any() or match_point_events(sample["p2_stock"]).any())
