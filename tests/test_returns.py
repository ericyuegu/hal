"""Reward and return primitives shared by AWR experiments.

Every case is hand-computed. The functions are the audited 020/022 formulas moved
to one shared home; these tests pin the same properties the experiment tests
pinned, against the shared module.
"""

import numpy as np
import pytest

from hal.training.returns import damage_taken
from hal.training.returns import discounted_returns
from hal.training.returns import frame_reward
from hal.training.returns import is_terminal
from hal.training.returns import match_point_events
from hal.training.returns import replay_reward_columns
from hal.training.returns import stock_loss_events
from hal.wire import MASK_INT32


def test_stock_loss_events_fire_only_on_drops() -> None:
    stock = np.array([4, 4, 3, 3, 4, 2], dtype=np.int32)
    np.testing.assert_array_equal(stock_loss_events(stock), [0.0, 0.0, 1.0, 0.0, 0.0, 1.0])


def test_stock_loss_events_frame_zero_never_fires() -> None:
    assert stock_loss_events(np.array([1], dtype=np.int32)).tolist() == [0.0]


def test_stock_loss_events_mask_sentinel_suppresses_the_event() -> None:
    stock = np.array([4, MASK_INT32, 3, 3], dtype=np.int32)
    np.testing.assert_array_equal(stock_loss_events(stock), [0.0, 0.0, 0.0, 0.0])


def test_match_point_events_fire_only_on_the_deciding_stock() -> None:
    stock = np.array([2, 1, 1, 0, 0], dtype=np.int32)
    np.testing.assert_array_equal(match_point_events(stock), [0.0, 0.0, 0.0, 1.0, 0.0])


def test_match_point_events_quit_out_has_no_event() -> None:
    stock = np.array([2, 1, 1], dtype=np.int32)
    assert match_point_events(stock).sum() == 0.0


def test_damage_taken_counts_rises_only() -> None:
    percent = np.array([0.0, 10.0, 10.0, 55.0, 0.0, 5.0], dtype=np.float32)
    np.testing.assert_allclose(damage_taken(percent), [0.0, 10.0, 0.0, 45.0, 0.0, 5.0])


def test_damage_taken_nan_frames_contribute_zero() -> None:
    percent = np.array([0.0, np.nan, 20.0], dtype=np.float32)
    assert np.isfinite(damage_taken(percent)).all()


def _toy_sample() -> dict[str, np.ndarray]:
    return {
        "p1_stock": np.array([2, 2, 1, 1, 1, 1], dtype=np.int32),
        "p2_stock": np.array([1, 1, 1, 1, 0, 0], dtype=np.int32),
        "p1_percent": np.array([0.0, 40.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "p2_percent": np.array([50.0, 50.0, 50.0, 90.0, 0.0, 0.0], dtype=np.float32),
    }


def test_frame_reward_hand_computed() -> None:
    reward = frame_reward(_toy_sample(), ego="p1", opp="p2", damage_shaping=0.01, win_reward=0.5)
    # frame 1: ego takes 40 percent -> -0.4 * 0.01 * 100 = -0.40 damage term.
    # frame 2: ego loses a stock -> -1.
    # frame 3: opp takes 40 percent -> +0.40.
    # frame 4: opp loses the deciding stock -> +1 + 0.5.
    np.testing.assert_allclose(reward, [0.0, -0.40, -1.0, 0.40, 1.5, 0.0], atol=1e-6)


def test_frame_reward_port_swap_flips_every_sign() -> None:
    sample = _toy_sample()
    ego_view = frame_reward(sample, ego="p1", opp="p2", damage_shaping=0.01, win_reward=0.5)
    opp_view = frame_reward(sample, ego="p2", opp="p1", damage_shaping=0.01, win_reward=0.5)
    np.testing.assert_allclose(ego_view, -opp_view, atol=1e-6)


def test_discounted_returns_match_the_reverse_scan() -> None:
    reward = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32)
    gamma = 0.9
    expected = [
        0.0 + gamma * (1.0 + gamma * (0.0 + gamma * -1.0)),
        1.0 + gamma * (0.0 + gamma * -1.0),
        0.0 + gamma * -1.0,
        -1.0,
    ]
    out = discounted_returns(reward, gamma)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_discounted_returns_cover_the_full_episode_not_a_window() -> None:
    reward = np.zeros(500, dtype=np.float32)
    reward[-1] = 1.0
    out = discounted_returns(reward, 0.999)
    assert out[0] == pytest.approx(0.999**499, rel=1e-5)


def test_replay_reward_columns_keys_and_td_identity() -> None:
    columns = replay_reward_columns(_toy_sample(), gamma=0.9, damage_shaping=0.01, win_reward=0.5)
    assert set(columns) == {"p1_awr_reward", "p1_awr_return", "p2_awr_reward", "p2_awr_return"}
    for port in ("p1", "p2"):
        reward = columns[f"{port}_awr_reward"]
        ret = columns[f"{port}_awr_return"]
        np.testing.assert_allclose(ret[:-1], reward[:-1] + 0.9 * ret[1:], rtol=1e-5, atol=1e-6)
        assert ret[-1] == pytest.approx(reward[-1])


def test_is_terminal_true_only_when_a_last_stock_falls() -> None:
    assert is_terminal(_toy_sample())
    quit_out = _toy_sample()
    quit_out["p2_stock"] = np.array([1, 1, 1, 1, 1, 1], dtype=np.int32)
    assert not is_terminal(quit_out)
