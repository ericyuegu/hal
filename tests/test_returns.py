"""Contracts for the shared reward and discounted-return formulas."""

import numpy as np

from hal.data.policy_schema import pack_player_state
from hal.training import returns
from hal.wire import MASK_INT32

# One six-frame replay, checked by hand below. Player 2's last stock empties on
# the final frame, so the replay is terminal.
_P1_STOCK = np.array([2, 2, 1, 1, 1, 1])
_P2_STOCK = np.array([2, 2, 2, 1, 1, 0])
_P1_PERCENT = np.array([0.0, 10.0, 0.0, 0.0, 5.0, 5.0])  # the drop at index 2 is a respawn reset
_P2_PERCENT = np.array([0.0, 0.0, 20.0, 20.0, 20.0, 20.0])


def _sample() -> dict:
    return {
        "p1_stock": _P1_STOCK.copy(),
        "p2_stock": _P2_STOCK.copy(),
        "p1_percent": _P1_PERCENT.copy(),
        "p2_percent": _P2_PERCENT.copy(),
    }


def _compact_sample() -> dict[str, object]:
    def packed_state(stock: np.ndarray) -> np.ndarray:
        zeros = np.zeros(stock.shape, dtype=np.int32)
        return pack_player_state(
            {
                "action": zeros,
                "stock": stock,
                "jumps_used": zeros,
                "hurtbox_state": zeros,
                "airborne": zeros,
                "direction": np.zeros(stock.shape, dtype=np.float32),
            }
        )

    return {
        "num_frames": np.int64(len(_P1_STOCK)),
        "p1_percent": _P1_PERCENT.copy(),
        "p2_percent": _P2_PERCENT.copy(),
        "p1_state": packed_state(_P1_STOCK),
        "p2_state": packed_state(_P2_STOCK),
    }


# Hand-derived per-frame reward for player 1 with damage_shaping=0.1, win_reward=0.5:
#   index 1: takes 10 percent                          -> -1.0
#   index 2: loses a stock, deals 20 percent           -> -1.0 + 2.0 = +1.0
#   index 3: opponent loses a stock                    -> +1.0
#   index 4: takes 5 percent                           -> -0.5
#   index 5: opponent loses the deciding stock         -> +1.0 + 0.5 = +1.5
_REWARD_P1 = np.array([0.0, -1.0, 1.0, 1.0, -0.5, 1.5], dtype=np.float32)


def test_frame_reward_matches_the_hand_derivation() -> None:
    reward = returns.frame_reward(_sample(), ego="p1", opp="p2", damage_shaping=0.1, win_reward=0.5)
    np.testing.assert_allclose(reward, _REWARD_P1, atol=1e-6)


def test_stock_value_scales_only_stock_events() -> None:
    reward = returns.frame_reward(_sample(), ego="p1", opp="p2", damage_shaping=0.1, win_reward=0.5, stock_value=120.0)
    expected = np.array([0.0, -1.0, -118.0, 120.0, -0.5, 120.5], dtype=np.float32)
    np.testing.assert_allclose(reward, expected, atol=1e-6)


def test_default_stock_value_is_bitwise_compatible() -> None:
    sample = _sample()
    stock = returns.stock_loss_events(sample["p2_stock"]) - returns.stock_loss_events(sample["p1_stock"])
    legacy = stock + 0.5 * (
        returns.match_point_events(sample["p2_stock"]) - returns.match_point_events(sample["p1_stock"])
    )
    legacy = legacy + 0.1 * (returns.damage_taken(sample["p2_percent"]) - returns.damage_taken(sample["p1_percent"]))
    actual = returns.frame_reward(sample, ego="p1", opp="p2", damage_shaping=0.1, win_reward=0.5)
    np.testing.assert_array_equal(actual, legacy.astype(np.float32, copy=False))


def test_swapping_ports_flips_every_reward_and_return_sign() -> None:
    sample = _sample()
    kwargs = dict(damage_shaping=0.1, win_reward=0.5)
    p1 = returns.frame_reward(sample, ego="p1", opp="p2", **kwargs)
    p2 = returns.frame_reward(sample, ego="p2", opp="p1", **kwargs)
    np.testing.assert_allclose(p1, -p2, atol=1e-6)
    labeled = returns.replay_returns(sample, gamma=0.5, suffix="r", **kwargs)
    np.testing.assert_allclose(labeled["p1_r"], -labeled["p2_r"], atol=1e-5)


def test_discounted_return_matches_the_hand_scan() -> None:
    # G_t = r_t + 0.5 * G_{t+1}, computed by hand from _REWARD_P1.
    expected = np.array([-0.109375, -0.21875, 1.5625, 1.125, 0.25, 1.5], dtype=np.float32)
    np.testing.assert_allclose(returns.discounted_return(_REWARD_P1, 0.5), expected, atol=1e-6)


def test_stock_sentinels_suppress_events_on_both_sides() -> None:
    assert returns.stock_loss_events(np.array([2, MASK_INT32, 1])).sum() == 0
    assert returns.stock_loss_events(np.array([MASK_INT32, 2, 2])).sum() == 0
    assert returns.stock_loss_events(np.array([2, 1, MASK_INT32])).sum() == 1


def test_match_point_fires_only_when_the_count_empties() -> None:
    np.testing.assert_array_equal(returns.match_point_events(_P2_STOCK), [0, 0, 0, 0, 0, 1])
    np.testing.assert_array_equal(returns.match_point_events(_P1_STOCK), np.zeros(6))


def test_damage_counts_rises_only() -> None:
    percent = np.array([0.0, 30.0, np.nan, 0.0, 10.0])
    np.testing.assert_allclose(returns.damage_taken(percent), [0.0, 30.0, 0.0, 0.0, 10.0])


def test_truncated_replay_gets_nan_returns_and_a_false_mask() -> None:
    sample = _sample()
    sample["p2_stock"] = np.array([2, 2, 2, 1, 1, 1])  # nobody empties a stock count
    labeled = returns.replay_returns(sample, gamma=0.9, damage_shaping=0.0, win_reward=0.0, suffix="r")
    assert np.isnan(labeled["p1_r"]).all() and np.isnan(labeled["p2_r"]).all()
    assert not labeled["p1_r_valid"].any() and not labeled["p2_r_valid"].any()
    forced = returns.replay_returns(sample, gamma=0.9, damage_shaping=0.0, win_reward=0.0, suffix="r", terminated=True)
    assert np.isfinite(forced["p1_r"]).all() and forced["p1_r_valid"].all()
    sample["mc_terminated"] = np.array([True])
    assert returns.infer_terminal_replay(sample)


def test_label_replay_keeps_every_source_column_and_adds_four() -> None:
    sample = _sample()
    labeled = returns.label_replay(sample, gamma=0.5, damage_shaping=0.1, win_reward=0.5, suffix="awr_return")
    assert set(labeled) - set(sample) == {
        "p1_awr_return",
        "p1_awr_return_valid",
        "p2_awr_return",
        "p2_awr_return_valid",
    }
    for name, value in sample.items():
        assert labeled[name] is value


def test_compact_policy_returns_match_decoded_replay_exactly() -> None:
    kwargs = {
        "gamma": 0.99618,
        "damage_shaping": 1.0,
        "win_reward": 50.0,
        "stock_value": 120.0,
        "suffix": "awr_return",
    }

    expected = returns.replay_returns(_sample(), **kwargs)
    actual = returns.compact_policy_returns(_compact_sample(), **kwargs)

    assert actual.keys() == expected.keys()
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])


def test_compact_policy_returns_preserve_truncation_mask() -> None:
    compact = _compact_sample()
    p2_stock = np.array([2, 2, 2, 1, 1, 1])
    zeros = np.zeros(p2_stock.shape, dtype=np.int32)
    compact["p2_state"] = pack_player_state(
        {
            "action": zeros,
            "stock": p2_stock,
            "jumps_used": zeros,
            "hurtbox_state": zeros,
            "airborne": zeros,
            "direction": np.zeros(p2_stock.shape, dtype=np.float32),
        }
    )

    actual = returns.compact_policy_returns(
        compact,
        gamma=0.9,
        damage_shaping=1.0,
        win_reward=50.0,
        stock_value=120.0,
        suffix="awr_return",
    )

    assert np.isnan(actual["p1_awr_return"]).all()
    assert np.isnan(actual["p2_awr_return"]).all()
    assert not actual["p1_awr_return_valid"].any()
    assert not actual["p2_awr_return_valid"].any()


def test_lfilter_matches_a_naive_float64_reverse_scan() -> None:
    rng = np.random.default_rng(0)
    reward = rng.normal(scale=0.2, size=4000)
    gamma = 0.99827
    expected = np.empty(len(reward))
    carry = 0.0
    for index in range(len(reward) - 1, -1, -1):
        carry = reward[index] + gamma * carry
        expected[index] = carry
    np.testing.assert_allclose(returns.discounted_return(reward, gamma), expected, atol=1e-5)


def test_output_arrays_are_full_length_with_declared_dtypes() -> None:
    labeled = returns.replay_returns(_sample(), gamma=0.5, damage_shaping=0.1, win_reward=0.5, suffix="r")
    for port in ("p1", "p2"):
        assert labeled[f"{port}_r"].dtype == np.float32 and labeled[f"{port}_r"].shape == (6,)
        assert labeled[f"{port}_r_valid"].dtype == np.bool_ and labeled[f"{port}_r_valid"].shape == (6,)
