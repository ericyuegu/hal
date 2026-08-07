"""Unit tests for the paired head-to-head statistics."""

import math

import pytest

from hal.eval.h2h import MatchOutcome
from hal.eval.h2h import MatchRecord
from hal.eval.paired import binomial_two_sided_p
from hal.eval.paired import config_stock_diffs
from hal.eval.paired import damage_diff_per_active_minute
from hal.eval.paired import mean_ci
from hal.eval.paired import opponent_of
from hal.eval.paired import sign_test
from hal.eval.paired import stock_diff
from hal.eval.paired import summarize_paired
from hal.eval.paired import wilson_ci

# ---------------------------------------------------------------------------
# Interval and test math
# ---------------------------------------------------------------------------


def test_wilson_ci_matches_the_hand_computed_value():
    # p = 49/96 = 0.5104167, z = 1.96:
    #   denominator = 1 + z^2/n            = 1.04001667
    #   center      = p + z^2/(2n)         = 0.53042500
    #   half width  = z*sqrt(pq/n + z^2/4n^2) = 0.10198107
    low, high = wilson_ci(49, 96)
    assert low == pytest.approx(0.4119584, abs=1e-6)
    assert high == pytest.approx(0.6080726, abs=1e-6)


def test_wilson_ci_brackets_the_point_estimate():
    low, high = wilson_ci(3, 10)
    assert low < 0.3 < high
    assert low >= 0.0 and high <= 1.0


def test_wilson_ci_of_an_empty_sample_is_undefined():
    low, high = wilson_ci(0, 0)
    assert math.isnan(low) and math.isnan(high)


def test_wilson_ci_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson_ci(5, 3)


def test_binomial_two_sided_p_small_cases():
    # n=5: P(X=1) = 5/32. Outcomes no more likely: 0, 1, 4, 5 -> 12/32.
    assert binomial_two_sided_p(1, 5) == pytest.approx(12 / 32)
    # Only the two extremes are as unlikely as 5 successes: 2/32.
    assert binomial_two_sided_p(5, 5) == pytest.approx(2 / 32)
    # A perfectly balanced split cannot be evidence against the fair coin.
    assert binomial_two_sided_p(48, 96) == pytest.approx(1.0)
    # n=1 has two equally likely outcomes, so both are "no more likely".
    assert binomial_two_sided_p(0, 1) == pytest.approx(1.0)


def test_binomial_two_sided_p_is_symmetric():
    assert binomial_two_sided_p(20, 96) == pytest.approx(binomial_two_sided_p(76, 96))


def test_binomial_two_sided_p_of_an_empty_sample_is_undefined():
    assert math.isnan(binomial_two_sided_p(0, 0))


def test_mean_ci_matches_the_hand_computed_value():
    # values 1, 2, 3, 4: mean 2.5, sample variance 5/3, standard error sqrt(5/12).
    mean, low, high = mean_ci([1.0, 2.0, 3.0, 4.0])
    half_width = 1.96 * math.sqrt(5 / 12)
    assert mean == pytest.approx(2.5)
    assert low == pytest.approx(2.5 - half_width)
    assert high == pytest.approx(2.5 + half_width)


def test_mean_ci_needs_two_values_for_an_interval():
    mean, low, high = mean_ci([3.0])
    assert mean == 3.0
    assert math.isnan(low) and math.isnan(high)
    assert all(math.isnan(x) for x in mean_ci([]))


def test_sign_test_counts_and_p_value():
    result = sign_test([2.0, 1.0, -1.0, 0.0, 0.0])
    assert (result.ahead, result.behind, result.even) == (2, 1, 2)
    assert result.pairs == 5
    assert result.p_value == pytest.approx(binomial_two_sided_p(2, 3))


def test_sign_test_of_all_ties_is_undefined():
    result = sign_test([0.0, 0.0])
    assert (result.ahead, result.behind, result.even) == (0, 0, 2)
    assert math.isnan(result.p_value)


# ---------------------------------------------------------------------------
# Aggregation over match records
# ---------------------------------------------------------------------------

_ACTIVE_FRAMES = 3600  # exactly one active minute, so damage rates read back directly


def _record(
    *,
    config_id: int,
    orientation: int,
    stage: str = "BATTLEFIELD",
    character_port_1: str = "FOX",
    character_port_2: str = "MARTH",
    stocks_lost_port_1: int,
    stocks_lost_port_2: int,
    damage_dealt_port_1: float = 0.0,
    damage_dealt_port_2: float = 0.0,
    active_frames: int = _ACTIVE_FRAMES,
    ran: bool = True,
) -> MatchRecord:
    """One synthetic record. Orientation 0 puts alpha on port 1; orientation 1 mirrors it."""
    model_port_1, model_port_2 = ("alpha", "beta") if orientation == 0 else ("beta", "alpha")
    winner_port: int | None = None
    if stocks_lost_port_1 != stocks_lost_port_2:
        winner_port = 1 if stocks_lost_port_1 < stocks_lost_port_2 else 2
    outcome = (
        MatchOutcome(
            total_frames=active_frames + 123,
            active_frames=active_frames,
            stocks_left_port_1=4 - stocks_lost_port_1,
            stocks_left_port_2=4 - stocks_lost_port_2,
            stocks_lost_port_1=stocks_lost_port_1,
            stocks_lost_port_2=stocks_lost_port_2,
            damage_taken_port_1=damage_dealt_port_2,
            damage_taken_port_2=damage_dealt_port_1,
            damage_dealt_port_1=damage_dealt_port_1,
            damage_dealt_port_2=damage_dealt_port_2,
            decided=max(stocks_lost_port_1, stocks_lost_port_2) >= 4,
            hit_frame_budget=False,
            winner_port=winner_port,
            winner_model=None if winner_port is None else (model_port_1 if winner_port == 1 else model_port_2),
        )
        if ran
        else None
    )
    return MatchRecord(
        match_id=f"config{config_id:04d}-{model_port_1}-on-port1",
        config_id=config_id,
        orientation=orientation,
        boot_index=config_id,
        stage=stage,
        stage_id=31,
        model_port_1=model_port_1,
        model_port_2=model_port_2,
        character_port_1=character_port_1,
        character_port_2=character_port_2,
        character_id_port_1=1,
        character_id_port_2=18,
        replay_path=None,
        replay_status="ok",
        identity_stamped=True,
        outcome=outcome,
        input_stats_port_1=None,
        input_stats_port_2=None,
    )


def test_stock_diff_is_focal_relative():
    forward = _record(config_id=0, orientation=0, stocks_lost_port_1=1, stocks_lost_port_2=4)
    mirror = _record(config_id=0, orientation=1, stocks_lost_port_1=1, stocks_lost_port_2=4)

    assert stock_diff(forward, "alpha") == 3.0  # alpha on port 1 took 4, lost 1
    assert stock_diff(forward, "beta") == -3.0
    assert stock_diff(mirror, "alpha") == -3.0  # same match, alpha now on port 2
    assert stock_diff(mirror, "beta") == 3.0


def test_damage_diff_per_active_minute():
    record = _record(
        config_id=0,
        orientation=0,
        stocks_lost_port_1=2,
        stocks_lost_port_2=2,
        damage_dealt_port_1=180.0,
        damage_dealt_port_2=120.0,
        active_frames=2 * _ACTIVE_FRAMES,
    )
    assert damage_diff_per_active_minute(record, "alpha") == pytest.approx(30.0)
    assert damage_diff_per_active_minute(record, "beta") == pytest.approx(-30.0)


def test_damage_diff_is_undefined_without_active_frames():
    record = _record(config_id=0, orientation=0, stocks_lost_port_1=0, stocks_lost_port_2=4, active_frames=0)
    assert damage_diff_per_active_minute(record, "alpha") is None


def test_opponent_of_needs_exactly_two_models():
    records = [_record(config_id=0, orientation=0, stocks_lost_port_1=1, stocks_lost_port_2=4)]
    assert opponent_of(records, "alpha") == "beta"
    with pytest.raises(ValueError, match="played none"):
        opponent_of(records, "gamma")


def test_config_stock_diffs_sum_both_orientations():
    records = [
        # Config 0: port 1 wins both orientations, so the two cancel to 0 for either model.
        _record(config_id=0, orientation=0, stocks_lost_port_1=1, stocks_lost_port_2=4),
        _record(config_id=0, orientation=1, stocks_lost_port_1=1, stocks_lost_port_2=4),
        # Config 1: alpha wins on either port, so its config sum is positive.
        _record(config_id=1, orientation=0, stocks_lost_port_1=0, stocks_lost_port_2=4),
        _record(config_id=1, orientation=1, stocks_lost_port_1=4, stocks_lost_port_2=0),
        # Config 2 lost one orientation, so it cannot be paired.
        _record(config_id=2, orientation=0, stocks_lost_port_1=2, stocks_lost_port_2=4),
        _record(config_id=2, orientation=1, stocks_lost_port_1=2, stocks_lost_port_2=4, ran=False),
    ]

    diffs = config_stock_diffs(records, "alpha")

    assert diffs == {0: 0.0, 1: 8.0}


def test_summarize_paired_over_a_mirrored_sweep():
    records = [
        # alpha sweeps config 0 on both ports.
        _record(config_id=0, orientation=0, stocks_lost_port_1=1, stocks_lost_port_2=4, damage_dealt_port_1=200.0),
        _record(config_id=0, orientation=1, stocks_lost_port_1=4, stocks_lost_port_2=1, damage_dealt_port_2=200.0),
        # Config 1 splits: port 1 wins both times, so the config sum is a draw.
        _record(config_id=1, orientation=0, stage="YOSHIS_STORY", stocks_lost_port_1=2, stocks_lost_port_2=4),
        _record(config_id=1, orientation=1, stage="YOSHIS_STORY", stocks_lost_port_1=2, stocks_lost_port_2=4),
        # Config 2 ends on equal stocks at the budget: a stock tie, no winner.
        _record(config_id=2, orientation=0, stocks_lost_port_1=2, stocks_lost_port_2=2),
        _record(config_id=2, orientation=1, stocks_lost_port_1=2, stocks_lost_port_2=2),
        # Config 3 never ran on one side, so it drops out of the paired view.
        _record(config_id=3, orientation=0, stocks_lost_port_1=0, stocks_lost_port_2=4),
        _record(config_id=3, orientation=1, stocks_lost_port_1=0, stocks_lost_port_2=4, ran=False),
    ]

    summary = summarize_paired(records, focal_model="alpha")

    assert (summary.focal_model, summary.opponent_model) == ("alpha", "beta")
    assert summary.matches == 7  # one record never ran
    assert (summary.wins, summary.losses, summary.stock_ties) == (4, 1, 2)
    assert summary.win_rate == pytest.approx(4 / 5)
    assert summary.win_rate_p_value == pytest.approx(binomial_two_sided_p(4, 5))
    assert summary.decided_by_knockout == 5
    # Per-config sums: 0 -> +6, 1 -> 0, 2 -> 0. Config 3 is unpaired.
    assert summary.paired_configs == 3
    assert (summary.config_sign_test.ahead, summary.config_sign_test.behind) == (1, 0)
    assert summary.config_sign_test.even == 2
    assert summary.mean_config_stock_diff == pytest.approx(2.0)
    # Per-match diffs: +3, +3, +2, -2, 0, 0, +4.
    assert summary.mean_stock_diff_per_match == pytest.approx(10 / 7)
    # Only config 0 recorded damage: +200 per active minute on each of its two matches.
    assert summary.mean_damage_diff_per_active_minute == pytest.approx(400 / 7)
    assert {g.label: g.matches for g in summary.by_stage} == {"BATTLEFIELD": 5, "YOSHIS_STORY": 2}
    assert sum(g.matches for g in summary.by_focal_character) == 7


def test_summarize_paired_is_antisymmetric():
    records = [
        _record(config_id=0, orientation=0, stocks_lost_port_1=1, stocks_lost_port_2=4),
        _record(config_id=0, orientation=1, stocks_lost_port_1=4, stocks_lost_port_2=1),
    ]

    alpha = summarize_paired(records, focal_model="alpha")
    beta = summarize_paired(records, focal_model="beta")

    assert alpha.wins == beta.losses == 2
    assert alpha.mean_stock_diff_per_match == pytest.approx(-beta.mean_stock_diff_per_match)
    assert alpha.mean_config_stock_diff == pytest.approx(-beta.mean_config_stock_diff)


def test_summarize_paired_needs_a_completed_match():
    with pytest.raises(ValueError, match="no completed matches"):
        summarize_paired(
            [_record(config_id=0, orientation=0, stocks_lost_port_1=0, stocks_lost_port_2=4, ran=False)],
            focal_model="alpha",
        )


def test_format_table_reports_both_views():
    records = [
        _record(config_id=0, orientation=0, stocks_lost_port_1=1, stocks_lost_port_2=4),
        _record(config_id=0, orientation=1, stocks_lost_port_1=4, stocks_lost_port_2=1),
    ]

    table = summarize_paired(records, focal_model="alpha").format_table()

    assert "alpha vs beta" in table
    assert "stock-lead rate over non-tied" in table
    assert "paired configs" in table
    assert "-- by stage --" in table
