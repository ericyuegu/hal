"""Tests for the closed-loop eval reductions (``hal.eval.cross_stage``).

These exercise the pure post-processing — active-frame accounting, per-match row
construction, bootstrap CIs, and paired cross-checkpoint deltas — on synthetic
trajectories / summaries / rows. No emulator, no model: the sweep functions that
boot Dolphin are covered elsewhere; here we pin the statistics they feed.
"""

import numpy as np
import pytest
from melee import Character
from melee import Stage

from hal.data.replay_stats import cumulative_damage
from hal.eval.cross_stage import FRAMES_PER_MINUTE
from hal.eval.cross_stage import PREGAME_FRAMES
from hal.eval.cross_stage import STARTING_STOCKS
from hal.eval.cross_stage import MatchRow
from hal.eval.cross_stage import match_rows
from hal.eval.cross_stage import paired_vs_cpu_deltas
from hal.eval.cross_stage import sweep_vs_cpu_prior_with_rows
from hal.eval.cross_stage import vs_cpu_metrics
from hal.eval.match_summary import MatchSummary
from hal.eval.match_summary import last_finite_stock
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.trajectory import Trajectory
from hal.sim.vec import VecMatch

_LEGACY_METRIC_KEYS = {
    "stocks_taken_per_min",
    "stocks_lost_per_min",
    "damage_dealt_per_min",
    "damage_taken_per_min",
    "frames",
    "matches",
    "crashed",
}
_ONE_ACTIVE_MINUTE = PREGAME_FRAMES + FRAMES_PER_MINUTE  # total frames whose active span == 1 min


# --------------------------------------------------------------------------- helpers


def _summary(frames: int, *, p1_left: int, p2_left: int, p1_dmg: float, p2_dmg: float) -> MatchSummary:
    return MatchSummary(
        frames=frames,
        p1_stocks_left=p1_left,
        p2_stocks_left=p2_left,
        p1_damage_taken=p1_dmg,
        p2_damage_taken=p2_dmg,
    )


def _traj(
    frame_ids: list[int],
    *,
    p1_stock: list[float],
    p2_stock: list[float],
    p1_pct: list[float],
    p2_pct: list[float],
) -> Trajectory:
    n = len(frame_ids)
    assert all(len(a) == n for a in (p1_stock, p2_stock, p1_pct, p2_pct))
    return Trajectory(
        frame_id=np.array(frame_ids, dtype=np.int32),
        post={
            1: {"stock": np.array(p1_stock, dtype=np.float64), "percent": np.array(p1_pct, dtype=np.float64)},
            2: {"stock": np.array(p2_stock, dtype=np.float64), "percent": np.array(p2_pct, dtype=np.float64)},
        },
        random_seed=np.zeros(n, dtype=np.uint32),
    )


def _vm(
    *,
    ego_char: Character = Character.FOX,
    opp_char: Character = Character.FALCO,
    stage: Stage = Stage.BATTLEFIELD,
    ego_port: int = 1,
    cpu_level: int = 9,
) -> VecMatch:
    cpu_port = 2 if ego_port == 1 else 1
    return VecMatch(
        matchup=Matchup(
            stage=stage,
            players=(
                PlayerSetup(port=ego_port, character=ego_char, cpu_level=0),
                PlayerSetup(port=cpu_port, character=opp_char, cpu_level=cpu_level),
            ),
        ),
        model_ports=(ego_port,),
    )


def _row(
    *,
    boot_index: int,
    match_ordinal: int,
    ego_character: int = int(Character.FOX.value),
    opp_character: int = int(Character.FALCO.value),
    stage: int = int(Stage.BATTLEFIELD.value),
    active_frames: int = FRAMES_PER_MINUTE,
    damage_dealt: float = 0.0,
    damage_taken: float = 0.0,
    stocks_taken: int = 0,
    stocks_lost: int = 0,
) -> MatchRow:
    return MatchRow(
        ego_character=ego_character,
        opp_character=opp_character,
        stage=stage,
        boot_index=boot_index,
        match_ordinal=match_ordinal,
        active_frames=active_frames,
        total_frames=active_frames + PREGAME_FRAMES,
        damage_dealt=damage_dealt,
        damage_taken=damage_taken,
        stocks_taken=stocks_taken,
        stocks_lost=stocks_lost,
    )


# ------------------------------------------------------------------ vs_cpu_metrics


def test_rates_are_per_active_minute_not_total() -> None:
    """A match of one active minute (+ the countdown) yields rates equal to its raw
    per-match numerators — the countdown frames are excluded from the denominator."""
    s = _summary(_ONE_ACTIVE_MINUTE, p1_left=2, p2_left=1, p1_dmg=45.0, p2_dmg=90.0)
    m = vs_cpu_metrics([(Stage.BATTLEFIELD, 0, s)], bootstrap_resamples=200, seed=0)

    assert m["damage_dealt_per_min"] == pytest.approx(90.0)  # ego dealt = p2 took
    assert m["damage_taken_per_min"] == pytest.approx(45.0)
    assert m["stocks_taken_per_min"] == pytest.approx(3.0)  # 4 - p2_left
    assert m["stocks_lost_per_min"] == pytest.approx(2.0)  # 4 - p1_left
    # The old total-frame denominator would have divided by 3723/3600 min -> ~87.0.
    old_style = 90.0 / (_ONE_ACTIVE_MINUTE / FRAMES_PER_MINUTE)
    assert m["damage_dealt_per_min"] != pytest.approx(old_style)
    assert m["dead_frame_frac"] == pytest.approx(PREGAME_FRAMES / _ONE_ACTIVE_MINUTE)


def test_keeps_legacy_keys_and_adds_new_ones() -> None:
    s = _summary(_ONE_ACTIVE_MINUTE, p1_left=3, p2_left=2, p1_dmg=10.0, p2_dmg=20.0)
    m = vs_cpu_metrics([(Stage.BATTLEFIELD, 0, s)], bootstrap_resamples=100)

    assert m.keys() >= _LEGACY_METRIC_KEYS
    assert m["frames"] == pytest.approx(float(_ONE_ACTIVE_MINUTE))  # mean total frames, unchanged meaning
    assert m["matches"] == 1.0
    assert m["crashed"] == 0.0
    assert "dead_frame_frac" in m
    for key in ("damage_dealt_per_min", "stocks_taken_per_min"):
        assert f"{key}_ci_lo" in m and f"{key}_ci_hi" in m


def test_all_crashed_returns_minimal_dict() -> None:
    assert vs_cpu_metrics([(Stage.BATTLEFIELD, 0, None), (Stage.BATTLEFIELD, 1, None)]) == {"crashed": 1.0}


def test_crashed_fraction_counts_none_rows() -> None:
    good = _summary(_ONE_ACTIVE_MINUTE, p1_left=4, p2_left=4, p1_dmg=0.0, p2_dmg=0.0)
    m = vs_cpu_metrics([(Stage.BATTLEFIELD, 0, good), (Stage.BATTLEFIELD, 1, None)], bootstrap_resamples=50)
    assert m["crashed"] == pytest.approx(0.5)
    assert m["matches"] == 1.0


def test_countdown_only_fragment_is_not_a_completed_match() -> None:
    partial = _summary(PREGAME_FRAMES - 73, p1_left=4, p2_left=4, p1_dmg=0.0, p2_dmg=0.0)
    m = vs_cpu_metrics([(Stage.BATTLEFIELD, 0, partial)], bootstrap_resamples=50)
    assert m == {"matches": 0.0, "zero_active": 1.0, "crashed": 0.0}


def test_countdown_only_fragment_does_not_change_rates_or_intervals() -> None:
    complete = _summary(_ONE_ACTIVE_MINUTE, p1_left=3, p2_left=2, p1_dmg=10.0, p2_dmg=20.0)
    partial = _summary(PREGAME_FRAMES - 1, p1_left=4, p2_left=4, p1_dmg=0.0, p2_dmg=0.0)
    baseline = vs_cpu_metrics([(Stage.BATTLEFIELD, 0, complete)], bootstrap_resamples=100, seed=3)
    with_partial = vs_cpu_metrics(
        [(Stage.BATTLEFIELD, 0, complete), (Stage.BATTLEFIELD, 1, partial)],
        bootstrap_resamples=100,
        seed=3,
    )
    assert with_partial == {**baseline, "zero_active": 1.0}


def test_bootstrap_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(7)
    result = []
    for i in range(60):
        dmg = 60.0 + float(rng.integers(0, 40))
        result.append(
            (Stage.BATTLEFIELD, i, _summary(_ONE_ACTIVE_MINUTE, p1_left=3, p2_left=2, p1_dmg=20.0, p2_dmg=dmg))
        )
    m = vs_cpu_metrics(result, bootstrap_resamples=1000, seed=0)
    for key in ("damage_dealt_per_min", "stocks_taken_per_min", "damage_taken_per_min"):
        assert m[f"{key}_ci_lo"] <= m[key] <= m[f"{key}_ci_hi"]
        assert m[f"{key}_ci_lo"] <= m[f"{key}_ci_hi"]


def test_bootstrap_ci_is_wider_with_fewer_matches() -> None:
    """Same per-match distribution, 8x more matches -> a tighter CI (~1/sqrt(n))."""

    def result(n: int) -> list:
        out = []
        for i in range(n):
            dmg = 50.0 + float(i % 10) * 6.0  # deterministic spread, real variance
            out.append(
                (Stage.BATTLEFIELD, i, _summary(_ONE_ACTIVE_MINUTE, p1_left=3, p2_left=2, p1_dmg=20.0, p2_dmg=dmg))
            )
        return out

    wide = vs_cpu_metrics(result(25), bootstrap_resamples=1000, seed=1)
    narrow = vs_cpu_metrics(result(200), bootstrap_resamples=1000, seed=1)
    width_wide = wide["damage_dealt_per_min_ci_hi"] - wide["damage_dealt_per_min_ci_lo"]
    width_narrow = narrow["damage_dealt_per_min_ci_hi"] - narrow["damage_dealt_per_min_ci_lo"]
    assert width_wide > width_narrow


def test_metrics_are_deterministic_under_fixed_seed() -> None:
    result = [
        (
            Stage.BATTLEFIELD,
            i,
            _summary(_ONE_ACTIVE_MINUTE, p1_left=3, p2_left=2, p1_dmg=float(i), p2_dmg=float(2 * i)),
        )
        for i in range(20)
    ]
    assert vs_cpu_metrics(result, seed=3) == vs_cpu_metrics(result, seed=3)


# ---------------------------------------------------------------------- match_rows


def test_match_rows_active_frames_ordinal_and_labels() -> None:
    flat = {"p1_stock": [4.0] * 5, "p2_stock": [4.0] * 5, "p1_pct": [0.0] * 5, "p2_pct": [0.0] * 5}
    boot0 = [
        _traj([-2, -1, 0, 1, 2], **flat),  # 3 active
        _traj([-1, 0, 1, 2, 3], **flat),  # 4 active
    ]
    boot1 = [_traj([-3, -2, -1, 0, 1], **flat)]  # 2 active
    matches = [
        _vm(ego_char=Character.FOX, opp_char=Character.FALCO, stage=Stage.BATTLEFIELD),
        _vm(ego_char=Character.MARTH, opp_char=Character.SHEIK, stage=Stage.FINAL_DESTINATION),
    ]

    rows = match_rows([boot0, boot1], matches, ego_port=1)

    assert [(r.boot_index, r.match_ordinal) for r in rows] == [(0, 0), (0, 1), (1, 0)]
    assert [r.active_frames for r in rows] == [3, 4, 2]
    assert [r.total_frames for r in rows] == [5, 5, 5]
    assert (rows[0].ego_character, rows[0].opp_character) == (int(Character.FOX.value), int(Character.FALCO.value))
    assert rows[0].stage == int(Stage.BATTLEFIELD.value)
    assert (rows[2].ego_character, rows[2].opp_character) == (int(Character.MARTH.value), int(Character.SHEIK.value))
    assert rows[2].stage == int(Stage.FINAL_DESTINATION.value)


def test_match_rows_empty_boot_contributes_no_rows() -> None:
    flat = {"p1_stock": [4.0] * 3, "p2_stock": [4.0] * 3, "p1_pct": [0.0] * 3, "p2_pct": [0.0] * 3}
    rows = match_rows([[], [_traj([-1, 0, 1], **flat)]], [_vm(), _vm()], ego_port=1)
    assert [r.boot_index for r in rows] == [1]
    assert rows[0].match_ordinal == 0


def test_match_rows_ego_relative_damage_and_stocks() -> None:
    # p1 takes 20 then loses a stock; p2 takes 70 then loses two stocks.
    p1_pct = [0.0, 20.0, 20.0, 0.0]
    p1_stock = [4.0, 4.0, 4.0, 3.0]
    p2_pct = [0.0, 70.0, 0.0, 0.0]
    p2_stock = [4.0, 4.0, 3.0, 2.0]
    traj = _traj([-1, 0, 1, 2], p1_stock=p1_stock, p2_stock=p2_stock, p1_pct=p1_pct, p2_pct=p2_pct)

    exp_p1_dmg = cumulative_damage(np.array(p1_pct), np.array(p1_stock))
    exp_p2_dmg = cumulative_damage(np.array(p2_pct), np.array(p2_stock))
    exp_p1_left = last_finite_stock(np.array(p1_stock))
    exp_p2_left = last_finite_stock(np.array(p2_stock))

    (ego1,) = match_rows([[traj]], [_vm(ego_port=1)], ego_port=1)
    assert ego1.damage_dealt == pytest.approx(exp_p2_dmg)  # ego dealt = opp (p2) took
    assert ego1.damage_taken == pytest.approx(exp_p1_dmg)
    assert ego1.stocks_taken == STARTING_STOCKS - exp_p2_left
    assert ego1.stocks_lost == STARTING_STOCKS - exp_p1_left

    # Same trajectory, ego on port 2 -> dealt/taken and stocks swap.
    (ego2,) = match_rows([[traj]], [_vm(ego_port=2)], ego_port=2)
    assert ego2.damage_dealt == pytest.approx(exp_p1_dmg)
    assert ego2.damage_taken == pytest.approx(exp_p2_dmg)
    assert ego2.stocks_taken == STARTING_STOCKS - exp_p1_left
    assert ego2.stocks_lost == STARTING_STOCKS - exp_p2_left


def test_match_row_dict_roundtrip() -> None:
    row = _row(boot_index=2, match_ordinal=1, damage_dealt=33.0, stocks_taken=2)
    assert MatchRow.from_dict(row.as_dict()) == row
    # Extra annotation keys are ignored on load.
    assert MatchRow.from_dict({**row.as_dict(), "run": "abc", "step": 1000}) == row


def test_combined_prior_sweep_reuses_identical_boots_for_summaries_and_rows(monkeypatch) -> None:
    flat = {"p1_stock": [4.0] * 3, "p2_stock": [4.0] * 3, "p1_pct": [0.0] * 3, "p2_pct": [0.0] * 3}
    matches = [_vm()]
    boots = [[_traj([-1, 0, 1], **flat)]]

    def fake_drive(*_args, **_kwargs):
        return matches, boots

    monkeypatch.setattr("hal.eval.cross_stage._drive_prior", fake_drive)
    result, rows = sweep_vs_cpu_prior_with_rows(
        lambda: None,
        session_cfg=None,
        n_matchups=1,
        max_parallel=1,
    )

    assert len(result) == len(rows) == 1
    assert result[0][1] == rows[0].boot_index == 0
    assert result[0][2] is not None and result[0][2].frames == rows[0].total_frames == 3


# --------------------------------------------------------------- paired comparison


def _paired_runs(
    per_boot_dmg: list[tuple[float, float]],
) -> tuple[list[MatchRow], list[MatchRow]]:
    """One match per boot; run A / B damage_dealt from ``per_boot_dmg`` (active = 1min
    so per-match rate == damage). Identical FOX/FALCO matchup on every boot."""
    rows_a, rows_b = [], []
    for boot_index, (da, db) in enumerate(per_boot_dmg):
        rows_a.append(_row(boot_index=boot_index, match_ordinal=0, damage_dealt=da))
        rows_b.append(_row(boot_index=boot_index, match_ordinal=0, damage_dealt=db))
    return rows_a, rows_b


def test_paired_delta_recovers_known_delta_and_excludes_zero() -> None:
    per_boot = [(50.0 + (i % 5), 28.0 + (i % 3)) for i in range(40)]
    rows_a, rows_b = _paired_runs(per_boot)
    expected = float(np.mean([da - db for da, db in per_boot]))

    out = paired_vs_cpu_deltas(rows_a, rows_b, bootstrap_resamples=1000, seed=0)

    assert out["n_pairs"] == 40.0
    assert out["pairing_rate"] == pytest.approx(1.0)
    assert out["damage_dealt_per_min_delta_mean"] == pytest.approx(expected)
    # A is uniformly ahead by >= 20 dmg/min, so the CI is entirely positive.
    assert out["damage_dealt_per_min_delta_ci_lo"] > 0.0
    assert out["damage_dealt_per_min_delta_ci_hi"] > 0.0


def test_paired_delta_null_case_includes_zero() -> None:
    rows_a, rows_b = _paired_runs([(40.0 + (i % 7), 40.0 + (i % 7)) for i in range(30)])
    out = paired_vs_cpu_deltas(rows_a, rows_b, bootstrap_resamples=500, seed=0)
    assert out["damage_dealt_per_min_delta_mean"] == pytest.approx(0.0)
    assert out["damage_dealt_per_min_delta_ci_lo"] <= 0.0 <= out["damage_dealt_per_min_delta_ci_hi"]


def test_paired_unequal_boot_counts_pair_by_shorter() -> None:
    # boot 0: A has 3 matches, B has 2 -> 2 pairs; boot 1: A has 2, B has 3 -> 2 pairs.
    rows_a = [_row(boot_index=0, match_ordinal=k, damage_dealt=10.0) for k in range(3)] + [
        _row(boot_index=1, match_ordinal=k, damage_dealt=10.0) for k in range(2)
    ]
    rows_b = [_row(boot_index=0, match_ordinal=k, damage_dealt=10.0) for k in range(2)] + [
        _row(boot_index=1, match_ordinal=k, damage_dealt=10.0) for k in range(3)
    ]
    out = paired_vs_cpu_deltas(rows_a, rows_b, bootstrap_resamples=100, seed=0)
    assert out["n_pairs"] == 4.0
    assert out["pairing_rate"] == pytest.approx(2 * 4 / (5 + 5))  # 0.8


def test_paired_raises_on_matchup_mismatch() -> None:
    rows_a = [_row(boot_index=0, match_ordinal=0, ego_character=int(Character.FOX.value))]
    rows_b = [
        _row(
            boot_index=0,
            match_ordinal=0,
            ego_character=int(Character.MARTH.value),
            opp_character=int(Character.SHEIK.value),
        )
    ]
    with pytest.raises(ValueError, match="matchup differs"):
        paired_vs_cpu_deltas(rows_a, rows_b)


def test_paired_raises_when_pairing_rate_below_half() -> None:
    rows_a = [_row(boot_index=i, match_ordinal=0) for i in range(4)]
    rows_b = [_row(boot_index=i, match_ordinal=0) for i in range(10, 14)]  # disjoint boots
    with pytest.raises(ValueError, match="pairing_rate"):
        paired_vs_cpu_deltas(rows_a, rows_b)


def test_paired_deltas_are_deterministic() -> None:
    rows_a, rows_b = _paired_runs([(50.0 + (i % 4), 30.0 + (i % 6)) for i in range(25)])
    assert paired_vs_cpu_deltas(rows_a, rows_b, seed=5) == paired_vs_cpu_deltas(rows_a, rows_b, seed=5)
