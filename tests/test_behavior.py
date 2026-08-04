"""Unit tests for hal.data.behavior — synthetic frame columns + fixture smoke."""

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from melee import Action
from melee import Character
from melee import Stage

from hal.data.behavior import BLASTZONES
from hal.data.behavior import DEATH_SIDE
from hal.data.behavior import EDGE_X
from hal.data.behavior import BehaviorFrames
from hal.data.behavior import PlayerBehaviorFrames
from hal.data.behavior import active_mask
from hal.data.behavior import behavior_frames
from hal.data.behavior import center_control_frac
from hal.data.behavior import death_percent_mean
from hal.data.behavior import find_deaths
from hal.data.behavior import frac_near_ledge_onstage
from hal.data.behavior import frac_offstage
from hal.data.behavior import mean_abs_x_minus_edge
from hal.data.behavior import mean_center_dist
from hal.data.behavior import movement
from hal.data.behavior import onsets
from hal.data.behavior import openings
from hal.paths import DEV_ARCHIVE_PATH
from hal.wire import BUTTON_BITS

FD_EDGE: float = EDGE_X[Stage.FINAL_DESTINATION]


def _fill(values: Sequence[Any] | None, n: int, default: Any, dtype: Any) -> np.ndarray:
    return np.full(n, default, dtype=dtype) if values is None else np.asarray(values, dtype=dtype)


def make_player(
    port: int,
    action: Sequence[int],
    *,
    character: int = int(Character.FOX.value),
    x: Sequence[float] | None = None,
    y: Sequence[float] | None = None,
    percent: Sequence[float] | None = None,
    stock: Sequence[int] | None = None,
    direction: Sequence[float] | None = None,
    airborne: Sequence[int] | None = None,
    state_age: Sequence[float] | None = None,
    last_attack_landed: Sequence[int] | None = None,
    buttons: Sequence[int] | None = None,
    main_stick_x: Sequence[float] | None = None,
    main_stick_y: Sequence[float] | None = None,
    c_stick_x: Sequence[float] | None = None,
    c_stick_y: Sequence[float] | None = None,
    trigger_l: Sequence[float] | None = None,
    trigger_r: Sequence[float] | None = None,
) -> PlayerBehaviorFrames:
    """One port of synthetic frames; every column defaults to a benign constant."""
    n = len(action)
    return PlayerBehaviorFrames(
        port=port,
        character=character,
        is_cpu=False,
        cpu_level=0,
        action=np.asarray(action, dtype=np.int32),
        x=_fill(x, n, 0.0, np.float32),
        y=_fill(y, n, 0.0, np.float32),
        percent=_fill(percent, n, 0.0, np.float32),
        stock=_fill(stock, n, 4, np.int32),
        direction=_fill(direction, n, 1.0, np.float32),
        airborne=_fill(airborne, n, 0, np.int8),
        state_age=_fill(state_age, n, 0.0, np.float32),
        last_attack_landed=_fill(last_attack_landed, n, -1, np.int32),
        buttons=_fill(buttons, n, 0, np.int32),
        main_stick_x=_fill(main_stick_x, n, 0.0, np.float32),
        main_stick_y=_fill(main_stick_y, n, 0.0, np.float32),
        c_stick_x=_fill(c_stick_x, n, 0.0, np.float32),
        c_stick_y=_fill(c_stick_y, n, 0.0, np.float32),
        trigger_l=_fill(trigger_l, n, 0.0, np.float32),
        trigger_r=_fill(trigger_r, n, 0.0, np.float32),
    )


def make_frames(
    p1: PlayerBehaviorFrames,
    p2: PlayerBehaviorFrames,
    *,
    stage: Stage = Stage.FINAL_DESTINATION,
    first_frame: int = 0,
) -> BehaviorFrames:
    n = len(p1.action)
    return BehaviorFrames(
        stage=stage,
        edge_x=EDGE_X[stage],
        blastzones=BLASTZONES[stage],
        frame_id=np.arange(first_frame, first_frame + n, dtype=np.int64),
        players=(p1, p2),
    )


def _standing(n: int) -> list[int]:
    return [Action.STANDING.value] * n


# ---------------------------------------------------------------------------
# Death taxonomy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("action_id", "side"), sorted(DEATH_SIDE.items()))
def test_death_side_mapping(action_id: int, side: str) -> None:
    # stocks 4,4,3,3 -> decrement between index 1 and 2; the DEAD_* state starts there.
    p = make_player(
        1,
        [Action.STANDING.value, Action.STANDING.value, action_id, action_id],
        percent=[0.0, 88.0, 0.0, 0.0],
        stock=[4, 4, 3, 3],
    )
    opp = make_player(2, _standing(4))
    deaths = find_deaths(p, opp, make_frames(p, opp))
    assert len(deaths) == 1
    assert deaths[0].side == side
    assert deaths[0].percent == pytest.approx(88.0)
    assert deaths[0].index == 2


def test_death_side_unknown_without_dead_state() -> None:
    """A stock decrement with no DEAD_* state inside the scan window is
    'unknown', not silently attributed to a blastzone."""
    p = make_player(1, _standing(20), stock=[4] * 10 + [3] * 10)
    opp = make_player(2, _standing(20))
    deaths = find_deaths(p, opp, make_frames(p, opp))
    assert [d.side for d in deaths] == ["unknown"]
    assert deaths[0].sd_like is False


def _death_case(
    *,
    side_action: int,
    hit: bool = False,
    opp_offstage: bool = False,
    self_offstage: bool = False,
) -> tuple[PlayerBehaviorFrames, PlayerBehaviorFrames, BehaviorFrames]:
    n = 200
    death_at = 150
    action = _standing(n)
    action[death_at:] = [side_action] * (n - death_at)
    if hit:
        # Hitstun 20 frames before the death, inside DEATH_HIT_LOOKBACK.
        action[death_at - 20 : death_at - 10] = [Action.DAMAGE_HIGH_1.value] * 10
    stock = [4] * death_at + [3] * (n - death_at)
    x = [FD_EDGE + 20.0 if self_offstage else 0.0] * n
    p = make_player(1, action, stock=stock, x=x)
    opp = make_player(
        2,
        _standing(n),
        x=[FD_EDGE + 30.0 if opp_offstage else 0.0] * n,
        airborne=[1 if opp_offstage else 0] * n,
    )
    return p, opp, make_frames(p, opp)


def test_sd_like_true_for_clean_bottom_death() -> None:
    p, opp, m = _death_case(side_action=Action.DEAD_DOWN.value)
    (death,) = find_deaths(p, opp, m)
    assert death.side == "bottom"
    assert death.sd_like is True
    assert death.hit_recently is False


def test_sd_like_false_for_top_death() -> None:
    """A star-KO is an opponent's kill by construction, never an SD."""
    p, opp, m = _death_case(side_action=Action.DEAD_FLY_STAR.value)
    (death,) = find_deaths(p, opp, m)
    assert death.side == "top"
    assert death.sd_like is False


def test_sd_like_false_when_hit_recently() -> None:
    p, opp, m = _death_case(side_action=Action.DEAD_LEFT.value, hit=True)
    (death,) = find_deaths(p, opp, m)
    assert death.hit_recently is True
    assert death.sd_like is False


def test_sd_like_false_when_opponent_offstage() -> None:
    """An airborne off-stage opponent was edgeguarding, so this is their kill."""
    p, opp, m = _death_case(side_action=Action.DEAD_DOWN.value, opp_offstage=True)
    (death,) = find_deaths(p, opp, m)
    assert death.opp_offstage is True
    assert death.sd_like is False


def test_edgeguarded_requires_offstage_and_a_hit() -> None:
    p, opp, m = _death_case(side_action=Action.DEAD_DOWN.value, hit=True, self_offstage=True)
    (death,) = find_deaths(p, opp, m)
    assert death.offstage_before is True
    assert death.hit_recently is True
    assert death.edgeguarded is True

    p, opp, m = _death_case(side_action=Action.DEAD_DOWN.value, self_offstage=True)
    (death,) = find_deaths(p, opp, m)
    assert death.offstage_before is True
    assert death.edgeguarded is False


def test_death_percent_mean_is_nan_without_deaths() -> None:
    assert math.isnan(death_percent_mean(()))


def test_masked_stock_transitions_are_not_deaths() -> None:
    """The -1 sentinel at a game-end trailer must not read as a stock loss."""
    p = make_player(1, _standing(4), stock=[4, 4, -1, -1])
    opp = make_player(2, _standing(4))
    assert find_deaths(p, opp, make_frames(p, opp)) == ()


# ---------------------------------------------------------------------------
# Active mask
# ---------------------------------------------------------------------------


def test_active_mask_excludes_pre_countdown_frames() -> None:
    p = make_player(1, _standing(10))
    opp = make_player(2, _standing(10))
    m = make_frames(p, opp, first_frame=-4)
    assert list(active_mask(p, opp, m)) == [False] * 4 + [True] * 6


def test_active_mask_excludes_either_player_respawning() -> None:
    p_action = _standing(10)
    p_action[2:4] = [Action.ON_HALO_WAIT.value] * 2
    opp_action = _standing(10)
    opp_action[7:9] = [Action.ENTRY.value] * 2
    p, opp = make_player(1, p_action), make_player(2, opp_action)
    active = active_mask(p, opp, make_frames(p, opp))
    assert int(active.sum()) == 6
    assert not active[2] and not active[3] and not active[7] and not active[8]


def test_active_mask_excludes_masked_positions() -> None:
    p = make_player(1, _standing(5), x=[0.0, np.nan, 0.0, 0.0, 0.0])
    opp = make_player(2, _standing(5), x=[0.0, 0.0, 0.0, np.nan, 0.0])
    assert int(active_mask(p, opp, make_frames(p, opp)).sum()) == 3


# ---------------------------------------------------------------------------
# Stage control / positioning
# ---------------------------------------------------------------------------


def test_center_control_and_ties() -> None:
    # 2 frames p1 closer to center, 2 ties, 1 frame p2 closer.
    p = make_player(1, _standing(5), x=[0.0, 10.0, 20.0, -20.0, 50.0])
    opp = make_player(2, _standing(5), x=[30.0, 40.0, 20.0, 20.0, 5.0])
    m = make_frames(p, opp)
    active = active_mask(p, opp, m)
    assert center_control_frac(p, opp, active) == pytest.approx(2 / 5)
    assert center_control_frac(opp, p, active) == pytest.approx(1 / 5)
    # Ties leave both numerators short, so the two shares sum below 1.
    assert center_control_frac(p, opp, active) + center_control_frac(opp, p, active) < 1.0


def test_mean_center_dist_is_stage_normalized() -> None:
    p = make_player(1, _standing(4), x=[0.0, FD_EDGE, -FD_EDGE, 2 * FD_EDGE])
    opp = make_player(2, _standing(4))
    m = make_frames(p, opp)
    assert mean_center_dist(p, m, active_mask(p, opp, m)) == pytest.approx(1.0, rel=1e-5)


def test_offstage_and_ledge_fractions() -> None:
    x = [0.0, FD_EDGE - 5.0, FD_EDGE + 1.0, -(FD_EDGE + 50.0)]
    p = make_player(1, _standing(4), x=x)
    opp = make_player(2, _standing(4))
    m = make_frames(p, opp)
    active = active_mask(p, opp, m)
    assert frac_offstage(p, m, active) == pytest.approx(0.5)
    assert frac_near_ledge_onstage(p, m, active) == pytest.approx(0.25)
    assert mean_abs_x_minus_edge(p, m, active) == pytest.approx(
        float(np.mean(np.abs(np.asarray(x, dtype=np.float32)) - FD_EDGE)), rel=1e-5
    )


def test_positioning_is_nan_without_active_frames() -> None:
    p = make_player(1, [Action.ENTRY.value] * 4)
    opp = make_player(2, _standing(4))
    m = make_frames(p, opp)
    active = active_mask(p, opp, m)
    assert int(active.sum()) == 0
    for value in (
        center_control_frac(p, opp, active),
        mean_center_dist(p, m, active),
        frac_offstage(p, m, active),
        mean_abs_x_minus_edge(p, m, active),
        frac_near_ledge_onstage(p, m, active),
    ):
        assert math.isnan(value)


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------


def test_onsets_counts_rising_edges_only() -> None:
    assert list(onsets(np.array([True, True, False, True, True]))) == [0, 3]


def test_wavedash_needs_a_jumpsquat_waveland_does_not() -> None:
    # index 0-1 jumpsquat, 2 airdodge, 3 landing special -> wavedash.
    jump = [Action.KNEE_BEND.value] * 2 + [Action.AIRDODGE.value, Action.LANDING_SPECIAL.value] + _standing(6)
    p = make_player(1, jump)
    opp = make_player(2, _standing(len(jump)))
    mv = movement(p, active_mask(p, opp, make_frames(p, opp)))
    assert (mv.wavedashes, mv.wavelands) == (1, 0)

    air = [Action.FALLING.value] * 2 + [Action.AIRDODGE.value, Action.LANDING_SPECIAL.value] + _standing(6)
    p = make_player(1, air)
    mv = movement(p, active_mask(p, opp, make_frames(p, opp)))
    assert (mv.wavedashes, mv.wavelands) == (0, 1)


def test_airdodge_without_landing_special_is_neither() -> None:
    action = [Action.KNEE_BEND.value] * 2 + [Action.AIRDODGE.value] + [Action.FALLING.value] * 20
    p = make_player(1, action)
    opp = make_player(2, _standing(len(action)))
    mv = movement(p, active_mask(p, opp, make_frames(p, opp)))
    assert (mv.wavedashes, mv.wavelands) == (0, 0)


def test_full_vs_short_hop_reads_the_held_jump_input() -> None:
    action = [Action.KNEE_BEND.value] * 3 + [Action.JUMPING_FORWARD.value] + [Action.FALLING.value] * 6
    n = len(action)
    held = [0] * n
    held[2] = BUTTON_BITS["x"]  # still held on the last jumpsquat frame
    p = make_player(1, action, buttons=held)
    opp = make_player(2, _standing(n))
    mv = movement(p, active_mask(p, opp, make_frames(p, opp)))
    assert (mv.full_hops, mv.short_hops) == (1, 0)

    p = make_player(1, action)  # nothing held -> short hop
    mv = movement(p, active_mask(p, opp, make_frames(p, opp)))
    assert (mv.full_hops, mv.short_hops) == (0, 1)


def test_tap_jump_counts_as_held() -> None:
    action = [Action.KNEE_BEND.value] * 3 + [Action.JUMPING_FORWARD.value] + [Action.FALLING.value] * 6
    stick_y = [0.0] * len(action)
    stick_y[2] = 0.9
    p = make_player(1, action, main_stick_y=stick_y)
    opp = make_player(2, _standing(len(action)))
    mv = movement(p, active_mask(p, opp, make_frames(p, opp)))
    assert (mv.full_hops, mv.short_hops) == (1, 0)


def test_dash_dance_requires_a_direction_flip() -> None:
    action = [Action.DASHING.value] * 3 + _standing(2) + [Action.DASHING.value] * 3 + _standing(2)
    n = len(action)
    p = make_player(1, action, direction=[1.0] * 5 + [-1.0] * (n - 5))
    opp = make_player(2, _standing(n))
    assert movement(p, active_mask(p, opp, make_frames(p, opp))).dash_dances == 1

    p = make_player(1, action)  # same direction both dashes
    assert movement(p, active_mask(p, opp, make_frames(p, opp))).dash_dances == 0


def test_ledge_grabs_count_onsets_not_frames() -> None:
    action = [Action.EDGE_CATCHING.value] * 5 + _standing(3) + [Action.EDGE_CATCHING.value] * 5
    p = make_player(1, action)
    opp = make_player(2, _standing(len(action)))
    assert movement(p, active_mask(p, opp, make_frames(p, opp))).ledge_grabs == 2


def test_idle_frac_ignores_runs_below_the_minimum() -> None:
    short_run = _standing(20) + [Action.DASHING.value] * 20
    p = make_player(1, short_run)
    opp = make_player(2, _standing(len(short_run)))
    assert movement(p, active_mask(p, opp, make_frames(p, opp))).idle_frac == 0.0

    long_run = _standing(40) + [Action.DASHING.value] * 20
    p = make_player(1, long_run)
    opp = make_player(2, _standing(len(long_run)))
    assert movement(p, active_mask(p, opp, make_frames(p, opp))).idle_frac == pytest.approx(40 / 60)


def test_idle_frac_needs_a_neutral_stick_and_no_buttons() -> None:
    action = _standing(60)
    p = make_player(1, action, main_stick_x=[0.5] * 60)
    opp = make_player(2, _standing(60))
    assert movement(p, active_mask(p, opp, make_frames(p, opp))).idle_frac == 0.0

    p = make_player(1, action, buttons=[BUTTON_BITS["a"]] * 60)
    assert movement(p, active_mask(p, opp, make_frames(p, opp))).idle_frac == 0.0


# ---------------------------------------------------------------------------
# Openings
# ---------------------------------------------------------------------------


def _victim(
    n: int, hits: Sequence[tuple[int, float]], *, state: int = Action.DAMAGE_HIGH_1.value
) -> PlayerBehaviorFrames:
    """A victim who enters ``state`` for 5 frames at each hit frame and whose
    percent steps up by that hit's damage on the same frame."""
    action = _standing(n)
    percent = [0.0] * n
    for start, damage in hits:
        action[start : start + 5] = [state] * 5
        for k in range(start, n):
            percent[k] += damage
    return make_player(2, action, percent=percent)


def test_openings_merges_hits_inside_the_reset_window() -> None:
    # Two hits 30 frames apart (inside the 60-frame reset) -> one opening.
    opp = _victim(300, [(50, 10.0), (80, 12.0)])
    result = openings(make_player(1, _standing(300)), opp)
    assert result.count == 1


def test_openings_damage_excludes_the_first_hits_delta() -> None:
    """Ported behavior, called out so it is not mistaken for a bug: the window
    starts ON the first punished frame, whose percent already includes the
    opening hit, so only the follow-up damage lands inside the tally."""
    opp = _victim(300, [(50, 10.0), (80, 12.0)])
    assert openings(make_player(1, _standing(300)), opp).damage_mean == pytest.approx(12.0)


def test_openings_splits_after_the_reset_window() -> None:
    opp = _victim(400, [(50, 10.0), (70, 8.0), (200, 12.0), (220, 15.0)], state=Action.GRABBED.value)
    result = openings(make_player(1, _standing(400)), opp)
    assert result.count == 2
    assert result.damage_mean == pytest.approx((8.0 + 15.0) / 2)


def test_openings_none() -> None:
    p, opp = make_player(1, _standing(50)), make_player(2, _standing(50))
    assert openings(p, opp) == type(openings(p, opp))(count=0, damage_mean=0.0)


# ---------------------------------------------------------------------------
# Fixture smoke
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not Path(DEV_ARCHIVE_PATH).exists(),
    reason=f"dev archive missing at {DEV_ARCHIVE_PATH}; run `python -m hal.scripts.fetch --name dev.7z`",
)
def test_behavior_frames_on_fixture(tmp_path: Path) -> None:
    """Smoke test against a real .slp from the dev archive."""
    import peppi_py
    import py7zr

    with py7zr.SevenZipFile(DEV_ARCHIVE_PATH, "r") as z:
        members = [m for m in z.getnames() if m.endswith(".slp")]
        assert members
        first = members[0]
        z.extract(path=tmp_path, targets=[first])
    g = peppi_py.read_slippi(str(tmp_path / first), skip_frames=False)

    m = behavior_frames(g)
    assert m is not None
    assert m.stage in EDGE_X
    assert [p.port for p in m.players] == [1, 2]
    assert m.frame_id[0] >= -123
    assert np.all(np.diff(m.frame_id) > 0)

    p, opp = m.players
    assert m.opponent_of(p) is opp
    active = active_mask(p, opp, m)
    assert 0 < int(active.sum()) <= len(m.frame_id)

    deaths = find_deaths(p, opp, m)
    assert len(deaths) == 4 - int(p.stock[-1])
    assert all(d.side in set(DEATH_SIDE.values()) | {"unknown"} for d in deaths)

    assert 0.0 <= center_control_frac(p, opp, active) <= 1.0
    assert 0.0 <= frac_offstage(p, m, active) <= 1.0
    assert 0.0 <= frac_near_ledge_onstage(p, m, active) <= 1.0
    assert mean_center_dist(p, m, active) >= 0.0

    mv = movement(p, active)
    assert 0.0 <= mv.idle_frac <= 1.0
    assert mv.full_hops >= 0 and mv.short_hops >= 0
    assert openings(p, opp).count >= 0
