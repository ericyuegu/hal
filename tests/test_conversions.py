"""Unit tests for hal.data.conversions.

Two layers: synthetic frames pinning each replicated slippi-js quirk, and a
parity suite that runs slippi-js's own consistency fixtures through our port and
checks the counts its ``test/conversion.test.ts`` asserts. The fixtures are
never copied into this repo; point ``HAL_SLIPPI_JS_SLP`` at a slippi-js checkout
(default ``~/src/slippi-js/slp``) to enable that layer.
"""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from melee import Action
from melee import Character
from melee import Stage

from hal.data.behavior import BLASTZONES
from hal.data.behavior import EDGE_X
from hal.data.behavior import BehaviorFrames
from hal.data.behavior import PlayerBehaviorFrames
from hal.data.behavior import behavior_frames
from hal.data.conversions import BARREL_WAIT
from hal.data.conversions import COMMAND_GRAB_RANGE1_END
from hal.data.conversions import COMMAND_GRAB_RANGE1_START
from hal.data.conversions import COMMAND_GRAB_RANGE2_END
from hal.data.conversions import COMMAND_GRAB_RANGE2_START
from hal.data.conversions import GRAB
from hal.data.conversions import GROUND_ATTACK_END
from hal.data.conversions import GROUND_ATTACK_START
from hal.data.conversions import PUNISH_RESET_FRAMES
from hal.data.conversions import Conversion
from hal.data.conversions import InputCounts
from hal.data.conversions import MoveLanded
from hal.data.conversions import Ratio
from hal.data.conversions import StockLoss
from hal.data.conversions import compute_conversions
from hal.data.conversions import compute_stock_losses
from hal.data.conversions import count_inputs
from hal.data.conversions import is_command_grabbed
from hal.data.conversions import is_damaged
from hal.data.conversions import is_grabbed
from hal.data.conversions import is_in_control
from hal.data.conversions import is_sd
from hal.data.conversions import overall_ratios
from hal.wire import BUTTON_BITS

WAIT: int = 0x0E  # grounded, actionable
FALL: int = 0x1D  # airborne, NOT actionable and NOT damaged
HITSTUN: int = Action.DAMAGE_HIGH_1.value  # 0x4b
JAB1: int = GROUND_ATTACK_START  # 0x2c
JAB2: int = GROUND_ATTACK_START + 1  # 0x2d


def _fill(values: Sequence[Any] | None, n: int, default: Any, dtype: Any) -> np.ndarray:
    return np.full(n, default, dtype=dtype) if values is None else np.asarray(values, dtype=dtype)


def make_player(
    port: int,
    action: Sequence[int],
    *,
    percent: Sequence[float] | None = None,
    stock: Sequence[int] | None = None,
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
    """One port of synthetic frames. Only the columns these machines read are
    parameterized; the rest default to a benign constant."""
    n = len(action)
    return PlayerBehaviorFrames(
        port=port,
        character=int(Character.FOX.value),
        is_cpu=False,
        cpu_level=0,
        action=np.asarray(action, dtype=np.int32),
        x=np.zeros(n, dtype=np.float32),
        y=np.zeros(n, dtype=np.float32),
        percent=_fill(percent, n, 0.0, np.float32),
        stock=_fill(stock, n, 4, np.int32),
        direction=np.ones(n, dtype=np.float32),
        airborne=np.zeros(n, dtype=np.int8),
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


def make_frames(p1: PlayerBehaviorFrames, p2: PlayerBehaviorFrames, *, first_frame: int = 0) -> BehaviorFrames:
    n = len(p1.action)
    return BehaviorFrames(
        stage=Stage.FINAL_DESTINATION,
        edge_x=EDGE_X[Stage.FINAL_DESTINATION],
        blastzones=BLASTZONES[Stage.FINAL_DESTINATION],
        frame_id=np.arange(first_frame, first_frame + n, dtype=np.int64),
        players=(p1, p2),
    )


def _mask(fn: Any, value: int) -> bool:
    return bool(fn(np.array([value], dtype=np.int32))[0])


# ---------------------------------------------------------------------------
# State predicates
# ---------------------------------------------------------------------------


def test_is_in_control_excludes_jab1_but_not_jab2() -> None:
    """slippi-js uses a STRICT lower bound on the ground-attack range, so jab1
    is the one ground attack that does not count as being in control."""
    assert _mask(is_in_control, JAB1) is False
    assert _mask(is_in_control, JAB2) is True
    assert _mask(is_in_control, GROUND_ATTACK_END) is True
    assert _mask(is_in_control, GROUND_ATTACK_END + 1) is False


def test_is_in_control_ranges() -> None:
    for state in (0x0E, 0x18, 0x27, 0x29, GRAB):
        assert _mask(is_in_control, state) is True
    for state in (0x0D, 0x19, 0x26, 0x2A, GRAB + 1):
        assert _mask(is_in_control, state) is False


def test_is_damaged_covers_hitstun_damage_fall_and_jab_resets() -> None:
    for state in (0x4B, 0x5B, 0x26, 0xB9, 0xC1):
        assert _mask(is_damaged, state) is True
    for state in (0x4A, 0x5C, 0x25, 0xB8, 0xC0):
        assert _mask(is_damaged, state) is False


def test_is_grabbed_range() -> None:
    assert _mask(is_grabbed, 0xDF) is True
    assert _mask(is_grabbed, 0xE8) is True
    assert _mask(is_grabbed, 0xDE) is False
    assert _mask(is_grabbed, 0xE9) is False


def test_is_command_grabbed_excludes_barrel_wait() -> None:
    for state in (
        COMMAND_GRAB_RANGE1_START,
        COMMAND_GRAB_RANGE1_END,
        COMMAND_GRAB_RANGE2_START,
        COMMAND_GRAB_RANGE2_END,
    ):
        assert _mask(is_command_grabbed, state) is True
    for state in (
        BARREL_WAIT,
        COMMAND_GRAB_RANGE1_START - 1,
        COMMAND_GRAB_RANGE1_END + 1,
        COMMAND_GRAB_RANGE2_END + 1,
    ):
        assert _mask(is_command_grabbed, state) is False


# ---------------------------------------------------------------------------
# Synthetic scenario helpers
# ---------------------------------------------------------------------------


def _punish(
    n: int,
    hits: list[tuple[int, float]],
    *,
    attacker_action: list[int] | None = None,
    attacker_state_age: list[float] | None = None,
    victim_stock: list[int] | None = None,
    victim_percent: list[float] | None = None,
) -> BehaviorFrames:
    """p1 hits p2. Each ``(index, damage)`` puts p2 in hitstun for one frame and
    steps their percent up on the same frame. p2 is actionable otherwise, so the
    45-frame reset counter starts on the frame after each hit."""
    victim_action = [WAIT] * n
    percent = [0.0] * n
    for at, damage in hits:
        victim_action[at] = HITSTUN
        for k in range(at, n):
            percent[k] += damage
    p1 = make_player(
        1,
        attacker_action if attacker_action is not None else [JAB2] * n,
        state_age=attacker_state_age,
        last_attack_landed=[3] * n,
    )
    p2 = make_player(
        2,
        victim_action,
        percent=victim_percent if victim_percent is not None else percent,
        stock=victim_stock,
    )
    return make_frames(p1, p2)


# ---------------------------------------------------------------------------
# Reset timer
# ---------------------------------------------------------------------------


def test_reset_counter_terminates_strictly_above_45_actionable_frames() -> None:
    """The hit is at index 5; the counter starts on the first actionable frame
    (6) and terminates only once it EXCEEDS PUNISH_RESET_FRAMES."""
    n = 200
    (conv,) = compute_conversions(_punish(n, [(5, 10.0)]))
    assert conv.start_frame == 5
    assert conv.end_frame == 6 + PUNISH_RESET_FRAMES


def test_conversion_stays_open_at_exactly_45_actionable_frames() -> None:
    last_open = 6 + PUNISH_RESET_FRAMES - 1
    (conv,) = compute_conversions(_punish(last_open + 1, [(5, 10.0)]))
    assert conv.end_frame is None
    assert conv.end_percent is None


def test_reset_counter_waits_for_the_victim_to_regain_control() -> None:
    """200 airborne (non-actionable, non-damaged) frames must not tick the
    counter; the conversion only ends once the victim can act again."""
    n = 400
    victim_action = [FALL] * n
    victim_action[5] = HITSTUN
    victim_action[300:] = [WAIT] * (n - 300)
    p1 = make_player(1, [JAB2] * n, last_attack_landed=[3] * n)
    p2 = make_player(2, victim_action, percent=[0.0] * 5 + [10.0] * (n - 5))
    (conv,) = compute_conversions(make_frames(p1, p2))
    assert conv.end_frame == 300 + PUNISH_RESET_FRAMES


def test_jab1_does_not_start_the_reset_counter() -> None:
    """The strict ground-attack bound is observable end-to-end: a victim who
    mashes jab1 forever keeps the conversion open."""
    n = 300
    victim_action = [JAB1] * n
    victim_action[5] = HITSTUN
    p1 = make_player(1, [JAB2] * n, last_attack_landed=[3] * n)
    p2 = make_player(2, victim_action, percent=[0.0] * 5 + [10.0] * (n - 5))
    (conv,) = compute_conversions(make_frames(p1, p2))
    assert conv.end_frame is None

    victim_action = [JAB2] * n
    victim_action[5] = HITSTUN
    p2 = make_player(2, victim_action, percent=[0.0] * 5 + [10.0] * (n - 5))
    (conv,) = compute_conversions(make_frames(p1, p2))
    assert conv.end_frame is not None


# ---------------------------------------------------------------------------
# Damage gate
# ---------------------------------------------------------------------------


def test_negative_percent_delta_still_lands_a_move() -> None:
    """JS gates on ``if (opntDamageTaken)`` — truthiness, not positivity — so a
    percent DROP inside hitstun opens a move with negative damage."""
    n = 100
    percent = [20.0] * n
    percent[5:] = [15.0] * (n - 5)
    m = _punish(n, [(5, 0.0)], victim_percent=percent)
    (conv,) = compute_conversions(m)
    assert len(conv.moves) == 1
    assert conv.moves[0].damage == pytest.approx(-5.0)


def test_zero_percent_delta_lands_no_move() -> None:
    m = _punish(100, [(5, 0.0)])
    (conv,) = compute_conversions(m)
    assert conv.moves == ()
    assert conv.start_frame == 5


def test_move_id_comes_from_the_attackers_last_attack_landed() -> None:
    (conv,) = compute_conversions(_punish(100, [(5, 10.0)]))
    assert [mv.move_id for mv in conv.moves] == [3]
    assert [mv.frame for mv in conv.moves] == [5]


# ---------------------------------------------------------------------------
# Move de-duplication
# ---------------------------------------------------------------------------


def test_multihit_move_counts_once() -> None:
    """Same attacker animation with a rising state_age: fox drill, one move."""
    n = 100
    m = _punish(n, [(5, 3.0), (6, 3.0)], attacker_state_age=[float(i) for i in range(n)])
    (conv,) = compute_conversions(m)
    assert len(conv.moves) == 1
    assert conv.moves[0].damage == pytest.approx(6.0)


def test_state_age_reset_splits_two_fast_hits_of_the_same_move() -> None:
    """Ganon's jab twice in a row: same action id, but the animation counter
    restarts, so slippi-js records two moves."""
    n = 100
    state_age = [float(i) for i in range(n)]
    state_age[6] = 0.0  # new animation on the second hit
    m = _punish(n, [(5, 3.0), (6, 3.0)], attacker_state_age=state_age)
    (conv,) = compute_conversions(m)
    assert len(conv.moves) == 2
    assert [mv.damage for mv in conv.moves] == [pytest.approx(3.0), pytest.approx(3.0)]


def test_missing_state_age_falls_back_to_the_action_id_test() -> None:
    """Pre-2.0 replays have no counter. JS compares ``undefined < undefined``
    (false) and we compare NaN (also false), so the two hits merge."""
    n = 100
    m = _punish(n, [(5, 3.0), (6, 3.0)], attacker_state_age=[float("nan")] * n)
    (conv,) = compute_conversions(m)
    assert len(conv.moves) == 1


def test_action_change_between_hits_splits_the_moves() -> None:
    n = 100
    attacker = [JAB2] * n
    attacker[5:] = [JAB2 + 1] * (n - 5)
    m = _punish(n, [(5, 3.0), (6, 3.0)], attacker_action=attacker)
    (conv,) = compute_conversions(m)
    assert len(conv.moves) == 2


# ---------------------------------------------------------------------------
# Kills and percents
# ---------------------------------------------------------------------------


def test_kill_terminates_and_reads_end_percent_from_the_previous_frame() -> None:
    n = 100
    percent = [0.0] * 5 + [120.0] * 5 + [0.0] * (n - 10)  # reset on the death frame
    stock = [4] * 10 + [3] * (n - 10)
    m = _punish(n, [(5, 0.0)], victim_percent=percent, victim_stock=stock)
    (conv,) = compute_conversions(m)
    assert conv.did_kill is True
    assert conv.end_frame == 10
    assert conv.end_percent == pytest.approx(120.0)


def test_current_percent_is_not_updated_on_the_stock_loss_frame() -> None:
    n = 100
    percent = [0.0] * 5 + [120.0] * 5 + [0.0] * (n - 10)
    stock = [4] * 10 + [3] * (n - 10)
    m = _punish(n, [(5, 0.0)], victim_percent=percent, victim_stock=stock)
    (conv,) = compute_conversions(m)
    # The post-respawn 0.0 on frame 10 must not overwrite the live percent.
    assert conv.current_percent == pytest.approx(120.0)


def test_start_percent_comes_from_the_frame_before_the_first_hit() -> None:
    (conv,) = compute_conversions(_punish(100, [(5, 10.0), (7, 5.0)]))
    assert conv.start_percent == pytest.approx(0.0)
    assert conv.current_percent == pytest.approx(15.0)


def test_masked_stock_transition_is_not_a_kill() -> None:
    n = 100
    stock = [4] * 10 + [-1] * (n - 10)
    m = _punish(n, [(5, 10.0)], victim_stock=stock)
    (conv,) = compute_conversions(m)
    assert conv.did_kill is False


# ---------------------------------------------------------------------------
# Opening classification
# ---------------------------------------------------------------------------


def _both_hit(n: int, p1_hit: int, p2_hit: int) -> Any:
    """Symmetric scenario: p1 hits p2 at ``p2_hit``, p2 hits p1 at ``p1_hit``."""

    def side(hit_at: int) -> tuple[list[int], list[float]]:
        action = [WAIT] * n
        action[hit_at] = HITSTUN
        percent = [0.0] * hit_at + [10.0] * (n - hit_at)
        return action, percent

    a1, pct1 = side(p1_hit)
    a2, pct2 = side(p2_hit)
    return make_frames(
        make_player(1, a1, percent=pct1, last_attack_landed=[3] * n),
        make_player(2, a2, percent=pct2, last_attack_landed=[3] * n),
    )


def test_shared_start_frame_is_a_trade() -> None:
    convs = compute_conversions(_both_hit(200, 5, 5))
    assert len(convs) == 2
    assert {c.opening for c in convs} == {"trade"}


def test_counter_attack_when_the_attacker_was_still_being_punished() -> None:
    # p2 opens on p1 at frame 5 (ends at 6+45=51); p1 opens on p2 at frame 20,
    # inside that window, so p1's conversion is a counter-attack.
    convs = compute_conversions(_both_hit(300, 5, 20))
    by_attacker = {c.attacker_port: c for c in convs}
    assert by_attacker[2].opening == "neutral-win"
    assert by_attacker[2].end_frame == 6 + PUNISH_RESET_FRAMES
    assert by_attacker[1].opening == "counter-attack"


def test_neutral_win_when_the_earlier_punish_already_ended() -> None:
    convs = compute_conversions(_both_hit(300, 5, 120))
    by_attacker = {c.attacker_port: c for c in convs}
    assert by_attacker[2].opening == "neutral-win"
    assert by_attacker[1].opening == "neutral-win"


def test_counter_attack_check_treats_end_frame_zero_as_falsy() -> None:
    """JS writes ``oppEndFrame && oppEndFrame > startFrame``. Frame 0 is a real
    slp frame but a falsy number, so a punish that ended exactly on frame 0
    never makes the next conversion a counter-attack."""
    n = 60
    first_frame = -20
    # p1 dies on frame 0 (row 20), closing p2's conversion started at frame -15.
    p1_action = [WAIT] * n
    p1_action[5] = HITSTUN  # frame -15
    p1_stock = [4] * 20 + [3] * (n - 20)  # decrement on row 20 == frame 0
    # p2 gets hit at frame -10 (row 10), after p2's own conversion started.
    p2_action = [WAIT] * n
    p2_action[10] = HITSTUN
    m = make_frames(
        make_player(1, p1_action, percent=[0.0] * 5 + [50.0] * (n - 5), stock=p1_stock, last_attack_landed=[3] * n),
        make_player(2, p2_action, percent=[0.0] * 10 + [10.0] * (n - 10), last_attack_landed=[3] * n),
        first_frame=first_frame,
    )
    convs = compute_conversions(m)
    by_attacker = {c.attacker_port: c for c in convs}
    assert by_attacker[2].end_frame == 0
    assert by_attacker[1].start_frame == -10
    assert by_attacker[1].opening == "neutral-win"


def test_move_less_conversion_classifies_as_a_counter_attack_against_itself() -> None:
    """A conversion with no move looks up its OWN victim slot, which it wrote a
    moment earlier — replicated so our counts match slippi-js's."""
    n = 200
    victim_action = [WAIT] * n
    victim_action[5:8] = [Action.GRABBED.value] * 3
    m = make_frames(
        make_player(1, [GRAB] * n, last_attack_landed=[3] * n),
        make_player(2, victim_action),
    )
    (conv,) = compute_conversions(m)
    assert conv.moves == ()
    assert conv.end_frame is not None
    assert conv.opening == "counter-attack"


# ---------------------------------------------------------------------------
# Stock losses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("action_id", "direction"),
    [(0, "down"), (1, "left"), (2, "right"), (3, "up"), (10, "up"), (11, "unknown")],
)
def test_stock_loss_direction(action_id: int, direction: str) -> None:
    p1 = make_player(1, [WAIT, WAIT, action_id, action_id], percent=[0.0, 77.0, 0.0, 0.0], stock=[4, 4, 3, 3])
    p2 = make_player(2, [WAIT] * 4)
    (loss,) = compute_stock_losses(make_frames(p1, p2))
    assert loss == StockLoss(port=1, frame=2, percent=pytest.approx(77.0), direction=direction)


def test_stock_losses_cover_both_players_in_frame_order() -> None:
    p1 = make_player(1, [WAIT] * 6 + [0] * 4, stock=[4] * 6 + [3] * 4)
    p2 = make_player(2, [WAIT] * 3 + [1] * 7, stock=[4] * 3 + [3] * 7)
    losses = compute_stock_losses(make_frames(p1, p2))
    assert [(loss.port, loss.frame, loss.direction) for loss in losses] == [(2, 3, "left"), (1, 6, "down")]


def test_masked_stock_transitions_are_not_losses() -> None:
    p1 = make_player(1, [WAIT] * 4, stock=[4, 4, -1, -1])
    p2 = make_player(2, [WAIT] * 4)
    assert compute_stock_losses(make_frames(p1, p2)) == ()


def _conversion(**kwargs: Any) -> Conversion:
    defaults: dict[str, Any] = dict(
        attacker_port=1,
        victim_port=2,
        start_frame=0,
        end_frame=10,
        start_percent=0.0,
        current_percent=10.0,
        end_percent=10.0,
        did_kill=False,
        moves=(),
        opening="neutral-win",
    )
    defaults.update(kwargs)
    return Conversion(**defaults)


def test_is_sd_when_no_conversion_killed_on_that_frame() -> None:
    loss = StockLoss(port=2, frame=10, percent=90.0, direction="down")
    assert is_sd(loss, []) is True
    assert is_sd(loss, [_conversion(did_kill=True, end_frame=9)]) is True
    assert is_sd(loss, [_conversion(did_kill=False, end_frame=10)]) is True
    # A kill on the loss frame against this player is the opponent's, not an SD.
    assert is_sd(loss, [_conversion(did_kill=True, end_frame=10)]) is False
    # A kill against the OTHER player on the same frame does not clear it.
    assert is_sd(loss, [_conversion(did_kill=True, end_frame=10, attacker_port=2, victim_port=1)]) is True


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _inputs(**cols: Any) -> InputCounts:
    n = len(next(iter(cols.values())))
    p = make_player(1, [WAIT] * n, **cols)
    return count_inputs(p, np.arange(0, n, dtype=np.int64))


def test_button_presses_count_rising_edges_only() -> None:
    a, b = BUTTON_BITS["a"], BUTTON_BITS["b"]
    counts = _inputs(buttons=[0, a, a, a | b, b, 0])
    assert counts.buttons == 2
    assert counts.total == 2


def test_button_mask_drops_start() -> None:
    """slippi-js masks with 0xfff, which excludes START (0x1000). This differs
    from ``replay_stats._count_input_edges``, which uses ``wire.BUTTON_BITS``."""
    assert _inputs(buttons=[0, BUTTON_BITS["start"]]).buttons == 0
    assert _inputs(buttons=[0, 0x0001]).buttons == 1  # d-pad left IS inside 0xfff


def test_joystick_transitions_into_the_dead_zone_do_not_count() -> None:
    # center -> E -> center -> NE: two counted transitions.
    counts = _inputs(main_stick_x=[0.0, 1.0, 0.0, 1.0], main_stick_y=[0.0, 0.0, 0.0, 1.0])
    assert counts.joystick == 2


def test_joystick_holding_one_region_counts_once() -> None:
    counts = _inputs(main_stick_x=[0.0, 1.0, 0.9, 0.5], main_stick_y=[0.0] * 4)
    assert counts.joystick == 1


def test_cstick_counted_separately() -> None:
    counts = _inputs(c_stick_x=[0.0, 1.0, -1.0], c_stick_y=[0.0, 0.0, 0.0])
    assert (counts.cstick, counts.joystick) == (2, 0)
    assert counts.total == 2


def test_trigger_edges_cross_the_threshold_upward() -> None:
    counts = _inputs(trigger_l=[0.0, 0.5, 0.5, 0.0, 0.31], trigger_r=[0.0, 0.0, 1.0, 1.0, 1.0])
    assert counts.triggers == 3


def test_inputs_are_not_counted_before_the_first_playable_frame() -> None:
    a = BUTTON_BITS["a"]
    p = make_player(1, [WAIT] * 6, buttons=[0, a, 0, a, 0, a])
    assert count_inputs(p, np.arange(-123, -117, dtype=np.int64)).buttons == 0
    # Edges land on frames -41, -39, -37; only the last two are playable.
    assert count_inputs(p, np.arange(-42, -36, dtype=np.int64)).buttons == 2
    assert count_inputs(p, np.arange(-40, -34, dtype=np.int64)).buttons == 3


def test_frame_gap_breaks_the_previous_frame_link() -> None:
    a = BUTTON_BITS["a"]
    p = make_player(1, [WAIT] * 3, buttons=[0, a, 0])
    assert count_inputs(p, np.array([0, 5, 6], dtype=np.int64)).buttons == 0


# ---------------------------------------------------------------------------
# Overall ratios
# ---------------------------------------------------------------------------


def _moves(count: int, damage: float) -> tuple[MoveLanded, ...]:
    return tuple(MoveLanded(frame=i, move_id=3, damage=damage) for i in range(count))


def test_ratio_is_none_when_the_total_is_zero() -> None:
    assert Ratio(count=0, total=0).ratio is None
    assert Ratio(count=3, total=4).ratio == pytest.approx(0.75)


def test_overall_ratios_counts_and_shares() -> None:
    convs = [
        _conversion(opening="neutral-win", moves=(), did_kill=False),  # move-less
        _conversion(opening="neutral-win", moves=_moves(2, 10.0), did_kill=True),
        _conversion(opening="counter-attack", moves=_moves(1, 4.0)),
        _conversion(attacker_port=2, victim_port=1, opening="neutral-win", moves=_moves(1, 7.0)),
    ]
    p1 = overall_ratios(convs, port=1, opponent_port=2)
    assert p1.conversion_count == 3
    assert p1.kill_count == 1
    assert p1.total_damage == pytest.approx(2 * 10.0 + 4.0)
    assert p1.successful_conversion_ratio == Ratio(count=1, total=3)
    assert p1.openings_per_kill == Ratio(count=3, total=1)
    assert p1.damage_per_opening.ratio == pytest.approx(24.0 / 3)
    # The move-less conversion is invisible to the opening ratios.
    assert p1.neutral_win_ratio == Ratio(count=1, total=2)
    assert p1.counter_hit_ratio == Ratio(count=1, total=1)

    p2 = overall_ratios(convs, port=2, opponent_port=1)
    assert p2.conversion_count == 1
    assert p2.kill_count == 0
    assert p2.openings_per_kill.ratio is None


# ---------------------------------------------------------------------------
# slippi-js fixture parity
# ---------------------------------------------------------------------------

SLIPPI_JS_SLP = Path(os.environ.get("HAL_SLIPPI_JS_SLP", "~/src/slippi-js/slp")).expanduser()
CONSISTENCY_DIR = SLIPPI_JS_SLP / "consistencyTest"

# slippi-js test/conversion.test.ts: `overall[0].conversionCount` for the lower
# port, plus the kill count that suite pins at 0 for every command-grab case.
PARITY_CASES: list[tuple[str, int]] = [
    ("PuffVFalcon-Sing.slp", 2),
    ("BowsVDK-SB-63.slp", 3),
    ("FalcVBows-5UB-67.slp", 3),
    ("GanonVDK-5UB-73.slp", 5),
    ("KirbyVDK-Neutral-17.slp", 3),
    ("YoshiVDK-Egg-13.slp", 2),
    ("MewTwoVDK-SB-42.slp", 1),
]


@pytest.mark.parametrize(("name", "expected"), PARITY_CASES)
def test_parity_conversion_counts(name: str, expected: int) -> None:
    import peppi_py

    path = CONSISTENCY_DIR / name
    if not path.exists():
        pytest.skip(f"slippi-js fixture missing at {path}; set HAL_SLIPPI_JS_SLP")
    m = behavior_frames(peppi_py.read_slippi(str(path), skip_frames=False))
    assert m is not None
    convs = compute_conversions(m)
    lower, upper = m.players
    overall = overall_ratios(convs, port=lower.port, opponent_port=upper.port)
    assert overall.conversion_count == expected, [
        (c.attacker_port, c.start_frame, c.end_frame, len(c.moves)) for c in convs
    ]
    assert overall.kill_count == 0
    # slippi-js asserts overall.totalDamage == the sum of that player's move
    # damage across every conversion; ours is that sum by construction, so the
    # assertion is a structural check on the port.
    assert overall.total_damage == pytest.approx(
        float(sum(mv.damage for c in convs if c.attacker_port == lower.port for mv in c.moves))
    )


@pytest.mark.parametrize(("name", "expected"), [("KirbyVMario-nB.slp", (1, 0)), ("DKVBows-nB.slp", (0, 0))])
def test_parity_neutral_b_edge_cases(name: str, expected: tuple[int, int]) -> None:
    """slippi-js pins these two so a windup animation cannot invent a punish."""
    import peppi_py

    path = CONSISTENCY_DIR / name
    if not path.exists():
        pytest.skip(f"slippi-js fixture missing at {path}; set HAL_SLIPPI_JS_SLP")
    m = behavior_frames(peppi_py.read_slippi(str(path), skip_frames=False))
    assert m is not None
    convs = compute_conversions(m)
    lower, upper = m.players
    got = (
        overall_ratios(convs, port=lower.port, opponent_port=upper.port).conversion_count,
        overall_ratios(convs, port=upper.port, opponent_port=lower.port).conversion_count,
    )
    assert got == expected


def test_parity_bowser_command_grab_damage() -> None:
    """slippi-js asserts Bowser dealt at least 63% in BowsVDK-SB-63."""
    import peppi_py

    path = CONSISTENCY_DIR / "BowsVDK-SB-63.slp"
    if not path.exists():
        pytest.skip(f"slippi-js fixture missing at {path}; set HAL_SLIPPI_JS_SLP")
    m = behavior_frames(peppi_py.read_slippi(str(path), skip_frames=False))
    assert m is not None
    convs = compute_conversions(m)
    lower, upper = m.players
    assert overall_ratios(convs, port=lower.port, opponent_port=upper.port).total_damage >= 63.0
