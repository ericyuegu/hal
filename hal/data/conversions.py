"""slippi-js-faithful conversion / stock-loss / input state machines.

Port of ``slippi-js`` ``src/common/stats/{conversions,stocks,inputs,overall}.ts``
so HAL's behavior rows speak the conversion vocabulary the Slippi community
publishes. Faithfulness is the contract: the quirks below are reproduced
deliberately and pinned by ``tests/test_conversions.py``, which checks the
conversion counts against slippi-js's own fixture expectations.

Input is a ``hal.data.behavior.BehaviorFrames`` — the same rollback-deduplicated
in-game frame stream every other metric reads. The action-state ranges here are
slippi-js's own numbering and stay quarantined in this module;
``hal.data.behavior`` owns HAL's action-state vocabulary and never inherits
these.

Replicated quirks
-----------------
- ``is_in_control`` treats the ground-attack range as ``(0x2c, 0x40]``: jab1
  (0x2c) is excluded by the strict lower bound.
- The damage gate is JS truthiness on the percent delta, so any non-zero delta
  opens or extends a move — negative deltas included.
- Move de-duplication keys on ``state_age`` (slippi-js ``actionStateCounter``).
  Replays before slp 2.0 record no counter; JS compares ``undefined <
  undefined`` (false) and we compare NaN (also false), so old files fall back to
  the action-id test alone.
- ``end_percent`` is read from the frame BEFORE termination, and
  ``current_percent`` is not updated on a stock-loss frame.
- ``opening`` classification is a post-pass in ascending start-frame order with
  per-victim ``last_end_frame`` bookkeeping. A conversion that landed no move
  looks up its OWN victim slot and so classifies as a counter-attack; the
  counter-attack test also keeps JS's falsy-zero check on the end frame.
- ``overall_ratios`` groups a conversion under its first move's attacker, so a
  move-less conversion counts toward ``conversion_count`` but drops out of the
  neutral-win / counter-hit ratios.

Damage semantics
----------------
Conversion damage is the sum of the gated percent deltas above and is
deliberately NOT ``hal.data.replay_stats.cumulative_damage`` (positive deltas
only, stock-change frames dropped). Both definitions ship: this one keeps our
numbers comparable to Slippi's, the other is HAL's own damage metric.

Deviation
---------
slippi-js stops processing at the first frame-id gap. We treat a gap as a
prev-frame boundary and keep going, because a truncated eval replay should
still yield stats for the frames it does have. A deduplicated stream from a
complete slp is contiguous, so the two agree except on damaged files.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Final
from typing import Literal

import numpy as np

from hal.data.behavior import DEATH_SIDE
from hal.data.behavior import BehaviorFrames
from hal.data.behavior import PlayerBehaviorFrames

# ---------------------------------------------------------------------------
# slippi-js action-state numbering (src/common/stats/common.ts, `State`)
# ---------------------------------------------------------------------------

DAMAGE_START: Final[int] = 0x4B
DAMAGE_END: Final[int] = 0x5B
DAMAGE_FALL: Final[int] = 0x26
JAB_RESET_UP: Final[int] = 0xB9
JAB_RESET_DOWN: Final[int] = 0xC1
CAPTURE_START: Final[int] = 0xDF
CAPTURE_END: Final[int] = 0xE8
GROUNDED_CONTROL_START: Final[int] = 0x0E
GROUNDED_CONTROL_END: Final[int] = 0x18
SQUAT_START: Final[int] = 0x27
SQUAT_END: Final[int] = 0x29
GROUND_ATTACK_START: Final[int] = 0x2C
GROUND_ATTACK_END: Final[int] = 0x40
GRAB: Final[int] = 0xD4
COMMAND_GRAB_RANGE1_START: Final[int] = 0x10A
COMMAND_GRAB_RANGE1_END: Final[int] = 0x130
COMMAND_GRAB_RANGE2_START: Final[int] = 0x147
COMMAND_GRAB_RANGE2_END: Final[int] = 0x152
BARREL_WAIT: Final[int] = 0x125

# slippi-js `Timers.PUNISH_RESET_FRAMES`. Strict: the conversion terminates once
# the counter EXCEEDS this, so 45 actionable frames still extend it.
PUNISH_RESET_FRAMES: Final[int] = 45

# slippi-js `Frames.FIRST_PLAYABLE`: inputs are not counted before this frame.
FIRST_PLAYABLE_FRAME: Final[int] = -39

# slippi-js input thresholds. The button mask drops START (0x1000) and every bit
# above it; this differs from `replay_stats._count_input_edges`, which masks with
# `wire.BUTTON_BITS` instead. Both are published definitions and both ship.
INPUT_BUTTON_MASK: Final[int] = 0xFFF
JOYSTICK_DEADZONE: Final[float] = 0.2875
TRIGGER_PRESS_THRESHOLD: Final[float] = 0.3

Opening = Literal["trade", "counter-attack", "neutral-win"]
StockDirection = Literal["down", "left", "right", "up", "unknown"]

# slippi-js reports a stock loss by death DIRECTION; `behavior.DEATH_SIDE` names
# the same action ids by stage SIDE. One id table, two published vocabularies.
_DIRECTION_BY_SIDE: Final[dict[str, StockDirection]] = {
    "bottom": "down",
    "left": "left",
    "right": "right",
    "top": "up",
}


# ---------------------------------------------------------------------------
# State predicates. Vectorized over an action column; the scalar ranges are the
# comment beside each so the port is checkable line-by-line against common.ts.
# ---------------------------------------------------------------------------


def is_damaged(action: np.ndarray) -> np.ndarray:
    """Hitstun (0x4b-0x5b), damage-fall (0x26), or either jab reset."""
    return (
        ((action >= DAMAGE_START) & (action <= DAMAGE_END))
        | (action == DAMAGE_FALL)
        | (action == JAB_RESET_UP)
        | (action == JAB_RESET_DOWN)
    )


def is_grabbed(action: np.ndarray) -> np.ndarray:
    """Held or thrown by a normal grab (0xdf-0xe8)."""
    return (action >= CAPTURE_START) & (action <= CAPTURE_END)


def is_command_grabbed(action: np.ndarray) -> np.ndarray:
    """Caught by a special-move grab, minus Barrel Wait (0x125)."""
    in_range = ((action >= COMMAND_GRAB_RANGE1_START) & (action <= COMMAND_GRAB_RANGE1_END)) | (
        (action >= COMMAND_GRAB_RANGE2_START) & (action <= COMMAND_GRAB_RANGE2_END)
    )
    return in_range & (action != BARREL_WAIT)


def is_in_control(action: np.ndarray) -> np.ndarray:
    """Actionable on the ground.

    Grounded control (0x0e-0x18), squat (0x27-0x29), ground attack and grab.
    The ground-attack lower bound is STRICT in slippi-js, so jab1 (0x2c) does
    not count as in control while jab2 and everything above it does. Replicated
    as-is; it shifts where the 45-frame reset counter starts after a jab.
    """
    ground = (action >= GROUNDED_CONTROL_START) & (action <= GROUNDED_CONTROL_END)
    squat = (action >= SQUAT_START) & (action <= SQUAT_END)
    ground_attack = (action > GROUND_ATTACK_START) & (action <= GROUND_ATTACK_END)
    return ground | squat | ground_attack | (action == GRAB)


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MoveLanded:
    frame: int  # slp frame id the move connected on
    move_id: int  # slp attack id; -1 when the replay records none
    damage: float  # summed percent delta over every hit of this move


@dataclass(frozen=True, slots=True)
class Conversion:
    """One punish: the victim stayed unactionable from ``start_frame`` until
    45 actionable frames passed (or they lost a stock).

    ``end_frame``/``end_percent`` are ``None`` for a conversion still open when
    the replay ends. slippi-js keeps those in its list too, and they count
    toward the conversion totals, so they are kept rather than dropped.
    """

    attacker_port: int  # libmelee 1..4
    victim_port: int
    start_frame: int  # slp frame id
    end_frame: int | None
    start_percent: float
    current_percent: float  # victim percent at the last non-stock-loss frame
    end_percent: float | None
    did_kill: bool
    moves: tuple[MoveLanded, ...]
    opening: Opening


@dataclass(frozen=True, slots=True)
class StockLoss:
    port: int  # libmelee 1..4
    frame: int  # slp frame id the stock counter decremented on
    percent: float  # percent on the frame BEFORE the decrement
    direction: StockDirection


@dataclass(frozen=True, slots=True)
class InputCounts:
    buttons: int
    joystick: int
    cstick: int
    triggers: int

    @property
    def total(self) -> int:
        return self.buttons + self.joystick + self.cstick + self.triggers


@dataclass(frozen=True, slots=True)
class Ratio:
    """A count over a total. ``ratio`` is ``None`` when the total is zero — an
    undefined rate is reported as undefined, never as a silent 0.0."""

    count: float
    total: float

    @property
    def ratio(self) -> float | None:
        return self.count / self.total if self.total else None


@dataclass(frozen=True, slots=True)
class OverallRatios:
    port: int
    conversion_count: int
    total_damage: float
    kill_count: int
    successful_conversion_ratio: Ratio
    openings_per_kill: Ratio
    damage_per_opening: Ratio
    neutral_win_ratio: Ratio
    counter_hit_ratio: Ratio


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _MoveBuilder:
    frame: int
    move_id: int
    damage: float = 0.0


@dataclass(slots=True)
class _ConversionBuilder:
    attacker_port: int
    victim_port: int
    start_frame: int
    start_percent: float
    current_percent: float
    end_frame: int | None = None
    end_percent: float | None = None
    did_kill: bool = False
    moves: list[_MoveBuilder] = field(default_factory=list)
    opening: Opening = "neutral-win"


@dataclass(slots=True)
class _PunishState:
    conversion: _ConversionBuilder | None = None
    move: _MoveBuilder | None = None
    reset_counter: int = 0
    last_hit_animation: int | None = None


@dataclass(frozen=True, slots=True)
class _Permutation:
    """One (attacker, victim) ordering plus the victim state masks it reads."""

    attacker: PlayerBehaviorFrames
    victim: PlayerBehaviorFrames
    damaged: np.ndarray
    grabbed: np.ndarray
    command_grabbed: np.ndarray
    in_control: np.ndarray


def _or_zero(value: float) -> float:
    """slippi-js reads every percent as ``percent ?? 0``; our masked percents
    are NaN, so the same fallback applies here."""
    return 0.0 if math.isnan(value) else float(value)


def _permutations(m: BehaviorFrames) -> tuple[_Permutation, ...]:
    ordered = (m.players, (m.players[1], m.players[0]))
    return tuple(
        _Permutation(
            attacker=attacker,
            victim=victim,
            damaged=is_damaged(victim.action),
            grabbed=is_grabbed(victim.action),
            command_grabbed=is_command_grabbed(victim.action),
            in_control=is_in_control(victim.action),
        )
        for attacker, victim in ordered
    )


def _has_prev_mask(frame_id: np.ndarray) -> np.ndarray:
    """True where the previous ROW is also the previous FRAME.

    slippi-js looks the previous frame up by number and gets ``undefined`` when
    it is missing; a gap in our deduplicated stream is the same condition.
    """
    has_prev = np.zeros(frame_id.size, dtype=bool)
    has_prev[1:] = frame_id[1:] == frame_id[:-1] + 1
    return has_prev


def _step(
    perm: _Permutation,
    st: _PunishState,
    i: int,
    frame: int,
    has_prev: bool,
    out: list[_ConversionBuilder],
) -> None:
    """One frame of ``handleConversionCompute`` for one attacker/victim pair."""
    attacker, victim = perm.attacker, perm.victim
    opnt_stunned = bool(perm.damaged[i] or perm.grabbed[i] or perm.command_grabbed[i])
    # No previous frame means no delta at all, not a delta against zero.
    damage_taken = _or_zero(victim.percent[i]) - _or_zero(victim.percent[i - 1]) if has_prev else 0.0

    # Track whether the attacker's animation changed since the last hit, so a
    # multi-hit move (fox drill) counts once but two fast jabs count twice.
    # `state_age` is NaN on pre-2.0 replays, and NaN comparisons are false —
    # exactly what `undefined < undefined` does in slippi-js.
    action_changed = st.last_hit_animation is None or int(attacker.action[i]) != st.last_hit_animation
    prev_state_age = float(attacker.state_age[i - 1]) if has_prev else 0.0
    counter_reset = bool(float(attacker.state_age[i]) < prev_state_age)
    if action_changed or counter_reset:
        st.last_hit_animation = None

    if opnt_stunned:
        if st.conversion is None:
            st.conversion = _ConversionBuilder(
                attacker_port=attacker.port,
                victim_port=victim.port,
                start_frame=frame,
                start_percent=_or_zero(victim.percent[i - 1]) if has_prev else 0.0,
                current_percent=_or_zero(victim.percent[i]),
            )
            out.append(st.conversion)
        # JS truthiness on the delta: zero is skipped, a NEGATIVE delta is not.
        if damage_taken != 0.0:
            if st.last_hit_animation is None:
                st.move = _MoveBuilder(frame=frame, move_id=int(attacker.last_attack_landed[i]))
                st.conversion.moves.append(st.move)
            if st.move is not None:
                st.move.damage += damage_taken
            # The PREVIOUS frame's animation is the move that actually connected.
            st.last_hit_animation = int(attacker.action[i - 1]) if has_prev else None

    if st.conversion is None:
        return

    # A missing stock reading is `undefined` in slippi-js, and `undefined -
    # undefined > 0` is false; the -1 sentinel needs the validity gate to match.
    prev_stock = int(victim.stock[i - 1]) if has_prev else -1
    stock = int(victim.stock[i])
    lost_stock = prev_stock >= 0 and stock >= 0 and prev_stock - stock > 0
    if not lost_stock:
        st.conversion.current_percent = _or_zero(victim.percent[i])
    if opnt_stunned:
        st.reset_counter = 0
    if (st.reset_counter == 0 and bool(perm.in_control[i])) or st.reset_counter > 0:
        st.reset_counter += 1

    should_terminate = False
    if lost_stock:
        st.conversion.did_kill = True
        should_terminate = True
    if st.reset_counter > PUNISH_RESET_FRAMES:
        should_terminate = True
    if should_terminate:
        st.conversion.end_frame = frame
        st.conversion.end_percent = _or_zero(victim.percent[i - 1]) if has_prev else 0.0
        st.conversion = None
        st.move = None


def _classify_openings(out: list[_ConversionBuilder]) -> None:
    """slippi-js ``_populateConversionTypes`` — the opening post-pass."""
    groups: dict[int, list[_ConversionBuilder]] = {}
    for b in out:
        groups.setdefault(b.start_frame, []).append(b)
    last_end_by_victim: dict[int, int | None] = {}
    for start in sorted(groups):
        group = groups[start]
        is_trade = len(group) >= 2
        for b in group:
            last_end_by_victim[b.victim_port] = b.end_frame
            if is_trade:
                b.opening = "trade"
                continue
            # JS keys the lookup on the LAST move's attacker, falling back to the
            # conversion's own victim when no move landed — a move-less
            # conversion therefore reads back the end frame it just wrote and
            # classifies itself as a counter-attack.
            key = b.attacker_port if b.moves else b.victim_port
            opp_end = last_end_by_victim.get(key)
            # `oppEndFrame && oppEndFrame > startFrame`: frame 0 is falsy in JS.
            is_counter = opp_end is not None and opp_end != 0 and opp_end > b.start_frame
            b.opening = "counter-attack" if is_counter else "neutral-win"


def _freeze(b: _ConversionBuilder) -> Conversion:
    return Conversion(
        attacker_port=b.attacker_port,
        victim_port=b.victim_port,
        start_frame=b.start_frame,
        end_frame=b.end_frame,
        start_percent=b.start_percent,
        current_percent=b.current_percent,
        end_percent=b.end_percent,
        did_kill=b.did_kill,
        moves=tuple(MoveLanded(frame=mv.frame, move_id=mv.move_id, damage=mv.damage) for mv in b.moves),
        opening=b.opening,
    )


def compute_conversions(m: BehaviorFrames) -> tuple[Conversion, ...]:
    """Every conversion in the replay, both directions, in start order.

    Ties on ``start_frame`` (a trade) keep slippi-js's ordering: the
    lower-ported attacker first.
    """
    frame_id = m.frame_id
    has_prev = _has_prev_mask(frame_id)
    perms = _permutations(m)
    states = [_PunishState() for _ in perms]
    out: list[_ConversionBuilder] = []
    for i in range(int(frame_id.size)):
        frame = int(frame_id[i])
        for perm, st in zip(perms, states, strict=True):
            _step(perm, st, i, frame, bool(has_prev[i]), out)
    _classify_openings(out)
    return tuple(_freeze(b) for b in out)


# ---------------------------------------------------------------------------
# Stock losses
# ---------------------------------------------------------------------------


def compute_stock_losses(m: BehaviorFrames) -> tuple[StockLoss, ...]:
    """One record per stock-counter decrement, both players, in frame order.

    The percent is read from the frame BEFORE the decrement (the decrement frame
    already carries the post-respawn reset) and the direction from the death
    action state ON the decrement frame, which is slippi-js's ``deathAnimation``.
    ``behavior.find_deaths`` scans a short window for that state instead; this
    one stays on the exact frame for parity.
    """
    out: list[StockLoss] = []
    for p in m.players:
        prev, nxt = p.stock[:-1], p.stock[1:]
        for i in np.flatnonzero((prev >= 0) & (nxt >= 0) & (nxt < prev)):
            at = int(i) + 1
            side = DEATH_SIDE.get(int(p.action[at]))
            out.append(
                StockLoss(
                    port=p.port,
                    frame=int(m.frame_id[at]),
                    percent=_or_zero(p.percent[at - 1]),
                    direction=_DIRECTION_BY_SIDE[side] if side is not None else "unknown",
                )
            )
    out.sort(key=lambda s: (s.frame, s.port))
    return tuple(out)


def is_sd(loss: StockLoss, conversions: Sequence[Conversion]) -> bool:
    """True when no opponent conversion killed on this frame.

    The launcher heuristic: a stock the opponent did not close out is one the
    player threw away. It is deliberately blunt — a stock lost to a stray hit
    that never opened a conversion reads as an SD — so read it beside
    ``behavior.Death.sd_like``, which asks the same question from positions and
    hitstun instead of from the punish machine.
    """
    return not any(c.victim_port == loss.port and c.did_kill and c.end_frame == loss.frame for c in conversions)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _joystick_region(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """slippi-js ``getJoystickRegion``: 8 compass sectors plus a dead zone (0).

    Evaluated in slippi-js's if/else order, so the diagonals win over the
    cardinals. NaN fails every comparison and falls through to the dead zone.
    """
    t = JOYSTICK_DEADZONE
    return np.select(
        [
            (x >= t) & (y >= t),
            (x >= t) & (y <= -t),
            (x <= -t) & (y <= -t),
            (x <= -t) & (y >= t),
            y >= t,
            x >= t,
            y <= -t,
            x <= -t,
        ],
        [1, 2, 3, 4, 5, 6, 7, 8],
        default=0,
    )


def count_inputs(p: PlayerBehaviorFrames, frame_id: np.ndarray) -> InputCounts:
    """slippi-js ``InputComputer`` over one port's controller columns.

    Counted from ``FIRST_PLAYABLE_FRAME`` on: newly pressed buttons under
    ``INPUT_BUTTON_MASK``, stick and c-stick transitions into a new non-dead-zone
    region, and each trigger crossing ``TRIGGER_PRESS_THRESHOLD`` upward.

    NOT bit-parity with slippi-js, by construction: HAL stores the LOGICAL
    controller (see ``hal.wire``), so the trigger columns are
    already zeroed below ``wire.TRIGGER_DEADZONE`` (43/140 = 0.307). A trigger
    press that peaks in the 0.300-0.307 band therefore counts in slippi-js and
    not here. The stick columns are the same slp-logical values slippi-js reads,
    so those two legs do agree. Treat the total as HAL's variant of the
    definition — the parity tests cover conversions and stock losses only.
    """
    n = int(frame_id.size)
    if n < 2:
        return InputCounts(buttons=0, joystick=0, cstick=0, triggers=0)
    countable = _has_prev_mask(frame_id) & (frame_id >= FIRST_PLAYABLE_FRAME)

    buttons = p.buttons & INPUT_BUTTON_MASK
    pressed = np.zeros(n, dtype=np.int32)
    pressed[1:] = ~buttons[:-1] & buttons[1:] & INPUT_BUTTON_MASK
    bit_counts = np.unpackbits(pressed.astype(np.uint16).view(np.uint8).reshape(n, 2), axis=1).sum(axis=1)

    def _stick_edges(x: np.ndarray, y: np.ndarray) -> int:
        region = _joystick_region(x, y)
        changed = np.zeros(n, dtype=bool)
        changed[1:] = region[1:] != region[:-1]
        return int((changed & (region != 0) & countable).sum())

    def _trigger_edges(t: np.ndarray) -> int:
        crossed = np.zeros(n, dtype=bool)
        crossed[1:] = (t[:-1] < TRIGGER_PRESS_THRESHOLD) & (t[1:] >= TRIGGER_PRESS_THRESHOLD)
        return int((crossed & countable).sum())

    return InputCounts(
        buttons=int(bit_counts[countable].sum()),
        joystick=_stick_edges(p.main_stick_x, p.main_stick_y),
        cstick=_stick_edges(p.c_stick_x, p.c_stick_y),
        triggers=_trigger_edges(p.trigger_l) + _trigger_edges(p.trigger_r),
    )


# ---------------------------------------------------------------------------
# Overall ratios
# ---------------------------------------------------------------------------


def overall_ratios(conversions: Sequence[Conversion], *, port: int, opponent_port: int) -> OverallRatios:
    """slippi-js ``generateOverallStats`` for one port of a 1v1.

    ``conversion_count`` is every conversion whose victim is the opponent.
    ``neutral_win_ratio`` / ``counter_hit_ratio`` are this player's openings of
    that type over both players' openings of that type — and, per slippi-js,
    they only see conversions that landed at least one move.
    """
    mine = [c for c in conversions if c.victim_port != port]
    kill_count = sum(1 for c in mine if c.did_kill and c.attacker_port == port)
    successful = sum(1 for c in mine if len(c.moves) > 1 and c.attacker_port == port)
    total_damage = float(sum(mv.damage for c in mine for mv in c.moves))

    def _openings(attacker: int, opening: Opening) -> int:
        return sum(1 for c in conversions if c.moves and c.attacker_port == attacker and c.opening == opening)

    def _opening_ratio(opening: Opening) -> Ratio:
        ours, theirs = _openings(port, opening), _openings(opponent_port, opening)
        return Ratio(count=ours, total=ours + theirs)

    return OverallRatios(
        port=port,
        conversion_count=len(mine),
        total_damage=total_damage,
        kill_count=kill_count,
        successful_conversion_ratio=Ratio(count=successful, total=len(mine)),
        openings_per_kill=Ratio(count=len(mine), total=kill_count),
        damage_per_opening=Ratio(count=total_damage, total=len(mine)),
        neutral_win_ratio=_opening_ratio("neutral-win"),
        counter_hit_ratio=_opening_ratio("counter-attack"),
    )
