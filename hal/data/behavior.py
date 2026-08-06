"""Hand-engineered Melee behavior metrics for one replay.

``behavior_frames`` turns a peppi ``Game`` into rollback-deduplicated, in-game
per-frame columns; the free functions below reduce those columns to death
taxonomy, movement quality, opening counts and stage control. Everything is
derived from peppi post-frame columns and libmelee ``Action`` ids.

Layering: this is the hand-engineered stratum. Melee-specific action-state
knowledge lives here and nothing above it (scalable outcome metrics, paired
statistics, training objectives) imports it. It is also the single source of the
action-state vocabulary: ``hal.data.conversions`` imports the death table from
here and adds only the parity-scoped slippi-js ranges that port needs.

Geometry comes from ``melee.stages``. Every positioning metric uses
``EDGE_GROUND_POSITION`` (the walkable stage edge), never ``EDGE_POSITION`` (the
grabbable ledge, 2-3 units further out). The two differ enough to move
``frac_offstage``, so the choice is made once here and not re-decided per metric.

Metric denominators are ``active_mask`` frames — both players alive, past the
countdown, not in a respawn animation.
"""

import math
from dataclasses import dataclass
from typing import Any
from typing import Final

import melee
import numpy as np
from melee import Action
from melee import Character
from melee import Stage
from peppi_py.frame import Post
from peppi_py.game import Game

from hal.data.replay_stats import cumulative_damage
from hal.policy import INCLUDED_STAGES
from hal.wire import BUTTON_BITS
from hal.wire import GAME_START_FRAME
from hal.wire import TRIGGER_DEADZONE
from hal.wire import dedupe_keep_idx
from hal.wire import peppi_port_to_libmelee
from hal.wire import slp_stage_to_libmelee

FPS: Final[float] = 60.0
STARTING_STOCKS: Final[int] = 4

# ---------------------------------------------------------------------------
# Stage geometry. Re-read from libmelee here so a stage missing from the table
# fails loud instead of silently defaulting to a wrong edge.
# ---------------------------------------------------------------------------

EDGE_X: Final[dict[Stage, float]] = dict(melee.stages.EDGE_GROUND_POSITION)
BLASTZONES: Final[dict[Stage, tuple[float, float, float, float]]] = dict(melee.stages.BLASTZONES)

# ---------------------------------------------------------------------------
# Action-state sets. Values below ~340 are shared across the cast; 341+ are
# character-specific, so anything using them is gated on the character.
# ---------------------------------------------------------------------------

# Death states -> which blastzone was crossed. Star-KO / screen-KO variants are
# all top-blastzone outcomes. The keys are exactly action ids 0..10.
DEATH_SIDE: Final[dict[int, str]] = {
    Action.DEAD_DOWN.value: "bottom",
    Action.DEAD_LEFT.value: "left",
    Action.DEAD_RIGHT.value: "right",
    Action.DEAD_UP.value: "top",
    Action.DEAD_FLY_STAR.value: "top",
    Action.DEAD_FLY_STAR_ICE.value: "top",
    Action.DEAD_FLY.value: "top",
    Action.DEAD_FLY_SPLATTER.value: "top",
    Action.DEAD_FLY_SPLATTER_FLAT.value: "top",
    Action.DEAD_FLY_SPLATTER_ICE.value: "top",
    Action.DEAD_FLY_SPLATTER_FLAT_ICE.value: "top",
}
DEAD_ACTIONS: Final[frozenset[int]] = frozenset(DEATH_SIDE)

# Hitstun / grabbed / thrown: "the opponent has a hold on you". Used for openings
# and for the "was this player hit recently" leg of the SD classifier.
HITSTUN_ACTIONS: Final[frozenset[int]] = frozenset(
    list(range(Action.DAMAGE_HIGH_1.value, Action.DAMAGE_FLY_ROLL.value + 1))
    + [
        Action.DAMAGE_GROUND.value,
        Action.DAMAGE_SCREW.value,
        Action.DAMAGE_SCREW_AIR.value,
        Action.DAMAGE_SONG.value,
        Action.DAMAGE_SONG_WAIT.value,
        Action.DAMAGE_SONG_RV.value,
        Action.DAMAGE_BIND.value,
        Action.DAMAGE_ICE.value,
        Action.DAMAGE_ICE_JUMP.value,
        Action.SHIELD_BREAK_FLY.value,
        Action.SHIELD_BREAK_FALL.value,
        Action.SHIELD_BREAK_DOWN_U.value,
        Action.SHIELD_BREAK_DOWN_D.value,
        Action.SHIELD_BREAK_STAND_U.value,
        Action.SHIELD_BREAK_STAND_D.value,
        Action.TECH_MISS_UP.value,
        Action.TECH_MISS_DOWN.value,
    ]
)
GRABBED_ACTIONS: Final[frozenset[int]] = frozenset(
    [
        Action.GRABBED_WAIT_HIGH.value,
        Action.PUMMELED_HIGH.value,
        Action.GRABBED.value,
        Action.GRAB_PUMMELED.value,
        Action.THROWN_FORWARD.value,
        Action.THROWN_BACK.value,
        Action.THROWN_UP.value,
        Action.THROWN_DOWN.value,
        Action.THROWN_DOWN_2.value,
        Action.THROWN_FF.value,
        Action.THROWN_FB.value,
        Action.THROWN_F_HIGH.value,
        Action.THROWN_F_LOW.value,
    ]
)
PUNISHED_ACTIONS: Final[frozenset[int]] = HITSTUN_ACTIONS | GRABBED_ACTIONS

# Any offensive commitment; used to decide whether two back-to-back full hops
# were "for no reason".
ATTACK_ACTIONS: Final[frozenset[int]] = frozenset(
    list(range(Action.NEUTRAL_ATTACK_1.value, Action.DAIR_LANDING.value + 1))
    + list(range(Action.GRAB.value, Action.THROW_DOWN.value + 1))
    + list(range(Action.NEUTRAL_B_CHARGING.value, Action.KIRBY_STONE_FALLING.value + 1))
)

# Non-combat frames: respawn platform + entry animations. Excluded from
# positioning / idle denominators so a long death animation isn't "idle".
INACTIVE_ACTIONS: Final[frozenset[int]] = DEAD_ACTIONS | frozenset(
    [
        Action.ON_HALO_DESCENT.value,
        Action.ON_HALO_WAIT.value,
        Action.ENTRY.value,
        Action.ENTRY_START.value,
        Action.ENTRY_END.value,
        Action.NOTHING_STATE.value,
    ]
)

# Post-special "helpless" fall — the recovery resource is spent and the player
# can only drift. Dying in one is the unambiguous "died trying to recover"
# signature, independent of whether the opponent landed a chip hit on the way out.
HELPLESS_ACTIONS: Final[frozenset[int]] = frozenset(
    [Action.DEAD_FALL.value, Action.SPECIAL_FALL_FORWARD.value, Action.SPECIAL_FALL_BACK.value]
)

GROUND_JUMP_ACTIONS: Final[frozenset[int]] = frozenset([Action.JUMPING_FORWARD.value, Action.JUMPING_BACKWARD.value])

# Shield is up: the raise (GuardOn), the hold (Guard), shield stun (GuardSetOff)
# and a reflect (GuardReflect). SHIELD_RELEASE (GuardOff) is the drop animation —
# the shield is already gone — so it stays out, and a broken shield is hitstun.
SHIELD_ACTIONS: Final[frozenset[int]] = frozenset(
    [
        Action.SHIELD_START.value,
        Action.SHIELD.value,
        Action.SHIELD_STUN.value,
        Action.SHIELD_REFLECT.value,
    ]
)

# Fox/Falco shine (character-specific ids).
SHINE_ACTIONS: Final[frozenset[int]] = frozenset(
    [
        Action.DOWN_B_GROUND_START.value,
        Action.DOWN_B_GROUND.value,
        Action.SHINE_TURN.value,
        Action.DOWN_B_STUN.value,
        Action.DOWN_B_AIR.value,
        Action.SHINE_RELEASE_AIR.value,
    ]
)
SPACIES: Final[frozenset[int]] = frozenset([Character.FOX.value, Character.FALCO.value])

# Heuristic windows (frames).
DEATH_OFFSTAGE_LOOKBACK: Final[int] = 60
DEATH_HIT_LOOKBACK: Final[int] = 120
DEATH_HELPLESS_LOOKBACK: Final[int] = 30
DEATH_STATE_SCAN: Final[int] = 10  # decrement frame -> DEAD_* state onset
WAVEDASH_LAND_WINDOW: Final[int] = 10  # airdodge -> LANDING_SPECIAL
WAVEDASH_JUMP_WINDOW: Final[int] = 12  # KNEE_BEND -> airdodge (jump-initiated)
WAVESHINE_WINDOW: Final[int] = 20  # shine onset -> wavedash onset
DASH_DANCE_WINDOW: Final[int] = 25  # opposite-direction dash re-entry
IDLE_MIN_RUN: Final[int] = 30
OPENING_RESET_FRAMES: Final[int] = 60
DOUBLE_FULL_HOP_WINDOW: Final[int] = 120
LEDGE_PROXIMITY: Final[float] = 10.0
TAP_JUMP_Y: Final[float] = 0.6  # logical stick y that keeps a tap-jump "held"

_JUMP_BUTTON_BITS: Final[int] = BUTTON_BITS["x"] | BUTTON_BITS["y"]
_ACTION_BUTTON_BITS: Final[int] = (
    BUTTON_BITS["a"]
    | BUTTON_BITS["b"]
    | BUTTON_BITS["x"]
    | BUTTON_BITS["y"]
    | BUTTON_BITS["z"]
    | BUTTON_BITS["l"]
    | BUTTON_BITS["r"]
)


# ---------------------------------------------------------------------------
# Frame containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlayerBehaviorFrames:
    """One port's rollback-deduped in-game frame columns.

    Gamestate columns are the post-frame block; controller columns are the
    pre-frame block in HAL's logical representation (slp-logical sticks,
    per-shoulder triggers with ``wire.TRIGGER_DEADZONE`` applied). Unavailable
    values carry the per-dtype sentinel named in each comment so downstream
    gates exclude them instead of reading a plausible-looking zero.
    """

    port: int  # libmelee 1..4
    character: int  # libmelee INTERNAL Character value; -1 if unavailable
    is_cpu: bool
    cpu_level: int
    action: np.ndarray  # int32, -1 where masked
    x: np.ndarray  # float32, NaN where masked
    y: np.ndarray  # float32
    percent: np.ndarray  # float32
    stock: np.ndarray  # int32, -1 where masked
    direction: np.ndarray  # float32
    airborne: np.ndarray  # int8, -1 where the column is unavailable
    state_age: np.ndarray  # float32, NaN pre-2.0 (slippi-js actionStateCounter)
    last_attack_landed: np.ndarray  # int32, -1 where masked
    buttons: np.ndarray  # int32, slp physical bitmask
    main_stick_x: np.ndarray  # float32, slp-logical [-1, 1]
    main_stick_y: np.ndarray  # float32
    c_stick_x: np.ndarray  # float32
    c_stick_y: np.ndarray  # float32
    trigger_l: np.ndarray  # float32 [0, 1], sub-deadzone zeroed
    trigger_r: np.ndarray  # float32


@dataclass(frozen=True, slots=True)
class BehaviorFrames:
    """One 1v1 replay's deduped in-game frames plus its stage geometry."""

    stage: Stage
    edge_x: float  # melee.stages.EDGE_GROUND_POSITION
    blastzones: tuple[float, float, float, float]  # left, right, top, bottom
    frame_id: np.ndarray  # int64 slp frame ids, ascending, >= GAME_START_FRAME
    players: tuple[PlayerBehaviorFrames, PlayerBehaviorFrames]  # ascending port

    def opponent_of(self, p: PlayerBehaviorFrames) -> PlayerBehaviorFrames:
        a, b = self.players
        if p is a:
            return b
        if p is b:
            return a
        raise ValueError(f"port {p.port} is not one of {[q.port for q in self.players]}")


@dataclass(frozen=True, slots=True)
class Death:
    """One stock loss with its blastzone side and the context around it."""

    frame_id: int
    index: int  # row index into the player's columns
    side: str  # bottom / left / right / top / unknown
    x: float
    y: float
    percent: float
    offstage_before: bool
    opp_offstage: bool
    hit_recently: bool
    helpless: bool
    sd_like: bool

    @property
    def edgeguarded(self) -> bool:
        """Died off stage after being touched: the opponent's edgeguard landed."""
        return self.offstage_before and self.hit_recently


@dataclass(frozen=True, slots=True)
class Movement:
    wavedashes: int
    wavelands: int
    waveshines: int
    full_hops: int
    short_hops: int
    double_full_hops: int
    dash_dances: int
    ledge_grabs: int
    idle_frac: float
    shield_frac: float
    jump_rise_mean: float


@dataclass(frozen=True, slots=True)
class Openings:
    count: int
    damage_mean: float


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _col(arr: Any, keep: np.ndarray, dtype: Any, missing: Any) -> np.ndarray:
    """peppi column -> deduped numpy, with None scalars and absent columns
    replaced by ``missing`` (so downstream gates exclude them)."""
    if arr is None:
        return np.full(len(keep), missing, dtype=dtype)
    values = arr.to_pylist()
    out = np.array([v if v is not None else missing for v in values], dtype=dtype)
    return out[keep]


def _trigger(arr: Any, keep: np.ndarray) -> np.ndarray:
    """One shoulder's physical trigger with the game's deadzone zeroed, matching
    ``hal.data.extract`` so the stored value is game-causal."""
    out = _col(arr, keep, np.float32, np.nan)
    out[out < TRIGGER_DEADZONE] = 0.0
    return out


def _character(post: Post, keep: np.ndarray) -> int:
    """The player's live libmelee INTERNAL character on the first kept frame.

    Read per-frame rather than from the start block: ``post.character`` is
    transform-aware (Sheik/Zelda) and already internal, while the start block
    carries the external character-select id.
    """
    return int(_col(post.character, keep, np.int32, -1)[0])


def behavior_frames(g: Game) -> BehaviorFrames | None:
    """Deduped in-game frames for a 1v1 replay, or ``None`` if unusable.

    ``None`` covers per-replay conditions a sweep must skip: no frames, not
    singles, a stage id peppi/libmelee cannot name, a nameable stage HAL has no
    geometry for, or fewer than two in-game frames.

    It raises for the one case that is a table bug rather than a replay
    condition: a stage in ``policy.INCLUDED_STAGES`` with no
    ``EDGE_GROUND_POSITION`` entry. Silently skipping those would drop training
    stages from an eval sweep without a word.
    """
    if g.frames is None or g.frames.id is None or g.start.is_teams or len(g.start.players) != 2:
        return None

    ids = np.asarray(g.frames.id.to_pylist(), dtype=np.int64)
    keep = dedupe_keep_idx(ids)
    frame_id = ids[keep]
    in_game = frame_id >= GAME_START_FRAME
    keep, frame_id = keep[in_game], frame_id[in_game]
    if keep.size < 2:
        return None

    try:
        stage = slp_stage_to_libmelee(int(g.start.stage))
    except ValueError:
        return None
    if stage not in EDGE_X:
        if stage in INCLUDED_STAGES:
            raise ValueError(f"{stage.name} is an included stage but melee.stages has no EDGE_GROUND_POSITION entry")
        return None

    players: list[PlayerBehaviorFrames] = []
    for i, start in enumerate(g.start.players):
        leader = g.frames.ports[i].leader
        post, pre = leader.post, leader.pre
        players.append(
            PlayerBehaviorFrames(
                port=peppi_port_to_libmelee(start.port),
                character=_character(post, keep),
                is_cpu=int(start.type) == 1,
                cpu_level=int(start.cpu_level) if start.cpu_level is not None else 0,
                action=_col(post.action, keep, np.int32, -1),
                x=_col(post.position.x, keep, np.float32, np.nan),
                y=_col(post.position.y, keep, np.float32, np.nan),
                percent=_col(post.percent, keep, np.float32, np.nan),
                stock=_col(post.stock, keep, np.int32, -1),
                direction=_col(post.direction, keep, np.float32, np.nan),
                airborne=_col(post.airborne, keep, np.int8, -1),
                state_age=_col(post.state_age, keep, np.float32, np.nan),
                last_attack_landed=_col(post.last_attack_landed, keep, np.int32, -1),
                buttons=_col(pre.buttons_physical, keep, np.int32, 0),
                main_stick_x=_col(pre.joystick.x, keep, np.float32, np.nan),
                main_stick_y=_col(pre.joystick.y, keep, np.float32, np.nan),
                c_stick_x=_col(pre.cstick.x, keep, np.float32, np.nan),
                c_stick_y=_col(pre.cstick.y, keep, np.float32, np.nan),
                trigger_l=_trigger(pre.triggers_physical.l if pre.triggers_physical is not None else None, keep),
                trigger_r=_trigger(pre.triggers_physical.r if pre.triggers_physical is not None else None, keep),
            )
        )
    players.sort(key=lambda p: p.port)
    return BehaviorFrames(
        stage=stage,
        edge_x=EDGE_X[stage],
        blastzones=BLASTZONES[stage],
        frame_id=frame_id,
        players=(players[0], players[1]),
    )


# ---------------------------------------------------------------------------
# Primitive frame masks
# ---------------------------------------------------------------------------


def in_set(action: np.ndarray, members: frozenset[int]) -> np.ndarray:
    return np.isin(action, np.fromiter(members, dtype=np.int32, count=len(members)))


def onsets(mask: np.ndarray) -> np.ndarray:
    """Indices where ``mask`` turns True (rising edges; index 0 counts)."""
    prev = np.concatenate(([False], mask[:-1]))
    return np.flatnonzero(mask & ~prev)


def any_in_window(mask: np.ndarray, i: int, lo: int, hi: int) -> bool:
    """``mask`` True anywhere in the half-open index window ``[i+lo, i+hi)``."""
    a, b = max(0, i + lo), min(len(mask), i + hi)
    return bool(a < b and mask[a:b].any())


def active_mask(p: PlayerBehaviorFrames, opp: PlayerBehaviorFrames, m: BehaviorFrames) -> np.ndarray:
    """Frames both players are alive and playing (post-countdown, not dead or
    respawning). This is the denominator for every rate and fraction."""
    return (
        (m.frame_id >= 0)
        & ~in_set(p.action, INACTIVE_ACTIONS)
        & ~in_set(opp.action, INACTIVE_ACTIONS)
        & np.isfinite(p.x)
        & np.isfinite(opp.x)
    )


def offstage_mask(p: PlayerBehaviorFrames, m: BehaviorFrames) -> np.ndarray:
    """Frames past the walkable stage edge. NaN positions read as on stage."""
    return np.abs(p.x) > m.edge_x


# ---------------------------------------------------------------------------
# Death taxonomy
# ---------------------------------------------------------------------------


def find_deaths(p: PlayerBehaviorFrames, opp: PlayerBehaviorFrames, m: BehaviorFrames) -> tuple[Death, ...]:
    """One record per stock loss.

    A death is a stock decrement between two valid stock readings; the blastzone
    side comes from the ``DEAD_*`` action state the player enters on (or within a
    few frames of) that decrement, which is Melee's own record of which boundary
    was crossed. ``x``/``y`` are read at the death frame — for a bottom death that
    is below the lower blastzone, which is the sanity check on this whole path.
    """
    prev, nxt = p.stock[:-1], p.stock[1:]
    death_idx = np.flatnonzero((prev >= 0) & (nxt >= 0) & (nxt < prev))
    if death_idx.size == 0:
        return ()

    dead = in_set(p.action, DEAD_ACTIONS)
    helpless = in_set(p.action, HELPLESS_ACTIONS)
    hit = in_set(p.action, PUNISHED_ACTIONS)
    took_damage = np.concatenate(([False], np.diff(p.percent) > 0))
    self_off = offstage_mask(p, m)
    # For "was the opponent standing on stage while this player died", a grounded
    # opponent counts as on stage even at |x| just past the teetering edge — the
    # CPU teeters at the ledge constantly and would otherwise read as off-stage.
    opp_off = offstage_mask(opp, m) & (opp.airborne != 0)

    out: list[Death] = []
    for i in death_idx:
        # The DEAD_* state begins on the decrement frame; scan a short window for
        # robustness against per-build off-by-one in when the stock ticks down.
        j = next((k for k in range(int(i), min(int(i) + DEATH_STATE_SCAN, len(p.action))) if dead[k]), None)
        side = DEATH_SIDE.get(int(p.action[j]), "unknown") if j is not None else "unknown"
        at = int(j) if j is not None else int(i)
        offstage_before = any_in_window(self_off, at, -DEATH_OFFSTAGE_LOOKBACK, 0)
        hit_recently = any_in_window(hit, at, -DEATH_HIT_LOOKBACK, 0) or any_in_window(
            took_damage, at, -DEATH_HIT_LOOKBACK, 0
        )
        opp_offstage = bool(opp_off[at]) if np.isfinite(opp.x[at]) else False
        out.append(
            Death(
                frame_id=int(m.frame_id[at]),
                index=at,
                side=side,
                x=float(p.x[at]),
                y=float(p.y[at]),
                percent=float(p.percent[max(0, at - 1)]),
                offstage_before=offstage_before,
                opp_offstage=opp_offstage,
                hit_recently=hit_recently,
                helpless=any_in_window(helpless, at, -DEATH_HELPLESS_LOOKBACK, 1),
                # Failed recovery / self-destruct: left through a bottom or side
                # blastzone with the opponent standing on stage and nobody having
                # touched this player in the last two seconds.
                sd_like=(side in ("bottom", "left", "right")) and not hit_recently and not opp_offstage,
            )
        )
    return tuple(out)


def death_percent_mean(deaths: tuple[Death, ...]) -> float:
    """Mean percent at the moment of death; NaN with no deaths."""
    return float(np.mean([d.percent for d in deaths])) if deaths else math.nan


def death_y_mean(deaths: tuple[Death, ...]) -> float:
    """Mean height at the moment of death; NaN with no deaths."""
    return float(np.mean([d.y for d in deaths])) if deaths else math.nan


# ---------------------------------------------------------------------------
# Movement quality
# ---------------------------------------------------------------------------


def _wavedash_onsets(p: PlayerBehaviorFrames) -> tuple[list[int], list[int]]:
    """(jump-initiated, from-air) airdodge onsets that land in LANDING_SPECIAL."""
    airdodge = onsets(p.action == Action.AIRDODGE.value)
    landing_special = p.action == Action.LANDING_SPECIAL.value
    knee_bend = p.action == Action.KNEE_BEND.value
    jump_initiated: list[int] = []
    from_air: list[int] = []
    for i in airdodge:
        if not any_in_window(landing_special, int(i), 1, WAVEDASH_LAND_WINDOW + 1):
            continue
        if any_in_window(knee_bend, int(i), -WAVEDASH_JUMP_WINDOW, 0):
            jump_initiated.append(int(i))
        else:
            from_air.append(int(i))
    return jump_initiated, from_air


def _jump_events(p: PlayerBehaviorFrames) -> tuple[list[int], list[int], float]:
    """(full-hop takeoffs, short-hop takeoffs, mean apex rise).

    Melee decides short vs full hop by whether the jump input is still held on
    the last frame of jumpsquat, so that is read directly rather than inferred
    from height (which fastfalling and aerials corrupt). Apex rise is carried
    alongside purely as a diagnostic.
    """
    ground_jump = in_set(p.action, GROUND_JUMP_ACTIONS)
    takeoffs = [int(i) for i in onsets(ground_jump) if i > 0 and p.action[i - 1] == Action.KNEE_BEND.value]
    full: list[int] = []
    short: list[int] = []
    rises: list[float] = []
    airborne = p.airborne == 1
    for i in takeoffs:
        held = bool(p.buttons[i - 1] & _JUMP_BUTTON_BITS) or (
            np.isfinite(p.main_stick_y[i - 1]) and p.main_stick_y[i - 1] >= TAP_JUMP_Y
        )
        (full if held else short).append(i)
        end = i
        while end < len(p.y) - 1 and airborne[end + 1]:
            end += 1
        if end > i and np.isfinite(p.y[i]):
            rises.append(float(np.nanmax(p.y[i : end + 1]) - p.y[i]))
    return full, short, float(np.mean(rises)) if rises else math.nan


def _double_full_hops(p: PlayerBehaviorFrames, full: list[int]) -> int:
    """Full hops immediately followed by another with no attack in between."""
    attack = in_set(p.action, ATTACK_ACTIONS)
    return sum(
        1 for a, b in zip(full, full[1:], strict=False) if b - a <= DOUBLE_FULL_HOP_WINDOW and not attack[a:b].any()
    )


def _dash_dances(p: PlayerBehaviorFrames) -> int:
    dash = onsets(p.action == Action.DASHING.value)
    return sum(
        1
        for a, b in zip(dash, dash[1:], strict=False)
        if b - a <= DASH_DANCE_WINDOW and np.isfinite(p.direction[a]) and p.direction[a] != p.direction[b]
    )


def _idle_frac(p: PlayerBehaviorFrames, active: np.ndarray) -> float:
    """Fraction of active frames inside a >= ``IDLE_MIN_RUN`` run of standing
    still with a neutral stick and no buttons held."""
    neutral = (np.abs(np.nan_to_num(p.main_stick_x)) == 0) & (np.abs(np.nan_to_num(p.main_stick_y)) == 0)
    idle = (p.action == Action.STANDING.value) & neutral & ((p.buttons & _ACTION_BUTTON_BITS) == 0)
    denom = int(active.sum())
    if not idle.any() or not denom:
        return 0.0
    # Zero out runs shorter than the minimum.
    padded = np.concatenate(([False], idle, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    long_idle = np.zeros_like(idle)
    for s, e in zip(starts, ends, strict=True):
        if e - s >= IDLE_MIN_RUN:
            long_idle[s:e] = True
    return float((long_idle & active).sum() / denom)


def _shield_frac(p: PlayerBehaviorFrames, active: np.ndarray) -> float:
    """Fraction of active frames with the shield up.

    Every frame counts, with no minimum run: one long hold and many short ones
    are the same amount of time not spent playing. Passivity shows up here and
    in ``idle_frac`` together, which is why both share the ``active``
    denominator."""
    denom = int(active.sum())
    if not denom:
        return 0.0
    return float((in_set(p.action, SHIELD_ACTIONS) & active).sum() / denom)


def movement(p: PlayerBehaviorFrames, active: np.ndarray) -> Movement:
    """Movement-quality event counts over the whole replay, plus the idle and
    shield fractions (the rates whose denominator is ``active``)."""
    jump_wd, air_wd = _wavedash_onsets(p)
    full, short, rise = _jump_events(p)
    waveshines = 0
    if p.character in SPACIES:
        shine = onsets(in_set(p.action, SHINE_ACTIONS))
        wd = np.array(sorted(jump_wd + air_wd), dtype=np.int64)
        for s in shine:
            if wd.size and ((wd > s) & (wd <= s + WAVESHINE_WINDOW)).any():
                waveshines += 1
    return Movement(
        wavedashes=len(jump_wd),
        wavelands=len(air_wd),
        waveshines=waveshines,
        full_hops=len(full),
        short_hops=len(short),
        double_full_hops=_double_full_hops(p, full),
        dash_dances=_dash_dances(p),
        ledge_grabs=len(onsets(p.action == Action.EDGE_CATCHING.value)),
        idle_frac=_idle_frac(p, active),
        shield_frac=_shield_frac(p, active),
        jump_rise_mean=rise,
    )


# ---------------------------------------------------------------------------
# Openings
# ---------------------------------------------------------------------------


def openings(p: PlayerBehaviorFrames, opp: PlayerBehaviorFrames) -> Openings:
    """``p``'s openings on ``opp``: a 60-clean-frame reset heuristic.

    An opening starts when ``opp`` enters hitstun/grab after
    ``OPENING_RESET_FRAMES`` clean frames, and runs until ``opp`` has been clean
    that long again. Its damage is ``opp``'s percent gained over the window,
    stock-change frames excluded — the window starts ON the first punished
    frame, whose percent already carries the opening hit, so the tally is the
    FOLLOW-UP damage rather than the whole punish.

    This coexists with ``hal.data.conversions`` by design: that module answers
    "what would Slippi report" (45-frame actionability reset, move-level
    detail), this one answers "how often did this policy get in" with a
    denominator that does not depend on the opponent's action-state timing.
    """
    punished = in_set(opp.action, PUNISHED_ACTIONS)
    n = len(punished)
    count, total = 0, 0.0
    i = 0
    while i < n:
        if not punished[i]:
            i += 1
            continue
        if i > 0 and punished[max(0, i - OPENING_RESET_FRAMES) : i].any():
            i += 1
            continue
        count += 1
        end = i
        clean = 0
        while end < n - 1 and clean < OPENING_RESET_FRAMES:
            end += 1
            clean = 0 if punished[end] else clean + 1
        total += cumulative_damage(opp.percent[i : end + 1], opp.stock[i : end + 1])
        i = end + 1
    return Openings(count=count, damage_mean=total / count if count else 0.0)


# ---------------------------------------------------------------------------
# Stage control / positioning
# ---------------------------------------------------------------------------


def center_control_frac(p: PlayerBehaviorFrames, opp: PlayerBehaviorFrames, active: np.ndarray) -> float:
    """Fraction of active frames where ``p`` is nearer the stage center than
    ``opp``. Ties (and NaN positions) are excluded from the numerator but stay
    in the denominator, so the two players' fractions sum to at most 1."""
    denom = int(active.sum())
    if not denom:
        return math.nan
    return float((active & (np.abs(p.x) < np.abs(opp.x))).sum() / denom)


def mean_center_dist(p: PlayerBehaviorFrames, m: BehaviorFrames, active: np.ndarray) -> float:
    """Mean ``|x| / edge_x`` over active frames: 0 at center, 1 at the ledge,
    above 1 off stage. Stage-normalized so it compares across stages."""
    if not int(active.sum()):
        return math.nan
    return float(np.mean(np.abs(p.x[active]) / m.edge_x))


def frac_offstage(p: PlayerBehaviorFrames, m: BehaviorFrames, active: np.ndarray) -> float:
    """Fraction of active frames spent past the walkable stage edge."""
    denom = int(active.sum())
    if not denom:
        return math.nan
    return float((offstage_mask(p, m) & active).sum() / denom)


def mean_abs_x_minus_edge(p: PlayerBehaviorFrames, m: BehaviorFrames, active: np.ndarray) -> float:
    """Mean ``|x| - edge_x`` over active frames: negative on stage, positive off.
    The un-normalized sibling of ``mean_center_dist``."""
    if not int(active.sum()):
        return math.nan
    return float(np.mean(np.abs(p.x[active]) - m.edge_x))


def frac_near_ledge_onstage(p: PlayerBehaviorFrames, m: BehaviorFrames, active: np.ndarray) -> float:
    """Fraction of active frames spent on stage but within ``LEDGE_PROXIMITY``
    of the edge — the ledge-camping / edgeguard-setup band."""
    denom = int(active.sum())
    if not denom:
        return math.nan
    onstage = active & ~offstage_mask(p, m)
    return float((onstage & ((m.edge_x - np.abs(p.x)) <= LEDGE_PROXIMITY)).sum() / denom)
