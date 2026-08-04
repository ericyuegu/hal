"""Per-match behavior rows: one row per player per replay.

This is the aggregation layer of the hand-engineered stratum. It reads a ``.slp``,
runs ``hal.data.behavior`` (death taxonomy, movement, openings, stage control) and
``hal.data.conversions`` (the slippi-js punish machine) over it, and reduces both
to one flat row per player that a CSV sweep or a paired comparison can consume.

Layering: this module imports the two heuristic libraries; nothing in the scalable
outcome stratum (``match_summary``, ``cross_stage``, ``paired``, ``h2h``) imports
this one. ``h2h.replay_display_names`` is read the other way for the policy labels
stamped into an eval replay, which is an eval-to-eval read inside one stratum.

Rates are per ACTIVE minute (``behavior.active_mask``: both players alive, past
the countdown, not respawning), so a match that ends early and one that reaches
the frame budget are directly comparable.

Two opening definitions ship side by side, and both are named after where they
come from: ``openings_per_min`` / ``opening_damage_mean`` are the 60-clean-frame
heuristic from ``behavior.openings``, while ``damage_per_opening``,
``openings_per_kill`` and the neutral / counter ratios are Slippi's own numbers
from ``conversions.overall_ratios``. The same holds for the death columns:
``deaths_bottom`` is HAL's taxonomy (the ``DEAD_*`` state within a short scan of
the stock decrement) and ``deaths_down`` is the slippi-js direction read exactly
on the decrement frame.
"""

import math
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import is_dataclass
from dataclasses import replace
from pathlib import Path

import melee
import numpy as np
import peppi_py
from loguru import logger
from peppi_py.game import Game

from hal.data.behavior import FPS
from hal.data.behavior import SPACIES
from hal.data.behavior import STARTING_STOCKS
from hal.data.behavior import BehaviorFrames
from hal.data.behavior import PlayerBehaviorFrames
from hal.data.behavior import active_mask
from hal.data.behavior import behavior_frames
from hal.data.behavior import center_control_frac
from hal.data.behavior import death_percent_mean
from hal.data.behavior import death_y_mean
from hal.data.behavior import find_deaths
from hal.data.behavior import frac_near_ledge_onstage
from hal.data.behavior import frac_offstage
from hal.data.behavior import mean_abs_x_minus_edge
from hal.data.behavior import mean_center_dist
from hal.data.behavior import movement
from hal.data.behavior import openings
from hal.data.conversions import Conversion
from hal.data.conversions import Ratio
from hal.data.conversions import StockLoss
from hal.data.conversions import compute_conversions
from hal.data.conversions import compute_stock_losses
from hal.data.conversions import is_sd
from hal.data.conversions import overall_ratios
from hal.data.replay_stats import cumulative_damage
from hal.data.slp_finalize import trim_to_last_frame
from hal.eval.h2h import replay_display_names

# ---------------------------------------------------------------------------
# Tolerant replay reading
# ---------------------------------------------------------------------------


def peppi_panicked(error: BaseException) -> bool:
    """True for peppi's arrow2 panic.

    pyo3 raises ``PanicException``, a ``BaseException`` whose module
    (``pyo3_runtime``) is synthetic and cannot be imported, so the class is
    identified by name. It escapes a plain ``except Exception``.
    """
    return type(error).__qualname__ == "PanicException"


def _read(path: Path) -> Game | None:
    """One peppi read; None on any failure that is not an interrupt."""
    try:
        return peppi_py.read_slippi(str(path), skip_frames=False)
    except BaseException as error:  # peppi panics are BaseExceptions; see peppi_panicked
        if not peppi_panicked(error) and not isinstance(error, Exception):
            raise
        logger.debug(f"peppi cannot read {path}: {type(error).__name__}: {error}")
        return None


def read_replay_tolerant(path: str | Path) -> Game | None:
    """Read a ``.slp``, repairing a mid-frame tear once; None if still unreadable.

    A match stopped at its frame budget ends between the two ports' post-frame
    events, on which peppi either panics (ragged port columns) or errors (a stream
    shorter than its own declared length). ``trim_to_last_frame`` cuts the torn
    frame off — IN PLACE, so the repair is paid once per file — and the read is
    tried again. A file that fails twice is reported as None with a warning; a
    sweep must skip it, not die on it.
    """
    replay = Path(path)
    game = _read(replay)
    if game is not None:
        return game
    if not trim_to_last_frame(replay):
        logger.warning(f"unreadable replay with no complete frame to salvage: {replay}")
        return None
    game = _read(replay)
    if game is None:
        logger.warning(f"unreadable replay even after trimming the torn frame: {replay}")
    return game


# ---------------------------------------------------------------------------
# Row value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Outcome:
    """Who won and by how much. ``stock_delta`` is this player's stocks left minus
    the opponent's (the same number as taken minus lost), so it is positive for the
    player who came out ahead."""

    frames_total: int
    frames_active: int
    minutes: float
    stocks_left: int
    stocks_lost: int
    stocks_taken: int
    stock_delta: int
    damage_dealt: float
    damage_taken: float
    damage_dealt_per_min: float
    damage_taken_per_min: float
    won: bool


@dataclass(frozen=True, slots=True)
class DeathTaxonomy:
    """How this player lost stocks.

    The first block is HAL's taxonomy (``behavior.find_deaths``). ``deaths_down``
    and ``sd_count`` are the slippi-parity pair: the death direction on the exact
    decrement frame, and the stocks no opponent conversion closed out.
    """

    deaths: int
    deaths_bottom: int
    deaths_side: int
    deaths_top: int
    deaths_unknown_side: int
    deaths_offstage_before: int
    deaths_edgeguarded: int
    deaths_helpless: int
    deaths_sd_like: int
    death_percent_mean: float
    death_y_mean: float
    deaths_down: int
    sd_count: int


@dataclass(frozen=True, slots=True)
class Positioning:
    """Where on the stage this player spent the match. Every value is over active
    frames; ``mean_center_dist`` is stage-normalized, the rest are not."""

    center_control_frac: float
    mean_center_dist: float
    frac_offstage: float
    mean_abs_x_minus_edge: float
    frac_near_ledge_onstage: float


@dataclass(frozen=True, slots=True)
class MovementRates:
    """Movement-quality events per active minute. ``waveshines_per_min`` is NaN
    for a character with no shine."""

    wavedashes_per_min: float
    wavelands_per_min: float
    waveshines_per_min: float
    full_hops_per_min: float
    short_hops_per_min: float
    double_full_hops_per_min: float
    dash_dances_per_min: float
    ledge_grabs_per_min: float
    idle_frac: float
    jump_rise_mean: float


@dataclass(frozen=True, slots=True)
class Neutral:
    """How often this player got in, by the 60-clean-frame opening heuristic."""

    openings: int
    openings_per_min: float
    opening_damage_mean: float


@dataclass(frozen=True, slots=True)
class ConversionStats:
    """Slippi's punish numbers for this player. A ratio with a zero denominator
    is NaN, never a silent 0."""

    conversions: int
    conversion_damage: float
    kills: int
    successful_conversion_ratio: float
    openings_per_kill: float
    damage_per_opening: float
    neutral_wins: int
    neutral_win_ratio: float
    counter_hits: int
    counter_hit_ratio: float


@dataclass(frozen=True, slots=True, kw_only=True)
class BehaviorRow:
    """One player's behavior in one match.

    ``model`` is the policy that drove the port. ``replay`` is empty until the row
    comes from ``analyze_replay``, which is the layer that knows the file.
    """

    replay: str = ""
    model: str
    opp_model: str
    port: int
    opp_port: int
    character: str
    opp_character: str
    is_cpu: bool
    cpu_level: int
    stage: str
    outcome: Outcome
    death_taxonomy: DeathTaxonomy
    positioning: Positioning
    movement: MovementRates
    neutral: Neutral
    conversion_stats: ConversionStats

    def as_flat_dict(self) -> dict[str, int | float | str]:
        """One flat level of descriptive keys: the row's own fields plus every
        field of its value objects, in declaration order.

        Raises on a duplicate key, so a rename that collides fails on the first
        row instead of silently dropping a column from the CSV.
        """
        out: dict[str, int | float | str] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            items = asdict(value).items() if is_dataclass(value) else [(f.name, value)]
            for key, item in items:
                if key in out:
                    raise ValueError(f"duplicate flat key {key!r} in BehaviorRow")
                out[key] = item
        return out


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------


def _per_minute(value: float, minutes: float) -> float:
    """A count as a rate, or NaN when the match had no active time."""
    return value / minutes if minutes > 0.0 else math.nan


def _or_nan(ratio: Ratio) -> float:
    return math.nan if ratio.ratio is None else ratio.ratio


def _final_stock(p: PlayerBehaviorFrames) -> int:
    """The last valid stock reading. Raises when the column is entirely masked:
    a replay with no stock at all cannot produce an outcome."""
    valid = p.stock[p.stock >= 0]
    if not valid.size:
        raise ValueError(f"port {p.port} has no valid stock reading")
    return int(valid[-1])


def _character_name(value: int) -> str:
    """libmelee INTERNAL character id as a name; anything else keeps its number."""
    try:
        return melee.Character(value).name
    except ValueError:
        return f"UNKNOWN_{value}"


def _label(p: PlayerBehaviorFrames, names: Mapping[int, str] | None) -> str:
    """The caller's name for this port, else what the replay itself says it was."""
    if names and p.port in names:
        return names[p.port]
    return "cpu" if p.is_cpu else f"port{p.port}"


def _outcome(p: PlayerBehaviorFrames, opp: PlayerBehaviorFrames, m: BehaviorFrames, active: np.ndarray) -> Outcome:
    frames_active = int(active.sum())
    minutes = frames_active / (FPS * 60.0)
    stocks_left, opp_stocks_left = _final_stock(p), _final_stock(opp)
    damage_taken = cumulative_damage(p.percent, p.stock)
    damage_dealt = cumulative_damage(opp.percent, opp.stock)
    return Outcome(
        frames_total=int(m.frame_id.size),
        frames_active=frames_active,
        minutes=minutes,
        stocks_left=stocks_left,
        stocks_lost=STARTING_STOCKS - stocks_left,
        stocks_taken=STARTING_STOCKS - opp_stocks_left,
        stock_delta=stocks_left - opp_stocks_left,
        damage_dealt=damage_dealt,
        damage_taken=damage_taken,
        damage_dealt_per_min=_per_minute(damage_dealt, minutes),
        damage_taken_per_min=_per_minute(damage_taken, minutes),
        won=stocks_left > opp_stocks_left,
    )


def _death_taxonomy(
    p: PlayerBehaviorFrames,
    opp: PlayerBehaviorFrames,
    m: BehaviorFrames,
    conversions: tuple[Conversion, ...],
    losses: tuple[StockLoss, ...],
) -> DeathTaxonomy:
    deaths = find_deaths(p, opp, m)
    mine = [loss for loss in losses if loss.port == p.port]
    return DeathTaxonomy(
        deaths=len(deaths),
        deaths_bottom=sum(d.side == "bottom" for d in deaths),
        deaths_side=sum(d.side in ("left", "right") for d in deaths),
        deaths_top=sum(d.side == "top" for d in deaths),
        deaths_unknown_side=sum(d.side == "unknown" for d in deaths),
        deaths_offstage_before=sum(d.offstage_before for d in deaths),
        deaths_edgeguarded=sum(d.edgeguarded for d in deaths),
        deaths_helpless=sum(d.helpless for d in deaths),
        deaths_sd_like=sum(d.sd_like for d in deaths),
        death_percent_mean=death_percent_mean(deaths),
        death_y_mean=death_y_mean(deaths),
        deaths_down=sum(loss.direction == "down" for loss in mine),
        sd_count=sum(is_sd(loss, conversions) for loss in mine),
    )


def _positioning(
    p: PlayerBehaviorFrames, opp: PlayerBehaviorFrames, m: BehaviorFrames, active: np.ndarray
) -> Positioning:
    return Positioning(
        center_control_frac=center_control_frac(p, opp, active),
        mean_center_dist=mean_center_dist(p, m, active),
        frac_offstage=frac_offstage(p, m, active),
        mean_abs_x_minus_edge=mean_abs_x_minus_edge(p, m, active),
        frac_near_ledge_onstage=frac_near_ledge_onstage(p, m, active),
    )


def _movement_rates(p: PlayerBehaviorFrames, active: np.ndarray, minutes: float) -> MovementRates:
    mv = movement(p, active)
    return MovementRates(
        wavedashes_per_min=_per_minute(mv.wavedashes, minutes),
        wavelands_per_min=_per_minute(mv.wavelands, minutes),
        waveshines_per_min=_per_minute(mv.waveshines, minutes) if p.character in SPACIES else math.nan,
        full_hops_per_min=_per_minute(mv.full_hops, minutes),
        short_hops_per_min=_per_minute(mv.short_hops, minutes),
        double_full_hops_per_min=_per_minute(mv.double_full_hops, minutes),
        dash_dances_per_min=_per_minute(mv.dash_dances, minutes),
        ledge_grabs_per_min=_per_minute(mv.ledge_grabs, minutes),
        idle_frac=mv.idle_frac,
        jump_rise_mean=mv.jump_rise_mean,
    )


def _conversion_stats(
    p: PlayerBehaviorFrames, opp: PlayerBehaviorFrames, conversions: tuple[Conversion, ...]
) -> ConversionStats:
    r = overall_ratios(conversions, port=p.port, opponent_port=opp.port)
    return ConversionStats(
        conversions=r.conversion_count,
        conversion_damage=r.total_damage,
        kills=r.kill_count,
        successful_conversion_ratio=_or_nan(r.successful_conversion_ratio),
        openings_per_kill=_or_nan(r.openings_per_kill),
        damage_per_opening=_or_nan(r.damage_per_opening),
        neutral_wins=int(r.neutral_win_ratio.count),
        neutral_win_ratio=_or_nan(r.neutral_win_ratio),
        counter_hits=int(r.counter_hit_ratio.count),
        counter_hit_ratio=_or_nan(r.counter_hit_ratio),
    )


def _row(
    m: BehaviorFrames,
    p: PlayerBehaviorFrames,
    opp: PlayerBehaviorFrames,
    conversions: tuple[Conversion, ...],
    losses: tuple[StockLoss, ...],
    names: Mapping[int, str] | None,
) -> BehaviorRow:
    active = active_mask(p, opp, m)
    outcome = _outcome(p, opp, m, active)
    op = openings(p, opp)
    return BehaviorRow(
        model=_label(p, names),
        opp_model=_label(opp, names),
        port=p.port,
        opp_port=opp.port,
        character=_character_name(p.character),
        opp_character=_character_name(opp.character),
        is_cpu=p.is_cpu,
        cpu_level=p.cpu_level,
        stage=m.stage.name,
        outcome=outcome,
        death_taxonomy=_death_taxonomy(p, opp, m, conversions, losses),
        positioning=_positioning(p, opp, m, active),
        movement=_movement_rates(p, active, outcome.minutes),
        neutral=Neutral(
            openings=op.count,
            openings_per_min=_per_minute(op.count, outcome.minutes),
            opening_damage_mean=op.damage_mean,
        ),
        conversion_stats=_conversion_stats(p, opp, conversions),
    )


def behavior_rows(m: BehaviorFrames, *, names: Mapping[int, str] | None = None) -> tuple[BehaviorRow, BehaviorRow]:
    """One row per player, in ascending port order.

    ``names`` maps a libmelee port (1..4) to the policy that drove it. A port with
    no entry is named from the replay itself: "cpu" for a CPU port, "portN"
    otherwise. The conversion machine runs once for the match and both rows read
    the same conversion list, so the two are consistent by construction.
    """
    conversions = compute_conversions(m)
    losses = compute_stock_losses(m)
    a, b = m.players
    return _row(m, a, b, conversions, losses, names), _row(m, b, a, conversions, losses, names)


def analyze_replay(
    path: str | Path,
    *,
    names: Mapping[int, str] | None = None,
    min_active_frames: int = 0,
) -> tuple[BehaviorRow, BehaviorRow] | None:
    """Both rows of one replay file, or None when it yields no usable match.

    None covers a replay peppi cannot read even after a trim, one that is not a
    usable 1v1 (``behavior_frames``), and one with fewer than
    ``min_active_frames`` live frames — a boot that died in the menus, where every
    rate would divide by about zero.

    Without ``names`` the port labels come from the replay's own game-start
    display names, which the head-to-head runner stamps with the policy that drove
    each port. A replay too old to carry that block simply has none.
    """
    replay = Path(path)
    game = read_replay_tolerant(replay)
    if game is None:
        return None
    m = behavior_frames(game)
    if m is None:
        return None
    a, b = m.players
    if int(active_mask(a, b, m).sum()) < min_active_frames:
        return None
    labels = dict(names) if names else _display_names(replay)
    first, second = behavior_rows(m, names=labels)
    return replace(first, replay=str(replay)), replace(second, replay=str(replay))


def _display_names(replay: Path) -> dict[int, str]:
    """Stamped policy labels, or none for a replay whose Game Start block predates
    the display-name field."""
    try:
        return replay_display_names(replay)
    except ValueError as error:
        logger.debug(f"no display names in {replay}: {error}")
        return {}
