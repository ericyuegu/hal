"""Sweep an experiment's policy across stages, in parallel.

Each sweep builds a grid of matches (``stages × replicas``) and runs them
concurrently through ``run_matches_vec`` — every live model-driven port across
all matches is fed to a single batched ``BatchPolicy`` call per frame. The
experiment passes in a ``policy_factory`` that builds a fresh ``BatchPolicy``
per wave (rolling-buffer state must reset between waves). Replicas of the same
stage diverge naturally via the policy's own per-step sampling.

Sweep flavors:

- ``sweep_vs_cpu`` — model on one port, in-game CPU on the other.
- ``sweep_self_play`` — both ports driven by the same batched policy.
- ``sweep_vs_cpu_prior`` — instant-restart vs-CPU over the training matchup prior.
- ``sweep_vs_cpu_prior_with_rows`` — the same prior sweep, retaining both pooled
  reduction input and exact per-match rows from one set of emulator trajectories.

Reductions and per-match views:

- ``vs_cpu_metrics`` — pool a ``SweepResult`` into per-active-minute rates with
  bootstrap CIs (the frozen active-frame protocol; see its docstring).
- ``match_rows`` — one ``MatchRow`` per completed
  match (characters, boot index, ordinal, active frames, damage, stocks), the
  source for matched-boot diagnostics.
- ``matched_vs_cpu_deltas`` — a character-matched delta that treats each boot as
  one cluster. It is not a common-random-numbers estimate.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from typing import Literal
from typing import cast

import melee
import numpy as np

from hal.eval.harness import DEFAULT_START_RETRIES
from hal.eval.harness import SessionConfig
from hal.eval.harness import run_matches_vec
from hal.eval.match_summary import MatchSummary
from hal.eval.match_summary import summarize_trajectory
from hal.eval.matchups import matchups_for_vs_cpu
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.trajectory import Trajectory
from hal.sim.vec import BatchPolicy
from hal.sim.vec import VecMatch

# (stage, replica index, summary-or-None-if-crashed) per match in the grid.
SweepResult = list[tuple[melee.Stage, int, MatchSummary | None]]

STARTING_STOCKS = 4  # Melee match default
FRAMES_PER_MINUTE = 3600  # 60 fps
# Frozen active-frame protocol: active frames are those with id >= 0. Canonical frame
# ids run -123..-1 before GO (hal/sim/trajectory.py; libmelee sets canonical id =
# gamestate.frame; hal/sim/vec.py segments a new match when the id resets). NOTE:
# players already control their characters from ~frame -39 (measured on ranked val
# replays: 79/80 player-sides show pre-0 input activity, some take pre-0 damage), so
# id >= 0 is a comparability CONVENTION that also trims those first controllable
# frames, not a claim that pre-0 frames are inert. Changing the cutoff would unfreeze
# the protocol.
PREGAME_FRAMES = 123

# Per-active-minute rate keys, in a fixed order shared by the reduction, the CI
# helpers, and the paired-delta helper. Order is (numerator source):
#   stocks_taken  = STARTING_STOCKS - opp_stocks_left
#   stocks_lost   = STARTING_STOCKS - ego_stocks_left
#   damage_dealt  = opp_damage_taken
#   damage_taken  = ego_damage_taken
_RATE_KEYS: tuple[str, ...] = (
    "stocks_taken_per_min",
    "stocks_lost_per_min",
    "damage_dealt_per_min",
    "damage_taken_per_min",
)
# Bootstrap resamples for the 95% CIs. Enough to stabilize the 2.5/97.5 percentiles
# at the ~32-96 boots a sweep schedules; overridable per call, seeded for
# determinism (byte-identical CIs across runs/boxes).
BOOTSTRAP_RESAMPLES = 2000

# Seed stage for the one menu navigation an instant-restart boot makes before the
# Gecko code takes over with random legal stages. Battlefield's cursor target sits
# near the menu origin, so it is the most reliable single nav under concurrent load.
PRIOR_SWEEP_SEED_STAGE = melee.Stage.BATTLEFIELD


def _active_frames(total_frames: int) -> int:
    """Active (post-GO, id >= 0) frame count for a match of ``total_frames``.

    A full match starts at the countdown (id -123), so the first ``PREGAME_FRAMES``
    frames are dead; a segment cut off inside the countdown (budget-cut partial)
    clamps to 0 active. Exact whenever capture begins at the countdown start, which
    ``start_match`` / instant-restart both do; ``match_rows`` counts ``frame_id >= 0``
    directly and does not rely on this."""
    return max(0, total_frames - PREGAME_FRAMES)


def _resample_index(rng: np.random.Generator, n: int, resamples: int) -> np.ndarray | None:
    """``(resamples, n)`` matrix of with-replacement row indices, or None if either
    dimension is empty (a degenerate bootstrap collapses to the point estimate)."""
    if resamples <= 0 or n <= 0:
        return None
    return rng.integers(0, n, size=(resamples, n))


def _ratio_ci(
    num: np.ndarray, active_minutes: np.ndarray, idx: np.ndarray | None, point: float
) -> tuple[float, float]:
    """95% bootstrap CI for the pooled ratio ``sum(num) / sum(active_minutes)``.

    Resamples matches (rows of ``idx``) and recomputes the frame-weighted pooled
    rate; a resample with zero active minutes is dropped as undefined. Falls back to
    ``(point, point)`` when there is nothing to resample."""
    if idx is None:
        return point, point
    numer = num[idx].sum(axis=1)
    denom = active_minutes[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = np.where(denom > 0.0, numer / denom, np.nan)
    if np.all(np.isnan(rates)):
        return point, point
    lo, hi = np.nanpercentile(rates, [2.5, 97.5])
    return float(lo), float(hi)


def _mean_ci(values: np.ndarray, idx: np.ndarray | None, point: float) -> tuple[float, float]:
    """95% bootstrap CI for the mean of ``values`` (resamples = rows of ``idx``)."""
    if idx is None or len(values) == 0:
        return point, point
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _summary_numerator(summary: MatchSummary, rate_key: str) -> float:
    return {
        "stocks_taken_per_min": float(STARTING_STOCKS - summary.p2_stocks_left),
        "stocks_lost_per_min": float(STARTING_STOCKS - summary.p1_stocks_left),
        "damage_dealt_per_min": summary.p2_damage_taken,
        "damage_taken_per_min": summary.p1_damage_taken,
    }[rate_key]


def vs_cpu_metrics(
    result: SweepResult,
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> dict[str, float]:
    """Reduce a ``sweep_vs_cpu`` grid to a flat metric dict for logging.

    Assumes the ego model is on port 1 vs a CPU on port 2 (the ``sweep_vs_cpu``
    default). Stocks/damage are reported as **per-active-minute rates**, pooled
    frame-weighted over every non-crashed match (``sum(metric) / sum(active_minutes)``)
    so the numbers are comparable across runs regardless of how many episodes ran or
    how long each lasted. ``crashed`` is the fraction of scheduled boots that produced no game.
    A countdown-only tail fragment is recorded as ``zero_active`` but is not a completed match and
    does not enter rates or confidence intervals.

    Protocol freeze (see ``PREGAME_FRAMES``): the denominator counts only **active**
    frames (canonical id >= 0). Each match's ``PREGAME_FRAMES`` pre-GO countdown frames
    are excluded, and a budget-cut partial that never reached GO contributes 0 active
    minutes. This intentionally **changes the rate values** versus the old total-frame
    denominator — rates are now per active minute, which removes the dead-frame bias
    that differed across eval styles (e.g. instant-restart's many short matches carry
    proportionally more countdown). ``dead_frame_frac`` reports the excluded fraction.

    Each rate also carries a seeded 95% cluster-bootstrap CI over boots
    (``<rate>_ci_lo`` / ``<rate>_ci_hi``); ``bootstrap_resamples <= 0`` collapses the
    CI to the point estimate.
    """
    boot_ids = {(stage, boot_index) for stage, boot_index, _ in result}
    completed_boot_ids = {(stage, boot_index) for stage, boot_index, summary in result if summary is not None}
    all_summaries = [s for _, _, s in result if s is not None]
    active_rows = [
        (stage, boot_index, s) for stage, boot_index, s in result if s is not None and _active_frames(s.frames) > 0
    ]
    summaries = [summary for _, _, summary in active_rows]
    zero_active = len(all_summaries) - len(summaries)
    crashed = len(boot_ids - completed_boot_ids) / len(boot_ids) if boot_ids else 1.0
    if not summaries:
        if not all_summaries:
            return {"crashed": crashed}
        return {"matches": 0.0, "zero_active": float(zero_active), "crashed": crashed}
    total_frames = sum(s.frames for s in summaries)

    total = np.array([s.frames for s in summaries], dtype=np.float64)
    active = np.array([_active_frames(s.frames) for s in summaries], dtype=np.float64)
    active_minutes = active / FRAMES_PER_MINUTE
    numerators = {
        key: np.array([_summary_numerator(summary, key) for summary in summaries], dtype=np.float64)
        for key in _RATE_KEYS
    }

    total_active_minutes = float(active_minutes.sum())
    by_boot: dict[tuple[melee.Stage, int], list[MatchSummary]] = {}
    for stage, boot_index, summary in active_rows:
        by_boot.setdefault((stage, boot_index), []).append(summary)
    boot_active_minutes = np.array(
        [sum(_active_frames(summary.frames) for summary in boot) / FRAMES_PER_MINUTE for boot in by_boot.values()],
        dtype=np.float64,
    )
    idx = _resample_index(np.random.default_rng(seed), len(by_boot), bootstrap_resamples)
    out: dict[str, float] = {}
    for key in _RATE_KEYS:
        point = float(numerators[key].sum() / total_active_minutes) if total_active_minutes > 0.0 else 0.0
        boot_numerators = np.array(
            [sum(_summary_numerator(summary, key) for summary in boot) for boot in by_boot.values()],
            dtype=np.float64,
        )
        lo, hi = _ratio_ci(boot_numerators, boot_active_minutes, idx, point)
        out[key] = point
        out[f"{key}_ci_lo"] = lo
        out[f"{key}_ci_hi"] = hi
    out["dead_frame_frac"] = float((total - active).sum() / total.sum())
    out["frames"] = total_frames / len(summaries)  # mean episode length (incl. countdown), a diagnostic
    out["matches"] = float(len(summaries))  # completed matches pooled (many per boot under instant-restart)
    out["boots"] = float(len(by_boot))
    out["zero_active"] = float(zero_active)
    out["crashed"] = crashed
    return out


@dataclass(frozen=True, slots=True)
class MatchRow:
    """One completed match, flat and self-describing for paired analysis.

    Characters/stage are libmelee-internal id **values** (the space the model trains
    on; see CLAUDE.md). ``stage`` is the boot's nominal/seed stage — under
    instant-restart the live per-match stage is randomized by the Gecko code and is
    NOT retained in the ``Trajectory``, so it cannot be recovered here. Damage/stocks
    are ego-relative (``dealt``/``taken`` = ego dealt-to / received-from the opponent;
    ``stocks_taken`` = opponent stocks the ego removed).
    """

    ego_character: int
    opp_character: int
    stage: int
    boot_index: int
    match_ordinal: int  # 0-based position within its boot's back-to-back matches
    active_frames: int  # frames with canonical id >= 0 (post-GO)
    total_frames: int
    damage_dealt: float
    damage_taken: float
    stocks_taken: int
    stocks_lost: int

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, int | float]) -> MatchRow:
        """Rebuild from a persisted flat dict (round-trips ``as_dict``); extra keys
        are ignored so logging can annotate rows without breaking the loader."""
        return cls(
            ego_character=cast(int, data["ego_character"]),
            opp_character=cast(int, data["opp_character"]),
            stage=cast(int, data["stage"]),
            boot_index=cast(int, data["boot_index"]),
            match_ordinal=cast(int, data["match_ordinal"]),
            active_frames=cast(int, data["active_frames"]),
            total_frames=cast(int, data["total_frames"]),
            damage_dealt=data["damage_dealt"],
            damage_taken=data["damage_taken"],
            stocks_taken=cast(int, data["stocks_taken"]),
            stocks_lost=cast(int, data["stocks_lost"]),
        )


def match_rows(
    boots: Sequence[Sequence[Trajectory]],
    matches: Sequence[VecMatch],
    *,
    ego_port: Literal[1, 2] = 1,
) -> list[MatchRow]:
    """One ``MatchRow`` per completed match from a vectorized sweep's raw output.

    ``boots[i]`` are the matches boot ``i`` played (from ``run_matches_vec`` /
    ``drive_vec``), aligned to ``matches[i]`` which supplies the boot's characters and
    seed stage. The match ordinal — implicit in the per-boot list order — is surfaced,
    and active frames are counted exactly from ``frame_id >= 0`` (no countdown
    constant). Ego is the model on ``ego_port``; the opponent is the other of ports
    {1, 2} (the vs-CPU port convention ``summarize_trajectory`` reads)."""
    opp_port = 2 if ego_port == 1 else 1
    rows: list[MatchRow] = []
    for boot_index, (boot, vm) in enumerate(zip(boots, matches, strict=True)):
        characters = {p.port: int(p.character.value) for p in vm.matchup.players}
        stage = int(vm.matchup.stage.value)
        for ordinal, traj in enumerate(boot):
            summary = summarize_trajectory(traj)
            ego_stocks_left = summary.p1_stocks_left if ego_port == 1 else summary.p2_stocks_left
            opp_stocks_left = summary.p2_stocks_left if ego_port == 1 else summary.p1_stocks_left
            ego_damage_taken = summary.p1_damage_taken if ego_port == 1 else summary.p2_damage_taken
            opp_damage_taken = summary.p2_damage_taken if ego_port == 1 else summary.p1_damage_taken
            rows.append(
                MatchRow(
                    ego_character=characters[ego_port],
                    opp_character=characters[opp_port],
                    stage=stage,
                    boot_index=boot_index,
                    match_ordinal=ordinal,
                    active_frames=int((traj.frame_id >= 0).sum()),
                    total_frames=len(traj),
                    damage_dealt=opp_damage_taken,
                    damage_taken=ego_damage_taken,
                    stocks_taken=STARTING_STOCKS - opp_stocks_left,
                    stocks_lost=STARTING_STOCKS - ego_stocks_left,
                )
            )
    return rows


def sweep_vs_cpu(
    policy_factory: Callable[[], BatchPolicy],
    *,
    session_cfg: SessionConfig,
    stages: Sequence[melee.Stage],
    max_parallel: int,
    replicas: int = 1,
    character: melee.Character = melee.Character.FOX,
    cpu_level: int = 9,
    ego_port: Literal[1, 2] = 1,
    max_frames: int = 15_000,
) -> SweepResult:
    """``replicas`` matches per stage, model on ``ego_port`` vs a level
    ``cpu_level`` CPU. All matches run concurrently in waves of ``max_parallel``."""
    cpu_port: Literal[1, 2] = 2 if ego_port == 1 else 1
    grid = [(stage, r) for stage in stages for r in range(replicas)]
    matches = [
        VecMatch(
            matchup=Matchup(
                stage=stage,
                players=(
                    PlayerSetup(port=ego_port, character=character, cpu_level=0),
                    PlayerSetup(port=cpu_port, character=character, cpu_level=cpu_level),
                ),
            ),
            model_ports=(ego_port,),
        )
        for stage, _ in grid
    ]
    boots = run_matches_vec(session_cfg, matches, policy_factory, max_frames=max_frames, max_parallel=max_parallel)
    # No instant-restart on this path → each boot is one match (or empty if it crashed).
    return [
        (stage, r, summarize_trajectory(boot[0]) if boot else None)
        for (stage, r), boot in zip(grid, boots, strict=True)
    ]


def sweep_self_play(
    policy_factory: Callable[[], BatchPolicy],
    *,
    session_cfg: SessionConfig,
    stages: Sequence[melee.Stage],
    max_parallel: int,
    replicas: int = 1,
    character: melee.Character = melee.Character.FOX,
    max_frames: int = 15_000,
) -> SweepResult:
    """``replicas`` matches per stage with both ports driven by the batched
    policy. All matches run concurrently in waves of ``max_parallel``."""
    grid = [(stage, r) for stage in stages for r in range(replicas)]
    matches = [
        VecMatch(
            matchup=Matchup(
                stage=stage,
                players=(
                    PlayerSetup(port=1, character=character, cpu_level=0),
                    PlayerSetup(port=2, character=character, cpu_level=0),
                ),
            ),
            model_ports=(1, 2),
        )
        for stage, _ in grid
    ]
    boots = run_matches_vec(session_cfg, matches, policy_factory, max_frames=max_frames, max_parallel=max_parallel)
    return [
        (stage, r, summarize_trajectory(boot[0]) if boot else None)
        for (stage, r), boot in zip(grid, boots, strict=True)
    ]


def _prior_vec_matches(
    n_matchups: int,
    *,
    cpu_level: int,
    ego_port: Literal[1, 2],
    seed_stage: melee.Stage,
) -> list[VecMatch]:
    """``n_matchups`` prior-drawn vs-CPU ``VecMatch`` boots.

    The schedule is the empirical matchup prior conditioned on a CPU-selectable
    opponent (CPU Sheik is impossible in local VS mode), and remains prefix-stable
    in ``n`` so boot ``i`` is the same matchup across runs.
    """
    cpu_port: Literal[1, 2] = 2 if ego_port == 1 else 1
    return [
        VecMatch(
            matchup=Matchup(
                stage=seed_stage,
                players=(
                    PlayerSetup(port=ego_port, character=ego_char, cpu_level=0),
                    PlayerSetup(port=cpu_port, character=opp_char, cpu_level=cpu_level),
                ),
            ),
            model_ports=(ego_port,),
        )
        for ego_char, opp_char in matchups_for_vs_cpu(n_matchups)
    ]


def _drive_prior(
    policy_factory: Callable[[], BatchPolicy],
    *,
    session_cfg: SessionConfig,
    n_matchups: int,
    max_parallel: int,
    cpu_level: int,
    ego_port: Literal[1, 2],
    seed_stage: melee.Stage,
    max_frames: int,
    start_retries: int,
) -> tuple[list[VecMatch], list[list[Trajectory]]]:
    """Drive the prior-distribution instant-restart sweep, returning the boot matches
    and their per-boot trajectory lists (aligned). Shared by the pooled-metric and
    per-row entry points so both see the identical schedule and boots."""
    matches = _prior_vec_matches(n_matchups, cpu_level=cpu_level, ego_port=ego_port, seed_stage=seed_stage)
    boots = run_matches_vec(
        session_cfg,
        matches,
        policy_factory,
        max_frames=max_frames,
        max_parallel=max_parallel,
        start_retries=start_retries,
    )
    return matches, boots


def sweep_vs_cpu_prior(
    policy_factory: Callable[[], BatchPolicy],
    *,
    session_cfg: SessionConfig,
    n_matchups: int,
    max_parallel: int,
    cpu_level: int = 9,
    ego_port: Literal[1, 2] = 1,
    seed_stage: melee.Stage = PRIOR_SWEEP_SEED_STAGE,
    max_frames: int = 15_000,
    start_retries: int = DEFAULT_START_RETRIES,
) -> SweepResult:
    """Prior-distribution vs-CPU sweep for instant-restart sessions.

    ``n_matchups`` deterministic ``(ego_char, opp_char)`` boots are drawn from the
    training matchup prior (``matchups_for``); each boots once to ``seed_stage`` and
    then — via the Gecko "Instant Match" code (``session_cfg.instant_match_restart``
    must be set) — plays many matches back-to-back on random legal stages within
    ``max_frames``. Every completed match becomes one ``SweepResult`` row (the
    ``stage`` label is the seed, kept only for shape compatibility with
    ``vs_cpu_metrics``); a boot that produced no match contributes one ``None`` row.
    Pool the rows with ``vs_cpu_metrics`` exactly as the fixed sweep."""
    matches, boots = _drive_prior(
        policy_factory,
        session_cfg=session_cfg,
        n_matchups=n_matchups,
        max_parallel=max_parallel,
        cpu_level=cpu_level,
        ego_port=ego_port,
        seed_stage=seed_stage,
        max_frames=max_frames,
        start_retries=start_retries,
    )
    return _prior_sweep_result(boots, seed_stage)


def _prior_sweep_result(boots: Sequence[Sequence[Trajectory]], seed_stage: melee.Stage) -> SweepResult:
    """Summarize already-driven prior boots without losing their boot indices."""
    out: SweepResult = []
    for bi, boot in enumerate(boots):
        if not boot:
            out.append((seed_stage, bi, None))  # boot never reached IN_GAME (hung/crashed)
        else:
            out.extend((seed_stage, bi, summarize_trajectory(t)) for t in boot)
    return out


def sweep_vs_cpu_prior_with_rows(
    policy_factory: Callable[[], BatchPolicy],
    *,
    session_cfg: SessionConfig,
    n_matchups: int,
    max_parallel: int,
    cpu_level: int = 9,
    ego_port: Literal[1, 2] = 1,
    seed_stage: melee.Stage = PRIOR_SWEEP_SEED_STAGE,
    max_frames: int = 15_000,
    start_retries: int = DEFAULT_START_RETRIES,
) -> tuple[SweepResult, list[MatchRow]]:
    """Run the prior sweep once and retain both pooled-metric input and exact rows.

    This avoids a second emulator sweep when an experiment needs the legacy
    ``SweepResult`` reduction plus trajectory-derived rows for paired checkpoint
    comparisons. Both outputs therefore describe the identical matches.
    """
    matches, boots = _drive_prior(
        policy_factory,
        session_cfg=session_cfg,
        n_matchups=n_matchups,
        max_parallel=max_parallel,
        cpu_level=cpu_level,
        ego_port=ego_port,
        seed_stage=seed_stage,
        max_frames=max_frames,
        start_retries=start_retries,
    )
    return _prior_sweep_result(boots, seed_stage), match_rows(boots, matches, ego_port=ego_port)


def _rows_by_boot(rows: Sequence[MatchRow]) -> dict[int, list[MatchRow]]:
    """Group rows by boot, ordinal-sorted, asserting one matchup per boot (all of a
    boot's back-to-back matches share the ego/opp characters — instant-restart varies
    only the stage). A boot with mixed matchups is malformed input; fail loud."""
    by_boot: dict[int, list[MatchRow]] = {}
    for row in rows:
        by_boot.setdefault(row.boot_index, []).append(row)
    for boot_index, boot_rows in by_boot.items():
        boot_rows.sort(key=lambda r: r.match_ordinal)
        pairs = {(r.ego_character, r.opp_character) for r in boot_rows}
        if len(pairs) != 1:
            raise ValueError(f"boot {boot_index} has inconsistent matchups across its matches: {sorted(pairs)}")
    return by_boot


def _match_numerator(row: MatchRow, rate_key: str) -> float:
    return {
        "stocks_taken_per_min": float(row.stocks_taken),
        "stocks_lost_per_min": float(row.stocks_lost),
        "damage_dealt_per_min": row.damage_dealt,
        "damage_taken_per_min": row.damage_taken,
    }[rate_key]


def _boot_rate(rows: Sequence[MatchRow], rate_key: str) -> float:
    active = [row for row in rows if row.active_frames > 0]
    numerator = sum(_match_numerator(row, rate_key) for row in active)
    minutes = sum(row.active_frames for row in active) / FRAMES_PER_MINUTE
    return numerator / minutes


def matched_vs_cpu_deltas(
    rows_a: Sequence[MatchRow],
    rows_b: Sequence[MatchRow],
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> dict[str, float]:
    """Return a character-matched boot-level delta of run A minus run B.

    Boot ``i`` must use the same character pair in both runs. Active games from one
    boot are pooled into one rate. The bootstrap samples boots, so games from one
    Dolphin process stay in one cluster. Restart stages and game randomness differ
    across runs. This is a blocked diagnostic, not a common-random-numbers estimate.
    """
    by_boot_a = _rows_by_boot(rows_a)
    by_boot_b = _rows_by_boot(rows_b)
    shared_boots = sorted(by_boot_a.keys() & by_boot_b.keys())
    for boot_index in shared_boots:
        boot_a, boot_b = by_boot_a[boot_index], by_boot_b[boot_index]
        matchup_a = (boot_a[0].ego_character, boot_a[0].opp_character)
        matchup_b = (boot_b[0].ego_character, boot_b[0].opp_character)
        if matchup_a != matchup_b:
            raise ValueError(
                f"boot {boot_index} matchup differs between runs ({matchup_a} vs {matchup_b}); "
                "the two runs did not use the same matchups_for schedule"
            )

    total_boots = len(by_boot_a) + len(by_boot_b)
    matching_rate = (2 * len(shared_boots) / total_boots) if total_boots else 0.0
    if matching_rate < 0.5:
        raise ValueError(
            f"matched_vs_cpu_deltas: matched {len(shared_boots)} of "
            f"{len(by_boot_a)}/{len(by_boot_b)} boots "
            f"(matching_rate {matching_rate:.2f} < 0.5); schedules likely differ"
        )

    out: dict[str, float] = {
        "matching_rate": matching_rate,
        "n_boots": float(len(shared_boots)),
        "n_rows_a": float(len(rows_a)),
        "n_rows_b": float(len(rows_b)),
    }
    rated = [
        (by_boot_a[index], by_boot_b[index])
        for index in shared_boots
        if any(row.active_frames > 0 for row in by_boot_a[index])
        and any(row.active_frames > 0 for row in by_boot_b[index])
    ]
    out["n_boots_rated"] = float(len(rated))
    idx = _resample_index(np.random.default_rng(seed), len(rated), bootstrap_resamples)
    for key in _RATE_KEYS:
        deltas = np.array([_boot_rate(a, key) - _boot_rate(b, key) for a, b in rated], dtype=np.float64)
        point = float(deltas.mean()) if len(deltas) else 0.0
        lo, hi = _mean_ci(deltas, idx, point)
        out[f"{key}_delta_mean"] = point
        out[f"{key}_delta_ci_lo"] = lo
        out[f"{key}_delta_ci_hi"] = hi
    return out
