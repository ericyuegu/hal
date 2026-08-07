"""Paired statistics for a mirrored head-to-head sweep.

Pure math over ``hal.eval.h2h`` match records: no emulator, no torch, and no
game-specific heuristics. The layering rule is deliberate — outcome statistics must stay
independent of any hand-engineered Melee vocabulary, so nothing here reads action states,
conversions or behavior metrics.

Two views of the same matches:

- **Per match.** Win / loss / stock tie, plus the mean stock and damage differential with
  a normal-approximation confidence interval. Every match counts once.
- **Per config (the paired view).** A config's two orientations are summed, which cancels
  the port advantage and the character-matchup advantage. A sign test over those sums
  then asks the direct question: on how many configs is the focal model ahead?
"""

import math
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

from hal.eval.cross_stage import FRAMES_PER_MINUTE
from hal.eval.h2h import MatchRecord

# Two-sided 95% normal quantile, the confidence level every interval here reports.
Z_95: float = 1.96


def wilson_ci(wins: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``wins / n``.

    The Wilson interval stays inside [0, 1] and keeps useful coverage near 0 and 1, where
    the normal approximation fails. An empty sample gives ``(nan, nan)``.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    if not 0 <= wins <= n:
        raise ValueError(f"wins must be in 0..{n}, got {wins}")
    p = wins / n
    denominator = 1 + z * z / n
    center = p + z * z / (2 * n)
    half_width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((center - half_width) / denominator, (center + half_width) / denominator)


def binomial_two_sided_p(wins: int, n: int) -> float:
    """Exact two-sided p-value of ``wins`` successes in ``n`` fair trials.

    This is the method of small p-values: sum the probability of every outcome that is no
    more likely than the observed one. A small tolerance absorbs float error, so the two
    mirror outcomes of a symmetric distribution always both count. An empty sample gives
    ``nan``.
    """
    if n == 0:
        return float("nan")
    if not 0 <= wins <= n:
        raise ValueError(f"wins must be in 0..{n}, got {wins}")
    observed = math.comb(n, wins) / 2**n
    total = sum(math.comb(n, i) / 2**n for i in range(n + 1) if math.comb(n, i) / 2**n <= observed + 1e-12)
    return min(1.0, total)


def mean_ci(values: Sequence[float], z: float = Z_95) -> tuple[float, float, float]:
    """``(mean, low, high)`` of ``values`` under the normal approximation.

    The interval is ``mean +- z * standard error`` with the sample standard deviation
    (one degree of freedom removed). Fewer than two values give ``(mean, nan, nan)``; an
    empty sequence gives all ``nan``.
    """
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = sum(values) / n
    if n < 2:
        return (mean, float("nan"), float("nan"))
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    standard_error = math.sqrt(variance / n)
    return (mean, mean - z * standard_error, mean + z * standard_error)


@dataclass(frozen=True, slots=True)
class SignTest:
    """Sign test over paired differences: how often is the focal side ahead?"""

    ahead: int
    behind: int
    even: int
    p_value: float

    @property
    def pairs(self) -> int:
        return self.ahead + self.behind + self.even


def sign_test(differences: Sequence[float]) -> SignTest:
    """Exact two-sided sign test. Zero differences are ties and leave the test."""
    ahead = sum(1 for d in differences if d > 0)
    behind = sum(1 for d in differences if d < 0)
    even = len(differences) - ahead - behind
    return SignTest(ahead=ahead, behind=behind, even=even, p_value=binomial_two_sided_p(ahead, ahead + behind))


@dataclass(frozen=True, slots=True)
class GroupDelta:
    """Mean per-match stock differential inside one slice of the sweep."""

    label: str
    matches: int
    mean_stock_diff: float


@dataclass(frozen=True, slots=True)
class PairedSummary:
    """Everything the head-to-head summary table shows, from the focal model's side.

    Differentials are focal minus opponent, so a positive number favors the focal model.
    A model ahead on stocks when the frame budget ends counts as a stock leader, not a game
    winner. Stock ties are excluded from ``stock_lead_rate``. ``decided_by_knockout``
    reports matches that actually ended.
    """

    focal_model: str
    opponent_model: str
    matches: int
    stock_leads: int
    stock_deficits: int
    stock_ties: int
    decided_by_knockout: int
    stock_lead_rate: float
    stock_lead_rate_ci_low: float
    stock_lead_rate_ci_high: float
    stock_lead_rate_p_value: float
    mean_stock_diff_per_match: float
    stock_diff_ci_low: float
    stock_diff_ci_high: float
    mean_damage_diff_per_active_minute: float
    damage_diff_ci_low: float
    damage_diff_ci_high: float
    paired_configs: int
    config_sign_test: SignTest
    mean_config_stock_diff: float
    config_stock_diff_ci_low: float
    config_stock_diff_ci_high: float
    by_focal_character: tuple[GroupDelta, ...]
    by_stage: tuple[GroupDelta, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_table(self) -> str:
        """Human-readable summary, one block per view."""
        lines = [
            f"=== {self.focal_model} vs {self.opponent_model}  ({self.matches} matches) ===",
            f"stock leads {self.stock_leads}  stock deficits {self.stock_deficits}  stock ties {self.stock_ties}"
            f"   (knockout-decided before budget: {self.decided_by_knockout})",
            f"stock-lead rate over non-tied: {self.stock_lead_rate:.3f}"
            f"  95% CI [{self.stock_lead_rate_ci_low:.3f}, {self.stock_lead_rate_ci_high:.3f}]"
            f"  binomial p={self.stock_lead_rate_p_value:.3f}",
            f"stock diff /match:          {self.mean_stock_diff_per_match:+.3f}"
            f"  95% CI [{self.stock_diff_ci_low:+.3f}, {self.stock_diff_ci_high:+.3f}]",
            f"damage diff /active-minute: {self.mean_damage_diff_per_active_minute:+.2f}"
            f"  95% CI [{self.damage_diff_ci_low:+.2f}, {self.damage_diff_ci_high:+.2f}]",
            f"paired configs (n={self.paired_configs}): ahead {self.config_sign_test.ahead},"
            f" behind {self.config_sign_test.behind}, even {self.config_sign_test.even};"
            f" sign-test p={self.config_sign_test.p_value:.3f}",
            f"per-config stock diff (both orientations summed): {self.mean_config_stock_diff:+.3f}"
            f"  95% CI [{self.config_stock_diff_ci_low:+.3f}, {self.config_stock_diff_ci_high:+.3f}]",
            f"-- by {self.focal_model}'s character (mean stock diff/match) --",
        ]
        lines += [f"  {g.label:<16} {g.mean_stock_diff:+.2f}  (n={g.matches})" for g in self.by_focal_character]
        lines.append("-- by stage --")
        lines += [f"  {g.label:<16} {g.mean_stock_diff:+.2f}  (n={g.matches})" for g in self.by_stage]
        return "\n".join(lines)


def completed(records: Sequence[MatchRecord]) -> list[MatchRecord]:
    """The records whose match actually ran."""
    return [r for r in records if r.outcome is not None]


def opponent_of(records: Sequence[MatchRecord], focal_model: str) -> str:
    """The other model in the sweep. Fails loud on anything but a two-model sweep."""
    names = {name for r in records for name in (r.model_port_1, r.model_port_2)}
    if focal_model not in names:
        raise ValueError(f"{focal_model!r} played none of these {len(records)} matches; saw {sorted(names)}")
    others = names - {focal_model}
    if len(others) != 1:
        raise ValueError(f"expected exactly two models, saw {sorted(names)}")
    return others.pop()


def stock_diff(record: MatchRecord, focal_model: str) -> float:
    """Stocks the focal model took minus stocks it lost, for one match."""
    if record.outcome is None:
        raise ValueError(f"match {record.match_id} never ran")
    focal_port = record.port_of_model(focal_model)
    opponent_port = 2 if focal_port == 1 else 1
    return float(record.outcome.stocks_lost(opponent_port) - record.outcome.stocks_lost(focal_port))


def damage_diff_per_active_minute(record: MatchRecord, focal_model: str) -> float | None:
    """Damage the focal model dealt minus damage it took, per active minute.

    None when the match recorded no active frame, where the rate is undefined.
    """
    if record.outcome is None:
        raise ValueError(f"match {record.match_id} never ran")
    if record.outcome.active_frames <= 0:
        return None
    focal_port = record.port_of_model(focal_model)
    opponent_port = 2 if focal_port == 1 else 1
    minutes = record.outcome.active_frames / FRAMES_PER_MINUTE
    return (record.outcome.damage_dealt(focal_port) - record.outcome.damage_dealt(opponent_port)) / minutes


def config_stock_diffs(records: Sequence[MatchRecord], focal_model: str) -> dict[int, float]:
    """Per-config stock differential, summed over both orientations.

    Only configs whose two orientations both completed are included: a half config carries
    the port advantage it was designed to cancel.
    """
    by_config: dict[int, list[MatchRecord]] = {}
    for record in completed(records):
        by_config.setdefault(record.config_id, []).append(record)
    out: dict[int, float] = {}
    for config_id, group in by_config.items():
        if {r.orientation for r in group} != {0, 1} or len(group) != 2:
            continue
        out[config_id] = sum(stock_diff(r, focal_model) for r in group)
    return out


def _group_deltas(values_by_label: Mapping[str, list[float]]) -> tuple[GroupDelta, ...]:
    """Mean per group, ordered by sample count then label so the table is stable."""
    return tuple(
        GroupDelta(label=label, matches=len(values), mean_stock_diff=sum(values) / len(values))
        for label, values in sorted(values_by_label.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    )


def summarize_paired(records: Sequence[MatchRecord], *, focal_model: str) -> PairedSummary:
    """Reduce a mirrored head-to-head sweep to its summary table, from ``focal_model``'s side.

    Records whose match never ran are dropped. The per-match view uses every completed
    match; the paired view uses only the configs that completed both orientations.
    """
    rows = completed(records)
    if not rows:
        raise ValueError("no completed matches to summarize")
    opponent_model = opponent_of(rows, focal_model)

    stock_leads = sum(1 for r in rows if r.outcome is not None and r.outcome.stock_leader_model == focal_model)
    stock_deficits = sum(1 for r in rows if r.outcome is not None and r.outcome.stock_leader_model == opponent_model)
    stock_ties = sum(1 for r in rows if r.outcome is not None and r.outcome.stock_leader_model is None)
    decided = sum(1 for r in rows if r.outcome is not None and r.outcome.decided)
    non_tied = stock_leads + stock_deficits
    lead_low, lead_high = wilson_ci(stock_leads, non_tied)

    stock_diffs = [stock_diff(r, focal_model) for r in rows]
    damage_diffs = [d for d in (damage_diff_per_active_minute(r, focal_model) for r in rows) if d is not None]
    stock_mean, stock_low, stock_high = mean_ci(stock_diffs)
    damage_mean, damage_low, damage_high = mean_ci(damage_diffs)

    per_config = config_stock_diffs(rows, focal_model)
    config_mean, config_low, config_high = mean_ci(list(per_config.values()))

    by_character: dict[str, list[float]] = {}
    by_stage: dict[str, list[float]] = {}
    for record, diff in zip(rows, stock_diffs, strict=True):
        character = record.character_of_port(record.port_of_model(focal_model))
        by_character.setdefault(character, []).append(diff)
        by_stage.setdefault(record.stage, []).append(diff)

    return PairedSummary(
        focal_model=focal_model,
        opponent_model=opponent_model,
        matches=len(rows),
        stock_leads=stock_leads,
        stock_deficits=stock_deficits,
        stock_ties=stock_ties,
        decided_by_knockout=decided,
        stock_lead_rate=stock_leads / non_tied if non_tied else float("nan"),
        stock_lead_rate_ci_low=lead_low,
        stock_lead_rate_ci_high=lead_high,
        stock_lead_rate_p_value=binomial_two_sided_p(stock_leads, non_tied),
        mean_stock_diff_per_match=stock_mean,
        stock_diff_ci_low=stock_low,
        stock_diff_ci_high=stock_high,
        mean_damage_diff_per_active_minute=damage_mean,
        damage_diff_ci_low=damage_low,
        damage_diff_ci_high=damage_high,
        paired_configs=len(per_config),
        config_sign_test=sign_test(list(per_config.values())),
        mean_config_stock_diff=config_mean,
        config_stock_diff_ci_low=config_low,
        config_stock_diff_ci_high=config_high,
        by_focal_character=_group_deltas(by_character),
        by_stage=_group_deltas(by_stage),
    )
