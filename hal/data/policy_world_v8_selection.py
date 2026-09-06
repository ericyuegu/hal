"""Selection and rank policy for the policy-world-v8 corpus."""

import dataclasses
import hashlib
from dataclasses import dataclass
from typing import Final

from hal.data.index import ReplayIndexEntry
from hal.data.schema import Rank
from hal.policy import INCLUDED_STAGES
from hal.wire import slp_stage_to_libmelee

RANK_IMPUTATION_PERSON: Final[bytes] = b"hal-pwv8-rank1"
RANK_DIAMOND_THRESHOLD: Final[int] = 3 * 2**64 // 10

_KNOWN_DUPLICATES: Final[dict[tuple[str, str, int], tuple[str, str]]] = {
    ("professional-monotheon-policy-world-v7", "train", 14_160): (
        "4b650975420bb169ff0b8fe4cb13df8bd8b99c5e",
        "professional-daniel-policy-world-v7 train row 3392",
    ),
    ("professional-monotheon-policy-world-v7", "train", 14_163): (
        "c4173bf6a44238e5c53dcea510debe5b2419e8fb",
        "professional-daniel-policy-world-v7 train row 3386",
    ),
}


@dataclass(frozen=True, slots=True)
class PolicyWorldV8QualityPolicy:
    min_frames: int = 1_500
    stages: tuple[str, ...] = tuple(stage.name for stage in INCLUDED_STAGES)
    min_damage_dealt: float = 100.0
    min_damage_taken: float = 100.0
    min_stock_losses: int = 3
    max_cheap_deaths_exclusive: int = 2
    cheap_death_percent: float = 10.0
    starting_stocks: int = 4
    min_damage_dealt_per_player_exclusive: float = 50.0
    players: int = 2
    player_type: str = "HUMAN"

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


V8_QUALITY_POLICY: Final[PolicyWorldV8QualityPolicy] = PolicyWorldV8QualityPolicy()


def failed_quality_rules(
    entry: ReplayIndexEntry,
    policy: PolicyWorldV8QualityPolicy = V8_QUALITY_POLICY,
) -> tuple[str, ...]:
    """Return every v8 quality rule that one manifest row fails."""
    stats = entry.stats
    if stats is None:
        raise ValueError(f"{entry.path}: policy-world-v8 selection requires replay statistics")

    failures: list[str] = []
    frame_count = entry.annotation.frame_count_actual if entry.annotation is not None else entry.frame_count
    if frame_count < policy.min_frames:
        failures.append("min_frames")
    try:
        stage_is_legal = slp_stage_to_libmelee(entry.stage) in INCLUDED_STAGES
    except ValueError:
        stage_is_legal = False
    if not stage_is_legal:
        failures.append("legal_stage")
    if len(entry.players) != policy.players or any(
        player.player_type != policy.player_type for player in entry.players
    ):
        failures.append("two_human_players")
    if len(stats.players) != policy.players:
        failures.append("two_player_statistics")
        return tuple(failures)
    if any(player.stocks_remaining + len(player.death_percents) != policy.starting_stocks for player in stats.players):
        failures.append("four_starting_stocks_per_player")
    if not any(player.damage_dealt >= policy.min_damage_dealt for player in stats.players):
        failures.append("one_player_dealt_100")
    if not any(player.damage_taken >= policy.min_damage_taken for player in stats.players):
        failures.append("one_player_took_100")
    if not any(len(player.death_percents) >= policy.min_stock_losses for player in stats.players):
        failures.append("one_player_lost_three_stocks")
    if any(
        sum(percent <= policy.cheap_death_percent for percent in player.death_percents)
        >= policy.max_cheap_deaths_exclusive
        for player in stats.players
    ):
        failures.append("fewer_than_two_cheap_deaths_per_player")
    if any(player.damage_dealt <= policy.min_damage_dealt_per_player_exclusive for player in stats.players):
        failures.append("each_player_dealt_over_50")
    return tuple(failures)


def known_duplicate_rejection(
    source_name: str,
    split: str,
    source_row: int,
    entry: ReplayIndexEntry,
) -> str | None:
    """Return the fixed duplicate rejection, and verify its expected SHA-1."""
    expected = _KNOWN_DUPLICATES.get((source_name, split, source_row))
    if expected is None:
        return None
    sha1, retained = expected
    if entry.sha1 != sha1:
        raise ValueError(
            f"known duplicate position {source_name} {split} row {source_row} has SHA-1 {entry.sha1!r}, "
            f"expected {sha1}"
        )
    return f"known_duplicate_of:{retained}"


def replay_id_bytes(replay_id: bytes | str) -> bytes:
    """Normalize a v7 hexadecimal replay identity to its raw 16 bytes."""
    if isinstance(replay_id, str):
        try:
            value = bytes.fromhex(replay_id)
        except ValueError as error:
            raise ValueError("replay_id must be a hexadecimal 16-byte identity") from error
    else:
        value = replay_id
    if len(value) != 16:
        raise ValueError(f"replay_id must contain exactly 16 bytes, got {len(value)}")
    return value


def _imputed_rank(replay_id: bytes, logical_side: int) -> Rank:
    digest = hashlib.blake2b(
        replay_id + bytes((logical_side,)),
        digest_size=8,
        person=RANK_IMPUTATION_PERSON,
    ).digest()
    value = int.from_bytes(digest, "big", signed=False)
    return Rank.DIAMOND if value < RANK_DIAMOND_THRESHOLD else Rank.MASTER


def assign_v8_ranks(
    source_name: str,
    replay_id: bytes | str,
    p1_rank: int,
    p2_rank: int,
) -> tuple[Rank, Rank, int]:
    """Apply the observed-PRO and deterministic professional fallback policy."""
    try:
        ranks = (Rank(p1_rank), Rank(p2_rank))
    except ValueError as error:
        raise ValueError(f"v7 row has an invalid rank pair {(p1_rank, p2_rank)}") from error
    if not source_name.startswith("professional-"):
        return ranks[0], ranks[1], 0

    identity = replay_id_bytes(replay_id)
    assigned: list[Rank] = []
    imputed_mask = 0
    for logical_side, rank in enumerate(ranks, start=1):
        if rank == Rank.PRO:
            assigned.append(rank)
            continue
        assigned.append(_imputed_rank(identity, logical_side))
        imputed_mask |= 1 << (logical_side - 1)
    return assigned[0], assigned[1], imputed_mask


def rank_imputation_rule() -> dict[str, object]:
    return {
        "algorithm": "blake2b-64",
        "input": "raw replay_id bytes followed by logical side byte (1 or 2)",
        "byte_order": "big",
        "personalization_hex": RANK_IMPUTATION_PERSON.hex(),
        "diamond_when_digest_below": RANK_DIAMOND_THRESHOLD,
        "professional_sources": "preserve PRO; impute every other side",
        "ranked_anonymous_sources": "preserve observed v7 rank",
    }
