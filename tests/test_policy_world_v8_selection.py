import dataclasses

from hal.data.index import PlayerEntry
from hal.data.index import ReplayIndexEntry
from hal.data.policy_world_v8_selection import assign_v8_ranks
from hal.data.policy_world_v8_selection import failed_quality_rules
from hal.data.policy_world_v8_selection import known_duplicate_rejection
from hal.data.replay_stats import PlayerStats
from hal.data.replay_stats import ReplayStats
from hal.data.schema import SCHEMA_VERSION
from hal.data.schema import Rank


def _player(port: int, player_type: str = "HUMAN") -> PlayerEntry:
    return PlayerEntry(port=port, character=1, costume=0, player_type=player_type, code=None, name=None)  # type: ignore[arg-type]


def _stats(port: int, damage: float = 120.0) -> PlayerStats:
    return PlayerStats(
        port=port,
        damage_dealt=damage,
        damage_taken=damage,
        stocks_remaining=1,
        inputs=100,
        death_percents=(80.0, 90.0, 100.0),
    )


def _entry() -> ReplayIndexEntry:
    return ReplayIndexEntry(
        path="archive://source!valid.slp",
        slp_version=(3, 18, 0),
        stage=31,
        players=[_player(1), _player(2)],
        frame_count=1_500,
        timestamp=None,
        played_on="network",
        outcome=None,
        rank_filename=None,
        sha1="0" * 40,
        schema_version=SCHEMA_VERSION,
        stats=ReplayStats((_stats(1), _stats(2))),
    )


def test_quality_policy_keeps_valid_nonterminal_replay() -> None:
    assert failed_quality_rules(_entry()) == ()


def test_quality_policy_reports_every_failed_rule() -> None:
    entry = _entry()
    assert entry.stats is not None
    bad_stats = ReplayStats(
        (
            dataclasses.replace(
                entry.stats.players[0],
                damage_dealt=50.0,
                damage_taken=40.0,
                stocks_remaining=1,
                death_percents=(10.0, 10.0),
            ),
            dataclasses.replace(
                entry.stats.players[1],
                damage_dealt=49.0,
                damage_taken=40.0,
                stocks_remaining=1,
                death_percents=(9.0, 9.0),
            ),
        )
    )
    bad = dataclasses.replace(
        entry,
        frame_count=1_499,
        stage=99,
        players=[_player(1), _player(2, "CPU")],
        stats=bad_stats,
    )

    assert set(failed_quality_rules(bad)) == {
        "min_frames",
        "legal_stage",
        "two_human_players",
        "four_starting_stocks_per_player",
        "one_player_dealt_100",
        "one_player_took_100",
        "one_player_lost_three_stocks",
        "fewer_than_two_cheap_deaths_per_player",
        "each_player_dealt_over_50",
    }


def test_professional_rank_imputation_preserves_only_pro_and_is_stable() -> None:
    replay_id = bytes(range(16))
    first = assign_v8_ranks("professional-franz-policy-world-v7", replay_id, Rank.MASTER, Rank.PRO)
    second = assign_v8_ranks("professional-franz-policy-world-v7", replay_id, Rank.MASTER, Rank.PRO)

    assert first == second
    assert first[0] == Rank.MASTER
    assert first[1] == Rank.PRO
    assert first[2] == 1


def test_ranked_anonymous_ranks_are_not_imputed() -> None:
    assert assign_v8_ranks(
        "ranked-anonymized-1-policy-world-v7",
        bytes(range(16)),
        Rank.PLATINUM,
        Rank.MASTER,
    ) == (Rank.PLATINUM, Rank.MASTER, 0)


def test_known_duplicate_position_requires_the_audited_sha1() -> None:
    entry = dataclasses.replace(_entry(), sha1="4b650975420bb169ff0b8fe4cb13df8bd8b99c5e")
    reason = known_duplicate_rejection(
        "professional-monotheon-policy-world-v7",
        "train",
        14_160,
        entry,
    )

    assert reason == "known_duplicate_of:professional-daniel-policy-world-v7 train row 3392"
