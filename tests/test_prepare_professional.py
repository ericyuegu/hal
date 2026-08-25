import dataclasses
import json
from pathlib import Path

from hal.data.index import PlayerEntry
from hal.data.index import ReplayIndexEntry
from hal.data.index import write_jsonl
from hal.data.replay_stats import PlayerStats
from hal.data.replay_stats import ReplayStats
from hal.data.schema import SCHEMA_VERSION
from hal.data.schema import Rank
from hal.scripts.prepare_professional import dedupe_index
from hal.scripts.prepare_professional import write_corpus_rank_overrides
from hal.scripts.prepare_professional import write_filterable_index
from hal.scripts.prepare_professional import write_owner_rank_overrides


def _entry(path: str, sha1: str, players: list[PlayerEntry]) -> ReplayIndexEntry:
    return ReplayIndexEntry(
        path=path,
        slp_version=(3, 18, 0),
        stage=31,
        players=players,
        frame_count=2_000,
        timestamp=None,
        played_on="network",
        outcome=None,
        rank_filename=None,
        sha1=sha1,
        schema_version=SCHEMA_VERSION,
    )


def _player(port: int, name: str | None, code: str | None, character: int = 1) -> PlayerEntry:
    return PlayerEntry(
        port=port,
        character=character,
        costume=0,
        player_type="HUMAN",
        code=code,
        name=name,
    )


def test_dedupe_is_stable_by_path(tmp_path: Path) -> None:
    source = tmp_path / "index.jsonl"
    destination = tmp_path / "deduped.jsonl"
    players = [_player(1, "A", "A#1"), _player(2, "B", "B#1")]
    write_jsonl(
        source,
        [
            _entry("archive://z!same.slp", "duplicate", players),
            _entry("archive://a!same.slp", "duplicate", players),
            _entry("archive://m!other.slp", "unique", players),
        ],
    )

    assert dedupe_index(source, destination) == (3, 2)
    assert '"path": "archive://a!same.slp"' in destination.read_text()
    assert '"path": "archive://z!same.slp"' not in destination.read_text()


def test_filterable_index_ledgers_rows_without_stats(tmp_path: Path) -> None:
    source = tmp_path / "deduped.jsonl"
    destination = tmp_path / "filterable.jsonl"
    failures = tmp_path / "filter.failures.jsonl"
    players = [_player(1, "A", "A#1"), _player(2, "B", "B#1")]
    entries = [
        _entry("archive://one!missing.slp", "missing", players),
        dataclasses.replace(
            _entry("archive://one!present.slp", "present", players),
            stats=ReplayStats(
                players=(
                    PlayerStats(1, 100.0, 100.0, 0, 10, (50.0, 60.0, 70.0)),
                    PlayerStats(2, 100.0, 100.0, 0, 10, (50.0, 60.0, 70.0)),
                )
            ),
        ),
    ]
    write_jsonl(source, entries)

    assert write_filterable_index(source, destination, failures) == (2, 1)
    assert "present.slp" in destination.read_text()
    failure = json.loads(failures.read_text())
    assert failure["phase"] == "filter_prerequisite"
    assert "missing.slp" in failure["path"]


def test_owner_rank_uses_logical_port_order_and_leaves_missing_unknown(tmp_path: Path) -> None:
    index = tmp_path / "index.jsonl"
    paths = tmp_path / "paths.txt"
    overrides = tmp_path / "ranks.jsonl"
    entries = [
        _entry(
            "archive://one!1.slp",
            "one",
            [_player(2, "Opponent", "OPP#1"), _player(3, "Krudo", "KRUDO#0")],
        ),
        _entry(
            "archive://one!2.slp",
            "two",
            [_player(1, None, None), _player(4, None, None)],
        ),
    ]
    write_jsonl(index, entries)
    paths.write_text("\n".join(entry.path for entry in entries) + "\n")

    report = write_owner_rank_overrides("krudo", index, paths, overrides)

    rows = overrides.read_text().splitlines()
    assert '"p1_rank": 0' in rows[0]
    assert f'"p2_rank": {int(Rank.PRO)}' in rows[0]
    assert '"p1_rank": 0' in rows[1]
    assert '"p2_rank": 0' in rows[1]
    assert report["owner_labeled_replays"] == 1
    assert report["unlabeled_replays"] == 1


def test_friend_alias_prefers_dominant_matching_component(tmp_path: Path) -> None:
    index = tmp_path / "index.jsonl"
    paths = tmp_path / "paths.txt"
    overrides = tmp_path / "ranks.jsonl"
    entries = [
        _entry(
            f"archive://one!{game}.slp",
            str(game),
            [_player(1, "RegularFriend", "FRIE#438"), _player(2, f"Opponent{game}", None)],
        )
        for game in range(4)
    ]
    entries.append(
        _entry(
            "archive://one!collision.slp",
            "collision",
            [_player(1, "Friend", "ROY#296"), _player(2, "Opponent", None)],
        )
    )
    write_jsonl(index, entries)
    paths.write_text("\n".join(entry.path for entry in entries) + "\n")

    report = write_owner_rank_overrides("friend", index, paths, overrides)

    assert report["method"] == "alias"
    assert report["owner_labeled_replays"] == 4
    assert "name:regularfriend" in report["owner_tokens"]


def test_trif_alias_resolves_archive_identity(tmp_path: Path) -> None:
    index = tmp_path / "index.jsonl"
    paths = tmp_path / "paths.txt"
    overrides = tmp_path / "ranks.jsonl"
    entries = [
        _entry(
            "archive://one!1.slp",
            "one",
            [_player(1, "Trifit", "TRIF#0"), _player(2, "Opponent", None)],
        )
    ]
    write_jsonl(index, entries)
    paths.write_text(entries[0].path + "\n")

    report = write_owner_rank_overrides("trif", index, paths, overrides)

    assert report["owner_labeled_replays"] == 1
    assert "code:trif0" in report["owner_tokens"]


def test_franz_uses_pro_owner_and_master_corpus_fallback(tmp_path: Path) -> None:
    index = tmp_path / "index.jsonl"
    paths = tmp_path / "paths.txt"
    overrides = tmp_path / "ranks.jsonl"
    entries = [
        _entry(
            "archive://one!identified.slp",
            "identified",
            [_player(1, "Opponent", None), _player(2, "Franz", "FLOW#242")],
        ),
        _entry(
            "archive://one!doctor.slp",
            "doctor",
            [_player(1, None, None, character=21), _player(2, None, None, character=2)],
        ),
        _entry(
            "archive://one!anonymous.slp",
            "anonymous",
            [_player(1, None, None, character=1), _player(2, None, None, character=2)],
        ),
    ]
    write_jsonl(index, entries)
    paths.write_text("\n".join(entry.path for entry in entries) + "\n")

    report = write_owner_rank_overrides("franz", index, paths, overrides)

    rows = [json.loads(line) for line in overrides.read_text().splitlines()]
    assert rows[0] == {
        "path": entries[0].path,
        "p1_rank": int(Rank.MASTER),
        "p2_rank": int(Rank.PRO),
    }
    assert rows[1] == {
        "path": entries[1].path,
        "p1_rank": int(Rank.PRO),
        "p2_rank": int(Rank.MASTER),
    }
    assert rows[2] == {
        "path": entries[2].path,
        "p1_rank": int(Rank.MASTER),
        "p2_rank": int(Rank.MASTER),
    }
    assert report["owner_labeled_replays"] == 2
    assert report["character_fallback_labeled_replays"] == 1
    assert report["fully_ranked_replays"] == 3


def test_corpus_rank_mode_labels_both_sides_explicitly(tmp_path: Path) -> None:
    paths = tmp_path / "paths.txt"
    overrides = tmp_path / "ranks.jsonl"
    paths.write_text("one\ntwo\n")

    report = write_corpus_rank_overrides(paths, overrides)

    assert overrides.read_text().count(f'"p1_rank": {int(Rank.PRO)}') == 2
    assert overrides.read_text().count(f'"p2_rank": {int(Rank.PRO)}') == 2
    assert report["method"] == "corpus"
