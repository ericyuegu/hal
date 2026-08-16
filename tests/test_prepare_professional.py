from pathlib import Path

from hal.data.index import PlayerEntry
from hal.data.index import ReplayIndexEntry
from hal.data.index import write_jsonl
from hal.data.schema import SCHEMA_VERSION
from hal.data.schema import Rank
from hal.scripts.prepare_professional import dedupe_index
from hal.scripts.prepare_professional import write_corpus_rank_overrides
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


def _player(port: int, name: str | None, code: str | None) -> PlayerEntry:
    return PlayerEntry(
        port=port,
        character=1,
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


def test_corpus_rank_mode_labels_both_sides_explicitly(tmp_path: Path) -> None:
    paths = tmp_path / "paths.txt"
    overrides = tmp_path / "ranks.jsonl"
    paths.write_text("one\ntwo\n")

    report = write_corpus_rank_overrides(paths, overrides)

    assert overrides.read_text().count(f'"p1_rank": {int(Rank.PRO)}') == 2
    assert overrides.read_text().count(f'"p2_rank": {int(Rank.PRO)}') == 2
    assert report["method"] == "corpus"
