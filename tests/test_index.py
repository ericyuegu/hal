"""Schema-version guard on the Stage-1 index (``ReplayIndexEntry`` / ``read_jsonl``).

A stale ``index.jsonl`` (built before the character-id normalization) would
silently feed external ids to ``materialize`` and bake them into a v5 manifest.
Every entry now carries ``schema_version`` and ``read_jsonl`` rejects mismatches.
"""

import json
from pathlib import Path

import pytest

from hal.data.index import PlayerEntry
from hal.data.index import ReplayIndexEntry
from hal.data.index import read_jsonl
from hal.data.index import write_jsonl
from hal.data.schema import SCHEMA_VERSION


def _entry(schema_version: int = SCHEMA_VERSION) -> ReplayIndexEntry:
    return ReplayIndexEntry(
        path="x.slp",
        slp_version=(3, 14, 0),
        stage=31,
        players=[PlayerEntry(port=1, character=1, costume=0, player_type="HUMAN", code=None, name=None)],
        frame_count=100,
        timestamp=None,
        played_on=None,
        outcome=None,
        rank_filename=None,
        sha1=None,
        schema_version=schema_version,
    )


def test_entry_roundtrips_schema_version() -> None:
    e = _entry()
    assert e.schema_version == SCHEMA_VERSION
    assert ReplayIndexEntry.from_dict(e.to_dict()).schema_version == SCHEMA_VERSION


def test_read_jsonl_rejects_stale_index(tmp_path: Path) -> None:
    p = tmp_path / "index.jsonl"
    write_jsonl(p, [_entry(schema_version=SCHEMA_VERSION - 1)])
    with pytest.raises(ValueError, match="schema_version"):
        list(read_jsonl(p))


def test_read_jsonl_rejects_unversioned_index(tmp_path: Path) -> None:
    """Pre-versioning indexes have no ``schema_version`` field — they must be
    rejected, not silently treated as current."""
    p = tmp_path / "index.jsonl"
    d = _entry().to_dict()
    del d["schema_version"]
    p.write_text(json.dumps(d) + "\n")
    with pytest.raises(ValueError, match="schema_version"):
        list(read_jsonl(p))


def test_read_jsonl_accepts_current_index(tmp_path: Path) -> None:
    p = tmp_path / "index.jsonl"
    write_jsonl(p, [_entry()])
    entries = list(read_jsonl(p))
    assert len(entries) == 1
    assert entries[0].schema_version == SCHEMA_VERSION


def test_read_jsonl_can_skip_verification(tmp_path: Path) -> None:
    p = tmp_path / "index.jsonl"
    write_jsonl(p, [_entry(schema_version=SCHEMA_VERSION - 1)])
    assert len(list(read_jsonl(p, verify_schema_version=False))) == 1
