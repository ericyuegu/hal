"""Contracts for the deterministic O49 player-identity sidecar."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hal.data.policy_schema import policy_replay_identity
from hal.data.schema import Rank
from hal.training import player_identity


def _write_manifest(path: Path) -> tuple[str, str, str]:
    train_path = "archive://players.zip!train.slp"
    missing_path = "archive://players.zip!missing.slp"
    validation_path = "archive://players.zip!validation.slp"
    rows = (
        {
            "path": train_path,
            "players": [
                {"port": 4, "code": "aa#1", "name": "Shared name"},
                {"port": 2, "code": " AA#1 ", "name": "Shared name"},
            ],
            "annotation": {"split": "train"},
        },
        {
            "path": missing_path,
            "players": [
                {"port": 1, "code": None, "name": "AA#1"},
                {"port": 3, "code": "BB#2", "name": None},
            ],
            "annotation": {"split": "train"},
        },
        {
            "path": validation_path,
            "players": [
                {"port": 2, "code": "NEW#3", "name": "New"},
                {"port": 4, "code": "BB#2", "name": "Known"},
            ],
            "annotation": {"split": "val"},
        },
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return train_path, missing_path, validation_path


def test_sidecar_is_deterministic_exact_and_train_only(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    train_path, missing_path, validation_path = _write_manifest(manifest)
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"
    source = (player_identity.ManifestInput("fixture", manifest),)

    first_summary = player_identity.build_player_identity_sidecar(source, first)
    second_summary = player_identity.build_player_identity_sidecar(source, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_summary["sha256"] == second_summary["sha256"]
    assert first_summary["unique_connect_codes_train"] == 3
    assert first_summary["unique_connect_codes_all_splits"] == 4
    assert first_summary["casefold_connect_code_collision_groups_train"] == 1
    assert first_summary["nicknames_are_identity_keys"] is False

    sidecar = player_identity.load_player_identity_sidecar(first, expected_sha256=first_summary["sha256"])
    assert sidecar.vocabulary.codes == ("AA#1", "BB#2", "aa#1")
    assert sidecar.vocabulary.id_for_code(" AA#1 ") != sidecar.vocabulary.id_for_code("aa#1")
    with pytest.raises(KeyError, match="absent"):
        sidecar.vocabulary.id_for_code("Aa#1")

    train_id = policy_replay_identity(train_path)
    missing_id = policy_replay_identity(missing_path)
    validation_id = policy_replay_identity(validation_path)
    assert sidecar.by_replay[train_id] == (
        sidecar.vocabulary.id_for_code("AA#1"),
        sidecar.vocabulary.id_for_code("aa#1"),
    )
    assert sidecar.by_replay[missing_id] == (0, sidecar.vocabulary.id_for_code("BB#2"))
    assert sidecar.by_replay[validation_id] == (0, sidecar.vocabulary.id_for_code("BB#2"))


def test_replay_lookup_uses_professional_ids_or_rank_aggregates() -> None:
    lookup = player_identity.ReplayPlayerLookup({"a" * 32: (7, 8)})

    professional = lookup({"replay_id": "a" * 32, "num_frames": 3})
    assert professional["p1_player_id"].tolist() == [7, 7, 7]
    assert professional["p2_player_id"].tolist() == [8, 8, 8]

    ranked = lookup(
        {
            "replay_id": "b" * 32,
            "num_frames": 2,
            "p1_rank": np.asarray(int(Rank.DIAMOND), dtype=np.uint8),
            "p2_rank": np.asarray(int(Rank.MASTER), dtype=np.uint8),
        }
    )
    assert ranked["p1_player_id"].tolist() == [int(Rank.DIAMOND)] * 2
    assert ranked["p2_player_id"].tolist() == [int(Rank.MASTER)] * 2

    with pytest.raises(KeyError, match="unsupported ranks"):
        lookup(
            {
                "replay_id": "c" * 32,
                "num_frames": 2,
                "p1_rank": np.asarray(int(Rank.PRO), dtype=np.uint8),
                "p2_rank": np.asarray(int(Rank.MASTER), dtype=np.uint8),
            }
        )


def test_rank_ids_and_checkpoint_vocabulary_round_trip() -> None:
    vocabulary = player_identity.PlayerVocabulary(("AA#1", "aa#1"))
    encoded = player_identity.vocabulary_buffer(vocabulary)
    restored = player_identity.vocabulary_from_checkpoint_buffer(encoded)

    assert restored == vocabulary
    assert restored.id_for_rank(Rank.PLATINUM) == 1
    assert restored.id_for_rank(Rank.DIAMOND) == 2
    assert restored.id_for_rank(Rank.MASTER) == 3
    with pytest.raises(ValueError, match="requires Platinum"):
        restored.id_for_rank(Rank.PRO)
