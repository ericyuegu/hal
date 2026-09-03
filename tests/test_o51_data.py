"""Contracts for O51 nested bands and training-only replay columns."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from streaming import MDSWriter
from streaming import StreamingDataset

from hal import streams
from hal.data.o51 import D0
from hal.data.o51 import O48_PROTOCOL
from hal.data.o51 import OFFICIAL_CORPUS_TARGETS
from hal.data.o51 import OFFICIAL_RAW_REPLAYS
from hal.data.o51 import OFFICIAL_TIER_REPLAYS
from hal.data.o51 import OFFICIAL_TIER_TARGETS
from hal.data.o51 import OFFICIAL_UNIQUE_REPLAYS
from hal.data.o51 import Inventory
from hal.data.o51 import InventoryEntry
from hal.data.o51 import build_nested_corpus
from hal.data.o51 import iter_band_manifest
from hal.data.o51 import read_o48_inventory
from hal.data.o51 import write_band_manifests
from hal.data.o51_schema import O51_MDS_COLUMNS
from hal.data.o51_schema import O51_MDS_SCHEMA_VERSION
from hal.data.o51_schema import O51_RETURN_SUFFIX
from hal.data.policy_schema import PACKED_STATE_SUFFIXES
from hal.data.policy_schema import pack_player_state
from hal.data.policy_world_schema import POLICY_WORLD_MDS_COLUMNS
from hal.data.policy_world_schema import POLICY_WORLD_SCHEMA_VERSION
from hal.training.o51_data import encode_o51_replay
from hal.training.o51_data import o51_replay_labels

_FIXTURE_CORPUS_HASH = "162a9f6e10f8d71d71a2fa539ccd8d1ed85c6236e5c73b7f96e39f6bcd78b4cf"
_FIXTURE_BAND_HASHES = {
    1: "68a2c2e554de829d36b812edaddd385ade9483240e6e0804ac57f41aee19d3bf",
    2: "ca9a15e6a37871c3863f3039780298e3dffa7faf2765920c3959cc2ff71d653e",
    4: "90e2a6236800fa6deb6206772d9da269c61b4a5fc6c789418d3fd545228a90a3",
    8: "f94b14bd8de9e48ea1c4c7c37540533ee383b04f67aae2c981214cbadccbdbea",
}
_FIXTURE_TIER_HASHES = {
    1: "68a2c2e554de829d36b812edaddd385ade9483240e6e0804ac57f41aee19d3bf",
    2: "3af5f1f850f734110b0e22f1b3d87bb50d32eaf3489944915be3a92a5f0a07e0",
    4: "dad47b75a2a208e62a350b8e12fac7c80a15c60503e9cc427f759738e8cd2ecf",
    8: "9eee83144b150a66de1f2905f92006efdea3012534604ca81c12872882542815",
}


def _inventory() -> Inventory:
    entries = []
    digest = hashlib.sha256()
    for source_index, source in enumerate(streams.POLICY_WORLD_V7_SOURCES):
        for row in range(8):
            entry = InventoryEntry(
                source=source.name,
                row=row,
                shard=f"shard-{row // 2}",
                replay_id=f"{source.name}-{row}",
                frames=100 + 8 * source_index + row,
                content_sha1=hashlib.sha1(f"{source.name}:{row}".encode()).hexdigest(),
            )
            entries.append(entry)
            digest.update(
                f"{entry.source}\t{entry.row}\t{entry.shard}\t{entry.replay_id}\t"
                f"{entry.frames}\t{entry.content_sha1}\n".encode()
            )
    assert digest.hexdigest() == _FIXTURE_CORPUS_HASH
    return Inventory({"corpus_hash": digest.hexdigest()}, tuple(entries))


def test_official_nested_data_accounting_is_pinned() -> None:
    assert D0 == 2**30
    assert (OFFICIAL_RAW_REPLAYS, OFFICIAL_UNIQUE_REPLAYS) == (1_300_640, 1_300_638)
    assert OFFICIAL_CORPUS_TARGETS == 26_582_742_076
    assert OFFICIAL_TIER_REPLAYS == {1: 162_598, 2: 325_176, 4: 650_331, 8: 1_300_638}
    assert OFFICIAL_TIER_TARGETS == {
        1: 3_321_597_594,
        2: 6_647_731_852,
        4: 13_297_093_392,
        8: 26_582_742_076,
    }


def test_nested_selection_is_strict_source_stratified_and_hash_pinned() -> None:
    inventory = _inventory()
    corpus = build_nested_corpus(inventory, strict_official=False)

    assert {scale: band.sha256 for scale, band in corpus.bands.items()} == _FIXTURE_BAND_HASHES
    assert {scale: tier.sha256 for scale, tier in corpus.tiers.items()} == _FIXTURE_TIER_HASHES
    assert {scale: tier.unique_replays for scale, tier in corpus.tiers.items()} == {
        1: 44,
        2: 88,
        4: 176,
        8: 352,
    }
    previous: set[str] = set()
    for scale in (1, 2, 4, 8):
        tier = corpus.tiers[scale]
        replay_ids = {entry.replay_id for entry in tier.entries}
        assert previous < replay_ids
        assert set(tier.source_replays) == {source.name for source in streams.POLICY_WORLD_V7_SOURCES}
        assert set(tier.source_replays.values()) == {scale}
        previous = replay_ids
    assert previous == {entry.replay_id for entry in inventory.entries}
    assert sum(band.unique_replays for band in corpus.bands.values()) == len(inventory.entries)


def test_o48_inventory_authentication_and_band_manifests_are_exact(tmp_path: Path) -> None:
    inventory = _inventory()
    path = tmp_path / "inventory.tsv.gz"
    header = {
        "schema_version": 1,
        "protocol": O48_PROTOCOL,
        "source_names": [source.name for source in streams.POLICY_WORLD_V7_SOURCES],
        "canonical_replays": len(inventory.entries),
        "canonical_loss_positions": sum(entry.potential_targets for entry in inventory.entries),
        "corpus_hash": inventory.corpus_hash,
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(f"#{json.dumps(header)}\n")
        for entry in inventory.entries:
            handle.write(
                f"{entry.source}\t{entry.row}\t{entry.shard}\t{entry.replay_id}\t"
                f"{entry.frames}\t{entry.content_sha1}\n"
            )

    loaded = read_o48_inventory(path)
    assert loaded.entries == inventory.entries
    assert loaded.corpus_hash == inventory.corpus_hash
    corpus = build_nested_corpus(loaded, strict_official=False)
    paths = write_band_manifests(corpus, tmp_path / "bands")
    first_payloads = [manifest.read_bytes() for manifest in paths]
    assert write_band_manifests(corpus, tmp_path / "bands") == paths
    assert [manifest.read_bytes() for manifest in paths] == first_payloads
    for scale, manifest in zip((1, 2, 4, 8), paths, strict=True):
        assert tuple(iter_band_manifest(manifest)) == corpus.bands[scale].entries


def _compact_replay(frames: int, *, terminal: bool) -> dict[str, object]:
    compact: dict[str, object] = {}
    for name, encoding in POLICY_WORLD_MDS_COLUMNS.items():
        if encoding == "str":
            compact[name] = "replay-1"
        elif encoding == "int":
            compact[name] = 0
        else:
            compact[name] = np.zeros(frames, dtype=np.dtype(encoding.removeprefix("ndarray:")))
    compact.update(
        policy_world_schema_version=POLICY_WORLD_SCHEMA_VERSION,
        source_schema_version=7,
        replay_id="replay-1",
        num_frames=frames,
    )
    for port in ("p1", "p2"):
        values = {name: np.zeros(frames, dtype=np.int32) for name in PACKED_STATE_SUFFIXES}
        values["stock"].fill(4)
        values["direction"] = np.ones(frames, dtype=np.float32)
        if terminal and port == "p2":
            values["stock"][-1] = 0
        compact[f"{port}_state"] = pack_player_state(values)
        compact[f"{port}_percent"] = np.arange(frames, dtype=np.float32)
    return compact


def _encode(frames: int, *, terminal: bool) -> dict[str, object]:
    labels = {
        "p1_player_id": np.full(frames, 7, dtype=np.int32),
        "p2_player_id": np.full(frames, 11, dtype=np.int32),
    }
    return encode_o51_replay(
        _compact_replay(frames, terminal=terminal),
        player_labels=labels,
        gamma=0.9,
        damage_shaping=1.0,
        win_reward=50.0,
        stock_value=120.0,
    )


def test_o51_schema_precomputes_returns_masks_and_scalar_player_ids(tmp_path: Path) -> None:
    frames = 12
    terminal = _encode(frames, terminal=True)
    truncated = _encode(frames, terminal=False)

    assert terminal["o51_schema_version"] == O51_MDS_SCHEMA_VERSION
    assert (terminal["p1_player_id"], terminal["p2_player_id"]) == (7, 11)
    for port in ("p1", "p2"):
        returns_name = f"{port}_{O51_RETURN_SUFFIX}"
        valid_name = f"{returns_name}_valid"
        assert np.isfinite(terminal[returns_name]).all()
        assert np.asarray(terminal[valid_name]).all()
        assert np.isnan(truncated[returns_name]).all()
        assert not np.asarray(truncated[valid_name]).any()

    labels = o51_replay_labels(terminal)
    assert labels["p1_player_id"].shape == labels["p2_player_id"].shape == ()
    assert (labels["p1_player_id"].item(), labels["p2_player_id"].item()) == (7, 11)

    with MDSWriter(out=str(tmp_path), columns=O51_MDS_COLUMNS, compression="zstd") as writer:
        writer.write(terminal)
    loaded = StreamingDataset(local=str(tmp_path), batch_size=1, shuffle=False)[0]
    loaded_labels = o51_replay_labels(loaded)
    for name, values in labels.items():
        np.testing.assert_array_equal(loaded_labels[name], values)
