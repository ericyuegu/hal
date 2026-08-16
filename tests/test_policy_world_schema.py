import functools
from pathlib import Path

import numpy as np
import pytest
from streaming import MDSWriter
from streaming import StreamingDataset

from hal.data.policy_schema import POLICY_MDS_COLUMNS
from hal.data.policy_schema import decode_policy_replay
from hal.data.policy_world_schema import POLICY_WORLD_MDS_COLUMNS
from hal.data.policy_world_schema import POLICY_WORLD_SCHEMA_VERSION
from hal.data.policy_world_schema import assert_policy_world_replay_equal
from hal.data.policy_world_schema import decode_policy_world_replay
from hal.data.policy_world_schema import decode_policy_world_replay_slice
from hal.data.policy_world_schema import encode_policy_world_replay
from hal.data.policy_world_schema import policy_row_from_world
from hal.data.schema import Rank
from hal.training.features import FeatureProjection
from hal.training.replay_reservoir import PolicyReplayPackDataset
from hal.wire import ITEM_FIELD_SUFFIXES
from hal.wire import ITEM_SLOTS
from hal.wire import item_column

_V7_TRAIN = (
    Path(__file__).resolve().parents[1] / "data" / "processed" / "ranked-anonymized-1" / "mds-v7-sub4" / "train"
)


@functools.lru_cache(maxsize=1)
def _source() -> dict[str, object]:
    if not _V7_TRAIN.is_dir():
        pytest.skip("local v7 subset is not available")
    return dict(StreamingDataset(local=str(_V7_TRAIN), batch_size=1, shuffle=False)[0])


def test_real_v7_world_round_trip_preserves_ranks_and_items() -> None:
    source = _source()
    encoded = encode_policy_world_replay(source, "ranked-0")
    assert encoded["policy_world_schema_version"] == POLICY_WORLD_SCHEMA_VERSION
    assert_policy_world_replay_equal(source, encoded, "ranked-0")
    decoded = decode_policy_world_replay(encoded)

    for name in ("p1_rank", "p2_rank"):
        assert np.array_equal(decoded[name], source[name])
    for slot in range(ITEM_SLOTS):
        for suffix in ITEM_FIELD_SUFFIXES:
            name = item_column(slot, suffix)
            assert np.array_equal(decoded[name], source[name], equal_nan=True), name


def test_world_slice_matches_full_decode() -> None:
    source = dict(_source())
    source["p1_rank"] = np.full(len(source["frame"]), Rank.PRO, dtype=np.uint8)
    encoded = encode_policy_world_replay(source, "pro-0")
    full = decode_policy_world_replay(encoded)
    for start, stop in ((0, 1), (7, 43), (len(source["frame"]) - 1, len(source["frame"]))):
        sliced = decode_policy_world_replay_slice(encoded, start, stop)
        assert sliced.keys() == full.keys()
        assert sliced["schema_version"] == full["schema_version"]
        for name in sliced.keys() - {"schema_version"}:
            assert np.array_equal(sliced[name], np.asarray(full[name])[start:stop], equal_nan=True), name


def test_world_to_policy_is_payload_exact() -> None:
    encoded = encode_policy_world_replay(_source(), "ranked-0")
    projected = policy_row_from_world(encoded)
    assert projected.keys() == POLICY_MDS_COLUMNS.keys()
    assert decode_policy_replay(projected).keys() <= decode_policy_world_replay(encoded).keys()
    for name in POLICY_MDS_COLUMNS:
        left = projected[name]
        right = encoded[name]
        if isinstance(left, np.ndarray):
            assert np.array_equal(left, right, equal_nan=True), name
        else:
            assert left == right


def test_world_columns_round_trip_through_mds(tmp_path: Path) -> None:
    encoded = encode_policy_world_replay(_source(), "ranked-0")
    with MDSWriter(out=str(tmp_path), columns=POLICY_WORLD_MDS_COLUMNS, compression="zstd") as writer:
        writer.write(encoded)
    loaded = StreamingDataset(local=str(tmp_path), batch_size=1, shuffle=False)[0]
    assert loaded["replay_id"] == "ranked-0"
    assert_policy_world_replay_equal(_source(), loaded, "mds")


def test_decoder_rejects_reserved_presence_and_metadata_bits() -> None:
    encoded = encode_policy_world_replay(_source(), "ranked-0")
    encoded["item_present"] = np.asarray(encoded["item_present"]).copy()
    encoded["item_present"][0] |= np.uint8(1 << ITEM_SLOTS)
    with pytest.raises(ValueError, match="reserved bits"):
        decode_policy_world_replay(encoded)


def test_replay_reservoir_slice_decoder_exposes_rank_and_items() -> None:
    encoded = encode_policy_world_replay(_source(), "ranked-0")
    projection = FeatureProjection(
        frozenset({"ego_rank", "item0_type", "item0_pos_x", "ego_main_stick_x"}),
        derive_spatial=False,
    )
    packs = PolicyReplayPackDataset(
        [encoded],
        32,
        2,
        seed=7,
        windows_per_replay=2,
        schema_version=7,
        projection=projection,
        replay_format="policy-world",
    )
    pack = next(iter(packs))
    assert pack.replay_id == "ranked-0"
    assert pack.windows
    for window in pack.windows:
        assert {"ego_rank", "item0_type", "item0_pos_x", "ego_main_stick_x", "ctx_pad"} == window.keys()
        assert len(window["item0_type"]) == 34

    encoded = encode_policy_world_replay(_source(), "ranked-0")
    encoded["item0_meta"] = np.asarray(encoded["item0_meta"]).copy()
    encoded["item0_meta"][0] |= np.uint32(1 << 31)
    with pytest.raises(ValueError, match="reserved bits"):
        decode_policy_world_replay(encoded)
