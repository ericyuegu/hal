from pathlib import Path

import numpy as np
import pytest
from streaming import MDSWriter
from streaming import StreamingDataset

from hal.data.policy_schema import BUTTON_SUFFIXES
from hal.data.policy_schema import FLOAT_STATE_SUFFIXES
from hal.data.policy_schema import LEADER_PREFIXES
from hal.data.policy_schema import PACKED_STATE_SUFFIXES
from hal.data.policy_schema import PLAYER_PREFIXES
from hal.data.policy_schema import POLICY_MDS_COLUMNS
from hal.data.policy_schema import POLICY_SCHEMA_VERSION
from hal.data.policy_schema import decode_policy_replay
from hal.data.policy_schema import decode_policy_replay_slice
from hal.data.policy_schema import encode_policy_replay
from hal.data.policy_schema import pack_buttons
from hal.data.policy_schema import pack_player_state
from hal.data.policy_schema import pack_stick
from hal.data.policy_schema import pack_trigger
from hal.data.policy_schema import policy_replay_identity
from hal.data.policy_schema import unpack_buttons
from hal.data.policy_schema import unpack_player_state
from hal.data.policy_schema import unpack_player_stock
from hal.data.policy_schema import unpack_stick
from hal.data.policy_schema import unpack_trigger
from hal.training.features import ACTION_CHANNELS
from hal.wire import MASK_INT32

_DEV_TRAIN = Path(__file__).resolve().parents[1] / "data" / "processed" / "dev" / "mds" / "train"


def _bits(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).view(np.uint32)


def test_policy_replay_identity_is_stable() -> None:
    assert policy_replay_identity("archive://ranked/example.slp") == "272a3b895dfedbf42a4bb86febb17ffc"


def test_all_stick_codes_round_trip_exactly() -> None:
    source = np.arange(-80, 81, dtype=np.float32) / np.float32(80)
    source = np.append(source, np.float32(np.nan)).astype(np.float32)
    decoded = unpack_stick(pack_stick(source))
    assert np.array_equal(_bits(decoded[:-1]), _bits(source[:-1]))
    assert np.isnan(decoded[-1])


def test_all_trigger_codes_round_trip_exactly() -> None:
    source = np.arange(141, dtype=np.float32) / np.float32(140)
    source = np.append(source, np.float32(np.nan)).astype(np.float32)
    decoded = unpack_trigger(pack_trigger(source))
    assert np.array_equal(_bits(decoded[:-1]), _bits(source[:-1]))
    assert np.isnan(decoded[-1])


@pytest.mark.parametrize("value", [0.11, -1.1, 1.1])
def test_off_grid_or_out_of_range_controller_values_raise(value: float) -> None:
    with pytest.raises(ValueError):
        pack_stick(np.array([value], dtype=np.float32))


def test_every_button_mask_round_trips() -> None:
    masks = np.arange(256, dtype=np.uint8)
    source = {name: ((masks >> bit) & 1).astype(np.int32) for bit, name in enumerate(BUTTON_SUFFIXES)}
    packed = pack_buttons(source)
    assert packed.tolist() == masks.tolist()
    decoded = unpack_buttons(packed)
    for name in BUTTON_SUFFIXES:
        assert decoded[name].tolist() == source[name].tolist()


def test_button_pack_rejects_missing_or_non_binary_values() -> None:
    source = {name: np.zeros(2, dtype=np.int32) for name in BUTTON_SUFFIXES}
    source[BUTTON_SUFFIXES[0]][1] = MASK_INT32
    with pytest.raises(ValueError):
        pack_buttons(source)


def test_packed_player_state_round_trips_values_and_sentinels() -> None:
    values = {
        "action": np.array([0, 525, 65535, MASK_INT32], dtype=np.int32),
        "stock": np.array([0, 4, 4, MASK_INT32], dtype=np.int32),
        "jumps_used": np.array([0, 8, 8, MASK_INT32], dtype=np.int32),
        "hurtbox_state": np.array([0, 2, 2, MASK_INT32], dtype=np.int32),
        "airborne": np.array([0, 1, 1, MASK_INT32], dtype=np.int32),
        "direction": np.array([-1.0, 1.0, 0.0, np.nan], dtype=np.float32),
    }
    packed = pack_player_state(values)
    decoded = unpack_player_state(packed)
    np.testing.assert_array_equal(unpack_player_stock(packed), values["stock"])
    for name, source in values.items():
        if name == "direction":
            assert np.array_equal(_bits(decoded[name][:-1]), _bits(source[:-1]))
            assert np.isnan(decoded[name][-1])
        else:
            assert decoded[name].tolist() == source.tolist()


def test_character_specific_action_states_above_model_vocab_round_trip() -> None:
    actions = np.arange(520, 526, dtype=np.int32)
    values = {
        "action": actions,
        "stock": np.zeros(actions.shape, dtype=np.int32),
        "jumps_used": np.zeros(actions.shape, dtype=np.int32),
        "hurtbox_state": np.zeros(actions.shape, dtype=np.int32),
        "airborne": np.zeros(actions.shape, dtype=np.int32),
        "direction": np.ones(actions.shape, dtype=np.float32),
    }
    decoded = unpack_player_state(pack_player_state(values))
    assert np.array_equal(decoded["action"], actions)


def test_all_uint16_action_states_round_trip() -> None:
    actions = np.arange(65536, dtype=np.int32)
    values = {
        "action": actions,
        "stock": np.zeros_like(actions),
        "jumps_used": np.zeros_like(actions),
        "hurtbox_state": np.zeros_like(actions),
        "airborne": np.zeros_like(actions),
        "direction": np.ones(actions.shape, dtype=np.float32),
    }
    assert np.array_equal(unpack_player_state(pack_player_state(values))["action"], actions)


@pytest.mark.parametrize(
    ("name", "bad"),
    [("action", 65536), ("stock", 5), ("jumps_used", 9), ("hurtbox_state", 3), ("airborne", 2)],
)
def test_state_pack_rejects_reserved_codes(name: str, bad: int) -> None:
    values = {
        "action": np.array([0], dtype=np.int32),
        "stock": np.array([0], dtype=np.int32),
        "jumps_used": np.array([0], dtype=np.int32),
        "hurtbox_state": np.array([0], dtype=np.int32),
        "airborne": np.array([0], dtype=np.int32),
        "direction": np.array([1.0], dtype=np.float32),
    }
    values[name][0] = bad
    with pytest.raises(ValueError):
        pack_player_state(values)


def test_state_unpack_rejects_reserved_high_bits() -> None:
    with pytest.raises(ValueError, match="reserved bits"):
        unpack_player_state(np.array([1 << 31], dtype=np.uint32))
    with pytest.raises(ValueError, match="reserved bits"):
        unpack_player_stock(np.array([1 << 31], dtype=np.uint32))


def test_compact_columns_round_trip_through_mds(tmp_path) -> None:
    n = 3
    sample: dict[str, object] = {
        "policy_schema_version": POLICY_SCHEMA_VERSION,
        "source_schema_version": 7,
        "replay_id": "fixture",
        "num_frames": n,
        "stage": 2,
        "p1_character": 1,
        "p2_character": 22,
        "p1_nana_present": 0,
        "p2_nana_present": 0,
    }
    for name, encoding in POLICY_MDS_COLUMNS.items():
        if name in sample:
            continue
        dtype = np.dtype(encoding.removeprefix("ndarray:"))
        sample[name] = np.zeros(1 if "nana" in name else n, dtype=dtype)

    with MDSWriter(out=str(tmp_path), columns=POLICY_MDS_COLUMNS, compression="zstd") as writer:
        writer.write(sample)
    loaded = StreamingDataset(local=str(tmp_path), batch_size=1, shuffle=False)[0]
    assert loaded["replay_id"] == "fixture"
    assert loaded["num_frames"] == n
    for name, value in sample.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(loaded[name], value), name


@pytest.mark.skipif(not _DEV_TRAIN.is_dir(), reason="local dev MDS is not available")
def test_compact_replay_preserves_every_p0_value() -> None:
    source = StreamingDataset(local=str(_DEV_TRAIN), batch_size=1, shuffle=False)[0]
    encoded = encode_policy_replay(source, "dev-0")
    decoded = decode_policy_replay(encoded)

    for name in ("stage", "p1_character", "p2_character"):
        assert np.array_equal(decoded[name], source[name]), name
    for prefix in PLAYER_PREFIXES:
        for name in (*FLOAT_STATE_SUFFIXES, *PACKED_STATE_SUFFIXES):
            key = f"{prefix}_{name}"
            assert np.array_equal(decoded[key], source[key], equal_nan=True), key
    for prefix in LEADER_PREFIXES:
        for name in (*ACTION_CHANNELS[:6], *(f"button_{name}" for name in BUTTON_SUFFIXES)):
            key = f"{prefix}_{name}"
            assert np.array_equal(decoded[key], source[key], equal_nan=True), key

    for prefix in ("p1_nana", "p2_nana"):
        if not int(encoded[f"{prefix}_present"]):
            assert all(np.asarray(encoded[f"{prefix}_{name}"]).shape == (1,) for name in FLOAT_STATE_SUFFIXES)
            assert np.asarray(encoded[f"{prefix}_state"]).shape == (1,)


@pytest.mark.skipif(not _DEV_TRAIN.is_dir(), reason="local dev MDS is not available")
def test_slice_decode_matches_full_decode_exactly() -> None:
    source = StreamingDataset(local=str(_DEV_TRAIN), batch_size=1, shuffle=False)[0]
    encoded = encode_policy_replay(source, "dev-0")
    full = decode_policy_replay(encoded)
    frames = int(encoded["num_frames"])

    for start, stop in ((0, 1), (0, min(17, frames)), (1, min(257, frames)), (frames - 1, frames)):
        sliced = decode_policy_replay_slice(encoded, start, stop)
        assert sliced.keys() == full.keys()
        assert sliced["schema_version"] == full["schema_version"]
        for name in sliced.keys() - {"schema_version"}:
            expected = np.asarray(full[name])[start:stop]
            assert np.array_equal(sliced[name], expected, equal_nan=True), (name, start, stop)


@pytest.mark.skipif(not _DEV_TRAIN.is_dir(), reason="local dev MDS is not available")
def test_slice_decode_rejects_out_of_bounds_ranges() -> None:
    source = StreamingDataset(local=str(_DEV_TRAIN), batch_size=1, shuffle=False)[0]
    encoded = encode_policy_replay(source, "dev-0")
    frames = int(encoded["num_frames"])
    for start, stop in ((-1, 1), (2, 1), (0, frames + 1)):
        with pytest.raises(ValueError, match="outside replay length"):
            decode_policy_replay_slice(encoded, start, stop)


@pytest.mark.skipif(not _DEV_TRAIN.is_dir(), reason="local dev MDS is not available")
def test_decode_rejects_invalid_nana_presence_flag() -> None:
    source = StreamingDataset(local=str(_DEV_TRAIN), batch_size=1, shuffle=False)[0]
    encoded = encode_policy_replay(source, "dev-0")
    encoded["p1_nana_present"] = 2
    with pytest.raises(ValueError, match="p1_nana_present must be 0 or 1"):
        decode_policy_replay(encoded)


@pytest.mark.skipif(not _DEV_TRAIN.is_dir(), reason="local dev MDS is not available")
def test_replay_encode_error_identifies_replay_player_character_and_value() -> None:
    source = dict(StreamingDataset(local=str(_DEV_TRAIN), batch_size=1, shuffle=False)[0])
    source["p2_action"] = np.asarray(source["p2_action"]).copy()
    source["p2_action"][0] = 65536
    with pytest.raises(
        ValueError,
        match=r"dev-bad: p2_state \(character=\d+\): action has 1 invalid value\(s\) \[65536\]",
    ):
        encode_policy_replay(source, "dev-bad")
