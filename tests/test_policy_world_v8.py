import struct

import numpy as np
import pytest
from streaming import MDSWriter
from streaming import StreamingDataset

from hal.data.feature_stats import FeatureStats
from hal.data.policy_schema import FLOAT_STATE_SUFFIXES
from hal.data.policy_schema import PACKED_STATE_SUFFIXES
from hal.data.policy_schema import POLICY_SCHEMA_VERSION
from hal.data.policy_schema import pack_player_state
from hal.data.policy_world_schema import ITEM_FLOAT_SUFFIXES
from hal.data.policy_world_schema import POLICY_WORLD_MDS_COLUMNS
from hal.data.policy_world_schema import POLICY_WORLD_SCHEMA_VERSION
from hal.data.policy_world_schema import decode_policy_world_replay
from hal.data.policy_world_v8 import P1_MATCH_POINT
from hal.data.policy_world_v8 import P1_STOCK_LOSS
from hal.data.policy_world_v8 import P2_MATCH_POINT
from hal.data.policy_world_v8 import P2_STOCK_LOSS
from hal.data.policy_world_v8 import decode_policy_world_v8_replay
from hal.data.policy_world_v8 import decode_policy_world_v8_reward_events
from hal.data.policy_world_v8 import decode_policy_world_v8_slice
from hal.data.policy_world_v8 import encode_policy_world_v8_replay
from hal.data.policy_world_v8 import policy_world_v7_row_from_v8
from hal.data.schema import POLICY_WORLD_V8_MDS_COLUMNS
from hal.data.schema import SCHEMA_VERSION
from hal.data.schema import Rank
from hal.training import dataloader
from hal.training import returns
from hal.training.features import BASE_ACTION_PROJECTION
from hal.training.features import FLOAT_FEATURES
from hal.training.features import FeatureProjection
from hal.wire import ACTION_CHANNELS
from hal.wire import ITEM_SLOTS
from hal.wire import MASK_INT32
from hal.wire import item_column

_FRAMES = 600


def _state(stock: np.ndarray) -> np.ndarray:
    zeros = np.zeros(stock.shape, dtype=np.int32)
    return pack_player_state(
        {
            "action": zeros,
            "stock": stock,
            "jumps_used": zeros,
            "hurtbox_state": zeros,
            "airborne": zeros,
            "direction": np.zeros(stock.shape, dtype=np.float32),
        }
    )


def _v7_row() -> dict[str, object]:
    row: dict[str, object] = {
        "policy_world_schema_version": POLICY_WORLD_SCHEMA_VERSION,
        "policy_schema_version": POLICY_SCHEMA_VERSION,
        "source_schema_version": SCHEMA_VERSION,
        "replay_id": bytes(range(16)).hex(),
        "num_frames": _FRAMES,
        "stage": 31,
        "p1_character": 1,
        "p2_character": 2,
        "p1_nana_present": 1,
        "p2_nana_present": 0,
        "p1_rank": int(Rank.DIAMOND),
        "p2_rank": int(Rank.MASTER),
    }
    base = np.linspace(-10, 10, _FRAMES, dtype=np.float32)
    for prefix in ("p1", "p2", "p1_nana"):
        for offset, suffix in enumerate(FLOAT_STATE_SUFFIXES):
            row[f"{prefix}_{suffix}"] = base + np.float32(offset)
    absent_nan = np.array([0x7FC01234], dtype=np.uint32).view(np.float32)
    for suffix in FLOAT_STATE_SUFFIXES:
        row[f"p2_nana_{suffix}"] = absent_nan.copy()

    p1_stock = np.full(_FRAMES, 4, dtype=np.int32)
    p1_stock[100:] = 3
    p1_stock[300:] = 2
    p1_stock[500:] = 1
    p1_stock[599:] = 0
    p2_stock = np.full(_FRAMES, 4, dtype=np.int32)
    p2_stock[256:] = 3
    row["p1_state"] = _state(p1_stock)
    row["p2_state"] = _state(p2_stock)
    row["p1_nana_state"] = _state(np.full(_FRAMES, 4, dtype=np.int32))
    missing = np.full(1, MASK_INT32, dtype=np.int32)
    row["p2_nana_state"] = pack_player_state(
        {
            name: np.full(1, np.nan, dtype=np.float32) if name == "direction" else missing
            for name in PACKED_STATE_SUFFIXES
        }
    )
    p1_percent = np.asarray(row["p1_percent"]).copy()
    p2_percent = np.asarray(row["p2_percent"]).copy()
    p1_percent[:] = 0
    p2_percent[:] = 0
    p1_percent[255:] = 12.5
    p2_percent[256:] = 20.25
    row["p1_percent"] = p1_percent
    row["p2_percent"] = p2_percent

    for prefix in ("p1", "p2"):
        for name in ACTION_CHANNELS[:4]:
            row[f"{prefix}_{name}"] = np.zeros(_FRAMES, dtype=np.int8)
        for name in ACTION_CHANNELS[4:6]:
            row[f"{prefix}_{name}"] = np.zeros(_FRAMES, dtype=np.uint8)
        row[f"{prefix}_buttons"] = np.arange(_FRAMES, dtype=np.uint8)

    presence = np.zeros(_FRAMES, dtype=np.uint8)
    presence[[255, 256]] = 1
    row["item_present"] = presence
    for slot in range(ITEM_SLOTS):
        present = ((presence >> slot) & 1).astype(bool)
        meta = np.zeros(_FRAMES, dtype=np.uint32)
        meta[present] = np.uint32(1 | (1 << 26))
        row[f"item{slot}_meta"] = meta
        for suffix in ITEM_FLOAT_SUFFIXES:
            values = np.full(_FRAMES, np.nan, dtype=np.float32)
            values[present] = np.float32(slot + 0.25)
            row[item_column(slot, suffix)] = values
    assert set(row) == set(POLICY_WORLD_MDS_COLUMNS)
    return row


def _encoded(source_name: str = "ranked-anonymized-1-policy-world-v7") -> dict[str, object]:
    return encode_policy_world_v8_replay(_v7_row(), source_name=source_name, p1_port=1, p2_port=3)


def test_full_codec_reconstructs_v7_bytes_and_outer_metadata() -> None:
    source = _v7_row()
    encoded = _encoded()
    compact = policy_world_v7_row_from_v8(encoded)

    assert set(encoded) == set(POLICY_WORLD_V8_MDS_COLUMNS)
    assert encoded["replay_id"] == bytes(range(16))
    assert encoded["p1_port"] == 1 and encoded["p2_port"] == 3
    assert encoded["mc_terminated"] == 1
    for name, expected in source.items():
        actual = compact[name]
        if isinstance(expected, np.ndarray):
            assert isinstance(actual, np.ndarray)
            assert actual.dtype == expected.dtype, name
            assert actual.tobytes() == expected.tobytes(), name
        else:
            assert actual == expected, name

    expected_full = decode_policy_world_replay(source)
    actual_full = decode_policy_world_v8_replay(encoded)
    for name, expected in expected_full.items():
        actual = actual_full[name]
        if isinstance(expected, np.ndarray):
            assert isinstance(actual, np.ndarray)
            assert actual.tobytes() == expected.tobytes(), name
        else:
            assert actual == expected, name


def test_slice_crosses_block_boundary_and_reads_only_core(monkeypatch: pytest.MonkeyPatch) -> None:
    from hal.data import policy_world_v8 as module

    encoded = _encoded()
    full = decode_policy_world_v8_replay(encoded)
    calls: list[tuple[int, int]] = []
    original = module._decompress_chunk

    def observe(index: object, group: int, block: int) -> bytes:
        calls.append((group, block))
        return original(index, group, block)  # type: ignore[arg-type]

    monkeypatch.setattr(module, "_decompress_chunk", observe)
    sliced = decode_policy_world_v8_slice(encoded, 255, 257, groups=frozenset({"core"}))

    assert calls == [(1, 0), (1, 1)]
    assert not any("nana" in name or name.startswith("item") for name in sliced)
    for name, actual in sliced.items():
        if name in {"p1_port", "p2_port", "p1_rank_imputed", "p2_rank_imputed"}:
            continue
        expected = full[name]
        if isinstance(actual, np.ndarray):
            assert actual.tobytes() == np.asarray(expected)[255:257].tobytes(), name


def test_sparse_reward_events_preserve_damage_and_stock_flags() -> None:
    events = decode_policy_world_v8_reward_events(_encoded())
    by_frame = {int(event["frame"]): event for event in events}

    assert float(by_frame[255]["p1_damage_taken"]) == 12.5
    assert float(by_frame[256]["p2_damage_taken"]) == 20.25
    assert int(by_frame[100]["flags"]) & P1_STOCK_LOSS
    assert int(by_frame[256]["flags"]) & P2_STOCK_LOSS
    assert int(by_frame[599]["flags"]) & (P1_STOCK_LOSS | P1_MATCH_POINT)
    assert not int(by_frame[256]["flags"]) & P2_MATCH_POINT


@pytest.mark.parametrize(
    ("gamma", "damage_shaping", "win_reward", "stock_value"),
    [
        (0.0, 0.0, 0.0, 1.0),
        (0.99618, 1.0, 50.0, 120.0),
        (1.0, 0.125, 2.0, 1.0),
    ],
)
def test_sparse_returns_match_full_decode_exactly(
    gamma: float,
    damage_shaping: float,
    win_reward: float,
    stock_value: float,
) -> None:
    encoded = _encoded()
    decoded = decode_policy_world_v8_replay(encoded)
    kwargs = {
        "gamma": gamma,
        "damage_shaping": damage_shaping,
        "win_reward": win_reward,
        "stock_value": stock_value,
        "suffix": "return",
    }

    expected = returns.replay_returns(decoded, **kwargs)
    actual = returns.compact_policy_returns(encoded, **kwargs)

    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])


def test_sparse_returns_reject_an_unobserved_terminal_tail() -> None:
    encoded = _encoded()
    encoded["mc_terminated"] = 0

    actual = returns.compact_policy_returns(
        encoded,
        gamma=0.9,
        damage_shaping=1.0,
        win_reward=50.0,
        stock_value=120.0,
        suffix="return",
    )

    for port in ("p1", "p2"):
        assert np.isnan(actual[f"{port}_return"]).all()
        assert not actual[f"{port}_return_valid"].any()


def test_crc_failure_is_rejected() -> None:
    encoded = _encoded()
    payload = bytearray(encoded["block_payload"])
    header_size = struct.calcsize("<8sBBHIIIII")
    crc_offset = header_size + struct.calcsize("<BBII")
    payload[crc_offset] ^= 1
    encoded["block_payload"] = bytes(payload)

    with pytest.raises(ValueError, match="CRC32"):
        decode_policy_world_v8_slice(encoded, 0, 1, groups=frozenset({"core"}))


def test_v8_columns_round_trip_through_raw_mds(tmp_path) -> None:
    encoded = _encoded()
    with MDSWriter(out=str(tmp_path), columns=POLICY_WORLD_V8_MDS_COLUMNS, compression=None) as writer:
        writer.write(encoded)
    loaded = dict(StreamingDataset(local=str(tmp_path), batch_size=1, shuffle=False)[0])

    assert loaded["replay_id"] == bytes(range(16))
    assert loaded["block_payload"] == encoded["block_payload"]
    assert decode_policy_world_v8_replay(loaded)["frame"].shape == (_FRAMES,)


def test_v8_window_dataset_decodes_only_the_selected_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    encoded = _encoded()
    calls: list[tuple[int, int, frozenset[str]]] = []
    original = dataloader.decode_policy_world_v8_slice

    def observe(
        source: dict[str, object],
        start: int,
        stop: int,
        *,
        groups: frozenset[str],
    ) -> dict[str, np.ndarray | int]:
        calls.append((start, stop, groups))
        return original(source, start, stop, groups=groups)  # type: ignore[arg-type]

    def labels(source: dict[str, object]) -> dict[str, np.ndarray]:
        frames = int(source["num_frames"])
        return {
            "p1_marker": np.arange(frames, dtype=np.int32),
            "p2_marker": np.arange(frames, dtype=np.int32) + 1000,
        }

    monkeypatch.setattr(dataloader, "decode_policy_world_v8_slice", observe)
    projection = FeatureProjection(
        frozenset({"frame", "ego_position_x", "opp_position_x", "ego_rank", "ego_rank_imputed", "ego_marker"}),
        derive_spatial=False,
    )
    dataset = dataloader._PolicyWorldV8WindowDataset(
        [encoded],
        6,
        4,
        seed=17,
        schema_version=SCHEMA_VERSION,
        projection=projection,
        replay_labels=labels,
        require_full_context=False,
    )

    window = next(iter(dataset))

    assert calls and calls[0][2] == frozenset({"core"})
    assert all(len(value) == 10 for name, value in window.items() if name != "ctx_pad")
    first_real = int(window["ctx_pad"])
    frame = int(window["frame"][first_real])
    marker = int(window["ego_marker"][first_real])
    assert marker in (frame, frame + 1000)
    assert not any("nana" in name or name.startswith("item") for name in window)


def test_make_loader_routes_v8_directly_to_the_window_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class CapturingLoader:
        def __init__(self, dataset: object, **_kwargs: object) -> None:
            captured["dataset"] = dataset

    monkeypatch.setattr(dataloader, "DataLoader", CapturingLoader)
    monkeypatch.setattr(
        dataloader,
        "make_streaming_dataset",
        lambda *_args, **_kwargs: ([dict(_encoded())], ()),
    )

    dataloader.make_loader(
        "unused",
        "train",
        stats={},
        L_ctx=6,
        L_chunk=4,
        batch_size=1,
        seed=17,
        num_workers=0,
        replay_format="policy-world-v8",
    )

    assert isinstance(captured["dataset"], dataloader._PolicyWorldV8WindowDataset)


def test_v8_window_uses_the_existing_training_preprocessor() -> None:
    dataset = dataloader._PolicyWorldV8WindowDataset(
        [_encoded()],
        6,
        4,
        seed=17,
        schema_version=SCHEMA_VERSION,
        projection=BASE_ACTION_PROJECTION,
        replay_labels=None,
        require_full_context=True,
    )
    window = next(iter(dataset))
    unit = FeatureStats(mean=0.0, std=1.0, min=-1.0, max=1.0)
    stats = {name: unit for name in FLOAT_FEATURES}
    stats.update({f"nana_{name}": unit for name in FLOAT_FEATURES})

    batch = dataloader.collate_train_batch(
        [window],
        stats=stats,
        L_ctx=6,
        projection=BASE_ACTION_PROJECTION,
    )

    assert batch.target.shape == (1, 4, 14)
    assert all(value.shape[:2] == (1, 6) for value in batch.context.features.values())


def test_v8_window_resume_reproduces_the_remaining_epoch() -> None:
    first = _encoded()
    second = dict(_encoded())
    second["replay_id"] = bytes(reversed(range(16)))

    complete = dataloader._PolicyWorldV8WindowDataset(
        [first, second],
        6,
        4,
        seed=17,
        schema_version=SCHEMA_VERSION,
        projection=BASE_ACTION_PROJECTION,
        replay_labels=None,
        require_full_context=True,
    )
    expected = list(complete)[1]
    resumed = dataloader._PolicyWorldV8WindowDataset(
        [second],
        6,
        4,
        seed=17,
        schema_version=SCHEMA_VERSION,
        projection=BASE_ACTION_PROJECTION,
        replay_labels=None,
        require_full_context=True,
    )
    resumed.resume_epoch(0)
    actual = next(iter(resumed))

    assert actual.keys() == expected.keys()
    for name in actual:
        np.testing.assert_array_equal(actual[name], expected[name])


def test_real_mosaic_v8_loader_yields_one_window_per_row(tmp_path) -> None:
    with MDSWriter(
        out=str(tmp_path / "train"),
        columns=POLICY_WORLD_V8_MDS_COLUMNS,
        compression=None,
    ) as writer:
        for index in range(4):
            row = dict(_encoded())
            row["replay_id"] = index.to_bytes(16, "little")
            writer.write(row)
    unit = FeatureStats(mean=0.0, std=1.0, min=-1.0, max=1.0)
    stats = {name: unit for name in FLOAT_FEATURES}
    stats.update({f"nana_{name}": unit for name in FLOAT_FEATURES})
    loader = dataloader.make_loader(
        str(tmp_path),
        "train",
        stats=stats,
        L_ctx=6,
        L_chunk=4,
        batch_size=2,
        seed=17,
        shuffle=False,
        num_workers=0,
        resumable=True,
        replay_format="policy-world-v8",
        projection=BASE_ACTION_PROJECTION,
        require_full_context=True,
    )

    batches = list(loader)

    assert len(batches) == 2
    assert sum(len(batch.target) for batch in batches) == 4
